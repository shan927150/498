"""Frozen-routing 2x2 factorial ablation of MoE GPTQ calibration (helpers).

Two manipulated factors of the CALIBRATION-HESSIAN construction only:
  activation source  X in {O, A}: the cached pristine-BF16 layer inputs from the ORIGINAL extraction (O) or from
                                  grouping G20261111 (A); aligned by (original sequence id, token position)
  routing source     R in {O, A}: the ordered GPU top-k expert assignments of the natural batch-1 replay of the
                                  layer on X_O (R_O) or on X_A (R_A)
Condition names are <activation><routing>: OO, AA (natural), AO (frozen original routing), OA (routing only).

Mechanism (transformers 4.51.3 Qwen2MoeSparseMoeBlock, verified by tests): the router weights multiply the expert
OUTPUT, so they never enter the E3 Profiler's Hessians, which are UNWEIGHTED sums over the tokens routed to the
expert of x x^T (gate/up pre-hook input = MoE-block input rows) and z z^T (down_proj pre-hook input
z = act(gate_proj x) * up_proj x, computed by the expert module with its ORIGINAL BF16 weights). The override
therefore only replaces `selected_experts` inside a verbatim copy of the block forward, for the Hessian-building
Profiler run only. Everything else (the Profiler hooks, BlockHessianAccumulator, block assignment, the GPTQ
solver, reconstruction scoring, PPL evaluation) is the unmodified original code. The block output under a forced
routing is computed but discarded (non-sequential calibration: every layer reads its own precomputed cache).
"""
from __future__ import annotations
import hashlib
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from . import model_utils as MU
from .profiler import Profiler

CONDITIONS = {"OO": ("O", "O"), "AO": ("A", "O"), "OA": ("O", "A"), "AA": ("A", "A")}   # name -> (activation, routing)
NATURAL, HYBRID = ("OO", "AA"), ("AO", "OA")
DIAG_LAYERS = (1, 8, 16, 23)
DIAG_EXPERTS = (0, 30)                                       # prespecified (= EP.CHECK_EXPERTS), not outcome-based
DIAG_KEYS = [(L, e, p) for L in DIAG_LAYERS for e in DIAG_EXPERTS for p in ("gu", "down")]


# ---------------------------------------------------------------- forced MoE forward
def make_forced_forward(blk, provider):
    """Verbatim Qwen2MoeSparseMoeBlock.forward (transformers 4.51.3) except that `selected_experts` is passed through
    provider(selected_experts) -> LongTensor [T, k]. If the provider returns the natural tensor itself the computation
    is op-for-op identical to the original forward. Router weights for a forced assignment are the softmax
    probabilities of the forced experts (they only scale the discarded block output, never a Hessian)."""
    def forward(hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = blk.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, blk.top_k, dim=-1)
        if blk.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        forced = provider(selected_experts)
        if forced is not selected_experts:
            forced = forced.to(selected_experts.device, selected_experts.dtype)
            assert forced.shape == selected_experts.shape, (forced.shape, selected_experts.shape)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float).gather(1, forced)
            if blk.norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            selected_experts = forced
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype,
                                          device=hidden_states.device)
        expert_mask = F.one_hot(selected_experts, num_classes=blk.num_experts).permute(2, 1, 0)
        for expert_idx in range(blk.num_experts):
            expert_layer = blk.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        shared_expert_output = blk.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(blk.shared_expert_gate(hidden_states)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
    return forward


@contextmanager
def block_override(model, L, provider):
    blk = MU.moe_block(model, L)
    assert "forward" not in blk.__dict__, "MoE block already overridden"
    blk.forward = make_forced_forward(blk, provider)
    try:
        yield blk
    finally:
        del blk.forward


def assert_no_override(model, layers=None):
    """Inference must run the class forward of every MoE block (no forced routing during evaluation)."""
    n = MU.num_layers(model)
    for L in (layers if layers is not None else range(n)):
        blk = MU.moe_block(model, L)
        assert "forward" not in blk.__dict__, f"layer {L}: MoE forward is overridden"


# ---------------------------------------------------------------- routing capture
@torch.no_grad()
def capture_routing(model, L, cache, rows, sids):
    """Natural ordered top-k of the batch-1 replay (the E3 Profiler path, on the layer's device) for cache rows
    `rows` (original sequence ids `sids`). Returns {sid: LongTensor [T, k] (cpu)}, {sid: FloatTensor [T, k] probs}."""
    sel, prob, cur = {}, {}, {}

    def provider(s):
        sel[cur["sid"]] = s.detach().to("cpu").clone()
        return s

    def gate_hook(m, i, o):
        p = torch.softmax(o, dim=-1, dtype=torch.float)
        cur["probs"] = p
    blk = MU.moe_block(model, L)
    h = blk.gate.register_forward_hook(gate_hook)
    try:
        with block_override(model, L, provider):
            for r, sid in zip(rows, sids):
                cur["sid"] = int(sid)
                MU.run_layer(model, L, cache[r:r + 1].to(MU.GPU, non_blocking=True))
                prob[int(sid)] = cur["probs"].gather(1, sel[int(sid)].to(cur["probs"].device)).cpu()
    finally:
        h.remove()
    return sel, prob


def expert_seq_counts(sel_by_sid, E):
    """{sid: {e: number of tokens with e in their top-k}} (the Profiler's seq_counts semantics)."""
    out = {}
    for sid, s in sel_by_sid.items():
        c = torch.zeros(E, dtype=torch.long).scatter_add_(0, s.reshape(-1).long(), torch.ones(s.numel(), dtype=torch.long))
        out[int(sid)] = {int(e): int(c[e]) for e in torch.nonzero(c).flatten().tolist()}
    return out


def expert_token_counts(sel_by_sid, sids, E):
    c = torch.zeros(E, dtype=torch.long)
    for sid in sids:
        s = sel_by_sid[int(sid)].reshape(-1).long()
        c.scatter_add_(0, s, torch.ones_like(s))
    return c


def routing_sha(sel_by_sid):
    h = hashlib.sha256()
    for sid in sorted(sel_by_sid):
        h.update(int(sid).to_bytes(8, "little"))
        h.update(sel_by_sid[sid].to(torch.int64).contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- forced-routing Hessian Profiler
class ForcedProfiler(Profiler):
    """The original Profiler; while attached, layer L's MoE block uses `table[row_to_sid[cur_seq]]` as its
    selected experts. `expect_natural=True` asserts that the forced table equals the natural selection for every
    token (natural-condition regression). Statistics of forced-vs-natural disagreement are recorded."""
    def __init__(self, model, layer, table, row_to_sid, expect_natural=False, **kw):
        super().__init__(model, layer, **kw)
        self.table, self.row_to_sid, self.expect_natural = table, row_to_sid, expect_natural
        self.n_tok = self.n_ordered_diff = self.n_set_diff = self.n_seq = 0
        self._ov = None

    def _provider(self, natural):
        sid = int(self.row_to_sid[self.cur_seq])
        forced = self.table[sid].to(natural.device, natural.dtype)
        assert forced.shape == natural.shape, (sid, forced.shape, natural.shape)
        od = (forced != natural).any(-1)
        sd = (forced.sort(-1).values != natural.sort(-1).values).any(-1)
        self.n_seq += 1
        self.n_tok += natural.shape[0]
        self.n_ordered_diff += int(od.sum())
        self.n_set_diff += int(sd.sum())
        if self.expect_natural:
            if bool(od.any()):
                raise AssertionError(f"layer {self.L} sid {sid}: forced routing != natural routing in a natural condition")
            return natural
        return forced

    def attach(self, routing_only=False):
        super().attach(routing_only)
        self._ov = block_override(self.model, self.L, self._provider)
        self._ov.__enter__()
        return self

    def detach(self):
        if self._ov is not None:
            self._ov.__exit__(None, None, None)
            self._ov = None
        super().detach()

    def stats(self):
        return dict(n_seq=self.n_seq, n_tokens=self.n_tok, n_ordered_diff=self.n_ordered_diff,
                    n_set_diff=self.n_set_diff, frac_set_diff=self.n_set_diff / max(self.n_tok, 1))


@contextmanager
def forced_e3(E3, tables, row_to_sid, expect_natural, on_hessian=None):
    """Patch e3_allocation so that ONLY the Hessian-building Profiler (collect_hessian=True, no holdout) is a
    ForcedProfiler; the held-out reconstruction Profiler stays the plain original. `tables[L]` = {sid: [T, k]}.
    on_hessian(L, e, acc) is called before each expert is quantized (Hessian capture)."""
    orig_prof, orig_q = E3.Profiler, E3.quantize_from_acc_or_fallback
    stats = {}

    def factory(model, layer, K=4, collect_hessian=True, collect_holdout=False, **kw):
        if collect_hessian and not collect_holdout:
            fp = ForcedProfiler(model, layer, tables[layer], row_to_sid, expect_natural, K=K,
                                collect_hessian=True, collect_holdout=False, **kw)
            stats[layer] = fp
            return fp
        return orig_prof(model, layer, K=K, collect_hessian=collect_hessian, collect_holdout=collect_holdout, **kw)

    def qwrap(model, acc, L, e, *a, **k):
        if on_hessian is not None:
            on_hessian(L, e, acc)
        return orig_q(model, acc, L, e, *a, **k)

    E3.Profiler, E3.quantize_from_acc_or_fallback = factory, qwrap
    try:
        yield stats
    finally:
        E3.Profiler, E3.quantize_from_acc_or_fallback = orig_prof, orig_q


# ---------------------------------------------------------------- Hessian / act-order metrics
def tensor_sha(t):
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def actorder_perm(H):
    """The solver's column order: dead columns (diag 0) get diag 1, then argsort(diag, descending) (gptq.py)."""
    d = torch.diag(H).clone().float()
    d[d == 0] = 1.0
    return torch.argsort(d, descending=True)


def perm_metrics(p, p_ref, group_size=128):
    p, p_ref = torch.as_tensor(p).long(), torch.as_tensor(p_ref).long()
    n = p.numel()
    pos, pos_ref = torch.empty(n, dtype=torch.long), torch.empty(n, dtype=torch.long)
    pos[p] = torch.arange(n)
    pos_ref[p_ref] = torch.arange(n)
    return dict(frac_positions_changed=float((p != p_ref).float().mean()),
                frac_columns_changed_group=float(((pos // group_size) != (pos_ref // group_size)).float().mean()),
                identical=bool(torch.equal(p, p_ref)))


def hessian_metrics(H, H_ref, n=None, n_ref=None):
    """Raw (as accumulated: unnormalized sums) and scale-free (H / tr H; GPTQ with relative damping is invariant to
    the scale of H) relative Frobenius differences, diagonal relative difference, trace ratio."""
    H, H_ref = H.double(), H_ref.double()
    t, tr = float(torch.trace(H)), float(torch.trace(H_ref))
    out = dict(rel_fro_raw=float((H - H_ref).norm() / H_ref.norm()),
               rel_fro_trace_normalized=float((H / t - H_ref / tr).norm() / (H_ref / tr).norm()),
               diag_rel_diff=float((torch.diag(H) - torch.diag(H_ref)).norm() / torch.diag(H_ref).norm()),
               trace_ratio=t / tr, bitwise_identical=bool(torch.equal(H, H_ref)))
    if n is not None and n_ref:
        out.update(tokens=float(n), tokens_ref=float(n_ref),
                   rel_fro_mean_normalized=float((H / n - H_ref / n_ref).norm() / (H_ref / n_ref).norm()))
    return out


def weight_diff(W, W_ref):
    """Elementwise comparison of two {name: tensor} dicts (BF16 effective weights)."""
    n = dif = 0
    d2 = o2 = 0.0
    same = 0
    for k in W_ref:
        a, b = W[k].float(), W_ref[k].float()
        n += b.numel()
        dif += int((a != b).sum())
        d2 += float(((a - b) ** 2).sum())
        o2 += float((b ** 2).sum())
        same += bool(torch.equal(W[k], W_ref[k]))
    return dict(n=n, n_differ=dif, frac_differ=dif / max(n, 1), d2=d2, o2=o2, rel_l2=(d2 / o2) ** .5 if o2 else 0.0,
                n_tensors=len(W_ref), n_tensors_identical=same)


# ---------------------------------------------------------------- factorial contrasts
def factorial_contrasts(y00, y10, y01, y11):
    """y_ar: a = activation (0 = original, 1 = alternative), r = routing (0 = original, 1 = alternative)."""
    return dict(A_effect_R0=y10 - y00, R_effect_X0=y01 - y00, A_effect_R1=y11 - y01, R_effect_X1=y11 - y10,
                interaction=y11 - y10 - y01 + y00, natural_effect=y11 - y00)


# ---------------------------------------------------------------- frozen decision rule
TAU = 1e-4 * 9.237089824789335          # deterministic PPL tolerance used by the grouping studies (0.000924)
DECISION_RULE = {
    "metrics_per_strategy": {
        "h_C": "median over the 16 prespecified diagnostic Hessians (layers 1/8/16/23 x experts 0/30 x gu/down) of the "
               "trace-normalized relative Frobenius difference ||H_C/tr H_C - H_OO/tr H_OO||_F / ||H_OO/tr H_OO||_F",
        "w_C": "relative L2 difference of all 720 effective weight tensors of condition C from OO",
        "dY_C": "PPL(C) - PPL(OO); contrasts A_R0 = Y_AO - Y_OO, R_X0 = Y_OA - Y_OO, I = Y_AA - Y_AO - Y_OA + Y_OO",
        "TAU": "0.000924 = 1e-4 x BF16 PPL (deterministic tolerance of the grouping studies); |x| < TAU is treated as 0",
    },
    "flags_per_strategy": {
        "local_suppression": "h_AO <= 0.5 h_AA AND w_AO <= 0.5 w_AA",
        "local_routing_dominant": "h_OA >= h_AO AND w_OA >= w_AO",
        "ppl_suppression": "|dY_AA| > TAU AND |dY_AO| <= 0.5 |dY_AA|",
        "activation_dominant": "h_AO > h_OA AND w_AO > w_OA AND |dY_AO| > |dY_OA| (with |dY_AO| > TAU)",
        "interaction_large": "|I| > TAU AND |I| >= 0.5 max(|A_R0|, |R_X0|)",
    },
    "ranking_preserved": "|Delta_AO - Delta_OO| <= 0.5 |Delta_AA - Delta_OO|, Delta_C = PPL_FC(C) - PPL_FR(C)",
    "precedence": [
        "RUN-BLOCKED: a natural-condition regression (routing, Hessians, weights, reconstruction) fails, a hybrid "
        "configuration is missing / invalid, or forced routing was active during evaluation",
        "FROZEN-ROUTING-SUPPRESSES-SENSITIVITY: local_suppression AND ppl_suppression in BOTH strategies AND "
        "ranking_preserved AND NOT interaction_large in either strategy",
        "ACTIVATION-VALUES-DOMINATE: activation_dominant in BOTH strategies",
        "LOCAL-ROUTING-EFFECT-WITHOUT-PPL-EXPLANATION: (local_suppression OR local_routing_dominant) in BOTH "
        "strategies, AND (ppl_suppression fails in at least one strategy OR NOT ranking_preserved)",
        "ROUTING-AND-ACTIVATION-INTERACT: otherwise (neither factor alone explains the natural effect, or the "
        "interaction is large); the interaction flags are always reported",
    ],
    "frozen": "fixed before any hybrid (AO / OA) quantization or PPL; never adjusted after outcomes",
}


def flags(h, w, y, tau=TAU):
    """h, w: {cond: metric vs OO} for AO / OA / AA; y: {cond: PPL} for OO / AO / OA / AA."""
    z = lambda x: 0.0 if abs(x) < tau else x
    dY = {c: z(y[c] - y["OO"]) for c in ("AO", "OA", "AA")}
    fc = factorial_contrasts(y["OO"], y["AO"], y["OA"], y["AA"])
    I = z(fc["interaction"])
    return dict(
        local_suppression=bool(h["AO"] <= 0.5 * h["AA"] and w["AO"] <= 0.5 * w["AA"]),
        local_routing_dominant=bool(h["OA"] >= h["AO"] and w["OA"] >= w["AO"]),
        ppl_suppression=bool(abs(dY["AA"]) > 0 and abs(dY["AO"]) <= 0.5 * abs(dY["AA"])),
        activation_dominant=bool(h["AO"] > h["OA"] and w["AO"] > w["OA"] and abs(dY["AO"]) > abs(dY["OA"])
                                 and abs(dY["AO"]) > 0),
        interaction_large=bool(abs(I) > 0 and abs(I) >= 0.5 * max(abs(dY["AO"]), abs(dY["OA"]))),
        dY=dY, contrasts=fc)


def ranking_preserved(delta):
    return bool(abs(delta["AO"] - delta["OO"]) <= 0.5 * abs(delta["AA"] - delta["OO"]))


def decide(per_strategy_flags, delta, blocked=False):
    if blocked:
        return "RUN-BLOCKED"
    f = list(per_strategy_flags.values())
    rp = ranking_preserved(delta)
    if all(x["local_suppression"] and x["ppl_suppression"] for x in f) and rp and not any(x["interaction_large"] for x in f):
        return "FROZEN-ROUTING-SUPPRESSES-SENSITIVITY"
    if all(x["activation_dominant"] for x in f):
        return "ACTIVATION-VALUES-DOMINATE"
    if all(x["local_suppression"] or x["local_routing_dominant"] for x in f) and \
            (not all(x["ppl_suppression"] for x in f) or not rp):
        return "LOCAL-ROUTING-EFFECT-WITHOUT-PPL-EXPLANATION"
    return "ROUTING-AND-ACTIVATION-INTERACT"


def obj_sha(o):
    import json
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def summarize(values):
    v = np.asarray(values, float)
    return dict(median=float(np.median(v)), mean=float(v.mean()), min=float(v.min()), max=float(v.max()), n=int(v.size))
