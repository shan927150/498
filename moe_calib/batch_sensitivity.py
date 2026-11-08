"""Calibration batch-composition sensitivity: pure helpers (no model, no I/O).

The manipulation under study is ONLY which sequences share a BF16 forward batch
while the calibration layer-input cache is built. Everything here is either a
deterministic batch plan, the reassembly of per-sequence activations by original
sequence id, or a comparison metric between two variants aligned by sequence id
and token position (never by batch position).
"""
from __future__ import annotations

import numpy as np
import torch

VARIANTS = ("ORIGINAL", "COMPACT", "SHUFFLED")
SHUFFLE_SEED = 20261110


# ---------------------------------------------------------------- batch plans
def original_batches(rows, B, n_total):
    """cache_inputs.py: consecutive batches of B original rows from row 0. Every needed row is
    forwarded inside its original batch (with its original batch-mates, incl. rows not needed)."""
    groups = sorted({int(r) // B for r in rows})
    return [[g * B + j for j in range(B) if g * B + j < n_total] for g in groups]


def compact_batches(rows, B):
    """v1 (failed) construction: the needed rows, in the given order, regrouped in batches of B
    (the final batch may be smaller)."""
    rows = list(rows)
    return [rows[i:i + B] for i in range(0, len(rows), B)]


def shuffled_batches(rows, B, seed=SHUFFLE_SEED):
    """One deterministic shuffle of the SAME rows, partitioned exactly like compact_batches."""
    perm = np.random.default_rng(seed).permutation(len(rows))
    return compact_batches([int(rows[i]) for i in perm], B)


def batch_sizes(batches):
    return [len(b) for b in batches]


def reassembly_index(batches, rows):
    """For each needed row (in `rows` order): (batch index, position in batch)."""
    where = {r: (bi, j) for bi, b in enumerate(batches) for j, r in enumerate(b)}
    missing = [r for r in rows if r not in where]
    if missing:
        raise ValueError(f"rows {missing[:5]} not in any batch")
    return [where[r] for r in rows]


def reassemble(batch_tensors, index):
    """Stack per-sequence tensors back into `rows` order (original sequence ids)."""
    return torch.stack([batch_tensors[bi][j] for bi, j in index])


def plan_manifest(name, batches, rows, construction):
    return dict(variant=name, construction=construction, batch_size_pattern=batch_sizes(batches),
                n_batches=len(batches), n_rows_forwarded=sum(map(len, batches)),
                execution_order="batches executed in list order", batches=[list(map(int, b)) for b in batches],
                needed_rows_in_cache_order=list(map(int, rows)))


# ---------------------------------------------------------------- activation drift
@torch.no_grad()
def activation_drift(X, R, eps=1e-12):
    """X: variant, R: reference; both [N, T, d] aligned by sequence id and token position."""
    assert X.shape == R.shape
    Xf, Rf = X.float(), R.float()
    D = Xf - Rf
    tok = D.norm(dim=-1) / (Rf.norm(dim=-1) + eps)                       # [N, T]
    q = torch.quantile(tok.flatten().double(), torch.tensor([.5, .9, .99], dtype=torch.float64))
    return dict(relative_l2=float(D.norm() / (Rf.norm() + eps)),
                max_abs=float(D.abs().max()), mean_abs=float(D.abs().mean()),
                token_rel_l2_p50=float(q[0]), token_rel_l2_p90=float(q[1]), token_rel_l2_p99=float(q[2]),
                frac_elements_exactly_equal=float((X == R).float().mean()),
                n_sequences_with_differences=int((X != R).flatten(1).any(1).sum()), n_sequences=int(X.shape[0]),
                nan_count=int(torch.isnan(Xf).sum()), inf_count=int(torch.isinf(Xf).sum()))


# ---------------------------------------------------------------- routing
def topk_route(logits, k):
    """The model's op: softmax(float) then topk(k). Returns (probs [T,E], ordered top-k ids [T,k])."""
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    return probs, torch.topk(probs, k, dim=-1).indices


def routing_compare(sel_v, sel_o, probs_v=None, probs_o=None, n_experts=None):
    """sel_*: [n_tok, k] ordered top-k expert ids aligned by token. Separates a re-ordering of the
    same experts (set identical, order not) from a change of the selected set."""
    sel_v, sel_o = torch.as_tensor(sel_v).long(), torch.as_tensor(sel_o).long()
    n, k = sel_o.shape
    E = n_experts or int(max(sel_v.max(), sel_o.max())) + 1
    ordered = (sel_v == sel_o).all(1)
    mv = torch.zeros(n, E, dtype=torch.bool).scatter_(1, sel_v, True)
    mo = torch.zeros(n, E, dtype=torch.bool).scatter_(1, sel_o, True)
    overlap = (mv & mo).sum(1).float()
    same_set = overlap == k
    cv, co = mv.sum(0).double(), mo.sum(0).double()
    out = dict(n_tokens=n, frac_ordered_identical=float(ordered.float().mean()),
               frac_set_identical=float(same_set.float().mean()),
               frac_reordered_same_set=float((same_set & ~ordered).float().mean()),
               frac_any_expert_changed=float((~same_set).float().mean()),
               mean_topk_overlap=float(overlap.mean() / k),
               per_expert_count_abs_diff=(cv - co).abs().tolist(),
               max_expert_count_abs_diff=float((cv - co).abs().max()),
               routing_distribution_l1=float((cv / cv.sum() - co / co.sum()).abs().sum()))
    if probs_v is not None:
        pv, po = probs_v.float(), probs_o.float()
        out["mean_abs_prob_diff"] = float((pv - po).abs().mean())
        tv, to = torch.topk(pv, k, -1).values, torch.topk(po, k, -1).values
        out["mean_abs_topk_score_diff"] = float((tv - to).abs().mean())
        out["max_abs_topk_score_diff"] = float((tv - to).abs().max())
    return out


def expert_token_mask(sel, e):
    """bool [n_tok]: tokens routed to expert e."""
    return (torch.as_tensor(sel) == e).any(-1)


# ---------------------------------------------------------------- Hessians
@torch.no_grad()
def hessian_compare(Hv, Ho, group_size=128):
    """Alternative vs reference input Hessian of one expert/projection (summed over blocks).
    Also: does the act-order permutation (argsort of diag(H), descending; frozen GPTQ) or the
    resulting column -> quantization-group assignment change?"""
    Hv, Ho = Hv.double(), Ho.double()
    dv, do = torch.diagonal(Hv), torch.diagonal(Ho)
    out = dict(rel_fro=float((Hv - Ho).norm() / Ho.norm().clamp_min(1e-300)),
               diag_rel=float((dv - do).norm() / do.norm().clamp_min(1e-300)),
               trace_ratio=float(dv.sum() / do.sum()) if float(do.sum()) != 0 else float("nan"),
               max_abs=float((Hv - Ho).abs().max()))
    pv, po = actorder_perm(Hv.float()), actorder_perm(Ho.float())
    gv, go = torch.empty_like(pv), torch.empty_like(po)
    gv[pv] = torch.arange(len(pv)) // group_size
    go[po] = torch.arange(len(po)) // group_size
    out.update(actorder_perm_identical=bool(torch.equal(pv, po)),
               frac_actorder_positions_changed=float((pv != po).float().mean()),
               frac_columns_group_changed=float((gv != go).float().mean()))
    return out


def actorder_perm(H):
    """Exactly gptq_quantize's permutation (dead columns get diag 1 first)."""
    d = torch.diagonal(H).clone().float()
    d[d == 0] = 1.0
    return torch.argsort(d, descending=True)


@torch.no_grad()
def gram(X):
    """X [t, d] -> fp32 X^T X (the same accumulation op as BlockHessianAccumulator.add, one shot)."""
    Xf = X.float()
    return Xf.T @ Xf


# ---------------------------------------------------------------- weights
@torch.no_grad()
def weight_compare(Wv, Wo):
    Wv, Wo = Wv.float(), Wo.float()
    D = Wv - Wo
    return dict(frac_differ=float((Wv != Wo).float().mean()), rel_l2=float(D.norm() / Wo.norm().clamp_min(1e-30)),
                max_abs=float(D.abs().max()))


# ---------------------------------------------------------------- PPL / decision
def strategy_deltas(ppl):
    """ppl[(strategy, variant)] -> Delta(variant) = PPL(FC, v) - PPL(FR, v) and the
    strategy-by-batching interaction Delta(v) - Delta(ORIGINAL)."""
    vs = [v for v in VARIANTS if ("FR", v) in ppl and ("FC", v) in ppl]
    d = {v: ppl[("FC", v)] - ppl[("FR", v)] for v in vs}
    inter = {v: d[v] - d["ORIGINAL"] for v in vs}
    rng = {s: max(ppl[(s, v)] for v in vs) - min(ppl[(s, v)] for v in vs) for s in ("FR", "FC")}
    return dict(delta=d, interaction=inter, strategy_range=rng,
                delta_range=max(d.values()) - min(d.values()),
                n_fc_worse=sum(1 for v in vs if d[v] > 0),
                ranking={v: ("FC worse" if d[v] > 0 else "FR worse" if d[v] < 0 else "tie") for v in vs},
                ranking_reverses=any(np.sign(d[v]) != np.sign(d["ORIGINAL"]) for v in vs),
                max_shift={s: max(abs(ppl[(s, v)] - ppl[(s, "ORIGINAL")]) for v in vs) for s in ("FR", "FC")})


DECISION_RULE = {
    "quantities": "Delta_O = PPL(FC,ORIGINAL) - PPL(FR,ORIGINAL); for the valid alternative variants v "
                  "(COMPACT, SHUFFLED): shift = max_{s,v} |PPL(s,v) - PPL(s,ORIGINAL)|, "
                  "I = max_v |Delta(v) - Delta_O|, reversal = sign(Delta(v)) != sign(Delta_O) for any v; "
                  "tau = 1e-4 x PPL_BF16",
    "RUN-BLOCKED": "a configuration is missing, the ORIGINAL four-layer regression does not reproduce the "
                   "partial-PPL weights and PPL exactly, an installation / contamination check fails, or the "
                   "ORIGINAL execution-order negative control fails without explanation",
    "BATCH-VARIANT-INVALID": "an alternative variant differs from ORIGINAL in token ids, position ids, attention "
                             "mask semantics, model weights, dtype or inference configuration",
    "BATCH-SENSITIVITY-DOMINANT": "shift >= 0.5 |Delta_O| OR I >= 0.5 |Delta_O| OR any reversal",
    "BATCH-SENSITIVITY-NEGLIGIBLE": "shift <= max(tau, 0.1 |Delta_O|) AND I <= max(tau, 0.1 |Delta_O|)",
    "BATCH-SENSITIVITY-MODERATE": "otherwise (measurable, below half of |Delta_O|, no reversal)",
}


def decide(ppl, bf16, valid=True, blocked=False):
    if blocked:
        return "RUN-BLOCKED"
    if not valid:
        return "BATCH-VARIANT-INVALID"
    s = strategy_deltas(ppl)
    d0 = abs(s["delta"]["ORIGINAL"])
    shift = max(s["max_shift"].values())
    inter = max(abs(v) for k, v in s["interaction"].items() if k != "ORIGINAL")
    tau = 1e-4 * bf16
    if shift >= 0.5 * d0 or inter >= 0.5 * d0 or s["ranking_reverses"]:
        return "BATCH-SENSITIVITY-DOMINANT"
    floor = max(tau, 0.1 * d0)
    if shift <= floor and inter <= floor:
        return "BATCH-SENSITIVITY-NEGLIGIBLE"
    return "BATCH-SENSITIVITY-MODERATE"
