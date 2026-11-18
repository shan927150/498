"""Single-layer profiler over CACHED layer inputs (proposal §3, component 1).

Each run replays ONE decoder layer (attention + MoE block) on cached input
hidden states, one sequence at a time (batch=1 -> exact sequence attribution),
with hooks on that layer's router and experts. Because characterization is
non-sequential (every layer sees full-precision inputs), layers are
independent and can be profiled one at a time: GPU memory = 1 layer
(~1.2 GB) + that layer's block Hessians (~1.5 GB per block).

Collected:
  * routing profile c_{s,e} (seq_counts[s][(L,e)])
  * router uncertainty per expert: mean top-k margin p_k - p_{k+1}, entropy
  * block Hessians (BlockHessianAccumulator), optional
  * held-out activations per expert, grouped BY SEQUENCE, optional
Natural routing is preserved: statistics use tokens actually routed to e.
"""
from __future__ import annotations
import time

import torch

from .hessian import BlockHessianAccumulator
from . import model_utils as MU


class Profiler:
    WEIGHTINGS = ("none", "router_affinity", "unit_affinity")

    def __init__(self, model, layer: int, K: int = 4,
                 collect_hessian: bool = True, collect_holdout: bool = False,
                 holdout_tokens_per_seq: int = 128, hessian_weighting: str = "none",
                 collect_affinity: bool = False, verify_alignment: bool = False, affinity_power: int = 1):
        """hessian_weighting (default "none" = standard GPTQ, code path unchanged):
          "router_affinity": AGQ (MoEQuant Eq. 19) - every routed row of expert e is weighted by its router
              affinity c = softmax(router logits, fp32)[selected top-k] (no renormalization: the checkpoint has
              norm_topk_prob=False), aligned to the expert's dispatch rows exactly as the model builds them.
          "unit_affinity": regression control - the same weighted code path with every c forced to 1.
        affinity_power q (default 1 = published AGQ, code path unchanged): with "router_affinity", the row weight is
              c^q (q in {1, 2}; q = 2 is the EXPERIMENTAL c^2 variant, H = X^T diag(c^2) X). q = 0 is standard GPTQ
              and is requested with hessian_weighting="none".
        collect_affinity: keep (seq, token, slot, c) per (layer, expert) for statistics.
        verify_alignment: assert each expert's gate_proj input rows equal the router input at the recorded
              token positions (bitwise) and that down_proj rows are the same tokens (row counts)."""
        if hessian_weighting not in self.WEIGHTINGS:
            raise ValueError(f"hessian_weighting must be one of {self.WEIGHTINGS}")
        self.model, self.L, self.K = model, layer, K
        if affinity_power not in (1, 2):
            raise ValueError("affinity_power must be 1 (AGQ) or 2 (experimental c^2); use hessian_weighting='none' "
                             "for q = 0")
        if affinity_power != 1 and hessian_weighting != "router_affinity":
            raise ValueError("affinity_power applies to hessian_weighting='router_affinity' only")
        self.affinity_power = affinity_power
        self.hessian_weighting = hessian_weighting
        self.collect_affinity, self.verify_alignment = collect_affinity, verify_alignment
        self._need_rows = hessian_weighting != "none" or collect_affinity or verify_alignment
        self._cur_rows: dict[int, tuple] = {}
        self._gate_in = None
        self.affinity: dict[tuple, list] = {}
        self.n_alignment_checks = 0
        self.E, self.topk = MU.num_experts(model), MU.top_k(model)
        d_gu, d_down = MU.expert_dims(model)
        self.acc = (BlockHessianAccumulator([layer], self.E, d_gu, d_down, K,
                                            device=MU.GPU, store_device=MU.GPU)
                    if collect_hessian else None)
        self.collect_holdout = collect_holdout
        self.ho_cap = holdout_tokens_per_seq
        self.holdout: dict[tuple, list] = {}
        self.seq_counts: dict[int, dict[tuple, int]] = {}
        self.margin_sum = torch.zeros(self.E, dtype=torch.float64)
        self.entropy_sum = torch.zeros(self.E, dtype=torch.float64)
        self.tok_sum = torch.zeros(self.E, dtype=torch.float64)
        self.cur_seq, self.cur_block, self.cur_weight = -1, 0, 1.0
        self._hooks = []
        self.seconds = 0.0

    # ------------------------------------------------------------- hooks
    def _gate_hook(self, module, inputs, output):
        # Selected experts: the model's exact op (Qwen2MoeSparseMoeBlock:
        # softmax(dim=1, dtype=float) then topk(k)). Router logits are bf16, so
        # exact ties are common and topk(k+1)[:k] may break them differently
        # from topk(k). The k+1 call is used ONLY for the margin values.
        probs = torch.softmax(output, dim=-1, dtype=torch.float)   # [T, E]
        if self._need_rows:
            vals, sel = torch.topk(probs, self.topk, dim=-1)
            self._record_rows(vals, sel)
            if self.verify_alignment:
                self._gate_in = inputs[0].reshape(-1, inputs[0].shape[-1])
        else:
            sel = torch.topk(probs, self.topk, dim=-1).indices
        topv = torch.topk(probs, self.topk + 1, dim=-1).values
        margin = topv[:, self.topk - 1] - topv[:, self.topk]
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)
        sc = self.seq_counts.setdefault(self.cur_seq, {})
        for e in sel.unique().tolist():
            idx = (sel == e).any(dim=-1).nonzero(as_tuple=True)[0]
            cnt = int(idx.numel())
            sc[(self.L, e)] = cnt
            self.tok_sum[e] += cnt
            self.margin_sum[e] += float(margin[idx].sum())
            self.entropy_sum[e] += float(ent[idx].sum())

    def _expert_hook(self, e, proj):
        key = (self.L, e, proj)

        def fn(module, args):
            X = args[0].reshape(-1, args[0].shape[-1])
            if self._need_rows:
                self._check_rows(e, proj, X)
            if self.acc is not None:
                if self.hessian_weighting == "none":
                    self.acc.add(key, X, self.cur_block, self.cur_weight)
                elif X.shape[0] > 0:
                    c = self._cur_rows[e][2]
                    if self.hessian_weighting == "unit_affinity":
                        c = torch.ones_like(c)
                    elif self.affinity_power != 1:
                        c = c.pow(self.affinity_power)          # weight c^q -> rows scaled by c^(q/2)
                    self.acc.add(key, X, self.cur_block, self.cur_weight, row_weight=c)
            if self.collect_holdout and X.shape[0] > 0:
                self.holdout.setdefault(key, []).append(
                    (self.cur_seq, X[:self.ho_cap].detach().to("cpu", torch.float16)))
        return fn

    # ------------------------------------------------- AGQ row bookkeeping
    @torch.no_grad()
    def _record_rows(self, vals, sel):
        """Per expert e: (token positions top_x, top-k slot idx, affinity c) in the EXACT row order of the
        model's dispatch (Qwen2MoeSparseMoeBlock: one_hot(sel).permute(2,1,0); idx, top_x = where(mask[e]);
        rows = hidden[top_x]) - slot-major, NOT token order. c = vals[top_x, idx] (fp32)."""
        mask = torch.nn.functional.one_hot(sel, num_classes=self.E).permute(2, 1, 0)
        self._cur_rows = {}
        for e in range(self.E):
            idx, top_x = torch.where(mask[e])
            if top_x.numel():
                self._cur_rows[e] = (top_x, idx, vals[top_x, idx])
                if self.collect_affinity:
                    self.affinity.setdefault((self.L, e), []).append(
                        (self.cur_seq, top_x.cpu(), idx.cpu(), vals[top_x, idx].detach().cpu()))

    @torch.no_grad()
    def _check_rows(self, e, proj, X):
        rows = self._cur_rows.get(e)
        if rows is None and X.shape[0] == 0:          # the model calls unrouted experts with an empty batch
            return
        if rows is None or rows[0].shape[0] != X.shape[0]:
            raise RuntimeError(f"AGQ alignment: layer {self.L} expert {e} {proj}: {X.shape[0]} activation rows vs "
                               f"{0 if rows is None else rows[0].shape[0]} routed (token, slot) pairs")
        if self.verify_alignment and proj == "gu":
            if not torch.equal(X, self._gate_in[rows[0]].to(X.dtype)):
                raise RuntimeError(f"AGQ alignment: layer {self.L} expert {e}: gate_proj rows != router input "
                                   "at recorded token positions")
            self.n_alignment_checks += 1

    def attach(self, routing_only: bool = False):
        blk = MU.moe_block(self.model, self.L)
        self._hooks.append(blk.gate.register_forward_hook(self._gate_hook))
        if not routing_only:
            for e in range(self.E):
                ex = blk.experts[e]
                self._hooks.append(ex.gate_proj.register_forward_pre_hook(
                    self._expert_hook(e, "gu")))
                self._hooks.append(ex.down_proj.register_forward_pre_hook(
                    self._expert_hook(e, "down")))
        return self

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ------------------------------------------------------------ driving
    @torch.no_grad()
    def run(self, cache: torch.Tensor, seq_ids: list[int],
            blocks: list[int] | None = None, weights: list[float] | None = None,
            routing_only: bool = False):
        """cache: [N, T, d] layer-input tensor (mmap ok). seq_ids index rows.
        blocks[i]: Hessian block of seq i (default contiguous K-way split)."""
        n = len(seq_ids)
        if blocks is None:
            blocks = [(i * self.K) // max(n, 1) for i in range(n)]
        if weights is None:
            weights = [1.0] * n
        self.attach(routing_only=routing_only)
        t0 = time.time()
        try:
            with MU.layer_on_gpu(self.model, self.L):
                for i, sid in enumerate(seq_ids):
                    self.cur_seq, self.cur_block, self.cur_weight = sid, blocks[i], weights[i]
                    MU.run_layer(self.model, self.L,
                                 cache[sid:sid + 1].to(MU.GPU, non_blocking=True))
        finally:
            self.detach()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds += time.time() - t0
        return self

    def free(self):
        self.acc = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ signals
    def router_signals(self, e: int):
        t = float(self.tok_sum[e])
        if t == 0:
            return float("nan"), float("nan")
        return float(self.margin_sum[e]) / t, float(self.entropy_sum[e]) / t

    def usage_freq(self, e: int):
        tot = float(self.tok_sum.sum())
        return 0.0 if tot == 0 else float(self.tok_sum[e]) / tot
