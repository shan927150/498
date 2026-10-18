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
    def __init__(self, model, layer: int, K: int = 4,
                 collect_hessian: bool = True, collect_holdout: bool = False,
                 holdout_tokens_per_seq: int = 128):
        self.model, self.L, self.K = model, layer, K
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
            if self.acc is not None:
                self.acc.add(key, X, self.cur_block, self.cur_weight)
            if self.collect_holdout and X.shape[0] > 0:
                self.holdout.setdefault(key, []).append(
                    (self.cur_seq, X[:self.ho_cap].detach().to("cpu", torch.float16)))
        return fn

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
