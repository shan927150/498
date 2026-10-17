"""Block-partitioned Hessian accumulation.

Design note / provenance
------------------------
GPTQ's layer Hessian H = sum_t x_t x_t^T is an (unnormalized) *empirical
expectation* over calibration activations.  Canevet, Jose & Fleuret,
"Importance Sampling Tree for Large-scale Empirical Expectation" (ICML 2016)
make two points we reuse:

 (1) you may acquire the samples of an empirical expectation NON-uniformly
     (prioritizing "heavy" / informative ones) and still keep the estimate
     unbiased by importance re-weighting;
 (2) keeping *partial sums* of the expectation lets you re-estimate it under
     different sample subsets without re-touching the data.

We implement (2) as K per-expert Hessian blocks: block k holds
H_k = sum_{t in block k} w_t x_t x_t^T and its token count n_k.  From the
blocks we get, with NO extra forward passes:
    H(n0)        = sum of blocks 0..K/2-1         (E1: first-half budget)
    H(2n0)       = sum of all blocks              (E1)
    drift        = compare normalized prefix means (E2 signal, as in proposal)
    rel. std err = between-block variability -> t-style CI width (E2 signal)
Point (1) is used in E3: sequences selected with inclusion prob pi_s are
accumulated with weight w = 1/pi_s (Horvitz-Thompson), i.e. activations are
scaled by sqrt(w) — the solver itself stays standard GPTQ.

Keys are (layer, expert, proj) with proj in {"gu", "down"}; gate_proj and
up_proj share "gu" since they see the same input (halves memory, and GPTQ
only needs the INPUT Hessian).
"""
from __future__ import annotations
import torch

Key = tuple[int, int, str]   # (layer, expert, "gu"|"down")


class BlockHessianAccumulator:
    def __init__(self, layers: list[int], n_experts: int, d_gu: int, d_down: int,
                 K: int = 8, device: str = "cuda", store_device: str = "cuda"):
        self.K = K
        self.device = device
        self.store_device = store_device
        self.H: dict[Key, torch.Tensor] = {}   # [K, d, d] fp32
        self.n: dict[Key, torch.Tensor] = {}   # [K] weighted token counts
        self.raw_n: dict[Key, torch.Tensor] = {}  # [K] unweighted token counts
        for L in layers:
            for e in range(n_experts):
                for proj, d in (("gu", d_gu), ("down", d_down)):
                    k = (L, e, proj)
                    self.H[k] = torch.zeros(K, d, d, dtype=torch.float32,
                                            device=store_device)
                    self.n[k] = torch.zeros(K, dtype=torch.float64)
                    self.raw_n[k] = torch.zeros(K, dtype=torch.float64)

    @torch.no_grad()
    def add(self, key: Key, X: torch.Tensor, block: int, weight: float = 1.0):
        """X: [t, d] activations routed to this expert (any float dtype)."""
        if X.numel() == 0:
            return
        # compute X^T X on whatever GPU this layer lives on (multi-GPU safe),
        # then move the d x d result to the storage device
        Xf = X.to(torch.float32) if X.is_cuda else X.to(self.device, torch.float32)
        outer = Xf.T @ Xf                       # [d, d]
        if weight != 1.0:
            outer *= weight
        self.H[key][block] += outer.to(self.store_device)
        self.n[key][block] += weight * X.shape[0]
        self.raw_n[key][block] += X.shape[0]

    # ---------- readouts (all O(K d^2), no new forwards) ----------

    def H_sum(self, key: Key, blocks: list[int]) -> torch.Tensor:
        return self.H[key][blocks].sum(dim=0)

    def tokens(self, key: Key, blocks: list[int] | None = None) -> float:
        b = blocks if blocks is not None else list(range(self.K))
        return float(self.raw_n[key][b].sum())

    def normalized_block_means(self, key: Key, blocks: list[int]):
        """H_bar_k = H_k / n_k for non-empty blocks (token-count normalized,
        per the proposal, so stats don't just reflect cumulative growth)."""
        Hs, ns = self.H[key], self.n[key]
        out = []
        for k in blocks:
            if ns[k] > 0:
                out.append(Hs[k].float() / float(ns[k]))
        return out

    def rel_std_err(self, key: Key, blocks: list[int]) -> float:
        """Between-block relative standard error of the normalized Hessian:
        relSE = ||std_k(H_bar_k)||_F / (sqrt(K_eff) * ||mean_k(H_bar_k)||_F).
        This is the t-statistic-style CI half-width (up to z_alpha) that E2
        tests as a sufficiency signal and E3 turns into token needs."""
        ms = self.normalized_block_means(key, blocks)
        if len(ms) < 2:
            return float("nan")
        Mstack = torch.stack(ms)                     # [K_eff, d, d]
        mean = Mstack.mean(dim=0)
        std = Mstack.std(dim=0, unbiased=True)
        denom = torch.linalg.norm(mean)
        if denom == 0:
            return float("nan")
        return float(torch.linalg.norm(std) / (len(ms) ** 0.5 * denom))

    def drift(self, key: Key, blocks: list[int]) -> float:
        """Prefix drift within `blocks`: || H_bar(first half) - H_bar(all) ||_F
        / ||H_bar(all)||_F — the proposal's original signal, on normalized
        moments."""
        half = blocks[: max(1, len(blocks) // 2)]
        def nmean(bl):
            ms = self.normalized_block_means(key, bl)
            return None if not ms else torch.stack(ms).mean(dim=0)
        a, b = nmean(half), nmean(blocks)
        if a is None or b is None:
            return float("nan")
        d = torch.linalg.norm(b)
        return float("nan") if d == 0 else float(torch.linalg.norm(a - b) / d)
