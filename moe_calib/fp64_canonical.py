"""FP64-canonical Hessian accumulation helpers (numerical-reproducibility study).

The canonical Hessian of (layer, expert, projection) is: the ORIGINAL FP32 per-sequence contributions
outer_s = Xf.T @ Xf (one FP32 GEMM per sequence, exactly as BlockHessianAccumulator.add), accumulated per ORIGINAL
statistical block in FP64, the K blocks summed in FP64, then cast ONCE to FP32 and passed to the unchanged GPTQ solver.
Only accumulation precision changes; activations, GEMMs, routing, act-order, damping and the solver do not.

LoadedHessianAccumulator stands in for BlockHessianAccumulator inside the unchanged Profiler / E3 quantization path:
it counts tokens exactly as the original (per block, per key) but serves a stored canonical FP32 Hessian from H_sum.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch


class LoadedHessianAccumulator:
    """Drop-in for moe_calib.hessian.BlockHessianAccumulator (constructor signature, add / H_sum / tokens / raw_n / n).
    STORE: {"L.e.pj": FP32 tensor} set by `serve_hessians`. H_sum only answers the full-block request the E3 path makes
    (all K blocks); any partial request raises, so no statistic can silently use a non-canonical value."""
    STORE: dict | None = None

    def __init__(self, layers, n_experts, d_gu, d_down, K=4, device="cuda", store_device="cuda"):
        if LoadedHessianAccumulator.STORE is None:
            raise RuntimeError("no canonical Hessians loaded")
        self.K, self.device = K, device
        self.n, self.raw_n = {}, {}
        for L in layers:
            for e in range(n_experts):
                for pj in ("gu", "down"):
                    self.n[(L, e, pj)] = torch.zeros(K, dtype=torch.float64)
                    self.raw_n[(L, e, pj)] = torch.zeros(K, dtype=torch.float64)
        self.n_served = 0

    def add(self, key, X, block, weight=1.0, row_weight=None):
        if row_weight is not None or weight != 1.0:
            raise ValueError("canonical path supports unit-weight standard GPTQ only")
        if X.numel() == 0:
            return
        self.n[key][block] += X.shape[0]
        self.raw_n[key][block] += X.shape[0]

    def H_sum(self, key, blocks):
        if list(blocks) != list(range(self.K)):
            raise ValueError(f"canonical Hessian requested for a partial block set {blocks}")
        self.n_served += 1
        L, e, pj = key
        return LoadedHessianAccumulator.STORE[f"{L}.{e}.{pj}"].to(self.device)

    def tokens(self, key, blocks=None):
        b = blocks if blocks is not None else list(range(self.K))
        return float(self.raw_n[key][b].sum())


@contextmanager
def serve_hessians(store: dict):
    """Within the block every Profiler(collect_hessian=True) gets a LoadedHessianAccumulator serving `store`."""
    from moe_calib import profiler as PR
    orig = PR.BlockHessianAccumulator
    LoadedHessianAccumulator.STORE = store
    PR.BlockHessianAccumulator = LoadedHessianAccumulator
    try:
        yield
    finally:
        PR.BlockHessianAccumulator = orig
        LoadedHessianAccumulator.STORE = None


def canonical_cast(H64_blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[K, d, d] FP64 block sums -> (FP64 total, FP32 cast once)."""
    assert H64_blocks.dtype == torch.float64
    H64 = H64_blocks.sum(dim=0)
    return H64, H64.to(torch.float32)


# ----------------------------------------------------------------------------- statistics (pure)
def sd(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def hierarchical_bootstrap(d: dict, n_boot=20000, seed=0):
    """d: {seed: [paired differences over groupings]}. Resample seeds with replacement, then groupings within each
    chosen seed; statistic = mean over all resampled values. Returns (2.5 %, 97.5 %)."""
    rng = np.random.default_rng(seed)
    keys = sorted(d)
    out = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for k in rng.choice(keys, len(keys), replace=True):
            v = np.asarray(d[k], float)
            vals.append(v[rng.integers(0, len(v), len(v))])
        out[b] = np.concatenate(vals).mean()
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def pooled_within_sd(d: dict):
    """sqrt of the df-weighted mean within-group variance."""
    num = sum((len(v) - 1) * np.var(v, ddof=1) for v in d.values() if len(v) > 1)
    den = sum(len(v) - 1 for v in d.values() if len(v) > 1)
    return float(np.sqrt(num / den))


def ppl_from_nll(sum_nll, n_tokens):
    return float(np.exp(sum_nll / n_tokens))


def classify_canonical(blocked: bool, order_invariant: bool, reproducible: bool) -> str:
    if blocked:
        return "CANONICAL-BLOCKED"
    if not order_invariant or not reproducible:
        return "CANONICAL-NOT-ORDER-INVARIANT"
    return "CANONICAL-VALIDATED"


def classify_execution(sign_changes: bool, magnitude_ratio: float, eval_deterministic: bool,
                       material_ratio: float = 2.0) -> str:
    """MATERIAL if matched execution variants change the sign of FC-FR, or the spread of FC-FR across variants
    (max - min) is at least `material_ratio` x |reference difference|, provided identical weights evaluate identically."""
    if eval_deterministic and (sign_changes or magnitude_ratio >= material_ratio):
        return "EXECUTION-VARIABILITY-MATERIAL"
    return "EXECUTION-VARIABILITY-LIMITED"
