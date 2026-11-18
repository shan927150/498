"""Layer-pooled Hessian shrinkage (feasibility MVP): pure estimators, an opt-in pooled-Hessian hook, covariance
metrics and the frozen Stage 1 / Stage 2 decision rules. The GPTQ solver is NOT touched: a shrunk Hessian is passed to
the unchanged `gptq_quantize`, which then applies its usual relative damping.

Estimators for expert e with raw routed Hessian H_e (n_e routed tokens) and layer-pooled Hessian H_L (N tokens, every
MoE-input token once):
    Hbar_e = H_e / n_e,  Hbar_L = H_L / N
    pooled   : n_e [(1-a) Hbar_e + a P_e],   P_e = Hbar_L tr(Hbar_e) / tr(Hbar_L)   (trace-matched, PRIMARY)
    identity : n_e [(1-a) Hbar_e + a I*],    I*  = tr(Hbar_e) / d * I               (trace-matched control)
    untraced : n_e [(1-a) Hbar_e + a Hbar_L]                                         (diagnostic only, no GPTQ)
alpha = 0 returns the raw H_e itself (a clone), never a renormalized copy, so standard GPTQ is reproduced bitwise.
"""
from __future__ import annotations

import numpy as np
import torch

ALPHAS = (0.0, 0.1, 0.3, 0.5)
BUDGETS = (16, 32, 64, 128)
PRIMARY_BUDGETS = (16, 32)
TARGETS = ("pooled", "identity", "untraced")
TOPK = 16
STRATA = dict(low=(0, 1), median=(29, 30), high=(58, 59))   # ascending B32-exposure ranks within a layer (60 experts)


# ----------------------------------------------------------------------------- expert selection
def select_experts(counts_b32, strata=STRATA) -> dict:
    """counts_b32: [60] routed-token counts at B32 for one layer. Ascending sort with ties broken by expert id
    (lexsort). Returns {stratum: [expert ids]}; depends on exposure only."""
    c = np.asarray(counts_b32)
    order = np.lexsort((np.arange(len(c)), c))
    return {s: [int(order[r]) for r in ranks] for s, ranks in strata.items()}


# ----------------------------------------------------------------------------- estimators
@torch.no_grad()
def trace_matched_prior(Hbar_e: torch.Tensor, Hbar_L: torch.Tensor) -> torch.Tensor:
    return Hbar_L * (torch.trace(Hbar_e) / torch.trace(Hbar_L))


@torch.no_grad()
def shrink(H_e: torch.Tensor, n_e: float, H_L: torch.Tensor, N: float, alpha: float, target: str = "pooled",
           out_dtype=torch.float32) -> torch.Tensor:
    """Shrunk Hessian in the original SUM convention (scale n_e). Mixing is done in float64, the result is cast to
    out_dtype. alpha == 0 -> H_e.clone() (no normalization round trip)."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if alpha == 0.0:
        return H_e.clone()
    if n_e <= 0:
        raise ValueError("shrinkage of an empty-exposure expert is undefined (selected experts have n_e > 0)")
    He = H_e.double() / n_e
    if target == "identity":
        T = torch.eye(He.shape[0], dtype=He.dtype, device=He.device) * (torch.trace(He) / He.shape[0])
    else:
        HL = H_L.to(He.device, torch.float64) / N
        T = trace_matched_prior(He, HL) if target == "pooled" else HL
    return (n_e * ((1.0 - alpha) * He + alpha * T)).to(out_dtype)


# ----------------------------------------------------------------------------- opt-in pooled accumulation
class PooledHessian:
    """Opt-in diagnostic: forward_pre_hook on the MoE router gate (`blk.gate`). Its input is exactly the normalized
    MoE input that the routed experts' gate/up projections receive. Every token is added once (no top-k duplication)
    into a chronological block buffer [K, d, d] fp32 using the same ops as BlockHessianAccumulator.add."""

    def __init__(self, d: int, K: int, device, keep_input: bool = False):
        self.H = torch.zeros(K, d, d, dtype=torch.float32, device=device)
        self.n = torch.zeros(K, dtype=torch.float64)
        self.cur_block = 0
        self.keep_input = keep_input
        self.last_input = None
        self.enabled = True
        self._h = None

    @torch.no_grad()
    def _hook(self, module, args):
        X = args[0].reshape(-1, args[0].shape[-1])
        if self.keep_input:
            self.last_input = X
        if not self.enabled or X.shape[0] == 0:
            return
        Xf = X.to(torch.float32)
        self.H[self.cur_block] += (Xf.T @ Xf).to(self.H.device)
        self.n[self.cur_block] += X.shape[0]

    def attach(self, blk):
        self._h = blk.gate.register_forward_pre_hook(self._hook)
        return self

    def detach(self):
        if self._h is not None:
            self._h.remove()
            self._h = None

    def H_sum(self, blocks):
        return self.H[list(blocks)].sum(dim=0)

    def tokens(self, blocks):
        return float(self.n[list(blocks)].sum())


# ----------------------------------------------------------------------------- numerical checks / metrics
@torch.no_grad()
def numerics(H: torch.Tensor, ref_trace: float | None = None, eig_tol: float = 1e-6) -> dict:
    """Symmetry, finiteness, min eigenvalue relative to the max, trace preservation (relative)."""
    Hd = H.double()
    w = torch.linalg.eigvalsh(Hd)
    mx = float(w.max().abs()) or 1.0
    out = dict(shape=list(H.shape), dtype=str(H.dtype), finite=bool(torch.isfinite(H).all()),
               sym_rel=float(torch.linalg.norm(Hd - Hd.T) / (torch.linalg.norm(Hd) + 1e-300)),
               min_eig_rel=float(w.min()) / mx, psd_ok=bool(float(w.min()) >= -eig_tol * mx))
    if ref_trace is not None:
        out["trace_rel_err"] = abs(float(torch.trace(Hd)) - ref_trace) / (abs(ref_trace) + 1e-300)
    return out


@torch.no_grad()
def frob_mismatch(H: torch.Tensor, R: torch.Tensor, eps: float = 1e-12) -> float:
    H, R = H.double(), R.double()
    return float(torch.linalg.norm(H - R) / (torch.linalg.norm(R) + eps))


@torch.no_grad()
def diag_mismatch(H: torch.Tensor, R: torch.Tensor, eps: float = 1e-12) -> float:
    dh, dr = torch.diagonal(H.double()), torch.diagonal(R.double())
    return float(torch.linalg.norm(dh - dr) / (torch.linalg.norm(dr) + eps))


@torch.no_grad()
def top_eigvecs(H: torch.Tensor, k: int = TOPK) -> torch.Tensor:
    """Top-k eigenvectors (same convention as diagnostics/covariance_mismatch.spectral)."""
    _, U = torch.linalg.eigh(H.double())
    return U[:, -k:]


@torch.no_grad()
def subspace_alignment(U: torch.Tensor, V: torch.Tensor) -> float:
    """||U^T V||_F^2 / k in [0, 1] (covariance_mismatch.py convention)."""
    return float(torch.linalg.norm(U.T @ V) ** 2 / U.shape[1])


# ----------------------------------------------------------------------------- frozen rules
RULES = dict(
    s1_min_rel_improvement=0.03,      # mean D_F(pooled, a) <= 0.97 mean D_F(raw)
    s1_min_experts=8,                 # of 12 experts with D_F(pooled, a) < D_F(raw)
    s1_layer_max_regression=0.01,     # a layer whose mean D_F rises > 1 % vs raw = large systematic regression
    s2_min_rel_improvement=0.01,      # mean J (12 experts) >= 1 % below standard GPTQ, same budget
    s2_tuned_tolerance=0.005,         # competitive with tuned damping: mean J_pool <= 1.005 mean J_tuned
    s2_min_experts=8,                 # experts with J_pool < J_std
    s2_max_expert_regression=0.05,    # no expert worse than +5 % vs standard
    s2_cost_max_overhead=0.05,        # pooled accumulation <= 5 % of (replay + 60-expert GPTQ) per layer
)


def screen_budget(df_raw, df_pool: dict, df_id: dict, layer_of, rules=RULES) -> dict:
    """Stage 1 screen at one budget. df_raw: [12] D_F of the raw estimate; df_pool / df_id: {alpha: [12]} for
    nonzero alphas; layer_of: [12] layer ids. Returns per-alpha criteria and the selected alpha (lowest mean pooled
    D_F among passing alphas) or None."""
    raw = np.asarray(df_raw, float)
    lay = np.asarray(layer_of)
    per = {}
    for a in sorted(df_pool):
        pl, idn = np.asarray(df_pool[a], float), np.asarray(df_id[a], float)
        rel = 1.0 - pl.mean() / raw.mean()
        n_imp = int((pl < raw).sum())
        layer_reg = {int(L): float(pl[lay == L].mean() / raw[lay == L].mean() - 1.0) for L in np.unique(lay)}
        c = dict(c1_rel_improvement=bool(rel >= rules["s1_min_rel_improvement"]),
                 c2_experts_improved=bool(n_imp >= rules["s1_min_experts"]),
                 c3_beats_identity=bool(pl.mean() < idn.mean()),
                 c4_no_layer_regression=bool(all(v <= rules["s1_layer_max_regression"] for v in layer_reg.values())))
        per[a] = dict(rel_improvement=float(rel), n_improved=n_imp, mean_pooled=float(pl.mean()),
                      mean_identity=float(idn.mean()), mean_raw=float(raw.mean()),
                      pooled_vs_identity_rel=float(pl.mean() / idn.mean() - 1.0), layer_rel_change=layer_reg,
                      criteria=c, passes=all(c.values()))
    passing = [a for a in per if per[a]["passes"]]
    sel = min(passing, key=lambda a: per[a]["mean_pooled"]) if passing else None
    return dict(per_alpha=per, passing_alphas=passing, selected_alpha=sel)


def stage1_label(screens: dict, numerics_ok: bool) -> str:
    """screens: {B: screen_budget(...)} for the primary budgets."""
    if not numerics_ok:
        return "POOL-PRIOR-BLOCKED"
    return "POOL-PRIOR-PROMISING" if any(s["selected_alpha"] is not None for s in screens.values()) \
        else "POOL-PRIOR-NO-GAIN"


def budget_criteria(J: dict, layer_of, rules=RULES) -> dict:
    """Stage 2 per-budget criteria 1-6. J: {"std","pool","identity","tuned"} -> [12] per-expert J_e."""
    s, pl, idn, tu = (np.asarray(J[k], float) for k in ("std", "pool", "identity", "tuned"))
    lay = np.asarray(layer_of)
    rel = 1.0 - pl.mean() / s.mean()
    d = pl / s - 1.0
    layer_rel = {int(L): float(1.0 - pl[lay == L].mean() / s[lay == L].mean()) for L in np.unique(lay)}
    c = dict(c1_min_improvement=bool(rel >= rules["s2_min_rel_improvement"]),
             c2_beats_identity=bool(pl.mean() < idn.mean()),
             c3_competitive_with_tuned=bool(pl.mean() <= (1.0 + rules["s2_tuned_tolerance"]) * tu.mean()),
             c4_experts_improved=bool(int((d < 0).sum()) >= rules["s2_min_experts"]),
             c5_no_tail_regression=bool(d.max() <= rules["s2_max_expert_regression"]),
             c6_both_layers=bool(all(v > 0 for v in layer_rel.values())))
    return dict(rel_improvement_vs_std=float(rel), rel_vs_identity=float(1.0 - pl.mean() / idn.mean()),
                rel_vs_tuned=float(1.0 - pl.mean() / tu.mean()), n_improved=int((d < 0).sum()),
                worst_expert_rel_change=float(d.max()), p90_expert_rel_change=float(np.quantile(d, 0.9)),
                layer_rel_improvement=layer_rel, criteria=c)


def classify(stage1: str, per_budget: dict, subset_ok: bool | None, correctness_ok: bool, cost_ok: bool) -> dict:
    """Frozen precedence: BLOCKED > (Stage 1 failed: POOL-PRIOR-NO-GAIN) > PROMISING (some eligible budget meets 1-6,
    plus 7-9) > REGULARIZATION-ONLY (some budget meets 1 but fails 2 or 3) > MISMATCH-ONLY (no budget meets 1:
    the prior improved mismatch, Stage 1 passed, but not reconstruction) > NO-GAIN (1-3 met, robustness 4-9 fail)."""
    if not correctness_ok or stage1 == "POOL-PRIOR-BLOCKED":
        return dict(verdict="POOL-SHRINKAGE-BLOCKED", recommendation="B")
    if stage1 != "POOL-PRIOR-PROMISING":
        return dict(verdict="POOL-PRIOR-NO-GAIN", recommendation="B")
    cr = {B: v["criteria"] for B, v in per_budget.items()}
    glob = (subset_ok is not False) and cost_ok
    if glob and any(all(c.values()) for c in cr.values()):
        return dict(verdict="POOL-SHRINKAGE-PROMISING", recommendation="A")
    if any(c["c1_min_improvement"] and not (c["c2_beats_identity"] and c["c3_competitive_with_tuned"])
           for c in cr.values()):
        return dict(verdict="POOL-SHRINKAGE-REGULARIZATION-ONLY", recommendation="B")
    if not any(c["c1_min_improvement"] for c in cr.values()):
        return dict(verdict="POOL-SHRINKAGE-MISMATCH-ONLY", recommendation="C")
    return dict(verdict="POOL-SHRINKAGE-NO-GAIN", recommendation="B")


def gap_recovery(e_std_b, e_new_b, e_std_2b, min_gap_frac=0.01):
    """R = (E_B^std - E_B^new) / (E_B^std - E_2B^std); None when the budget gap is < min_gap_frac of E_B^std."""
    gap = e_std_b - e_std_2b
    if gap <= min_gap_frac * e_std_b:
        return None
    return float((e_std_b - e_new_b) / gap)
