"""Shared covariance-aware (SC) sequence selection for full E3 (240 experts, 4 layers), fixed warm start.

Same surrogate as the expert-wise MVP (moe_calib/expertwise_calibration.py): per (layer, expert, proj) term t
  D_t(S) = || sum_{s in S} M_t[s] / sum_{s in S} n_t[s] - R_t ||_F / (||R_t||_F + eps),    R_t = full-pool reference
with M_t[s] the svech of the 64-d Gaussian sketch sum_{tokens of s routed to e} z z^T (z = R_p^T x) and the
empty-token rule H(S) := 0. The global shared objective is the equal-expert mean
  J(S) = 1/240 sum_{L, e} 1/2 sum_p D_{L,e,p}(S)       (term weight 1/480; every layer has 60 experts -> weight 1/4)

Optimizer = the MVP's (seeded stochastic greedy over a uniform 10 % sample of the remaining candidates, then <= 5
best-improvement swap passes accepted only if the EXACT objective decreases), with two differences required by the
E3 protocol: (1) the greedy starts from the frozen ordered 32-sequence warm start W and fills 96 slots;
(2) swap passes only visit the 96 non-warm slots (W is never removed). Candidates = entire original pool minus S.

Scalability: all terms are stacked into one [T, N, V] tensor (fp32 storage, fp64 accumulation of every exact
quantity), so candidate scoring for the whole pool is one batched mat-vec per update. Closed-form scores only
PROPOSE moves; acceptance always recomputes the exact objective in fp64 from the stored sketches.
"""
from __future__ import annotations

import math

import numpy as np
import torch

EPS = 1e-12
SC_STREAM = 1          # MVP COND_CODE["SC"]: rng = default_rng(SeedSequence([seed, 1]))


def sc_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), SC_STREAM]))


class StackedObjective:
    """J(S) = sum_t w_t ||A_t(S)/n_t(S) - R_t|| / (||R_t|| + eps) over stacked terms.

    M: [T, N, V] per-sequence svech sketch sums (fp32 or fp64, any device); n: [T, N] routed-token counts;
    w: [T] term weights. References are computed from the full pool in fp64."""

    def __init__(self, M: torch.Tensor, n: torch.Tensor, w: torch.Tensor, eps: float = EPS, chunk: int = 32):
        if M.dim() != 3 or n.shape != M.shape[:2] or w.shape != (M.shape[0],):
            raise ValueError(f"shape mismatch M {tuple(M.shape)} n {tuple(n.shape)} w {tuple(w.shape)}")
        self.M, self.eps, self.chunk = M, eps, chunk
        self.dev = M.device
        self.T, self.N, self.V = M.shape
        self.n = n.to(self.dev, torch.float64)
        self.w = w.to(self.dev, torch.float64)
        tot = self.n.sum(1)                                   # [T]
        if bool((tot <= 0).any()):
            raise ValueError("a term has zero pool tokens: its reference is undefined")
        R = torch.zeros(self.T, self.V, dtype=torch.float64, device=self.dev)
        MM = torch.zeros(self.T, self.N, dtype=torch.float64, device=self.dev)
        for a in range(0, self.T, chunk):
            Mc = M[a:a + chunk].double()
            R[a:a + chunk] = Mc.sum(1) / tot[a:a + chunk, None]
            MM[a:a + chunk] = (Mc * Mc).sum(2)
        self.R, self.MM = R, MM
        self.Rn = torch.linalg.norm(R, dim=1)                 # [T]
        self.MR = self._matvec(R)                            # [T, N]
        self.coef = self.w / (self.Rn + eps)                  # [T]

    # ------------------------------------------------------------------ helpers
    def _matvec(self, X: torch.Tensor) -> torch.Tensor:
        """[T, V] fp64 -> M @ X per term, [T, N] fp64 (fp32 product if M is fp32)."""
        out = torch.empty(self.T, self.N, dtype=torch.float64, device=self.dev)
        for a in range(0, self.T, self.chunk):
            Mc = self.M[a:a + self.chunk]
            out[a:a + self.chunk] = torch.bmm(Mc, X[a:a + self.chunk].to(Mc.dtype).unsqueeze(2)).squeeze(2).double()
        return out

    def _sum_rows(self, idx: torch.Tensor) -> torch.Tensor:
        A = torch.zeros(self.T, self.V, dtype=torch.float64, device=self.dev)
        if idx.numel():
            for a in range(0, self.T, self.chunk):
                A[a:a + self.chunk] = self.M[a:a + self.chunk][:, idx].double().sum(1)
        return A

    def _idx(self, sel) -> torch.Tensor:
        return torch.as_tensor(list(sel), dtype=torch.long, device=self.dev)

    # ------------------------------------------------------------------ exact (fp64)
    def per_term(self, sel) -> torch.Tensor:
        """D_t(S) for every term, [T] fp64 (exact; empty-token rule)."""
        idx = self._idx(sel)
        A = self._sum_rows(idx)
        ns = self.n[:, idx].sum(1) if idx.numel() else torch.zeros(self.T, dtype=torch.float64, device=self.dev)
        safe = torch.where(ns > 0, ns, torch.ones_like(ns))
        cur = torch.where((ns > 0)[:, None], A / safe[:, None], torch.zeros_like(A))
        return torch.linalg.norm(cur - self.R, dim=1) / (self.Rn + self.eps)

    def exact(self, sel) -> float:
        return float((self.w * self.per_term(sel)).sum())

    # ------------------------------------------------------------------ incremental state
    def state(self, sel) -> dict:
        idx = self._idx(sel)
        A = self._sum_rows(idx)
        ns = self.n[:, idx].sum(1) if idx.numel() else torch.zeros(self.T, dtype=torch.float64, device=self.dev)
        return dict(A=A, n=ns, MA=self._matvec(A))

    def _scores(self, A, ns, MA) -> torch.Tensor:
        """Objective of S + c for every candidate c, [N] fp64."""
        AA = (A * A).sum(1)[:, None]
        AR = (A * self.R).sum(1)[:, None]
        den = ns[:, None] + self.n                           # [T, N]
        num2 = AA + 2.0 * MA + self.MM
        numR = AR + self.MR
        safe = torch.where(den > 0, den, torch.ones_like(den))
        q = num2 / safe ** 2 - 2.0 * numR / safe + (self.Rn ** 2)[:, None]
        val = torch.sqrt(q.clamp_min(0.0))
        val = torch.where(den > 0, val, self.Rn[:, None].expand_as(val))
        return (self.coef[:, None] * val).sum(0)

    def add_scores(self, st) -> torch.Tensor:
        return self._scores(st["A"], st["n"], st["MA"])

    def swap_scores(self, st, i: int) -> torch.Tensor:
        Mi = self._sum_rows(self._idx([i]))
        return self._scores(st["A"] - Mi, st["n"] - self.n[:, i], st["MA"] - self._matvec(Mi))

    def apply_add(self, st, c: int):
        Mc = self._sum_rows(self._idx([c]))
        st["A"] = st["A"] + Mc
        st["n"] = st["n"] + self.n[:, c]
        st["MA"] = st["MA"] + self._matvec(Mc)

    def apply_remove(self, st, i: int):
        Mi = self._sum_rows(self._idx([i]))
        st["A"] = st["A"] - Mi
        st["n"] = st["n"] - self.n[:, i]
        st["MA"] = st["MA"] - self._matvec(Mi)


# ===================================================================== optimizer (MVP, with fixed warm start)
def greedy_from_warm(obj: StackedObjective, warm: list[int], B: int, rng: np.random.Generator,
                     sample_frac: float = 0.1) -> tuple[list[int], dict]:
    """Start from the ordered warm start; add B - |W| sequences, each the argmin of the closed-form objective over
    a uniform ceil(sample_frac * N) sample of the remaining candidates (ties -> lowest id). Returns slots in order."""
    N = obj.N
    sel = list(map(int, warm))
    if len(set(sel)) != len(sel) or len(sel) > B:
        raise ValueError("invalid warm start")
    in_sel = np.zeros(N, dtype=bool)
    in_sel[sel] = True
    st = obj.state(sel)
    m = N if sample_frac >= 1 else max(1, int(math.ceil(sample_frac * N)))
    n_scored = 0
    for _ in range(B - len(sel)):
        rem = np.flatnonzero(~in_sel)
        cand = rem if m >= rem.size else np.sort(rng.choice(rem, size=m, replace=False))
        sc = obj.add_scores(st)[torch.as_tensor(cand, device=obj.dev)]
        c = int(cand[int(torch.argmin(sc))])
        n_scored += cand.size
        sel.append(c)
        in_sel[c] = True
        obj.apply_add(st, c)
    return sel, dict(candidates_scored=n_scored)


def swap_refine_fixed(obj: StackedObjective, sel: list[int], n_fixed: int, rng: np.random.Generator,
                      max_passes: int = 5, rel_tol: float = 1e-9) -> tuple[list[int], list[float], dict]:
    """MVP best-improvement swaps over the slots n_fixed..B-1 only (warm slots never removed). Each pass visits those
    slots in a seeded random order; the best out-of-set candidate (whole pool) is proposed in closed form and
    accepted only if the exact fp64 objective strictly decreases by more than rel_tol (relative)."""
    sel = list(sel)
    in_sel = torch.zeros(obj.N, dtype=torch.bool, device=obj.dev)
    in_sel[torch.as_tensor(sel, device=obj.dev)] = True
    st = obj.state(sel)
    cur = obj.exact(sel)
    trace = [cur]
    n_acc = n_prop = passes = n_scored = 0
    free = np.arange(n_fixed, len(sel))
    for _ in range(max_passes):
        passes += 1
        improved = False
        for slot in free[rng.permutation(free.size)]:
            i = sel[slot]
            sc = obj.swap_scores(st, i)
            n_scored += int(obj.N - len(sel))
            sc[in_sel] = float("inf")
            c = int(torch.argmin(sc))
            if not math.isfinite(float(sc[c])) or float(sc[c]) >= cur * (1 - rel_tol):
                continue
            n_prop += 1
            trial = sel.copy()
            trial[slot] = c
            new = obj.exact(trial)
            if new < cur * (1 - rel_tol):
                obj.apply_remove(st, i)
                obj.apply_add(st, c)
                in_sel[i], in_sel[c] = False, True
                sel, cur = trial, new
                trace.append(cur)
                n_acc += 1
                improved = True
        if not improved:
            break
    if any(b > a for a, b in zip(trace, trace[1:])):
        raise AssertionError("objective increased during accepted swaps")
    return sel, trace, dict(passes=passes, accepted=n_acc, proposed=n_prop, candidates_scored=n_scored)


def select_shared(obj: StackedObjective, warm: list[int], B: int, seed: int, sample_frac: float = 0.1,
                  max_passes: int = 5, rel_tol: float = 1e-9) -> tuple[list[int], dict]:
    """SC for one allocation seed. Returns the ordered 128 ids (warm start first, then the 96 slots) and optimizer
    info. One RNG stream (seed, 1) feeds the greedy samples and then the swap visiting order, as in the MVP."""
    import time
    rng = sc_rng(seed)
    t0 = time.time()
    j_warm = obj.exact(warm)
    g, gi = greedy_from_warm(obj, warm, B, rng, sample_frac)
    t_g = time.time() - t0
    j_greedy = obj.exact(g)
    t1 = time.time()
    sel, trace, info = swap_refine_fixed(obj, g, len(warm), rng, max_passes, rel_tol)
    info.update(warm_objective=j_warm, greedy_objective=j_greedy, final_objective=trace[-1], trace=trace,
                greedy_candidates_scored=gi["candidates_scored"], greedy_s=t_g, swap_s=time.time() - t1,
                greedy_order=g)
    return sel, info


def validate_shared(sel, warm, B: int, pool_N: int, forbidden=()):
    """Hard checks for one shared selection; raises AssertionError."""
    if len(sel) != B or len(set(sel)) != B:
        raise AssertionError(f"{len(sel)} ids / {len(set(sel))} distinct, need {B}")
    if list(sel[:len(warm)]) != list(warm):
        raise AssertionError("ordered warm start not preserved")
    if min(sel) < 0 or max(sel) >= pool_N:
        raise AssertionError("id outside the original candidate pool")
    if set(sel) & set(forbidden):
        raise AssertionError("tune/test/confirmation row selected")


# ===================================================================== diagnostics helpers
def gini(x) -> float:
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return 0.0
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def overlap(a, b) -> dict:
    a, b = set(a), set(b)
    return dict(intersection=len(a & b), jaccard=len(a & b) / len(a | b), frac_of_128=len(a & b) / max(1, len(a)))


def relative_improvement(err_x, err_ref) -> float:
    """(err_ref - err_x) / err_ref: positive = x better."""
    return (err_ref - err_x) / err_ref


# ===================================================================== frozen decision rule
THRESHOLD = 0.005
DECISION_RULE = {
    "quantities": "r_s = (E_test(FR,s) - E_test(SC,s)) / E_test(FR,s), E = mean held-out relative reconstruction over "
                  "the 240-expert test universe, ORIGINAL extraction; r_conf_s same on the confirmation split; "
                  "r_L = 3-seed mean relative improvement in layer L; mean r = mean_s r_s",
    "SUPPORTED": "r_s > 0 for all 3 seeds AND r_L > 0 for >= 3 of 4 layers AND mean r >= 0.5 % AND mean r_conf > 0",
    "ROBUST-SUPPORTED": "SUPPORTED AND, under the paired grouping design (Stage B), the grouping-averaged reconstruction "
                        "advantage is > 0 in every allocation seed AND its hierarchical-bootstrap 95 % interval "
                        "(seeds, then groupings) excludes zero",
    "NO-GAIN": "mean r < 0.5 % (no material improvement on the primary test split)",
    "INCONCLUSIVE": "mean r >= 0.5 % but a SUPPORTED condition fails (seed, layer or confirmation direction), OR "
                    "SUPPORTED but the grouping-averaged advantage is <= 0 in some allocation seed",
    "precedence": "NO-GAIN is checked first; then ROBUST-SUPPORTED > SUPPORTED > INCONCLUSIVE",
    "scope": "primary comparison SC vs FR; the same rule is applied, descriptively labelled, to SC vs TR and SC vs TC",
    "ppl": "PPL is reported descriptively; no PPL claim if the grouping SD of paired PPL differences is >= |mean|",
}


def decide(r_seed, r_layer, r_conf_seed, grouping=None, threshold=THRESHOLD):
    """r_seed / r_conf_seed: per-seed relative improvements (lists); r_layer: {L: 3-seed mean};
    grouping: None (Stage B not run) or dict(seed_means=[...], ci95=[lo, hi]) of the reconstruction ADVANTAGE
    (positive = SC better). Returns (label, details)."""
    m = float(np.mean(r_seed))
    det = dict(mean_r=m, all_seeds_positive=all(v > 0 for v in r_seed),
               layers_positive=int(sum(v > 0 for v in r_layer.values())),
               mean_r_conf=float(np.mean(r_conf_seed)), material=m >= threshold)
    if m < threshold:
        return "NO-GAIN", det
    sup = det["all_seeds_positive"] and det["layers_positive"] >= 3 and det["mean_r_conf"] > 0
    det["supported_conditions"] = sup
    if not sup:
        return "INCONCLUSIVE", det
    if grouping is None:
        det["grouping"] = "not run"
        return "SUPPORTED", det
    pos = all(v > 0 for v in grouping["seed_means"])
    det.update(grouping_seed_means_positive=pos, grouping_ci95=grouping["ci95"])
    if not pos:
        return "INCONCLUSIVE", det
    if grouping["ci95"][0] > 0:
        return "ROBUST-SUPPORTED", det
    return "SUPPORTED", det
