"""Aggregate-shift-constrained Coverage-LP (coverage_lp_aggregate; 11/07, final).

Constraint on the FINAL set S of B sequences (warm start included):

    D_L(S) = sum_e | p_S(L,e) - p_pool(L,e) | <= eps_L      for every layer L,

p_pool = routing distribution of the ORIGINAL full candidate pool (fixed),
p_S(L,e) = sum_{s in S} C[s,L,e] / (B * T), where T = routed tokens per
sequence per layer (512 x top-4 = 2048; VERIFIED on the cache, never
assumed: `token_invariant`). With |S| = B fixed this is linear in z.

Exact linear form (auxiliary t[L,e] >= 0):
    t >=  (n_fix + C^T x) / M - p_pool
    t >= -(n_fix + C^T x) / M + p_pool        M = B * T
    sum_e t[L,e] <= eps_L
where n_fix = routed counts of sequences already fixed (warm start + earlier
rounds) and x = all still-free selection mass (this round z + future f).

Controller per adaptive round r (budget b_r, k_future slots after it):
  1. LP relaxation over (z, f, y, t): coverage_lp's objective sum_e u_e y_e on
     the round's picks z only (y_e <= C^T z, 0 <= y_e <= n_e, sum z = b_r),
     plus the future mass f (sum f = k_future, z + f <= 1) and the aggregate
     rows on n_fix + C^T (z + f): the final-set constraint with the remaining
     horizon relaxed, so a round is never optimised as if it were the last.
  2. Rounding exactly as coverage_lp: descending-z positive-marginal walk, then
     seeded random fill (rng.sample) to b_r.
  3. Discrete certificate: an exact MILP feasibility problem over binary f
     decides whether the rounded picks still admit an INTEGER completion of
     k_future sequences meeting all four bounds (k_future = 0: direct check).
  4. If not, REPAIR: the exact constrained round MILP (binary z and f):
     max coverage objective of z s.t. a binary completion f exists. Its
     solution carries its own completion certificate, so the next round is
     always feasible. Repairs and objective loss vs step 2 are logged.
  Infeasible -> AggInfeasible; solver without a certified integer solution ->
  AggUnresolved. eps is never relaxed and there is no unconstrained fallback.

This is a CONTROLLER difference from coverage_lp (horizon-aware LP, discrete
certificate, MILP repair), not only a utility difference.

Thresholds (prespecified, `build_epsilon`): for each seed's fixed warm start,
1,000 completions of 96 uniform draws from pool \\ W_s with an RNG seeded by
(EPS_SEED, seed); eps_L = median of D_L over all 3,000 pooled draws.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

EPS_SEED = 20261107
TOL = 1e-12
# Deterministic solver budget: HiGHS branch-and-bound stops at a relative gap of
# 1e-4 or after 200 nodes (a NODE limit, unlike a time limit, gives the same
# incumbent on every run). The incumbent is re-checked discretely; its gap to
# the LP bound is logged. Fixed before any confirmation evaluation.
MILP_OPTS = dict(disp=False, presolve=True, mip_rel_gap=1e-4, node_limit=200)


class AggInfeasible(RuntimeError):
    """No discrete selection satisfies the aggregate bounds."""


class AggUnresolved(RuntimeError):
    """The solver returned no certified integer solution."""


# ---------------------------------------------------------------- basics

def token_invariant(C, n_layers, E):
    """Routed tokens per sequence per layer; must be one constant for every row
    and layer (top-k routing over a fixed sequence length). Raises otherwise."""
    C = np.asarray(C, float)
    assert C.shape[1] == n_layers * E
    tots = np.stack([C[:, l * E:(l + 1) * E].sum(1) for l in range(n_layers)], 1)
    T = float(tots[0, 0])
    if T <= 0 or not np.all(tots == T):
        raise ValueError(f"routed tokens per sequence/layer are not constant "
                         f"(min {tots.min()}, max {tots.max()}): aggregate normalisation invalid")
    return T


def pool_reference_flat(C, n_layers, E):
    """p_pool as a flat layer-major vector (j = l * E + e)."""
    C = np.asarray(C, float)
    out = []
    for l in range(n_layers):
        X = C[:, l * E:(l + 1) * E].sum(0)
        out.append(X / X.sum())
    return np.concatenate(out)


def aggregate_l1(counts, ref, M, n_layers, E):
    """D_L for a selected set with summed routed counts `counts` and size*T = M."""
    p = np.asarray(counts, float) / M
    return np.array([np.abs(p[l * E:(l + 1) * E] - ref[l * E:(l + 1) * E]).sum()
                     for l in range(n_layers)])


class AggSpec:
    def __init__(self, ref, eps, B, T, n_layers, E):
        self.ref, self.eps = np.asarray(ref, float), np.asarray(eps, float)
        self.B, self.T, self.nL, self.E = int(B), float(T), int(n_layers), int(E)
        self.M = self.B * self.T
        self.J = self.nL * self.E

    def D(self, counts):
        return aggregate_l1(counts, self.ref, self.M, self.nL, self.E)

    def ok(self, counts):
        return bool(np.all(self.D(counts) <= self.eps + TOL))


def build_epsilon(C, pool_ids, warm_by_seed, B, n_draws=1000, eps_seed=EPS_SEED, n_layers=4, E=60):
    """Prespecified thresholds from random completions of the fixed warm starts.
    Returns dict(eps, D[seed][draw][layer], draws (row indices), spec)."""
    C = np.asarray(C, float)
    T = token_invariant(C, n_layers, E)
    ref = pool_reference_flat(C, n_layers, E)
    row = {int(s): i for i, s in enumerate(pool_ids)}
    D, draws = {}, {}
    for sd in sorted(warm_by_seed, key=int):
        W = [row[int(s)] for s in warm_by_seed[sd]]
        rest = np.setdiff1d(np.arange(len(pool_ids)), W)
        rng = np.random.default_rng([eps_seed, int(sd)])
        nW = C[W].sum(0)
        dd, ii = [], []
        for _ in range(n_draws):
            pick = rng.choice(rest, B - len(W), replace=False)
            dd.append(aggregate_l1(nW + C[pick].sum(0), ref, B * T, n_layers, E))
            ii.append(np.sort(pick).tolist())
        D[str(sd)], draws[str(sd)] = np.array(dd), ii
    allD = np.concatenate([D[k] for k in sorted(D, key=int)])
    eps = np.median(allD, axis=0)
    return dict(eps=eps, D=D, draws=draws, T=T, ref=ref,
                rule=f"eps_L = median over {len(D)} seeds x {n_draws} uniform completions of the fixed "
                     f"warm start (numpy default_rng([{eps_seed}, seed]), choice without replacement)",
                eps_seed=eps_seed, n_draws=n_draws)


# ---------------------------------------------------------------- programs

def _program(Cc, value, cap, b, k, n_fix, spec, with_round=True):
    """Variables [z (S) | f (S if k) | y (J if with_round) | t (J)]. Returns
    (c, constraints(list of (A, lo, hi)), lb, ub, sizes)."""
    S, J = Cc.shape
    nz = S if with_round else 0
    nf = S if k > 0 else 0
    ny = J if with_round else 0
    nt = J
    N = nz + nf + ny + nt
    oz, of, oy, ot = 0, nz, nz + nf, nz + nf + ny
    CT = sparse.csr_matrix(np.asarray(Cc, float).T)
    rows, lo, hi = [], [], []

    def block(parts):
        mats = []
        for off, width, mat in parts:
            mats.append((off, width, mat))
        r = mats[0][2].shape[0]
        cols = []
        cur = 0
        for off, width, mat in sorted(mats, key=lambda x: x[0]):
            if off > cur:
                cols.append(sparse.csr_matrix((r, off - cur)))
            cols.append(mat)
            cur = off + width
        if cur < N:
            cols.append(sparse.csr_matrix((r, N - cur)))
        return sparse.hstack(cols).tocsr()

    if with_round:
        rows.append(block([(oz, nz, -CT), (oy, ny, sparse.eye(J))]))       # y - C^T z <= 0
        lo.append(np.full(J, -np.inf)); hi.append(np.zeros(J))
        rows.append(block([(oz, nz, sparse.csr_matrix(np.ones((1, S))))]))  # sum z = b
        lo.append([b]); hi.append([b])
    if nf:
        rows.append(block([(of, nf, sparse.csr_matrix(np.ones((1, S))))]))  # sum f = k
        lo.append([k]); hi.append([k])
        if with_round:
            rows.append(block([(oz, nz, sparse.eye(S)), (of, nf, sparse.eye(S))]))   # z + f <= 1
            lo.append(np.full(S, -np.inf)); hi.append(np.ones(S))
    rhs = spec.ref - np.asarray(n_fix, float) / spec.M
    sel = [(oz, nz, CT / spec.M)] if with_round else []
    if nf:
        sel.append((of, nf, CT / spec.M))
    if sel:
        rows.append(block(sel + [(ot, nt, -sparse.eye(J))]))                       # p - t <= ref
        lo.append(np.full(J, -np.inf)); hi.append(rhs)
        rows.append(block([(o, w, -m) for o, w, m in sel] + [(ot, nt, -sparse.eye(J))]))
        lo.append(np.full(J, -np.inf)); hi.append(-rhs)
    else:                                                                          # nothing free
        rows.append(block([(ot, nt, -sparse.eye(J))]))
        lo.append(np.full(J, -np.inf)); hi.append(-np.abs(rhs))
    Lsum = sparse.lil_matrix((spec.nL, J))
    for l in range(spec.nL):
        Lsum[l, l * spec.E:(l + 1) * spec.E] = 1
    rows.append(block([(ot, nt, Lsum.tocsr())]))
    lo.append(np.full(spec.nL, -np.inf)); hi.append(spec.eps)
    c = np.zeros(N)
    lb, ub = np.zeros(N), np.ones(N)
    if with_round:
        c[oy:oy + ny] = -np.asarray(value, float)
        ub[oy:oy + ny] = np.maximum(0.0, np.asarray(cap, float))
    ub[ot:ot + nt] = np.inf
    A = sparse.vstack(rows).tocsr()
    return c, A, np.concatenate([np.asarray(x, float) for x in lo]), \
        np.concatenate([np.asarray(x, float) for x in hi]), lb, ub, dict(oz=oz, nz=nz, of=of, nf=nf, oy=oy, ot=ot)


def lp_round(Cc, value, cap, b, k, n_fix, spec):
    """Horizon-aware LP relaxation. Returns (z, f, info). Raises AggInfeasible."""
    c, A, lo, hi, lb, ub, o = _program(Cc, value, cap, b, k, n_fix, spec)
    res, sec = _milp(c, A, lo, hi, lb, ub, np.zeros(len(c)))      # all continuous = the LP (HiGHS)
    if res.status == 2:
        raise AggInfeasible(f"horizon LP infeasible: {res.message}")
    if res.x is None:
        raise AggUnresolved(f"horizon LP: {res.message}")
    z = res.x[o["oz"]:o["oz"] + o["nz"]]
    f = res.x[o["of"]:o["of"] + o["nf"]] if o["nf"] else np.zeros(0)
    return z, f, dict(lp_status=int(res.status), lp_objective=float(-res.fun), lp_s=sec)


def _milp(c, A, lo, hi, lb, ub, integ, opts=None):
    """HiGHS via scipy. On a solver ERROR (status 4 without a solution; seen on
    fully presolved degenerate problems) retry once with presolve off - same
    problem, different solve path; every solution is re-checked discretely."""
    t0 = time.time()
    o = dict(MILP_OPTS, **(opts or {}))
    res = milp(c, constraints=[LinearConstraint(A, lo, hi)], integrality=integ, bounds=Bounds(lb, ub), options=o)
    if res.status == 4 and res.x is None and o.get("presolve", True):
        res = milp(c, constraints=[LinearConstraint(A, lo, hi)], integrality=integ, bounds=Bounds(lb, ub),
                   options=dict(o, presolve=False))
    return res, time.time() - t0


def completion_feasible(C_avail, n_fix, k, spec):
    """Does an INTEGER set of k rows of C_avail exist with n_fix + their counts
    within all bounds? Exact MILP (k = 0: direct check). Returns (ok, rows, info)."""
    if k == 0:
        return spec.ok(n_fix), [], dict(status="direct", seconds=0.0)
    if len(C_avail) < k:
        return False, [], dict(status="too_few_candidates", seconds=0.0)
    if len(C_avail) == k:                                  # forced completion: exact direct check
        okd = spec.ok(np.asarray(n_fix, float) + np.asarray(C_avail, float).sum(0))
        return okd, (list(range(k)) if okd else []), dict(status="forced_direct", seconds=0.0)
    c, A, lo, hi, lb, ub, o = _program(C_avail, None, None, 0, k, n_fix, spec, with_round=False)
    integ = np.zeros(len(c))
    integ[o["of"]:o["of"] + o["nf"]] = 1
    res, sec = _milp(c, A, lo, hi, lb, ub, integ)
    if res.status == 2:
        return False, [], dict(status="infeasible", seconds=sec)
    if res.x is None:
        raise AggUnresolved(f"completion MILP returned no solution: {res.message}")
    f = np.flatnonzero(res.x[o["of"]:o["of"] + o["nf"]] > 0.5)
    if len(f) != k or not spec.ok(np.asarray(n_fix, float) + np.asarray(C_avail, float)[f].sum(0)):
        raise AggUnresolved("completion MILP solution fails the discrete re-check")
    return True, f.tolist(), dict(status="feasible", seconds=sec, milp_status=int(res.status))


def milp_round(Cc, value, cap, b, k, n_fix, spec):
    """Exact constrained discrete round: max coverage objective of z subject to an
    integer completion f. Returns (z_rows, f_rows, info)."""
    c, A, lo, hi, lb, ub, o = _program(Cc, value, cap, b, k, n_fix, spec)
    integ = np.zeros(len(c))
    integ[:o["nz"] + o["nf"]] = 1
    res, sec = _milp(c, A, lo, hi, lb, ub, integ)
    if res.status == 2:
        raise AggInfeasible(f"round MILP infeasible: {res.message}")
    if res.x is None:
        raise AggUnresolved(f"round MILP returned no solution: {res.message}")
    z = np.flatnonzero(res.x[:o["nz"]] > 0.5)
    f = np.flatnonzero(res.x[o["of"]:o["of"] + o["nf"]] > 0.5) if o["nf"] else np.zeros(0, int)
    Cc = np.asarray(Cc, float)
    tot = np.asarray(n_fix, float) + Cc[z].sum(0) + (Cc[f].sum(0) if len(f) else 0)
    if len(z) != b or len(f) != k or set(z) & set(f) or not spec.ok(tot):
        raise AggUnresolved("round MILP solution fails the discrete re-check")
    return z.tolist(), f.tolist(), dict(milp_status=int(res.status), milp_message=str(res.message),
                                        milp_objective=float(-res.fun), milp_gap=getattr(res, "mip_gap", None),
                                        milp_s=sec)


def coverage_objective(value, cap, Crows):
    cov = np.asarray(Crows, float).sum(0) if len(Crows) else np.zeros(len(value))
    return float(np.asarray(value) @ np.minimum(cap, cov))


# ---------------------------------------------------------------- one round

def select_round_aggregate(Cc, cand, value, cap, targets, budget, k_future, n_fix, spec, rng,
                           positive_marginal_selection, tol=1e-9):
    """One adaptive round of coverage_lp_aggregate (steps 1-4 in the module doc).
    Exploration (zero-exposure targets) is not supported by this controller and
    raises if any target exists (never triggered with a 32-sequence warm start;
    the existing exploration rule could not be certified cheaply)."""
    Cc = np.asarray(Cc, float)
    S = len(cand)
    if targets.any():
        raise AggUnresolved(f"{int(targets.sum())} zero-exposure exploration targets: not supported "
                            "by coverage_lp_aggregate")
    log = dict(n_exploration_targets=0, n_exploration_targets_reached=0,
               n_exploration_targets_unreached=0, n_exploration_sequences=0,
               exploration_fraction_of_round=0.0, n_positive_utility=int((value > 0).sum()),
               used_random_utility_fallback=False, n_random_fallback_sequences=0,
               n_lp_positive_value_selected=0, n_zero_marginal_random_fill=0, k_future=int(k_future))
    t0 = time.time()
    z, f, info = lp_round(Cc, value, cap, budget, k_future, n_fix, spec)
    log.update(info, lp_fractional_mass_z=float(z.sum()), lp_fractional_mass_f=float(f.sum()) if len(f) else 0.0)
    pos = positive_marginal_selection(z, Cc, value, cap, budget)
    chosen = set(pos)
    fill = rng.sample([i for i in range(S) if i not in chosen], budget - len(pos))
    naive = list(pos) + list(fill)
    naive_obj = coverage_objective(value, cap, Cc[naive])
    log.update(naive_n_lp=len(pos), naive_n_fill=len(fill), naive_objective=naive_obj)
    avail = np.setdiff1d(np.arange(S), naive)
    tc = time.time()
    ok, comp, cinfo = completion_feasible(Cc[avail], np.asarray(n_fix, float) + Cc[naive].sum(0),
                                          k_future, spec)
    log.update(certificate_s=time.time() - tc, certificate_status=cinfo["status"],
               naive_rounding_feasible=bool(ok))
    if ok:
        rows = naive
        log.update(repaired=False, n_rounding_repairs=0, n_lp_positive_value_selected=len(pos),
                   n_zero_marginal_random_fill=len(fill), final_objective=naive_obj,
                   rounding_objective_loss=0.0, rounding_objective_loss_rel=0.0,
                   completion_certificate=[cand[i] for i in avail[comp]] if k_future else [])
    else:
        zr, fr, minfo = milp_round(Cc, value, cap, budget, k_future, n_fix, spec)
        rows = zr
        fin = coverage_objective(value, cap, Cc[zr])
        log.update(minfo, repaired=True, n_rounding_repairs=len(set(zr) ^ set(naive)) // 2,
                   n_lp_positive_value_selected=len(zr), n_zero_marginal_random_fill=0,
                   final_objective=fin, rounding_objective_loss=naive_obj - fin,
                   rounding_objective_loss_rel=(naive_obj - fin) / naive_obj if naive_obj > 0 else 0.0,
                   completion_certificate=[cand[i] for i in fr])
    take = [cand[i] for i in rows]
    assert len(take) == budget == len(set(take))
    part = np.asarray(n_fix, float) + Cc[rows].sum(0)
    size = part[:spec.E].sum() / spec.T
    log.update(round_select_s=time.time() - t0, n_selected_after_round=int(round(size)),
               D_partial_set_after_round=aggregate_l1(part, spec.ref, size * spec.T, spec.nL, spec.E).tolist())
    return take, rows, log
