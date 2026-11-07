"""Representativeness-constrained Coverage-LP (coverage_lp_repr; 11/07).

Same expert-coverage objective, capacities, warm start, schedule and LP as
coverage_lp (u_e = 1 / n_e, y_e <= sum_s C[s,e] z_s, 0 <= y_e <= n_e,
sum_s z_s = round budget), plus ONE additional constraint on the routing
atypicality of the selected sequences:

    sum_{s in round} d(s) z_s <= round cap.

d(s) is the existing typicality metric (moe_calib/typicality.py; ORIGINAL
2,000-sequence pool reference, equal layer weights). The scientific
constraint applies to the final collection of N_post = B - |warm| adaptive
picks:

    mean_{s in post-warm} d(s) <= tau,   i.e.  sum d(s) <= A = N_post * tau.

Cumulative accounting (prespecified, not tuned):
  * the warm start is fixed and consumes nothing;
  * before round r, the unspent budget A_rem = A - (d spent by earlier rounds)
    and the remaining slot count k_rem are known;
  * round cap = min( A_rem * b_r / k_rem,                       (pacing; slack carries forward)
                     A_rem - R_future )                          (feasibility reserve)
    where R_future = sum of sorted-ascending candidate d at positions
    [b_r, b_r + k_future) is a provable upper bound on the cheapest possible
    completion of the later rounds, whatever this round takes. The last round
    gets cap = A_rem exactly.
  Pacing prevents a myopic first round from spending the whole shared budget
  on atypical sequences and forcing the last round onto only the most typical.

Discrete feasibility (never only the fractional LP):
  constraint-aware rounding walks the LP solution in descending z exactly like
  e3_allocation.positive_marginal_selection, but a candidate is kept only if
      spent + d_i + minsum(k_left - 1 cheapest other available) <= cap.
  Skipped positive-marginal candidates are REPAIRS (counted, with the
  objective loss versus naive rounding of the same z). Remaining slots are a
  seeded random fill restricted the same way. The invariant
  spent + minsum(k_left) <= cap holds after every accepted pick and the
  cheapest available candidate always preserves it, so a feasible completion
  always exists: there is no unconstrained fallback.
"""
from __future__ import annotations

import numpy as np

from moe_calib import typicality as TY

TOL = 1e-12            # absolute tolerance on summed d (float64)


class ReprInfeasible(RuntimeError):
    """The atypicality budget cannot be met by any discrete selection."""


def pool_typicality(C_full, n_layers, E):
    """d(s) for every row of the ORIGINAL full-pool routing matrix (reference =
    the full pool itself). Returns (d, ref). Raises on malformed rows."""
    ref = TY.pool_reference(C_full, n_layers, E)
    sc = TY.sequence_scores(C_full, n_layers, E, ref)
    if sc["malformed"].any() or not np.isfinite(sc["d"]).all():
        raise ValueError(f"{int(sc['malformed'].sum())} malformed candidate rows: d(s) undefined")
    return sc["d"].astype(float), ref


def resolve_tau(spec, d):
    """tau from the prespecified rule: 'median' -> median d over the full pool."""
    if spec in (None, "median"):
        return float(np.median(d)), "median d(s) over the original candidate pool"
    return float(spec), "fixed numeric threshold (tests only)"


def minsum(d, k, exclude=None):
    """Sum of the k smallest values of d (optionally excluding index `exclude`)."""
    if k <= 0:
        return 0.0
    if exclude is not None:
        d = np.delete(d, exclude)
    if k > len(d):
        return np.inf
    return float(np.sum(np.partition(d, k - 1)[:k]))


def reserve_bound(d_cand, budget, k_future):
    """Upper bound on min-sum of k_future picks from d_cand after ANY `budget`
    picks are removed: sorted positions [budget, budget + k_future)."""
    if k_future <= 0:
        return 0.0
    s = np.sort(d_cand)
    if budget + k_future > len(s):
        return np.inf
    return float(s[budget:budget + k_future].sum())


class ReprBudget:
    """Shared atypicality budget A = n_post * tau over the adaptive rounds."""

    def __init__(self, tau, n_post):
        self.tau, self.n_post = float(tau), int(n_post)
        self.total = self.tau * self.n_post
        self.spent, self.n_taken = 0.0, 0
        self.history = []

    @property
    def remaining(self):
        return self.total - self.spent

    def round_cap(self, d_cand, budget):
        k_rem = self.n_post - self.n_taken
        if budget > k_rem:
            raise ValueError(f"round budget {budget} exceeds the {k_rem} remaining post-warm slots")
        k_future = k_rem - budget
        reserve = reserve_bound(d_cand, budget, k_future)
        paced = self.remaining * budget / k_rem if k_future > 0 else self.remaining
        cap = min(paced, self.remaining - reserve)
        need = minsum(d_cand, budget)
        if need > cap + TOL:
            raise ReprInfeasible(
                f"round needs >= {need:.6f} atypicality for {budget} picks but cap is {cap:.6f} "
                f"(remaining {self.remaining:.6f}, reserve {reserve:.6f}, tau {self.tau:.6f})")
        return cap, dict(paced_cap=paced, reserve_future=reserve, k_future=k_future,
                         min_feasible_round_sum=need, remaining_before=self.remaining)

    def commit(self, d_round):
        d_round = np.asarray(d_round, float)
        self.spent += float(d_round.sum())
        self.n_taken += len(d_round)
        self.history.append(float(d_round.sum()))
        if self.spent > self.total + TOL:
            raise ReprInfeasible(f"cumulative atypicality {self.spent:.6f} > budget {self.total:.6f}")

    def final_check(self):
        if self.n_taken != self.n_post:
            raise ReprInfeasible(f"{self.n_taken} post-warm picks, expected {self.n_post}")
        mean = self.spent / self.n_post
        return dict(tau=self.tau, n_post=self.n_post, budget_total=self.total, spent=self.spent,
                    mean_post_warm_d=mean, satisfied=bool(mean <= self.tau + TOL),
                    slack=self.total - self.spent, tightness=self.spent / self.total,
                    per_round_spent=list(self.history))


def objective(value, cap, Crows):
    cov = np.asarray(Crows, float).sum(0) if len(Crows) else np.zeros(len(value))
    return float(value @ np.minimum(cap, cov))


class _Picker:
    """Feasibility-tracking accumulator over candidate rows (indices into Cc)."""

    def __init__(self, d_cand, k, cap_d):
        self.d, self.k, self.cap = np.asarray(d_cand, float), int(k), float(cap_d)
        self.avail = np.ones(len(self.d), bool)
        self.picks, self.spent = [], 0.0

    def feasible(self, i):
        if not self.avail[i] or len(self.picks) >= self.k:
            return False
        k_left = self.k - len(self.picks) - 1
        rest = self.d[self.avail]
        pos = int(np.flatnonzero(self.avail).searchsorted(i))
        return self.spent + self.d[i] + minsum(rest, k_left, exclude=pos) <= self.cap + TOL

    def take(self, i):
        self.picks.append(int(i))
        self.avail[i] = False
        self.spent += float(self.d[i])

    def full(self):
        return len(self.picks) >= self.k


def select_round_repr(Cc, cand, value, cap, targets, budget, d_cand, cap_d, rng, solve_lp,
                      tol=1e-9):
    """One constrained adaptive round (mirrors e3_allocation.select_round):
      1. exploration cover for `targets`, restricted to feasible picks;
      2. LP with the extra row d^T z <= cap_d - spent; constraint-aware
         rounding of z (positive-marginal walk + feasibility; skips = repairs);
      3. feasibility-restricted seeded random fill for the remaining slots.
    Returns (take, log). Raises ReprInfeasible if no feasible set exists."""
    d_cand = np.asarray(d_cand, float)
    S = len(cand)
    if minsum(d_cand, budget) > cap_d + TOL:
        raise ReprInfeasible(f"no {budget} candidates fit the round cap {cap_d:.6f}")
    pk = _Picker(d_cand, budget, cap_d)

    # 1. exploration cover (same greedy rule as utility.exploration_cover, feasible rows only)
    hit = Cc > 0
    uncovered = targets & hit.any(axis=0)
    reached = np.zeros_like(targets)
    n_explore_skipped = 0
    while uncovered.any() and not pk.full():
        gain = hit[:, uncovered].sum(axis=1).astype(float)
        gain[~pk.avail] = -1
        order = np.argsort(-gain, kind="stable")
        chosen = None
        for i in order:
            if gain[i] <= 0:
                break
            if pk.feasible(i):
                chosen = int(i)
                break
            n_explore_skipped += 1
        if chosen is None:
            break
        pk.take(chosen)
        reached |= targets & hit[chosen]
        uncovered &= ~hit[chosen]
    n_explore = len(pk.picks)
    log = dict(n_exploration_targets=int(targets.sum()),
               n_exploration_targets_reached=int(reached.sum()),
               n_exploration_targets_unreached=int((targets & ~reached).sum()),
               n_exploration_sequences=n_explore,
               exploration_fraction_of_round=n_explore / budget,
               n_exploration_skipped_for_constraint=n_explore_skipped,
               n_positive_utility=int((value > 0).sum()),
               used_random_utility_fallback=False, n_random_fallback_sequences=0,
               n_lp_positive_value_selected=0, n_zero_marginal_random_fill=0,
               lp_status=None, lp_fractional_objective=None, lp_fractional_d_sum=None,
               naive_rounding_violates=None, naive_rounding_d_sum=None,
               n_rounding_repairs=0, naive_lp_objective=None, constrained_lp_objective=None,
               rounding_objective_loss=None, rounding_objective_loss_rel=None)
    rest = budget - n_explore
    keep = np.flatnonzero(pk.avail)
    if rest > 0:
        eligible = (value > 0) & (cap > 0) & (Cc[keep].sum(axis=0) > 0)
        if eligible.any():
            cap_lp = cap_d - pk.spent
            z, lpinfo = solve_lp(Cc[keep], value, cap, rest, d=d_cand[keep], d_cap=cap_lp)
            log.update(lp_status=lpinfo["status"], lp_fractional_objective=lpinfo["objective"],
                       lp_fractional_d_sum=float(d_cand[keep] @ z))
            # naive (unconstrained) rounding of the SAME z, for logging only:
            # e3's positive-marginal walk, then top-z completion to `rest`
            cov = np.zeros(Cc.shape[1])
            naive = []
            for j in np.argsort(-z, kind="stable"):
                if len(naive) >= rest or z[j] <= tol:
                    break
                new = np.minimum(cap, cov + Cc[keep[j]])
                if float(value @ (new - np.minimum(cap, cov))) > tol:
                    naive.append(int(j))
                    cov = new
            topz = [int(j) for j in np.argsort(-z, kind="stable")[:rest]]
            naive_d = float(d_cand[keep[topz]].sum())
            log.update(naive_rounding_d_sum=naive_d,
                       naive_rounding_violates=bool(naive_d > cap_lp + TOL),
                       naive_lp_objective=objective(value, cap, Cc[keep[naive]]) if naive else 0.0)
            # constrained rounding: same walk, feasibility-checked
            cov = np.zeros(Cc.shape[1])
            for i in Cc[pk.picks]:
                cov = cov + i
            cov0 = cov.copy()
            n_lp = repairs = 0
            for j in np.argsort(-z, kind="stable"):
                if pk.full() or z[j] <= tol:
                    break
                i = int(keep[j])
                new = np.minimum(cap, cov + Cc[i])
                if float(value @ (new - np.minimum(cap, cov))) <= tol:
                    continue
                if not pk.feasible(i):
                    repairs += 1
                    continue
                pk.take(i)
                cov = new
                n_lp += 1
            lp_rows = pk.picks[n_explore:]
            cons_obj = float(value @ (np.minimum(cap, cov) - np.minimum(cap, cov0)))
            naive_obj = log["naive_lp_objective"]
            log.update(n_lp_positive_value_selected=n_lp, n_rounding_repairs=repairs,
                       constrained_lp_objective=cons_obj,
                       rounding_objective_loss=naive_obj - cons_obj,
                       rounding_objective_loss_rel=((naive_obj - cons_obj) / naive_obj
                                                    if naive_obj > 0 else 0.0))
            assert len(lp_rows) == n_lp
        else:
            log["used_random_utility_fallback"] = True
        # 3. feasibility-restricted seeded random fill
        n_fill = 0
        order = list(np.flatnonzero(pk.avail))
        rng.shuffle(order)
        for i in order:
            if pk.full():
                break
            if pk.feasible(int(i)):
                pk.take(int(i))
                n_fill += 1
        if log["used_random_utility_fallback"]:
            log["n_random_fallback_sequences"] = n_fill
        else:
            log["n_zero_marginal_random_fill"] = n_fill
    if not pk.full():
        raise ReprInfeasible(f"constrained rounding filled {len(pk.picks)} of {budget} slots")
    take = [cand[i] for i in pk.picks]
    assert len(set(take)) == budget
    d_round = float(d_cand[pk.picks].sum())
    if d_round > cap_d + TOL:
        raise ReprInfeasible(f"round atypicality {d_round:.6f} > cap {cap_d:.6f}")
    log.update(round_d_sum=d_round, round_d_cap=float(cap_d), round_d_mean=d_round / budget)
    return take, pk.picks, log
