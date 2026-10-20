"""P2 gate — does any calibration-state signal predict FUTURE benefit beyond
routed-token exposure?  Read-only on E2's observation table; CPU only.

Stricter than E2's own selection in three ways:

 1. A FLEXIBLE exposure baseline. "Beyond count" must mean beyond any smooth
    function of count and budget, otherwise a signal can "win" just by
    absorbing count non-linearity (E1's benefit is non-monotone in budget).
      count_linear : 1, log n                              (E2's old baseline)
      count_flex   : 1, log n, (log n)^2, (log n)^3, budget dummies,
                     budget x log n                       (budget-aware exposure)
    Signal models add a standardized signal z(s) to count_flex:
      +s           : one common slope
      +s x budget  : a separate slope per starting budget
 2. Two validation axes: leave-one-characterization-seed-out (LOSO) and
    leave-one-layer-out (LOLO). Improvement per fold = 1 - MSE / MSE(count_flex).
 3. A permutation null: the signal is shuffled within (seed, budget, layer,
    count-quintile) strata, which keeps its relation to exposure and destroys
    any extra information; p = share of permutations with pooled-LOSO
    improvement >= the observed one.
 Also: n0-specific slope signs/t-values (stability), and the replicate-noise
 ceiling of the target (how much variance is explainable at all).

Decision (case labels follow the team's plan):
  A           a signal passes with +s: LOSO gain >= --min-gain in every fold,
              LOLO gain > 0 in every fold, permutation p < --alpha, and no two
              budgets with |t| >= 2 disagree in slope sign      -> go to E3
  A_budget    fails A with +s but passes the same tests with +s x budget
              -> go to E3 with the budget-aware predictor
  B           best signal has pooled LOSO gain >= --weak-gain and p < alpha, but
              fails the stability tests
  C           no signal beats flexible exposure                -> report negative finding
Exit code: 0 for A / A_budget, 1 for B, 2 for C.
"""
from __future__ import annotations
import argparse, csv, json, os, sys

import numpy as np

from moe_calib.runid import damp_tag, load_e1_runs

STATE_SIGNALS = ("drift", "relse", "margin", "entropy")


# --------------------------------------------------------------- data

def load_obs(path):
    rows = list(csv.DictReader(open(path)))
    col = lambda k: np.array([float(r[k]) for r in rows])
    d = dict(seed=col("characterization_seed"), layer=col("layer"), budget=col("starting_budget"),
             count=col("routed_count"), y=col("future_rel_delta"),
             **{s: col(s) for s in STATE_SIGNALS})
    ok = np.isfinite(d["y"]) & (d["count"] > 0)
    for s in STATE_SIGNALS:
        ok &= np.isfinite(d[s])
    return {k: v[ok] for k, v in d.items()}, int((~ok).sum())


# ------------------------------------------------------------ designs

class Design:
    """Builds design matrices with all normalisation fitted on the TRAIN rows."""

    def __init__(self, kind, signal=None):
        self.kind, self.signal = kind, signal

    def fit(self, d, idx):
        lc = np.log(d["count"][idx])
        self.lc_mu, self.lc_sd = lc.mean(), lc.std() + 1e-12
        self.budgets = sorted(set(d["budget"][idx].tolist()))
        if self.signal:
            v = d[self.signal][idx]
            self.s_mu, self.s_sd = v.mean(), v.std() + 1e-12
        return self

    def X(self, d, idx):
        lc = (np.log(d["count"][idx]) - self.lc_mu) / self.lc_sd
        cols = [np.ones_like(lc), lc]
        if self.kind not in ("count_linear", "lin_sig"):
            cols += [lc ** 2, lc ** 3]
            dummies = [(d["budget"][idx] == b).astype(float) for b in self.budgets[1:]]
            cols += dummies + [D * lc for D in dummies]
        if self.signal:
            z = (d[self.signal][idx] - self.s_mu) / self.s_sd
            if self.kind == "sig_budget":
                cols += [(d["budget"][idx] == b).astype(float) * z for b in self.budgets]
            else:
                cols.append(z)
        return np.stack(cols, axis=1)


def cv_errors(d, kind, signal, groups):
    """Held-out squared errors for every row under leave-one-group-out."""
    se = np.full(d["y"].size, np.nan)
    per_fold = {}
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        des = Design(kind, signal).fit(d, tr)
        beta, *_ = np.linalg.lstsq(des.X(d, tr), d["y"][tr], rcond=None)
        r = d["y"][te] - des.X(d, te) @ beta
        se[te] = r ** 2
        per_fold[g] = float(np.mean(r ** 2))
    return se, per_fold


def gains(d, kind, signal, groups, base_folds):
    se, folds = cv_errors(d, kind, signal, groups)
    return se, {g: 1 - folds[g] / base_folds[g] for g in folds}


# ------------------------------------------------------- permutation

def strata_ids(d):
    key = np.zeros(d["y"].size, dtype=np.int64)
    for comp in (d["seed"], d["budget"], d["layer"]):
        _, inv = np.unique(comp, return_inverse=True)
        key = key * 1000 + inv
    q = np.zeros_like(key)
    for k in np.unique(key):
        m = key == k
        c = d["count"][m]
        edges = np.quantile(c, [0.2, 0.4, 0.6, 0.8])
        q[m] = np.searchsorted(edges, c, side="right")
    return key * 10 + q


def permutation_p(d, kind, signal, groups, base_se, obs_gain, R, rng):
    strata = strata_ids(d)
    hits = 0
    for _ in range(R):
        dp = dict(d)
        v = d[signal].copy()
        for k in np.unique(strata):
            m = np.flatnonzero(strata == k)
            v[m] = v[rng.permutation(m)]
        dp[signal] = v
        se, _ = cv_errors(dp, kind, signal, groups)
        hits += (1 - se.sum() / base_se.sum()) >= obs_gain
    return (1 + hits) / (1 + R)


# ---------------------------------------------------------- stability

def budget_slopes(d, signal):
    """Within each starting budget: y ~ 1 + lc + lc^2 + z(s); slope and OLS t
    (rows within an expert are treated as independent: t is optimistic)."""
    out = {}
    for b in sorted(set(d["budget"].tolist())):
        m = d["budget"] == b
        lc = np.log(d["count"][m])
        lc = (lc - lc.mean()) / (lc.std() + 1e-12)
        z = (d[signal][m] - d[signal][m].mean()) / (d[signal][m].std() + 1e-12)
        X = np.stack([np.ones_like(lc), lc, lc ** 2, z], axis=1)
        y = d["y"][m]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        dof = max(len(y) - X.shape[1], 1)
        cov = np.linalg.pinv(X.T @ X) * (r @ r / dof)
        out[int(b)] = dict(beta=float(beta[3]), t=float(beta[3] / np.sqrt(max(cov[3, 3], 1e-300))),
                           n=int(m.sum()))
    sig = [v["beta"] for v in out.values() if abs(v["t"]) >= 2]
    consistent = not (any(b > 0 for b in sig) and any(b < 0 for b in sig))
    return out, consistent


# --------------------------------------------------------- noise ceiling

def noise_ceiling(runs, y_var):
    """Share of target variance NOT explained by replicate noise:
    1 - E[Var_rep(rel_delta)] / R / Var(y). An upper bound on achievable R^2."""
    groups = {}
    for run in runs:
        for r in csv.DictReader(open(run["labels"])):
            k = (run["char_seed"], r["layer"], r["expert"], r["n0"])
            groups.setdefault(k, []).append(float(r["rel_delta"]))
    v = [np.var(x, ddof=1) / len(x) for x in groups.values() if len(x) >= 2]
    if not v:
        return None
    return dict(noise_var_of_mean=float(np.mean(v)), target_var=float(y_var),
                explainable_fraction=float(1 - np.mean(v) / y_var))


# ---------------------------------------------------------------- main

def run_gate(d, R=200, min_gain=0.02, alpha=0.05, seed=0, weak_gain=0.005):
    rng = np.random.default_rng(seed)
    axes = {"LOSO": d["seed"], "LOLO": d["layer"]}
    res = dict(n_obs=int(d["y"].size), baselines={}, signals={})
    base_se, base_folds, lin_se = {}, {}, {}
    for ax, g in axes.items():
        se_lin, f_lin = cv_errors(d, "count_linear", None, g)
        se_flex, f_flex = cv_errors(d, "count_flex", None, g)
        base_se[ax], base_folds[ax], lin_se[ax] = se_flex, f_flex, se_lin
        res["baselines"][ax] = dict(
            mse_count_linear=float(se_lin.mean()), mse_count_flex=float(se_flex.mean()),
            flex_gain_over_linear=float(1 - se_flex.sum() / se_lin.sum()),
            r2_count_flex=float(1 - se_flex.mean() / d["y"].var()))
    for s in STATE_SIGNALS:
        res["signals"][s] = {}
        slopes, consistent = budget_slopes(d, s)
        for kind in ("sig", "sig_budget"):
            r = {}
            for ax, g in axes.items():
                se, fg = gains(d, kind, s, g, base_folds[ax])
                r[ax] = dict(pooled_gain=float(1 - se.sum() / base_se[ax].sum()),
                             fold_gains={str(k): v for k, v in fg.items()})
            obs = r["LOSO"]["pooled_gain"]
            r["perm_p_LOSO"] = (permutation_p(d, kind, s, axes["LOSO"], base_se["LOSO"],
                                              obs, R, rng) if obs > 0 and R > 0 else 1.0)
            r["passes"] = bool(all(v >= min_gain for v in r["LOSO"]["fold_gains"].values())
                               and all(v > 0 for v in r["LOLO"]["fold_gains"].values())
                               and r["perm_p_LOSO"] < alpha
                               and (consistent or kind == "sig_budget"))
            res["signals"][s][kind] = r
        se_ls, _ = cv_errors(d, "lin_sig", s, axes["LOSO"])
        res["signals"][s]["old_style_LOSO_gain_vs_linear_count"] = float(
            1 - se_ls.sum() / lin_se["LOSO"].sum())
        res["signals"][s]["budget_slopes"] = slopes
        res["signals"][s]["slope_sign_consistent"] = consistent
    best = max(STATE_SIGNALS, key=lambda s: max(res["signals"][s][k]["LOSO"]["pooled_gain"]
                                                for k in ("sig", "sig_budget")))
    a = [s for s in STATE_SIGNALS if res["signals"][s]["sig"]["passes"]]
    ab = [s for s in STATE_SIGNALS if res["signals"][s]["sig_budget"]["passes"]]
    bs = res["signals"][best]
    weak = any(bs[k]["LOSO"]["pooled_gain"] >= weak_gain and bs[k]["perm_p_LOSO"] < alpha
               for k in ("sig", "sig_budget"))
    if a:
        case, sel = "A", max(a, key=lambda s: res["signals"][s]["sig"]["LOSO"]["pooled_gain"])
    elif ab:
        case, sel = "A_budget", max(ab, key=lambda s: res["signals"][s]["sig_budget"]["LOSO"]["pooled_gain"])
    elif weak:
        case, sel = "B", best
    else:
        case, sel = "C", None
    res.update(case=case, selected_signal=sel, best_signal=best,
               thresholds=dict(min_gain=min_gain, weak_gain=weak_gain, alpha=alpha,
                               permutations=R))
    return res


CASE_TEXT = {
    "A": "a state signal robustly beats flexible exposure -> proceed to E3 with it",
    "A_budget": "a state signal helps only with budget-specific slopes -> proceed to E3 with the "
                "budget-aware predictor (state this in the report)",
    "B": "some gain but unstable across folds/budgets -> do not run E3 yet; investigate",
    "C": "no signal beats flexible exposure -> negative finding on the core hypothesis",
}


def render(res, tag):
    M = [f"# E2 gate ({tag}): CASE {res['case']}", "", CASE_TEXT[res["case"]], "",
         f"{res['n_obs']} observations; thresholds {res['thresholds']}.", ""]
    if res.get("noise_ceiling"):
        nc = res["noise_ceiling"]
        M += [f"Replicate-noise ceiling: at most {100 * nc['explainable_fraction']:.1f}% of target "
              "variance is explainable.", ""]
    M += ["## Exposure baselines", "", "| axis | MSE count_linear | MSE count_flex | flex gain | R2 count_flex |",
          "|---|---|---|---|---|"]
    for ax, b in res["baselines"].items():
        M.append(f"| {ax} | {b['mse_count_linear']:.6f} | {b['mse_count_flex']:.6f} | "
                 f"{100 * b['flex_gain_over_linear']:+.2f}% | {b['r2_count_flex']:.3f} |")
    M += ["", "## Signals (gain = 1 - MSE / MSE(count_flex))", "",
          "| signal | model | LOSO pooled | LOSO folds | LOLO pooled | LOLO folds | perm p | passes |",
          "|---|---|---|---|---|---|---|---|"]
    for s, v in res["signals"].items():
        for kind in ("sig", "sig_budget"):
            r = v[kind]
            fmt = lambda f: ", ".join(f"{100 * x:+.1f}%" for x in f.values())
            M.append(f"| {s} | {'+s' if kind == 'sig' else '+s x budget'} | "
                     f"{100 * r['LOSO']['pooled_gain']:+.2f}% | {fmt(r['LOSO']['fold_gains'])} | "
                     f"{100 * r['LOLO']['pooled_gain']:+.2f}% | {fmt(r['LOLO']['fold_gains'])} | "
                     f"{r['perm_p_LOSO']:.3f} | {'yes' if r['passes'] else 'no'} |")
    M += ["", "Old-style comparison (signal + linear log-count vs linear log-count, LOSO) — what "
          "E2 used to report; compare with the flexible-baseline gains above:", "",
          ", ".join(f"{s} {100 * v['old_style_LOSO_gain_vs_linear_count']:+.2f}%"
                    for s, v in res["signals"].items()), ""]
    M += ["", "## Per-budget slope of each signal (y ~ 1 + lc + lc^2 + z(s), within budget)", "",
          "| signal | " + " | ".join(f"n0={b}" for b in next(iter(res['signals'].values()))["budget_slopes"])
          + " | sign-consistent |", "|---|" + "---|" * (len(next(iter(res['signals'].values()))["budget_slopes"]) + 1)]
    for s, v in res["signals"].items():
        M.append(f"| {s} | " + " | ".join(f"{x['beta']:+.4f} (t={x['t']:+.1f})"
                                          for x in v["budget_slopes"].values())
                 + f" | {'yes' if v['slope_sign_consistent'] else 'NO'} |")
    return "\n".join(M)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--percdamp", type=float, default=None)
    ap.add_argument("--signals-csv", default=None,
                    help="default: <outdir>/e2_signals_b{bits}[_d{damp}].csv")
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--weak-gain", type=float, default=0.005)
    ap.add_argument("--outdir", default="runs")
    a = ap.parse_args(argv)
    tag = f"b{a.bits}" + (f"_{damp_tag(a.percdamp)}" if a.percdamp is not None else "")
    path = a.signals_csv or os.path.join(a.outdir, f"e2_signals_{tag}.csv")
    d, n_drop = load_obs(path)
    if len(np.unique(d["seed"])) < 2:
        raise SystemExit("need >= 2 characterization seeds for LOSO")
    res = run_gate(d, a.permutations, a.min_gain, a.alpha, weak_gain=a.weak_gain)
    res["n_dropped_nonfinite"] = n_drop
    try:
        runs, _ = load_e1_runs(a.outdir, a.bits, None, a.percdamp)
        res["noise_ceiling"] = noise_ceiling(runs, float(d["y"].var()))
    except (FileNotFoundError, RuntimeError) as ex:
        res["noise_ceiling"] = None
        print(f"[E2-gate] noise ceiling skipped: {ex}")
    base = os.path.join(a.outdir, f"e2_gate_{tag}")
    json.dump(res, open(base + ".json", "w"), indent=2, default=float)
    md = render(res, tag)
    open(base + ".md", "w").write(md + "\n")
    print(md)
    return {"A": 0, "A_budget": 0, "B": 1, "C": 2}[res["case"]]


if __name__ == "__main__":
    sys.exit(main())
