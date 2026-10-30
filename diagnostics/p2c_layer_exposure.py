"""P2c — is the marginal value of routed-token exposure layer-dependent?
DIAGNOSTIC ONLY, read-only on the final P2 observation table (CPU; no E1/E2
regeneration; final test split never touched; nothing here feeds E3).
Exposure-only: uses routed count, layer, budget and future_rel_delta; the
state signals (drift / relSE / margin / entropy) are deliberately NOT read.

Nested ladder (lc = log routed count, standardized on the training rows):
  M0  1 + lc                                                    (reference)
  M1  1 + lc + lc^2 + lc^3 + budget dummies + budget x lc       (= P2 count_flex)
  M2  M1 + layer dummies                                        (H1: layer level)
  M3  M2 + layer x lc                                           (H2: layer slope)
  M4  M3 + layer x lc^2                                         (H2: layer curve)
  M4c M4 + layer x lc^3        (kept only if CV prefers it over M4)
  Dlb M2 + layer x budget dummies   DIAGNOSTIC: a layer-specific *budget* curve
      without layer x count. Counts are ~proportional to budget (within-budget
      spread is +-10-20 %), so if Dlb matches M3/M4 the "layer curve" is
      identified from between-budget shifts, not from within-budget exposure.
  DlbS Dlb + layer x lc             DIAGNOSTIC: does within-budget exposure have a
      layer-specific slope once each (layer, budget) has its own mean?

Validation
  LOSO   leave one characterization seed out (primary)
  EXPERT 5-fold grouped CV on (layer, expert), balanced within layer, all seeds
         of an expert held out together (no expert-id feature anywhere)
  JOINT  hold out seed s AND expert fold k together (strictest; supplementary)
  Uncertainty: cluster bootstrap over (layer, expert) of the held-out gain.
  Interaction nulls (M2 -> M3/M4):
    Freedman-Lane: y* = yhat_M2 + residuals permuted within (seed, budget,
      count-quintile); the reduced model has no interaction by construction.
    Layer-label shuffle within (seed, budget, count-quintile) (secondary: it
      destroys the layer intercept too).
  In-sample nested F tests are reported as secondary (rows are not independent).

Ranking (fixed budget, held-out seed; models fit on the other seeds):
  coverage 1/n,  global max(f_M1, 0)/n,  layer-intercept max(f_M2, 0)/n,
  layer-aware max(f_best, 0)/n;  oracle y/n (realized E1 benefit per token).
  Spearman vs oracle, captured share of the oracle top-q% utility (q = 10/25/50).

Gate (exactly one): LA-A / LA-B / LA-C / LA-D, see decide().
"""
from __future__ import annotations
import argparse, csv, json, os, sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p2b_regime_diagnostic import check_table, fold_gains, gain   # noqa: E402

MODELS = ("M0", "M1", "M2", "M3", "M4", "M4c", "Dlb", "DlbS")
LAYER_AWARE = ("M3", "M4", "M4c")
POINTS = {16: 500, 32: 1000, 64: 2000, 128: 4000}     # representative n per budget
QS = (0.10, 0.25, 0.50)


# --------------------------------------------------------------- data

def load(path):
    rows = list(csv.DictReader(open(path)))
    col = lambda k: np.array([float(r[k]) for r in rows])
    d = dict(seed=col("characterization_seed"), layer=col("layer"), budget=col("starting_budget"),
             expert=col("expert"), count=col("routed_count"), y=col("future_rel_delta"))
    ok = np.isfinite(d["y"]) & (d["count"] > 0)
    return {k: v[ok] for k, v in d.items()}, int((~ok).sum())


def sub(d, m):
    return {k: v[m] for k, v in d.items()}


# ------------------------------------------------------------ designs

class LDesign:
    """Design matrix for one ladder model; normalisation fitted on TRAIN rows."""

    def __init__(self, name):
        self.name = name

    def fit(self, d):
        lc = np.log(d["count"])
        self.mu, self.sd = lc.mean(), lc.std() + 1e-12
        self.budgets = sorted(set(d["budget"].tolist()))
        self.layers = sorted(set(d["layer"].tolist()))
        return self

    def X(self, d):
        lc = (np.log(d["count"]) - self.mu) / self.sd
        cols = [np.ones_like(lc), lc]
        if self.name == "M0":
            return np.stack(cols, 1)
        B = [(d["budget"] == b).astype(float) for b in self.budgets[1:]]
        cols += [lc ** 2, lc ** 3] + B + [x * lc for x in B]
        if self.name == "M1":
            return np.stack(cols, 1)
        L = [(d["layer"] == l).astype(float) for l in self.layers[1:]]
        cols += L
        if self.name in ("Dlb", "DlbS"):
            cols += [x * y for x in L for y in B]
        powers = {"M3": 1, "M4": 2, "M4c": 3, "DlbS": 1}.get(self.name, 0)
        for p in range(1, powers + 1):
            cols += [x * lc ** p for x in L]
        return np.stack(cols, 1)


def fit(d, name):
    des = LDesign(name).fit(d)
    beta, *_ = np.linalg.lstsq(des.X(d), d["y"], rcond=None)
    return lambda dd: des.X(dd) @ beta


def cv_pred(d, name, folds):
    pred = np.full(d["y"].size, np.nan)
    for tr, te in folds:
        pred[te] = fit(sub(d, tr), name)(sub(d, te))
    return pred


def loso_folds(d):
    return [(d["seed"] != s, d["seed"] == s) for s in np.unique(d["seed"])]


def expert_fold_ids(d, k=5, seed=0):
    """Deterministic grouped folds on (layer, expert), balanced within each layer."""
    rng = np.random.default_rng(seed)
    f = np.zeros(d["y"].size, int)
    for L in np.unique(d["layer"]):
        ex = np.unique(d["expert"][d["layer"] == L])
        order = rng.permutation(ex)
        for i, e in enumerate(order):
            f[(d["layer"] == L) & (d["expert"] == e)] = i % k
    return f


def expert_folds(fid):
    return [(fid != k, fid == k) for k in np.unique(fid)]


def joint_folds(d, fid):
    return [((d["seed"] != s) & (fid != k), (d["seed"] == s) & (fid == k))
            for s in np.unique(d["seed"]) for k in np.unique(fid)]


# --------------------------------------------------------- statistics

def cluster_ids(d):
    _, inv = np.unique(np.stack([d["layer"], d["expert"]], 1), axis=0, return_inverse=True)
    return inv.ravel()


def boot_gain(se_a, se_b, cl, B=2000, rng=None):
    """95 % cluster-bootstrap CI of 1 - SSE_a / SSE_b over (layer, expert)."""
    rng = rng or np.random.default_rng(0)
    A, Bs = np.bincount(cl, se_a), np.bincount(cl, se_b)
    idx = rng.integers(0, A.size, (B, A.size))
    g = 1 - A[idx].sum(1) / Bs[idx].sum(1)
    return [float(np.quantile(g, 0.025)), float(np.quantile(g, 0.975))]


def strata(d):
    key = (d["seed"] * 1000 + d["budget"]).astype(np.int64)
    q = np.zeros_like(key)
    for k in np.unique(key):
        m = key == k
        c = d["count"][m]
        q[m] = np.searchsorted(np.quantile(c, [0.2, 0.4, 0.6, 0.8]), c, side="right")
    return key * 10 + q


def shuffle_within(v, st, rng):
    v = v.copy()
    for k in np.unique(st):
        m = np.flatnonzero(st == k)
        v[m] = v[rng.permutation(m)]
    return v


def loso_gain(d, a, b):
    folds = loso_folds(d)
    return gain((d["y"] - cv_pred(d, a, folds)) ** 2, (d["y"] - cv_pred(d, b, folds)) ** 2)


def interaction_nulls(d, models, obs, R, rng):
    st = strata(d)
    yhat = fit(d, "M2")(d)
    res = d["y"] - yhat
    out = {}
    for m in models:
        fl = lab = 0
        for _ in range(R):
            dp = dict(d, y=yhat + shuffle_within(res, st, rng))
            fl += loso_gain(dp, m, "M2") >= obs[m]
            dl = dict(d, layer=shuffle_within(d["layer"], st, rng))
            lab += loso_gain(dl, m, "M2") >= obs[m]
        out[m] = dict(freedman_lane_p=(1 + fl) / (1 + R), layer_shuffle_p=(1 + lab) / (1 + R))
    return out


def f_tests(d):
    out, n = {}, d["y"].size
    rss = {}
    for m in ("M1", "M2", "M3", "M4", "M4c", "Dlb", "DlbS"):
        des = LDesign(m).fit(d)
        X = des.X(d)
        beta, *_ = np.linalg.lstsq(X, d["y"], rcond=None)
        rss[m] = (float(((d["y"] - X @ beta) ** 2).sum()), np.linalg.matrix_rank(X))
    for a, b in (("M1", "M2"), ("M2", "M3"), ("M3", "M4"), ("M4", "M4c"), ("M2", "Dlb"), ("Dlb", "DlbS")):
        (ra, pa), (rb, pb) = rss[a], rss[b]
        F = ((ra - rb) / (pb - pa)) / (rb / (n - pb))
        out[f"{a}->{b}"] = dict(F=float(F), df=[int(pb - pa), int(n - pb)],
                                p=float(stats.f.sf(F, pb - pa, n - pb)))
    return out


# ------------------------------------------------------------- blocks

def cv_block(d, folds, groups, models, base="M1"):
    se = {m: (d["y"] - cv_pred(d, m, folds)) ** 2 for m in models}
    out = {}
    for m in models:
        out[m] = dict(mse=float(se[m].mean()),
                      r2=float(1 - se[m].sum() / ((d["y"] - d["y"].mean()) ** 2).sum()),
                      mse_by_group={str(int(g)): float(se[m][groups == g].mean()) for g in np.unique(groups)},
                      gain_vs_M1=gain(se[m], se[base]),
                      fold_gains_vs_M1=fold_gains(se[m], se[base], groups),
                      gain_by_layer_vs_M1=fold_gains(se[m], se[base], d["layer"]))
        if m not in ("M0", "M1"):
            out[m]["gain_vs_M2"] = gain(se[m], se["M2"])
            out[m]["fold_gains_vs_M2"] = fold_gains(se[m], se["M2"], groups)
    return out, se


def curves(d, models, sel):
    """f_L(n) and u_L(n) = max(f, 0)/n at each budget's representative n, only
    inside the observed (layer, budget) count range; full data and per seed."""
    def table(dd, m):
        f = fit(dd, m)
        t = {}
        for b, n in POINTS.items():
            row = {}
            for L in np.unique(d["layer"]):
                c = d["count"][(d["layer"] == L) & (d["budget"] == b)]
                if not (c.min() <= n <= c.max()):
                    row[str(int(L))] = None
                    continue
                q = dict(count=np.array([float(n)]), budget=np.array([float(b)]),
                         layer=np.array([float(L)]))
                v = float(f(q)[0])
                row[str(int(L))] = dict(f=v, u=max(v, 0.0) / n)
            ok = {k: v for k, v in row.items() if v}
            fv = [v["f"] for v in ok.values()]
            t[str(b)] = dict(n=n, layers=row,
                             order=sorted(ok, key=lambda k: -ok[k]["f"]) if np.ptp(fv) > 1e-9 else [],
                             spread_f=float(max(v["f"] for v in ok.values()) - min(v["f"] for v in ok.values())),
                             rel_spread_f=float((max(v["f"] for v in ok.values()) - min(v["f"] for v in ok.values()))
                                                / np.mean([v["f"] for v in ok.values()])))
        return t
    out = {m: table(d, m) for m in models}
    out["per_seed"] = {str(int(s)): table(sub(d, d["seed"] == s), sel) for s in np.unique(d["seed"])}
    out["order_stable_across_seeds"] = {
        b: len({tuple(out["per_seed"][s][b]["order"]) for s in out["per_seed"]}) == 1 for b in map(str, POINTS)}
    out["top_layer_by_budget"] = {b: out[sel][b]["order"][0] for b in map(str, POINTS)}
    return out


def ranking(d, sel):
    rules = {"coverage": None, "global": "M1", "layer_intercept": "M2",
             "layer_budget_means": "Dlb", "layer_aware": sel}
    res = {r: {} for r in rules}
    for s in np.unique(d["seed"]):
        tr = d["seed"] != s
        fs = {r: fit(sub(d, tr), m) for r, m in rules.items() if m}
        for b in np.unique(d["budget"]):
            te = sub(d, (d["seed"] == s) & (d["budget"] == b))
            n, y = te["count"], te["y"]
            oracle = y / n
            for r, m in rules.items():
                if m is None:
                    u, f = 1 / n, None
                else:
                    f = fs[r](te)
                    u = np.maximum(f, 0) / n
                cell = dict(spearman_u=float(stats.spearmanr(u, oracle)[0]),
                            spearman_benefit=None if f is None else float(stats.spearmanr(f, y)[0]))
                for q in QS:
                    k = max(1, int(round(q * n.size)))
                    best = np.sort(oracle)[::-1][:k].sum()
                    cell[f"capture@{int(100 * q)}"] = float(oracle[np.argsort(-u, kind="stable")[:k]].sum() / best)
                res[r][f"s{int(s)}_n{int(b)}"] = cell
    summ = {}
    for r, cells in res.items():
        keys = [k for k in next(iter(cells.values())) if cells[next(iter(cells))][k] is not None]
        summ[r] = {k: float(np.mean([c[k] for c in cells.values()])) for k in keys}
        summ[r]["mean_capture"] = float(np.mean([summ[r][f"capture@{int(100 * q)}"] for q in QS]))
        summ[r]["by_seed_mean_capture"] = {
            s: float(np.mean([c[f"capture@{int(100 * q)}"] for k, c in cells.items() if k.startswith(f"s{s}_")
                              for q in QS])) for s in sorted({k[1:].split("_")[0] for k in cells})}
    return dict(cells=res, summary=summ)


# ----------------------------------------------------------- decision

def decide(L, E, rk, nulls, sel, min_gain, alpha, catastrophic=0.02):
    m2, ms = L["M2"], L[sel]
    c = {}
    c["1_M2_beats_M1_every_seed"] = all(v > 0 for v in m2["fold_gains_vs_M1"].values())
    c["2_layer_curve_beats_M2"] = bool(ms["gain_vs_M2"] >= min_gain
                                       and all(v > 0 for v in ms["fold_gains_vs_M2"].values())
                                       and nulls[sel]["freedman_lane_p"] < alpha)
    c["3_layer_aware_beats_M1_every_seed"] = all(v > 0 for v in ms["fold_gains_vs_M1"].values())
    c["4_expert_cv_positive"] = bool(E[sel]["gain_vs_M1"] > 0
                                     and all(v > 0 for v in E[sel]["fold_gains_vs_M1"].values()))
    S = rk["summary"]
    la, cov, glo = S["layer_aware"], S["coverage"], S["global"]
    c["5_ranking_beats_coverage_and_global"] = bool(
        la["mean_capture"] > cov["mean_capture"] and la["mean_capture"] > glo["mean_capture"]
        and all(la["by_seed_mean_capture"][s] >= cov["by_seed_mean_capture"][s] - catastrophic
                and la["by_seed_mean_capture"][s] >= glo["by_seed_mean_capture"][s] - catastrophic
                for s in la["by_seed_mean_capture"]))
    c["6_not_intercept_only"] = bool(c["2_layer_curve_beats_M2"] and E[sel]["gain_vs_M2"] > 0)
    predictive = c["2_layer_curve_beats_M2"] and c["3_layer_aware_beats_M1_every_seed"] \
        and c["4_expert_cv_positive"] and c["6_not_intercept_only"]
    if predictive and c["5_ranking_beats_coverage_and_global"] and c["1_M2_beats_M1_every_seed"]:
        case = "LA-A"
    elif predictive:
        case = "LA-C"
    elif c["1_M2_beats_M1_every_seed"] and m2["gain_vs_M1"] >= min_gain:
        case = "LA-B"
    else:
        case = "LA-D"
    return dict(case=case, selected_layer_model=sel, criteria=c, min_gain=min_gain, alpha=alpha,
                catastrophic_capture_drop=catastrophic)


CASE_TEXT = {
    "LA-A": "Layer-aware marginal-benefit hypothesis supported — proceed to minimal allocation experiment.",
    "LA-B": "Layers differ in baseline calibration benefit, but evidence for layer-specific marginal "
            "exposure curves is weak. Do not call the method layer-aware marginal-benefit allocation; "
            "consider a simpler layer-weighted coverage baseline.",
    "LA-C": "Layer-conditioned prediction improves fit but does not provide useful allocation priorities. "
            "Do not proceed to E3.",
    "LA-D": "Layer-aware exposure hypothesis not supported.",
}


# ------------------------------------------------------------- render

pct = lambda x: f"{100 * x:+.1f}%"
folds_s = lambda f: ", ".join(f"{k}:{pct(v)}" for k, v in f.items())


def render(res):
    D, L, E, J = res["decision"], res["loso"], res["expert_cv"], res["joint_cv"]
    sel = D["selected_layer_model"]
    M = [f"# P2c layer-aware exposure: {D['case']}", "", CASE_TEXT[D["case"]], "",
         f"Input `{res['input']}`: {res['n_obs']} rows, {res['design']}. Exposure-only (no state signals). "
         f"Selected layer-aware model (best LOSO gain over M2 among {', '.join(LAYER_AWARE)}): **{sel}**.", "",
         "## Gate criteria", "", "| criterion | pass |", "|---|---|"]
    M += [f"| {k} | {'yes' if v else 'NO'} |" for k, v in D["criteria"].items()]
    M += ["", "## LOSO (primary)", "",
          "| model | MSE | R2 | MSE by seed | gain vs M1 [95% CI] | seed gains vs M1 | gain vs M2 [95% CI] | seed gains vs M2 |",
          "|---|---|---|---|---|---|---|---|"]
    for m, r in L.items():
        ci1, ci2 = res["bootstrap"]["vs_M1"].get(m), res["bootstrap"]["vs_M2"].get(m)
        M.append(f"| {m} | {r['mse']:.3e} | {r['r2']:.3f} | "
                 + ", ".join(f"{k}:{v:.2e}" for k, v in r["mse_by_group"].items())
                 + f" | {pct(r['gain_vs_M1'])}" + (f" [{pct(ci1[0])}, {pct(ci1[1])}]" if ci1 else "")
                 + f" | {folds_s(r['fold_gains_vs_M1'])} | "
                 + (f"{pct(r['gain_vs_M2'])}" + (f" [{pct(ci2[0])}, {pct(ci2[1])}]" if ci2 else "")
                    + f" | {folds_s(r['fold_gains_vs_M2'])} |" if "gain_vs_M2" in r else "- | - |"))
    dec = res["decomposition"]
    M += ["", "### Intercept vs curve (LOSO, MSE)", "",
          f"- M1 -> M2 (layer intercept): {pct(dec['M1_to_M2_rel'])} of M1's MSE "
          f"(absolute {dec['M1_to_M2_abs']:.3e}).",
          f"- M2 -> {sel} (layer-specific curve): {pct(dec['M2_to_sel_rel'])} of M2's MSE "
          f"(absolute {dec['M2_to_sel_abs']:.3e}).",
          f"- Share of the total M1 -> {sel} MSE reduction due to the intercept: "
          f"{100 * dec['intercept_share']:.0f}%; due to the curve: {100 * (1 - dec['intercept_share']):.0f}%.",
          f"- Diagnostic Dlb (layer x budget means, no layer x count): gain vs M2 {pct(L['Dlb']['gain_vs_M2'])}"
          f" vs {sel} {pct(L[sel]['gain_vs_M2'])}; DlbS (Dlb + layer x lc) vs Dlb: "
          f"{pct(1 - L['DlbS']['mse'] / L['Dlb']['mse'])} (seed folds "
          + ", ".join(f"{k}:{pct(1 - L['DlbS']['mse_by_group'][k] / L['Dlb']['mse_by_group'][k])}"
                      for k in L['Dlb']['mse_by_group']) + ").", "",
          "Interaction nulls (LOSO gain over M2; secondary in-sample F below):", ""]
    for m, v in res["interaction_nulls"].items():
        M.append(f"- {m}: Freedman–Lane p = {v['freedman_lane_p']:.3f}, layer-label shuffle p = "
                 f"{v['layer_shuffle_p']:.3f} ({res['permutations']} perms)")
    M += ["", "In-sample nested F (rows not independent -> optimistic): " +
          "; ".join(f"{k} F={v['F']:.1f} (df {v['df'][0]},{v['df'][1]}) p={v['p']:.1e}"
                    for k, v in res["f_tests"].items()), ""]
    for title, blk, grp in (("Expert-grouped CV (5 folds on (layer, expert); all seeds held out together)", E, "fold"),
                            ("Joint CV (held-out seed AND held-out experts; supplementary)", J, "seed")):
        M += [f"## {title}", "", f"| model | MSE | gain vs M1 | {grp} gains vs M1 | gain vs M2 | gain by layer vs M1 |",
              "|---|---|---|---|---|---|"]
        for m, r in blk.items():
            M.append(f"| {m} | {r['mse']:.3e} | {pct(r['gain_vs_M1'])} | {folds_s(r['fold_gains_vs_M1'])} | "
                     f"{pct(r['gain_vs_M2']) if 'gain_vs_M2' in r else '-'} | {folds_s(r['gain_by_layer_vs_M1'])} |")
        M.append("")
    C = res["curves"]
    M += ["## Layer curves f_L(n) and u_L(n) = max(f,0)/n (full-data fits; '-' = outside support)", "",
          "| model | n0 | n | " + " | ".join(f"L{l} f (u x1e5)" for l in res["design"]["layer"])
          + " | order (high -> low) | spread (rel) |", "|---|---|---|" + "---|" * (len(res["design"]["layer"]) + 2)]
    for m in ("M1", "M2", sel):
        for b, t in C[m].items():
            cells = [f"{v['f']:.4f} ({1e5 * v['u']:.2f})" if v else "-" for v in t["layers"].values()]
            M.append(f"| {m} | {b} | {t['n']} | " + " | ".join(cells)
                     + f" | {' > '.join('L' + x for x in t['order']) or 'all equal'} | {t['spread_f']:.4f} ({pct(t['rel_spread_f'])}) |")
    M += ["", f"Per-seed {sel} fits — layer order at each budget's n:", ""]
    for b in map(str, POINTS):
        M.append(f"- n0={b}: " + "; ".join(f"seed {s}: {' > '.join('L' + x for x in C['per_seed'][s][b]['order'])}"
                                           for s in C["per_seed"])
                 + f" -> {'stable' if C['order_stable_across_seeds'][b] else 'CHANGES'}")
    M += ["", f"Top layer by budget ({sel}): " + ", ".join(f"n0={b}: L{v}" for b, v in C["top_layer_by_budget"].items()),
          "", "## Expert ranking at fixed budget (held-out seed; oracle = realized future_rel_delta / n)", "",
          "| rule | Spearman(u, oracle) | Spearman(f, y) | capture@10 | capture@25 | capture@50 | mean | mean by seed |",
          "|---|---|---|---|---|---|---|---|"]
    for r, s in res["ranking"]["summary"].items():
        M.append(f"| {r} | {s['spearman_u']:+.3f} | {s.get('spearman_benefit', float('nan')):+.3f} | "
                 f"{s['capture@10']:.3f} | {s['capture@25']:.3f} | {s['capture@50']:.3f} | {s['mean_capture']:.3f} | "
                 + ", ".join(f"{k}:{v:.3f}" for k, v in s["by_seed_mean_capture"].items()) + " |")
    M += ["", "Per (seed, budget) mean capture (layer_aware - coverage): " +
          ", ".join(f"{k} {np.mean([res['ranking']['cells']['layer_aware'][k][f'capture@{int(100 * q)}'] - res['ranking']['cells']['coverage'][k][f'capture@{int(100 * q)}'] for q in QS]):+.3f}"
                    for k in res["ranking"]["cells"]["coverage"]),
          "", "Within a fixed budget the global model's f depends on n only, so `global` ranks like "
          "`coverage` whenever f(n)/n is monotone; any difference from coverage must come from layer terms.",
          "", "## Caveats", ""] + [f"- {x}" for x in res["caveats"]]
    return "\n".join(M)


# --------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-csv", default="runs/e2_signals_b4_d1.csv")
    ap.add_argument("--out", default="runs/p2c_layer_exposure")
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--min-gain", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    d, n_drop = load(a.signals_csv)
    design = check_table(d)
    if n_drop:
        raise SystemExit(f"{n_drop} non-finite rows in the P2 table")
    rng = np.random.default_rng(a.seed)
    res = dict(input=a.signals_csv, n_obs=int(d["y"].size), design=design, permutations=a.permutations)

    L, se = cv_block(d, loso_folds(d), d["seed"], MODELS)
    # M4c is kept as a candidate only if CV prefers it over M4
    cands = [m for m in LAYER_AWARE if m != "M4c" or L["M4c"]["mse"] < L["M4"]["mse"]]
    sel = max(cands, key=lambda m: L[m]["gain_vs_M2"])
    res["loso"] = L
    cl = cluster_ids(d)
    res["bootstrap"] = {
        "vs_M1": {m: boot_gain(se[m], se["M1"], cl, a.bootstrap, rng) for m in MODELS if m != "M1"},
        "vs_M2": {m: boot_gain(se[m], se["M2"], cl, a.bootstrap, rng) for m in (*LAYER_AWARE, "Dlb")}}
    t1, t2, ts = se["M1"].sum(), se["M2"].sum(), se[sel].sum()
    n = d["y"].size
    res["decomposition"] = dict(M1_to_M2_rel=float(1 - t2 / t1), M1_to_M2_abs=float((t1 - t2) / n),
                                M2_to_sel_rel=float(1 - ts / t2), M2_to_sel_abs=float((t2 - ts) / n),
                                intercept_share=float((t1 - t2) / (t1 - ts)) if t1 > ts else None)
    obs = {m: L[m]["gain_vs_M2"] for m in ("M3", "M4")}
    res["interaction_nulls"] = interaction_nulls(d, ("M3", "M4"), obs, a.permutations, rng)
    if sel == "M4c":
        res["interaction_nulls"]["M4c"] = interaction_nulls(d, ("M4c",), {"M4c": L["M4c"]["gain_vs_M2"]},
                                                            a.permutations, rng)["M4c"]
    res["f_tests"] = f_tests(d)
    fid = expert_fold_ids(d, 5, a.seed)
    res["expert_cv"] = cv_block(d, expert_folds(fid), fid, ("M1", "M2", "M3", "M4", "Dlb", "DlbS"))[0]
    if sel not in res["expert_cv"]:
        res["expert_cv"][sel] = cv_block(d, expert_folds(fid), fid, ("M1", "M2", sel))[0][sel]
    res["joint_cv"] = cv_block(d, joint_folds(d, fid), d["seed"], ("M1", "M2", sel, "Dlb"))[0]
    res["curves"] = curves(d, ("M1", "M2", sel), sel)
    res["ranking"] = ranking(d, sel)
    res["decision"] = decide(L, res["expert_cv"], res["ranking"], res["interaction_nulls"], sel,
                             a.min_gain, a.alpha)
    res["caveats"] = [
        "Routed count is ~proportional to budget (within-budget spread +-10-20 %), so a layer-specific "
        "exposure curve is identified mostly from between-budget changes of each layer's mean benefit; "
        "compare with Dlb, which models exactly that without using count.",
        f"{sel} was selected on the same LOSO folds it is scored on (mild optimism for the M2 -> {sel} gain); "
        "expert-grouped and joint CV are not used for selection.",
        "Only layers 1/8/16/23 are characterized; a production allocator over all 24 layers needs either "
        "per-layer characterization or a parameter-sharing scheme across layers.",
        "Ranking assumes exposure can be given to experts directly; it ignores sequence-routing coupling "
        "and is not a substitute for E3.",
        "The oracle utility y/n shares the 1/n factor with every rule, so Spearman(u, oracle) is inflated "
        "for all rules alike; Spearman(f, y) and capture differences are the informative comparisons."]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=2)
    md = render(res)
    open(a.out + ".md", "w").write(md + "\n")
    print(md)
    print(f"\nP2c decision: {res['decision']['case']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
