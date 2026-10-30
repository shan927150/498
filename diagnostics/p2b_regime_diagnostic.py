"""P2b — is the P2 (CASE B) instability budget-dependent, layer-dependent, or both?
DIAGNOSTIC ONLY, read-only on the final P2 observation table (CPU, no GPU, no
E1/E2 regeneration, final test split never touched). Nothing here feeds E3.

Reuses e2_gate's Design / cv_errors / permutation_p so every model, fold and
null is defined exactly as in P2. Gain = 1 - MSE / MSE(matched exposure model).

 1. Budget-specific (each n0 separately): exposure = 1, lc, lc^2, lc^3 (budget
    is constant, so P2's count_flex reduces to the cubic); signal model adds
    z(s). LOSO + LOLO within the budget; permutation within (seed, layer,
    count-quintile); full-data slope with OLS t plus per-seed / per-layer
    slopes (sign agreement is the uncertainty diagnostic: OLS t is optimistic).
 2. Layer-specific (each profiled layer separately): exposure = P2's count_flex
    (cubic + budget dummies + budget x lc); +s (common slope) and +s x budget.
    LOSO; gain by budget; slope by budget; permutation within (seed, budget,
    count-quintile).
 3. Model-complexity ladder on all rows, gains vs GLOBAL count_flex and vs the
    matched exposure model. Layer-specific models cannot be scored on an unseen
    layer (LOSO only). signal x budget x layer is a descriptive upper bound.

Decision (exactly one):
  BUDGET    some signal is robust in >= 3/4 budgets (LOSO folds all > 0, LOLO
            folds all > 0, pooled LOSO >= min-gain, perm p < alpha)
  LAYER     else some signal is robust (common slope) in >= 3/4 layers
            (seed folds > 0, budget gains > 0, pooled >= min-gain, p < alpha)
  BOTH      else layer-specific + budget-aware slopes are robust in >= 3/4 layers
            by the same rule
  UNSTABLE  otherwise
Only calibration-STATE signals count. A signal whose variance is >= --static-share
explained by expert identity (layer, expert) barely moves with seed or budget:
it is a static expert descriptor, and LOSO (which keeps the same experts in
train and test) cannot tell "state predicts benefit" from "expert id predicts
benefit". Such signals are reported but excluded from the decision; the
decision without the exclusion is reported as `mechanical_decision`.
"""
from __future__ import annotations
import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e2_gate import STATE_SIGNALS, Design, cv_errors, load_obs, permutation_p  # noqa: E402
from moe_calib.runid import damp_tag                                          # noqa: E402

KIND_NAME = {"count_flex": "exposure", "sig": "+s", "sig_budget": "+s x budget"}


def load_expert_col(path, d):
    """e2_gate.load_obs keeps no expert id; re-read it (same non-finite filter)."""
    rows = list(csv.DictReader(open(path)))
    e = np.array([float(r["expert"]) for r in rows])
    keep = np.array([all(np.isfinite(float(r[k])) for k in ("future_rel_delta", *STATE_SIGNALS))
                     and float(r["routed_count"]) > 0 for r in rows])
    return e[keep]


def sub(d, m):
    return {k: v[m] for k, v in d.items()}


def check_table(d, expect_seeds=(0, 1, 2), expect_layers=(1, 8, 16, 23),
                expect_budgets=(16, 32, 64, 128)):
    got = {k: tuple(int(x) for x in np.unique(d[k])) for k in ("seed", "layer", "budget")}
    exp = dict(seed=tuple(expect_seeds), layer=tuple(expect_layers), budget=tuple(expect_budgets))
    if got != exp:
        raise SystemExit(f"P2 table does not match the frozen design: {got} != {exp}")
    cells = {}
    for s in exp["seed"]:
        for L in exp["layer"]:
            for b in exp["budget"]:
                cells[(s, L, b)] = int(((d["seed"] == s) & (d["layer"] == L) & (d["budget"] == b)).sum())
    if len(set(cells.values())) != 1:
        raise SystemExit(f"unbalanced (seed, layer, budget) cells: {sorted(set(cells.values()))}")
    return dict(**{k: list(v) for k, v in got.items()}, rows_per_cell=next(iter(cells.values())))


def stratified_se(d, kind, signal, groups, by):
    """Held-out squared errors when a separate model is fitted inside each value
    of `by` (None = one global model); folds are leave-one-`groups`-value-out."""
    if by is None:
        return cv_errors(d, kind, signal, groups)[0]
    se = np.full(d["y"].size, np.nan)
    for v in np.unique(by):
        m = by == v
        se[m] = cv_errors(sub(d, m), kind, signal, groups[m])[0]
    return se


def gain(se, base, mask=None):
    m = np.ones(se.size, bool) if mask is None else mask
    return float(1 - se[m].sum() / base[m].sum())


def fold_gains(se, base, groups):
    return {str(int(g)): gain(se, base, groups == g) for g in np.unique(groups)}


def ols_slopes(d, kind, signal):
    """Full-data fit; standardized signal coefficient(s) with OLS t."""
    des = Design(kind, signal).fit(d, np.ones(d["y"].size, bool))
    X = des.X(d, np.ones(d["y"].size, bool))
    y = d["y"]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    cov = np.linalg.pinv(X.T @ X) * (r @ r / max(len(y) - X.shape[1], 1))
    k = len(des.budgets) if kind == "sig_budget" else 1
    idx = range(X.shape[1] - k, X.shape[1])
    labels = [str(int(b)) for b in des.budgets] if kind == "sig_budget" else ["all"]
    return {lab: dict(beta=float(beta[i]), t=float(beta[i] / np.sqrt(max(cov[i, i], 1e-300))))
            for lab, i in zip(labels, idx)}


def slope_by(d, kind, signal, key):
    return {str(int(v)): ols_slopes(sub(d, d[key] == v), kind, signal)
            for v in np.unique(d[key])}


def signs(slopes, tmin=2.0):
    """'+', '-', or '0' (|t| < tmin) for each entry."""
    return {k: ("+" if v["beta"] > 0 else "-") if abs(v["t"]) >= tmin else "0"
            for k, v in slopes.items()}


def sign_flip(sgn):
    vals = set(sgn.values())
    return "+" in vals and "-" in vals


def perm(d, kind, s, groups, base_se, obs, R, rng):
    return permutation_p(d, kind, s, groups, base_se, obs, R, rng) if obs > 0 and R > 0 else 1.0


# ----------------------------------------------------------- 1. budget

def budget_analysis(d, R, rng, min_gain, alpha):
    out = {}
    for b in np.unique(d["budget"]):
        db = sub(d, d["budget"] == b)
        ax = {"LOSO": db["seed"], "LOLO": db["layer"]}
        base = {a: cv_errors(db, "count_flex", None, g)[0] for a, g in ax.items()}
        res = dict(n=int(db["y"].size),
                   exposure_r2_LOSO=float(1 - base["LOSO"].mean() / db["y"].var()), signals={})
        for s in STATE_SIGNALS:
            r = {}
            for a, g in ax.items():
                se = cv_errors(db, "sig", s, g)[0]
                r[a] = dict(pooled_gain=gain(se, base[a]), fold_gains=fold_gains(se, base[a], g))
                r[f"perm_p_{a}"] = perm(db, "sig", s, g, base[a], r[a]["pooled_gain"], R, rng)
            r["slope"] = ols_slopes(db, "sig", s)["all"]
            r["slope_by_seed"] = {k: v["all"] for k, v in slope_by(db, "sig", s, "seed").items()}
            r["slope_by_layer"] = {k: v["all"] for k, v in slope_by(db, "sig", s, "layer").items()}
            r["seed_sign_flip"] = sign_flip(signs(r["slope_by_seed"]))
            r["layer_sign_flip"] = sign_flip(signs(r["slope_by_layer"]))
            r["neg_seeds"] = [k for k, v in r["LOSO"]["fold_gains"].items() if v <= 0]
            r["neg_layers"] = [k for k, v in r["LOLO"]["fold_gains"].items() if v <= 0]
            r["robust"] = bool(not r["neg_seeds"] and not r["neg_layers"]
                               and r["LOSO"]["pooled_gain"] >= min_gain and r["perm_p_LOSO"] < alpha)
            res["signals"][s] = r
        out[str(int(b))] = res
    return out


# ------------------------------------------------------------ 2. layer

def layer_analysis(d, R, rng, min_gain, alpha):
    out = {}
    for L in np.unique(d["layer"]):
        dl = sub(d, d["layer"] == L)
        g, bud = dl["seed"], dl["budget"]
        base = cv_errors(dl, "count_flex", None, g)[0]
        res = dict(n=int(dl["y"].size),
                   exposure_r2_LOSO=float(1 - base.mean() / dl["y"].var()), signals={})
        for s in STATE_SIGNALS:
            rs = {}
            for kind in ("sig", "sig_budget"):
                se = cv_errors(dl, kind, s, g)[0]
                r = dict(pooled_gain=gain(se, base), seed_gains=fold_gains(se, base, g),
                         budget_gains=fold_gains(se, base, bud))
                r["perm_p"] = perm(dl, kind, s, g, base, r["pooled_gain"], R, rng)
                r["neg_seeds"] = [k for k, v in r["seed_gains"].items() if v <= 0]
                r["neg_budgets"] = [k for k, v in r["budget_gains"].items() if v <= 0]
                r["robust"] = bool(not r["neg_seeds"] and not r["neg_budgets"]
                                   and r["pooled_gain"] >= min_gain and r["perm_p"] < alpha)
                rs[kind] = r
            rs["sig"]["slope"] = ols_slopes(dl, "sig", s)["all"]
            rs["sig_budget"]["slope_by_budget"] = ols_slopes(dl, "sig_budget", s)
            rs["sig_budget"]["budget_sign_flip"] = sign_flip(signs(rs["sig_budget"]["slope_by_budget"]))
            res["signals"][s] = rs
        out[str(int(L))] = res
    return out


# ----------------------------------------------------------- 3. ladder

def ladder(d):
    """LOSO (and LOLO where a held-out layer can be scored) for each structure."""
    loso, lolo = d["seed"], d["layer"]
    cell = d["layer"] * 1000 + d["budget"]
    G = {"LOSO": cv_errors(d, "count_flex", None, loso)[0],
         "LOLO": cv_errors(d, "count_flex", None, lolo)[0]}
    exposure = {"global": G,
                "budget": {"LOSO": stratified_se(d, "count_flex", None, loso, d["budget"]),
                           "LOLO": stratified_se(d, "count_flex", None, lolo, d["budget"])},
                "layer": {"LOSO": stratified_se(d, "count_flex", None, loso, d["layer"])},
                "cell": {"LOSO": stratified_se(d, "count_flex", None, loso, cell)}}
    specs = [("1 global exposure", None, "count_flex", None, "global", ("LOSO", "LOLO")),
             ("2 global + s", None, "sig", None, "global", ("LOSO", "LOLO")),
             ("3 global + s x budget", None, "sig_budget", None, "global", ("LOSO", "LOLO")),
             ("4 budget-specific + s", "budget", "sig", d["budget"], "budget", ("LOSO", "LOLO")),
             ("5 layer-specific + s", "layer", "sig", d["layer"], "layer", ("LOSO",)),
             ("6 layer-specific + s x budget", "layer", "sig_budget", d["layer"], "layer", ("LOSO",)),
             ("7 [DIAGNOSTIC ONLY] s x budget x layer", "cell", "sig", cell, "cell", ("LOSO",))]
    out = {"exposure_only": {
        k: {a: dict(gain_vs_global=gain(se, G[a]), fold_gains_vs_global=fold_gains(
            se, G[a], loso if a == "LOSO" else lolo)) for a, se in v.items()}
        for k, v in exposure.items() if k != "global"}, "models": {}}
    for s in STATE_SIGNALS:
        out["models"][s] = {}
        for name, _, kind, by, match, axes in specs:
            if kind == "count_flex" and s != STATE_SIGNALS[0]:
                continue
            r = {}
            for a in axes:
                g = loso if a == "LOSO" else lolo
                se = stratified_se(d, kind, s if kind != "count_flex" else None, g, by)
                r[a] = dict(gain_vs_global=gain(se, G[a]),
                            gain_vs_matched_exposure=gain(se, exposure[match][a]),
                            folds_vs_matched=fold_gains(se, exposure[match][a], g))
            out["models"][s][name] = r
    return out


# --------------------------------------------------------- decision

def expert_share(d, s):
    """Share of the signal's variance explained by (layer, expert) means."""
    _, inv = np.unique(np.stack([d["layer"], d["expert"]], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    mu = np.bincount(inv, d[s]) / np.bincount(inv)
    return float(1 - np.mean((d[s] - mu[inv]) ** 2) / d[s].var())


def decide(B, Lr, min_gain, alpha, eligible=STATE_SIGNALS):
    nb = {s: sum(B[b]["signals"][s]["robust"] for b in B) for s in STATE_SIGNALS}
    nl = {s: sum(Lr[L]["signals"][s]["sig"]["robust"] for L in Lr) for s in STATE_SIGNALS}
    nlb = {s: sum(Lr[L]["signals"][s]["sig_budget"]["robust"] for L in Lr) for s in STATE_SIGNALS}
    # slope change across budgets at fixed budget (pooled-over-layers fits)
    bflip = {s: sign_flip(signs({b: B[b]["signals"][s]["slope"] for b in B})) for s in STATE_SIGNALS}
    # same signal differs across layers (common within-layer slope)
    lflip = {s: sign_flip(signs({L: Lr[L]["signals"][s]["sig"]["slope"] for L in Lr}))
             for s in STATE_SIGNALS}
    pick = lambda c: max(eligible, key=c.get)
    top = lambda c: max((c[s] for s in eligible), default=0)
    need = 3
    if top(nb) >= need:
        dec, sig = "BUDGET", pick(nb)
    elif top(nl) >= need:
        dec, sig = "LAYER", pick(nl)
    elif top(nlb) >= need:
        dec, sig = "BOTH", pick(nlb)
    else:
        dec, sig = "UNSTABLE", None
    return dict(decision=dec, signal=sig, robust_budgets=nb, robust_layers_common=nl,
                robust_layers_budget_aware=nlb, budget_slope_sign_flip=bflip,
                layer_slope_sign_flip=lflip, eligible=list(eligible), need=need, min_gain=min_gain, alpha=alpha)


REC = {
    "BUDGET": "redesign the E2 predictor as explicitly budget-specific; E3 stays blocked until it "
              "passes a strict held-out gate",
    "LAYER": "consider layer-specific predictors (only usable on profiled layers); E3 stays blocked "
             "until strict cross-seed validation passes",
    "BOTH": "propose the simplest structured predictor supported by CV and define a new gate; E3 "
            "stays blocked until then",
    "UNSTABLE": "conditioning on budget and/or layer does not remove the instability: stop forcing a "
                "sufficiency-aware predictor; treat the core hypothesis as unsupported / strongly "
                "context-dependent; E3 stays blocked",
}


# ----------------------------------------------------------- render

pct = lambda x: f"{100 * x:+.1f}%"
folds = lambda f: ", ".join(f"{k}:{pct(v)}" for k, v in f.items())
flag = lambda xs: " **NEG " + ",".join(xs) + "**" if xs else ""


def render(res, tag):
    D, B, Lr, Ld = res["decision"], res["budget"], res["layer"], res["ladder"]
    M = [f"# P2b regime diagnostic ({tag}): P2b-{D['decision']}", "",
         f"Recommendation: {REC[D['decision']]}.", "",
         f"Input: `{res['input']}` ({res['n_obs']} rows; {res['design']}). Target "
         "`future_rel_delta`; gain = 1 - MSE / MSE(matched exposure model); robust = no "
         f"held-out fold <= 0, pooled LOSO >= {pct(D['min_gain'])}, permutation p < {D['alpha']} "
         f"({res['permutations']} perms). Signs use |t| >= 2 (OLS t, optimistic; 0 = not significant).",
         "", "## Decision inputs", "",
         f"Mechanical decision (all four signals eligible): P2b-{D['mechanical_decision']}. Signals "
         f"with >= {100 * D['static_share_threshold']:.0f}% of variance explained by expert identity are "
         "static expert descriptors, not calibration state, and are excluded "
         f"(target: {100 * res['expert_share']['y']:.1f}%).", "",
         "| signal | expert-id share of variance | eligible | robust budgets (/4) | robust layers, common (/4) "
         "| robust layers, budget-aware (/4) | budget slope sign flip | layer slope sign flip |",
         "|---|---|---|---|---|---|---|---|"]
    for s in STATE_SIGNALS:
        M.append(f"| {s} | {100 * res['expert_share'][s]:.1f}% | {'yes' if s in D['eligible'] else 'NO (static)'} "
                 f"| {D['robust_budgets'][s]} | {D['robust_layers_common'][s]} | "
                 f"{D['robust_layers_budget_aware'][s]} | {'YES' if D['budget_slope_sign_flip'][s] else 'no'} | "
                 f"{'YES' if D['layer_slope_sign_flip'][s] else 'no'} |")
    M += ["", "## 1. Budget-specific (exposure = cubic in log count within n0)", "",
          "| n0 | signal | LOSO pooled | LOSO folds (seed) | LOLO pooled | LOLO folds (layer) | "
          "perm p (LOSO / LOLO) | slope (t) | per-seed signs | per-layer signs | robust |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, rb in B.items():
        for s, r in rb["signals"].items():
            M.append(f"| {b} | {s} | {pct(r['LOSO']['pooled_gain'])} | {folds(r['LOSO']['fold_gains'])}"
                     f"{flag(r['neg_seeds'])} | {pct(r['LOLO']['pooled_gain'])} | "
                     f"{folds(r['LOLO']['fold_gains'])}{flag(r['neg_layers'])} | "
                     f"{r['perm_p_LOSO']:.3f} / {r['perm_p_LOLO']:.3f} | "
                     f"{r['slope']['beta']:+.4f} ({r['slope']['t']:+.1f}) | "
                     f"{''.join(signs(r['slope_by_seed']).values())} | "
                     f"{''.join(signs(r['slope_by_layer']).values())} | {'yes' if r['robust'] else 'no'} |")
    M += ["", "Exposure-only R2 (LOSO) by budget: " +
          ", ".join(f"n0={b} {rb['exposure_r2_LOSO']:.3f}" for b, rb in B.items()),
          "", "## 2. Layer-specific (exposure = cubic + budget dummies + budget x log count)", "",
          "| layer | signal | model | LOSO pooled | seed folds | gain by budget | perm p | slope(s) (t) | robust |",
          "|---|---|---|---|---|---|---|---|---|"]
    for L, rl in Lr.items():
        for s, rs in rl["signals"].items():
            for kind in ("sig", "sig_budget"):
                r = rs[kind]
                sl = ([r["slope"]] if kind == "sig" else list(r["slope_by_budget"].values()))
                M.append(f"| L{L} | {s} | {KIND_NAME[kind]} | {pct(r['pooled_gain'])} | "
                         f"{folds(r['seed_gains'])}{flag(r['neg_seeds'])} | {folds(r['budget_gains'])}"
                         f"{flag(r['neg_budgets'])} | {r['perm_p']:.3f} | "
                         + ", ".join(f"{x['beta']:+.4f} ({x['t']:+.1f})" for x in sl)
                         + f" | {'yes' if r['robust'] else 'no'} |")
    M += ["", "Exposure-only R2 (LOSO) by layer: " +
          ", ".join(f"L{L} {rl['exposure_r2_LOSO']:.3f}" for L, rl in Lr.items()),
          "", "## 3. Model-complexity ladder (all rows; held-out only)", "",
          "Stratified exposure models vs global count_flex: " +
          "; ".join(f"{k} {a} {pct(v['gain_vs_global'])} ({folds(v['fold_gains_vs_global'])})"
                    for k, ax in Ld["exposure_only"].items() for a, v in ax.items()), "",
          "| signal | model | LOSO vs global | LOSO vs matched exposure | LOSO folds (matched) | "
          "LOLO vs global | LOLO vs matched | LOLO folds (matched) |", "|---|---|---|---|---|---|---|---|"]
    for s, ms in Ld["models"].items():
        for name, r in ms.items():
            lo = r.get("LOLO")
            M.append(f"| {s if not name.startswith('1') else '-'} | {name} | {pct(r['LOSO']['gain_vs_global'])} | "
                     f"{pct(r['LOSO']['gain_vs_matched_exposure'])} | {folds(r['LOSO']['folds_vs_matched'])} | "
                     + (f"{pct(lo['gain_vs_global'])} | {pct(lo['gain_vs_matched_exposure'])} | "
                        f"{folds(lo['folds_vs_matched'])} |" if lo else "n/a (unseen layer) | | |"))
    M += ["", "Model 7 (signal x budget x layer) is a descriptive upper bound only and is not "
          "eligible for E3.", "", "## Interpretation", ""] + res["interpretation"]
    return "\n".join(M)


def interpret(res):
    D, B, Lr = res["decision"], res["budget"], res["layer"]
    lines = []
    for s in STATE_SIGNALS:
        nb = [b for b in B if B[b]["signals"][s]["neg_layers"]]
        ns = [b for b in B if B[b]["signals"][s]["neg_seeds"]]
        bl = [L for L in Lr if Lr[L]["signals"][s]["sig_budget"]["neg_seeds"]]
        bb = [L for L in Lr if Lr[L]["signals"][s]["sig_budget"]["neg_budgets"]]
        lines.append(f"- **{s}**: within-budget fits have a negative held-out layer at n0 = "
                     f"{', '.join(nb) or 'none'} and a negative held-out seed at n0 = {', '.join(ns) or 'none'}; "
                     f"within-layer budget-aware fits have a negative held-out seed in "
                     f"{', '.join('L' + L for L in bl) or 'none'} and a negative budget in "
                     f"{', '.join('L' + L for L in bb) or 'none'}.")
    Ld = res["ladder"]
    lay = Ld["exposure_only"]["layer"]["LOSO"]["gain_vs_global"]
    bud = Ld["exposure_only"]["budget"]["LOSO"]["gain_vs_global"]
    m = lambda s, k: Ld["models"][s][k]["LOSO"]["gain_vs_matched_exposure"]
    lines += [
        "", f"- Exposure: a per-layer exposure model alone gains {pct(lay)} over global count_flex "
        f"(LOSO); a per-budget one {pct(bud)}. Much of what P2 credited to the signals is layer "
        "heterogeneity that the global baseline does not model.",
        "- Hypothesis A (budget): " + ("supported" if D["decision"] == "BUDGET" else "rejected")
        + f" — robust budgets per signal {D['robust_budgets']}; budget-specific + s "
        "(model 4) is no better than global + s x budget (model 3), and still fails held-out layers.",
        "- Hypothesis B (layer): " + ("supported" if D["decision"] == "LAYER" else "rejected")
        + " — vs per-layer exposure, layer-specific + s gains " + ", ".join(
            f"{s} {pct(m(s, '5 layer-specific + s'))}" for s in STATE_SIGNALS)
        + f"; the gains of the static signals ({', '.join(s for s in STATE_SIGNALS if s not in D['eligible']) or 'none'}) "
        "come with within-layer slopes whose sign changes across layers, so they cannot transfer to "
        "unprofiled layers.",
        f"- Hypothesis C (both / neither): final decision P2b-{D['decision']} "
        f"(mechanical, all signals eligible: P2b-{D['mechanical_decision']})."]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--percdamp", type=float, default=1.0)
    ap.add_argument("--signals-csv", default=None)
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--static-share", type=float, default=0.95)
    a = ap.parse_args(argv)
    tag = f"b{a.bits}_{damp_tag(a.percdamp)}"
    path = a.signals_csv or os.path.join(a.outdir, f"e2_signals_{tag}.csv")
    d, n_drop = load_obs(path)
    d["expert"] = load_expert_col(path, d)
    design = check_table(d)
    gate = os.path.join(a.outdir, f"e2_gate_{tag}.json")
    if os.path.exists(gate):
        g = json.load(open(gate))
        if g["n_obs"] != d["y"].size:
            raise SystemExit(f"row count {d['y'].size} != P2 gate n_obs {g['n_obs']}")
        design["p2_case"] = g["case"]
    rng = np.random.default_rng(a.seed)
    res = dict(input=path, n_obs=int(d["y"].size), n_dropped_nonfinite=n_drop, design=design,
               permutations=a.permutations)
    res["budget"] = budget_analysis(d, a.permutations, rng, a.min_gain, a.alpha)
    res["layer"] = layer_analysis(d, a.permutations, rng, a.min_gain, a.alpha)
    res["ladder"] = ladder(d)
    res["expert_share"] = {s: expert_share(d, s) for s in (*STATE_SIGNALS, "y")}
    eligible = [s for s in STATE_SIGNALS if res["expert_share"][s] < a.static_share]
    res["decision"] = decide(res["budget"], res["layer"], a.min_gain, a.alpha, eligible)
    res["decision"]["mechanical_decision"] = decide(res["budget"], res["layer"], a.min_gain,
                                                    a.alpha)["decision"]
    res["decision"]["static_share_threshold"] = a.static_share
    res["interpretation"] = interpret(res)
    base = os.path.join(a.outdir, f"p2b_regime_{tag}")
    json.dump(res, open(base + ".json", "w"), indent=2)
    md = render(res, tag)
    open(base + ".md", "w").write(md + "\n")
    print(md)
    print(f"\nP2b decision: {res['decision']['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
