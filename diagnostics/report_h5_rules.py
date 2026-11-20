"""Report for diagnostics/h5_rules.py -> runs_h5/report.{json,md} + figures (frozen protocol runs_h5/protocol.json).

  python diagnostics/report_h5_rules.py
"""
from __future__ import annotations
import json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5_rules as H5                                               # noqa: E402
import theorem1_check as T1                                         # noqa: E402
from report_theorem1_check import COL                               # noqa: E402

GROUPS = H5.GROUPS
PREFIXES = H5.PREFIXES
RULES = H5.RULES
FIG = f"{H5.OUT}/figures"
NBOOT = H5.NBOOT
METRICS = ("all", "gu", "down")


def ci(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if x.size else [float("nan")] * 2


# ---------------------------------------------------------------- measured evaluation (primary)

def load_eval(subset=None):
    """metric -> rule -> prefix -> array over clusters (layer, expert) [nan if missing]; cluster keys."""
    recs = T1.read_jsonl(H5.RAW_EVAL)
    keys = sorted({(r["layer"], int(e)) for r in recs for e in r["experts"]
                   if subset is None or int(e) in subset.get(r["layer"], [])})
    idx = {k: i for i, k in enumerate(keys)}
    out = {m: {r: {n: np.full(len(keys), np.nan) for n in PREFIXES} for r in RULES} for m in METRICS}
    nus = {r: {n: {} for n in PREFIXES} for r in RULES}
    for rec in recs:
        for e, x in rec["experts"].items():
            k = (rec["layer"], int(e))
            if k not in idx:
                continue
            for rule, err in x["err"].items():
                for m in METRICS:
                    out[m][rule][rec["n"]][idx[k]] = err[m]
            for rule, v in x["nus"].items():
                nus[rule][rec["n"]][k] = v
    return out, keys, nus


def measured_tables(D, keys, rules):
    """Gain vs R0 / oracle capture per prefix and overall, for one metric dict D[rule][n] -> array."""
    rng = np.random.default_rng(0)
    nk = len(keys)
    mask = {n: np.all([np.isfinite(D[r][n]) for r in rules], 0) for n in PREFIXES}
    boots = [rng.integers(0, nk, nk) for _ in range(NBOOT)]

    def stats(sel):
        E = {r: {n: float(np.mean(D[r][n][sel[n]])) for n in PREFIXES} for r in rules}
        return E

    full = {n: np.where(mask[n])[0] for n in PREFIXES}
    E = stats(full)
    BE = []
    for b in boots:
        sel = {n: b[mask[n][b]] for n in PREFIXES}
        BE.append(stats(sel))
    out = {}
    for r in rules:
        row = {}
        for n in PREFIXES:
            g = 100 * (E["R0"][n] - E[r][n]) / E["R0"][n]
            gb = [100 * (x["R0"][n] - x[r][n]) / x["R0"][n] for x in BE]
            den = E["R0"][n] - E["R5"][n]
            cap = (E["R0"][n] - E[r][n]) / den if den != 0 else float("nan")
            capb = [(x["R0"][n] - x[r][n]) / (x["R0"][n] - x["R5"][n]) if x["R0"][n] != x["R5"][n] else np.nan
                    for x in BE]
            row[str(n)] = dict(E=E[r][n], gain_pct=g, gain_ci95=ci(gb), capture=cap, capture_ci95=ci(capb),
                               cells=int(mask[n].sum()))
        num = sum(E["R0"][n] - E[r][n] for n in PREFIXES)
        den = sum(E["R0"][n] - E["R5"][n] for n in PREFIXES)
        capb = [sum(x["R0"][n] - x[r][n] for n in PREFIXES) / sum(x["R0"][n] - x["R5"][n] for n in PREFIXES)
                for x in BE]
        gall = 100 * num / sum(E["R0"][n] for n in PREFIXES)
        gallb = [100 * sum(x["R0"][n] - x[r][n] for n in PREFIXES) / sum(x["R0"][n] for n in PREFIXES) for x in BE]
        row["overall"] = dict(capture=num / den if den else float("nan"), capture_ci95=ci(capb),
                              gain_pct=gall, gain_ci95=ci(gallb))
        out[r] = row
    diff = {}
    if "R4" in rules and "R3" in rules:
        for n in PREFIXES:
            d = 100 * (E["R4"][n] - E["R3"][n]) / E["R0"][n]
            db = [100 * (x["R4"][n] - x["R3"][n]) / x["R0"][n] for x in BE]
            diff[str(n)] = dict(diff_pct=d, ci95=ci(db))
        d = 100 * sum(E["R4"][n] - E["R3"][n] for n in PREFIXES) / sum(E["R0"][n] for n in PREFIXES)
        db = [100 * sum(x["R4"][n] - x["R3"][n] for n in PREFIXES) / sum(x["R0"][n] for n in PREFIXES) for x in BE]
        diff["overall"] = dict(diff_pct=d, ci95=ci(db))
    return out, diff


def verdict(T):
    passed = {}
    for r in ("R2", "R3", "R4"):
        row = T[r]
        passed[r] = bool(row["overall"]["capture"] >= 0.80 and row["16"]["gain_ci95"][0] > 0
                         and row["32"]["gain_ci95"][0] > 0)
    label = ("H5-PASS-R2" if passed["R2"] else "H5-PASS-R3" if passed["R3"] else
             "H5-PASS-R4" if passed["R4"] else "H5-FAIL")
    return label, passed


# ---------------------------------------------------------------- R_ideal-scored tables (both folds)

def ideal_scored(fold, fit, nus_primary=None):
    """E_rule = mean interpolated R_ideal(nu_rule) per (prefix, group) on the fold's eval cells."""
    ev = H5.FOLDS[fold]["eval"]
    cells = H5.e1_cells(ev["seed"], ev["layers"])
    P = fit[fold]
    rules = ["R0", "R1g", "R2", "R2f", "R3", "R3f", "R5"] + (["R1", "R4"] if nus_primary else [])
    res = {}
    for gi, g in enumerate(GROUPS):
        cg = cells[g]
        meta = cg["meta"]
        nvec = np.array([m[2] for m in meta])
        nu = {}
        nu["R0"] = np.ones(len(meta))
        nu["R5"] = cg["nu_star"]
        nu["R1g"] = np.array([P["R1g"][g][str(m[2])]["nu"] for m in meta])
        for rule, kind in (("R2", "budget"), ("R2f", "budget"), ("R3", "expert"), ("R3f", "expert")):
            p = P["laws"][g][rule]
            nu[rule] = H5.law_nu(p["logc"], p["b"], H5.law_x(cg, g, kind))
        if nus_primary:
            for rule in ("R1", "R4"):
                nu[rule] = np.array([nus_primary[rule][m[2]].get((m[1], m[3]), [np.nan, np.nan])[gi] for m in meta])
        Rv = {r: cg["interp"](np.nan_to_num(nu[r], nan=1.0)) for r in rules}
        if nus_primary:
            Rv["R4"] = np.where(np.isfinite(nu["R4"]), Rv["R4"], np.nan)
        res[g] = dict(nvec=nvec, R=Rv, cluster=cg["cluster"])
    out = {}
    for scope in ("gu", "down", "pooled"):
        gs = GROUPS if scope == "pooled" else (scope,)
        E = {r: {n: float(np.nanmean(np.concatenate([res[g]["R"][r][res[g]["nvec"] == n] for g in gs])))
                 for n in PREFIXES} for r in rules}
        rows = {}
        for r in rules:
            rows[r] = {str(n): dict(E=E[r][n], capture=(E["R0"][n] - E[r][n]) / (E["R0"][n] - E["R5"][n]))
                       for n in PREFIXES}
            rows[r]["overall"] = dict(capture=sum(E["R0"][n] - E[r][n] for n in PREFIXES) /
                                      sum(E["R0"][n] - E["R5"][n] for n in PREFIXES))
        out[scope] = rows
    return out


# ---------------------------------------------------------------- Part K

def load_kappa():
    out = {}
    for r in T1.read_jsonl(H5.RAW_KAPPA):
        for k, c in r["cells"].items():
            e, g = k.split(":")
            if not c.get("dropped"):
                out[(r["layer"], int(e), g)] = c
    return out


def ols(X, y, w=None):
    if w is None:
        w = np.ones(len(y))
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def r2(y, yh, sel=None):
    if sel is not None:
        y, yh = y[sel], yh[sel]
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


def part_k(kap):
    rows = []
    for r in T1.read_jsonl(H5.RAW_E1):
        for k, c in r["cells"].items():
            e, g = k.split(":")
            e = int(e)
            tok = r["tokens"][e]
            kk = kap.get((r["layer"], e, g))
            if "nu_star" not in c or c["edge"] or tok <= 0 or kk is None:
                continue
            rows.append(dict(y=math.log(c["nu_star"]), x=math.log(H5.D_IN[g] / tok), down=float(g == "down"),
                             cl=(r["layer"], e), g=g, layer=r["layer"], expert=e, nu=c["nu_star"], tok=tok,
                             **{v: math.log(kk[v]) for v in ("kappa_coord", "kappa_dir", "kappa_top", "erank_d")}))
    y = np.array([r["y"] for r in rows])
    base = np.column_stack([np.ones(len(rows)), [r["x"] for r in rows], [r["down"] for r in rows]])
    col = {v: np.array([r[v] for r in rows]) for v in ("kappa_coord", "kappa_dir", "kappa_top", "erank_d")}
    isd = np.array([r["g"] == "down" for r in rows])
    clus = [r["cl"] for r in rows]
    u = sorted(set(clus))
    ci_ = np.array([u.index(c) for c in clus])
    rng = np.random.default_rng(0)
    boots = [np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[ci_].astype(float) for _ in range(NBOOT)]

    def model(extra):
        X = np.column_stack([base] + [col[v] for v in extra])
        b = ols(X, y)
        yh = X @ b
        bs = [ols(X, y, w)[2] for w in boots]
        names = ["const", "log(d/n)", "1[down]"] + [f"log {v}" for v in extra]
        return dict(predictors=extra, coef=dict(zip(names, map(float, b))), group_coef=float(b[2]),
                    group_ci95=ci(bs), r2=r2(y, yh), r2_down=r2(y, yh, isd), r2_gu=r2(y, yh, ~isd), n=len(y))

    M = {"M0": model([])}
    for v in ("kappa_coord", "kappa_dir", "kappa_top"):
        M[f"M1_{v}"] = model([v])
    best = max(("kappa_coord", "kappa_dir", "kappa_top"), key=lambda v: M[f"M1_{v}"]["r2"])
    M["M1"] = M[f"M1_{best}"]
    M["M2"] = model(["erank_d"])
    M["M3"] = model([best, "erank_d"])

    def cond(m):
        shrink = 1 - abs(M[m]["group_coef"]) / abs(M["M0"]["group_coef"])
        dr = M[m]["r2_down"] - M["M0"]["r2_down"]
        return dict(shrink=shrink, delta_r2_down=dr, holds=bool(shrink >= 0.5 and dr >= 0.10))
    tails, spec = cond("M1"), cond("M2")
    label = ("K-BOTH" if tails["holds"] and spec["holds"] else "K-TAILS" if tails["holds"] else
             "K-SPECTRUM" if spec["holds"] else "K-NEITHER")
    # descriptive: group medians of the statistics
    desc = {g: {v: float(np.median([kap[k][v] for k in kap if k[2] == g]))
                for v in ("kappa_coord", "kappa_dir", "kappa_top", "erank_d", "pr_d")} for g in GROUPS}
    return dict(models={k: v for k, v in M.items() if k != "M1"}, kappa_best=best, tails=tails, spectrum=spec,
                verdict=label, medians=desc, n_cells=len(y)), rows


# ---------------------------------------------------------------- figures

def figures(rep, krows, kap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"axes.edgecolor": COL["ink2"], "axes.labelcolor": COL["ink"], "xtick.color": COL["ink2"],
                         "ytick.color": COL["ink2"], "axes.grid": True, "grid.color": COL["grid"],
                         "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": COL["surface"], "axes.facecolor": COL["surface"], "font.size": 10})
    gc = {"gu": COL["blue"], "down": COL["orange"]}
    gname = {"gu": "gate+up", "down": "down"}

    # 1. kappa_best vs nu* (n/d)^(2/3), per (layer, expert, group): median over seeds / prefixes
    kb = rep["K"]["kappa_best"]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for g in GROUPS:
        agg = {}
        for r in krows:
            if r["g"] == g:
                agg.setdefault((r["layer"], r["expert"]), []).append(r["nu"] * (r["tok"] / H5.D_IN[g]) ** (2 / 3))
        xs = [kap[(L, e, g)][kb] for (L, e) in agg]
        ys = [np.median(v) for v in agg.values()]
        ax.scatter(xs, ys, s=14, color=gc[g], alpha=0.7, linewidths=0, label=gname[g])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{kb} (tuning activations)")
    ax.set_ylabel("ν* · (n_e/d)^(2/3)  (median over seeds, prefixes)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/kappa_vs_scaled_nu.png", dpi=150)
    plt.close(fig)

    # 2. group coefficient across models
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    names = ["M0", f"M1_{kb}", "M2", "M3"] + [f"M1_{v}" for v in ("kappa_coord", "kappa_dir", "kappa_top") if v != kb]
    for i, m in enumerate(names):
        M = rep["K"]["models"][m]
        ax.errorbar([i], [M["group_coef"]], yerr=[[M["group_coef"] - M["group_ci95"][0]],
                                                   [M["group_ci95"][1] - M["group_coef"]]],
                    fmt="o", color=COL["orange"] if i < 4 else COL["ink2"], ms=7, capsize=3)
    ax.axhline(0, color=COL["ink2"], lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_kappa_", "\nκ_") for n in names], fontsize=8.5)
    ax.set_ylabel("1[down] coefficient (log ν*)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/group_coef.png", dpi=150)
    plt.close(fig)

    # 3. measured gain vs R0 per prefix, all rules
    T = rep["measured"]["all"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    rules = [r for r in RULES if r != "R0"]
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#c23b6e", "#d4a017", "#52514e", "#0b0b0b"]
    for j, r in enumerate(rules):
        xs = np.arange(len(PREFIXES)) + (j - len(rules) / 2) * 0.09
        g = [T[r][str(n)]["gain_pct"] for n in PREFIXES]
        lo = [T[r][str(n)]["gain_pct"] - T[r][str(n)]["gain_ci95"][0] for n in PREFIXES]
        hi = [T[r][str(n)]["gain_ci95"][1] - T[r][str(n)]["gain_pct"] for n in PREFIXES]
        ax.errorbar(xs, g, yerr=[lo, hi], fmt="o", ms=4.5, capsize=2, color=pal[j], label=r)
    ax.axhline(0, color=COL["ink2"], lw=0.8)
    ax.set_xticks(range(len(PREFIXES)))
    ax.set_xticklabels([f"B{n}" for n in PREFIXES])
    ax.set_ylabel("measured gain vs R0 (ν=1), %")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/gain_by_rule.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main / render

def main():
    P = json.load(open(H5.PROTOCOL))
    fit = json.load(open(H5.FIT))
    ev = T1.read_jsonl(H5.EVENTS)
    recs = T1.read_jsonl(H5.RAW_EVAL)
    complete = len(recs) == 10
    D, keys, nus = load_eval()
    r4_mode = next((e["r4_mode"] for e in ev if "r4_mode" in e), None)
    rules_all = [r for r in RULES if any(np.isfinite(D["all"][r][n]).any() for n in PREFIXES)]
    measured, diff = {}, {}
    for m in METRICS:
        measured[m], diff[m] = measured_tables(D[m], keys, rules_all)
    label, passed = verdict(measured["all"]) if complete else ("INCOMPLETE", {})
    rep = dict(protocol_frozen_utc=P["frozen_utc"], complete=complete, eval_units=len(recs), r4_mode=r4_mode,
               verdict=label, passed=passed, fit={f: {k: v for k, v in fit[f].items()} for f in fit},
               measured=measured, r4_minus_r3=diff,
               events=[e for e in ev if "checkpoint" not in e])
    if r4_mode == "24":
        sub = {r["layer"]: r["r4_sub"] for r in recs}
        Ds, ks, _ = load_eval(sub)
        rep["measured_r4_subset"] = {m: measured_tables(Ds[m], ks, rules_all)[0] for m in METRICS}
    rep["ideal_scored"] = dict(primary=ideal_scored("primary", fit, nus), swapped=ideal_scored("swapped", fit))
    # R4 diagnostics
    sk = [v["skipped_folds"] for r in recs for v in r["r4_detail"].values()]
    ed = [v.get("edge") for r in recs for v in r["r4_detail"].values() if "edge" in v]
    rep["r4_diag"] = dict(cells=len(sk), skipped_folds_total=int(sum(sk)), cells_with_skips=int(sum(s > 0 for s in sk)),
                          edge_frac=float(np.mean(ed)) if ed else None,
                          failed=sum("failed" in v for r in recs for v in r["r4_detail"].values()))
    rep["timing"] = dict(eval_h=sum(r["sec"] for r in recs) / 3600, r4_h=sum(r["sec_r4"] for r in recs) / 3600)
    kap = load_kappa()
    krows = []
    if len({k[0] for k in kap}) == 4:
        rep["K"], krows = part_k(kap)
    else:
        rep["K"] = dict(verdict="INCOMPLETE")
    if complete and rep["K"].get("verdict") != "INCOMPLETE":
        figures(rep, krows, kap)
    json.dump(rep, open(f"{H5.OUT}/report.json", "w"), indent=1, default=float)
    open(f"{H5.OUT}/report.md", "w").write(render(rep))
    print(render(rep))


def pct(x, ci_=None):
    s = f"{x:+.2f}%"
    return s + (f" [{ci_[0]:+.2f}, {ci_[1]:+.2f}]" if ci_ else "")


def render(rep):
    L = ["# H5 deployable damping rules + Part K\n",
         f"**H5 verdict: `{rep['verdict']}`** (pass per rule: {rep['passed']})  ",
         f"**K verdict: `{rep['K'].get('verdict')}`**\n",
         f"Eval units {rep['eval_units']}/10, R4 mode {rep['r4_mode']}. R4: {rep['r4_diag']['cells']} cells, "
         f"{rep['r4_diag']['cells_with_skips']} with skipped (empty) folds ({rep['r4_diag']['skipped_folds_total']} "
         f"folds), CV argmin at a grid edge {100 * (rep['r4_diag']['edge_frac'] or 0):.1f}%, failed "
         f"{rep['r4_diag']['failed']}. Eval time {rep['timing']['eval_h']:.2f} h (R4 {rep['timing']['r4_h']:.2f} h).\n",
         "## 1. Fitted constants\n", "| fold | group | R2 c | R2f c, b | R3 c' | R3f c', b' | fit excess R2 / R3 |",
         "|---|---|---|---|---|---|---|"]
    for f, F in rep["fit"].items():
        for g in GROUPS:
            l = F["laws"][g]
            L.append(f"| {f} | {g} | {l['R2']['c']:.3f} [{l['R2']['c_ci95'][0]:.3f}, {l['R2']['c_ci95'][1]:.3f}] | "
                     f"{l['R2f']['c']:.3f}, {l['R2f']['b']:.3f} [{l['R2f']['b_ci95'][0]:.3f}, {l['R2f']['b_ci95'][1]:.3f}] | "
                     f"{l['R3']['c']:.3f} [{l['R3']['c_ci95'][0]:.3f}, {l['R3']['c_ci95'][1]:.3f}] | "
                     f"{l['R3f']['c']:.3f}, {l['R3f']['b']:.3f} [{l['R3f']['b_ci95'][0]:.3f}, {l['R3f']['b_ci95'][1]:.3f}] | "
                     f"{100 * l['R2']['objective']:.2f}% / {100 * l['R3']['objective']:.2f}% |")
    L += ["\nR1 (primary fit set, shared ν, measured 'all') and R1g (per group, R_ideal) per prefix:\n",
          "| prefix | R1 ν (boot 2.5–97.5%) | R1g gu [CI] | R1g down [CI] | R1g swapped gu / down |", "|---|---|---|---|---|"]
    for n in map(str, PREFIXES):
        r1 = rep["fit"]["primary"]["R1"][n]
        a, b = rep["fit"]["primary"]["R1g"]["gu"][n], rep["fit"]["primary"]["R1g"]["down"][n]
        s = rep["fit"]["swapped"]["R1g"]
        L.append(f"| {n} | {r1['nu']} ({r1['boot_p2_5']}–{r1['boot_p97_5']}) | {a['nu']:.2f} [{a['ci95'][0]:.2f}, "
                 f"{a['ci95'][1]:.2f}] | {b['nu']:.2f} [{b['ci95'][0]:.2f}, {b['ci95'][1]:.2f}] | "
                 f"{s['gu'][n]['nu']:.2f} / {s['down'][n]['nu']:.2f} |")
    for m, title in (("all", "2. Eval set, measured `all`"), ("gu", "3a. per group: gate+up"),
                     ("down", "3b. per group: down")):
        T = rep["measured"][m]
        L += [f"\n## {title} (seed 1, layers 16/23; gain vs R0 [95% CI]; oracle capture)\n",
              "| rule | " + " | ".join(f"B{n} gain / capture" for n in PREFIXES) + " | overall capture [CI] |",
              "|---|" + "---|" * (len(PREFIXES) + 1)]
        for r, row in T.items():
            L.append(f"| {r} | " + " | ".join(f"{pct(row[str(n)]['gain_pct'], row[str(n)]['gain_ci95'])} / "
                                               f"{row[str(n)]['capture']:.2f}" for n in PREFIXES) +
                     f" | {row['overall']['capture']:.2f} [{row['overall']['capture_ci95'][0]:.2f}, "
                     f"{row['overall']['capture_ci95'][1]:.2f}] |")
        L.append("\nE_rule: " + "; ".join(f"B{n} R0 {T['R0'][str(n)]['E']:.5f} R5 {T['R5'][str(n)]['E']:.5f}"
                                          for n in PREFIXES))
    L += ["\n## 4. R4 − R3 (paired, % of E_R0; negative = R4 better)\n", "| metric | " +
          " | ".join(f"B{n}" for n in PREFIXES) + " | overall |", "|---|" + "---|" * (len(PREFIXES) + 1)]
    for m in METRICS:
        d = rep["r4_minus_r3"][m]
        if d:
            L.append(f"| {m} | " + " | ".join(pct(d[str(n)]["diff_pct"], d[str(n)]["ci95"]) for n in PREFIXES) +
                     f" | {pct(d['overall']['diff_pct'], d['overall']['ci95'])} |")
    L += ["\n## 5. R_ideal-scored oracle capture (overall, per group and pooled)\n",
          "| rule | primary gu | primary down | primary pooled | swapped gu | swapped down | swapped pooled |",
          "|---|---|---|---|---|---|---|"]
    IP, IS = rep["ideal_scored"]["primary"], rep["ideal_scored"]["swapped"]
    for r in RULES:
        if r not in IP["gu"]:
            continue
        sw = [f"{IS[s][r]['overall']['capture']:.2f}" if r in IS[s] else "–" for s in ("gu", "down", "pooled")]
        L.append(f"| {r} | {IP['gu'][r]['overall']['capture']:.2f} | {IP['down'][r]['overall']['capture']:.2f} | "
                 f"{IP['pooled'][r]['overall']['capture']:.2f} | " + " | ".join(sw) + " |")
    K = rep["K"]
    if K.get("verdict") != "INCOMPLETE":
        L += [f"\n## 6. Part K regressions (log ν*, {K['n_cells']} non-edge cells; κ_best = {K['kappa_best']})\n",
              "| model | 1[down] coef [95% CI] | R² | R² down | R² gu | other coefs |", "|---|---|---|---|---|---|"]
        for m, M in K["models"].items():
            oth = ", ".join(f"{k} {v:+.3f}" for k, v in M["coef"].items() if k not in ("const", "1[down]"))
            L.append(f"| {m} | {M['group_coef']:+.3f} [{M['group_ci95'][0]:+.3f}, {M['group_ci95'][1]:+.3f}] | "
                     f"{M['r2']:.3f} | {M['r2_down']:.3f} | {M['r2_gu']:.3f} | {oth} |")
        L.append(f"\nTails (M1 vs M0): shrink {100 * K['tails']['shrink']:.1f}%, ΔR² down "
                 f"{K['tails']['delta_r2_down']:+.3f}. Spectrum (M2 vs M0): shrink {100 * K['spectrum']['shrink']:.1f}%, "
                 f"ΔR² down {K['spectrum']['delta_r2_down']:+.3f}.\n")
        L.append("Group medians: " + "; ".join(f"{g}: " + ", ".join(f"{k} {v:.3g}" for k, v in K["medians"][g].items())
                                              for g in GROUPS) + "\n")
    L.append("Figures: `figures/gain_by_rule.png`, `figures/kappa_vs_scaled_nu.png`, `figures/group_coef.png`.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
