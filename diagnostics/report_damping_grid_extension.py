"""Report for diagnostics/damping_grid_extension.py -> runs_grid_ext/report.{json,md} + figures.

Implements the frozen protocol runs_grid_ext/protocol.json: Q1-Q5 and the EXT-* verdict.

  python diagnostics/report_damping_grid_extension.py
"""
from __future__ import annotations
import json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theorem1_check as T1                                          # noqa: E402
from damping_grid_extension import (EVENTS, E2_PREFIXES, NU_E1, NU_E2, NU_MERGED, OUT, PREFIXES,  # noqa: E402
                                    PROTOCOL, RAW_E1, RAW_E2, T1_OUTLIERS)
from report_theorem1_check import COL, cell_stats, fit_slope, is_num, med_iqr, q    # noqa: E402

D_IN = {"gu": 2048, "down": 1408}
GROUPS = ("gu", "down")
FIG = f"{OUT}/figures"
LX = np.log(NU_E1)
NBOOT = 2000


def pct(x, p):
    return q(x, p)


# ---------------------------------------------------------------- loading

def load_e1():
    """list of cells: dict(seed, layer, n, expert, group, tokens, R (42), nu_star, edge, argmin_idx, flat)."""
    cells = []
    for r in T1.read_jsonl(RAW_E1):
        for k, c in r["cells"].items():
            e, g = k.split(":")
            e = int(e)
            if "R_fine" not in c or not all(is_num(v) for v in c["R_fine"]):
                cells.append(dict(seed=r["seed"], layer=r["layer"], n=r["n"], expert=e, group=g, failed=True,
                                  tokens=r["tokens"][e]))
                continue
            cells.append(dict(seed=r["seed"], layer=r["layer"], n=r["n"], expert=e, group=g, tokens=r["tokens"][e],
                              R=np.array(c["R_fine"]), nu_star=c["nu_star"], R_star=c["R_star"], edge=c["edge"],
                              argmin_idx=c["argmin_idx"], flat=c["flat"], failed=False))
    return cells


def load_merged():
    """Merged 18-point B2 cells: (seed, layer, n, expert, group) -> dict(pred, meas, emeas, clampW, clampWork)."""
    pb = {(r["seed"], r["layer"], r["n"], float(r["nu"])): r for r in T1.read_jsonl(T1.RAW_GRID)}
    e2 = {(r["seed"], r["layer"], r["n"], float(r["nu"])): r for r in T1.read_jsonl(RAW_E2)}
    out = {}
    keys = sorted({k[:3] for k in pb})
    for (s, L, n) in keys:
        r0 = pb.get((s, L, n, T1.NU_GRID[0]))
        if r0 is None:
            continue
        for e in r0["b2_experts"]:
            for g in GROUPS:
                pts = []
                try:
                    for nu in T1.NU_GRID:
                        c = pb[(s, L, n, nu)]["b2"][str(e)]
                        pts.append(dict(nu=nu, pred=c[g]["R_ideal"], meas=c["gptq_err"][g] / c["rtn_err"][g],
                                        emeas=c[g]["e_meas"] / c[g]["e_rtn"], clampW=c[g]["clamp_frac_W"],
                                        clampWork=c[g]["clamp_frac_work"]))
                    if n in E2_PREFIXES[g]:
                        for nu in NU_E2[1:]:
                            c = e2[(s, L, n, nu)]["cells"][f"{e}:{g}"]
                            pts.append(dict(nu=nu, pred=c["R_ideal"], meas=c["ratio"], emeas=c["e_meas"] / c["e_rtn"],
                                            clampW=c["clamp_frac_W"], clampWork=c["clamp_frac_work"]))
                except (KeyError, TypeError):
                    continue
                out[(s, L, n, e, g)] = {k: [p[k] for p in pts] for k in pts[0]}
    return out


# ---------------------------------------------------------------- core helpers

def vertex(lx, y):
    """argmin + parabola vertex in log nu; returns (nu*, value at vertex, argmin index, interior)."""
    y = np.asarray(y, float)
    i = int(np.argmin(y))
    interior = 0 < i < len(y) - 1
    v = T1.parabola_min(lx, y, i) if interior else None
    if v is None:
        return float(np.exp(lx[i])), float(y[i]), i, interior
    c = np.polyfit(lx[i - 1:i + 2], y[i - 1:i + 2], 2)
    val = float(np.polyval(c, v))
    if val > y[i]:
        return float(np.exp(lx[i])), float(y[i]), i, interior
    return float(math.exp(v)), val, i, interior


def boot_vertex(cells, rng, nboot=NBOOT):
    cl = sorted({(c["layer"], c["expert"]) for c in cells})
    idx = {k: [] for k in cl}
    for j, c in enumerate(cells):
        idx[(c["layer"], c["expert"])].append(j)
    Rm = np.array([c["R"] for c in cells])
    out = []
    for _ in range(nboot):
        pick = np.concatenate([idx[cl[j]] for j in rng.integers(0, len(cl), len(cl))])
        out.append(vertex(LX, Rm[pick].mean(0))[0])
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def power_law(g, ratio):
    t4 = json.load(open(f"{T1.OUT}/report.json"))["T4"][g]
    f = t4["fit_n_lt_d"] if ratio < 1 else t4["fit_n_gt_d"]
    return float(math.exp(f["a"]) * (1 / ratio) ** f["b"]), ("n<d" if ratio < 1 else "n>d"), f["a"], f["b"]


def old_partb_best():
    return json.load(open(f"{T1.OUT}/report.json"))["T1"]["best_nu_per_prefix"]


# ---------------------------------------------------------------- Q1 / Q3 (predicted) / Q5

def q1(cells):
    rng = np.random.default_rng(0)
    old = old_partb_best()
    out = {}
    j1, j5 = NU_E1.index(1.0), NU_E1.index(5.0)
    for g in GROUPS:
        out[g] = {}
        for n in PREFIXES:
            cn = [c for c in cells if c["group"] == g and c["n"] == n and not c["failed"]]
            if not cn:
                continue
            Rbar = np.mean([c["R"] for c in cn], 0)
            nu, val, i, interior = vertex(LX, Rbar)
            med_ratio = float(np.median([c["tokens"] / D_IN[g] for c in cn]))
            pl, side, a, b = power_law(g, med_ratio)
            ok = [c for c in cn if not c["flat"]]
            out[g][str(n)] = dict(
                cells=len(cn), nu_star=nu, ci95=boot_vertex(cn, rng), interior=bool(interior),
                grid_argmin=NU_E1[i], R_at_star=val, R_at_1=float(Rbar[j1]), R_at_5=float(Rbar[j5]),
                pred_excess_5=float(Rbar[j5] / val - 1), pred_excess_1=float(Rbar[j1] / val - 1),
                cell_excess_5=med_iqr([c["R"][j5] / c["R_star"] - 1 for c in ok]),
                cell_excess_1=med_iqr([c["R"][j1] / c["R_star"] - 1 for c in ok]),
                old_partb_pred=old[g][str(n)]["pred_best"], old_partb_meas=old[g][str(n)]["meas_best"],
                median_n_over_d=med_ratio, power_law=pl, power_law_side=side, mean_curve=Rbar.tolist())
    return out


def q5(cells):
    out = {}
    for g in GROUPS:
        out[g] = {}
        for n in PREFIXES:
            cn = [c for c in cells if c["group"] == g and c["n"] == n and not c["failed"] and not c["flat"]]
            ns = [c["nu_star"] for c in cn]
            hi = [c for c in cn if c["argmin_idx"] == len(NU_E1) - 1]
            lo = [c for c in cn if c["argmin_idx"] == 0]
            out[g][str(n)] = dict(n=len(cn), median=pct(ns, 50), q25=pct(ns, 25), q75=pct(ns, 75), p5=pct(ns, 5),
                                  p95=pct(ns, 95), frac_edge_hi=len(hi) / max(len(cn), 1),
                                  frac_edge_lo=len(lo) / max(len(cn), 1),
                                  edge_hi=[f"s{c['seed']} L{c['layer']} e{c['expert']}" for c in hi],
                                  edge_lo=[f"s{c['seed']} L{c['layer']} e{c['expert']}" for c in lo])
    # edge experts overall and overlap with T1 outliers
    edge = {}
    for c in cells:
        if not c["failed"] and not c["flat"] and c["edge"]:
            edge.setdefault((c["layer"], c["expert"], c["group"]), []).append(
                (c["seed"], c["n"], "hi" if c["argmin_idx"] else "lo"))
    outl = {(L, e) for L, es in T1_OUTLIERS.items() for e in es}
    out["edge_experts"] = [dict(layer=L, expert=e, group=g, cells=v, t1_outlier=(L, e) in outl)
                           for (L, e, g), v in sorted(edge.items())]
    # per-expert nu* of the T1 outlier experts (down) vs the layer's median, by prefix
    rows = []
    for (L, e) in sorted(outl):
        for n in PREFIXES:
            mine = [c["nu_star"] for c in cells if c["layer"] == L and c["expert"] == e and c["n"] == n
                    and c["group"] == "down" and not c["failed"]]
            lay = [c["nu_star"] for c in cells if c["layer"] == L and c["n"] == n and c["group"] == "down"
                   and not c["failed"] and not c["flat"]]
            rows.append(dict(layer=L, expert=e, n=n, nu_star=mine, layer_median=pct(lay, 50)))
    out["t1_outliers_down"] = rows
    return out


# ---------------------------------------------------------------- Q2 / Q3 (measured)

def q2(merged):
    out = {}
    for g in GROUPS:
        cs = []
        for (s, L, n, e, gg), m in merged.items():
            if gg != g or n not in E2_PREFIXES[g] or len(m["nu"]) != len(NU_MERGED):
                continue
            rho, agree = cell_stats(m["pred"], m["meas"])
            rho_e, agree_e = cell_stats(m["pred"], m["emeas"])
            cs.append(dict(seed=s, layer=L, n=n, expert=e, spearman=rho, agree=agree, spearman_emeas=rho_e,
                           agree_emeas=agree_e, argmin_pred=NU_MERGED[int(np.argmin(m["pred"]))],
                           argmin_meas=NU_MERGED[int(np.argmin(m["meas"]))]))
        rows = {}
        for n in E2_PREFIXES[g] + ["all"]:
            cn = [c for c in cs if n == "all" or c["n"] == n]
            if not cn:
                continue
            ag = [c["agree"] for c in cn if c["agree"] is not None]
            rows[str(n)] = dict(cells=len(cn), median_spearman=q([c["spearman"] for c in cn], 50),
                                frac_argmin_agree=float(np.mean(ag)) if ag else float("nan"),
                                median_spearman_emeas=q([c["spearman_emeas"] for c in cn], 50),
                                median_argmin_pred=pct([c["argmin_pred"] for c in cn], 50),
                                median_argmin_meas=pct([c["argmin_meas"] for c in cn], 50),
                                frac_meas_argmin_at_50=float(np.mean([c["argmin_meas"] == 50.0 for c in cn])),
                                frac_pred_argmin_at_50=float(np.mean([c["argmin_pred"] == 50.0 for c in cn])))
        out[g] = rows
    a = out["down"].get("all")
    gate = (a is not None and a["frac_argmin_agree"] >= 0.70 and a["median_spearman"] >= 0.8)
    # clamp fractions vs nu (down, all merged cells; gu at its prefixes)
    clamp = {}
    for g in GROUPS:
        clamp[g] = {}
        for j, nu in enumerate(NU_MERGED):
            v = [m for (s, L, n, e, gg), m in merged.items() if gg == g and len(m["nu"]) == len(NU_MERGED)]
            if v:
                clamp[g][str(nu)] = dict(clampW_mean=float(np.mean([m["clampW"][j] for m in v])),
                                         clampW_max=float(np.max([m["clampW"][j] for m in v])),
                                         clampWork_mean=float(np.mean([m["clampWork"][j] for m in v])),
                                         clampWork_max=float(np.max([m["clampWork"][j] for m in v])))
    return out, bool(gate), clamp


def q3_measured(merged):
    out = {}
    for g in GROUPS:
        out[g] = {}
        for n in PREFIXES:
            ms = [m for (s, L, nn, e, gg), m in merged.items() if gg == g and nn == n]
            if not ms:
                continue
            nus = ms[0]["nu"]
            ms = [m for m in ms if m["nu"] == nus]
            Mbar = np.mean([m["meas"] for m in ms], 0)
            Pbar = np.mean([m["pred"] for m in ms], 0)
            i1, i5 = nus.index(1.0), nus.index(5.0)
            jm = int(np.argmin(Mbar))
            out[g][str(n)] = dict(
                cells=len(ms), grid=len(nus), meas_argmin=nus[jm], pred_argmin_same_cells=nus[int(np.argmin(Pbar))],
                meas_excess_5=float(Mbar[i5] / Mbar[jm] - 1), meas_excess_1=float(Mbar[i1] / Mbar[jm] - 1),
                cell_meas_excess_5=med_iqr([m["meas"][i5] / min(m["meas"]) - 1 for m in ms]),
                cell_meas_excess_1=med_iqr([m["meas"][i1] / min(m["meas"]) - 1 for m in ms]),
                meas_curve=Mbar.tolist(), pred_curve=Pbar.tolist(), nus=nus)
    return out


# ---------------------------------------------------------------- Q4

def q4(cells, Q1):
    rng = np.random.default_rng(1)
    budget = {str(n): Q1["down"][str(n)]["nu_star"] / Q1["gu"][str(n)]["nu_star"] for n in PREFIXES
              if str(n) in Q1["down"] and str(n) in Q1["gu"]}
    by = {}
    for c in cells:
        by.setdefault((c["seed"], c["layer"], c["n"], c["expert"]), {})[c["group"]] = c
    rows = []
    n_edge = 0
    for (s, L, n, e), d in by.items():
        if len(d) < 2 or any(d[g]["failed"] or d[g]["flat"] for g in GROUPS):
            continue
        if any(d[g]["edge"] for g in GROUPS):
            n_edge += 1
            continue
        rows.append(dict(seed=s, layer=L, n=n, expert=e, tokens=d["gu"]["tokens"],
                         ratio=d["down"]["nu_star"] / d["gu"]["nu_star"]))
    per_prefix = {str(n): med_iqr([r["ratio"] for r in rows if r["n"] == n]) for n in PREFIXES}
    trend = fit_slope([((r["layer"], r["expert"]), math.log(r["tokens"]), math.log(r["ratio"])) for r in rows
                       if r["tokens"] > 0], rng)
    bt = [(math.log(np.median([c["tokens"] for c in cells if c["n"] == n])), math.log(budget[str(n)]))
          for n in PREFIXES if str(n) in budget]
    btrend = float(np.polyfit(*np.array(bt).T, 1)[0]) if len(bt) >= 3 else float("nan")
    return dict(budget=budget, budget_range=[min(budget.values()), max(budget.values())] if budget else None,
                per_cell_all=med_iqr([r["ratio"] for r in rows]), per_cell_by_prefix=per_prefix,
                n_matched=len(rows), n_excluded_edge=n_edge,
                trend_log_ratio_vs_log_tokens=trend, budget_trend_slope=btrend)


# ---------------------------------------------------------------- verdict

def verdict(Q1, gate):
    edges = [f"{g}@B{n}" for g in GROUPS for n, v in Q1[g].items() if not v["interior"]]
    if edges:
        return "EXT-EDGE", edges
    return ("EXT-INTERIOR-CONFIRMED" if gate else "EXT-INTERIOR-PREDICTED-ONLY"), []


# ---------------------------------------------------------------- figures

def figures(cells, Q1, Q3m, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"axes.edgecolor": COL["ink2"], "axes.labelcolor": COL["ink"], "xtick.color": COL["ink2"],
                         "ytick.color": COL["ink2"], "axes.grid": True, "grid.color": COL["grid"],
                         "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": COL["surface"], "axes.facecolor": COL["surface"], "font.size": 10})
    pcol = dict(zip(PREFIXES, ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#c23b6e"]))
    gname = {"gu": "gate+up", "down": "down"}

    # 1. expert-mean R_ideal vs nu per prefix + measured B2 means
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, g in zip(axs, GROUPS):
        for n in PREFIXES:
            v = Q1[g].get(str(n))
            if v is None:
                continue
            ax.plot(NU_E1, v["mean_curve"], color=pcol[n], lw=2, label=f"B{n} R_ideal (all 60)")
            ax.plot([v["nu_star"]], [v["R_at_star"]], "v", color=pcol[n], ms=8, mec=COL["surface"], mew=1.5)
            m = Q3m[g].get(str(n))
            if m is not None:
                ax.plot(m["nus"], m["meas_curve"], "o", color=pcol[n], ms=4.5, mfc=COL["surface"], mew=1.3)
        ax.set_xscale("log")
        ax.axvline(5.0, color=COL["ink2"], lw=1, ls=":")
        ax.text(5.2, ax.get_ylim()[1], "old grid max ν=5", va="top", fontsize=8, color=COL["ink2"])
        ax.set_xlabel("ν (damping / mean diag Ĥ)")
        ax.set_ylabel("GPTQ / RTN error ratio")
        ax.set_title(f"{gname[g]}  (line: mean R_ideal, ○: measured B2 mean, ▼: ν*)", fontsize=9.5)
        ax.set_ylim(top=min(ax.get_ylim()[1], 1.6))
    axs[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/mean_curves.png", dpi=150)
    plt.close(fig)

    # 2. nu* vs median n/d, both groups, with power law and George-Saab
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    gc = {"gu": COL["blue"], "down": COL["orange"]}
    t4 = json.load(open(f"{T1.OUT}/report.json"))["T4"]
    xs = np.logspace(np.log10(0.05), np.log10(6), 100)
    for g in GROUPS:
        r = [Q1[g][str(n)]["median_n_over_d"] for n in PREFIXES]
        ns = [Q1[g][str(n)]["nu_star"] for n in PREFIXES]
        lo = [Q1[g][str(n)]["nu_star"] - Q1[g][str(n)]["ci95"][0] for n in PREFIXES]
        hi = [Q1[g][str(n)]["ci95"][1] - Q1[g][str(n)]["nu_star"] for n in PREFIXES]
        ax.errorbar(r, ns, yerr=[lo, hi], fmt="o", color=gc[g], ms=7, capsize=3, label=f"{gname[g]} budget ν* (E1)")
        pl = [math.exp(t4[g]["fit_n_lt_d" if x < 1 else "fit_n_gt_d"]["a"]) *
              (1 / x) ** t4[g]["fit_n_lt_d" if x < 1 else "fit_n_gt_d"]["b"] for x in xs]
        ax.plot(xs, pl, color=gc[g], lw=1.5, ls="--", label=f"{gname[g]} Part C power law")
        for x, y, n in zip(r, ns, PREFIXES):
            ax.annotate(f"B{n}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8, color=COL["ink2"])
    ax.plot(xs, [T1.george_saab(x * 1000, 1000) for x in xs], color=COL["ink2"], lw=1.2, ls=":",
            label="George–Saab (depends on n/d only)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("median tokens / d_in")
    ax.set_ylabel("ν*")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/nu_star_vs_ratio.png", dpi=150)
    plt.close(fig)

    # 3. histogram of per-cell nu*
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=False)
    bins = np.logspace(-2, 2, 33)
    for ax, g in zip(axs, GROUPS):
        for n in PREFIXES:
            v = [c["nu_star"] for c in cells if c["group"] == g and c["n"] == n and not c["failed"] and not c["flat"]]
            ax.hist(v, bins=bins, histtype="step", lw=1.8, color=pcol[n], label=f"B{n}")
        ax.set_xscale("log")
        ax.set_xlabel("per-cell ν* (E1)")
        ax.set_ylabel("cells")
        ax.set_title(gname[g], fontsize=10)
    axs[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/nu_star_hist.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main / render

def main():
    P = json.load(open(PROTOCOL))
    cells = load_e1()
    n_e1 = len(T1.read_jsonl(RAW_E1))
    n_e2 = len(T1.read_jsonl(RAW_E2))
    complete = n_e1 == 2 * 4 * 5 and n_e2 == 2 * 4 * 5 * len(NU_E2)
    merged = load_merged()
    Q1 = q1(cells)
    Q2, gate, clamp = q2(merged)
    Q3m = q3_measured(merged)
    Q4 = q4(cells, Q1)
    Q5 = q5(cells)
    label, edges = verdict(Q1, gate) if complete else ("INCOMPLETE", [])
    ev = T1.read_jsonl(EVENTS)
    anchors = [v for r in T1.read_jsonl(RAW_E2) if r["nu"] == 5.0 for c in r["cells"].values()
               for v in c.get("anchor_rel", {}).values()]
    rep = dict(protocol_frozen_utc=P["frozen_utc"], complete=complete, records=dict(e1=n_e1, e2=n_e2),
               verdict=label, edges=edges, down_gate_pass=gate,
               invariants=dict(e1=[e for e in ev if "e1_invariant" in e],
                               e2_anchor=dict(n=len(anchors), max_rel=max(anchors) if anchors else None)),
               failed_cells=sum(c["failed"] for c in cells), flat_cells=sum(c.get("flat", False) for c in cells),
               Q1={g: {n: {k: v for k, v in r.items() if k != "mean_curve"} for n, r in Q1[g].items()} for g in GROUPS},
               Q2=Q2, clamp=clamp, Q3_measured={g: {n: {k: v for k, v in r.items() if k not in ("meas_curve", "pred_curve")}
                                                     for n, r in Q3m[g].items()} for g in GROUPS},
               Q4=Q4, Q5=Q5, run_events=[e for e in ev if "est_total_sec" in e or "checkpoint" in e])
    figures(cells, Q1, Q3m, rep)
    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1, default=float)
    open(f"{OUT}/report.md", "w").write(render(rep))
    print(render(rep))


def f(x, d=3):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}g}"


def pc(x):
    return "–" if x is None or not math.isfinite(x) else f"{100 * x:+.2f}%"


def render(rep):
    L = [f"# Damping-grid extension report\n", f"**Verdict: `{rep['verdict']}`**" +
         (f" — edges: {', '.join(rep['edges'])}" if rep["edges"] else "") +
         f"  (measured down gate: {'PASS' if rep['down_gate_pass'] else 'FAIL'})\n",
         f"Records: E1 {rep['records']['e1']}/40, E2 {rep['records']['e2']}/280. Failed cells {rep['failed_cells']}, "
         f"flat cells {rep['flat_cells']}. E2 anchor (ν=5 vs Part B): {rep['invariants']['e2_anchor']['n']} "
         f"comparisons, max rel {f(rep['invariants']['e2_anchor']['max_rel'])}.\n",
         "## Q1 budget-level ν* (E1, expert-mean R_ideal, 42-point grid)\n",
         "| group | prefix | median n/d | ν* [95% CI] | interior | old Part B argmin pred/meas | Part C power law |",
         "|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        for n, r in rep["Q1"][g].items():
            L.append(f"| {g} | {n} | {r['median_n_over_d']:.2f} | {r['nu_star']:.2f} [{r['ci95'][0]:.2f}, "
                     f"{r['ci95'][1]:.2f}] | {'yes' if r['interior'] else '**EDGE**'} | {r['old_partb_pred']} / "
                     f"{r['old_partb_meas']} | {r['power_law']:.2f} ({r['power_law_side']}) |")
    L += ["\n## Q2 measured confirmation (B2, 18-point merged grid)\n",
          "| group | prefix | cells | median Spearman | argmin agree ±1 | median argmin pred / meas | "
          "frac meas argmin at 50 |", "|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        for n, r in rep["Q2"][g].items():
            L.append(f"| {g} | {n} | {r['cells']} | {r['median_spearman']:.3f} | {100 * r['frac_argmin_agree']:.1f}% | "
                     f"{f(r['median_argmin_pred'])} / {f(r['median_argmin_meas'])} | "
                     f"{100 * r['frac_meas_argmin_at_50']:.1f}% |")
    L += ["\nClamp fraction (down, mean / max over B2 cells), original W and working weights:\n",
          "| ν | clamp W mean | clamp W max | clamp work mean | clamp work max |", "|---|---|---|---|---|"]
    for nu, r in rep["clamp"]["down"].items():
        L.append(f"| {nu} | {r['clampW_mean']:.2e} | {r['clampW_max']:.2e} | {r['clampWork_mean']:.2e} | "
                 f"{r['clampWork_max']:.2e} |")
    L += ["\n## Q3 practical size (excess of the old ν over ν*)\n",
          "| group | prefix | pred excess ν=5 (budget / cell median) | pred excess ν=1 (budget / cell median) | "
          "meas argmin (B2 mean) | meas excess ν=5 (mean / cell median) | meas excess ν=1 (mean / cell median) |",
          "|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        for n, r in rep["Q1"][g].items():
            m = rep["Q3_measured"][g].get(n)
            ms = (f"{f(m['meas_argmin'])} ({m['grid']}-pt) | {pc(m['meas_excess_5'])} / "
                  f"{pc(m['cell_meas_excess_5']['median'])} | {pc(m['meas_excess_1'])} / "
                  f"{pc(m['cell_meas_excess_1']['median'])}") if m else "– | – | –"
            L.append(f"| {g} | {n} | {pc(r['pred_excess_5'])} / {pc(r['cell_excess_5']['median'])} | "
                     f"{pc(r['pred_excess_1'])} / {pc(r['cell_excess_1']['median'])} | {ms} |")
    q4 = rep["Q4"]
    L += ["\n## Q4 ν*_down / ν*_gu\n", "| prefix | budget-level | per-cell median [IQR] |", "|---|---|---|"]
    for n in map(str, PREFIXES):
        b, c = q4["budget"].get(n), q4["per_cell_by_prefix"][n]
        L.append(f"| {n} | {f(b)} | {c['median']:.2f} [{c['q25']:.2f}, {c['q75']:.2f}] (n={c['n']}) |")
    t = q4["trend_log_ratio_vs_log_tokens"]
    L.append(f"\nAll matched cells: {q4['per_cell_all']['median']:.2f} [{q4['per_cell_all']['q25']:.2f}, "
             f"{q4['per_cell_all']['q75']:.2f}] (n={q4['n_matched']}, {q4['n_excluded_edge']} excluded for an edge ν*). "
             f"Trend: slope of log ratio on log tokens = {t.get('b', float('nan')):.3f} "
             f"[{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}] (cluster bootstrap); budget-level slope "
             f"{q4['budget_trend_slope']:.3f}.\n")
    L += ["## Q5 per-cell ν* distribution\n",
          "| group | prefix | n | median | IQR | p5 / p95 | at 1e2 edge | at 1e-2 edge |", "|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        for n in map(str, PREFIXES):
            r = rep["Q5"][g][n]
            L.append(f"| {g} | {n} | {r['n']} | {f(r['median'])} | {f(r['q25'])}–{f(r['q75'])} | {f(r['p5'])} / "
                     f"{f(r['p95'])} | {100 * r['frac_edge_hi']:.1f}% | {100 * r['frac_edge_lo']:.1f}% |")
    L.append("\nEdge experts (layer, expert, group: cells (seed, prefix, side)):\n")
    for x in rep["Q5"]["edge_experts"]:
        L.append(f"- L{x['layer']} e{x['expert']} {x['group']}{' (T1 outlier)' if x['t1_outlier'] else ''}: "
                 + ", ".join(f"s{s}/B{n}/{sd}" for s, n, sd in x["cells"]))
    L.append("\nT1 outlier experts, down ν* (seed 0, seed 1) vs layer median:\n")
    for x in rep["Q5"]["t1_outliers_down"]:
        L.append(f"- L{x['layer']} e{x['expert']} B{x['n']}: " + ", ".join(f(v) for v in x["nu_star"]) +
                 f" (layer median {f(x['layer_median'])})")
    L.append("\nFigures: `figures/mean_curves.png`, `figures/nu_star_vs_ratio.png`, `figures/nu_star_hist.png`.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
