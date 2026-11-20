"""Report for diagnostics/theorem1_check.py -> runs_theorem1_check/report.{json,md} + figures.

Implements the frozen protocol (runs_theorem1_check/protocol.json): T1 verdict (gate), T2-T4 descriptive.

  python diagnostics/report_theorem1_check.py
"""
from __future__ import annotations
import json, math, os, sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theorem1_check import (NU_CURVE, NU_FINE, NU_GRID, OUT, PROTOCOL, RAW_GRID, RAW_SWEEP,  # noqa: E402
                            read_jsonl)

RATIO_BINS = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, np.inf]   # == damping_threshold.RATIO_BINS
D_IN = {"gu": 2048, "down": 1408}
PREFIXES = [16, 32, 64, 128, 256]
PRIMARY = [0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
ADAPTIVE_BEST = {16: 3.0, 32: 2.0, 64: 1.0, 128: 1.0, 256: 0.5}       # runs_diag/adaptive_damping.md
FIG = f"{OUT}/figures"
COL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#0b0b0b", "ink2": "#52514e",
       "grid": "#e4e3df", "surface": "#fcfcfb"}
NU_COL = {0.01: COL["blue"], 0.1: COL["orange"], 1.0: COL["aqua"]}


def q(x, p):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return float(np.percentile(x, p)) if x.size else float("nan")


def med_iqr(x):
    return dict(median=q(x, 50), q25=q(x, 25), q75=q(x, 75),
                n=int(np.sum([v is not None and np.isfinite(v) for v in x])))


def fmt(m, d=3):
    return f"{m['median']:.{d}f} [{m['q25']:.{d}f}, {m['q75']:.{d}f}]" if m["n"] else "–"


# ---------------------------------------------------------------- loading

def load_grid():
    by = {}
    for r in read_jsonl(RAW_GRID):
        by[(r["seed"], r["layer"], r["n"], r["nu"])] = r
    return by


def load_measured():
    """(seed, layer, n) -> dict(rtn=[per-expert dict], err={nu: [per-expert dict]}) from damping_threshold_raw."""
    out = {}
    for r in read_jsonl("runs_diag/damping_threshold_raw.jsonl"):
        m = out.setdefault((r["seed"], r["layer"], r["n"]), dict(err={}))
        if "rtn" in r:
            m["rtn"] = r["rtn"]
        m["err"].update({float(d): v for d, v in r["err"].items()})
    return out


def is_num(v):
    return isinstance(v, (int, float)) and v is not None and math.isfinite(v)


# ---------------------------------------------------------------- T1

def cell_stats(pred, meas):
    if len(pred) < 3 or np.ptp(pred) == 0 or np.ptp(meas) == 0:
        return float("nan"), None
    rho = float(spearmanr(pred, meas).statistic)
    return rho, abs(int(np.argmin(pred)) - int(np.argmin(meas))) <= 1


def t1(grid, meas):
    """Primary: B1 experts x PRIMARY nu grid vs damping_threshold_raw; secondary: B2 x full grid."""
    keys = sorted({k[:3] for k in grid})
    cells = {"primary": [], "secondary": []}
    for (s, L, n) in keys:
        recs = {nu: grid.get((s, L, n, nu)) for nu in NU_GRID}
        if any(v is None for v in recs.values()):
            continue
        r0 = recs[NU_GRID[0]]
        m = meas.get((s, L, n))
        for e in r0["b1_experts"]:
            for g in ("gu", "down"):
                pred = [recs[nu]["R_ideal"][g].get(str(e)) for nu in PRIMARY]
                if m is None or not all(is_num(v) for v in pred):
                    continue
                rt = m["rtn"][e][g]
                mm = [m["err"][nu][e][g] / rt for nu in PRIMARY]
                if not all(is_num(v) for v in mm):
                    continue
                rho, agree = cell_stats(pred, mm)
                cells["primary"].append(dict(seed=s, layer=L, n=n, expert=e, group=g, tokens=r0["tokens"][e],
                                             spearman=rho, agree=agree, pred=pred, meas=mm,
                                             argmin_pred=PRIMARY[int(np.argmin(pred))],
                                             argmin_meas=PRIMARY[int(np.argmin(mm))]))
        for e in r0["b2_experts"]:
            for g in ("gu", "down"):
                try:
                    pred = [recs[nu]["b2"][str(e)][g]["R_ideal"] for nu in NU_GRID]
                    mm = [recs[nu]["b2"][str(e)]["gptq_err"][g] / recs[nu]["b2"][str(e)]["rtn_err"][g]
                          for nu in NU_GRID]
                    em = [recs[nu]["b2"][str(e)][g]["e_meas"] / recs[nu]["b2"][str(e)][g]["e_rtn"] for nu in NU_GRID]
                except (KeyError, TypeError, ZeroDivisionError):
                    continue
                rho, agree = cell_stats(pred, mm)
                rho_e, agree_e = cell_stats(pred, em)
                cells["secondary"].append(dict(seed=s, layer=L, n=n, expert=e, group=g, tokens=r0["tokens"][e],
                                               spearman=rho, agree=agree, spearman_emeas=rho_e,
                                               agree_emeas=agree_e, pred=pred, meas=mm, emeas=em,
                                               argmin_pred=NU_GRID[int(np.argmin(pred))],
                                               argmin_meas=NU_GRID[int(np.argmin(mm))]))
    return cells


def summarize_t1(cells):
    out = {}
    for g in ("gu", "down"):
        cg = [c for c in cells if c["group"] == g]
        rows = {}
        for n in PREFIXES + ["all"]:
            cn = [c for c in cg if n == "all" or c["n"] == n]
            if not cn:
                continue
            ag = [c["agree"] for c in cn if c["agree"] is not None]
            rows[str(n)] = dict(cells=len(cn), median_spearman=q([c["spearman"] for c in cn], 50),
                                frac_argmin_agree=float(np.mean(ag)) if ag else float("nan"),
                                median_argmin_pred=q([c["argmin_pred"] for c in cn], 50),
                                median_argmin_meas=q([c["argmin_meas"] for c in cn], 50),
                                frac_pred_argmin_lt_meas=float(np.mean([c["argmin_pred"] < c["argmin_meas"] for c in cn])),
                                frac_pred_argmin_gt_meas=float(np.mean([c["argmin_pred"] > c["argmin_meas"] for c in cn])),
                                median_level_ratio_at_001=q([c["pred"][0] / c["meas"][0] for c in cn], 50),
                                median_level_ratio_at_1=q([c["pred"][PRIMARY.index(1.0) if len(c["pred"]) == 10
                                                                     else NU_GRID.index(1.0)] /
                                                           c["meas"][PRIMARY.index(1.0) if len(c["pred"]) == 10
                                                                     else NU_GRID.index(1.0)] for c in cn], 50))
        out[g] = rows
    return out


def verdict(summary):
    if not all("all" in summary[g] for g in ("gu", "down")):
        return "INCOMPLETE", {"gu": None, "down": None}
    ok = {g: (summary[g]["all"]["frac_argmin_agree"] >= 0.70 and summary[g]["all"]["median_spearman"] >= 0.8)
          for g in ("gu", "down")}
    label = ("T1-PASS-BOTH" if all(ok.values()) else "T1-PASS-GU-ONLY" if ok["gu"] else
             "T1-PASS-DOWN-ONLY" if ok["down"] else "T1-FAIL")
    return label, ok


def best_nu_per_prefix(cells):
    """Mean over experts of R_ideal / measured ratio per nu -> argmin per prefix and group (descriptive)."""
    out = {}
    for g in ("gu", "down"):
        out[g] = {}
        for n in PREFIXES:
            cn = [c for c in cells if c["group"] == g and c["n"] == n]
            if not cn:
                continue
            P = np.mean([c["pred"] for c in cn], 0)
            Mm = np.mean([c["meas"] for c in cn], 0)
            out[g][str(n)] = dict(pred_best=PRIMARY[int(np.argmin(P))], meas_best=PRIMARY[int(np.argmin(Mm))],
                                  pred_curve=P.tolist(), meas_curve=Mm.tolist())
    return out


# ---------------------------------------------------------------- T2

def t2(grid):
    out = {}
    for g in ("gu", "down"):
        out[g] = {}
        for nu in NU_GRID:
            vals = [c[g] for (s, L, n, v), r in grid.items() if v == nu for c in r["b2"].values()
                    if isinstance(c.get(g), dict) and "gap_R1" in c[g]]
            if not vals:
                continue
            out[g][str(nu)] = dict(
                cells=len(vals),
                gap_R1=med_iqr([x["gap_R1"] for x in vals]), gap_R2=med_iqr([x["gap_R2"] for x in vals]),
                clamp_frac_W=med_iqr([x["clamp_frac_W"] for x in vals]),
                clamp_frac_work=med_iqr([x["clamp_frac_work"] for x in vals]),
                emeas_over_Rideal_rtn=med_iqr([x["e_meas"] / x["e_rtn"] / x["R_ideal"] for x in vals]),
                # remaining factor (not in the frozen decomposition): (e_meas/e_rtn)/R_ideal = (1+R1)(1+R2)(1+R3),
                # 1+R3 = mean_ce2_gptq * tr(S) / e_rtn, i.e. GPTQ's mean residual variance vs RTN's S-weighted error
                gap_R3=med_iqr([x["e_meas"] / x["e_rtn"] / x["R_ideal"] / ((1 + x["gap_R1"]) * (1 + x["gap_R2"])) - 1
                                for x in vals]),
                delta_recon_relerr_max=float(np.max([x["delta_recon_relerr"] for x in vals])),
                n_dead_max=int(np.max([x["n_dead"] for x in vals])))
    return out


# ---------------------------------------------------------------- T3 / T4 (Part C)

def load_sweep():
    raw = read_jsonl(RAW_SWEEP)
    return ([r for r in raw if r.get("kind") == "curve"], [r for r in raw if r.get("kind") == "anchor"],
            [r for r in raw if r.get("invariant")])


def t3(grid, curves, anchors):
    observed = json.load(open("runs_diag/damping_threshold.json"))["threshold"]
    out = dict(binned={}, partC_peak={}, anchors={})
    for nu in (0.01, 1.0):
        out["binned"][str(nu)] = {}
        for g in ("gu", "down"):
            pts = [(r["tokens"][int(e)] / D_IN[g], v) for (s, L, n, v_), r in grid.items() if v_ == nu
                   for e, v in r["R_ideal"][g].items() if is_num(v)]
            rr = np.array([p[0] for p in pts])
            vv = np.array([p[1] for p in pts])
            rows = []
            obs = observed.get(str(nu), {}).get(g, {})
            for i, (lo, hi) in enumerate(zip(RATIO_BINS[:-1], RATIO_BINS[1:])):
                m = (rr >= lo) & (rr < hi)
                rows.append(dict(bin=[lo, hi], n=int(m.sum()),
                                 R_ideal_mean=float(vv[m].mean()) if m.sum() >= 20 else float("nan"),
                                 R_ideal_mean_any=float(vv[m].mean()) if m.sum() else float("nan"),
                                 observed=(obs.get("ratio") or [float("nan")] * 10)[i],
                                 observed_n=(obs.get("n") or [0] * 10)[i]))
            pk = [x["R_ideal_mean"] for x in rows]
            out["binned"][str(nu)][g] = dict(rows=rows, peak_bin_pred=(
                None if np.all(np.isnan(pk)) else rows[int(np.nanargmax(pk))]["bin"]),
                peak_bin_obs=obs.get("peak_bin"))
    for g in ("gu", "down"):
        peaks = []
        for r in curves:
            if r["group"] != g:
                continue
            per = {}
            for s in r["subsets"]:
                v = s.get("R_curve", {}).get("0.01")
                if is_num(v):
                    per.setdefault(s["rho"], []).append((s["ratio"], v))
            pts = sorted((np.mean([a for a, _ in l]), np.mean([b for _, b in l])) for l in per.values())
            if not pts:
                continue
            i = int(np.argmax([b for _, b in pts]))
            peaks.append(dict(seed=r["seed"], layer=r["layer"], expert=r["expert"], peak_ratio=pts[i][0],
                              peak_R=pts[i][1], at_edge=i in (0, len(pts) - 1),
                              max_ratio=pts[-1][0]))
        out["partC_peak"][g] = dict(per_expert=peaks, peak_ratio=med_iqr([p["peak_ratio"] for p in peaks]),
                                    peak_R=med_iqr([p["peak_R"] for p in peaks]),
                                    n_at_edge=int(sum(p["at_edge"] for p in peaks)))
    rows = {}
    for r in anchors:
        for s in r["subsets"]:
            for nu in ("0.01", "1.0"):
                c = s.get(nu)
                if not isinstance(c, dict) or "e_meas" not in c:
                    continue
                rows.setdefault((s["rho"], nu), []).append(dict(
                    ratio=s["ratio"], R_ideal=c["R_ideal"], emeas_ratio=c["e_meas"] / c["e_rtn"],
                    tune_ratio=c["gptq_err"]["down"] / r["rtn_err"]["down"], gap_R1=c["gap_R1"], gap_R2=c["gap_R2"]))
    out["anchors"] = {f"{rho}|{nu}": dict(ratio=q([x["ratio"] for x in v], 50),
                                          R_ideal=med_iqr([x["R_ideal"] for x in v]),
                                          emeas_ratio=med_iqr([x["emeas_ratio"] for x in v]),
                                          tune_ratio=med_iqr([x["tune_ratio"] for x in v]),
                                          gap_R1=med_iqr([x["gap_R1"] for x in v]), n=len(v))
                      for (rho, nu), v in sorted(rows.items())}
    return out


def fit_slope(pts, rng, nboot=2000):
    """pts: list of (cluster, x=log(d/n), y=log nu*). OLS slope b with cluster bootstrap 95% CI."""
    if len(pts) < 5:
        return dict(n=len(pts))
    x = np.array([p[1] for p in pts])
    y = np.array([p[2] for p in pts])
    b, a = np.polyfit(x, y, 1)
    cl = sorted({p[0] for p in pts})
    idx = {c: [i for i, p in enumerate(pts) if p[0] == c] for c in cl}
    bs = []
    for _ in range(nboot):
        pick = sum((idx[cl[j]] for j in rng.integers(0, len(cl), len(cl))), [])
        if np.ptp(x[pick]) > 0:
            bs.append(np.polyfit(x[pick], y[pick], 1)[0])
    return dict(n=len(pts), clusters=len(cl), a=float(a), b=float(b),
                ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def t4(curves):
    rng = np.random.default_rng(0)
    out = {}
    allpts = {"lt": [], "gt": []}
    for g in ("gu", "down"):
        pts = {"lt": [], "gt": []}
        n_edge = n_tot = 0
        for r in curves:
            if r["group"] != g:
                continue
            for s in r["subsets"]:
                if "nu_star" not in s:
                    continue
                n_tot += 1
                if s["edge"]:
                    n_edge += 1
                    continue
                side = "lt" if s["n"] < r["d"] else "gt"
                p = ((r["seed"], r["layer"], r["expert"], g), math.log(r["d"] / s["n"]), math.log(s["nu_star"]))
                pts[side].append(p)
                allpts[side].append(p)
        out[g] = dict(n_subsets=n_tot, n_edge=n_edge,
                      fit_n_lt_d=fit_slope(pts["lt"], rng), fit_n_gt_d=fit_slope(pts["gt"], rng),
                      gs_slope_note="nu_GS: slope b = 1 for n<d (nu_GS=(d/n)), b = 1/3 for n>d")
    out["pooled"] = dict(fit_n_lt_d=fit_slope(allpts["lt"], rng), fit_n_gt_d=fit_slope(allpts["gt"], rng))
    # nu* at the median B-prefix token counts, for the adaptive_damping comparison
    out["R_star_vs_R_GS"] = {g: med_iqr([s["R_GS"] / s["R_star"] - 1 for r in curves if r["group"] == g
                                         for s in r["subsets"] if is_num(s.get("R_GS")) and is_num(s.get("R_star"))])
                             for g in ("gu", "down")}
    return out


# ---------------------------------------------------------------- figures

def figures(grid, curves, anchors, cells, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"axes.edgecolor": COL["ink2"], "axes.labelcolor": COL["ink"], "xtick.color": COL["ink2"],
                         "ytick.color": COL["ink2"], "axes.grid": True, "grid.color": COL["grid"],
                         "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": COL["surface"], "axes.facecolor": COL["surface"], "font.size": 10})

    # 1. R_ideal vs n/d at three nu (Part C), one panel per group, binned medians + anchors
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, g in zip(axs, ("gu", "down")):
        for nu in NU_CURVE:
            pts = [(s["ratio"], s["R_curve"][str(nu)]) for r in curves if r["group"] == g for s in r["subsets"]
                   if is_num(s["R_curve"].get(str(nu)))]
            if not pts:
                continue
            x, y = np.array(pts).T
            ax.scatter(x, y, s=8, color=NU_COL[nu], alpha=0.18, linewidths=0)
            edges = np.quantile(x, np.linspace(0, 1, 16))
            mx = [np.median(x[(x >= a) & (x <= b)]) for a, b in zip(edges[:-1], edges[1:])]
            my = [np.median(y[(x >= a) & (x <= b)]) for a, b in zip(edges[:-1], edges[1:])]
            ax.plot(mx, my, color=NU_COL[nu], lw=2, label=f"ν = {nu} (predicted)")
        if g == "down":
            for nu, mk in (("0.01", "o"), ("1.0", "s")):
                a = [v for k, v in rep["T3"]["anchors"].items() if k.endswith("|" + nu)]
                if a:
                    ax.plot([v["ratio"] for v in a], [v["tune_ratio"]["median"] for v in a], mk, ms=8,
                            mfc="none", mec=NU_COL[float(nu)], mew=2, ls="none",
                            label=f"measured tune ratio, ν = {nu}")
        ax.axhline(1, color=COL["ink2"], lw=1, ls="--")
        ax.axvline(1, color=COL["ink2"], lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("routed tokens / d_in")
        ax.set_ylabel("R_ideal (predicted GPTQ / RTN)")
        ax.set_title(f"{'gate+up (d=2048)' if g == 'gu' else 'down (d=1408)'}", color=COL["ink"], loc="left")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/r_ideal_vs_ratio.png", dpi=150)
    plt.close(fig)

    # 2. nu* vs n/d with George-Saab and adaptive-damping optima
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    tok_med = {}
    for (s, L, n, nu), r in grid.items():
        tok_med.setdefault(n, []).extend(r["tokens"])
    for ax, g in zip(axs, ("gu", "down")):
        pts = [(s["ratio"], s["nu_star"], s["edge"]) for r in curves if r["group"] == g for s in r["subsets"]
               if "nu_star" in s]
        if pts:
            x = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            ed = np.array([p[2] for p in pts])
            ax.scatter(x[~ed], y[~ed], s=9, color=COL["blue"], alpha=0.35, linewidths=0, label="ν* (Part C)")
            if ed.any():
                ax.scatter(x[ed], y[ed], s=12, marker="x", color=COL["ink2"], label="ν* at grid edge")
            xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
            d = D_IN[g]
            ax.plot(xs, [d / ((r * d) ** (1 / 3) * min(r * d, d) ** (2 / 3)) for r in xs], color=COL["orange"],
                    lw=2, label="George–Saab ν_GS")
        bx = [np.median(tok_med[n]) / D_IN[g] for n in PREFIXES if n in tok_med]
        by = [ADAPTIVE_BEST[n] for n in PREFIXES if n in tok_med]
        ax.plot(bx, by, "D", color=COL["aqua"], ms=8, label="measured per-budget optimum (all projections)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("routed tokens / d_in")
        ax.set_ylabel("best ν")
        ax.set_title(f"{'gate+up' if g == 'gu' else 'down'}", color=COL["ink"], loc="left")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/nu_star_vs_ratio.png", dpi=150)
    plt.close(fig)

    # 3. measured vs predicted ratio scatter (B1 primary cells), one panel per group, colored by nu
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, g in zip(axs, ("gu", "down")):
        for nu in NU_CURVE:
            i = PRIMARY.index(nu)
            x = [c["pred"][i] for c in cells if c["group"] == g]
            y = [c["meas"][i] for c in cells if c["group"] == g]
            ax.scatter(x, y, s=7, alpha=0.3, color=NU_COL[nu], linewidths=0, label=f"ν = {nu}")
        lim = ax.get_xlim()[0], max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot(lim, lim, color=COL["ink2"], lw=1, ls="--", label="y = x")
        ax.set_xlabel("predicted R_ideal")
        ax.set_ylabel("measured GPTQ / RTN (tune_err_groups)")
        ax.set_title(f"{'gate+up' if g == 'gu' else 'down'}", color=COL["ink"], loc="left")
        ax.legend(frameon=False, fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(f"{FIG}/measured_vs_predicted.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    P = json.load(open(PROTOCOL))
    grid = load_grid()
    meas = load_measured()
    cells = t1(grid, meas)
    S1 = summarize_t1(cells["primary"])
    S1b = summarize_t1(cells["secondary"])
    label, ok = verdict(S1)
    curves, anchors, inv = load_sweep()
    rep = dict(protocol_frozen_utc=P["frozen_utc"], verdict=label, group_pass=ok,
               T1=dict(primary=S1, secondary_B2_full_grid=S1b,
                       secondary_emeas=dict((g, dict(
                           median_spearman=q([c["spearman_emeas"] for c in cells["secondary"] if c["group"] == g], 50),
                           frac_argmin_agree=float(np.mean([c["agree_emeas"] for c in cells["secondary"]
                                                            if c["group"] == g and c["agree_emeas"] is not None]))))
                                            for g in ("gu", "down") if cells["secondary"]),
                       best_nu_per_prefix=best_nu_per_prefix(cells["primary"])),
               T2=t2(grid), T3=t3(grid, curves, anchors), T4=t4(curves),
               partC_invariant=[dict(seed=r["seed"], layer=r["layer"], status=r["invariant"],
                                     max_rel=max(c["rel_frob"] for c in r["checks"]),
                                     tokens_equal=all(c["tokens"] == c["tokens_ref"] for c in r["checks"]))
                                for r in inv],
               coverage=dict(grid_records=len(grid), partC_curves=len(curves), partC_anchor_records=len(anchors),
                             t1_primary_cells=len(cells["primary"]), t1_secondary_cells=len(cells["secondary"])))
    events = read_jsonl(f"{OUT}/grid_events.jsonl")
    rep["run_events"] = [e for e in events if "b1_mode" in e or "repro_check" in e]
    figures(grid, curves, anchors, cells["primary"], rep)
    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1, default=float)
    open(f"{OUT}/report.md", "w").write(render(rep))
    print(render(rep))


def render(rep):
    M = [f"# Theorem-1 check — report", "", f"**Frozen verdict: `{rep['verdict']}`** "
         f"(gu pass={rep['group_pass']['gu']}, down pass={rep['group_pass']['down']}; "
         "pass = ≥70 % argmin agreement within one grid step AND median Spearman ≥ 0.8)", "",
         "## T1 agreement (primary: B1 experts × 10-ν grid measured in damping_threshold_raw)", "",
         "| group | prefix | cells | median Spearman | argmin agree (±1 step) | median argmin pred | "
         "median argmin meas | pred<meas | pred>meas | R/meas @0.01 | R/meas @1.0 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, S in (("primary", rep["T1"]["primary"]),):
        for g in ("gu", "down"):
            for n, r in S[g].items():
                M.append(f"| {g} | {n} | {r['cells']} | {r['median_spearman']:.3f} | {100 * r['frac_argmin_agree']:.1f}% | "
                         f"{r['median_argmin_pred']} | {r['median_argmin_meas']} | "
                         f"{100 * r['frac_pred_argmin_lt_meas']:.0f}% | {100 * r['frac_pred_argmin_gt_meas']:.0f}% | "
                         f"{r['median_level_ratio_at_001']:.3f} | {r['median_level_ratio_at_1']:.3f} |")
    M += ["", "Secondary (B2 experts, full 12-ν grid incl. 0.03/0.05, measured in this run):", "",
          "| group | cells | median Spearman | argmin agree | Spearman vs e_meas ratio | argmin agree vs e_meas |",
          "|---|---|---|---|---|---|"]
    for g in ("gu", "down"):
        r = rep["T1"]["secondary_B2_full_grid"].get(g, {}).get("all")
        e = rep["T1"]["secondary_emeas"].get(g, {})
        if r:
            M.append(f"| {g} | {r['cells']} | {r['median_spearman']:.3f} | {100 * r['frac_argmin_agree']:.1f}% | "
                     f"{e.get('median_spearman', float('nan')):.3f} | {100 * e.get('frac_argmin_agree', float('nan')):.1f}% |")
    M += ["", "Best ν per prefix from expert-mean curves (predicted R_ideal vs measured ratio):", "",
          "| group | " + " | ".join(str(n) for n in PREFIXES) + " |", "|---|" + "---|" * len(PREFIXES)]
    for g in ("gu", "down"):
        bb = rep["T1"]["best_nu_per_prefix"][g]
        M.append(f"| {g} | " + " | ".join(f"{bb[str(n)]['pred_best']} / {bb[str(n)]['meas_best']}"
                                          if str(n) in bb else "–" for n in PREFIXES) + " |")
    M += ["", "## T2 decomposition (B2: 12 experts/layer), median [IQR]", "",
          "| group | ν | cells | R1 gap e_meas/R_w−1 | R2 gap R_w/(mean ce²·tr)−1 | (e_meas/e_rtn)/R_ideal | "
          "R3 gap σ²_GPTQ·trS/e_rtn−1 | clamp frac W | clamp frac working | max Δ-recon relerr |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g in ("gu", "down"):
        for nu, r in rep["T2"][g].items():
            M.append(f"| {g} | {nu} | {r['cells']} | {fmt(r['gap_R1'])} | {fmt(r['gap_R2'])} | "
                     f"{fmt(r['emeas_over_Rideal_rtn'])} | {fmt(r['gap_R3'])} | {fmt(r['clamp_frac_W'], 5)} | "
                     f"{fmt(r['clamp_frac_work'], 5)} | {r['delta_recon_relerr_max']:.1e} |")
    M += ["", "## T3 peak location", ""]
    for nu, gg in rep["T3"]["binned"].items():
        M += [f"### ν = {nu}: mean R_ideal (B1) vs observed mean GPTQ/RTN ratio (damping_threshold §2)", "",
              "| tokens/d_in | gu R_ideal (n) | gu observed (n) | down R_ideal (n) | down observed (n) |",
              "|---|---|---|---|---|"]
        for i in range(len(RATIO_BINS) - 1):
            cell = []
            for g in ("gu", "down"):
                r = gg[g]["rows"][i]
                cell.append(f"{r['R_ideal_mean']:.3f} ({r['n']})" if np.isfinite(r["R_ideal_mean"])
                            else f"– ({r['n']})")
                cell.append(f"{r['observed']:.3f} ({r['observed_n']})" if r["observed"] is not None and
                            np.isfinite(r["observed"]) else f"– ({r['observed_n']})")
            M.append(f"| [{RATIO_BINS[i]}, {RATIO_BINS[i + 1]}) | " + " | ".join(cell) + " |")
        M += ["", f"Peak bin predicted / observed: gu {gg['gu']['peak_bin_pred']} / {gg['gu']['peak_bin_obs']}; "
              f"down {gg['down']['peak_bin_pred']} / {gg['down']['peak_bin_obs']}", ""]
    M += ["### Part C per-expert peak of R_ideal(ν=0.01) over n/d", "",
          "| group | experts | peak n/d median [IQR] | peak R median [IQR] | peak at sweep edge |", "|---|---|---|---|---|"]
    for g, r in rep["T3"]["partC_peak"].items():
        M.append(f"| {g} | {len(r['per_expert'])} | {fmt(r['peak_ratio'], 2)} | {fmt(r['peak_R'])} | {r['n_at_edge']} |")
    M += ["", "Down anchors (actual GPTQ on Part C subsets), median over experts × draws:", "",
          "| target ρ | ν | n/d | R_ideal | e_meas/e_rtn | tune_err ratio | R1 gap |", "|---|---|---|---|---|---|---|"]
    for k, v in rep["T3"]["anchors"].items():
        rho, nu = k.split("|")
        M.append(f"| {rho} | {nu} | {v['ratio']:.2f} | {v['R_ideal']['median']:.3f} | {v['emeas_ratio']['median']:.3f} | "
                 f"{v['tune_ratio']['median']:.3f} | {v['gap_R1']['median']:+.3f} |")
    M += ["", "## T4 best ν vs n/d (Part C): log ν* = a + b·log(d/n)", "",
          "| group | side | points | clusters | b | 95% CI (cluster bootstrap) | a |", "|---|---|---|---|---|---|---|"]
    for g in ("gu", "down", "pooled"):
        for side in ("fit_n_lt_d", "fit_n_gt_d"):
            f = rep["T4"][g][side]
            if "b" in f:
                M.append(f"| {g} | {'n<d' if side.endswith('lt_d') else 'n>d'} | {f['n']} | {f['clusters']} | "
                         f"{f['b']:.3f} | [{f['ci95'][0]:.3f}, {f['ci95'][1]:.3f}] | {f['a']:.3f} |")
            else:
                M.append(f"| {g} | {side} | {f['n']} | – | – | – | – |")
    for g in ("gu", "down"):
        M.append(f"\n{g}: {rep['T4'][g]['n_edge']}/{rep['T4'][g]['n_subsets']} subsets with ν* at the fine-grid edge "
                 f"(excluded from the fit); R(ν_GS)/R(ν*)−1 median [IQR] = {fmt(rep['T4']['R_star_vs_R_GS'][g], 4)}")
    M += ["", "## Invariants / run events", ""]
    for r in rep["partC_invariant"]:
        M.append(f"- Part C Ĥ invariant seed {r['seed']} L{r['layer']}: {r['status']}, max rel Frobenius "
                 f"{r['max_rel']:.1e}, token counts equal: {r['tokens_equal']}")
    for e in rep["run_events"]:
        M.append(f"- {json.dumps(e)}")
    M += ["", f"Coverage: {json.dumps(rep['coverage'])}", "",
          "Figures: figures/r_ideal_vs_ratio.png, figures/nu_star_vs_ratio.png, figures/measured_vs_predicted.png"]
    return "\n".join(M) + "\n"


if __name__ == "__main__":
    main()
