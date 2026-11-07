"""Matched-warm-start typicality experiment: validation, analysis, figures (CPU).

    --validate F       one run: B unique, frozen settings, prescribed warm start
                       (exact ordered ids + hash), pool membership/fingerprint
    --check-warm SEED  the six runs of SEED share the exact ordered warm start
    (default)          <outdir>/summary.{md,json} and <outdir>/figures/*.{png,pdf}

E(X, S, s): mean expert reconstruction error, pool X in {F, T, H}, strategy
S in {R = random, C = coverage_lp}, seed s. Every run of seed s starts from the
same 32 sequences W_s (drawn from T & H). Lower is better.
  P_X(s) = E(X,C,s) - E(X,R,s)    coverage penalty in pool X
  I(s)   = P_T(s) - P_F(s)        < 0: typical eligibility shrinks the penalty
  J(s)   = P_T(s) - P_H(s)        < 0: ... more than a random half does
  D(s)   = E(T,C,s) - E(F,C,s)    effect of typical eligibility on coverage_lp

Prespecified decision (fixed before any matched result was read):
  SUPPORTED      D < 0, I < 0, J < 0 on all 3 seeds and mean |D| / E(F,C) >= 0.5 %
  PARTIAL        D < 0 and I < 0 on >= 2 seeds and not SUPPORTED
                 (incl. consistent signs but an effect below 0.5 %)
  NOT-SUPPORTED  otherwise (D >= 0 or I >= 0 on >= 2 seeds)
Three seeds = three paired replicates; no significance claims; experts are
not independent replicates.
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import report_minimal_allocation as RM               # noqa: E402
import report_typicality as RT                       # noqa: E402
from moe_calib import typicality as TY               # noqa: E402
from moe_calib.frozen import load_frozen             # noqa: E402

POOLS = {"F": "full", "T": "typical_half", "H": "random_half"}
STRATS = {"R": "random", "C": "coverage_lp"}
SEEDS, B, WARM, EFFECT = (0, 1, 2), 128, 32, 0.005


def warm_file(outdir):
    return json.load(open(os.path.join(outdir, "warm_starts.json")))


def run_path(outdir, pool, strat, seed, ws):
    h = ws["seeds"][str(seed)]["hash"][:8]
    tag = "" if pool == "full" else f"_pool-{pool}"
    return os.path.join(outdir, f"e3_{strat}_B{B}_b4_d1_s{seed}{tag}_warm-{h}.json")


def validate(path, outdir, cfg, ws=None):
    ws = ws or warm_file(outdir)
    r = RM.validate(path, cfg)
    s = r["summary"]
    seed = s["seed"]
    want = ws["seeds"][str(seed)]
    errs = []
    wf = s.get("warm_start_fixed") or {}
    if wf.get("hash") != want["hash"] or r["warm_start"] != want["ids"] or r["chosen"][:WARM] != want["ids"]:
        errs.append("warm start differs from the prescribed ordered W_s")
    if len(r["chosen"]) - WARM != B - WARM or set(r["chosen"][WARM:]) & set(want["ids"]):
        errs.append("not exactly 96 additional sequences after the warm start")
    cs = s.get("candidate_subset")
    pool = cs["name"] if cs else "full"
    if cs and cs["fingerprint"] != ws["pools"].get(pool):
        errs.append(f"candidate pool {pool} fingerprint not in the warm-start file")
    ids = json.load(open(os.path.join(ws["pool_dir"], "pools.json")))["pools"][pool]["ids"]
    if not set(r["chosen"]) <= set(ids):
        errs.append(f"selection outside pool {pool}")
    if errs:
        raise SystemExit(f"INVALID {path}:\n  " + "\n  ".join(errs))
    return r


def check_warm(outdir, cfg, seed):
    ws = warm_file(outdir)
    lists = {}
    for p in POOLS.values():
        for st in STRATS.values():
            r = validate(run_path(outdir, p, st, seed, ws), outdir, cfg, ws)
            lists[f"{p}/{st}"] = r["chosen"][:WARM]
    ref = ws["seeds"][str(seed)]["ids"]
    bad = [k for k, v in lists.items() if v != ref]
    if bad:
        raise SystemExit(f"seed {seed}: warm start differs in {bad}")
    print(f"seed {seed}: all 6 runs share the exact ordered warm start {ws['seeds'][str(seed)]['hash']}")


# --------------------------------------------------------------- analysis

def build(outdir, cfg, prev_dir):
    ws = warm_file(outdir)
    z = np.load(os.path.join(outdir, "routing_profile_v2.npz"))
    layers, E = z["layers"].tolist(), int(z["E"])
    C = z["C"].astype(float)
    row = {int(s): i for i, s in enumerate(z["pool_ids"])}
    ref = TY.pool_reference(C, len(layers), E)
    scores = TY.sequence_scores(C, len(layers), E, ref)
    runs, missing = {}, []
    for pk, p in POOLS.items():
        for sk, st in STRATS.items():
            for sd in SEEDS:
                f = run_path(outdir, p, st, sd, ws)
                if os.path.exists(f):
                    runs[(pk, sk, sd)] = validate(f, outdir, cfg, ws)
                else:
                    missing.append(f"{p}/{st}/s{sd}")
    if len({json.dumps(r["eval_universe"]) for r in runs.values()}) > 1:
        raise SystemExit("runs do not share one evaluation universe")
    seeds = sorted({k[2] for k in runs})
    groups = [(p, s) for p in POOLS for s in STRATS
              if seeds and all((p, s, sd) in runs for sd in seeds)]
    res = dict(seeds=seeds, missing=missing, warm_starts={s: v["hash"] for s, v in ws["seeds"].items()},
               eligible_size=ws["eligible"]["size"], groups={}, runs={}, primary={}, relative={},
               fc_vs_tc={}, overlap={}, warm_check={})
    Ev = lambda p, s, sd: runs[(p, s, sd)]["summary"]["recon_mean"]
    for sd in seeds:
        res["warm_check"][str(sd)] = all(runs[(p, s, sd)]["chosen"][:WARM] == ws["seeds"][str(sd)]["ids"]
                                         for p in POOLS for s in STRATS if (p, s, sd) in runs)
    for (p, s, sd), r in runs.items():
        sm = r["summary"]
        m = RT.set_metrics(r, C, row, ref, scores, layers, E)
        lp = sum(t["n_lp_positive_value_selected"] for t in r["trajectory"])
        fb = sum(t["n_random_fallback_sequences"] + t["n_zero_marginal_random_fill"] for t in r["trajectory"])
        res["runs"][f"{p}{s}_s{sd}"] = dict(
            recon_mean=sm["recon_mean"], expert_median=sm["recon_expert_median"],
            expert_p90=sm["recon_expert_p90"], expert_p95=sm["recon_expert_p95"],
            per_layer=sm["recon_per_layer"], warm_start=r["warm_start"], chosen=r["chosen"],
            lp_selected=lp if s == "C" else None, fallback_or_fill=fb if s == "C" else None,
            selection_s=sm["cost"]["selection_s"], hessian_s=sm["cost"]["hessian_replay_s"],
            gptq_s=sm["cost"]["gptq_s"],
            routing_profile_full_forward_equiv_s=sm["cost"].get("routing_full_forward_equiv_s", 0.0),
            peak_gpu_mem_gb=sm["peak_gpu_mem_gb"], **m)
    for p, s in groups:
        rs = [res["runs"][f"{p}{s}_s{sd}"] for sd in seeds]
        v = np.array([r["recon_mean"] for r in rs])
        g = dict(pool=POOLS[p], strategy=STRATS[s], recon_mean=float(v.mean()),
                 recon_sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                 recon_by_seed=dict(zip(map(str, seeds), v.tolist())))
        for k in ("expert_median", "expert_p90", "expert_p95", "d_mean", "d_post_warm", "entropy_mean",
                  "js_set_vs_original_pool", "median_expert_tokens", "gini_expert_tokens",
                  "median_effective_sources_per_expert", "mean_effective_sources_per_expert",
                  "selection_s", "hessian_s", "gptq_s", "routing_profile_full_forward_equiv_s"):
            g[k] = float(np.mean([r[k] for r in rs]))
        g["peak_gpu_mem_gb"] = float(max(r["peak_gpu_mem_gb"] or 0 for r in rs))
        g["per_layer"] = {str(L): float(np.mean([r["per_layer"][str(L)] for r in rs])) for L in layers}
        g["set_routing_l1_by_layer"] = {str(L): float(np.mean([r["set_routing_l1_by_layer"][str(L)] for r in rs]))
                                        for L in layers}
        if s == "C":
            g["lp_selected_total"] = int(sum(r["lp_selected"] for r in rs))
            g["fallback_or_fill_total"] = int(sum(r["fallback_or_fill"] for r in rs))
        res["groups"][f"{p}{s}"] = g
    if len(groups) == 6 and len(seeds) == 3:
        P = {p: {str(sd): Ev(p, "C", sd) - Ev(p, "R", sd) for sd in seeds} for p in POOLS}
        Prel = {p: {str(sd): P[p][str(sd)] / Ev(p, "R", sd) for sd in seeds} for p in POOLS}
        I = {k: P["T"][k] - P["F"][k] for k in P["F"]}
        J = {k: P["T"][k] - P["H"][k] for k in P["F"]}
        D = {str(sd): Ev("T", "C", sd) - Ev("F", "C", sd) for sd in seeds}
        Drel = {str(sd): D[str(sd)] / Ev("F", "C", sd) for sd in seeds}
        st = lambda d: dict(by_seed=d, mean=float(np.mean(list(d.values()))),
                            sd=float(np.std(list(d.values()), ddof=1)), n_negative=sum(v < 0 for v in d.values()))
        res["primary"] = dict(P={p: st(v) for p, v in P.items()}, P_rel={p: st(v) for p, v in Prel.items()},
                              I=st(I), J=st(J), D=st(D), D_rel=st(Drel))
        rel = lambda a, b: {str(sd): (Ev(*a, sd) - Ev(*b, sd)) / Ev(*b, sd) for sd in seeds}
        for name, a, b in (("TC_vs_FC", ("T", "C"), ("F", "C")), ("TR_vs_FR", ("T", "R"), ("F", "R")),
                           ("TC_vs_TR", ("T", "C"), ("T", "R")), ("FC_vs_FR", ("F", "C"), ("F", "R")),
                           ("HC_vs_FC", ("H", "C"), ("F", "C")), ("TC_vs_HC", ("T", "C"), ("H", "C")),
                           ("HC_vs_HR", ("H", "C"), ("H", "R")), ("TC_vs_FR", ("T", "C"), ("F", "R"))):
            res["relative"][name] = st(rel(a, b))
        for sd in seeds:
            fc, tc = res["runs"][f"FC_s{sd}"], res["runs"][f"TC_s{sd}"]
            res["fc_vs_tc"][str(sd)] = dict(
                recon=tc["recon_mean"] - fc["recon_mean"],
                d_post_warm=tc["d_post_warm"] - fc["d_post_warm"],
                d_mean=tc["d_mean"] - fc["d_mean"],
                eff_sources_median=tc["median_effective_sources_per_expert"] - fc["median_effective_sources_per_expert"],
                gini=tc["gini_expert_tokens"] - fc["gini_expert_tokens"],
                js=tc["js_set_vs_original_pool"] - fc["js_set_vs_original_pool"],
                per_layer_rel={L: (tc["per_layer"][L] - fc["per_layer"][L]) / fc["per_layer"][L] for L in fc["per_layer"]})
        for a, b in (("TC", "FC"), ("HC", "FC"), ("TC", "TR"), ("TR", "FR"), ("TC", "HC")):
            res["overlap"][f"{a}_vs_{b}"] = {str(sd): len(set(res["runs"][f"{a}_s{sd}"]["chosen"])
                                                         & set(res["runs"][f"{b}_s{sd}"]["chosen"])) / B
                                             for sd in seeds}
        pr = res["primary"]
        nD, nI, nJ = pr["D"]["n_negative"], pr["I"]["n_negative"], pr["J"]["n_negative"]
        if nD == 3 and nI == 3 and nJ == 3 and -pr["D_rel"]["mean"] >= EFFECT:
            lab = "MATCHED-TYPICALITY-SUPPORTED"
        elif nD >= 2 and nI >= 2:
            lab = "MATCHED-TYPICALITY-PARTIAL"
        else:
            lab = "MATCHED-TYPICALITY-NOT-SUPPORTED"
        res["decision"] = dict(label=lab, nD=nD, nI=nI, nJ=nJ, D_rel_mean=pr["D_rel"]["mean"],
                               outcome_E_TC_beats_TR_all_seeds=res["relative"]["TC_vs_TR"]["n_negative"] == 3)
        prev = os.path.join(prev_dir, "summary.json")
        if os.path.exists(prev):
            q = json.load(open(prev))
            G = q["groups"]
            res["unmatched"] = dict(
                FC=G["FC"]["recon_mean"], TC=G["TC"]["recon_mean"], FR=G["FR"]["recon_mean"], TR=G["TR"]["recon_mean"],
                P_F=q["primary"]["P_mean"]["F"], P_T=q["primary"]["P_mean"]["T"], I=q["primary"]["I_mean"],
                J=q["primary"]["T_minus_H_mean"],
                D_rel=float(np.mean(list(q["relative"]["TC_vs_FC"]["by_seed"].values()))),
                overlap_TC_FC=q["overlap"]["TC_vs_FC"])
            res["shrinkage_of_D_vs_unmatched"] = pr["D_rel"]["mean"] / res["unmatched"]["D_rel"]
    else:
        res["decision"] = dict(label="INCOMPLETE")
    return res, layers


# ---------------------------------------------------------------- figures

COL = {"F": "#2a78d6", "T": "#eb6834", "H": "#1baf7a"}        # reference palette slots 1-3
MARK = {"R": "o", "C": "s"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def figures(res, layers, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(outdir, "figures")
    os.makedirs(fd, exist_ok=True)
    seeds = res["seeds"]
    R = res["runs"]
    out = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(fd, f"{name}.{ext}"), dpi=200)
        plt.close(fig)
        out.append(os.path.join(fd, name))

    # 1 paired Full vs Typical coverage_lp per seed
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    pairs = [(sd, R[f"FC_s{sd}"]["recon_mean"] * 1e3, R[f"TC_s{sd}"]["recon_mean"] * 1e3) for sd in seeds]
    for sd, a, b in pairs:
        ax.plot([0, 1], [a, b], color=INK2, lw=1.2, zorder=1)
        ax.scatter([0], [a], s=46, color=COL["F"], marker="s", zorder=2, edgecolor="white", lw=1.5)
        ax.scatter([1], [b], s=46, color=COL["T"], marker="s", zorder=2, edgecolor="white", lw=1.5)
    ax.text(0.62, 0.95, "Typical − Full, per seed:\n" + "\n".join(
        f"seed {sd}: {100 * (b - a) / a:+.2f}%" for sd, a, b in pairs),
        transform=ax.transAxes, fontsize=8, color=INK2, va="top")
    ax.text(0.0, -0.17, "y-axis does not start at zero (paired differences ≈ 2.5 %)", transform=ax.transAxes,
            fontsize=7, color=INK2)
    ax.set_xticks([0, 1], ["Full pool", "Typical half"])
    ax.set_xlim(-0.3, 1.75)
    ax.set_ylabel("Mean expert recon. error (×10⁻³)", color=INK, fontsize=9)
    ax.set_title("Coverage-LP, matched warm start", fontsize=10, color=INK, loc="left")
    _style(ax)
    save(fig, "fig1_paired_full_vs_typical_coverage")

    # 2 coverage penalty by pool (absolute, ×1e-6), seeds as points, mean as bar
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    P = res["primary"]["P"]
    for i, p in enumerate("FTH"):
        vals = [P[p]["by_seed"][str(sd)] * 1e6 for sd in seeds]
        ax.bar(i, np.mean(vals), width=0.55, color=COL[p], alpha=0.35, edgecolor="none")
        ax.scatter(i + np.array([-0.12, 0, 0.12])[:len(vals)], vals, color=COL[p], s=30, zorder=3,
                   edgecolor="white", lw=1)
        ax.annotate(f"{np.mean(vals):+.1f}", (i, max(max(vals), 0) + 3), ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(3), ["Full", "Typical half", "Random half"])
    ax.set_ylabel("Coverage-LP − Random error (×10⁻⁶)", color=INK, fontsize=9)
    ax.set_title("Coverage penalty by candidate pool (points = seeds)", fontsize=10, color=INK, loc="left")
    _style(ax)
    save(fig, "fig2_coverage_penalty_by_pool")

    # 3 atypicality vs error, all 18 runs
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for p in "FTH":
        for s in "RC":
            xs = [R[f"{p}{s}_s{sd}"]["d_post_warm"] for sd in seeds]
            ys = [R[f"{p}{s}_s{sd}"]["recon_mean"] * 1e3 for sd in seeds]
            ax.scatter(xs, ys, color=COL[p], marker=MARK[s], s=40, edgecolor="white", lw=1,
                       label=f"{POOLS[p].replace('_', ' ')} · {STRATS[s].replace('_', '-')}")
    ax.set_xlabel("Routing atypicality d(s) of the 96 post-warm-start picks", color=INK, fontsize=9)
    ax.set_ylabel("Mean expert recon. error (×10⁻³)", color=INK, fontsize=9)
    ax.set_title("Atypicality vs reconstruction (18 runs)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _style(ax)
    save(fig, "fig3_atypicality_vs_error")

    # 4 per-layer relative effect of typical filtering on coverage_lp
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for i, L in enumerate(layers):
        vals = [res["fc_vs_tc"][str(sd)]["per_layer_rel"][str(L)] * 100 for sd in seeds]
        ax.bar(i, np.mean(vals), width=0.55, color=COL["T"], alpha=0.35, edgecolor="none")
        ax.scatter(i + np.array([-0.12, 0, 0.12])[:len(vals)], vals, color=COL["T"], s=28, zorder=3,
                   edgecolor="white", lw=1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(layers)), [f"L{L}" for L in layers])
    ax.set_ylabel("Typical − Full Coverage-LP error (%)", color=INK, fontsize=9)
    ax.set_title("Per-layer effect of typical eligibility (points = seeds)", fontsize=10, color=INK, loc="left")
    _style(ax)
    save(fig, "fig4_per_layer_effect")
    return out


# ---------------------------------------------------------------- render

pct = lambda x: f"{100 * x:+.2f}%"
e6 = lambda x: f"{1e6 * x:+.1f}"


def render(res, layers):
    G, M = res["groups"], []
    M += [f"# Matched-warm-start typicality experiment (B = {B}, seeds {res['seeds']})", "",
          f"Prespecified decision: **{res['decision']['label']}**.", "",
          f"Every run of seed s starts from the same ordered 32 sequences W_s (drawn from typical ∩ random half, "
          f"{res['eligible_size']} candidates). Warm-start hashes: "
          + ", ".join(f"s{k} {v}" for k, v in res["warm_starts"].items())
          + ". Exact ordered warm-start identity across the 6 conditions: "
          + ", ".join(f"s{k} {'yes' if v else 'NO'}" for k, v in res["warm_check"].items()) + ".", ""]
    if res["missing"]:
        M += ["Missing: " + ", ".join(res["missing"]), ""]
    M += ["## Reconstruction (mean expert relative error, test split; lower is better)", "",
          "| pool | strategy | mean | sd | seed 0 / 1 / 2 | expert median | p90 | p95 |", "|---|---|---|---|---|---|---|---|"]
    for g in G.values():
        M.append(f"| {g['pool']} | {g['strategy']} | {g['recon_mean']:.6f} | "
                 + (f"{g['recon_sd']:.1e}" if g["recon_sd"] else "–") + " | "
                 + " / ".join(f"{v:.6f}" for v in g["recon_by_seed"].values())
                 + f" | {g['expert_median']:.6f} | {g['expert_p90']:.6f} | {g['expert_p95']:.6f} |")
    if res["primary"]:
        pr = res["primary"]
        M += ["", "## Paired interactions (absolute ×1e-6; negative = typical eligibility helps)", "",
              "| quantity | seed 0 | seed 1 | seed 2 | mean | sd | seeds < 0 |", "|---|---|---|---|---|---|---|"]
        rows = [("P_F = E(F,C) − E(F,R)", pr["P"]["F"]), ("P_T = E(T,C) − E(T,R)", pr["P"]["T"]),
                ("P_H = E(H,C) − E(H,R)", pr["P"]["H"]), ("**I = P_T − P_F**", pr["I"]),
                ("**J = P_T − P_H**", pr["J"]), ("**D = E(T,C) − E(F,C)**", pr["D"])]
        for lab, d in rows:
            M.append(f"| {lab} | " + " | ".join(e6(v) for v in d["by_seed"].values())
                     + f" | {e6(d['mean'])} | {1e6 * d['sd']:.1f} | {d['n_negative']}/3 |")
        M.append(f"| D relative to E(F,C) | " + " | ".join(pct(v) for v in pr["D_rel"]["by_seed"].values())
                 + f" | {pct(pr['D_rel']['mean'])} | | {pr['D_rel']['n_negative']}/3 |")
        M.append("| relative penalty P_X / E(X,R): F / T / H | | | | "
                 + " / ".join(pct(pr["P_rel"][p]["mean"]) for p in "FTH") + " | | |")
        M += ["", "## Relative changes ((A − B) / B; negative = A better)", "",
              "| A vs B | seed 0 / 1 / 2 | mean | seeds A better |", "|---|---|---|---|"]
        for k, v in res["relative"].items():
            M.append(f"| {k.replace('_vs_', ' vs ')} | " + " / ".join(pct(x) for x in v["by_seed"].values())
                     + f" | {pct(v['mean'])} | {v['n_negative']}/3 |")
    M += ["", "## Per-layer error (seed mean)", "", "| pool | strategy | " + " | ".join(f"L{L}" for L in layers) + " |",
          "|---|---|" + "---|" * len(layers)]
    for g in G.values():
        M.append(f"| {g['pool']} | {g['strategy']} | " + " | ".join(f"{v:.6f}" for v in g["per_layer"].values()) + " |")
    M += ["", "## Routing, exposure and source diversity (seed mean; typicality vs the ORIGINAL pool)", "",
          "| pool | strategy | d(s) all 128 | d(s) post-warm | entropy | set JS | set L1 per layer | median tokens | Gini | eff. sources median / mean |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g in G.values():
        M.append(f"| {g['pool']} | {g['strategy']} | {g['d_mean']:.4f} | {g['d_post_warm']:.4f} | {g['entropy_mean']:.4f} | "
                 f"{g['js_set_vs_original_pool']:.5f} | "
                 + ", ".join(f"{v:.3f}" for v in g["set_routing_l1_by_layer"].values())
                 + f" | {g['median_expert_tokens']:.0f} | {g['gini_expert_tokens']:.3f} | "
                 f"{g['median_effective_sources_per_expert']:.1f} / {g['mean_effective_sources_per_expert']:.1f} |")
    if res["fc_vs_tc"]:
        M += ["", "## Typical vs Full Coverage-LP within seed (T-C − F-C)", "",
              "| seed | Δ error | Δ d post-warm | Δ set JS | Δ eff. sources (median) | Δ Gini | per-layer Δ error (rel.) |",
              "|---|---|---|---|---|---|---|"]
        for sd, v in res["fc_vs_tc"].items():
            M.append(f"| {sd} | {e6(v['recon'])}e-6 | {v['d_post_warm']:+.4f} | {v['js']:+.5f} | {v['eff_sources_median']:+.1f} | "
                     f"{v['gini']:+.3f} | " + ", ".join(f"L{L} {pct(x)}" for L, x in v["per_layer_rel"].items()) + " |")
        M += ["", "## Selected-set overlap (fraction of 128; 0.25 = warm start only)", "", "| sets | seed 0 / 1 / 2 |", "|---|---|"]
        for k, v in res["overlap"].items():
            M.append(f"| {k.replace('_vs_', ' vs ')} | " + " / ".join(f"{x:.3f}" for x in v.values()) + " |")
    if "unmatched" in res:
        u, pr = res["unmatched"], res["primary"]
        Gm = res["groups"]
        M += ["", "## Unmatched (earlier) vs matched warm start — not pooled", "",
              "| experiment | F-C | T-C | F-R | T-R | P_F | P_T | I (×1e-6) | J (×1e-6) | T-C vs F-C | warm start |",
              "|---|---|---|---|---|---|---|---|---|---|---|",
              f"| unmatched (`runs_typicality/`) | {u['FC']:.6f} | {u['TC']:.6f} | {u['FR']:.6f} | {u['TR']:.6f} | "
              f"{e6(u['P_F'])} | {e6(u['P_T'])} | {e6(u['I'])} | {e6(u['J'])} | {pct(u['D_rel'])} | "
              f"different per pool (T-C/F-C overlap " + "/".join(f"{x:.3f}" for x in u["overlap_TC_FC"].values()) + ") |",
              f"| matched (this run) | {Gm['FC']['recon_mean']:.6f} | {Gm['TC']['recon_mean']:.6f} | {Gm['FR']['recon_mean']:.6f} | "
              f"{Gm['TR']['recon_mean']:.6f} | {e6(pr['P']['F']['mean'])} | {e6(pr['P']['T']['mean'])} | {e6(pr['I']['mean'])} | "
              f"{e6(pr['J']['mean'])} | {pct(pr['D_rel']['mean'])} | identical ordered W_s in all 6 conditions |",
              "", f"Matched D / unmatched D = {res['shrinkage_of_D_vs_unmatched']:.2f}."]
    M += ["", "## Controller and cost (seed mean)", "",
          "| pool | strategy | LP-selected / fill+fallback (3 seeds) | selection s | Hessian s | GPTQ s | routing profile (full-forward equiv.) s | peak GPU GB |",
          "|---|---|---|---|---|---|---|---|"]
    for g in G.values():
        lp = f"{g['lp_selected_total']} / {g['fallback_or_fill_total']}" if "lp_selected_total" in g else "–"
        M.append(f"| {g['pool']} | {g['strategy']} | {lp} | {g['selection_s']:.1f} | {g['hessian_s']:.0f} | {g['gptq_s']:.0f} | "
                 f"{g['routing_profile_full_forward_equiv_s']:.0f} | {g['peak_gpu_mem_gb']:.1f} |")
    M += ["", "Notes: estimand is conditional on a shared warm start drawn from typical ∩ random half. Three seeds = "
          "three paired replicates; no significance claims. Random from a filtered pool also relies on the full-pool "
          "routing profile to define the pool (not charged in its cost field). Parallel runs (2 at a time) inflate "
          "timings. Equal-budget comparison only."]
    return "\n".join(M)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs_typicality_matched")
    ap.add_argument("--prev-dir", default="runs_typicality")
    ap.add_argument("--frozen", default="configs/frozen_b4.json")
    ap.add_argument("--validate", default=None)
    ap.add_argument("--check-warm", type=int, default=None)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_frozen(a.frozen)
    if a.validate:
        validate(a.validate, a.outdir, cfg)
        print(f"valid {a.validate}")
        return 0
    if a.check_warm is not None:
        check_warm(a.outdir, cfg, a.check_warm)
        return 0
    res, layers = build(a.outdir, cfg, a.prev_dir)
    if res["primary"] and not a.no_figures:
        res["figures"] = figures(res, layers, a.outdir)
    json.dump(res, open(os.path.join(a.outdir, "summary.json"), "w"), indent=2)
    md = render(res, layers)
    open(os.path.join(a.outdir, "summary.md"), "w").write(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
