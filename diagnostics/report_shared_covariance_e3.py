"""Report for the full-E3 Shared Covariance-aware (SC) validation: summary.json, summary.md, figures.

Interpretation text is not generated: runs_shared_covariance_e3/interpretation.md is appended verbatim if present.
"""
from __future__ import annotations

import json, os, sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import shared_covariance_e3 as D                            # noqa: E402
from moe_calib import cross_seed_grouping as CS             # noqa: E402
from moe_calib import shared_covariance_selection as SCS    # noqa: E402

OUT = os.path.join(ROOT, D.OUT)
S, L4, ST = D.SEEDS, D.LAYERS, D.STRATS
COL = dict(FR="#2a78d6", TR="#eb6834", TC="#1baf7a", SC="#eda100")     # reference categorical slots 1-4, fixed order
NAMES = dict(FR="Full Random", TR="Typical Random", TC="Typical Coverage-LP", SC="Shared Covariance-aware")


def jl(*a):
    return json.load(open(os.path.join(OUT, *a)))


def per_layer(per):
    return {L: float(np.mean([r["mean"] for r in per if r["layer"] == L])) for L in L4}


def emap(per):
    return {(r["layer"], r["expert"]): r["mean"] for r in per}


def pct(v, d=2):
    return f"{100 * v:+.{d}f} %"


def main():
    pr = D.protocol()
    A, reuse = D.stage_a_results()
    have = {k for k in A}
    complete_A = all((c, s) in have for c in ST for s in S)
    bf16 = json.load(open(os.path.join(ROOT, "runs_partial_ppl", "bf16_baseline.json")))["ppl"]
    sel = {s: jl("selections", f"seed{s}.json") for s in S}
    summ = dict(protocol_sha256=pr["protocol_sha256"], baselines_reused=reuse, stage_a_complete=complete_A, bf16_ppl=bf16)
    val = {k: jl("results", f"A_{k}.json")["reproduction"] for k in ("FR_s0", "TR_s1", "TC_s2")
           if os.path.exists(os.path.join(OUT, "results", f"A_{k}.json"))}
    summ["baseline_reproduction"] = val
    if not complete_A:
        json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
        print("[report] Stage A incomplete:", sorted(set((c, s) for c in ST for s in S) - have)); return

    # ---------------- Stage A tables
    tab = {c: {s: dict(test=A[(c, s)]["recon_test"], conf=A[(c, s)]["recon_conf"], ppl=A[(c, s)]["ppl"],
                       layers=per_layer(A[(c, s)]["per_expert_test"]),
                       q=dict(median=float(np.median([r["mean"] for r in A[(c, s)]["per_expert_test"]])),
                              p90=float(np.quantile([r["mean"] for r in A[(c, s)]["per_expert_test"]], .9)),
                              p95=float(np.quantile([r["mean"] for r in A[(c, s)]["per_expert_test"]], .95))))
               for s in S} for c in ST}
    summ["stage_a"] = {c: {str(s): v for s, v in d.items()} for c, d in tab.items()}
    comps, labels = {}, {}
    for ref in ("FR", "TR", "TC"):
        for x in [c for c in ST if c != ref]:
            r = [SCS.relative_improvement(tab[x][s]["test"], tab[ref][s]["test"]) for s in S]
            rc = [SCS.relative_improvement(tab[x][s]["conf"], tab[ref][s]["conf"]) for s in S]
            rl = {L: float(np.mean([SCS.relative_improvement(tab[x][s]["layers"][L], tab[ref][s]["layers"][L])
                                     for s in S])) for L in L4}
            ex = []
            for s in S:
                a, b = emap(A[(x, s)]["per_expert_test"]), emap(A[(ref, s)]["per_expert_test"])
                ex.append(int(sum(a[k] < b[k] for k in a)))
            a3 = {k: np.mean([emap(A[(x, s)]["per_expert_test"])[k] for s in S]) for k in emap(A[(x, 0)]["per_expert_test"])}
            b3 = {k: np.mean([emap(A[(ref, s)]["per_expert_test"])[k] for s in S]) for k in a3}
            dppl = [tab[x][s]["ppl"] - tab[ref][s]["ppl"] for s in S]
            comps[f"{x}_vs_{ref}"] = dict(r_test=r, mean_test=float(np.mean(r)), sd_test=float(np.std(r, ddof=1)),
                                          r_conf=rc, mean_conf=float(np.mean(rc)), r_layer={str(L): v for L, v in rl.items()},
                                          experts_improved_per_seed=ex,
                                          experts_improved_seed_avg=int(sum(a3[k] < b3[k] for k in a3)),
                                          dppl=dppl, mean_dppl=float(np.mean(dppl)))
            if x == "SC":
                labels[f"SC_vs_{ref}_stageA"] = SCS.decide(r, rl, rc)[0]
    summ["comparisons"] = comps

    # ---------------- overlap
    summ["overlap"] = {str(s): sel[s]["overlap_with_baselines"] for s in S}
    summ["optimizer"] = {str(s): dict({k: v for k, v in sel[s]["optimizer"].items() if k != "trace"},
                                      n_trace=len(sel[s]["optimizer"]["trace"]),
                                      n_post_warm_from_typical_half=sel[s]["n_post_warm_from_typical_half"],
                                      cost=sel[s]["cost"]) for s in S}

    # ---------------- diagnostics
    diag = None
    if os.path.exists(os.path.join(OUT, "diagnostics", "sets.json")):
        diag = jl("diagnostics", "sets.json")["sets"]
        rows = {}
        for c in ST:
            for s in S:
                d = diag[f"{c}_s{s}"]
                rr = d["routing"]
                rows[(c, s)] = dict(sketch=d["sketch"]["objective"], sketch_gu=d["sketch"]["gu"],
                                    sketch_down=d["sketch"]["down"], full=d["full_summary"]["objective"],
                                    full_gu=d["full_summary"]["gu"], full_down=d["full_summary"]["down"],
                                    d_mean=rr["d_mean"], d_post=rr["d_post_warm_mean"], js=rr["js_set_vs_original_pool"],
                                    l1=rr["set_routing_l1_mean"], tok_median=rr["tokens_per_expert"]["median"],
                                    tok_p10=rr["tokens_per_expert"]["p10"], gini=rr["gini_expert_tokens"],
                                    eff_src=rr["median_effective_sources_per_expert"],
                                    distinct=rr["distinct_sources_per_expert_mean"],
                                    test=tab[c][s]["test"], conf=tab[c][s]["conf"], ppl=tab[c][s]["ppl"])
        summ["diagnostics_mean"] = {c: {k: float(np.mean([rows[(c, s)][k] for s in S])) for k in rows[(c, 0)]} for c in ST}
        keys = ["sketch", "full", "full_gu", "full_down", "d_mean", "js", "l1", "tok_median", "gini", "eff_src"]
        # within-seed paired deltas vs FR (TR, TC, SC x 3 seeds = 9) and across all 12 sets (descriptive)
        assoc = {}
        for k in keys:
            dx = [rows[(c, s)][k] - rows[("FR", s)][k] for s in S for c in ("TR", "TC", "SC")]
            dy = [rows[(c, s)]["test"] - rows[("FR", s)]["test"] for s in S for c in ("TR", "TC", "SC")]
            xs = [rows[(c, s)][k] for c in ST for s in S]
            ys = [rows[(c, s)]["test"] for c in ST for s in S]
            assoc[k] = dict(paired_vs_FR_spearman=float(spearmanr(dx, dy).correlation),
                            all12_spearman=float(spearmanr(xs, ys).correlation))
        summ["association_with_test_error"] = assoc
        # per-expert: SC - FR delta sketch mismatch vs delta error, 240 experts per seed (descriptive)
        pe = []
        for s in S:
            ds = np.array(diag[f"SC_s{s}"]["sketch"]["per_expert"]) - np.array(diag[f"FR_s{s}"]["sketch"]["per_expert"])
            dfull = np.concatenate([np.array(diag[f"SC_s{s}"]["full"][str(L)]["per_expert"]) -
                                    np.array(diag[f"FR_s{s}"]["full"][str(L)]["per_expert"]) for L in L4])
            a, b = emap(A[("SC", s)]["per_expert_test"]), emap(A[("FR", s)]["per_expert_test"])
            de = np.array([(a[(L, e)] - b[(L, e)]) / b[(L, e)] for L in L4 for e in range(60)])
            pe.append(dict(seed=s, frac_experts_lower_sketch=float((ds < 0).mean()),
                           frac_experts_lower_full=float((dfull < 0).mean()),
                           spearman_dsketch_derr=float(spearmanr(ds, de).correlation),
                           spearman_dfull_derr=float(spearmanr(dfull, de).correlation)))
        summ["per_expert_association_SC_vs_FR"] = pe
        summ["_rows"] = {f"{c}_s{s}": v for (c, s), v in rows.items()}

    # ---------------- Stage B
    gb = None
    gdir = os.path.join(OUT, "grouping_robustness", "results")
    if os.path.isdir(gdir):
        R = {}
        for c in ST:
            for s in S:
                for g in D.GSEEDS:
                    f = os.path.join(gdir, f"{c}_s{s}_{D.gname(g)}.json")
                    if os.path.exists(f):
                        R[(c, s, g)] = json.load(open(f))
        gb = dict(n_configs=len(R), complete=len(R) == 60, all_install_ok=all(r["install_ok"] for r in R.values()))
        if gb["complete"]:
            for ref in ("FR", "TR", "TC"):
                adv = {s: {g: (R[(ref, s, g)]["recon_test"] - R[("SC", s, g)]["recon_test"]) / R[(ref, s, g)]["recon_test"]
                           for g in D.GSEEDS} for s in S}
                dppl = {s: {g: R[("SC", s, g)]["ppl"] - R[(ref, s, g)]["ppl"] for g in D.GSEEDS} for s in S}
                vr, vp = CS.variance_summary(adv), CS.variance_summary(dppl)
                br, bp = CS.hierarchical_bootstrap(adv), CS.hierarchical_bootstrap(dppl)
                gb[f"SC_vs_{ref}"] = dict(
                    recon_adv=dict(values={str(s): {str(g): v for g, v in d.items()} for s, d in adv.items()},
                                   seed_means=[vr["mean_delta"][s] for s in S], sd_grouping=[vr["sd_grouping"][s] for s in S],
                                   pooled_within_sd=vr["pooled_within_sd"], sd_between_seed_means=vr["sd_between_seed_means"],
                                   overall=vr["overall_mean_delta"], n_positive=int(sum(v > 0 for d in adv.values() for v in d.values())),
                                   boot=br),
                    ppl_delta=dict(values={str(s): {str(g): v for g, v in d.items()} for s, d in dppl.items()},
                                   seed_means=[vp["mean_delta"][s] for s in S], sd_grouping=[vp["sd_grouping"][s] for s in S],
                                   pooled_within_sd=vp["pooled_within_sd"], sd_between_seed_means=vp["sd_between_seed_means"],
                                   overall=vp["overall_mean_delta"], n_sc_better=int(sum(v < 0 for d in dppl.values() for v in d.values())),
                                   abs_mean_over_W=abs(vp["overall_mean_delta"]) / vp["pooled_within_sd"], boot=bp))
                lab = SCS.decide(comps[f"SC_vs_{ref}"]["r_test"],
                                 {int(k): v for k, v in comps[f"SC_vs_{ref}"]["r_layer"].items()},
                                 comps[f"SC_vs_{ref}"]["r_conf"],
                                 dict(seed_means=gb[f"SC_vs_{ref}"]["recon_adv"]["seed_means"], ci95=br["ci95"]))
                labels[f"SC_vs_{ref}_final"] = lab[0]
            gb["strategy_means"] = {c: dict(recon=float(np.mean([R[(c, s, g)]["recon_test"] for s in S for g in D.GSEEDS])),
                                            ppl=float(np.mean([R[(c, s, g)]["ppl"] for s in S for g in D.GSEEDS])))
                                    for c in ST}
            ctl = os.path.join(gdir, f"SC_s0_{D.gname(D.GSEEDS[0])}_CTRL.json")
            gb["determinism_control"] = json.load(open(ctl))["determinism"] if os.path.exists(ctl) else None
            gb["_R"] = {f"{c}_s{s}_{g}": dict(recon=r["recon_test"], ppl=r["ppl"]) for (c, s, g), r in R.items()}
    summ["stage_b"] = gb
    summ["labels"] = labels
    summ["decision"] = labels.get("SC_vs_FR_final", labels.get("SC_vs_FR_stageA"))
    # ---------------- cost
    prof = {str(L): jl("profiling", f"sketch_L{L}.json") for L in L4}
    pre = jl("preflight.json")
    summ["cost"] = dict(
        pool_cache_s=pre["estimate"]["existing_costs"]["pool_cache_s"], routing_profile_s=357.0,
        sketch_s={L: v["seconds"] for L, v in prof.items()}, sketch_total_s=float(sum(v["seconds"] for v in prof.values())),
        sketch_peak_gpu_gb=max(v["peak_gpu_gb"] for v in prof.values()),
        sketch_peak_rss_gb=max(v["peak_rss_gb"] for v in prof.values()),
        sketch_disk_gb=float(sum(v["sketch_file_gb"] for v in prof.values())),
        hfull_disk_gb=float(sum(v["hfull_file_gb"] for v in prof.values())),
        selection_s={str(s): sel[s]["cost"]["select_s"] for s in S},
        selection_load_s={str(s): sel[s]["cost"]["load_s"] for s in S},
        selection_peak_gpu_gb=max(sel[s]["cost"]["peak_gpu_gb"] for s in S),
        sc_stage_a={str(s): {k: A[("SC", s)]["cost"].get(k) for k in ("hessian_s", "gptq_s", "quantize_wall_s", "eval_s",
                                                                    "conf_wall_s", "peak_gpu_gb", "peak_cpu_gb")} for s in S},
        forward_s_per_seq=pre["estimate"]["existing_costs"]["forward_s_per_seq"])
    c = summ["cost"]
    extra = c["sketch_total_s"] + float(np.mean(list(c["selection_s"].values())))
    c["sc_extra_over_FR_s"] = extra
    c["sc_extra_full_forward_equiv_seqs"] = extra / c["forward_s_per_seq"]
    c["note"] = ("FR needs no pool profiling; TR / TC need the 2,000-sequence routing profile (357 s); SC needs the "
                 "4-layer covariance sketch pass + selection. Equivalent extra random sequences = extra seconds / "
                 "measured full-model forward cost per sequence (E3 cost model).")
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    figures(summ, tab, A)
    write_md(summ, tab, pr)
    print("[report] decision:", summ["decision"], labels)


# ===================================================================== figures
def figures(summ, tab, A):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": ink})

    def clean(ax):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=grid, linewidth=0.6, zorder=0)
        ax.axhline(0, color=ink2, linewidth=0.8, zorder=1)
    # fig1: paired improvement vs FR per seed (test, confirmation)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, key, title in ((axs[0], "r_test", "test split (primary)"), (axs[1], "r_conf", "confirmation split")):
        for i, c in enumerate(("TR", "TC", "SC")):
            v = [100 * x for x in summ["comparisons"][f"{c}_vs_FR"][key]]
            ax.bar(i, np.mean(v), width=0.55, color=COL[c], zorder=2)
            ax.scatter([i] * 3, v, s=16, facecolor="white", edgecolor=ink, linewidth=0.8, zorder=3)
        ax.axhline(100 * SCS.THRESHOLD, color=ink2, linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xticks(range(3), ["TR", "TC", "SC"])
        ax.set_title(title)
        clean(ax)
    axs[0].set_ylabel("reconstruction improvement vs FR (%)\nbar = 3-seed mean, dots = seeds; dashed = 0.5 %")
    fig.suptitle("Paired held-out reconstruction improvement over Full Random (ORIGINAL extraction, 240 experts)",
                 color=ink, fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "fig1_improvement_vs_FR.png"), dpi=150); plt.close(fig)
    # fig2: per-layer SC improvement vs each baseline
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.25
    for j, ref in enumerate(("FR", "TR", "TC")):
        v = [100 * summ["comparisons"][f"SC_vs_{ref}"]["r_layer"][str(L)] for L in L4]
        ax.bar(np.arange(4) + (j - 1) * w, v, width=w - 0.03, color=COL[ref], label=f"vs {ref}", zorder=2)
    ax.set_xticks(range(4), [f"L{L}" for L in L4])
    ax.set_ylabel("SC improvement (%), 3-seed mean")
    ax.legend(frameon=False, ncol=3, loc="best")
    ax.set_title("SC reconstruction improvement by layer (test split)")
    clean(ax)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "fig2_per_layer.png"), dpi=150); plt.close(fig)
    # fig3: covariance mismatch vs test error (12 sets)
    if "_rows" in summ:
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
        for ax, k, lab in ((axs[0], "sketch", "64-d sketch objective J (selection surrogate)"),
                           (axs[1], "full", "full-dim relative Frobenius mismatch"),
                           (axs[2], "d_mean", "mean routing atypicality d(s)")):
            for c in ST:
                xs = [summ["_rows"][f"{c}_s{s}"][k] for s in S]
                ys = [1e3 * summ["_rows"][f"{c}_s{s}"]["test"] for s in S]
                ax.scatter(xs, ys, s=44, color=COL[c], edgecolor="white", linewidth=1.5, label=c, zorder=3)
            ax.set_xlabel(lab)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.grid(color=grid, linewidth=0.6, zorder=0)
        axs[0].set_ylabel("test reconstruction error (×1e-3)")
        axs[2].legend(frameon=False, title="strategy")
        fig.suptitle("Selected-set diagnostics vs held-out reconstruction (each point = strategy × seed)", color=ink,
                     fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "fig3_mismatch_vs_error.png"), dpi=150); plt.close(fig)
    # fig4: Stage B paired SC - FR per (seed, grouping)
    gb = summ.get("stage_b")
    if gb and gb.get("complete"):
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, key, lab, scale in ((axs[0], "recon_adv", "SC reconstruction advantage over reference (%)", 100),
                                    (axs[1], "ppl_delta", "PPL(SC) − PPL(reference)", 1)):
            for j, ref in enumerate(("FR", "TR", "TC")):
                d = gb[f"SC_vs_{ref}"][key]
                for i, s in enumerate(S):
                    v = [scale * x for x in d["values"][str(s)].values()]
                    xpos = i * 4 + j
                    ax.scatter([xpos] * len(v), v, s=14, color=COL[ref], alpha=0.8, zorder=3,
                               label=f"vs {ref}" if i == 0 else None)
                    ax.plot([xpos - 0.3, xpos + 0.3], [np.mean(v)] * 2, color=ink, linewidth=1.5, zorder=4)
            ax.set_xticks([i * 4 + 1 for i in range(3)], [f"seed {s}" for s in S])
            ax.set_ylabel(lab)
            clean(ax)
        axs[0].legend(frameon=False, loc="best")
        fig.suptitle("Stage B: paired comparisons per (allocation seed, grouping); dots = 5 groupings, bar = mean",
                     color=ink, fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "fig4_grouping_robustness.png"), dpi=150); plt.close(fig)


# ===================================================================== markdown
def write_md(summ, tab, pr):
    Lx = []
    w = Lx.append
    w("# Full E3 validation of Shared Covariance-aware calibration (SC) — summary\n")
    w(f"Protocol `{pr['protocol_sha256'][:16]}` (frozen before sketches); selections frozen in `freeze.json` before any SC "
      f"reconstruction. Qwen1.5-MoE-A2.7B, layers 1/8/16/23 (240 routed experts), B = 128 shared sequences, matched "
      f"32-sequence warm start, GPTQ 4-bit / g128 / act-order / percdamp 1.0, seeds 0/1/2.\n")
    w(f"**Decision (frozen rule, SC vs FR): {summ['decision']}**. Other labels: "
      + ", ".join(f"{k} = {v}" for k, v in summ["labels"].items()) + "\n")
    w(f"Baselines: stored FR/TR/TC results reused = **{summ['baselines_reused']}** (FR s0, TR s1, TC s2 re-run "
      "through this pipeline: " + "; ".join(f"{k}: {v['all']}" for k, v in summ["baseline_reproduction"].items()) + ").\n")
    # Stage A main table
    w("## 1. Stage A — ORIGINAL extraction (3-seed means)\n")
    w("| strategy | test error ↓ | confirmation error ↓ | test improvement vs FR | confirmation improvement vs FR | "
      "PPL ↓ | Δ PPL vs BF16 | Δ PPL vs FR |")
    w("|---|---|---|---|---|---|---|---|")
    for c in ST:
        te = np.mean([tab[c][s]["test"] for s in S]); cf = np.mean([tab[c][s]["conf"] for s in S])
        pp = np.mean([tab[c][s]["ppl"] for s in S])
        if c == "FR":
            ri = rc = dp = "—"
        else:
            cp = summ["comparisons"][f"{c}_vs_FR"]
            ri, rc, dp = pct(cp["mean_test"]), pct(cp["mean_conf"]), f"{cp['mean_dppl']:+.4f}"
        w(f"| {c} ({NAMES[c]}) | {te:.6f} | {cf:.6f} | {ri} | {rc} | {pp:.4f} | {pp - summ['bf16_ppl']:+.4f} | {dp} |")
    w(f"\nBF16 PPL {summ['bf16_ppl']:.4f}.\n")
    w("Per seed (test / confirmation / PPL):\n")
    w("| strategy | " + " | ".join(f"seed {s}" for s in S) + " |")
    w("|---|" + "---|" * len(S))
    for c in ST:
        w(f"| {c} | " + " | ".join(f"{tab[c][s]['test']:.6f} / {tab[c][s]['conf']:.6f} / {tab[c][s]['ppl']:.4f}"
                                   for s in S) + " |")
    w("")
    # comparisons
    w("## 2. Paired comparisons (positive = first strategy better; paired by allocation seed)\n")
    w("| comparison | test r per seed | mean ± sd | confirmation r per seed | conf mean | experts improved per seed (of 240) "
      "| experts improved, 3-seed avg | Δ PPL per seed | layers improved |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k in ("SC_vs_FR", "SC_vs_TR", "SC_vs_TC", "TR_vs_FR", "TC_vs_FR", "TC_vs_TR"):
        cp = summ["comparisons"][k]
        w(f"| {k.replace('_vs_', ' vs ')} | " + " / ".join(pct(v) for v in cp["r_test"]) +
          f" | {pct(cp['mean_test'])} ± {100 * cp['sd_test']:.2f} | " + " / ".join(pct(v) for v in cp["r_conf"]) +
          f" | {pct(cp['mean_conf'])} | " + " / ".join(map(str, cp["experts_improved_per_seed"])) +
          f" | {cp['experts_improved_seed_avg']} | " + " / ".join(f"{v:+.4f}" for v in cp["dppl"]) +
          f" | {sum(v > 0 for v in cp['r_layer'].values())}/4 |")
    w("\nPer-layer improvement (3-seed mean):\n")
    w("| comparison | " + " | ".join(f"L{L}" for L in L4) + " |")
    w("|---|" + "---|" * 4)
    for k in ("SC_vs_FR", "SC_vs_TR", "SC_vs_TC"):
        w(f"| {k.replace('_vs_', ' vs ')} | " + " | ".join(pct(summ["comparisons"][k]["r_layer"][str(L)]) for L in L4) + " |")
    w("\nExpert error quantiles (3-seed mean over the 240-expert distribution, test):\n")
    w("| strategy | median | p90 | p95 |")
    w("|---|---|---|---|")
    for c in ST:
        w(f"| {c} | " + " | ".join(f"{np.mean([tab[c][s]['q'][q] for s in S]):.6f}" for q in ("median", "p90", "p95")) + " |")
    w("\n![](figures/fig1_improvement_vs_FR.png)\n\n![](figures/fig2_per_layer.png)\n")
    # overlap + optimizer
    w("## 3. SC selection and overlap\n")
    w("| seed | J(warm) | J(greedy) | J(final) | accepted swaps | passes | select s | SC ∩ FR | SC ∩ TR | SC ∩ TC | "
      "post-warm picks in typical half |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in S:
        o, ov = summ["optimizer"][str(s)], summ["overlap"][str(s)]
        w(f"| {s} | {o['warm_objective']:.4f} | {o['greedy_objective']:.4f} | {o['final_objective']:.4f} | {o['accepted']} | "
          f"{o['passes']} | {o['cost']['select_s']:.0f} | {ov['FR']['intersection']} | {ov['TR']['intersection']} | "
          f"{ov['TC']['intersection']} | {o['n_post_warm_from_typical_half']}/96 |")
    w("\n(All overlaps include the 32 shared warm-start sequences.)\n")
    # diagnostics
    if "diagnostics_mean" in summ:
        dm = summ["diagnostics_mean"]
        w("## 4. Mechanism diagnostics (3-seed means; descriptive)\n")
        w("| strategy | sketch J | sketch gu / down | full-dim mismatch | full gu / down | mean d(s) | post-warm d(s) | "
          "set JS | set L1 | median tokens/expert | p10 tokens | Gini | eff. sources (median) | distinct sources |")
        w("|---|" + "---|" * 13)
        for c in ST:
            m = dm[c]
            w(f"| {c} | {m['sketch']:.4f} | {m['sketch_gu']:.4f} / {m['sketch_down']:.4f} | {m['full']:.4f} | "
              f"{m['full_gu']:.4f} / {m['full_down']:.4f} | {m['d_mean']:.4f} | {m['d_post']:.4f} | {m['js']:.5f} | "
              f"{m['l1']:.4f} | {m['tok_median']:.0f} | {m['tok_p10']:.0f} | {m['gini']:.4f} | {m['eff_src']:.1f} | "
              f"{m['distinct']:.1f} |")
        w("\nSpearman association with test error (descriptive; 9 within-seed deltas vs FR / all 12 sets):\n")
        w("| diagnostic | paired Δ vs FR (n = 9) | all 12 sets |")
        w("|---|---|---|")
        for k, v in summ["association_with_test_error"].items():
            w(f"| {k} | {v['paired_vs_FR_spearman']:+.2f} | {v['all12_spearman']:+.2f} |")
        w("\nPer-expert SC − FR (240 experts per seed): fraction of experts with lower mismatch and Spearman(Δ mismatch, "
          "Δ relative error):\n")
        w("| seed | lower sketch mismatch | lower full-dim mismatch | ρ(Δsketch, Δerr) | ρ(Δfull, Δerr) |")
        w("|---|---|---|---|---|")
        for r in summ["per_expert_association_SC_vs_FR"]:
            w(f"| {r['seed']} | {100 * r['frac_experts_lower_sketch']:.0f} % | {100 * r['frac_experts_lower_full']:.0f} % | "
              f"{r['spearman_dsketch_derr']:+.3f} | {r['spearman_dfull_derr']:+.3f} |")
        w("\n![](figures/fig3_mismatch_vs_error.png)\n")
    # Stage B
    gb = summ.get("stage_b")
    w("## 5. Stage B — paired calibration-grouping robustness\n")
    if not gb or not gb.get("complete"):
        w(f"Not complete ({0 if not gb else gb['n_configs']}/60 configurations).\n")
    else:
        w("Per allocation seed: union of the four strategies' ids, 5 groupings (20261111–15, batches of 8), all four "
          "strategies quantized from the same grouping cache. Advantage = (E_ref − E_SC)/E_ref; Δ PPL = PPL(SC) − PPL(ref). "
          "Nested analysis (seed > grouping); hierarchical bootstrap over seeds then groupings.\n")
        w("| comparison | recon advantage seed means | pooled grouping SD | overall | 95 % hier. CI | (s,g) SC better | "
          "Δ PPL seed means | PPL pooled grouping SD W | overall Δ PPL | 95 % hier. CI | (s,g) SC better PPL | |M|/W |")
        w("|---|" + "---|" * 11)
        for ref in ("FR", "TR", "TC"):
            r, q = gb[f"SC_vs_{ref}"]["recon_adv"], gb[f"SC_vs_{ref}"]["ppl_delta"]
            w(f"| SC vs {ref} | " + " / ".join(pct(v, 3) for v in r["seed_means"]) + f" | {100 * r['pooled_within_sd']:.3f} % | "
              f"{pct(r['overall'], 3)} | [{pct(r['boot']['ci95'][0], 3)}, {pct(r['boot']['ci95'][1], 3)}] | {r['n_positive']}/15 | "
              + " / ".join(f"{v:+.4f}" for v in q["seed_means"]) + f" | {q['pooled_within_sd']:.4f} | {q['overall']:+.4f} | "
              f"[{q['boot']['ci95'][0]:+.4f}, {q['boot']['ci95'][1]:+.4f}] | {q['n_sc_better']}/15 | {q['abs_mean_over_W']:.2f} |")
        sm = gb["strategy_means"]
        w("\nGrouping-averaged means (15 configurations each): " + "; ".join(
            f"{c} recon {sm[c]['recon']:.6f}, PPL {sm[c]['ppl']:.4f}" for c in ST) + ".")
        dc = gb.get("determinism_control")
        w(f"\nFresh-process determinism control (SC s0 G20261111): {dc}\n")
        w("![](figures/fig4_grouping_robustness.png)\n")
    # cost
    c = summ["cost"]
    w("## 6. Compute (equal sequence budget ≠ equal profiling compute)\n")
    w("| item | FR | TR / TC | SC |")
    w("|---|---|---|---|")
    w(f"| initial full-model activation cache (reused, all) | {c['pool_cache_s']:.0f} s | same | same |")
    w(f"| candidate-pool routing profile | — | {c['routing_profile_s']:.0f} s | — (counts come from the sketch pass) |")
    w(f"| covariance sketch pass (4 layers × 2,000 seqs, 240 experts) | — | — | {c['sketch_total_s']:.0f} s; peak GPU "
      f"{c['sketch_peak_gpu_gb']:.1f} GB, RSS {c['sketch_peak_rss_gb']:.1f} GB |")
    w(f"| selection | ~0 | LP (TC) | " + " / ".join(f"{v:.0f}" for v in c["selection_s"].values()) +
      f" s (+ {np.mean(list(c['selection_load_s'].values())):.0f} s load); peak GPU {c['selection_peak_gpu_gb']:.1f} GB |")
    w(f"| disk | — | — | sketches {c['sketch_disk_gb']:.1f} GB (+ {c['hfull_disk_gb']:.1f} GB full-dim diagnostic reference) |")
    sa = c["sc_stage_a"]["0"]
    w(f"| calibration Hessian replay / GPTQ / PPL (per config) | same | same | {sa['hessian_s']:.0f} s / {sa['gptq_s']:.0f} s / "
      f"{sa['eval_s']:.0f} s; peak GPU {sa['peak_gpu_gb']:.1f} GB, peak CPU {sa['peak_cpu_gb']:.0f} GB |")
    w(f"\nSC extra cost over FR ≈ {c['sc_extra_over_FR_s']:.0f} s ≈ **{c['sc_extra_full_forward_equiv_seqs']:.0f} extra "
      f"random sequences** at {c['forward_s_per_seq']:.3f} s per full-model sequence forward (E3 cost model).\n")
    ip = os.path.join(OUT, "interpretation.md")
    if os.path.exists(ip):
        w(open(ip).read())
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(Lx) + "\n")


if __name__ == "__main__":
    main()
