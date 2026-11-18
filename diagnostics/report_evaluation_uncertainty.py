"""Figures for runs_evaluation_uncertainty (reads the JSON outputs + logs/parts/*.npz; no recomputation)."""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation_uncertainty as E   # noqa: E402

COL = {"WT2": "#2a78d6", "C4HO": "#eb6834"}
SEEDC = ["#2a78d6", "#1baf7a", "#e87ba4"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LBL = {"WT2": "WikiText-2", "C4HO": "C4-val held-out"}
NAME = {"moe": "MoE FC−FR (15)", "dense4": "Dense 4-bit (5)", "dense3": "Dense 3-bit (5)"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
FIG = E.p("figures")


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=160, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    unc = E.jload(E.p("variance_component_uncertainty.json"))["sets"]
    spl = E.jload(E.p("split_half_reliability.json"))["sets"]
    cross = E.jload(E.p("cross_dataset_reliability.json"))["sets"]
    mci = E.jload(E.p("moe_window_ci_analysis.json"))
    parts = {n: np.load(E.p("logs", "parts", f"{n}.npz")) for n in unc}
    PS = dict(WT2="article", C4HO="window")
    main3 = ["moe", "dense4", "dense3"]

    # ---- Fig 1: observed vs sampling vs corrected
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    for ax, n in zip(axes, main3):
        for k, ds in enumerate(E.DS):
            r = unc[n][ds][PS[ds]]
            x0 = k * 4
            s = 1e7
            ax.plot([x0], [r["V_obs"] * s], "o", color=COL[ds], ms=7, mfc="white", mew=2)
            ax.plot([x0 + 1], [r["V_samp"] * s], "s", color=COL[ds], ms=7, mfc="white", mew=2)
            lo, hi = r["V_corr_pivot_ci"]
            ax.errorbar([x0 + 2], [r["V_corr"] * s], yerr=[[(r["V_corr"] - lo) * s], [(hi - r["V_corr"]) * s]],
                        fmt="D", color=COL[ds], ms=7, capsize=3, lw=2)
            for sch in ("block4", "block8", "block16"):
                if sch in unc[n][ds]:
                    ax.plot([x0 + 2.35], [unc[n][ds][sch]["V_corr"] * s], "_", color=COL[ds], ms=8, mew=1.5)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks([0, 1, 2, 4, 5, 6])
        ax.set_xticklabels(["obs", "samp", "corr", "obs", "samp", "corr"])
        ax.set_title(NAME[n], color=INK)
        ax.text(1, -0.14, "WikiText-2", transform=ax.get_xaxis_transform(), ha="center", va="top", color=COL["WT2"])
        ax.text(5, -0.14, "C4-val", transform=ax.get_xaxis_transform(), ha="center", va="top", color=COL["C4HO"])
    axes[0].set_ylabel("between-variant variance of mean NLL (×10⁻⁷ nats²)")
    fig.suptitle("Fig. 1  Observed vs evaluation-sampling vs corrected variance (pivot 95% CI; dashes = WT2 block 4/8/16; "
                 "negative estimates not clipped)", color=INK, fontsize=9.5, y=1.06)
    save(fig, "fig1_variance_components")

    # ---- Fig 2: split-half distributions
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4))
    bins = np.linspace(-1, 1, 41)
    for j, n in enumerate(main3):
        for ds in E.DS:
            rr = parts[n][f"{ds}_{PS[ds]}_pearson"]
            axes[0, j].hist(rr, bins=bins, color=COL[ds], alpha=0.55, label=LBL[ds])
            axes[0, j].axvline(np.median(rr), color=COL[ds], lw=2)
            sk = "spearman_within_seed_ranks" if n == "moe" else "spearman"
            s_ = spl[n][ds][PS[ds]][sk]
            axes[1, j].errorbar([s_["median"]], [0.5 if ds == "WT2" else -0.5], xerr=[[s_["median"] - s_["p05"]], [s_["p95"] - s_["median"]]],
                                fmt="o", color=COL[ds], capsize=4, lw=2, ms=7)
            axes[1, j].text(s_["p95"] + 0.04, 0.5 if ds == "WT2" else -0.5, f"median {s_['median']:.2f}", va="center",
                            color=INK2, fontsize=8)
        axes[0, j].set_title(NAME[n], color=INK)
        axes[0, j].set_xlim(-1, 1); axes[1, j].set_xlim(-1, 1.45); axes[1, j].set_ylim(-1.2, 1.2)
        axes[1, j].set_yticks([0.5, -0.5]); axes[1, j].set_yticklabels(["WT2", "C4HO"])
        axes[1, j].axvline(0, color=INK2, lw=1); axes[0, j].axvline(0, color=INK2, lw=1)
        axes[1, j].set_xlabel("Spearman (median, 5–95% over partitions)")
    axes[0, 0].set_ylabel("partitions (of 1000)")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[0, 1].set_xlabel("split-half Pearson of centred effects")
    fig.suptitle("Fig. 2  Within-dataset split-half reliability (WT2: article split; C4: document split). Partitions reuse "
                 "the same windows — a sensitivity distribution, not independent replicates", color=INK, fontsize=9.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_split_half_distributions")

    # ---- Fig 3: within vs cross
    sets7 = main3 + ["dose_natural_k1", "dose_natural_k4", "dose_natural_k16", "dose_natural_k64"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, n in enumerate(sets7):
        for k, ds in enumerate(E.DS):
            sp = spl[n][ds][PS[ds]]["pearson"]
            sb = lambda r: 2 * r / (1 + r) if r > -1 else np.nan
            m, lo, hi = sb(sp["median"]), sb(sp["p05"]), sb(sp["p95"])
            xx = i - 0.22 + k * 0.22
            if not np.isfinite(lo) or lo < -1.25:
                lo = -1.25
            if m < -1.25:
                ax.annotate(f"{m:.1f}", (xx, -1.25), xytext=(xx + 0.05, -1.12), fontsize=7, color=COL[ds],
                            arrowprops=dict(arrowstyle="->", color=COL[ds], lw=1))
                m = np.nan
            ax.errorbar([i - 0.22 + k * 0.22], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=COL[ds], capsize=3, ms=6,
                        label=f"within {LBL[ds]} (split-half, Spearman–Brown to full length)" if i == 0 else None)
        c = cross[n]
        ax.errorbar([i + 0.22], [c["pearson_full_WT2_vs_C4"]], yerr=[[c["pearson_full_WT2_vs_C4"] - c["pearson_bootstrap_ci"][0]],
                    [c["pearson_bootstrap_ci"][1] - c["pearson_full_WT2_vs_C4"]]], fmt="D", color=INK, capsize=3, ms=6,
                    label="cross-dataset full WT2 vs full C4 (bootstrap 95%)" if i == 0 else None)
        if c["expected_cross_r_if_identical_true_effects"]:
            ax.plot([i + 0.22], [c["expected_cross_r_if_identical_true_effects"]], "*", color="#1baf7a", ms=11,
                    label="expected cross r if true effects identical (√λ₁λ₂)" if i == 0 else None)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(sets7)))
    ax.set_xticklabels(["MoE", "dense 4b", "dense 3b", "dose k=1", "k=4", "k=16", "k=64"])
    ax.set_ylim(-1.3, 1.15)
    ax.set_ylabel("correlation of centred effects")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("Fig. 3  Within-dataset reliability (stepped up to n = 128) vs cross-dataset agreement; bars: 5–95% of "
                 "partitions / bootstrap 95%. Arrows: stepped-up value < −1 (V_obs < V_samp), axis clipped at −1.25", color=INK, fontsize=9)
    save(fig, "fig3_within_vs_cross")

    # ---- Fig 4: null
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
    for j, n in enumerate(main3):
        for i, ds in enumerate(E.DS):
            ax = axes[i, j]
            med = parts[n][f"{ds}_{PS[ds]}_null_med"]
            ax.hist(med, bins=np.linspace(-1, 1, 41), color=GRID, edgecolor=INK2, lw=0.4, label="null (common effect)")
            obs = spl[n][ds][PS[ds]]["pearson"]["median"]
            ax.axvline(obs, color=COL[ds], lw=2.5, label="observed")
            ax.text(0.02, 0.92, f"{LBL[ds]}\nobs {obs:.2f}, p = {spl[n][ds][PS[ds]]['null']['p_value']:.3f}",
                    transform=ax.transAxes, va="top", fontsize=8, color=INK)
            if i == 0:
                ax.set_title(NAME[n], color=INK)
            ax.set_xlim(-1, 1)
    axes[1, 1].set_xlabel("median split-half Pearson (null: 500 datasets × 40 partitions)")
    axes[0, 0].legend(fontsize=7.5, loc="center left")
    fig.suptitle("Fig. 4  Observed split-half reliability vs a common-effect null preserving the joint window covariance",
                 color=INK, fontsize=9.5, y=1.01)
    fig.tight_layout()
    save(fig, "fig4_sampling_null")

    # ---- Fig 5: MoE scatter
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for c, v in mci["pairs"].items():
        s = int(c[1])
        w, h = v["WT2"]["article"], v["C4HO"]["window"]
        dis = not v["point_signs_agree"]
        ax.errorbar([w["mean_nll_diff"] * 1e3], [h["mean_nll_diff"] * 1e3],
                    xerr=[[(w["mean_nll_diff"] - w["ci"][0]) * 1e3], [(w["ci"][1] - w["mean_nll_diff"]) * 1e3]],
                    yerr=[[(h["mean_nll_diff"] - h["ci"][0]) * 1e3], [(h["ci"][1] - h["mean_nll_diff"]) * 1e3]],
                    fmt="o" if not dis else "X", color=SEEDC[s], ms=7 if not dis else 9, mec=INK if dis else SEEDC[s],
                    elinewidth=0.8, alpha=0.9, capsize=0)
    for s in range(3):
        ax.plot([], [], "o", color=SEEDC[s], label=f"seed {s}")
    ax.plot([], [], "X", color=INK2, ms=9, label="sign disagrees (8/15)")
    ax.axhline(0, color=INK2, lw=1); ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("WT2 paired mean-NLL diff FC−FR (×10⁻³ nats; >0 = FR better)")
    ax.set_ylabel("C4HO paired mean-NLL diff (×10⁻³ nats)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Fig. 5  MoE paired effects across datasets\nr = {cross['moe']['pearson_full_WT2_vs_C4']:.2f} "
                 f"(95% {cross['moe']['pearson_bootstrap_ci'][0]:.2f}…{cross['moe']['pearson_bootstrap_ci'][1]:.2f}); "
                 "bars = per-pair 95% eval CI", color=INK, fontsize=9)
    save(fig, "fig5_moe_cross_dataset")

    # ---- Fig 6: window CI vs execution dispersion
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    cols = list(mci["pairs"])
    ex = mci["summary"]["execution_dispersion"]
    for ax, ds in zip(axes, E.DS):
        schs = ["window", "article", "block16"] if ds == "WT2" else ["window"]
        for y, c in enumerate(cols):
            for k, sch in enumerate(schs):
                e = mci["pairs"][c][ds][sch]
                yy = y + (k - (len(schs) - 1) / 2) * 0.22
                ax.plot([e["ci"][0] * 1e3, e["ci"][1] * 1e3], [yy, yy], color=COL[ds], lw=[1.2, 2.2, 1.2][k],
                        alpha=[0.5, 1, 0.5][k], ls=["-", "-", ":"][k])
                if k == 0 or sch == E.primary_scheme(ds):
                    ax.plot([e["mean_nll_diff"] * 1e3], [yy], "o", color=COL[ds], ms=4)
        sd_o = ex[ds]["within_seed_SD_obs"] * 1e3
        sd_c = (ex[ds]["corrected_SD"] or 0) * 1e3
        ax.axvspan(-sd_o, sd_o, color=GRID, alpha=0.6, label=f"±within-seed SD across executions (obs {sd_o:.2f})")
        ax.axvspan(-sd_c, sd_c, color="#1baf7a", alpha=0.18, label=f"±corrected SD ({sd_c:.2f})")
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(f"{LBL[ds]}: {mci['counts'][ds][E.primary_scheme(ds)]['n_excluding_zero']}/15 CIs exclude 0"
                     + (" (article); naive 4, block16 4" if ds == "WT2" else ""), color=INK, fontsize=9)
        ax.set_xlabel("FC−FR mean-NLL diff (×10⁻³ nats)")
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    axes[0].set_yticks(range(len(cols))); axes[0].set_yticklabels(cols, fontsize=7.5)
    fig.suptitle("Fig. 6  Fixed-weight window CIs (WT2: naive / article / block-16 dotted) vs dispersion across executions",
                 color=INK, fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, "fig6_window_ci_vs_execution")

    # ---- Fig 7: dense / dose reliability
    sets = ["dense4", "dense3", "dose_natural_k1", "dose_natural_k4", "dose_natural_k16", "dose_natural_k64",
            "dose_fixed_k1", "dose_fixed_k4", "dose_fixed_k16", "dose_fixed_k64"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for i, n in enumerate(sets):
        for k, ds in enumerate(E.DS):
            sp = spl[n][ds][PS[ds]]
            x = i + (k - 0.5) * 0.3
            nm = sp["null"]["null_median_pearson"]
            ax.add_patch(plt.Rectangle((x - 0.1, nm["p05"]), 0.2, nm["p95"] - nm["p05"], color=GRID, zorder=0))
            ax.errorbar([x], [sp["pearson"]["median"]], yerr=[[sp["pearson"]["median"] - sp["pearson"]["p05"]],
                        [sp["pearson"]["p95"] - sp["pearson"]["median"]]], fmt="o", color=COL[ds], capsize=3, ms=6,
                        label=LBL[ds] if i == 0 else None)
    ax.axhline(0, color=INK2, lw=1); ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(["dense 4b", "dense 3b", "nat k1", "nat k4", "nat k16", "nat k64", "fix k1", "fix k4", "fix k16",
                        "fix k64"])
    ax.set_ylabel("split-half Pearson (median, 5–95%)")
    ax.plot([], [], "s", color=GRID, ms=10, label="null median 5–95%")
    ax.legend(fontsize=8, loc="lower left", ncol=3)
    ax.set_title("Fig. 7  Dense grouping and dose-level ranking reliability (dashed = 0.5 reliability criterion; "
                 "fixed act-order exploratory)", color=INK, fontsize=9)
    save(fig, "fig7_dense_dose_reliability")
    print("figures written")


if __name__ == "__main__":
    main()
