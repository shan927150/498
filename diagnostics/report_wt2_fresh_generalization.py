"""Figures for runs_wt2_fresh_generalization (reads JSON outputs + logs/figure_data.npz only)."""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wt2_fresh_generalization as W   # noqa: E402

p, jload = W.p, W.jload
COL = {"original": "#2a78d6", "fresh_article": "#1baf7a", "fresh_window": "#eda100", "C4HO": "#eb6834"}
SEEDC = ["#2a78d6", "#1baf7a", "#e87ba4"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LBL = {"original": "original WT2 (49 art.)", "fresh_article": "fresh articles (10)", "fresh_window": "fresh windows, same art. (48)",
       "C4HO": "C4-val (128 docs)"}
NAME = {"moe": "MoE FC−FR (15)", "dense4": "Dense 4-bit (5)", "dense3": "Dense 3-bit (5)"}
COH = ["original", "fresh_article", "fresh_window", "C4HO"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


_orig_errorbar = matplotlib.axes.Axes.errorbar


def _safe_errorbar(self, x, y, yerr=None, xerr=None, **kw):
    """errorbar that accepts intervals not containing the point (pivot / percentile CIs): draws lo..hi explicitly."""
    if yerr is None and xerr is None:
        return _orig_errorbar(self, x, y, **kw)
    col = kw.get("color", INK)
    lw = kw.get("lw", kw.get("elinewidth", 1.2))
    for i in range(len(x)):
        if yerr is not None:
            lo, hi = y[i] - yerr[0][i], y[i] + yerr[1][i]
            self.plot([x[i], x[i]], [lo, hi], color=col, lw=lw, alpha=kw.get("alpha", 1))
        if xerr is not None:
            lo, hi = x[i] - xerr[0][i], x[i] + xerr[1][i]
            self.plot([lo, hi], [y[i], y[i]], color=col, lw=kw.get("elinewidth", lw), alpha=kw.get("alpha", 1))
    k2 = {k: v for k, v in kw.items() if k not in ("capsize", "elinewidth", "lw")}
    fmt = k2.pop("fmt", "o")
    return self.plot(x, y, fmt, **k2)


matplotlib.axes.Axes.errorbar = _safe_errorbar


def save(fig, name):
    os.makedirs(p("figures"), exist_ok=True)
    fig.savefig(p("figures", name + ".png"), dpi=160, bbox_inches="tight")
    fig.savefig(p("figures", name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    var = jload(p("fresh_variance_components.json"))["results"]
    spl = jload(p("fresh_split_half_reliability.json"))["results"]
    tr = jload(p("cross_cohort_rank_transfer.json"))["results"]
    wci = jload(p("fresh_window_ci_analysis.json"))
    hist = jload(p("historical_significance_audit.json"))
    fd = np.load(p("logs", "figure_data.npz"))
    moe = jload(p("moe_fresh_ppl_results.json"))["moe"]
    cols = list(moe["original"]["paired"])

    # Fig 1 -- MoE paired effects original vs fresh (two panels)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, c in zip(axes, ("fresh_article", "fresh_window")):
        for k in cols:
            s = int(k[1])
            o, f = wci["original"]["pairs"][k], wci[c]["pairs"][k]
            dis = np.sign(o["mean_nll_diff"]) != np.sign(f["mean_nll_diff"])
            ax.errorbar([o["mean_nll_diff"] * 1e3], [f["mean_nll_diff"] * 1e3],
                        xerr=[[(o["mean_nll_diff"] - o["ci95"][0]) * 1e3], [(o["ci95"][1] - o["mean_nll_diff"]) * 1e3]],
                        yerr=[[(f["mean_nll_diff"] - f["ci95"][0]) * 1e3], [(f["ci95"][1] - f["mean_nll_diff"]) * 1e3]],
                        fmt="X" if dis else "o", color=SEEDC[s], ms=8 if dis else 6, mec=INK if dis else SEEDC[s],
                        elinewidth=0.7, alpha=0.9)
        ax.axhline(0, color=INK2, lw=1); ax.axvline(0, color=INK2, lw=1)
        t = tr["moe"][f"original__vs__{c}"]
        ax.set_title(f"{LBL[c]}: r = {t['pearson']:.2f} (95% {t['pearson_ci95'][0]:.2f}…{t['pearson_ci95'][1]:.2f}), "
                     f"signs agree {round(t['sign_agreement'] * 15)}/15", color=INK, fontsize=8.5)
        ax.set_xlabel("original WT2 FC−FR mean-NLL diff (×10⁻³; >0 = FR better)")
        ax.set_ylabel(f"{LBL[c]} (×10⁻³)")
    for s in range(3):
        axes[0].plot([], [], "o", color=SEEDC[s], label=f"seed {s}")
    axes[0].plot([], [], "X", color=INK2, label="sign differs")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Fig. 1  MoE paired effects: original vs fresh WikiText-2 (bars: article-cluster 95% CIs; r on raw paired "
                 "effects in plot, transfer r uses within-seed-centred effects)", color=INK, fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, "fig1_moe_original_vs_fresh")

    # Fig 2 -- variance decomposition
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, m in zip(axes, ("moe", "dense4", "dense3")):
        for i, c in enumerate(COH):
            r = var[m][c]
            x = i * 3.5
            ax.plot([x], [r["V_obs"] * 1e7], "o", color=COL[c], mfc="white", mew=2, ms=7)
            ax.plot([x + 1], [r["V_samp"] * 1e7], "s", color=COL[c], mfc="white", mew=2, ms=7)
            lo, hi = r["V_corr_pivot_ci"]
            ax.errorbar([x + 2], [r["V_corr"] * 1e7], yerr=[[(r["V_corr"] - lo) * 1e7], [(hi - r["V_corr"]) * 1e7]], fmt="D",
                        color=COL[c], capsize=3, ms=7, lw=2)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks([i * 3.5 + 1 for i in range(4)]); ax.set_xticklabels(["orig", "fresh\nart.", "fresh\nwin.", "C4"])
        ax.set_title(NAME[m], color=INK)
    axes[0].set_ylabel("variance of mean NLL (×10⁻⁷ nats²)\n○ observed  □ sampling  ◆ corrected (95%)")
    fig.suptitle("Fig. 2  Execution-variance decomposition per cohort (negative corrected estimates not clipped)", color=INK,
                 fontsize=9.5, y=1.03)
    fig.tight_layout()
    save(fig, "fig2_variance_decomposition")

    # Fig 3 -- split-half reliability
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, m in zip(axes, ("moe", "dense4", "dense3")):
        for i, c in enumerate(COH):
            rr = fd[f"{m}__{c}__pearson"]; nm = fd[f"{m}__{c}__null_med"]
            vp = ax.violinplot([rr[np.isfinite(rr)]], positions=[i], widths=0.7, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(COL[c]); b.set_alpha(0.55)
            ax.plot([i - 0.3, i + 0.3], [np.median(rr)] * 2, color=COL[c], lw=2.5)
            ax.add_patch(plt.Rectangle((i + 0.33, np.quantile(nm, .05)), 0.12, np.quantile(nm, .95) - np.quantile(nm, .05),
                                       color=INK2, alpha=0.35))
            ax.text(i, 1.08, f"p={spl[m][c]['null']['p_value']:.2f}", ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(range(4)); ax.set_xticklabels(["orig", "fresh art.", "fresh win.", "C4"])
        ax.set_ylim(-1.05, 1.18); ax.axhline(0, color=INK2, lw=1)
        ax.set_title(NAME[m], color=INK, pad=14)
    axes[0].set_ylabel("split-half Pearson (article splits)\ngrey bar: null median 5–95%")
    fig.suptitle("Fig. 3  Within-cohort split-half reliability vs common-effect null", color=INK, fontsize=9.5, y=1.04)
    fig.tight_layout()
    save(fig, "fig3_split_half")

    # Fig 4 -- article vs window generalization
    fig, ax = plt.subplots(figsize=(9, 3.8))
    comps = [("original__vs__fresh_article", "orig vs fresh art."), ("original__vs__fresh_window", "orig vs fresh win."),
             ("fresh_article__vs__fresh_window", "fresh art. vs fresh win."), ("original__vs__C4HO", "orig vs C4")]
    for mi, m in enumerate(("moe", "dense4", "dense3")):
        for j, (k, lab) in enumerate(comps):
            t = tr[m][k]
            x = j + (mi - 1) * 0.22
            ax.errorbar([x], [t["pearson"]], yerr=[[t["pearson"] - t["pearson_ci95"][0]], [t["pearson_ci95"][1] - t["pearson"]]],
                        fmt="o", color=["#2a78d6", "#1baf7a", "#e87ba4"][mi], capsize=3, ms=6, label=NAME[m] if j == 0 else None)
            if t["expected_r_if_identical_true_effects"]:
                ax.plot([x], [t["expected_r_if_identical_true_effects"]], "*", color=INK, ms=9,
                        label="expected r if identical true effects" if (mi == 0 and j == 0) else None)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([c[1] for c in comps])
    ax.set_ylabel("correlation of centred effects (95% CI)")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.set_title("Fig. 4  Ranking transfer: article-disjoint vs same-article fresh text", color=INK, fontsize=9.5)
    save(fig, "fig4_article_vs_window")

    # Fig 5 -- FR/FC per pair CIs across cohorts
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for j, c in enumerate(("original", "fresh_article", "fresh_window", "C4HO")):
        for y, k in enumerate(cols):
            e = wci[c]["pairs"][k]
            yy = y + (j - 1.5) * 0.18
            ax.plot([e["ci95"][0] * 1e3, e["ci95"][1] * 1e3], [yy, yy], color=COL[c], lw=2 if e["excludes_zero"] else 1,
                    alpha=1 if e["excludes_zero"] else 0.6)
            ax.plot([e["mean_nll_diff"] * 1e3], [yy], "o", color=COL[c], ms=4)
    for c in ("original", "fresh_article", "fresh_window", "C4HO"):
        ax.plot([], [], color=COL[c], lw=2, label=f"{LBL[c]}: {wci[c]['n_excluding_zero']}/15 exclude 0")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(15)); ax.set_yticklabels(cols, fontsize=7.5)
    ax.set_xlabel("FC−FR mean-NLL diff (×10⁻³ nats; >0 = FR better); thick = CI excludes 0")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title("Fig. 5  Per-grouping FR/FC effects and article-cluster 95% CIs across cohorts", color=INK, fontsize=9.5)
    save(fig, "fig5_fr_fc_instability")

    # Fig 6 -- dense bit-width
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for i, c in enumerate(COH):
        for k, m in enumerate(("dense4", "dense3")):
            r = var[m][c]
            x = i + (k - 0.5) * 0.3
            lo, hi = r["V_corr_pivot_ci"]
            ax.errorbar([x], [r["V_corr"] * 1e7], yerr=[[(r["V_corr"] - lo) * 1e7], [(hi - r["V_corr"]) * 1e7]], fmt="D" if k else "o",
                        color=COL[c], capsize=3, ms=6, mfc=COL[c] if k else "white", mew=2)
            ax.plot([x], [r["V_obs"] * 1e7], "_", color=INK2, ms=10)
    ax.plot([], [], "o", color=INK2, mfc="white", label="4-bit corrected (95%)"); ax.plot([], [], "D", color=INK2, label="3-bit corrected")
    ax.plot([], [], "_", color=INK2, ms=10, label="observed")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([LBL[c] for c in COH], fontsize=8)
    ax.set_ylabel("between-grouping variance\n(×10⁻⁷ nats²)")
    ax.legend(fontsize=8)
    ax.set_title("Fig. 6  Dense grouping variance at 4 vs 3 bits across cohorts", color=INK, fontsize=9.5)
    save(fig, "fig6_dense_bitwidth")

    # Fig 7 -- historical significance audit
    fig, ax = plt.subplots(figsize=(8, 5.2))
    rows = hist["table"]
    for y, r in enumerate(rows):
        sig = r["article_significant"]
        c = "#2a78d6" if r["favored"] == "FR" else "#eb6834"
        ax.plot([r["article_ci"][0] * 1e3, r["article_ci"][1] * 1e3], [y, y], color=c if sig else GRID, lw=3 if sig else 2)
        ax.plot([r["mean_nll_diff_FC_minus_FR"] * 1e3], [y], "o", color=c if sig else INK2, ms=6)
        lab = f"s{r['seed']} {r['grouping']}"
        if sig:
            lab += f"  ★ favours {r['favored']} (p={r['p_boot_article']:.3f}, BH {r['p_BH_article']:.2f})"
        elif r["naive_window_significant"]:
            lab += "  (naive-window only)"
        ax.text(2.75, y, lab, va="center", fontsize=7.5, color=INK if sig else INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks([]); ax.set_xlim(-2.8, 2.7)
    ax.set_xlabel("original WT2 FC−FR mean-NLL diff (×10⁻³; >0 = FR better), article-cluster 95% CI")
    ax.plot([], [], color="#2a78d6", lw=3, label="significant, favours FR"); ax.plot([], [], color="#eb6834", lw=3, label="significant, favours FC")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Fig. 7  Historical significance audit: 3/15 marginal, opposite directions; 0/15 after BH or max-|t|",
                 color=INK, fontsize=9)
    save(fig, "fig7_historical_significance")
    print("figures written")


if __name__ == "__main__":
    main()
