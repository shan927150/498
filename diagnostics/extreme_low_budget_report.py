"""Markdown + figures for diagnostics/extreme_low_budget.py (routing counts only; no GPTQ)."""
from __future__ import annotations
import os

import numpy as np

import extreme_low_budget as X

FAM_COL = {"full": X.COL["FC"], "typical": X.COL["TC"]}
FAM_NAME = {"full": "Full: Coverage-LP − Random", "typical": "Typical: Coverage-LP − Random"}


def _f(v, d=1):
    return "—" if v is None else f"{v:.{d}f}"


def _pct(v):
    return "—" if v is None else f"{100 * v:+.1f} %"


def write_scarcity_md(sc, layers, path):
    L = []
    w = L.append
    w("# Extreme low-budget exposure scarcity — random Monte Carlo (routing counts only)\n")
    w(f"{sc['n_sets']} deterministic random calibration sets per (pool, B), sampled without replacement; "
      "real routed-token counts from `routing_profile_v2.npz`; 240 experts (layers "
      f"{'/'.join(map(str, layers))}). Uniform expectation per expert = B × 512 × 4 / 60 = "
      f"{sc['uniform_expectation']['4']:.1f} (B = 4) / {sc['uniform_expectation']['8']:.1f} (B = 8). "
      "Thresholds are descriptive scarcity indicators, **not** validated GPTQ sufficiency thresholds. "
      "These are exposure distributions, not GPTQ replicates.\n")
    w(f"RNG streams (numpy default_rng): {sc['mc_seeds']}\n")
    w("## Probability over random sets (all 240 experts; 95 % Wilson CI)\n")
    w("| pool | B | P(any zero) | P(any < 32) | P(any < 64) | P(any < 128) | P(any < 256) | P(≥ 10 % of experts < 128) |")
    w("|---|---|---|---|---|---|---|---|")
    for key, r in sc["results"].items():
        p = r["all"]["probabilities"]
        cell = lambda k: f"{p[k]['p']:.3f} [{p[k]['ci95'][0]:.3f}, {p[k]['ci95'][1]:.3f}]"
        w(f"| {key.rsplit('_B', 1)[0]} | {r['B']} | " + " | ".join(cell(k) for k in (
            "any_zero", "any_below_32", "any_below_64", "any_below_128", "any_below_256",
            "frac_below_128_ge_10pct")) + " |")
    w("\n## Distribution over random sets of per-set statistics (all 240 experts): median [q05, q95]\n")
    keys = ("min", "p1", "p5", "p10", "median", "p90", "max", "gini", "cv", "zero", "below_32", "below_64",
            "below_128", "below_256", "frac_below_128", "unique_sources_median", "effective_sources_median")
    w("| pool | B | " + " | ".join(keys) + " |")
    w("|---|---|" + "---|" * len(keys))
    for key, r in sc["results"].items():
        D = r["all"]["distributions"]
        dd = 3 if True else 1
        fmt = lambda k: (f"{D[k]['q50']:.3f} [{D[k]['q05']:.3f}, {D[k]['q95']:.3f}]" if k in ("gini", "cv", "frac_below_128")
                         else f"{D[k]['q50']:.1f} [{D[k]['q05']:.1f}, {D[k]['q95']:.1f}]")
        w(f"| {key.rsplit('_B', 1)[0]} | {r['B']} | " + " | ".join(fmt(k) for k in keys) + " |")
    w("\n## Per layer: median [q05, q95] over random sets, and probabilities\n")
    w("| pool | B | layer | min | p10 | median | Gini | experts < 128 | P(any zero) | P(any < 64) | P(any < 128) | P(≥ 10 % < 128) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, r in sc["results"].items():
        for Ls in map(str, layers):
            D, P = r["per_layer"][Ls]["distributions"], r["per_layer"][Ls]["probabilities"]
            m = lambda k, d=1: f"{D[k]['q50']:.{d}f} [{D[k]['q05']:.{d}f}, {D[k]['q95']:.{d}f}]"
            w(f"| {key.rsplit('_B', 1)[0]} | {r['B']} | L{Ls} | {m('min')} | {m('p10')} | {m('median')} | "
              f"{m('gini', 3)} | {m('below_128')} | {P['any_zero']['p']:.3f} | {P['any_below_64']['p']:.3f} | "
              f"{P['any_below_128']['p']:.3f} | {P['frac_below_128_ge_10pct']['p']:.3f} |")
    open(path, "w").write("\n".join(L) + "\n")


def write_summary_md(sel, summ, sc, layers, path):
    L = []
    w = L.append
    S = sel["stats"]
    w("# Extreme low-budget (B = 4 / 8) scarcity study — summary (CPU, routing counts only; no GPTQ)\n")
    w(f"**Decision (prespecified gate): {summ['decision']}**\n")
    w(f"All 24 matched selections valid: {summ['valid']}. Schedules: "
      + "; ".join(f"B = {B}: warm {v['warm']}, rounds {v['rounds']}" for B, v in sel["schedule"].items())
      + f". Controller unchanged; K passed = {sel['k_passed_to_controller']} (see module docstring).\n")
    w("## Random Monte Carlo (1,000 sets each) — headline\n")
    w("| pool | B | median min tokens [q05, q95] | median p10 | P(any zero) | P(any < 64) | P(any < 128) | P(≥ 10 % < 128) |")
    w("|---|---|---|---|---|---|---|---|")
    for key, r in sc["results"].items():
        D, P = r["all"]["distributions"], r["all"]["probabilities"]
        w(f"| {key.rsplit('_B', 1)[0]} | {r['B']} | {D['min']['q50']:.0f} [{D['min']['q05']:.0f}, {D['min']['q95']:.0f}] | "
          f"{D['p10']['q50']:.0f} | {P['any_zero']['p']:.3f} | {P['any_below_64']['p']:.3f} | "
          f"{P['any_below_128']['p']:.3f} | {P['frac_below_128_ge_10pct']['p']:.3f} |")
    w("\n## Matched selections (all 240 experts)\n")
    w("| B | seed | strategy | min | p5 | p10 | median | zero | < 64 | < 128 | < 256 | Gini (layer mean) | eff. sources (median) | post-warm d(s) | set JS | agg. L1 (L1/L8/L16/L23) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for B in X.BUDGETS:
        for sd in X.SEEDS:
            for c in X.STRATS:
                t, r = S[f"{c}_B{B}_s{sd}"]["tail"]["all"], S[f"{c}_B{B}_s{sd}"]["repr"]
                w(f"| {B} | {sd} | {X.NAMES[c]} | {t['min']:.0f} | {t['p5']:.0f} | {t['p10']:.0f} | {t['median']:.0f} | "
                  f"{t['zero']} | {t['below_64']} | {t['below_128']} | {t['below_256']} | {t['gini_layer_mean']:.4f} | "
                  f"{t['effective_sources_median']:.2f} | {r['d_post_warm']:.3f} | {r['set_js']:.5f} | "
                  + " / ".join(f"{v:.3f}" for v in r["aggregate_l1_by_layer"].values()) + " |")
    w("\n## Paired Coverage − Random (same seed, same warm start)\n")
    w("| B | family | seed | Δmin | Δp5 | Δp10 (rel) | Δzero | Δ<64 | Δ<128 (rel. reduction) | Δ<256 | ΔGini | Δeff. sources | Δpost-warm d(s) | Δset JS |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for B in (*X.BUDGETS, 16):
        for fam in X.PAIRS:
            for sd in X.SEEDS:
                d = summ["deltas"][f"{fam}_B{B}_s{sd}"]
                w(f"| {B}{' (ref.)' if B == 16 else ''} | {fam} | {sd} | {d['min']:+.0f} | {d['p5']:+.0f} | "
                  f"{d['p10']:+.0f} ({_pct(d['p10_rel'])}) | {d['zero']:+d} | {d['below_64']:+d} | "
                  f"{d['below_128']:+d} ({_pct(d['below_128_rel_reduction'])}) | {d['below_256']:+d} | "
                  f"{d['gini_layer_mean']:+.4f} | {d['effective_sources_median']:+.2f} | {d['d_post_warm']:+.3f} | "
                  f"{d['set_js']:+.5f} |")
    w("\n## Zero-exposure handling (Coverage-LP trajectories)\n")
    w("| B | seed | strategy | round | budget | zero-exposure experts before | targets reached / unreached | exploration seqs | LP positive | random fill |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for B in X.BUDGETS:
        for sd in X.SEEDS:
            for c in ("FC", "TC"):
                for t in sel["sets"][f"{c}_B{B}_s{sd}"]["trajectory"]:
                    w(f"| {B} | {sd} | {X.NAMES[c]} | {t['round']} | {t['round_budget']} | {t['n_zero_exposure']} | "
                      f"{t['n_exploration_targets_reached']} / {t['n_exploration_targets_unreached']} | "
                      f"{t['n_exploration_sequences']} | {t['n_lp_positive_value_selected']} | "
                      f"{t['n_zero_marginal_random_fill']} |")
    w("\n## Selection overlap (Jaccard of post-warm picks)\n")
    w("| B | seed | FC vs FR | TC vs TR | FC vs TC | FR vs TR |")
    w("|---|---|---|---|---|---|")
    for B in X.BUDGETS:
        for sd in X.SEEDS:
            o = S[f"overlap_B{B}_s{sd}"]
            w(f"| {B} | {sd} | {o['FC_FR']:.2f} | {o['TC_TR']:.2f} | {o['FC_TC']:.2f} | {o['FR_TR']:.2f} |")
    w("\n## Prespecified gate\n")
    g = summ["gate"]
    w(f"1. Full Random scarce (≥ {100 * g['scarcity_frac_below_128']:.0f} % of experts < 128 tokens, or ≥ 1 zero) "
      f"in ≥ {g['min_seeds']}/3 matched seeds; 2. Coverage (full or typical) raises p10 by ≥ "
      f"{100 * g['p10_rel_gain']:.0f} % or cuts experts < 128 by ≥ {100 * g['below128_rel_reduction']:.0f} % in "
      f"≥ {g['min_seeds']}/3 seeds; 3. its minimum is not worse in ≥ {g['min_seeds']}/3 seeds; 4. valid selections.\n")
    w("| B | FR scarce by seed | c1 | family | lower tail improved by seed | c2 | min worse by seed | c3 | Gini better by seed |")
    w("|---|---|---|---|---|---|---|---|---|")
    for B in X.BUDGETS:
        dB = summ["gate_detail"].get(str(B))
        if not dB:
            continue
        for fam, f in dB["families"].items():
            yn = lambda v: "/".join("Y" if x else "n" for x in v)
            w(f"| {B} | {yn(dB['fr_scarce_by_seed'])} | {dB['c1']} | {fam} | {yn(f['lower_tail_improved'])} | {f['c2']} | "
              f"{yn(f['min_worse'])} | {f['c3']} | {yn(f['gini_better'])} |")
    c = summ["compute"]
    w("\n## Compute feasibility of a targeted GPTQ comparison (estimates from the B = 16–128 runs)\n")
    w(f"Hessian replay fit: {c['hessian_fit']['intercept_s']:.1f} s + {c['hessian_fit']['slope_s_per_seq']:.3f} s × B; "
      f"GPTQ ≈ {c['gptq_s_per_run']:.0f} s per run (independent of B); routing profile ≈ {c['routing_profile_s']:.0f} s "
      f"forward-equivalent (once; needed by FC, TC and TR); LP selection ≤ {c['lp_selection_s_max_observed']:.1f} s; "
      f"peak GPU ≈ {c['peak_gpu_gb']:.1f} GB.\n")
    w("| B | Hessian (extrap.) s | GPTQ s | per run s | calibration forward-equiv s | profile / calibration forwards | 12-run comparison GPU s |")
    w("|---|---|---|---|---|---|---|")
    for B, v in c["by_budget"].items():
        w(f"| {B} | {v['hessian_s_extrapolated']:.1f} | {v['gptq_s']:.0f} | {v['run_s']:.0f} | "
          f"{v['calib_forward_equiv_s']:.2f} | {v['profile_over_calib_forwards']:.0f}× | {v['minimal_comparison_gpu_s']:.0f} |")
    w("\nFigures (`figures/`, PDF + PNG): fig1_exposure_by_budget, fig2_scarcity_probability, "
      "fig3_coverage_lower_tail, fig4_balance_vs_atypicality, fig5_source_diversity. "
      "None of them measures quantization quality.")
    open(path, "w").write("\n".join(L) + "\n")


# ---------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.6, "axes.axisbelow": True})
    return plt


def _save(fig, d, name):
    fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(d, name + ".png"), dpi=200, bbox_inches="tight")


def figures(sc, mc_counts, sel, summ, ref16, layers, E, d):
    plt = _style()
    S = sel["stats"]
    # ---- fig1 exposure distribution per layer
    fig, axes = plt.subplots(1, len(layers), figsize=(11, 3.4), sharey=True)
    lab = ["B=4\n(1,000 sets)", "B=8\n(1,000 sets)", "B=16\n(3 matched)"]
    cols = ["#9ec5f0", "#2a78d6", "#123f75"]
    for li, (ax, Lr) in enumerate(zip(axes, layers)):
        sl = slice(li * E, (li + 1) * E)
        data = [mc_counts["full_B4"][:, sl].ravel(), mc_counts["full_B8"][:, sl].ravel(),
                np.concatenate([np.asarray(ref16[f"FR_B16_s{s}"]["counts"])[sl] for s in X.SEEDS])]
        bp = ax.boxplot(data, widths=0.55, whis=(1, 99), showfliers=False, patch_artist=True,
                        medianprops=dict(color="white", lw=1.5))
        for p, c in zip(bp["boxes"], cols):
            p.set_facecolor(c); p.set_edgecolor(c)
        for t in (32, 64, 128, 256):
            ax.axhline(t, color="#888", lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 3], lab, fontsize=7)
        ax.set_title(f"layer {Lr}")
    axes[0].set_ylabel("routed tokens per expert (log)")
    for t in (32, 64, 128, 256):
        axes[-1].text(3.45, t, str(t), fontsize=6.5, color="#666", va="center")
    fig.suptitle("Full Random: per-expert routed-token exposure (whiskers p1–p99; dotted = 32/64/128/256)", y=1.02)
    _save(fig, d, "fig1_exposure_by_budget"); plt.close(fig)

    # ---- fig2 scarcity probability
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    th = list(X.THRESH)
    sty = {("full", 4): (X.COL["FR"], "o", "--"), ("full", 8): (X.COL["FR"], "o", "-"),
           ("typical_half", 4): (X.COL["TR"], "s", "--"), ("typical_half", 8): (X.COL["TR"], "s", "-")}
    for (pool, B), (c, m, ls) in sty.items():
        P = sc["results"][f"{pool}_B{B}"]["all"]["probabilities"]
        p = [P[f"any_below_{t}"]["p"] for t in th]
        lo = [p_ - P[f"any_below_{t}"]["ci95"][0] for p_, t in zip(p, th)]
        hi = [P[f"any_below_{t}"]["ci95"][1] - p_ for p_, t in zip(p, th)]
        ax.errorbar(th, p, yerr=[lo, hi], color=c, marker=m, ls=ls, lw=2, ms=6, capsize=2,
                    label=f"{'Full' if pool == 'full' else 'Typical'} Random, B = {B}")
    ax.set_xscale("log", base=2); ax.set_xticks(th, [str(t) for t in th])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("token threshold t"); ax.set_ylabel("P(at least one of 240 experts < t)")
    ax.set_title("Scarcity probability over 1,000 random sets (95 % CI)")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _save(fig, d, "fig2_scarcity_probability"); plt.close(fig)

    # ---- fig3 coverage lower-tail deltas
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (k, t) in zip(axes, (("min", "Δ minimum exposure (tokens)"), ("p10", "Δ p10 exposure (tokens)"),
                                 ("below_128", "Δ experts < 128 tokens"))):
        for fi, fam in enumerate(X.PAIRS):
            for bi, B in enumerate((*X.BUDGETS, 16)):
                v = [summ["deltas"][f"{fam}_B{B}_s{s}"][k] for s in X.SEEDS]
                x = bi + (fi - 0.5) * 0.22
                ax.scatter([x] * 3, v, color=FAM_COL[fam], s=26, edgecolor="white", lw=0.6, zorder=3,
                           label=FAM_NAME[fam] if bi == 0 else None)
                ax.plot([x - 0.08, x + 0.08], [np.mean(v)] * 2, color=FAM_COL[fam], lw=2)
        ax.axhline(0, color="#555", lw=0.9)
        ax.set_xticks([0, 1, 2], ["B=4", "B=8", "B=16 (ref.)"])
        ax.set_title(t, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Matched Coverage-LP − Random (dots = seeds, bar = mean). Left/middle: > 0 = more tokens; right: > 0 = more scarce experts",
                 y=1.03, fontsize=9)
    _save(fig, d, "fig3_coverage_lower_tail"); plt.close(fig)

    # ---- fig4 Gini vs atypicality
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    mk = {4: "o", 8: "s", 16: "^"}
    for fam in X.PAIRS:
        for B in (*X.BUDGETS, 16):
            xs = [summ["deltas"][f"{fam}_B{B}_s{s}"]["gini_layer_mean"] for s in X.SEEDS]
            ys = [summ["deltas"][f"{fam}_B{B}_s{s}"]["d_post_warm"] for s in X.SEEDS]
            ax.scatter(xs, ys, color=FAM_COL[fam], marker=mk[B], s=36, edgecolor="white", lw=0.6,
                       label=f"{fam}, B={B}{' (ref.)' if B == 16 else ''}")
    ax.axhline(0, color="#555", lw=0.9); ax.axvline(0, color="#555", lw=0.9)
    ax.set_xlabel("Δ expert-token Gini (Coverage − Random; < 0 = more balanced)")
    ax.set_ylabel("Δ mean post-warm d(s) (> 0 = more atypical)")
    ax.set_title("Exposure balance vs routing atypicality (each point = budget × seed)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, d, "fig4_balance_vs_atypicality"); plt.close(fig)

    # ---- fig5 effective source diversity
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    allstats = dict(S); allstats.update(ref16)
    for ax, (k, t) in zip(axes, (("effective_sources_median", "median effective sources per expert"),
                                 ("unique_sources_median", "median unique source sequences per expert"))):
        for ci, c in enumerate(X.STRATS):
            for bi, B in enumerate((4, 8, 16)):
                v = [allstats[f"{c}_B{B}_s{s}"]["tail"]["all"][k] / B for s in X.SEEDS]
                x = bi + (ci - 1.5) * 0.18
                ax.bar(x, np.mean(v), width=0.16, color=X.COL[c], label=X.NAMES[c] if bi == 0 else None)
                ax.scatter([x] * 3, v, color="#333", s=8, zorder=3)
        ax.set_xticks([0, 1, 2], ["B=4", "B=8", "B=16 (ref.)"])
        ax.set_ylabel(f"{t} / B"); ax.set_ylim(0, 1.05)
        ax.set_title(t, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(1.1, -0.12))
    fig.suptitle("Source diversity, normalised by budget (1 = every selected sequence contributes equally)", y=1.03,
                 fontsize=9)
    _save(fig, d, "fig5_source_diversity"); plt.close(fig)
