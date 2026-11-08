"""Low-budget sweep report: reconstruction by budget x strategy, paired
Coverage-vs-Random gains, diminishing returns, mechanism diagnostics, figures.

Primary quantities (lower error is better):
  G_full(B,s)    = [E(FC,B,s) - E(FR,B,s)] / E(FR,B,s)
  G_typical(B,s) = [E(TC,B,s) - E(TR,B,s)] / E(TR,B,s)      (G < 0: Coverage better)
  R(B)           = [E(B) - E(2B)] / E(B)                     (per strategy, per seed)
Decision (prespecified in the task; EFFECT = 0.5 % practical reference):
  PROMISING  some B in {16, 32, 64}: G < 0 on 3/3 seeds for FULL or TYPICAL with
             mean G <= -0.5 %, no layer with mean relative regression > +0.5 %
             and expert-p95 relative change <= +0.5 %
  PARTIAL    not PROMISING, and for FULL or TYPICAL the mean G strictly increases
             with B over 16 < 32 < 64 < 128 (penalty shrinks as B decreases)
             AND mean G(16) < mean G(128)
  NO-GAIN    otherwise
  RUN-BLOCKED  any of the 48 runs missing / invalid
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import low_budget as LB                                # noqa: E402

BUDGETS, SEEDS, STRATS, NAMES = LB.BUDGETS, LB.SEEDS, list(LB.STRATS), LB.NAMES
EFFECT = 0.005
COLORS = {"FR": "#2a78d6", "FC": "#eda100", "TR": "#eb6834", "TC": "#1baf7a"}
MARKERS = {"FR": "o", "FC": "D", "TR": "s", "TC": "^"}


def coverage_gain(E, cov, rnd, B, s):
    return (E[(cov, B, s)] - E[(rnd, B, s)]) / E[(rnd, B, s)]


def doubling_return(E, code, B, s):
    return (E[(code, B, s)] - E[(code, 2 * B, s)]) / E[(code, B, s)]


def decide(G, per_layer_rel, p95_rel):
    """G: {'full'|'typical': {B: [g_s0, g_s1, g_s2]}}."""
    for fam in ("full", "typical"):
        for B in (16, 32, 64):
            g = np.array(G[fam][B])
            if (g < 0).all() and g.mean() <= -EFFECT and max(per_layer_rel[fam][B].values()) <= EFFECT \
                    and p95_rel[fam][B] <= EFFECT:
                return "LOW-BUDGET-COVERAGE-PROMISING", dict(family=fam, budget=B)
    for fam in ("full", "typical"):
        m = [float(np.mean(G[fam][B])) for B in BUDGETS]
        if all(m[i] < m[i + 1] for i in range(len(m) - 1)) and m[0] < m[-1]:
            return "LOW-BUDGET-COVERAGE-PARTIAL", dict(family=fam, means=m)
    return "LOW-BUDGET-COVERAGE-NO-GAIN", {}


def run_files(outdir, matched_dir):
    """(code, B, seed) -> run json path; B = 128 anchors are the matched runs."""
    sel = json.load(open(os.path.join(outdir, "selected_sets.json")))
    out = {}
    for B in BUDGETS:
        d = matched_dir if B == 128 else outdir
        for sd in SEEDS:
            h = sel["sets"][f"FR_B{B}_s{sd}"]["warm_hash"][:8]
            for code, (st, pool) in LB.STRATS.items():
                tag = "" if pool == "full" else f"_pool-{pool}"
                out[(code, B, sd)] = os.path.join(d, f"e3_{st}_B{B}_b4_d1_s{sd}{tag}_warm-{h}.json")
    return out, sel


def build(a):
    files, sel = run_files(a.outdir, a.matched_dir)
    runs, missing, invalid = {}, [], []
    for k, f in files.items():
        if not os.path.exists(f):
            missing.append(f"{k[0]}_B{k[1]}_s{k[2]}")
            continue
        try:
            r, code = LB.validate(f, a)
            assert code == k[0]
            runs[k] = r
        except SystemExit as e:
            invalid.append(str(e))
    res = dict(missing=missing, invalid=invalid, n_valid=len(runs), n_expected=len(files))
    if missing or invalid or len({json.dumps(r["eval_universe"]) for r in runs.values()}) != 1:
        res["decision"] = dict(label="RUN-BLOCKED", reason=dict(missing=missing, invalid=invalid))
        return res
    layers = runs[("FR", 16, 0)]["summary"]["layers"]
    E = {k: r["summary"]["recon_mean"] for k, r in runs.items()}
    pe = {k: {(x["layer"], x["expert"]): x["mean"] for x in r["per_expert"]} for k, r in runs.items()}
    res["table"] = {f"{c}_B{B}": dict(name=NAMES[c], B=B, by_seed={str(s): E[(c, B, s)] for s in SEEDS},
                                      mean=float(np.mean([E[(c, B, s)] for s in SEEDS])),
                                      sd=float(np.std([E[(c, B, s)] for s in SEEDS], ddof=1)),
                                      expert_median=float(np.mean([runs[(c, B, s)]["summary"]["recon_expert_median"] for s in SEEDS])),
                                      p90=float(np.mean([runs[(c, B, s)]["summary"]["recon_expert_p90"] for s in SEEDS])),
                                      p95=float(np.mean([runs[(c, B, s)]["summary"]["recon_expert_p95"] for s in SEEDS])),
                                      per_layer={str(L): float(np.mean([runs[(c, B, s)]["summary"]["recon_per_layer"][str(L)]
                                                                        for s in SEEDS])) for L in layers},
                                      n_rtn_fallback=int(sum(runs[(c, B, s)]["summary"]["n_zero_token_quantization_fallback"] for s in SEEDS)))
                    for c in STRATS for B in BUDGETS}
    G, pl_rel, p95_rel, G_layer, G_expert = {}, {}, {}, {}, {}
    for fam, cov, rnd in (("full", "FC", "FR"), ("typical", "TC", "TR")):
        G[fam] = {B: [coverage_gain(E, cov, rnd, B, s) for s in SEEDS] for B in BUDGETS}
        pl_rel[fam] = {B: {str(L): float(np.mean([(runs[(cov, B, s)]["summary"]["recon_per_layer"][str(L)]
                                                    - runs[(rnd, B, s)]["summary"]["recon_per_layer"][str(L)])
                                                   / runs[(rnd, B, s)]["summary"]["recon_per_layer"][str(L)] for s in SEEDS]))
                           for L in layers} for B in BUDGETS}
        p95_rel[fam] = {B: float(np.mean([(runs[(cov, B, s)]["summary"]["recon_expert_p95"]
                                           - runs[(rnd, B, s)]["summary"]["recon_expert_p95"])
                                          / runs[(rnd, B, s)]["summary"]["recon_expert_p95"] for s in SEEDS])) for B in BUDGETS}
        G_expert[fam] = {B: dict(frac_experts_improved=float(np.mean([np.mean([pe[(cov, B, s)][k] < pe[(rnd, B, s)][k]
                                                                             for k in pe[(rnd, B, s)]]) for s in SEEDS])))
                         for B in BUDGETS}
    res["G"] = {fam: {str(B): dict(by_seed=v, mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)),
                                   n_negative=int(sum(x < 0 for x in v)))
                      for B, v in G[fam].items()} for fam in G}
    res["G_per_layer_rel"] = {fam: {str(B): v for B, v in pl_rel[fam].items()} for fam in pl_rel}
    res["G_p95_rel"] = {fam: {str(B): v for B, v in p95_rel[fam].items()} for fam in p95_rel}
    res["G_expert"] = {fam: {str(B): v for B, v in G_expert[fam].items()} for fam in G_expert}
    res["R"] = {c: {f"{B}->{2 * B}": dict(by_seed=[doubling_return(E, c, B, s) for s in SEEDS],
                                         mean=float(np.mean([doubling_return(E, c, B, s) for s in SEEDS])))
                    for B in (16, 32, 64)} for c in STRATS}
    lab, info = decide(G, pl_rel, p95_rel)
    res["decision"] = dict(label=lab, **info)
    # mechanism: selection-side statistics from selected_sets.json
    st = sel["stats"]
    res["mechanism"] = {f"{c}_B{B}": dict(
        gini=float(np.mean([st[f"{c}_B{B}_s{s}"]["overall"]["gini_mean_over_layers"] for s in SEEDS])),
        min_tokens=float(np.mean([st[f"{c}_B{B}_s{s}"]["overall"]["min"] for s in SEEDS])),
        p10_tokens=float(np.mean([st[f"{c}_B{B}_s{s}"]["overall"]["p10"] for s in SEEDS])),
        median_tokens=float(np.mean([st[f"{c}_B{B}_s{s}"]["overall"]["median"] for s in SEEDS])),
        zero=float(np.mean([st[f"{c}_B{B}_s{s}"]["overall"]["zero"] for s in SEEDS])),
        below_256=float(np.mean([st[f"{c}_B{B}_s{s}"]["overall"]["below_256"] for s in SEEDS])),
        eff_sources=float(np.mean([st[f"{c}_B{B}_s{s}"]["overall"]["effective_sources_median_mean_over_layers"] for s in SEEDS])),
        d_mean=float(np.mean([st[f"{c}_B{B}_s{s}"]["representativeness"]["d_mean"] for s in SEEDS])),
        d_post_warm=float(np.mean([st[f"{c}_B{B}_s{s}"]["representativeness"]["d_post_warm"] for s in SEEDS])),
        set_js=float(np.mean([st[f"{c}_B{B}_s{s}"]["representativeness"]["set_js"] for s in SEEDS])),
        set_entropy=float(np.mean([st[f"{c}_B{B}_s{s}"]["representativeness"]["set_entropy_mean"] for s in SEEDS])),
        seq_entropy=float(np.mean([st[f"{c}_B{B}_s{s}"]["representativeness"]["seq_entropy_mean"] for s in SEEDS])),
        aggregate_l1={str(L): float(np.mean([st[f"{c}_B{B}_s{s}"]["representativeness"]["aggregate_l1_by_layer"][str(L)]
                                             for s in SEEDS])) for L in layers},
        trajectories=[sel["sets"][f"{c}_B{B}_s{s}"]["trajectory"] for s in SEEDS])
        for c in STRATS for B in BUDGETS}
    res["cost"] = {f"{c}_B{B}": dict(
        selection_s=float(np.mean([runs[(c, B, s)]["summary"]["cost"]["selection_s"] for s in SEEDS])),
        hessian_s=float(np.mean([runs[(c, B, s)]["summary"]["cost"]["hessian_replay_s"] for s in SEEDS])),
        gptq_s=float(np.mean([runs[(c, B, s)]["summary"]["cost"]["gptq_s"] for s in SEEDS])),
        calib_forward_equiv_s=float(np.mean([runs[(c, B, s)]["summary"]["cost"]["calib_full_forward_equiv_s"] for s in SEEDS])),
        routing_forward_equiv_s=float(np.mean([runs[(c, B, s)]["summary"]["cost"].get("routing_full_forward_equiv_s", 0.0)
                                               for s in SEEDS])),
        peak_gpu_mem_gb=float(max(runs[(c, B, s)]["summary"]["peak_gpu_mem_gb"] for s in SEEDS)))
        for c in STRATS for B in BUDGETS}
    res["layers"] = layers
    return res


def pct(x):
    return f"{100 * x:+.2f} %"


def figures(res, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e6e6e6"})
    fd = os.path.join(outdir, "figures")
    os.makedirs(fd, exist_ok=True)
    T = res["table"]
    xs = np.log2(BUDGETS)
    off = {"FR": -0.09, "FC": -0.03, "TR": 0.03, "TC": 0.09}
    # Fig 1: error vs budget, seeds visible
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    for c in STRATS:
        m = [T[f"{c}_B{B}"]["mean"] for B in BUDGETS]
        ax.plot(xs + off[c], m, color=COLORS[c], linewidth=2, marker=MARKERS[c], markersize=7, label=NAMES[c],
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        for i, B in enumerate(BUDGETS):
            ax.scatter([xs[i] + off[c]] * 3, list(T[f"{c}_B{B}"]["by_seed"].values()), s=10, color=COLORS[c], alpha=0.6, zorder=2)
    ax.set_xticks(xs, [str(B) for B in BUDGETS])
    ax.set_xlabel("calibration budget B (sequences, log scale)")
    ax.set_ylabel("mean relative expert error (old test split)")
    ax.set_title("Reconstruction error vs budget (line = mean; small dots = seeds)", loc="left")
    ax.grid(True); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"fig1_error_vs_budget.{e}"), dpi=200)
    plt.close(fig)
    # Fig 2: coverage penalty G vs budget
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for fam, c, o in (("full", "FC", -0.05), ("typical", "TC", 0.05)):
        m = [100 * res["G"][fam][str(B)]["mean"] for B in BUDGETS]
        ax.plot(xs + o, m, color=COLORS[c], linewidth=2, marker=MARKERS[c], markersize=7, markeredgecolor="white",
                markeredgewidth=1.5, label=f"G_{fam}: {NAMES[c]} vs {'Full' if fam == 'full' else 'Typical'} Random", zorder=3)
        for i, B in enumerate(BUDGETS):
            ax.scatter([xs[i] + o] * 3, [100 * v for v in res["G"][fam][str(B)]["by_seed"]], s=14, color=COLORS[c], alpha=0.6)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks(xs, [str(B) for B in BUDGETS])
    ax.set_xlabel("calibration budget B")
    ax.set_ylabel("Coverage − matched Random (% error)\nbelow 0 = Coverage better")
    ax.set_title("Coverage penalty G(B) (line = mean; dots = seeds)", loc="left")
    ax.grid(True); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"fig2_coverage_penalty.{e}"), dpi=200)
    plt.close(fig)
    # Fig 4: Gini reduction vs reconstruction change (per budget x seed x family)
    sel = json.load(open(os.path.join(outdir, "selected_sets.json")))["stats"]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    bm = {16: "o", 32: "s", 64: "^", 128: "D"}
    for fam, cov, rnd in (("full", "FC", "FR"), ("typical", "TC", "TR")):
        for B in BUDGETS:
            dg = [sel[f"{cov}_B{B}_s{s}"]["overall"]["gini_mean_over_layers"] - sel[f"{rnd}_B{B}_s{s}"]["overall"]["gini_mean_over_layers"]
                  for s in SEEDS]
            ax.scatter(dg, [100 * v for v in res["G"][fam][str(B)]["by_seed"]], s=36, marker=bm[B], color=COLORS[cov],
                       edgecolor="white", linewidth=1.2, label=f"{fam}, B={B}", zorder=3)
    ax.axhline(0, color="#555555", linewidth=1); ax.axvline(0, color="#555555", linewidth=1)
    ax.set_xlabel("Δ expert-token Gini (Coverage − matched Random; < 0 = more balanced)")
    ax.set_ylabel("G: Δ error (%), < 0 = Coverage better")
    ax.set_title("Exposure balance vs reconstruction (each point = budget × seed)", loc="left")
    ax.grid(True); ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"fig4_balance_vs_recon.{e}"), dpi=200)
    plt.close(fig)
    # Fig 5: routing atypicality vs budget
    M = res["mechanism"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for c in STRATS:
        axes[0].plot(xs + off[c], [M[f"{c}_B{B}"]["d_post_warm"] for B in BUDGETS], color=COLORS[c], marker=MARKERS[c],
                     linewidth=2, markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=NAMES[c])
        axes[1].plot(xs + off[c], [M[f"{c}_B{B}"]["set_js"] for B in BUDGETS], color=COLORS[c], marker=MARKERS[c],
                     linewidth=2, markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=NAMES[c])
    for ax, t, yl in ((axes[0], "Per-sequence atypicality of post-warm picks", "mean d(s)"),
                      (axes[1], "Selected-set routing JS to pool", "JS divergence (nats)")):
        ax.set_xticks(xs, [str(B) for B in BUDGETS]); ax.set_xlabel("calibration budget B")
        ax.set_title(t, loc="left"); ax.set_ylabel(yl); ax.grid(True); ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"fig5_atypicality_vs_budget.{e}"), dpi=200)
    plt.close(fig)


def write_md(res, path):
    L = []
    w = L.append
    w("# Low-budget calibration allocation sweep — summary\n")
    w("Mean relative expert reconstruction error, 240 routed experts (layers 1/8/16/23), frozen 4-bit GPTQ "
      "(g128, percdamp 1.0, act-order), non-sequential partial-model expert GPTQ. Evaluation: the OLD test split "
      "(rows 2128–2255; previously inspected, so exploratory). Lower is better. B = 128 = the reused matched runs.\n")
    w(f"**Decision (prespecified rule): {res['decision']['label']}**\n")
    if res["decision"]["label"] == "RUN-BLOCKED":
        w(json.dumps(res["decision"]["reason"]))
        open(path, "w").write("\n".join(L) + "\n")
        return
    T = res["table"]
    w(f"Valid runs: {res['n_valid']} / {res['n_expected']} (one evaluation universe of 240 experts).\n")
    w("## Reconstruction by budget and strategy\n")
    w("| B | strategy | mean | SD | s0 | s1 | s2 | expert median | p95 | " + " | ".join(f"L{l}" for l in res["layers"]) + " | RTN fallbacks |")
    w("|---" * (10 + len(res["layers"])) + "|")
    for B in BUDGETS:
        for c in STRATS:
            t = T[f"{c}_B{B}"]
            w(f"| {B} | {NAMES[c]} | {t['mean']:.6f} | {t['sd']:.6f} | " + " | ".join(f"{v:.6f}" for v in t["by_seed"].values()) +
              f" | {t['expert_median']:.6f} | {t['p95']:.6f} | " + " | ".join(f"{v:.6f}" for v in t["per_layer"].values()) +
              f" | {t['n_rtn_fallback']} |")
    w("\n## Paired Coverage vs matched Random: G(B) (< 0 = Coverage better)\n")
    w("| B | G_full s0 / s1 / s2 | mean ± SD | seeds < 0 | G_typical s0 / s1 / s2 | mean ± SD | seeds < 0 |")
    w("|---|---|---|---|---|---|---|")
    for B in BUDGETS:
        f, t = res["G"]["full"][str(B)], res["G"]["typical"][str(B)]
        w(f"| {B} | " + " / ".join(pct(v) for v in f["by_seed"]) + f" | {pct(f['mean'])} ± {100 * f['sd']:.2f} | {f['n_negative']}/3 | " +
          " / ".join(pct(v) for v in t["by_seed"]) + f" | {pct(t['mean'])} ± {100 * t['sd']:.2f} | {t['n_negative']}/3 |")
    w("\nPer-layer mean relative difference and expert-level detail:\n")
    w("| B | family | L1 | L8 | L16 | L23 | p95 rel | experts where Coverage better |")
    w("|---|---|---|---|---|---|---|---|")
    for B in BUDGETS:
        for fam in ("full", "typical"):
            pl = res["G_per_layer_rel"][fam][str(B)]
            w(f"| {B} | {fam} | " + " | ".join(pct(v) for v in pl.values()) + f" | {pct(res['G_p95_rel'][fam][str(B)])} | "
              f"{100 * res['G_expert'][fam][str(B)]['frac_experts_improved']:.1f} % |")
    w("\n## Diminishing returns R(B) = [E(B) − E(2B)] / E(B) (mean over seeds)\n")
    w("| strategy | 16→32 | 32→64 | 64→128 |")
    w("|---|---|---|---|")
    for c in STRATS:
        w(f"| {NAMES[c]} | " + " | ".join(pct(res["R"][c][k]["mean"]) for k in ("16->32", "32->64", "64->128")) + " |")
    w("\n## Mechanism (selection side, mean over seeds)\n")
    w("| B | strategy | Gini | min tokens | p10 tokens | experts < 256 | eff. sources | post-warm d(s) | set JS | aggregate L1 (L1/L8/L16/L23) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for B in BUDGETS:
        for c in STRATS:
            m = res["mechanism"][f"{c}_B{B}"]
            w(f"| {B} | {NAMES[c]} | {m['gini']:.4f} | {m['min_tokens']:.0f} | {m['p10_tokens']:.0f} | {m['below_256']:.1f} | "
              f"{m['eff_sources']:.1f} | {m['d_post_warm']:.4f} | {m['set_js']:.5f} | " +
              " / ".join(f"{v:.4f}" for v in m["aggregate_l1"].values()) + " |")
    w("\n## Cost (mean per run)\n")
    w("| B | strategy | selection s | Hessian replay s | GPTQ s | calibration forward-equiv s | routing-profile forward-equiv s | peak GPU GB |")
    w("|---|---|---|---|---|---|---|---|")
    for B in BUDGETS:
        for c in STRATS:
            k = res["cost"][f"{c}_B{B}"]
            prof = 357 if c != "FR" else 0
            w(f"| {B} | {NAMES[c]} | {k['selection_s']:.2f} | {k['hessian_s']:.1f} | {k['gptq_s']:.1f} | "
              f"{k['calib_forward_equiv_s']:.1f} | {prof} | {k['peak_gpu_mem_gb']:.1f} |")
    w("\nTypical Random also needs the ~357 s routing profile (to build the typical half) even though its run JSON "
      "charges 0; only Full Random avoids it. At B = 16 the profile (≈ 2,000 sequence forwards) is ~125× the "
      "calibration forwards themselves.\n")
    w("Figures: `figures/fig1_error_vs_budget`, `fig2_coverage_penalty`, `fig3_exposure_scarcity`, "
      "`fig4_balance_vs_recon`, `fig5_atypicality_vs_budget` (.pdf/.png).")
    open(path, "w").write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs_low_budget")
    ap.add_argument("--matched-dir", default="runs_typicality_matched")
    ap.add_argument("--pool-dir", default="runs_typicality")
    ap.add_argument("--frozen", default="configs/frozen_b4.json")
    a = ap.parse_args(argv)
    res = build(a)
    json.dump(res, open(os.path.join(a.outdir, "summary.json"), "w"), indent=1)
    if res["decision"]["label"] != "RUN-BLOCKED":
        figures(res, a.outdir)
    write_md(res, os.path.join(a.outdir, "summary.md"))
    print(open(os.path.join(a.outdir, "summary.md")).read())


if __name__ == "__main__":
    main()
