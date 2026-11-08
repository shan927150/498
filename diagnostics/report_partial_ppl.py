"""Summary, decision and figures for runs_partial_ppl (partial-model WikiText-2 PPL).

The decision rule is the one frozen in evaluation_protocol.json before any
quantized evaluation (eval_partial_ppl.DECISION_RULE)."""
from __future__ import annotations
import json, math, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import eval_partial_ppl as EP      # noqa: E402

C4 = ("FR", "TR", "FC", "TC")
COL = {"FR": "#2a78d6", "FC": "#eda100", "TR": "#eb6834", "TC": "#1baf7a"}
PAIRS = [("FC", "FR"), ("TC", "TR"), ("TR", "FR"), ("FC", "TR")]
N_BOOT, BOOT_SEED = 2000, 20261108


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):                          # average ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def load():
    P = lambda *a: os.path.join(ROOT, EP.OUT, *a)
    prot = json.load(open(P("evaluation_protocol.json")))
    man = json.load(open(P("selected_set_manifest.json")))
    base = json.load(open(P("bf16_baseline.json"))) if os.path.exists(P("bf16_baseline.json")) else None
    ref = json.load(open(P("reference_check.json"))) if os.path.exists(P("reference_check.json")) else None
    res, missing = {}, []
    for c in C4:
        for s in EP.SEEDS:
            if EP.valid_result(c, s, man, prot):
                res[(c, s)] = json.load(open(EP.result_path(c, s)))
            else:
                missing.append(f"{c}_s{s}")
    rep = json.load(open(EP.result_path("FR", 0, "_repeat"))) if EP.valid_result("FR", 0, man, prot, "_repeat") else None
    logs = {n: json.load(open(P("logs", n))) for n in ("quant_process_smoke.json", "quant_process_main.json")
            if os.path.exists(P("logs", n))}
    return prot, man, base, ref, res, missing, rep, logs


def boot_ci(a, b):
    """Window-bootstrap 95 % CI of PPL(a) - PPL(b) on the SAME windows (descriptive:
    benchmark-sampling uncertainty for one calibration set, not seed variance)."""
    g = np.random.default_rng(BOOT_SEED)
    na, nb = np.array(a["per_window_nll"]), np.array(b["per_window_nll"])
    t = np.array(a["per_window_tokens"])
    idx = g.integers(0, len(na), size=(N_BOOT, len(na)))
    d = np.exp(na[idx].sum(1) / t[idx].sum(1)) - np.exp(nb[idx].sum(1) / t[idx].sum(1))
    return [float(np.quantile(d, .025)), float(np.quantile(d, .975))]


def build():
    prot, man, base, ref, res, missing, rep, logs = load()
    out = dict(missing=missing, n_valid=len(res), n_expected=12, model_scope=prot["model_scope"])
    if base is None or missing:
        out["decision"] = "PPL-RUN-BLOCKED"
        return out, prot, man, base, ref, res, rep, logs
    bppl = base["ppl"]
    tab = {}
    for c in C4:
        v = [res[(c, s)]["ppl"] for s in EP.SEEDS]
        tab[c] = dict(by_seed=v, mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)),
                      abs_increase=float(np.mean(v)) - bppl, rel_degradation=(float(np.mean(v)) - bppl) / bppl,
                      rel_by_seed=[(x - bppl) / bppl for x in v],
                      recon_test=[man["configs"][f"{c}_s{s}"]["recon_test"] for s in EP.SEEDS],
                      recon_confirmation=[man["configs"][f"{c}_s{s}"]["recon_confirmation"] for s in EP.SEEDS])
    paired = {}
    for a, b in PAIRS:
        d = [res[(a, s)]["ppl"] - res[(b, s)]["ppl"] for s in EP.SEEDS]
        dr = [res[(a, s)]["recon_recomputed"] - res[(b, s)]["recon_recomputed"] for s in EP.SEEDS]
        paired[f"{a}-{b}"] = dict(delta_ppl=d, mean=float(np.mean(d)),
                                  rel=[x / res[(b, s)]["ppl"] for x, s in zip(d, EP.SEEDS)],
                                  delta_share_of_degradation=[x / (res[(b, s)]["ppl"] - bppl) for x, s in zip(d, EP.SEEDS)],
                                  boot_ci=[boot_ci(res[(a, s)], res[(b, s)]) for s in EP.SEEDS],
                                  delta_recon_test=dr,
                                  delta_recon_rel=[x / res[(b, s)]["recon_recomputed"] for x, s in zip(dr, EP.SEEDS)])
    keys = [(c, s) for c in C4 for s in EP.SEEDS]
    rt = [res[k]["recon_recomputed"] for k in keys]
    rc = [man["configs"][f"{c}_s{s}"]["recon_confirmation"] for c, s in keys]
    pp = [res[k]["ppl"] for k in keys]
    corr = dict(spearman_test=spearman(rt, pp), pearson_test=pearson(rt, pp),
                spearman_confirmation=spearman(rc, pp), pearson_confirmation=pearson(rc, pp),
                ppl_rank_of_means=sorted(C4, key=lambda c: tab[c]["mean"]),
                recon_test_rank_of_means=sorted(C4, key=lambda c: np.mean(tab[c]["recon_test"])),
                recon_confirmation_rank_of_means=sorted(C4, key=lambda c: np.mean(tab[c]["recon_confirmation"])))
    # numerical repeatability
    rel = dict(bf16_repeat_abs=base["repeat_abs_diff_ppl"],
               bf16_cross_process_abs=max([l.get("bf16_cross_process_abs_diff", 0.0) for l in logs.values()] or [0.0]),
               bf16_end_absdiff=logs.get("quant_process_main.json", {}).get("bf16_end_abs_diff"),
               quant_repeat_abs=abs(rep["ppl"] - res[("FR", 0)]["ppl"]) if rep else None,
               quant_repeat_identical_weights=(rep["installed_weights_sha256"] == res[("FR", 0)]["installed_weights_sha256"])
               if rep else None)
    rep_max = max(v for k in ("bf16_repeat_abs", "bf16_cross_process_abs", "quant_repeat_abs") if (v := rel[k]) is not None)
    tau = max(3 * rep_max, 1e-4 * bppl)
    rel.update(max_numerical_abs=rep_max, tau=tau)
    # decision (frozen rule)
    d_fc = paired["FC-FR"]["delta_ppl"]
    fc_worst = corr["ppl_rank_of_means"][-1] == "FC"
    if all(x > tau for x in d_fc) and fc_worst and corr["spearman_test"] >= 0.5:
        dec = "PPL-RECON-ALIGNED"
    elif all(x < -tau for x in d_fc) or corr["spearman_test"] <= -0.5:
        dec = "PPL-RECON-DISAGREE"
    else:
        dec = "PPL-RECON-WEAKLY-RELATED"
    checks = dict(
        max_recon_rel_diff=max(r["recon_max_rel_diff"] for r in res.values()),
        max_installed_vs_e3_ratio=max(ck["ratio_to_quant_effect"] for r in res.values() for ck in r["expert_checks"]),
        max_installed_over_tolerance=max(ck["installed_vs_e3_rel"] / EP.INSTALL_TOL(ck)
                                         for r in res.values() for ck in r["expert_checks"]),
        max_installed_vs_e3_rel=max(ck["installed_vs_e3_rel"] for r in res.values() for ck in r["expert_checks"]),
        median_quant_effect_rel=float(np.median([ck["quant_effect_rel"] for r in res.values() for ck in r["expert_checks"]])),
        max_weight_rounding_rel=max(ck["weight_rounding_only_rel"] for r in res.values() for ck in r["expert_checks"]),
        max_bf16_compute_noise_rel=max(ck["bf16_compute_noise_rel"] for r in res.values() for ck in r["expert_checks"]),
        all_weights_exact=all(ck["installed_equals_bf16_of_e3_wq"] for r in res.values() for ck in r["expert_checks"]),
        untouched_changed=sum(r["untouched_params_changed"] for r in res.values()),
        n_untouched_params=res[("FR", 0)]["n_untouched_params_checked"],
        rtn_fallbacks=sum(r["rtn_fallbacks"] for r in res.values()),
        distinct_installed_weights=len({r["installed_weights_sha256"] for r in res.values()}))
    cost = {c: dict(hessian_s=float(np.mean([res[(c, s)]["hessian_s"] for s in EP.SEEDS])),
                    gptq_s=float(np.mean([res[(c, s)]["gptq_s"] for s in EP.SEEDS])),
                    quantize_wall_s=float(np.mean([res[(c, s)]["quantize_wall_s"] for s in EP.SEEDS])),
                    ppl_eval_s=float(np.mean([res[(c, s)]["eval_s"] for s in EP.SEEDS])),
                    peak_gpu_gb=max(res[(c, s)]["peak_gpu_gb"] for s in EP.SEEDS),
                    peak_cpu_gb=max(res[(c, s)]["peak_cpu_gb"] for s in EP.SEEDS),
                    routing_profile_s=0.0 if c == "FR" else 357.0) for c in C4}
    out.update(bf16_ppl=bppl, table=tab, paired=paired, correlation=corr, reliability=rel, decision=dec,
               install_checks=checks, cost=cost, bf16=dict(model_load_s=base["model_load_s"],
                                                           eval_s=base["runs"][0]["eval_s"],
                                                           peak_gpu_gb=base["runs"][0]["peak_gpu_gb"],
                                                           peak_cpu_gb=base["peak_cpu_gb"]),
               reference=ref, packed_artifact_bytes=res[("FR", 0)]["quant_artifact_bytes_if_packed"],
               process_logs=logs)
    return out, prot, man, base, ref, res, rep, logs


def figures(out, res, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e4e4", "grid.linewidth": 0.6, "axes.axisbelow": True})
    sv = lambda f, n: (f.savefig(os.path.join(d, n + ".pdf"), bbox_inches="tight"),
                       f.savefig(os.path.join(d, n + ".png"), dpi=200, bbox_inches="tight"))
    T, b = out["table"], out["bf16_ppl"]
    # fig1 PPL by strategy
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for i, c in enumerate(C4):
        ax.scatter([i] * 3, T[c]["by_seed"], color=COL[c], s=30, edgecolor="white", zorder=3)
        ax.plot([i - .22, i + .22], [T[c]["mean"]] * 2, color=COL[c], lw=2.5)
        ax.text(i + .26, T[c]["mean"], f"{T[c]['mean']:.4f}", va="center", fontsize=8)
    ax.axhline(b, color="#555", lw=1, ls="--")
    ax.text(3.45, b, f"BF16 {b:.4f}", va="bottom", ha="right", fontsize=8, color="#555")
    ax.set_xticks(range(4), [EP.NAMES[c] for c in C4], fontsize=8)
    ax.set_xlim(-.5, 3.6)
    ax.set_ylabel("WikiText-2 PPL (128 × 512 windows)")
    ax.set_title("Partial-model PPL (layers 1/8/16/23 routed experts GPTQ-4bit; rest BF16)", fontsize=9)
    sv(fig, "fig1_ppl_by_strategy"); plt.close(fig)
    # fig2 degradation
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for i, c in enumerate(C4):
        absd = [x - b for x in T[c]["by_seed"]]
        axes[0].bar(i, np.mean(absd), color=COL[c], width=.6)
        axes[0].scatter([i] * 3, absd, color="#333", s=10, zorder=3)
        axes[1].bar(i, 100 * T[c]["rel_degradation"], color=COL[c], width=.6)
        axes[1].scatter([i] * 3, [100 * x for x in T[c]["rel_by_seed"]], color="#333", s=10, zorder=3)
    for ax, t in zip(axes, ("PPL − PPL(BF16)", "relative degradation (%)")):
        ax.set_xticks(range(4), [EP.NAMES[c].replace(" ", "\n", 1) for c in C4], fontsize=8); ax.set_title(t, fontsize=9)
    fig.suptitle("Degradation from BF16 (bars = mean, dots = seeds); partial model", y=1.02, fontsize=9)
    sv(fig, "fig2_ppl_degradation"); plt.close(fig)
    # fig3 recon vs PPL
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, key, t in ((axes[0], "recon_test", "old test split"),
                       (axes[1], "recon_confirmation", "confirmation split")):
        for c in C4:
            ax.scatter(T[c][key], T[c]["by_seed"], color=COL[c], s=34, edgecolor="white", label=EP.NAMES[c])
        ax.set_xlabel(f"expert recon. error ×1e-3 ({t})")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 1e3:.2f}"))
        r = out["correlation"]["spearman_test" if key == "recon_test" else "spearman_confirmation"]
        ax.set_title(f"Spearman ρ = {r:+.2f} over 12 configurations (descriptive)", fontsize=9)
    axes[0].set_ylabel("partial-model WikiText-2 PPL")
    axes[0].legend(frameon=False, fontsize=7)
    sv(fig, "fig3_recon_vs_ppl"); plt.close(fig)
    # fig4 paired differences
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    tau = out["reliability"]["tau"]
    for i, k in enumerate(("FC-FR", "TC-TR", "TR-FR", "FC-TR")):
        P = out["paired"][k]
        for s, (v, ci) in enumerate(zip(P["delta_ppl"], P["boot_ci"])):
            x = i + (s - 1) * 0.15
            ax.errorbar(x, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="o", color=COL[k.split("-")[0]], ms=5, capsize=2,
                        lw=1)
        ax.plot([i - .25, i + .25], [P["mean"]] * 2, color="#333", lw=1.5)
    ax.axhline(0, color="#555", lw=0.9)
    ax.axhspan(-tau, tau, color="#bbb", alpha=0.35, lw=0, label=f"±τ numerical repeatability ({tau:.4f})")
    ax.set_xticks(range(4), ["Full Cov − Full Rnd", "Typ Cov − Typ Rnd", "Typ Rnd − Full Rnd", "Full Cov − Typ Rnd"],
                  fontsize=8)
    ax.set_ylabel("paired Δ PPL (> 0: first is worse)")
    ax.set_title("Paired PPL differences per seed (bars: 95 % window bootstrap; line = mean)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    sv(fig, "fig4_paired_differences"); plt.close(fig)


def write_md(out, res, rep, path):
    L = []; w = L.append
    w("# Partial-model WikiText-2 perplexity — summary\n")
    w("**PARTIAL model:** only the 240 routed experts (gate/up/down) of layers 1/8/16/23 are GPTQ-quantized "
      "(4-bit, g128, percdamp 1.0, act-order; fake-quantized weights in BF16); attention, routers, shared experts, "
      "embeddings, norms, lm_head and the other 20 layers are BF16. Not a fully 4-bit model.\n")
    w(f"**Decision (rule frozen in evaluation_protocol.json): {out['decision']}**\n")
    w(f"Valid evaluations: {out['n_valid']} / {out['n_expected']}" + (f"; missing {out['missing']}" if out["missing"] else "") + "\n")
    if out["decision"] == "PPL-RUN-BLOCKED":
        open(path, "w").write("\n".join(L) + "\n"); return
    b = out["bf16_ppl"]
    w(f"BF16 PPL: **{b:.4f}** (128 windows × 511 scored targets = 65,408 targets)\n")
    w("## PPL by strategy and seed\n")
    w("| strategy | s0 | s1 | s2 | mean | SD | Δ vs BF16 | rel. degradation | recon (test) mean | recon (confirmation) mean |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for c in C4:
        t = out["table"][c]
        w(f"| {EP.NAMES[c]} | " + " | ".join(f"{x:.4f}" for x in t["by_seed"]) +
          f" | {t['mean']:.4f} | {t['sd']:.4f} | +{t['abs_increase']:.4f} | +{100 * t['rel_degradation']:.3f} % | "
          f"{np.mean(t['recon_test']):.6f} | {np.mean(t['recon_confirmation']):.6f} |")
    w("\n## Paired differences (same seed; > 0 = first method worse)\n")
    w("| pair | ΔPPL s0 / s1 / s2 | mean | rel. ΔPPL | share of the baseline's degradation | 95 % window bootstrap per seed | Δrecon (rel.) s0 / s1 / s2 |")
    w("|---|---|---|---|---|---|---|")
    for k, P in out["paired"].items():
        a, bb = k.split("-")
        w(f"| {EP.NAMES[a]} − {EP.NAMES[bb]} | " + " / ".join(f"{x:+.4f}" for x in P["delta_ppl"]) +
          f" | {P['mean']:+.4f} | " + " / ".join(f"{100 * x:+.3f} %" for x in P["rel"]) + " | " +
          " / ".join(f"{100 * x:+.1f} %" for x in P["delta_share_of_degradation"]) + " | " +
          "; ".join(f"[{c[0]:+.4f}, {c[1]:+.4f}]" for c in P["boot_ci"]) + " | " +
          " / ".join(f"{100 * x:+.2f} %" for x in P["delta_recon_rel"]) + " |")
    C = out["correlation"]
    w("\n## Reconstruction vs PPL (12 method × seed configurations; descriptive, not independent replicates)\n")
    w(f"- Spearman / Pearson with old-test reconstruction: {C['spearman_test']:+.3f} / {C['pearson_test']:+.3f}")
    w(f"- Spearman / Pearson with locked-confirmation reconstruction: {C['spearman_confirmation']:+.3f} / {C['pearson_confirmation']:+.3f}")
    w(f"- Ranking of means (best → worst): PPL {C['ppl_rank_of_means']}; recon test {C['recon_test_rank_of_means']}; "
      f"recon confirmation {C['recon_confirmation_rank_of_means']}")
    R = out["reliability"]
    w("\n## Numerical reliability\n")
    w(f"- BF16 repeated in one process: |ΔPPL| = {R['bf16_repeat_abs']:.2e}; across processes: {R['bf16_cross_process_abs']:.2e}; "
      f"after all 13 quantize/restore cycles: {R['bf16_end_absdiff']}")
    w(f"- Full Random s0 re-quantized and re-evaluated: |ΔPPL| = {R['quant_repeat_abs']}; identical installed weights: "
      f"{R['quant_repeat_identical_weights']}")
    w(f"- τ = max(3 × max numerical difference, 1e-4 × PPL_BF16) = {R['tau']:.5f}")
    if out.get("reference"):
        for r in out["reference"]["windows"]:
            w(f"- Independent HF forward (accelerate offload, labels=) window {r['window']}: mean NLL {r['hf_loss_mean_nll']:.6f} "
              f"vs layer-wise {r['layerwise_mean_nll']:.6f} (rel. diff {r['rel_diff_layerwise_vs_hf']:.1e})")
    I = out["install_checks"]
    w("\n## Installation checks (all 12 configurations)\n")
    w(f"- E3 per-expert reconstruction recomputed from the installed run vs the frozen run: max rel. diff {I['max_recon_rel_diff']:.1e}")
    w(f"- 8 check experts × 12: installed BF16 weights bitwise = BF16 cast of E3's W_q: {I['all_weights_exact']}; "
      f"installed-module output vs E3 FP32 output: max rel. {I['max_installed_vs_e3_rel']:.2e} = at most "
      f"{100 * I['max_installed_vs_e3_ratio']:.1f} % of the quantization perturbation itself (median {I['median_quant_effect_rel']:.3f}); "
      f"of which weight BF16 rounding alone ≤ {I['max_weight_rounding_rel']:.1e}; BF16-compute noise with original weights ≤ "
      f"{I['max_bf16_compute_noise_rel']:.1e}; largest deviation / amended tolerance = {I['max_installed_over_tolerance']:.2f} "
      "(tolerance = weight-rounding deviation + 1.5 × BF16-compute deviation; amended after smoke attempt 1, before any "
      "quantized PPL — logs/amendment_install_tolerance.json)")
    w(f"- Untouched parameters/buffers changed: {I['untouched_changed']} (of {I['n_untouched_params']} checked per run); "
      f"RTN fallbacks: {I['rtn_fallbacks']}; distinct installed weight sets: {I['distinct_installed_weights']} / 12")
    w("\n## Compute (mean per run)\n")
    w("| strategy | Hessian replay s | GPTQ s | quantize wall s (incl. E3 scoring and checks) | PPL eval s | peak GPU GB | peak CPU GB (process) | routing profile s |")
    w("|---|---|---|---|---|---|---|---|")
    for c in C4:
        k = out["cost"][c]
        w(f"| {EP.NAMES[c]} | {k['hessian_s']:.0f} | {k['gptq_s']:.0f} | {k['quantize_wall_s']:.0f} | {k['ppl_eval_s']:.0f} | "
          f"{k['peak_gpu_gb']:.1f} | {k['peak_cpu_gb']:.1f} | {k['routing_profile_s']:.0f} |")
    bf = out["bf16"]
    w(f"\nBF16: model load {bf['model_load_s']:.0f} s, PPL eval {bf['eval_s']:.0f} s, peak GPU {bf['peak_gpu_gb']:.1f} GB, "
      f"process peak CPU {bf['peak_cpu_gb']:.1f} GB. Packed 4-bit artifact for 240 experts would be ≈ "
      f"{out['packed_artifact_bytes'] / 1e9:.2f} GB (not produced; weights are fake-quantized).\n")
    w("Figures: `figures/fig1_ppl_by_strategy`, `fig2_ppl_degradation`, `fig3_recon_vs_ppl`, `fig4_paired_differences` (PDF + PNG).")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    out, prot, man, base, ref, res, rep, logs = build()
    d = os.path.join(ROOT, EP.OUT)
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
    write_md(out, res, rep, os.path.join(d, "summary.md"))
    if out["decision"] != "PPL-RUN-BLOCKED":
        figures(out, res, os.path.join(d, "figures"))
    print(f"decision: {out['decision']}")


if __name__ == "__main__":
    main()
