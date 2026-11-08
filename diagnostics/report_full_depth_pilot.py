"""Summary, decision (frozen in protocol.json) and figures for the full-depth pilot."""
from __future__ import annotations
import json, math, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import eval_partial_ppl as EP         # noqa: E402
import full_depth_pilot as FD         # noqa: E402

COL = {"FR": "#2a78d6", "FC": "#eda100"}
NAMES = {"FR": "Full Random s0", "FC": "Full Coverage-LP s0"}


def build():
    P = FD.p
    proto = json.load(open(P("protocol.json")))
    reg = json.load(open(P("regression_checks.json"))) if os.path.exists(P("regression_checks.json")) else None
    cman = json.load(open(P("cache_manifest.json"))) if os.path.exists(P("cache_manifest.json")) else None
    base = json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))
    part = {c: json.load(open(os.path.join(ROOT, EP.OUT, f"ppl_{c}_s0.json"))) for c in FD.STRATS}
    full = {c: json.load(open(P(f"{n}.json"))) for c, n in FD.STRATS.items() if os.path.exists(P(f"{n}.json"))}
    out = dict(proto_decision_rule=proto["decision_rule"], n_complete=len(full), n_expected=2,
               regression_passed=bool(reg and reg["passed"]))
    if len(full) < 2 or not out["regression_passed"]:
        out["decision"] = "RUN-BLOCKED"
        return out, proto, reg, cman, base, part, full
    b = base["ppl"]
    bn = base["runs"][0]["mean_nll"]
    rows = {}
    for c in FD.STRATS:
        f, q = full[c], part[c]
        rows[c] = dict(ppl4=q["ppl"], ppl24=f["ppl"], nll4=q["mean_nll"], nll24=f["mean_nll"],
                       deg4=(q["ppl"] - b) / b, deg24=(f["ppl"] - b) / b, abs4=q["ppl"] - b, abs24=f["ppl"] - b,
                       recon4=q["recon_recomputed"], recon24_mean=float(np.mean(list(f["recon_by_layer"].values()))))
    d4 = rows["FC"]["ppl4"] - rows["FR"]["ppl4"]
    d24 = rows["FC"]["ppl24"] - rows["FR"]["ppl24"]
    bs24 = FD.paired_bootstrap(full["FC"]["per_window_nll"], full["FR"]["per_window_nll"], full["FR"]["per_window_tokens"])
    bs4 = FD.paired_bootstrap(part["FC"]["per_window_nll"], part["FR"]["per_window_nll"], part["FR"]["per_window_tokens"])
    finite = all(f["finite_nll"] and f["nonfinite_weights"] == 0 for f in full.values())
    dec = FD.decide(b, rows["FR"]["ppl24"], rows["FC"]["ppl24"], d4, bs24["ci95"], finite)
    t = np.array(full["FR"]["per_window_tokens"], float)
    wdiff = (np.array(full["FC"]["per_window_nll"]) - np.array(full["FR"]["per_window_nll"])) / t
    wdiff4 = (np.array(part["FC"]["per_window_nll"]) - np.array(part["FR"]["per_window_nll"])) / t
    rl = {c: full[c]["recon_by_layer"] for c in FD.STRATS}
    layer_rel = {L: (rl["FC"][L] - rl["FR"][L]) / rl["FR"][L] for L in rl["FR"]}
    v1p = P("logs", "regression_attempt1_FAILED_v1cache.json")
    sens = None
    if os.path.exists(v1p):
        v1 = json.load(open(v1p))
        sens = dict(note="four-layer GPTQ re-run with BF16 calibration inputs that differ only by batch-composition "
                         "rounding (v1 cache); same selections, solver and settings",
                    ppl_original={c: v1["configs"][c]["original_cache"]["ppl"] for c in FD.STRATS},
                    ppl_perturbed={c: v1["configs"][c]["pilot_cache"]["ppl"] for c in FD.STRATS},
                    recon_original={c: v1["configs"][c]["original_cache"]["recon"] for c in FD.STRATS},
                    recon_perturbed={c: v1["configs"][c]["pilot_cache"]["recon"] for c in FD.STRATS},
                    delta4_original=v1["delta4_reference"], delta4_perturbed=v1["delta4_pilot_cache"])
        sens["max_abs_ppl_shift"] = max(abs(sens["ppl_perturbed"][c] - sens["ppl_original"][c]) for c in FD.STRATS)
    out.update(bf16_ppl=b, bf16_mean_nll=bn, rows=rows, delta4=d4, delta24=d24, delta_change=d24 - d4,
               calibration_numerics_sensitivity=sens,
               delta_ratio=d24 / d4 if d4 else None, delta4_rel=d4 / rows["FR"]["ppl4"], delta24_rel=d24 / rows["FR"]["ppl24"],
               delta_nll4=rows["FC"]["nll4"] - rows["FR"]["nll4"], delta_nll24=rows["FC"]["nll24"] - rows["FR"]["nll24"],
               bootstrap24=bs24, bootstrap4=bs4, tau=1e-4 * b, decision=dec,
               window_diff=dict(fc_worse_windows=int((wdiff > 0).sum()), n=len(wdiff), mean=float(wdiff.mean()),
                                median=float(np.median(wdiff)), q05=float(np.quantile(wdiff, .05)), q95=float(np.quantile(wdiff, .95)),
                                corr_with_4layer=float(np.corrcoef(wdiff, wdiff4)[0, 1])),
               per_window_nll_diff=wdiff.tolist(), per_window_nll_diff4=wdiff4.tolist(),
               recon_by_layer=rl, recon_rel_diff_by_layer=layer_rel,
               recon_fc_worse_layers=int(sum(v > 0 for v in layer_rel.values())),
               correctness={c: dict(layers=f["n_quantized_layers"], experts=f["n_quantized_experts"],
                                    projections=f["n_quantized_projections"],
                                    unchanged=f["projections_unchanged_from_bf16"], nonfinite=f["nonfinite_weights"],
                                    untouched_ok=f["untouched_params_unchanged"], rtn=f["rtn_fallbacks"],
                                    finite_nll=f["finite_nll"], repeat=f["repeat_eval_abs_diff_ppl"],
                                    sha=f["installed_weights_sha256"],
                                    checks_pass=all(ck["pass"] for ck in f["expert_checks"]),
                                    max_dev_over_tol=max(ck["installed_vs_e3_rel"] / ck["tolerance"] for ck in f["expert_checks"]))
                            for c, f in full.items()},
               cost={c: dict(quantize_min=f["quantize_total_s"] / 60, hessian_min=f["hessian_total_s"] / 60,
                             gptq_min=f["gptq_total_s"] / 60, eval_s=f["eval_s"], load_s=f["model_load_s"],
                             verify_s=f["pristine_verify_s"], peak_gpu_gb=f["peak_gpu_gb"], peak_cpu_gb=f["peak_cpu_gb"],
                             artifacts_gb=f["weight_artifacts_gb"], resumed=f["resumed_layers"]) for c, f in full.items()},
               cache=dict(seconds=cman["seconds"], disk_gb=cman["disk_gb"], rows=cman["n_rows"],
                          comparison=cman["comparison_with_original_cache"]) if cman else None,
               regression=dict((c, {k: {kk: v[kk] for kk in ("ppl", "ppl_partial_reference", "abs_diff_ppl",
                                                              "digest_equals_partial_ppl", "recon", "recon_reference")}
                                    for k, v in reg["configs"][c].items()}) for c in FD.STRATS) | dict(
                   delta4_pilot_cache=reg["delta4_pilot_cache"], original_cache_exact=reg["original_cache_exact"],
                   pilot_cache_ok=reg["pilot_cache_ok"]))
    return out, proto, reg, cman, base, part, full


def figures(o, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e4e4", "grid.linewidth": 0.6, "axes.axisbelow": True})
    sv = lambda f, n: (f.savefig(os.path.join(d, n + ".pdf"), bbox_inches="tight"),
                       f.savefig(os.path.join(d, n + ".png"), dpi=200, bbox_inches="tight"))
    R, b = o["rows"], o["bf16_ppl"]
    # fig1
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xs = [0, 4, 24]
    for c in ("FR", "FC"):
        ys = [b, R[c]["ppl4"], R[c]["ppl24"]]
        ax.plot(xs, ys, "-o", color=COL[c], lw=2, ms=6, label=NAMES[c])
        for x, y in zip(xs[1:], ys[1:]):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(6, -12 if c == "FR" else 6), fontsize=7.5)
    ax.set_xticks(xs, ["BF16\n(0 layers)", "4 layers\n(1/8/16/23)", "24 layers\n(all)"])
    ax.set_ylabel("WikiText-2 PPL (128 × 512 windows)")
    ax.set_title("Measured PPL by number of routed-expert layers quantized (seed 0)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    sv(fig, "fig1_ppl_by_depth"); plt.close(fig)
    # fig2 penalty
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i, (k, bs) in enumerate((("delta4", o["bootstrap4"]), ("delta24", o["bootstrap24"]))):
        nll_frs = R["FR"]["nll4" if k == "delta4" else "nll24"]
        ppl_fr = R["FR"]["ppl4" if k == "delta4" else "ppl24"]
        lo = ppl_fr * (math.exp(bs["ci95"][0]) - 1)                 # PPL-scale CI from the NLL-difference CI
        hi = ppl_fr * (math.exp(bs["ci95"][1]) - 1)
        v = o[k]
        ax.bar(i, v, color="#eda100", width=.5)
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color="#333", capsize=4, lw=1.2)
        ax.text(i + .28, v, f"{v:+.4f}", va="center", fontsize=8)
    ax.axhline(0, color="#555", lw=.9)
    ax.set_xticks([0, 1], ["4 layers", "24 layers"])
    ax.set_ylabel("PPL(FC) − PPL(FR), seed 0")
    ax.set_title("Coverage penalty (bars: measured; whiskers: 95 % paired window bootstrap)", fontsize=8.5)
    sv(fig, "fig2_coverage_penalty_by_depth"); plt.close(fig)
    # fig3 window distribution
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    w24, w4 = np.array(o["per_window_nll_diff"]), np.array(o["per_window_nll_diff4"])
    lim = max(abs(w24).max(), abs(w4).max())
    bins = np.linspace(-lim, lim, 41)
    ax.hist(w4, bins=bins, color="#9ec5f0", alpha=.8, label="4 layers")
    ax.hist(w24, bins=bins, color="#eda100", alpha=.6, label="24 layers")
    ax.axvline(0, color="#555", lw=.9)
    ax.axvline(w24.mean(), color="#b06f00", lw=1.5, ls="--", label=f"24-layer mean {w24.mean():+.4f}")
    ax.set_xlabel("per-window mean-NLL difference, FC − FR (nats/token)")
    ax.set_ylabel("windows (of 128)")
    ax.set_title("Paired per-window NLL differences (measured)", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    sv(fig, "fig3_window_nll_differences"); plt.close(fig)
    # fig4 runtime / memory
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    C = o["cost"]
    for i, c in enumerate(("FR", "FC")):
        k = C[c]
        axes[0].bar(i - .2, k["hessian_min"], .4, color="#9ec5f0", label="Hessian replay" if i == 0 else None)
        axes[0].bar(i - .2, k["gptq_min"], .4, bottom=k["hessian_min"], color="#2a78d6", label="GPTQ" if i == 0 else None)
        axes[0].bar(i + .2, k["quantize_min"], .4, color="#888", label="total quantize wall" if i == 0 else None)
        axes[1].bar(i - .2, k["peak_gpu_gb"], .4, color=COL[c])
        axes[1].bar(i + .2, k["peak_cpu_gb"], .4, color=COL[c], alpha=.45)
    axes[0].set_xticks([0, 1], [NAMES["FR"], NAMES["FC"]], fontsize=8); axes[0].set_ylabel("minutes (24 layers)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xticks([-.2, .2, .8, 1.2], ["GPU FR", "CPU FR", "GPU FC", "CPU FC"], fontsize=8)
    axes[1].set_ylabel("peak GB (measured)")
    fig.suptitle("Full-depth quantization cost (measured; resumed layers excluded from timings)", fontsize=9, y=1.02)
    sv(fig, "fig4_runtime_memory"); plt.close(fig)


def write_md(o, path):
    L = []; w = L.append
    w("# Full-depth routed-expert GPTQ pilot — summary (seed 0)\n")
    w("Routed experts (gate/up/down) of **all 24 MoE layers** (1,440 experts, 4,320 projections) GPTQ-quantized "
      "(4-bit, g128, percdamp 1.0, act-order; fake-quantized weights in BF16); attention, routers, shared experts, "
      "embeddings, norms, lm_head BF16. Not fully 4-bit LLM quantization. Calibration non-sequential on BF16 "
      "inputs; evaluation a sequential 24-layer forward on the frozen 128 × 512 WikiText-2 windows.\n")
    w(f"**Decision (rule frozen in protocol.json): {o['decision']}**\n")
    if o["decision"] == "RUN-BLOCKED":
        w(f"complete {o['n_complete']}/2; regression passed {o['regression_passed']}")
        open(path, "w").write("\n".join(L) + "\n"); return
    R, b = o["rows"], o["bf16_ppl"]
    w("| configuration | PPL | mean NLL | Δ vs BF16 | rel. degradation | mean expert recon |")
    w("|---|---|---|---|---|---|")
    w(f"| BF16 | {b:.6f} | {o['bf16_mean_nll']:.6f} | — | — | — |")
    for k, lab in (("4", "4 layers"), ("24", "24 layers")):
        for c in ("FR", "FC"):
            r = R[c]
            rec = r["recon4"] if k == "4" else r["recon24_mean"]
            w(f"| {NAMES[c]}, {lab} | {r['ppl' + k]:.6f} | {r['nll' + k]:.6f} | +{r['abs' + k]:.4f} | "
              f"+{100 * r['deg' + k]:.3f} % | {rec:.6f}{' (layer mean)' if k == '24' else ''} |")
    w("\n## Coverage penalty\n")
    w(f"- Δ_4 = PPL(FC4) − PPL(FR4) = {o['delta4']:+.6f} ({100 * o['delta4_rel']:+.3f} %); ΔNLL {o['delta_nll4']:+.6f}; "
      f"paired window bootstrap 95 % CI of ΔNLL [{o['bootstrap4']['ci95'][0]:+.6f}, {o['bootstrap4']['ci95'][1]:+.6f}]")
    w(f"- Δ_24 = PPL(FC24) − PPL(FR24) = {o['delta24']:+.6f} ({100 * o['delta24_rel']:+.3f} %); ΔNLL {o['delta_nll24']:+.6f}; "
      f"95 % CI of ΔNLL [{o['bootstrap24']['ci95'][0]:+.6f}, {o['bootstrap24']['ci95'][1]:+.6f}]")
    w(f"- Δ_24 − Δ_4 = {o['delta_change']:+.6f}; Δ_24 / Δ_4 = {o['delta_ratio']:.2f}; τ = {o['tau']:.6f}")
    W = o["window_diff"]
    w(f"- Windows where FC24 is worse: {W['fc_worse_windows']}/{W['n']}; per-window ΔNLL median {W['median']:+.5f}, "
      f"5–95 % [{W['q05']:+.5f}, {W['q95']:+.5f}]; correlation with the 4-layer per-window differences {W['corr_with_4layer']:+.2f}")
    w(f"- Expert reconstruction (test rows): FC worse than FR in {o['recon_fc_worse_layers']}/24 layers; "
      f"layer-mean relative difference {100 * np.mean(list(o['recon_rel_diff_by_layer'].values())):+.2f} %")
    S = o.get("calibration_numerics_sensitivity")
    if S:
        w("\n## Calibration-numerics sensitivity (four layers; from the failed v1-cache regression)\n")
        w("Same selections, solver and settings; BF16 calibration inputs differing only by batch-composition rounding:")
        for c in ("FR", "FC"):
            w(f"- {NAMES[c]}: PPL {S['ppl_original'][c]:.6f} → {S['ppl_perturbed'][c]:.6f} "
              f"({S['ppl_perturbed'][c] - S['ppl_original'][c]:+.6f}); recon {S['recon_original'][c]:.7f} → "
              f"{S['recon_perturbed'][c]:.7f}")
        w(f"- Δ_4: {S['delta4_original']:+.6f} → {S['delta4_perturbed']:+.6f}. PPL shifts up to "
          f"{S['max_abs_ppl_shift']:.4f} from numerically equivalent calibration inputs — the same order as Δ_4; the "
          "window bootstrap does not capture this source of variation.")
    w("\n## Correctness\n")
    for c, k in o["correctness"].items():
        w(f"- {NAMES[c]}: {k['layers']} layers, {k['experts']} experts, {k['projections']} projections; "
          f"unchanged from BF16 {k['unchanged']}; non-finite weights {k['nonfinite']}; untouched params unchanged "
          f"{k['untouched_ok']}; RTN fallbacks {k['rtn']}; finite NLL {k['finite_nll']}; repeated evaluation |ΔPPL| "
          f"{k['repeat']:.1e}; expert checks pass {k['checks_pass']} (max deviation / tolerance {k['max_dev_over_tol']:.2f}); "
          f"weights sha256 {k['sha'][:16]}")
    G = o["regression"]
    w(f"- Four-layer regression: original cache reproduces the partial-PPL weights and PPL exactly: "
      f"{G['original_cache_exact']}; new cache within tolerance: {G['pilot_cache_ok']} (Δ_4 with new cache "
      f"{G['delta4_pilot_cache']:+.6f})")
    if o.get("cache"):
        for L_, v in o["cache"]["comparison"].items():
            w(f"  - cache L{L_}: {v['rows_bitwise_equal']}/{v['rows']} rows bitwise equal to ~/moe_cache; max abs diff "
              f"{v['max_abs_diff']:.3g}; max rel. (Frobenius) {v['max_rel_fro']:.2e}")
    w("\n## Cost\n")
    w("| strategy | quantize (min) | Hessian (min) | GPTQ (min) | PPL eval (s) | peak GPU GB | peak CPU GB | weight artifacts GB |")
    w("|---|---|---|---|---|---|---|---|")
    for c, k in o["cost"].items():
        w(f"| {NAMES[c]} | {k['quantize_min']:.1f} | {k['hessian_min']:.1f} | {k['gptq_min']:.1f} | {k['eval_s']:.0f} | "
          f"{k['peak_gpu_gb']:.1f} | {k['peak_cpu_gb']:.1f} | {k['artifacts_gb']:.1f} |")
    if o.get("cache"):
        w(f"\nBF16 calibration cache: {o['cache']['rows']} rows × 24 layers, {o['cache']['disk_gb']:.1f} GB, "
          f"{o['cache']['seconds'] / 60:.1f} min. Full Coverage-LP additionally needs the ~357 s routing profile.")
    w("\nFigures: `figures/fig1_ppl_by_depth`, `fig2_coverage_penalty_by_depth`, `fig3_window_nll_differences`, "
      "`fig4_runtime_memory` (PDF + PNG).")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    o, *_ = build()
    json.dump(o, open(FD.p("summary.json"), "w"), indent=1)
    write_md(o, FD.p("summary.md"))
    if o["decision"] != "RUN-BLOCKED":
        figures(o, FD.p("figures"))
    print(f"decision: {o['decision']}")


if __name__ == "__main__":
    main()
