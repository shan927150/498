"""Results, summary, decision (rule frozen in protocol.json) and figures for the batch-sensitivity study."""
from __future__ import annotations
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import batch_sensitivity as DB        # noqa: E402
import eval_partial_ppl as EP         # noqa: E402
import full_depth_pilot as FD         # noqa: E402
from moe_calib import batch_sensitivity as BS   # noqa: E402

P = DB.p
VCOL = {"ORIGINAL": "#6b6b6b", "COMPACT": "#2a78d6", "SHUFFLED": "#e07b00"}
SMARK = {"FR": "o", "FC": "s"}
ALT = ("COMPACT", "SHUFFLED")


def load(name):
    path = P(name)
    return json.load(open(path)) if os.path.exists(path) else None


def build():
    proto = load("protocol.json")
    val, reg = load("cache_validation.json"), load("regression_checks.json")
    act, rout, hess, wts = (load(n) for n in ("activation_drift.json", "routing_disagreement.json",
                                               "hessian_sensitivity.json", "weight_sensitivity.json"))
    base = json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))
    res = {(c, v): load(os.path.join("results", f"{DB.cfg_name(c, v)}.json")) for c in DB.STRATS for v in BS.VARIANTS}
    missing = [f"{c}_{v}" for (c, v), r in res.items() if r is None]
    invalid = [f"{c}_{v}" for (c, v), r in res.items() if r is not None and not r["valid"]]
    valid_variants = bool(val and val.get("valid"))
    blocked = bool(missing or invalid or not (reg and reg["passed"]) or not val
                   or not val.get("negative_control", {}).get("passed"))
    out = dict(decision_rule=proto["decision_rule"], n_configs_complete=6 - len(missing), n_configs_expected=6,
               missing=missing, invalid=invalid, regression_passed=bool(reg and reg["passed"]),
               cache_semantics_valid=valid_variants)
    if missing or invalid or not out["regression_passed"]:
        out["decision"] = BS.decide({}, base["ppl"], valid_variants, blocked=True)
        return out, None
    b = base["ppl"]
    ppl = {k: r["ppl"] for k, r in res.items()}
    sd = BS.strategy_deltas(ppl)
    out["decision"] = BS.decide(ppl, b, valid=valid_variants, blocked=blocked)
    tok = res[("FR", "ORIGINAL")]["per_window_tokens"]
    boot = {v: FD.paired_bootstrap(res[("FC", v)]["per_window_nll"], res[("FR", v)]["per_window_nll"], tok)
            for v in BS.VARIANTS}
    table = {f"{c}|{v}": dict(ppl=r["ppl"], mean_nll=r["mean_nll"], ppl_minus_bf16=r["ppl"] - b,
                              degradation_rel=(r["ppl"] - b) / b,
                              change_vs_original=r["ppl"] - res[(c, "ORIGINAL")]["ppl"],
                              rel_change_vs_original=(r["ppl"] - res[(c, "ORIGINAL")]["ppl"]) / res[(c, "ORIGINAL")]["ppl"],
                              recon=r["recon"], recon_rel_change_vs_original=(r["recon"] - res[(c, "ORIGINAL")]["recon"]) /
                              res[(c, "ORIGINAL")]["recon"], weights_sha256=r["installed_weights_sha256"],
                              quantize_s=r["quantize_s"], eval_s=r["eval_s"], repeat_eval_abs_diff_ppl=r["repeat_eval_abs_diff_ppl"],
                              peak_gpu_gb=r["peak_gpu_gb"], peak_cpu_gb=r["peak_cpu_gb"])
             for (c, v), r in res.items()}
    # per-expert reconstruction change vs ORIGINAL (same 240 experts, same eval rows)
    rec_exp = {}
    for c in DB.STRATS:
        o = {(x["layer"], x["expert"]): x["mean"] for x in res[(c, "ORIGINAL")]["per_expert"]}
        for v in ALT:
            d = np.array([(x["mean"] - o[(x["layer"], x["expert"])]) / o[(x["layer"], x["expert"])]
                          for x in res[(c, v)]["per_expert"]])
            rec_exp[f"{c}|{v}"] = dict(n=len(d), median_abs_rel=float(np.median(np.abs(d))),
                                       p90_abs_rel=float(np.quantile(np.abs(d), .9)), max_abs_rel=float(np.abs(d).max()),
                                       frac_worse=float((d > 0).mean()))
    d0 = abs(sd["delta"]["ORIGINAL"])
    shift = max(sd["max_shift"].values())
    inter = max(abs(sd["interaction"][v]) for v in ALT)
    wd = {v: (np.array(res[("FC", v)]["per_window_nll"]) - np.array(res[("FR", v)]["per_window_nll"])) / np.array(tok)
          for v in BS.VARIANTS}
    v1ref = proto["references"]["v1_compact_regression"]
    out.update(
        bf16_ppl=b, table=table, delta=sd["delta"], interaction=sd["interaction"], strategy_range=sd["strategy_range"],
        delta_range=sd["delta_range"], n_variants_fc_worse=sd["n_fc_worse"], ranking=sd["ranking"],
        ranking_reverses=sd["ranking_reverses"], max_shift_vs_original=sd["max_shift"],
        decision_quantities=dict(abs_delta_original=d0, shift=shift, max_abs_interaction=inter, tau=1e-4 * b,
                                 shift_over_abs_delta_original=shift / d0, interaction_over_abs_delta_original=inter / d0,
                                 strategy_range_over_abs_delta_original={c: r / d0 for c, r in sd["strategy_range"].items()}),
        paired_window_bootstrap=boot,
        bootstrap_note="the window bootstrap measures evaluation-text variability for a FIXED pair of quantized models; "
                       "it says nothing about calibration-batch sensitivity",
        window_diff_correlation={v: float(np.corrcoef(wd[v], wd["ORIGINAL"])[0, 1]) for v in ALT},
        recon_per_expert_change=rec_exp,
        compact_reproduces_v1=dict(
            cache_bitwise=val["COMPACT"].get("v1_reproduced_bitwise_all_layers"),
            weights={c: res[(c, "COMPACT")]["installed_weights_sha256"] == v1ref[c]["installed_sha256"] for c in DB.STRATS},
            ppl={c: res[(c, "COMPACT")]["ppl"] == v1ref[c]["ppl"] for c in DB.STRATS}),
        correctness={f"{c}|{v}": dict(experts=r["n_quantized_experts"], rtn_fallbacks=r["rtn_fallbacks"],
                                      unchanged=r["projections_unchanged_from_bf16"], nonfinite=r["nonfinite_weights"],
                                      untouched_ok=r["untouched_params_unchanged"],
                                      expert_checks_pass=all(k["pass"] for k in r["expert_checks"]),
                                      max_check_ratio=max(k["installed_vs_e3_rel"] / k["tolerance"] for k in r["expert_checks"]),
                                      repeat_eval=r["repeat_eval_abs_diff_ppl"], valid=r["valid"])
                     for (c, v), r in res.items()},
        regression=reg, negative_control=dict(passed=val["negative_control"]["passed"],
                                              layers_bitwise=sum(x["bitwise_equal"] for x in val["negative_control"]["per_layer"].values())),
        semantics=val["semantics"])
    if act:
        out["activation"] = dict(first_layer_with_differences=act["first_layer_with_differences"],
                                 by_layer={L: {v: {k: x[v]["calibration_rows"][k] for k in
                                                   ("relative_l2", "max_abs", "token_rel_l2_p50", "token_rel_l2_p99",
                                                    "frac_elements_exactly_equal", "n_sequences_with_differences", "nan_count", "inf_count")}
                                               for v in ALT} for L, x in act["per_layer"].items()})
    if rout:
        out["routing"] = {L: {v: {k: x[v][k] for k in ("frac_ordered_identical", "frac_set_identical", "frac_reordered_same_set",
                                                        "frac_any_expert_changed", "mean_topk_overlap", "routing_distribution_l1",
                                                        "max_expert_count_abs_diff", "mean_abs_topk_score_diff",
                                                        "n_sequences_with_set_change")}
                              for v in ALT} for L, x in rout["per_layer"].items()}
    if hess:
        out["hessian"] = hess["per_layer"]
        rs = [r for x in hess["per_layer"].values() for rr in x.values() for r in rr.values()]
        out["hessian_validation"] = dict(
            routing_matches_profiler=f"{sum(r['routing_matches_profiler'] for r in rs)}/{len(rs)}",
            max_gram_vs_profiler_rel=max(r["decomp_gram_vs_profiler_rel"] for r in rs if "decomp_gram_vs_profiler_rel" in r),
            share_membership_dominant=f"{sum(r['decomp_rel_fro_membership_change'] > r['decomp_rel_fro_common_tokens_activation'] for r in rs if 'decomp_rel_fro_membership_change' in r)}/"
                                      f"{sum('decomp_rel_fro_membership_change' in r for r in rs)}",
            actorder_perm_identical=f"{sum(r['actorder_perm_identical'] for r in rs)}/{len(rs)}")
    if wts:
        out["weights"] = {k: {kk: vv for kk, vv in x.items() if kk != "per_expert"} for k, x in wts["comparisons"].items()}
        out["weights_note"] = wts["scales_zero_points"]
    return out, dict(res=res, wd=wd, act=act, rout=rout, hess=hess, wts=wts)


# ---------------------------------------------------------------- figures
def figures(out, raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e4e4", "grid.linewidth": 0.6, "axes.axisbelow": True})
    d = P("figures")
    sv = lambda f, n: (f.savefig(os.path.join(d, n + ".pdf"), bbox_inches="tight"),
                       f.savefig(os.path.join(d, n + ".png"), dpi=200, bbox_inches="tight"), plt.close(f))
    Ls = list(range(DB.N_LAYERS))
    act, rout, hess, wts, res = raw["act"], raw["rout"], raw["hess"], raw["wts"], raw["res"]

    # 1 activation drift
    f, ax = plt.subplots(figsize=(6.6, 3.8))
    for v in ALT:
        y = [act["per_layer"][str(L)][v]["calibration_rows"]["relative_l2"] for L in Ls]
        ax.plot(Ls, np.maximum(y, 1e-9), "-o", ms=4, lw=2, color=VCOL[v], label=f"{v} vs ORIGINAL")
    ax.set_yscale("log"); ax.set_xlabel("decoder layer (cached layer INPUT; layer 0 = embeddings)")
    ax.set_ylabel("relative L2 difference (log scale)")
    ax.set_title("BF16 calibration-activation drift by layer (217 calibration rows; 0 plotted at 1e-9)", fontsize=9.5)
    for L in DB.LAYERS:
        ax.axvline(L, color="#bbb", lw=.8, ls=":")
    ax.legend(frameon=False)
    sv(f, "fig1_activation_drift")

    # 2 routing disagreement
    f, ax = plt.subplots(figsize=(6.6, 3.8))
    for v in ALT:
        ax.plot(Ls, [100 * rout["per_layer"][str(L)][v]["frac_any_expert_changed"] for L in Ls], "-o", ms=4, lw=2,
                color=VCOL[v], label=f"{v}: top-4 set changed")
        ax.plot(Ls, [100 * rout["per_layer"][str(L)][v]["frac_reordered_same_set"] for L in Ls], "--", lw=1.2,
                color=VCOL[v], alpha=.8, label=f"{v}: same set, order changed")
    ax.set_xlabel("decoder layer"); ax.set_ylabel("% of calibration tokens")
    ax.set_title("Routing disagreement vs ORIGINAL (E3 profiler replay of each cache)", fontsize=9.5)
    for L in DB.LAYERS:
        ax.axvline(L, color="#bbb", lw=.8, ls=":")
    ax.legend(frameon=False, fontsize=8)
    sv(f, "fig2_routing_disagreement")

    # 3 Hessian and weight sensitivity
    f, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for v in ALT:
        for c in DB.STRATS:
            for pj, ls in (("gu", "-"), ("down", "--")):
                y = [np.mean([hess["per_layer"][str(L)][c][f"{v}|e{e}|{pj}"]["rel_fro"] for e in DB.DIAG_EXPERTS])
                     for L in DB.LAYERS]
                ax.plot(DB.LAYERS, y, ls, marker=SMARK[c], color=VCOL[v], ms=5, lw=1.3,
                        label=f"{v} {c} {pj}")
    ax.set_yscale("log"); ax.set_xticks(DB.LAYERS); ax.set_xlabel("layer")
    ax.set_ylabel("Hessian rel. Frobenius diff (log; mean of experts 0, 30)")
    ax.set_title("Expert input Hessians vs ORIGINAL", fontsize=9.5)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax = axes[1]
    for v in ALT:
        for c in DB.STRATS:
            y = [wts["comparisons"][f"{c}|{v}"]["per_layer"][str(L)]["rel_l2"] for L in DB.LAYERS]
            ax.plot(DB.LAYERS, y, "-", marker=SMARK[c], color=VCOL[v], ms=5, lw=1.5, label=f"{v} {c}")
    ax.set_xticks(DB.LAYERS); ax.set_xlabel("layer"); ax.set_ylabel("GPTQ weight rel. L2 diff vs ORIGINAL")
    ax.set_ylim(bottom=0)
    ax.set_title("Effective GPTQ weights vs ORIGINAL (all 60 experts)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5)
    sv(f, "fig3_hessian_weight_sensitivity")

    # 4 PPL by calibration batching
    f, ax = plt.subplots(figsize=(6.4, 3.9))
    for i, c in enumerate(DB.STRATS):
        for j, v in enumerate(BS.VARIANTS):
            y = res[(c, v)]["ppl"]
            x = i + (j - 1) * .22
            ax.plot(x, y, SMARK[c], color=VCOL[v], ms=9, label=v if i == 0 else None)
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels([DB.STRATS["FR"], DB.STRATS["FC"]]); ax.set_xlim(-.6, 1.6)
    ax.set_ylabel("WikiText-2 PPL (axis does not start at 0)")
    ax.set_title(f"PPL by calibration batching (layers 1/8/16/23 quantized; BF16 = {out['bf16_ppl']:.4f})", fontsize=9.5)
    lo = min(r["ppl"] for r in res.values()); hi = max(r["ppl"] for r in res.values())
    ax.set_ylim(lo - .006, hi + .006)
    ax.legend(frameon=False, title="calibration cache", fontsize=8)
    sv(f, "fig4_ppl_by_batching")

    # 5 coverage penalty across variants
    f, ax = plt.subplots(figsize=(6.0, 3.8))
    for j, v in enumerate(BS.VARIANTS):
        dv = out["delta"][v]
        ax.bar(j, dv, color=VCOL[v], width=.55)
        ppl_fr = res[("FR", v)]["ppl"]
        lo, hi = out["paired_window_bootstrap"][v]["ci95"]
        ax.errorbar(j, dv, yerr=[[dv - ppl_fr * (np.exp(lo) - 1)], [ppl_fr * (np.exp(hi) - 1) - dv]],
                    color="#333", capsize=4, lw=1.1)
        ax.annotate(f"{dv:+.4f}", (j + .29, dv), textcoords="offset points", xytext=(3, 0), va="center", fontsize=8)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(BS.VARIANTS); ax.set_xlim(-.5, 2.75)
    ax.set_ylabel("PPL(FC) − PPL(FR)")
    ax.set_title("Coverage penalty by calibration batching\n(whiskers: window bootstrap = evaluation-text "
                 "variability only)", fontsize=9.5)
    sv(f, "fig5_coverage_penalty_by_batching")

    # 6 reconstruction change vs PPL change
    f, ax = plt.subplots(figsize=(6.0, 3.9))
    for c in DB.STRATS:
        for v in ALT:
            t = out["table"][f"{c}|{v}"]
            ax.plot(100 * t["recon_rel_change_vs_original"], t["change_vs_original"], SMARK[c], color=VCOL[v], ms=9,
                    label=f"{c} {v}")
    ax.axhline(0, color="#555", lw=.9); ax.axvline(0, color="#555", lw=.9)
    ax.axhspan(-abs(out["delta"]["ORIGINAL"]), abs(out["delta"]["ORIGINAL"]), color="#eda100", alpha=.12,
               label="± |Δ ORIGINAL| (FC − FR)")
    ax.set_xlabel("mean expert reconstruction error change vs ORIGINAL (%)")
    ax.set_ylabel("PPL change vs ORIGINAL")
    ax.set_title("Local reconstruction vs model-level PPL (same strategy)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    sv(f, "fig6_recon_vs_ppl_change")


# ---------------------------------------------------------------- markdown
def markdown(out):
    L = [f"# Calibration batch-composition sensitivity — summary (four layers, seed 0)", "",
         f"**Decision (rule frozen in protocol.json): {out['decision']}**", ""]
    if "table" not in out:
        L += [f"Incomplete: missing {out['missing']}, invalid {out['invalid']}, regression {out['regression_passed']}"]
        return "\n".join(L)
    t, b = out["table"], out["bf16_ppl"]
    L += ["Routed experts of layers 1/8/16/23 GPTQ-quantized (4-bit, g128, percdamp 1.0, act-order; fake-quantized BF16 "
          "weights), everything else BF16. Only the grouping of calibration sequences into BF16 forward batches during "
          "activation caching differs between variants.", "",
          "| strategy | cache variant | PPL | mean NLL | Δ vs BF16 | change vs ORIGINAL | mean recon | recon change |",
          "|---|---|---|---|---|---|---|---|",
          f"| BF16 | — | {b:.6f} | — | — | — | — | — |"]
    for c in DB.STRATS:
        for v in BS.VARIANTS:
            r = t[f"{c}|{v}"]
            L.append(f"| {DB.STRATS[c]} | {v} | {r['ppl']:.6f} | {r['mean_nll']:.6f} | {r['ppl_minus_bf16']:+.4f} "
                     f"({100 * r['degradation_rel']:+.3f} %) | {r['change_vs_original']:+.6f} | {r['recon']:.7f} | "
                     f"{100 * r['recon_rel_change_vs_original']:+.3f} % |")
    L += ["", "| cache variant | FR PPL | FC PPL | FC − FR | ranking | interaction Δ(v) − Δ(ORIGINAL) | window-bootstrap 95 % CI of ΔNLL |",
          "|---|---|---|---|---|---|---|"]
    for v in BS.VARIANTS:
        ci = out["paired_window_bootstrap"][v]["ci95"]
        L.append(f"| {v} | {t['FR|' + v]['ppl']:.6f} | {t['FC|' + v]['ppl']:.6f} | {out['delta'][v]:+.6f} | "
                 f"{out['ranking'][v]} | {out['interaction'][v]:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |")
    q = out["decision_quantities"]
    L += ["", "## Numerical sensitivity (observed ranges over three deterministic groupings; not variance estimates)", "",
          f"- PPL range across cache variants: FR {out['strategy_range']['FR']:.6f}, FC {out['strategy_range']['FC']:.6f} "
          f"(= {q['strategy_range_over_abs_delta_original']['FR']:.2f}× and {q['strategy_range_over_abs_delta_original']['FC']:.2f}× "
          f"|Δ ORIGINAL| = {q['abs_delta_original']:.6f})",
          f"- Largest change from ORIGINAL: {q['shift']:.6f} ({q['shift_over_abs_delta_original']:.2f}× |Δ ORIGINAL|); "
          f"largest |interaction| {q['max_abs_interaction']:.6f} ({q['interaction_over_abs_delta_original']:.2f}×)",
          f"- Range of FC − FR across variants: {out['delta_range']:.6f}; FC worse in {out['n_variants_fc_worse']}/3 variants; "
          f"ranking reverses: {out['ranking_reverses']}",
          f"- Correlation of per-window FC − FR differences with ORIGINAL's: "
          + ", ".join(f"{v} {x:+.2f}" for v, x in out["window_diff_correlation"].items()),
          f"- {out['bootstrap_note']}", ""]
    L += ["## Reconstruction (same 240 experts, same ~/moe_cache test rows for every variant)", ""]
    for k, x in out["recon_per_expert_change"].items():
        L.append(f"- {k}: per-expert |relative change| median {100 * x['median_abs_rel']:.3f} %, p90 "
                 f"{100 * x['p90_abs_rel']:.3f} %, max {100 * x['max_abs_rel']:.3f} %; worse for {100 * x['frac_worse']:.0f} % of experts")
    if "activation" in out:
        a = out["activation"]
        L += ["", "## Activation drift (217 calibration rows, aligned by sequence id and position)", "",
              f"First layer input with any difference: {a['first_layer_with_differences']}", "",
              "| layer | COMPACT rel. L2 | COMPACT % elements equal | SHUFFLED rel. L2 | SHUFFLED % elements equal | p99 token rel. L2 (C / S) |",
              "|---|---|---|---|---|---|"]
        for Ls in ["0", "1", "2", "4", "8", "12", "16", "20", "23"]:
            x = a["by_layer"][Ls]
            L.append(f"| {Ls} | {x['COMPACT']['relative_l2']:.2e} | {100 * x['COMPACT']['frac_elements_exactly_equal']:.1f} | "
                     f"{x['SHUFFLED']['relative_l2']:.2e} | {100 * x['SHUFFLED']['frac_elements_exactly_equal']:.1f} | "
                     f"{x['COMPACT']['token_rel_l2_p99']:.2e} / {x['SHUFFLED']['token_rel_l2_p99']:.2e} |")
        nan = sum(x[v]["nan_count"] + x[v]["inf_count"] for x in a["by_layer"].values() for v in ALT)
        L.append(f"\nNaN/Inf elements in alternative caches: {nan}")
    if "routing" in out:
        L += ["", "## Routing disagreement (E3 profiler replay, 217 calibration rows = 111,104 tokens per layer)", "",
              "| layer | C: top-4 set changed | C: same set, reordered | C: mean overlap | S: top-4 set changed | S: same set, reordered | S: routing-distribution L1 |",
              "|---|---|---|---|---|---|---|"]
        for Ls in ["0", "1", "2", "4", "8", "12", "16", "20", "23"]:
            x = out["routing"][Ls]
            L.append(f"| {Ls} | {100 * x['COMPACT']['frac_any_expert_changed']:.3f} % | {100 * x['COMPACT']['frac_reordered_same_set']:.3f} % | "
                     f"{x['COMPACT']['mean_topk_overlap']:.5f} | {100 * x['SHUFFLED']['frac_any_expert_changed']:.3f} % | "
                     f"{100 * x['SHUFFLED']['frac_reordered_same_set']:.3f} % | {x['SHUFFLED']['routing_distribution_l1']:.2e} |")
    if "hessian" in out:
        L += ["", "## Hessian sensitivity (existing E3 Profiler Hessians; layers 1/8/16/23 × experts 0, 30)", "",
              "| layer | strategy | variant | expert | proj | tokens O / V | membership change (−/+) | rel. Frob | diag rel | trace ratio | act-order perm identical | cols changing group | gu decomposition: common-token / membership |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        hv = out["hessian_validation"]
        L.insert(len(L) - 3, f"Replay routing equals the Profiler's routed-token counts for {hv['routing_matches_profiler']} "
                 f"entries; max relative difference between the replay Gram matrix and the Profiler gu Hessian "
                 f"{hv['max_gram_vs_profiler_rel']:.1e}; membership term larger than the common-token activation term in "
                 f"{hv['share_membership_dominant']} gu comparisons; act-order permutation identical in {hv['actorder_perm_identical']}.\n")
        for Ls, xs in out["hessian"].items():
            for c, rr in xs.items():
                for key, h in rr.items():
                    v, e, pj = key.split("|")
                    dec = (f"{h['decomp_rel_fro_common_tokens_activation']:.2e} / {h['decomp_rel_fro_membership_change']:.2e}"
                           if pj == "gu" else "—")
                    L.append(f"| {Ls} | {c} | {v} | {e[1:]} | {pj} | {h['tokens_original']:.0f} / {h['tokens_variant']:.0f} | "
                             f"{h['tokens_only_original']} / {h['tokens_only_variant']} | {h['rel_fro']:.2e} | {h['diag_rel']:.2e} | "
                             f"{h['trace_ratio']:.6f} | {h['actorder_perm_identical']} | {100 * h['frac_columns_group_changed']:.1f} % | {dec} |")
    if "weights" in out:
        L += ["", "## Effective GPTQ weights vs ORIGINAL", "",
              "| strategy | variant | % weights differ | rel. L2 | max abs | per-projection rel. L2 median [min, max] | identical projections |",
              "|---|---|---|---|---|---|---|"]
        for k, x in out["weights"].items():
            c, v = k.split("|")
            qq = x["per_expert_rel_l2_quantiles"]
            L.append(f"| {c} | {v} | {100 * x['frac_differ']:.2f} | {x['rel_l2']:.3e} | {x['max_abs']:.3g} | "
                     f"{qq['0.5']:.3e} [{qq['0']:.3e}, {qq['1']:.3e}] | {x['n_projections_identical']}/720 |")
        L.append(f"\n{out['weights_note']}")
    cv = out["compact_reproduces_v1"]
    L += ["", "## Correctness and provenance", "",
          f"- ORIGINAL regression (fresh processes): weights, PPL and reconstruction bitwise equal to the partial-PPL run: "
          f"{out['regression_passed']}",
          f"- Negative control (ORIGINAL batches in reversed execution order): {out['negative_control']['layers_bitwise']}/24 "
          f"layers bitwise equal; passed {out['negative_control']['passed']}",
          f"- COMPACT regenerated: cache bitwise equal to the v1 cache at all 24 layers: {cv['cache_bitwise']}; GPTQ weights "
          f"equal to v1 regression: {cv['weights']}; PPL equal: {cv['ppl']}",
          f"- Input semantics: token ids identical {out['semantics']['token_ids_identical']}; same model weights "
          f"{out['semantics']['same_model_weights']}; eval mode {out['semantics']['eval_mode']}; dtype "
          f"{out['semantics']['same_dtype']}; batch pattern COMPACT = SHUFFLED {out['semantics']['batch_pattern_compact_equals_shuffled']}; "
          f"{out['semantics']['position_ids_and_mask_identical']}"]
    for k, x in out["correctness"].items():
        L.append(f"- {k}: {x['experts']} experts, RTN fallbacks {x['rtn_fallbacks']}, unchanged {x['unchanged']}, "
                 f"non-finite {x['nonfinite']}, untouched ok {x['untouched_ok']}, expert checks pass {x['expert_checks_pass']} "
                 f"(max {x['max_check_ratio']:.2f} of tolerance), repeat eval |ΔPPL| {x['repeat_eval']:.1e}")
    L += ["", "| config | quantize (s) | PPL eval (s) | peak GPU GB | peak CPU GB |", "|---|---|---|---|---|"]
    for k, r in t.items():
        L.append(f"| {k} | {r['quantize_s']:.0f} | {r['eval_s']:.0f} | {r['peak_gpu_gb']:.1f} | {r['peak_cpu_gb']:.1f} |")
    L += ["", "Figures: `figures/fig1_activation_drift`, `fig2_routing_disagreement`, `fig3_hessian_weight_sensitivity`, "
          "`fig4_ppl_by_batching`, `fig5_coverage_penalty_by_batching`, `fig6_recon_vs_ppl_change` (PDF + PNG)."]
    return "\n".join(L)


def main():
    out, raw = build()
    if raw:
        res = raw["res"]
        DB.jdump({f"{c}|{v}": {k: x for k, x in r.items()} for (c, v), r in res.items()}, P("results.json"))
        figures(out, raw)
    DB.jdump(out, P("summary.json"))
    open(P("summary.md"), "w").write(markdown(out) + "\n")
    print(f"decision: {out['decision']}")


if __name__ == "__main__":
    main()
