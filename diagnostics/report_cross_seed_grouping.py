"""Results, paired differences, variance analysis, decision (frozen in protocol.json) and figures for the
cross-seed x calibration-grouping study. Seed 0 = reused runs_multigroup_robustness results."""
from __future__ import annotations
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import cross_seed_grouping as XG        # noqa: E402
import multigroup_robustness as MR      # noqa: E402
import eval_partial_ppl as EP           # noqa: E402
import full_depth_pilot as FD           # noqa: E402
from moe_calib import cross_seed_grouping as CS     # noqa: E402
from moe_calib import multigroup_robustness as MG   # noqa: E402

P, GS, SEEDS = XG.p, XG.GSEEDS, list(CS.ALLOC_SEEDS)
SCOL = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def result_path(c, s, g):
    if s == 0:
        return os.path.join(ROOT, MR.OUT, "results", f"{c}_{MG.gname(g)}.json")
    return P("results", f"{XG.rname(c, s, g)}.json")


def dir_bytes(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) if os.path.isdir(d) else 0


def build():
    pr = XG.protocol()
    pre = load(P("logs", "preflight.json"))
    cm, act = load(P("cache_manifest.json")), load(P("activation_diagnostics.json"))
    base = json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))
    b = base["ppl"]
    res = {(c, s, g): load(result_path(c, s, g)) for c in XG.STRATS for s in SEEDS for g in GS}
    ctrl = load(P("results", f"{XG.rname('FR', 0, GS[0])}_CTRL_repeat.json"))
    missing = [XG.rname(c, s, g) for (c, s, g), r in res.items() if r is None]
    invalid = [XG.rname(c, s, g) for (c, s, g), r in res.items() if r is not None and not r["valid"]]
    checks = dict(seed0_reuse_valid=bool(pre and pre["seed0_reuse"]["valid"]),
                  new_caches_valid=bool(cm and cm.get("valid")),
                  repeat_quant_control=bool(ctrl and ctrl["determinism"]["reproduced_bitwise"]),
                  repeat_ppl_identical=all(r["repeat_eval_abs_diff_ppl"] == 0 for r in res.values() if r),
                  selection_hashes_match=all(r["ids_hash"] == pr["seeds"][str(s)]["selection_hashes"][c]
                                             for (c, s, g), r in res.items() if r),
                  manifests_match=all(r.get("manifest_sha256", pr["groupings"][CS.cname(s, g)]["manifest_sha256"]) ==
                                      pr["groupings"][CS.cname(s, g)]["manifest_sha256"] for (c, s, g), r in res.items() if r and s),
                  windows_tokens=all(r["total_tokens"] == 65408 for r in res.values() if r))
    blocked = bool(missing or invalid or not all(checks.values()))
    out = dict(protocol_sha256=pr["protocol_sha256"], decision_rule=pr["decision_rule"], checks=checks, missing=missing,
               invalid=invalid, n_new_complete=sum(1 for (c, s, g), r in res.items() if s and r and r["valid"]),
               n_new_expected=20, n_seed0_reused=sum(1 for (c, s, g), r in res.items() if s == 0 and r))
    if blocked:
        out["decision"] = "RUN-BLOCKED"
        return out, None
    ppl = {k: r["ppl"] for k, r in res.items()}
    D = CS.paired_deltas(ppl)
    v = CS.variance_summary(D)
    cat, det = CS.decide(D)
    hb = CS.hierarchical_bootstrap(D)
    out["decision"], out["decision_details"] = cat, det
    tok = res[("FR", 0, GS[0])]["per_window_tokens"]
    assert all(r["per_window_tokens"] == tok for r in res.values())

    # ---------------- ppl_results.json
    cfgs = {}
    for (c, s, g), r in res.items():
        cfgs[XG.rname(c, s, g)] = dict(
            strategy=c, allocation_seed=s, grouping_seed=g, ppl=r["ppl"], mean_nll=r["mean_nll"],
            ppl_degradation_rel_bf16=(r["ppl"] - b) / b, per_window_nll=r["per_window_nll"], recon=r["recon"],
            recon_per_layer=r["recon_per_layer"], installed_weights_sha256=r["installed_weights_sha256"],
            ids_hash=r["ids_hash"], manifest_sha256=pr["groupings"][CS.cname(s, g)]["manifest_sha256"],
            quantize_s=r["quantize_s"], hessian_s=r["hessian_s"], gptq_s=r["gptq_s"], eval_s=r["eval_s"],
            peak_gpu_gb=r["peak_gpu_gb"], peak_cpu_gb=r["peak_cpu_gb"], status="valid" if r["valid"] else "invalid",
            source="reused (runs_multigroup_robustness)" if s == 0 else "new")
    XG.jdump(dict(bf16_ppl=b, evaluation=pr["evaluation"], configs=cfgs), P("ppl_results.json"))

    # ---------------- paired differences
    pairs = {}
    for s in SEEDS:
        for g in GS:
            fr, fc = res[("FR", s, g)], res[("FC", s, g)]
            wb = FD.paired_bootstrap(fc["per_window_nll"], fr["per_window_nll"], tok)
            pairs[CS.cname(s, g)] = dict(allocation_seed=s, grouping_seed=g, fr_ppl=fr["ppl"], fc_ppl=fc["ppl"],
                                         delta_ppl=D[s][g], delta_mean_nll=fc["mean_nll"] - fr["mean_nll"],
                                         recon_fr=fr["recon"], recon_fc=fc["recon"],
                                         recon_penalty_rel=(fc["recon"] - fr["recon"]) / fr["recon"],
                                         window_bootstrap=dict(point=wb["point"], ci95=wb["ci95"]))
    XG.jdump(dict(definition="Delta[s,g] = PPL(FC,s,g) - PPL(FR,s,g), paired within (allocation seed, grouping)",
                  window_bootstrap_note=CS.UNCERTAINTY_LEVELS["window_bootstrap"], pairs=pairs), P("paired_differences.json"))

    # ---------------- variance analysis
    XG.jdump(dict(summary=v, hierarchical_bootstrap=hb, decision=cat, decision_details=det,
                  uncertainty_levels=CS.UNCERTAINTY_LEVELS,
                  pseudoreplication_note="the 15 Delta[s,g] are NOT 15 independent replicates: groupings are nested "
                                         "within 3 allocation seeds; no flat 15-observation bootstrap is used"),
             P("variance_analysis.json"))

    # ---------------- reconstruction
    rec = {}
    for s in SEEDS:
        per = {}
        for c in XG.STRATS:
            x = np.array([res[(c, s, g)]["recon"] for g in GS])
            per[c] = dict(values={MG.gname(g): float(res[(c, s, g)]["recon"]) for g in GS}, mean=float(x.mean()),
                          rel_range=float((x.max() - x.min()) / x.mean()))
        gap = np.array([pairs[CS.cname(s, g)]["recon_penalty_rel"] for g in GS])
        dl = np.array([D[s][g] for g in GS])
        per["fc_penalty_rel"] = dict(values={MG.gname(g): float(y) for g, y in zip(GS, gap)}, mean=float(gap.mean()),
                                     min=float(gap.min()), max=float(gap.max()), n_positive=int((gap > 0).sum()))
        per["corr_penalty_vs_delta_ppl"] = MG.pearson(gap, dl)
        per["corr_recon_vs_ppl_within_strategy"] = {c: MG.pearson([res[(c, s, g)]["recon"] for g in GS],
                                                                  [ppl[(c, s, g)] for g in GS]) for c in XG.STRATS}
        rec[str(s)] = per
    allgap = [pairs[k]["recon_penalty_rel"] for k in pairs]
    rec_sum = dict(per_seed=rec, fc_recon_worse_in=f"{sum(x > 0 for x in allgap)}/15",
                   penalty_range_all=[min(allgap), max(allgap)],
                   seed_mean_penalty_vs_seed_mean_delta=dict(penalty={s: rec[str(s)]["fc_penalty_rel"]["mean"] for s in SEEDS},
                                                             delta={s: v["mean_delta"][s] for s in SEEDS}),
                   note="correlations have n=5 per seed; descriptive only, no causal claim")
    XG.jdump(dict(evaluator="E3 held-out relative expert reconstruction, ~/moe_cache test rows 2128-2255 (eval_cache_dir), "
                            "same for all 30 configurations", configs={k: dict(recon=x["recon"], recon_per_layer=x["recon_per_layer"])
                                                                        for k, x in cfgs.items()}, summary=rec_sum),
             P("reconstruction_results.json"))

    # ---------------- historical ORIGINAL (partial-PPL) per seed
    hist = {}
    for s in SEEDS:
        fr = json.load(open(os.path.join(ROOT, EP.OUT, f"ppl_FR_s{s}.json")))
        fc = json.load(open(os.path.join(ROOT, EP.OUT, f"ppl_FC_s{s}.json")))
        hist[s] = dict(fr=fr["ppl"], fc=fc["ppl"], delta=fc["ppl"] - fr["ppl"],
                       recon_penalty_rel=(fc["recon_recomputed"] - fr["recon_recomputed"]) / fr["recon_recomputed"])
    new = [r for (c, s, g), r in res.items() if s]
    disk = dict(new_caches_gb=dir_bytes(XG.XCACHE) / 1e9 - dir_bytes(os.path.join(XG.XCACHE, "weights")) / 1e9,
                new_weights_gb=dir_bytes(os.path.join(XG.XCACHE, "weights")) / 1e9)
    out.update(
        bf16_ppl=b, deltas={s: {MG.gname(g): D[s][g] for g in GS} for s in SEEDS},
        table=[dict(seed=s, grouping=g, fr=ppl[("FR", s, g)], fc=ppl[("FC", s, g)], delta=D[s][g]) for s in SEEDS for g in GS],
        variance=v, hierarchical_bootstrap=hb, reconstruction=rec_sum, original_extraction_reference=hist,
        window_bootstrap_summary={k: x["window_bootstrap"]["ci95"] for k, x in pairs.items()},
        n_window_ci_excluding_zero=dict(positive=sum(x["window_bootstrap"]["ci95"][0] > 0 for x in pairs.values()),
                                        negative=sum(x["window_bootstrap"]["ci95"][1] < 0 for x in pairs.values())),
        repeat_quant_control=ctrl["determinism"],
        cache_controls=dict(execution_order={k: e["negative_control"]["passed"] for k, e in cm["entries"].items()},
                            token_ids={k: e["token_ids_identical"] for k, e in cm["entries"].items()}),
        correctness={XG.rname(c, s, g): dict(experts=r["n_quantized_experts"], projections=r["n_projection_tensors"],
                                             rtn=r["rtn_fallbacks"], unchanged=r["projections_unchanged_from_bf16"],
                                             nonfinite=r["nonfinite_weights"], untouched_ok=r["untouched_params_unchanged"],
                                             expert_checks_pass=all(k["pass"] for k in r["expert_checks"]), valid=r["valid"])
                     for (c, s, g), r in res.items() if s},
        compute=dict(n_new_configs=len(new), mean_quantize_s=float(np.mean([r["quantize_s"] for r in new])),
                     mean_hessian_s=float(np.mean([r["hessian_s"] for r in new])),
                     mean_gptq_and_recon_s=float(np.mean([r["gptq_s"] for r in new])),
                     mean_eval_s=float(np.mean([r["eval_s"] for r in new])),
                     cache_generation_s={k: e["generation_s"] for k, e in cm["entries"].items()},
                     cache_negative_control_s={k: e["negative_control"]["seconds"] for k, e in cm["entries"].items()},
                     diagnostics_s=act["seconds"] if act else None, control_quantize_s=ctrl["quantize_s"],
                     peak_gpu_gb=max(r["peak_gpu_gb"] for r in new + [ctrl]), peak_cpu_gb=max(r["peak_cpu_gb"] for r in new + [ctrl]),
                     disk=disk, note="reconstruction scoring runs inside the GPTQ loop (gptq_s includes it)"))
    if act:
        out["activation_routing"] = {s: {L: dict(rel_l2=[act["per_seed"][s][L][MG.gname(g)]["relative_l2"] for g in GS],
                                                 pct_set_changed=[100 * act["per_seed"][s][L][MG.gname(g)]["frac_tokens_top4_set_changed"] for g in GS],
                                                 routing_l1=[act["per_seed"][s][L][MG.gname(g)]["routing_distribution_l1"] for g in GS])
                                         for L in act["per_seed"][s]} for s in act["per_seed"]}
    return out, dict(D=D, v=v, ppl=ppl, res=res, pairs=pairs, act=act, hist=hist)


# ---------------------------------------------------------------- figures
def figures(raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    d = P("figures")
    sv = lambda f, n: (f.savefig(os.path.join(d, n + ".pdf"), bbox_inches="tight"),
                       f.savefig(os.path.join(d, n + ".png"), dpi=200, bbox_inches="tight"), plt.close(f))
    D, v, pairs, hist = raw["D"], raw["v"], raw["pairs"], raw["hist"]
    off = np.linspace(-.24, .24, len(GS))
    marks = ["o", "s", "^", "D", "v"]

    # 1 all 15 paired deltas
    f, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(0, color=INK, lw=.8)
    for s in SEEDS:
        for j, g in enumerate(GS):
            ax.scatter(s + off[j], D[s][g], color=SCOL[s], marker=marks[j], s=34, edgecolor="white", lw=1, zorder=3,
                       label=f"grouping …{str(g)[-2:]}" if s == 0 else None)
        ax.hlines(v["mean_delta"][s], s - .32, s + .32, color=SCOL[s], lw=2.2, zorder=2)
        ax.scatter(s + .4, hist[s]["delta"], marker="D", s=30, facecolor="none", edgecolor=MUTED, lw=1, zorder=3,
                   label="ORIGINAL extraction (ref.)" if s == 0 else None)
    ax.set_xticks(SEEDS, [f"seed {s}" for s in SEEDS]); ax.set_xlabel("allocation seed (bar: mean over 5 groupings)")
    ax.set_ylabel("Δ[s,g] = PPL(FC) − PPL(FR)")
    ax.set_title("Paired FC − FR PPL per allocation seed and grouping", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.18))
    sv(f, "fig1_paired_deltas")

    # 2 within-seed grouping sensitivity
    f, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.axhline(0, color=INK, lw=.8)
    for s in SEEDS:
        y = [D[s][g] for g in GS]
        ax.scatter([s] * 5, y, color=SCOL[s], s=30, edgecolor="white", lw=1, zorder=3)
        ax.errorbar(s + .18, v["mean_delta"][s], yerr=v["sd_grouping"][s], color=SCOL[s], capsize=4, lw=1.4, marker="_", ms=10)
        ax.text(s + .26, v["mean_delta"][s], f"SD {v['sd_grouping'][s]:.4f}", va="center", fontsize=7.5, color=INK)
    ax.set_xticks(SEEDS, [f"seed {s}" for s in SEEDS]); ax.set_xlim(-.4, 2.8)
    ax.set_ylabel("Δ[s,g]"); ax.set_title("Within-seed grouping spread (bar: mean ± 1 SD over 5 groupings)", fontsize=9, loc="left")
    sv(f, "fig2_within_seed_sensitivity")

    # 3 grouping-averaged effect
    f, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.axhline(0, color=INK, lw=.8)
    for s in SEEDS:
        ax.errorbar(s, v["mean_delta"][s], yerr=v["se_grouping_mean"][s], color=SCOL[s], marker="o", ms=7, capsize=4, lw=1.4)
        ax.text(s + .08, v["mean_delta"][s], f"{v['mean_delta'][s]:+.4f}", fontsize=7.5, va="bottom", color=INK)
    ax.axhline(v["overall_mean_delta"], color=MUTED, ls="--", lw=1.2)
    ax.text(2.45, v["overall_mean_delta"], f"overall {v['overall_mean_delta']:+.4f}", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xticks(SEEDS, [f"seed {s}" for s in SEEDS]); ax.set_xlim(-.5, 2.5)
    ax.set_ylabel("mean_g Δ[s,g]")
    ax.set_title("Grouping-averaged FC − FR (± grouping SE, 3 seeds only)", fontsize=9, loc="left")
    sv(f, "fig3_grouping_averaged_effect")

    # 4 reconstruction penalty vs delta
    f, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.axhline(0, color=INK, lw=.8)
    for s in SEEDS:
        for j, g in enumerate(GS):
            x = pairs[CS.cname(s, g)]
            ax.scatter(100 * x["recon_penalty_rel"], x["delta_ppl"], color=SCOL[s], marker=marks[j], s=34, edgecolor="white",
                       lw=1, zorder=3, label=f"seed {s}" if j == 0 else None)
    ax.set_xlabel("FC − FR relative reconstruction penalty (%)"); ax.set_ylabel("Δ PPL (FC − FR)")
    ax.set_title("Reconstruction penalty is stable; PPL difference is not (marker = grouping)", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    sv(f, "fig4_recon_vs_ppl")

    # 5 sources of variation (descriptive)
    f, ax = plt.subplots(figsize=(5.8, 3.0))
    items = [("pooled within-seed\ngrouping SD", v["pooled_within_sd"]),
             ("SD of seed means\n(observed)", v["sd_between_seed_means"]),
             ("SD of seed means expected\nfrom grouping alone (W/√5)", v["expected_sd_of_seed_means_from_grouping_only"]),
             ("|overall mean Δ|", abs(v["overall_mean_delta"]))]
    y = np.arange(len(items))[::-1]
    ax.barh(y, [x[1] for x in items], color=[MUTED, "#2a78d6", "#9fbfe8", "#eb6834"], height=.55)
    for yy, (_, val) in zip(y, items):
        ax.text(val, yy, f" {val:.4f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [x[0] for x in items], fontsize=7.5); ax.set_xlabel("PPL units")
    ax.set_title("Sources of variation — descriptive, not a variance decomposition", fontsize=9, loc="left")
    ax.set_xlim(0, max(x[1] for x in items) * 1.3)
    sv(f, "fig5_sources_of_variation")

    # 6 activation / routing by seed
    act = raw["act"]
    if act:
        Ls = [str(L) for L in XG.LAYERS]
        f, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        for s in SEEDS:
            ps = act["per_seed"][str(s)]
            for ai, key, sc in ((0, "relative_l2", 1), (1, "frac_tokens_top4_set_changed", 100)):
                vals = np.array([[sc * ps[L][MG.gname(g)][key] for g in GS] for L in Ls])
                axs[ai].plot(XG.LAYERS, vals.mean(1), color=SCOL[s], marker="o", ms=4, lw=1.2, label=f"seed {s}")
                axs[ai].fill_between(XG.LAYERS, vals.min(1), vals.max(1), color=SCOL[s], alpha=.15, lw=0)
        axs[0].set_yscale("log"); axs[0].set_ylabel("relative L2 vs ORIGINAL"); axs[0].set_xlabel("layer input")
        axs[1].set_ylabel("% tokens, top-4 set changed"); axs[1].set_xlabel("layer"); axs[1].set_ylim(bottom=0)
        axs[0].legend(frameon=False, fontsize=7.5)
        for a_ in axs:
            a_.set_xticks(XG.LAYERS)
        f.suptitle("Grouping-induced activation / routing drift vs each seed's ORIGINAL extraction (band: min–max over 5 groupings)",
                   fontsize=9, x=.02, ha="left")
        f.tight_layout()
        sv(f, "fig6_activation_routing")


# ---------------------------------------------------------------- markdown
def markdown(o):
    v = o["variance"]
    L = ["# Cross-seed x calibration-grouping robustness (FR vs FC, four-layer GPTQ)", "",
         f"**Decision (rule frozen in protocol.json, sha256 `{o['protocol_sha256'][:16]}`): {o['decision']}**", "",
         f"Decision details: {o['decision_details']}", "",
         "| seed | grouping | FR PPL | FC PPL | FC − FR |", "|---|---|---|---|---|"]
    for r in o["table"]:
        L.append(f"| {r['seed']} | {r['grouping']} | {r['fr']:.6f} | {r['fc']:.6f} | {r['delta']:+.6f} |")
    L += ["", "| seed | mean Δ | SD (ddof 1) | FC worse / better | ORIGINAL-extraction Δ |", "|---|---|---|---|---|"]
    for s in SEEDS:
        L.append(f"| {s} | {v['mean_delta'][s]:+.6f} | {v['sd_grouping'][s]:.6f} | {v['n_fc_worse'][s]}/{v['n_fc_better'][s]} | "
                 f"{o['original_extraction_reference'][s]['delta']:+.6f} |")
    hb = o["hierarchical_bootstrap"]
    L += ["", f"- overall mean Δ {v['overall_mean_delta']:+.6f}; overall median {v['overall_median_delta']:+.6f}; "
              f"min {v['min_delta']:+.6f}, max {v['max_delta']:+.6f}; FC worse {v['n_fc_worse_total']}/15",
          f"- pooled within-seed SD {v['pooled_within_sd']:.6f}; SD of seed means {v['sd_between_seed_means']:.6f} "
          f"(grouping-only expectation {v['expected_sd_of_seed_means_from_grouping_only']:.6f}); between-seed component SD "
          f"≈ {v['between_seed_sd_component']:.6f} (descriptive)",
          f"- hierarchical bootstrap 95% interval [{hb['ci95'][0]:+.6f}, {hb['ci95'][1]:+.6f}] ({hb['n_distinct_seed_resamples']} "
          "distinct seed resamples; unstable, descriptive)",
          f"- checks: {o['checks']}", ""]
    return "\n".join(L) + "\n"


def main():
    out, raw = build()
    XG.jdump(out, P("summary.json"))
    if raw is None:
        print(f"[report] blocked: missing {out['missing']} invalid {out['invalid']} checks {out['checks']}"); return
    open(P("summary.md"), "w").write(markdown(out))
    figures(raw)
    print(f"[report] decision {out['decision']}; seed means " +
          " ".join(f"{out['variance']['mean_delta'][s]:+.5f}" for s in SEEDS))


if __name__ == "__main__":
    main()
