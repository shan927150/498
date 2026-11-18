"""Aggregate the Hessian early-stopping MVP: timing breakdown, quality-cost table, frozen decision, figures, summary."""
from __future__ import annotations

import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import hessian_early_stop_mvp as HM     # noqa: E402
from moe_calib import early_stop as ES  # noqa: E402

p, jdump, jload = HM.p, HM.jdump, HM.jload
LAYERS = HM.LAYERS


def main():
    prot = jload("protocol.json")
    bench = jload("logs", "bench_raw.json")
    C = bench["conditions"]
    med = {c: C[c]["median_s"] for c in C}
    ev = bench["cuda_events_full"]["total"]
    e2e = {c: dict(replay_s=sum(v["replay_s"] for v in d.values()), solve_s=sum(v["solve_s"] for v in d.values()),
                   peak_gpu_gb=max(v["peak_gpu_gb"] for v in d.values()), peak_rss_gb=max(v["peak_rss_gb"] for v in d.values()))
           for c, d in bench["end_to_end"].items()}
    for c in e2e:
        e2e[c]["total_s"] = e2e[c]["replay_s"] + e2e[c]["solve_s"]
    cnt = bench["counts"]
    def frac(c):
        d = sum(cnt[f"L{L}_{c}"]["rows_done"] for L in LAYERS); s = sum(cnt[f"L{L}_{c}"]["rows_skipped"] for L in LAYERS)
        return s / (d + s)
    hook = med["noacc"] - med["forward"]
    accum = med["full"] - med["noacc"]
    cm = json.load(open(os.path.join(ROOT, "runs_ebss_official_validation", "logs", "capture_manifest.json")))
    capture_128 = cm["per_row_compute_s_mean"] * 128
    timing = dict(
        kind="MEASURED (median of 3 interleaved repetitions, 4 layers x 128 sequences, cuda-synchronized)",
        model_forward_s=med["forward"], hook_overhead_s=hook, accumulation_s=accum,
        accumulation_split_cuda_events=dict(conversion_s=ev["conversion_s"], gemm_s=ev["gemm_s"], update_s=ev["update_s"],
                                            note="GPU time from CUDA events in a separate instrumented full run"),
        total_hessian_replay_s=med["full"], replay_range_s={c: [C[c]["min_s"], C[c]["max_s"]] for c in C},
        gptq_solve_s=e2e["full"]["solve_s"], end_to_end_replay_plus_solve_s=e2e["full"]["total_s"],
        accumulation_fraction_of_replay=accum / med["full"],
        accumulation_fraction_of_end_to_end=accum / e2e["full"]["total_s"],
        context=dict(canonical_b1_capture_for_128_rows_s=capture_128,
                     note="activation capture (24-layer forward per calibration row) is upstream and unaffected by "
                          "stopping; the historical quantize_total (~347 s) additionally includes E3 scoring"))
    jdump(timing, p("timing_breakdown.json"))
    fm = {}
    for c in ("full", "stop50", "stop75"):
        fm[c] = dict(expert_ids_stopped_frac=0.0 if c == "full" else (0.5 if c == "stop50" else 0.75),
                     accumulation_rows_skipped_frac=frac(c), replay_median_s=med[c], replay_range_s=[C[c]["min_s"], C[c]["max_s"]],
                     replay_reduction=1 - med[c] / med["full"], speedup_H=med["full"] / med[c],
                     e2e_measured=e2e[c], e2e_reduction=1 - e2e[c]["total_s"] / e2e["full"]["total_s"])
    fmb = jload("fixed_mask_benchmark.json")
    outputs_ok = all(v["outputs_identical"] for v in fmb["validity"].values())
    baseline_ok = all(v.get("equals_historical_weights", True) for v in bench["end_to_end"]["full"].values())
    fmb.update(dict(conditions=fm, outputs_identical_all_layers=outputs_ok, full_equals_historical_weights=baseline_ok,
                    ceiling=dict(skip_all_accumulation_replay_reduction=accum / med["full"],
                                 skip_all_accumulation_e2e_reduction=accum / e2e["full"]["total_s"])))
    jdump(fmb, p("fixed_mask_benchmark.json"))
    sg = prot["experiment1"]["systems_gate"]
    systems_ok = bool(fm["stop50"]["replay_reduction"] >= sg["min_hessian_replay_reduction"]
                      and fm["stop50"]["e2e_reduction"] >= sg["min_end_to_end_reduction"] and outputs_ok)
    # ---------------- science
    sp = jload("stopping_policy_results.json")
    val = jload("hessian_checkpoint_validation.json")
    pol, fixed = sp["policies"], sp["fixed"]
    primary = {k: v for k, v in pol.items() if v["primary"]}
    passing = [k for k, v in primary.items() if v["passes_scientific_gate"]]
    science = dict(any_rule_passes=bool(passing), passing_rules=passing,
                   any_passing_rule_beats_frontier=any(primary[k]["beats_frontier"] for k in passing))
    costm = dict(forward_s=med["forward"], hook_overhead_s=hook, accumulation_s=accum, solve_s=e2e["full"]["solve_s"])
    qc = {}
    for name, o in list({f"fixed_B{B}": fixed[f"fixed_B{B}"] for B in (32, 64, 128)}.items()) + list(pol.items()):
        pr = ES.project_time(costm, o["work_fraction_done"])
        qc[name] = dict(mean_err=o["mean_err"], mean_rel_degradation=o["mean_rel_degradation"], p90_err=o["p90_err"],
                        p90_rel_degradation=o["p90_rel_degradation"], worst_expert_degradation=o["worst_expert_degradation"],
                        false_safe_1pct=o["false_safe_1pct"], false_safe_2pct=o["false_safe_2pct"],
                        work_fraction_skipped=o["work_fraction_skipped"], n_stop=o["n_stop"],
                        mean_effective_budget=o["mean_effective_budget"], projected=pr,
                        primary=o.get("primary", False), beats_frontier=o.get("beats_frontier"),
                        passes_scientific_gate=o.get("passes_scientific_gate"))
    qc["measured_fixed_masks"] = {c: dict(kind="MEASURED", replay_s=fm[c]["replay_median_s"], total_s=fm[c]["e2e_measured"]["total_s"],
                                          rows_skipped=fm[c]["accumulation_rows_skipped_frac"]) for c in fm}
    jdump(qc, p("quality_cost_comparison.json"))
    blocked = not (val["passed"] and outputs_ok and baseline_ok)
    verdict = ES.classify(science, systems_ok, blocked)
    decision = dict(verdict=verdict, science=science, systems=dict(passed=systems_ok, gate=sg,
                    stop50_replay_reduction=fm["stop50"]["replay_reduction"], stop50_e2e_reduction=fm["stop50"]["e2e_reduction"],
                    stop75_replay_reduction=fm["stop75"]["replay_reduction"], stop75_e2e_reduction=fm["stop75"]["e2e_reduction"],
                    ceiling_e2e=accum / e2e["full"]["total_s"]),
                    correctness=dict(checkpoints=val["passed"], outputs_identical=outputs_ok, full_equals_historical=baseline_ok),
                    gates=prot["experiment2"]["scientific_gate"])
    jdump(decision, p("decision.json"))
    summ = dict(verdict=verdict, decision=decision, timing=timing, fixed_masks=fm, quality_cost=qc)
    jdump(summ, p("summary.json"))
    figures(timing, fm, sp)
    write_md(summ)
    print("[report]", verdict, science, "systems", systems_ok)


def figures(timing, fm, sp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(p("figures"), exist_ok=True)
    def save(fig, nm):
        fig.tight_layout(); fig.savefig(p("figures", nm + ".png"), dpi=140); fig.savefig(p("figures", nm + ".pdf")); plt.close(fig)
    st = jload("expert_statistics.json")
    rb = jload("reconstruction_by_budget.json")
    # 1 timing
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    comps = [("model forward", timing["model_forward_s"]), ("hook overhead", timing["hook_overhead_s"]),
             ("Hessian accumulation", timing["accumulation_s"]), ("GPTQ solve", timing["gptq_solve_s"])]
    b = 0
    for i, (lab, v) in enumerate(comps):
        ax[0].barh(0, v, left=b, label=f"{lab} {v:.1f}s"); b += v
    ax[0].set_yticks([]); ax[0].set(xlabel="seconds (4 layers, B=128, measured)", title="Pipeline breakdown (full)"); ax[0].legend(fontsize=8)
    cs = ["full", "stop50", "stop75"]
    ax[1].bar(np.arange(3) - .2, [fm[c]["replay_median_s"] for c in cs], .4, yerr=[[fm[c]["replay_median_s"] - fm[c]["replay_range_s"][0] for c in cs],
              [fm[c]["replay_range_s"][1] - fm[c]["replay_median_s"] for c in cs]], label="Hessian replay")
    ax[1].bar(np.arange(3) + .2, [fm[c]["e2e_measured"]["total_s"] for c in cs], .4, label="replay + GPTQ solve")
    for i, c in enumerate(cs):
        ax[1].text(i, fm[c]["e2e_measured"]["total_s"], f"{fm[c]['e2e_reduction'] * 100:+.1f}%", ha="center", va="bottom", fontsize=8)
    ax[1].set_xticks(range(3), [f"{c}\n({fm[c]['accumulation_rows_skipped_frac'] * 100:.0f}% rows skipped)" for c in cs])
    ax[1].set(ylabel="seconds (measured)", title="Fixed-mask benchmark"); ax[1].legend()
    save(fig, "fig1_timing")
    # 2 drift vs budget
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, f in enumerate(("trace", "frob")):
        for pj, col in (("gu", "C0"), ("down", "C3")):
            data = [[v[pj][f] for v in st["per_checkpoint"][str(B)].values() if np.isfinite(v[pj][f])] for B in (32, 64, 128)]
            bp = ax[i].boxplot(data, positions=np.arange(3) + (0 if pj == "gu" else .3), widths=.25, patch_artist=True,
                               boxprops=dict(facecolor=col, alpha=.4), showfliers=False)
        ax[i].set_xticks(np.arange(3) + .15, ["B32 vs 16", "B64 vs 32", "B128 vs 64"]); ax[i].set_yscale("log")
        ax[i].set(title=f"{f} drift (blue gu, red down)", ylabel="normalized drift")
    save(fig, "fig2_drift_vs_budget")
    # 3 drift vs degradation
    keys = [tuple(k) for k in rb["keys"]]
    e128 = np.array(rb["per_expert"]["B128"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, B in enumerate((32, 64)):
        d = (np.array(rb["per_expert"][f"B{B}"]) - e128) / e128 * 100
        x = np.array([max(st["per_checkpoint"][str(B)][f"{L}.{e}"]["gu"]["frob"], st["per_checkpoint"][str(B)][f"{L}.{e}"]["down"]["frob"])
                      for L, e in keys])
        ax[i].scatter(x, d, s=8, c=[LAYERS.index(L) for L, e in keys], cmap="viridis")
        ax[i].axhline(1, color="r", ls="--", lw=.8); ax[i].set_xscale("log")
        ax[i].set(xlabel=f"max(gu, down) Frobenius drift at B{B}", ylabel=f"E_B{B} vs E_B128 (%)", title=f"stopping at B{B}")
    save(fig, "fig3_drift_vs_degradation")
    # 4 false-safe vs skipped work, 5 frontier
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig5, ax5 = plt.subplots(figsize=(6.5, 4.5))
    mk = dict(trace="o", frob="s", frob_exposure="^", exposure="x")
    for name, o in sp["policies"].items():
        ax.scatter(o["work_fraction_skipped"] * 100, o["false_safe_1pct"]["rate"] * 100, marker=mk[o["rule"]],
                   s=60 if o["primary"] else 25, label=name)
        ax5.scatter(o["work_fraction_skipped"] * 100, o["mean_rel_degradation"] * 100, marker=mk[o["rule"]],
                    s=60 if o["primary"] else 25, label=name)
    fx = sorted((v["work_fraction_skipped"] * 100, v["mean_rel_degradation"] * 100) for v in sp["fixed"].values())
    ax5.plot([a for a, b in fx], [b for a, b in fx], "k-o", label="fixed B128 / B64 / B32")
    ax.axhline(10, color="r", ls="--", lw=.8)
    ax.set(xlabel="Hessian accumulation work skipped (%)", ylabel="false-safe stops (> 1 %) among stopped (%)",
           title="False-safe rate vs skipped work"); ax.legend(fontsize=6)
    ax5.axhline(0.5, color="r", ls="--", lw=.8)
    ax5.set(xlabel="Hessian accumulation work skipped (%)", ylabel="mean error vs B128 (%)", title="Quality-cost frontier")
    ax5.legend(fontsize=6)
    save(fig, "fig4_false_safe_vs_work"); save(fig5, "fig5_quality_cost_frontier")


def write_md(s):
    t, fm, qc = s["timing"], s["fixed_masks"], s["quality_cost"]
    L = ["# Hessian early-stopping MVP", "", f"**Verdict (frozen rule): `{s['verdict']}`**", "",
         "## Measured timing (4 layers, B=128)", "",
         f"forward {t['model_forward_s']:.2f} s | hook overhead {t['hook_overhead_s']:.2f} s | accumulation {t['accumulation_s']:.2f} s "
         f"(events: conversion {t['accumulation_split_cuda_events']['conversion_s']:.2f}, GEMM {t['accumulation_split_cuda_events']['gemm_s']:.2f}, "
         f"update {t['accumulation_split_cuda_events']['update_s']:.2f}) | replay {t['total_hessian_replay_s']:.2f} s | GPTQ solve {t['gptq_solve_s']:.1f} s", "",
         "| Condition | rows skipped | replay (s) | replay reduction | replay+solve (s) | e2e reduction | peak GPU (GB) |",
         "|---|---|---|---|---|---|---|"]
    for c, v in fm.items():
        L.append(f"| {c} | {v['accumulation_rows_skipped_frac'] * 100:.1f} % | {v['replay_median_s']:.2f} | {v['replay_reduction'] * 100:+.1f} % | "
                 f"{v['e2e_measured']['total_s']:.1f} | {v['e2e_reduction'] * 100:+.2f} % | {v['e2e_measured']['peak_gpu_gb']:.2f} |")
    L += ["", "## Quality-cost (offline; times are PROJECTED from the measured cost model)", "",
          "| Strategy | mean err | Δ vs B128 | p90 Δ | false-safe >1 % | work skipped | projected total (s) |", "|---|---|---|---|---|---|---|"]
    for k, v in qc.items():
        if k == "measured_fixed_masks":
            continue
        fs = v["false_safe_1pct"]
        L.append(f"| {k}{' (primary)' if v['primary'] else ''} | {v['mean_err']:.6f} | {v['mean_rel_degradation'] * 100:+.2f} % | "
                 f"{v['p90_rel_degradation'] * 100:+.2f} % | {fs['n_false']}/{fs['n_stopped']} | {v['work_fraction_skipped'] * 100:.1f} % | "
                 f"{v['projected']['total_s']:.1f} |")
    open(p("summary.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
