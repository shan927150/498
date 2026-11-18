"""Aggregate the layer-pooled Hessian shrinkage MVP: Stage 1 summaries, Stage 2 quality tables, controls, quality-cost,
timing, correctness, frozen verdict, figures and summary.md. Reads only files written by hessian_pool_shrinkage_mvp.py."""
from __future__ import annotations

import glob, json, os, subprocess, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import hessian_pool_shrinkage_mvp as M   # noqa: E402
from moe_calib import hessian_shrinkage as HS  # noqa: E402

p, jload, jdump = M.p, M.jload, M.jdump
LAYERS = M.LAYERS
NBOOT = 2000


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ----------------------------------------------------------------------------- Stage 1 summaries
def stage1_tables():
    cm = jload("covariance_mismatch.json")["records"]
    SL = M.sel_list()
    tags = ["raw"] + [f"{t}_{a}" for a in HS.ALPHAS[1:] for t in HS.TARGETS] + ["pure_prior", "pure_prior_untraced"]
    out = {}
    for B in HS.BUDGETS:
        rb = {}
        for tg in tags:
            v = {m: np.array([cm[f"{L}.{e}.{B}"]["metrics"][tg][m] for L, e, _ in SL]) for m in ("D_F", "D_diag", "align16")}
            raw = np.array([cm[f"{L}.{e}.{B}"]["metrics"]["raw"]["D_F"] for L, e, _ in SL])
            rb[tg] = dict(mean_D_F=float(v["D_F"].mean()), median_D_F=float(np.median(v["D_F"])),
                          mean_D_diag=float(v["D_diag"].mean()), mean_align16=float(v["align16"].mean()),
                          rel_vs_raw=float(1 - v["D_F"].mean() / raw.mean()), n_improved=int((v["D_F"] < raw).sum()),
                          by_layer={str(L): float(v["D_F"][[i for i, x in enumerate(SL) if x[0] == L]].mean()) for L in LAYERS},
                          by_stratum={st: float(v["D_F"][[i for i, x in enumerate(SL) if x[2] == st]].mean())
                                      for st in ("low", "median", "high")},
                          per_expert={f"{L}.{e}": float(v["D_F"][i]) for i, (L, e, _) in enumerate(SL)})
        best = min([t for t in tags if t.startswith("pooled_")], key=lambda t: rb[t]["mean_D_F"])
        best_id = min([t for t in tags if t.startswith("identity_")], key=lambda t: rb[t]["mean_D_F"])
        out[str(B)] = dict(targets=rb, best_pooled_alpha_by_mean_D_F=best, best_identity_alpha_by_mean_D_F=best_id)
    return out


# ----------------------------------------------------------------------------- Stage 2 loading
def load_split(split="confirm"):
    """{cand: {metric: [12] ratio-of-sums per expert}}, per-seq arrays for bootstrap, expert list."""
    per, seqs, experts = {}, {}, []
    for L in LAYERS:
        z = np.load(p("results", split, f"L{L}.npz"))
        meta = jload("results", split, f"L{L}_meta.json")
        experts += [(L, int(e)) for e in z["experts"]]
        seqs[L] = z
        for c in meta["candidates"]:
            for m in ("J", "U", "gate", "up", "e3"):
                v = z[f"num_{c}_{m}"].sum(0) / (z[f"den_{m}"].sum(0) + 1e-30)
                per.setdefault(c, {}).setdefault(m, []).extend(v.tolist())
    per = {c: {m: np.array(v) for m, v in d.items()} for c, d in per.items()}
    return per, seqs, experts


def boot_rel(seqs, a, b, m="J", seed=0):
    """Paired sequence bootstrap (independently per layer) of 1 - mean_e J_a / mean_e J_b over the 12 experts."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(NBOOT):
        ja, jb = [], []
        for L in LAYERS:
            z = seqs[L]
            idx = rng.integers(0, z["tokens"].shape[0], z["tokens"].shape[0])
            den = z[f"den_{m}"][idx].sum(0) + 1e-30
            ja += list(z[f"num_{a}_{m}"][idx].sum(0) / den); jb += list(z[f"num_{b}_{m}"][idx].sum(0) / den)
        out.append(1 - np.mean(ja) / np.mean(jb))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def rel(a, b):
    return float(1 - np.mean(a) / np.mean(b))


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(p("figures"), exist_ok=True)
    proto = jload("protocol.json")
    s1 = jload("stage1_decision.json")
    asel = jload("alpha_selection.json")["selected"]
    st1 = stage1_tables()
    elig = s1["eligible_budgets"]
    SL = M.sel_list()
    strata = np.array([x[2] for x in SL]); lay = np.array([x[0] for x in SL])
    have2 = os.path.exists(p("results", "confirm", "L16.npz"))
    summary = dict(stage1_label=s1["label"], selected_alpha=asel, eligible_budgets=elig)
    per_budget, qres = {}, {}
    if have2:
        per, seqs, experts = load_split("confirm")
        assert [(L, e) for L, e, _ in SL] == experts
        # ---- quantization results (all candidates, all metrics)
        qres = dict(split="confirm", experts=[f"{L}.{e}" for L, e in experts], strata=strata.tolist(),
                    metric_definitions=proto["stage2"]["primary_metric"],
                    candidates={c: {m: dict(per_expert=v.tolist(), mean=float(v.mean()), median=float(np.median(v)),
                                            by_layer={str(L): float(v[lay == L].mean()) for L in LAYERS})
                                    for m, v in d.items()} for c, d in per.items()})
        cm = jload("covariance_mismatch.json")["records"]
        for B in elig:
            J = {k: per[f"B{B}_{c}"]["J"] for k, c in (("std", "std"), ("pool", "pool"), ("identity", "identity"),
                                                         ("tuned", "tuned"))}
            bc = HS.budget_criteria(J, lay)
            d = J["pool"] / J["std"] - 1
            al = asel[str(B)]
            dDF = np.array([cm[f"{L}.{e}.{B}"]["metrics"][f"pooled_{al}"]["D_F"] / cm[f"{L}.{e}.{B}"]["metrics"]["raw"]["D_F"] - 1
                            for L, e, _ in SL])
            bc.update(
                alpha=al, mean_J=dict((k, float(v.mean())) for k, v in J.items()),
                ci95_rel_vs_std=boot_rel(seqs, f"B{B}_pool", f"B{B}_std"),
                ci95_rel_vs_identity=boot_rel(seqs, f"B{B}_pool", f"B{B}_identity"),
                ci95_rel_vs_tuned=boot_rel(seqs, f"B{B}_pool", f"B{B}_tuned"),
                identity_rel_vs_std=rel(J["identity"], J["std"]), tuned_rel_vs_std=rel(J["tuned"], J["std"]),
                per_expert_rel_change=dict(zip([f"{L}.{e}" for L, e in experts], d.tolist())),
                by_stratum={st: float(1 - J["pool"][strata == st].mean() / J["std"][strata == st].mean())
                            for st in ("low", "median", "high")},
                secondary={m: dict(pool_vs_std=rel(per[f"B{B}_pool"][m], per[f"B{B}_std"][m]),
                                   identity_vs_std=rel(per[f"B{B}_identity"][m], per[f"B{B}_std"][m]),
                                   tuned_vs_std=rel(per[f"B{B}_tuned"][m], per[f"B{B}_std"][m]))
                           for m in ("U", "gate", "up", "e3")},
                spearman_dDF_vs_dJ=spearman(dDF, d), dDF_per_expert=dDF.tolist(),
                gap_recovery=dict(pool=HS.gap_recovery(J["std"].mean(), J["pool"].mean(), per[f"B{2 * B}_std"]["J"].mean()),
                                  identity=HS.gap_recovery(J["std"].mean(), J["identity"].mean(),
                                                           per[f"B{2 * B}_std"]["J"].mean()),
                                  tuned=HS.gap_recovery(J["std"].mean(), J["tuned"].mean(), per[f"B{2 * B}_std"]["J"].mean()),
                                  doubling_rel_gain=rel(per[f"B{2 * B}_std"]["J"], J["std"])))
            per_budget[str(B)] = bc
        qres["per_budget"] = per_budget
        jdump(qres, p("quantization_results.json"))
        jdump({B: dict(alpha=v["alpha"], identity_mean_J=v["mean_J"]["identity"], pool_mean_J=v["mean_J"]["pool"],
                       std_mean_J=v["mean_J"]["std"], pool_vs_identity_rel=v["rel_vs_identity"],
                       ci95=v["ci95_rel_vs_identity"], identity_vs_std_rel=v["identity_rel_vs_std"],
                       equivalent_percdamp=(1 + v["alpha"]) / (1 - v["alpha"]),
                       note="identity shrinkage at alpha with percdamp 1.0 equals raw H with percdamp (1+a)/(1-a) up to a "
                            "global scale and fp rounding (gate/up only)")
               for B, v in per_budget.items()}, p("identity_control_results.json"))
        jdump({B: dict(percdamp=M.TUNED_DAMP[int(B)], tuned_mean_J=v["mean_J"]["tuned"], pool_mean_J=v["mean_J"]["pool"],
                       pool_vs_tuned_rel=v["rel_vs_tuned"], ci95=v["ci95_rel_vs_tuned"], tuned_vs_std_rel=v["tuned_rel_vs_std"],
                       source=proto["tuned_damping"]["source"]) for B, v in per_budget.items()},
              p("tuned_damping_control.json"))
    # ---- timing
    tr = jload("logs", "timing_raw.json") if os.path.exists(p("logs", "timing_raw.json")) else None
    qm = jload("logs", "quantize_meta.json") if os.path.exists(p("logs", "quantize_meta.json")) else None
    timing = {}
    if tr:
        med = {c: {L: float(np.median(v)) for L, v in d.items()} for c, d in tr["conditions"].items()}
        for L in map(str, LAYERS):
            fwd, pl, ex, both = (med[c][L] for c in ("forward", "pooled", "experts", "experts+pooled"))
            solve6 = qm["timing"][f"L{L}.B16_std"]["solve_s"] if qm else None
            solve60 = solve6 * 10 if solve6 else None
            timing[L] = dict(forward_s=fwd, expert_hessian_s=ex - fwd, pooled_overhead_s=both - ex,
                             pooled_only_overhead_s=pl - fwd, pooled_gemm_cuda_event_s=tr["pooled_gemm_cuda_event_s"][L],
                             replay_with_experts_s=ex, solve_6_experts_measured_s=solve6,
                             solve_60_experts_projected_s=solve60,
                             mix_s_per_candidate={k.split(".", 1)[1]: v["mix_s"] for k, v in (qm or {}).get("timing", {}).items()
                                                  if k.startswith(f"L{L}.")},
                             pooled_overhead_frac_of_pipeline=(max(both - ex, 0) / (ex + solve60)) if solve60 else None,
                             raw_repetitions={c: tr["conditions"][c][L] for c in tr["conditions"]})
        cost_ok = all(v["pooled_overhead_frac_of_pipeline"] is not None and
                      v["pooled_overhead_frac_of_pipeline"] <= HS.RULES["s2_cost_max_overhead"] for v in timing.values())
    else:
        cost_ok = False
    ev = {L: jload("results", "confirm", f"L{L}_meta.json") for L in LAYERS} if have2 else {}
    flops = dict(pooled_per_token=2048 ** 2, experts_per_token=4 * (2048 ** 2 + 1408 ** 2),
                 pooled_frac_of_expert_accumulation_flops=2048 ** 2 / (4 * (2048 ** 2 + 1408 ** 2)))
    jdump(dict(per_layer=timing, flops=flops,
               mvp_cost=dict(quantize=qm["timing"] if qm else None, evaluation_s={str(L): v["seconds"] for L, v in ev.items()},
                             peak_gpu_gb=max([qm["peak_gpu_gb"] if qm else 0] + [v["peak_gpu_gb"] for v in ev.values()] +
                                             [jload("logs", f"hessians_L{L}.json")["peak_gpu_gb"] for L in LAYERS]),
                             peak_rss_gb=max([qm["peak_rss_gb"] if qm else 0] + [v["peak_rss_gb"] for v in ev.values()] +
                                             [jload("logs", f"hessians_L{L}.json")["peak_rss_gb"] for L in LAYERS]),
                             replay_with_pooled_and_alignment_s={str(L): jload("logs", f"hessians_L{L}.json")["replay_s"]
                                                                 for L in LAYERS}),
               deployment_cost="one routed Hessian per expert + one pooled 2048x2048 Hessian per layer, one fp64 mix per "
                               "expert, ONE GPTQ solve per projection (frozen alpha): extra = pooled accumulation + mixing",
               cost_criterion_ok=bool(cost_ok)), p("timing_breakdown.json"))
    # ---- subset
    sub = jload("subset_layer_output.json") if os.path.exists(p("subset_layer_output.json")) else None
    subset_ok = None if sub is None else all(v["subset_rel_err"] <= v["std_rel_err"] for v in sub["layers"].values())
    # ---- correctness
    hv = jload("pooled_hessian_validation.json")
    hl = {L: jload("logs", f"hessians_L{L}.json") for L in LAYERS}
    cn = jload("logs", "candidate_numerics.json")
    rm = jload("logs", "reference_meta.json")
    inv = jload("artifact_inventory.json")
    git_gptq = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "moe_calib/gptq.py"], cwd=ROOT).returncode == 0
    hist_ok = all(M.sha256_file(os.path.join(ROOT, f)) == h for f, h in inv["historical_inputs_sha256"].items())
    qc = qm["checks"] if qm else {}
    exposure_ok = all(v > 0 for B in jload("selected_experts.json")["exposure"].values() for d in B.values() for v in d.values())
    chk = {
        "01_chronological_prefixes": bool(hv["passed"] and all(hl[L]["drift_stats_equal_early_stop_bitwise"] for L in LAYERS)),
        "02_pooled_uses_actual_moe_input": all(hl[L]["pooled_input_equals_router_input"] and hl[L]["n_alignment_checks"] > 0
                                               for L in LAYERS),
        "03_pooled_counts_each_token_once": all(hl[L]["pooled_tokens_ok"] and hl[L]["routed_sum_equals_topk_times_tokens"]
                                                for L in LAYERS),
        "04_expert_hessians_natural_routing": all(hl[L]["expert_counts_match_routing_profile"] for L in LAYERS),
        "05_calibration_tune_disjoint": bool(rm["calibration_disjoint"]),
        "06_alpha0_reproduces_hessian": bool(cn["alpha0_returns_raw_bitwise_and_no_mutation"]),
        "07_alpha0_reproduces_gptq_bitwise": bool(qc) and all(v.get(f"B{B}_alpha0_equals_std_bitwise", False)
                                                              for v in qc.values() for B in elig)
                                             and all(hl[L]["B128_weights_equal_historical"] for L in LAYERS),
        "08_pooled_trace_preserved": all(v["trace_rel_err"] < 1e-6 for k, v in cn["per_candidate"].items() if "pooled" in k),
        "09_identity_trace_preserved": all(v["trace_rel_err"] < 1e-6 for k, v in cn["per_candidate"].items() if "identity" in k),
        "10_finite_symmetric": all(v["finite"] and v["sym_rel"] < 1e-12 for v in cn["per_candidate"].values()),
        "11_no_negative_eigenvalues": all(v["psd_ok"] for v in cn["per_candidate"].values()),
        "12_gptq_solver_unchanged": bool(git_gptq and M.sha256_file(os.path.join(ROOT, "moe_calib/gptq.py")) ==
                                         proto["code_sha256"]["moe_calib/gptq.py"]),
        "13_down_unchanged_ABC": bool(qc) and all(v.get(f"B{B}_down_identical_std_pool_identity", False)
                                                  for v in qc.values() for B in elig),
        "14_tuned_damping_documented": M.TUNED_DAMP == {16: 3.0, 32: 2.0} and all(
            jload("logs", "quantize_meta.json")["candidates"][f"B{B}_tuned"]["percdamp"] == M.TUNED_DAMP[B] for B in elig)
            if qm else False,
        "15_bf16_weights_restored": all(hl[L]["layer_weights_unchanged"] for L in LAYERS) and
                                    all(v.get("model_weights_unchanged", False) for v in qc.values()),
        "16_no_test_data_in_alpha_selection": bool(s1["decided_before_any_quantization"] and s1["confirm_results_absent"]),
        "17_router_and_experts_match": all(hl[L]["outputs_identical_with_hooks"] for L in LAYERS),
        "18_no_rtn_fallback": bool(exposure_ok),
        "19_gpu_memory_within_20gb": json.load(open(p("timing_breakdown.json")))["mvp_cost"]["peak_gpu_gb"] < 20.0,
        "20_historical_files_unchanged": bool(hist_ok),
    }
    tl = p("logs", "pytest_full.log")
    tests_ok = os.path.exists(tl) and " passed" in open(tl).read() and "failed" not in open(tl).read()
    chk["21_all_tests_pass"] = bool(tests_ok)
    correctness_ok = all(chk.values())
    jdump(dict(checks=chk, passed=correctness_ok,
               notes={"07": "alpha=0 path through shrink() == standard weights bitwise at the eligible budgets; standard "
                            "B128 == historical FR s0 weights bitwise; standard B32/B64/B128 E3 == early-stop logs bitwise",
                      "15": "evaluation never installs weights (fp32 copies); layer digests checked before/after",
                      "17": "layer outputs (and therefore routing) bitwise identical with and without the pooled hook"}),
          p("correctness_results.json"))
    # ---- decision
    v = HS.classify(s1["label"], {int(B): x for B, x in per_budget.items()}, subset_ok, correctness_ok, cost_ok)
    decision = dict(v, stage1=s1["label"], per_budget_criteria={B: x["criteria"] for B, x in per_budget.items()},
                    c7_subset_ok=subset_ok, c8_correctness_ok=correctness_ok, c9_cost_ok=cost_ok,
                    rules=HS.RULES, recommendation_A_executed=False, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(decision, p("decision.json"))
    jdump(dict(stage1=st1, stage1_decision=s1["label"]), p("logs", "stage1_tables.json"))
    # quality-cost table
    if have2:
        rows = []
        for B in elig:
            for nm, c, strat, pd in (("Standard GPTQ", "std", "Routed", 1.0), ("Identity shrinkage", "identity",
                                     f"Expert -> Identity (a={asel[str(B)]})", 1.0),
                                    ("Layer-pooled shrinkage", "pool", f"Expert -> Pooled (a={asel[str(B)]})", 1.0),
                                    ("Tuned damping", "tuned", "Routed", M.TUNED_DAMP[B])):
                j = per[f"B{B}_{c}"]["J"]
                rows.append(dict(method=nm, budget=B, hessian=strat, percdamp=pd, mean_J=float(j.mean()),
                                 rel_vs_std_same_budget=rel(j, per[f"B{B}_std"]["J"])))
        for B in (64, 128):
            j = per[f"B{B}_std"]["J"]
            rows.append(dict(method="Standard GPTQ", budget=B, hessian="Routed", percdamp=1.0, mean_J=float(j.mean()),
                             rel_vs_std_B16=rel(j, per["B16_std"]["J"]), rel_vs_std_B32=rel(j, per["B32_std"]["J"])))
        jdump(dict(rows=rows, gap_recovery={B: x["gap_recovery"] for B, x in per_budget.items()},
                   cost=json.load(open(p("timing_breakdown.json")))["per_layer"],
                   note="different budgets carry different calibration cost; B64/B128 are references only"),
              p("quality_cost_comparison.json"))
    summary.update(decision=v, correctness_ok=correctness_ok, cost_ok=cost_ok, subset=None if sub is None else
                   {L: dict(rel_change=x["rel_change"], selected_share=x["selected_share_of_diag_error"])
                    for L, x in sub["layers"].items()},
                   stage1={B: dict(raw=st1[B]["targets"]["raw"]["mean_D_F"],
                                   **{f"{t}_{a}": st1[B]["targets"][f"{t}_{a}"]["mean_D_F"] for t in HS.TARGETS
                                      for a in HS.ALPHAS[1:]}) for B in st1},
                   stage2={B: dict(mean_J=x["mean_J"], rel_vs_std=x["rel_improvement_vs_std"],
                                   rel_vs_identity=x["rel_vs_identity"], rel_vs_tuned=x["rel_vs_tuned"],
                                   n_improved=x["n_improved"], worst=x["worst_expert_rel_change"],
                                   by_layer=x["layer_rel_improvement"], by_stratum=x["by_stratum"],
                                   ci95_vs_std=x["ci95_rel_vs_std"], gap=x["gap_recovery"])
                           for B, x in per_budget.items()})
    jdump(summary, p("summary.json"))
    figures(st1, per if have2 else None, per_budget, asel, elig, timing)
    print(json.dumps(dict(verdict=v, checks_failed=[k for k, x in chk.items() if not x]), indent=1))


def figures(st1, per, per_budget, asel, elig, timing):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = dict(raw="#555555", pooled="#1f77b4", identity="#ff7f0e", untraced="#2ca02c", tuned="#9467bd")

    def save(fig, n):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(p("figures", f"{n}.{ext}"), dpi=150)
        plt.close(fig)
    # fig1: mismatch at B16 / B32, raw vs pooled(alpha*) vs identity(alpha*)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, B in zip(axs, ("16", "32")):
        al = asel.get(B) or 0.3
        tg = st1[B]["targets"]
        keys = [("raw", "raw"), (f"pooled_{al}", "pooled"), (f"identity_{al}", "identity")]
        for i, (k, c) in enumerate(keys):
            vals = list(tg[k]["per_expert"].values())
            ax.bar(i, np.mean(vals), color=col[c], alpha=.7)
            ax.scatter(np.full(len(vals), i) + np.linspace(-.25, .25, len(vals)), vals, s=8, color="k")
        ax.set_xticks(range(3), ["raw", f"pooled a={al}", f"identity a={al}"])
        ax.set_title(f"B{B}: gu second-moment mismatch vs tune ref"); ax.set_ylabel("rel. Frobenius D_F")
    save(fig, "fig1_hessian_mismatch")
    # fig2: alpha vs D_F
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, B in zip(axs, ("16", "32")):
        tg = st1[B]["targets"]
        for t in HS.TARGETS:
            ys = [tg["raw"]["mean_D_F"]] + [tg[f"{t}_{a}"]["mean_D_F"] for a in HS.ALPHAS[1:]]
            ax.plot(HS.ALPHAS, ys, "o-", color=col[t], label=t + (" (diag.)" if t == "untraced" else ""))
            if t == "pooled":
                for ex in tg["raw"]["per_expert"]:
                    ax.plot(HS.ALPHAS, [tg["raw"]["per_expert"][ex]] + [tg[f"pooled_{a}"]["per_expert"][ex]
                                                                          for a in HS.ALPHAS[1:]],
                            color=col[t], alpha=.15, lw=.8)
        ax.axhline(tg["pure_prior"]["mean_D_F"], ls=":", color=col["pooled"], label="pure prior a=1 (diag.)")
        ax.set_xlabel("alpha"); ax.set_title(f"B{B}"); ax.set_ylabel("mean D_F")
    axs[0].legend(fontsize=7)
    save(fig, "fig2_alpha_vs_mismatch")
    if per is None:
        return
    # fig3: quality at B16/B32 relative to standard
    fig, ax = plt.subplots(figsize=(7, 3.6))
    meths = [("pool", "pooled"), ("identity", "identity"), ("tuned", "tuned")]
    for bi, B in enumerate(elig):
        s = per[f"B{B}_std"]["J"]
        for mi, (c, cc) in enumerate(meths):
            d = 100 * (per[f"B{B}_{c}"]["J"] / s - 1)
            x = bi * 4 + mi
            ax.bar(x, 100 * (per[f"B{B}_{c}"]["J"].mean() / s.mean() - 1), color=col[cc], alpha=.7,
                   label=cc if bi == 0 else None)
            ax.scatter(np.full(len(d), x) + np.linspace(-.25, .25, len(d)), d, s=8, color="k")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks([bi * 4 + 1 for bi in range(len(elig))], [f"B{B} (a*={asel[str(B)]})" for B in elig])
    ax.set_ylabel("J vs standard GPTQ, same budget (%)"); ax.legend(fontsize=8)
    ax.set_title("Complete-expert c^2-weighted output error (confirm split; dots = experts)")
    save(fig, "fig3_quantization_quality")
    # fig4: J vs budget
    fig, ax = plt.subplots(figsize=(6, 3.8))
    Bs = list(HS.BUDGETS)
    ax.plot(Bs, [per[f"B{B}_std"]["J"].mean() for B in Bs], "o-", color=col["raw"], label="standard GPTQ")
    for c, cc in meths:
        ax.plot(elig, [per[f"B{B}_{c}"]["J"].mean() for B in elig], "s", color=col[cc], label=cc, ms=7)
    ax.set_xscale("log", base=2); ax.set_xticks(Bs, [str(b) for b in Bs]); ax.set_xlabel("calibration sequences")
    ax.set_ylabel("mean J (12 experts)"); ax.legend(fontsize=8); ax.set_title("Quality vs calibration budget")
    save(fig, "fig4_quality_vs_budget")
    # fig5: quality vs estimated compute cost (deployment: replay ~ B/128 x replay + solve; pooled adds its overhead)
    if timing:
        rep = np.mean([v["replay_with_experts_s"] for v in timing.values()])
        sol = np.mean([v["solve_60_experts_projected_s"] for v in timing.values()])
        po = np.mean([max(v["pooled_overhead_s"], 0) for v in timing.values()])
        fig, ax = plt.subplots(figsize=(6, 3.8))
        cost = lambda B, extra=0.0: rep * B / 128 + sol + extra * B / 128
        ax.plot([cost(B) for B in Bs], [per[f"B{B}_std"]["J"].mean() for B in Bs], "o-", color=col["raw"], label="standard")
        for c, cc in meths:
            ax.plot([cost(B, po if c == "pool" else 0) for B in elig], [per[f"B{B}_{c}"]["J"].mean() for B in elig], "s",
                    color=col[cc], label=cc, ms=7)
        ax.set_xlabel("estimated per-layer deployment cost (s): replay x B/128 + 60-expert solve (+ pooled overhead)")
        ax.set_ylabel("mean J"); ax.legend(fontsize=8)
        ax.set_title("Quality vs cost (MVP's repeated candidate solves excluded)", fontsize=9)
        save(fig, "fig5_quality_vs_cost")


if __name__ == "__main__":
    main()
