"""Analysis + report for the expert-wise GPTQ/AGQ hybrid MVP (runs_expertwise_agq_hybrid/).

Reads results/{tune,confirm,oldtest}/L{L}.json + expert_mats_L{L}.npz, the frozen protocol / selection manifest and the
historical cost records; writes every analysis JSON, the figures and the decision (rule frozen in protocol.json and
implemented in moe_calib.hybrid.classify)."""
from __future__ import annotations

import json, os, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import expertwise_agq_hybrid as X   # noqa: E402
from moe_calib import hybrid as HY  # noqa: E402

p, jdump, jload, LAYERS, E, METHODS = X.p, X.jdump, X.jload, X.LAYERS, X.E, X.METHODS
TEST_SPLITS = ("confirm", "oldtest")
N_BOOT, BOOT_SEED = 2000, 20261113
LABEL = dict(GPTQ="Uniform GPTQ", AGQ="Uniform AGQ", LAYERWISE="Layer-wise selection", HYBRID="Expert-wise hybrid")
COLOR = dict(GPTQ="#2a78d6", AGQ="#eb6834", LAYERWISE="#1baf7a", HYBRID="#8a4fd8")


def load(split):
    R, M = {}, {}
    for L in LAYERS:
        R[L] = json.load(open(p("results", split, f"L{L}.json")))
        M[L] = dict(np.load(p("results", split, f"expert_mats_L{L}.npz")))
    return R, M


def E_layer(R, L, m):
    return R[L]["configs"][m]["moe_layer_output_rel_err"]


# ============================================================================ layer output + paired
def layer_output(splits):
    out = {}
    for split, (R, M) in splits.items():
        ms = [m for m in METHODS if m in R[LAYERS[0]]["configs"]]
        rec = dict(layers={}, mean={}, pooled={}, native_mean={})
        for L in LAYERS:
            rec["layers"][str(L)] = {m: dict(E_layer=E_layer(R, L, m), native=R[L]["configs"][m]["native_rel_err"],
                                             n_agq=R[L]["configs"][m]["n_agq"],
                                             config_hash=R[L]["configs"][m].get("installed_config_hash"))
                                     for m in ms}
            for m in ms:
                for ref in ms:
                    if ref != m:
                        rec["layers"][str(L)][m][f"rel_vs_{ref}"] = HY.rel(E_layer(R, L, m), E_layer(R, L, ref))
        for m in ms:
            rec["mean"][m] = float(np.mean([E_layer(R, L, m) for L in LAYERS]))
            rec["native_mean"][m] = float(np.mean([R[L]["configs"][m]["native_rel_err"] for L in LAYERS]))
            rec["pooled"][m] = sum(R[L]["configs"][m]["num"] for L in LAYERS) / sum(R[L]["configs"][m]["den"]
                                                                                 for L in LAYERS)
        rec["mean_rel"] = {f"{m}_vs_{ref}": HY.rel(rec["mean"][m], rec["mean"][ref]) for m in ms for ref in ms
                           if m != ref}
        rec["native_mean_rel"] = {f"{m}_vs_{ref}": HY.rel(rec["native_mean"][m], rec["native_mean"][ref])
                                  for m in ms for ref in ms if m != ref}
        rank = lambda d: sorted(ms, key=lambda m: d[m])
        rec["ranking_fp32"] = rank(rec["mean"])
        rec["ranking_native"] = rank(rec["native_mean"])
        rec["native_ranking_agrees"] = rec["ranking_fp32"] == rec["ranking_native"]
        rec["per_layer_ranking_agrees_native"] = {
            str(L): sorted(ms, key=lambda m: E_layer(R, L, m)) ==
            sorted(ms, key=lambda m: R[L]["configs"][m]["native_rel_err"]) for L in LAYERS}
        rec["scope"] = "2-layer (L8, L16) mean; NOT comparable to the historical 4-layer means"
        out[split] = rec
    return out


def paired(splits):
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    pairs = [("HYBRID", "GPTQ"), ("HYBRID", "AGQ"), ("HYBRID", "LAYERWISE"), ("LAYERWISE", "GPTQ"),
             ("LAYERWISE", "AGQ"), ("AGQ", "GPTQ")]
    for split, (R, M) in splits.items():
        if "HYBRID" not in R[LAYERS[0]]["configs"]:
            continue
        S = len(R[LAYERS[0]]["configs"]["GPTQ"]["per_seq"]["num"])
        num = {m: {L: np.array(R[L]["configs"][m]["per_seq"]["num"]) for L in LAYERS} for m in METHODS}
        den = {L: np.array(R[L]["configs"]["GPTQ"]["per_seq"]["den"]) for L in LAYERS}
        idx = rng.integers(0, S, size=(N_BOOT, S))
        rec = {}
        for a, b in pairs:
            r = dict(layers={})
            for L in LAYERS:
                ra, rb = num[a][L] / den[L], num[b][L] / den[L]
                dlt = ra - rb
                r["layers"][str(L)] = dict(mean_window_ratio_diff=float(dlt.mean()),
                                           median_window_rel_diff=float(np.median(dlt / rb)),
                                           frac_windows_improved=float((dlt < 0).mean()),
                                           frac_windows_equal=float((dlt == 0).mean()),
                                           rel_diff=HY.rel(num[a][L].sum() / den[L].sum(),
                                                           num[b][L].sum() / den[L].sum()))
            # bootstrap of the 2-layer mean relative difference (windows resampled jointly across layers)
            ea = np.mean([num[a][L][idx].sum(1) / den[L][idx].sum(1) for L in LAYERS], 0)
            eb = np.mean([num[b][L][idx].sum(1) / den[L][idx].sum(1) for L in LAYERS], 0)
            bd = (ea - eb) / eb
            pa = np.mean([num[a][L].sum() / den[L].sum() for L in LAYERS])
            pb = np.mean([num[b][L].sum() / den[L].sum() for L in LAYERS])
            r.update(mean_rel_diff=float((pa - pb) / pb), boot_ci95=[float(np.quantile(bd, .025)),
                                                                  float(np.quantile(bd, .975))],
                     boot_frac_a_better=float((bd < 0).mean()),
                     frac_windows_improved_both_layers_pooled=float(np.mean(
                         [((num[a][L] - num[b][L]) < 0).mean() for L in LAYERS])))
            rec[f"{a}_vs_{b}"] = r
        rec["note"] = ("descriptive: windows are not independent draws of the deployment distribution; bootstrap "
                       f"{N_BOOT} reps, seed {BOOT_SEED}")
        out[split] = rec
    return out


# ============================================================================ expert scores / transfer
def expert_scores(M):
    J0 = np.concatenate([M[L]["J0"].sum(0) for L in LAYERS])
    J1 = np.concatenate([M[L]["J1"].sum(0) for L in LAYERS])
    return J0, J1


def test_scores(splits, man):
    tJ = {}
    val = jload("validation_expert_scores.json")["experts"]
    vJ0 = np.array([x["val_J0"] for x in val]); vJ1 = np.array([x["val_J1"] for x in val])
    z = np.concatenate([np.array(man["layers"][str(L)]["z"]) for L in LAYERS])
    out = dict(note="evaluation only; never used for selection", splits={})
    for split in TEST_SPLITS:
        R, M = splits[split]
        J0, J1 = expert_scores(M)
        tok = np.concatenate([M[L]["tokens"].sum(0) for L in LAYERS])
        ri = HY.rel_improvement(J0, J1)
        ex = [dict(layer=val[i]["layer"], expert=val[i]["expert"], z=int(z[i]), test_J0=float(J0[i]),
                   test_J1=float(J1[i]), test_rel_improvement_agq=float(ri[i]), test_tokens=int(tok[i]),
                   test_prefers=HY.CANDIDATES[int(J1[i] < J0[i])], val_rel_improvement_agq=val[i][
                       "val_rel_improvement_agq"], agree=bool(int(J1[i] < J0[i]) == z[i]))
              for i in range(len(z))]
        tr = dict(pooled=HY.transfer_metrics(vJ0, vJ1, J0, J1, z), layers={})
        for li, L in enumerate(LAYERS):
            s = slice(li * E, (li + 1) * E)
            tr["layers"][str(L)] = HY.transfer_metrics(vJ0[s], vJ1[s], J0[s], J1[s], z[s])
        out["splits"][split] = dict(experts=ex, transfer=tr)
        tJ[split] = (J0, J1)
    return out, (vJ0, vJ1, z), tJ


def decomposition(splits):
    out = {}
    for split, (R, M) in splits.items():
        rec = {}
        for L in LAYERS:
            c = R[L]["configs"]
            rl = {}
            for m in c:
                x = c[m]
                rl[m] = dict(total_rel=x["num"] / x["den"], D_rel=x["D_rel"], C_rel=x["C_rel"],
                             cross_fraction=x["cross_fraction"],
                             identity_rel_residual=abs(x["D"] + x["C"] - x["decomp_total"]) / x["decomp_total"],
                             decomp_vs_direct_rel=abs(x["decomp_total"] - x["num"]) / x["num"])
            if "AGQ" in rl:
                for m in rl:
                    rl[m]["D_change_vs_AGQ_rel"] = HY.rel(rl[m]["D_rel"], rl["AGQ"]["D_rel"])
                    rl[m]["total_change_vs_AGQ_rel"] = HY.rel(rl[m]["total_rel"], rl["AGQ"]["total_rel"])
                    rl[m]["C_change_vs_AGQ_abs"] = rl[m]["C_rel"] - rl["AGQ"]["C_rel"]
            rec[str(L)] = rl
        rec["max_identity_rel_residual"] = max(v["identity_rel_residual"] for L in LAYERS for v in rec[str(L)].values())
        out[split] = rec
    return out


# ============================================================================ selection distribution (Stage 5A)
def selection_distribution():
    val = jload("validation_expert_scores.json")["experts"]
    st = jload("selection_stability.json")
    z = np.array([x["z"] for x in val]); lay = np.array([x["layer"] for x in val])

    def by_bins(key, nb, label):
        v = np.array([x[key] for x in val], dtype=float)
        res = {}
        for L in LAYERS:
            m = lay == L
            qs = np.quantile(v[m], np.linspace(0, 1, nb + 1)[1:-1])
            b = np.digitize(v[m], qs)
            res[str(L)] = [dict(bin=int(i), n=int((b == i).sum()), n_agq=int(z[m][b == i].sum()),
                                frac_agq=float(z[m][b == i].mean()) if (b == i).any() else None,
                                range=[float(v[m][b == i].min()), float(v[m][b == i].max())] if (b == i).any() else None)
                           for i in range(nb)]
        return dict(variable=label, bins_within_layer=nb, layers=res)

    dec = {}
    for L in LAYERS:
        m = lay == L
        nd = np.array([x["natural_decile"] for x in val])[m]
        dec[str(L)] = {int(d): dict(n=int((nd == d).sum()), n_agq=int(z[m][nd == d].sum())) for d in np.unique(nd)}
    # Is the heterogeneity real? compare spread of val relative improvement with its half-split reproducibility.
    het = {}
    for L in LAYERS:
        m = lay == L
        ri = np.array([x["val_rel_improvement_agq"] for x in val])[m]
        het[str(L)] = dict(n_agq=int(z[m].sum()), n_gptq=int((1 - z[m]).sum()), mean_rel_improvement=float(ri.mean()),
                           sd_rel_improvement=float(ri.std()), min=float(ri.min()), max=float(ri.max()),
                           frac_abs_margin_below_1pct=float((np.abs(ri) < 0.01).mean()),
                           half_split_agreement=st["layers"][str(L)]["agreement_halves"],
                           half_split_spearman=st["layers"][str(L)]["spearman_rel_improvement_halves"])
    return dict(counts=dict(total_agq=int(z.sum()), total_gptq=int((1 - z).sum()),
                            per_layer={str(L): dict(agq=int(z[lay == L].sum()), gptq=int((1 - z[lay == L]).sum()))
                                       for L in LAYERS}),
                heterogeneity=het,
                by_natural_exposure_decile=dec,
                by_calibration_tokens_tercile=by_bins("calib_tokens", 3, "FR s0 calibration routed tokens"),
                by_calibration_affinity_mean_tercile=by_bins("calib_affinity_mean", 3, "mean router affinity (calib)"),
                by_calibration_affinity_cv_tercile=by_bins("calib_affinity_cv", 3, "router-affinity CV (calib)"),
                by_validation_tokens_tercile=by_bins("val_tokens", 3, "tuning-split routed tokens"),
                spearman_with_val_rel_improvement={
                    str(L): {k: HY._corr(HY._rank(np.array([x[k] for x in val])[lay == L]),
                                         HY._rank(np.array([x["val_rel_improvement_agq"] for x in val])[lay == L]))
                             for k in ("calib_tokens", "calib_affinity_mean", "calib_affinity_cv", "val_tokens")}
                    for L in LAYERS},
                note="bins are within-layer terciles (60 experts per layer); no claim of an exposure->method "
                     "relationship is made unless the correlation is clearly non-zero")


# ============================================================================ cost
def cost():
    hc = json.load(open(os.path.join(ROOT, "runs_agq_only_validation", "cost_results.json")))
    reg, sc = jload("candidate_regression.json"), jload("logs", "cost_score_tune.json")
    ev = {s: jload("logs", f"cost_evaluate_{s}.json") for s in TEST_SPLITS}
    sel = jload("logs", "selection_freeze.json")
    g, a = hc["gptq_historical_FR_s0"], hc["agq_FR_s0"]
    per2 = lambda r: dict(hessian_s=r["hessian_s"] / 2, gptq_s=r["gptq_s"] / 2,
                          hessian_plus_gptq_s=(r["hessian_s"] + r["gptq_s"]) / 2)
    tune_s = sum(sc[str(L)]["seconds"] for L in LAYERS)
    tune_lean = sum(sc[str(L)]["lean_s"] for L in LAYERS)
    tune_native = sum(sc[str(L)]["native_s"] for L in LAYERS)
    q_g, q_a = per2(g)["hessian_plus_gptq_s"], per2(a)["hessian_plus_gptq_s"]
    wsize = {L: os.path.getsize(os.path.join(X.cand_dirs()[0], f"L{L}.pt")) for L in LAYERS}
    return dict(
        incremental_mvp=dict(
            candidate_generation_s=0.0,
            candidate_generation_note="historical q=0 / q=1 weights reused (bitwise verified); no quantization run",
            regression_s=reg["seconds"], validation_scoring_s=tune_s, validation_scoring_lean_s=tune_lean,
            validation_native_check_s=tune_native, expert_selection_s=sel["selection_s"],
            test_evaluation_s={s: sum(ev[s][str(L)]["seconds"] for L in LAYERS) for s in TEST_SPLITS},
            peak_gpu_gb=max([reg["peak_gpu_gb"], sc["peak_gpu_gb"]] + [ev[s]["peak_gpu_gb"] for s in TEST_SPLITS]),
            peak_cpu_gb=max([reg["peak_cpu_gb"], sc["peak_cpu_gb"]] + [ev[s]["peak_cpu_gb"] for s in TEST_SPLITS])),
        historical_quantization_4_layers=dict(GPTQ_FR_s0=g, AGQ_FR_s0=a,
                                              source="runs_agq_only_validation/cost_results.json"),
        estimated_fresh_deployment_2_layers=dict(
            method="historical 4-layer Hessian + GPTQ seconds / 2 (L8, L16 only; layers have equal shapes); the "
                   "'other' term (E3 scoring / weight IO / PPL checks) is excluded",
            uniform_GPTQ_s=q_g, uniform_AGQ_s=q_a,
            layerwise_selection_s=q_g + q_a + tune_lean,
            expert_wise_hybrid_s=q_g + q_a + tune_lean,
            hybrid_vs_single_run_ratio=(q_g + q_a + tune_lean) / q_g,
            note="the layer-wise baseline needs the same two candidates and the same tuning-split layer evaluation as "
                 "the hybrid (its per-expert scores come for free from that pass), so both cost ~2x one GPTQ run + "
                 "validation; a joint profiler pass could share the Hessian forward but not the GPTQ solves"),
        storage=dict(final_weights_bytes_per_layer_bf16_fake_quant=wsize,
                     final_storage_vs_uniform="identical: one 4-bit candidate per expert, same format/group size",
                     offline_peak_storage="both candidates (2x) must be held until selection",
                     selection_manifest_bytes=os.path.getsize(p("selection_manifest.json"))),
        inference=dict(expected_change="none (one existing weight set per expert; same kernels/format)",
                       measured=False, note="no inference speed or memory measurement was made"))


# ============================================================================ correctness
def correctness(splits, dec_all, test_sc):
    inv, reg, man = jload("artifact_inventory.json"), jload("candidate_regression.json"), jload("selection_manifest.json")
    fr = jload("logs", "selection_freeze.json")
    R, _ = splits["confirm"]
    leg = inv["historical_confirm_gptq"]
    metas = {s: {L: splits[s][0][L]["meta"] for L in LAYERS} for s in ("tune",) + TEST_SPLITS}
    acc = {L: metas["confirm"][L]["accumulator_check"] for L in LAYERS}
    test_mtime = min(os.path.getmtime(p("results", s, f"L{L}.json")) for s in TEST_SPLITS for L in LAYERS)
    man_mtime = os.path.getmtime(p("selection_manifest.json"))
    hashes_ok = all(splits[s][0][L]["configs"]["HYBRID"]["installed_config_hash"] ==
                    man["layers"][str(L)]["installed_config_hash"] for s in TEST_SPLITS for L in LAYERS)
    checks = {
        "01_gptq_candidates_match_historical_q0": all(reg["layers"][str(L)]["weights"]["gptq_file_sha_match"]
                                                      for L in LAYERS),
        "02_agq_candidates_match_historical_q1": all(reg["layers"][str(L)]["weights"]["agq_file_sha_match"]
                                                     for L in LAYERS),
        "03_same_frozen_calibration_manifest": all(inv["candidates"][n]["cache_rows_equal_frozen_manifest"]
                                                   for n in ("GPTQ", "AGQ")),
        "04_router_coefficients_identical_across_methods": True,
        "05_top4_routing_unchanged": True,
        "06_only_routed_expert_weights_replaced": all(
            metas[s][L]["install_all_three_from_same_candidate_and_shared_untouched"] for s in metas for L in LAYERS),
        "07_shared_expert_untouched": all(metas[s][L]["install_all_three_from_same_candidate_and_shared_untouched"]
                                          for s in metas for L in LAYERS),
        "08_each_expert_all_three_projections_same_candidate": all(
            metas[s][L]["install_all_three_from_same_candidate_and_shared_untouched"] for s in metas for L in LAYERS),
        "09_no_mixed_candidate_tensors": hashes_ok,
        "10_pristine_weights_restored": all(metas[s][L]["pristine_restored"] for s in metas for L in LAYERS),
        "11_manifest_fully_determines_hybrid": hashes_ok and X.sha256_file(p("selection_manifest.json")) ==
                                               fr["manifest_sha256"],
        "12_no_test_data_used_for_selection": bool(man_mtime < test_mtime and not man["test_results_exist_at_freeze"]
                                                   and man["split"] == "tune"),
        "13_layer_evaluator_matches_validated_accumulator": all(c["abs_diff"] <= 1e-12 for L in LAYERS
                                                                for c in acc[L].values()) and reg["passed"],
        "14_D_plus_C_reproduces_total": all(dec_all[s]["max_identity_rel_residual"] < 1e-6 for s in dec_all) and
                                        all(metas[s][L]["decomposition_vs_err_max_rel"] < 1e-5 for s in metas
                                            for L in LAYERS),
        "15_historical_uniform_errors_reproduced": reg["passed"] and all(
            abs(R[L]["configs"]["GPTQ"]["moe_layer_output_rel_err"] / leg[str(L)] - 1) < 1e-9 for L in LAYERS),
    }
    ev = dict(
        router_note="the native BF16 forward asserts torch.equal on router logits for every window and configuration; "
                    "all configurations share the same routing/c tensors within one pass (checks 04/05)",
        confirm_gptq_vs_legacy={str(L): dict(now=R[L]["configs"]["GPTQ"]["moe_layer_output_rel_err"],
                                             legacy=leg[str(L)]) for L in LAYERS},
        oldtest_regression={str(L): {n: reg["layers"][str(L)][n] for n in ("GPTQ", "AGQ")} for L in LAYERS},
        accumulator_check_confirm={str(L): acc[L] for L in LAYERS},
        manifest_frozen_before_tests=dict(manifest_mtime=time.ctime(man_mtime), first_test_result=time.ctime(test_mtime)),
        tests=_tests())
    ok = all(checks.values()) and ev["tests"].get("hybrid_passed", False)
    return dict(checks=checks, evidence=ev, passed=bool(ok))


def _tests():
    out = {}
    for tag in ("hybrid", "all"):
        f = p("logs", f"pytest_{tag}.log")
        if os.path.exists(f):
            last = [l for l in open(f).read().strip().splitlines() if l.strip()][-1]
            out[tag] = last
            out[f"{tag}_passed"] = ("failed" not in last and "error" not in last and "passed" in last)
    return out


# ============================================================================ figures
def figures(splits, lo, tsc, vt, dec, cst):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25})
    os.makedirs(p("figures"), exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(p("figures", f"{name}.{ext}"), dpi=170)
        plt.close(fig)

    val = jload("validation_expert_scores.json")["experts"]
    # Fig 1: expert objective preference (validation)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, L in zip(axes, LAYERS):
        ri = np.array([x["val_rel_improvement_agq"] for x in val if x["layer"] == L]) * 100
        bins = np.linspace(min(ri.min(), -1), max(ri.max(), 1), 25)
        ax.hist(ri[ri > 0], bins=bins, color=COLOR["AGQ"], label=f"AGQ chosen ({(ri > 0).sum()})")
        ax.hist(ri[ri <= 0], bins=bins, color=COLOR["GPTQ"], label=f"GPTQ chosen ({(ri <= 0).sum()})")
        ax.axvline(0, color="k", lw=.8)
        ax.set_title(f"L{L}: validation c²-weighted expert error, AGQ vs GPTQ")
        ax.set_xlabel("relative improvement of AGQ over GPTQ (%)")
        ax.legend(frameon=False)
    axes[0].set_ylabel("experts")
    save(fig, "fig1_expert_objective_preference")

    # Fig 2: aggregated layer output
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, split in zip(axes, TEST_SPLITS):
        rec = lo[split]
        xs = np.arange(len(LAYERS) + 1)
        w = .2
        for i, m in enumerate(METHODS):
            vals = [rec["layers"][str(L)][m]["E_layer"] for L in LAYERS] + [rec["mean"][m]]
            base = [rec["layers"][str(L)]["GPTQ"]["E_layer"] for L in LAYERS] + [rec["mean"]["GPTQ"]]
            ax.bar(xs + (i - 1.5) * w, [(v / b - 1) * 100 for v, b in zip(vals, base)], w, color=COLOR[m],
                   label=LABEL[m])
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(xs, [f"L{L}" for L in LAYERS] + ["2-layer mean"])
        ax.set_ylabel("E_layer relative to uniform GPTQ (%)")
        ax.set_title(f"{split}: MoE-layer output error vs GPTQ (GPTQ = 0; lower better)", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "fig2_layer_output_comparison")

    # Fig 3: validation vs test preference
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, split in zip(axes, TEST_SPLITS):
        ex = tsc["splits"][split]["experts"]
        for L, mk in zip(LAYERS, ("o", "s")):
            v = np.array([x["val_rel_improvement_agq"] for x in ex if x["layer"] == L]) * 100
            t = np.array([x["test_rel_improvement_agq"] for x in ex if x["layer"] == L]) * 100
            ax.scatter(v, t, s=14, marker=mk, alpha=.75, label=f"L{L}", color=COLOR["GPTQ"] if L == 8 else COLOR["AGQ"])
        lim = max(abs(np.array([x["val_rel_improvement_agq"] for x in ex])).max(),
                  abs(np.array([x["test_rel_improvement_agq"] for x in ex])).max()) * 105
        ax.plot([-lim, lim], [-lim, lim], color="grey", lw=.7, ls="--")
        ax.set_xscale("symlog", linthresh=1.0); ax.set_yscale("symlog", linthresh=1.0)   # e40 (L16) is ~49 %
        ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6)
        tr = tsc["splits"][split]["transfer"]["pooled"]
        ax.set_title(f"{split}: sign agreement {tr['sign_agreement']:.2f}, Spearman {tr['spearman_rel_improvement']:.2f}")
        ax.set_xlabel("validation: AGQ rel. improvement (%, symlog)"); ax.set_ylabel(f"{split}: AGQ rel. improvement (%)")
        ax.legend(frameon=False)
    save(fig, "fig3_validation_vs_test_preference")

    # Fig 4: error decomposition (confirm): diagonal change vs uniform AGQ and cross-term share of the total
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, L in zip(axes, LAYERS):
        d = dec["confirm"][str(L)]
        xs = np.arange(len(METHODS))
        Dc = [d[m]["D_change_vs_AGQ_rel"] * 100 for m in METHODS]
        Cf = [d[m]["cross_fraction"] * 100 for m in METHODS]
        ax.bar(xs - .2, Dc, .4, color=[COLOR[m] for m in METHODS], label="diagonal D: change vs uniform AGQ (%)")
        ax.bar(xs + .2, Cf, .4, color="#555555", alpha=.6, label="cross C: share of total (%)")
        for i in range(len(METHODS)):
            ax.text(xs[i] - .2, Dc[i], f"{Dc[i]:+.3f}", ha="center", va="bottom" if Dc[i] >= 0 else "top", fontsize=7)
            ax.text(xs[i] + .2, Cf[i], f"{Cf[i]:+.3f}", ha="center", va="bottom" if Cf[i] >= 0 else "top", fontsize=7)
        ax.axhline(0, color="k", lw=.7)
        ax.set_xticks(xs, [LABEL[m].replace(" ", "\n") for m in METHODS], fontsize=7)
        ax.set_ylabel("%")
        ax.set_title(f"confirm L{L}: D vs uniform AGQ, and C share")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "fig4_error_decomposition")

    # Fig 5: quality vs preparation cost
    fe = cst["estimated_fresh_deployment_2_layers"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    costs = dict(GPTQ=fe["uniform_GPTQ_s"], AGQ=fe["uniform_AGQ_s"], LAYERWISE=fe["layerwise_selection_s"],
                 HYBRID=fe["expert_wise_hybrid_s"])
    for m in METHODS:
        q = lo["confirm"]["mean_rel"].get(f"{m}_vs_GPTQ", 0.0) * 100
        ax.scatter(costs[m] / 60, q, s=60, color=COLOR[m], label=LABEL[m], zorder=3,
                   marker="D" if m == "HYBRID" else "o")
    ax.axhline(0, color="k", lw=.7)
    ax.set_xlabel("estimated offline preparation, 2 layers (min)")
    ax.set_ylabel("confirm 2-layer mean E_layer vs uniform GPTQ (%)")
    ax.set_title("Quality vs preparation cost (lower-left is better)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig5_quality_vs_cost")


# ============================================================================ main
def main():
    splits = {s: load(s) for s in ("tune",) + TEST_SPLITS}
    man = jload("selection_manifest.json")
    lo = layer_output(splits)
    pc = paired({s: splits[s] for s in TEST_SPLITS})
    tsc, vt, tJ = test_scores(splits, man)
    dec = decomposition(splits)
    sd = selection_distribution()
    cst = cost()
    corr = correctness(splits, dec, tsc)
    prot = jload("protocol.json")
    rules = prot["decision_rule"]["rules"]
    verdicts = {}
    for s in TEST_SPLITS:
        le = {L: {m: lo[s]["layers"][str(L)][m]["E_layer"] for m in METHODS} for L in LAYERS}
        verdicts[s] = HY.classify(rules, le, tsc["splits"][s]["transfer"]["pooled"], corr["passed"])
        verdicts[s]["recommendation"] = HY.recommend(verdicts[s], rules["tau_rel"])
    prim = verdicts["confirm"]
    jdump(lo, p("layer_output_results.json"))
    jdump(pc, p("paired_comparisons.json"))
    jdump(tsc, p("test_expert_scores.json"))
    jdump(dec, p("error_decomposition.json"))
    jdump(cst, p("cost_results.json"))
    jdump(corr, p("correctness_results.json"))
    jdump(dict(primary_split="confirm", verdict=prim["verdict"], recommendation=prim["recommendation"],
               primary=prim, secondary_oldtest=verdicts["oldtest"], rules=rules,
               rule_source="protocol.json (frozen before selection) / moe_calib.hybrid.classify",
               decided=time.strftime("%Y-%m-%d %H:%M:%S")), p("decision.json"))
    summ = dict(
        verdict=prim["verdict"], recommendation=prim["recommendation"],
        selection=sd["counts"], heterogeneity=sd["heterogeneity"],
        stability=jload("selection_stability.json"),
        layer_output={s: dict(per_layer={str(L): {m: lo[s]["layers"][str(L)][m]["E_layer"] for m in METHODS}
                                         for L in LAYERS}, mean=lo[s]["mean"], mean_rel=lo[s]["mean_rel"],
                              native_ranking_agrees=lo[s]["native_ranking_agrees"]) for s in TEST_SPLITS},
        validation_layer_output={str(L): {m: splits["tune"][0][L]["configs"][m]["moe_layer_output_rel_err"]
                                          for m in ("GPTQ", "AGQ")} for L in LAYERS},
        layerwise_choice={str(L): jload("layerwise_uniform_selection.json")["layers"][str(L)]["choice"] for L in LAYERS},
        transfer={s: dict(pooled={k: v for k, v in tsc["splits"][s]["transfer"]["pooled"].items()
                                  if k != "margin_terciles"},
                          layers={L: {k: v for k, v in t.items() if k != "margin_terciles"}
                                  for L, t in tsc["splits"][s]["transfer"]["layers"].items()},
                          margin_terciles=tsc["splits"][s]["transfer"]["pooled"]["margin_terciles"])
                  for s in TEST_SPLITS},
        decomposition={s: {str(L): {m: dict(D_rel=dec[s][str(L)][m]["D_rel"], C_rel=dec[s][str(L)][m]["C_rel"],
                                            cross_fraction=dec[s][str(L)][m]["cross_fraction"])
                                    for m in METHODS} for L in LAYERS} for s in TEST_SPLITS},
        paired={s: {k: {kk: vv for kk, vv in v.items() if kk != "layers"} for k, v in pc[s].items() if k != "note"}
                for s in TEST_SPLITS},
        cost=dict(incremental=cst["incremental_mvp"], fresh=cst["estimated_fresh_deployment_2_layers"]),
        correctness_passed=corr["passed"], selection_distribution=sd)
    jdump(summ, p("summary.json"))
    figures(splits, lo, tsc, vt, dec, cst)
    print(f"[report] verdict {prim['verdict']} -> recommendation {prim['recommendation']}; correctness "
          f"{corr['passed']}; oldtest verdict {verdicts['oldtest']['verdict']}")
    print(json.dumps({k: prim[k] for k in ("mean", "rel_vs_gptq", "rel_vs_agq", "rel_vs_layerwise", "criteria")},
                     indent=1))


if __name__ == "__main__":
    main()
