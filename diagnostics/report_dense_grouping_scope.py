"""Analysis, decision, figures and summary for the dense-model grouping control (runs_dense_grouping_scope)."""
from __future__ import annotations
import itertools, json, math, os, time

import numpy as np

import dense_grouping_scope as S

p, jload, jdump, GROUPS, REF, CONFIGS = S.p, S.jload, S.jdump, S.GROUPS, S.REF, S.CONFIGS
ROOT = S.ROOT
PROJ = S.PROJ


def sd(x):
    return float(np.std(np.asarray(x, float), ddof=1))


def sd_ci(s, n, alpha=0.05):
    """chi-square CI of a normal-sample SD (descriptive; n=5 -> df 4)."""
    from scipy.stats import chi2
    df = n - 1
    return [float(s * math.sqrt(df / chi2.ppf(1 - alpha / 2, df))), float(s * math.sqrt(df / chi2.ppf(alpha / 2, df)))]


def ratio_ci(r, n1, n2, alpha=0.05):
    """CI of an SD ratio from F(n1-1, n2-1) (descriptive)."""
    from scipy.stats import f
    lo, hi = f.ppf(alpha / 2, n1 - 1, n2 - 1), f.ppf(1 - alpha / 2, n1 - 1, n2 - 1)
    return [float(r / math.sqrt(hi)), float(r / math.sqrt(lo))]


def med(x):
    return float(np.median(np.asarray(list(x), float)))


# ---------------------------------------------------------------- historical MoE comparators
def moe_reference():
    au = jload(p("logs", "audit.json"))
    w = json.load(open(os.path.join(ROOT, "runs_grouping_routing_amplification/logs/weight_differences_raw.json")))
    pairs = {k: v for k, v in w["within_seed_pairs"].items() if k.startswith("FR_s0_")}
    vs_o = {k: v for k, v in w["s0_vs_original"].items() if k.startswith("FR_s0_")}
    assert len(pairs) == 10 and len(vs_o) == 5
    hc = json.load(open(os.path.join(ROOT, "runs_frozen_routing/hessian_comparison.json")))
    toks = np.array([e["tokens"] for e in json.load(open(os.path.join(ROOT, "runs_multigroup_robustness/results/FR_G20261111.json")))["per_expert"]])
    mm = au["moe_model"]

    def agg(d):
        return dict(frac_changed=float(np.mean([v["frac_changed"] for v in d.values()])),
                    rel_frobenius=float(np.mean([v["rel_frobenius"] for v in d.values()])),
                    mean_abs_diff=float(np.mean([v["mean_abs_diff"] for v in d.values()])),
                    median_abs_diff=float(np.mean([v["median_abs_diff_subsample"] for v in d.values()])),
                    ulp=({b: float(np.mean([v["ulp_share_of_changed"][b] for v in d.values()]))
                          for b in ("1", "2", "3-7", ">=8", "sign_change")}),
                    per_projection={k: float(np.mean([v["frac_changed_per_projection"][k] for v in d.values()])) for k in PROJ},
                    per_layer={k: float(np.mean([v["frac_changed_per_layer"][k] for v in d.values()])) for k in ("1", "8", "16", "23")})
    fc = np.array([json.load(open(os.path.join(ROOT, f"runs_multigroup_robustness/results/FC_{g}.json")))["ppl"] for g in GROUPS])
    return dict(
        ppl=au["moe_FR_s0"], ppl_FC_s0=dict(sd=sd(fc), sd_log=sd(np.log(fc)), cv=sd(fc) / fc.mean()),
        pooled_3seed_sd=dict(FR=0.005198221444771882, FC=0.003811771415524038, paired=0.006871891170852839,
                             note="pooled within-seed (3 seeds) from runs_grouping_routing_amplification; NOT a single-seed estimate"),
        weights_10pairs=agg(pairs), weights_vs_original=agg(vs_o),
        hessian=dict(source="runs_frozen_routing/hessian_comparison.json, FR, G20261111 vs ORIGINAL (OO), 16 diagnostic Hessians "
                            "(layers 1/8/16/23 x experts 0/30 x gu/down)",
                     AA_natural_median=hc["summary"]["FR"]["AA_vs_OO"]["rel_fro_trace_normalized"]["median"],
                     AO_activation_only_median=hc["summary"]["FR"]["AO_vs_OO"]["rel_fro_trace_normalized"]["median"],
                     AA_by_projection={k: v["median"] for k, v in hc["by_projection"]["FR"]["AA_vs_OO"].items()},
                     AO_by_projection={k: v["median"] for k, v in hc["by_projection"]["FR"]["AO_vs_OO"].items()},
                     actorder_AA=dict(frac_positions_changed=hc["actorder"]["FR"]["AA_vs_OO"]["mean_frac_positions_changed"],
                                      frac_columns_group_changed=hc["actorder"]["FR"]["AA_vs_OO"]["mean_frac_columns_changed_group"]),
                     actorder_AO=dict(frac_positions_changed=hc["actorder"]["FR"]["AO_vs_OO"]["mean_frac_positions_changed"],
                                      frac_columns_group_changed=hc["actorder"]["FR"]["AO_vs_OO"]["mean_frac_columns_changed_group"]),
                     note="dense has no routing: the matched MoE quantity is AO (activation-only, routing frozen); AA adds "
                          "routing-membership changes"),
        geometry=dict(n_matrices=720, n_quantized_params=mm["n_quantized_params"], d_gu=mm["hidden"], d_down=mm["moe_intermediate"],
                      tokens_per_expert=dict(median=float(np.median(toks)), mean=float(toks.mean()), p10=float(np.quantile(toks, .1)),
                                             p90=float(np.quantile(toks, .9)), min=float(toks.min()), max=float(toks.max())),
                      tokens_per_input_dim_gu_median=float(np.median(toks) / mm["hidden"]),
                      tokens_per_input_dim_down_median=float(np.median(toks) / mm["moe_intermediate"]),
                      frac_quantized_of_model=mm["n_quantized_params"] / 14315784192))


# ---------------------------------------------------------------- analysis
def analyze():
    t0 = time.time()
    pr = S.check_protocol()
    LAY = [int(x) for x in pr["quantization"]["layer_mapping"].values()]
    au = jload(p("logs", "audit.json"))
    bf = jload(p("bf16_baseline.json"))
    gc_ = jload(p("logs", "gpu_checks.json"))
    cm = jload(p("cache_manifest.json"))
    rep = jload(p("logs", "repeat_control.json")) or {}
    res = {g: jload(p("results", f"{g}.json")) for g in CONFIGS}
    hraw = jload(p("logs", "hessian_raw.json"))
    wraw = jload(p("logs", "weight_differences_raw.json"))
    pyt = jload(p("logs", "pytest_results.json")) or {}
    moe = moe_reference()

    # ---------- correctness
    ck = dict(
        A_bf16_checkpoint_equal=gc_["checkpoint"]["all_equal"], A_bf16_evaluated=bf is not None,
        B_bf16_repeat_bitwise=bf["repeat_per_window_bitwise"],
        B_all_config_repeat_bitwise=all(r["B"]["repeat_per_window_bitwise"] for r in res.values()),
        B_fresh_process_requantization_bitwise=rep.get("passed"),
        C_union_multiset_identical=all(jload(p("grouping_manifest.json"))["groupings"][g]["multiset_equals_union"] for g in GROUPS),
        C_tokens_identical=all(cm[g]["token_sha_equals_union"] for g in CONFIGS),
        C_reversed_execution_order_bitwise=all(cm[g]["reversed_ok"] for g in CONFIGS),
        C_tokens_per_projection_65536=all(r["D"]["tokens_per_projection_ok"] for r in res.values()),
        C_same_layers=all(r["layers"] == LAY for r in res.values()),
        D_shapes=all(r["D"]["all_shapes_ok"] for r in res.values()), D_finite=all(r["D"]["all_finite"] for r in res.values()),
        D_all_targets_quantized=all(r["D"]["all_targets_quantized"] for r in res.values()),
        D_installed_equals_solver=all(r["D"]["installed_equals_solver"] for r in res.values()),
        D_non_targets_unchanged=all(r["D"]["untouched_unchanged"] for r in res.values()),
        D_integer_codes_exact=all(r["D"]["codes_exact"] for r in res.values()),
        D_info_capture_bitwise_default=gc_.get("info_path_bitwise_default"),
        E_pristine_start=all(r["E"]["started_pristine"] for r in res.values()),
        F_restoration=all(all(r["F"].values()) for r in res.values()),
        F_end_bf16_bitwise=gc_["end"]["bf16_reeval_bitwise"], F_end_pristine=gc_["end"]["target_digest_pristine"],
        F_end_untouched=gc_["end"]["untouched_unchanged"],
        all_configs_valid=all(r["valid"] for r in res.values()),
        targeted_tests=pyt.get("passed"))
    blocked = not all(v for k, v in ck.items() if k != "targeted_tests")
    jdump(dict(checks=ck, blocked=blocked, n_untouched=gc_["n_untouched"]), p("correctness_results.json"))

    # ---------- PPL
    ppl = np.array([res[g]["ppl"] for g in GROUPS])
    pw = {g: np.array(res[g]["per_window_nll"]) / np.array(res[g]["per_window_tokens"]) for g in CONFIGS}
    rows = {g: dict(ppl=res[g]["ppl"], mean_nll=res[g]["mean_nll"], ppl_minus_ref=res[g]["ppl"] - res[REF]["ppl"],
                    ppl_minus_bf16=res[g]["ppl"] - bf["ppl"], installed_weights_sha256=res[g]["installed_weights_sha256"])
            for g in CONFIGS}
    jdump(dict(bf16=dict(ppl=bf["ppl"]), configs=rows, reference_grouping=REF,
               note="ORIGINAL is the cache_inputs.py batching reference (descriptive; not one of the 5 groupings)"),
          p("grouping_ppl_results.json"))
    pairs10 = list(itertools.combinations(GROUPS, 2))
    pwd = {}
    for a, b in pairs10 + [(g, "ORIGINAL") for g in GROUPS]:
        d = pw[a] - pw[b]
        pwd[f"{a}__{b}"] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), max_abs=float(np.abs(d).max()),
                               frac_windows_bitwise_equal=float((np.array(res[a]["per_window_nll"]) == np.array(res[b]["per_window_nll"])).mean()),
                               ppl_diff=res[a]["ppl"] - res[b]["ppl"])
    s_ppl = sd(ppl)
    var = dict(
        n=5, mean=float(ppl.mean()), sd=s_ppl, sd_ci95=sd_ci(s_ppl, 5), sd_log=sd(np.log(ppl)), cv=s_ppl / float(ppl.mean()),
        min=float(ppl.min()), max=float(ppl.max()), max_pairwise_diff=float(ppl.max() - ppl.min()),
        mean_minus_bf16=float(ppl.mean() - bf["ppl"]), sd_over_quant_penalty=s_ppl / float(ppl.mean() - bf["ppl"]),
        relative_to_ref={g: res[g]["ppl"] - res[REF]["ppl"] for g in GROUPS},
        per_window_paired=pwd,
        mean_per_window_sd_over_pairs=float(np.mean([v["sd"] for k, v in pwd.items() if "ORIGINAL" not in k])),
        original_reference=dict(ppl=res["ORIGINAL"]["ppl"], minus_grouping_mean=res["ORIGINAL"]["ppl"] - float(ppl.mean())),
        n_distinct_installed_weights=len({res[g]["installed_weights_sha256"] for g in GROUPS}))
    jdump(var, p("ppl_variability.json"))

    # ---------- weights + codes
    W, Cd = wraw["weights"], wraw["codes"]
    k10 = [f"{a}__{b}" for a, b in pairs10]
    k10 = [k if k in W else "__".join(k.split("__")[::-1]) for k in k10]
    kref = [f"{g}__{REF}" for g in GROUPS if g != REF]
    korig = [f"{g}__ORIGINAL" for g in GROUPS]

    def agg(keys):
        return dict(frac_changed=float(np.mean([W[k]["frac_changed"] for k in keys])),
                    frac_changed_range=[float(min(W[k]["frac_changed"] for k in keys)), float(max(W[k]["frac_changed"] for k in keys))],
                    rel_frobenius=float(np.mean([W[k]["rel_frobenius"] for k in keys])),
                    mean_abs_diff=float(np.mean([W[k]["mean_abs_diff"] for k in keys])),
                    median_abs_diff=float(np.mean([W[k]["median_abs_diff_subsample"] for k in keys])),
                    ulp={b: float(np.mean([W[k]["ulp_share_of_changed"][b] for k in keys])) for b in ("1", "2", "3-7", ">=8", "sign_change")},
                    per_layer={str(L): float(np.mean([W[k]["frac_changed_per_layer"][str(L)] for k in keys])) for L in LAY},
                    per_projection={pj: float(np.mean([W[k]["frac_changed_per_projection"][pj] for k in keys])) for pj in PROJ},
                    codes=dict(frac_codes_changed=float(np.mean([Cd[k]["frac_codes_changed"] for k in keys])),
                               frac_bf16_changed_with_code_change=float(np.mean([Cd[k]["frac_bf16_changed_with_code_change"] for k in keys])),
                               frac_bf16_changed_same_code=float(np.mean([Cd[k]["frac_bf16_changed_same_code"] for k in keys])),
                               frac_code_changed_same_bf16=float(np.mean([Cd[k]["frac_code_changed_same_bf16"] for k in keys])),
                               frac_elements_in_group_changed_columns=float(np.mean([Cd[k]["frac_elements_in_group_changed_columns"] for k in keys]))),
                    per_layer_proj={lp: dict(frac_bf16_changed=float(np.mean([Cd[k]["per_layer_proj"][lp]["frac_bf16_changed"] for k in keys])),
                                             frac_codes_changed=float(np.mean([Cd[k]["per_layer_proj"][lp]["frac_codes_changed"] for k in keys])),
                                             frac_columns_group_changed=float(np.mean([Cd[k]["per_layer_proj"][lp]["frac_columns_group_changed"] for k in keys])))
                                    for lp in Cd[keys[0]]["per_layer_proj"]})
    wd = dict(ten_pairs=agg(k10), vs_reference=agg(kref), vs_original=agg(korig),
              per_pair={k: dict(frac_changed=W[k]["frac_changed"], rel_frobenius=W[k]["rel_frobenius"],
                                frac_codes_changed=Cd[k]["frac_codes_changed"]) for k in W},
              moe_FR_s0=dict(ten_pairs=moe["weights_10pairs"], vs_original=moe["weights_vs_original"]),
              note=wraw["note"])
    jdump(wd, p("weight_difference_analysis.json"))

    # ---------- Hessians
    H = hraw["pairs"]

    def hagg(pairs_, pj=None):
        ks = [k for k in H if k.split("|")[2] in pairs_ and (pj is None or k.split("|")[1] == pj)]
        if not ks:
            return None
        return {m: dict(median=med(H[k][m] for k in ks), mean=float(np.mean([H[k][m] for k in ks])),
                        min=float(min(H[k][m] for k in ks)), max=float(max(H[k][m] for k in ks)), n=len(ks))
                for m in ("rel_fro_trace_normalized", "rel_fro_raw", "diag_rel_diff", "frac_actorder_positions_changed",
                          "frac_columns_group_changed", "mean_abs_diag_rank_shift", "frac_elements_bitwise_equal")}
    p10 = {f"{a}__{b}" if f"L{LAY[0]}|gu|{a}__{b}" in H else f"{b}__{a}" for a, b in pairs10}
    pref = {f"{g}__{REF}" for g in GROUPS if g != REF}
    porig = {f"{g}__ORIGINAL" for g in GROUPS}
    hs = dict(
        ten_pairs=dict(all=hagg(p10), gu=hagg(p10, "gu"), down=hagg(p10, "down")),
        vs_reference=dict(all=hagg(pref), gu=hagg(pref, "gu"), down=hagg(pref, "down")),
        vs_original=dict(all=hagg(porig), gu=hagg(porig, "gu"), down=hagg(porig, "down")),
        matched_to_moe=dict(dense_G20261111_vs_ORIGINAL={f"L{L}|{pj}": H[f"L{L}|{pj}|{REF}__ORIGINAL"]["rel_fro_trace_normalized"]
                                                         if f"L{L}|{pj}|{REF}__ORIGINAL" in H else None
                                                         for L in LAY for pj in ("gu", "down")},
                            moe=moe["hessian"]),
        per_layer_proj={f"L{L}|{pj}": dict(
            ten_pairs_median_rel_fro_tn=med(H[f"L{L}|{pj}|{k}"]["rel_fro_trace_normalized"] for k in p10),
            ten_pairs_median_actorder_pos=med(H[f"L{L}|{pj}|{k}"]["frac_actorder_positions_changed"] for k in p10),
            ten_pairs_median_group_changed=med(H[f"L{L}|{pj}|{k}"]["frac_columns_group_changed"] for k in p10))
            for L in LAY for pj in ("gu", "down")},
        conditioning=hraw["conditioning"], cholesky=hraw["per_config"],
        tokens=dict(per_projection=65536, per_input_dim_gu=65536 / 1536, per_input_dim_down=65536 / 8960,
                    moe=moe["geometry"]),
        activation_drift=gc_["activation_drift"],
        note="rel_fro_trace_normalized = ||H/trH - H0/trH0||_F / ||H0/trH0||_F (moe_calib.frozen_routing.hessian_metrics), the "
             "historical Family D decision metric; gate/up share H_gu")
    jdump(hs, p("hessian_sensitivity.json"))

    # ---------- dense vs MoE + decision
    sd_log_d, sd_log_m = var["sd_log"], moe["ppl"]["sd_log"]
    R_PPL = sd_log_d / sd_log_m
    R_W = wd["ten_pairs"]["frac_changed"] / moe["weights_10pairs"]["frac_changed"]
    if blocked:
        outcome = "BLOCKED"
    elif R_W >= 0.5 and R_PPL >= 0.5:
        outcome = "A-SUBSTANTIAL-DENSE-SENSITIVITY"
    elif R_W < 0.1 and R_PPL < 0.1:
        outcome = "B-SMALL-DENSE-SENSITIVITY"
    elif R_W >= 0.5 and R_PPL < 0.5:
        outcome = "C-WEIGHT-SENSITIVE-PPL-STABLE"
    elif R_W < 0.1 and R_PPL >= 0.5:
        outcome = "D-WEIGHT-STABLE-PPL-SENSITIVE"
    else:
        outcome = "E-INTERMEDIATE"
    g = moe["geometry"]
    cmp_ = dict(
        rows={
            "model": ["Qwen2.5-1.5B (dense, 28 layers)", "Qwen1.5-MoE-A2.7B (24 layers, 60 routed experts, top-4)"],
            "quantized": [f"4 MLP layers {LAY}, 12 matrices", "4 routed-expert layers [1, 8, 16, 23], 720 matrices"],
            "quantized_params": [pr["quantization"]["n_quantized_params"], g["n_quantized_params"]],
            "frac_of_model_params": [pr["quantization"]["frac_quantized"], g["frac_quantized_of_model"]],
            "matrix_dims_gu_down_in": [[1536, 8960], [g["d_gu"], g["d_down"]]],
            "calibration": ["FR s0 128 x 512, identical token ids", "FR s0 128 x 512"],
            "tokens_per_matrix": [65536, f"median {g['tokens_per_expert']['median']:.0f} (p10 {g['tokens_per_expert']['p10']:.0f}, p90 {g['tokens_per_expert']['p90']:.0f})"],
            "tokens_per_input_dim_gu": [65536 / 1536, g["tokens_per_input_dim_gu_median"]],
            "tokens_per_input_dim_down": [65536 / 8960, g["tokens_per_input_dim_down_median"]],
            "groupings": ["5 (seed 0)", "5 (seed 0)"],
            "bf16_ppl": [bf["ppl"], moe["ppl"]["bf16_ppl"]],
            "mean_quantized_ppl": [var["mean"], moe["ppl"]["mean"]],
            "within_seed_sd_ppl": [var["sd"], moe["ppl"]["sd"]],
            "sd_log_ppl": [sd_log_d, sd_log_m],
            "cv_ppl": [var["cv"], moe["ppl"]["cv"]],
            "sd_over_quant_penalty": [var["sd_over_quant_penalty"], moe["ppl"]["sd"] / (moe["ppl"]["mean"] - moe["ppl"]["bf16_ppl"])],
            "max_pairwise_ppl_diff": [var["max_pairwise_diff"], moe["ppl"]["range"]],
            "bf16_elements_changed_10pairs": [wd["ten_pairs"]["frac_changed"], moe["weights_10pairs"]["frac_changed"]],
            "rel_frobenius_10pairs": [wd["ten_pairs"]["rel_frobenius"], moe["weights_10pairs"]["rel_frobenius"]],
            "ge8ulp_share_of_changed": [wd["ten_pairs"]["ulp"][">=8"], moe["weights_10pairs"]["ulp"][">=8"]],
            "bf16_elements_changed_vs_original": [wd["vs_original"]["frac_changed"], moe["weights_vs_original"]["frac_changed"]],
            "hessian_rel_fro_tn_G1111_vs_ORIGINAL_median": [med(v for v in hs["matched_to_moe"]["dense_G20261111_vs_ORIGINAL"].values() if v is not None),
                                                            f"AO {moe['hessian']['AO_activation_only_median']:.4f} / AA {moe['hessian']['AA_natural_median']:.4f}"],
            "hessian_rel_fro_tn_10pairs_median": [hs["ten_pairs"]["all"]["rel_fro_trace_normalized"]["median"], "not measured"],
            "actorder_positions_changed_vs_original": [hs["vs_original"]["all"]["frac_actorder_positions_changed"]["mean"],
                                                       f"AO {moe['hessian']['actorder_AO']['frac_positions_changed']:.3f} / AA {moe['hessian']['actorder_AA']['frac_positions_changed']:.3f}"],
            "columns_group_changed_vs_original": [hs["vs_original"]["all"]["frac_columns_group_changed"]["mean"],
                                                  f"AO {moe['hessian']['actorder_AO']['frac_columns_group_changed']:.4f} / AA {moe['hessian']['actorder_AA']['frac_columns_group_changed']:.4f}"],
        },
        columns=["dense control", "MoE Family D (FR s0)"],
        moe_between_seed_note=moe["pooled_3seed_sd"],
        R_PPL=R_PPL, R_PPL_ci95=ratio_ci(R_PPL, 5, 5) if R_PPL > 0 else [0.0, 0.0], R_W=R_W,
        R_relfrob=wd["ten_pairs"]["rel_frobenius"] / moe["weights_10pairs"]["rel_frobenius"],
        R_CV=var["cv"] / moe["ppl"]["cv"],
        outcome=outcome, rule=pr["comparison"]["rule"], caveats=pr["comparison"]["caveats"])
    jdump(cmp_, p("dense_moe_comparison.json"))

    # ---------- cost + inventory
    cr = jload(p("logs", "gpu_cost_raw.json"))
    import glob
    store = sum(os.path.getsize(f) for f in glob.glob(S.c("**", "*.pt"), recursive=True))
    jdump(dict(model_loads_s=cr["loads"], per_config=cr["configs"], peak_cpu_gb=cr.get("peak_cpu_gb"),
               peak_gpu_gb=max(v["peak_gpu_gb"] for v in cr["configs"].values()),
               cache_build_s={k: v["build_s"] for k, v in cm.items()}, storage_bytes=store, analysis_s=time.time() - t0),
          p("cost_results.json"))
    inv = {g: dict(installed_weights_sha256=res[g]["installed_weights_sha256"],
                   files={L: {k: res[g]["per_layer"][L][k] for k in ("weight_sha256", "codes_sha256", "hessian_sha256")}
                          for L in res[g]["per_layer"]},
                   cache_files={L: v["sha256"] for L, v in cm[g]["files"].items()},
                   weights_dir=S.c("weights", g), codes_dir=S.c("codes", g), hessians_dir=S.c("hessians", g),
                   cache_dir=S.c("cache", g)) for g in CONFIGS}
    for g in CONFIGS:                                    # re-hash every stored artifact
        for L, f in inv[g]["files"].items():
            inv[g].setdefault("rehash_ok", True)
            inv[g]["rehash_ok"] &= all(S.sha256_file(os.path.join(S.c(sub, g), f"L{L}.pt")) == f[key]
                                       for sub, key in (("weights", "weight_sha256"), ("codes", "codes_sha256"), ("hessians", "hessian_sha256")))
    jdump(dict(configs=inv, protocol_sha256=pr["protocol_sha256"]), p("artifact_inventory.json"))
    figures(LAY, var, wd, hs, moe, res, cmp_)
    write_summary(pr, au, bf, ck, blocked, var, wd, hs, cmp_, moe, res, LAY, rep)
    print(json.dumps(dict(outcome=outcome, R_PPL=R_PPL, R_W=R_W, sd=var["sd"], sd_log=var["sd_log"],
                          frac_changed=wd["ten_pairs"]["frac_changed"], blocked=blocked), indent=1))


# ---------------------------------------------------------------- figures
def figures(LAY, var, wd, hs, moe, res, cmp_):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C2, C3 = "#2a6fbb", "#d1603d", "#7a7a7a"
    fd = p("figures")
    labels = [g[-2:] for g in GROUPS]
    # Fig 1
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    d = np.array([res[g]["ppl"] for g in GROUPS])
    m = np.array(list(moe["ppl"]["ppl"].values()))
    for ax, y, col, name, bfp in ((axs[0], d, C1, "dense Qwen2.5-1.5B", None), (axs[1], m, C2, "MoE Qwen1.5-MoE FR s0", None)):
        ax.plot(range(5), y, "o", color=col, ms=8)
        ax.axhline(y.mean(), color=col, lw=1)
        ax.axhspan(y.mean() - sd(y), y.mean() + sd(y), color=col, alpha=.12)
        ax.set_xticks(range(5), [f"G…{l}" for l in labels])
        ax.set_title(f"{name}\nmean {y.mean():.4f}, SD {sd(y):.2e}, SD(log) {sd(np.log(y)):.2e}", fontsize=9)
        ax.set_ylabel("WikiText-2 PPL")
    axs[0].axhline(res["ORIGINAL"]["ppl"], color=C3, ls="--", lw=1, label="ORIGINAL batching (ref.)")
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig1_dense_grouping_ppl.png"), dpi=150); fig.savefig(os.path.join(fd, "fig1_dense_grouping_ppl.pdf")); plt.close(fig)
    # Fig 2
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, key, lab in ((axs[0], "sd_log_ppl", "within-seed SD(log PPL)"), (axs[1], "cv_ppl", "CV(PPL)")):
        vals = cmp_["rows"][key]
        ci = [sd_ci(v, 5) if v > 0 else [0, 0] for v in vals]
        ax.bar([0, 1], vals, color=[C1, C2], yerr=np.array([[v - c[0] for v, c in zip(vals, ci)], [c[1] - v for v, c in zip(vals, ci)]]),
               capsize=4)
        ax.set_xticks([0, 1], ["dense", "MoE FR s0"]); ax.set_title(lab, fontsize=9)
    fig.suptitle(f"5 groupings, seed 0 (95% chi-square CI); R_PPL = {cmp_['R_PPL']:.3f}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig2_dense_vs_moe_sensitivity.png"), dpi=150); fig.savefig(os.path.join(fd, "fig2_dense_vs_moe_sensitivity.pdf")); plt.close(fig)
    # Fig 3
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    cats = ["10 pairs", "vs ORIGINAL"]
    dv = [wd["ten_pairs"]["frac_changed"], wd["vs_original"]["frac_changed"]]
    mv = [moe["weights_10pairs"]["frac_changed"], moe["weights_vs_original"]["frac_changed"]]
    x = np.arange(2)
    axs[0].bar(x - .18, dv, .36, color=C1, label="dense"); axs[0].bar(x + .18, mv, .36, color=C2, label="MoE FR s0")
    axs[0].bar(x - .18, [wd["ten_pairs"]["codes"]["frac_codes_changed"], wd["vs_original"]["codes"]["frac_codes_changed"]], .12,
               color="k", label="dense integer codes")
    axs[0].set_xticks(x, cats); axs[0].set_ylabel("fraction of BF16 weight elements changed"); axs[0].legend(fontsize=8)
    bins = ["1", "2", "3-7", ">=8", "sign_change"]
    for i, (lab, u) in enumerate((("dense", wd["ten_pairs"]["ulp"]), ("MoE FR s0", moe["weights_10pairs"]["ulp"]))):
        bottom = 0
        for j, b in enumerate(bins):
            axs[1].bar(i, u[b], bottom=bottom, color=plt.cm.viridis(j / 4), label=b if i == 0 else None)
            bottom += u[b]
    axs[1].set_xticks([0, 1], ["dense", "MoE FR s0"]); axs[1].set_ylabel("share of changed elements (10 pairs)")
    axs[1].legend(title="BF16 ULP", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig3_weight_differences.png"), dpi=150); fig.savefig(os.path.join(fd, "fig3_weight_differences.pdf")); plt.close(fig)
    # Fig 4: Hessian perturbation vs weight change per (pair, layer, projection)
    hr = jload(p("logs", "hessian_raw.json"))["pairs"]
    Cd = jload(p("logs", "weight_differences_raw.json"))["codes"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for pj, mk in (("gate_proj", "o"), ("up_proj", "s"), ("down_proj", "^")):
        hx, wy = [], []
        for key, cd in Cd.items():
            for L in LAY:
                hk = f"L{L}|{'down' if pj == 'down_proj' else 'gu'}|{key}"
                if hk in hr:
                    hx.append(hr[hk]["rel_fro_trace_normalized"]); wy.append(cd["per_layer_proj"][f"{L}.{pj}"]["frac_bf16_changed"])
        ax.scatter(np.maximum(hx, 1e-12), wy, marker=mk, s=18, alpha=.7, label=f"dense {pj}")
    ax.scatter([moe["hessian"]["AO_activation_only_median"], moe["hessian"]["AA_natural_median"]],
               [moe["weights_vs_original"]["frac_changed"]] * 2, color=C2, marker="*", s=120, label="MoE FR s0 (AO / AA median)")
    ax.set_xscale("log"); ax.set_xlabel("Hessian rel. Frobenius diff. (trace-normalized)"); ax.set_ylabel("frac. BF16 elements changed")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(fd, "fig4_hessian_vs_weight_change.png"), dpi=150); fig.savefig(os.path.join(fd, "fig4_hessian_vs_weight_change.pdf")); plt.close(fig)
    # Fig 5
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    plp = wd["ten_pairs"]["per_layer_proj"]
    x = np.arange(len(LAY))
    for i, pj in enumerate(PROJ):
        axs[0].bar(x + (i - 1) * .25, [plp[f"{L}.{pj}"]["frac_bf16_changed"] for L in LAY], .25, label=pj)
        axs[1].bar(x + (i - 1) * .25, [plp[f"{L}.{pj}"]["frac_codes_changed"] for L in LAY], .25, label=pj)
    for ax, t in ((axs[0], "BF16 elements changed (mean of 10 pairs)"), (axs[1], "integer codes changed (mean of 10 pairs)")):
        ax.set_xticks(x, [f"L{L}" for L in LAY]); ax.set_title(t, fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig5_per_projection_sensitivity.png"), dpi=150); fig.savefig(os.path.join(fd, "fig5_per_projection_sensitivity.pdf")); plt.close(fig)


# ---------------------------------------------------------------- summary
def write_summary(pr, au, bf, ck, blocked, var, wd, hs, cmp_, moe, res, LAY, rep):
    L = []
    w = L.append
    R = cmp_["rows"]
    w("# Dense-model control: GPTQ calibration-grouping reproducibility (Qwen2.5-1.5B)\n")
    w(f"**Outcome: {cmp_['outcome']}** - R_PPL = {cmp_['R_PPL']:.3f} (SD(log PPL) dense / MoE FR s0), R_W = {cmp_['R_W']:.3f} "
      f"(mean pairwise BF16 elements changed, dense / MoE FR s0); protocol sha256 {pr['protocol_sha256'][:16]}.\n")
    w("VERDICT_PARAGRAPH\n")
    w("## Setup\n")
    w(f"- Model `{pr['model']['id']}` @ `{pr['model']['revision'][:12]}` (checkpoint sha256 `{pr['model']['checkpoint_sha256'][:16]}…`), "
      f"depth {au['dense_model']['depth']}, hidden {au['dense_model']['hidden']}, intermediate {au['dense_model']['intermediate']}; BF16, eager attention, layer-wise offload.")
    w(f"- Quantized: mlp.gate/up/down_proj of layers {LAY} (MoE 1/8/16/23 by relative depth); 12 tensors, "
      f"{pr['quantization']['n_quantized_params']:,} parameters = {100 * pr['quantization']['frac_quantized']:.1f} % of the model.")
    w("- GPTQ: existing `moe_calib.gptq.quantize_expert`, 4-bit, g128, percdamp 1.0 (x mean diag H), act-order, BF16 effective weights.")
    w("- Calibration: FR s0 (128 x 512 C4 rows, identical token ids to Family D), K = 4 blocks; groupings G20261111-15 of the same "
      "217-row union exactly as Family D (batched pristine BF16 forward -> layer-input cache; Hessians one sequence at a time).")
    w(f"- Evaluation: the historical 128 x 512 WikiText-2 windows (dense re-tokenization reproduces them bitwise), batch 8. BF16 PPL {bf['ppl']:.6f}.\n")
    w("## Mandatory checks\n")
    w("| check | result |\n|---|---|")
    for k, v in ck.items():
        w(f"| {k} | {v} |")
    w(f"\nBlocked: {blocked}.\n")
    w("## Grouping PPL\n")
    w("| config | PPL | − PPL(G20261111) | − BF16 | installed sha256 |\n|---|---|---|---|---|")
    for g in CONFIGS:
        w(f"| {g} | {res[g]['ppl']:.6f} | {res[g]['ppl'] - res[REF]['ppl']:+.6f} | {res[g]['ppl'] - bf['ppl']:+.6f} | `{res[g]['installed_weights_sha256'][:12]}` |")
    w(f"\n| statistic (5 groupings) | dense | MoE FR s0 |\n|---|---|---|")
    w(f"| mean PPL | {var['mean']:.6f} | {moe['ppl']['mean']:.6f} |")
    w(f"| SD(PPL) | {var['sd']:.6f} (95 % {var['sd_ci95'][0]:.6f}–{var['sd_ci95'][1]:.6f}) | {moe['ppl']['sd']:.6f} |")
    w(f"| SD(log PPL) | {var['sd_log']:.3e} | {moe['ppl']['sd_log']:.3e} |")
    w(f"| CV(PPL) | {var['cv']:.3e} | {moe['ppl']['cv']:.3e} |")
    w(f"| min / max | {var['min']:.6f} / {var['max']:.6f} | {moe['ppl']['min']:.6f} / {moe['ppl']['max']:.6f} |")
    w(f"| max pairwise difference | {var['max_pairwise_diff']:.6f} | {moe['ppl']['range']:.6f} |")
    w(f"| SD / (mean − BF16) | {R['sd_over_quant_penalty'][0]:.4f} | {R['sd_over_quant_penalty'][1]:.4f} |")
    w(f"| distinct installed weight sets | {var['n_distinct_installed_weights']} | 5 |")
    w(f"\nR_PPL = {cmp_['R_PPL']:.3f} (F(4,4) 95 % {cmp_['R_PPL_ci95'][0]:.3f}–{cmp_['R_PPL_ci95'][1]:.3f}); R_CV = {cmp_['R_CV']:.3f}. "
      f"MoE pooled 3-seed within-seed SDs (FR {moe['pooled_3seed_sd']['FR']:.4f}) are not used as single-seed estimates. "
      f"Mean per-window paired NLL SD over the 10 pairs: {var['mean_per_window_sd_over_pairs']:.2e}.\n")
    w("## Weights and integer codes\n")
    w("| | dense 10 pairs | dense vs ORIGINAL | MoE FR s0 10 pairs | MoE FR s0 vs ORIGINAL |\n|---|---|---|---|---|")
    t, o, mt, mo = wd["ten_pairs"], wd["vs_original"], moe["weights_10pairs"], moe["weights_vs_original"]
    w(f"| BF16 elements changed | {t['frac_changed']:.4f} | {o['frac_changed']:.4f} | {mt['frac_changed']:.4f} | {mo['frac_changed']:.4f} |")
    w(f"| rel. Frobenius | {t['rel_frobenius']:.2e} | {o['rel_frobenius']:.2e} | {mt['rel_frobenius']:.2e} | {mo['rel_frobenius']:.2e} |")
    w(f"| mean abs diff | {t['mean_abs_diff']:.2e} | {o['mean_abs_diff']:.2e} | {mt['mean_abs_diff']:.2e} | {mo['mean_abs_diff']:.2e} |")
    for b in ("1", "2", "3-7", ">=8", "sign_change"):
        w(f"| ULP share {b} | {t['ulp'][b]:.3f} | {o['ulp'][b]:.3f} | {mt['ulp'][b]:.3f} | {mo['ulp'][b]:.3f} |")
    w(f"| integer codes changed | {t['codes']['frac_codes_changed']:.4f} | {o['codes']['frac_codes_changed']:.4f} | n/a | n/a |")
    w(f"| changed BF16 elements with a code change | {t['codes']['frac_bf16_changed_with_code_change']:.3f} | {o['codes']['frac_bf16_changed_with_code_change']:.3f} | n/a | n/a |")
    w(f"| elements in columns whose group changed | {t['codes']['frac_elements_in_group_changed_columns']:.4f} | {o['codes']['frac_elements_in_group_changed_columns']:.4f} | n/a | n/a |")
    w("\nPer projection (10 pairs, BF16 changed): " + ", ".join(f"{k} {v:.4f}" for k, v in t["per_projection"].items()) +
      "; per layer: " + ", ".join(f"L{k} {v:.4f}" for k, v in t["per_layer"].items()) + ".\n")
    w("## Hessians and act-order\n")
    w("| | gu | down | all |\n|---|---|---|---|")
    for lab, blk in (("10 pairs", hs["ten_pairs"]), ("vs G20261111", hs["vs_reference"]), ("vs ORIGINAL", hs["vs_original"])):
        w(f"| {lab}: rel. Frob. (trace-norm.), median | {blk['gu']['rel_fro_trace_normalized']['median']:.2e} | "
          f"{blk['down']['rel_fro_trace_normalized']['median']:.2e} | {blk['all']['rel_fro_trace_normalized']['median']:.2e} |")
        w(f"| {lab}: act-order positions changed, mean | {blk['gu']['frac_actorder_positions_changed']['mean']:.4f} | "
          f"{blk['down']['frac_actorder_positions_changed']['mean']:.4f} | {blk['all']['frac_actorder_positions_changed']['mean']:.4f} |")
        w(f"| {lab}: columns changing group, mean | {blk['gu']['frac_columns_group_changed']['mean']:.4f} | "
          f"{blk['down']['frac_columns_group_changed']['mean']:.4f} | {blk['all']['frac_columns_group_changed']['mean']:.4f} |")
    mh = moe["hessian"]
    w(f"\nMoE FR s0 (G20261111 vs ORIGINAL, 16 diagnostic Hessians): trace-normalized median AA {mh['AA_natural_median']:.4f} "
      f"(gu {mh['AA_by_projection']['gu']:.4f}, down {mh['AA_by_projection']['down']:.4f}); activation-only AO {mh['AO_activation_only_median']:.4f} "
      f"(gu {mh['AO_by_projection']['gu']:.4f}, down {mh['AO_by_projection']['down']:.4f}); act-order positions changed AA {mh['actorder_AA']['frac_positions_changed']:.3f}, "
      f"columns changing group AA {mh['actorder_AA']['frac_columns_group_changed']:.4f}.\n")
    cond = hs["conditioning"]
    w("| layer.proj | cond(damped H), ref | cond(H), ref | eff. rank frac | diag CV |\n|---|---|---|---|---|")
    for Lx in LAY:
        for pj in ("gu", "down"):
            cc = cond[f"L{Lx}|{pj}|{REF}"]
            w(f"| L{Lx}.{pj} | {cc['cond_damped']:.2f} | {cc['cond_undamped']:.2e} | {cc['eff_rank_frac']:.3f} | {cc['diag_cv']:.2f} |")
    ad = hs["activation_drift"]
    w("\n**Calibration activations (layer-input cache) vs G20261111:** " + "; ".join(
        f"L{Lx}: rows bitwise " + ", ".join(f"{g[-2:]} {ad[f'L{Lx}|{g}_vs_{REF}']['frac_rows_bitwise']:.3f}" for g in GROUPS if g != REF)
        for Lx in LAY) + ".")
    w("\n**vs ORIGINAL:** " + "; ".join(f"L{Lx}: rel L2 " + ", ".join(f"{ad[f'L{Lx}|{g}_vs_ORIGINAL']['rel_l2']:.1e}" for g in GROUPS) for Lx in LAY) + ".\n")
    w("## Dense vs MoE\n")
    w("| metric | dense control | MoE Family D (FR s0) |\n|---|---|---|")
    for k, v in R.items():
        fm = lambda x: f"{x:.4g}" if isinstance(x, float) else str(x)
        w(f"| {k} | {fm(v[0])} | {fm(v[1])} |")
    w(f"\nR_W = {cmp_['R_W']:.3f}; R_relFrob = {cmp_['R_relfrob']:.3f}; R_PPL = {cmp_['R_PPL']:.3f}. {cmp_['caveats']}.\n")
    w("ANSWERS_AND_LIMITATIONS\n")
    w("## Files\n\n- Code: `moe_calib/dense_gptq.py`, `diagnostics/dense_grouping_scope.py`, `diagnostics/report_dense_grouping_scope.py`, "
      "`tests/test_dense_grouping_scope.py`.\n- Outputs: `runs_dense_grouping_scope/` (required JSONs, `figures/fig1`-`fig5`, `logs/`).\n"
      f"- Weights, integer codes, Hessians and caches: `{S.CACHE}/{{weights,codes,hessians,cache}}/<config>/`.\n")
    txt = "\n".join(L) + "\n"
    it = jload(p("logs", "interpretation.json")) or {}
    txt = txt.replace("VERDICT_PARAGRAPH", it.get("verdict_paragraph", "_(interpretation pending)_"))
    txt = txt.replace("ANSWERS_AND_LIMITATIONS", it.get("answers_and_limitations", "_(interpretation pending)_"))
    open(p("summary.md"), "w").write(txt)
