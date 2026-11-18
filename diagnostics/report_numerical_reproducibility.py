"""FP64 canonical GPTQ results + numerical-reproducibility synthesis. Reads only stored artifacts (this study's
outputs and the historical order-null / act-order / frozen-routing / multi-grouping / cross-seed / partial-PPL /
legacy-metric runs). No GPU work."""
from __future__ import annotations

import json, os, subprocess, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import fp64_canonical_gptq as M           # noqa: E402
import full_depth_pilot as FD             # noqa: E402
from moe_calib import cross_seed_grouping as CSG   # noqa: E402
from moe_calib import fp64_canonical as FC64       # noqa: E402
from moe_calib import frozen_routing as FRT        # noqa: E402

p, jload, jdump, hon = M.p, M.jload, M.jdump, M.hon
LAYERS, STRATS = M.LAYERS, M.STRATS
SEEDS = (20261121, 20261122, 20261123, 20261124, 20261125)
PROJ = ("gate_proj", "up_proj", "down_proj")


def J(path):
    return json.load(open(os.path.join(ROOT, path)))


def sd(x):
    return float(np.std(np.asarray(x, float), ddof=1))


# ----------------------------------------------------------------------------- canonical results
def load_results():
    can = {c: jload("results", f"{c}_s0_FP64_CANONICAL.json") for c in STRATS}
    rep = jload("results", "FR_s0_FP64_CANONICAL_repeat.json")
    oo = {c: json.load(open(M.XR.oo_result_path(c))) for c in STRATS}
    on = {c: {s: json.load(open(hon("results", f"{c}_O{s}.json"))) for s in SEEDS} for c in STRATS}
    return can, rep, oo, on


def weight_quantizer(can, oo):
    base = {c: J(f"runs_hessian_order_null/baseline_regression/{c}.json") for c in STRATS}
    out = {}
    for c in STRATS:
        Wo = {L: torch.load(os.path.join(oo[c]["weights_dir"], f"L{L}.pt")) for L in LAYERS}
        per_layer, n_id, ch_exp = {}, 0, 0
        for L in LAYERS:
            W = torch.load(os.path.join(can[c]["weights_dir"], f"L{L}.pt"))
            per_layer[str(L)] = FRT.weight_diff(W, Wo[L])
            for e in range(60):
                ks = [f"{e}.{k}" for k in PROJ]
                d = FRT.weight_diff({k: W[k] for k in ks}, {k: Wo[L][k] for k in ks})
                n_id += d["n_tensors_identical"]
                ch_exp += d["n_differ"] > 0
            del W
        tot = {k: sum(v[k] for v in per_layer.values()) for k in ("n", "n_differ", "d2", "o2")}
        tot.update(frac_differ=tot["n_differ"] / tot["n"], rel_l2=(tot["d2"] / tot["o2"]) ** .5)
        Qb, Q = torch.load(base[c]["quantizer_file"]), torch.load(can[c]["quantizer_file"])
        rows = []
        for ek, ent in Qb["check_experts"].items():
            for k in PROJ:
                a, b = ent[k], Q["check_experts"][ek][k]
                sdiff = (b["scale"] - a["scale"]).abs() / a["scale"].abs()
                cd = b["codes"] != a["codes"]
                rows.append(dict(key=f"{ek}.{k}", frac_scale_differ=float((b["scale"] != a["scale"]).float().mean()),
                                 median_scale_rel_diff=float(sdiff.median()), max_scale_rel_diff=float(sdiff.max()),
                                 frac_zero_differ=float((b["zero"] != a["zero"]).float().mean()),
                                 frac_codes_differ=float(cd.float().mean()), n_codes_differ=int(cd.sum()), n=int(cd.numel())))
        # BF16 changes in the check experts with unchanged integer codes
        same_code_changed, changed = 0, 0
        for ek in Qb["check_experts"]:
            L, e = map(int, ek.split("."))
            W = torch.load(os.path.join(can[c]["weights_dir"], f"L{L}.pt"))
            for k in PROJ:
                dw = W[f"{e}.{k}"] != Wo[L][f"{e}.{k}"]
                cd = Q["check_experts"][ek][k]["codes"] != Qb["check_experts"][ek][k]["codes"]
                changed += int(dw.sum()); same_code_changed += int((dw & ~cd).sum())
            del W
        out[c] = dict(vs_OO=dict(total=tot, per_layer=per_layer, n_projections=720, n_projections_identical=n_id,
                                 n_experts_changed=ch_exp),
                      quantizer_check_experts=dict(
                          n_projections=len(rows), frac_scale_differ=float(np.mean([r["frac_scale_differ"] for r in rows])),
                          median_scale_rel_diff=float(np.median([r["median_scale_rel_diff"] for r in rows])),
                          max_scale_rel_diff=float(max(r["max_scale_rel_diff"] for r in rows)),
                          frac_zero_differ=float(np.mean([r["frac_zero_differ"] for r in rows])),
                          n_codes_differ=int(sum(r["n_codes_differ"] for r in rows)),
                          frac_codes_differ=sum(r["n_codes_differ"] for r in rows) / sum(r["n"] for r in rows),
                          bf16_changed_elements=changed, bf16_changed_with_same_code=same_code_changed,
                          frac_bf16_changes_with_same_code=same_code_changed / changed if changed else None,
                          per_projection=rows))
        del Wo
        print(f"[report] weights {c}: frac {tot['frac_differ']:.3e} rel {tot['rel_l2']:.3e}", flush=True)
    return out


def recon_vs_oo(can, oo):
    out = {}
    for c in STRATS:
        rp = {(d["layer"], d["expert"]): d["mean"] for d in oo[c]["per_expert"]}
        rel = np.array([(x["mean"] - rp[(x["layer"], x["expert"])]) / rp[(x["layer"], x["expert"])] for x in can[c]["per_expert"]])
        out[c] = dict(recon=can[c]["recon"], recon_OO=oo[c]["recon"], rel_change=(can[c]["recon"] - oo[c]["recon"]) / oo[c]["recon"],
                      n_experts_changed=int((rel != 0).sum()), per_expert_median_abs_rel=float(np.median(np.abs(rel))),
                      per_expert_max_abs_rel=float(np.abs(rel).max()))
    out["FC_penalty_over_FR"] = dict(canonical=can["FC"]["recon"] / can["FR"]["recon"] - 1,
                                     OO=oo["FC"]["recon"] / oo["FR"]["recon"] - 1)
    return out


def main():
    os.makedirs(p("figures"), exist_ok=True)
    proto = jload("protocol.json")
    inv = jload("artifact_inventory.json")
    can, rep, oo, on = load_results()
    hs = {c: jload("logs", f"hessians_{c}.json") for c in STRATS}
    reg = jload("logs", "regression_FR.json")
    ctl = {c: J(f"runs_hessian_order_null/fp64_control/{c}.json")["summary"] for c in STRATS}
    tok = np.array(can["FR"]["per_window_tokens"], float)

    # ---------------- canonical Hessian manifest + order invariance
    jdump({c: dict(store=hs[c]["summary"]["store"], store_sha256=hs[c]["summary"]["store_sha256"],
                   store_bytes=hs[c]["summary"]["store_bytes"], n_hessians=hs[c]["summary"]["n_hessians"],
                   sha_canonical={k: v["sha_cast_O0"] for k, v in hs[c]["keys"].items()},
                   sha64={k: v["sha64_O0"] for k, v in hs[c]["keys"].items()},
                   construction=proto["hessian_accumulation"]) for c in STRATS}, p("canonical_hessian_manifest.json"))
    oi = {c: dict(orders_tested=["O0", "O20261121", "O20261122"], n_orders=3,
                  **{k: hs[c]["summary"][k] for k in ("n_hessians", "fp32_path_equals_preflight_all", "block_tokens_equal_all",
                                                      "fp64_order_invariant_all", "cast_order_invariant_all",
                                                      "reproduces_control_fp64_all", "finite_all", "symmetric_all")},
                  historical_control=ctl[c]) for c in STRATS}
    oi["scope"] = "bitwise invariance verified for 3 accumulation orders (original + the 2 historical control orders); " \
                  "not a universal order-independence claim"
    jdump(oi, p("fp64_order_invariance.json"))
    order_ok = all(oi[c]["fp64_order_invariant_all"] and oi[c]["cast_order_invariant_all"] for c in STRATS)

    # ---------------- historical regression
    hreg = dict(loaded_pathway_FR=dict(weights_equal_OO=reg["weights_equal_OO"], recon_equal_OO=reg["recon_equal_OO"],
                                       valid=reg["valid"], passed=reg["passed"]),
                fp32_path_equals_stored_preflight={c: hs[c]["summary"]["fp32_path_equals_preflight_all"] for c in STRATS},
                fp64_reproduces_historical_control={c: hs[c]["summary"]["reproduces_control_fp64_all"] for c in STRATS},
                order_null_ppl_exact={c: dict(OO=oo[c]["ppl"], **{f"O{s}": on[c][s]["ppl"] for s in SEEDS}) for c in STRATS},
                oo_ppl_equals_protocol={c: oo[c]["ppl"] == proto["regression"]["expected"][c] for c in STRATS},
                order_null_equals_protocol={c: all(on[c][s]["ppl"] == proto["regression"]["order_null_expected"][c][f"O{s}"]
                                                   for s in SEEDS) for c in STRATS})
    hreg["passed"] = bool(reg["passed"] and all(hreg["fp32_path_equals_stored_preflight"].values()) and
                          all(hreg["fp64_reproduces_historical_control"].values()) and all(hreg["oo_ppl_equals_protocol"].values())
                          and all(hreg["order_null_equals_protocol"].values()))
    jdump(hreg, p("historical_regression.json"))

    # ---------------- canonical quantization + determinism
    det = dict(weights_sha_equal=rep["installed_weights_sha256"] == can["FR"]["installed_weights_sha256"],
               weight_files_equal=all(rep["weight_files"][L]["sha256"] == can["FR"]["weight_files"][L]["sha256"] for L in can["FR"]["weight_files"]),
               recon_equal=rep["recon"] == can["FR"]["recon"],
               per_expert_equal=rep["per_expert"] == can["FR"]["per_expert"],
               ppl_equal=rep["ppl"] == can["FR"]["ppl"], per_window_equal=rep["per_window_nll"] == can["FR"]["per_window_nll"])
    reproducible = all(det.values())
    keys_q = ["n_quantized_experts", "n_projection_tensors", "rtn_fallbacks", "projections_unchanged_from_bf16",
              "nonfinite_weights", "untouched_params_unchanged", "router_weights_unchanged", "token_counts_equal_R_O",
              "routing_equal_R_O_all_tokens", "perm_equal_original", "hessians_served_equal_canonical", "integrity_pass",
              "I1_all_installed_equal_cast", "I2_all_module_equal_reference", "I3_all_cast_correctly_rounded", "I4_all_finite",
              "frozen_old_check_all_pass", "natural_actorder", "installed_weights_sha256", "weights_equal_OO", "valid",
              "quantize_s", "hessian_s", "gptq_s", "peak_gpu_gb", "peak_cpu_gb"]
    jdump(dict(configs={c: {k: can[c].get(k) for k in keys_q} | dict(weight_files=can[c]["weight_files"]) for c in STRATS},
               repeat_FR={k: rep.get(k) for k in keys_q}, determinism_FR=det, reproducible=reproducible),
          p("canonical_quantization_results.json"))

    # ---------------- canonical PPL
    ppl = {}
    for c in STRATS:
        r = can[c]
        ppl[c] = dict(ppl=r["ppl"], mean_nll=r["mean_nll"], total_nll=r["total_nll"], total_tokens=r["total_tokens"],
                      per_window_nll=r["per_window_nll"], per_window_tokens=r["per_window_tokens"],
                      installed_weights_sha256=r["installed_weights_sha256"], eval_s=r.get("eval_s_both"),
                      repeat_eval_identical=r["repeat_eval_per_window_identical"])
    d_fp64 = can["FC"]["ppl"] - can["FR"]["ppl"]
    d_fp64_nll = can["FC"]["mean_nll"] - can["FR"]["mean_nll"]
    wb = FD.paired_bootstrap(can["FC"]["per_window_nll"], can["FR"]["per_window_nll"], tok)
    jdump(dict(per_strategy=ppl, delta_fp64_ppl=d_fp64, delta_fp64_mean_nll=d_fp64_nll,
               window_bootstrap_mean_nll=dict(wb, note="paired window bootstrap (same definition as the historical "
                                                       "studies, full_depth_pilot.paired_bootstrap); conditional on these "
                                                       "two fixed quantized models")),
          p("canonical_ppl_results.json"))

    # ---------------- order-null comparison
    deltas = [on["FC"][s]["ppl"] - on["FR"][s]["ppl"] for s in SEEDS]
    d_oo = oo["FC"]["ppl"] - oo["FR"]["ppl"]
    rows = [dict(policy="Original FP32", FR=oo["FR"]["ppl"], FC=oo["FC"]["ppl"], delta=d_oo)] + \
           [dict(policy=f"FP32 order {s}", FR=on["FR"][s]["ppl"], FC=on["FC"][s]["ppl"], delta=d) for s, d in zip(SEEDS, deltas)] + \
           [dict(policy="FP64 canonical", FR=can["FR"]["ppl"], FC=can["FC"]["ppl"], delta=d_fp64)]
    frs = [on["FR"][s]["ppl"] for s in SEEDS]; fcs = [on["FC"][s]["ppl"] for s in SEEDS]
    onc = dict(
        table=rows,
        q1_delta_inside_order_range=bool(min(deltas) <= d_fp64 <= max(deltas)),
        q1_delta_inside_range_incl_OO=bool(min(deltas + [d_oo]) <= d_fp64 <= max(deltas + [d_oo])),
        q1_FR_inside_range=bool(min(frs) <= can["FR"]["ppl"] <= max(frs)), q1_FC_inside_range=bool(min(fcs) <= can["FC"]["ppl"] <= max(fcs)),
        order_range_delta=[min(deltas), max(deltas)], order_range_FR=[min(frs), max(frs)], order_range_FC=[min(fcs), max(fcs)],
        q2_ranking_preserved=bool(np.sign(d_fp64) == np.sign(d_oo)),
        q3_vs_original=dict(FR=can["FR"]["ppl"] - oo["FR"]["ppl"], FC=can["FC"]["ppl"] - oo["FC"]["ppl"], delta=d_fp64 - d_oo),
        q4_vs_order_mean=dict(FR=can["FR"]["ppl"] - np.mean(frs), FC=can["FC"]["ppl"] - np.mean(fcs), delta=d_fp64 - np.mean(deltas)),
        order_sd=dict(FR=sd(frs), FC=sd(fcs), delta=sd(deltas)),
        q4_in_order_sd_units=dict(FR=(can["FR"]["ppl"] - np.mean(frs)) / sd(frs), FC=(can["FC"]["ppl"] - np.mean(fcs)) / sd(fcs),
                                  delta=(d_fp64 - np.mean(deltas)) / sd(deltas)),
        q5="FP64 accumulation removes the accumulation-ORDER variability by construction (bitwise identical Hessians "
           "for the tested orders), so repeated canonical runs give one value; it does not reduce, and cannot measure, "
           "variability from any other execution choice",
        interpretation="the canonical point is one deterministic configuration; its distance from the order mean is not an "
                       "estimation error")
    jdump(onc, p("order_null_comparison.json"))

    # ---------------- weight / quantizer / recon / per-window sensitivity
    wq = weight_quantizer(can, oo)
    rc = recon_vs_oo(can, oo)
    pw = {}
    for c in STRATS:
        a, b = np.array(can[c]["per_window_nll"]), np.array(oo[c]["per_window_nll"])
        pw[c] = dict(abs_ppl_diff_vs_OO=abs(can[c]["ppl"] - oo[c]["ppl"]), n_windows_differ=int((a != b).sum()),
                     mean_abs_window_nll_diff=float(np.abs(a - b).mean()), max_abs_window_nll_diff=float(np.abs(a - b).max()),
                     exceeds_deterministic_repeatability=bool(can[c]["ppl"] != oo[c]["ppl"]),
                     repeatability_abs_diff=can[c]["repeat_eval_abs_diff_ppl"])
    on_w = J("runs_hessian_order_null/weight_differences.json")
    hlev = {c: {k: hs[c]["summary"][k] for k in ("frac_elements_differ_from_original", "rel_fro_vs_original_median",
                                                 "rel_fro_vs_original_max", "trace_rel_diff_max_abs", "natural_perm_equal_PO",
                                                 "original_natural_perm_equal_PO", "n_hessians_with_group_change_without_override",
                                                 "natural_frac_columns_changed_group_max")} |
                dict(factorization_diag_subset=dict(
                    n=len(hs[c]["summary"]["factorization_diag"]),
                    rel_damped_median=float(np.median([x["rel_damped_perturbation"] for x in hs[c]["summary"]["factorization_diag"]])),
                    rel_hinv_factor_median=float(np.median([x["rel_hinv_factor"] for x in hs[c]["summary"]["factorization_diag"]])),
                    rel_hinv_factor_max=float(max(x["rel_hinv_factor"] for x in hs[c]["summary"]["factorization_diag"]))))
            for c in STRATS}
    jdump(dict(hessian_level=hlev, weights_and_quantizer=wq, reconstruction=rc, ppl_level=pw,
               order_null_reference_weights={c: on_w.get(c, on_w) if isinstance(on_w, dict) else None for c in STRATS},
               distinction="integer-code flips vs FP32 scale changes vs BF16 effective-weight changes are reported "
                           "separately (check experts: 8 x 3 projections)"), p("weight_sensitivity.json"))

    # ---------------- mechanism comparison (existing studies)
    aof = J("runs_actorder_ablation/ppl_sensitivity.json")
    aws = J("runs_actorder_ablation/weight_sensitivity.json")["summary"]
    wr = J("runs_frozen_routing_robustness/weight_robustness.json")["summary"]
    hr = J("runs_frozen_routing_robustness/hessian_robustness.json")["summary"]
    pn = J("runs_hessian_order_null/paired_differences.json")
    pa = aof["ppl"]
    dAO = [b - a for a, b in zip(pa["FR"]["AO"], pa["FC"]["AO"])]
    dAOF = [b - a for a, b in zip(pa["FR"]["AOF"], pa["FC"]["AOF"])]
    dAA = [b - a for a, b in zip(pa["FR"]["AA"], pa["FC"]["AA"])]
    G = ["G20261111", "G20261112", "G20261113", "G20261114", "G20261115"]
    wmean = lambda c, cmp, k: float(np.mean([aws[c][g][cmp][k] for g in G]))
    wmean_r = lambda c, cmp, k: float(np.mean([wr[c][g][cmp][k] for g in G]))
    hmed = lambda c, cmp: [float(min(hr[c][g][cmp]["median"] for g in G)), float(max(hr[c][g][cmp]["median"] for g in G))]
    onw = {c: wq_ for c, wq_ in ((c, on_w[c]) for c in STRATS)} if all(c in on_w for c in STRATS) else on_w
    cs = J("runs_cross_seed_grouping/ppl_results.json")["configs"]
    cs_fr = {s: [cs[f"FR_s{s}_G{g[1:]}"]["ppl"] for g in G] for s in (0, 1, 2)} if f"FR_s0_{G[0]}" in cs else None
    mech = dict(
        A_accumulation_order=dict(
            seeds=1, variants=5, paired=True, perturbation="FP32 summation order of identical per-sequence contributions",
            fixed="sequences, activations, routing, contributions, act-order, groups, windows",
            hessian_metric=dict(kind="rel. Frobenius vs OO, median", value="8.4e-8 to 9.3e-8 (report table)"),
            weights=dict(frac_differ={c: [min(v["total"]["frac_differ"] for v in onw[c].values()),
                                          max(v["total"]["frac_differ"] for v in onw[c].values())] for c in STRATS},
                         rel_l2={c: [min(v["total"]["rel_l2"] for v in onw[c].values()),
                                     max(v["total"]["rel_l2"] for v in onw[c].values())] for c in STRATS}) if isinstance(onw, dict) and "FR" in onw else "see runs_hessian_order_null/weight_differences.json",
            recon_variation="mean relative change <= 1.3e-6", ppl=dict(FR_sd=pn["sd_fr"], FC_sd=pn["sd_fc"], delta_sd=pn["sd_delta"],
                                                                      delta_range=[pn["min_delta"], pn["max_delta"]], sign_changes_vs_OO=pn["sign_reversals"]),
            limitation="one seed, 5 deterministic orders, one hardware/software stack"),
        B_AOF=dict(seeds=1, variants=5, paired=True,
                   perturbation="BF16 calibration activations from alternative batch groupings; routing R_O and act-order P_O frozen",
                   hessian_metric=dict(kind="trace-normalized rel. diff vs OO, median per grouping", FR=hmed("FR", "AO_vs_OO"), FC=hmed("FC", "AO_vs_OO"),
                                       note="AOF Hessians are bitwise the AO Hessians"),
                   weights=dict(frac_differ={c: wmean(c, "AOF_vs_OO", "frac_differ") for c in STRATS},
                                rel_l2={c: wmean(c, "AOF_vs_OO", "rel_l2") for c in STRATS}),
                   ppl=dict(FR_sd=sd(pa["FR"]["AOF"]), FC_sd=sd(pa["FC"]["AOF"]), delta_sd=sd(dAOF),
                            delta_range=[min(dAOF), max(dAOF)], sign_changes_vs_OO=int(sum(np.sign(d) != np.sign(d_oo) for d in dAOF))),
                   limitation="act-order and group membership fixed jointly; one seed"),
        C_AO=dict(seeds=1, variants=5, paired=True,
                  perturbation="as B, act-order and groups follow each Hessian",
                  hessian_metric=dict(kind="trace-normalized rel. diff vs OO, median per grouping", FR=hmed("FR", "AO_vs_OO"), FC=hmed("FC", "AO_vs_OO")),
                  weights=dict(frac_differ={c: wmean(c, "AO_vs_OO", "frac_differ") for c in STRATS},
                               rel_l2={c: wmean(c, "AO_vs_OO", "rel_l2") for c in STRATS}),
                  ppl=dict(FR_sd=sd(pa["FR"]["AO"]), FC_sd=sd(pa["FC"]["AO"]), delta_sd=sd(dAO),
                           delta_range=[min(dAO), max(dAO)], sign_changes_vs_OO=int(sum(np.sign(d) != np.sign(d_oo) for d in dAO))),
                  limitation="one seed"),
        D_natural_grouping_seed0=dict(
            seeds=1, variants=5, paired=True, perturbation="batch grouping: activations, routing, Hessians, act-order, weights",
            hessian_metric=dict(kind="trace-normalized rel. diff vs OO, median per grouping", FR=hmed("FR", "AA_vs_OO"), FC=hmed("FC", "AA_vs_OO")),
            weights=dict(frac_differ={c: wmean_r(c, "AA_vs_OO", "frac_differ") for c in STRATS},
                         rel_l2={c: wmean_r(c, "AA_vs_OO", "rel_l2") for c in STRATS}),
            ppl=dict(FR_sd=sd(pa["FR"]["AA"]), FC_sd=sd(pa["FC"]["AA"]), delta_sd=sd(dAA), delta_range=[min(dAA), max(dAA)],
                     sign_changes_vs_OO=int(sum(np.sign(d) != np.sign(d_oo) for d in dAA))),
            limitation="one seed; ORIGINAL extraction is a different policy and is not one of the 5 draws"),
        D3_natural_grouping_3seeds=dict(
            seeds=3, variants="5 per seed (nested)", paired=True, perturbation="as D",
            ppl=dict(pooled_within_seed_delta_sd=J("runs_cross_seed_grouping/variance_analysis.json")["summary"]["pooled_within_sd"]),
            limitation="3 allocation seeds; groupings nested, not crossed"),
        statistic_types="FR_sd / FC_sd = SD (ddof 1) of one strategy's PPL over the variants; delta_sd = SD of the paired "
                        "FC-FR difference. Different interventions; SDs are NOT additive and are not decomposed.",
        canonical_reference=dict(delta_fp64=d_fp64, delta_OO=d_oo))
    jdump(mech, p("mechanism_comparison.json"))

    # ---------------- cross-seed grouping reanalysis
    pdx = J("runs_cross_seed_grouping/paired_differences.json")["pairs"]
    va_hist = J("runs_cross_seed_grouping/variance_analysis.json")
    nested = {s: {g: pdx[f"s{s}_G{g}"]["delta_ppl"] for g in (20261111, 20261112, 20261113, 20261114, 20261115)} for s in (0, 1, 2)}
    nested_nll = {s: {g: pdx[f"s{s}_G{g}"]["delta_mean_nll"] for g in nested[s]} for s in nested}
    vs = CSG.variance_summary(nested)
    hb = CSG.hierarchical_bootstrap(nested)
    vs_nll = CSG.variance_summary(nested_nll)
    hb_nll = CSG.hierarchical_bootstrap(nested_nll)
    orig = J("runs_partial_ppl/summary.json")["paired"]["FC-FR"]
    allv = [v for s in nested for v in nested[s].values()]
    M_ = vs["overall_mean_delta"]
    csa = dict(per_seed_mean=vs["mean_delta"], per_seed_sd=vs["sd_grouping"], pooled_within_sd=vs["pooled_within_sd"],
               overall_mean=M_, overall_median=vs["overall_median_delta"], n_fc_worse=vs["n_fc_worse_total"],
               n_fc_better=vs["n_fc_better_total"], opposite_sign_fraction=float(np.mean(np.sign(allv) != np.sign(M_))),
               sd_between_seed_means=vs["sd_between_seed_means"], hierarchical_bootstrap=hb,
               matches_historical=dict(mean=M_ == va_hist["summary"]["overall_mean_delta"],
                                       pooled_sd=vs["pooled_within_sd"] == va_hist["summary"]["pooled_within_sd"],
                                       interval=hb["ci95"] == va_hist["hierarchical_bootstrap"]["ci95"]),
               mean_nll_estimand=dict(per_seed_mean=vs_nll["mean_delta"], overall_mean=vs_nll["overall_mean_delta"],
                                      pooled_within_sd=vs_nll["pooled_within_sd"], hierarchical_bootstrap=hb_nll,
                                      n_fc_worse=vs_nll["n_fc_worse_total"]),
               original_policy=dict(delta_ppl_by_seed=orig["delta_ppl"], window_boot_ci_ppl=orig["boot_ci"],
                                    note="ORIGINAL extraction (batch-8 cache_inputs.py) is a different policy; never pooled"),
               historical_decision=va_hist["decision"],
               caveat="3 allocation seeds -> 10 distinct outer resamples; the interval is descriptive and unstable")
    jdump(csa, p("cross_seed_grouping_analysis.json"))

    # ---------------- evaluation uncertainty vs execution variability
    mg = J("runs_multigroup_robustness/summary.json")["paired_window_bootstrap"]
    fixed = {k: dict(point=v["window_bootstrap"]["point"], ci95=v["window_bootstrap"]["ci95"], delta_ppl=v["delta_ppl"],
                     excludes_zero=bool(v["window_bootstrap"]["ci95"][0] > 0 or v["window_bootstrap"]["ci95"][1] < 0),
                     favours=("FR" if v["window_bootstrap"]["ci95"][0] > 0 else "FC" if v["window_bootstrap"]["ci95"][1] < 0 else None))
             for k, v in pdx.items()}
    eu = dict(fixed_model_window_intervals=fixed,
              n_excluding_zero=dict(favour_FR=sum(v["favours"] == "FR" for v in fixed.values()),
                                    favour_FC=sum(v["favours"] == "FC" for v in fixed.values())),
              original_s0_window_ci_mean_nll=mg["ORIGINAL"],
              canonical_window_ci_mean_nll=wb,
              execution_level=dict(hierarchical_interval_ppl=hb["ci95"], hierarchical_interval_mean_nll=hb_nll["ci95"],
                                   pooled_within_seed_sd_ppl=vs["pooled_within_sd"]),
              conditioning=dict(window="conditional on ONE fixed pair of quantized models; resamples evaluation windows",
                                execution="resamples allocation seeds and, within them, execution groupings; windows fixed"),
              conclusion="fixed-model window intervals exclude zero in BOTH directions for different valid groupings of "
                         "the same selections, so they do not measure execution-induced variability")
    jdump(eu, p("evaluation_uncertainty_comparison.json"))

    # ---------------- reconstruction vs PPL
    lm = J("runs_legacy_metric_reeval/moe_layer_output.json")
    rpa = {k: dict(seed=v["allocation_seed"], grouping=v["grouping_seed"], recon_penalty_rel=v["recon_penalty_rel"],
                   delta_ppl=v["delta_ppl"], layer_output_penalty=None) for k, v in pdx.items()}
    b1 = {}
    for s in (0, 1, 2):
        try:
            fr = lm["confirm"]["methods"][f"FR_s{s}"]["E_layer_mean_of_layers"]; fc = lm["confirm"]["methods"][f"FC_s{s}"]["E_layer_mean_of_layers"]
            b1[str(s)] = fc / fr - 1
        except KeyError:
            b1[str(s)] = None
    rp = [v["recon_penalty_rel"] for v in rpa.values()]
    jdump(dict(grouping_configs=rpa,
               recon_penalty_range=[min(rp), max(rp)], n_fc_recon_worse=sum(x > 0 for x in rp),
               delta_ppl_range=[min(allv), max(allv)], n_fc_ppl_worse=sum(x > 0 for x in allv),
               canonical=dict(recon_penalty=rc["FC_penalty_over_FR"]["canonical"], delta_ppl=d_fp64),
               OO=dict(recon_penalty=rc["FC_penalty_over_FR"]["OO"], delta_ppl=d_oo),
               layer_output_grouping_configs="unavailable (never evaluated for the grouping weight sets; not imputed)",
               layer_output_canonical_b1_separate_protocol=dict(
                   FC_over_FR_confirm=b1, source="runs_legacy_metric_reeval (CANONICAL-B1 extraction, different weights; "
                                                 "no matched PPL exists; shown separately, never mixed)"),
               hypotheses=["local objectives (expert / layer reconstruction) average over many tokens, PPL is driven by a "
                           "small set of high-loss predictions", "nonlinear downstream propagation through 20 BF16 layers",
                           "BF16 effective-weight and scale-bit sensitivity", "different local vs model-level objectives"],
               note="hypotheses only; no causal attribution to experts or tokens is measured"),
          p("reconstruction_ppl_alignment.json"))

    # ---------------- estimands
    jdump(dict(canonical=dict(definition=proto["estimands"]["canonical"], value_ppl=d_fp64, value_mean_nll=d_fp64_nll),
               order_null=dict(definition=proto["estimands"]["order_null"], mean_ppl=float(np.mean(deltas)), sd_ppl=sd(deltas)),
               execution_marginal=dict(definition=proto["estimands"]["execution_marginal"], ppl=M_, ppl_interval=hb["ci95"],
                                       mean_nll=vs_nll["overall_mean_delta"], mean_nll_interval=hb_nll["ci95"]),
               window_conditional=dict(definition=proto["estimands"]["window_conditional"]),
               nonlinearity="E_G[PPL] != exp(E_G[NLL]); the PPL estimand averages paired PPL differences, the NLL estimand "
                            "averages paired mean-NLL differences; both reported, both nested over (seed, grouping)"),
          p("statistical_estimands.json"))

    # ---------------- cost
    tr = jload("logs", "timing_raw.json")
    hsz = {c: hs[c]["summary"]["store_bytes"] for c in STRATS}
    wsz = {}
    for nm in ("FR_s0_FP64_CANONICAL", "FC_s0_FP64_CANONICAL", "FR_s0_FP64_CANONICAL_repeat", "FR_FP32_LOADED_REGRESSION"):
        d = os.path.join(M.STORE, "weights", nm)
        wsz[nm] = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) if os.path.isdir(d) else None
    jdump(dict(hessian_reconstruction={c: dict(replay_s_3orders=hs[c]["summary"]["replay_s_total"], wall_s=hs[c]["summary"]["wall_s"],
                                               peak_gpu_gb=hs[c]["summary"]["peak_gpu_gb"], peak_rss_gb=hs[c]["summary"]["peak_rss_gb"])
                                       for c in STRATS},
               accumulation_timing=tr,
               fp64_overhead=dict(fp32_minus_none=tr["median"]["fp32"] - tr["median"]["none"],
                                  fp64_minus_none=tr["median"]["fp64"] - tr["median"]["none"],
                                  ratio_fp64_over_fp32_accumulation=(tr["median"]["fp64"] - tr["median"]["none"]) /
                                  max(tr["median"]["fp32"] - tr["median"]["none"], 1e-9)),
               quantization={c: dict(quantize_s=can[c]["quantize_s"], gptq_s=can[c]["gptq_s"], peak_gpu_gb=can[c]["peak_gpu_gb"],
                                     peak_cpu_gb=can[c]["peak_cpu_gb"], eval_s=can[c].get("eval_s_both")) for c in STRATS},
               storage_bytes=dict(canonical_hessians=hsz, weights=wsz),
               deployment_note="a deployed canonical policy needs FP64 block buffers (2x FP32 memory: ~5.9 GB -> ~11.9 GB "
                               "per layer at K=4, or chunking) and one cast; the MVP's 3-order replays, stored FP32+FP64 "
                               "copies and repeat solves are experimental overhead"), p("cost_results.json"))

    # ---------------- correctness
    tl = p("logs", "pytest_full.log")
    tests_ok = os.path.exists(tl) and " passed" in open(tl).read() and " failed" not in open(tl).read()
    git_ok = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "moe_calib/gptq.py"], cwd=ROOT).returncode == 0
    hist_ok = all(M.sha256_file(os.path.join(ROOT, f)) == h for f, h in inv["historical_sha256"].items())
    oo_ok = all(M.sha256_file(os.path.join(oo[c]["weights_dir"], f"L{L}.pt")) == v["sha256"]
                for c in STRATS for L, v in oo[c]["weight_files"].items())
    cq = [can["FR"], can["FC"], rep]
    win = json.load(open(os.path.join(ROOT, "runs_partial_ppl", "evaluation_protocol.json")))
    chk = {
        "01_selection_hashes": all(v["match"] and v["matches_order_null_manifest"] for v in inv["selections"].values()),
        "02_no_calibration_test_leakage": inv["calibration_test_disjoint"]["ok"],
        "03_original_activation_hashes": all(v["sha256"] == v["expected"] for v in inv["original_cache"]["files"].values()),
        "04_contributions_unchanged": all(hs[c]["summary"]["fp32_path_equals_preflight_all"] for c in STRATS),
        "05_block_membership_unchanged": all(hs[c]["summary"]["block_tokens_equal_all"] for c in STRATS) and all(r["token_counts_equal_R_O"] for r in cq),
        "06_fp64_protocol": all(hs[c]["summary"]["reproduces_control_fp64_all"] for c in STRATS),
        "07_fp64_order_invariance": order_ok,
        "08_cast_once_and_served": all(r["hessians_served_equal_canonical"] == "480/480" for r in cq),
        "09_finite_symmetric": all(hs[c]["summary"]["finite_all"] and hs[c]["summary"]["symmetric_all"] for c in STRATS),
        "10_original_actorder": all(r["perm_equal_original"] == "240/240" for r in cq),
        "11_gptq_params": proto["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True),
        "12_solver_unchanged": git_ok and M.sha256_file(os.path.join(ROOT, "moe_calib/gptq.py")) == proto["code_sha256"]["moe_calib/gptq.py"],
        "13_no_rtn_fallback": all(r["rtn_fallbacks"] == 0 for r in cq),
        "14_240_experts": all(r["n_quantized_experts"] == 240 for r in cq),
        "15_720_tensors": all(r["n_projection_tensors"] == 720 and r["projections_unchanged_from_bf16"] == 0 for r in cq),
        "16_bf16_installation": all(r["integrity_pass"] and r["I1_all_installed_equal_cast"] and r["I3_all_cast_correctly_rounded"] and r["I4_all_finite"] for r in cq),
        "17_routers_shared_unchanged": all(r["router_weights_unchanged"] and r["routing_equal_R_O_all_tokens"] for r in cq),
        "18_nontarget_unchanged": all(r["untouched_params_unchanged"] and r["nonfinite_weights"] == 0 for r in cq),
        "19_windows_and_targets": all(r["total_tokens"] == 65408 for r in cq) and win["windows_sha256"] == proto["evaluation"]["windows_sha256"],
        "20_repeat_eval_identical": all(r["repeat_eval_per_window_identical"] for r in cq),
        "21_repeat_quantization_identical": reproducible,
        "22_historical_fp32_controls": hreg["passed"],
        "23_no_artifact_overwritten": bool(hist_ok and oo_ok),
        "24_tests_pass": bool(tests_ok),
    }
    cok = all(chk.values())
    jdump(dict(checks=chk, passed=cok), p("correctness_results.json"))

    # ---------------- verdicts
    blocked = not (cok and hreg["passed"])
    v_can = FC64.classify_canonical(blocked, order_ok, reproducible)
    # matched execution variants: cross-seed groupings (sign changes within seeds), AO/AOF/AA, accumulation order
    sign_changes = any(len({np.sign(v) for v in nested[s].values()}) > 1 for s in nested)
    ratio = (max(deltas) - min(deltas)) / abs(d_oo)
    v_exec = FC64.classify_execution(sign_changes, ratio, all(r["repeat_eval_per_window_identical"] for r in cq))
    dec = dict(canonical_verdict=v_can, execution_verdict=v_exec,
               execution_inputs=dict(sign_changes_within_seed=sign_changes, order_null_range_over_abs_delta_OO=ratio,
                                     n_seeds_with_sign_change=sum(len({np.sign(v) for v in nested[s].values()}) > 1 for s in nested)),
               fp64_not_used_for_execution_verdict=True, created=time.strftime("%F %T"))
    jdump(dec, p("decision.json"))
    summ = dict(decision=dec, fp64=dict(FR=can["FR"]["ppl"], FC=can["FC"]["ppl"], delta=d_fp64, delta_mean_nll=d_fp64_nll),
                order_null=onc, weights={c: wq[c]["vs_OO"]["total"] | dict(n_identical=wq[c]["vs_OO"]["n_projections_identical"],
                                                                           n_experts_changed=wq[c]["vs_OO"]["n_experts_changed"]) for c in STRATS},
                quantizer={c: {k: v for k, v in wq[c]["quantizer_check_experts"].items() if k != "per_projection"} for c in STRATS},
                recon=rc, ppl_level=pw, hessian_level=hlev, cross_seed=csa | dict(hierarchical_bootstrap=hb),
                correctness=cok)
    jdump(summ, p("summary.json"))
    figures(rows, deltas, d_oo, d_fp64, mech, nested, fixed, hb, rpa, hlev, wq, rc, pw, on_w, oo, can)
    print(json.dumps(dict(dec, checks_failed=[k for k, v in chk.items() if not v]), indent=1))


# ----------------------------------------------------------------------------- figures
def figures(rows, deltas, d_oo, d_fp64, mech, nested, fixed, hb, rpa, hlev, wq, rc, pw, on_w, oo, can):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, n):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(p("figures", f"{n}.{ext}"), dpi=150)
        plt.close(fig)
    # fig1: perturbation chain (FR canonical vs OO, measured at each level; log scale, not additive)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, c in enumerate(STRATS):
        vals = [hlev[c]["rel_fro_vs_original_median"], hlev[c]["factorization_diag_subset"]["rel_hinv_factor_median"],
                wq[c]["quantizer_check_experts"]["frac_codes_differ"], wq[c]["vs_OO"]["total"]["frac_differ"],
                wq[c]["vs_OO"]["total"]["rel_l2"], abs(rc[c]["rel_change"]) or 1e-12,
                pw[c]["abs_ppl_diff_vs_OO"] / oo[c]["ppl"]]
        ax.plot(range(len(vals)), vals, "o-", label=c)
    ax.set_yscale("log")
    ax.set_xticks(range(7), ["Hessian\nrel. Frob.", "H^-1 factor\nrel. diff", "int codes\nfrac. changed", "BF16 weights\nfrac. changed",
                             "weights\nrel. L2", "recon\n|rel. change|", "PPL\n|rel. change|"], fontsize=7)
    ax.set_title("FP64 canonical vs ORIGINAL FP32: measured response at each level (different units; not additive)", fontsize=9)
    ax.legend()
    save(fig, "fig1_perturbation_chain")
    # fig2: canonical vs order-null
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, key, lab in zip(axs, ("FR", "FC", "delta"), ("FR PPL", "FC PPL", "FC - FR")):
        ys = [r[key] for r in rows[1:6]]
        ax.scatter(np.zeros(5) + np.linspace(-.1, .1, 5), ys, color="#1f77b4", label="FP32 shuffled orders (5)")
        ax.scatter([0.35], [rows[0][key]], marker="s", color="#555555", label="ORIGINAL FP32")
        ax.scatter([0.7], [rows[6][key]], marker="*", s=180, color="#d62728", label="FP64 canonical")
        if key == "delta":
            ax.axhline(0, color="k", lw=.7)
        ax.set_xticks([]); ax.set_title(lab); ax.set_xlim(-.4, 1.0)
    axs[0].legend(fontsize=7)
    save(fig, "fig2_fp64_vs_order_null")
    # fig3: interventions, separate panels
    names = [("A_accumulation_order", "acc. order\n(n=5)"), ("B_AOF", "AOF\n(n=5)"), ("C_AO", "AO\n(n=5)"),
             ("D_natural_grouping_seed0", "natural\ngrouping s0\n(n=5)")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, k, t in zip(axs, ("FR_sd", "FC_sd", "delta_sd"), ("Panel A: FR PPL SD", "Panel B: FC PPL SD", "Panel C: paired FC-FR SD")):
        ax.bar(range(4), [mech[n]["ppl"][k] for n, _ in names], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"])
        ax.set_xticks(range(4), [l for _, l in names], fontsize=7); ax.set_title(t, fontsize=9)
    axs[2].axhline(mech["D3_natural_grouping_3seeds"]["ppl"]["pooled_within_seed_delta_sd"], ls="--", color="k", lw=.8)
    axs[2].text(0, mech["D3_natural_grouping_3seeds"]["ppl"]["pooled_within_seed_delta_sd"] * 1.02, "3-seed pooled within-seed SD", fontsize=7)
    save(fig, "fig3_intervention_comparison")
    # fig4: cross-seed matched differences
    fig, ax = plt.subplots(figsize=(7, 3.8))
    orig = J("runs_partial_ppl/summary.json")["paired"]["FC-FR"]["delta_ppl"]
    for i, s in enumerate(sorted(nested)):
        v = list(nested[s].values())
        ax.scatter(np.full(5, i) + np.linspace(-.15, .15, 5), v, color="#1f77b4", s=18)
        ax.hlines(np.mean(v), i - .25, i + .25, color="#1f77b4", lw=2)
        ax.scatter([i + .35], [orig[i]], marker="s", color="#555555", label="ORIGINAL extraction (separate policy)" if i == 0 else None)
    ax.axhline(np.mean([np.mean(list(nested[s].values())) for s in nested]), color="#d62728", ls="--", label="overall paired mean")
    ax.axhline(0, color="k", lw=.7)
    ax.axhspan(hb["ci95"][0], hb["ci95"][1], color="#d62728", alpha=.08, label="hierarchical 95 % interval")
    ax.set_xticks(range(3), ["seed 0", "seed 1", "seed 2"]); ax.set_ylabel("PPL(FC) - PPL(FR)"); ax.legend(fontsize=7)
    ax.set_title("Execution-dependent ranking: 15 matched (seed, grouping) differences", fontsize=9)
    save(fig, "fig4_execution_dependent_ranking")
    # fig5: window bootstrap vs execution
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[2, 1]))
    ks = sorted(fixed)
    for i, k in enumerate(ks):
        f = fixed[k]
        col = "#d62728" if f["favours"] == "FR" else "#2ca02c" if f["favours"] == "FC" else "#888888"
        axs[0].errorbar(i, f["point"], yerr=[[f["point"] - f["ci95"][0]], [f["ci95"][1] - f["point"]]], fmt="o", color=col, ms=4)
    axs[0].axhline(0, color="k", lw=.7)
    axs[0].set_xticks(range(len(ks)), ks, rotation=70, fontsize=6)
    axs[0].set_ylabel("mean-NLL difference FC-FR")
    axs[0].set_title("Fixed-model window bootstrap (conditional on one weight pair); red favours FR, green FC", fontsize=8)
    allv = [v for s in nested for v in nested[s].values()]
    axs[1].scatter(np.zeros(15) + np.linspace(-.2, .2, 15), allv, s=14)
    axs[1].errorbar([0.4], [hb["point"]], yerr=[[hb["point"] - hb["ci95"][0]], [hb["ci95"][1] - hb["point"]]], fmt="s", color="#d62728")
    axs[1].axhline(0, color="k", lw=.7); axs[1].set_xticks([])
    axs[1].set_title("Execution level: 15 paired PPL diffs +\nhierarchical interval (seeds, then groupings)", fontsize=8)
    save(fig, "fig5_window_vs_execution")
    # fig6: recon vs PPL
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    ks = sorted(rpa)
    axs[0].bar(range(15), [100 * rpa[k]["recon_penalty_rel"] for k in ks], color="#ff7f0e")
    axs[0].set_title("FC - FR expert reconstruction penalty (%), 15 grouping configs", fontsize=8)
    axs[1].bar(range(15), [rpa[k]["delta_ppl"] for k in ks], color=["#d62728" if rpa[k]["delta_ppl"] > 0 else "#2ca02c" for k in ks])
    axs[1].axhline(0, color="k", lw=.7)
    axs[1].set_title("FC - FR PPL, same configs (layer-output metric: not evaluated for these)", fontsize=8)
    for ax in axs:
        ax.set_xticks(range(15), ks, rotation=70, fontsize=6)
    save(fig, "fig6_reconstruction_vs_ppl")


if __name__ == "__main__":
    main()
