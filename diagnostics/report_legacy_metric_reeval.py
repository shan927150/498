"""Analysis + report for the legacy metric re-evaluation (runs_legacy_metric_reeval/).

Reads results/{split}/L{L}.json (+ e3 matrices), the frozen protocol and the historical result JSONs; writes every
analysis artifact, the figures and the decision. The decision rule is the one frozen in protocol.json."""
from __future__ import annotations

import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import legacy_metric_reeval as LM   # noqa: E402

OUT, LAYERS, SEEDS, TAU = LM.OUT, LM.LAYERS, LM.SEEDS, LM.TAU
p, jdump = LM.p, LM.jdump
SPLIT_ORDER = ("confirm", "oldtest")
B128 = [f"{c}_s{s}" for c in ("FR", "TR", "FC", "TC") for s in SEEDS]
B16 = ["C4R16_s0", "C4R16_s1", "C4R16_s2", "RS_PO16", "RS_EBSS16", "PX_PO16", "PX_EBSS16"]
COLORS = dict(FR="#2a78d6", TR="#eb6834", FC="#1baf7a", TC="#eda100", EBSS="#e87ba4",
              C4R16="#2a78d6", RS_PO16="#eb6834", RS_EBSS16="#1baf7a", PX_PO16="#eda100", PX_EBSS16="#e87ba4")
MARKERS = dict(FR="o", TR="s", FC="^", TC="D", EBSS="*", C4R16="o", RS_PO16="s", RS_EBSS16="^", PX_PO16="D",
               PX_EBSS16="*")


def strat(n):
    return LM.methods_registry()[n]["strategy"]


# ============================================================================ loading
def load_split(split):
    """{method: dict(layer={L: rec}, e3={L: matrix})} for every method evaluated on all 4 layers."""
    recs, meta = {}, {}
    for L in LAYERS:
        path = p("results", split, f"L{L}.json")
        if not os.path.exists(path):
            return {}, {}
        d = json.load(open(path))
        meta[L] = d.get("meta_runs", [])
        for n, r in d["methods"].items():
            recs.setdefault(n, dict(layer={}, e3={}))["layer"][L] = r
            recs[n]["e3"][L] = np.load(p("results", split, f"e3_matrix_{n}_L{L}.npy"))
    return {n: v for n, v in recs.items() if len(v["layer"]) == len(LAYERS)}, meta


def summarize(m):
    lay = {L: m["layer"][L]["moe_layer_output_rel_err"] for L in LAYERS}
    den = {L: m["layer"][L]["den"] for L in LAYERS}
    pe = np.concatenate([np.array([np.nan if v is None else v for v in m["layer"][L]["e3_per_expert_mean"]])
                         for L in LAYERS])
    return dict(
        E_layer=float(np.mean(list(lay.values()))), E_layer_per_layer=lay,
        E_layer_pooled=sum(m["layer"][L]["num"] for L in LAYERS) / sum(den.values()),
        native=float(np.mean([m["layer"][L]["native_rel_err"] for L in LAYERS])),
        native_per_layer={L: m["layer"][L]["native_rel_err"] for L in LAYERS},
        D_per_layer={L: m["layer"][L]["D"] / den[L] for L in LAYERS},
        C_per_layer={L: m["layer"][L]["C"] / den[L] for L in LAYERS},
        D=float(np.mean([m["layer"][L]["D"] / den[L] for L in LAYERS])),
        C=float(np.mean([m["layer"][L]["C"] / den[L] for L in LAYERS])),
        cross_fraction_per_layer={L: m["layer"][L]["C"] / m["layer"][L]["decomp_total"] for L in LAYERS},
        c2_weighted_expert_err_per_layer={L: m["layer"][L]["c2_weighted_expert_err"] for L in LAYERS},
        E_expert=float(np.nanmean(pe)), E_expert_median=float(np.nanmedian(pe)),
        E_expert_p90=float(np.nanquantile(pe, .9)), E_expert_worst=float(np.nanmax(pe)),
        E_expert_worst_id=int(np.nanargmax(pe)),
        E_expert_per_layer={L: float(np.nanmean(pe[i * 60:(i + 1) * 60])) for i, L in enumerate(LAYERS)},
        per_expert=pe)


# ============================================================================ bootstrap (descriptive)
def boot_stats(recs, names, nseq):
    g = np.random.default_rng(LM.BOOT_SEED)
    idx = g.integers(0, nseq, size=(LM.N_BOOT, nseq))
    out = {}
    for n in names:
        lay = np.zeros(LM.N_BOOT)
        for L in LAYERS:
            ps = recs[n]["layer"][L]["per_seq"]
            num, den = np.asarray(ps["num"]), np.asarray(ps["den"])
            lay += num[idx].sum(1) / den[idx].sum(1)
        lay /= len(LAYERS)
        M = np.concatenate([recs[n]["e3"][L] for L in LAYERS])          # [240, nseq]
        e3 = np.empty(LM.N_BOOT)
        for b in range(LM.N_BOOT):
            with np.errstate(all="ignore"):
                e3[b] = np.nanmean(np.nanmean(M[:, idx[b]], axis=1))
        out[n] = dict(layer=lay, expert=e3)
    return out


# ============================================================================ comparisons
def pairs():
    P = []
    for s in SEEDS:
        P.append(("main_FC_vs_FR", f"FC_s{s}", f"FR_s{s}", s))
        P.append(("secondary_TC_vs_TR", f"TC_s{s}", f"TR_s{s}", s))
        P.append(("additional_TR_vs_FR", f"TR_s{s}", f"FR_s{s}", s))
        P.append(("extra_TC_vs_FR", f"TC_s{s}", f"FR_s{s}", s))
        P.append(("ebss_EBSS_vs_FR", "EBSS", f"FR_s{s}", s))
        P.append(("ebss_EBSS_vs_TR", "EBSS", f"TR_s{s}", s))
        P.append(("b16_RS_EBSS_vs_C4R16", "RS_EBSS16", f"C4R16_s{s}", s))
        P.append(("b16_RS_PO_vs_C4R16", "RS_PO16", f"C4R16_s{s}", s))
        P.append(("b16_PX_EBSS_vs_C4R16", "PX_EBSS16", f"C4R16_s{s}", s))
        P.append(("b16_PX_PO_vs_C4R16", "PX_PO16", f"C4R16_s{s}", s))
    P.append(("b16_PX_EBSS_vs_PX_PO", "PX_EBSS16", "PX_PO16", None))
    return P


def rel(a, b):
    return (a - b) / b


def category(de, dl):
    if de < 0 and dl < 0:
        return "improves both"
    if de > 0 and dl > 0:
        return "worsens both"
    if de < 0 < dl:
        return "improves expert recon, worsens MoE output"
    if de > 0 > dl:
        return "worsens expert recon, improves MoE output"
    return "tie"


def compare(S, recs, BS, fam, m, r, s):
    a, b = S[m], S[r]
    rec = dict(family=fam, method=m, ref=r, seed=s,
               d_expert=rel(a["E_expert"], b["E_expert"]), d_layer=rel(a["E_layer"], b["E_layer"]),
               d_layer_pooled=rel(a["E_layer_pooled"], b["E_layer_pooled"]), d_native=rel(a["native"], b["native"]),
               d_layer_per_layer={L: rel(a["E_layer_per_layer"][L], b["E_layer_per_layer"][L]) for L in LAYERS},
               d_expert_per_layer={L: rel(a["E_expert_per_layer"][L], b["E_expert_per_layer"][L]) for L in LAYERS},
               # decomposition differences in units of the reference's E_layer: dD + dC ~= d_layer
               dD_over_ref={L: (a["D_per_layer"][L] - b["D_per_layer"][L]) / b["E_layer_per_layer"][L] for L in LAYERS},
               dC_over_ref={L: (a["C_per_layer"][L] - b["C_per_layer"][L]) / b["E_layer_per_layer"][L] for L in LAYERS},
               dD_agg_over_ref=(a["D"] - b["D"]) / b["E_layer"], dC_agg_over_ref=(a["C"] - b["C"]) / b["E_layer"],
               d_expert_median=rel(a["E_expert_median"], b["E_expert_median"]),
               d_expert_p90=rel(a["E_expert_p90"], b["E_expert_p90"]),
               frac_experts_improved=float(np.nanmean(a["per_expert"] < b["per_expert"])))
    rec["category"] = category(rec["d_expert"], rec["d_layer"])
    rec["ranking_agreement"] = bool(np.sign(rec["d_expert"]) == np.sign(rec["d_layer"]))
    rec["layer_agreement"] = {L: bool(np.sign(rec["d_expert_per_layer"][L]) == np.sign(rec["d_layer_per_layer"][L]))
                              for L in LAYERS}
    for k in ("layer", "expert"):
        d = (BS[m][k] - BS[r][k]) / BS[r][k]
        rec[f"boot95_d_{k}"] = [float(np.quantile(d, .025)), float(np.quantile(d, .975))]
    return rec


def classify(rows, single_set=False):
    """Frozen rule (protocol.json decision_rule). rows: paired comparisons of one family on one split."""
    de = np.array([x["d_expert"] for x in rows]); dl = np.array([x["d_layer"] for x in rows])
    n = len(rows)
    maj = int(np.ceil(2 * n / 3))
    if de.mean() >= TAU and (de > 0).sum() >= maj:
        leg = "penalty"
    elif de.mean() <= -TAU and (de < 0).sum() >= maj:
        leg = "gain"
    else:
        leg = "tie"
    if leg == "penalty":
        if (dl <= -TAU).all():
            v = "METRIC-RANKING-REVERSAL"
        elif (dl > 0).sum() >= maj and not (dl <= -TAU).any():
            v = ("LEGACY-NEGATIVE-CONFIRMED" if dl.mean() >= max(TAU, 0.5 * de.mean())
                 else "METRIC-ALIGNED-BUT-SMALL")
        else:
            v = "MIXED-OR-LAYER-SPECIFIC"
    else:
        v = ("NEW-METRIC-GAIN" if (dl <= -TAU).all() else
             "NEW-METRIC-PENALTY" if (dl >= TAU).all() else "NO-ROBUST-DIFFERENCE")
    mean_sign = np.sign(dl.mean())
    lay_rev = [L for L in LAYERS if all(np.sign(x["d_layer_per_layer"][L]) == -mean_sign and
                                        abs(x["d_layer_per_layer"][L]) >= TAU for x in rows)]
    return dict(verdict=v, legacy_direction=leg, n_pairs=n, mean_d_expert=float(de.mean()),
                mean_d_layer=float(dl.mean()), d_expert=de.tolist(), d_layer=dl.tolist(),
                magnitude_ratio_layer_over_expert=float(dl.mean() / de.mean()) if de.mean() != 0 else None,
                layer_specific_reversals=lay_rev,
                note="one calibration set compared with several references; NOT replications" if single_set else "")


# ============================================================================ correlations
def spearman(x, y):
    from scipy.stats import kendalltau, spearmanr
    return dict(spearman=float(spearmanr(x, y).statistic), kendall=float(kendalltau(x, y).statistic), n=len(x))


# ============================================================================ main analysis
def analyze():
    prot = json.load(open(p("protocol.json")))
    reg = LM.methods_registry()
    out = {}
    for split in SPLIT_ORDER:
        recs, meta = load_split(split)
        if not recs:
            continue
        S = {n: summarize(m) for n, m in recs.items()}
        nseq = len(next(iter(recs.values()))["layer"][LAYERS[0]]["per_seq"]["num"])
        BS = boot_stats(recs, list(recs), nseq)
        comps = [compare(S, recs, BS, f, m, r, s) for f, m, r, s in pairs() if m in S and r in S]
        fams = {}
        for c in comps:
            fams.setdefault(c["family"], []).append(c)
        cls = {f: classify(rows, single_set=f.startswith(("ebss", "b16"))) for f, rows in fams.items()}
        b128 = [n for n in B128 + ["EBSS"] if n in S]
        corr = dict(
            b128_with_ebss=spearman([S[n]["E_expert"] for n in b128], [S[n]["E_layer"] for n in b128]),
            b128_without_ebss=spearman([S[n]["E_expert"] for n in b128 if n != "EBSS"],
                                       [S[n]["E_layer"] for n in b128 if n != "EBSS"]),
            per_layer_b128_without_ebss={L: spearman([S[n]["E_expert_per_layer"][L] for n in b128 if n != "EBSS"],
                                                     [S[n]["E_layer_per_layer"][L] for n in b128 if n != "EBSS"])
                                         for L in LAYERS},
            b16=spearman([S[n]["E_expert"] for n in B16 if n in S], [S[n]["E_layer"] for n in B16 if n in S])
            if sum(n in S for n in B16) >= 3 else None)
        # per-expert vs c^2-weighted: correlation of the per-layer metrics across all B128 configs
        out[split] = dict(S=S, comps=comps, cls=cls, corr=corr, meta=meta, nseq=nseq, BS=BS)
    write_artifacts(out, prot, reg)
    return out


def strip(S):
    return {n: {k: v for k, v in s.items() if k != "per_expert"} for n, s in S.items()}


def write_artifacts(out, prot, reg):
    hist = {n: json.load(open(os.path.join(ROOT, reg[n]["result"]))) for n in reg
            if os.path.exists(os.path.join(ROOT, reg[n]["result"]))}
    # ---------------- evaluation manifest
    em = dict(primary_split=prot["evaluation"]["primary_split"], splits={})
    for split, o in out.items():
        m0 = o["meta"][LAYERS[0]][0]
        em["splits"][split] = dict(LM.SPLITS[split], n_sequences=o["nseq"], tokens_per_layer=o["nseq"] * 512,
                                   methods=sorted(o["S"]),
                                   universe_size={L: len(o["meta"][L][0]["universe"]) for L in LAYERS},
                                   evaluation_runs={L: [dict(methods=r["methods"], seconds=r["seconds"],
                                                             peak_gpu_gb=r["peak_gpu_gb"]) for r in o["meta"][L]]
                                                    for L in LAYERS})
    jdump(em, p("evaluation_manifest.json"))

    # ---------------- baseline regression (merge with the quantize-stage record)
    br = json.load(open(p("baseline_regression.json"))) if os.path.exists(p("baseline_regression.json")) else {}
    if "oldtest" in out:
        S = out["oldtest"]["S"]
        agq = json.load(open(os.path.join(ROOT, "runs_agq_only_validation", "results", "eval_FR_s0.json")))
        ref_lay = {L: agq["layers"][str(L)]["GPTQ"]["moe_layer_output_rel_err"] for L in LAYERS}
        ref_nat = {L: agq["layers"][str(L)]["GPTQ"]["moe_layer_output_native_rel_err"] for L in LAYERS}
        if "FR_s0" in S:
            lay = S["FR_s0"]["E_layer_per_layer"]
            br["agq_q0_layer_output"] = dict(
                reference="runs_agq_only_validation/results/eval_FR_s0.json (GPTQ arm = same FR_s0 weights)",
                per_layer=lay, per_layer_reference=ref_lay,
                per_layer_exact={L: lay[L] == ref_lay[L] for L in LAYERS},
                max_rel_diff=max(abs(lay[L] - ref_lay[L]) / ref_lay[L] for L in LAYERS),
                mean=S["FR_s0"]["E_layer"], mean_reference=prot["regression"]["agq_q0_layer_output_FR_s0_oldtest"],
                native_per_layer=S["FR_s0"]["native_per_layer"], native_reference=ref_nat,
                native_max_rel_diff=max(abs(S["FR_s0"]["native_per_layer"][L] - ref_nat[L]) / ref_nat[L]
                                        for L in LAYERS))
            br["agq_q0_layer_output"]["passed"] = br["agq_q0_layer_output"]["max_rel_diff"] <= 1e-9
        e3 = {}
        for n, s in S.items():
            h = hist[n]
            hp = np.array([x["mean"] for x in h["per_expert"]])
            e3[n] = dict(recomputed_installed_bf16=s["E_expert"], historical_solver_fp32=h["recon"],
                         rel_diff=rel(s["E_expert"], h["recon"]),
                         per_expert_max_abs_rel_diff=float(np.nanmax(np.abs(s["per_expert"] - hp) / hp)),
                         per_expert_spearman=float(__import__("scipy.stats").stats.spearmanr(s["per_expert"], hp).statistic))
        tol = 0.005
        hist_pairs = []
        for s_ in SEEDS:
            for m, r in ((f"FC_s{s_}", f"FR_s{s_}"), (f"TR_s{s_}", f"FR_s{s_}"), ("EBSS", f"FR_s{s_}")):
                if m in S and r in S:
                    hist_pairs.append(dict(method=m, ref=r, hist=rel(hist[m]["recon"], hist[r]["recon"]),
                                           recomputed=rel(S[m]["E_expert"], S[r]["E_expert"])))
        br["expert_reconstruction"] = dict(
            note="historical E3 used the solver's fp32 W_q; the installed (and evaluated) weights are their BF16 "
                 "rounding, so equality is not expected; tolerance fixed at 0.5 % relative per configuration and "
                 "every historical paired sign preserved",
            tolerance_rel=tol, configs=e3,
            max_abs_rel_diff=max(abs(v["rel_diff"]) for v in e3.values()),
            paired_differences=hist_pairs,
            paired_signs_preserved=all(np.sign(x["hist"]) == np.sign(x["recomputed"]) for x in hist_pairs))
        br["expert_reconstruction"]["passed"] = bool(br["expert_reconstruction"]["max_abs_rel_diff"] <= tol and
                                                     br["expert_reconstruction"]["paired_signs_preserved"])
    cap = json.load(open(p("logs", "tc_capture_manifest.json"))) if os.path.exists(p("logs", "tc_capture_manifest.json")) else None
    if cap:
        br["tc_capture"] = {k: cap[k] for k in ("n_rows", "n_shared_with_historical_b1", "shared_rows_bitwise_equal",
                                                "shared_rows_routing_equal", "passed")}
    br["passed"] = all(br.get(k, {}).get("passed", False) for k in
                       ("fr_s0_newcache", "agq_q0_layer_output", "expert_reconstruction", "tc_capture"))
    jdump(br, p("baseline_regression.json"))

    # ---------------- correctness
    corr = dict(gates={})
    for split, o in out.items():
        g = {}
        for L in LAYERS:
            runs = o["meta"][L]
            g[L] = dict(
                accumulator_check=[c for r in runs for c in r["accumulator_check"].values()],
                e3_fast_equals_production=all(r["e3_fast_vs_production"]["bitwise_equal"] for r in runs),
                decomposition_identity_max_rel=max(r["decomposition_identity_max_rel"] for r in runs),
                decomposition_vs_fp32_err_max_rel=max(r["decomposition_vs_fp32_err_max_rel"] for r in runs),
                pristine_restored=all(r["pristine_restored"] for r in runs),
                same_universe=len({tuple(r["universe"]) for r in runs}) == 1,
                weights_all_differ_from_bf16=all(o_["layer"][L]["projections_differing_from_bf16"] == 180
                                                 for o_ in load_split(split)[0].values()))
        corr["gates"][split] = g
    inv = json.load(open(p("artifact_inventory.json")))
    corr["weight_files_match_records"] = {n: v.get("weight_files_match_record") for n, v in inv["methods"].items()}
    corr["all_methods_standard_q0_gptq"] = {n: dict(gptq=hist[n]["gptq"], rtn_fallbacks=hist[n]["rtn_fallbacks"],
                                                     n_quantized=hist[n]["n_quantized_experts"],
                                                     policy=hist[n].get("policy"),
                                                     hessian_weighting=hist[n].get("hessian_weighting", "none"),
                                                     routers_unchanged=hist[n]["routers_unchanged"],
                                                     untouched_params_unchanged=hist[n]["untouched_params_unchanged"],
                                                     valid=hist[n]["valid"])
                                          for n in hist}
    corr["router_frozen"] = "asserted per sequence: native block router logits under installed weights == BF16 logits"
    corr["shared_expert"] = "never modified (only routed gate/up/down are installed; untouched digests checked by quant_config)"
    ok = all(all(v["e3_fast_equals_production"] and v["pristine_restored"] and v["same_universe"] and
                 v["weights_all_differ_from_bf16"] and v["decomposition_identity_max_rel"] < 1e-9 and
                 all(c["abs_diff"] <= 1e-12 * c["accumulator"] for c in v["accumulator_check"]) for v in g.values())
             for g in corr["gates"].values())
    ok &= all(v["valid"] and v["rtn_fallbacks"] == 0 and v["n_quantized"] == 240 and v["gptq"]["bits"] == 4
              and v["gptq"]["group_size"] == 128 and v["gptq"]["percdamp"] == 1.0 and v["gptq"]["act_order"]
              and v["hessian_weighting"] == "none" and v["policy"] == "CANONICAL-B1"
              and v["routers_unchanged"] and v["untouched_params_unchanged"]
              for v in corr["all_methods_standard_q0_gptq"].values())
    ok &= all(v is not False for v in corr["weight_files_match_records"].values())
    corr["passed"] = bool(ok and br["passed"])
    jdump(corr, p("correctness_results.json"))

    # ---------------- metric tables
    jdump({split: {n: dict(E_expert=s["E_expert"], median=s["E_expert_median"], p90=s["E_expert_p90"],
                           worst=s["E_expert_worst"], worst_expert_global_id=s["E_expert_worst_id"],
                           per_layer=s["E_expert_per_layer"],
                           historical_oldtest_solver_fp32=hist[n]["recon"])
                   for n, s in o["S"].items()} | dict(
        frac_experts_improved_vs_FR_same_seed={c["method"]: c["frac_experts_improved"] for c in o["comps"]
                                               if c["ref"].startswith("FR") and c["method"] != "EBSS"})
           for split, o in out.items()}, p("expert_reconstruction.json"))
    jdump({split: dict(definition=prot["evaluation"]["primary_metric"], aggregate=prot["evaluation"]["aggregate"],
                       methods={n: dict(E_layer_mean_of_layers=s["E_layer"], per_layer=s["E_layer_per_layer"],
                                        E_layer_pooled_reference_only=s["E_layer_pooled"],
                                        native_bf16_block_mean=s["native"], native_per_layer=s["native_per_layer"])
                                for n, s in o["S"].items()})
           for split, o in out.items()}, p("moe_layer_output.json"))
    dec = {}
    for split, o in out.items():
        dec[split] = dict(methods={n: dict(D_over_den=s["D_per_layer"], C_over_den=s["C_per_layer"],
                                           cross_fraction_signed=s["cross_fraction_per_layer"],
                                           D_agg=s["D"], C_agg=s["C"], E_layer=s["E_layer"],
                                           c2_weighted_expert_err=s["c2_weighted_expert_err_per_layer"])
                                   for n, s in o["S"].items()},
                          differences={f'{c["method"]}-{c["ref"]}': dict(dD_over_ref=c["dD_over_ref"],
                                                                         dC_over_ref=c["dC_over_ref"],
                                                                         dD_agg=c["dD_agg_over_ref"],
                                                                         dC_agg=c["dC_agg_over_ref"],
                                                                         d_layer=c["d_layer"])
                                       for c in o["comps"]})
    jdump(dec, p("error_decomposition.json"))
    jdump({split: dict(comparisons=o["comps"], classifications=o["cls"]) for split, o in out.items()},
          p("paired_comparisons.json"))
    ra = {}
    for split, o in out.items():
        cats = {}
        for c in o["comps"]:
            cats.setdefault(c["category"], []).append(f'{c["method"]} vs {c["ref"]}')
        ra[split] = dict(table=[dict(method=c["method"], ref=c["ref"], seed=c["seed"], d_expert=c["d_expert"],
                                     d_layer=c["d_layer"], agreement=c["ranking_agreement"],
                                     category=c["category"], layer_agreement=c["layer_agreement"])
                                for c in o["comps"]],
                         categories=cats, rank_correlation=o["corr"],
                         method_level=method_ranks(o["S"], hist))
    jdump(ra, p("ranking_agreement.json"))

    # ---------------- cost
    cost = dict(evaluation={split: {L: [dict(methods=len(r["methods"]), seconds=r["seconds"],
                                             peak_gpu_gb=r["peak_gpu_gb"]) for r in o["meta"][L]] for L in LAYERS}
                            for split, o in out.items()})
    if cap:
        cost["tc_capture"] = dict(rows=cap["n_rows"], seconds=cap["seconds_total"], peak_gpu_gb=cap["peak_gpu_gb"])
    for n in ("FR_s0_newcache", "TC_s0", "TC_s1", "TC_s2"):
        if os.path.exists(p("results", f"{n}.json")):
            r = json.load(open(p("results", f"{n}.json")))
            cost[n] = dict(quantize_total_s=r["quantize_total_s"], hessian_s=r["hessian_s"], gptq_s=r["gptq_s"],
                           eval_ppl_s=r["eval_s"], peak_gpu_gb=r["peak_gpu_gb"], peak_cpu_gb=r["peak_cpu_gb"])
    jdump(cost, p("cost_results.json"))

    # ---------------- decision
    prim = prot["evaluation"]["primary_split"]
    dec_out = dict(primary_split=prim, gates_passed=corr["passed"])
    for split, o in out.items():
        dec_out[split] = o["cls"]
    c = out.get(prim, {}).get("cls", {})
    if not corr["passed"] or "main_FC_vs_FR" not in c:
        rec, main = "EVALUATION-BLOCKED", "EVALUATION-BLOCKED"
    else:
        main = c["main_FC_vs_FR"]["verdict"]
        tc = c.get("secondary_TC_vs_TR", {}).get("verdict")
        eb = c.get("ebss_EBSS_vs_FR", {}).get("verdict")
        if main == "METRIC-RANKING-REVERSAL" or tc in ("NEW-METRIC-GAIN", "METRIC-RANKING-REVERSAL"):
            rec = "A"
        elif main in ("LEGACY-NEGATIVE-CONFIRMED", "METRIC-ALIGNED-BUT-SMALL") and \
                tc not in ("NEW-METRIC-GAIN", "METRIC-RANKING-REVERSAL") and \
                eb not in ("NEW-METRIC-GAIN", "METRIC-RANKING-REVERSAL"):
            rec = "B"
        else:
            rec = "C"
    dec_out.update(primary_verdict=main, recommendation=rec,
                   split_agreement={f: out[SPLIT_ORDER[0]]["cls"][f]["verdict"] == out[SPLIT_ORDER[1]]["cls"][f]["verdict"]
                                    for f in out.get(SPLIT_ORDER[0], {}).get("cls", {})
                                    if len(out) == 2 and f in out[SPLIT_ORDER[1]]["cls"]})
    jdump(dec_out, p("decision.json"))
    summ = dict(decision=dec_out, metrics={split: strip(o["S"]) for split, o in out.items()},
                correlations={split: o["corr"] for split, o in out.items()})
    jdump(summ, p("summary.json"))


def method_ranks(S, hist):
    """Strategy-level means over seeds; ranks under expert recon, layer output, and compatible historical PPL."""
    groups = {}
    for n in S:
        if n in B128 or n == "EBSS":
            groups.setdefault(strat(n), []).append(n)
    rows = {}
    for g, ns in groups.items():
        rows[g] = dict(E_expert=float(np.mean([S[n]["E_expert"] for n in ns])),
                       E_layer=float(np.mean([S[n]["E_layer"] for n in ns])),
                       ppl_canonical_b1=float(np.mean([hist[n]["ppl"] for n in ns])) if all(
                           hist.get(n, {}).get("ppl") for n in ns) else None, n=len(ns))
    for k in ("E_expert", "E_layer", "ppl_canonical_b1"):
        vals = [(rows[g][k], g) for g in rows if rows[g][k] is not None]
        for i, (_, g) in enumerate(sorted(vals)):
            rows[g][f"rank_{k}"] = i + 1
    return rows


# ============================================================================ figures
def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
    os.makedirs(p("figures"), exist_ok=True)
    splits = [s for s in SPLIT_ORDER if s in out]

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(p("figures", f"{name}.{ext}"), dpi=170)
        plt.close(fig)

    # Fig 1: expert recon vs MoE output; per split: all B128 (+EBSS), zoom on C4 strategies, B16 (exploratory)
    fig, axes = plt.subplots(len(splits), 3, figsize=(13.5, 4.0 * len(splits)), squeeze=False)
    panels = ((B128 + ["EBSS"], "B = 128 incl. official EBSS"), (B128, "B = 128, C4 strategies (zoom)"),
              (B16, "B = 16 (exploratory, separate)"))
    for i, split in enumerate(splits):
        S = out[split]["S"]
        for j, (names, title) in enumerate(panels):
            ax = axes[i, j]
            seen = set()
            for n in names:
                if n not in S:
                    continue
                st = strat(n)
                big = MARKERS[st] == "*"
                ax.scatter(S[n]["E_expert"], S[n]["E_layer"], s=150 if big else 46, c=COLORS[st],
                           marker=MARKERS[st], edgecolors="#fcfcfb", linewidths=0.8 if big else 1.5, zorder=3,
                           label=LM.STRAT_NAMES[st] if st not in seen else None)
                seen.add(st)
            ax.set_xlabel("mean expert reconstruction error (E3, installed weights)")
            ax.set_ylabel("MoE-layer output error (mean of 4 layers)")
            ax.set_title(f"{split} split — {title}", fontsize=9, loc="left")
            ax.ticklabel_format(style="sci", scilimits=(-3, -3), axis="both")
            ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.text(0.01, 0.002, "one point per calibration set (3 seeds per C4 strategy; EBSS and B=16 synthetic sets are single "
                          "sets). Points to the lower left are better under both metrics.", fontsize=7, color="#52514e")
    save(fig, "fig1_expert_vs_moe_output")

    # Fig 2: paired differences, per seed, both metrics
    fams = [("main_FC_vs_FR", "FC − FR"), ("secondary_TC_vs_TR", "TC − TR"), ("additional_TR_vs_FR", "TR − FR")]
    fig, axes = plt.subplots(len(splits), 3, figsize=(11, 3.4 * len(splits)), squeeze=False, sharey="row")
    for i, split in enumerate(splits):
        comps = out[split]["comps"]
        for j, (f, title) in enumerate(fams):
            ax = axes[i, j]
            rows = sorted([c for c in comps if c["family"] == f], key=lambda c: c["seed"])
            if not rows:
                ax.set_title(f"{title} ({split}) - not evaluated", fontsize=9, loc="left"); continue
            x = np.arange(len(rows))
            for k, (key, lab, col) in enumerate((("d_expert", "expert recon", "#2a78d6"),
                                                 ("d_layer", "MoE output", "#eb6834"))):
                vals = np.array([c[key] for c in rows]) * 100
                ci = np.array([c[f"boot95_{key}"] for c in rows]) * 100
                ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=col, label=lab, edgecolor="#fcfcfb", linewidth=2)
                ax.errorbar(x + (k - 0.5) * 0.36, vals, yerr=[vals - ci[:, 0], ci[:, 1] - vals], fmt="none",
                            ecolor="#52514e", elinewidth=0.8, capsize=2)
            ax.axhline(0, color="#52514e", lw=0.8)
            ax.set_xticks(x, [f"s{c['seed']}" for c in rows])
            ax.set_title(f"{title} ({split})", fontsize=9, loc="left")
            if j == 0:
                ax.set_ylabel("relative difference (%)  — negative = method better")
            ax.legend(fontsize=7, frameon=False)
    fig.text(0.01, 0.002, "error bars: descriptive paired window bootstrap 95 % (windows, not calibration draws)",
             fontsize=7, color="#52514e")
    save(fig, "fig2_paired_differences")

    # Fig 3: layer-level output errors (primary split), absolute, strategy x seed
    split = splits[0]
    S = out[split]["S"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    strats = ["FR", "TR", "FC", "TC", "EBSS"]
    for li, L in enumerate(LAYERS):
        ax = axes[li]
        for si, st in enumerate(strats):
            ns = [n for n in B128 + ["EBSS"] if n in S and strat(n) == st]
            v = [S[n]["E_layer_per_layer"][L] for n in ns]
            if not v:
                continue
            ax.bar(si, np.mean(v), 0.6, color=COLORS[st], edgecolor="#fcfcfb", linewidth=2, alpha=0.9)
            ax.scatter([si] * len(v), v, s=14, c="#0b0b0b", marker=MARKERS[st], zorder=3)
        ax.set_xticks(range(len(strats)), strats)
        ax.set_title(f"L{L}", fontsize=9, loc="left")
        lo = min(S[n]["E_layer_per_layer"][L] for n in S if n in B128 + ["EBSS"])
        hi = max(S[n]["E_layer_per_layer"][L] for n in S if n in B128 + ["EBSS"])
        ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.15 * (hi - lo))
        if li == 0:
            ax.set_ylabel(f"MoE-layer output error ({split})")
    fig.text(0.01, 0.002, "bars: mean over seeds; dots: individual seeds (EBSS: one set). Axis not zero-based "
                          "(differences are small relative to the level).", fontsize=7, color="#52514e")
    save(fig, "fig3_layer_output_errors")

    # Fig 4: diagonal vs cross changes per layer (primary split), mean over seeds
    fams4 = [("main_FC_vs_FR", "FC − FR"), ("secondary_TC_vs_TR", "TC − TR"), ("additional_TR_vs_FR", "TR − FR"),
             ("ebss_EBSS_vs_FR", "EBSS − FR")]
    fig, axes = plt.subplots(1, len(fams4), figsize=(12, 3.3), sharey=False)
    comps = out[split]["comps"]
    for j, (f, title) in enumerate(fams4):
        ax = axes[j]
        rows = [c for c in comps if c["family"] == f]
        if not rows:
            ax.set_title(f"{title} - not evaluated", fontsize=9, loc="left"); continue
        x = np.arange(len(LAYERS))
        dD = np.array([[c["dD_over_ref"][L] for L in LAYERS] for c in rows]) * 100
        dC = np.array([[c["dC_over_ref"][L] for L in LAYERS] for c in rows]) * 100
        ax.bar(x - 0.19, dD.mean(0), 0.36, color="#2a78d6", label="Δ diagonal (c²-weighted)", edgecolor="#fcfcfb",
               linewidth=2)
        ax.bar(x + 0.19, dC.mean(0), 0.36, color="#eb6834", label="Δ cross-expert", edgecolor="#fcfcfb", linewidth=2)
        for r in range(dD.shape[0]):
            ax.scatter(x - 0.19, dD[r], s=8, c="#0b0b0b", zorder=3)
            ax.scatter(x + 0.19, dC[r], s=8, c="#0b0b0b", zorder=3)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xticks(x, [f"L{L}" for L in LAYERS])
        ax.set_title(title, fontsize=9, loc="left")
        if j == 0:
            ax.set_ylabel(f"% of reference E_layer ({split})")
            ax.legend(fontsize=7, frameon=False)
    save(fig, "fig4_diagonal_vs_cross")

    # Fig 5: ranking stability (strategy level), slope chart
    keys = []
    for sp in splits:
        keys += [(sp, "E_expert", f"expert recon\n({sp})"), (sp, "E_layer", f"MoE output\n({sp})")]
    keys.append((splits[0], "ppl_canonical_b1", "partial PPL\n(historical, same weights)"))
    ranks = {sp: json.load(open(p("ranking_agreement.json")))[sp]["method_level"] for sp in splits}
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for st in ["FR", "TR", "FC", "TC", "EBSS"]:
        ys = []
        for sp, k, _ in keys:
            ys.append(ranks[sp].get(st, {}).get(f"rank_{k}", np.nan))
        ax.plot(range(len(keys)), ys, color=COLORS[st], lw=2, marker=MARKERS[st], ms=8,
                markeredgecolor="#fcfcfb", label=LM.STRAT_NAMES[st])
        ax.annotate(st, (len(keys) - 1, ys[-1]), xytext=(8, -3), textcoords="offset points", fontsize=8)
    ax.set_xticks(range(len(keys)), [k[2] for k in keys], fontsize=7)
    ax.set_yticks(range(1, 6)); ax.invert_yaxis(); ax.set_ylabel("rank (1 = best)")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.04, 1))
    ax.set_title("Strategy-level ranks (means over seeds; EBSS one set). Ranks only — not significance.",
                 fontsize=8, loc="left")
    save(fig, "fig5_ranking_stability")


def main(stage="analyze"):
    out = analyze()
    figures(out)
