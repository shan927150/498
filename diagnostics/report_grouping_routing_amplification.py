"""Analysis, decision, figures and summary for the Family D frozen-routing intervention (CPU only)."""
from __future__ import annotations
import json, math, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import grouping_routing_amplification as G                     # noqa: E402
import report_inference_routing_amplification as RIR           # noqa: E402
import full_depth_pilot as FD                                  # noqa: E402
from moe_calib import cross_seed_grouping as CS               # noqa: E402
from moe_calib import inference_routing_override as IRO       # noqa: E402

p, jdump, jload = G.p, G.jdump, G.jload
SEEDS, GS, STRATS = G.SEEDS, G.GROUPINGS, G.STRATS
FA_RES = os.path.join(ROOT, "runs_inference_routing_amplification/results")
FA_RUNS = os.path.expanduser("~/moe_cache_inference_routing/runs")
FR_C, FC_C, REF_C, INK, INK2, GRID = "#2a78d6", "#eb6834", "#7a7975", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"FR": FR_C, "FC": FC_C}
HIST_ORIG_DELTA = {0: 0.010056899696323285}                        # filled from files below


def res(n, mode):
    return jload(G.res_file(n, mode))


def run(n, mode):
    return torch.load(G.run_file(n, mode))


def fa_res(n, mode):
    return jload(os.path.join(FA_RES, f"{n}_{mode}.json"))


def fa_run(n, mode):
    return torch.load(os.path.join(FA_RUNS, f"{n}_{mode}.pt"))


def sd(x):
    return float(np.std(np.asarray(x, float), ddof=1))


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def spearman(a, b):
    return RIR.spearman(a, b)


def nested(f):
    return {s: {g: f(s, g) for g in GS} for s in SEEDS}


def ci_info(fc, fr, tok):
    wb = FD.paired_bootstrap(fc, fr, tok)
    lo, hi = wb["ci95"]
    return dict(point=wb["point"], ci95=wb["ci95"], width=hi - lo, n_boot=wb["n_boot"], seed=wb["seed"],
                favours="FR" if lo > 0 else "FC" if hi < 0 else None)


def analyze():
    t0 = time.time()
    pr = G.check_protocol()
    inv, wm = jload(p("artifact_inventory.json")), jload(p("weight_manifest.json"))["configs"]
    val, cost_raw = jload(p("routing_override_validation.json")), jload(p("logs", "gpu_cost_raw.json"))
    names = G.CONFIGS
    have = {(n, m): res(n, m) for n in names for m in ("natural", "frozen")}
    missing = [f"{n}:{m}" for (n, m), r in have.items() if not (r and r.get("valid"))]
    ref = torch.load(G.FA_REF)
    ref_idx, ref_m = ref["idx"], RIR.margin(ref)
    tok = have[(names[0], "natural")]["per_window_tokens"]
    orig = {(st, m): fa_res(f"{st}_ORIGINAL", m) for st in STRATS for m in ("natural", "frozen")}
    hist_orig = {0: orig[("FC", "natural")]["ppl"] - orig[("FR", "natural")]["ppl"]}
    for s in (1, 2):
        hist_orig[s] = jload(os.path.join(ROOT, f"runs_partial_ppl/ppl_FC_s{s}.json"))["ppl"] - \
            jload(os.path.join(ROOT, f"runs_partial_ppl/ppl_FR_s{s}.json"))["ppl"]
    fro_orig0 = orig[("FC", "frozen")]["ppl"] - orig[("FR", "frozen")]["ppl"]

    # ---------- PPL results
    ppl = {m: {n: have[(n, m)]["ppl"] for n in names} for m in ("natural", "frozen")}
    jdump(dict(per_config={n: dict(ppl=ppl["natural"][n], historical_ppl=wm[n]["historical_ppl"],
                                   historical_bitwise=have[(n, "natural")]["historical_regression"]["bitwise"])
                           for n in names},
               s0_ORIGINAL={st: orig[(st, "natural")]["ppl"] for st in STRATS}), p("natural_ppl_results.json"))
    jdump(dict(per_config={n: dict(ppl=ppl["frozen"][n], natural_minus_frozen=ppl["natural"][n] - ppl["frozen"][n])
                           for n in names},
               s0_ORIGINAL={st: orig[(st, "frozen")]["ppl"] for st in STRATS}), p("frozen_ppl_results.json"))

    # ---------- paired comparisons
    D = {m: nested(lambda s, g, m=m: ppl[m][G.name("FC", s, g)] - ppl[m][G.name("FR", s, g)]) for m in ppl}
    pw = lambda n, m: have[(n, m)]["per_window_nll"]
    pairs = {}
    for s in SEEDS:
        for g in GS:
            fr, fc = G.name("FR", s, g), G.name("FC", s, g)
            pairs[G.cname(s, g)] = dict(
                allocation_seed=s, grouping=g, FR_natural=ppl["natural"][fr], FC_natural=ppl["natural"][fc],
                FR_frozen=ppl["frozen"][fr], FC_frozen=ppl["frozen"][fc], delta_natural=D["natural"][s][g],
                delta_frozen=D["frozen"][s][g], delta_g=D["natural"][s][g] - D["frozen"][s][g],
                ci_natural=ci_info(pw(fc, "natural"), pw(fr, "natural"), tok),
                ci_frozen=ci_info(pw(fc, "frozen"), pw(fr, "frozen"), tok),
                FR_frac_windows_improved=float(np.mean(np.array(pw(fr, "frozen")) < np.array(pw(fr, "natural")))),
                FC_frac_windows_improved=float(np.mean(np.array(pw(fc, "frozen")) < np.array(pw(fc, "natural")))))
    orig_ci = {m: ci_info(orig[("FC", m)]["per_window_nll"], orig[("FR", m)]["per_window_nll"], tok) for m in ("natural", "frozen")}
    jdump(dict(definition="Delta = PPL(FC) - PPL(FR) (positive = FR better); delta_g = Delta_nat - Delta_fro; "
                          "CI = paired window bootstrap of the mean-NLL difference (full_depth_pilot.paired_bootstrap)",
               pairs=pairs, s0_ORIGINAL=dict(delta_natural=hist_orig[0], delta_frozen=fro_orig0, ci=orig_ci)),
          p("paired_grouping_comparisons.json"))

    # ---------- variance
    V = {m: CS.variance_summary(D[m]) for m in D}
    R = V["frozen"]["pooled_within_sd"] / V["natural"]["pooled_within_sd"]
    flat = {m: sd([D[m][s][g] for s in SEEDS for g in GS]) for m in D}
    pp = lambda st, m: CS.variance_summary(nested(lambda s, g: ppl[m][G.name(st, s, g)]))["pooled_within_sd"]
    dec_hist = {m: CS.decide(D[m]) for m in D}
    hb = {m: CS.hierarchical_bootstrap(D[m]) for m in D}
    var = dict(
        pooled_within_sd_natural=V["natural"]["pooled_within_sd"], pooled_within_sd_frozen=V["frozen"]["pooled_within_sd"],
        R_SD=R, pct_reduction=100 * (1 - R),
        historical_pooled_within_sd=jload(os.path.join(ROOT, "runs_cross_seed_grouping/variance_analysis.json"))["summary"]["pooled_within_sd"],
        per_seed_sd={m: V[m]["sd_grouping"] for m in D},
        per_seed_R={str(s): V["frozen"]["sd_grouping"][s] / V["natural"]["sd_grouping"][s] for s in SEEDS},
        flat_sd_15={**flat, "R_flat": flat["frozen"] / flat["natural"], "note": "descriptive only (nested design)"},
        pooled_ppl_sd={f"{st}_{m}": pp(st, m) for st in STRATS for m in D},
        mean_delta={m: V[m]["overall_mean_delta"] for m in D}, seed_means={m: V[m]["mean_delta"] for m in D},
        change_in_mean_delta=V["frozen"]["overall_mean_delta"] - V["natural"]["overall_mean_delta"],
        range_delta={m: [V[m]["min_delta"], V[m]["max_delta"]] for m in D},
        n_FR_better={m: V[m]["n_fc_worse_total"] for m in D}, n_FC_better={m: V[m]["n_fc_better_total"] for m in D},
        historical_decision_rule={m: dict(category=dec_hist[m][0], details=dec_hist[m][1]) for m in D},
        hierarchical_bootstrap=hb, sd_between_seed_means={m: V[m]["sd_between_seed_means"] for m in D},
        delta_g=dict(values={G.cname(s, g): pairs[G.cname(s, g)]["delta_g"] for s in SEEDS for g in GS},
                     mean=float(np.mean([x["delta_g"] for x in pairs.values()])), sd=sd([x["delta_g"] for x in pairs.values()])),
        seed0_only=dict(sd_natural=V["natural"]["sd_grouping"][0], sd_frozen=V["frozen"]["sd_grouping"][0],
                        R=V["frozen"]["sd_grouping"][0] / V["natural"]["sd_grouping"][0],
                        note="single seed x 5 variants: same structure as Family A"),
        natural_reproduces_historical_pooled=abs(V["natural"]["pooled_within_sd"] - jload(os.path.join(
            ROOT, "runs_cross_seed_grouping/variance_analysis.json"))["summary"]["pooled_within_sd"]) < 1e-12,
        note="descriptive; groupings nested in 3 seeds; no additive decomposition; frozen is off-policy")
    jdump(var, p("variance_comparison.json"))

    # ---------- ranking reversals
    M = {m: V[m]["overall_mean_delta"] for m in D}
    rows = []
    for s in SEEDS:
        for g in GS:
            dn, df = D["natural"][s][g], D["frozen"][s][g]
            ref_fro = fro_orig0 if s == 0 else hist_orig[s]
            rows.append(dict(comparison=G.cname(s, g), delta_natural=dn, delta_frozen=df,
                             sign_natural=sign(dn), sign_frozen=sign(df), sign_flipped_by_freezing=sign(dn) != sign(df),
                             R1_natural=sign(dn) == -sign(M["natural"]), R1_frozen=sign(df) == -sign(M["frozen"]),
                             R1_frozen_vs_natural_M=sign(df) == -sign(M["natural"]),
                             R2_natural=sign(dn) != sign(hist_orig[s]), R2_frozen=sign(df) != sign(ref_fro),
                             R2_frozen_reference="frozen s0 ORIGINAL" if s == 0 else "natural ORIGINAL (mixed-regime; frozen unavailable)",
                             ci_natural_favours=pairs[G.cname(s, g)]["ci_natural"]["favours"],
                             ci_frozen_favours=pairs[G.cname(s, g)]["ci_frozen"]["favours"]))
    cnt = lambda k: int(sum(r[k] for r in rows))
    rr = dict(
        rows=rows, M=M, original_reference_delta=dict(natural=hist_orig, frozen_s0=fro_orig0),
        R1=dict(definition=pr["metrics"]["reversal_R1"], natural=cnt("R1_natural"), frozen=cnt("R1_frozen"),
                frozen_vs_natural_M=cnt("R1_frozen_vs_natural_M"),
                natural_reversals_remaining=[r["comparison"] for r in rows if r["R1_natural"] and r["R1_frozen_vs_natural_M"]]),
        R2=dict(definition=pr["metrics"]["reversal_R2"], natural=cnt("R2_natural"), frozen=cnt("R2_frozen"),
                frozen_seed0_only=int(sum(r["R2_frozen"] for r in rows if r["comparison"].startswith("s0"))),
                natural_seed0_only=int(sum(r["R2_natural"] for r in rows if r["comparison"].startswith("s0"))),
                natural_reversals_remaining=[r["comparison"] for r in rows if r["R2_natural"] and r["R2_frozen"]]),
        n_sign_flipped_by_freezing=cnt("sign_flipped_by_freezing"),
        ci=dict(natural_excluding_zero={k: sum(r["ci_natural_favours"] == k for r in rows) for k in ("FR", "FC")},
                frozen_excluding_zero={k: sum(r["ci_frozen_favours"] == k for r in rows) for k in ("FR", "FC")},
                mean_width_natural=float(np.mean([pairs[r["comparison"]]["ci_natural"]["width"] for r in rows])),
                mean_width_frozen=float(np.mean([pairs[r["comparison"]]["ci_frozen"]["width"] for r in rows])),
                s0_ORIGINAL=orig_ci,
                note="window CIs condition on one fixed weight pair (evaluation-text uncertainty); they do not sample "
                     "grouping (execution) variability"))
    jdump(rr, p("ranking_reversal_analysis.json"))

    # ---------- routing divergence
    div = {}
    for n in names:
        d = run(n, "natural")
        dv, _ = RIR.divergence(d["idx"], ref_idx, ref_m, RIR.margin(d))
        dv["own_exact_tie_rate"] = float((RIR.margin(d) == 0).double().mean())
        dv.pop("margin", None)
        div[n] = dict(frac_set_changed_all=dv["frac_set_changed_all"], frac_tokens_any_layer_changed=dv["frac_tokens_any_layer_changed"],
                      per_layer=[x["frac_set_changed"] for x in dv["per_layer"]], first_layer_with_any_change=dv["first_layer_with_any_change"],
                      layers_0_1_identical=dv["layers_0_1_identical"], own_exact_tie_rate=dv["own_exact_tie_rate"],
                      natural_minus_frozen_ppl=ppl["natural"][n] - ppl["frozen"][n])
    bf16_tie = float((ref_m == 0).double().mean())

    def pair_stats(a, b, mref=None):
        sc, oo = IRO.set_disagreement(a, b)
        nm = IRO.n_membership_changes(a, b)
        out = dict(frac_set_changed=float(sc.float().mean()), per_layer=sc.float().mean(dim=(1, 2)).tolist(),
                   frac_order_only=float(oo.float().mean()),
                   exchanged_per_changed=dict(zip(["1", "2", "3", "4"], [float((nm[sc] == k).float().mean()) for k in (1, 2, 3, 4)])))
        if mref is not None:
            rm, ch = mref.flatten().numpy(), sc.flatten().numpy()
            edges = np.quantile(rm[::5], np.linspace(0, 1, 11))
            dec = np.clip(np.searchsorted(edges, rm, side="right") - 1, 0, 9)
            out.update(frac_changed_by_ref_margin_decile=[float(ch[dec == i].mean()) for i in range(10)],
                       frac_changed_on_exact_ref_tie=float(ch[rm == 0].mean()),
                       share_of_changes_on_exact_ref_tie=float((rm[ch] == 0).mean()),
                       ref_margin_changed_median=float(np.median(rm[ch])), ref_margin_stable_median=float(np.median(rm[~ch][::5])))
        return out
    within = {}
    for st in STRATS:
        for s in SEEDS:
            I = {g: run(G.name(st, s, g), "natural")["idx"] for g in GS}
            for i, g in enumerate(GS):
                for g2 in GS[i + 1:]:
                    within[f"{st}_s{s}_{g}__{g2}"] = pair_stats(I[g], I[g2])
            del I
    vs_orig = {}
    fa_vs_orig, fa_pairs = {}, {}
    for st in STRATS:
        o = fa_run(f"{st}_ORIGINAL", "natural")
        om = RIR.margin(o)
        for g in GS:
            vs_orig[G.name(st, 0, g)] = pair_stats(run(G.name(st, 0, g), "natural")["idx"], o["idx"], om)
        FAI = {g: fa_run(f"{st}_O{g}", "natural")["idx"] for g in (20261121, 20261122, 20261123, 20261124, 20261125)}
        for g in FAI:
            fa_vs_orig[f"{st}_O{g}"] = pair_stats(FAI[g], o["idx"], om)
        ks = sorted(FAI)
        for i, g in enumerate(ks):
            for g2 in ks[i + 1:]:
                fa_pairs[f"{st}_O{g}__O{g2}"] = pair_stats(FAI[g], FAI[g2])
        del FAI, o
    mean_of = lambda d, k="frac_set_changed": float(np.mean([v[k] for v in d.values()]))
    mean_layer = lambda d: np.mean([v["per_layer"] for v in d.values()], axis=0).tolist()
    mean_dec = lambda d: np.mean([v["frac_changed_by_ref_margin_decile"] for v in d.values()], axis=0).tolist()
    # association: per comparison |delta_g| vs mean disagreement vs BF16 of FR and FC
    dg = [abs(pairs[G.cname(s, g)]["delta_g"]) for s in SEEDS for g in GS]
    dis = [np.mean([div[G.name(st, s, g)]["frac_set_changed_all"] for st in STRATS]) for s in SEEDS for g in GS]
    rdiv = dict(
        reference="Family A BF16 reference routing (sha-verified)", bf16_exact_tie_rate=bf16_tie, per_config_vs_bf16=div,
        mean_frac_vs_bf16=float(np.mean([v["frac_set_changed_all"] for v in div.values()])),
        within_seed_grouping_pairs=dict(pairs=within, mean_frac_set_changed=mean_of(within), mean_per_layer=mean_layer(within),
                                        mean_frac_order_only=mean_of(within, "frac_order_only")),
        s0_vs_ORIGINAL=dict(pairs=vs_orig, mean_frac_set_changed=mean_of(vs_orig), mean_per_layer=mean_layer(vs_orig),
                            mean_decile_profile=mean_dec(vs_orig),
                            mean_share_of_changes_on_exact_tie=mean_of(vs_orig, "share_of_changes_on_exact_ref_tie"),
                            mean_frac_changed_on_exact_tie=mean_of(vs_orig, "frac_changed_on_exact_ref_tie")),
        family_A_vs_ORIGINAL=dict(pairs=fa_vs_orig, mean_frac_set_changed=mean_of(fa_vs_orig), mean_per_layer=mean_layer(fa_vs_orig),
                                  mean_decile_profile=mean_dec(fa_vs_orig),
                                  mean_share_of_changes_on_exact_tie=mean_of(fa_vs_orig, "share_of_changes_on_exact_ref_tie"),
                                  mean_frac_changed_on_exact_tie=mean_of(fa_vs_orig, "frac_changed_on_exact_ref_tie")),
        family_A_pairwise_orders=dict(mean_frac_set_changed=mean_of(fa_pairs), mean_per_layer=mean_layer(fa_pairs)),
        association=dict(spearman_abs_delta_g_vs_mean_bf16_disagreement=spearman(dg, dis),
                         spearman_abs_nat_minus_fro_vs_bf16_disagreement=spearman(
                             [abs(div[n]["natural_minus_frozen_ppl"]) for n in names], [div[n]["frac_set_changed_all"] for n in names]),
                         note="descriptive, not causal; 15 comparisons nested in 3 seeds"))
    jdump(rdiv, p("routing_divergence.json"))

    # ---------- weights
    wraw = jload(p("logs", "weight_differences_raw.json"))
    wd = None
    if wraw:
        wp = wraw["within_seed_pairs"]
        assoc_rows = []
        for k, v in wp.items():
            st, s, g = k.split("__")[0].split("_")
            g2 = k.split("__")[1]
            s = int(s[1:])
            a, b = G.name(st, s, g), G.name(st, s, g2)
            assoc_rows.append(dict(pair=k, rel_frobenius=v["rel_frobenius"], frac_changed=v["frac_changed"],
                                   abs_ppl_diff_natural=abs(ppl["natural"][a] - ppl["natural"][b]),
                                   abs_ppl_diff_frozen=abs(ppl["frozen"][a] - ppl["frozen"][b]),
                                   routing_frac_set_changed=within[k]["frac_set_changed"]))
        o_rows = []
        for n, v in wraw["s0_vs_original"].items():
            st = n.split("_")[0]
            o_rows.append(dict(config=n, rel_frobenius=v["rel_frobenius"], frac_changed=v["frac_changed"],
                               abs_ppl_diff_natural=abs(ppl["natural"][n] - orig[(st, "natural")]["ppl"]),
                               abs_ppl_diff_frozen=abs(ppl["frozen"][n] - orig[(st, "frozen")]["ppl"]),
                               routing_frac_set_changed=vs_orig[n]["frac_set_changed"]))
        f = lambda rows_, k: [r[k] for r in rows_]
        wd = dict(
            within_seed_pairs=wp, s0_vs_ORIGINAL=wraw["s0_vs_original"], family_A_vs_ORIGINAL=wraw["family_a_vs_original"],
            summary=dict(
                within_seed_mean_frac_changed=float(np.mean(f(wp.values(), "frac_changed"))),
                within_seed_range_frac_changed=[min(f(wp.values(), "frac_changed")), max(f(wp.values(), "frac_changed"))],
                within_seed_mean_rel_frobenius=float(np.mean(f(wp.values(), "rel_frobenius"))),
                s0_vs_original_frac_changed={k: v["frac_changed"] for k, v in wraw["s0_vs_original"].items()},
                s0_vs_original_mean_rel_frobenius=float(np.mean(f(wraw["s0_vs_original"].values(), "rel_frobenius"))),
                family_A_frac_changed={k: v["frac_changed"] for k, v in wraw["family_a_vs_original"].items()},
                family_A_rel_frobenius={k: v["rel_frobenius"] for k, v in wraw["family_a_vs_original"].items()},
                ulp_share_within_seed_mean={k: float(np.mean([v["ulp_share_of_changed"][k] for v in wp.values()]))
                                            for k in ("1", "2", "3-7", ">=8", "sign_change")}),
            association=dict(
                within_seed_rows=assoc_rows, s0_vs_original_rows=o_rows,
                spearman_relfrob_vs_abs_frozen_ppl_diff=spearman(f(assoc_rows, "rel_frobenius"), f(assoc_rows, "abs_ppl_diff_frozen")),
                spearman_relfrob_vs_abs_natural_ppl_diff=spearman(f(assoc_rows, "rel_frobenius"), f(assoc_rows, "abs_ppl_diff_natural")),
                spearman_fracchanged_vs_abs_frozen_ppl_diff=spearman(f(assoc_rows, "frac_changed"), f(assoc_rows, "abs_ppl_diff_frozen")),
                spearman_routing_vs_abs_natural_ppl_diff=spearman(f(assoc_rows, "routing_frac_set_changed"), f(assoc_rows, "abs_ppl_diff_natural")),
                mean_abs_ppl_diff_natural=float(np.mean(f(assoc_rows, "abs_ppl_diff_natural"))),
                mean_abs_ppl_diff_frozen=float(np.mean(f(assoc_rows, "abs_ppl_diff_frozen"))),
                note="descriptive association over 60 within-seed pairs (not independent); not causal"))
        jdump(wd, p("weight_difference_analysis.json"))

    # ---------- correctness / decision
    endc = cost_raw.get("end_checks", {})
    allv = lambda key: bool(val.get(key)) and all(v.get("passed", all(x for x in v.values() if isinstance(x, bool)))
                                                  for v in val[key].values())
    corr = dict(
        audit_all_valid=inv["all_valid"], no_duplicates=not inv["duplicates"], no_missing_runs=not missing,
        A_natural_parity_all_bitwise=all(v["bitwise"] for v in val["A_natural_parity"].values()) and len(val["A_natural_parity"]) == 30,
        B_record_only=allv("B_record_only"), C_self_forcing=allv("C_self_forcing"),
        D_bf16_self_forcing=bool(val["D_bf16_self_forcing"]["passed"]),
        E_frozen_dispatch=all(all(x for x in v.values() if isinstance(x, bool)) for v in val["E_frozen_dispatch"].values())
                          and len(val["E_frozen_dispatch"]) == 30,
        F_restoration=all(all(v.values()) for v in val["F_restoration"].values()) and len(val["F_restoration"]) == 30,
        end_untouched_params_unchanged=endc.get("untouched_params_unchanged"), end_targets_pristine=endc.get("targets_pristine"),
        end_gates_unchanged=endc.get("gates_unchanged"), end_no_hooks=endc.get("no_hooks"),
        end_bf16_bitwise=endc.get("bf16_end_per_window_bitwise"),
        reference_routing_sha_verified=True, gptq_solves=0,
        token_nll_sums_reproduce_window_nll=all(np.array_equal(run(n, m)["token_nll"].numpy().sum(1), np.array(have[(n, m)]["per_window_nll"]))
                                                for n in names[:4] for m in ("natural", "frozen")))
    parity_ok = all(corr[k] for k in ("A_natural_parity_all_bitwise", "B_record_only", "C_self_forcing", "D_bf16_self_forcing",
                                      "E_frozen_dispatch", "F_restoration"))
    pt = jload(p("logs", "pytest_results.json")) or {}
    corr["targeted_tests_pass"] = pt.get("targeted_passed")
    corr["family_A_artifacts_unchanged"] = pt.get("family_a_unchanged")
    jdump(dict(checks=corr, all_pass=all(v is True or (k == "gptq_solves" and v == 0) for k, v in corr.items()),
               missing=missing), p("correctness_results.json"))
    r1n, r1f = rr["R1"]["natural"], rr["R1"]["frozen"]
    if not parity_ok or missing:
        verdict = "BLOCKED"
    elif R <= 0.5 and r1f <= r1n / 2:
        verdict = "STRONG-COLLAPSE (A)"
    elif R >= 0.9 and r1f > r1n / 2:
        verdict = "WEAK-COLLAPSE (B)"
    else:
        verdict = "PARTIAL-COLLAPSE (C)"
    dec = dict(verdict=verdict, R_SD=R, R1_natural=r1n, R1_frozen=r1f, R2_natural=rr["R2"]["natural"], R2_frozen=rr["R2"]["frozen"],
               parity_ok=parity_ok, rules=pr["decision_rules"], protocol_sha256=pr["protocol_sha256"])
    jdump(dec, p("decision.json"))
    # cost
    runs_ = cost_raw["runs"]
    ev = lambda suf: [v["eval_s"] for k, v in runs_.items() if k.endswith(suf) and "eval_s" in v]
    store = sum(os.path.getsize(os.path.join(G.c("runs"), f)) for f in os.listdir(G.c("runs")))
    jdump(dict(model_load_s=[x["model_load_s"] for x in cost_raw["loads"]], median_natural_eval_s=float(np.median(ev("_natural"))),
               median_frozen_eval_s=float(np.median(ev("_frozen"))),
               install_s_median=float(np.median([v["install_s"] for k, v in runs_.items() if k.endswith("_install")])),
               peak_gpu_gb=max(v.get("peak_gpu_gb", 0) for v in runs_.values()), peak_cpu_gb=cost_raw.get("peak_cpu_gb"),
               storage_bytes=store, gptq_solves=0, n_eval_forwards=sum(1 for v in runs_.values() if "eval_s" in v),
               analysis_s=time.time() - t0), p("cost_results.json"))
    print(json.dumps(dec, indent=1))
    report()
    return dec


# ---------------------------------------------------------------- figures + summary
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    return plt


def save(fig, nm):
    fig.tight_layout()
    fig.savefig(p("figures", f"{nm}.png"), dpi=160)
    fig.savefig(p("figures", f"{nm}.pdf"))


def report():
    plt = _plt()
    pc = jload(p("paired_grouping_comparisons.json"))
    pairs, var, rr = pc["pairs"], jload(p("variance_comparison.json")), jload(p("ranking_reversal_analysis.json"))
    rdiv, wd, dec = jload(p("routing_divergence.json")), jload(p("weight_difference_analysis.json")), jload(p("decision.json"))
    corr, val, cost = jload(p("correctness_results.json")), jload(p("routing_override_validation.json")), jload(p("cost_results.json"))
    inv = jload(p("artifact_inventory.json"))
    keys = [G.cname(s, g) for s in SEEDS for g in GS]
    seed_col = {0: "#2a78d6", 1: "#1baf7a", 2: "#6250d6"}
    # fig1
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k in keys:
        x = pairs[k]
        ax.plot([0, 1], [x["delta_natural"], x["delta_frozen"]], color=seed_col[x["allocation_seed"]], marker="o", ms=5, lw=1.4, alpha=0.85)
        ax.text(1.03, x["delta_frozen"], k, fontsize=6, va="center", color=INK2)
    for s in SEEDS:
        ax.plot([], [], color=seed_col[s], marker="o", label=f"allocation seed {s}")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks([0, 1], ["natural routing", "BF16-frozen routing"]); ax.set_xlim(-0.2, 1.45)
    ax.set_ylabel("PPL(FC) - PPL(FR)   (> 0: FR better)")
    ax.set_title("Family D: paired difference per (seed, grouping)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    save(fig, "fig1_natural_vs_frozen_paired"); plt.close(fig)
    # fig2
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for i, m in enumerate(("natural", "frozen")):
        v = [pairs[k][f"delta_{m}"] for k in keys]
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=[seed_col[pairs[k]["allocation_seed"]] for k in keys], s=22, zorder=3)
        ax.hlines(var["mean_delta"][m], i - 0.25, i + 0.25, color=INK, lw=2)
        ax.text(i + 0.28, var["mean_delta"][m], f"mean {var['mean_delta'][m]:+.4f}\npooled SD "
                f"{var['pooled_within_sd_' + m]:.4f}", fontsize=7, va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks([0, 1], ["natural", "frozen"]); ax.set_xlim(-0.5, 1.9)
    ax.set_ylabel("PPL(FC) - PPL(FR)")
    ax.set_title(f"Grouping variability; R_SD (pooled within-seed) = {var['R_SD']:.3f}", fontsize=9)
    save(fig, "fig2_grouping_variability"); plt.close(fig)
    # fig3 ranking map
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    rows = {r["comparison"]: r for r in rr["rows"]}
    lab = ["natural Δ sign", "natural window CI", "frozen Δ sign", "frozen window CI"]
    for j, k in enumerate(keys):
        r = rows[k]
        vals = [("FR" if r["sign_natural"] > 0 else "FC"), r["ci_natural_favours"], ("FR" if r["sign_frozen"] > 0 else "FC"), r["ci_frozen_favours"]]
        for i, v in enumerate(vals):
            col = FR_C if v == "FR" else FC_C if v == "FC" else "#ffffff"
            ax.add_patch(__import__("matplotlib").patches.Rectangle((j, 3 - i), 0.92, 0.88, color=col, ec=GRID))
            ax.text(j + 0.46, 3 - i + 0.44, v or "–", ha="center", va="center", fontsize=6, color="white" if v else INK2)
    ax.set_xlim(0, len(keys)); ax.set_ylim(0, 4)
    ax.set_yticks([3.44, 2.44, 1.44, 0.44], lab); ax.set_xticks(np.arange(len(keys)) + 0.46, keys, rotation=60, fontsize=6)
    ax.grid(False)
    ax.set_title(f"Which strategy each comparison favours (blue FR, orange FC). Seed ORIGINAL Δ: s0 nat "
                 f"{rr['original_reference_delta']['natural']['0']:+.4f} / frozen {rr['original_reference_delta']['frozen_s0']:+.4f}; "
                 f"s1 {rr['original_reference_delta']['natural']['1']:+.4f}, s2 {rr['original_reference_delta']['natural']['2']:+.4f} (natural)",
                 fontsize=7)
    save(fig, "fig3_ranking_reversal_map"); plt.close(fig)
    # fig4 routing per layer
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax = axs[0]
    L = np.arange(24)
    ax.plot(L, 100 * np.array(rdiv["s0_vs_ORIGINAL"]["mean_per_layer"]), color=FC_C, marker="o", ms=3, label="Family D s0 grouping vs ORIGINAL")
    ax.plot(L, 100 * np.array(rdiv["within_seed_grouping_pairs"]["mean_per_layer"]), color=FC_C, ls="--", label="Family D grouping vs grouping (within seed)")
    ax.plot(L, 100 * np.array(rdiv["family_A_vs_ORIGINAL"]["mean_per_layer"]), color=FR_C, marker="o", ms=3, label="Family A order vs ORIGINAL")
    ax.plot(L, 100 * np.array(rdiv["family_A_pairwise_orders"]["mean_per_layer"]), color=FR_C, ls="--", label="Family A order vs order")
    ax.set_xlabel("MoE layer"); ax.set_ylabel("% tokens with changed top-4 set"); ax.legend(fontsize=6.5)
    ax.set_title("Routing disagreement between quantized variants", fontsize=9)
    ax = axs[1]
    ax.plot(np.arange(1, 11), 100 * np.array(rdiv["s0_vs_ORIGINAL"]["mean_decile_profile"]), color=FC_C, marker="o", label="Family D s0 vs ORIGINAL")
    ax.plot(np.arange(1, 11), 100 * np.array(rdiv["family_A_vs_ORIGINAL"]["mean_decile_profile"]), color=FR_C, marker="o", label="Family A vs ORIGINAL")
    ax.set_xlabel("ORIGINAL p4-p5 margin decile (1 = smallest)"); ax.set_ylabel("% of slots with changed set")
    ax.set_title("Flips concentrate at near-ties", fontsize=9); ax.legend(fontsize=7)
    save(fig, "fig4_routing_divergence_family_A_vs_D"); plt.close(fig)
    # fig5 weights vs residual
    if wd:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        rows_ = wd["association"]["within_seed_rows"]
        fa = wd["summary"]["family_A_rel_frobenius"]
        for ax, m in zip(axs, ("natural", "frozen")):
            for r in rows_:
                ax.scatter(r["rel_frobenius"], r[f"abs_ppl_diff_{m}"], color=COL[r["pair"][:2]], s=14, alpha=0.7)
            for r in wd["association"]["s0_vs_original_rows"]:
                ax.scatter(r["rel_frobenius"], r[f"abs_ppl_diff_{m}"], facecolor="white", edgecolor=COL[r["config"][:2]], s=22)
            ax.set_xlabel("relative Frobenius weight difference")
            ax.set_title(f"{m} routing: |PPL difference| between two variants of one strategy", fontsize=8)
            ax.axvline(np.mean(list(fa.values())), color=REF_C, ls=":", lw=1.2)
            ax.text(np.mean(list(fa.values())), ax.get_ylim()[1] * 0.95, " Family A", fontsize=7, color=REF_C)
        axs[0].set_ylabel("|PPL(variant) - PPL(variant')|")
        for st in STRATS:
            axs[1].scatter([], [], color=COL[st], s=14, label=f"{st} within-seed grouping pair")
            axs[1].scatter([], [], facecolor="white", edgecolor=COL[st], s=22, label=f"{st} s0 grouping vs ORIGINAL")
        axs[1].legend(fontsize=6.5)
        save(fig, "fig5_weight_diff_vs_ppl_diff"); plt.close(fig)
    write_summary(pairs, var, rr, rdiv, wd, dec, corr, val, cost, inv, pc)


def write_summary(pairs, var, rr, rdiv, wd, dec, corr, val, cost, inv, pc):
    fa_var = jload(os.path.join(ROOT, "runs_inference_routing_amplification/variance_comparison.json"))
    L = []
    w = L.append
    keys = [G.cname(s, g) for s in SEEDS for g in GS]
    w("# Family D grouping: frozen-routing causal intervention\n")
    w(f"**Verdict: {dec['verdict']}** - R_SD = {dec['R_SD']:.3f} (pooled within-seed SD of FC-FR, frozen / natural); "
      f"R1 reversals {dec['R1_natural']}/15 -> {dec['R1_frozen']}/15; protocol sha256 {dec['protocol_sha256'][:16]}.\n")
    w("VERDICT_PARAGRAPH\n")
    w("## 1. Artifact and parity audit\n")
    w(f"- {inv['n_valid']}/{inv['n_configs']} weight sets valid; duplicates: {inv['duplicates']}; missing: {inv['missing'] or 'none'}; "
      f"historical Δ table reproduced from stored results for 15/15 pairs: {all(all(v.values()) for v in inv['pairs'].values())}.")
    w("- References: seed-0 ORIGINAL reused from Family A; seed-1/2 ORIGINAL-extraction weights were never saved "
      "(runs_partial_ppl quantized in-process) -> frozen ORIGINAL Δ unavailable for seeds 1/2.")
    w("- Parity (all exact):")
    for k in ("A_natural_parity_all_bitwise", "B_record_only", "C_self_forcing", "D_bf16_self_forcing", "E_frozen_dispatch", "F_restoration"):
        w(f"  - {k}: {corr['checks'][k]}")
    w(f"- End-of-process: non-target params unchanged {corr['checks']['end_untouched_params_unchanged']}, BF16 bitwise "
      f"{corr['checks']['end_bf16_bitwise']}, no hooks {corr['checks']['end_no_hooks']}; GPTQ solves 0.\n")
    w("## 2. Primary paired PPL table\n")
    w("| comparison | FR nat | FC nat | FR frozen | FC frozen | Δ nat | Δ frozen | δ_g | CI nat | CI frozen |\n|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        x = pairs[k]
        w(f"| {k} | {x['FR_natural']:.6f} | {x['FC_natural']:.6f} | {x['FR_frozen']:.6f} | {x['FC_frozen']:.6f} | "
          f"{x['delta_natural']:+.6f} | {x['delta_frozen']:+.6f} | {x['delta_g']:+.6f} | {x['ci_natural']['favours'] or '–'} | {x['ci_frozen']['favours'] or '–'} |")
    w(f"\nseed-0 ORIGINAL (reference, not in SDs): Δ natural {pc['s0_ORIGINAL']['delta_natural']:+.6f}, frozen {pc['s0_ORIGINAL']['delta_frozen']:+.6f}.\n")
    w("## 3. Variability and mean preservation\n")
    w("| statistic | natural | frozen |\n|---|---|---|")
    w(f"| pooled within-seed SD(Δ) | {var['pooled_within_sd_natural']:.6f} (historical {var['historical_pooled_within_sd']:.6f}) | {var['pooled_within_sd_frozen']:.6f} |")
    for s in SEEDS:
        w(f"| seed {s} SD(Δ) | {var['per_seed_sd']['natural'][str(s)]:.6f} | {var['per_seed_sd']['frozen'][str(s)]:.6f} (ratio {var['per_seed_R'][str(s)]:.3f}) |")
    w(f"| flat SD over 15 (descriptive) | {var['flat_sd_15']['natural']:.6f} | {var['flat_sd_15']['frozen']:.6f} |")
    for st in STRATS:
        w(f"| pooled within-seed SD PPL {st} | {var['pooled_ppl_sd'][st + '_natural']:.6f} | {var['pooled_ppl_sd'][st + '_frozen']:.6f} |")
    w(f"| mean Δ (mean of seed means) | {var['mean_delta']['natural']:+.6f} | {var['mean_delta']['frozen']:+.6f} |")
    w(f"| seed means | {', '.join(f'{v:+.5f}' for v in var['seed_means']['natural'].values())} | {', '.join(f'{v:+.5f}' for v in var['seed_means']['frozen'].values())} |")
    w(f"| range Δ | {var['range_delta']['natural'][0]:+.6f} .. {var['range_delta']['natural'][1]:+.6f} | {var['range_delta']['frozen'][0]:+.6f} .. {var['range_delta']['frozen'][1]:+.6f} |")
    w(f"| FR better / FC better | {var['n_FR_better']['natural']}/{var['n_FC_better']['natural']} | {var['n_FR_better']['frozen']}/{var['n_FC_better']['frozen']} |")
    w(f"| hierarchical bootstrap 95% | [{var['hierarchical_bootstrap']['natural']['ci95'][0]:+.5f}, {var['hierarchical_bootstrap']['natural']['ci95'][1]:+.5f}] | [{var['hierarchical_bootstrap']['frozen']['ci95'][0]:+.5f}, {var['hierarchical_bootstrap']['frozen']['ci95'][1]:+.5f}] |")
    w(f"| historical decision rule | {var['historical_decision_rule']['natural']['category']} | {var['historical_decision_rule']['frozen']['category']} |")
    w(f"\n**R_SD = {var['R_SD']:.3f}** ({var['pct_reduction']:.1f} % reduction); seed-0-only ratio {var['seed0_only']['R']:.3f}; "
      f"change in mean Δ {var['change_in_mean_delta']:+.6f}; δ_g mean {var['delta_g']['mean']:+.6f}, SD {var['delta_g']['sd']:.6f}.\n")
    w("## 4. Ranking reversals and window CIs\n")
    w(f"- R1 (historical decision rule: sign opposite to overall mean M): natural {rr['R1']['natural']}/15 (M {rr['M']['natural']:+.6f}), "
      f"frozen {rr['R1']['frozen']}/15 (M {rr['M']['frozen']:+.6f}); natural R1 reversals still opposite natural M after freezing: "
      f"{rr['R1']['natural_reversals_remaining'] or 'none'}.")
    w(f"- R2 (vs same seed's ORIGINAL Δ sign, fp64 mechanism table): natural {rr['R2']['natural']}/15, frozen {rr['R2']['frozen']}/15 "
      f"(seed 0 fully frozen: {rr['R2']['natural_seed0_only']}/5 -> {rr['R2']['frozen_seed0_only']}/5; seeds 1/2 frozen vs natural ORIGINAL sign, mixed-regime). "
      f"Remaining: {rr['R2']['natural_reversals_remaining']}.")
    w(f"- Comparisons whose sign flips under freezing: {rr['n_sign_flipped_by_freezing']}/15.")
    w(f"- Window CIs excluding zero: natural FR {rr['ci']['natural_excluding_zero']['FR']}, FC {rr['ci']['natural_excluding_zero']['FC']}; "
      f"frozen FR {rr['ci']['frozen_excluding_zero']['FR']}, FC {rr['ci']['frozen_excluding_zero']['FC']}; mean width "
      f"{rr['ci']['mean_width_natural']:.6f} -> {rr['ci']['mean_width_frozen']:.6f} (mean-NLL units). Window CIs condition on one weight pair; "
      "they measure text-sampling uncertainty, not grouping variability.\n")
    w("## 5. Routing divergence\n")
    w(f"- vs BF16: mean {100 * rdiv['mean_frac_vs_bf16']:.2f} % of (layer, token) sets differ; layers 0-1 identical in all 30: "
      f"{all(v['layers_0_1_identical'] for v in rdiv['per_config_vs_bf16'].values())}.")
    w(f"- Family D seed-0 grouping vs ORIGINAL: {100 * rdiv['s0_vs_ORIGINAL']['mean_frac_set_changed']:.2f} % (Family A order vs ORIGINAL "
      f"{100 * rdiv['family_A_vs_ORIGINAL']['mean_frac_set_changed']:.2f} %).")
    w(f"- Family D grouping vs grouping within seed: {100 * rdiv['within_seed_grouping_pairs']['mean_frac_set_changed']:.2f} % "
      f"(Family A order vs order {100 * rdiv['family_A_pairwise_orders']['mean_frac_set_changed']:.2f} %); order-only changes "
      f"{100 * rdiv['within_seed_grouping_pairs']['mean_frac_order_only']:.2f} % (excluded).")
    w(f"- BF16 exact-tie rate (reference file) {100 * rdiv['bf16_exact_tie_rate']:.2f} %; share of D s0-vs-ORIGINAL flips on exact ORIGINAL ties "
      f"{100 * rdiv['s0_vs_ORIGINAL']['mean_share_of_changes_on_exact_tie']:.1f} % (Family A {100 * rdiv['family_A_vs_ORIGINAL']['mean_share_of_changes_on_exact_tie']:.1f} %); "
      f"flip rate on exact ties {100 * rdiv['s0_vs_ORIGINAL']['mean_frac_changed_on_exact_tie']:.1f} % vs {100 * rdiv['family_A_vs_ORIGINAL']['mean_frac_changed_on_exact_tie']:.1f} %.")
    w(f"- Margin-decile flip rate (1st..10th), D: {', '.join(f'{100 * x:.1f}' for x in rdiv['s0_vs_ORIGINAL']['mean_decile_profile'])}; "
      f"A: {', '.join(f'{100 * x:.1f}' for x in rdiv['family_A_vs_ORIGINAL']['mean_decile_profile'])} (%).")
    w(f"- Descriptive: Spearman(|δ_g|, BF16 disagreement) = {rdiv['association']['spearman_abs_delta_g_vs_mean_bf16_disagreement']}.\n")
    if wd:
        sm = wd["summary"]
        w("## 6. Weight differences\n")
        w(f"- Seed-0 grouping vs ORIGINAL changed BF16 elements: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in sm["s0_vs_original_frac_changed"].items()))
        w(f"- Within-seed grouping pairs: mean {100 * sm['within_seed_mean_frac_changed']:.1f} % (range {100 * sm['within_seed_range_frac_changed'][0]:.1f}-"
          f"{100 * sm['within_seed_range_frac_changed'][1]:.1f} %), mean rel. Frobenius {sm['within_seed_mean_rel_frobenius']:.4f}.")
        w(f"- Family A (same function): " + ", ".join(f"{k} {100 * v:.4f} % / rel Frob {sm['family_A_rel_frobenius'][k]:.2e}" for k, v in sm["family_A_frac_changed"].items()))
        w(f"- ULP distance of changed elements (within-seed mean): " + ", ".join(f"{k}: {100 * v:.1f} %" for k, v in sm["ulp_share_within_seed_mean"].items()))
        a = wd["association"]
        w(f"- Descriptive over 60 within-seed pairs: Spearman(rel Frob, |ΔPPL| frozen) = {a['spearman_relfrob_vs_abs_frozen_ppl_diff']}, "
          f"natural = {a['spearman_relfrob_vs_abs_natural_ppl_diff']}; mean |ΔPPL| between two variants {a['mean_abs_ppl_diff_natural']:.5f} natural -> "
          f"{a['mean_abs_ppl_diff_frozen']:.5f} frozen.\n")
    w("## 7. Family A vs Family D\n")
    w("| metric | Family A (accumulation order) | Family D (grouping) |\n|---|---|---|")
    if wd:
        w(f"| BF16 weight elements changed vs ORIGINAL | {100 * np.mean(list(wd['summary']['family_A_frac_changed'].values())):.4f} % | "
          f"{100 * np.mean(list(wd['summary']['s0_vs_original_frac_changed'].values())):.1f} % (seed 0) |")
        w(f"| rel. Frobenius vs ORIGINAL | {np.mean(list(wd['summary']['family_A_rel_frobenius'].values())):.2e} | {wd['summary']['s0_vs_original_mean_rel_frobenius']:.4f} |")
    w(f"| natural paired SD | {fa_var['sd_delta_natural']:.6f} (5 orders, 1 seed) | {var['pooled_within_sd_natural']:.6f} pooled within-seed; seed 0 {var['seed0_only']['sd_natural']:.6f} |")
    w(f"| frozen paired SD | {fa_var['sd_delta_frozen']:.6f} | {var['pooled_within_sd_frozen']:.6f}; seed 0 {var['seed0_only']['sd_frozen']:.6f} |")
    w(f"| R_SD | {fa_var['R_SD']:.3f} | {var['R_SD']:.3f} (seed 0 only {var['seed0_only']['R']:.3f}) |")
    w(f"| sign reversals vs ORIGINAL sign (natural) | 0/5 | {rr['R2']['natural']}/15 (seed 0: {rr['R2']['natural_seed0_only']}/5) |")
    w(f"| sign reversals vs ORIGINAL sign (frozen) | 0/5 | {rr['R2']['frozen']}/15 (seed 0: {rr['R2']['frozen_seed0_only']}/5) |")
    w(f"| routing disagreement vs ORIGINAL | {100 * rdiv['family_A_vs_ORIGINAL']['mean_frac_set_changed']:.2f} % | {100 * rdiv['s0_vs_ORIGINAL']['mean_frac_set_changed']:.2f} % (seed 0) |")
    w(f"| routing disagreement variant vs variant | {100 * rdiv['family_A_pairwise_orders']['mean_frac_set_changed']:.2f} % | {100 * rdiv['within_seed_grouping_pairs']['mean_frac_set_changed']:.2f} % |")
    w(f"| mean Δ natural -> frozen | {fa_var['mean_delta_natural']:+.5f} -> {fa_var['mean_delta_frozen']:+.5f} | {var['mean_delta']['natural']:+.5f} -> {var['mean_delta']['frozen']:+.5f} |")
    w("\nANSWERS_AND_LIMITATIONS\n")
    w(f"## Cost\n\nmedian natural eval {cost['median_natural_eval_s']:.1f} s, frozen {cost['median_frozen_eval_s']:.1f} s, install "
      f"{cost['install_s_median']:.0f} s; peak GPU {cost['peak_gpu_gb']:.2f} GB; storage {cost['storage_bytes'] / 1e9:.2f} GB; GPTQ solves 0.\n")
    txt = "\n".join(L) + "\n"
    interp = jload(p("logs", "interpretation.json")) or {}          # hand-written after inspecting the numbers
    txt = txt.replace("VERDICT_PARAGRAPH", interp.get("verdict_paragraph", "_(interpretation pending)_"))
    txt = txt.replace("ANSWERS_AND_LIMITATIONS", interp.get("answers_and_limitations", "_(interpretation pending)_"))
    open(p("summary.md"), "w").write(txt)
    jdump(dict(decision=dec, R_SD=var["R_SD"], pooled_sd=dict(natural=var["pooled_within_sd_natural"], frozen=var["pooled_within_sd_frozen"]),
               mean_delta=var["mean_delta"], R1=rr["R1"], R2={k: v for k, v in rr["R2"].items() if k != "definition"},
               correctness_all_pass=corr["all_pass"]), p("summary.json"))
    print("report written")
