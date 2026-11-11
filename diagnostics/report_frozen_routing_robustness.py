"""Report for the frozen-routing robustness study: OO / AA_g / AO_g PPL across the five prespecified groupings,
grouping spread, distance to OO, FR-vs-FC ranking, paired intervention effects, Hessian / act-order / weight /
reconstruction diagnostics, the frozen decision, figures 1-6, summary.md."""
from __future__ import annotations
import json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import frozen_routing_robustness as XR      # noqa: E402
import full_depth_pilot as FD               # noqa: E402
from moe_calib import frozen_routing as FRT              # noqa: E402
from moe_calib import frozen_routing_robustness as RB    # noqa: E402

P, STRATS, LAYERS, G = XR.p, XR.STRATS, XR.LAYERS, list(RB.GROUPINGS)
KCOL = {"OO": "#52514e", "AO": "#2a78d6", "AA": "#1baf7a"}
SCOL = {"FR": "#2a78d6", "FC": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
GL = [g[-2:] for g in G]                                # short labels "11".."15"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def ppl_view(r):
    return dict(ppl=r["ppl"], mean_nll=r["mean_nll"], total_nll=r["total_nll"], total_tokens=r["total_tokens"],
                per_window_nll=r["per_window_nll"], per_window_tokens=r["per_window_tokens"])


def build():
    pr = XR.protocol()
    rp, reg = load(P("routing_preparation.json")), load(P("regression.json"))
    oo = {c: load(XR.oo_result_path(c)) for c in STRATS}
    aa = {c: {g: load(XR.aa_result_path(c, g)) for g in G} for c in STRATS}
    ao = {c: {g: load(XR.ao_result_path(c, g)) for g in G} for c in STRATS}
    missing = [f"{c}_AO_{g}" for c in STRATS for g in G if ao[c][g] is None]
    # an AO configuration that ran to completion but failed a correctness check blocks the frozen decision; its
    # numbers are still reported descriptively (never as the decision) together with the exact failed check
    ao_failed = [f"{c}_AO_{g}" for c in STRATS for g in G if ao[c][g] is not None and not ao[c][g]["valid"]]
    invalid = [f"{c}_AA_{g}" for c in STRATS for g in G if not (aa[c][g] and aa[c][g]["valid"])]
    blockers = dict(missing=missing, invalid=invalid, routing_preparation_valid=bool(rp and rp["valid"]),
                    regression_passed=bool(reg and reg["passed"]),
                    aa_hessians_missing=[f"{c}_{g}" for c in STRATS for g in G if not os.path.exists(XR.aa_hessian_path(c, g))])
    if missing or invalid or not blockers["routing_preparation_valid"] or not blockers["regression_passed"] \
            or blockers["aa_hessians_missing"]:
        return dict(protocol_sha256=pr["protocol_sha256"], decision="RUN-BLOCKED", blockers=blockers), None
    man = json.load(open(P("input_manifest.json")))
    for c in STRATS:
        assert XR.sha256_file(XR.oo_result_path(c)) == man["references"][f"{c}_OO_ORIGINAL"]["result_sha256"]
        for g in G:
            assert XR.sha256_file(XR.aa_result_path(c, g)) == man["references"][f"{c}_AA_{g}"]["result_sha256"]

    # ---------------- AO correctness + token-count invariant
    H = {c: {("OO", None): torch.load(os.path.join(XR.FRD.FCACHE, "hessians", f"{c}_OO.pt"))} for c in STRATS}
    for c in STRATS:
        for g in G:
            assert XR.sha256_file(ao[c][g]["hessian_file"]) == ao[c][g]["hessian_file_sha256"]
            H[c][("AO", g)] = torch.load(ao[c][g]["hessian_file"])
            H[c][("AA", g)] = torch.load(XR.aa_hessian_path(c, g))
    inv = {}
    for c in STRATS:
        cnt = {g: {k: float(v) for k, v in H[c][("AO", g)]["tokens"].items()} for g in G}
        exp = rp["expected_ao_token_counts"][c]
        inv[c] = dict(identical_across_groupings=RB.ao_counts_invariant(cnt),
                      equal_R_O_expected=all(cnt[g][f"{L}.{e}.gu"] == float(exp[str(L)][e]) for g in G for L in LAYERS for e in range(60)),
                      aa_counts_vary_across_groupings=not RB.ao_counts_invariant(
                          {g: {k: float(v) for k, v in H[c][("AA", g)]["tokens"].items()} for g in G}),
                      n_experts_with_tokens=int(sum(float(exp[str(L)][e]) > 0 for L in LAYERS for e in range(60))))
    ao_checks = {f"{c}_AO_{g}": dict(
        reused=g in RB.REUSED_AO, valid=ao[c][g]["valid"], ppl=ao[c][g]["ppl"], mean_nll=ao[c][g]["mean_nll"],
        recon=ao[c][g]["recon"], installed_weights_sha256=ao[c][g]["installed_weights_sha256"],
        n_quantized_experts=ao[c][g]["n_quantized_experts"], n_projection_tensors=ao[c][g]["n_projection_tensors"],
        rtn_fallbacks=ao[c][g]["rtn_fallbacks"], nonfinite_weights=ao[c][g]["nonfinite_weights"],
        projections_unchanged_from_bf16=ao[c][g]["projections_unchanged_from_bf16"],
        router_weights_unchanged=ao[c][g]["router_weights_unchanged"],
        untouched_params_unchanged=ao[c][g]["untouched_params_unchanged"],
        forced_token_counts_match_table=ao[c][g]["forced_token_counts_match_table"],
        expert_checks_pass=all(x["pass"] for x in ao[c][g]["expert_checks"]),
        repeat_eval_abs_diff_ppl=ao[c][g]["repeat_eval_abs_diff_ppl"],
        repeat_eval_per_window_identical=ao[c][g]["repeat_eval_per_window_identical"],
        forced_routing_frac_set_diff={L: v["frac_set_diff"] for L, v in ao[c][g]["forced_routing_stats"].items()},
        quantize_s=ao[c][g]["quantize_s"], eval_s=ao[c][g]["eval_s"], peak_gpu_gb=ao[c][g]["peak_gpu_gb"],
        peak_cpu_gb=ao[c][g]["peak_cpu_gb"], result=os.path.relpath(XR.ao_result_path(c, g), ROOT))
        for c in STRATS for g in G}
    XR.jdump(dict(configs=ao_checks, token_count_invariant=inv, regression=reg), P("ao_results.json"))
    XR.jdump({f"{c}_{g}": dict(ppl=aa[c][g]["ppl"], mean_nll=aa[c][g]["mean_nll"], recon=aa[c][g]["recon"],
                               installed_weights_sha256=aa[c][g]["installed_weights_sha256"],
                               result=os.path.relpath(XR.aa_result_path(c, g), ROOT)) for c in STRATS for g in G}
             | {f"{c}_OO": dict(ppl=oo[c]["ppl"], mean_nll=oo[c]["mean_nll"], recon=oo[c]["recon"],
                                installed_weights_sha256=oo[c]["installed_weights_sha256"],
                                result=os.path.relpath(XR.oo_result_path(c), ROOT)) for c in STRATS},
             P("aa_reference_results.json"))
    if not all(v["identical_across_groupings"] and v["equal_R_O_expected"] for v in inv.values()):
        return dict(protocol_sha256=pr["protocol_sha256"], decision="RUN-BLOCKED",
                    blockers=dict(token_count_invariant=inv)), None

    # ---------------- PPL robustness
    y = {c: dict(OO=oo[c]["ppl"], AO=[ao[c][g]["ppl"] for g in G], AA=[aa[c][g]["ppl"] for g in G]) for c in STRATS}
    nl = {c: dict(OO=oo[c]["mean_nll"], AO=[ao[c][g]["mean_nll"] for g in G], AA=[aa[c][g]["mean_nll"] for g in G])
          for c in STRATS}
    sp = {c: RB.spread(y[c]["AO"], y[c]["AA"]) for c in STRATS}
    sp_nll = {c: RB.spread(nl[c]["AO"], nl[c]["AA"]) for c in STRATS}
    dist = {c: RB.distances(y[c]["AO"], y[c]["AA"], y[c]["OO"]) for c in STRATS}
    eff = {c: RB.intervention_effects(y[c]["AO"], y[c]["AA"]) for c in STRATS}
    boot = {c: {g: dict(AO_vs_OO=FD.paired_bootstrap(ao[c][g]["per_window_nll"], oo[c]["per_window_nll"], oo[c]["per_window_tokens"]),
                        AA_vs_OO=FD.paired_bootstrap(aa[c][g]["per_window_nll"], oo[c]["per_window_nll"], oo[c]["per_window_tokens"]),
                        AO_vs_AA=FD.paired_bootstrap(ao[c][g]["per_window_nll"], aa[c][g]["per_window_nll"], oo[c]["per_window_tokens"]))
                for g in G} for c in STRATS}
    XR.jdump(dict(groupings=G, bf16_ppl=pr["evaluation"]["bf16_ppl"], ppl=y, mean_nll=nl, spread=sp, spread_mean_nll=sp_nll,
                  distance_to_OO=dist, intervention_effect_AO_minus_AA=eff, window_bootstrap_mean_nll=boot,
                  per_window={c: dict(OO=ppl_view(oo[c]), AO={g: ppl_view(ao[c][g]) for g in G},
                                      AA={g: ppl_view(aa[c][g]) for g in G}) for c in STRATS},
                  note="OO is one fixed reference per strategy; distance to OO is a protocol-stability metric, not a "
                       "quantization-quality metric"), P("ppl_robustness.json"))
    delta_oo = y["FC"]["OO"] - y["FR"]["OO"]
    rk = RB.ranking({k: y["FR"][k] for k in ("AO", "AA")}, {k: y["FC"][k] for k in ("AO", "AA")}, delta_oo)
    XR.jdump(dict(groupings=G, **rk), P("ranking_robustness.json"))

    # ---------------- Hessians (16 diagnostic Hessians; act-order over all 480)
    hz = {}
    for c in STRATS:
        hz[c] = {}
        for g in G:
            pairs = {"AO_vs_OO": (("AO", g), ("OO", None)), "AA_vs_OO": (("AA", g), ("OO", None)),
                     "AO_vs_AA": (("AO", g), ("AA", g))}
            hz[c][g] = {}
            for key, (a, b) in pairs.items():
                Ha, Hb = H[c][a], H[c][b]
                m = {kk: FRT.hessian_metrics(Ha["full"][kk], Hb["full"][kk], Ha["tokens"][kk], Hb["tokens"][kk])
                     for kk in Hb["full"]}
                pm = {kk: FRT.perm_metrics(Ha["perm"][kk], Hb["perm"][kk]) for kk in Hb["perm"]}
                v = [x["rel_fro_trace_normalized"] for x in m.values()]
                hz[c][g][key] = dict(
                    rel_fro_trace_normalized=FRT.summarize(v),
                    by_projection={pj: FRT.summarize([x["rel_fro_trace_normalized"] for kk, x in m.items() if kk.endswith(pj)])
                                   for pj in ("gu", "down")},
                    actorder=dict(n_hessians=len(pm), n_identical_perm=sum(x["identical"] for x in pm.values()),
                                  mean_frac_positions_changed=float(np.mean([x["frac_positions_changed"] for x in pm.values()])),
                                  mean_frac_columns_changed_group=float(np.mean([x["frac_columns_changed_group"] for x in pm.values()]))),
                    per_hessian=m)
    hsum = {c: {g: {key: dict(median=v["rel_fro_trace_normalized"]["median"], max=v["rel_fro_trace_normalized"]["max"],
                              cols_changed_group=v["actorder"]["mean_frac_columns_changed_group"])
                    for key, v in hz[c][g].items()} for g in G} for c in STRATS}
    hagg = {c: {key: dict(median_over_groupings=float(np.median([hsum[c][g][key]["median"] for g in G])),
                          min=float(min(hsum[c][g][key]["median"] for g in G)), max=float(max(hsum[c][g][key]["median"] for g in G)),
                          cols_changed_group_mean=float(np.mean([hsum[c][g][key]["cols_changed_group"] for g in G])))
                for key in ("AO_vs_OO", "AA_vs_OO", "AO_vs_AA")} for c in STRATS}
    for c in STRATS:
        hagg[c]["AO_over_AA_median_ratio_per_grouping"] = [hsum[c][g]["AO_vs_OO"]["median"] / hsum[c][g]["AA_vs_OO"]["median"] for g in G]
        hagg[c]["n_groupings_AO_below_AA"] = int(sum(hsum[c][g]["AO_vs_OO"]["median"] < hsum[c][g]["AA_vs_OO"]["median"] for g in G))
    XR.jdump(dict(subset=pr["diagnostic_subset"], summary=hsum, aggregate=hagg, detail=hz), P("hessian_robustness.json"))

    # ---------------- effective weights (all 720 tensors)
    wz = {c: {g: {k: dict(per_layer={}) for k in ("AO_vs_OO", "AA_vs_OO", "AO_vs_AA")} for g in G} for c in STRATS}
    for c in STRATS:
        for L in LAYERS:
            Wo = torch.load(os.path.join(oo[c]["weights_dir"], f"L{L}.pt"))
            for g in G:
                Wao = torch.load(os.path.join(ao[c][g]["weights_dir"], f"L{L}.pt"))
                Waa = torch.load(os.path.join(aa[c][g]["weights_dir"], f"L{L}.pt"))
                for key, (A, B) in {"AO_vs_OO": (Wao, Wo), "AA_vs_OO": (Waa, Wo), "AO_vs_AA": (Wao, Waa)}.items():
                    wz[c][g][key]["per_layer"][str(L)] = FRT.weight_diff(A, B)
                del Wao, Waa
            del Wo
        for g in G:
            for key, v in wz[c][g].items():
                pl = v["per_layer"].values()
                n, dif = sum(x["n"] for x in pl), sum(x["n_differ"] for x in pl)
                d2, o2 = sum(x["d2"] for x in pl), sum(x["o2"] for x in pl)
                v["total"] = dict(frac_differ=dif / n, rel_l2=(d2 / o2) ** .5,
                                  n_tensors_identical=sum(x["n_tensors_identical"] for x in pl), n_tensors=sum(x["n_tensors"] for x in pl))
    wsum = {c: {g: {k: v["total"] for k, v in wz[c][g].items()} for g in G} for c in STRATS}
    XR.jdump(dict(summary=wsum, detail=wz), P("weight_robustness.json"))

    # ---------------- reconstruction
    rec = {}
    for c in STRATS:
        ref = {(r["layer"], r["expert"]): r["mean"] for r in oo[c]["per_expert"]}
        rec[c] = dict(OO=dict(recon=oo[c]["recon"], recon_per_layer=oo[c]["recon_per_layer"]), groupings={})
        for g in G:
            refa = {(r["layer"], r["expert"]): r["mean"] for r in aa[c][g]["per_expert"]}
            ch = np.array([(r["mean"] - ref[(r["layer"], r["expert"])]) / ref[(r["layer"], r["expert"])] for r in ao[c][g]["per_expert"]])
            rec[c]["groupings"][g] = dict(
                AO=ao[c][g]["recon"], AA=aa[c][g]["recon"], AO_per_layer=ao[c][g]["recon_per_layer"],
                AA_per_layer=aa[c][g]["recon_per_layer"],
                AO_rel_change_vs_OO=(ao[c][g]["recon"] - oo[c]["recon"]) / oo[c]["recon"],
                AA_rel_change_vs_OO=(aa[c][g]["recon"] - oo[c]["recon"]) / oo[c]["recon"],
                AO_rel_change_vs_AA=(ao[c][g]["recon"] - aa[c][g]["recon"]) / aa[c][g]["recon"],
                AO_per_expert_abs_rel_change_vs_OO_max=float(np.max(np.abs(ch))),
                AO_per_expert_abs_rel_change_vs_AA_max=float(max(abs(r["mean"] - refa[(r["layer"], r["expert"])]) /
                                                                 refa[(r["layer"], r["expert"])] for r in ao[c][g]["per_expert"])))
    rec["FC_penalty_rel"] = {k: {g: ((ao if k == "AO" else aa)["FC"][g]["recon"] - (ao if k == "AO" else aa)["FR"][g]["recon"])
                                 / (ao if k == "AO" else aa)["FR"][g]["recon"] for g in G} for k in ("AO", "AA")}
    rec["FC_penalty_rel"]["OO"] = (oo["FC"]["recon"] - oo["FR"]["recon"]) / oo["FR"]["recon"]
    rec["max_abs_rel_change_vs_OO"] = max(abs(rec[c]["groupings"][g][k]) for c in STRATS for g in G
                                          for k in ("AO_rel_change_vs_OO", "AA_rel_change_vs_OO"))
    XR.jdump(rec, P("reconstruction_robustness.json"))

    # ---------------- decision (frozen rule)
    cr = RB.criteria(sp, rk, dist)
    fail_detail = {}
    for name in ao_failed:
        c, g = name.split("_AO_")
        r = ao[c][g]
        fail_detail[name] = dict(
            failed_expert_checks=[{k: v for k, v in ck.items() if not isinstance(v, (list, dict))}
                                  for ck in r["expert_checks"] if not ck["pass"]],
            other_checks=dict(rtn_fallbacks=r["rtn_fallbacks"], nonfinite_weights=r["nonfinite_weights"],
                              router_weights_unchanged=r["router_weights_unchanged"],
                              untouched_params_unchanged=r["untouched_params_unchanged"],
                              forced_token_counts_match_table=r["forced_token_counts_match_table"],
                              projections_unchanged_from_bf16=r["projections_unchanged_from_bf16"],
                              repeat_eval_abs_diff_ppl=r["repeat_eval_abs_diff_ppl"]))
    cond_decision, cond_hits = RB.decide(cr)
    decision, partial_hits = RB.decide(cr, blocked=bool(ao_failed))
    out = dict(protocol_sha256=pr["protocol_sha256"], decision=decision, partial_items_satisfied=partial_hits,
               blocked_by_failed_ao_checks=fail_detail,
               conditional_decision_if_failed_configs_accepted=dict(
                   decision=cond_decision, partial_items_satisfied=cond_hits,
                   note="NOT the frozen decision: computed only to show what the frozen rule would return if the "
                        "failed configuration(s) were accepted; no tolerance was changed") if ao_failed else None,
               criteria=cr,
               decision_rule=RB.DECISION_RULE, groupings=G, ppl=y, mean_nll=nl, spread=sp, spread_mean_nll=sp_nll,
               distance_to_OO=dist, ranking=rk, intervention_effect=eff, hessian=hsum, hessian_aggregate=hagg,
               weights=wsum, reconstruction=rec, token_count_invariant=inv, regression=reg,
               ao_checks={k: {kk: v[kk] for kk in ("valid", "rtn_fallbacks", "nonfinite_weights", "router_weights_unchanged",
                                                   "forced_token_counts_match_table", "expert_checks_pass",
                                                   "repeat_eval_abs_diff_ppl", "reused")} for k, v in ao_checks.items()},
               compute={k: {kk: v[kk] for kk in ("quantize_s", "eval_s", "peak_gpu_gb", "peak_cpu_gb")}
                        for k, v in ao_checks.items() if not v["reused"]})
    return out, dict(hz=hz, wz=wz)


def figures(o, raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    d = P("figures")
    sv = lambda f, n: (f.savefig(os.path.join(d, n + ".pdf"), bbox_inches="tight"),
                       f.savefig(os.path.join(d, n + ".png"), dpi=200, bbox_inches="tight"), plt.close(f))
    x = np.arange(len(G))
    y = o["ppl"]

    # 1 PPL across groupings
    f, axs = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    for ax, c in zip(axs, STRATS):
        ax.axhline(y[c]["OO"], color=KCOL["OO"], lw=1.2, ls="--", label=f"OO {y[c]['OO']:.4f}")
        for k, m in (("AA", "s"), ("AO", "o")):
            ax.plot(x, y[c][k], color=KCOL[k], marker=m, lw=1.2, label=f"{k}_g (X_g, R_{'g' if k == 'AA' else 'O'})")
        ax.set_xticks(x, [f"G…{s}" for s in GL]); ax.set_title(c, loc="left")
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    axs[0].set_ylabel("WikiText-2 PPL")
    f.suptitle("Fig 1 — PPL across the five prespecified groupings (exact values: ppl_robustness.json)", x=.02, y=1.02,
               ha="left", fontsize=9.5)
    sv(f, "fig1_ppl_groupings")

    # 2 grouping SDs
    f, ax = plt.subplots(figsize=(5.8, 3.3))
    names = ["FR", "FC", "Δ (FC−FR)"]
    aa_sd = [o["spread"]["FR"]["SD_AA"], o["spread"]["FC"]["SD_AA"], o["ranking"]["AA"]["SD"]]
    ao_sd = [o["spread"]["FR"]["SD_AO"], o["spread"]["FC"]["SD_AO"], o["ranking"]["AO"]["SD"]]
    rr = [o["spread"]["FR"]["SD_ratio"], o["spread"]["FC"]["SD_ratio"], o["ranking"]["SD_Delta_ratio"]]
    xi = np.arange(3)
    ax.bar(xi - .19, aa_sd, .38, color=KCOL["AA"], label="AA (natural routing)")
    ax.bar(xi + .19, ao_sd, .38, color=KCOL["AO"], label="AO (frozen R_O)")
    for i in range(3):
        ax.annotate(f"ratio {rr[i]:.2f}" if rr[i] is not None else "ratio n/a", (i, max(aa_sd[i], ao_sd[i])),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.5)
    ax.set_xticks(xi, names); ax.set_ylabel("SD over 5 groupings (ddof=1), PPL")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Fig 2 — Grouping sensitivity: AA vs AO", loc="left", fontsize=9.5)
    sv(f, "fig2_sd_reduction")

    # 3 ranking
    f, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.axhline(0, color=INK, lw=.8)
    ax.axhline(o["ranking"]["Delta_OO"], color=KCOL["OO"], ls="--", lw=1.2, label=f"Δ_OO {o['ranking']['Delta_OO']:+.4f}")
    ax.plot(x, o["ranking"]["AA"]["Delta"], color=KCOL["AA"], marker="s", lw=1.2, label="Δ_AA,g")
    ax.plot(x, o["ranking"]["AO"]["Delta"], color=KCOL["AO"], marker="o", lw=1.2, label="Δ_AO,g")
    ax.set_xticks(x, [f"G…{s}" for s in GL]); ax.set_ylabel("Δ = PPL(FC) − PPL(FR)")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Fig 3 — FR/FC ranking per grouping (positive: FC worse)", loc="left", fontsize=9.5)
    sv(f, "fig3_ranking")

    # 4 Hessian stability
    f, axs = plt.subplots(1, 2, figsize=(8.0, 3.3), sharey=True)
    for ax, c in zip(axs, STRATS):
        for k, off in (("AA", -.12), ("AO", .12)):
            for i, g in enumerate(G):
                v = [m["rel_fro_trace_normalized"] for m in raw["hz"][c][g][f"{k}_vs_OO"]["per_hessian"].values()]
                ax.scatter(np.full(len(v), i + off), v, color=KCOL[k], s=9, alpha=.55, lw=0,
                           label=f"{k}_g vs OO (16 Hessians)" if i == 0 else None)
                ax.hlines(np.median(v), i + off - .1, i + off + .1, color=KCOL[k], lw=2.2)
        ax.set_yscale("log"); ax.set_xticks(x, [f"G…{s}" for s in GL]); ax.set_title(c, loc="left")
    axs[0].set_ylabel("rel. Frobenius diff. from OO\n(trace-normalized; bar = median)")
    axs[1].legend(frameon=False, fontsize=7.5)
    f.suptitle("Fig 4 — Hessian stability (layers 1/8/16/23 × experts 0/30 × gu/down)", x=.02, ha="left", fontsize=9.5)
    sv(f, "fig4_hessian_stability")

    # 5 weight stability
    f, axs = plt.subplots(1, 2, figsize=(8.0, 3.3), sharey=True)
    for ax, c in zip(axs, STRATS):
        for k, m in (("AA", "s"), ("AO", "o")):
            ax.plot(x, [o["weights"][c][g][f"{k}_vs_OO"]["frac_differ"] * 100 for g in G], color=KCOL[k], marker=m, lw=1.2,
                    label=f"{k}_g vs OO")
        ax.plot(x, [o["weights"][c][g]["AO_vs_AA"]["frac_differ"] * 100 for g in G], color=MUTED, marker="^", lw=1, ls=":",
                label="AO_g vs AA_g")
        ax.set_xticks(x, [f"G…{s}" for s in GL]); ax.set_title(c, loc="left"); ax.set_ylim(0, None)
    axs[0].set_ylabel("effective weight elements differing (%)\n(720 BF16 tensors)")
    axs[1].legend(frameon=False, fontsize=7.5)
    f.suptitle("Fig 5 — GPTQ effective-weight stability", x=.02, ha="left", fontsize=9.5)
    sv(f, "fig5_weight_stability")

    # 6 reconstruction vs PPL
    f, ax = plt.subplots(figsize=(6.0, 3.7))
    for c in STRATS:
        for k in ("AA", "AO"):
            xr = [o["reconstruction"][c]["groupings"][g][f"{k}_rel_change_vs_OO"] * 100 for g in G]
            yp = [v - y[c]["OO"] for v in y[c][k]]
            ax.scatter(xr, yp, color=KCOL[k], marker="o" if c == "FR" else "s", s=34, edgecolor="white", lw=.8, zorder=3,
                       label=f"{c} {k}")
    ax.axhline(0, color=INK, lw=.8); ax.axvline(0, color=INK, lw=.8)
    ax.set_xlabel("mean reconstruction change vs OO (%)"); ax.set_ylabel("PPL − PPL(OO)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.set_title("Fig 6 — Reconstruction vs PPL (○ FR, □ FC)", loc="left", fontsize=9.5)
    sv(f, "fig6_recon_vs_ppl")


def markdown(o):
    y, sp, rk, dist = o["ppl"], o["spread"], o["ranking"], o["distance_to_OO"]
    f6 = lambda v: "n/a" if v is None else f"{v:.3f}"
    L = ["# Frozen-routing robustness across five calibration groupings (FR s0 / FC s0, four layers)", "",
         f"**Decision (rule frozen in protocol.json, sha256 `{o['protocol_sha256'][:16]}`): {o['decision']}**"
         + (f" — satisfied partial items: {', '.join(o['partial_items_satisfied'])}" if o["partial_items_satisfied"] else ""), ""]
    for name, d in o["blocked_by_failed_ao_checks"].items():
        for ck in d["failed_expert_checks"]:
            L.append(f"- blocker: {name} installation check L{ck['layer']} e{ck['expert']}: installed_vs_e3_rel "
                     f"{ck['installed_vs_e3_rel']:.6f} > tolerance {ck['tolerance']:.6f} "
                     f"({ck['installed_vs_e3_rel'] / ck['tolerance']:.3f}x); installed == BF16(E3 Wq): "
                     f"{ck['installed_equals_bf16_of_e3_wq']}")
    if o["conditional_decision_if_failed_configs_accepted"]:
        cd = o["conditional_decision_if_failed_configs_accepted"]
        L += [f"- conditional (NOT the decision; failed configuration accepted as is): {cd['decision']} "
              f"{cd['partial_items_satisfied']}", "", "All numbers below include the blocked configuration (descriptive only).", ""]
    L += [
         "| grouping | FR OO | FR AA_g | FR AO_g | FC OO | FC AA_g | FC AO_g | Δ_AA,g | Δ_AO,g |", "|---|---|---|---|---|---|---|---|---|"]
    for i, g in enumerate(G):
        L.append(f"| {g} | {y['FR']['OO']:.6f} | {y['FR']['AA'][i]:.6f} | {y['FR']['AO'][i]:.6f} | {y['FC']['OO']:.6f} | "
                 f"{y['FC']['AA'][i]:.6f} | {y['FC']['AO'][i]:.6f} | {rk['AA']['Delta'][i]:+.6f} | {rk['AO']['Delta'][i]:+.6f} |")
    L += ["", "| quantity | AA | AO | ratio AO/AA |", "|---|---|---|---|"]
    for c in STRATS:
        L.append(f"| {c} SD | {sp[c]['SD_AA']:.6f} | {sp[c]['SD_AO']:.6f} | {f6(sp[c]['SD_ratio'])} |")
        L.append(f"| {c} range | {sp[c]['Range_AA']:.6f} | {sp[c]['Range_AO']:.6f} | {f6(sp[c]['Range_ratio'])} |")
        L.append(f"| {c} mean |dist to OO| | {dist[c]['mean_Dist_AA']:.6f} | {dist[c]['mean_Dist_AO']:.6f} | AO closer in "
                 f"{dist[c]['n_AO_closer']}/5 |")
    L.append(f"| Δ SD | {rk['AA']['SD']:.6f} | {rk['AO']['SD']:.6f} | {f6(rk['SD_Delta_ratio'])} |")
    L.append(f"| Δ mean | {rk['AA']['mean']:+.6f} | {rk['AO']['mean']:+.6f} | Δ_OO {rk['Delta_OO']:+.6f} |")
    L.append(f"| ranking preserved vs OO | {rk['AA']['n_preserved_vs_OO']}/5 | {rk['AO']['n_preserved_vs_OO']}/5 | |")
    L += ["", "criteria: " + ", ".join(f"{k}={v}" for k, v in o["criteria"].items()), ""]
    L += ["| strategy | grouping | Hessian median AO vs OO | AA vs OO | cols changing group AO / AA | weights differ AO / AA | rel L2 AO / AA | recon AO / AA rel vs OO |",
          "|---|---|---|---|---|---|---|---|"]
    for c in STRATS:
        for g in G:
            h, w, r = o["hessian"][c][g], o["weights"][c][g], o["reconstruction"][c]["groupings"][g]
            L.append(f"| {c} | {g} | {h['AO_vs_OO']['median']:.4f} | {h['AA_vs_OO']['median']:.4f} | "
                     f"{h['AO_vs_OO']['cols_changed_group'] * 100:.1f}% / {h['AA_vs_OO']['cols_changed_group'] * 100:.1f}% | "
                     f"{w['AO_vs_OO']['frac_differ'] * 100:.1f}% / {w['AA_vs_OO']['frac_differ'] * 100:.1f}% | "
                     f"{w['AO_vs_OO']['rel_l2']:.4f} / {w['AA_vs_OO']['rel_l2']:.4f} | "
                     f"{r['AO_rel_change_vs_OO'] * 100:+.4f}% / {r['AA_rel_change_vs_OO'] * 100:+.4f}% |")
    return "\n".join(L) + "\n"


def main():
    out, raw = build()
    XR.jdump(out, P("summary.json"))
    if raw is None:
        print(f"[report] RUN-BLOCKED: {out['blockers']}"); return
    open(P("summary.md"), "w").write(markdown(out))
    figures(out, raw)
    print(f"[report] decision {out['decision']} {out['partial_items_satisfied']}; criteria {out['criteria']}")


if __name__ == "__main__":
    main()
