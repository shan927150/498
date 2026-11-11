"""Report for the GPTQ act-order ablation: invariance checks, weight / quantizer / reconstruction / PPL / ranking
sensitivity of AOF vs AO (and OO), the frozen decision, figures 1-6, summary.md."""
from __future__ import annotations
import json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import actorder_ablation as XA              # noqa: E402
import frozen_routing_robustness as XR      # noqa: E402
import full_depth_pilot as FD               # noqa: E402
from moe_calib import actorder_ablation as AA   # noqa: E402
from moe_calib import frozen_routing as FRT     # noqa: E402

P, STRATS, LAYERS, G = XA.p, XA.STRATS, XA.LAYERS, XA.G
GL = [g[-2:] for g in G]
KCOL = {"OO": "#52514e", "AO": "#2a78d6", "AOF": "#eb6834", "AA": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def wtotal(per_layer):
    pl = per_layer.values()
    n, dif = sum(x["n"] for x in pl), sum(x["n_differ"] for x in pl)
    d2, o2 = sum(x["d2"] for x in pl), sum(x["o2"] for x in pl)
    return dict(frac_differ=dif / n, rel_l2=(d2 / o2) ** .5, n_tensors_identical=sum(x["n_tensors_identical"] for x in pl),
                n_tensors=sum(x["n_tensors"] for x in pl))


def pw(r):
    return dict(ppl=r["ppl"], mean_nll=r["mean_nll"], total_nll=r["total_nll"], total_tokens=r["total_tokens"],
                per_window_nll=r["per_window_nll"], per_window_tokens=r["per_window_tokens"])


def build():
    pr = XA.protocol()
    audit, sreg = load(P("installation_audit.json")), load(P("solver_regression.json"))
    aof = {c: {g: load(XA.aof_result_path(c, g)) for g in G} for c in STRATS}
    ao = {c: {g: load(XR.ao_result_path(c, g)) for g in G} for c in STRATS}
    oo = {c: load(XR.oo_result_path(c)) for c in STRATS}
    aa = {c: {g: load(XR.aa_result_path(c, g)) for g in G} for c in STRATS}
    blockers = dict(audit=(audit or {}).get("decision"), solver_regression_passed=bool(sreg and sreg["passed"]),
                    missing=[XA.aof_name(c, g) for c in STRATS for g in G if aof[c][g] is None],
                    invalid=[XA.aof_name(c, g) for c in STRATS for g in G if aof[c][g] is not None and not aof[c][g]["valid"]])
    if blockers["audit"] != "AUDIT-CONFIRMED-BF16-REPRESENTATION" or not blockers["solver_regression_passed"] \
            or blockers["missing"] or blockers["invalid"]:
        return dict(protocol_sha256=pr["protocol_sha256"], decision="RUN-BLOCKED", blockers=blockers), None
    man = json.load(open(P("input_manifest.json")))
    for c in STRATS:
        for g in G:
            assert XA.sha256_file(XR.ao_result_path(c, g)) == man["references"][f"{c}_AO_{g}"]["result_sha256"]
    XA.jdump({XA.aof_name(c, g): {k: v for k, v in aof[c][g].items() if k not in ("per_window_nll", "per_window_tokens", "per_expert",
                                                                               "integrity_checks", "frozen_old_check")}
              for c in STRATS for g in G}, P("aof_results.json"))
    XA.jdump({f"{c}_AO_{g}": dict(ppl=ao[c][g]["ppl"], recon=ao[c][g]["recon"], valid_under_frozen_old_check=ao[c][g]["valid"],
                                  installed_weights_sha256=ao[c][g]["installed_weights_sha256"],
                                  result=os.path.relpath(XR.ao_result_path(c, g), ROOT)) for c in STRATS for g in G}
             | {f"{c}_OO": dict(ppl=oo[c]["ppl"], recon=oo[c]["recon"], installed_weights_sha256=oo[c]["installed_weights_sha256"])
                for c in STRATS}, P("ao_reference_results.json"))

    # ---------------- Hessian / act-order invariance and differences
    perms_oo = {c: torch.load(P("original_actorder.pt"))[c] for c in STRATS}
    hz = {}
    rob_h = json.load(open(os.path.join(ROOT, XR.OUT, "hessian_robustness.json")))["summary"]
    for c in STRATS:
        hz[c] = {}
        for g in G:
            Hao = torch.load(ao[c][g]["hessian_file"])
            pa = {}
            for k in perms_oo[c]:
                L, e, pj = k.split(".")
                pa[k] = AA.perm_compare(Hao["perm"][f"{L}.{e}.{AA.PROJ_H[pj]}"], perms_oo[c][k])
            hz[c][g] = dict(
                AOF_hessians_equal_AO=aof[c][g]["hessians_equal_AO"], AOF_perm_equal_OO=aof[c][g]["perm_equal_original"],
                AOF_vs_OO_actorder=dict(frac_positions_changed=0.0, frac_columns_changed_group=0.0,
                                        basis="every applied permutation verified equal to the ORIGINAL one"),
                AO_vs_OO_actorder=dict(n_identical=sum(v["identical"] for v in pa.values()), n=len(pa),
                                       mean_frac_positions_changed=float(np.mean([v["frac_positions_changed"] for v in pa.values()])),
                                       mean_frac_columns_changed_group=float(np.mean([v["frac_columns_changed_group"] for v in pa.values()])),
                                       by_proj={pj: float(np.mean([v["frac_columns_changed_group"] for kk, v in pa.items() if kk.endswith(pj)]))
                                                for pj in AA.PROJ}),
                hessian_median_rel_fro_vs_OO=dict(AO=rob_h[c][g]["AO_vs_OO"]["median"], AOF=rob_h[c][g]["AO_vs_OO"]["median"],
                                                  note="AOF Hessians are bitwise the AO Hessians"),
                routed_token_counts_equal_R_O=aof[c][g]["token_counts_equal_R_O"])
    XA.jdump(hz, P("hessian_checks.json"))

    # ---------------- effective weights (720 tensors)
    wz = {c: {g: {k: dict(per_layer={}, per_expert_rel_l2=[]) for k in ("AO_vs_OO", "AOF_vs_OO", "AOF_vs_AO")} for g in G} for c in STRATS}
    for c in STRATS:
        for L in LAYERS:
            Wo = torch.load(os.path.join(oo[c]["weights_dir"], f"L{L}.pt"))
            for g in G:
                Wa = torch.load(os.path.join(ao[c][g]["weights_dir"], f"L{L}.pt"))
                Wf = torch.load(os.path.join(aof[c][g]["weights_dir"], f"L{L}.pt"))
                for key, (A, B) in {"AO_vs_OO": (Wa, Wo), "AOF_vs_OO": (Wf, Wo), "AOF_vs_AO": (Wf, Wa)}.items():
                    wz[c][g][key]["per_layer"][str(L)] = FRT.weight_diff(A, B)
                    for e in range(60):
                        ks = [f"{e}.{pj}" for pj in AA.PROJ]
                        wz[c][g][key]["per_expert_rel_l2"].append(FRT.weight_diff({k: A[k] for k in ks}, {k: B[k] for k in ks})["rel_l2"])
                del Wa, Wf
            del Wo
        for g in G:
            for key, v in wz[c][g].items():
                v["total"] = wtotal(v["per_layer"])
                v["per_expert_summary"] = FRT.summarize(v["per_expert_rel_l2"])
                v["per_layer_rel_l2"] = {L: x["rel_l2"] for L, x in v["per_layer"].items()}
    D = {c: {k: [wz[c][g][f"{k}_vs_OO"]["total"]["rel_l2"] for g in G] for k in ("AO", "AOF")} for c in STRATS}
    FD_ = {c: {k: [wz[c][g][f"{k}_vs_OO"]["total"]["frac_differ"] for g in G] for k in ("AO", "AOF")} for c in STRATS}
    wred = {c: AA.weight_reduction(D[c]["AOF"], D[c]["AO"]) for c in STRATS}
    fred = {c: AA.weight_reduction(FD_[c]["AOF"], FD_[c]["AO"]) for c in STRATS}
    XA.jdump(dict(summary={c: {g: {k: v["total"] | dict(per_layer_rel_l2=v["per_layer_rel_l2"], per_expert=v["per_expert_summary"])
                                   for k, v in wz[c][g].items()} for g in G} for c in STRATS},
                  D_rel_l2_vs_OO=D, frac_differ_vs_OO=FD_, WeightReduction=wred, ChangedFractionReduction=fred,
                  detail={c: {g: {k: v["per_layer"] for k, v in wz[c][g].items()} for g in G} for c in STRATS}),
             P("weight_sensitivity.json"))

    # ---------------- quantizer diagnostics (check experts: scales / zeros; all 720: weight-space column error)
    qd = {}
    for c in STRATS:
        Qo = torch.load(os.path.join(XA.ACACHE, "quantizer", f"{c}_OO.pt"))
        qd[c] = {}
        for g in G:
            Qf = torch.load(aof[c][g]["quantizer_file"])
            assert XA.sha256_file(aof[c][g]["quantizer_file"]) == aof[c][g]["quantizer_file_sha256"]
            ent = {"AOF_vs_OO": [], "AO_vs_OO": [], "AOF_vs_AO": []}
            for L, e in XA.CHECK:
                o, f, a = Qo["check_experts"][f"OO.{L}.{e}"], Qf["check_experts"][f"AOF.{L}.{e}"], Qf["check_experts"][f"AO.{L}.{e}"]
                for k in AA.PROJ:
                    for key, (x, y) in {"AOF_vs_OO": (f[k], o[k]), "AO_vs_OO": (a[k], o[k]), "AOF_vs_AO": (f[k], a[k])}.items():
                        ent[key].append(dict(L=L, e=e, proj=k,
                                             scale_rel_diff=float((x["scale"] - y["scale"]).norm() / y["scale"].norm()),
                                             frac_scale_equal=float((x["scale"] == y["scale"]).float().mean()),
                                             frac_zero_equal=float((x["zero"] == y["zero"]).float().mean()),
                                             col_err2_ratio=float(x["col_err2"].sum() / y["col_err2"].sum())))
            ce = {k: dict(OO=float(Qo["col_err2"][k].double().sum()), AOF=float(Qf["col_err2"][k].double().sum())) for k in Qo["col_err2"]}
            qd[c][g] = {key: dict(median_scale_rel_diff=float(np.median([x["scale_rel_diff"] for x in v])),
                                  mean_frac_zero_equal=float(np.mean([x["frac_zero_equal"] for x in v])),
                                  mean_frac_scale_equal=float(np.mean([x["frac_scale_equal"] for x in v])),
                                  median_col_err2_ratio=float(np.median([x["col_err2_ratio"] for x in v])), per_projection=v)
                        for key, v in ent.items()}
            qd[c][g]["all720_weight_space_quant_error_AOF_over_OO"] = FRT.summarize([v["AOF"] / v["OO"] for v in ce.values()])
    XA.jdump(dict(subset="8 prespecified check experts x gate/up/down (AO values recomputed on the bitwise-identical AO "
                         "Hessian inside each AOF run; their BF16 casts equal the stored AO weights); col_err2 = sum over "
                         "rows of (w - dq)^2 per column BEFORE error feedback, captured read-only in the solver",
                  per_config=qd), P("quantizer_diagnostics.json"))

    # ---------------- PPL / ranking / reconstruction
    y = {c: dict(OO=oo[c]["ppl"], AO=[ao[c][g]["ppl"] for g in G], AOF=[aof[c][g]["ppl"] for g in G],
                 AA=[aa[c][g]["ppl"] for g in G]) for c in STRATS}
    sp = {c: AA.ppl_spread(y[c]["AO"], y[c]["AOF"], y[c]["OO"]) for c in STRATS}
    for c in STRATS:
        sp[c]["SD_AA_context"] = AA.sd(y[c]["AA"])
        sp[c]["mean_nll"] = dict(AO=[ao[c][g]["mean_nll"] for g in G], AOF=[aof[c][g]["mean_nll"] for g in G], OO=oo[c]["mean_nll"])
    boot = {c: {g: FD.paired_bootstrap(aof[c][g]["per_window_nll"], ao[c][g]["per_window_nll"], oo[c]["per_window_tokens"]) for g in G}
            for c in STRATS}
    XA.jdump(dict(groupings=G, ppl=y, spread=sp, window_bootstrap_AOF_vs_AO=boot,
                  per_window={c: dict(OO=pw(oo[c]), AO={g: pw(ao[c][g]) for g in G}, AOF={g: pw(aof[c][g]) for g in G}) for c in STRATS},
                  note="AO G20261114 FC failed only the frozen BF16 tolerance check of the earlier study (that study stays "
                       "RUN-BLOCKED); the installation audit confirmed its installation bitwise"), P("ppl_sensitivity.json"))
    d_oo = y["FC"]["OO"] - y["FR"]["OO"]
    dl = {k: AA.delta_stats(y["FR"][k], y["FC"][k], d_oo) for k in ("AO", "AOF", "AA")}
    dl["Delta_OO"] = d_oo
    dl["SD_Delta_ratio"] = AA.ratio(dl["AOF"]["SD"], dl["AO"]["SD"])
    dl["Range_Delta_ratio"] = AA.ratio(dl["AOF"]["range"], dl["AO"]["range"])
    XA.jdump(dict(groupings=G, **dl), P("ranking_analysis.json"))
    rec = {}
    for c in STRATS:
        rec[c] = dict(OO=oo[c]["recon"], groupings={})
        for g in G:
            ref_ao = {(r["layer"], r["expert"]): r["mean"] for r in ao[c][g]["per_expert"]}
            ch = [abs(r["mean"] - ref_ao[(r["layer"], r["expert"])]) / ref_ao[(r["layer"], r["expert"])] for r in aof[c][g]["per_expert"]]
            rec[c]["groupings"][g] = dict(AO=ao[c][g]["recon"], AOF=aof[c][g]["recon"], AOF_per_layer=aof[c][g]["recon_per_layer"],
                                          AO_per_layer=ao[c][g]["recon_per_layer"],
                                          AOF_rel_change_vs_AO=(aof[c][g]["recon"] - ao[c][g]["recon"]) / ao[c][g]["recon"],
                                          AOF_rel_change_vs_OO=(aof[c][g]["recon"] - oo[c]["recon"]) / oo[c]["recon"],
                                          AO_rel_change_vs_OO=(ao[c][g]["recon"] - oo[c]["recon"]) / oo[c]["recon"],
                                          per_expert_abs_rel_change_vs_AO_median=float(np.median(ch)),
                                          per_expert_abs_rel_change_vs_AO_max=float(np.max(ch)))
    rec["FC_penalty_rel"] = {k: {g: ((aof if k == "AOF" else ao)["FC"][g]["recon"] - (aof if k == "AOF" else ao)["FR"][g]["recon"])
                                 / (aof if k == "AOF" else ao)["FR"][g]["recon"] for g in G} for k in ("AO", "AOF")}
    rec["FC_penalty_rel"]["OO"] = (oo["FC"]["recon"] - oo["FR"]["recon"]) / oo["FR"]["recon"]
    XA.jdump(rec, P("reconstruction_results.json"))

    cr = AA.criteria(wred, sp, dl)
    decision = AA.decide(cr)
    out = dict(protocol_sha256=pr["protocol_sha256"], decision=decision, criteria=cr, decision_rule=AA.DECISION_RULE,
               audit=dict(decision=audit["decision"], summary=audit["summary"], failure_explained=audit["failure_explained"],
                          failing_case=dict(expert=audit["failing_case"]["expert"], per_projection=audit["failing_case"]["per_projection"]),
                          negative_controls=audit["negative_controls"]),
               earlier_decision=man["earlier_decision"],
               solver_regression={k: v for k, v in sreg.items() if k.startswith("regression") or k == "passed"},
               groupings=G, ppl=y, spread=sp, ranking=dl, WeightReduction=wred, ChangedFractionReduction=fred, D=D, frac_differ=FD_,
               weights={c: {g: {k: v["total"] for k, v in wz[c][g].items()} for g in G} for c in STRATS},
               hessian=hz, quantizer={c: {g: {k: {kk: vv for kk, vv in v.items() if kk != "per_projection"} if isinstance(v, dict) and "per_projection" in v else v
                                              for k, v in qd[c][g].items()} for g in G} for c in STRATS},
               reconstruction=rec,
               aof_checks={XA.aof_name(c, g): dict(valid=aof[c][g]["valid"], integrity_pass=aof[c][g]["integrity_pass"],
                                                   frozen_old_check_all_pass=aof[c][g]["frozen_old_check_all_pass"],
                                                   old_check_max_ratio=max(ck["installed_vs_e3_rel"] / ck["tolerance"] for ck in aof[c][g]["frozen_old_check"]),
                                                   perm_equal_original=aof[c][g]["perm_equal_original"], hessians_equal_AO=aof[c][g]["hessians_equal_AO"],
                                                   ao_default_on_same_hessian_equals_stored_AO=aof[c][g]["ao_default_on_same_hessian_equals_stored_AO"],
                                                   repeat_eval_abs_diff_ppl=aof[c][g]["repeat_eval_abs_diff_ppl"])
                           for c in STRATS for g in G},
               compute={XA.aof_name(c, g): dict(quantize_s=aof[c][g]["quantize_s"], eval_s=aof[c][g]["eval_s"],
                                                peak_gpu_gb=aof[c][g]["peak_gpu_gb"], peak_cpu_gb=aof[c][g]["peak_cpu_gb"])
                        for c in STRATS for g in G})
    return out, dict(wz=wz)


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
    lab = [f"G…{s}" for s in GL]

    # 1 Hessian + act-order vs OO
    f, axs = plt.subplots(1, 2, figsize=(8.2, 3.3))
    for i, c in enumerate(STRATS):
        h = [o["hessian"][c][g]["hessian_median_rel_fro_vs_OO"]["AO"] for g in G]
        axs[0].plot(x, h, marker="o" if c == "FR" else "s", color=KCOL["AO"], ls="-" if c == "FR" else "--",
                    label=f"{c}: AO = AOF (bitwise)")
        a = [o["hessian"][c][g]["AO_vs_OO_actorder"]["mean_frac_columns_changed_group"] * 100 for g in G]
        axs[1].plot(x, a, marker="o" if c == "FR" else "s", color=KCOL["AO"], ls="-" if c == "FR" else "--", label=f"{c} AO")
        axs[1].plot(x, [0] * len(G), marker="o" if c == "FR" else "s", color=KCOL["AOF"], ls="-" if c == "FR" else "--", label=f"{c} AOF")
    axs[0].set_ylabel("median rel. Frobenius diff. vs OO\n(trace-normalized, 16 Hessians)"); axs[0].set_ylim(0, None)
    axs[1].set_ylabel("columns changing GPTQ group vs OO (%)\n(all 720 projections)")
    for ax in axs:
        ax.set_xticks(x, lab); ax.legend(frameon=False, fontsize=7)
    f.subplots_adjust(wspace=.45)
    f.suptitle("Fig 1 — Hessians unchanged by AOF; act-order / group membership reset to ORIGINAL", x=.02, ha="left", fontsize=9.5)
    sv(f, "fig1_hessian_actorder")

    # 2 weight sensitivity
    f, axs = plt.subplots(1, 2, figsize=(8.2, 3.3))
    for c in STRATS:
        for k in ("AO", "AOF"):
            m, ls = ("o" if c == "FR" else "s"), ("-" if c == "FR" else "--")
            axs[0].plot(x, [v * 100 for v in o["frac_differ"][c][k]], marker=m, ls=ls, color=KCOL[k], label=f"{c} {k}")
            axs[1].plot(x, o["D"][c][k], marker=m, ls=ls, color=KCOL[k], label=f"{c} {k}")
    axs[0].set_ylabel("effective weight elements ≠ OO (%)"); axs[1].set_ylabel("relative L2 vs OO (720 tensors)")
    for ax in axs:
        ax.set_xticks(x, lab); ax.set_ylim(0, None)
    axs[1].legend(frameon=False, fontsize=7, ncol=2)
    f.suptitle("Fig 2 — Effective GPTQ weight sensitivity vs OO", x=.02, ha="left", fontsize=9.5)
    sv(f, "fig2_weight_sensitivity")

    # 3 PPL across groupings
    f, axs = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, c in zip(axs, STRATS):
        ax.axhline(o["ppl"][c]["OO"], color=KCOL["OO"], ls="--", lw=1.2, label="OO")
        ax.plot(x, o["ppl"][c]["AA"], color=KCOL["AA"], marker="^", lw=.8, alpha=.6, label="AA (context)")
        ax.plot(x, o["ppl"][c]["AO"], color=KCOL["AO"], marker="o", lw=1.2, label="AO")
        ax.plot(x, o["ppl"][c]["AOF"], color=KCOL["AOF"], marker="s", lw=1.2, label="AOF")
        ax.set_xticks(x, lab); ax.set_title(c, loc="left"); ax.legend(frameon=False, fontsize=7, loc="lower left")
    axs[0].set_ylabel("WikiText-2 PPL")
    f.suptitle("Fig 3 — PPL across groupings (exact values: ppl_sensitivity.json)", x=.02, y=1.02, ha="left", fontsize=9.5)
    sv(f, "fig3_ppl_groupings")

    # 4 grouping sensitivity
    f, ax = plt.subplots(figsize=(5.8, 3.3))
    names = ["FR", "FC", "Δ (FC−FR)"]
    v_ao = [o["spread"]["FR"]["SD_AO"], o["spread"]["FC"]["SD_AO"], o["ranking"]["AO"]["SD"]]
    v_aof = [o["spread"]["FR"]["SD_AOF"], o["spread"]["FC"]["SD_AOF"], o["ranking"]["AOF"]["SD"]]
    rr = [o["spread"]["FR"]["SD_ratio"], o["spread"]["FC"]["SD_ratio"], o["ranking"]["SD_Delta_ratio"]]
    xi = np.arange(3)
    ax.bar(xi - .19, v_ao, .38, color=KCOL["AO"], label="AO")
    ax.bar(xi + .19, v_aof, .38, color=KCOL["AOF"], label="AOF")
    for i in range(3):
        ax.annotate(f"ratio {rr[i]:.2f}" if rr[i] is not None else "n/a", (i, max(v_ao[i], v_aof[i])), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7.5)
    ax.set_xticks(xi, names); ax.set_ylabel("SD over 5 groupings (ddof=1), PPL"); ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Fig 4 — Grouping sensitivity: AO vs AOF", loc="left", fontsize=9.5)
    sv(f, "fig4_grouping_sensitivity")

    # 5 reconstruction vs PPL
    f, ax = plt.subplots(figsize=(6.0, 3.7))
    for c in STRATS:
        for k in ("AO", "AOF"):
            xr = [o["reconstruction"][c]["groupings"][g][f"{k}_rel_change_vs_OO"] * 100 for g in G]
            yp = [v - o["ppl"][c]["OO"] for v in o["ppl"][c][k]]
            ax.scatter(xr, yp, color=KCOL[k], marker="o" if c == "FR" else "s", s=34, edgecolor="white", lw=.8, zorder=3, label=f"{c} {k}")
    ax.axhline(0, color=INK, lw=.8); ax.axvline(0, color=INK, lw=.8)
    ax.set_xlabel("mean reconstruction change vs OO (%)"); ax.set_ylabel("PPL − PPL(OO)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.set_title("Fig 5 — Reconstruction vs PPL (○ FR, □ FC)", loc="left", fontsize=9.5)
    sv(f, "fig5_recon_vs_ppl")

    # 6 quantizer parameters (check experts)
    f, axs = plt.subplots(1, 2, figsize=(8.2, 3.3))
    for c in STRATS:
        m, ls = ("o" if c == "FR" else "s"), ("-" if c == "FR" else "--")
        for key, col in (("AO_vs_OO", KCOL["AO"]), ("AOF_vs_OO", KCOL["AOF"]), ("AOF_vs_AO", MUTED)):
            axs[0].plot(x, [o["quantizer"][c][g][key]["median_scale_rel_diff"] for g in G], marker=m, ls=ls, color=col, label=f"{c} {key}")
            axs[1].plot(x, [(1 - o["quantizer"][c][g][key]["mean_frac_zero_equal"]) * 100 for g in G], marker=m, ls=ls, color=col)
    axs[0].set_ylabel("median rel. diff. of per-group scales"); axs[1].set_ylabel("zero-points differing (%)")
    for ax in axs:
        ax.set_xticks(x, lab); ax.set_ylim(0, None)
    axs[0].legend(frameon=False, fontsize=6.5, ncol=2)
    f.suptitle("Fig 6 — Quantizer parameters (8 check experts × 3 projections; read-only solver capture)", x=.02, ha="left", fontsize=9.5)
    sv(f, "fig6_quantizer_params")


def markdown(o):
    y, sp, dl = o["ppl"], o["spread"], o["ranking"]
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
    L = ["# GPTQ act-order ablation after frozen-routing calibration (FR s0 / FC s0, four layers, five groupings)", "",
         f"**Decision (rule frozen in protocol.json, sha256 `{o['protocol_sha256'][:16]}`): {o['decision']}**", "",
         f"- installation audit: {o['audit']['decision']}; earlier study decision unchanged: {o['earlier_decision']['decision']}",
         "- criteria: " + ", ".join(f"{k}={v}" for k, v in o["criteria"].items()), "",
         "| grouping | FR AO | FR AOF | FC AO | FC AOF | Δ_AO | Δ_AOF |", "|---|---|---|---|---|---|---|",
         f"| OO | {y['FR']['OO']:.6f} | | {y['FC']['OO']:.6f} | | {dl['Delta_OO']:+.6f} | |"]
    for i, g in enumerate(G):
        L.append(f"| {g} | {y['FR']['AO'][i]:.6f} | {y['FR']['AOF'][i]:.6f} | {y['FC']['AO'][i]:.6f} | {y['FC']['AOF'][i]:.6f} | "
                 f"{dl['AO']['Delta'][i]:+.6f} | {dl['AOF']['Delta'][i]:+.6f} |")
    L += ["", "| quantity | AO | AOF | ratio |", "|---|---|---|---|"]
    for c in STRATS:
        L.append(f"| {c} SD | {sp[c]['SD_AO']:.6f} | {sp[c]['SD_AOF']:.6f} | {f3(sp[c]['SD_ratio'])} |")
        L.append(f"| {c} range | {sp[c]['Range_AO']:.6f} | {sp[c]['Range_AOF']:.6f} | {f3(sp[c]['Range_ratio'])} |")
        L.append(f"| {c} mean \\|PPL−OO\\| | {sp[c]['mean_dist_OO_AO']:.6f} | {sp[c]['mean_dist_OO_AOF']:.6f} | AOF closer {sp[c]['n_AOF_closer_to_OO']}/5 |")
        L.append(f"| {c} weight rel L2 vs OO (mean) | {np.mean(o['D'][c]['AO']):.4f} | {np.mean(o['D'][c]['AOF']):.4f} | "
                 f"reduction {o['WeightReduction'][c]['mean']:+.3f} |")
        L.append(f"| {c} elements ≠ OO (mean) | {np.mean(o['frac_differ'][c]['AO']) * 100:.1f}% | {np.mean(o['frac_differ'][c]['AOF']) * 100:.1f}% | |")
    L.append(f"| Δ SD | {dl['AO']['SD']:.6f} | {dl['AOF']['SD']:.6f} | {f3(dl['SD_Delta_ratio'])} |")
    L.append(f"| ranking preserved vs OO | {dl['AO']['n_preserved_vs_OO']}/5 | {dl['AOF']['n_preserved_vs_OO']}/5 | (AA {dl['AA']['n_preserved_vs_OO']}/5) |")
    return "\n".join(L) + "\n"


def main():
    out, raw = build()
    XA.jdump(out, P("summary.json"))
    if raw is None:
        print(f"[report] RUN-BLOCKED: {out['blockers']}"); return
    open(P("summary.md"), "w").write(markdown(out))
    figures(out, raw)
    print(f"[report] decision {out['decision']}; criteria {out['criteria']}")


if __name__ == "__main__":
    main()
