"""Report for the frozen-routing factorial ablation: natural regression, Hessian / act-order / weight / reconstruction
comparisons, PPL factorial contrasts, FR-vs-FC ranking, the frozen decision, figures 1-6 and summary.md."""
from __future__ import annotations
import json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import frozen_routing as XF            # noqa: E402
import full_depth_pilot as FD          # noqa: E402
from moe_calib import frozen_routing as FRT   # noqa: E402

P, STRATS, LAYERS = XF.p, list(XF.STRATS), XF.LAYERS
CONDS = ["OO", "AO", "OA", "AA"]
CCOL = {"OO": "#52514e", "AO": "#2a78d6", "OA": "#eb6834", "AA": "#1baf7a"}
SCOL = {"FR": "#2a78d6", "FC": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
CLABEL = {"OO": "OO (X_O, R_O)", "AO": "AO (X_A, R_O)", "OA": "OA (X_O, R_A)", "AA": "AA (X_A, R_A)"}


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def build():
    pr = XF.protocol()
    rv = load(P("routing_validation.json"))
    res = {c: {k: load(P("results", f"{c}_{k}.json")) for k in CONDS} for c in STRATS}
    missing = [f"{c}_{k}" for c in STRATS for k in CONDS if res[c][k] is None]
    invalid = [f"{c}_{k}" for c in STRATS for k in CONDS if res[c][k] is not None and not res[c][k]["valid"]]
    reg_fail = [f"{c}_{k}" for c in STRATS for k in FRT.NATURAL if res[c][k] is not None and not res[c][k]["regression_passed"]]
    blocked = bool(missing or invalid or reg_fail or not (rv and rv["valid"]))
    natural = {f"{c}_{k}": dict(res[c][k]["regression"], passed=res[c][k]["regression_passed"])
               for c in STRATS for k in FRT.NATURAL if res[c][k] is not None}
    XF.jdump(dict(passed=not reg_fail and len(natural) == 4, configs=natural), P("natural_regression.json"))
    if blocked:
        return dict(protocol_sha256=pr["protocol_sha256"], decision="RUN-BLOCKED", missing=missing, invalid=invalid,
                    regression_failed=reg_fail, routing_valid=bool(rv and rv["valid"])), None

    # ---------------- Hessians (diagnostic subset: full matrices; all 480: act-order + token counts)
    H = {c: {k: torch.load(res[c][k]["hessian_file"]) for k in CONDS} for c in STRATS}
    for c in STRATS:
        for k in CONDS:
            assert XF.sha256_file(res[c][k]["hessian_file"]) == res[c][k]["hessian_file_sha256"]
    pairs = [("AO", "OO"), ("OA", "OO"), ("AA", "OO"), ("AA", "AO"), ("AA", "OA")]
    hcmp, act, tokc = {}, {}, {}
    for c in STRATS:
        hcmp[c], act[c], tokc[c] = {}, {}, {}
        for a, b in pairs:
            key = f"{a}_vs_{b}"
            hcmp[c][key] = {kk: FRT.hessian_metrics(H[c][a]["full"][kk], H[c][b]["full"][kk], H[c][a]["tokens"][kk],
                                                    H[c][b]["tokens"][kk]) for kk in H[c][b]["full"]}
            pm = {kk: FRT.perm_metrics(H[c][a]["perm"][kk], H[c][b]["perm"][kk]) for kk in H[c][b]["perm"]}
            act[c][key] = dict(
                n_hessians=len(pm), n_identical_perm=sum(v["identical"] for v in pm.values()),
                mean_frac_positions_changed=float(np.mean([v["frac_positions_changed"] for v in pm.values()])),
                mean_frac_columns_changed_group=float(np.mean([v["frac_columns_changed_group"] for v in pm.values()])),
                by_proj={pj: float(np.mean([v["frac_columns_changed_group"] for kk, v in pm.items() if kk.endswith(pj)]))
                         for pj in ("gu", "down")},
                diag_subset={kk: pm[kk] for kk in H[c][b]["full"]})
            ta = np.array([H[c][a]["tokens"][kk] for kk in H[c][b]["tokens"] if kk.endswith("gu")])
            tb = np.array([H[c][b]["tokens"][kk] for kk in H[c][b]["tokens"] if kk.endswith("gu")])
            tokc[c][key] = dict(n_experts_count_changed=int((ta != tb).sum()),
                                mean_abs_rel_count_change=float(np.mean(np.abs(ta - tb) / np.maximum(tb, 1))),
                                max_abs_rel_count_change=float(np.max(np.abs(ta - tb) / np.maximum(tb, 1))),
                                total_tokens=[float(ta.sum()), float(tb.sum())])
    hsum = {c: {key: {m: FRT.summarize([v[m] for v in hcmp[c][key].values()])
                      for m in ("rel_fro_trace_normalized", "rel_fro_raw", "diag_rel_diff", "trace_ratio")}
                for key in hcmp[c]} for c in STRATS}
    hproj = {c: {key: {pj: FRT.summarize([v["rel_fro_trace_normalized"] for kk, v in hcmp[c][key].items() if kk.endswith(pj)])
                       for pj in ("gu", "down")} for key in hcmp[c]} for c in STRATS}
    XF.jdump(dict(subset=pr["diagnostic_subset"], summary=hsum, by_projection=hproj, per_hessian=hcmp,
                  actorder=act, token_counts=tokc,
                  note="trace-normalized relative Frobenius is the decision metric (GPTQ with relative damping is "
                       "invariant to the scale of H); raw = the accumulated unnormalized sums"), P("hessian_comparison.json"))

    # ---------------- effective weights (all 720 tensors; BF16 installed)
    wcmp = {}
    for c in STRATS:
        wcmp[c] = {f"{a}_vs_{b}": dict(per_layer={}, per_expert_rel_l2=[]) for a, b in pairs}
        for L in LAYERS:
            W = {k: torch.load(os.path.join(XF.weights_dir(c, k), f"L{L}.pt")) for k in CONDS}
            for a, b in pairs:
                key = f"{a}_vs_{b}"
                wcmp[c][key]["per_layer"][str(L)] = FRT.weight_diff(W[a], W[b])
                for e in range(60):
                    ks = [f"{e}.{pj}" for pj in ("gate_proj", "up_proj", "down_proj")]
                    d = FRT.weight_diff({kk: W[a][kk] for kk in ks}, {kk: W[b][kk] for kk in ks})
                    wcmp[c][key]["per_expert_rel_l2"].append(d["rel_l2"])
            del W
        for key, v in wcmp[c].items():
            pl = v["per_layer"].values()
            n, dif = sum(x["n"] for x in pl), sum(x["n_differ"] for x in pl)
            d2, o2 = sum(x["d2"] for x in pl), sum(x["o2"] for x in pl)
            v["total"] = dict(frac_differ=dif / n, rel_l2=(d2 / o2) ** .5, n_tensors_identical=sum(x["n_tensors_identical"] for x in pl),
                              n_tensors=sum(x["n_tensors"] for x in pl))
            v["per_expert_summary"] = FRT.summarize(v["per_expert_rel_l2"])
    XF.jdump(wcmp, P("weight_comparison.json"))

    # ---------------- reconstruction
    rec = {c: {k: dict(recon=res[c][k]["recon"], recon_per_layer=res[c][k]["recon_per_layer"]) for k in CONDS} for c in STRATS}
    for c in STRATS:
        ref = {(r["layer"], r["expert"]): r["mean"] for r in res[c]["OO"]["per_expert"]}
        for k in CONDS:
            ch = np.array([(r["mean"] - ref[(r["layer"], r["expert"])]) / ref[(r["layer"], r["expert"])]
                           for r in res[c][k]["per_expert"]])
            rec[c][k].update(rel_change_vs_OO=(res[c][k]["recon"] - res[c]["OO"]["recon"]) / res[c]["OO"]["recon"],
                             per_expert_abs_rel_change_median=float(np.median(np.abs(ch))),
                             per_expert_abs_rel_change_max=float(np.max(np.abs(ch))))
    rec["FC_minus_FR_rel"] = {k: (rec["FC"][k]["recon"] - rec["FR"][k]["recon"]) / rec["FR"][k]["recon"] for k in CONDS}
    XF.jdump(rec, P("reconstruction_results.json"))

    # ---------------- PPL + factorial contrasts
    ppl = {c: {k: dict(ppl=res[c][k]["ppl"], mean_nll=res[c][k]["mean_nll"], total_nll=res[c][k]["total_nll"],
                       total_tokens=res[c][k]["total_tokens"], per_window_nll=res[c][k]["per_window_nll"],
                       per_window_tokens=res[c][k]["per_window_tokens"],
                       source="natural reference (bitwise-reproduced weights)" if k in FRT.NATURAL else "evaluated")
               for k in CONDS} for c in STRATS}
    XF.jdump(dict(bf16_ppl=pr["evaluation"]["bf16_ppl"], configs=ppl), P("ppl_results.json"))
    fac = {}
    for c in STRATS:
        y = {k: ppl[c][k]["ppl"] for k in CONDS}
        nl = {k: ppl[c][k]["mean_nll"] for k in CONDS}
        fac[c] = dict(ppl=FRT.factorial_contrasts(y["OO"], y["AO"], y["OA"], y["AA"]),
                      mean_nll=FRT.factorial_contrasts(nl["OO"], nl["AO"], nl["OA"], nl["AA"]))
        bt = {}
        for k in ("AO", "OA", "AA"):
            bt[f"{k}_vs_OO"] = FD.paired_bootstrap(ppl[c][k]["per_window_nll"], ppl[c]["OO"]["per_window_nll"],
                                                   ppl[c]["OO"]["per_window_tokens"])
        fac[c]["window_bootstrap_mean_nll_vs_OO"] = bt
    delta = {k: ppl["FC"][k]["ppl"] - ppl["FR"][k]["ppl"] for k in CONDS}
    dboot = {k: FD.paired_bootstrap(ppl["FC"][k]["per_window_nll"], ppl["FR"][k]["per_window_nll"],
                                    ppl["FR"][k]["per_window_tokens"]) for k in CONDS}
    dfac = FRT.factorial_contrasts(delta["OO"], delta["AO"], delta["OA"], delta["AA"])
    XF.jdump(dict(per_strategy=fac, delta_fc_minus_fr=delta, delta_window_bootstrap=dboot, delta_factorial=dfac,
                  tau=FRT.TAU), P("factorial_contrasts.json"))

    # ---------------- decision (frozen rule)
    hm = {c: {k: hsum[c][f"{k}_vs_OO"]["rel_fro_trace_normalized"]["median"] for k in ("AO", "OA", "AA")} for c in STRATS}
    wm = {c: {k: wcmp[c][f"{k}_vs_OO"]["total"]["rel_l2"] for k in ("AO", "OA", "AA")} for c in STRATS}
    fl = {c: FRT.flags(hm[c], wm[c], {k: ppl[c][k]["ppl"] for k in CONDS}) for c in STRATS}
    decision = FRT.decide(fl, delta)
    rs = {str(L): {c: {k: res[c][k]["forced_routing_stats"][str(L)] for k in CONDS} for c in STRATS} for L in LAYERS}
    out = dict(protocol_sha256=pr["protocol_sha256"], decision=decision, decision_rule=FRT.DECISION_RULE,
               flags=fl, ranking_preserved=FRT.ranking_preserved(delta), h_metric=hm, w_metric=wm,
               ppl={c: {k: ppl[c][k]["ppl"] for k in CONDS} for c in STRATS},
               mean_nll={c: {k: ppl[c][k]["mean_nll"] for k in CONDS} for c in STRATS},
               factorial=fac, delta_fc_minus_fr=delta, delta_factorial=dfac, delta_window_bootstrap=dboot,
               hessian_summary=hsum, actorder={c: {k: {kk: v for kk, v in d.items() if kk != "diag_subset"}
                                                   for k, d in act[c].items()} for c in STRATS},
               token_counts=tokc, weights={c: {k: v["total"] | dict(per_expert=v["per_expert_summary"])
                                                for k, v in wcmp[c].items()} for c in STRATS},
               reconstruction=rec, routing_capture={L: {kk: v for kk, v in rv["per_layer"][L]["A_vs_O"].items()}
                                                    for L in rv["per_layer"]},
               forced_routing_stats=rs, natural_regression=natural,
               compute={f"{c}_{k}": dict(quantize_s=res[c][k]["quantize_s"], hessian_s=res[c][k]["hessian_s"],
                                         gptq_s=res[c][k]["gptq_s"], eval_s=res[c][k].get("eval_s"),
                                         reference_profiler_s=res[c][k]["reference_profiler_s"],
                                         peak_gpu_gb=res[c][k]["peak_gpu_gb"], peak_cpu_gb=res[c][k]["peak_cpu_gb"])
                        for c in STRATS for k in CONDS}, capture_s=rv["capture_s"])
    raw = dict(hcmp=hcmp, hsum=hsum, wcmp=wcmp, rec=rec, ppl=ppl, fac=fac, delta=delta, dboot=dboot)
    return out, raw


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
    hc, wc, ppl, fac, delta = raw["hcmp"], raw["wcmp"], raw["ppl"], raw["fac"], raw["delta"]
    comp = ["AO", "OA", "AA"]

    # 1 Hessian sensitivity vs OO (16 diagnostic Hessians, each point one Hessian)
    f, axs = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, c in zip(axs, STRATS):
        for i, k in enumerate(comp):
            v = hc[c][f"{k}_vs_OO"]
            for j, (kk, m) in enumerate(sorted(v.items())):
                ax.scatter(i + (0.12 if kk.endswith("down") else -0.12), m["rel_fro_trace_normalized"], color=CCOL[k],
                           marker="s" if kk.endswith("down") else "o", s=22, edgecolor="white", lw=.6, zorder=3,
                           label=("down" if kk.endswith("down") else "gate/up") if (i == 0 and j < 2) else None)
            ax.hlines(o["h_metric"][c][k], i - .3, i + .3, color=CCOL[k], lw=2.2)
        ax.set_xticks(range(3), [CLABEL[k] for k in comp], fontsize=7.5)
        ax.set_title(c, loc="left")
        ax.set_yscale("log")
    axs[0].set_ylabel("rel. Frobenius diff. from OO\n(trace-normalized Hessian)")
    axs[0].legend(frameon=False, fontsize=7.5)
    f.suptitle("Fig 1 — Hessian sensitivity (layers 1/8/16/23 × experts 0/30; bar = median)", x=.02, ha="left", fontsize=9.5)
    sv(f, "fig1_hessian_factorial")

    # 2 weight sensitivity vs OO per layer
    f, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    xw = np.arange(len(LAYERS))
    for ax, c in zip(axs, STRATS):
        for i, k in enumerate(comp):
            ax.bar(xw + (i - 1) * .26, [wc[c][f"{k}_vs_OO"]["per_layer"][str(L)]["rel_l2"] for L in LAYERS], .26,
                   color=CCOL[k], label=CLABEL[k])
        ax.set_xticks(xw, [f"L{L}" for L in LAYERS]); ax.set_title(c, loc="left")
    axs[0].set_ylabel("rel. L2 of effective weights vs OO")
    axs[1].legend(frameon=False, fontsize=7.5)
    f.suptitle("Fig 2 — GPTQ effective-weight sensitivity", x=.02, ha="left", fontsize=9.5)
    sv(f, "fig2_weight_factorial")

    # 3 PPL of the 4 conditions per strategy
    f, ax = plt.subplots(figsize=(6.0, 3.4))
    for c in STRATS:
        ax.plot(range(4), [ppl[c][k]["ppl"] for k in CONDS], color=SCOL[c], marker="o", lw=1.4, label=c)
        for i, k in enumerate(CONDS):
            ax.annotate(f"{ppl[c][k]['ppl']:.4f}", (i, ppl[c][k]["ppl"]), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7, color=SCOL[c])
    ax.set_xticks(range(4), [CLABEL[k] for k in CONDS], fontsize=7.5); ax.set_ylabel("WikiText-2 PPL")
    ax.legend(frameon=False)
    ax.set_title("Fig 3 — PPL under the four calibration conditions", loc="left", fontsize=9.5)
    sv(f, "fig3_ppl_factorial")

    # 4 factorial effects
    names = ["A_effect_R0", "R_effect_X0", "A_effect_R1", "R_effect_X1", "interaction", "natural_effect"]
    lab = ["act | R_O\n(AO−OO)", "route | X_O\n(OA−OO)", "act | R_A\n(AA−OA)", "route | X_A\n(AA−AO)", "interaction",
           "natural\n(AA−OO)"]
    f, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(names))
    for i, c in enumerate(STRATS):
        ax.bar(x + (i - .5) * .36, [fac[c]["ppl"][n] for n in names], .36, color=SCOL[c], label=c)
    ax.axhline(0, color=INK, lw=.8)
    ax.axhspan(-FRT.TAU, FRT.TAU, color=GRID, alpha=.6, lw=0, label="± deterministic tolerance τ")
    ax.set_xticks(x, lab, fontsize=7.5); ax.set_ylabel("Δ PPL"); ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Fig 4 — Factorial PPL effects and interaction", loc="left", fontsize=9.5)
    sv(f, "fig4_factorial_effects")

    # 5 FR vs FC ranking
    f, ax = plt.subplots(figsize=(5.6, 3.2))
    for i, k in enumerate(CONDS):
        ci = raw["dboot"][k]["ci95"]
        ax.bar(i, delta[k], .6, color=CCOL[k])
        ax.annotate(f"{delta[k]:+.4f}", (i, delta[k]), textcoords="offset points",
                    xytext=(0, 4 if delta[k] >= 0 else -11), ha="center", fontsize=7.5)
    ax.axhline(0, color=INK, lw=.8)
    ax.set_xticks(range(4), [CLABEL[k] for k in CONDS], fontsize=7.5)
    ax.set_ylabel("Δ_C = PPL(FC) − PPL(FR)")
    ax.set_title("Fig 5 — FR vs FC ranking per condition (positive: FC worse)", loc="left", fontsize=9.5)
    sv(f, "fig5_ranking")

    # 6 reconstruction change vs PPL change (vs OO)
    f, ax = plt.subplots(figsize=(5.6, 3.6))
    for c in STRATS:
        for k in comp:
            xr = raw["rec"][c][k]["rel_change_vs_OO"] * 100
            yp = ppl[c][k]["ppl"] - ppl[c]["OO"]["ppl"]
            ax.scatter(xr, yp, color=CCOL[k], marker="o" if c == "FR" else "s", s=40, edgecolor="white", lw=1, zorder=3)
            ax.annotate(f"{c} {k}", (xr, yp), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.axhline(0, color=INK, lw=.8); ax.axvline(0, color=INK, lw=.8)
    ax.set_xlabel("mean reconstruction change vs OO (%)"); ax.set_ylabel("PPL change vs OO")
    ax.set_title("Fig 6 — Local reconstruction vs model-level PPL (○ FR, □ FC)", loc="left", fontsize=9.5)
    sv(f, "fig6_recon_vs_ppl")


def markdown(o):
    L = ["# Frozen-routing 2×2 factorial ablation (FR s0 / FC s0, four layers, ORIGINAL vs G20261111)", "",
         f"**Decision (rule frozen in protocol.json, sha256 `{o['protocol_sha256'][:16]}`): {o['decision']}**", "",
         "| strategy | condition | activation | routing | PPL | mean NLL | recon | h (median Hessian diff vs OO) | "
         "w (weight rel. L2 vs OO) |", "|---|---|---|---|---|---|---|---|---|"]
    for c in STRATS:
        for k in CONDS:
            a, r = FRT.CONDITIONS[k]
            h = o["h_metric"][c].get(k, 0.0)
            w = o["w_metric"][c].get(k, 0.0)
            L.append(f"| {c} | {k} | {a} | {r} | {o['ppl'][c][k]:.6f} | {o['mean_nll'][c][k]:.7f} | "
                     f"{o['reconstruction'][c][k]['recon']:.7f} | {h:.4f} | {w:.4f} |")
    L += ["", "| strategy | AO−OO | OA−OO | AA−OA | AA−AO | interaction | AA−OO |", "|---|---|---|---|---|---|---|"]
    for c in STRATS:
        f = o["factorial"][c]["ppl"]
        L.append(f"| {c} | {f['A_effect_R0']:+.6f} | {f['R_effect_X0']:+.6f} | {f['A_effect_R1']:+.6f} | "
                 f"{f['R_effect_X1']:+.6f} | {f['interaction']:+.6f} | {f['natural_effect']:+.6f} |")
    L += ["", "| condition | Δ = FC − FR | window-bootstrap 95% CI (mean NLL) |", "|---|---|---|"]
    for k in CONDS:
        ci = o["delta_window_bootstrap"][k]["ci95"]
        L.append(f"| {k} | {o['delta_fc_minus_fr'][k]:+.6f} | [{ci[0]:+.5f}, {ci[1]:+.5f}] |")
    L += ["", f"- ranking preserved (|Δ_AO − Δ_OO| ≤ 0.5 |Δ_AA − Δ_OO|): {o['ranking_preserved']}",
          f"- flags: " + "; ".join(f"{c}: " + ", ".join(f"{k}={v}" for k, v in o['flags'][c].items()
                                                         if isinstance(v, bool)) for c in STRATS),
          f"- natural regression: " + ", ".join(f"{k} {'passed' if v['passed'] else 'FAILED'} (Hessians "
                                                 f"{v['hessians_bitwise_identical']})" for k, v in o['natural_regression'].items()),
          ""]
    return "\n".join(L) + "\n"


def main():
    out, raw = build()
    XF.jdump(out, P("summary.json"))
    if raw is None:
        print(f"[report] RUN-BLOCKED: {out}"); return
    open(P("summary.md"), "w").write(markdown(out))
    figures(out, raw)
    print(f"[report] decision {out['decision']}; Δ " + " ".join(f"{k} {v:+.5f}" for k, v in out["delta_fc_minus_fr"].items()))


if __name__ == "__main__":
    main()
