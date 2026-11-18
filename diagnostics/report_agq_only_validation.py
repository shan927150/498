"""Aggregate the AGQ-only validation: metric JSONs, frozen decision, figures, summary."""
from __future__ import annotations

import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import agq_only_validation as AV     # noqa: E402
import full_depth_pilot as FD        # noqa: E402
from moe_calib import agq as AG      # noqa: E402

p, jdump, jload = AV.p, AV.jdump, AV.jload
LAYERS, E = AV.LAYERS, AV.E
KEYS = [(L, e) for L in LAYERS for e in range(E)]


def seeds_done():
    return [s for s in AV.SEEDS if os.path.exists(p("results", f"AGQ_FR_s{s}.json"))
            and os.path.exists(p("results", f"eval_FR_s{s}.json"))]


def rel(a, b):
    return AG.rel_diff(a, b)


def per_expert_vec(r):
    m = {(x["layer"], x["expert"]): x["mean"] for x in r["per_expert"]}
    return np.array([m[k] for k in KEYS])


def recon_block(v, dec):
    return dict(mean=float(v.mean()), median=float(np.median(v)), p90=float(np.quantile(v, .9)), worst=float(v.max()),
                worst_expert=list(KEYS[int(v.argmax())]),
                per_layer={str(L): float(v[i * E:(i + 1) * E].mean()) for i, L in enumerate(LAYERS)},
                by_natural_exposure_decile=[float(v[dec == d].mean()) for d in range(10)])


def compare(vg, va, dec):
    d = (va - vg) / vg
    return dict(gptq=recon_block(vg, dec), agq=recon_block(va, dec), rel_diff_mean=rel(va.mean(), vg.mean()),
                rel_diff_median=rel(np.median(va), np.median(vg)),
                per_layer_rel_diff={str(L): rel(va[i * E:(i + 1) * E].mean(), vg[i * E:(i + 1) * E].mean())
                                    for i, L in enumerate(LAYERS)},
                frac_experts_improved=float((va < vg).mean()), n_experts_improved=int((va < vg).sum()),
                per_expert_rel_diff_quantiles=dict(zip(["min", "q25", "median", "q75", "max"],
                                                       map(float, np.quantile(d, [0, .25, .5, .75, 1])))),
                by_natural_exposure_decile_rel_diff=[rel(va[dec == k].mean(), vg[dec == k].mean()) for k in range(10)])


def main():
    prot = jload("protocol.json")
    thr = prot["decision_rule"]
    pk, dec = AV.natural_exposure()
    seeds = seeds_done()
    names = [f"FR_s{s}" for s in seeds]
    unit = jload("unit_regression.json")
    gates = dict(source_review=os.path.exists(p("logs", "source_review.json")),
                 routing_definition=jload("router_affinity_definition.json")["verified"],
                 capture_alignment=jload("capture_alignment_results.json")["passed"],
                 hessian_oracle=jload("hessian_oracle_results.json")["passed"],
                 unit_regression=unit["passed"])
    diag = json.load(open(p("logs", "diagnostics_raw.json"))) if os.path.exists(p("logs", "diagnostics_raw.json")) else None
    if diag is not None:
        gates["diagnostics_recompute_equals_installed"] = diag["recomputed_weights_equal_installed"]

    base, agq, stdrec, wrec, moe, ppl, cost = {}, {}, {}, {}, {}, {}, {}
    bf16 = json.load(open(os.path.join(ROOT, "runs_partial_ppl", "bf16_baseline.json")))["ppl"]
    for n in names:
        rg, ra = AV.hist_result(n), json.load(open(p("results", f"AGQ_{n}.json")))
        ev = json.load(open(p("results", f"eval_{n}.json")))
        gates[f"AGQ_{n}_valid"] = bool(ra["valid"]) and ra["rtn_fallbacks"] == 0
        keep = ("ppl", "recon", "rtn_fallbacks", "n_quantized_experts", "n_projections", "installed_weights_sha256",
                "weights_dir", "valid", "quantize_total_s", "hessian_s", "gptq_s", "peak_gpu_gb", "peak_cpu_gb")
        base[n] = {k: rg[k] for k in keep}
        agq[n] = {k: ra[k] for k in keep}
        vg, va = per_expert_vec(rg), per_expert_vec(ra)
        stdrec[n] = compare(vg, va, dec)
        # weighted / unweighted full-expert and projection-level losses (held-out, all routed tokens)
        def vec(m, key):
            return np.concatenate([np.array(ev["layers"][str(L)][m][key], float) for L in LAYERS])
        w = {}
        for key in ("expert_weighted", "expert_unweighted", "proj_weighted_all", "proj_unweighted_all",
                    *[f"proj_weighted_{pj}" for pj in AG.PROJ], *[f"proj_unweighted_{pj}" for pj in AG.PROJ]):
            g_, a_ = vec("GPTQ", key), vec("AGQ", key)
            w[key] = compare(g_, a_, dec)
        wrec[n] = w
        lay = {}
        for L in LAYERS:
            s = ev["layers"][str(L)]
            g_, a_ = s["GPTQ"]["moe_layer_output_rel_err"], s["AGQ"]["moe_layer_output_rel_err"]
            gn, an = s["GPTQ"]["moe_layer_output_native_rel_err"], s["AGQ"]["moe_layer_output_native_rel_err"]
            ps_g = np.array(s["GPTQ"]["per_seq_moe_layer_output_rel_err"]); ps_a = np.array(s["AGQ"]["per_seq_moe_layer_output_rel_err"])
            lay[str(L)] = dict(gptq=g_, agq=a_, rel_diff=rel(a_, g_), native_gptq=gn, native_agq=an,
                               native_rel_diff=rel(an, gn), frac_heldout_seqs_improved=float((ps_a < ps_g).mean()),
                               per_seq_rel_diff_median=float(np.median((ps_a - ps_g) / ps_g)))
        mg = float(np.mean([lay[str(L)]["gptq"] for L in LAYERS])); ma = float(np.mean([lay[str(L)]["agq"] for L in LAYERS]))
        mgn = float(np.mean([lay[str(L)]["native_gptq"] for L in LAYERS])); man = float(np.mean([lay[str(L)]["native_agq"] for L in LAYERS]))
        moe[n] = dict(per_layer=lay, mean_gptq=mg, mean_agq=ma, rel_diff_mean=rel(ma, mg),
                      mean_of_layer_rel_diffs=float(np.mean([lay[str(L)]["rel_diff"] for L in LAYERS])),
                      layers_improved=int(sum(lay[str(L)]["rel_diff"] < 0 for L in LAYERS)),
                      native_mean_gptq=mgn, native_mean_agq=man, native_rel_diff_mean=rel(man, mgn))
        bt = FD.paired_bootstrap(np.array(ra["per_window_nll"]), np.array(rg["per_window_nll"]),
                                 np.array(ra["per_window_tokens"]))
        dw = np.array(ra["per_window_nll"]) / np.array(ra["per_window_tokens"]) - \
            np.array(rg["per_window_nll"]) / np.array(rg["per_window_tokens"])
        ppl[n] = dict(bf16=bf16, gptq=rg["ppl"], agq=ra["ppl"], dppl=ra["ppl"] - rg["ppl"],
                      gptq_minus_bf16=rg["ppl"] - bf16, agq_minus_bf16=ra["ppl"] - bf16,
                      mean_nll_diff_bootstrap=bt, frac_windows_agq_lower_nll=float((dw < 0).mean()),
                      per_window_mean_nll_diff=dw.tolist(),
                      class_=AG.classify_ppl(ra["ppl"] - rg["ppl"], thr["secondary"]["ppl_noise"]))

    # ---------------- cost (matched fresh GPTQ rerun vs AGQ, same process conditions)
    gr = json.load(open(p("logs", "result_GPTQ_rerun_FR_s0.json")))
    cost = dict(gptq_rerun_FR_s0=AV.cost_row(gr))
    for n in names:
        cost[f"agq_{n}"] = AV.cost_row(json.load(open(p("results", f"AGQ_{n}.json"))))
        cost[f"gptq_historical_{n}"] = AV.cost_row(AV.hist_result(n))
    a0 = cost.get("agq_FR_s0")
    if a0:
        cost["incremental_FR_s0"] = {k: a0[k] - cost["gptq_rerun_FR_s0"][k] for k in a0}
        cost["incremental_FR_s0_rel_total"] = cost["incremental_FR_s0"]["quantize_total_s"] / cost["gptq_rerun_FR_s0"]["quantize_total_s"]
    if diag is not None:
        cost["isolated_profiler_and_solver_s"] = diag["timing"]
    if os.path.exists(p("affinity_statistics.json")):
        af = jload("affinity_statistics.json")["configs"]
        cost["affinity_metadata"] = {k: dict(sparse_bytes=v["sparse_metadata_bytes"],
                                             dense_alternative_bytes=v["dense_alternative_bytes"],
                                             router_replay_s=v["seconds"]) for k, v in af.items()}
    cm = json.load(open(os.path.join(ROOT, AV.OV.OUT, "logs", "capture_manifest.json")))
    cost["shared_activation_capture_s"] = dict(seconds_total=cm["seconds_total"], rows=cm["n_rows"],
                                               note="CANONICAL-B1 capture of all calibration rows; shared by both "
                                                    "methods; affinities are computed in the same Profiler replay")
    cost["no_new_generation"] = True

    # ---------------- decision (frozen rule)
    correctness = all(gates.values())
    if names:
        p0 = moe.get(AV.PRIMARY)
        layer_rel = {L: p0["per_layer"][str(L)]["rel_diff"] for L in LAYERS} if p0 else {}
        wr = wrec[AV.PRIMARY]["expert_weighted"]["rel_diff_mean"] if p0 else 0.0
        seeds_rel = {n: moe[n]["rel_diff_mean"] for n in names}
        dec_ = AG.classify(dict(tau_rel=thr["tau_rel"], min_layers_improved=thr["min_layers_improved"]),
                           layer_rel, wr, correctness and bool(p0), seeds_rel,
                           primary_rel=p0["rel_diff_mean"] if p0 else None)
    else:
        dec_ = dict(verdict="AGQ-RUN-BLOCKED")
    tau = thr["secondary"]["tau_rel"]
    secondary = {n: dict(standard_reconstruction=AG.classify_secondary(stdrec[n]["rel_diff_mean"], tau),
                         weighted_reconstruction=AG.classify_secondary(wrec[n]["expert_weighted"]["rel_diff_mean"], tau),
                         ppl=ppl[n]["class_"],
                         systems_overhead_rel_total=(cost[f"agq_{n}"]["quantize_total_s"] /
                                                     cost["gptq_rerun_FR_s0"]["quantize_total_s"] - 1))
                 for n in names}
    decision = dict(primary=dec_, secondary=secondary, gates=gates, seeds_measured=names,
                    thresholds=thr, robust=bool(len(names) >= 3 and dec_.get("seeds_all_improve")))

    # ---------------- diagnostics summaries
    if diag is not None:
        H, Wd = summarize_diag(diag, pk, dec, wrec.get(AV.PRIMARY), stdrec.get(AV.PRIMARY))
        jdump(H, p("hessian_diagnostics.json")); jdump(Wd, p("weight_diagnostics.json"))

    jdump(dict(configs=base, note="C4R128_GPTQ = historical CANONICAL-B1 FR results (unit_regression reproduced FR_s0 "
                                  "bitwise with the modified code)"), p("baseline_results.json"))
    jdump(dict(configs=agq), p("agq_results.json"))
    jdump(dict(metric="E3 held-out relative reconstruction (unchanged)", configs=stdrec), p("reconstruction_results.json"))
    jdump(dict(metric="held-out affinity-weighted vs unweighted full-expert / projection errors (all routed tokens, "
                      "held-out routing coefficients)", configs=wrec), p("weighted_reconstruction_results.json"))
    jdump(dict(metric=prot["evaluation"]["moe_layer_output"], configs=moe), p("moe_layer_output_results.json"))
    jdump(dict(configs=ppl), p("ppl_results.json"))
    jdump(cost, p("cost_results.json"))
    jdump(decision, p("decision.json"))
    summ = summary(names, stdrec, wrec, moe, ppl, cost, decision)
    cp = p("logs", f"c2_probe_{AV.PRIMARY}.json")
    if os.path.exists(cp):
        c2 = json.load(open(cp))
        summ["exploratory_c2_probe"] = dict(
            note="POST HOC, not part of the frozen decision: MoE output error vs the c^2-weighted expert error",
            per_layer={L: dict(total_rel=v["rel_diff"]["total"], c2_weighted_rel=v["rel_diff"]["c2_weighted_expert_err"],
                               c1_pooled_rel=v["rel_diff"]["c1_weighted_expert_err_pooled"],
                               gptq_diag_fraction=v["GPTQ"]["diag_fraction"]) for L, v in c2["layers"].items()})
    jdump(summ, p("summary.json"))
    figures(names, stdrec, wrec, moe, ppl, cost, diag, pk, dec)
    write_md(summ, names, stdrec, wrec, moe, ppl, cost, decision)
    print("[report]", decision["primary"])


def summarize_diag(diag, pk, dec, w0, s0):
    ex = diag["experts"]
    def agg(sel):
        out = {}
        for pj in AG.PROJ:
            hs = [x["proj"][pj]["hessian"] for x in ex if sel(x)]
            out[pj] = {k: dict(mean=float(np.nanmean([h[k] for h in hs])), median=float(np.nanmedian([h[k] for h in hs])),
                               min=float(np.nanmin([h[k] for h in hs])), max=float(np.nanmax([h[k] for h in hs])))
                       for k in hs[0] if isinstance(hs[0][k], (int, float))}
        return out
    H = dict(config=AV.PRIMARY, overall=agg(lambda x: True),
             per_layer={str(L): agg(lambda x, L=L: x["layer"] == L) for L in LAYERS},
             per_expert=[dict(layer=x["layer"], expert=x["expert"], tokens=x["tokens"],
                              **{pj: x["proj"][pj]["hessian"] for pj in AG.PROJ}) for x in ex],
             note="rel_frob_raw includes pure global scaling (trace ratio ~ mean affinity); rel_frob_trace_norm and "
                  "shape_residual measure the change of relative sensitivity structure only")
    def wagg(sel):
        out = {}
        for pj in AG.PROJ:
            ws = [x["proj"][pj]["weights"] for x in ex if sel(x)]
            out[pj] = {k: dict(mean=float(np.mean([w[k] for w in ws])), max=float(np.max([w[k] for w in ws])))
                       for k in ws[0] if isinstance(ws[0][k], (int, float, bool))}
        return out
    Wd = dict(config=AV.PRIMARY, recomputed_weights_equal_installed=diag["recomputed_weights_equal_installed"],
              overall=wagg(lambda x: True), per_layer={str(L): wagg(lambda x, L=L: x["layer"] == L) for L in LAYERS},
              per_expert=[dict(layer=x["layer"], expert=x["expert"], **{pj: x["proj"][pj]["weights"] for pj in AG.PROJ})
                          for x in ex])
    # exploratory: affinity variability vs AGQ effect (descriptive only)
    if w0 is not None:
        cv = np.array([x["affinity"]["cv"] for x in ex]); nf = np.array([x["affinity"]["n_eff_frac"] for x in ex])
        mc = np.array([x["affinity"]["mean"] for x in ex])
        H["affinity_vs_effect"] = affinity_effect(ex, cv, nf, mc, pk)
    return H, Wd


def affinity_effect(ex, cv, nf, mc, pk):
    ev = json.load(open(p("results", f"eval_{AV.PRIMARY}.json")))
    def vec(m, key):
        return np.concatenate([np.array(ev["layers"][str(L)][m][key], float) for L in LAYERS])
    dw = (vec("AGQ", "expert_weighted") - vec("GPTQ", "expert_weighted")) / vec("GPTQ", "expert_weighted")
    du = (vec("AGQ", "expert_unweighted") - vec("GPTQ", "expert_unweighted")) / vec("GPTQ", "expert_unweighted")
    sp = lambda a, b: float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
    return dict(spearman_cv_vs_weighted_rel=sp(cv, dw), spearman_neff_frac_vs_weighted_rel=sp(nf, dw),
                spearman_mean_c_vs_weighted_rel=sp(mc, dw), spearman_natural_p_vs_mean_c=sp(pk, mc),
                spearman_natural_p_vs_cv=sp(pk, cv), spearman_cv_vs_unweighted_rel=sp(cv, du),
                note="240 experts are not independent replicates; descriptive only",
                per_expert=dict(cv=cv.tolist(), n_eff_frac=nf.tolist(), mean_c=mc.tolist(), weighted_rel=dw.tolist(),
                                unweighted_rel=du.tolist(), natural_p=pk.tolist()))


def summary(names, stdrec, wrec, moe, ppl, cost, decision):
    rows = {}
    for n in names:
        g0, a0 = cost["gptq_rerun_FR_s0"], cost[f"agq_{n}"]
        rows[n] = {
            "Standard expert reconstruction": (stdrec[n]["gptq"]["mean"], stdrec[n]["agq"]["mean"], stdrec[n]["rel_diff_mean"]),
            "Weighted expert reconstruction": (wrec[n]["expert_weighted"]["gptq"]["mean"], wrec[n]["expert_weighted"]["agq"]["mean"],
                                               wrec[n]["expert_weighted"]["rel_diff_mean"]),
            "MoE layer output reconstruction": (moe[n]["mean_gptq"], moe[n]["mean_agq"], moe[n]["rel_diff_mean"]),
            "MoE layer output (native BF16 block)": (moe[n]["native_mean_gptq"], moe[n]["native_mean_agq"], moe[n]["native_rel_diff_mean"]),
            "Partial-model WikiText-2 PPL": (ppl[n]["gptq"], ppl[n]["agq"], rel(ppl[n]["agq"], ppl[n]["gptq"])),
            "Quantization wall time (s)": (g0["quantize_total_s"], a0["quantize_total_s"], rel(a0["quantize_total_s"], g0["quantize_total_s"])),
            "Peak GPU memory (GB)": (g0["peak_gpu_gb"], a0["peak_gpu_gb"], rel(a0["peak_gpu_gb"], g0["peak_gpu_gb"])),
        }
    return dict(verdict=decision["primary"]["verdict"], decision=decision, table=rows)


def figures(names, stdrec, wrec, moe, ppl, cost, diag, pk, dec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(p("figures"), exist_ok=True)
    def save(fig, nm):
        fig.tight_layout(); fig.savefig(p("figures", nm + ".png"), dpi=140); fig.savefig(p("figures", nm + ".pdf")); plt.close(fig)
    if os.path.exists(p("affinity_statistics.json")):
        per = jload("affinity_statistics.json")["configs"][AV.PRIMARY]["per_expert"]
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for L in LAYERS:
            xs = [x for x in per if x["layer"] == L]
            ax[0].scatter([x["natural_p"] for x in xs], [x["mean"] for x in xs], s=10, label=f"L{L}")
            ax[1].scatter([x["natural_p"] for x in xs], [x["cv"] for x in xs], s=10, label=f"L{L}")
        ax[0].set(xlabel="natural routing frequency p", ylabel="mean affinity c", xscale="log", title="mean c vs exposure")
        ax[1].set(xlabel="natural routing frequency p", ylabel="CV of c", xscale="log", title="affinity variability")
        ax[0].legend(fontsize=7)
        ax[2].boxplot([[x["median"] for x in per if x["layer"] == L] for L in LAYERS], tick_labels=[f"L{L}" for L in LAYERS])
        ax[2].set(ylabel="per-expert median c", title="affinity by layer (FR s0 calibration)")
        save(fig, "fig1_affinity")
    if diag is not None:
        ex = diag["experts"]
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for i, pj in enumerate(("gate_proj", "down_proj")):
            ax[0].hist([x["proj"][pj]["hessian"]["rel_frob_trace_norm"] for x in ex], bins=30, alpha=.6, label=pj)
            ax[1].hist([x["proj"][pj]["hessian"]["diag_corr"] for x in ex], bins=30, alpha=.6, label=pj)
            ax[2].hist([x["proj"][pj]["hessian"]["actorder_frac_changed"] for x in ex], bins=30, alpha=.6, label=pj)
        ax[0].set(xlabel="||Ha/tr - Hs/tr|| / ||Hs/tr||", title="trace-normalized Hessian change"); ax[0].legend()
        ax[1].set(xlabel="corr(diag Hs, diag Ha)", title="diagonal correlation")
        ax[2].set(xlabel="fraction of act-order positions changed", title="act-order change")
        save(fig, "fig2_hessian")
    if names:
        n = AV.PRIMARY
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for i, (key, t) in enumerate((("expert_unweighted", "unweighted full-expert"), ("expert_weighted", "affinity-weighted full-expert"))):
            ev = json.load(open(p("results", f"eval_{n}.json")))
            G = np.concatenate([ev["layers"][str(L)]["GPTQ"][key] for L in LAYERS]); A = np.concatenate([ev["layers"][str(L)]["AGQ"][key] for L in LAYERS])
            ax[i].scatter(G, (A - G) / G * 100, s=8, c=dec, cmap="viridis")
            ax[i].axhline(0, color="k", lw=.8); ax[i].set(xscale="log", xlabel=f"GPTQ error ({t})", ylabel="AGQ vs GPTQ (%)", title=t)
        save(fig, "fig3_expert_reconstruction")
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.8 / max(1, len(names))
        for j, nm in enumerate(names):
            ax.bar(np.arange(4) + j * w, [moe[nm]["per_layer"][str(L)]["rel_diff"] * 100 for L in LAYERS], w, label=nm)
        ax.axhline(0, color="k", lw=.8); ax.axhline(-0.5, color="gray", ls="--", lw=.8); ax.axhline(0.5, color="gray", ls="--", lw=.8)
        ax.set_xticks(np.arange(4) + w * (len(names) - 1) / 2, [f"L{L}" for L in LAYERS])
        ax.set(ylabel="MoE layer output error, AGQ vs GPTQ (%)", title="Aggregated MoE output (frozen BF16 routing)"); ax.legend()
        save(fig, "fig4_moe_output")
        if diag is not None:
            H = jload("hessian_diagnostics.json").get("affinity_vs_effect")
            if H:
                pe = H["per_expert"]
                fig, ax = plt.subplots(1, 2, figsize=(11, 4))
                ax[0].scatter(pe["cv"], np.array(pe["weighted_rel"]) * 100, s=8); ax[0].axhline(0, color="k", lw=.8)
                ax[0].set(xlabel="CV of calibration affinity", ylabel="weighted expert error, AGQ vs GPTQ (%)")
                ax[1].scatter(pe["n_eff_frac"], np.array(pe["weighted_rel"]) * 100, s=8); ax[1].axhline(0, color="k", lw=.8)
                ax[1].set(xlabel="N_eff / N", ylabel="weighted expert error, AGQ vs GPTQ (%)")
                save(fig, "fig5_affinity_variability")
        g0, a0 = cost["gptq_rerun_FR_s0"], cost.get("agq_FR_s0")
        if a0:
            fig, ax = plt.subplots(figsize=(6, 4))
            for i, (lab, c) in enumerate((("GPTQ", g0), ("AGQ", a0))):
                b = 0
                for k, col in (("hessian_s", "C0"), ("gptq_s", "C1"), ("other_s", "C2")):
                    ax.bar(i, c[k], bottom=b, color=col, label=k if i == 0 else None); b += c[k]
            ax.set_xticks([0, 1], ["GPTQ (rerun)", "AGQ"]); ax.set(ylabel="seconds (FR s0, 240 experts)",
                                                                   title="Quantization wall time (incl. E3 scoring)")
            ax.legend(); save(fig, "fig6_runtime")
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for nm in names:
                ax.scatter(cost[f"agq_{nm}"]["quantize_total_s"], moe[nm]["rel_diff_mean"] * 100, label=f"MoE out {nm}")
                ax.scatter(cost[f"agq_{nm}"]["quantize_total_s"], wrec[nm]["expert_weighted"]["rel_diff_mean"] * 100, marker="x", label=f"weighted {nm}")
            ax.scatter(g0["quantize_total_s"], 0, c="k", marker="s", label="GPTQ")
            ax.axhline(0, color="k", lw=.8); ax.set(xlabel="quantization wall time (s)", ylabel="error vs GPTQ (%)",
                                                    title="Quality-cost"); ax.legend(fontsize=7)
            save(fig, "fig7_quality_cost")


def fmt(x):
    return f"{x:.6g}"


def write_md(summ, names, stdrec, wrec, moe, ppl, cost, decision):
    L = ["# AGQ-only validation: standard GPTQ vs AGQ-GPTQ on matched C4 Random B=128", "",
         f"**Primary verdict (frozen rule): `{summ['verdict']}`**", "",
         f"Seeds measured: {', '.join(names) or 'none'}. Gates: " +
         ", ".join(f"{k}={v}" for k, v in decision["gates"].items()), ""]
    for n in names:
        L += [f"## {n}", "", "| Metric | GPTQ | AGQ | Relative difference |", "|---|---|---|---|"]
        for k, (g, a, r) in summ["table"][n].items():
            L.append(f"| {k} | {fmt(g)} | {fmt(a)} | {r * 100:+.3f} % |")
        L += ["", "Per-layer MoE output error (fp32, frozen routing): " + ", ".join(
            f"L{l} {moe[n]['per_layer'][str(l)]['rel_diff'] * 100:+.3f} %" for l in LAYERS),
            f"Experts improved (standard / weighted): {stdrec[n]['n_experts_improved']}/240 / "
            f"{wrec[n]['expert_weighted']['n_experts_improved']}/240", ""]
    open(p("summary.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
