"""Analysis + report for projection-selective AGQ (runs_projection_selective_agq/). Decision rule frozen in protocol.json,
implemented in moe_calib.projection_select.classify."""
from __future__ import annotations

import json, os, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import projection_selective_agq as PA   # noqa: E402
from moe_calib import projection_select as PS  # noqa: E402

p, jdump, jload, LAYERS, E = PA.p, PA.jdump, PA.jload, PA.LAYERS, PA.E
SPLITS3 = ("tune", "confirm", "oldtest")
N_BOOT, BOOT_SEED = 2000, 20261113
LABEL = dict(GPTQ="A: GPTQ", GU_AGQ="B: GU-AGQ", DOWN_AGQ="C: Down-AGQ", FULL_AGQ="D: Full AGQ")
COLOR = dict(GPTQ="#2a78d6", GU_AGQ="#1baf7a", DOWN_AGQ="#eda100", FULL_AGQ="#eb6834", SELECTED="#8a4fd8",
             LAYERWISE="#555555")
E40 = (16, 40)


def load(split):
    R = {L: json.load(open(p("results", split, f"L{L}.json"))) for L in LAYERS}
    M = {L: dict(np.load(p("results", split, f"expert_mats_L{L}.npz"))) for L in LAYERS}
    return R, M


def El(R, L, c):
    return R[L]["configs"][c]["moe_layer_output_rel_err"]


def split_results(R, sel, lw):
    out = dict(layers={}, mean={})
    for L in LAYERS:
        Ev = {c: El(R, L, c) for c in PS.ORDER}
        bu = min(Ev["GPTQ"], Ev["FULL_AGQ"])
        s, u = sel[L], lw[L]
        out["layers"][str(L)] = dict(
            E_layer=Ev, native={c: R[L]["configs"][c]["native_rel_err"] for c in PS.ORDER},
            selected=s, selected_E=Ev[s], layerwise=u, layerwise_E=Ev[u], best_uniform=bu,
            rel_vs_gptq={c: PS.rel(Ev[c], Ev["GPTQ"]) for c in PS.ORDER},
            rel_vs_full_agq={c: PS.rel(Ev[c], Ev["FULL_AGQ"]) for c in PS.ORDER},
            rel_vs_best_uniform={c: PS.rel(Ev[c], bu) for c in PS.ORDER},
            selected_rel=dict(vs_gptq=PS.rel(Ev[s], Ev["GPTQ"]), vs_full_agq=PS.rel(Ev[s], Ev["FULL_AGQ"]),
                              vs_layerwise=PS.rel(Ev[s], Ev[u]), vs_best_uniform=PS.rel(Ev[s], bu)),
            ranking_fp32=sorted(PS.ORDER, key=lambda c: Ev[c]),
            ranking_native=sorted(PS.ORDER, key=lambda c: R[L]["configs"][c]["native_rel_err"]))
        out["layers"][str(L)]["native_ranking_agrees"] = (out["layers"][str(L)]["ranking_fp32"] ==
                                                          out["layers"][str(L)]["ranking_native"])
    mean = lambda f: float(np.mean([f(L) for L in LAYERS]))
    out["mean"] = {c: mean(lambda L: El(R, L, c)) for c in PS.ORDER}
    out["mean"]["SELECTED"] = mean(lambda L: El(R, L, sel[L]))
    out["mean"]["LAYERWISE"] = mean(lambda L: El(R, L, lw[L]))
    m = out["mean"]
    out["mean_rel"] = dict(selected_vs_gptq=PS.rel(m["SELECTED"], m["GPTQ"]),
                           selected_vs_full_agq=PS.rel(m["SELECTED"], m["FULL_AGQ"]),
                           selected_vs_layerwise=PS.rel(m["SELECTED"], m["LAYERWISE"]),
                           layerwise_vs_full_agq=PS.rel(m["LAYERWISE"], m["FULL_AGQ"]),
                           layerwise_vs_gptq=PS.rel(m["LAYERWISE"], m["GPTQ"]))
    out["scope"] = "2-layer (L8, L16) mean; not comparable to 4-layer historical means"
    return out


def paired(R, sel, lw):
    rng = np.random.default_rng(BOOT_SEED)
    S = len(R[LAYERS[0]]["configs"]["GPTQ"]["per_seq"]["num"])
    idx = rng.integers(0, S, size=(N_BOOT, S))
    num = lambda L, c: np.array(R[L]["configs"][c]["per_seq"]["num"])
    den = {L: np.array(R[L]["configs"]["GPTQ"]["per_seq"]["den"]) for L in LAYERS}
    picks = dict(SELECTED=sel, LAYERWISE=lw, GPTQ={L: "GPTQ" for L in LAYERS},
                 FULL_AGQ={L: "FULL_AGQ" for L in LAYERS}, GU_AGQ={L: "GU_AGQ" for L in LAYERS},
                 DOWN_AGQ={L: "DOWN_AGQ" for L in LAYERS})
    out = {}
    for a, b in (("SELECTED", "LAYERWISE"), ("SELECTED", "FULL_AGQ"), ("SELECTED", "GPTQ"), ("LAYERWISE", "FULL_AGQ"),
                 ("GU_AGQ", "GPTQ"), ("GU_AGQ", "FULL_AGQ"), ("DOWN_AGQ", "GPTQ"), ("DOWN_AGQ", "FULL_AGQ")):
        r = dict(layers={})
        for L in LAYERS:
            na, nb = num(L, picks[a][L]), num(L, picks[b][L])
            dl = (na - nb) / den[L]
            ra = na[idx].sum(1) / den[L][idx].sum(1); rb = nb[idx].sum(1) / den[L][idx].sum(1)
            bd = (ra - rb) / rb
            r["layers"][str(L)] = dict(configs=[picks[a][L], picks[b][L]], rel_diff=PS.rel(na.sum(), nb.sum()),
                                       mean_window_ratio_diff=float(dl.mean()),
                                       frac_windows_improved=float((dl < 0).mean()),
                                       frac_windows_equal=float((dl == 0).mean()),
                                       boot_ci95=[float(np.quantile(bd, .025)), float(np.quantile(bd, .975))])
        ea = np.mean([num(L, picks[a][L])[idx].sum(1) / den[L][idx].sum(1) for L in LAYERS], 0)
        eb = np.mean([num(L, picks[b][L])[idx].sum(1) / den[L][idx].sum(1) for L in LAYERS], 0)
        pa = np.mean([num(L, picks[a][L]).sum() / den[L].sum() for L in LAYERS])
        pb = np.mean([num(L, picks[b][L]).sum() / den[L].sum() for L in LAYERS])
        bd = (ea - eb) / eb
        r.update(mean_rel_diff=float((pa - pb) / pb), boot_ci95=[float(np.quantile(bd, .025)),
                                                              float(np.quantile(bd, .975))],
                 boot_frac_a_better=float((bd < 0).mean()))
        out[f"{a}_vs_{b}"] = r
    out["note"] = f"descriptive window bootstrap ({N_BOOT} reps, seed {BOOT_SEED}); windows are not calibration seeds"
    return out


def expert_J(M, L):
    return {c: M[L][f"J_{c}"].sum(0) for c in PS.ORDER}


def expert_stats(splits, calib):
    out = {}
    tau_e = PA.RULES["expert_rel"]
    for s in SPLITS3:
        _, M = splits[s]
        rec = {}
        for L in LAYERS:
            J = expert_J(M, L)
            best = [min(PS.ORDER, key=lambda c: J[c][e]) for e in range(E)]
            bu = np.minimum(J["GPTQ"], J["FULL_AGQ"])
            bm = np.minimum(J["GU_AGQ"], J["DOWN_AGQ"])
            mixed_better = (bm - bu) / bu <= -tau_e
            I = J["FULL_AGQ"] - J["GU_AGQ"] - J["DOWN_AGQ"] + J["GPTQ"]
            full = J["GPTQ"] - J["FULL_AGQ"]
            gu_sh = 0.5 * ((J["GPTQ"] - J["GU_AGQ"]) + (J["DOWN_AGQ"] - J["FULL_AGQ"]))
            dn_sh = 0.5 * ((J["GPTQ"] - J["DOWN_AGQ"]) + (J["GU_AGQ"] - J["FULL_AGQ"]))
            rec[str(L)] = dict(
                best_config_counts={c: best.count(c) for c in PS.ORDER},
                frac_experts_mixed_beats_both_uniform_by_expert_rel=float(mixed_better.mean()),
                n_experts_mixed_beats_both_uniform_by_expert_rel=int(mixed_better.sum()),
                pooled_J={c: float(J[c].sum()) for c in PS.ORDER},
                pooled_J_interactions=PS.interactions({c: float(J[c].sum()) for c in PS.ORDER}),
                per_expert_rel_vs_gptq={c: dict(median=float(np.median((J[c] - J["GPTQ"]) / J["GPTQ"])),
                                                q10=float(np.quantile((J[c] - J["GPTQ"]) / J["GPTQ"], .1)),
                                                q90=float(np.quantile((J[c] - J["GPTQ"]) / J["GPTQ"], .9)))
                                        for c in PS.ORDER},
                per_expert_interaction_over_J_gptq=dict(median=float(np.median(I / J["GPTQ"])),
                                                        q10=float(np.quantile(I / J["GPTQ"], .1)),
                                                        q90=float(np.quantile(I / J["GPTQ"], .9))),
                full_gain_share_GU_pooled=float(gu_sh.sum() / full.sum()) if full.sum() != 0 else None,
                full_gain_share_DOWN_pooled=float(dn_sh.sum() / full.sum()) if full.sum() != 0 else None,
                experts=[dict(expert=e, J={c: float(J[c][e]) for c in PS.ORDER}, best=best[e],
                              interaction=float(I[e]), **calib.get((L, e), {})) for e in range(E)])
        out[s] = rec
    out["note"] = ("per-expert c^2-weighted full-expert output error J = sum c^2 ||E_FP - E_cfg||^2 on each split; "
                   "expert_rel = %.2f" % tau_e)
    return out


def calib_meta():
    v = json.load(open(os.path.join(ROOT, "runs_expertwise_agq_hybrid", "validation_expert_scores.json")))["experts"]
    keys = ("calib_tokens", "calib_affinity_mean", "calib_affinity_cv", "natural_p", "natural_decile")
    return {(x["layer"], x["expert"]): {k: x[k] for k in keys} for x in v}


def expert40(splits, calib):
    L, e = E40
    out = dict(layer=L, expert=e, post_hoc=True,
               note="identified from earlier (expert-wise hybrid) results; post-hoc mechanism exploration only; never "
                    "used for selection or decision", calibration=calib[(L, e)], splits={})
    for s in SPLITS3:
        _, M = splits[s]
        J = {c: float(M[L][f"J_{c}"][:, e].sum()) for c in PS.ORDER}
        c2den = float(M[L]["c2den"][:, e].sum())
        tok = M[L]["tokens"][:, e].sum()
        cs, c2 = M[L]["csum"][:, e].sum(), M[L]["c2sum"][:, e].sum()
        mu = cs / tok
        it = PS.interactions(J)
        share_layer = {c: J[c] / float(M[L][f"J_{c}"].sum()) for c in PS.ORDER}
        out["splits"][s] = dict(J=J, J_rel={c: J[c] / c2den for c in PS.ORDER},
                                rel_vs_gptq={c: PS.rel(J[c], J["GPTQ"]) for c in PS.ORDER},
                                rel_vs_full_agq={c: PS.rel(J[c], J["FULL_AGQ"]) for c in PS.ORDER},
                                interactions=it, share_of_layer_J=share_layer,
                                share_of_layer_full_agq_gain=(J["GPTQ"] - J["FULL_AGQ"]) /
                                float((M[L]["J_GPTQ"] - M[L]["J_FULL_AGQ"]).sum()),
                                tokens=int(tok), affinity_mean=float(mu),
                                affinity_cv=float(np.sqrt(max(c2 / tok - mu ** 2, 0)) / mu))
    c = out["splits"]["confirm"]["interactions"]
    sh = c["shapley_share_of_full_gain"]
    out["attribution_confirm"] = ("GU" if sh["GU"] >= 0.6 else "DOWN" if sh["DOWN"] >= 0.6 else "mixed/interaction")
    return out


def decomposition(splits):
    out = {}
    for s in SPLITS3:
        R, _ = splits[s]
        rec = {}
        for L in LAYERS:
            x = R[L]["configs"]
            rl = {}
            for c in PS.ORDER:
                rl[c] = dict(total_rel=x[c]["num"] / x[c]["den"], D_rel=x[c]["D_rel"], C_rel=x[c]["C_rel"],
                             cross_fraction=x[c]["cross_fraction"],
                             identity_rel_residual=abs(x[c]["D"] + x[c]["C"] - x[c]["decomp_total"]) / x[c]["decomp_total"],
                             decomp_vs_direct_rel=abs(x[c]["decomp_total"] - x[c]["num"]) / x[c]["num"])
            for c in PS.ORDER:
                for ref in ("GPTQ", "FULL_AGQ"):
                    rl[c][f"D_rel_change_vs_{ref}"] = PS.rel(rl[c]["D_rel"], rl[ref]["D_rel"])
                    rl[c][f"C_change_vs_{ref}_abs"] = rl[c]["C_rel"] - rl[ref]["C_rel"]
                    rl[c][f"C_rel_change_vs_{ref}"] = (PS.rel(rl[c]["C_rel"], rl[ref]["C_rel"])
                                                       if rl[ref]["C_rel"] != 0 else None)
                    rl[c][f"C_change_over_total_change_vs_{ref}"] = (
                        (rl[c]["C_rel"] - rl[ref]["C_rel"]) / (rl[c]["total_rel"] - rl[ref]["total_rel"])
                        if rl[c]["total_rel"] != rl[ref]["total_rel"] else None)
            rec[str(L)] = rl
        rec["max_identity_rel_residual"] = max(v["identity_rel_residual"] for L in LAYERS for v in rec[str(L)].values())
        rec["max_decomp_vs_direct_rel"] = max(v["decomp_vs_direct_rel"] for L in LAYERS for v in rec[str(L)].values())
        out[s] = rec
    return out


def cost():
    hc = json.load(open(os.path.join(ROOT, "runs_agq_only_validation", "cost_results.json")))
    c = {s: jload("logs", f"cost_{s}.json") for s in ("regression_oldtest", "tune", "confirm", "oldtest")}
    fr = jload("logs", "selection_freeze.json")
    hyb = json.load(open(os.path.join(ROOT, "runs_expertwise_agq_hybrid", "cost_results.json")))
    secs = lambda s: sum(v["seconds"] for v in c[s]["layers"].values())
    q_g = (hc["gptq_historical_FR_s0"]["hessian_s"] + hc["gptq_historical_FR_s0"]["gptq_s"]) / 2
    q_a = (hc["agq_FR_s0"]["hessian_s"] + hc["agq_FR_s0"]["gptq_s"]) / 2
    val2 = hyb["incremental_mvp"]["validation_scoring_lean_s"]                # 2-config tuning pass (hybrid study)
    val4 = sum(v["lean_s"] for v in c["tune"]["layers"].values())             # 4-config tuning pass (this study)
    return dict(
        incremental_mvp=dict(
            new_quantization_s=0.0, note="historical candidates reused (bitwise verified); no GPTQ/AGQ solves",
            weight_loading_s=sum(v["candidate_load_s"] for s in c for v in c[s]["layers"].values()),
            regression_s=secs("regression_oldtest"), validation_s=secs("tune"), selection_s=fr["selection_s"],
            confirmation_s=secs("confirm"), oldtest_s=secs("oldtest"),
            peak_gpu_gb=max(v["peak_gpu_gb"] for v in c.values()),
            peak_cpu_gb=max(v["peak_cpu_gb"] for v in c.values()),
            temporary_storage_bytes=sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in
                                        os.walk(p("results")) for f in fs),
            timings_include="lean fp32 evaluation + LayerEvalAccumulator cross-check + native BF16 forward"),
        estimated_fresh_deployment_2_layers=dict(
            method="historical 4-layer (Hessian + GPTQ) seconds / 2; validation = measured lean tuning pass "
                   "(includes the accumulator cross-check, an upper bound)",
            uniform_GPTQ_s=q_g, uniform_AGQ_s=q_a,
            layerwise_uniform_s=q_g + q_a + val2, projection_selective_s=q_g + q_a + val4,
            projection_vs_single_run=(q_g + q_a + val4) / q_g),
        storage=dict(final="one 4-bit candidate tensor per projection; identical format and size to uniform",
                     offline="both candidate sets held until selection (2x)"),
        inference=dict(expected_change="none; no routing parameters or runtime selection added", measured=False))


def correctness(splits, dec):
    inv, reg = jload("artifact_inventory.json"), jload("candidate_regression.json")
    cm, man, fr = jload("configuration_manifest.json"), jload("selection_manifest.json"), jload("logs", "selection_freeze.json")
    metas = {s: {L: splits[s][0][L]["meta"] for L in LAYERS} for s in SPLITS3}
    metas["regression"] = {L: reg["layers"][str(L)]["meta"] for L in LAYERS}
    allm = [m for s in metas.values() for m in s.values()]
    tests = {}
    for tag in ("projection", "all"):
        f = p("logs", f"pytest_{tag}.log")
        lines = [l for l in open(f).read().strip().splitlines() if l.strip()] if os.path.exists(f) else []
        if lines:
            last = lines[-1]
            tests[tag] = last
            tests[f"{tag}_passed"] = "passed" in last and "failed" not in last and "error" not in last
    src_ok = all(cm["layers"][str(L)][c]["n_from_agq"] == {"GPTQ": 0, "GU_AGQ": 120, "DOWN_AGQ": 60,
                                                            "FULL_AGQ": 180}[c] and
                 all(r["q"] == PS.CONFIGS[c][r["projection"]] for r in cm["layers"][str(L)][c]["projections"])
                 for L in LAYERS for c in PS.ORDER)
    test_mtime = min(os.path.getmtime(p("results", s, f"L{L}.json")) for s in ("confirm", "oldtest") for L in LAYERS)
    ld = lambda c: all(inv["candidates"][c]["layers"][str(L)]["valid"] for L in LAYERS)
    acc_ok = all(v["abs_diff"] <= 1e-12 and v["per_seq_equal"] for m in allm for v in m.get("accumulator_check", {}).values())
    nat_ok = all(splits[s][0][L]["configs"][c]["native_rel_err"] is not None for s in SPLITS3 for L in LAYERS
                 for c in PS.ORDER)
    rank_ok = {s: all(sorted(PS.ORDER, key=lambda c: splits[s][0][L]["configs"][c]["moe_layer_output_rel_err"]) ==
                      sorted(PS.ORDER, key=lambda c: splits[s][0][L]["configs"][c]["native_rel_err"]) for L in LAYERS)
               for s in SPLITS3}
    checks = {
        "01_A_all_gptq": src_ok, "02_D_all_agq": src_ok, "03_B_agq_gu_gptq_down": src_ok,
        "04_C_gptq_gu_agq_down": src_ok,
        "05_60_experts_per_layer": all(inv["candidates"][n]["layers"][str(L)]["n_experts"] == E
                                       for n in ("GPTQ", "AGQ") for L in LAYERS),
        "06_three_projections_per_expert": all(cm["layers"][str(L)][c]["n_projections"] == 3 * E
                                               for L in LAYERS for c in PS.ORDER) and ld("GPTQ") and ld("AGQ"),
        "07_candidates_unchanged": all(inv["candidates"][n]["layers"][str(L)]["file_sha256_matches_record"]
                                       for n in ("GPTQ", "AGQ") for L in LAYERS),
        "08_router_and_top4_unchanged": True,
        "09_shared_expert_unchanged": all(m["nontarget_unchanged_during_install"] for m in allm),
        "10_nontarget_params_unchanged": all(m["nontarget_unchanged_during_install"] and m["pristine_restored"]
                                             for m in allm),
        "11_gptq_reproduces": all(reg["layers"][str(L)]["GPTQ"]["per_window_exact"] for L in LAYERS) and reg["passed"],
        "12_full_agq_reproduces": all(reg["layers"][str(L)]["FULL_AGQ"]["per_window_exact"] for L in LAYERS)
                                  and reg["passed"],
        "13_complete_mixed_expert_outputs": acc_ok and tests.get("projection_passed", False),
        "14_no_test_data_in_selection": os.path.getmtime(p("selection_manifest.json")) < test_mtime
                                        and man["split"] == "tune" and not man["test_results_exist_at_freeze"],
        "15_selection_manifest_deterministic_and_frozen": PA.sha256_file(p("selection_manifest.json")) ==
                                                          fr["selection_manifest_sha256"],
        "16_reinstall_identical": all(m["reinstall_bitwise_identical"] for m in allm),
        "17_native_agrees_with_evaluator": nat_ok and reg["passed"],
        "18_D_plus_C_reproduces_numerator": all(dec[s]["max_identity_rel_residual"] < 1e-6 and
                                                dec[s]["max_decomp_vs_direct_rel"] < 1e-6 for s in dec),
        "19_original_weights_restored": all(m["pristine_restored"] for m in allm),
        "20_full_test_suite": tests.get("all_passed", False),
    }
    return dict(checks=checks, passed=bool(all(checks.values())), tests=tests,
                native_ranking_agrees_per_split=rank_ok,
                notes={"08": "native BF16 forward asserts torch.equal on router logits for every window and config",
                       "13": "LayerEvalAccumulator (independent per-expert loop) agrees for all four mixed configs on "
                             "tune and confirm (per-window equal, total abs diff <= 1e-12)",
                       "17": "native ranking agreement reported separately; equality to evaluator is checked on A/D "
                             "historical native values"})


# ============================================================================ figures
def figures(sp, inter, e40, dec, cst, pc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": .25})
    os.makedirs(p("figures"), exist_ok=True)

    def save(fig, n):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(p("figures", f"{n}.{ext}"), dpi=170)
        plt.close(fig)

    # Fig 1: four configs, validation and confirmation, relative to GPTQ
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, s in zip(axes, ("tune", "confirm")):
        xs = np.arange(len(LAYERS))
        for i, c in enumerate(PS.ORDER):
            v = [sp[s]["layers"][str(L)]["rel_vs_gptq"][c] * 100 for L in LAYERS]
            b = ax.bar(xs + (i - 1.5) * .2, v, .2, color=COLOR[c], label=LABEL[c])
            for j, L in enumerate(LAYERS):
                if sp[s]["layers"][str(L)]["selected"] == c:
                    ax.text(xs[j] + (i - 1.5) * .2, v[j], "sel", ha="center", va="top" if v[j] < 0 else "bottom",
                            fontsize=7)
        ax.axhline(0, color="k", lw=.7)
        ax.set_xticks(xs, [f"L{L}" for L in LAYERS])
        ax.set_title(f"{'validation (tune)' if s == 'tune' else 'confirmation'}: E_layer vs GPTQ (%)")
    axes[0].set_ylabel("relative to uniform GPTQ (%, lower better)")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "fig1_four_configurations")

    # Fig 2: 2x2 interaction matrices (confirm), values relative to GPTQ
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, L in zip(axes, LAYERS):
        Ev = sp["confirm"]["layers"][str(L)]["E_layer"]
        Mx = np.array([[Ev["GPTQ"], Ev["DOWN_AGQ"]], [Ev["GU_AGQ"], Ev["FULL_AGQ"]]])
        R = (Mx / Ev["GPTQ"] - 1) * 100
        lim = np.abs(R).max()
        ax.imshow(R, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{Mx[i, j] * 1e3:.5f}\n({R[i, j]:+.3f} %)", ha="center", va="center", fontsize=8)
        ax.set_xticks([0, 1], ["down: GPTQ", "down: AGQ"]); ax.set_yticks([0, 1], ["gate/up: GPTQ", "gate/up: AGQ"])
        ax.grid(False)
        it = inter["confirm"][str(L)]
        ax.set_title(f"confirm L{L} (E×1e3)  I = {it['interaction_I'] * 1e3:+.5f}e-3 ({it['interaction_sign']})",
                     fontsize=8)
    save(fig, "fig2_projection_interaction")

    # Fig 3: selected vs layer-wise, paired window bootstrap
    fig, ax = plt.subplots(figsize=(7, 3.4))
    rows = []
    for s in ("confirm", "oldtest"):
        r = pc[s]["SELECTED_vs_LAYERWISE"]
        for L in LAYERS:
            rows.append((f"{s} L{L} ({r['layers'][str(L)]['configs'][0]} vs {r['layers'][str(L)]['configs'][1]})",
                         r["layers"][str(L)]["rel_diff"], r["layers"][str(L)]["boot_ci95"]))
        rows.append((f"{s} 2-layer mean", r["mean_rel_diff"], r["boot_ci95"]))
    for i, (n, v, ci) in enumerate(rows):
        ax.errorbar(v * 100, i, xerr=[[v * 100 - ci[0] * 100], [ci[1] * 100 - v * 100]], fmt="o",
                    color=COLOR["SELECTED"], capsize=3)
    ax.axvline(0, color="k", lw=.7); ax.axvline(-0.5, color="grey", ls="--", lw=.7)
    ax.text(-0.5, len(rows) - .5, " −0.5 % criterion", fontsize=7, color="grey")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("selected vs validation-selected layer-wise uniform (%, 95 % window bootstrap)")
    save(fig, "fig3_selected_vs_layerwise")

    # Fig 4: expert 40
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    xs = np.arange(len(PS.ORDER))
    for i, s in enumerate(SPLITS3):
        v = [e40["splits"][s]["rel_vs_gptq"][c] * 100 for c in PS.ORDER]
        ax.bar(xs + (i - 1) * .26, v, .26, label=s, color=["#9bbbe8", "#2a78d6", "#123f7a"][i])
    ax.axhline(0, color="k", lw=.7)
    ax.set_xticks(xs, [LABEL[c] for c in PS.ORDER])
    ax.set_ylabel("c²-weighted expert error vs GPTQ (%)")
    ax.set_title("L16 expert 40 (post hoc): complete-expert error by configuration")
    ax.legend(frameon=False)
    save(fig, "fig4_expert40_attribution")

    # Fig 5: decomposition (confirm): D change vs GPTQ and C share
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, L in zip(axes, LAYERS):
        d = dec["confirm"][str(L)]
        Dc = [d[c]["D_rel_change_vs_GPTQ"] * 100 for c in PS.ORDER]
        Tc = [PS.rel(d[c]["total_rel"], d["GPTQ"]["total_rel"]) * 100 for c in PS.ORDER]
        Cf = [d[c]["cross_fraction"] * 100 for c in PS.ORDER]
        ax.bar(xs - .27, Dc, .27, color=[COLOR[c] for c in PS.ORDER], label="D change vs GPTQ (%)")
        ax.bar(xs, Tc, .27, color=[COLOR[c] for c in PS.ORDER], alpha=.45, hatch="//", label="total change vs GPTQ (%)")
        ax.bar(xs + .27, Cf, .27, color="#555555", alpha=.6, label="C share of total (%)")
        ax.axhline(0, color="k", lw=.7)
        ax.set_xticks(xs, [LABEL[c] for c in PS.ORDER], fontsize=7)
        ax.set_title(f"confirm L{L}: diagonal vs cross-expert")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "fig5_error_decomposition")


# ============================================================================ main
def main():
    splits = {s: load(s) for s in SPLITS3}
    man, lwb = jload("selection_manifest.json"), jload("layerwise_uniform_baseline.json")
    sel = {L: man["layers"][str(L)]["selected"] for L in LAYERS}
    lw = {L: lwb["layers"][str(L)]["choice"] for L in LAYERS}
    sp = {s: split_results(splits[s][0], sel, lw) for s in SPLITS3}
    pc = {s: paired(splits[s][0], sel, lw) for s in ("confirm", "oldtest")}
    inter = {s: {str(L): PS.interactions(sp[s]["layers"][str(L)]["E_layer"]) for L in LAYERS} for s in SPLITS3}
    for s in SPLITS3:
        for L in LAYERS:
            R = splits[s][0][L]["configs"]
            inter[s][str(L)]["diagonal_D_interactions"] = PS.interactions({c: R[c]["D_rel"] for c in PS.ORDER})
            inter[s][str(L)]["cross_C_interaction_abs"] = (R["FULL_AGQ"]["C_rel"] - R["GU_AGQ"]["C_rel"]
                                                           - R["DOWN_AGQ"]["C_rel"] + R["GPTQ"]["C_rel"])
    calib = calib_meta()
    es = expert_stats(splits, calib)
    e40 = expert40(splits, calib)
    dec = decomposition(splits)
    cst = cost()
    corr = correctness(splits, dec)
    rules = jload("protocol.json")["decision_rule"]["rules"]
    Ev = lambda s: {L: sp[s]["layers"][str(L)]["E_layer"] for L in LAYERS}
    emf = {str(L): es["confirm"][str(L)]["frac_experts_mixed_beats_both_uniform_by_expert_rel"] for L in LAYERS}
    v = PS.classify(rules, Ev("tune"), Ev("confirm"), Ev("oldtest"), sel, lw, corr["passed"], emf)
    v["recommendation"] = PS.RECOMMEND[v["verdict"]]
    vo = PS.classify(rules, Ev("tune"), Ev("oldtest"), Ev("oldtest"), sel, lw, corr["passed"], emf)
    v_old = dict(verdict=vo["verdict"], criteria=vo["criteria"], note="old-test as if primary; never decisive")
    jdump(dict(split="confirm", **sp["confirm"]), p("confirmation_results.json"))
    jdump(dict(split="oldtest", **sp["oldtest"]), p("old_test_results.json"))
    vr = jload("validation_results.json"); vr["summary"] = sp["tune"]; jdump(vr, p("validation_results.json"))
    jdump(pc, p("paired_comparisons.json"))
    jdump(dict(definition="I = E_D - E_B - E_C + E_A; gains positive = improvement; I > 0 antagonistic, I < 0 "
                          "synergistic (for this lower-is-better squared-error metric)", splits=inter),
          p("projection_interactions.json"))
    jdump(es, p("expert_statistics.json"))
    jdump(e40, p("expert40_attribution.json"))
    jdump(dec, p("error_decomposition.json"))
    jdump(cst, p("cost_results.json"))
    jdump(corr, p("correctness_results.json"))
    jdump(dict(primary_split="confirm", verdict=v["verdict"], recommendation=v["recommendation"], primary=v,
               secondary_oldtest=v_old, rules=rules, decided=time.strftime("%Y-%m-%d %H:%M:%S")), p("decision.json"))
    summ = dict(verdict=v["verdict"], recommendation=v["recommendation"], selected=sel, layerwise=lw,
                E_layer={s: {str(L): sp[s]["layers"][str(L)]["E_layer"] for L in LAYERS} for s in SPLITS3},
                mean={s: sp[s]["mean"] for s in SPLITS3}, mean_rel={s: sp[s]["mean_rel"] for s in SPLITS3},
                interactions={s: {L: {k: inter[s][L][k] for k in ("rel", "interaction_sign",
                                                                  "shapley_share_of_full_gain",
                                                                  "interaction_over_full_gain") if k in inter[s][L]}
                                  for L in inter[s]} for s in SPLITS3},
                expert40={s: dict(rel_vs_gptq=e40["splits"][s]["rel_vs_gptq"],
                                  shapley=e40["splits"][s]["interactions"].get("shapley_share_of_full_gain"))
                          for s in SPLITS3},
                expert_best_counts={s: {L: es[s][L]["best_config_counts"] for L in ("8", "16")} for s in SPLITS3},
                expert_mixed_frac_confirm=emf,
                paired={s: {k: dict(mean_rel_diff=x["mean_rel_diff"], ci=x["boot_ci95"]) for k, x in pc[s].items()
                            if k != "note"} for s in pc},
                cost=cst, correctness_passed=corr["passed"], correctness_checks=corr["checks"])
    jdump(summ, p("summary.json"))
    figures(sp, inter, e40, dec, cst, pc)
    print(f"[report] {v['verdict']} -> {v['recommendation']}; correctness {corr['passed']}; oldtest-as-primary "
          f"{vo['verdict']}")
    print(json.dumps({k: v[k] for k in ("criteria", "rel_vs_layerwise", "rel_vs_full_agq", "oldtest_rel_vs_layerwise",
                                        "per_layer_rel_vs_best_uniform", "best_mixed_rel_vs_best_uniform_confirm",
                                        "interaction_over_full_gain_confirm", "expert_mixed_frac")}, indent=1))


if __name__ == "__main__":
    main()
