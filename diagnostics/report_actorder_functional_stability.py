"""Analysis, deliverable JSONs, figures and summary for runs_actorder_functional_stability."""
from __future__ import annotations
import itertools, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import actorder_functional_stability as S                                  # noqa: E402

PROJ, UNITS = S.PROJ, S.UNITS
GROUPS, CONFIGS = S.GROUPS, S.CONFIGS
pk = S.pk


def jl(*a):
    return S.jload(S.p(*a))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return dict(rho=float("nan"), n=int(m.sum()))
    from scipy.stats import spearmanr, pearsonr
    r = spearmanr(x[m], y[m])
    lp = pearsonr(np.log(np.maximum(x[m], 1e-30)), np.log(np.maximum(y[m], 1e-30))) if (x[m] > 0).all() and (y[m] > 0).all() else None
    return dict(rho=float(r.statistic), n=int(m.sum()), pearson_log=float(lp.statistic) if lp is not None else float("nan"))


def hist_ppl():
    d = {g: S.jload(os.path.join(S.ROOT, S.DENSE_RUN, "results", f"{g}.json"))["ppl"] for g in CONFIGS}
    m = {g: S.jload(os.path.join(S.ROOT, S.MG_RUN, "results", "CTRL_FR_ORIGINAL.json" if g == "ORIGINAL" else f"FR_{g}.json"))["ppl"]
         for g in CONFIGS}
    return d, m


def sdlog(v):
    v = np.asarray(v, float)
    return float(np.log(v).std(ddof=1))


# ---------------------------------------------------------------- Experiment 1
def exp1():
    dres = jl("dense_functional_difference.json")
    mraw = jl("logs", "moe_functional_raw.json")
    dppl, mppl = hist_ppl()
    # ---- dense table (per pair, pooled + per layer x proj)
    dense_pairs = {}
    for a, b in S.PAIRS:
        key = pk(a, b)
        row = dict(pooled={u: dres["pooled_over_layers"][key][u] for u in UNITS})
        if b != "BF16":
            n = nc = nb = 0
            rf_num = rf_den = 0.0
            for L in S.DLAY:
                for k in PROJ:
                    r = dres["per_layer"][str(L)][key][k]
                    n += r["n"]; nc += r["frac_code_changed"] * r["n"]; nb += r["frac_bf16_changed"] * r["n"]
            row.update(frac_code_changed=nc / n, frac_bf16_changed=nb / n,
                       ppl_abs_diff=abs(dppl[a] - dppl[b]), log_ppl_abs_diff=abs(math.log(dppl[a] / dppl[b])))
        dense_pairs[key] = row
    # ---- MoE
    mlay = mraw["layers"]
    moe_pairs = {}
    for a, b in S.PAIRS:
        key = pk(a, b)
        agg_num = sum(mlay[str(L)]["aggregate"][key]["vs_block"]["sumsq_diff"] for L in S.MLAY)
        agg_den = sum(mlay[str(L)]["aggregate"][key]["vs_block"]["sumsq_ref"] for L in S.MLAY)
        agr_den = sum(mlay[str(L)]["aggregate"][key]["vs_routed"]["sumsq_ref"] for L in S.MLAY)
        row = dict(pooled={k: math.sqrt(sum(mlay[str(L)]["pooled_projection"][key][k]["sumsq_diff"] for L in S.MLAY) /
                                        sum(mlay[str(L)]["pooled_projection"][key][k]["sumsq_ref"] for L in S.MLAY)) for k in PROJ})
        row["pooled"]["moe_block"] = math.sqrt(agg_num / agg_den)
        row["pooled"]["moe_routed"] = math.sqrt(agg_num / agr_den)
        # expert-level MLP (unweighted)
        em, n, nb = [], 0, 0.0
        rf = {k: [0.0, 0.0] for k in PROJ}
        for L in S.MLAY:
            for e, rec in mlay[str(L)]["per_expert"].items():
                if not rec.get("tokens"):
                    continue
                r = rec["pairs"][key]
                em.append(r["mlp"]["E_func"])
                for k in PROJ:
                    n += r[k]["n"]; nb += r[k]["frac_bf16_changed"] * r[k]["n"]
        row["expert_mlp_E"] = dict(median=float(np.median(em)), p10=float(np.percentile(em, 10)), p90=float(np.percentile(em, 90)),
                                   max=float(np.max(em)), min=float(np.min(em)), n_experts=len(em))
        if b != "BF16":
            row.update(frac_bf16_changed_all_experts=nb / n, ppl_abs_diff=abs(mppl[a] - mppl[b]),
                       log_ppl_abs_diff=abs(math.log(mppl[a] / mppl[b])))
        moe_pairs[key] = row
    # sampled 20 experts: aligned code + functional
    sc = mraw["sampled_codes"]
    samp_units = []
    for L in S.MLAY:
        for e, rec in mlay[str(L)]["per_expert"].items():
            if not rec.get("sampled"):
                continue
            for a, b in S.PAIRS:
                if b == "BF16":
                    continue
                key = pk(a, b)
                for k in PROJ:
                    cm = sc[f"{L}|{e}|{key}|{k}"]
                    r = rec["pairs"][key][k]
                    samp_units.append(dict(layer=L, expert=int(e), pair=key, proj=k, tokens=rec["tokens"], E_func=r["E_func"],
                                           E_mlp=rec["pairs"][key]["mlp"]["E_func"], rel_frob=r["rel_frob"],
                                           frac_bf16_changed=r["frac_bf16_changed"], **cm))
    # dense units
    dense_units = []
    for L in S.DLAY:
        for a, b in S.PAIRS:
            if b == "BF16" or (a, b) == S.DENSE_NULL or "G20261115" in (a, b):
                continue
            key = pk(a, b)
            for k in PROJ:
                r = dres["per_layer"][str(L)][key][k]
                dense_units.append(dict(layer=L, pair=key, proj=k, E_func=r["E_func"], E_mlp=dres["per_layer"][str(L)][key]["mlp"]["E_func"],
                                        rel_frob=r["rel_frob"], frac_bf16_changed=r["frac_bf16_changed"],
                                        frac_code_changed=r["frac_code_changed"], frac_level_changed=r["frac_level_changed"],
                                        frac_actorder_pos_changed=r["frac_actorder_pos_changed"],
                                        frac_group_changed=r["frac_group_changed"], hessian_rel_fro=r["hessian_rel_fro"],
                                        primary=b == "ORIGINAL"))
    # MoE all-expert units (no codes)
    moe_units = []
    for L in S.MLAY:
        for e, rec in mlay[str(L)]["per_expert"].items():
            if not rec.get("tokens"):
                continue
            for a, b in S.MOE_PRIMARY:
                key = pk(a, b)
                for k in PROJ:
                    r = rec["pairs"][key][k]
                    moe_units.append(dict(layer=L, expert=int(e), pair=key, proj=k, tokens=rec["tokens"], E_func=r["E_func"],
                                          rel_frob=r["rel_frob"], frac_bf16_changed=r["frac_bf16_changed"]))
    return dict(dense_pairs=dense_pairs, moe_pairs=moe_pairs, dense_units=dense_units, samp_units=samp_units,
                moe_units=moe_units, dppl=dppl, mppl=mppl, mlay=mlay, dres=dres)


def ratios(E):
    icm = S.jload(os.path.join(S.ROOT, S.ICM_RUN, "dense_moe_code_comparison.json"))
    dp, mp = E["dense_pairs"], E["moe_pairs"]
    dprim = [pk(*x) for x in S.DENSE_PRIMARY]
    mprim = [pk(*x) for x in S.MOE_PRIMARY]
    dmean = {u: float(np.mean([dp[k]["pooled"][u] for k in dprim])) for u in UNITS}
    mmean = {u: float(np.mean([mp[k]["pooled"][u] for k in mprim])) for u in list(PROJ) + ["moe_block", "moe_routed"]}
    r_code = icm.get("R_q") or icm.get("ratios", {}).get("R_q")
    r_code_p = icm.get("R_q_by_projection") or icm.get("ratios", {}).get("R_q_by_projection")
    rows = icm["rows"]
    dense_code, moe_code = rows["integer_code_changed"]
    r_bf16_sampled = rows["bf16_changed"][1] / rows["bf16_changed"][0]
    moe_bf16_all = float(np.mean([mp[k]["frac_bf16_changed_all_experts"] for k in mprim]))
    dense_bf16 = float(np.mean([dp[k]["frac_bf16_changed"] for k in dprim]))
    dppl, mppl = E["dppl"], E["mppl"]
    sd_d, sd_m = sdlog([dppl[g] for g in GROUPS]), sdlog([mppl[g] for g in GROUPS])
    out = dict(
        dense_primary_mean=dmean, moe_primary_mean=mmean,
        R_code=r_code, R_code_by_projection=r_code_p, code_dense=dense_code, code_moe_sampled=moe_code,
        R_bf16_sampled=r_bf16_sampled, R_bf16_all_experts=moe_bf16_all / dense_bf16, bf16_dense=dense_bf16, bf16_moe_all=moe_bf16_all,
        R_func={k: mmean[k] / dmean[k] for k in PROJ},
        R_MLP=mmean["moe_block"] / dmean["mlp"], R_MLP_routed=mmean["moe_routed"] / dmean["mlp"],
        sd_log_ppl=dict(dense=sd_d, moe=sd_m), R_SDlog=sd_m / sd_d,
        quant_error_vs_bf16=dict(dense_mlp=float(np.mean([dp[pk(g, "BF16")]["pooled"]["mlp"] for g in CONFIGS])),
                                 moe_block=float(np.mean([mp[pk(g, "BF16")]["pooled"]["moe_block"] for g in CONFIGS])),
                                 moe_routed=float(np.mean([mp[pk(g, "BF16")]["pooled"]["moe_routed"] for g in CONFIGS])),
                                 dense_proj={k: float(np.mean([dp[pk(g, "BF16")]["pooled"][k] for g in CONFIGS])) for k in PROJ},
                                 moe_proj={k: float(np.mean([mp[pk(g, "BF16")]["pooled"][k] for g in CONFIGS])) for k in PROJ}),
        note="exploratory cross-model comparison of different quantities; not a causal decomposition")
    # classification (protocol)
    Rm, Rc = out["R_MLP"], r_code
    rp = {k: out["R_func"][k] / r_code_p[k] for k in PROJ}
    conflict = any(v <= 0.25 for v in rp.values()) and any(v >= 0.5 for v in rp.values())
    if Rm >= 0.5 * Rc:
        cls = "F2-FUNCTIONAL-ALSO-MUCH-LARGER-IN-MOE"
    elif Rm <= 3 and all(v < 0.5 for v in rp.values()) and not conflict:
        cls = "F1-CODE-OVERSTATES-FUNCTIONAL"
    else:
        cls = "F3-MIXED"
    out.update(R_func_over_R_code_by_projection=rp, projection_conflict=conflict, classification=cls)
    return out


def correlations(E):
    vars_ = ["frac_code_changed", "frac_bf16_changed", "rel_frob", "frac_actorder_pos_changed", "hessian_rel_fro"]
    out = dict(dense={}, moe_sampled={}, moe_all={})
    du = E["dense_units"]
    for scope, units in (("all_nonnull_pairs", du), ("primary_pairs", [u for u in du if u["primary"]])):
        out["dense"][scope] = {}
        for pj in list(PROJ) + ["pooled"]:
            us = units if pj == "pooled" else [u for u in units if u["proj"] == pj]
            out["dense"][scope][pj] = {v: spearman([u[v] for u in us], [u["E_func"] for u in us]) for v in vars_}
    su = E["samp_units"]
    for scope, units in (("all_pairs", su), ("primary_pairs", [u for u in su if u["pair"].endswith("ORIGINAL")])):
        out["moe_sampled"][scope] = {}
        for pj in list(PROJ) + ["pooled"]:
            us = units if pj == "pooled" else [u for u in units if u["proj"] == pj]
            out["moe_sampled"][scope][pj] = {v: spearman([u[v] for u in us], [u["E_func"] for u in us]) for v in vars_ + ["tokens"]}
    mu = E["moe_units"]
    for pj in list(PROJ) + ["pooled"]:
        us = mu if pj == "pooled" else [u for u in mu if u["proj"] == pj]
        out["moe_all"][pj] = {v: spearman([u[v] for u in us], [u["E_func"] for u in us]) for v in ("frac_bf16_changed", "rel_frob", "tokens")}
    # cross-model pooled (dense + MoE sampled units, primary pairs)
    both = [u for u in du if u["primary"]] + [u for u in su if u["pair"].endswith("ORIGINAL")]
    out["cross_model_primary"] = {v: spearman([u[v] for u in both], [u["E_func"] for u in both]) for v in vars_}
    # pair-level (model-level) association with |delta log PPL|
    dp, mp = E["dense_pairs"], E["moe_pairs"]
    pl = {}
    for name, P in (("dense", dp), ("moe", mp)):
        keys = [k for k in P if not k.endswith("BF16") and (name == "moe" or "G20261115" not in k)]
        unit = "mlp" if name == "dense" else "moe_block"
        pl[name] = dict(n_pairs=len(keys),
                        E_vs_abs_dlogppl=spearman([P[k]["pooled"][unit] for k in keys], [P[k]["log_ppl_abs_diff"] for k in keys]),
                        rows={k: dict(E=P[k]["pooled"][unit], abs_dlogppl=P[k]["log_ppl_abs_diff"]) for k in keys})
    out["pair_level_ppl"] = pl
    out["independence_note"] = ("dense: 4 independent single-row perturbations (G20261115 = ORIGINAL excluded) -> 5 distinct "
                                "weight sets; MoE: 6 distinct configurations. Units (layer x projection x pair) share "
                                "configurations and are NOT independent samples; correlations are descriptive.")
    return out


def curvature_summary():
    cr = jl("logs", "curvature_raw.json")
    out = {}
    for model, lays in (("dense", S.DLAY), ("moe", S.MLAY)):
        out[model] = {}
        for pj in PROJ:
            recs = [v for L in lays for k, v in cr[model][str(L)].items() if k.endswith("|" + pj)]
            o = dict(n_tensors=len(recs))
            for bt in ("exec_position", "calib_diag", "heldout_diag"):
                o[bt] = {f: [float(np.mean([r[bt]["bins"][i][f] for r in recs])) for i in range(S.NB)]
                         for f in ("flip_share", "flip_rate", "diag_share", "block_share")}
                o[bt]["diag_over_full_median"] = float(np.median([r[bt]["diag_over_full"] for r in recs]))
                o[bt]["sum_block_over_full_median"] = float(np.median([r[bt]["sum_block_over_full"] for r in recs]))
                fs, ds = np.array(o[bt]["flip_share"]), np.array(o[bt]["block_share"])
                # share of flips in the bins that together carry the least 50 % of (block) energy
            sp = [r["split"] for r in recs]
            o["split"] = {f: float(np.median([s[f] for s in sp])) for f in ("share_flip", "share_grid", "share_cross",
                                                                              "frac_elements_flip", "frac_elements_grid_changed")}
            o["split"]["energy_per_element_flip_over_grid_median"] = float(np.median(
                [(s["energy_flip"] / max(s["frac_elements_flip"], 1e-30)) / (s["energy_grid"] / max(s["frac_elements_grid_changed"], 1e-30))
                 for s in sp if s["energy_grid"] > 0 and s["frac_elements_flip"] > 0]))
            # low-impact concentration: flips in the half of held-out-diag bins with lowest curvature
            hd = o["heldout_diag"]
            o["low_curv_half"] = dict(flip_share=float(sum(hd["flip_share"][:S.NB // 2])), diag_energy_share=float(sum(hd["diag_share"][:S.NB // 2])),
                                      block_energy_share=float(sum(hd["block_share"][:S.NB // 2])))
            out[model][pj] = o
    return out


# ---------------------------------------------------------------- Experiment 2
def exp2():
    pr = S.check_protocol()
    corr = jl("actorder_correctness.json")
    bf = S.jload(os.path.join(S.ROOT, S.DENSE_RUN, "bf16_baseline.json"))
    runs = [r for r in S.POLICY_RUNS + S.POLICY_RUNS_EXPL if all(os.path.exists(S.p("results", f"{r}__{g}.json")) for g in CONFIGS)]
    R = {r: {g: jl("results", f"{r}__{g}.json") for g in CONFIGS} for r in runs}
    aof = jl("logs", "ao_functional_raw.json") or {}
    ppl, stab, code, qual = {}, {}, {}, {}
    nat = R["natural"]
    nat_ppl = np.array([nat[g]["ppl"] for g in GROUPS])
    nat_sdlog = sdlog(nat_ppl)
    rng = np.random.default_rng(20261215)
    for r in runs:
        v = np.array([R[r][g]["ppl"] for g in GROUPS])
        ppl[r] = dict(per_grouping={g: R[r][g]["ppl"] for g in GROUPS}, ORIGINAL=R[r]["ORIGINAL"]["ppl"],
                      mean_nll={g: R[r][g]["mean_nll"] for g in CONFIGS}, valid=all(R[r][g]["valid"] for g in CONFIGS),
                      exploratory=r in S.POLICY_RUNS_EXPL)
        sdl = sdlog(v)
        pw = [abs(x - y) for x, y in itertools.combinations(v, 2)]
        st = dict(sd_ppl=float(v.std(ddof=1)), sd_log_ppl=sdl, cv=float(v.std(ddof=1) / v.mean()), R_SD=sdl / nat_sdlog,
                  max_pairwise_ppl_diff=float(max(pw)), range=float(v.max() - v.min()),
                  sd_log_ppl_4_nonnull=sdlog(v[:4]), R_SD_4_nonnull=sdlog(v[:4]) / sdlog(nat_ppl[:4]))
        stab[r] = st
        # quality + window-paired vs natural
        paired = {}
        for g in CONFIGS:
            a_ = np.array(R[r][g]["per_window_nll"]) / np.array(R[r][g]["per_window_tokens"])
            b_ = np.array(nat[g]["per_window_nll"]) / np.array(nat[g]["per_window_tokens"])
            d = a_ - b_
            bs = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(2000)]
            paired[g] = dict(mean_window_nll_diff=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                             frac_windows_worse=float((d > 0).mean()))
        qual[r] = dict(bf16_ppl=bf["ppl"], mean_ppl=float(v.mean()), penalty_vs_bf16=float(v.mean() - bf["ppl"]),
                       rel_penalty_vs_bf16=float(v.mean() / bf["ppl"] - 1), mean_change_vs_natural=float(v.mean() - nat_ppl.mean()),
                       best=float(v.min()), worst=float(v.max()), best_grouping=GROUPS[int(v.argmin())], worst_grouping=GROUPS[int(v.argmax())],
                       paired_vs_natural=paired,
                       quality_acceptable=bool(v.mean() - nat_ppl.mean() <= 0.10 * (nat_ppl.mean() - bf["ppl"])))
        # code / functional
        if r in aof:
            a = aof[r]
            dec = a["decomposition"]
            agg = {}
            prim = [g for g in GROUPS[:4]]
            for f in ("frac_code_changed", "frac_level_changed", "frac_bf16_changed", "frac_scale_changed", "frac_zero_changed",
                      "frac_group_changed"):
                agg[f] = float(np.mean([np.average([dec[f"{g}|L{L}|{k}"][f] for L in S.DLAY for k in PROJ],
                                                    weights=[dec[f"{g}|L{L}|{k}"]["n"] for L in S.DLAY for k in PROJ]) for g in prim]))
            byp = {k: {f: float(np.mean([np.average([dec[f"{g}|L{L}|{k}"][f] for L in S.DLAY], weights=[dec[f"{g}|L{L}|{k}"]["n"] for L in S.DLAY])
                                         for g in prim])) for f in ("frac_code_changed", "frac_bf16_changed", "frac_group_changed")}
                   for k in PROJ}
            pos = float(np.mean([a["per_layer"][str(L)][pk(g, "ORIGINAL")][k]["frac_actorder_pos_changed"] for g in prim for L in S.DLAY for k in PROJ]))
            rf = float(np.mean([np.mean([a["per_layer"][str(L)][pk(g, "ORIGINAL")][k]["rel_frob"] for L in S.DLAY for k in PROJ]) for g in prim]))
            E_mlp = float(np.mean([a["pooled_over_layers"][pk(g, "ORIGINAL")]["mlp"] for g in prim]))
            E_proj = {k: float(np.mean([a["pooled_over_layers"][pk(g, "ORIGINAL")][k] for g in prim])) for k in PROJ}
            E_q = float(np.mean([a["pooled_over_layers"][pk(g, "BF16")]["mlp"] for g in CONFIGS]))
            null_zero = all(a["pooled_over_layers"][pk("G20261115", "ORIGINAL")][u] == 0 for u in UNITS)
            code[r] = dict(**agg, frac_actorder_pos_changed=pos, rel_frob_mean=rf,
                           by_projection=byp, E_mlp=E_mlp, E_proj=E_proj, E_mlp_quant_vs_bf16=E_q, null_functional_zero=null_zero,
                           grouping_pairs_E_mlp=float(np.mean([a["pooled_over_layers"][pk(x, y)]["mlp"]
                                                               for x, y in itertools.combinations(GROUPS[:4], 2)])))
    # descriptors / hypotheses
    nc = code.get("natural", {})
    desc = {}
    for r in runs:
        if r == "natural" or r not in code:
            continue
        cs = code[r]["frac_code_changed"] <= 0.5 * nc["frac_code_changed"]
        fs = code[r]["E_mlp"] <= 0.5 * nc["E_mlp"]
        ps = stab[r]["R_SD"] <= 0.5
        pu = stab[r]["R_SD"] >= 0.75
        qa = qual[r]["quality_acceptable"]
        hyp = "C-PRACTICAL-STABILITY-CONTROL" if (cs and fs and ps and qa) else (
            "D-CODE-STABILITY-ONLY" if (cs and (pu or not qa)) else "INTERMEDIATE")
        desc[r] = dict(code_stabilized=cs, functional_stabilized=fs, ppl_stabilized=ps, ppl_unchanged=pu, quality_acceptable=qa,
                       code_ratio_vs_natural=code[r]["frac_code_changed"] / nc["frac_code_changed"],
                       functional_ratio_vs_natural=code[r]["E_mlp"] / nc["E_mlp"], R_SD=stab[r]["R_SD"], hypothesis=hyp)
    checks = {k: (all(v.values()) if isinstance(v, dict) and v else v) for k, v in corr.items() if k in "ABCDEFGH"}
    checks["end"] = corr.get("end")
    return dict(runs=runs, ppl=ppl, stab=stab, code=code, qual=qual, desc=desc, checks=checks, corr=corr,
                all_checks_pass=all(v for k, v in checks.items() if k != "end") and all((corr.get("end") or {}).values()))


# ---------------------------------------------------------------- figures
def figures(E, Rt, C, X2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(S.p("figures"), exist_ok=True)
    cols = {"gate_proj": "tab:blue", "up_proj": "tab:orange", "down_proj": "tab:green"}
    # fig1
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, units, title in ((axs[0], [u for u in E["dense_units"]], "dense (12 tensors x 10 non-null pairs)"),
                             (axs[1], E["samp_units"], "MoE (20 sampled experts x 3 proj x 15 pairs)")):
        for pj in PROJ:
            us = [u for u in units if u["proj"] == pj]
            ax.scatter([u["frac_code_changed"] for u in us], [u["E_func"] for u in us], s=8, alpha=.6, color=cols[pj], label=pj)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("integer-code disagreement"); ax.set_ylabel("E_func (rel. to BF16 output)")
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(S.p("figures", "fig1_code_vs_functional.png"), dpi=140); plt.close(fig)
    # fig2
    fig, ax = plt.subplots(figsize=(8, 4))
    labs = ["code (sampled)", "BF16 (all exp.)", "E gate", "E up", "E down", "E MLP/MoE block", "SD(log PPL)"]
    vals = [Rt["R_code"], Rt["R_bf16_all_experts"], Rt["R_func"]["gate_proj"], Rt["R_func"]["up_proj"], Rt["R_func"]["down_proj"],
            Rt["R_MLP"], Rt["R_SDlog"]]
    ax.bar(range(len(vals)), vals, color=["tab:gray", "tab:gray", "tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(1, color="k", lw=.6); ax.set_yscale("log"); ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, rotation=20, fontsize=8)
    ax.set_ylabel("MoE / dense ratio")
    ax.set_title("Dense vs MoE sensitivity ratios (different quantities; exploratory, not a decomposition)", fontsize=9)
    fig.tight_layout(); fig.savefig(S.p("figures", "fig2_dense_moe_ratios.png"), dpi=140); plt.close(fig)
    if X2 and X2["runs"]:
        runs = X2["runs"]
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        x = np.arange(len(runs))
        axs[0].bar(x, [X2["code"].get(r, {}).get("frac_code_changed", np.nan) for r in runs], color="tab:blue")
        axs[0].set_title("integer-code disagreement vs policy ORIGINAL (4 pairs)", fontsize=8)
        axs[1].bar(x, [X2["code"].get(r, {}).get("E_mlp", np.nan) for r in runs], color="tab:purple")
        axs[1].set_title("E_MLP vs policy ORIGINAL (held-out)", fontsize=8)
        axs[2].bar(x, [X2["stab"][r]["sd_log_ppl"] for r in runs], color="tab:red")
        axs[2].set_title("SD(log PPL), 5 groupings", fontsize=8)
        for ax in axs:
            ax.set_xticks(x); ax.set_xticklabels(runs, rotation=25, fontsize=8)
        fig.tight_layout(); fig.savefig(S.p("figures", "fig3_actorder_stability.png"), dpi=140); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for r in runs:
            ax.scatter(X2["stab"][r]["sd_log_ppl"], X2["qual"][r]["mean_ppl"], s=40, marker="o" if r in S.POLICY_RUNS else "x")
            ax.annotate(r, (X2["stab"][r]["sd_log_ppl"], X2["qual"][r]["mean_ppl"]), fontsize=8, xytext=(4, 2), textcoords="offset points")
        ax.axhline(X2["qual"]["natural"]["bf16_ppl"], color="gray", lw=.6, ls=":")
        ax.set_xlabel("SD(log PPL) over 5 groupings (descriptive)"); ax.set_ylabel("mean quantized PPL")
        ax.set_title("quality-stability (o primary, x exploratory)", fontsize=9)
        fig.tight_layout(); fig.savefig(S.p("figures", "fig4_quality_stability_pareto.png"), dpi=140); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, r in enumerate(runs):
            ax.plot(range(5), [X2["ppl"][r]["per_grouping"][g] for g in GROUPS], "-o", ms=4, label=r)
        ax.set_xticks(range(5)); ax.set_xticklabels(GROUPS, fontsize=8); ax.set_ylabel("PPL"); ax.legend(fontsize=7)
        ax.set_title("grouping-level PPL per act-order policy (G20261115 = exact null)", fontsize=9)
        fig.tight_layout(); fig.savefig(S.p("figures", "fig6_grouping_ppl.png"), dpi=140); plt.close(fig)
    # fig5
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    xb = (np.arange(S.NB) + .5) / S.NB
    for row, model in enumerate(("dense", "moe")):
        for j, bt in enumerate(("exec_position", "calib_diag", "heldout_diag")):
            ax = axs[row, j]
            for pj in PROJ:
                d = C[model][pj][bt]
                ax.plot(xb, d["flip_share"], "-o", ms=3, color=cols[pj], label=f"{pj} flip share")
                ax.plot(xb, d["block_share"], ":", color=cols[pj], label=f"{pj} functional energy share")
            ax.set_yscale("log"); ax.set_title(f"{model}: by {bt} ({'0 = first executed' if bt == 'exec_position' else 'ascending'})", fontsize=8)
            if row == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(S.p("figures", "fig5_curvature_importance.png"), dpi=140); plt.close(fig)


def inventory():
    """Artifact inventory: inputs verified by the audit + every new cache file (path, bytes, sha256)."""
    au = jl("logs", "audit.json")
    man = jl("functional_activation_manifest.json")
    new = {}
    for root, _, files in os.walk(S.CACHE):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            new[os.path.relpath(fp, S.CACHE)] = dict(bytes=os.path.getsize(fp), sha256=S.sha256_file(fp))
    S.jdump(dict(inputs=dict(dense=au["dense"], moe=au["moe"], icm=au["icm"], dense_null_equal=au["dense_null_equal"],
                             windows=au["windows"], all_ok=au["all_ok"]),
                 activation_capture=dict(dense={k: v for k, v in man["dense"].items() if k != "files"},
                                         moe={k: v for k, v in man["moe"].items() if k != "files"}),
                 cache_root=S.CACHE, cache_files=new,
                 results=sorted(os.listdir(S.p("results"))) if os.path.isdir(S.p("results")) else []),
            S.p("artifact_inventory.json"))


def main():
    pr = S.check_protocol()
    E = exp1()
    Rt = ratios(E)
    Co = correlations(E)
    C = curvature_summary()
    S.jdump(dict(note="see dense_functional_difference.json for per-layer detail", pairs=E["dense_pairs"]),
            S.p("logs", "dense_pair_summary.json"))
    mlay = E["mlay"]
    S.jdump(dict(note="all 60 routed experts x 4 layers; shared reference routing; FP32 arithmetic; E relative to BF16 reference",
                 pairs=E["moe_pairs"],
                 per_layer_aggregate={L: mlay[L]["aggregate"] for L in mlay},
                 per_layer_pooled_projection={L: mlay[L]["pooled_projection"] for L in mlay},
                 per_expert={L: mlay[L]["per_expert"] for L in mlay},
                 sampled_subset_units=E["samp_units"]),
            S.p("moe_functional_difference.json"))
    S.jdump(dict(ratios=Rt, correlations=Co, curvature=C,
                 dense_pairs={k: v for k, v in E["dense_pairs"].items()}, moe_pairs=E["moe_pairs"]),
            S.p("functional_vs_code_comparison.json"))
    X2 = None
    if os.path.exists(S.p("actorder_correctness.json")) and os.path.exists(S.p("results", "natural__ORIGINAL.json")):
        X2 = exp2()
        S.jdump(dict(policies={r: dict(spec=S.policy_spec(r), exploratory=r in S.POLICY_RUNS_EXPL) for r in X2["runs"]},
                     definitions=pr["experiment2"]["policies"], weights=S.c("ao", "<policy>", "<config>", "L<l>.pt")),
                S.p("actorder_policy_manifest.json"))
        S.jdump(X2["ppl"], S.p("actorder_ppl_results.json"))
        S.jdump(dict(stability=X2["stab"], descriptors=X2["desc"], checks=X2["checks"]), S.p("actorder_stability_comparison.json"))
        S.jdump(X2["code"], S.p("actorder_code_difference.json"))
        S.jdump(X2["qual"], S.p("actorder_quality_tradeoff.json"))
    figures(E, Rt, C, X2)
    inventory()
    S.jdump(dict(E1=dict(ratios=Rt), E2=dict(desc=X2["desc"], stab=X2["stab"], qual=X2["qual"], checks=X2["checks"]) if X2 else None),
            S.p("logs", "analysis_digest.json"))
    print(json.dumps(dict(R=Rt, desc=X2["desc"] if X2 else None), indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
