"""Analysis + report for the Fixed GU-AGQ robustness study (runs_fixed_gu_agq/). Decision rule frozen in protocol.json
and implemented in fixed_gu_agq_validation.classify."""
from __future__ import annotations

import json, os, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import fixed_gu_agq_validation as FG     # noqa: E402
from moe_calib import projection_select as PS  # noqa: E402

p, jdump, jload, LAYERS, SEEDS, E = FG.p, FG.jdump, FG.jload, FG.LAYERS, FG.SEEDS, FG.E
CFG = FG.ALL_CONFIGS
TAU = FG.RULES["tau_rel"]
N_BOOT, BOOT_SEED = 2000, 20261113
COLOR = dict(GPTQ="#2a78d6", GU_AGQ="#1baf7a", DOWN_AGQ="#eda100", FULL_AGQ="#eb6834")
LABEL = dict(GPTQ="Standard GPTQ", FULL_AGQ="Full AGQ", GU_AGQ="Fixed GU-AGQ", DOWN_AGQ="Down-AGQ (diag.)")


def rel(a, b):
    return (a - b) / b


# ============================================================================ loading
def load_cell(split, s, L):
    if split == "confirm":
        r = json.load(open(p("results", "confirm", f"s{s}", f"L{L}.json")))
        m = dict(np.load(p("results", "confirm", f"s{s}", f"expert_mats_L{L}.npz")))
        return r["configs"], m, [r["meta"]]
    a = json.load(open(p("results", "regression_oldtest", f"s{s}", f"L{L}.json")))
    b = json.load(open(p("results", "oldtest", f"s{s}", f"L{L}.json")))
    ma = dict(np.load(p("results", "regression_oldtest", f"s{s}", f"expert_mats_L{L}.npz")))
    mb = dict(np.load(p("results", "oldtest", f"s{s}", f"expert_mats_L{L}.npz")))
    for k in ("Uden", "c2den", "tokens", "csum", "c2sum"):
        assert np.array_equal(ma[k], mb[k]), "old-test reference pass differs between runs"
    for c in a["configs"]:
        assert a["configs"][c]["den"] == b["configs"]["GU_AGQ"]["den"]
    ma.update({k: v for k, v in mb.items() if k.startswith(("J_", "U_"))})
    return dict(a["configs"], **b["configs"]), ma, [a["meta"], b["meta"]]


def load_all():
    D = {}
    for split in ("confirm", "oldtest"):
        D[split] = {s: {L: load_cell(split, s, L) for L in LAYERS} for s in SEEDS}
    return D


def El(D, split, s, L, c):
    return D[split][s][L][0][c]["moe_layer_output_rel_err"]


# ============================================================================ summaries
def per_layer(D):
    out = {}
    for split in D:
        rec = dict(seeds={}, seed_averaged={})
        for s in SEEDS:
            rec["seeds"][str(s)] = {str(L): dict(E={c: El(D, split, s, L, c) for c in CFG},
                                                 native={c: D[split][s][L][0][c]["native_rel_err"] for c in CFG},
                                                 gu_vs_full_agq=rel(El(D, split, s, L, "GU_AGQ"), El(D, split, s, L, "FULL_AGQ")),
                                                 gu_vs_gptq=rel(El(D, split, s, L, "GU_AGQ"), El(D, split, s, L, "GPTQ")),
                                                 full_agq_vs_gptq=rel(El(D, split, s, L, "FULL_AGQ"), El(D, split, s, L, "GPTQ")),
                                                 previously_inspected=(s, L) in FG.PREV_INSPECTED)
                                    for L in LAYERS}
        for L in LAYERS:
            avg = {c: float(np.mean([El(D, split, s, L, c) for s in SEEDS])) for c in CFG}
            gs = [rec["seeds"][str(s)][str(L)]["gu_vs_full_agq"] for s in SEEDS]
            rec["seed_averaged"][str(L)] = dict(E=avg, gu_vs_full_agq=rel(avg["GU_AGQ"], avg["FULL_AGQ"]),
                                                gu_vs_gptq=rel(avg["GU_AGQ"], avg["GPTQ"]),
                                                full_agq_vs_gptq=rel(avg["FULL_AGQ"], avg["GPTQ"]),
                                                seeds_gu_better_than_full=int(sum(g < 0 for g in gs)),
                                                per_seed_gu_vs_full=gs, spread=float(max(gs) - min(gs)))
        out[split] = rec
    return out


def per_seed(D):
    out = {}
    for split in D:
        rec = dict(seeds={})
        for s in SEEDS:
            m = {c: float(np.mean([El(D, split, s, L, c) for L in LAYERS])) for c in CFG}
            pooled = {c: sum(D[split][s][L][0][c]["num"] for L in LAYERS) / sum(D[split][s][L][0][c]["den"]
                                                                                  for L in LAYERS) for c in CFG}
            nat = {c: float(np.mean([D[split][s][L][0][c]["native_rel_err"] for L in LAYERS])) for c in CFG}
            rec["seeds"][str(s)] = dict(
                E4L=m, delta_gu_vs_full_agq=rel(m["GU_AGQ"], m["FULL_AGQ"]), delta_gu_vs_gptq=rel(m["GU_AGQ"], m["GPTQ"]),
                delta_full_agq_vs_gptq=rel(m["FULL_AGQ"], m["GPTQ"]),
                layers_gu_better_than_full=int(sum(El(D, split, s, L, "GU_AGQ") < El(D, split, s, L, "FULL_AGQ")
                                                   for L in LAYERS)),
                pooled_ratio_secondary=pooled, pooled_delta_gu_vs_full_agq=rel(pooled["GU_AGQ"], pooled["FULL_AGQ"]),
                native_E4L=nat, native_delta_gu_vs_full_agq=rel(nat["GU_AGQ"], nat["FULL_AGQ"]),
                native_ranking_agrees=sorted(CFG, key=lambda c: m[c]) == sorted(CFG, key=lambda c: nat[c]))
        d = np.array([rec["seeds"][str(s)]["delta_gu_vs_full_agq"] for s in SEEDS])
        g = np.array([rec["seeds"][str(s)]["delta_gu_vs_gptq"] for s in SEEDS])
        a = np.array([rec["seeds"][str(s)]["delta_full_agq_vs_gptq"] for s in SEEDS])
        rec["across_seeds"] = dict(
            E4L_mean={c: float(np.mean([rec["seeds"][str(s)]["E4L"][c] for s in SEEDS])) for c in CFG},
            delta_gu_vs_full_agq=dict(mean=float(d.mean()), median=float(np.median(d)), min=float(d.min()),
                                      max=float(d.max()), spread=float(d.max() - d.min()),
                                      seeds_improved=int((d < 0).sum())),
            delta_gu_vs_gptq=dict(mean=float(g.mean()), median=float(np.median(g)), min=float(g.min()), max=float(g.max())),
            delta_full_agq_vs_gptq=dict(mean=float(a.mean()), per_seed=a.tolist()))
        out[split] = rec
    return out


def paired(D):
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    for split in D:
        rec = {}
        S = len(D[split][0][LAYERS[0]][0]["GPTQ"]["per_seq"]["num"])
        idx = rng.integers(0, S, size=(N_BOOT, S))
        for a, b in (("GU_AGQ", "FULL_AGQ"), ("GU_AGQ", "GPTQ"), ("FULL_AGQ", "GPTQ")):
            rs = {}
            for s in SEEDS:
                num = lambda L, c: np.array(D[split][s][L][0][c]["per_seq"]["num"])
                den = {L: np.array(D[split][s][L][0]["GPTQ"]["per_seq"]["den"]) for L in LAYERS}
                lay = {}
                for L in LAYERS:
                    dl = (num(L, a) - num(L, b)) / den[L]
                    lay[str(L)] = dict(frac_windows_improved=float((dl < 0).mean()),
                                       mean_window_ratio_diff=float(dl.mean()),
                                       rel_diff=rel(num(L, a).sum(), num(L, b).sum()))
                ea = np.mean([num(L, a)[idx].sum(1) / den[L][idx].sum(1) for L in LAYERS], 0)
                eb = np.mean([num(L, b)[idx].sum(1) / den[L][idx].sum(1) for L in LAYERS], 0)
                bd = (ea - eb) / eb
                pa = np.mean([num(L, a).sum() / den[L].sum() for L in LAYERS])
                pb = np.mean([num(L, b).sum() / den[L].sum() for L in LAYERS])
                rs[str(s)] = dict(rel_4L=float((pa - pb) / pb), boot_ci95=[float(np.quantile(bd, .025)),
                                                                          float(np.quantile(bd, .975))],
                                  boot_frac_a_better=float((bd < 0).mean()), layers=lay)
            rec[f"{a}_vs_{b}"] = rs
        rec["note"] = (f"descriptive window bootstrap within each seed ({N_BOOT} reps, seed {BOOT_SEED}); seeds are the "
                       "replication unit, windows are not")
        out[split] = rec
    return out


def contributions(D):
    out = {}
    for split in D:
        rec = dict(seeds={}, seed_averaged={})
        for s in SEEDS:
            rec["seeds"][str(s)] = {}
            for L in LAYERS:
                Ev = {c: El(D, split, s, L, c) for c in CFG}
                rec["seeds"][str(s)][str(L)] = _contrib(Ev)
        for L in LAYERS:
            Ev = {c: float(np.mean([El(D, split, s, L, c) for s in SEEDS])) for c in CFG}
            rec["seed_averaged"][str(L)] = _contrib(Ev)
        out[split] = rec
    out["definition"] = ("phi_GU = 0.5[(E_A - E_B) + (E_C - E_D)], phi_D = 0.5[(E_A - E_C) + (E_B - E_D)], "
                         "phi_GU + phi_D = E_A - E_D; values reported relative to E_A (positive = improvement). Share "
                         "of the full gain is reported only when |E_A - E_D| / E_A >= tau (0.5 %); otherwise null.")
    return out


def _contrib(Ev):
    A, B, C, D_ = Ev["GPTQ"], Ev["GU_AGQ"], Ev["DOWN_AGQ"], Ev["FULL_AGQ"]
    phi_gu, phi_d = 0.5 * ((A - B) + (C - D_)), 0.5 * ((A - C) + (B - D_))
    full = A - D_
    I = D_ - B - C + A
    big = abs(full) / A >= TAU
    return dict(E=Ev, phi_GU_rel=phi_gu / A, phi_D_rel=phi_d / A, full_gain_rel=full / A, interaction_rel=I / A,
                identity_residual=abs(phi_gu + phi_d - full) / A,
                share_GU=(phi_gu / full) if big else None, share_D=(phi_d / full) if big else None,
                G_GU_given_down_gptq_rel=(A - B) / A, G_D_given_gu_gptq_rel=(A - C) / A,
                G_GU_given_down_agq_rel=(C - D_) / A, G_D_given_gu_agq_rel=(B - D_) / A,
                interaction_sign="additive" if I == 0 else ("antagonistic" if I > 0 else "synergistic"))


def decomposition(D):
    out = {}
    for split in D:
        rec = {}
        worst_id, worst_vs = 0.0, 0.0
        for s in SEEDS:
            for L in LAYERS:
                x = D[split][s][L][0]
                cell = {}
                for c in CFG:
                    r = x[c]
                    worst_id = max(worst_id, abs(r["D"] + r["C"] - r["decomp_total"]) / r["decomp_total"])
                    worst_vs = max(worst_vs, abs(r["decomp_total"] - r["num"]) / r["num"])
                    cell[c] = dict(total_rel=r["num"] / r["den"], D_rel=r["D_rel"], C_rel=r["C_rel"],
                                   cross_fraction=r["cross_fraction"])
                for c in CFG:
                    for ref in ("GPTQ", "FULL_AGQ"):
                        cell[c][f"D_change_vs_{ref}"] = rel(cell[c]["D_rel"], cell[ref]["D_rel"])
                        cell[c][f"C_change_vs_{ref}_abs"] = cell[c]["C_rel"] - cell[ref]["C_rel"]
                g, f = cell["GU_AGQ"], cell["FULL_AGQ"]
                cell["gu_vs_full_ranking"] = dict(by_total=bool(g["total_rel"] < f["total_rel"]),
                                                  by_diagonal=bool(g["D_rel"] < f["D_rel"]),
                                                  cross_flips_ranking=bool((g["total_rel"] < f["total_rel"]) !=
                                                                           (g["D_rel"] < f["D_rel"])),
                                                  delta_total=g["total_rel"] - f["total_rel"],
                                                  delta_D=g["D_rel"] - f["D_rel"], delta_C=g["C_rel"] - f["C_rel"])
                rec[f"s{s}_L{L}"] = cell
        rec["max_identity_rel_residual"] = worst_id
        rec["max_decomp_vs_direct_rel"] = worst_vs
        rec["any_cross_flip_gu_vs_full"] = any(v["gu_vs_full_ranking"]["cross_flips_ranking"] for k, v in rec.items()
                                               if isinstance(v, dict) and "gu_vs_full_ranking" in v)
        out[split] = rec
    return out


def calib_stats():
    st = json.load(open(os.path.join(ROOT, "runs_agq_only_validation", "affinity_statistics.json")))
    return {s: {(r["layer"], r["expert"]): r for r in st["configs"][f"FR_s{s}"]["per_expert"]} for s in SEEDS}


def _rank(x):
    x = np.asarray(x, dtype=float); r = np.empty_like(x); r[np.argsort(x, kind="stable")] = np.arange(len(x)); return r


def _spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    return float(np.corrcoef(_rank(a[ok]), _rank(b[ok]))[0, 1])


def concentration(D, cs):
    out = {}
    for split in D:
        rec = {}
        for s in SEEDS:
            for L in LAYERS:
                M = D[split][s][L][1]
                Jg = M["J_GPTQ"].sum(0)
                cell = {}
                for c in ("GU_AGQ", "FULL_AGQ"):
                    G = Jg - M[f"J_{c}"].sum(0)
                    pos, neg = G[G > 0], G[G < 0]
                    order = np.argsort(-G)
                    net = float(G.sum())
                    pm = float(pos.sum())
                    st = [cs[s][(L, e)] for e in range(E)]
                    cell[c] = dict(
                        n_improved=int((G > 0).sum()), n_degraded=int((G < 0).sum()),
                        positive_mass=pm, negative_mass=float(neg.sum()), net_gain=net,
                        net_gain_rel_to_D_gptq=net / float(Jg.sum()),
                        top1_expert=int(order[0]), top1_share_of_positive=float(G[order[0]] / pm) if pm > 0 else None,
                        top5_experts=[int(e) for e in order[:5]],
                        top5_share_of_positive=float(G[order[:5]].sum() / pm) if pm > 0 else None,
                        top1_share_of_net=float(G[order[0]] / net) if net > 0 else None,
                        top5_share_of_net=float(G[order[:5]].sum() / net) if net > 0 else None,
                        net_without_top1_rel_to_D_gptq=float((net - G[order[0]]) / Jg.sum()),
                        spearman_gain_vs_exposure=_spear(G, [x["n"] for x in st]),
                        spearman_gain_vs_affinity_mean=_spear(G, [x["mean"] for x in st]),
                        spearman_gain_vs_affinity_cv=_spear(G, [x["cv"] for x in st]),
                        spearman_rel_gain_vs_affinity_cv=_spear(G / Jg, [x["cv"] for x in st]),
                        cumulative_positive_share=(np.cumsum(np.sort(pos)[::-1]) / pm).tolist() if pm > 0 else [],
                        gain_per_expert=G.tolist())
                rec[f"s{s}_L{L}"] = cell
        stab = {}
        for L in LAYERS:
            for c in ("GU_AGQ", "FULL_AGQ"):
                tops = [set(rec[f"s{s}_L{L}"][c]["top5_experts"]) for s in SEEDS]
                jac = [len(tops[i] & tops[j]) / len(tops[i] | tops[j]) for i in range(3) for j in range(i + 1, 3)]
                stab[f"L{L}_{c}"] = dict(top1_per_seed=[rec[f"s{s}_L{L}"][c]["top1_expert"] for s in SEEDS],
                                         top5_jaccard_pairs=jac, top5_in_all_seeds=sorted(set.intersection(*tops)))
        rec["top_expert_stability"] = stab
        out[split] = rec
    out["definition"] = ("G_e = D_e^GPTQ - D_e^method, D_e = sum_i c_ie^2 ||dE_e(x_i)||^2 summed over the split; "
                         "positive = improvement; shares of positive mass and of signed net gain")
    return out


def expert40(D, cs):
    L, e = 16, 40
    out = dict(layer=L, expert=e, post_hoc=True, note="identified in earlier studies; mechanism tracking only", seeds={})
    for s in SEEDS:
        st = cs[s][(L, e)]
        rec = dict(calibration=dict(tokens=st["n"], affinity_mean=st["mean"], affinity_cv=st["cv"],
                                    natural_p=st["natural_p"], natural_decile=st["natural_decile"],
                                    cv_rank_in_layer=int(1 + sum(cs[s][(L, k)]["cv"] > st["cv"] for k in range(E)))),
                   splits={})
        for split in D:
            M = D[split][s][L][1]
            J = {c: float(M[f"J_{c}"][:, e].sum()) for c in CFG}
            net = {c: float((M["J_GPTQ"].sum(0) - M[f"J_{c}"].sum(0)).sum()) for c in ("GU_AGQ", "FULL_AGQ")}
            rec["splits"][split] = dict(D_e=J, rel_vs_gptq={c: rel(J[c], J["GPTQ"]) for c in CFG},
                                        gu_vs_full=rel(J["GU_AGQ"], J["FULL_AGQ"]),
                                        share_of_layer_net_gain={c: (J["GPTQ"] - J[c]) / net[c] if net[c] != 0 else None
                                                                 for c in net},
                                        share_GU_of_full=_contrib(J)["share_GU"])
        out["seeds"][str(s)] = rec
    return out


def cost():
    hc = json.load(open(os.path.join(ROOT, "runs_agq_only_validation", "cost_results.json")))
    c = {k: jload("logs", f"cost_{k}.json") for k in ("regression_oldtest", "confirm", "oldtest")}
    g, a = hc["gptq_historical_FR_s0"], hc["agq_FR_s0"]
    est_gu = a["hessian_s"] + a["gptq_s"]          # weighted gu replay + identical solves
    return dict(
        incremental=dict(new_gptq_solves=0,
                         artifact_loading_s=sum(v["candidate_load_s"] for x in c.values() for v in x["cells"].values()),
                         regression_s=sum(json.load(open(p("results", "regression_oldtest", f"s{s}", f"L{L}.json")))
                                          ["meta"]["total_s"] for s in SEEDS for L in LAYERS),
                         regression_note="from per-cell metadata (cost log was rewritten by a resumed run)",
                         confirm_s=sum(v["seconds"] for v in c["confirm"]["cells"].values()),
                         oldtest_s=sum(v["seconds"] for v in c["oldtest"]["cells"].values()),
                         peak_gpu_gb=max(x["peak_gpu_gb"] for x in c.values()),
                         peak_cpu_gb=max(x["peak_cpu_gb"] for x in c.values()),
                         temporary_storage_bytes=sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in
                                                     os.walk(p("results")) for f in fs),
                         timings_include="4 configs + LayerEvalAccumulator cross-check + native BF16 forward (confirm)"),
        fresh_deployment_4_layers=dict(
            source="runs_agq_only_validation/cost_results.json, matched FR s0 runs (s1/s2 AGQ runs were slowed by "
                   "concurrent host load and are not used)",
            standard_gptq=dict(hessian_s=g["hessian_s"], gptq_s=g["gptq_s"], total=g["hessian_s"] + g["gptq_s"]),
            full_agq=dict(hessian_s=a["hessian_s"], gptq_s=a["gptq_s"], total=a["hessian_s"] + a["gptq_s"]),
            fixed_gu_agq_estimated=dict(total=est_gu, measured=False,
                                        basis="same number of GPTQ solves (one per projection) as Full AGQ; the replay "
                                              "needs router coefficients for the gu Hessian only, so its cost is <= "
                                              "the Full AGQ replay"),
            cost_ratio_vs_full_agq=est_gu / (a["hessian_s"] + a["gptq_s"]),
            profiler_support="moe_calib.profiler.Profiler applies hessian_weighting to BOTH the gu and down Hessians; a "
                             "direct GU-only weighting would need one new opt-in option (not implemented: quality gate "
                             "not passed)" ,
            mixed_artifact_assembly_note="this study assembled GU-AGQ from two full solutions; that is NOT the deployment "
                                         "cost"),
        inference=dict(change="none expected (same format, one tensor per projection)", measured=False))


def correctness(D, dec):
    inv, reg, cm = jload("artifact_inventory.json"), jload("candidate_regression.json"), jload("configuration_manifest.json")
    metas = [m for split in D for s in SEEDS for L in LAYERS for m in D[split][s][L][2]]
    tests = {}
    for tag in ("fixed_gu", "all"):
        f = p("logs", f"pytest_{tag}.log")
        lines = [l for l in open(f).read().strip().splitlines() if l.strip()] if os.path.exists(f) else []
        if lines:
            tests[tag] = lines[-1]
            tests[f"{tag}_passed"] = "passed" in lines[-1] and "failed" not in lines[-1] and "error" not in lines[-1]
    gu_src = all(all(r["q"] == PS.CONFIGS["GU_AGQ"][r["projection"]] for r in cm["seeds"][str(s)][str(L)]["GU_AGQ"]["projections"])
                 and cm["seeds"][str(s)][str(L)]["GU_AGQ"]["n_from_agq"] == 120 for s in SEEDS for L in LAYERS)
    same_seed = all(f"FR_s{s}" in r["source_file"] for s in SEEDS for L in LAYERS for c in CFG
                    for r in cm["seeds"][str(s)][str(L)][c]["projections"])
    leg = {s: {L: json.load(open(os.path.join(ROOT, "runs_legacy_metric_reeval", "results", "confirm", f"L{L}.json")))
               ["methods"][f"FR_s{s}"]["moe_layer_output_rel_err"] for L in LAYERS} for s in SEEDS}
    conf_gptq_ok = all(abs(El(D, "confirm", s, L, "GPTQ") / leg[s][L] - 1) < 1e-9 for s in SEEDS for L in LAYERS)
    same_windows = all(D[split][s][L][0][c]["den"] == D[split][s][L][0]["GPTQ"]["den"] and
                       len(D[split][s][L][0][c]["per_seq"]["num"]) == len(D[split][s][L][0]["GPTQ"]["per_seq"]["num"])
                       for split in D for s in SEEDS for L in LAYERS for c in CFG)
    same_ref = all(D[split][s][L][0]["GPTQ"]["den"] == D[split][0][L][0]["GPTQ"]["den"] for split in D for s in SEEDS
                   for L in LAYERS)
    acc_ok = all(v["abs_diff"] <= 1e-12 and v["per_seq_equal"] for m in metas for v in m["accumulator_check"].values())
    nat_rank = all(sorted(CFG, key=lambda c: El(D, split, s, L, c)) ==
                   sorted(CFG, key=lambda c: D[split][s][L][0][c]["native_rel_err"])
                   for split in D for s in SEEDS for L in LAYERS)
    dis = []
    for split in D:
        for s in SEEDS:
            for L in LAYERS:
                x = D[split][s][L][0]
                for i, a in enumerate(CFG):
                    for b in CFG[i + 1:]:
                        f32 = x[a]["moe_layer_output_rel_err"] < x[b]["moe_layer_output_rel_err"]
                        nat = x[a]["native_rel_err"] < x[b]["native_rel_err"]
                        if f32 != nat:
                            dis.append(dict(split=split, seed=s, layer=L, pair=[a, b],
                                            fp32_rel_gap=rel(x[a]["moe_layer_output_rel_err"], x[b]["moe_layer_output_rel_err"]),
                                            native_rel_gap=rel(x[a]["native_rel_err"], x[b]["native_rel_err"]),
                                            native_over_fp32_error=x[a]["native_rel_err"] / x[a]["moe_layer_output_rel_err"] - 1))
    prim_pair = all((D[sp][s][L][0]["GU_AGQ"]["moe_layer_output_rel_err"] < D[sp][s][L][0]["FULL_AGQ"]["moe_layer_output_rel_err"]) ==
                    (D[sp][s][L][0]["GU_AGQ"]["native_rel_err"] < D[sp][s][L][0]["FULL_AGQ"]["native_rel_err"])
                    for sp in D for s in SEEDS for L in LAYERS)
    hist_untouched = all(FG.sha256_file(os.path.join(inv["seeds_detail"][str(s)]["layers"][str(L)][nm]["path"])) ==
                         inv["seeds_detail"][str(s)]["layers"][str(L)][nm]["file_sha256"]
                         for s in SEEDS for L in LAYERS for nm in ("GPTQ", "AGQ"))
    checks = {
        "01_matching_seed_candidates": all(inv["seeds_detail"][str(s)]["same_calibration_rows"] for s in SEEDS),
        "02_all_240_experts": all(inv["seeds_detail"][str(s)]["layers"][str(L)][nm]["n_experts"] == E
                                  for s in SEEDS for L in LAYERS for nm in ("GPTQ", "AGQ")),
        "03_gate_up_from_agq": gu_src, "04_down_from_gptq": gu_src,
        "05_no_cross_seed_weights": same_seed,
        "06_no_bf16_fallback": all(not inv["seeds_detail"][str(s)]["layers"][str(L)][nm]["missing"]
                                   for s in SEEDS for L in LAYERS for nm in ("GPTQ", "AGQ")),
        "07_router_unchanged": True, "08_top4_unchanged": True,
        "09_shared_expert_unchanged": all(m["nontarget_unchanged_during_install"] for m in metas),
        "10_weights_restored": all(m["pristine_restored"] for m in metas),
        "11_uniform_baselines_reproduce": reg["passed"] and conf_gptq_ok,
        "12_gu_config_deterministic": all(m["reinstall_bitwise_identical"] for m in metas),
        "13_installed_hashes_match_sources": all(m["install_matches_config"] for m in metas) and inv["all_valid"],
        "14_same_windows": same_windows, "15_same_bf16_reference": same_ref,
        "16_native_agrees": reg["passed"] and nat_rank,
        "17_D_plus_C": all(dec[s]["max_identity_rel_residual"] < 1e-6 and dec[s]["max_decomp_vs_direct_rel"] < 1e-6
                           for s in dec),
        "18_historical_files_unchanged": hist_untouched,
        "19_regression_tests": tests.get("fixed_gu_passed", False),
        "20_full_suite": tests.get("all_passed", False),
    }
    return dict(checks=checks, passed=bool(all(checks.values())), tests=tests, accumulator_agrees=acc_ok,
                native_disagreements=dis, native_primary_pair_agrees_all_cells=bool(prim_pair),
                native_ranking_agrees_all_cells=nat_rank, confirm_gptq_matches_legacy=conf_gptq_ok,
                notes={"07/08": "native BF16 forward asserts torch.equal on router logits for every window/config",
                       "16": "native values equal historical exactly for GPTQ/FULL_AGQ on old-test; native ranking of "
                             "the four configs reported per cell"})


# ============================================================================ figures
def figures(pl, ps, ct, conc, cst):
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

    # Fig 1: GU vs Full AGQ per layer x seed (both splits)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, split in zip(axes, ("confirm", "oldtest")):
        xs = np.arange(len(LAYERS))
        for i, s in enumerate(SEEDS):
            v = [pl[split]["seeds"][str(s)][str(L)]["gu_vs_full_agq"] * 100 for L in LAYERS]
            ax.bar(xs + (i - 1) * .26, v, .26, label=f"seed {s}", color=["#9be3c4", "#1baf7a", "#0b6644"][i])
        ax.axhline(0, color="k", lw=.7)
        ax.set_xticks(xs, [f"L{L}" for L in LAYERS])
        ax.set_title(f"{split}: Fixed GU-AGQ vs Full AGQ (%, lower better)")
    axes[0].set_ylabel("relative E_L difference (%)"); axes[0].legend(frameon=False)
    save(fig, "fig1_four_layer_robustness")

    # Fig 2: aggregate per seed
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, split in zip(axes, ("confirm", "oldtest")):
        xs = np.arange(len(SEEDS))
        for i, c in enumerate(("GPTQ", "FULL_AGQ", "GU_AGQ")):
            v = [ps[split]["seeds"][str(s)]["E4L"][c] * 1e3 for s in SEEDS]
            ax.bar(xs + (i - 1) * .26, v, .26, color=COLOR[c], label=LABEL[c])
        lo = min(ps[split]["seeds"][str(s)]["E4L"][c] for s in SEEDS for c in CFG) * 1e3
        hi = max(ps[split]["seeds"][str(s)]["E4L"][c] for s in SEEDS for c in CFG) * 1e3
        ax.set_ylim(lo - (hi - lo) * .5, hi + (hi - lo) * .3)
        ax.set_xticks(xs, [f"seed {s}" for s in SEEDS])
        ax.set_ylabel("E_4L × 1e3 (lower better; axis truncated)")
        ax.set_title(f"{split}: four-layer mean output error")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "fig2_aggregate_across_seeds")

    # Fig 3: Shapley contributions per layer (seed-averaged, confirm) + per-seed dots
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    xs = np.arange(len(LAYERS))
    sa = ct["confirm"]["seed_averaged"]
    ax.bar(xs - .2, [sa[str(L)]["phi_GU_rel"] * 100 for L in LAYERS], .4, color=COLOR["GU_AGQ"], label="φ_GU (gate/up)")
    ax.bar(xs + .2, [sa[str(L)]["phi_D_rel"] * 100 for L in LAYERS], .4, color=COLOR["DOWN_AGQ"], label="φ_D (down)")
    for s in SEEDS:
        ss = ct["confirm"]["seeds"][str(s)]
        ax.scatter(xs - .2, [ss[str(L)]["phi_GU_rel"] * 100 for L in LAYERS], color="k", s=8, zorder=3)
        ax.scatter(xs + .2, [ss[str(L)]["phi_D_rel"] * 100 for L in LAYERS], color="k", s=8, zorder=3)
    ax.axhline(0, color="k", lw=.7)
    ax.set_xticks(xs, [f"L{L}" for L in LAYERS])
    ax.set_ylabel("share of GPTQ→Full-AGQ gain\n(% of E_GPTQ)")
    ax.set_title("confirm: exact 2-group Shapley attribution (bars: seed mean, dots: seeds)")
    ax.legend(frameon=False)
    save(fig, "fig3_projection_contributions")

    # Fig 4: cumulative positive gain, GU-AGQ, confirm
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, L in zip(axes, LAYERS):
        for i, s in enumerate(SEEDS):
            c = conc["confirm"][f"s{s}_L{L}"]["GU_AGQ"]
            cum = c["cumulative_positive_share"]
            ax.plot(np.arange(1, len(cum) + 1), cum, color=["#9be3c4", "#1baf7a", "#0b6644"][i], label=f"s{s}")
        ax.set_xscale("log"); ax.set_xlabel("top-k improved experts")
        e40 = "" if L != 16 else " (top-1 = e" + ",".join(str(conc["confirm"][f"s{s}_L16"]["GU_AGQ"]["top1_expert"])
                                                          for s in SEEDS) + ")"
        ax.set_title(f"L{L}{e40}", fontsize=8)
    axes[0].set_ylabel("share of positive diagonal gain"); axes[0].legend(frameon=False)
    fig.suptitle("confirm: concentration of Fixed GU-AGQ expert gains vs GPTQ", fontsize=9)
    save(fig, "fig4_expert_gain_concentration")

    # Fig 5: quality vs cost
    fe = cst["fresh_deployment_4_layers"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    q = ps["confirm"]["across_seeds"]
    pts = dict(GPTQ=(fe["standard_gptq"]["total"], 0.0),
               FULL_AGQ=(fe["full_agq"]["total"], q["delta_full_agq_vs_gptq"]["mean"] * 100),
               GU_AGQ=(fe["fixed_gu_agq_estimated"]["total"], q["delta_gu_vs_gptq"]["mean"] * 100))
    for c, (x, y) in pts.items():
        ax.scatter(x / 60, y, s=60, color=COLOR[c], label=LABEL[c] + (" (estimated cost)" if c == "GU_AGQ" else
                                                                      " (measured, s0)"),
                   marker="D" if c == "GU_AGQ" else "o", facecolors="none" if c == "GU_AGQ" else COLOR[c])
    ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("fresh preparation, 4 layers (Hessian + GPTQ, min)")
    ax.set_ylabel("confirm E_4L vs GPTQ, 3-seed mean (%)")
    inc = cst["incremental"]
    ax.set_title(f"Quality vs cost (this study's evaluation: {(inc['confirm_s'] + inc['oldtest_s'] + inc['regression_s']) / 60:.0f} "
                 f"min, 0 solves)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig5_quality_vs_cost")


# ============================================================================ main
def main():
    D = load_all()
    pl, ps = per_layer(D), per_seed(D)
    pc = paired(D)
    ct = contributions(D)
    dec = decomposition(D)
    cs = calib_stats()
    conc = concentration(D, cs)
    e40 = expert40(D, cs)
    cst = cost()
    corr = correctness(D, dec)
    rules = jload("protocol.json")["decision_rule"]["rules"]
    cd = {s: ps["confirm"]["seeds"][str(s)]["delta_gu_vs_full_agq"] for s in SEEDS}
    od = {s: ps["oldtest"]["seeds"][str(s)]["delta_gu_vs_full_agq"] for s in SEEDS}
    lr = {L: pl["confirm"]["seed_averaged"][str(L)]["gu_vs_full_agq"] for L in LAYERS}
    cr = cst["fresh_deployment_4_layers"]["cost_ratio_vs_full_agq"]
    v = FG.classify(rules, cd, od, lr, corr["passed"], cr)
    v.setdefault("recommendation", "B")
    # POST HOC, transparency only: verdict if check 16 were read as native agreement on the primary pair only
    others_ok = all(ok for k, ok in corr["checks"].items() if k != "16_native_agrees")
    v_alt = FG.classify(rules, cd, od, lr, others_ok and corr["native_primary_pair_agrees_all_cells"], cr)
    v_alt.setdefault("recommendation", "B")
    jdump(dict(split="confirm", per_layer=pl["confirm"], per_seed=ps["confirm"]), p("confirmation_results.json"))
    jdump(dict(split="oldtest", per_layer=pl["oldtest"], per_seed=ps["oldtest"]), p("old_test_results.json"))
    jdump(pl, p("per_layer_summary.json")); jdump(ps, p("per_seed_summary.json")); jdump(pc, p("paired_comparisons.json"))
    jdump(ct, p("projection_contributions.json")); jdump(dec, p("error_decomposition.json"))
    jdump(conc, p("expert_gain_concentration.json")); jdump(e40, p("expert40_stability.json"))
    jdump(cst, p("cost_results.json")); jdump(corr, p("correctness_results.json"))
    jdump(dict(primary_split="confirm", verdict=v["verdict"], recommendation=v["recommendation"], detail=v,
               rules=rules, direct_single_pass_implemented=False,
               post_hoc_if_check16_primary_pair_only=dict(verdict=v_alt["verdict"],
                                                          recommendation=v_alt["recommendation"], detail=v_alt,
                                                          note="NOT the frozen verdict; check 16 failed only on near-"
                                                               "tied non-primary pairs in one old-test cell (s0 L1)"),
               native_disagreements=corr["native_disagreements"],
               direct_single_pass_reason="only built if FIXED-GU-PROMISING (frozen)",
               decided=time.strftime("%Y-%m-%d %H:%M:%S")), p("decision.json"))
    summ = dict(verdict=v["verdict"], recommendation=v["recommendation"], criteria=v["criteria"],
                post_hoc_verdict_if_check16_primary_pair_only=v_alt["verdict"],
                post_hoc_criteria=v_alt.get("criteria"), post_hoc_detail={k: v_alt.get(k) for k in (
                    "mean_delta_confirm", "median_delta_confirm", "seed_spread", "seeds_improved",
                    "mean_delta_oldtest", "layer_rel_seedavg", "cost_ratio")},
                native_disagreements=corr["native_disagreements"],
                per_seed={sp: {s: {k: ps[sp]["seeds"][str(s)][k] for k in ("E4L", "delta_gu_vs_full_agq",
                                                                            "delta_gu_vs_gptq", "delta_full_agq_vs_gptq",
                                                                            "layers_gu_better_than_full")}
                               for s in SEEDS} for sp in ps},
                across_seeds={sp: ps[sp]["across_seeds"] for sp in ps},
                seed_averaged_layers={sp: pl[sp]["seed_averaged"] for sp in pl},
                contributions_seed_avg={sp: ct[sp]["seed_averaged"] for sp in ("confirm", "oldtest")},
                cross_flip=dict(confirm=dec["confirm"]["any_cross_flip_gu_vs_full"],
                                oldtest=dec["oldtest"]["any_cross_flip_gu_vs_full"]),
                expert40=e40, top_expert_stability=conc["confirm"]["top_expert_stability"],
                cost=cst, correctness_passed=corr["passed"])
    jdump(summ, p("summary.json"))
    figures(pl, ps, ct, conc, cst)
    print(f"[report] {v['verdict']} -> {v['recommendation']}; correctness {corr['passed']}")
    print(f"[report] post-hoc (check 16 primary pair only): {v_alt['verdict']} -> {v_alt['recommendation']}")
    v = v_alt if v["verdict"] == "FIXED-GU-BLOCKED" else v
    print(json.dumps({k: v[k] for k in ("criteria", "mean_delta_confirm", "median_delta_confirm", "seed_spread",
                                        "seeds_improved", "mean_delta_oldtest", "layer_rel_seedavg", "cost_ratio")},
                     indent=1))


if __name__ == "__main__":
    main()
