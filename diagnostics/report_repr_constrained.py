"""Representativeness-constrained Coverage-LP: validation and analysis (CPU).

    --validate F       one coverage_lp_repr run: frozen settings, B unique,
                       prescribed ordered warm start, full pool, constraint
                       satisfied on the DISCRETE set, same evaluation universe
    --check-warm SEED  the new run and the 4 reused baselines of SEED share the
                       exact ordered warm start
    (default)          <outdir>/summary.{md,json}

Conditions (seed s; every one starts from the same ordered 32-sequence W_s):
  FR Full Random   FC Full Coverage-LP   TR Typical Random   TC Typical Coverage-LP
  HC Random-half Coverage-LP (context)   RC Coverage-LP-Repr (new; full pool)
Baselines are REUSED from runs_typicality_matched/ after validation (same
warm-start ids/hash, frozen GPTQ, model, split, evaluation universe, schema).

Prespecified decision (written before any coverage_lp_repr result existed;
EFFECT = 0.5 % relative is a practical-size reference, not a significance test):
  PASS     RC < TR and RC < TC on 3/3 seeds, mean relative reduction vs TR
           >= EFFECT, and no per-layer mean or expert-p95 regression vs TR
           larger than EFFECT
  PARTIAL  not PASS, and RC < FC on 3/3 seeds with mean relative reduction
           vs FC >= EFFECT (the constraint mitigates the coverage penalty but
           active allocation adds no practical value over Typical Random)
  NO-GAIN  otherwise
Three seeds = three paired replicates; experts are not independent
replicates; the test split has been inspected before (exploratory result).
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import report_minimal_allocation as RM               # noqa: E402
import report_typicality as RT                       # noqa: E402
import report_typicality_matched as RTM              # noqa: E402
from moe_calib import repr_constraint as RC          # noqa: E402
from moe_calib import typicality as TY               # noqa: E402
from moe_calib.frozen import load_frozen             # noqa: E402

SEEDS, B, WARM, EFFECT = (0, 1, 2), 128, 32, 0.005
BASE = {"FR": ("full", "random"), "FC": ("full", "coverage_lp"), "TR": ("typical_half", "random"),
        "TC": ("typical_half", "coverage_lp"), "HR": ("random_half", "random"),
        "HC": ("random_half", "coverage_lp")}
NAMES = {"FR": "Full Random", "FC": "Full Coverage-LP", "TR": "Typical Random",
         "TC": "Typical Coverage-LP", "HR": "Random-half Random", "HC": "Random-half Coverage-LP",
         "RC": "Coverage-LP-Repr"}
FORWARD_EQUIV_PROFILE_S = None   # read from the runs (meta.forward_s_per_seq x 2,000)


def warm_file(d):
    return json.load(open(os.path.join(d, "warm_starts.json")))


def rc_path(outdir, seed, ws):
    return os.path.join(outdir, f"e3_coverage_lp_repr_B{B}_b4_d1_s{seed}_warm-{ws['seeds'][str(seed)]['hash'][:8]}.json")


def validate(path, outdir, cfg, ws=None):
    ws = ws or warm_file(outdir)
    r = RM.validate(path, cfg)
    s = r["summary"]
    want = ws["seeds"][str(s["seed"])]
    errs = []
    if s["strategy"] != "coverage_lp_repr":
        errs.append(f"strategy {s['strategy']}")
    wf = s.get("warm_start_fixed") or {}
    if wf.get("hash") != want["hash"] or r["warm_start"] != want["ids"] or r["chosen"][:WARM] != want["ids"]:
        errs.append("warm start differs from the prescribed ordered W_s")
    post = r["chosen"][WARM:]
    if len(post) != B - WARM or set(post) & set(want["ids"]) or len(set(r["chosen"])) != B:
        errs.append("not exactly 96 unique post-warm sequences")
    if s.get("candidate_subset") is not None:
        errs.append("coverage_lp_repr must select from the full pool")
    rc = s.get("repr_constraint") or {}
    if not rc.get("satisfied") or rc.get("n_post") != B - WARM or rc.get("tau_spec") != "median":
        errs.append(f"representativeness constraint not satisfied / not prespecified: {rc}")
    if rc.get("scores_max_abs_diff") is None or rc["scores_max_abs_diff"] > 1e-9:
        errs.append("d(s) not verified against the typicality-score artifact")
    if s.get("n_experts_test_observed") != 240 or s.get("n_experts_quantized_total") != 240:
        errs.append("not all 240 experts quantized / evaluated")
    # recompute the constraint on the DISCRETE set from the routing cache
    z = np.load(os.path.join(outdir, "routing_profile_v2.npz"))
    d, _ = RC.pool_typicality(z["C"], len(z["layers"]), int(z["E"]))
    row = {int(x): i for i, x in enumerate(z["pool_ids"])}
    tau = float(np.median(d))
    m = float(np.mean([d[row[x]] for x in post]))
    if abs(tau - rc.get("tau", -1)) > 1e-12 or m > tau + RC.TOL:
        errs.append(f"recomputed post-warm mean d {m:.6f} vs tau {tau:.6f}")
    # split isolation: only original candidate-pool ids
    P = cfg["split"]["pool"]
    if not all(P[0] <= x < P[1] for x in r["chosen"]):
        errs.append("selected ids outside the calibration candidate pool (tune/test leakage)")
    if errs:
        raise SystemExit(f"INVALID {path}:\n  " + "\n  ".join(errs))
    return r


def load_seed(outdir, matched_dir, cfg, seed, need_rc=True):
    ws = warm_file(matched_dir)
    runs = {}
    for k, (pool, st) in BASE.items():
        f = RTM.run_path(matched_dir, pool, st, seed, ws)
        runs[k] = RTM.validate(f, matched_dir, cfg, ws)
    f = rc_path(outdir, seed, ws)
    if os.path.exists(f) or need_rc:
        runs["RC"] = validate(f, outdir, cfg, warm_file(outdir))
    return runs, ws


def check_warm(outdir, matched_dir, cfg, seed):
    if warm_file(outdir) != warm_file(matched_dir):
        raise SystemExit("warm_starts.json differs from the matched experiment's")
    runs, ws = load_seed(outdir, matched_dir, cfg, seed)
    ref = ws["seeds"][str(seed)]["ids"]
    bad = [k for k, r in runs.items() if r["chosen"][:WARM] != ref]
    univ = {json.dumps(r["eval_universe"]) for r in runs.values()}
    keys = ("bits", "percdamp", "layers", "split", "K", "eval_split", "B")
    cfgs = {json.dumps({k: r["summary"][k] for k in keys}) for r in runs.values()}
    if bad or len(univ) != 1 or len(cfgs) != 1:
        raise SystemExit(f"seed {seed}: warm start differs in {bad}; universes {len(univ)}; configs {len(cfgs)}")
    print(f"seed {seed}: {len(runs)} runs share warm start {ws['seeds'][str(seed)]['hash']}, "
          "one evaluation universe and identical GPTQ settings")


# --------------------------------------------------------------- analysis

def count_profile(n, layers, E):
    out = {}
    for li, L in enumerate(layers):
        x = np.asarray(n[li * E:(li + 1) * E], float)
        out[str(L)] = dict(min=float(x.min()), p10=float(np.quantile(x, .1)), median=float(np.median(x)),
                           p90=float(np.quantile(x, .9)), max=float(x.max()), gini=float(RT.gini(x)))
    return out


def run_metrics(r, C, row, ref, scores, layers, E, typ, ws_ids):
    sm = r["summary"]
    m = RT.set_metrics(r, C, row, ref, scores, layers, E)
    post = r["chosen"][WARM:]
    traj = r["trajectory"]
    lp = bool(traj)
    out = dict(recon_mean=sm["recon_mean"], expert_median=sm["recon_expert_median"],
               expert_p90=sm["recon_expert_p90"], expert_p95=sm["recon_expert_p95"],
               per_layer=sm["recon_per_layer"], chosen=r["chosen"],
               frac_post_outside_typical=float(np.mean([x not in typ for x in post])),
               n_post_outside_typical=int(sum(x not in typ for x in post)),
               coverage_objective=(float(sum(t["round_objective"] for t in traj)) if lp else None),
               count_profile=count_profile(r["selected_counts"], layers, E),
               selection_s=sm["cost"]["selection_s"], hessian_s=sm["cost"]["hessian_replay_s"],
               gptq_s=sm["cost"]["gptq_s"], typicality_s=sm["cost"].get("typicality_s", 0.0),
               lp_round_s=(float(sum(t.get("round_select_s", 0.0) for t in traj)) if lp else None),
               routing_full_forward_equiv_s=sm["cost"].get("routing_full_forward_equiv_s", 0.0),
               calib_full_forward_equiv_s=sm["cost"]["calib_full_forward_equiv_s"],
               equiv_extra_random_seqs=sm["cost"]["equiv_extra_random_seqs"],
               peak_gpu_mem_gb=sm["peak_gpu_mem_gb"],
               set_entropy_mean=float(np.mean([
                   -(lambda q: (q[q > 0] * np.log(q[q > 0])).sum())(
                       C[[row[x] for x in r["chosen"]]][:, li * E:(li + 1) * E].sum(0)
                       / C[[row[x] for x in r["chosen"]]][:, li * E:(li + 1) * E].sum())
                   for li in range(len(layers))])), **m)
    if sm.get("repr_constraint"):
        rc = sm["repr_constraint"]
        out.update(tau=rc["tau"], constraint_tightness=rc["tightness"], constraint_slack=rc["slack"],
                   constraint_satisfied=rc["satisfied"],
                   n_rounding_repairs=int(sum(t["n_rounding_repairs"] for t in traj)),
                   n_rounds_naive_rounding_violates=int(sum(bool(t["naive_rounding_violates"]) for t in traj)),
                   rounding_objective_loss=float(sum(t["rounding_objective_loss"] or 0 for t in traj)),
                   rounding_objective_loss_rel=float(sum(t["rounding_objective_loss"] or 0 for t in traj)
                                                     / max(sum(t["naive_lp_objective"] or 0 for t in traj), 1e-12)),
                   n_fill=int(sum(t["n_zero_marginal_random_fill"] + t["n_random_fallback_sequences"] for t in traj)),
                   n_lp_selected=int(sum(t["n_lp_positive_value_selected"] for t in traj)))
    return out


def stat(d):
    v = np.array(list(d.values()))
    return dict(by_seed=d, mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                n_negative=int((v < 0).sum()))


def build(outdir, matched_dir, pool_dir, cfg):
    z = np.load(os.path.join(outdir, "routing_profile_v2.npz"))
    layers, E = z["layers"].tolist(), int(z["E"])
    C = z["C"].astype(float)
    row = {int(s): i for i, s in enumerate(z["pool_ids"])}
    ref = TY.pool_reference(C, len(layers), E)
    scores = TY.sequence_scores(C, len(layers), E, ref)
    typ = set(json.load(open(os.path.join(pool_dir, "pools.json")))["pools"]["typical_half"]["ids"])
    feas = json.load(open(os.path.join(outdir, "feasibility.json")))
    per, seeds, missing = {}, [], []
    for sd in SEEDS:
        if not os.path.exists(rc_path(outdir, sd, warm_file(outdir))):
            missing.append(sd)
            continue
        check_warm(outdir, matched_dir, cfg, sd)
        runs, ws = load_seed(outdir, matched_dir, cfg, sd)
        per[sd] = {k: run_metrics(r, C, row, ref, scores, layers, E, typ, ws["seeds"][str(sd)]["ids"])
                   for k, r in runs.items()}
        seeds.append(sd)
    res = dict(seeds=seeds, missing_seeds=missing, feasibility=feas, effect_reference=EFFECT,
               conditions=NAMES, groups={}, paired={}, overlap={}, mechanism={}, runs={})
    if not seeds:
        return res
    keys = [k for k in per[seeds[0]]]
    scal = ("recon_mean", "expert_median", "expert_p90", "expert_p95", "d_mean", "d_post_warm",
            "entropy_mean", "set_entropy_mean", "js_set_vs_original_pool", "median_expert_tokens",
            "gini_expert_tokens", "n_low_exposure_experts", "median_sources_per_expert",
            "median_effective_sources_per_expert", "mean_effective_sources_per_expert",
            "frac_post_outside_typical", "selection_s", "hessian_s", "gptq_s", "typicality_s",
            "routing_full_forward_equiv_s", "calib_full_forward_equiv_s", "equiv_extra_random_seqs")
    for k in keys:
        rs = [per[sd][k] for sd in seeds]
        g = dict(name=NAMES[k], recon_by_seed={str(sd): per[sd][k]["recon_mean"] for sd in seeds},
                 recon_sd=float(np.std([r["recon_mean"] for r in rs], ddof=1)) if len(rs) > 1 else None)
        for f in scal:
            g[f] = float(np.mean([r[f] for r in rs]))
        g["peak_gpu_mem_gb"] = float(max(r["peak_gpu_mem_gb"] or 0 for r in rs))
        for f in ("per_layer", "set_routing_l1_by_layer"):
            g[f] = {str(L): float(np.mean([r[f][str(L)] for r in rs])) for L in layers}
        g["count_profile"] = {str(L): {q: float(np.mean([r["count_profile"][str(L)][q] for r in rs]))
                                       for q in rs[0]["count_profile"][str(L)]} for L in layers}
        for f in ("coverage_objective", "lp_round_s", "constraint_tightness", "n_rounding_repairs",
                  "rounding_objective_loss", "rounding_objective_loss_rel", "n_fill", "n_lp_selected",
                  "n_rounds_naive_rounding_violates", "tau"):
            if rs[0].get(f) is not None:
                g[f] = float(np.mean([r[f] for r in rs])) if "n_" not in f else int(sum(r[f] for r in rs))
        res["groups"][k] = g
    Ev = lambda k, sd: per[sd][k]["recon_mean"]
    for b in ("TR", "TC", "FR", "FC", "HC"):
        diff = {str(sd): Ev("RC", sd) - Ev(b, sd) for sd in seeds}
        rel = {str(sd): diff[str(sd)] / Ev(b, sd) for sd in seeds}
        pl = {str(L): stat({str(sd): (per[sd]["RC"]["per_layer"][str(L)] - per[sd][b]["per_layer"][str(L)])
                            / per[sd][b]["per_layer"][str(L)] for sd in seeds}) for L in layers}
        tail = {q: stat({str(sd): (per[sd]["RC"][q] - per[sd][b][q]) / per[sd][b][q] for sd in seeds})
                for q in ("expert_median", "expert_p90", "expert_p95")}
        res["paired"][f"RC_vs_{b}"] = dict(abs=stat(diff), rel=stat(rel), per_layer_rel=pl, tail_rel=tail)
    for b in ("FC", "TC", "TR", "FR"):
        res["overlap"][f"RC_vs_{b}"] = {str(sd): len(set(per[sd]["RC"]["chosen"]) & set(per[sd][b]["chosen"])) / B
                                        for sd in seeds}
        res["overlap"][f"RC_vs_{b}_post_warm"] = {
            str(sd): len(set(per[sd]["RC"]["chosen"][WARM:]) & set(per[sd][b]["chosen"][WARM:])) / (B - WARM)
            for sd in seeds}
    G = res["groups"]
    res["mechanism"] = dict(
        q1_balance_better_than_TR=dict(gini_RC=G["RC"]["gini_expert_tokens"], gini_TR=G["TR"]["gini_expert_tokens"],
                                       low_RC=G["RC"]["n_low_exposure_experts"], low_TR=G["TR"]["n_low_exposure_experts"]),
        q2_atypicality_vs_TR=dict(d_post_RC=G["RC"]["d_post_warm"], d_post_TR=G["TR"]["d_post_warm"],
                                  d_post_TC=G["TC"]["d_post_warm"], d_post_FC=G["FC"]["d_post_warm"],
                                  js_RC=G["RC"]["js_set_vs_original_pool"], js_TR=G["TR"]["js_set_vs_original_pool"]),
        q3_outside_typical=dict(frac_RC=G["RC"]["frac_post_outside_typical"]),
        q5_eff_sources=dict(RC=G["RC"]["median_effective_sources_per_expert"],
                            TR=G["TR"]["median_effective_sources_per_expert"],
                            TC=G["TC"]["median_effective_sources_per_expert"],
                            FC=G["FC"]["median_effective_sources_per_expert"]),
        q6_layers_better_than_TR=[L for L, v in res["paired"]["RC_vs_TR"]["per_layer_rel"].items() if v["mean"] < 0],
        q6_layers_better_than_FC=[L for L, v in res["paired"]["RC_vs_FC"]["per_layer_rel"].items() if v["mean"] < 0])
    for sd in seeds:
        res["runs"][str(sd)] = {k: {f: v for f, v in m.items() if f != "chosen"} for k, m in per[sd].items()}
    if len(seeds) == 3:
        P = res["paired"]
        tr, tc, fc = P["RC_vs_TR"], P["RC_vs_TC"], P["RC_vs_FC"]
        worst_layer = max(v["mean"] for v in tr["per_layer_rel"].values())
        worst_p95 = tr["tail_rel"]["expert_p95"]["mean"]
        if (tr["abs"]["n_negative"] == 3 and tc["abs"]["n_negative"] == 3 and -tr["rel"]["mean"] >= EFFECT
                and worst_layer <= EFFECT and worst_p95 <= EFFECT):
            lab = "REPR-ALLOCATION-PASS"
        elif fc["abs"]["n_negative"] == 3 and -fc["rel"]["mean"] >= EFFECT:
            lab = "REPR-ALLOCATION-PARTIAL"
        else:
            lab = "REPR-ALLOCATION-NO-GAIN"
        res["decision"] = dict(label=lab, rel_vs_TR=tr["rel"]["mean"], n_better_TR=tr["abs"]["n_negative"],
                               rel_vs_TC=tc["rel"]["mean"], n_better_TC=tc["abs"]["n_negative"],
                               rel_vs_FC=fc["rel"]["mean"], n_better_FC=fc["abs"]["n_negative"],
                               worst_layer_rel_vs_TR=worst_layer, p95_rel_vs_TR=worst_p95)
    return res


# --------------------------------------------------------------- markdown

def pct(x):
    return f"{100 * x:+.2f} %"


def write_md(res, path):
    L = []
    w = L.append
    G, seeds = res["groups"], res["seeds"]
    w("# Representativeness-constrained Coverage-LP — summary\n")
    w("Generated by `diagnostics/report_repr_constrained.py`. Mean expert relative reconstruction error "
      "on the fixed test-split evaluation universe (240 experts, layers 1/8/16/23); lower is better. "
      "Partial-model, non-sequential expert GPTQ (attention and shared experts BF16). "
      "**Exploratory**: the test split was inspected by earlier allocation/typicality experiments.\n")
    if res.get("missing_seeds"):
        w(f"Missing coverage_lp_repr seeds: {res['missing_seeds']}\n")
    if "decision" in res:
        dd = res["decision"]
        w(f"**Decision (prespecified rule): {dd['label']}**\n")
        if dd["label"] == "REPR-ALLOCATION-PARTIAL" and not abs(dd["rel_vs_TR"]) < EFFECT:
            w(f"Note: the task specification defines PARTIAL as improving over Full Coverage-LP while *matching* "
              f"Typical Random. Coverage-LP-Repr is {pct(dd['rel_vs_TR'])} vs Typical Random "
              f"({dd['n_better_TR']}/3 seeds better), beyond the {100 * EFFECT:.1f} % reference, so under the task's "
              f"definitions the outcome is **REPR-ALLOCATION-NO-GAIN**. The coded rule is left unchanged.\n")
    fe = res["feasibility"]
    w("## Constraint\n")
    w(f"- tau = median d(s) over the 2,000 original candidates = **{fe['tau']:.6f}**; d(s) recomputed from "
      f"the routing cache matches `typicality_scores.csv` (max abs diff {fe['d_max_abs_diff_vs_scores_csv']:.1e}); "
      f"typical half reproduced: {fe['typical_half_reproduced']}.")
    w("- Bound: mean d(s) over the 96 post-warm picks <= tau (shared budget 96 tau across both rounds).\n")
    if not seeds:
        open(path, "w").write("\n".join(L) + "\n")
        return
    order = [k for k in ("FR", "FC", "TR", "TC", "HC", "RC") if k in G]
    w("## Reconstruction\n")
    w("| condition | mean | SD (seeds) | " + " | ".join(f"s{sd}" for sd in seeds) +
      " | expert median | p90 | p95 | " + " | ".join(f"L{L}" for L in G["RC"]["per_layer"]) + " |")
    w("|---" * (4 + len(seeds) + 3 + len(G["RC"]["per_layer"]) - 1) + "|")
    for k in order:
        g = G[k]
        w(f"| {'**' + NAMES[k] + '**' if k == 'RC' else NAMES[k]} | {g['recon_mean']:.6f} | "
          f"{'—' if g['recon_sd'] is None else format(g['recon_sd'], '.6f')} | " + " | ".join(f"{g['recon_by_seed'][str(sd)]:.6f}" for sd in seeds) +
          f" | {g['expert_median']:.6f} | {g['expert_p90']:.6f} | {g['expert_p95']:.6f} | " +
          " | ".join(f"{v:.6f}" for v in g["per_layer"].values()) + " |")
    w("\n## Paired differences: Coverage-LP-Repr − baseline (per seed)\n")
    w("| baseline | Δ ×1e-6 per seed | mean Δ ×1e-6 | rel per seed | mean rel | seeds RC better | "
      "per-layer mean rel (L1/L8/L16/L23) | p95 rel |")
    w("|---|---|---|---|---|---|---|---|")
    for b in ("TR", "TC", "FR", "FC", "HC"):
        p = res["paired"][f"RC_vs_{b}"]
        w(f"| {NAMES[b]} | " + " / ".join(f"{1e6 * v:+.1f}" for v in p["abs"]["by_seed"].values()) +
          f" | {1e6 * p['abs']['mean']:+.1f}" + ("" if p['abs']['sd'] is None else f" ± {1e6 * p['abs']['sd']:.1f}") + " | " +
          " / ".join(pct(v) for v in p["rel"]["by_seed"].values()) + f" | {pct(p['rel']['mean'])} | "
          f"{p['abs']['n_negative']}/{len(seeds)} | " +
          " / ".join(pct(v["mean"]) for v in p["per_layer_rel"].values()) +
          f" | {pct(p['tail_rel']['expert_p95']['mean'])} |")
    w("\n## Selection and representativeness (mean over seeds)\n")
    w("| condition | d all 128 | d post-warm | set JS | set L1 (L1/L8/L16/L23) | seq entropy | set entropy | "
      "post-warm outside typical half | coverage objective |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k in order:
        g = G[k]
        co = g.get("coverage_objective")
        w(f"| {NAMES[k]} | {g['d_mean']:.4f} | {g['d_post_warm']:.4f} | {g['js_set_vs_original_pool']:.5f} | " +
          " / ".join(f"{v:.4f}" for v in g["set_routing_l1_by_layer"].values()) +
          f" | {g['entropy_mean']:.4f} | {g['set_entropy_mean']:.4f} | {100 * g['frac_post_outside_typical']:.1f} % | "
          f"{'—' if co is None else f'{co:.1f}'} |")
    g = G["RC"]
    w(f"\nCoverage-LP-Repr constraint: tightness (spent / budget) {g['constraint_tightness']:.4f}; rounds where "
      f"naive rounding of the LP solution would violate the round cap: {g['n_rounds_naive_rounding_violates']} / "
      f"{2 * len(seeds)}; rounding repairs {g['n_rounding_repairs']}; rounding objective loss "
      f"{100 * g['rounding_objective_loss_rel']:.3f} % of the naive-rounding objective; LP picks "
      f"{g['n_lp_selected']}, fill {g['n_fill']}.\n")
    w("Selected-set overlap with Coverage-LP-Repr (all 128 / post-warm 96, mean over seeds): " + "; ".join(
        f"{NAMES[b]} {np.mean(list(res['overlap'][f'RC_vs_{b}'].values())):.3f} / "
        f"{np.mean(list(res['overlap'][f'RC_vs_{b}_post_warm'].values())):.3f}" for b in ("FC", "TC", "TR", "FR")) + "\n")
    w("## Expert exposure (mean over seeds)\n")
    w("| condition | median tokens / expert | Gini | low-exposure experts | median sources | median effective sources |")
    w("|---|---|---|---|---|---|")
    for k in order:
        g = G[k]
        w(f"| {NAMES[k]} | {g['median_expert_tokens']:.0f} | {g['gini_expert_tokens']:.4f} | "
          f"{g['n_low_exposure_experts']:.1f} | {g['median_sources_per_expert']:.1f} | "
          f"{g['median_effective_sources_per_expert']:.1f} |")
    w("\nPer-layer routed-token distribution (min / p10 / median / p90 / max, Gini):\n")
    w("| condition | " + " | ".join(f"L{L}" for L in G["RC"]["count_profile"]) + " |")
    w("|---" * (1 + len(G["RC"]["count_profile"])) + "|")
    for k in order:
        cp = G[k]["count_profile"]
        w(f"| {NAMES[k]} | " + " | ".join(
            f"{v['min']:.0f}/{v['p10']:.0f}/{v['median']:.0f}/{v['p90']:.0f}/{v['max']:.0f}, {v['gini']:.3f}"
            for v in cp.values()) + " |")
    w("\n## Systems cost (mean per run)\n")
    w("| condition | typicality s | selection s (LP+rounding) | Hessian replay s | GPTQ s | peak GPU GB | "
      "routing-profile full-forward equiv s | extra random seqs at equal compute |")
    w("|---|---|---|---|---|---|---|---|")
    for k in order:
        g = G[k]
        w(f"| {NAMES[k]} | {g['typicality_s']:.2f} | {g['selection_s']:.2f} | {g['hessian_s']:.1f} | "
          f"{g['gptq_s']:.1f} | {g['peak_gpu_mem_gb']:.2f} | {g['routing_full_forward_equiv_s']:.0f} | "
          f"{g['equiv_extra_random_seqs']:.0f} |")
    w("\nThe routing profile (~357 s of full-model forwarding over the 2,000 candidates) is charged in the run "
      "JSON only to strategies that read it. Typical Random's row shows 0, but building the typical-half pool "
      "also needs that profile (d(s)), so its true overhead is the same ~357 s; only Full Random avoids it. "
      "Coverage-LP-Repr's higher mean GPTQ / Hessian wall time comes from seed 0 (597 s / 103 s), which ran "
      "while the 14-minute CPU test suite competed for host cores; seeds 1 and 2 took 280-284 s / 62-63 s, "
      "like the baselines. GPTQ itself is identical across conditions. "
      "Equal-B results do not establish equal-compute efficiency.\n")
    m = res["mechanism"]
    w("## Mechanism checks\n")
    w(f"1. Token-count balance vs Typical Random: Gini {m['q1_balance_better_than_TR']['gini_RC']:.4f} vs "
      f"{m['q1_balance_better_than_TR']['gini_TR']:.4f}; low-exposure experts "
      f"{m['q1_balance_better_than_TR']['low_RC']:.1f} vs {m['q1_balance_better_than_TR']['low_TR']:.1f}.")
    q = m["q2_atypicality_vs_TR"]
    w(f"2. Post-warm d: RC {q['d_post_RC']:.4f}, TR {q['d_post_TR']:.4f}, TC {q['d_post_TC']:.4f}, "
      f"FC {q['d_post_FC']:.4f}; set JS RC {q['js_RC']:.5f} vs TR {q['js_TR']:.5f}.")
    w(f"3. Post-warm picks outside the typical half: {100 * m['q3_outside_typical']['frac_RC']:.1f} %.")
    w(f"4. Reconstruction: see paired table (decision {res.get('decision', {}).get('label')}).")
    q = m["q5_eff_sources"]
    w(f"5. Median effective sources per expert: RC {q['RC']:.1f}, TR {q['TR']:.1f}, TC {q['TC']:.1f}, FC {q['FC']:.1f}.")
    w(f"6. Layers where RC beats TR on average: {m['q6_layers_better_than_TR']}; beats FC: {m['q6_layers_better_than_FC']}.")
    open(path, "w").write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs_repr_constrained")
    ap.add_argument("--matched-dir", default="runs_typicality_matched")
    ap.add_argument("--pool-dir", default="runs_typicality")
    ap.add_argument("--frozen", default="configs/frozen_b4.json")
    ap.add_argument("--validate", default=None)
    ap.add_argument("--check-warm", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_frozen(a.frozen)
    if a.validate:
        validate(a.validate, a.outdir, cfg)
        print(f"valid: {a.validate}")
        return
    if a.check_warm is not None:
        check_warm(a.outdir, a.matched_dir, cfg, a.check_warm)
        return
    res = build(a.outdir, a.matched_dir, a.pool_dir, cfg)
    json.dump(res, open(os.path.join(a.outdir, "summary.json"), "w"), indent=2)
    write_md(res, os.path.join(a.outdir, "summary.md"))
    print(open(os.path.join(a.outdir, "summary.md")).read())


if __name__ == "__main__":
    main()
