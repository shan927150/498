"""Extreme low-budget (B = 4 / 8) exposure-scarcity study — CPU only, no GPTQ.

Experiment A  Monte Carlo: 1,000 random calibration sets per (pool, B) from the
              full pool and the frozen typical half; real routed-token counts
              from routing_profile_v2.npz; per-expert lower-tail scarcity.
Experiment B  Matched selection: Full/Typical x Random/Coverage-LP, seeds 0/1/2,
              warm start W(seed,B) = first B/4 of the matched W(seed,128),
              schedules (1; 1+2) and (2; 3+3), through the UNCHANGED
              e3_allocation.select_lp_utility (u_e = 1/n_e, existing exploration).

Controller compatibility (caller side only, no code change): round_schedule
sets warm = min(B, max(K, B//4)). With the E3 default K = 4 the warm-start
floor is 4, which at B = 4 leaves no allocation round at all and at B = 8 gives
(4; 2+2). K has no other role in select_lp_utility, so the caller passes
K = B // 4, which makes the floor equal B // 4. For B >= 16 (B // 4 >= 4) this
is identical to K = 4, so the B = 16..128 schedules are unchanged.

    python diagnostics/extreme_low_budget.py [--outdir runs_extreme_low_budget]

Thresholds (32/64/128/256 tokens) are DESCRIPTIVE scarcity indicators, not
validated GPTQ sufficiency thresholds. No reconstruction error is read.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import low_budget as LB                    # noqa: E402  (shared helpers; unchanged)

BUDGETS, SEEDS, ROUNDS = (4, 8), (0, 1, 2), 2
THRESH = (32, 64, 128, 256)
N_MC = 1000
MC_SEEDS = {("full", 4): 2026111241, ("full", 8): 2026111281,          # independent fixed streams
            ("typical_half", 4): 2026111242, ("typical_half", 8): 2026111282}
QS = (0.05, 0.25, 0.5, 0.75, 0.95)
STRATS, NAMES = LB.STRATS, LB.NAMES
COL = {"FR": "#2a78d6", "FC": "#eda100", "TR": "#eb6834", "TC": "#1baf7a"}
SEQ_LEN, TOPK = 512, 4

# ---------------------------------------------------------------- prespecified decision gate
GATE = dict(scarcity_frac_below_128=0.10, p10_rel_gain=0.10, below128_rel_reduction=0.50, min_seeds=2)


def k_eff(B):
    """K passed to the unchanged controller so the warm-start floor is B // 4."""
    return max(1, B // 4)


def schedule(B):
    import e3_allocation as E3
    w, s = E3.round_schedule(B, k_eff(B), ROUNDS)
    return w, [r[2] for r in s]


# ---------------------------------------------------------------- metrics
def tail_metrics(n):
    """Lower-tail exposure metrics of one expert-count vector (60 or 240 experts)."""
    n = np.asarray(n, float)
    q = lambda p: float(np.quantile(n, p))
    out = dict(min=float(n.min()), p1=q(.01), p5=q(.05), p10=q(.10), median=q(.5), p90=q(.9),
               max=float(n.max()), mean=float(n.mean()), gini=LB.gini(n),
               cv=float(n.std() / n.mean()) if n.mean() > 0 else None, zero=int((n == 0).sum()))
    for t in THRESH:
        out[f"below_{t}"] = int((n < t).sum())
    out["frac_below_128"] = out["below_128"] / len(n)
    return out


def set_tail(X, layers, E):
    """X [B, L*E] routed tokens of one selected set -> per-layer and all-240 metrics,
    plus unique / effective source counts per expert."""
    n = X.sum(0)
    src = (X > 0).sum(0)
    eff = LB.eff_sources(X)
    per = {}
    for li, L in enumerate(layers):
        sl = slice(li * E, (li + 1) * E)
        per[str(L)] = dict(tail_metrics(n[sl]), unique_sources_median=float(np.median(src[sl])),
                           unique_sources_min=float(src[sl].min()),
                           effective_sources_median=float(np.median(eff[sl])),
                           effective_sources_p10=float(np.quantile(eff[sl], .1)))
    allm = dict(tail_metrics(n), gini_layer_mean=float(np.mean([per[str(L)]["gini"] for L in layers])),
                unique_sources_median=float(np.median(src)), unique_sources_min=float(src.min()),
                effective_sources_median=float(np.median(eff)), effective_sources_p10=float(np.quantile(eff, .1)))
    return dict(per_layer=per, all=allm), n


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def prob(flags):
    flags = np.asarray(flags, bool)
    k, n = int(flags.sum()), len(flags)
    lo, hi = wilson(k, n)
    return dict(p=k / n, k=k, n=n, ci95=[lo, hi])


def dist(vals):
    v = np.asarray(vals, float)
    return dict(mean=float(v.mean()), **{f"q{int(p * 100):02d}": float(np.quantile(v, p)) for p in QS},
                min=float(v.min()), max=float(v.max()))


# ---------------------------------------------------------------- Experiment A
def mc_sets(pool_ids, B, seed, n_sets=N_MC):
    """n_sets deterministic random sets (without replacement within a set)."""
    g = np.random.default_rng(seed)
    pool = np.asarray(pool_ids)
    return [sorted(int(s) for s in pool[g.choice(len(pool), B, replace=False)]) for _ in range(n_sets)]


def scarcity_flags(m):
    out = {f"any_below_{t}": m[f"below_{t}"] > 0 for t in THRESH}
    out["any_zero"] = m["zero"] > 0
    out["frac_below_128_ge_10pct"] = m["frac_below_128"] >= 0.10
    return out


METRICS = ("min", "p1", "p5", "p10", "median", "p90", "max", "gini", "cv", "zero",
           "below_32", "below_64", "below_128", "below_256", "frac_below_128",
           "unique_sources_median", "effective_sources_median")


def run_mc(C, frow, pool_ids, B, seed, layers, E):
    sets = mc_sets(pool_ids, B, seed)
    recs, layer_recs, counts = [], {str(L): [] for L in layers}, []
    for ids in sets:
        st, n = set_tail(C[[frow[s] for s in ids]], layers, E)
        recs.append(st["all"])
        for L in layers:
            layer_recs[str(L)].append(st["per_layer"][str(L)])
        counts.append(n)
    counts = np.asarray(counts)
    summ = lambda rs: dict(
        distributions={k: dist([r[k] for r in rs]) for k in METRICS},
        probabilities={k: prob([scarcity_flags(r)[k] for r in rs]) for k in scarcity_flags(rs[0])})
    out = dict(seed=seed, n_sets=len(sets), B=B, all=summ(recs),
               per_layer={L: summ(r) for L, r in layer_recs.items()},
               gini_layer_mean=dist([r["gini_layer_mean"] for r in recs]),
               first_set=sets[0], set_hash=__import__("hashlib").sha256(json.dumps(sets).encode()).hexdigest()[:16])
    return out, counts


# ---------------------------------------------------------------- Experiment B
def warm_prefix(ws, B, seed):
    return ws["seeds"][str(seed)]["ids"][:B // 4]


def select(code, B, seed, C, full, typ, layers, E, ws):
    """Exactly e3_allocation.main's selection path (rng = Random(seed)), with K = B // 4."""
    import e3_allocation as E3
    strategy, pool = STRATS[code]
    pool_ids = typ["ids"] if pool == "typical_half" else full
    frow = {s: i for i, s in enumerate(full)}
    Cp = C[[frow[s] for s in pool_ids]] if pool == "typical_half" else C
    W = warm_prefix(ws, B, seed)
    rng = random.Random(seed)
    if strategy == "random":
        wset = set(W)
        return W + rng.sample([s for s in pool_ids if s not in wset], B - len(W)), None
    ch, warm, traj = E3.select_lp_utility("coverage_lp", Cp, pool_ids, layers, E, B, k_eff(B), ROUNDS, rng,
                                          None, warm_start=W)
    assert warm == W
    return ch, traj


TRAJ_KEYS = ("round", "round_budget", "n_selected_before", "n_zero_exposure", "n_exploration_targets",
             "n_exploration_targets_reached", "n_exploration_targets_unreached", "n_exploration_sequences",
             "n_positive_utility", "n_lp_positive_value_selected", "n_zero_marginal_random_fill",
             "n_random_fallback_sequences", "used_random_utility_fallback")


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


PAIRS = {"full": ("FC", "FR"), "typical": ("TC", "TR")}
DELTA_KEYS = ("min", "p5", "p10", "below_64", "below_128", "below_256", "zero", "gini_layer_mean",
              "effective_sources_median")


def paired(stats, B, seed, fam):
    c, r = PAIRS[fam]
    a, b = stats[f"{c}_B{B}_s{seed}"], stats[f"{r}_B{B}_s{seed}"]
    d = {k: a["tail"]["all"][k] - b["tail"]["all"][k] for k in DELTA_KEYS}
    rb = b["tail"]["all"]
    d["p10_rel"] = (a["tail"]["all"]["p10"] - rb["p10"]) / rb["p10"] if rb["p10"] > 0 else None
    d["below_128_rel_reduction"] = ((rb["below_128"] - a["tail"]["all"]["below_128"]) / rb["below_128"]
                                    if rb["below_128"] > 0 else None)
    for k in ("d_post_warm", "set_js", "set_entropy_mean", "seq_entropy_mean"):
        d[k] = a["repr"][k] - b["repr"][k]
    d["aggregate_l1_by_layer"] = {L: a["repr"]["aggregate_l1_by_layer"][L] - b["repr"]["aggregate_l1_by_layer"][L]
                                  for L in a["repr"]["aggregate_l1_by_layer"]}
    return d


def decide(stats, valid):
    """Prespecified gate (task section 11). Returns (label, per-budget detail)."""
    if not valid:
        return "CPU-RUN-BLOCKED", {}
    det, promising, partial = {}, [], []
    for B in BUDGETS:
        scarce = [stats[f"FR_B{B}_s{s}"]["tail"]["all"]["frac_below_128"] >= GATE["scarcity_frac_below_128"]
                  or stats[f"FR_B{B}_s{s}"]["tail"]["all"]["zero"] >= 1 for s in SEEDS]
        c1 = sum(scarce) >= GATE["min_seeds"]
        fams = {}
        for fam in PAIRS:
            ds = [paired(stats, B, s, fam) for s in SEEDS]
            imp = [(d["p10_rel"] is not None and d["p10_rel"] >= GATE["p10_rel_gain"]) or
                   (d["below_128_rel_reduction"] is not None and
                    d["below_128_rel_reduction"] >= GATE["below128_rel_reduction"]) for d in ds]
            min_worse = [d["min"] < 0 for d in ds]
            gini_better = [d["gini_layer_mean"] < 0 for d in ds]
            fams[fam] = dict(lower_tail_improved=imp, c2=sum(imp) >= GATE["min_seeds"],
                             min_worse=min_worse, c3=sum(min_worse) < GATE["min_seeds"],
                             gini_better=gini_better)
        det[str(B)] = dict(fr_scarce_by_seed=scarce, c1=c1, families=fams)
        if c1 and any(f["c2"] and f["c3"] for f in fams.values()):
            promising.append(B)
        elif c1 and any(sum(f["gini_better"]) >= GATE["min_seeds"] or any(f["lower_tail_improved"])
                        for f in fams.values()):
            partial.append(B)
    if promising:
        # most promising: most seeds improved, then larger mean p10 gain
        def score(B):
            best = max((sum(f["lower_tail_improved"]), fam) for fam, f in det[str(B)]["families"].items()
                       if f["c2"] and f["c3"])
            return best[0]
        det["recommended_budget"] = max(promising, key=score)
        return "EXTREME-SCARCITY-PROMISING", det
    if partial:
        return "EXTREME-SCARCITY-PARTIAL", det
    return "EXTREME-SCARCITY-NO-OPPORTUNITY", det


# ---------------------------------------------------------------- compute feasibility
def compute_estimate(low_budget_summary):
    cost = json.load(open(low_budget_summary))["cost"]
    Bs = np.array([16, 32, 64, 128], float)
    hs = np.array([np.mean([cost[f"{c}_B{int(B)}"]["hessian_s"] for c in STRATS]) for B in Bs])
    slope, icpt = np.polyfit(Bs, hs, 1)
    fwd = np.mean([cost[f"FR_B{int(B)}"]["calib_forward_equiv_s"] / B for B in Bs])
    gptq = float(np.mean([cost[k]["gptq_s"] for k in cost]))
    prof = float(max(cost[k]["routing_forward_equiv_s"] for k in cost))
    sel = float(max(cost[k]["selection_s"] for k in cost))
    peak = float(max(cost[k]["peak_gpu_mem_gb"] for k in cost))
    out = dict(hessian_fit=dict(slope_s_per_seq=float(slope), intercept_s=float(icpt)),
               gptq_s_per_run=gptq, routing_profile_s=prof, lp_selection_s_max_observed=sel,
               peak_gpu_gb=peak, by_budget={})
    for B in BUDGETS:
        h = float(icpt + slope * B)
        run = h + gptq
        out["by_budget"][str(B)] = dict(
            hessian_s_extrapolated=h, gptq_s=gptq, run_s=run, calib_forward_equiv_s=float(fwd * B),
            profile_over_calib_forwards=prof / (fwd * B),
            minimal_comparison_runs=12, minimal_comparison_gpu_s=12 * run,
            note="12 = 4 strategies x 3 seeds; profile is paid once by FC/TC/TR, never by FR")
    return out


# ---------------------------------------------------------------- driver
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs_extreme_low_budget")
    ap.add_argument("--matched-dir", default="runs_typicality_matched")
    ap.add_argument("--pool-dir", default="runs_typicality")
    ap.add_argument("--low-budget-dir", default="runs_low_budget")
    a = ap.parse_args(argv)
    if os.path.exists(os.path.join(a.outdir, "summary.json")):
        raise SystemExit(f"{a.outdir}/summary.json exists; refusing to overwrite earlier outputs")
    for d in ("figures", "logs"):
        os.makedirs(os.path.join(a.outdir, d), exist_ok=True)
    from moe_calib import typicality as TY
    t0 = time.time()
    C, layers, E, full, typ, hal, ws, pools = LB.load_inputs(a.matched_dir, a.pool_dir)
    assert C.shape[1] == len(layers) * E == 240
    assert np.allclose(C.reshape(len(C), len(layers), E).sum(2), SEQ_LEN * TOPK), "routed tokens != 512 x 4"
    frow = {s: i for i, s in enumerate(full)}
    ref = TY.pool_reference(C, len(layers), E)
    scores = TY.sequence_scores(C, len(layers), E, ref)
    elig = set(typ["ids"]) & set(full)

    # ---- Experiment A
    mc, mc_counts = {}, {}
    for (pool, B), seed in MC_SEEDS.items():
        ids = typ["ids"] if pool == "typical_half" else full
        mc[f"{pool}_B{B}"], mc_counts[f"{pool}_B{B}"] = run_mc(C, frow, ids, B, seed, layers, E)
        print(f"[mc] {pool} B={B}: P(any zero)={mc[f'{pool}_B{B}']['all']['probabilities']['any_zero']['p']:.3f}",
              flush=True)
    sc = dict(uniform_expectation={str(B): B * SEQ_LEN * TOPK / E for B in BUDGETS}, thresholds=list(THRESH),
              mc_seeds={f"{p}_B{B}": s for (p, B), s in MC_SEEDS.items()}, n_sets=N_MC,
              note="Monte Carlo over random sets: an exposure-scarcity distribution, NOT GPTQ replicates; "
                   "thresholds are descriptive scarcity indicators, not validated sufficiency thresholds",
              results=mc, typical_fingerprint=typ["fingerprint"],
              routing_sha256=pools["routing_cache"]["sha256"])
    json.dump(sc, open(os.path.join(a.outdir, "scarcity.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.outdir, "logs", "mc_counts.npz"), **mc_counts)

    # ---- Experiment B
    sel = dict(budgets=list(BUDGETS), seeds=list(SEEDS), strategies=NAMES,
               schedule={str(B): dict(zip(("warm", "rounds"), schedule(B))) for B in BUDGETS},
               k_passed_to_controller={str(B): k_eff(B) for B in BUDGETS},
               controller_note="unchanged e3_allocation.select_lp_utility; caller passes K = B//4 (the only "
                               "role of K is the warm-start floor max(K, B//4) in round_schedule)",
               warm_starts={}, sets={}, stats={})
    valid = True
    for B in BUDGETS:
        for sd in SEEDS:
            W = warm_prefix(ws, B, sd)
            assert set(W) <= elig and set(W) <= set(hal), "warm prefix outside typical_half & random_half"
            import e3_allocation as E3
            sel["warm_starts"][f"B{B}_s{sd}"] = dict(ids=W, hash=E3.warm_hash(W),
                                                     prefix_of=ws["seeds"][str(sd)]["hash"])
            for code in STRATS:
                ch, traj = select(code, B, sd, C, full, typ, layers, E, ws)
                pool = typ["ids"] if STRATS[code][1] == "typical_half" else full
                ok = len(ch) == B == len(set(ch)) and ch[:B // 4] == W and set(ch) <= set(pool)
                valid &= ok
                key = f"{code}_B{B}_s{sd}"
                sel["sets"][key] = dict(chosen=ch, valid=ok,
                                        trajectory=[{k: t[k] for k in TRAJ_KEYS} for t in traj] if traj else None)
                tail, _ = set_tail(C[[frow[s] for s in ch]], layers, E)
                full_st = LB.set_stats(ch, C, frow, layers, E, ref, scores)
                sel["stats"][key] = dict(tail=tail, repr=full_st["representativeness"])
    for B in BUDGETS:
        for sd in SEEDS:
            g = lambda c: sel["sets"][f"{c}_B{B}_s{sd}"]["chosen"][B // 4:]
            sel["stats"][f"overlap_B{B}_s{sd}"] = dict(FC_FR=jaccard(g("FC"), g("FR")), TC_TR=jaccard(g("TC"), g("TR")),
                                                       FC_TC=jaccard(g("FC"), g("TC")), FR_TR=jaccard(g("FR"), g("TR")))
    json.dump(sel, open(os.path.join(a.outdir, "matched_selection.json"), "w"), indent=1)

    # ---- B = 16 reference (existing matched selections; no new computation of policy)
    ref16 = json.load(open(os.path.join(a.low_budget_dir, "selected_sets.json")))
    ref16_stats = {}
    for code in STRATS:
        for sd in SEEDS:
            ch = ref16["sets"][f"{code}_B16_s{sd}"]["chosen"]
            tail, n = set_tail(C[[frow[s] for s in ch]], layers, E)
            ref16_stats[f"{code}_B16_s{sd}"] = dict(tail=tail, repr=ref16["stats"][f"{code}_B16_s{sd}"]["representativeness"],
                                                    counts=n.tolist())

    label, det = decide(sel["stats"], valid)
    deltas = {f"{fam}_B{B}_s{sd}": paired(sel["stats"], B, sd, fam) for fam in PAIRS for B in BUDGETS for sd in SEEDS}
    deltas.update({f"{fam}_B16_s{sd}": paired(ref16_stats, 16, sd, fam) for fam in PAIRS for sd in SEEDS})
    summary = dict(decision=label, gate=GATE, gate_detail=det, valid=valid, deltas=deltas,
                   b16_reference={k: dict(tail=v["tail"], repr=v["repr"]) for k, v in ref16_stats.items()},
                   compute=compute_estimate(os.path.join(a.low_budget_dir, "summary.json")),
                   runtime_s=time.time() - t0)
    json.dump(summary, open(os.path.join(a.outdir, "summary.json"), "w"), indent=1)

    import extreme_low_budget_report as R
    R.write_scarcity_md(sc, layers, os.path.join(a.outdir, "scarcity.md"))
    R.write_summary_md(sel, summary, sc, layers, os.path.join(a.outdir, "summary.md"))
    R.figures(sc, mc_counts, sel, summary, ref16_stats, layers, E, os.path.join(a.outdir, "figures"))
    print(f"decision: {label}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
