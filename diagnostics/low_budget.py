"""Low-budget calibration allocation sweep: CPU side (warm starts, selections,
scarcity diagnostics) and per-run validation.

    --prepare          warm_starts_B{16,32,64,128}.json (nested prefixes of the
                       matched 32-sequence warm starts), selected_sets.json
                       (every strategy x budget x seed, computed with the SAME
                       code path and RNG as e3_allocation.main), scarcity.{json,md}
                       and figures/scarcity.{pdf,png}. Reads no reconstruction.
    --validate F       one E3 run JSON: frozen settings, B unique ids, exact
                       warm start, pool membership, 240 experts, ids equal to
                       the precomputed selection.

Strategies: FR full_random, FC coverage_lp (full pool), TR typical_random,
TC typical_coverage_lp (coverage_lp restricted to the typical half).
Schedule (existing round_schedule, K = 4): warm = B/4, two equal rounds.
"""
from __future__ import annotations
import argparse, json, os, random, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))

BUDGETS, SEEDS, K, ROUNDS = (16, 32, 64, 128), (0, 1, 2), 4, 2
THRESH = (64, 128, 256, 512)
STRATS = {"FR": ("random", "full"), "FC": ("coverage_lp", "full"),
          "TR": ("random", "typical_half"), "TC": ("coverage_lp", "typical_half")}
NAMES = {"FR": "Full Random", "FC": "Full Coverage-LP", "TR": "Typical Random", "TC": "Typical Coverage-LP"}


def load_inputs(matched_dir, pool_dir):
    import e3_allocation as E3
    z = np.load(os.path.join(matched_dir, "routing_profile_v2.npz"))
    pools = json.load(open(os.path.join(pool_dir, "pools.json")))
    sha = __import__("hashlib").sha256(open(os.path.join(matched_dir, "routing_profile_v2.npz"), "rb").read()).hexdigest()
    assert sha == pools["routing_cache"]["sha256"], "routing cache differs from the pools' provenance"
    C = z["C"].astype(float)
    layers, E = z["layers"].tolist(), int(z["E"])
    full = [int(s) for s in z["pool_ids"]]
    typ = E3.load_candidate_subset(os.path.join(pool_dir, "pool_typical_half.json"), full)
    hal = pools["pools"]["random_half"]["ids"]
    ws = json.load(open(os.path.join(matched_dir, "warm_starts.json")))
    return C, layers, E, full, typ, hal, ws, pools


def warm_file(ws, B):
    """Nested prefix W(seed, B) = first B/4 of the matched W(seed, 128)."""
    import e3_allocation as E3
    w = B // 4
    seeds = {}
    for sd, ent in ws["seeds"].items():
        assert E3.warm_hash(ent["ids"]) == ent["hash"] and len(ent["ids"]) == 32
        ids = ent["ids"][:w]
        seeds[sd] = dict(ids=ids, hash=E3.warm_hash(ids), prefix_of=ent["hash"])
    return dict(design=f"W(seed,{B}) = first {w} of the matched warm start (runs_typicality_matched)",
                pools=ws["pools"], pool_dir=ws["pool_dir"], warm_size=w, eligible=ws["eligible"],
                seeds=seeds, budget=B)


def select(code, B, seed, C, full, typ, layers, E, wfile):
    """Exactly e3_allocation.main's selection for this strategy (CPU)."""
    import e3_allocation as E3
    strategy, pool = STRATS[code]
    pool_ids = typ["ids"] if pool == "typical_half" else full
    frow = {s: i for i, s in enumerate(full)}
    Cp = C[[frow[s] for s in pool_ids]] if pool == "typical_half" else C
    W = wfile["seeds"][str(seed)]["ids"]
    rng = random.Random(seed)
    if strategy == "random":
        ws = set(W)
        return W + rng.sample([s for s in pool_ids if s not in ws], B - len(W)), None
    ch, warm, traj = E3.select_lp_utility("coverage_lp", Cp, pool_ids, layers, E, B, K, ROUNDS, rng, None,
                                          warm_start=W)
    return ch, traj


def eff_sources(X):
    q = X / np.maximum(X.sum(0, keepdims=True), 1)
    return np.exp(-(np.where(q > 0, q * np.log(np.where(q > 0, q, 1)), 0)).sum(0))


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def exposure_stats(X):
    """X: [n_selected, E] routed tokens of one layer. Per-expert statistics."""
    n = X.sum(0)
    src = (X > 0).sum(0)
    eff = eff_sources(X)
    out = dict(min=float(n.min()), p10=float(np.quantile(n, .1)), median=float(np.median(n)),
               p90=float(np.quantile(n, .9)), max=float(n.max()), mean=float(n.mean()), sd=float(n.std()),
               gini=gini(n), cv=float(n.std() / n.mean()) if n.mean() > 0 else None, zero=int((n == 0).sum()),
               distinct_sources_median=float(np.median(src)), distinct_sources_min=float(src.min()),
               effective_sources_median=float(np.median(eff)), effective_sources_p10=float(np.quantile(eff, .1)))
    for t in THRESH:
        out[f"below_{t}"] = int((n < t).sum())
    return out, n


def set_stats(ids, C, frow, layers, E, ref, scores):
    from moe_calib import typicality as TY
    from moe_calib import aggregate_constraint as AG
    X = C[[frow[s] for s in ids]]
    per, allN = {}, []
    for li, L in enumerate(layers):
        st, n = exposure_stats(X[:, li * E:(li + 1) * E])
        per[str(L)] = st
        allN.append(n)
    allN = np.concatenate(allN)
    glob = dict(min=float(allN.min()), p10=float(np.quantile(allN, .1)), median=float(np.median(allN)),
                p90=float(np.quantile(allN, .9)), max=float(allN.max()), mean=float(allN.mean()),
                sd=float(allN.std()), cv=float(allN.std() / allN.mean()), zero=int((allN == 0).sum()),
                gini_mean_over_layers=float(np.mean([per[str(L)]["gini"] for L in layers])),
                effective_sources_median_mean_over_layers=float(np.mean([per[str(L)]["effective_sources_median"]
                                                                       for L in layers])),
                distinct_sources_median_mean_over_layers=float(np.mean([per[str(L)]["distinct_sources_median"]
                                                                      for L in layers])),
                **{f"below_{t}": int((allN < t).sum()) for t in THRESH})
    T = AG.token_invariant(C, len(layers), E)
    reff = AG.pool_reference_flat(C, len(layers), E)
    D = AG.aggregate_l1(X.sum(0), reff, len(ids) * T, len(layers), E)
    idx = [frow[s] for s in ids]
    rep = dict(d_mean=float(scores["d"][idx].mean()),
               d_post_warm=float(scores["d"][idx[len(ids) // 4:]].mean()),
               seq_entropy_mean=float(scores["entropy"][idx].mean()),
               set_js=TY.selected_set_js(X, ref, len(layers), E),
               aggregate_l1_by_layer={str(L): float(v) for L, v in zip(layers, D)},
               set_entropy_mean=float(np.mean([(lambda q: -(q[q > 0] * np.log(q[q > 0])).sum())(
                   X[:, li * E:(li + 1) * E].sum(0) / X[:, li * E:(li + 1) * E].sum()) for li in range(len(layers))])))
    return dict(per_layer=per, overall=glob, representativeness=rep, counts=allN.tolist())


def prepare(a):
    from moe_calib import typicality as TY
    os.makedirs(os.path.join(a.outdir, "figures"), exist_ok=True)
    C, layers, E, full, typ, hal, ws, pools = load_inputs(a.matched_dir, a.pool_dir)
    elig = set(typ["ids"]) & set(hal)
    frow = {s: i for i, s in enumerate(full)}
    ref = TY.pool_reference(C, len(layers), E)
    scores = TY.sequence_scores(C, len(layers), E, ref)
    import e3_allocation as E3
    sel = dict(budgets=list(BUDGETS), seeds=list(SEEDS), strategies=NAMES, typical_fingerprint=typ["fingerprint"],
               schedule={str(B): dict(warm=E3.round_schedule(B, K, ROUNDS)[0],
                                      rounds=[r[2] for r in E3.round_schedule(B, K, ROUNDS)[1]]) for B in BUDGETS},
               sets={}, stats={})
    for B in BUDGETS:
        wf = warm_file(ws, B)
        for sd, ent in wf["seeds"].items():
            assert set(ent["ids"]) <= elig, "warm prefix not in typical_half & random_half"
            assert ent["ids"] == ws["seeds"][sd]["ids"][:B // 4]
        json.dump(wf, open(os.path.join(a.outdir, f"warm_starts_B{B}.json"), "w"), indent=1)
        for sd in SEEDS:
            for code in STRATS:
                ch, traj = select(code, B, sd, C, full, typ, layers, E, wf)
                W = wf["seeds"][str(sd)]["ids"]
                assert len(ch) == B == len(set(ch)) and ch[:B // 4] == W
                pool = typ["ids"] if STRATS[code][1] == "typical_half" else full
                assert set(ch) <= set(pool)
                key = f"{code}_B{B}_s{sd}"
                sel["sets"][key] = dict(chosen=ch, warm_hash=wf["seeds"][str(sd)]["hash"],
                                        trajectory=[{k: t[k] for k in (
                                            "round", "round_budget", "n_zero_exposure", "n_exploration_targets",
                                            "n_exploration_sequences", "n_lp_positive_value_selected",
                                            "n_zero_marginal_random_fill", "n_random_fallback_sequences")}
                                            for t in traj] if traj else None)
                sel["stats"][key] = set_stats(ch, C, frow, layers, E, ref, scores)
                if code in ("FC", "TC") and B == 128:      # B = 128 anchors: must equal the matched runs
                    import report_typicality_matched as RTM
                    old = json.load(open(RTM.run_path(a.matched_dir, STRATS[code][1], "coverage_lp", sd, ws)))
                    assert old["chosen"] == ch, f"B=128 {code} s{sd} does not reproduce the matched selection"
                if code in ("FR", "TR") and B == 128:
                    import report_typicality_matched as RTM
                    old = json.load(open(RTM.run_path(a.matched_dir, STRATS[code][1], "random", sd, ws)))
                    assert old["chosen"] == ch, f"B=128 {code} s{sd} does not reproduce the matched selection"
    json.dump(sel, open(os.path.join(a.outdir, "selected_sets.json"), "w"))
    scarcity(sel, layers, E, a.outdir)
    print("prepare ok: B=128 selections reproduce the matched experiment for all 4 strategies x 3 seeds")


def scarcity(sel, layers, E, outdir):
    """Scarcity of Full Random selections (the reference policy), plus every strategy for context."""
    res = dict(uniform_expectation={str(B): B * 512 * 4 / E for B in BUDGETS}, thresholds=list(THRESH),
               note="thresholds are descriptive, not a learned sufficiency definition", by_budget={})
    for B in BUDGETS:
        ent = {}
        for code in STRATS:
            sts = [sel["stats"][f"{code}_B{B}_s{sd}"] for sd in SEEDS]
            ov = {k: float(np.mean([s["overall"][k] for s in sts])) for k in sts[0]["overall"]}
            pl = {str(L): {k: float(np.mean([s["per_layer"][str(L)][k] for s in sts]))
                           for k in sts[0]["per_layer"][str(L)] if sts[0]["per_layer"][str(L)][k] is not None}
                  for L in layers}
            ent[code] = dict(overall_mean_over_seeds=ov, per_layer_mean_over_seeds=pl,
                             overall_by_seed={str(sd): s["overall"] for sd, s in zip(SEEDS, sts)})
        res["by_budget"][str(B)] = ent
    json.dump(res, open(os.path.join(outdir, "scarcity.json"), "w"), indent=1)
    L_ = ["# Routing scarcity of the calibration selections (CPU, routing cache only)\n",
          "Routed tokens per expert over the 240 experts of layers 1/8/16/23 (selected sets of the sweep, "
          "mean over seeds 0/1/2). Uniform expectation = B x 512 x 4 / 60. Thresholds are descriptive.\n",
          "## Full Random (reference policy)\n",
          "| B | uniform exp. | min | p10 | median | p90 | max | CV | Gini (layer mean) | zero | <64 | <128 | <256 | <512 | "
          "distinct sources (median) | effective sources (median) |",
          "|---" * 16 + "|"]
    for B in BUDGETS:
        o = res["by_budget"][str(B)]["FR"]["overall_mean_over_seeds"]
        L_.append(f"| {B} | {B * 512 * 4 / E:.0f} | {o['min']:.1f} | {o['p10']:.0f} | {o['median']:.0f} | {o['p90']:.0f} | "
                  f"{o['max']:.0f} | {o['cv']:.3f} | {o['gini_mean_over_layers']:.3f} | {o['zero']:.2f} | {o['below_64']:.1f} | "
                  f"{o['below_128']:.1f} | {o['below_256']:.1f} | {o['below_512']:.1f} | "
                  f"{o['distinct_sources_median_mean_over_layers']:.1f} | {o['effective_sources_median_mean_over_layers']:.1f} |")
    L_.append("\n## Per layer, Full Random (min / p10 / median / p90 / max; Gini; experts < 128 tokens; zero)\n")
    L_.append("| B | " + " | ".join(f"L{L}" for L in layers) + " |")
    L_.append("|---" * (1 + len(layers)) + "|")
    for B in BUDGETS:
        pl = res["by_budget"][str(B)]["FR"]["per_layer_mean_over_seeds"]
        L_.append(f"| {B} | " + " | ".join(
            f"{pl[str(L)]['min']:.0f}/{pl[str(L)]['p10']:.0f}/{pl[str(L)]['median']:.0f}/{pl[str(L)]['p90']:.0f}/"
            f"{pl[str(L)]['max']:.0f}; {pl[str(L)]['gini']:.3f}; {pl[str(L)]['below_128']:.1f}; {pl[str(L)]['zero']:.1f}"
            for L in layers) + " |")
    L_.append("\n## All strategies (overall, mean over seeds)\n")
    L_.append("| B | strategy | min | p10 | median | Gini | zero | <128 | <256 | effective sources |")
    L_.append("|---|---|---|---|---|---|---|---|---|---|")
    for B in BUDGETS:
        for code in STRATS:
            o = res["by_budget"][str(B)][code]["overall_mean_over_seeds"]
            L_.append(f"| {B} | {NAMES[code]} | {o['min']:.1f} | {o['p10']:.0f} | {o['median']:.0f} | "
                      f"{o['gini_mean_over_layers']:.3f} | {o['zero']:.2f} | {o['below_128']:.1f} | {o['below_256']:.1f} | "
                      f"{o['effective_sources_median_mean_over_layers']:.1f} |")
    open(os.path.join(outdir, "scarcity.md"), "w").write("\n".join(L_) + "\n")
    # figure: per-layer routed-token distributions (Full Random) by budget
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e6e6e6"})
    fig, axes = plt.subplots(1, len(layers), figsize=(10, 3.4), sharey=True)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for li, (ax, L) in enumerate(zip(axes, layers)):
        data = []
        for B in BUDGETS:
            v = np.concatenate([np.asarray(sel["stats"][f"FR_B{B}_s{sd}"]["counts"])[li * E:(li + 1) * E] for sd in SEEDS])
            data.append(np.maximum(v, 1))
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                        flierprops=dict(markersize=3, markerfacecolor="#777777", markeredgecolor="none"),
                        medianprops=dict(color="#333333", linewidth=1.5))
        for p, c in zip(bp["boxes"], cols):
            p.set_facecolor(c); p.set_alpha(0.85); p.set_edgecolor("white")
        ax.set_xticks(range(1, len(BUDGETS) + 1), [f"B={B}" for B in BUDGETS])
        ax.set_title(f"layer {L}", loc="left")
        ax.set_yscale("log")
        ax.axhline(128, color="#999999", linewidth=1, linestyle="--")
        ax.grid(axis="y", which="both")
    axes[0].set_ylabel("routed tokens per expert (log scale)")
    fig.suptitle("Expert exposure under Full Random selection (box = 3 seeds × 60 experts; dashed line = 128 tokens)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, "figures", f"fig3_exposure_scarcity.{ext}"), dpi=200)
    plt.close(fig)


def validate(path, a):
    from moe_calib.frozen import load_frozen
    cfg = load_frozen(a.frozen)
    r = json.load(open(path))
    s = r["summary"]
    B = s["B"]
    sel = json.load(open(os.path.join(a.outdir, "selected_sets.json")))
    code = {("random", None): "FR", ("coverage_lp", None): "FC", ("random", "typical_half"): "TR",
            ("coverage_lp", "typical_half"): "TC"}[(s["strategy"], (s.get("candidate_subset") or {}).get("name"))]
    exp = sel["sets"][f"{code}_B{B}_s{s['seed']}"]
    g = cfg["gptq"]
    errs = [f"{k}={v!r}" for k, v, w in (
        ("bits", s["bits"], g["bits"]), ("percdamp", s["percdamp"], float(g["percdamp"])),
        ("layers", s["layers"], cfg["layers"]), ("split", s["split"], cfg["split"]), ("eval_split", s["eval_split"], "test"),
        ("n_experts_quantized_total", s["n_experts_quantized_total"], 240),
        ("n_experts_test_observed", s["n_experts_test_observed"], 240),
        ("n_unique_selected", s.get("n_unique_selected"), B),
        ("routing_counts_match_hessian_tokens", s.get("routing_counts_match_hessian_tokens"), True)) if v != w]
    if r["chosen"] != exp["chosen"]:
        errs.append("chosen ids differ from the precomputed selection")
    if (s.get("warm_start_fixed") or {}).get("hash") != exp["warm_hash"] or r["chosen"][:B // 4] != exp["chosen"][:B // 4]:
        errs.append("warm start differs")
    if not all(0 <= x < 2000 for x in r["chosen"]):
        errs.append("ids outside the candidate pool")
    if errs:
        raise SystemExit(f"INVALID {path}: " + "; ".join(errs))
    return r, code


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs_low_budget")
    ap.add_argument("--matched-dir", default="runs_typicality_matched")
    ap.add_argument("--pool-dir", default="runs_typicality")
    ap.add_argument("--frozen", default="configs/frozen_b4.json")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--validate", default=None)
    a = ap.parse_args(argv)
    if a.prepare:
        prepare(a)
    if a.validate:
        validate(a.validate, a)
        print(f"valid: {a.validate}")


if __name__ == "__main__":
    main()
