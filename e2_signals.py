"""E2 — Fit and select a predictor of FUTURE GPTQ benefit from n0 signals.

Observations: one per (characterization seed, layer, expert, starting budget
n0), with signals computed on the SAME base draw E1 used (read from each E1
run dir) and target future_rel_delta = E1 rel_delta averaged over reps. E1
labels use only the TUNING holdout, so E2 never sees the test holdout.

Signals (candidate predictors, NOT definitions of sufficiency):
  count    routed tokens n_e                          (exposure baseline)
  drift    half-vs-all drift of normalized Hessian
  relse    between-block relative std. error of H_bar
  margin   mean router top-k margin p_k - p_{k+1}     (oriented as -margin)
  entropy  mean routing entropy of tokens routed to e
Blocks are the fixed per-sequence map hessian.block_of(seq, K, block_seed),
the same map E3 uses.

Predictor per signal (moe_calib/utility.py):
  future_rel_delta ~ b0 + b_count*log(count) + b_sig*z(oriented signal)
Selection: deterministic out-of-sample criterion (utility.select_signal).
Support report: per starting budget, routed-count min/p5/p25/median/p75/
p95/max of the fitted observations and every excluded observation with its
reason (E1 exclusion, or E2: non-finite target / signal).
Output: e2_model_b{bits}.json (consumed by E3 --signal ...) and
e2_signals_b{bits}.csv (observations). AUROC / Spearman / binned Spearman /
incremental R^2 are kept as diagnostics. Template: Katharopoulos & Fleuret
(ICML 2018) — predict whether non-uniform treatment will pay off first.
"""
from __future__ import annotations
import argparse, csv, json, os

import numpy as np

from moe_calib import model_utils as MU
from moe_calib.data_utils import load_layer_cache
from moe_calib.hessian import assign_blocks
from moe_calib.profiler import Profiler
from moe_calib.runid import load_e1_runs, damp_tag, fp_percdamp
from moe_calib.stats import auroc, spearman, binned_spearman, incremental_r2
from moe_calib.utility import SIGNALS, ORIENT, fit_predictor, select_signal


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--e1-dirs", nargs="+", default=None,
                    help="E1 run dirs (one per characterization seed); "
                         "default: all finished e1_b{bits}_s*_* in --outdir")
    ap.add_argument("--block-seed", type=int, default=0)
    ap.add_argument("--min-char-seeds", type=int, default=1,
                    help="fail unless at least this many characterization seeds "
                         "are combined (real experiments: 3)")
    ap.add_argument("--percdamp", type=float, default=None,
                    help="use the E1 runs made with this GPTQ damping; artifacts are "
                         "then named e2_*_b{bits}_d{percdamp}.* (omit = legacy runs/names)")
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args(argv)
    tag = f"b{args.bits}" + (f"_{damp_tag(args.percdamp)}" if args.percdamp is not None else "")
    runs, common = load_e1_runs(args.outdir, args.bits, args.e1_dirs, args.percdamp)
    K = common["K"]
    if len(runs) < args.min_char_seeds:
        raise RuntimeError(f"found {len(runs)} characterization seed(s) "
                           f"{[r['char_seed'] for r in runs]}, need >= {args.min_char_seeds}")
    print(f"[E2] combining characterization seeds {[r['char_seed'] for r in runs]}", flush=True)

    model, _ = MU.load_model()
    E = MU.num_experts(model)
    obs = []
    excluded: list[dict] = []
    for run in runs:
        # E1 exclusions: (layer, expert, n0) with NO labelled replicate
        why: dict[tuple, set] = {}
        exc_path = os.path.join(run["dir"], "excluded.csv")
        if os.path.exists(exc_path):
            for row in csv.DictReader(open(exc_path)):
                k = (int(row["layer"]), int(row["expert"]), int(row["n0"]))
                why.setdefault(k, set()).add(row["excluded"])
        lab: dict[tuple, dict] = {}
        for row in csv.DictReader(open(run["labels"])):
            k = (int(row["layer"]), int(row["expert"]), int(row["n0"]))
            d = lab.setdefault(k, {"delta": [], "sig": []})
            d["delta"].append(float(row["rel_delta"]))
            d["sig"].append(int(row["significant"]))
        for k, reasons in sorted(why.items()):
            if k not in lab:
                excluded.append(dict(characterization_seed=run["char_seed"], layer=k[0],
                                     expert=k[1], starting_budget=k[2],
                                     reason="e1:" + "+".join(sorted(reasons))))
        for L in sorted({k[0] for k in lab}):
            cache = load_layer_cache(args.cache_dir, L)
            for n0 in sorted({k[2] for k in lab}):
                base = run["draws"][n0]["base"]
                prof = Profiler(model, L, K=K).run(
                    cache, base, blocks=assign_blocks(base, K, args.block_seed))
                blocks = list(range(K))
                for e in range(E):
                    if (L, e, n0) not in lab:
                        continue
                    key = (L, e, "gu")
                    margin, ent = prof.router_signals(e)
                    obs.append(dict(
                        layer=L, expert=e, characterization_seed=run["char_seed"],
                        starting_budget=n0, routed_count=prof.acc.tokens(key),
                        drift=prof.acc.drift(key, blocks),
                        relse=prof.acc.rel_std_err(key, blocks),
                        margin=margin, entropy=ent,
                        future_rel_delta=float(np.nanmean(lab[(L, e, n0)]["delta"])),
                        significant=int(np.mean(lab[(L, e, n0)]["sig"]) >= 0.5)))
                prof.free()
                print(f"[E2] seed={run['char_seed']} L={L} n0={n0} done", flush=True)
            del cache

    # ---- fit + select on rows where every signal and the target are finite ----
    col = lambda name: np.array([r[name] for r in obs], float)
    cnt, y = col("routed_count"), col("future_rel_delta")
    vals = {s: (cnt if s == "count" else col(s)) for s in SIGNALS}
    ok = np.isfinite(y) & (cnt > 0)
    for s in SIGNALS:
        ok &= np.isfinite(vals[s])
    for i, r in enumerate(obs):
        if not ok[i]:
            bad = [s for s in SIGNALS if not np.isfinite(vals[s][i])]
            reason = ("e2:nonfinite_target" if not np.isfinite(y[i]) else
                      "e2:zero_count" if cnt[i] <= 0 else
                      "e2:nonfinite_signal:" + "+".join(bad))
            excluded.append(dict(characterization_seed=r["characterization_seed"],
                                 layer=r["layer"], expert=r["expert"],
                                 starting_budget=r["starting_budget"], reason=reason))
    n0_all = col("starting_budget")
    support = {}
    for n0 in sorted({r["starting_budget"] for r in obs} |
                     {x["starting_budget"] for x in excluded}):
        c = cnt[ok & (n0_all == n0)]
        q = (np.percentile(c, [0, 5, 25, 50, 75, 95, 100]).tolist() if c.size
             else [float("nan")] * 7)
        reasons: dict[str, int] = {}
        for x in excluded:
            if x["starting_budget"] == n0:
                reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
        support[f"n0={n0}"] = dict(n_fitted=int(c.size), count_min=q[0], count_p5=q[1],
                                   count_p25=q[2], count_median=q[3], count_p75=q[4],
                                   count_p95=q[5], count_max=q[6],
                                   n_excluded=sum(reasons.values()), excluded_by_reason=reasons)
    if ok.sum() < 10:
        raise RuntimeError(f"only {int(ok.sum())} usable observations; see "
                           f"support report / excluded reasons: {support}")
    fit_obs = {"count": cnt[ok], "target": y[ok],
               **{s: vals[s][ok] for s in SIGNALS if s != "count"}}
    seeds = col("characterization_seed")[ok]
    layers = col("layer")[ok]
    selected, criterion, prelim, scores = select_signal(fit_obs, seeds, layers)
    predictors = {s: fit_predictor(s, fit_obs["count"],
                                   None if s == "count" else fit_obs[s], fit_obs["target"])
                  for s in SIGNALS}

    sigl = col("significant")[ok].astype(int)
    metrics = {}
    for s in SIGNALS:
        v = ORIENT[s] * (fit_obs["count"] if s == "count" else fit_obs[s])
        r0, r1, dr2 = incremental_r2(v, fit_obs["target"], fit_obs["count"])
        metrics[s] = dict(selection_score=scores[s], auroc_significant=auroc(v, sigl),
                          spearman=spearman(v, fit_obs["target"]),
                          binned_spearman=binned_spearman(v, fit_obs["target"], fit_obs["count"]),
                          r2_count_only=r0, r2_with_signal=r1, incremental_r2=dr2)

    artifact = dict(
        selected_signal=selected, selection_criterion=criterion,
        selection_lower_is_better=True, preliminary=prelim,
        target="future_rel_delta (E1 rel_delta, n0 -> 2n0, tuning holdout, mean over reps)",
        predictors=predictors, metrics=metrics,
        n_obs_total=len(obs), n_obs_fit=int(ok.sum()),
        support_by_starting_budget=support, excluded_observations=excluded,
        characterization_seeds=sorted(r["char_seed"] for r in runs),
        e1_runs=[r["dir"] for r in runs], e1_config=common,
        split=common["split"], K=K, block_seed=args.block_seed, bits=args.bits,
        percdamp=fp_percdamp(common))
    with open(os.path.join(args.outdir, f"e2_model_{tag}.json"), "w") as f:
        json.dump(artifact, f, indent=2, default=float)
    with open(os.path.join(args.outdir, f"e2_signals_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(obs[0].keys()))
        w.writeheader()
        w.writerows(obs)

    print(f"\n[E2] {ok.sum()}/{len(obs)} observations; criterion={criterion}")
    if prelim:
        print("[E2] *** PRELIMINARY SIGNAL SELECTION: fewer than 2 characterization "
              "seeds, not validated out of sample across seeds ***")
    for k, v in support.items():
        print(f"    {k}: fitted {v['n_fitted']}, count min/p5/p25/med/p75/p95/max = "
              f"{v['count_min']:.0f}/{v['count_p5']:.0f}/{v['count_p25']:.0f}/"
              f"{v['count_median']:.0f}/{v['count_p75']:.0f}/{v['count_p95']:.0f}/"
              f"{v['count_max']:.0f}; excluded {v['n_excluded']} {v['excluded_by_reason']}")
    print(f"    {'signal':8s} {'score':>9s} {'AUROC':>6s} {'rho':>6s} {'bin_rho':>7s} {'dR2':>7s}")
    for s in SIGNALS:
        m = metrics[s]
        print(f"    {s:8s} {m['selection_score']:9.4g} {m['auroc_significant']:6.3f} "
              f"{m['spearman']:6.3f} {m['binned_spearman']:7.3f} {m['incremental_r2']:7.4f}")
    print(f"[E2] selected signal: {selected} -> {args.outdir}/e2_model_{tag}.json")
    return artifact


if __name__ == "__main__":
    main()
