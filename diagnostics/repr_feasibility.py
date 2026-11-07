"""CPU pre-GPU checks for coverage_lp_repr (runs before any reconstruction).

  1. recompute d(s) from routing_profile_v2.npz (ORIGINAL pool reference) and
     require exact agreement with runs_typicality/typicality_scores.csv and
     with the typical_half pool membership;
  2. tau = median d(s) over the 2,000 candidates (prespecified);
  3. per seed: the matched warm start W_s (fingerprint-checked), discrete
     feasibility of 96 post-warm picks with mean d <= tau from pool \\ W_s;
  4. a full CPU dry run of the selection (the controller is CPU-only, so this
     is the exact selection the GPU run will use, same rng seed), plus a replay
     of unconstrained coverage_lp that must reproduce the stored matched
     Full Coverage-LP selections (backward compatibility on real data).

Writes <out>/feasibility.json. Exits non-zero if anything fails; tau is never
relaxed.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, random, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import e3_allocation as E3                       # noqa: E402
from moe_calib import repr_constraint as RC      # noqa: E402

SEEDS, B, ROUNDS, K = (0, 1, 2), 128, 2, 4


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default="runs_typicality")
    ap.add_argument("--matched-dir", default="runs_typicality_matched")
    ap.add_argument("--out", default="runs_repr_constrained")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rp = os.path.join(a.matched_dir, "routing_profile_v2.npz")
    z = np.load(rp)
    layers, E = z["layers"].tolist(), int(z["E"])
    C = z["C"]
    pool_ids = [int(s) for s in z["pool_ids"]]
    pools = json.load(open(os.path.join(a.pool_dir, "pools.json")))
    if sha(rp) != pools["routing_cache"]["sha256"]:
        raise SystemExit("routing cache differs from the one the typicality pools were built from")
    d, _ = RC.pool_typicality(C, len(layers), E)
    rows = list(csv.DictReader(open(os.path.join(a.pool_dir, "typicality_scores.csv"))))
    assert [int(r["id"]) for r in rows] == pool_ids, "score ids differ"
    dif = float(np.max(np.abs(np.array([float(r["d"]) for r in rows]) - d)))
    typ = set(pools["pools"]["typical_half"]["ids"])
    order = sorted(range(len(d)), key=lambda i: (d[i], pool_ids[i]))
    typ_re = {pool_ids[i] for i in order[:1000]}
    tau, rule = RC.resolve_tau("median", d)
    out = dict(routing_cache=rp, routing_sha256=sha(rp), n_pool=len(pool_ids), layers=layers, E=E,
               d_max_abs_diff_vs_scores_csv=dif, typical_half_reproduced=typ_re == typ,
               tau=tau, tau_rule=rule, d_quantiles={q: float(np.quantile(d, q)) for q in
                                                    (0, .1, .25, .5, .75, .9, 1)},
               typical_half_max_d=float(max(d[pool_ids.index(s)] for s in typ)), seeds={})
    ok = dif <= 1e-9 and typ_re == typ
    ws = json.load(open(os.path.join(a.matched_dir, "warm_starts.json")))
    if ws["pools"]["full"] != pools["pools"]["full"]["fingerprint"]:
        raise SystemExit("warm-start file was not built for this full pool")
    row = {s: i for i, s in enumerate(pool_ids)}
    for sd in SEEDS:
        W = ws["seeds"][str(sd)]["ids"]
        assert E3.warm_hash(W) == ws["seeds"][str(sd)]["hash"]
        rest = np.array([d[row[s]] for s in pool_ids if s not in set(W)])
        need = RC.minsum(rest, B - len(W))
        feas = need <= (B - len(W)) * tau
        chosen, warm, traj = E3.select_lp_utility(
            "coverage_lp_repr", C, pool_ids, layers, E, B, K, ROUNDS, random.Random(sd), None,
            warm_start=W, repr_d=d, repr_tau=tau)
        unc, _, _ = E3.select_lp_utility("coverage_lp", C, pool_ids, layers, E, B, K, ROUNDS,
                                         random.Random(sd), None, warm_start=W)
        stored = json.load(open(os.path.join(
            a.matched_dir, f"e3_coverage_lp_B128_b4_d1_s{sd}_warm-{ws['seeds'][str(sd)]['hash'][:8]}.json")))
        fin = traj[-1]["repr_final"]
        post = chosen[len(W):]
        ent = dict(warm_hash=ws["seeds"][str(sd)]["hash"], min_post_warm_sum=need,
                   budget=(B - len(W)) * tau, feasible=bool(feas), final=fin,
                   n_unique=len(set(chosen)), warm_ok=chosen[:len(W)] == W,
                   n_outside_typical_half=sum(s not in typ for s in post),
                   overlap_with_FC=len(set(chosen) & set(stored["chosen"])) / B,
                   unconstrained_replay_matches_stored_FC=unc == stored["chosen"],
                   FC_post_warm_mean_d=float(np.mean([d[row[s]] for s in stored["chosen"][len(W):]])),
                   rounds=[{k: t.get(k) for k in (
                       "round_d_sum", "round_d_cap", "round_d_mean", "paced_cap", "reserve_future",
                       "lp_fractional_d_sum", "naive_rounding_d_sum", "naive_rounding_violates",
                       "n_rounding_repairs", "rounding_objective_loss", "rounding_objective_loss_rel",
                       "n_lp_positive_value_selected", "n_zero_marginal_random_fill",
                       "n_exploration_sequences", "round_select_s", "round_objective")} for t in traj])
        ok &= feas and fin["satisfied"] and ent["n_unique"] == B and ent["warm_ok"] \
            and ent["unconstrained_replay_matches_stored_FC"]
        out["seeds"][str(sd)] = ent
        print(f"seed {sd}: feasible={feas} mean post-warm d={fin['mean_post_warm_d']:.5f} (tau {tau:.5f}), "
              f"outside typical half {ent['n_outside_typical_half']}/96, repairs "
              f"{[r['n_rounding_repairs'] for r in ent['rounds']]}, FC replay ok "
              f"{ent['unconstrained_replay_matches_stored_FC']}")
    out["all_ok"] = bool(ok)
    json.dump(out, open(os.path.join(a.out, "feasibility.json"), "w"), indent=2)
    print(f"d diff {dif:.2e}, typical half reproduced {typ_re == typ}, tau {tau:.6f}; all_ok={ok}")
    if not ok:
        raise SystemExit("FEASIBILITY CHECK FAILED")


if __name__ == "__main__":
    main()
