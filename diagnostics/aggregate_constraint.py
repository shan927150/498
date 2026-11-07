"""Final aggregate-shift-constrained allocation experiment: staged driver.

    --prepare     CPU. Validate routing cache / pool / warm starts, build the
                  prespecified eps_L, CPU feasibility study (explicit integer
                  certificates + MILP), construct coverage_lp_aggregate
                  selections (twice: determinism), collect the 4 baseline
                  selections from runs_typicality_matched/. Writes
                  epsilon.json, epsilon_reference.npz, feasibility.json,
                  selections.json. Reads NO reconstruction result.
    --verify-old  GPU. Scores one reused baseline selection on the OLD test
                  split with the confirmation scorer and requires the stored
                  recon_mean to be reproduced exactly (scoring-path check; uses
                  no confirmation data).
    --lock        Freezes locked_protocol.json (+ .sha256): settings, hashes,
                  every selected id list, eps, confirmation fingerprint,
                  metrics, decision rule, git commit, code hashes.
    --evaluate    GPU, read-only w.r.t. the protocol: scores every locked
                  selection on the confirmation split. Resumable; prints no
                  errors (blind until the report).

Strategies (selected ids; all start from the same ordered warm start W_s):
  FR Full Random, TR Typical Random, TC Typical Coverage-LP, FC Full
  Coverage-LP (reused ids from runs_typicality_matched/), AG
  coverage_lp_aggregate (new).
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, random, subprocess, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))

SEEDS, B, WARM = (0, 1, 2), 128, 32
LAYERS = [1, 8, 16, 23]
BASE = {"FR": ("full", "random"), "TR": ("typical_half", "random"),
        "TC": ("typical_half", "coverage_lp"), "FC": ("full", "coverage_lp")}
NAMES = {"FR": "Full Random", "TR": "Typical Random", "TC": "Typical Coverage-LP",
         "FC": "Full Coverage-LP", "AG": "Aggregate Coverage-LP"}
STRATS = ("FR", "TR", "TC", "FC", "AG")
CODE = ["e3_allocation.py", "moe_calib/aggregate_constraint.py", "moe_calib/alloc_models.py",
        "moe_calib/quant_eval.py", "moe_calib/gptq.py", "moe_calib/profiler.py", "moe_calib/hessian.py",
        "moe_calib/model_utils.py", "moe_calib/data_utils.py", "moe_calib/typicality.py",
        "diagnostics/aggregate_constraint.py", "diagnostics/report_aggregate_constraint.py",
        "diagnostics/build_confirmation_split.py", "configs/frozen_b4.json"]


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_json(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def ids_hash(ids):
    return hashlib.sha256(json.dumps([int(s) for s in ids]).encode()).hexdigest()[:16]


def git(*args):
    return subprocess.run(["git", "-c", f"safe.directory={ROOT}", *args], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- prepare

def load_routing(matched_dir, pool_dir):
    from moe_calib import aggregate_constraint as AG
    rp = os.path.join(matched_dir, "routing_profile_v2.npz")
    pools = json.load(open(os.path.join(pool_dir, "pools.json")))
    if sha_file(rp) != pools["routing_cache"]["sha256"]:
        raise SystemExit("routing cache differs from the provenance-recorded one")
    z = np.load(rp)
    assert z["layers"].tolist() == LAYERS and str(z["rule"]) == "model_topk"
    C, E = z["C"].astype(float), int(z["E"])
    pool_ids = [int(s) for s in z["pool_ids"]]
    assert pool_ids == list(range(2000)) == pools["pools"]["full"]["ids"]
    T = AG.token_invariant(C, len(LAYERS), E)
    return C, E, pool_ids, T, rp, pools


def prepare(a):
    import e3_allocation as E3
    import report_typicality_matched as RTM
    from moe_calib import aggregate_constraint as AG
    from moe_calib.frozen import load_frozen
    cfg = load_frozen(a.frozen)
    os.makedirs(a.outdir, exist_ok=True)
    C, E, pool_ids, T, rp, pools = load_routing(a.matched_dir, a.pool_dir)
    ws = json.load(open(os.path.join(a.matched_dir, "warm_starts.json")))
    assert ws["pools"]["full"] == pools["pools"]["full"]["fingerprint"]
    warm = {}
    for sd in SEEDS:
        ids = ws["seeds"][str(sd)]["ids"]
        assert E3.warm_hash(ids) == ws["seeds"][str(sd)]["hash"] and len(set(ids)) == WARM
        assert set(ids) <= set(pool_ids)
        warm[str(sd)] = ids
    # ---- thresholds ----
    t0 = time.time()
    eb = AG.build_epsilon(C, pool_ids, warm, B, n_draws=1000, eps_seed=AG.EPS_SEED,
                          n_layers=len(LAYERS), E=E)
    eps_s = time.time() - t0
    eps = eb["eps"]
    np.savez(os.path.join(a.outdir, "epsilon_reference.npz"),
             **{f"D_s{k}": v for k, v in eb["D"].items()},
             **{f"draws_s{k}": np.array(v) for k, v in eb["draws"].items()})
    ej = dict(eps=eps.tolist(), layers=LAYERS, rule=eb["rule"], eps_seed=eb["eps_seed"],
              n_draws=eb["n_draws"], tokens_per_seq_layer=T, B=B, routing_cache=rp,
              routing_sha256=sha_file(rp), pool_fingerprint=pools["pools"]["full"]["fingerprint"],
              warm_hashes={k: ws["seeds"][k]["hash"] for k in warm}, construction_s=eps_s,
              reference="p_pool = original 2,000-sequence candidate pool routing (fixed)")
    json.dump(ej, open(os.path.join(a.outdir, "epsilon.json"), "w"), indent=1)
    spec = AG.AggSpec(eb["ref"], eps, B, T, len(LAYERS), E)
    # ---- feasibility ----
    feas = dict(eps=eps.tolist(), tokens_per_seq_layer=T, token_invariant_verified=True, seeds={})
    ok_all = True
    for sd in SEEDS:
        Dd = eb["D"][str(sd)]
        sat = (Dd <= eps + AG.TOL).all(1)
        W = warm[str(sd)]
        cert = None
        if sat.any():
            i = int(np.flatnonzero(sat)[0])
            cert = W + [pool_ids[r] for r in eb["draws"][str(sd)][i]]
            assert len(set(cert)) == B and spec.ok(C[cert].sum(0))
        rest = [r for r in range(len(pool_ids)) if pool_ids[r] not in set(W)]
        t1 = time.time()
        okm, comp, inf = AG.completion_feasible(C[rest], C[W].sum(0), B - WARM, spec)
        feas["seeds"][str(sd)] = dict(
            warm_hash=ws["seeds"][str(sd)]["hash"], frac_random_draws_all_layers_ok=float(sat.mean()),
            frac_random_draws_layer_ok=(Dd <= eps + AG.TOL).mean(0).tolist(),
            explicit_integer_certificate=cert,
            explicit_certificate_D=(spec.D(C[cert].sum(0)).tolist() if cert else None),
            milp_completion_feasible=bool(okm), milp_s=time.time() - t1,
            D_warm_only_completion_free=None)
        ok_all &= bool(sat.any()) and okm
    feas["all_feasible"] = bool(ok_all)
    if not ok_all:
        json.dump(feas, open(os.path.join(a.outdir, "feasibility.json"), "w"), indent=1)
        raise SystemExit("INFEASIBLE: median-random thresholds admit no integer completion; stopping")
    # ---- selections ----
    sel = dict(strategies=NAMES, seeds={})
    for sd in SEEDS:
        W = warm[str(sd)]
        runs = {}
        t1 = time.time()
        ch, wids, traj = E3.select_lp_utility("coverage_lp_aggregate", C, pool_ids, LAYERS, E, B,
                                              cfg["e1"]["K"], 2, random.Random(sd), None, warm_start=W,
                                              agg_spec=spec)
        sel_s = time.time() - t1
        ch2, _, _ = E3.select_lp_utility("coverage_lp_aggregate", C, pool_ids, LAYERS, E, B,
                                         cfg["e1"]["K"], 2, random.Random(sd), None, warm_start=W,
                                         agg_spec=spec)
        assert ch2 == ch, "coverage_lp_aggregate selection is not deterministic"
        runs["AG"] = dict(chosen=ch, source="coverage_lp_aggregate (this experiment)", selection_s=sel_s,
                          trajectory=traj, deterministic_rerun_identical=True)
        for k, (pool, st) in BASE.items():
            f = RTM.run_path(a.matched_dir, pool, st, sd, ws)
            r = json.load(open(f))
            assert r["chosen"][:WARM] == W and len(set(r["chosen"])) == B
            runs[k] = dict(chosen=r["chosen"], source=f, source_sha256=sha_file(f))
        for k, r in runs.items():
            assert len(r["chosen"]) == B == len(set(r["chosen"])) and r["chosen"][:WARM] == W, k
            assert set(r["chosen"]) <= set(pool_ids), k
            r["ids_hash"] = ids_hash(r["chosen"])
            r["D"] = spec.D(C[r["chosen"]].sum(0)).tolist()
            r["satisfies_aggregate_bounds"] = spec.ok(C[r["chosen"]].sum(0))
        sel["seeds"][str(sd)] = runs
        feas["seeds"][str(sd)].update(
            controller_D=runs["AG"]["D"], controller_feasible=runs["AG"]["satisfies_aggregate_bounds"],
            controller_slack=(eps - np.array(runs["AG"]["D"])).tolist(),
            controller_repaired_rounds=[bool(t["repaired"]) for t in traj],
            controller_deterministic=True, controller_selection_s=sel_s)
        print(f"seed {sd}: AG D={np.round(runs['AG']['D'], 5)} eps={np.round(eps, 5)} repaired="
              f"{[t['repaired'] for t in traj]} ({sel_s:.1f}s)", flush=True)
    json.dump(sel, open(os.path.join(a.outdir, "selections.json"), "w"), indent=1)
    json.dump(feas, open(os.path.join(a.outdir, "feasibility.json"), "w"), indent=1)
    print("prepare ok")


# ---------------------------------------------------------------- lock

DECISION_RULE = {
    "primary_metric": "mean held-out (confirmation split) relative expert reconstruction error over the "
                      "240 routed experts of layers 1/8/16/23; lower is better",
    "primary_comparison": "AG vs TR; Delta_s = (E_AG,s - E_TR,s) / E_TR,s for s in 0,1,2",
    "PASS": "mean E_AG < mean E_TR AND Delta_s < 0 for all 3 seeds AND mean Delta <= -0.005 AND "
            "mean relative expert-p95 change vs TR <= +0.005 AND every layer's mean relative change vs TR "
            "<= +0.005 AND all aggregate bounds satisfied",
    "PARTIAL": "not PASS AND E_AG < E_FC on all 3 seeds AND mean per-layer aggregate L1 of AG < that of FC",
    "NO-GAIN": "otherwise",
    "RUN-BLOCKED": "fewer than 15 valid confirmation runs",
    "effect_reference": 0.005,
    "notes": "3 paired seeds; no significance claims; experts are not independent replicates",
}
SECONDARY = ["AG vs FR", "AG vs TC", "AG vs FC", "per-layer error", "expert median/p90/p95",
             "per-layer aggregate L1", "set routing JS", "expert token Gini", "median tokens per expert",
             "median effective sources", "selection overlap", "compute overhead"]


def lock(a):
    from moe_calib.frozen import load_frozen
    pth = os.path.join(a.outdir, "locked_protocol.json")
    if os.path.exists(pth):
        raise SystemExit(f"{pth} exists: the protocol is already locked (never re-locked)")
    cfg = load_frozen(a.frozen)
    sel = json.load(open(os.path.join(a.outdir, "selections.json")))
    feas = json.load(open(os.path.join(a.outdir, "feasibility.json")))
    ej = json.load(open(os.path.join(a.outdir, "epsilon.json")))
    prov = json.load(open(os.path.join(a.conf_dir, "provenance.json")))
    cmeta = json.load(open(os.path.join(a.conf_dir, "meta.json")))
    tests = json.load(open(os.path.join(a.outdir, "tests_passed.json")))
    assert feas["all_feasible"] and prov["disjoint_ok"] and tests["failed"] == 0
    assert all(sel["seeds"][str(sd)]["AG"]["satisfies_aggregate_bounds"] for sd in SEEDS)
    dirty = git("status", "--porcelain", "--", *CODE)
    proto = dict(
        experiment="final aggregate-shift-constrained allocation (confirmation)",
        locked_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        git_commit=git("rev-parse", "HEAD"), git_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        code_dirty_at_lock=bool(dirty), code_sha256={p: sha_file(os.path.join(ROOT, p)) for p in CODE},
        model=cfg["model"], gptq=cfg["gptq"], layers=cfg["layers"], K=cfg["e1"]["K"], B=B, warm=WARM,
        split=cfg["split"], seeds=list(SEEDS),
        candidate_pool_fingerprint=ej["pool_fingerprint"], routing_sha256=ej["routing_sha256"],
        warm_starts={str(sd): sel["seeds"][str(sd)]["AG"]["chosen"][:WARM] for sd in SEEDS},
        warm_hashes=ej["warm_hashes"],
        epsilon=dict(eps=ej["eps"], rule=ej["rule"], eps_seed=ej["eps_seed"], n_draws=ej["n_draws"],
                     epsilon_json_sha256=sha_file(os.path.join(a.outdir, "epsilon.json")),
                     reference_npz_sha256=sha_file(os.path.join(a.outdir, "epsilon_reference.npz"))),
        selections={str(sd): {k: dict(ids=v["chosen"], ids_hash=v["ids_hash"], source=v["source"])
                              for k, v in sel["seeds"][str(sd)].items()} for sd in SEEDS},
        selections_json_sha256=sha_file(os.path.join(a.outdir, "selections.json")),
        feasibility_json_sha256=sha_file(os.path.join(a.outdir, "feasibility.json")),
        confirmation=dict(dir=a.conf_dir, n=prov["confirmation_size"], fingerprint=prov["confirmation_fingerprint"],
                          ids_pt_sha256=sha_file(os.path.join(a.conf_dir, "ids.pt")),
                          layer_cache_sha256={str(L): sha_file(os.path.join(a.conf_dir, f"L{L}.pt"))
                                              for L in LAYERS},
                          document_disjoint=prov["disjoint_ok"], pipeline_check=cmeta["pipeline_check"]),
        strategies=NAMES, baselines={k: f"reused selected ids of runs_typicality_matched {v}" for k, v in BASE.items()},
        decision_rule=DECISION_RULE, secondary_metrics=SECONDARY, tests=tests,
        evaluation="each of 5 strategies x 3 seeds re-scored on the confirmation split with "
                   "e3_allocation.quantize_and_score(eval_cache_dir=confirmation); no old scores reused")
    json.dump(proto, open(pth, "w"), indent=1)
    h = sha_file(pth)
    open(pth + ".sha256", "w").write(h + "\n")
    os.chmod(pth, 0o444)
    print(f"locked {pth} sha256={h} commit={proto['git_commit']} dirty={proto['code_dirty_at_lock']}")


def load_protocol(outdir):
    pth = os.path.join(outdir, "locked_protocol.json")
    h = open(pth + ".sha256").read().strip()
    if sha_file(pth) != h:
        raise SystemExit("locked protocol hash mismatch: the protocol was modified after locking")
    return json.load(open(pth)), h


# ---------------------------------------------------------------- evaluate

def eval_one(model, proto, chosen, eval_cache_dir, eval_ids, cache_dir):
    import e3_allocation as E3
    t0 = time.time()
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    results, counts, _, t_h, t_q = E3.quantize_and_score(
        model, cache_dir, proto["layers"], chosen, {}, eval_ids, proto["K"], proto["gptq"]["bits"],
        proto["gptq"]["group_size"], p=None, write_back=False, percdamp=float(proto["gptq"]["percdamp"]),
        eval_cache_dir=eval_cache_dir)
    means = np.array([r["mean"] for r in results])
    return dict(recon_mean=float(means.mean()), recon_expert_median=float(np.median(means)),
                recon_expert_p90=float(np.quantile(means, .9)), recon_expert_p95=float(np.quantile(means, .95)),
                recon_per_layer={str(L): float(np.mean([r["mean"] for r in results if r["layer"] == L]))
                                 for L in proto["layers"]},
                n_experts_quantized_total=counts["n_experts_quantized_total"],
                n_experts_eval_observed=counts["n_experts_test_observed"],
                n_zero_token_fallback=counts["n_zero_token_quantization_fallback"],
                eval_universe=counts["eval_universe"], per_expert=results,
                hessian_replay_s=t_h, gptq_s=t_q, wall_s=time.time() - t0,
                peak_gpu_mem_gb=(torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None))


def evaluate(a):
    from moe_calib import model_utils as MU
    proto, h = load_protocol(a.outdir)
    prov = json.load(open(os.path.join(proto["confirmation"]["dir"], "provenance.json")))
    assert prov["confirmation_fingerprint"] == proto["confirmation"]["fingerprint"]
    for L in proto["layers"]:
        assert sha_file(os.path.join(proto["confirmation"]["dir"], f"L{L}.pt")) == \
            proto["confirmation"]["layer_cache_sha256"][str(L)], "confirmation cache changed after lock"
    ev = os.path.join(a.outdir, "eval")
    os.makedirs(ev, exist_ok=True)
    model, _ = MU.load_model()
    eval_ids = list(range(proto["confirmation"]["n"]))
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else proto["seeds"]
    for sd in seeds:
        for k in STRATS:
            ids = proto["selections"][str(sd)][k]["ids"]
            f = os.path.join(ev, f"{k}_s{sd}.json")
            if os.path.exists(f):
                r = json.load(open(f))
                if r.get("protocol_sha256") == h and r.get("ids_hash") == ids_hash(ids):
                    print(f"[skip] {k} s{sd} (complete for this protocol)", flush=True)
                    continue
            out = eval_one(model, proto, ids, proto["confirmation"]["dir"], eval_ids, a.cache_dir)
            out.update(strategy=k, seed=sd, ids_hash=ids_hash(ids), protocol_sha256=h,
                       eval_split="confirmation", chosen=ids)
            json.dump(out, open(f + ".tmp", "w"))
            os.replace(f + ".tmp", f)
            print(f"[done] {k} s{sd}", flush=True)          # no scores printed (blind)


def verify_old(a):
    """Old-test-split scoring-path check on one reused baseline (FR, seed 0)."""
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_meta, split_ids
    from moe_calib.frozen import load_frozen
    import report_typicality_matched as RTM
    cfg = load_frozen(a.frozen)
    ws = json.load(open(os.path.join(a.matched_dir, "warm_starts.json")))
    f = RTM.run_path(a.matched_dir, "full", "random", 0, ws)
    r = json.load(open(f))
    proto = dict(layers=cfg["layers"], K=cfg["e1"]["K"], gptq=cfg["gptq"])
    model, _ = MU.load_model()
    test = split_ids(load_meta(a.cache_dir))["test"]
    out = eval_one(model, proto, r["chosen"], None, test, a.cache_dir)
    res = dict(run=f, stored=r["summary"]["recon_mean"], recomputed=out["recon_mean"],
               exact=out["recon_mean"] == r["summary"]["recon_mean"],
               max_abs_expert_diff=float(max(abs(x["mean"] - y["mean"]) for x, y in zip(out["per_expert"], r["per_expert"]))))
    json.dump(res, open(os.path.join(a.outdir, "verify_old_scoring.json"), "w"), indent=1)
    print(res)
    if not res["exact"] and res["max_abs_expert_diff"] > 1e-12:
        raise SystemExit("confirmation scorer does not reproduce the stored old-split score")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs_aggregate_constraint")
    ap.add_argument("--matched-dir", default="runs_typicality_matched")
    ap.add_argument("--pool-dir", default="runs_typicality")
    ap.add_argument("--conf-dir", default=os.path.expanduser("~/moe_cache_confirmation"))
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--frozen", default="configs/frozen_b4.json")
    ap.add_argument("--seeds", default=None, help="--evaluate subset, e.g. 0,1")
    g = ap.add_mutually_exclusive_group(required=True)
    for s in ("prepare", "verify-old", "lock", "evaluate"):
        g.add_argument(f"--{s}", action="store_true")
    a = ap.parse_args(argv)
    if a.prepare:
        prepare(a)
    elif a.verify_old:
        verify_old(a)
    elif a.lock:
        lock(a)
    elif a.evaluate:
        evaluate(a)


if __name__ == "__main__":
    main()
