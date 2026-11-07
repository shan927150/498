"""Matched warm starts for the matched-warm-start typicality experiment (CPU).

Reuses the pools of runs_typicality/ (F = full, T = typical_half,
H = random_half) after re-verifying their fingerprints, and draws, for each
allocation seed s, one ordered warm start
    W_s = random.Random(s).sample(sorted(T & H), 32)
used by EVERY pool x strategy of that seed (so it is valid in F, T and H).
Only pool membership is read: no tuning/test data, no reconstruction results.

Writes <out>/warm_starts.json:
  {pools: {name: fingerprint}, eligible: {ids, hash, size}, warm_size,
   seeds: {"s": {ids, hash}}, design}
and copies the routing cache into <out> so E3 never re-profiles.

The estimand is conditional on this construction: the effect of candidate
eligibility for the 96 sequences chosen AFTER a common warm start drawn from
T & H, not the unconditional effect of selecting 128 sequences from a pool.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e3_allocation import subset_fingerprint, warm_hash        # noqa: E402
from moe_calib.frozen import load_frozen                       # noqa: E402

WARM = 32


def load_pools(pool_dir):
    pj = json.load(open(os.path.join(pool_dir, "pools.json")))
    pools = {}
    for name, d in pj["pools"].items():
        ids = [int(s) for s in d["ids"]]
        if subset_fingerprint(name, ids) != d["fingerprint"]:
            raise SystemExit(f"pool {name}: stored fingerprint does not match its ids")
        if name != "full":
            f = json.load(open(os.path.join(pool_dir, f"pool_{name}.json")))
            if f["fingerprint"] != d["fingerprint"] or sorted(f["ids"]) != sorted(ids):
                raise SystemExit(f"pool_{name}.json differs from pools.json")
        pools[name] = dict(ids=ids, fingerprint=d["fingerprint"])
    return pools


def build(pool_dir, frozen, out, seeds=(0, 1, 2), warm=WARM):
    cfg = load_frozen(frozen)
    pools = load_pools(pool_dir)
    F, T, H = (set(pools[k]["ids"]) for k in ("full", "typical_half", "random_half"))
    held = set(range(cfg["split"]["tune"][0], cfg["split"]["test"][1]))
    P = cfg["split"]["pool"]
    if F != set(range(P[0], P[1])) or (T | H) - F or (F & held):
        raise SystemExit("pools are not subsets of the frozen calibration pool")
    elig = sorted(T & H)
    if len(elig) < warm:
        raise SystemExit(f"T & H has only {len(elig)} sequences < warm start {warm}; design not feasible")
    res = dict(design="W_s = random.Random(s).sample(sorted(typical_half & random_half), 32); shared by "
                      "all pools x strategies of seed s",
               pools={k: v["fingerprint"] for k, v in pools.items()},
               pool_dir=pool_dir, warm_size=warm,
               eligible=dict(size=len(elig), hash=warm_hash(elig), ids=elig), seeds={})
    for s in seeds:
        ids = random.Random(s).sample(elig, warm)
        assert len(ids) == warm == len(set(ids))
        assert all(i in F and i in T and i in H for i in ids) and not set(ids) & held
        res["seeds"][str(s)] = dict(ids=ids, hash=warm_hash(ids))
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "warm_starts.json"), "w"), indent=2)
    src, dst = os.path.join(pool_dir, "routing_profile_v2.npz"), os.path.join(out, "routing_profile_v2.npz")
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    if hashlib.sha256(open(src, "rb").read()).digest() != hashlib.sha256(open(dst, "rb").read()).digest():
        raise SystemExit("routing cache in the output dir differs from the pool dir's")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default="runs_typicality")
    ap.add_argument("--frozen", default="configs/frozen_b4.json")
    ap.add_argument("--out", default="runs_typicality_matched")
    a = ap.parse_args(argv)
    r = build(a.pool_dir, a.frozen, a.out)
    print(f"eligible T&H = {r['eligible']['size']} sequences; pools {r['pools']}")
    for s, v in r["seeds"].items():
        print(f"seed {s}: warm hash {v['hash']}  first ids {v['ids'][:6]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
