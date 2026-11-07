"""Build the candidate pools of the routing-typicality experiment (CPU, seconds).

Reads the corrected full-pool routing cache (routing_profile_v2.npz, rule
model_topk), scores every candidate with moe_calib.typicality, and writes

  <out>/typicality_scores.csv          id, d, d_L*, js, entropy, in_typical, in_random_half
  <out>/pool_typical_half.json         {name, ids, fingerprint}  (E3 --candidate-subset-file)
  <out>/pool_random_half.json          idem
  <out>/pools.json                     all pool definitions + construction metadata
  <out>/pool_diagnostics.json          score distributions, per-layer routing, sanity checks
  <out>/routing_profile_v2.npz         copy of the cache, so E3 never re-profiles

Only candidate-pool routing is read: no tuning/test data, no reconstruction
errors. The metric (mean-over-layers L1 to the ORIGINAL pool distribution)
and the pool size (1000) are fixed here, before any typicality-run result.

    python diagnostics/typicality_pools.py --routing runs_allocation_minimal/routing_profile_v2.npz
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e3_allocation import subset_fingerprint                  # noqa: E402
from moe_calib import typicality as TY                         # noqa: E402
from moe_calib.frozen import load_frozen                       # noqa: E402

POOL_SEED = 20261106          # independent of experiment seeds 0/1/2
HALF = 1000


def describe(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), sd=float(x.std()), min=float(x.min()),
                q10=float(np.quantile(x, .1)), median=float(np.median(x)),
                q90=float(np.quantile(x, .9)), max=float(x.max()))


def build(routing, frozen, out, half=HALF, pool_seed=POOL_SEED, min_size=128):
    cfg = load_frozen(frozen)
    z = np.load(routing)
    layers, E = z["layers"].tolist(), int(z["E"])
    ids = z["pool_ids"].tolist()
    P = cfg["split"]["pool"]
    if str(z["rule"]) != "model_topk":
        raise SystemExit(f"{routing} is not the corrected (model_topk) routing cache")
    if layers != list(cfg["layers"]):
        raise SystemExit(f"routing layers {layers} != frozen {cfg['layers']}")
    if ids != list(range(P[0], P[1])):
        raise SystemExit("routing rows are not the original candidate ids in cache order")
    C = z["C"]
    ref = TY.pool_reference(C, len(layers), E)
    sc = TY.sequence_scores(C, len(layers), E, ref)
    typ = TY.typical_half(ids, sc["d"], half)
    rnd = TY.random_half(ids, half, pool_seed)
    pools = {"full": ids, "typical_half": typ, "random_half": rnd}
    os.makedirs(out, exist_ok=True)
    defs = {}
    for name, p in pools.items():
        defs[name] = dict(name=name, size=len(p), fingerprint=subset_fingerprint(name, p), ids=p)
        if name != "full":
            json.dump(dict(name=name, ids=p, fingerprint=defs[name]["fingerprint"]),
                      open(os.path.join(out, f"pool_{name}.json"), "w"))
    row = {s: i for i, s in enumerate(ids)}
    diag = dict(metric="d(s) = mean_L L1(p(s,L,.), p(original pool,L,.)); layers equal weight",
                layers=layers, E=E, pool_seed=pool_seed, half=half,
                routing_cache=dict(path=routing,
                                   sha256=hashlib.sha256(open(routing, "rb").read()).hexdigest()),
                n_malformed_rows=int(sc["malformed"].sum()),
                typical_threshold_d=float(max(sc["d"][row[s]] for s in typ)),
                pools={})
    for name, p in pools.items():
        r = [row[s] for s in p]
        lay = TY.pool_reference(C[r], len(layers), E)
        diag["pools"][name] = dict(
            size=len(p), fingerprint=defs[name]["fingerprint"],
            d=describe(sc["d"][r]), js=describe(sc["js"][r]), entropy=describe(sc["entropy"][r]),
            d_by_layer={str(L): float(sc["d_layer"][r, li].mean()) for li, L in enumerate(layers)},
            routing_l1_to_original_pool_by_layer={str(L): float(np.abs(lay[li] - ref[li]).sum())
                                                  for li, L in enumerate(layers)},
            routing_by_layer={str(L): lay[li].tolist() for li, L in enumerate(layers)})
    D = diag["pools"]
    held = set(range(cfg["split"]["tune"][0], cfg["split"]["test"][1]))
    checks = {"typical_mean_d_lt_full": D["typical_half"]["d"]["mean"] < D["full"]["d"]["mean"],
              "typical_mean_d_lt_random_half": D["typical_half"]["d"]["mean"] < D["random_half"]["d"]["mean"],
              "all_pools_ge_128": all(len(set(p)) >= min_size for p in pools.values()),
              "no_tune_or_test_ids": not any(set(p) & held for p in pools.values()),
              "ids_map_to_routing_rows": all(row[s] == s - P[0] for s in ids),
              "typical_random_overlap": len(set(typ) & set(rnd))}
    diag["sanity"] = checks
    if not all(v for k, v in checks.items() if k != "typical_random_overlap"):
        raise SystemExit(f"pool sanity checks failed: {checks}")
    json.dump(dict(pool_seed=pool_seed, half=half, original_ids=ids,
                   pools={k: {kk: vv for kk, vv in v.items()} for k, v in defs.items()},
                   metric=diag["metric"], routing_cache=diag["routing_cache"]),
              open(os.path.join(out, "pools.json"), "w"))
    json.dump(diag, open(os.path.join(out, "pool_diagnostics.json"), "w"), indent=2)
    with open(os.path.join(out, "typicality_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "d", *[f"d_L{L}" for L in layers], "js", "entropy", "malformed",
                    "in_typical_half", "in_random_half"])
        st, sr = set(typ), set(rnd)
        for i, s in enumerate(ids):
            w.writerow([s, sc["d"][i], *sc["d_layer"][i], sc["js"][i], sc["entropy"][i],
                        int(sc["malformed"][i]), int(s in st), int(s in sr)])
    dst = os.path.join(out, "routing_profile_v2.npz")
    if os.path.abspath(dst) != os.path.abspath(routing):
        shutil.copyfile(routing, dst)
    return diag


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", default="runs_allocation_minimal/routing_profile_v2.npz")
    ap.add_argument("--frozen", default="configs/frozen_b4.json")
    ap.add_argument("--out", default="runs_typicality")
    a = ap.parse_args(argv)
    diag = build(a.routing, a.frozen, a.out)
    for k, v in diag["pools"].items():
        print(f"{k:13s} n={v['size']}  d mean {v['d']['mean']:.4f} (median {v['d']['median']:.4f})  "
              f"JS {v['js']['mean']:.4f}  entropy {v['entropy']['mean']:.4f}  fp {v['fingerprint']}")
    print("sanity:", diag["sanity"], " malformed rows:", diag["n_malformed_rows"],
          " typical threshold d <=", round(diag["typical_threshold_d"], 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
