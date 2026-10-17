"""E1 — Label per-expert sufficiency with a paired t-test (ground truth).

Q1: for expert e at starting budget n0, does doubling the calibration data
(n0 -> 2n0) SIGNIFICANTLY reduce its held-out reconstruction error?

Protocol (per profiled layer, one layer on the GPU at a time):
 1. Draws (fixed across layers, saved to runs/e1_draws.json for E2/E3):
    n0 base sequences + R disjoint extensions of n0 sequences each.
 2. Base Hessians -> accumulator A (K blocks); each extension -> accumulator
    B (K blocks, rebuilt per replicate).  H(n0) = sum A, H(2n0) = sum A + sum B.
    Partial-sum reuse: the base is profiled once for all replicates
    (cf. Canevet et al., ICML 2016).
 3. Per expert: GPTQ(H(n0)), GPTQ(H(2n0)) from FP weights; per-held-out-
    sequence relative error; paired one-sided t-test on
    d_s = err_s(n0) - err_s(2n0)  (Korattikara et al., ICML 2014 spirit).
 4. BH-FDR within each (n0, rep) across all (layer, expert) hypotheses.
Non-sequential GPTQ: every layer sees full-precision cached inputs.

Crash-safe: each finished (layer, n0, rep) is appended to runs/e1_raw.jsonl;
re-running the same command resumes. Final labels -> runs/e1_labels.csv.
"""
from __future__ import annotations
import argparse, csv, json, os, random

import numpy as np
import torch

from moe_calib import model_utils as MU
from moe_calib.data_utils import load_meta, load_layer_cache, pool_and_holdout_ids
from moe_calib.profiler import Profiler
from moe_calib.quant_eval import quantize_from_H, per_seq_rel_errors
from moe_calib.stats import paired_t_one_sided, bh_fdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="default: all layers in the cache")
    ap.add_argument("--n0", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--K", type=int, default=4, help="Hessian blocks per half")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-tokens", type=int, default=256)
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    tag = f"b{args.bits}"
    raw_path = os.path.join(args.outdir, f"e1_raw_{tag}.jsonl")
    draws_path = os.path.join(args.outdir, "e1_draws.json")

    meta = load_meta(args.cache_dir)
    layers = args.layers or meta["layers"]
    pool_ids, ho_ids = pool_and_holdout_ids(meta)

    # ---- draws: fixed across layers, bit-widths and reruns ----
    draws = {}
    if os.path.exists(draws_path):
        draws = {int(k): v for k, v in json.load(open(draws_path)).items()}
    for n0 in args.n0:
        if n0 not in draws:   # deterministic per (seed, n0)
            rng = random.Random(args.seed * 100_003 + n0)
            idx = rng.sample(pool_ids, n0 * (1 + args.reps))
            draws[n0] = {"base": idx[:n0],
                         "ext": [idx[n0 * (r + 1):n0 * (r + 2)] for r in range(args.reps)]}
    json.dump(draws, open(draws_path, "w"))

    done = set()
    if os.path.exists(raw_path):
        for line in open(raw_path):
            r = json.loads(line)
            done.add((r["layer"], r["n0"], r["rep"]))

    model, _ = MU.load_model()
    E = MU.num_experts(model)

    for L in layers:
        todo = [(n0, r) for n0 in args.n0 for r in range(len(draws[n0]["ext"]))
                if (L, n0, r) not in done]
        if not todo:
            continue
        cache = load_layer_cache(args.cache_dir, L)
        HO = Profiler(model, L, collect_hessian=False, collect_holdout=True) \
            .run(cache, ho_ids).holdout
        for n0 in args.n0:
            if all((L, n0, r) in done for r in range(len(draws[n0]["ext"]))):
                continue
            A = Profiler(model, L, K=args.K).run(cache, draws[n0]["base"])
            for r, ext in enumerate(draws[n0]["ext"]):
                if (L, n0, r) in done:
                    continue
                Bp = Profiler(model, L, K=args.K).run(cache, ext)
                rows = []
                for e in range(E):
                    kg, kd = (L, e, "gu"), (L, e, "down")
                    if A.acc.tokens(kg) < args.min_tokens:
                        continue
                    Hg0, Hd0 = A.acc.H_sum(kg, list(range(args.K))), A.acc.H_sum(kd, list(range(args.K)))
                    Hg1 = Hg0 + Bp.acc.H_sum(kg, list(range(args.K)))
                    Hd1 = Hd0 + Bp.acc.H_sum(kd, list(range(args.K)))
                    e0 = per_seq_rel_errors(model, L, e, quantize_from_H(
                        model, L, e, Hg0, Hd0, args.bits, args.group_size), HO)
                    e1 = per_seq_rel_errors(model, L, e, quantize_from_H(
                        model, L, e, Hg1, Hd1, args.bits, args.group_size), HO)
                    common = sorted(set(e0) & set(e1))
                    d = np.array([e0[s] - e1[s] for s in common])
                    t, p, delta, nseq = paired_t_one_sided(d)
                    base_err = float(np.mean([e0[s] for s in common])) if common else float("nan")
                    rows.append(dict(layer=L, expert=e, n0=n0, rep=r, bits=args.bits,
                                     tokens_n0=A.acc.tokens(kg), t=t, p=p, delta=delta,
                                     rel_delta=delta / base_err if base_err else float("nan"),
                                     n_seqs=nseq))
                with open(raw_path, "a") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                    # marker so empty layers also count as done
                    f.write(json.dumps(dict(layer=L, n0=n0, rep=r, marker=True)) + "\n")
                done.add((L, n0, r))
                print(f"[E1] L={L} n0={n0} rep={r}: {len(rows)} experts tested", flush=True)
                Bp.free(); del Bp
            A.free(); del A
        del cache, HO

    # ---- BH-FDR within each (n0, rep) across all layers/experts ----
    rows = [json.loads(l) for l in open(raw_path)]
    rows = [r for r in rows if not r.get("marker")]
    out = []
    for n0 in args.n0:
        for r in range(args.reps):
            grp = [x for x in rows if x["n0"] == n0 and x["rep"] == r]
            if not grp:
                continue
            q = bh_fdr(np.array([x["p"] for x in grp], dtype=float))
            for x, qq in zip(grp, q):
                x["q"] = float(qq)
                x["significant"] = int(np.isfinite(qq) and qq < 0.1)
                out.append(x)
            print(f"[E1] n0={n0} rep={r}: {sum(x['significant'] for x in grp)}/{len(grp)} significant@q<0.1")
    path = os.path.join(args.outdir, f"e1_labels_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"[E1] wrote {len(out)} rows -> {path}")


if __name__ == "__main__":
    main()
