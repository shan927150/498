"""E1 — Label per-expert sufficiency = FUTURE held-out GPTQ benefit (ground truth).

Q1: for expert e at starting budget n0, how much does doubling the
calibration data (n0 -> 2n0) reduce its held-out reconstruction error?

Primary target: rel_delta = mean_s(err_s(n0) - err_s(2n0)) / mean_s err_s(n0).
Secondary diagnostic: paired one-sided t-test + BH-FDR "significant".

Low-count policy: routed-token count is NOT a quality threshold. An expert
is labelled whenever it has >= --min-calib-tokens base tokens (technical
minimum; default 1 = a Hessian exists), both quantizations give finite
errors, and >= --min-paired-seqs tuning sequences evaluate it at both n0 and
2n0. Every exclusion is written to excluded.csv with its reason.

Protocol (per profiled layer, one layer on the GPU at a time):
 1. Draws (fixed per characterization seed, saved in the run dir for E2):
    n0 base sequences + R disjoint extensions of n0 sequences each.
 2. Base Hessians -> accumulator A (K blocks); each extension -> accumulator
    B (K blocks, rebuilt per replicate).  H(n0) = sum A, H(2n0) = sum A + sum B.
    Partial-sum reuse: the base is profiled once for all replicates
    (cf. Canevet et al., ICML 2016).
 3. Per expert: GPTQ(H(n0)), GPTQ(H(2n0)) from FP weights; per-TUNING-
    holdout-sequence relative error; paired one-sided t-test on
    d_s = err_s(n0) - err_s(2n0)  (Korattikara et al., ICML 2014 spirit).
 4. BH-FDR within each (n0, rep) across all (layer, expert) hypotheses.
Non-sequential GPTQ: every layer sees full-precision cached inputs.
Only the TUNING holdout is used; the test holdout is reserved for E3.

Run identity: outputs go to <outdir>/e1_b{bits}_s{seed}_{hash}/ where the
hash covers the full config (moe_calib/runid.py). Same config -> resume from
raw.jsonl; different config -> new directory; mismatch -> error.
"""
from __future__ import annotations
import argparse, csv, json, os, random

import numpy as np
import torch

from moe_calib import model_utils as MU
from moe_calib.data_utils import load_meta, load_layer_cache, split_ids
from moe_calib.profiler import Profiler
from moe_calib.quant_eval import quantize_from_H, per_seq_rel_errors
from moe_calib.runid import e1_run_dir
from moe_calib.stats import paired_t_one_sided, bh_fdr

FDR_Q = 0.1
LABEL_FIELDS = ["layer", "expert", "n0", "rep", "bits", "char_seed", "tokens_n0", "t", "p",
                "delta", "rel_delta", "base_err", "n_seqs", "q", "significant"]


def label_expert(e0: dict, e1: dict, tokens: float, min_calib_tokens: int,
                 min_paired_seqs: int):
    """Continuous future-benefit label for one expert, or (None, reason).
    e0/e1: per-tuning-sequence relative errors at n0 / 2n0."""
    if tokens <= 0:
        return None, "zero_calib_tokens"
    if tokens < min_calib_tokens:
        return None, "below_min_calib_tokens"
    common = sorted(set(e0) & set(e1))
    if len(common) < min_paired_seqs:
        return None, "insufficient_paired_eval_seqs"
    a = np.array([e0[s] for s in common])
    b = np.array([e1[s] for s in common])
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None, "nonfinite_error"
    base_err = float(a.mean())
    if base_err <= 0:
        return None, "nonpositive_base_error"
    d = a - b
    t, p, _, _ = paired_t_one_sided(d)   # secondary diagnostic only
    return dict(t=t, p=p, delta=float(d.mean()), rel_delta=float(d.mean()) / base_err,
                base_err=base_err, n_seqs=len(common)), None


def make_draws(pool_ids, n0s, reps, char_seed):
    """Deterministic per (char_seed, n0): base + `reps` disjoint extensions."""
    draws = {}
    for n0 in n0s:
        rng = random.Random(char_seed * 100_003 + n0)
        idx = rng.sample(pool_ids, n0 * (1 + reps))
        draws[n0] = {"base": idx[:n0],
                     "ext": [idx[n0 * (r + 1):n0 * (r + 2)] for r in range(reps)]}
    return draws


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="default: all layers in the cache")
    ap.add_argument("--n0", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--K", type=int, default=4, help="Hessian blocks per half")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--char-seed", "--seed", dest="char_seed", type=int, default=0,
                    help="characterization seed (changes the draws)")
    ap.add_argument("--min-calib-tokens", type=int, default=1,
                    help="technical minimum of routed base tokens for a Hessian")
    ap.add_argument("--min-paired-seqs", type=int, default=8,
                    help="minimum tuning sequences evaluating the expert at both n0 and 2n0")
    ap.add_argument("--tune-size", type=int, default=None,
                    help="rows of the cached holdout used as the tuning split "
                         "(default: meta tune_size or holdout_size // 2)")
    ap.add_argument("--percdamp", type=float, default=None,
                    help="GPTQ damping (fraction of mean diag H). Recorded in the run "
                         "fingerprint and dir name. Omit only to resume legacy 1%% runs.")
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args(argv)
    percdamp = 0.01 if args.percdamp is None else args.percdamp

    meta = load_meta(args.cache_dir)
    layers = args.layers or meta["layers"]
    split = split_ids(meta, args.tune_size)
    pool_ids, tune_ids = split["pool"], split["tune"]

    fp = dict(model=MU.MODEL_NAME, data_seed=meta.get("data_seed"),
              pool_size=meta["pool_size"], holdout_size=meta["holdout_size"],
              seqlen=meta.get("seqlen"), char_seed=args.char_seed,
              layers=list(layers), n0=list(args.n0), reps=args.reps, K=args.K,
              bits=args.bits, group_size=args.group_size,
              min_calib_tokens=args.min_calib_tokens, min_paired_seqs=args.min_paired_seqs,
              fdr_q=FDR_Q, split=split["definition"])
    if args.percdamp is not None:
        fp["percdamp"] = float(args.percdamp)
    run_dir = e1_run_dir(args.outdir, fp)
    raw_path = os.path.join(run_dir, "raw.jsonl")
    draws_path = os.path.join(run_dir, "draws.json")
    print(f"[E1] run dir {run_dir} (GPTQ percdamp={percdamp:g})", flush=True)

    draws = make_draws(pool_ids, args.n0, args.reps, args.char_seed)
    if os.path.exists(draws_path):
        with open(draws_path) as f:
            stored = {int(k): v for k, v in json.load(f).items()}
        if stored != draws:
            raise RuntimeError(f"{draws_path} disagrees with the deterministic draws")
    else:
        with open(draws_path, "w") as f:
            json.dump(draws, f)

    done = set()
    if os.path.exists(raw_path):
        for line in open(raw_path):
            r = json.loads(line)
            done.add((r["layer"], r["n0"], r["rep"]))

    model, _ = MU.load_model()
    E = MU.num_experts(model)

    for L in layers:
        todo = [(n0, r) for n0 in args.n0 for r in range(args.reps)
                if (L, n0, r) not in done]
        if not todo:
            continue
        cache = load_layer_cache(args.cache_dir, L)
        HO = Profiler(model, L, collect_hessian=False, collect_holdout=True) \
            .run(cache, tune_ids).holdout
        for n0 in args.n0:
            if all((L, n0, r) in done for r in range(args.reps)):
                continue
            A = Profiler(model, L, K=args.K).run(cache, draws[n0]["base"])
            for r, ext in enumerate(draws[n0]["ext"]):
                if (L, n0, r) in done:
                    continue
                Bp = Profiler(model, L, K=args.K).run(cache, ext)
                rows = []
                for e in range(E):
                    kg, kd = (L, e, "gu"), (L, e, "down")
                    ntok = A.acc.tokens(kg)
                    if ntok <= 0 or ntok < args.min_calib_tokens:
                        _, why = label_expert({}, {}, ntok, args.min_calib_tokens, 0)
                        rows.append(dict(layer=L, expert=e, n0=n0, rep=r, tokens_n0=ntok,
                                         excluded=why, n_paired=0))
                        continue
                    Hg0, Hd0 = A.acc.H_sum(kg, list(range(args.K))), A.acc.H_sum(kd, list(range(args.K)))
                    Hg1 = Hg0 + Bp.acc.H_sum(kg, list(range(args.K)))
                    Hd1 = Hd0 + Bp.acc.H_sum(kd, list(range(args.K)))
                    e0 = per_seq_rel_errors(model, L, e, quantize_from_H(
                        model, L, e, Hg0, Hd0, args.bits, args.group_size, percdamp), HO)
                    e1 = per_seq_rel_errors(model, L, e, quantize_from_H(
                        model, L, e, Hg1, Hd1, args.bits, args.group_size, percdamp), HO)
                    lab, why = label_expert(e0, e1, ntok, args.min_calib_tokens,
                                            args.min_paired_seqs)
                    if lab is None:
                        rows.append(dict(layer=L, expert=e, n0=n0, rep=r, tokens_n0=ntok,
                                         excluded=why, n_paired=len(set(e0) & set(e1))))
                        continue
                    rows.append(dict(layer=L, expert=e, n0=n0, rep=r, bits=args.bits,
                                     char_seed=args.char_seed, tokens_n0=ntok, **lab))
                with open(raw_path, "a") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                    # marker so empty layers also count as done
                    f.write(json.dumps(dict(layer=L, n0=n0, rep=r, marker=True)) + "\n")
                done.add((L, n0, r))
                n_ok = sum("excluded" not in x for x in rows)
                print(f"[E1] L={L} n0={n0} rep={r}: {n_ok} labelled, "
                      f"{len(rows) - n_ok} excluded", flush=True)
                Bp.free(); del Bp
            A.free(); del A
        del cache, HO

    # ---- BH-FDR within each (n0, rep) across all layers/experts ----
    rows = [json.loads(l) for l in open(raw_path)]
    excl = [r for r in rows if r.get("excluded")]
    rows = [r for r in rows if not r.get("marker") and not r.get("excluded")]
    with open(os.path.join(run_dir, "excluded.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "expert", "n0", "rep", "tokens_n0",
                                          "excluded", "n_paired"])
        w.writeheader()
        w.writerows(excl)
    out = []
    for n0 in args.n0:
        for r in range(args.reps):
            grp = [x for x in rows if x["n0"] == n0 and x["rep"] == r]
            if not grp:
                continue
            q = bh_fdr(np.array([x["p"] for x in grp], dtype=float))
            for x, qq in zip(grp, q):
                x["q"] = float(qq)
                x["significant"] = int(np.isfinite(qq) and qq < FDR_Q)
                out.append(x)
            print(f"[E1] n0={n0} rep={r}: {sum(x['significant'] for x in grp)}/{len(grp)} significant@q<{FDR_Q}")
    path = os.path.join(run_dir, "labels.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"[E1] wrote {len(out)} rows -> {path}; {len(excl)} excluded "
          f"({', '.join(sorted({x['excluded'] for x in excl})) or 'none'})")
    return run_dir


if __name__ == "__main__":
    main()
