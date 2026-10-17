"""E3 — Fixed-budget calibration allocation: t-test needs + LP selection.

  Phase A  routing profile of the whole pool: c_{s,(L,e)} and usage p_{(L,e)}.
           (Here a cheap single-layer replay on the cache; its REAL cost is a
           full-model forward per candidate, which is charged in `cost` via
           meta.forward_s_per_seq — see "Total cost?" below.)
  Phase B  random warm start C0 (B/4), then `--rounds` of:
             per-expert t-statistic CI  ->  remaining token need
                 relSE(n) ~ 1/sqrt(n)  =>  n_target = n (z relSE / eps)^2
                 need_e = max(0, n_target - n)
             LP:  max sum_e p_e y_e
                  s.t. y_e <= sum_s c_{s,e} z_s,  y_e <= need_e,
                       sum_s z_s = batch,  0 <= z_s <= 1
             take top-z (or randomized rounding with --unbiased)
  Phase C  per layer: Hessians over the final set (optionally Horvitz-
           Thompson weighted), UNCHANGED GPTQ, held-out evaluation,
           optional write-back + layer-wise WikiText-2 PPL.

Strategies: random | random_plus | coverage | tlp (ours) | oracle (E1 labels).

Total cost: the summary reports `equiv_extra_random_seqs` = how many extra
random calibration sequences the same compute would have bought. Run
`random_plus --delta-seqs <that number>` for the equal-compute baseline.

Provenance: Neyman allocation / Zhao & Zhang ICML'15 (p_e x sigma_e in the LP
objective); Canevet, Jose & Fleuret ICML'16 (non-uniform acquisition of an
empirical expectation kept unbiased by importance weights: --unbiased);
Korattikara et al. ICML'14 (t-test stopping rule -> need_e).
"""
from __future__ import annotations
import argparse, csv, json, os, random, time

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog

from moe_calib import model_utils as MU
from moe_calib.data_utils import (load_meta, load_layer_cache,
                                  pool_and_holdout_ids, wikitext2_ppl)
from moe_calib.profiler import Profiler
from moe_calib.quant_eval import quantize_from_acc, per_seq_rel_errors

Z_ALPHA = 1.645
RARE_NEED = 10_000.0


# --------------------------------------------------------------- helpers

def routing_profile(model, cache_dir, layers, pool_ids):
    """Dense C[s, j] (j = layer-major expert index) and usage p[j]."""
    E = MU.num_experts(model)
    C = np.zeros((len(pool_ids), len(layers) * E), dtype=np.float32)
    p = np.zeros(len(layers) * E)
    row = {s: i for i, s in enumerate(pool_ids)}
    for li, L in enumerate(layers):
        cache = load_layer_cache(cache_dir, L)
        prof = Profiler(model, L, collect_hessian=False).run(cache, pool_ids,
                                                             routing_only=True)
        for s, d in prof.seq_counts.items():
            for (_, e), c in d.items():
                C[row[s], li * E + e] = c
        tot = prof.tok_sum.sum().item()
        p[li * E:(li + 1) * E] = prof.tok_sum.numpy() / max(tot, 1)
        del cache
    return C, p


def expert_stats(model, cache_dir, layers, chosen, weights, K, rng):
    """Per-expert (tokens, relSE) from Hessians over `chosen` — one layer at
    a time; blocks assigned by random round-robin so selection order does
    not masquerade as between-block variance."""
    E = MU.num_experts(model)
    order = chosen[:]
    rng.shuffle(order)
    blocks = [i % K for i in range(len(order))]
    w = [weights.get(s, 1.0) for s in order]
    n = np.zeros(len(layers) * E)
    relse = np.full(len(layers) * E, np.nan)
    for li, L in enumerate(layers):
        cache = load_layer_cache(cache_dir, L)
        prof = Profiler(model, L, K=K).run(cache, order, blocks, w)
        for e in range(E):
            key = (L, e, "gu")
            n[li * E + e] = prof.acc.tokens(key)
            relse[li * E + e] = prof.acc.rel_std_err(key, list(range(K)))
        prof.free()
        del cache
    return n, relse


def needs_from_stats(n, relse, eps):
    need = np.full_like(n, RARE_NEED)
    ok = (n > 0) & np.isfinite(relse)
    n_target = n[ok] * (Z_ALPHA * relse[ok] / eps) ** 2
    need[ok] = np.maximum(0.0, n_target - n[ok])
    return need


def solve_lp(Ccand, p, need, budget):
    """Variables [z (S), y (J)]. Returns fractional z."""
    S, J = Ccand.shape
    obj = np.concatenate([np.zeros(S), -p])
    Csp = sparse.coo_matrix(-Ccand.T)                 # J x S
    A_ub = sparse.hstack([Csp, sparse.eye(J)]).tocsr()
    A_eq = sparse.hstack([sparse.csr_matrix(np.ones((1, S))),
                          sparse.csr_matrix((1, J))]).tocsr()
    bounds = [(0, 1)] * S + [(0, float(max(0.0, x))) for x in need]
    res = linprog(obj, A_ub=A_ub, b_ub=np.zeros(J), A_eq=A_eq, b_eq=[budget],
                  bounds=bounds, method="highs")
    assert res.success, res.message
    return res.x[:S]


def round_selection(z, cand, budget, rng, unbiased):
    if not unbiased:
        return [cand[i] for i in np.argsort(-z)[:budget]], {}
    # randomized rounding, HT weights 1/pi clipped at p99 (variance guard)
    pi = np.clip(z, 1e-3, 1.0)
    take = [cand[i] for i in range(len(cand)) if rng.random() < pi[i]]
    for i in np.argsort(-z):
        if len(take) >= budget:
            break
        if cand[i] not in take:
            take.append(cand[i])
    take = take[:budget]
    ci = {s: i for i, s in enumerate(cand)}
    w = np.array([1.0 / pi[ci[s]] for s in take])
    w = np.minimum(w, np.quantile(w, 0.99))
    return take, dict(zip(take, w.tolist()))


def select_coverage(C, pool_ids, B, rng, lam=64.0):
    """Stand-in coverage balancing (NOT MoEQuant/EAQuant): greedy on
    sum_e c_{s,e} / (n_e + lam) — routed-token count only."""
    C0 = rng.sample(range(len(pool_ids)), B // 4)
    n = C[C0].sum(0)
    avail = np.ones(len(pool_ids), bool)
    avail[C0] = False
    chosen = list(C0)
    while len(chosen) < B:
        v = C @ (1.0 / (n + lam))
        v[~avail] = -1
        j = int(np.argmax(v))
        chosen.append(j)
        avail[j] = False
        n += C[j]
    return [pool_ids[i] for i in chosen]


def oracle_need(labels_path, layers, E):
    agg = {}
    for r in csv.DictReader(open(labels_path)):
        agg.setdefault((int(r["layer"]), int(r["expert"])), []).append(float(r["rel_delta"]))
    ben = np.zeros(len(layers) * E)
    for li, L in enumerate(layers):
        for e in range(E):
            if (L, e) in agg:
                ben[li * E + e] = max(0.0, float(np.mean(agg[(L, e)])))
    return RARE_NEED * ben / (ben.max() + 1e-12)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True,
                    choices=["random", "random_plus", "coverage", "tlp", "oracle"])
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--unbiased", action="store_true")
    ap.add_argument("--delta-seqs", type=int, default=0)
    ap.add_argument("--oracle-labels", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ppl", action="store_true")
    ap.add_argument("--fp-ppl", action="store_true", help="also report unquantized PPL")
    ap.add_argument("--ppl-windows", type=int, default=40,
                    help="WikiText-2 windows of 2048 (full set ~140; 40 is a fast proxy)")
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    name = f"e3_{args.strategy}{'_ht' if args.unbiased else ''}_B{args.B}_b{args.bits}_s{args.seed}"
    out_path = os.path.join(args.outdir, name + ".json")

    meta = load_meta(args.cache_dir)
    layers = args.layers or meta["layers"]
    pool_ids, ho_ids = pool_and_holdout_ids(meta)
    model, tok = MU.load_model()
    E = MU.num_experts(model)
    cost = {}

    # ---- Phase A ----
    C = p = None
    if args.strategy in ("coverage", "tlp", "oracle"):
        t0 = time.time()
        C, p = routing_profile(model, args.cache_dir, layers, pool_ids)
        cost["routing_replay_s"] = time.time() - t0
        cost["routing_full_forward_equiv_s"] = meta["forward_s_per_seq"] * len(pool_ids)

    # ---- Phase B ----
    t0 = time.time()
    weights: dict[int, float] = {}
    if args.strategy == "random":
        chosen = rng.sample(pool_ids, args.B)
    elif args.strategy == "random_plus":
        chosen = rng.sample(pool_ids, args.B + args.delta_seqs)
    elif args.strategy == "coverage":
        chosen = select_coverage(C, pool_ids, args.B, rng)
    else:
        fixed_need = None
        if args.strategy == "oracle":
            lp = args.oracle_labels or os.path.join(args.outdir, f"e1_labels_b{args.bits}.csv")
            fixed_need = oracle_need(lp, layers, E)
        chosen = rng.sample(pool_ids, max(args.K, args.B // 4))
        per_round = (args.B - len(chosen)) // args.rounds
        for rd in range(args.rounds):
            budget = per_round if rd < args.rounds - 1 else args.B - len(chosen)
            if budget <= 0:
                break
            if fixed_need is None:
                n, rse = expert_stats(model, args.cache_dir, layers, chosen,
                                      weights, args.K, rng)
                need = needs_from_stats(n, rse, args.eps)
            else:
                need = fixed_need
            taken = set(chosen)
            cand = [s for s in pool_ids if s not in taken]
            z = solve_lp(C[cand], p, need, budget)
            take, w = round_selection(z, cand, budget, rng, args.unbiased)
            chosen += take
            weights.update(w)
            print(f"[E3] round {rd}: +{len(take)} (total {len(chosen)}), "
                  f"experts with need>0: {(need > 0).sum()}/{len(need)}", flush=True)
    cost["selection_s"] = time.time() - t0

    # ---- Phase C ----
    results, orig = [], {}
    t_h = t_q = 0.0
    for li, L in enumerate(layers):
        cache = load_layer_cache(args.cache_dir, L)
        HO = Profiler(model, L, collect_hessian=False, collect_holdout=True) \
            .run(cache, ho_ids).holdout
        fin = Profiler(model, L, K=args.K).run(
            cache, chosen, weights=[weights.get(s, 1.0) for s in chosen])
        t_h += fin.seconds
        tq = time.time()
        for e in range(E):
            if fin.acc.tokens((L, e, "gu")) < 32:
                continue
            Wq = quantize_from_acc(model, fin.acc, L, e, None, args.bits, args.group_size)
            errs = per_seq_rel_errors(model, L, e, Wq, HO)
            if errs:
                v = np.array(list(errs.values()))
                results.append(dict(layer=L, expert=e,
                                    p=float(p[li * E + e]) if p is not None else None,
                                    tokens=fin.acc.tokens((L, e, "gu")),
                                    mean=float(v.mean()), p90=float(np.quantile(v, .9)),
                                    p99=float(np.quantile(v, .99))))
            if args.ppl:
                orig[(L, e)] = {k: w.detach().clone()
                                for k, w in MU.get_expert_weights(model, L, e).items()}
                MU.set_expert_weights(model, L, e, Wq)
        t_q += time.time() - tq
        fin.free()
        del cache, HO, fin
        print(f"[E3] layer {L} quantized", flush=True)
    cost["hessian_replay_s"] = t_h
    cost["gptq_s"] = t_q
    cost["calib_full_forward_equiv_s"] = meta["forward_s_per_seq"] * len(chosen)
    overhead = cost.get("routing_full_forward_equiv_s", 0.0) + \
        (cost["selection_s"] if args.strategy in ("tlp", "oracle", "coverage") else 0.0)
    cost["equiv_extra_random_seqs"] = int(round(overhead / meta["forward_s_per_seq"]))

    means = np.array([r["mean"] for r in results])
    summary = dict(strategy=args.strategy, unbiased=args.unbiased, B=args.B,
                   n_selected=len(chosen), bits=args.bits, seed=args.seed,
                   layers=layers, n_experts_scored=len(results),
                   recon_mean=float(means.mean()),
                   recon_p90=float(np.mean([r["p90"] for r in results])),
                   recon_p99=float(np.mean([r["p99"] for r in results])),
                   cost=cost)
    if p is not None:
        summary["recon_usage_weighted"] = float(np.average(
            means, weights=[max(r["p"], 1e-9) for r in results]))
    if args.ppl:
        summary["wikitext2_ppl"] = wikitext2_ppl(model, tok, max_windows=args.ppl_windows)
        for (L, e), w in orig.items():
            MU.set_expert_weights(model, L, e, w)
    if args.fp_ppl:
        summary["wikitext2_ppl_fp"] = wikitext2_ppl(model, tok, max_windows=args.ppl_windows)

    json.dump(dict(summary=summary, chosen=chosen, per_expert=results),
              open(out_path, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[E3] wrote {out_path}")


if __name__ == "__main__":
    main()
