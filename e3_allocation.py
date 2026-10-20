"""E3 — Fixed-budget calibration allocation by predicted FUTURE GPTQ benefit.

  Phase A  routing profile of the whole pool: C[s,(L,e)] (routed tokens) and
           usage p_(L,e). (A cheap single-layer replay on the cache; its REAL
           cost is a full-model forward per candidate, charged in `cost` via
           meta.forward_s_per_seq.)
  Phase B  random warm start C0 (B/4), then `--rounds` of:
             per-expert stats on the chosen set (fixed blocks block_of(seq))
             delta_hat_e = E2 predictor for --signal (predicted rel. held-out
                           GPTQ improvement from doubling n_e -> 2 n_e)
                           ONLY inside E2's fitted count support (never
                           extrapolated)
             exploration: experts with n_e = 0 or n_e below the support have
                         no delta_hat; a greedy cover (<= round budget) gives
                         reachable ones more tokens
             no positive utility left -> seeded uniform random fill
             LP:  max sum_e w_e v_e y_e,   v_e = max(delta_hat_e, 0) / n_e
                  s.t. y_e <= sum_s C[s,e] z_s,  0 <= y_e <= n_e,
                       sum_s z_s = batch,  0 <= z_s <= 1
                  w_e = 1 (--weighting uniform) or p_e (usage)
             take top-z (or randomized rounding with --unbiased)
  Phase C  per layer: Hessians over the final set (optionally Horvitz-
           Thompson weighted), UNCHANGED GPTQ for EVERY routed expert (a fixed
           RTN fallback when a strategy gave an expert zero tokens), scored
           on the untouched TEST holdout; optional write-back + PPL.

Strategies: random | random_plus | coverage | tlp (ours) | oracle (E1
tuning-split labels used as delta_hat). `--allocation relse_precision` keeps
the old CI-width token need (n (z relSE/eps)^2 - n) as an ABLATION only.

Total cost: the summary reports `equiv_extra_random_seqs` = how many extra
random calibration sequences the same compute would have bought. Run
`random_plus --delta-seqs <that number>` for the equal-compute baseline.

Provenance: Canevet, Jose & Fleuret ICML'16 (non-uniform acquisition of an
empirical expectation kept unbiased by importance weights: --unbiased);
Korattikara et al. ICML'14 (t-test stopping rule, relse_precision ablation).
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, random, time

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog

from moe_calib import model_utils as MU
from moe_calib.data_utils import load_meta, load_layer_cache, split_ids, wikitext2_ppl
from moe_calib.hessian import assign_blocks
from moe_calib.profiler import Profiler
from moe_calib.quant_eval import quantize_from_acc_or_fallback, per_seq_rel_errors
from moe_calib.runid import DEFAULT_PERCDAMP, damp_tag
from moe_calib.utility import (SIGNALS, predicted_delta, allocation_terms,
                               exploration_cover)

DEFAULT_ROUNDS = 2
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


ROUTING_CACHE = "routing_profile_v2.npz"   # v2 = model's own topk(k) rule (post tie fix)


def cached_routing_profile(model, cache_dir, layers, pool_ids, outdir):
    """Pool routing matrix for LOGGING / diagnostics only (never charged as
    selection cost). Cached in <outdir>/routing_profile_v2.npz; the pre-fix
    routing_profile.npz (k+1 tie bug) is never read."""
    path = os.path.join(outdir, ROUTING_CACHE)
    if os.path.exists(path):
        z = np.load(path)
        if (z["layers"].tolist() == list(layers) and z["pool_ids"].tolist() == list(pool_ids)
                and str(z["rule"]) == "model_topk"):
            return z["C"], z["p"]
    C, p = routing_profile(model, cache_dir, layers, pool_ids)
    os.makedirs(outdir, exist_ok=True)
    np.savez(path, C=C, p=p, layers=np.array(layers), pool_ids=np.array(pool_ids),
             E=MU.num_experts(model), rule="model_topk")
    return C, p


def routing_shift(C, pool_ids, chosen, layers, E):
    """Selected-set vs candidate-pool routing distribution, per layer:
    L1 = ||p_C - p_D||_1 and Jensen-Shannon divergence (nats); mean over layers.
    Records the distribution shift that adaptive selection may introduce."""
    row = {s: i for i, s in enumerate(pool_ids)}
    idx = [row[s] for s in chosen if s in row]
    out = {"per_layer": {}}
    for li, L in enumerate(layers):
        cols = slice(li * E, (li + 1) * E)
        pD = C[:, cols].sum(0).astype(float)
        pC = C[idx][:, cols].sum(0).astype(float)
        pD /= max(pD.sum(), 1e-12)
        pC /= max(pC.sum(), 1e-12)
        m = 0.5 * (pD + pC)
        def kl(a, b):
            k = a > 0
            return float(np.sum(a[k] * np.log(a[k] / b[k])))
        out["per_layer"][str(L)] = dict(l1=float(np.abs(pC - pD).sum()),
                                        js=0.5 * kl(pD, m) + 0.5 * kl(pC, m),
                                        n_experts_unseen=int((pC == 0).sum()))
    out["mean_l1"] = float(np.mean([v["l1"] for v in out["per_layer"].values()]))
    out["mean_js"] = float(np.mean([v["js"] for v in out["per_layer"].values()]))
    return out


def round_schedule(B: int, K: int, rounds: int = DEFAULT_ROUNDS):
    """THE adaptive schedule (used by main and support_diagnostic.py).
    Returns (warm, [(round, chosen_size_at_decision, round_budget), ...]):
    warm start = min(B, max(K, B // 4)) random sequences; each round's
    benefit-predictor decision is made on `chosen_size_at_decision`
    sequences; the last round takes whatever is left of B."""
    warm = min(B, max(K, B // 4))
    per_round = (B - warm) // rounds
    sched, size = [], warm
    for rd in range(rounds):
        budget = per_round if rd < rounds - 1 else B - size
        if budget <= 0:
            break
        sched.append((rd, size, budget))
        size += budget
    return warm, sched


def expert_stats(model, cache_dir, layers, chosen, weights, K, block_seed):
    """Per-expert count and candidate signals over `chosen` — one layer at a
    time. Blocks come from the fixed map block_of(seq, K, block_seed), the
    same one E2 used, so a sequence never changes block between rounds."""
    E = MU.num_experts(model)
    J = len(layers) * E
    st = {"count": np.zeros(J), **{s: np.full(J, np.nan) for s in SIGNALS if s != "count"}}
    blocks = assign_blocks(chosen, K, block_seed)
    w = [weights.get(s, 1.0) for s in chosen]
    for li, L in enumerate(layers):
        cache = load_layer_cache(cache_dir, L)
        prof = Profiler(model, L, K=K).run(cache, chosen, blocks, w)
        for e in range(E):
            j, key = li * E + e, (L, e, "gu")
            st["count"][j] = prof.acc.tokens(key)
            st["relse"][j] = prof.acc.rel_std_err(key, list(range(K)))
            st["drift"][j] = prof.acc.drift(key, list(range(K)))
            st["margin"][j], st["entropy"][j] = prof.router_signals(e)
        prof.free()
        del cache
    return st


def needs_from_stats(n, relse, eps):
    """ABLATION ONLY (--allocation relse_precision): tokens until the Hessian
    CI width reaches eps. Statistical precision, NOT predicted benefit."""
    need = np.full_like(n, RARE_NEED)
    ok = (n > 0) & np.isfinite(relse)
    n_target = n[ok] * (Z_ALPHA * relse[ok] / eps) ** 2
    need[ok] = np.maximum(0.0, n_target - n[ok])
    return need


def solve_lp(Ccand, value, cap, budget):
    """max sum_j value_j y_j  s.t. y_j <= sum_s C[s,j] z_s, 0 <= y_j <= cap_j,
    sum_s z_s = budget, 0 <= z_s <= 1. Variables [z (S), y (J)].
    Returns fractional z."""
    S, J = Ccand.shape
    obj = np.concatenate([np.zeros(S), -np.asarray(value, float)])
    Csp = sparse.coo_matrix(-Ccand.T)                 # J x S
    A_ub = sparse.hstack([Csp, sparse.eye(J)]).tocsr()
    A_eq = sparse.hstack([sparse.csr_matrix(np.ones((1, S))),
                          sparse.csr_matrix((1, J))]).tocsr()
    bounds = [(0, 1)] * S + [(0, float(max(0.0, x))) for x in cap]
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


def positive_marginal_selection(z, Ccand, value, cap, budget, tol=1e-9):
    """Walk the LP's fractional solution in descending z and keep a sequence
    only if, given the sequences kept so far, it still adds positive
    objective  sum_e value_e * (min(cap_e, cov_e + C[s,e]) - min(cap_e, cov_e)).
    Sequences with zero marginal value (all useful capacity already
    saturated) are NOT kept: the LP's choice among them is solver/tie order,
    not evidence. Returns row indices into Ccand (<= budget)."""
    cov = np.zeros(Ccand.shape[1])
    keep = []
    for i in np.argsort(-z, kind="stable"):
        if len(keep) >= budget or z[i] <= tol:
            break
        new = np.minimum(cap, cov + Ccand[i])
        if float(value @ (new - np.minimum(cap, cov))) > tol:
            keep.append(int(i))
            cov = new
    return keep


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


def select_round(Cc, cand, value, cap, targets, budget, rng, unbiased):
    """One adaptive round, never more than `budget` sequences:
      1. exploration cover for `targets` (benefit undefined), <= budget;
      2. if any reachable expert still has positive utility: LP, keeping only
         sequences with positive marginal objective value; the slots left
         once useful capacity is saturated are a seeded uniform random fill;
      3. else the whole remainder is a seeded uniform random fill.
    Principle: when the validated benefit model gives no evidence for
    preferring one remaining sequence over another, acquire at random.
    (--unbiased keeps its randomized HT rounding unchanged.)"""
    explore, reached = exploration_cover(Cc, targets, budget)
    take = [cand[i] for i in explore]
    rest = budget - len(take)
    ex = set(explore)
    keep = [i for i in range(len(cand)) if i not in ex]
    eligible = (value > 0) & (cap > 0) & (Cc[keep].sum(axis=0) > 0)
    log = dict(n_exploration_targets=int(targets.sum()),
               n_exploration_targets_reached=int(reached.sum()),
               n_exploration_targets_unreached=int((targets & ~reached).sum()),
               n_exploration_sequences=len(explore),
               exploration_fraction_of_round=len(explore) / budget,
               n_positive_utility=int((value > 0).sum()),
               used_random_utility_fallback=False, n_random_fallback_sequences=0,
               n_lp_positive_value_selected=0, n_zero_marginal_random_fill=0)
    w = {}
    if rest > 0:
        pool_rest = [cand[i] for i in keep]
        if eligible.any():
            z = solve_lp(Cc[keep], value, cap, rest)
            if unbiased:
                t2, w = round_selection(z, pool_rest, rest, rng, unbiased)
                log["n_lp_positive_value_selected"] = len(t2)
            else:
                pos = positive_marginal_selection(z, Cc[keep], value, cap, rest)
                t2 = [pool_rest[i] for i in pos]
                chosen_pos = set(t2)
                fill = rng.sample([s for s in pool_rest if s not in chosen_pos], rest - len(t2))
                t2 += fill
                log.update(n_lp_positive_value_selected=len(pos),
                           n_zero_marginal_random_fill=len(fill))
        else:
            t2 = rng.sample(pool_rest, rest)
            log.update(used_random_utility_fallback=True, n_random_fallback_sequences=rest)
        take += t2
    assert len(take) <= budget
    return take, w, log


def oracle_delta(label_paths, layers, E):
    """Oracle delta_hat: E1's measured rel_delta (TUNING holdout), mean over
    starting budgets / reps / characterization seeds. NaN if never labelled."""
    agg = {}
    for path in label_paths:
        for r in csv.DictReader(open(path)):
            agg.setdefault((int(r["layer"]), int(r["expert"])), []).append(float(r["rel_delta"]))
    dh = np.full(len(layers) * E, np.nan)
    for li, L in enumerate(layers):
        for e in range(E):
            if (L, e) in agg:
                dh[li * E + e] = float(np.nanmean(agg[(L, e)]))
    return dh


def quantize_and_score(model, cache_dir, layers, chosen, weights, eval_ids, K,
                       bits, group_size, p=None, write_back=False, percdamp=0.01):
    """Quantize EVERY routed expert of every profiled layer from this
    strategy's calibration set (fixed RTN fallback for zero-token experts;
    all written back when write_back). Reconstruction is scored on the
    evaluation universe = experts with activations in `eval_ids`, which is
    built from the fixed test split alone, before any calibration data is
    used, so it is identical for every strategy."""
    E = MU.num_experts(model)
    results, orig, universe = [], {}, []
    n_fallback = n_quantized = 0
    t_h = t_q = 0.0
    for li, L in enumerate(layers):
        cache = load_layer_cache(cache_dir, L)
        HO = Profiler(model, L, collect_hessian=False, collect_holdout=True) \
            .run(cache, eval_ids).holdout
        universe += [(L, e) for e in range(E) if HO.get((L, e, "gu"))]
        fin = Profiler(model, L, K=K).run(
            cache, chosen, weights=[weights.get(s, 1.0) for s in chosen])
        t_h += fin.seconds
        tq = time.time()
        for e in range(E):
            Wq, fb = quantize_from_acc_or_fallback(model, fin.acc, L, e, bits, group_size,
                                                   percdamp)
            n_fallback += fb
            n_quantized += 1
            errs = per_seq_rel_errors(model, L, e, Wq, HO)
            if errs:
                v = np.array(list(errs.values()))
                results.append(dict(layer=L, expert=e,
                                    p=float(p[li * E + e]) if p is not None else None,
                                    tokens=fin.acc.tokens((L, e, "gu")), fallback=bool(fb),
                                    mean=float(v.mean()), p90=float(np.quantile(v, .9)),
                                    p99=float(np.quantile(v, .99))))
            if write_back:
                orig[(L, e)] = {k: w.detach().clone()
                                for k, w in MU.get_expert_weights(model, L, e).items()}
                MU.set_expert_weights(model, L, e, Wq)
        t_q += time.time() - tq
        fin.free()
        del cache, HO, fin
        print(f"[E3] layer {L} quantized", flush=True)
    counts = dict(n_experts_quantized_total=n_quantized,
                  n_experts_test_observed=len(universe),
                  n_zero_token_quantization_fallback=n_fallback,
                  eval_universe=universe)
    assert n_quantized == len(layers) * E, "every routed expert must be quantized"
    assert [(r["layer"], r["expert"]) for r in results] == universe, \
        "scored experts differ from the fixed test-split evaluation universe"
    return results, counts, orig, t_h, t_q


def check_eval_universe(outdir, split_def, layers, universe):
    """Persist the evaluation universe per (split, layers) and assert that
    every strategy run in this outdir uses the identical set."""
    key = hashlib.sha256(json.dumps([split_def, layers], sort_keys=True).encode()).hexdigest()[:10]
    path = os.path.join(outdir, f"eval_universe_{key}.json")
    universe = [list(u) for u in universe]
    if os.path.exists(path):
        stored = json.load(open(path))["eval_universe"]
        assert stored == universe, f"evaluation universe differs from {path}"
    else:
        json.dump(dict(split=split_def, layers=layers, eval_universe=universe),
                  open(path, "w"))
    return path


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True,
                    choices=["random", "random_plus", "coverage", "tlp", "oracle"])
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--K", type=int, default=None,
                    help="statistical blocks; default/required: the E2 model's K")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--signal", default="auto", choices=["auto", *SIGNALS],
                    help="predictor of future GPTQ benefit; auto = E2's selection")
    ap.add_argument("--signal-model", default=None,
                    help="E2 artifact (default: <outdir>/e2_model_b{bits}.json)")
    ap.add_argument("--weighting", default="uniform", choices=["uniform", "usage"],
                    help="w_e in the LP: 1, or routing usage p_e (ablation)")
    ap.add_argument("--allocation", default="utility",
                    choices=["utility", "relse_precision"],
                    help="relse_precision = old CI-width need (ablation only)")
    ap.add_argument("--eps", type=float, default=0.02, help="relse_precision only")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--unbiased", action="store_true")
    ap.add_argument("--delta-seqs", type=int, default=0)
    ap.add_argument("--oracle-labels", nargs="+", default=None,
                    help="default: labels.csv of the E1 runs named in the E2 model")
    ap.add_argument("--tune-size", type=int, default=None)
    ap.add_argument("--allow-preliminary", action="store_true",
                    help="accept an E2 model selected without >= 2 characterization "
                         "seeds (smoke tests only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ppl", action="store_true")
    ap.add_argument("--fp-ppl", action="store_true", help="also report unquantized PPL")
    ap.add_argument("--ppl-windows", type=int, default=40,
                    help="WikiText-2 windows of 2048 (full set ~140; 40 is a fast proxy)")
    ap.add_argument("--percdamp", type=float, default=None,
                    help="GPTQ damping; default = the E2 model's (legacy: 0.01). Selects "
                         "e2_model_b{bits}_d{percdamp}.json when --signal-model is omitted")
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    meta = load_meta(args.cache_dir)
    layers = args.layers or meta["layers"]
    split = split_ids(meta, args.tune_size)
    pool_ids, test_ids = split["pool"], split["test"]

    # ---- E2 signal model: required for tlp/oracle; split must match ----
    dtag = f"_{damp_tag(args.percdamp)}" if args.percdamp is not None else ""
    model_path = args.signal_model or os.path.join(args.outdir,
                                                   f"e2_model_b{args.bits}{dtag}.json")
    sigmodel = None
    if os.path.exists(model_path):
        sigmodel = json.load(open(model_path))
    elif args.strategy in ("tlp", "oracle"):
        raise FileNotFoundError(f"{args.strategy} needs the E2 model artifact {model_path}")
    if sigmodel is not None:
        assert sigmodel["split"] == split["definition"], (
            "E2 model was characterized on a different split; the test holdout "
            f"may have been used for tuning: {sigmodel['split']} vs {split['definition']}")
        assert sigmodel["bits"] == args.bits, "E2 model is for a different bit-width"
        sm_damp = float(sigmodel.get("percdamp", DEFAULT_PERCDAMP))
        if args.percdamp is not None and float(args.percdamp) != sm_damp:
            raise ValueError(f"--percdamp {args.percdamp} differs from the E2 model's "
                             f"GPTQ damping {sm_damp}: labels and evaluation would use "
                             "different quantizers")
        if (args.strategy in ("tlp", "oracle") and sigmodel["preliminary"]
                and not args.allow_preliminary):
            raise RuntimeError("E2 signal selection is PRELIMINARY (< 2 characterization "
                               "seeds); rerun E1 with more --char-seed values or pass "
                               "--allow-preliminary for a smoke test")
    percdamp = (float(args.percdamp) if args.percdamp is not None else
                float(sigmodel.get("percdamp", DEFAULT_PERCDAMP)) if sigmodel else DEFAULT_PERCDAMP)
    K = args.K or (sigmodel["K"] if sigmodel else 4)
    block_seed = sigmodel["block_seed"] if sigmodel else 0
    if args.strategy in ("tlp", "oracle") and K != sigmodel["K"]:
        raise ValueError(f"--K {K} differs from the E2 model's K={sigmodel['K']}: "
                         "signals would not be computed the way they were validated")
    signal = None
    if args.strategy == "tlp" and args.allocation == "utility":
        signal = sigmodel["selected_signal"] if args.signal == "auto" else args.signal

    suffix = ""
    if args.strategy == "tlp":
        suffix = (f"_{signal}" if args.allocation == "utility" else "_relseprec") + f"_{args.weighting}"
    elif args.strategy == "oracle":
        suffix = f"_{args.weighting}"
    name = (f"e3_{args.strategy}{suffix}{'_ht' if args.unbiased else ''}"
            f"_B{args.B}_b{args.bits}{dtag}_s{args.seed}")
    out_path = os.path.join(args.outdir, name + ".json")

    model, tok = MU.load_model()
    E = MU.num_experts(model)
    cost = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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
    trajectory = []
    if args.strategy == "random":
        chosen = rng.sample(pool_ids, args.B)
    elif args.strategy == "random_plus":
        want = args.B + args.delta_seqs
        if want > len(pool_ids):   # equal compute buys more than the whole pool
            print(f"[E3] random_plus wants {want} sequences > pool {len(pool_ids)}; "
                  "using the whole pool (record this: selection overhead exceeds "
                  "the cost of calibrating on every candidate)", flush=True)
        cost["random_plus_requested"] = want
        cost["random_plus_capped_at_pool"] = want > len(pool_ids)
        chosen = rng.sample(pool_ids, min(want, len(pool_ids)))
    elif args.strategy == "coverage":
        chosen = select_coverage(C, pool_ids, args.B, rng)
    else:
        w_e = p if args.weighting == "usage" else np.ones(len(layers) * E)
        oracle_dh = None
        if args.strategy == "oracle":
            paths = args.oracle_labels or [os.path.join(d, "labels.csv")
                                           for d in sigmodel["e1_runs"]]
            oracle_dh = oracle_delta(paths, layers, E)
        warm, schedule = round_schedule(args.B, K, args.rounds)
        chosen = rng.sample(pool_ids, warm)
        for rd, size, budget in schedule:
            assert len(chosen) == size, "selection drifted from round_schedule"
            st = expert_stats(model, args.cache_dir, layers, chosen, weights, K, block_seed)
            n = st["count"]
            zero = n == 0
            taken = set(chosen)
            cand = [s for s in pool_ids if s not in taken]
            Cc = C[cand]
            log = dict(round=rd, n_selected_before=len(chosen), round_budget=budget,
                       n_zero_exposure=int(zero.sum()))
            if args.allocation == "relse_precision":
                value, cap = w_e, needs_from_stats(n, st["relse"], args.eps)
                targets = np.zeros_like(zero)
            else:
                if oracle_dh is not None:
                    dh = oracle_dh.copy()
                    dh[zero] = np.nan
                    below = above = np.zeros_like(zero)
                else:
                    dh, info = predicted_delta(sigmodel, signal, st)
                    below, above = info["below"], info["above"]
                    log["n_signal_undefined_count_fallback"] = int(info["signal_undefined"].sum())
                log.update(n_below_fit_count_range=int(below.sum()),
                           n_above_fit_count_range=int(above.sum()),
                           n_out_of_support=int((below | above).sum()))
                value, cap = allocation_terms(n, dh, w_e)
                # above-support experts are NOT exploration targets: more tokens
                # cannot bring them into support; they just get no utility.
                targets = zero | below
            take, w2, rlog = select_round(Cc, cand, value, cap, targets, budget,
                                          rng, args.unbiased)
            weights.update(w2)
            chosen += take
            log.update(rlog, n_taken=len(take))
            trajectory.append(log)
            print(f"[E3] round {rd}: +{len(take)} (total {len(chosen)}), "
                  f"zero-exposure {log['n_zero_exposure']}, out-of-support "
                  f"{log.get('n_out_of_support', 0)}, exploration seqs "
                  f"{log['n_exploration_sequences']}, LP positive-value "
                  f"{log['n_lp_positive_value_selected']}, zero-marginal random fill "
                  f"{log['n_zero_marginal_random_fill']}, random fallback "
                  f"{log['n_random_fallback_sequences']}", flush=True)
        assert len(chosen) <= args.B, "global budget exceeded"
    cost["selection_s"] = time.time() - t0

    # ---- Phase C ----
    results, counts, orig, t_h, t_q = quantize_and_score(
        model, args.cache_dir, layers, chosen, weights, test_ids, K, args.bits,
        args.group_size, p=p, write_back=args.ppl, percdamp=percdamp)
    cost["hessian_replay_s"] = t_h
    cost["gptq_s"] = t_q
    cost["calib_full_forward_equiv_s"] = meta["forward_s_per_seq"] * len(chosen)
    overhead = cost.get("routing_full_forward_equiv_s", 0.0) + \
        (cost["selection_s"] if args.strategy in ("tlp", "oracle", "coverage") else 0.0)
    cost["equiv_extra_random_seqs"] = int(round(overhead / meta["forward_s_per_seq"]))

    C_log = C if C is not None else cached_routing_profile(
        model, args.cache_dir, layers, pool_ids, args.outdir)[0]
    shift = routing_shift(C_log, pool_ids, chosen, layers, E)
    means = np.array([r["mean"] for r in results])
    summary = dict(strategy=args.strategy, signal=signal,
                   signal_model=model_path if sigmodel else None,
                   signal_model_preliminary=sigmodel.get("preliminary") if sigmodel else None,
                   weighting=args.weighting, allocation=args.allocation,
                   unbiased=args.unbiased, B=args.B,
                   n_selected=len(chosen), bits=args.bits, seed=args.seed,
                   layers=layers, K=K, block_seed=block_seed,
                   split=split["definition"], eval_split="test",
                   **{k: v for k, v in counts.items() if k != "eval_universe"},
                   eval_universe_file=check_eval_universe(
                       args.outdir, split["definition"], layers, counts["eval_universe"]),
                   recon_mean=float(means.mean()),
                   recon_p90=float(np.mean([r["p90"] for r in results])),
                   recon_p99=float(np.mean([r["p99"] for r in results])),
                   percdamp=percdamp, routing_shift=shift,
                   peak_gpu_mem_gb=(torch.cuda.max_memory_allocated() / 1e9
                                    if torch.cuda.is_available() else None),
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

    json.dump(dict(summary=summary, chosen=chosen, trajectory=trajectory,
                   eval_universe=counts["eval_universe"], per_expert=results),
              open(out_path, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[E3] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
