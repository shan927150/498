"""Per-expert Hessian early-stopping feasibility MVP (FR s0, C4 Random B <= 128 x 512, CANONICAL-B1 cache, layers
1/8/16/23, GPTQ 4-bit g128 percdamp 1.0 act-order; solver, routing and data unchanged).

Experiment 1 (bench): measured Hessian-replay / GPTQ timing of the production Profiler path under fixed stopping masks
  (Full, 50 % / 75 % of expert IDs stop accumulating after the 64th sequence); forward execution untouched.
Experiment 2 (checkpoints / simulate): chronological prefix Hessians H16/H32/H64/H128 (frozen manifest order), drift
  statistics, GPTQ + held-out E3 reconstruction at B = 32/64/128, offline stopping rules without future information.

Stages: freeze | bench | checkpoints | simulate | report
"""
from __future__ import annotations

import argparse, hashlib, json, os, resource, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import agq_only_validation as AV        # noqa: E402  (frozen FR selection, caches, partial-model loader)
from moe_calib import early_stop as ES  # noqa: E402

OUT = "runs_hessian_early_stop_mvp"
CONFIG = "FR_s0"
LAYERS = AV.LAYERS
E = AV.E
K = 4                                    # production K for FR s0 (contiguous blocks of 32 = B32 / B64 prefixes)
REPS = 3
STOP_AFTER = 64
MASKS = {"full": lambda e: False, "stop50": lambda e: e % 2 == 0, "stop75": lambda e: e % 4 != 3}


def p(*a):
    return os.path.join(ROOT, OUT, *a)


jdump, sha256_file = AV.jdump, AV.sha256_file


def jload(*a):
    return json.load(open(p(*a)))


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ===================================================================== freeze
def stage_freeze(a):
    os.makedirs(p("logs"), exist_ok=True)
    if os.path.exists(p("protocol.json")):
        print("[freeze] already frozen"); return
    man = AV.jload("calibration_manifest.json")["configs"][CONFIG]
    rows = man["cache_rows"]
    pre = ES.prefixes(rows)
    jdump(dict(config=CONFIG, cache=AV.ECACHE, order="frozen FR_s0 manifest order (= production E3 `chosen` order)",
               cache_rows=rows, original_chosen=man["original_chosen"], ids_hash=man["ids_hash"],
               prefixes={str(B): v for B, v in pre.items()},
               production_blocks="Profiler.run default: block(i) = (i*K)//128, K=4 -> block 0 = S32, blocks 0-1 = S64, "
                                 "all = S128 (chronological, NOT hashed); S16 = mid-block CPU snapshot",
               masks={m: [e for e in range(E) if f(e)] for m, f in MASKS.items()}, stop_after=STOP_AFTER),
          p("checkpoint_manifest.json"))
    proto = dict(
        experiment="Per-expert Hessian early-stopping feasibility MVP",
        fixed=dict(model="Qwen/Qwen1.5-MoE-A2.7B @ 1a758c50", gptq=AV.jload("protocol.json")["gptq"], layers=LAYERS,
                   calibration=f"{CONFIG}, 128 x 512, CANONICAL-B1 cache, frozen order", checkpoints=list(ES.CHECKPOINTS),
                   heldout="~/moe_cache rows 2128-2255, E3 per-expert relative reconstruction (unchanged)"),
        experiment1=dict(
            conditions=dict(full="all experts accumulate all 128 sequences",
                            stop50="expert IDs with e % 2 == 0 (30/60 per layer) skip Hessian accumulation for sequence "
                                   "positions >= 64; forward unchanged",
                            stop75="expert IDs with e % 4 != 3 (45/60 per layer), same"),
            mask_rule="fixed by expert ID, independent of quality and exposure; same for gu and down",
            timing="per layer, sequences pre-materialized in RAM; 1 warm-up replay, then REPS=3 interleaved repetitions "
                   "per condition with cuda synchronize; median + range; forward-only and hooks-without-accumulation "
                   "conditions isolate forward and hook overhead; a separate CUDA-event run splits accumulation into "
                   "fp32 conversion / GEMM X^T X / update+storage; end-to-end = replay + 720 GPTQ solves per condition",
            validity="layer outputs bitwise identical across conditions (sha256 of all 128 outputs per layer)",
            systems_gate=dict(min_hessian_replay_reduction=0.15, min_end_to_end_reduction=0.03,
                              evaluated_on="stop50 (measured)")),
        experiment2=dict(
            statistics="token-normalized trace drift and Frobenius drift of H_B vs H_{B/2}, separately for gu and down; "
                       "routed-token count n_B; relSE not used (only 1-4 contiguous blocks available, would need extra "
                       "forwards)",
            decision_points=[32, 64], warmup=16,
            rules=dict(trace="both gu and down trace drift < tau_trace", frob="both Frobenius drift < tau_frob",
                       frob_exposure="frob AND n_B >= n_min", exposure_baseline="n_B >= n_min (optional baseline)"),
            thresholds=dict(primary="tau = MEDIAN of the pooled calibration-only statistic over (expert, gu/down) at "
                                    "that checkpoint; n_min = median n_B over experts at that checkpoint",
                            grid="quantiles {0.25, 0.5, 0.75} of the same calibration-only distributions (exploratory)",
                            tuning="no separate tuning split is used; thresholds are distribution quantiles fixed a priori "
                                   "and never chosen from held-out results"),
            fixed_baselines=[32, 64, 128],
            false_safe="among stopped experts, E_stop > 1.01 E_128 (also reported at 2 %)",
            scientific_gate=dict(min_work_skipped=0.25, max_mean_degradation=0.005, max_p90_degradation=0.01,
                                 max_false_safe_rate=0.10, applies_to="the three PRIMARY (median-threshold) rules"),
            advantage="a passing rule's mean degradation is below the fixed-budget frontier (linear interpolation of "
                      "fixed B32/B64/B128 in (work skipped, mean degradation)) at its own skipped work"),
        verdict_precedence="BLOCKED > SIGNAL-FAIL > NO-ADVANTAGE > SYSTEMS-FAIL > PROMISING (moe_calib.early_stop.classify)",
        hard_constraints=["no GPTQ change", "no routing / data / order change", "no AGQ", "forward never skipped",
                          "held-out quality never used for stopping decisions", "historical artifacts untouched"],
        code_sha256={f: sha256_file(os.path.join(ROOT, f)) for f in
                     ("moe_calib/early_stop.py", "diagnostics/hessian_early_stop_mvp.py", "moe_calib/profiler.py",
                      "moe_calib/hessian.py", "moe_calib/gptq.py")},
        created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(proto, p("protocol.json"))
    print("[freeze] protocol + manifest written")


def manifest():
    return jload("checkpoint_manifest.json")


# ===================================================================== Experiment 1
class Gate:
    """Wraps BlockHessianAccumulator.add: skips the call (conversion + GEMM + update) for frozen experts at
    sequence positions >= stop_after; counts rows/calls; optional CUDA-event timing of the three sub-steps."""

    def __init__(self, acc, mask, stop_after, mode="normal", events=False):
        self.acc, self.mask, self.stop_after, self.mode, self.events = acc, mask, stop_after, mode, events
        self.pos = 0
        self.rows_done = self.rows_skipped = self.calls_done = self.calls_skipped = 0
        self.ev = []
        self._orig = acc.add
        acc.add = self.add

    def add(self, key, X, block, weight=1.0, row_weight=None):
        if X.numel() == 0:
            return
        L, e, proj = key
        if self.mode == "noacc" or (self.mask(e) and self.pos >= self.stop_after):
            self.rows_skipped += X.shape[0]; self.calls_skipped += 1
            return
        self.rows_done += X.shape[0]; self.calls_done += 1
        if not self.events:
            return self._orig(key, X, block, weight)
        a, b, c, d = (torch.cuda.Event(enable_timing=True) for _ in range(4))
        a.record()
        Xf = X.to(torch.float32)
        b.record()
        outer = Xf.T @ Xf
        c.record()
        self.acc.H[key][block] += outer
        self.acc.n[key][block] += weight * X.shape[0]
        self.acc.raw_n[key][block] += X.shape[0]
        d.record()
        self.ev.append((a, b, c, d))

    def event_seconds(self):
        sync()
        conv = sum(a.elapsed_time(b) for a, b, c, d in self.ev) / 1e3
        gemm = sum(b.elapsed_time(c) for a, b, c, d in self.ev) / 1e3
        upd = sum(c.elapsed_time(d) for a, b, c, d in self.ev) / 1e3
        return dict(conversion_s=conv, gemm_s=gemm, update_s=upd)


def release(r):
    """Free a replay result's accumulator (GPU) immediately; returns the timing dict."""
    import gc
    if isinstance(r, tuple):
        r, pr = r
        pr.free(); del pr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return r


@torch.no_grad()
def replay(model, L, X, cond, out_hash=False, events=False):
    """One timed replay of layer L over the 128 pre-materialized sequences (position order). cond in
    forward | noacc | full | stop50 | stop75. Returns seconds (inside layer_on_gpu, synchronized), counts, digest."""
    from moe_calib import model_utils as MU
    from moe_calib.profiler import Profiler
    h = hashlib.sha256() if out_hash else None
    if cond == "forward":
        with MU.layer_on_gpu(model, L):
            sync(); t0 = time.time()
            for i in range(X.shape[0]):
                y = MU.run_layer(model, L, X[i:i + 1].to(MU.GPU, non_blocking=True))
                if h is not None:
                    h.update(y.contiguous().view(torch.int16).cpu().numpy().tobytes())
            sync()
            return dict(seconds=time.time() - t0, digest=h.hexdigest() if h else None)
    pr = Profiler(model, L, K=K)
    g = Gate(pr.acc, MASKS.get(cond, MASKS["full"]), STOP_AFTER, mode="noacc" if cond == "noacc" else "normal",
             events=events)
    pr.attach()
    try:
        with MU.layer_on_gpu(model, L):
            sync(); t0 = time.time()
            for i in range(X.shape[0]):
                pr.cur_seq, pr.cur_block, pr.cur_weight = i, (i * K) // X.shape[0], 1.0
                g.pos = i
                y = MU.run_layer(model, L, X[i:i + 1].to(MU.GPU, non_blocking=True))
                if h is not None:
                    h.update(y.contiguous().view(torch.int16).cpu().numpy().tobytes())
            sync()
            secs = time.time() - t0
    finally:
        pr.detach()
        pr.acc.add = g._orig                                           # break the Gate <-> accumulator reference cycle
        g.acc = None
    out = dict(seconds=secs, rows_done=g.rows_done, rows_skipped=g.rows_skipped, calls_done=g.calls_done,
               calls_skipped=g.calls_skipped, digest=h.hexdigest() if h else None)
    if events:
        out.update(g.event_seconds())
    return out, pr


def solve_all(model, L, acc, g):
    from moe_calib.quant_eval import quantize_from_acc_or_fallback
    sync(); t0 = time.time()
    W = {}
    for e in range(E):
        W[e], fb = quantize_from_acc_or_fallback(model, acc, L, e, g["bits"], g["group_size"], g["percdamp"])
        assert not fb
    sync()
    return time.time() - t0, W


@torch.no_grad()
def stage_bench(a):
    from moe_calib.data_utils import load_layer_cache
    if os.path.exists(p("fixed_mask_benchmark.json")):
        print("[bench] done"); return
    model, _ = AV.load_partial()
    g = AV.jload("protocol.json")["gptq"]
    rows = manifest()["cache_rows"]
    conds = ["forward", "noacc", "full", "stop50", "stop75"]
    res = {c: dict(seconds=[], per_layer={}) for c in conds}
    valid, counts, events, e2e = {}, {}, {}, {}
    for L in LAYERS:
        X = load_layer_cache(AV.ECACHE, L)[rows].contiguous()          # materialize in RAM (excluded from timing)
        # validity pass (untimed): output digests per condition + realised skipping
        dig = {}
        for c in conds:
            r = release(replay(model, L, X, c, out_hash=True))
            dig[c] = r["digest"]
            if c in ("full", "stop50", "stop75"):
                counts[f"L{L}_{c}"] = {k: r[k] for k in ("rows_done", "rows_skipped", "calls_done", "calls_skipped")}
        valid[str(L)] = dict(digests=dig, outputs_identical=len(set(dig.values())) == 1)
        # warm-up then interleaved repetitions
        release(replay(model, L, X, "full"))
        for rep in range(REPS):
            for c in (conds[rep:] + conds[:rep]):                      # rotate order across repetitions
                r = release(replay(model, L, X, c))
                res[c]["per_layer"].setdefault(str(L), []).append(r["seconds"])
        ev = release(replay(model, L, X, "full", events=True))
        events[str(L)] = {k: ev[k] for k in ("conversion_s", "gemm_s", "update_s", "seconds")}
        # end-to-end per condition: replay (full K=4 accumulator) + 60 GPTQ solves for this layer
        for c in ("full", "stop50", "stop75"):
            torch.cuda.reset_peak_memory_stats()
            r, pr = replay(model, L, X, c)
            ts, W = solve_all(model, L, pr.acc, g)
            e2e.setdefault(c, {})[str(L)] = dict(replay_s=r["seconds"], solve_s=ts,
                                                  peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
                                                  peak_rss_gb=rss_gb())
            if c == "full":                                           # baseline integrity: equals historical weights
                hist = torch.load(os.path.join(AV.weights_dir("GPTQ", CONFIG), f"L{L}.pt"))
                e2e[c][str(L)]["equals_historical_weights"] = all(
                    torch.equal(W[e][k].to(torch.bfloat16).cpu(), hist[f"{e}.{k}"]) for e in range(E) for k in W[e])
            del W
            release((r, pr)); del pr
        print(f"[bench] L{L}: outputs identical {valid[str(L)]['outputs_identical']}; median replay "
              + ", ".join(f"{c} {np.median(res[c]['per_layer'][str(L)]):.2f}s" for c in conds), flush=True)
        del X
    for c in conds:
        tot = [sum(res[c]["per_layer"][str(L)][r] for L in LAYERS) for r in range(REPS)]
        res[c]["seconds"] = tot
        res[c]["median_s"] = float(np.median(tot)); res[c]["min_s"] = float(min(tot)); res[c]["max_s"] = float(max(tot))
    ev = {k: sum(events[str(L)][k] for L in LAYERS) for k in ("conversion_s", "gemm_s", "update_s", "seconds")}
    jdump(dict(conditions=res, cuda_events_full=dict(per_layer=events, total=ev), end_to_end=e2e, counts=counts,
               validity=valid, reps=REPS), p("logs", "bench_raw.json"))
    jdump(dict(validity=valid, counts=counts), p("fixed_mask_benchmark.json"))
    print("[bench] done")


# ===================================================================== Experiment 2
@torch.no_grad()
def stage_checkpoints(a):
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    from moe_calib.quant_eval import fp_weights, per_seq_rel_errors
    from moe_calib.gptq import quantize_expert
    model, _ = AV.load_partial()
    g = AV.jload("protocol.json")["gptq"]
    rows = manifest()["cache_rows"]
    test_ids = list(range(*AV.jload("protocol.json")["evaluation"]["reconstruction_rows"]))
    hist_res = {(x["layer"], x["expert"]): x["mean"] for x in AV.hist_result(CONFIG)["per_expert"]}
    z = np.load(os.path.join(ROOT, AV.OV.OUT, "logs", "capture_routing_nll.npz"))["C"]
    blocks = ES.contiguous_blocks(128, K)
    for L in LAYERS:
        outp = p("logs", f"checkpoints_L{L}.json")
        if os.path.exists(outp):
            print(f"[checkpoints] L{L} done"); continue
        li = LAYERS.index(L)
        cache = load_layer_cache(AV.ECACHE, L)
        pr = Profiler(model, L, K=K)
        t0 = time.time()
        pr.run(cache, rows[:16], blocks=blocks[:16])                  # S16 (inside block 0)
        H = {16: {(e, pj): pr.acc.H[(L, e, pj)][0].cpu().clone() for e in range(E) for pj in ("gu", "down")}}
        n = {16: {e: float(pr.acc.raw_n[(L, e, "gu")][0]) for e in range(E)}}
        pr.run(cache, rows[16:], blocks=blocks[16:])                  # continue S16 -> S128, same blocks as production
        t_rep = time.time() - t0
        for B, bl in ((32, [0]), (64, [0, 1]), (128, [0, 1, 2, 3])):
            H[B] = {(e, pj): pr.acc.H_sum((L, e, pj), bl).cpu() for e in range(E) for pj in ("gu", "down")}
            n[B] = {e: pr.acc.tokens((L, e, "gu"), bl) for e in range(E)}
        # validation: H128 = H64 + H(64:128); counts = routing profile of the prefix rows; down counts = gu counts
        val = dict(max_rel_split=0.0, counts_match=True)
        for e in range(E):
            for pj in ("gu", "down"):
                tail = pr.acc.H_sum((L, e, pj), [2, 3]).cpu()
                ref = H[128][(e, pj)]
                val["max_rel_split"] = max(val["max_rel_split"],
                                           float(torch.linalg.norm(H[64][(e, pj)] + tail - ref) / torch.linalg.norm(ref)))
                if pj == "down":
                    val["counts_match"] &= all(pr.acc.tokens((L, e, "down"), bl) == pr.acc.tokens((L, e, "gu"), bl)
                                               for bl in ([0], [0, 1], [0, 1, 2, 3]))
            for B in ES.CHECKPOINTS:
                val["counts_match"] &= int(n[B][e]) == int(z[rows[:B], li * E + e].sum())
        pr.free(); del pr
        # statistics (calibration-only)
        stats = {}
        for B in (32, 64, 128):
            stats[B] = {}
            for e in range(E):
                s = {"n": n[B][e]}
                for pj in ("gu", "down"):
                    s[pj] = dict(trace=ES.trace_drift(H[B][(e, pj)], n[B][e], H[B // 2][(e, pj)], n[B // 2][e]),
                                 frob=ES.frob_drift(H[B][(e, pj)], n[B][e], H[B // 2][(e, pj)], n[B // 2][e]))
                stats[B][e] = s
        # GPTQ at B = 32 / 64 / 128 + held-out E3 reconstruction
        HO = Profiler(model, L, collect_hessian=False, collect_holdout=True).run(load_layer_cache(AV.OCACHE, L), test_ids).holdout
        hist = torch.load(os.path.join(AV.weights_dir("GPTQ", CONFIG), f"L{L}.pt"))
        err, w_eq, t_solve = {B: {} for B in (32, 64, 128)}, True, {}
        for B in (32, 64, 128):
            t1 = time.time()
            for e in range(E):
                Wq = quantize_expert(fp_weights(model, L, e), H[B][(e, "gu")], H[B][(e, "down")], g["bits"],
                                     g["group_size"], g["percdamp"])
                v = per_seq_rel_errors(model, L, e, Wq, HO)
                err[B][e] = float(np.mean(list(v.values())))
                if B == 128:
                    w_eq &= all(torch.equal(Wq[k].to(torch.bfloat16).cpu(), hist[f"{e}.{k}"]) for k in Wq)
            t_solve[B] = time.time() - t1
        val["B128_weights_equal_historical"] = bool(w_eq)
        val["B128_recon_equal_historical"] = all(err[128][e] == hist_res[(L, e)] for e in range(E))
        jdump(dict(layer=L, tokens={str(B): {str(e): n[B][e] for e in range(E)} for B in ES.CHECKPOINTS},
                   stats={str(B): {str(e): s for e, s in st.items()} for B, st in stats.items()},
                   err={str(B): {str(e): v for e, v in d.items()} for B, d in err.items()},
                   validation=val, replay_s=t_rep, solve_and_score_s={str(B): v for B, v in t_solve.items()}), outp)
        print(f"[checkpoints] L{L}: split rel {val['max_rel_split']:.1e}, counts {val['counts_match']}, B128 weights "
              f"{val['B128_weights_equal_historical']}, recon {val['B128_recon_equal_historical']}; mean err "
              + " ".join(f"B{B} {np.mean(list(err[B].values())):.6f}" for B in (32, 64, 128)), flush=True)
        del H, cache, HO, hist
        torch.cuda.empty_cache()


# ===================================================================== simulate (offline, no model)
def load_ckpt():
    D = {L: jload("logs", f"checkpoints_L{L}.json") for L in LAYERS}
    keys = [(L, e) for L in LAYERS for e in range(E)]
    tok = {B: {(L, e): D[L]["tokens"][str(B)][str(e)] for L, e in keys} for B in ES.CHECKPOINTS}
    st = {B: {(L, e): D[L]["stats"][str(B)][str(e)] for L, e in keys} for B in (32, 64, 128)}
    err = {B: np.array([D[L]["err"][str(B)][str(e)] for L, e in keys]) for B in (32, 64, 128)}
    return D, keys, tok, st, err


def thresholds_at(st, B, q):
    vals = lambda f: np.array([v[pj][f] for v in st[B].values() for pj in ("gu", "down")])
    tr, fr = vals("trace"), vals("frob")
    nn = np.array([v["n"] for v in st[B].values()])
    return dict(tau_trace=float(np.nanquantile(tr, q)), tau_frob=float(np.nanquantile(fr, q)),
                n_min=float(np.quantile(nn, q)))


def outcome(keys, stop, err, tok, d_gu=2048, d_down=1408):
    e_stop = np.array([err[stop[k]][i] for i, k in enumerate(keys)])
    d = ES.degradation(e_stop, err[128])
    stopped = np.array([stop[k] < 128 for k in keys])
    wl = ES.workload(tok, stop, d_gu, d_down)
    return dict(mean_err=float(e_stop.mean()), mean_rel_degradation=float(e_stop.mean() / err[128].mean() - 1),
                mean_per_expert_degradation=float(d.mean()), p90_err=float(np.quantile(e_stop, .9)),
                p90_rel_degradation=float(np.quantile(e_stop, .9) / np.quantile(err[128], .9) - 1),
                worst_err=float(e_stop.max()), worst_expert_degradation=float(d.max()),
                frac_experts_improved=float((d < 0).mean()), frac_degraded_gt1pct=float((d > .01).mean()),
                false_safe_1pct=ES.false_safe(e_stop, err[128], stopped, .01),
                false_safe_2pct=ES.false_safe(e_stop, err[128], stopped, .02),
                per_layer={str(L): float(e_stop[[i for i, k in enumerate(keys) if k[0] == L]].mean()) for L in LAYERS},
                **wl)


def stage_simulate(a):
    prot = jload("protocol.json")
    gate = prot["experiment2"]["scientific_gate"]
    D, keys, tok, st, err = load_ckpt()
    val = {str(L): D[L]["validation"] for L in LAYERS}
    valid = all(v["counts_match"] and v["B128_weights_equal_historical"] and v["B128_recon_equal_historical"]
                and v["max_rel_split"] < 1e-5 for v in val.values())
    jdump(dict(per_layer=val, passed=bool(valid)), p("hessian_checkpoint_validation.json"))
    # expert statistics + fixed budgets
    pk, dec = AV.natural_exposure()
    jdump(dict(keys=[list(k) for k in keys],
               per_checkpoint={str(B): {f"{k[0]}.{k[1]}": st[B][k] for k in keys} for B in st},
               tokens={str(B): {f"{k[0]}.{k[1]}": tok[B][k] for k in keys} for B in tok},
               summary={str(B): {pj: {f: dict(median=float(np.nanmedian([st[B][k][pj][f] for k in keys])),
                                              q25=float(np.nanquantile([st[B][k][pj][f] for k in keys], .25)),
                                              q75=float(np.nanquantile([st[B][k][pj][f] for k in keys], .75)))
                                      for f in ("trace", "frob")} for pj in ("gu", "down")} for B in st}),
          p("expert_statistics.json"))
    fixed = {}
    for B in (32, 64, 128):
        fixed[f"fixed_B{B}"] = outcome(keys, {k: B for k in keys}, err, tok)
    rb = {}
    for B in (32, 64):
        d = ES.degradation(err[B], err[128])
        rb[f"B{B}"] = dict(mean=float(err[B].mean()), median=float(np.median(err[B])), p90=float(np.quantile(err[B], .9)),
                           worst=float(err[B].max()), mean_rel_vs_B128=float(err[B].mean() / err[128].mean() - 1),
                           frac_improved=float((d < 0).mean()), frac_degraded_gt1pct=float((d > .01).mean()),
                           per_layer={str(L): float(err[B][[i for i, k in enumerate(keys) if k[0] == L]].mean()) for L in LAYERS},
                           by_exposure_decile=[float(d[dec == j].mean()) for j in range(10)])
    rb["B128"] = dict(mean=float(err[128].mean()), median=float(np.median(err[128])), p90=float(np.quantile(err[128], .9)),
                      worst=float(err[128].max()))
    jdump(dict(per_expert={f"B{B}": err[B].tolist() for B in err}, keys=[list(k) for k in keys], summary=rb),
          p("reconstruction_by_budget.json"))
    # policies
    pol = {}
    for q in (0.25, 0.5, 0.75):
        thr = {B: thresholds_at(st, B, q) for B in (32, 64)}
        for rule in ("trace", "frob", "frob_exposure", "exposure"):
            stop = ES.assign_stop_budgets({B: st[B] for B in (32, 64)}, rule, thr)
            o = outcome(keys, stop, err, tok)
            o.update(rule=rule, quantile=q, thresholds={str(B): v for B, v in thr.items()},
                     primary=(q == 0.5 and rule != "exposure"))
            pol[f"{rule}_q{int(q * 100)}"] = o
    frontier = [(fixed[f"fixed_B{B}"]["work_fraction_skipped"], fixed[f"fixed_B{B}"]["mean_rel_degradation"])
                for B in (32, 64, 128)]
    for name, o in pol.items():
        o["frontier_degradation_at_same_work"] = ES.frontier_interp(frontier, o["work_fraction_skipped"])
        o["beats_frontier"] = bool(o["mean_rel_degradation"] < o["frontier_degradation_at_same_work"])
        o["passes_scientific_gate"] = bool(o["work_fraction_skipped"] >= gate["min_work_skipped"]
                                           and o["mean_rel_degradation"] <= gate["max_mean_degradation"]
                                           and o["p90_rel_degradation"] <= gate["max_p90_degradation"]
                                           and o["false_safe_1pct"]["rate"] <= gate["max_false_safe_rate"])
    jdump(dict(fixed=fixed, policies=pol, frontier=frontier), p("stopping_policy_results.json"))
    # predictive analysis (descriptive; experts not independent)
    pred = {}
    for B in (32, 64):
        d = ES.degradation(err[B], err[128])
        for pj in ("gu", "down"):
            for f in ("trace", "frob"):
                x = np.array([st[B][k][pj][f] for k in keys])
                m = np.isfinite(x)
                rx = np.argsort(np.argsort(x[m])); rd = np.argsort(np.argsort(d[m]))
                pred[f"B{B}_{pj}_{f}"] = dict(spearman_vs_degradation=float(np.corrcoef(rx, rd)[0, 1]))
        nB = np.array([tok[B][k] for k in keys], float)
        pred[f"B{B}_exposure"] = dict(spearman_vs_degradation=float(np.corrcoef(np.argsort(np.argsort(nB)),
                                                                                np.argsort(np.argsort(d)))[0, 1]))
        # separability: among experts deemed stable by each primary rule at B, fraction truly safe (<= 1 %)
        thr = thresholds_at(st, B, 0.5)
        for rule in ("trace", "frob", "frob_exposure"):
            ok = np.array([ES.rule_ok(st[B][k], rule, thr) for k in keys])
            safe = d <= 0.01
            pred[f"B{B}_{rule}_q50_confusion"] = dict(
                n_flagged_stable=int(ok.sum()), false_safe_rate=float((ok & ~safe).sum() / max(1, ok.sum())),
                unnecessarily_continued_rate=float((~ok & safe).sum() / max(1, (~ok).sum())),
                base_safe_rate=float(safe.mean()), safe_rate_if_flagged=float((ok & safe).sum() / max(1, ok.sum())))
    jdump(pred, p("false_safe_stop_analysis.json"))
    print("[simulate] validation", valid, "| fixed",
          {k: (round(v["mean_rel_degradation"] * 100, 2), round(v["work_fraction_skipped"], 3)) for k, v in fixed.items()})
    for k, o in pol.items():
        print(f"  {k}: skipped {o['work_fraction_skipped']:.3f} mean deg {o['mean_rel_degradation'] * 100:+.2f}% "
              f"p90 {o['p90_rel_degradation'] * 100:+.2f}% false-safe {o['false_safe_1pct']['rate']:.2f} "
              f"(n={o['false_safe_1pct']['n_stopped']}) frontier {o['frontier_degradation_at_same_work'] * 100:+.2f}% "
              f"pass {o['passes_scientific_gate']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    if a.stage == "report":
        import report_hessian_early_stop_mvp as R
        return R.main()
    dict(freeze=stage_freeze, bench=stage_bench, checkpoints=stage_checkpoints, simulate=stage_simulate)[a.stage](a)


if __name__ == "__main__":
    main()
