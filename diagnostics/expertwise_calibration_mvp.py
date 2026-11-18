"""Expert-wise activation calibration MVP (Qwen1.5-MoE-A2.7B, layer 16, 8 experts, B = 128, seeds 0/1/2).

Diagnostic only: does relaxing the shared-sequence constraint (each expert selects its own 128 calibration
sequences) improve held-out isolated-expert GPTQ reconstruction, once per-expert routed-token counts are matched?
Conditions SR / SC / EC / ER (moe_calib/expertwise_calibration.py). Not a replacement for the fixed global
sequence budget E3 protocol; no full-model PPL.

Stages (--stage):
  protocol   choose the 8 experts from the existing pool routing profile; freeze all settings
  sketch     replay the 2,000 pool sequences once: per (seq, expert, proj) 64-d covariance sketches + counts,
             full-dimensional pool reference (diagnostic only); also time a routing-only replay for cost
  select     --seed s : SR / SC / EC / ER selections (CPU, sketches only), determinism + validity checks
  run        --seed s --tag smoke|final : token matching, aligned capture of matched Hessians, frozen GPTQ,
             held-out evaluation (test rows 2128-2255 primary, confirmation split secondary)
  freeze     after a passing smoke run: record code hashes; final runs refuse to start if they changed
  report     summary.json / summary.md / figures from the final runs
"""
from __future__ import annotations

import argparse, hashlib, json, os, resource, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_calib import expertwise_calibration as EW   # noqa: E402

OUT = "runs_expertwise_mvp"
STORE = os.path.expanduser("~/moe_cache_expertwise_mvp")
L = 16
CACHE = os.path.expanduser("~/moe_cache")
POOL = list(range(2000))
EVAL = {"test": dict(cache=os.path.expanduser("~/moe_cache"), rows=[2128, 2256], role="PRIMARY",
                     note="historical E3 test rows (never used for selection or references)"),
        "confirm": dict(cache=os.path.expanduser("~/moe_cache_confirmation"), rows=[0, 256], role="SECONDARY",
                        note="document-disjoint confirmation split")}
TUNE_ROWS = [2000, 2128]
ROUTING_NPZ = "runs_allocation_minimal/routing_profile_v2.npz"
SETTINGS = dict(layer=L, n_experts=8, B=128, seeds=[0, 1, 2], seq_len=512, sketch_dim=64, projection_seed=20261116,
                greedy_sample_frac=0.1, swap_max_passes=5, swap_rel_tol=1e-9, eps=EW.EPS,
                gptq=dict(bits=4, group_size=128, percdamp=1.0, actorder=True),
                holdout_tokens_per_seq=128)
FROZEN_CODE = ("moe_calib/gptq.py", "moe_calib/quant_eval.py", "moe_calib/profiler.py", "moe_calib/hessian.py",
               "moe_calib/model_utils.py", "e3_allocation.py")
EXP_CODE = ("moe_calib/expertwise_calibration.py", "diagnostics/expertwise_calibration_mvp.py")


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def s(*a):
    return os.path.join(STORE, *a)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def jload(*a):
    return json.load(open(p(*a)))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def hashes(files):
    return {f: sha256_file(os.path.join(ROOT, f)) for f in files}


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def gpu_peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def experts():
    return jload("protocol.json")["experts"]


def load_model():
    os.environ["MOE_LOAD_LAYERS"] = str(L)
    from moe_calib import model_utils as MU
    t0 = time.time()
    model, _ = MU.load_model()
    return model, time.time() - t0


def expert_digest(model, ex):
    from moe_calib import model_utils as MU
    h = hashlib.sha256()
    for e in range(MU.num_experts(model)):
        for k, w in MU.get_expert_weights(model, L, e).items():
            h.update(w.detach().contiguous().cpu().view(torch.int16).numpy().tobytes())
    h.update(MU.moe_block(model, L).gate.weight.detach().contiguous().cpu().view(torch.int16).numpy().tobytes())
    return h.hexdigest()


# ===================================================================== protocol
def stage_protocol(a):
    if os.path.exists(p("protocol.json")):
        print("[protocol] already frozen"); return
    z = np.load(os.path.join(ROOT, ROUTING_NPZ))
    assert z["layers"].tolist() == [1, 8, 16, 23] and z["pool_ids"].tolist() == POOL and str(z["rule"]) == "model_topk"
    li = z["layers"].tolist().index(L)
    counts = z["C"][:, li * 60:(li + 1) * 60].sum(0)
    ex = EW.select_experts(counts, SETTINGS["n_experts"])
    order = np.lexsort((np.arange(60), counts))
    jdump(dict(
        experiment="Expert-wise activation calibration MVP (diagnostic; not a replacement for fixed-budget E3)",
        model="Qwen/Qwen1.5-MoE-A2.7B (partial load: layer 16 only, BF16)",
        settings=SETTINGS, experts=ex,
        expert_rule="L16 candidate-pool routed-token counts (rows 0-1999, " + ROUTING_NPZ + ", Profiler routing, "
                    "model top-k); ascending sort (ties -> lower id); ranks round(i*59/7), i=0..7",
        expert_ranks={str(e): int(np.where(order == e)[0][0]) for e in ex},
        expert_pool_tokens={str(e): int(counts[e]) for e in ex},
        layer_pool_tokens=dict(min=int(counts.min()), median=float(np.median(counts)), max=int(counts.max())),
        candidate_pool=dict(cache=CACHE, rows=[0, 2000]), tune_rows_unused=TUNE_ROWS, evaluation=EVAL,
        conditions=dict(SR="one uniform random set of 128 sequences shared by all experts",
                        SC="one set minimizing mean_e D_e(S) (seeded stochastic greedy + best-improvement swaps)",
                        EC="per expert, S_e minimizing D_e(S_e) (same optimizer)",
                        ER="per expert, an independent uniform random set of 128"),
        objective="D_e(S) = 1/2 sum_{p in gu,down} ||H_ep(S) - H_ep^pool||_F / (||H_ep^pool||_F + eps), 64-d Gaussian "
                  "sketches z = R_p^T x (N(0,1/64), fixed seed), H token-normalized; empty-token -> H(S) := 0",
        optimizer="stochastic greedy (each step: argmin over a seeded uniform 10% sample of remaining candidates), "
                  "then swap passes visiting slots in seeded order, best out-of-set replacement scored in closed form, "
                  "accepted only if the exact objective decreases (rel tol 1e-9); <= 5 passes",
        rng_streams="SeedSequence([seed, cond]) for SR/SC, ([seed, cond, e]) for EC/ER, ([seed, 7919, cond, e]) "
                    "for token matching",
        token_matching="k_e = min over SR/SC/EC/ER of routed-token count; per condition choose exactly k_e of the "
                       "(seq, dispatch-row) pairs uniformly without replacement; the same row indices address the "
                       "gate/up input and the down input (alignment checked bitwise every sequence)",
        gptq="moe_calib.gptq.quantize_expert unchanged: 4-bit, g128, act-order, percdamp 1.0; H = sum x x^T fp32",
        metric="moe_calib.quant_eval.per_seq_rel_errors (E3 definition, gate+up+down), mean over held-out sequences, "
               "128 routed tokens/seq cap (Profiler default); summary = mean over the 8 experts",
        decision_rule=dict(
            paired_improvement="r_e(X over Y) = (err_Y - err_X) / err_Y per expert; seed-level = mean over 8 experts",
            consistent="seed-level r > 0 in all 3 seeds AND seed-averaged r_e > 0 for >= 6/8 experts",
            material="consistent AND 3-seed mean r >= 0.5 %",
            primary="EC vs SC (constraint relaxation under the covariance surrogate)",
            secondary=["EC vs ER (surrogate, expert-wise)", "SC vs SR (surrogate, shared)",
                       "ER vs SR (constraint relaxation with random selection: independence / noise reference)"]),
        secondary_analysis="UNMATCHED: the same four selections quantized with ALL routed tokens (token quantity not "
                           "controlled); reported separately, never mixed with the primary matched result",
        frozen_code_sha256=hashes(FROZEN_CODE), created=time.strftime("%Y-%m-%d %H:%M:%S")), p("protocol.json"))
    print("[protocol] experts", ex)


# ===================================================================== sketch
@torch.no_grad()
def stage_sketch(a):
    if os.path.exists(s("sketch.pt")) and os.path.exists(p("logs", "sketch.json")):
        print("[sketch] done"); return
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    os.makedirs(STORE, exist_ok=True)
    ex = experts()
    model, t_load = load_model()
    dig0 = expert_digest(model, ex)
    dims = {"gu": model.config.hidden_size, "down": model.config.moe_intermediate_size}
    k = SETTINGS["sketch_dim"]
    Rp = {q: r.to(MU.GPU) for q, r in EW.projection_matrices(dims, k, SETTINGS["projection_seed"]).items()}
    iu, w = EW.svech_index(k, MU.GPU)
    V = len(w)
    N = len(POOL)
    M = torch.zeros(len(ex), 2, N, V, dtype=torch.float64)
    n = torch.zeros(len(ex), N, dtype=torch.int64)
    Hfull = {(e, q): torch.zeros(dims[q], dims[q], dtype=torch.float32, device=MU.GPU) for e in ex for q in EW.PROJS}
    eidx = {e: i for i, e in enumerate(ex)}
    cache = load_layer_cache(CACHE, L)

    def cb(sid, e, Xg, Xd):
        i = eidx[e]
        n[i, sid] = Xg.shape[0]
        if Xg.shape[0] == 0:
            return
        for j, (q, X) in enumerate((("gu", Xg), ("down", Xd))):
            M[i, j, sid] = EW.svech(EW.sketch_outer(X, Rp[q]), iu, w).cpu()
            Xf = X.float()
            Hfull[(e, q)] += Xf.T @ Xf
    torch.cuda.reset_peak_memory_stats()
    with MU.layer_on_gpu(model, L):
        # hooks must not change layer outputs (first 16 sequences)
        plain = [MU.run_layer(model, L, cache[i:i + 1].to(MU.GPU)).cpu() for i in range(16)]
        sync(); t0 = time.time()
        info = EW.replay(model, L, cache, POOL, ex, cb)
        sync(); t_sketch = time.time() - t0
        hooked = [MU.run_layer(model, L, cache[i:i + 1].to(MU.GPU)).cpu() for i in range(16)]
        gpu_sketch = gpu_peak_gb()
        # cost reference: plain replay (no hooks) of the whole pool
        sync(); t0 = time.time()
        for sid in POOL:
            MU.run_layer(model, L, cache[sid:sid + 1].to(MU.GPU))
        sync(); t_plain = time.time() - t0
    # routing-only Profiler pass (E3's routing-profile cost)
    t0 = time.time()
    prof = Profiler(model, L, collect_hessian=False).run(cache, POOL, routing_only=True)
    t_route = time.time() - t0
    z = np.load(os.path.join(ROOT, ROUTING_NPZ))
    li = z["layers"].tolist().index(L)
    counts_ok = all(int(n[eidx[e], sid]) == int(z["C"][sid, li * 60 + e]) == prof.seq_counts.get(sid, {}).get((L, e), 0)
                    for e in ex for sid in POOL)
    ref = {f"{e}.{q}": Hfull[(e, q)].cpu() for e in ex for q in EW.PROJS}
    torch.save(dict(experts=ex, pool=POOL, M=M, n=n, svech_dim=V, sketch_dim=k, dims=dims,
                    projection_seed=SETTINGS["projection_seed"], Hfull=ref,
                    Nfull={str(e): int(n[eidx[e]].sum()) for e in ex}), s("sketch.pt"))
    val = dict(n_alignment_checks=info["n_alignment_checks"],
               alignment_checks_expected=len(POOL) * len(ex),
               outputs_identical_with_hooks=all(torch.equal(x, y) for x, y in zip(plain, hooked)),
               counts_equal_routing_profile_and_profiler=bool(counts_ok),
               weights_unchanged=expert_digest(model, ex) == dig0,
               sketch_finite=bool(torch.isfinite(M).all()),
               pool_tokens={str(e): int(n[eidx[e]].sum()) for e in ex},
               seqs_with_zero_tokens={str(e): int((n[eidx[e]] == 0).sum()) for e in ex},
               tokens_per_seq={str(e): dict(mean=float(n[eidx[e]].float().mean()), max=int(n[eidx[e]].max()))
                               for e in ex},
               full_ref_finite=all(bool(torch.isfinite(v).all()) for v in ref.values()),
               cost=dict(model_load_s=t_load, sketch_replay_s=t_sketch, plain_replay_s=t_plain,
                         routing_only_profiler_s=t_route, extra_over_plain_s=t_sketch - t_plain,
                         extra_over_routing_profile_s=t_sketch - t_route, peak_gpu_gb=gpu_sketch,
                         peak_rss_gb=rss_gb(), sketch_file_mb=os.path.getsize(s("sketch.pt")) / 1e6,
                         sketch_only_mb=M.numel() * 8 / 1e6 + n.numel() * 8 / 1e6),
               sketch_sha256=sha256_file(s("sketch.pt")), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    val["passed"] = bool(val["n_alignment_checks"] == val["alignment_checks_expected"] and
                         val["outputs_identical_with_hooks"] and counts_ok and val["weights_unchanged"] and
                         val["sketch_finite"] and val["full_ref_finite"])
    jdump(val, p("logs", "sketch.json"))
    print(f"[sketch] passed={val['passed']} sketch {t_sketch:.0f}s plain {t_plain:.0f}s route {t_route:.0f}s "
          f"gpu {gpu_sketch:.2f} GB file {val['cost']['sketch_file_mb']:.0f} MB")


def load_sketch():
    d = torch.load(s("sketch.pt"))
    ex = d["experts"]
    assert ex == experts()
    M = {(e, q): d["M"][i, j] for i, e in enumerate(ex) for j, q in enumerate(EW.PROJS)}
    n = {e: d["n"][i].double() for i, e in enumerate(ex)}
    return d, M, n


# ===================================================================== select
def stage_select(a):
    out = p("selections", f"seed{a.seed}.json")
    if os.path.exists(out):
        print(f"[select] seed {a.seed} done"); return
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    d, M, n = load_sketch()
    ex = d["experts"]
    B = SETTINGS["B"]
    t0 = time.time()
    r1 = EW.select_all(M, n, ex, B, a.seed, SETTINGS["greedy_sample_frac"], SETTINGS["swap_max_passes"])
    t_sel = time.time() - t0
    r2 = EW.select_all(M, n, ex, B, a.seed, SETTINGS["greedy_sample_frac"], SETTINGS["swap_max_passes"])
    sel = r1["selection"]
    EW.validate_selection(sel, ex, B, len(POOL))
    det = sel == r2["selection"]
    D = {c: {str(e): EW.expert_mismatch(M, n, e, sel[c][e]) for e in ex} for c in EW.CONDITIONS}
    counts = {c: {str(e): int(n[e][sel[c][e]].sum()) for e in ex} for c in EW.CONDITIONS}
    opt = r1["optimizer"]
    mono = all(all(y <= x for x, y in zip(i["trace"], i["trace"][1:]))
               for i in [opt["SC"], *opt["EC"].values()])
    jdump(dict(seed=a.seed, experts=ex, selection={c: {str(e): v for e, v in sel[c].items()} for c in EW.CONDITIONS},
               deterministic_rerun_identical=bool(det), objective_monotone=bool(mono),
               sketch_mismatch_pre_matching=D, routed_tokens=counts,
               quota={str(e): min(counts[c][str(e)] for c in EW.CONDITIONS) for e in ex},
               overlap=EW.overlap_stats(sel, ex),
               optimizer=dict(SC={k: v for k, v in opt["SC"].items()},
                              EC={str(e): v for e, v in opt["EC"].items()}),
               SC_mean_objective={c: float(np.mean([D[c][str(e)] for e in ex])) for c in EW.CONDITIONS},
               selection_cpu_s=t_sel, created=time.strftime("%Y-%m-%d %H:%M:%S")), out)
    print(f"[select] seed {a.seed}: {t_sel:.1f}s det={det} mono={mono} mean D:",
          {c: round(float(np.mean(list(D[c].values()))), 4) for c in EW.CONDITIONS})


# ===================================================================== run (capture + GPTQ + evaluate)
@torch.no_grad()
def stage_run(a):
    tag = a.tag
    out = p("results", f"{tag}_seed{a.seed}.json")
    if os.path.exists(out):
        print(f"[run] {tag} seed {a.seed} done"); return
    proto = jload("protocol.json")
    assert hashes(FROZEN_CODE) == proto["frozen_code_sha256"], "solver / evaluation / E3 code changed since protocol"
    if tag == "final":
        fz = jload("freeze.json")
        assert hashes(EXP_CODE) == fz["experiment_code_sha256"], "experiment code changed after freeze"
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.gptq import quantize_expert
    from moe_calib.profiler import Profiler
    from moe_calib.quant_eval import fp_weights, per_seq_rel_errors
    G = SETTINGS["gptq"]
    selj = jload("selections", f"seed{a.seed}.json")
    d, Msk, nsk = load_sketch()
    ex = d["experts"]
    sel = {c: {e: selj["selection"][c][str(e)] for e in ex} for c in EW.CONDITIONS}
    EW.validate_selection(sel, ex, SETTINGS["B"], len(POOL))
    nint = {e: nsk[e].long().numpy() for e in ex}
    t0 = time.time()
    plan = EW.matching_plan(sel, lambda e: nint[e], ex, a.seed)
    plan2 = EW.matching_plan(sel, lambda e: nint[e], ex, a.seed)
    t_plan = time.time() - t0
    plan_det = all(plan["plan"][c][e].keys() == plan2["plan"][c][e].keys() and
                   all(torch.equal(plan["plan"][c][e][q], plan2["plan"][c][e][q]) for q in plan["plan"][c][e])
                   for c in EW.CONDITIONS for e in ex)
    member = {}                                              # seq -> [(c, e)]
    for c in EW.CONDITIONS:
        for e in ex:
            for q in sel[c][e]:
                member.setdefault(q, []).append((c, e))
    union = sorted(member)
    model, t_load = load_model()
    dig0 = expert_digest(model, ex)
    dims = d["dims"]
    k = d["sketch_dim"]
    Rp = {q: r.to(MU.GPU) for q, r in EW.projection_matrices(dims, k, d["projection_seed"]).items()}
    iu, w = EW.svech_index(k, MU.GPU)
    keys = [(c, e, q) for c in EW.CONDITIONS for e in ex for q in EW.PROJS]
    Hm = {kk: torch.zeros(dims[kk[2]], dims[kk[2]], device=MU.GPU) for kk in keys}
    Hu = {kk: torch.zeros(dims[kk[2]], dims[kk[2]], device=MU.GPU) for kk in keys}
    Zm = {kk: torch.zeros(len(w), dtype=torch.float64) for kk in keys}
    Zu = {kk: torch.zeros(len(w), dtype=torch.float64) for kk in keys}
    cm = {(c, e): 0 for c in EW.CONDITIONS for e in ex}
    cu = {(c, e): 0 for c in EW.CONDITIONS for e in ex}
    chk = dict(count_mismatch=0, sketch_maxrel=0.0, rows_out_of_range=0)
    eidx = {e: i for i, e in enumerate(ex)}

    def cb(sid, e, Xg, Xd):
        conds = [c for c, ee in member.get(sid, []) if ee == e]
        if not conds:
            return
        nr = Xg.shape[0]
        if nr != int(nint[e][sid]):
            chk["count_mismatch"] += 1
        if nr == 0:
            return
        for j, (q, X) in enumerate((("gu", Xg), ("down", Xd))):
            Xf = X.float()
            full = Xf.T @ Xf
            zz = EW.svech(EW.sketch_outer(X, Rp[q]), iu, w).cpu()
            ref = Msk[(e, q)][sid]
            chk["sketch_maxrel"] = max(chk["sketch_maxrel"], float(torch.linalg.norm(zz - ref) /
                                                                   (torch.linalg.norm(ref) + 1e-30)))
            for c in conds:
                Hu[(c, e, q)] += full
                Zu[(c, e, q)] += zz
                if q == "gu":
                    cu[(c, e)] += nr
                rows = plan["plan"][c][e].get(sid)
                if rows is None:
                    continue
                if int(rows.max()) >= nr:
                    chk["rows_out_of_range"] += 1
                Xi = Xf[rows.to(Xf.device)]
                Hm[(c, e, q)] += Xi.T @ Xi
                Zm[(c, e, q)] += EW.svech(EW.sketch_outer(X[rows.to(X.device)], Rp[q]), iu, w).cpu()
                if q == "gu":
                    cm[(c, e)] += len(rows)
    cache = load_layer_cache(CACHE, L)
    torch.cuda.reset_peak_memory_stats()
    with MU.layer_on_gpu(model, L):
        sync(); t0 = time.time()
        info = EW.replay(model, L, cache, union, ex, cb)
        sync(); t_rep = time.time() - t0
    gpu_capture = gpu_peak_gb()
    # ---- held-out activations (identical for every condition)
    t0 = time.time()
    HO = {}
    for split, sp in EVAL.items():
        ids = list(range(*sp["rows"]))
        h = Profiler(model, L, collect_hessian=False, collect_holdout=True,
                     holdout_tokens_per_seq=SETTINGS["holdout_tokens_per_seq"]).run(load_layer_cache(sp["cache"], L), ids)
        HO[split] = {kk: v for kk, v in h.holdout.items() if kk[1] in ex}
        assert all(HO[split].get((L, e, "gu")) for e in ex), f"{split}: expert without held-out activations"
    t_ho = time.time() - t0
    ho_fp = {sp: hashlib.sha256(b"".join(x.numpy().tobytes() for kk in sorted(HO[sp]) for _, x in HO[sp][kk]))
             .hexdigest() for sp in HO}
    # ---- mismatches
    Nref = {e: float(d["Nfull"][str(e)]) for e in ex}
    skref = {e: {q: Msk[(e, q)].sum(0) / Nref[e] for q in EW.PROJS} for e in ex}
    mis = {}
    for c in EW.CONDITIONS:
        for e in ex:
            mis[f"{c}.{e}"] = dict(
                sketch_pre=EW.mismatch_from_sums({q: Zu[(c, e, q)] for q in EW.PROJS}, cu[(c, e)], skref[e]),
                sketch_post=EW.mismatch_from_sums({q: Zm[(c, e, q)] for q in EW.PROJS}, cm[(c, e)], skref[e]),
                full_pre=float(np.mean([EW.rel_frob(Hu[(c, e, q)].cpu(), cu[(c, e)], d["Hfull"][f"{e}.{q}"], Nref[e])
                                        for q in EW.PROJS])),
                full_post=float(np.mean([EW.rel_frob(Hm[(c, e, q)].cpu(), cm[(c, e)], d["Hfull"][f"{e}.{q}"],
                                                     Nref[e]) for q in EW.PROJS])),
                selection_time=selj["sketch_mismatch_pre_matching"][c][str(e)])
    # ---- GPTQ + evaluation
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res, perseq = {}, {}
    for variant, H, cnt in (("matched", Hm, cm), ("unmatched", Hu, cu)):
        for c in EW.CONDITIONS:
            for e in ex:
                W = fp_weights(model, L, e)
                Wq = quantize_expert(W, H[(c, e, "gu")], H[(c, e, "down")], G["bits"], G["group_size"], G["percdamp"])
                r = dict(tokens=cnt[(c, e)])
                for split in EVAL:
                    errs = per_seq_rel_errors(model, L, e, Wq, HO[split])
                    v = np.array([errs[q] for q in sorted(errs)])
                    perseq[f"{variant}.{split}.{c}.{e}"] = v
                    r[split] = dict(mean=float(v.mean()), median=float(np.median(v)), p90=float(np.quantile(v, .9)),
                                    n_seqs=int(v.size))
                res[f"{variant}.{c}.{e}"] = r
    t_q = time.time() - t0
    gpu_q = gpu_peak_gb()
    # ---- diversity / checks
    div = {f"{c}.{e}": dict(matched=EW.source_diversity(plan["plan"][c][e]),
                            unmatched=EW.source_diversity({q: int(nint[e][q]) for q in sel[c][e]}),
                            discarded_frac=1 - plan["quota"][e] / max(1, plan["counts"][c][e]))
           for c in EW.CONDITIONS for e in ex}
    checks = dict(
        exactly_8_experts=len(ex) == 8 and ex == proto["experts"],
        sets_of_128_distinct=True,                           # validate_selection raised otherwise
        matched_counts_identical_across_conditions=all(len({cm[(c, e)] for c in EW.CONDITIONS}) == 1 and
                                                       cm[("SR", e)] == plan["quota"][e] for e in ex),
        unmatched_counts_equal_selection=all(cu[(c, e)] == plan["counts"][c][e] for c in EW.CONDITIONS for e in ex),
        natural_routing_and_gu_down_alignment=info["n_alignment_checks"] == len(union) * len(ex),
        n_alignment_checks=info["n_alignment_checks"],
        replay_counts_equal_sketch=chk["count_mismatch"] == 0,
        rows_in_range=chk["rows_out_of_range"] == 0,
        per_seq_sketch_max_rel_diff=chk["sketch_maxrel"],
        per_seq_sketch_reproduced=chk["sketch_maxrel"] < 1e-9,
        pre_matching_sketch_equals_selection_time=max(abs(m["sketch_pre"] - m["selection_time"])
                                                     for m in mis.values()) < 1e-8,
        matching_plan_deterministic=bool(plan_det),
        selection_deterministic=selj["deterministic_rerun_identical"], objective_monotone=selj["objective_monotone"],
        selection_uses_pool_only=max(union) < 2000 and max(union) < TUNE_ROWS[0],
        eval_rows_disjoint_from_pool=EVAL["test"]["rows"][0] >= 2000,
        weights_unchanged=expert_digest(model, ex) == dig0,
        same_eval_activations_all_conditions=True,           # one HO dict per split, reused for every condition
        frozen_code_unchanged=hashes(FROZEN_CODE) == proto["frozen_code_sha256"])
    checks["passed"] = all(v for kk, v in checks.items() if isinstance(v, bool))
    os.makedirs(p("results"), exist_ok=True)
    np.savez_compressed(p("results", f"{tag}_seed{a.seed}_perseq.npz"), **perseq)
    jdump(dict(seed=a.seed, tag=tag, experts=ex, results=res, mismatch=mis, diversity=div,
               quota={str(e): plan["quota"][e] for e in ex},
               routed_tokens={c: {str(e): plan["counts"][c][e] for e in ex} for c in EW.CONDITIONS},
               union_replayed=len(union), heldout_fingerprint=ho_fp, checks=checks,
               cost=dict(model_load_s=t_load, matching_plan_s=t_plan, capture_replay_s=t_rep,
                         capture_s_per_seq=t_rep / len(union), heldout_capture_s=t_ho, gptq_eval_s=t_q,
                         peak_gpu_capture_gb=gpu_capture, peak_gpu_gptq_gb=gpu_q, peak_rss_gb=rss_gb()),
               created=time.strftime("%Y-%m-%d %H:%M:%S")), out)
    print(f"[run] {tag} seed {a.seed}: checks passed={checks['passed']} union={len(union)} replay {t_rep:.0f}s")
    for c in EW.CONDITIONS:
        print(f"   {c}: matched test {np.mean([res[f'matched.{c}.{e}']['test']['mean'] for e in ex]):.6f}  "
              f"unmatched {np.mean([res[f'unmatched.{c}.{e}']['test']['mean'] for e in ex]):.6f}  "
              f"Dsk post {np.mean([mis[f'{c}.{e}']['sketch_post'] for e in ex]):.4f}")
    if not checks["passed"]:
        print("[run] FAILED checks:", {kk: v for kk, v in checks.items() if v is False})
        sys.exit(1)


# ===================================================================== freeze
def stage_freeze(a):
    if os.path.exists(p("freeze.json")):
        print("[freeze] already frozen"); return
    sm = jload("results", "smoke_seed0.json")
    assert sm["checks"]["passed"], "smoke run did not pass its checks"
    assert jload("logs", "sketch.json")["passed"]
    jdump(dict(experiment_code_sha256=hashes(EXP_CODE), frozen_code_sha256=hashes(FROZEN_CODE),
               settings=SETTINGS, smoke_checks=sm["checks"],
               note="settings frozen after a passing seed-0 smoke run; final runs must reproduce smoke seed 0 exactly",
               created=time.strftime("%Y-%m-%d %H:%M:%S")), p("freeze.json"))
    print("[freeze] ok")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["protocol", "sketch", "select", "run", "freeze", "report"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="final", choices=["smoke", "final"])
    a = ap.parse_args(argv)
    if a.stage == "report":
        sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
        import report_expertwise_calibration_mvp as R
        R.main()
        return
    dict(protocol=stage_protocol, sketch=stage_sketch, select=stage_select, run=stage_run,
         freeze=stage_freeze)[a.stage](a)


if __name__ == "__main__":
    main()
