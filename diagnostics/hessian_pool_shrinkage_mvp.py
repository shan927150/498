"""Layer-pooled Hessian shrinkage for low-budget MoE GPTQ: feasibility MVP.

FR s0 (C4 Random, frozen 128 x 512 manifest order, CANONICAL-B1 cache), layers 8 / 16, chronological prefixes
S16 c S32 c S64 c S128, 12 experts selected by B32 routed-token exposure only. Gate/up Hessians are shrunk toward a
trace-matched layer-pooled MoE-input second moment (or a trace-matched identity control); down Hessians are unchanged.
The GPTQ solver (4-bit, g128, act-order, percdamp 1.0) is untouched.

Stages (--stage): protocol | hessians | reference | mismatch | quantize | evaluate | subset | timing | finalize | tests
"""
from __future__ import annotations

import argparse, glob, hashlib, json, os, resource, subprocess, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import agq_only_validation as AV          # noqa: E402  (frozen FR s0 manifest, caches)
import expertwise_agq_hybrid as X         # noqa: E402  (splits, partial-model loader for layers 8/16)
from moe_calib import early_stop as ES    # noqa: E402
from moe_calib import hessian_shrinkage as HS  # noqa: E402

OUT = "runs_hessian_pool_shrinkage_mvp"
STORE = os.path.expanduser("~/moe_cache_hessian_pool_shrinkage")
CONFIG = "FR_s0"
LAYERS = [8, 16]
ALL_LAYERS = AV.LAYERS                    # [1, 8, 16, 23]: column layout of the routing-count matrix
E = 60
K = 4                                     # production K for FR s0 -> block 0 = S32, blocks 0-1 = S64
SPLITS = X.SPLITS
TUNED_DAMP = {16: 3.0, 32: 2.0}           # runs_diag/damping_threshold_report.md, best percdamp per prefix
GPTQ = dict(bits=4, group_size=128, percdamp=1.0, act_order=True)
HIST_INPUTS = ["runs_hessian_early_stop_mvp/logs/checkpoints_L8.json", "runs_hessian_early_stop_mvp/logs/checkpoints_L16.json",
               "runs_hessian_early_stop_mvp/checkpoint_manifest.json", "runs_agq_only_validation/calibration_manifest.json",
               "runs_ebss_official_validation/logs/capture_routing_nll.npz", "runs_diag/damping_threshold_report.md",
               "runs_diag/adaptive_damping.md", "runs_repr_constrained/covariance/cov_summary.md",
               "runs_ebss_official_validation/results/FR_s0.json"]


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def s(*a):
    return os.path.join(STORE, *a)


jdump, sha256_file = AV.jdump, AV.sha256_file


def jload(*a):
    return json.load(open(p(*a)))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def gpu_peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def code_hashes():
    return {f: sha256_file(os.path.join(ROOT, f)) for f in
            ("moe_calib/hessian_shrinkage.py", "diagnostics/hessian_pool_shrinkage_mvp.py", "moe_calib/gptq.py",
             "moe_calib/profiler.py", "moe_calib/hessian.py", "moe_calib/quant_eval.py")}


def manifest_rows():
    return json.load(open(os.path.join(ROOT, "runs_hessian_early_stop_mvp", "checkpoint_manifest.json")))["cache_rows"]


def selected():
    return jload("selected_experts.json")


def sel_list():
    """[(L, e, stratum)] in a fixed order (layer, stratum low/median/high, rank)."""
    sx = selected()["experts"]
    return [(L, e, st) for L in LAYERS for st in ("low", "median", "high") for e in sx[str(L)][st]]


# ===================================================================== protocol (audit + freeze)
def stage_protocol(a):
    os.makedirs(p("logs"), exist_ok=True)
    if os.path.exists(p("protocol.json")):
        print("[protocol] already frozen; not rewritten"); return
    rows = manifest_rows()
    cm = json.load(open(os.path.join(ROOT, "runs_agq_only_validation", "calibration_manifest.json")))["configs"][CONFIG]
    es_man = json.load(open(os.path.join(ROOT, "runs_hessian_early_stop_mvp", "checkpoint_manifest.json")))
    pre = ES.prefixes(rows)
    z = np.load(os.path.join(ROOT, "runs_ebss_official_validation", "logs", "capture_routing_nll.npz"))["C"]
    counts = {B: {L: z[rows[:B], ALL_LAYERS.index(L) * E:(ALL_LAYERS.index(L) + 1) * E].sum(0) for L in LAYERS}
              for B in HS.BUDGETS}
    sel = {str(L): HS.select_experts(counts[32][L]) for L in LAYERS}
    jdump(dict(rule="per layer: ascending sort of B32 routed-token count (ties -> lower expert id); low = ranks "
                    "{0,1}, median = ranks {29,30}, high = ranks {58,59}. Exposure only: no covariance, AGQ or "
                    "reconstruction information was inspected.",
               source="runs_ebss_official_validation/logs/capture_routing_nll.npz (C[row, layer*60+e]) over the first "
                      "32 rows of the frozen FR s0 manifest",
               experts=sel,
               exposure={str(B): {str(L): {str(e): int(counts[B][L][e]) for st in sel[str(L)].values() for e in st}
                                  for L in LAYERS} for B in HS.BUDGETS},
               layer_exposure_b32={str(L): dict(min=int(counts[32][L].min()), median=float(np.median(counts[32][L])),
                                                max=int(counts[32][L].max())) for L in LAYERS},
               created=time.strftime("%Y-%m-%d %H:%M:%S")), p("selected_experts.json"))
    jdump(dict(config=CONFIG, cache=AV.ECACHE, cache_rows=rows, original_chosen=es_man["original_chosen"],
               ids_hash=es_man["ids_hash"], equals_agq_only_manifest=rows == cm["cache_rows"],
               prefixes={str(B): v for B, v in pre.items()},
               accumulation="Profiler.run default contiguous blocks (i*4)//128 -> block 0 = S32, 0-1 = S64, all = "
                            "S128; S16 = mid-block snapshot (identical to runs_hessian_early_stop_mvp)",
               pooled_tokens_expected={str(B): 512 * B for B in HS.BUDGETS}), p("calibration_manifest.json"))
    # ---- inventory
    hist_w = AV.weights_dir("GPTQ", CONFIG)
    inv = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
               expert_hessian_snapshots=dict(
                   stored=False, note="runs_hessian_early_stop_mvp kept only derived statistics (trace / Frobenius drift, "
                   "token counts) and B32/B64/B128 E3 errors, not the matrices -> Hessians are re-accumulated from the "
                   "same frozen rows with the same Profiler path and checked against those statistics bitwise"),
               chronological_checkpoints="runs_hessian_early_stop_mvp/checkpoint_manifest.json (S16/S32/S64/S128)",
               calibration_cache=dict(path=AV.ECACHE, files={str(L): os.path.exists(os.path.join(AV.ECACHE, f"L{L}.pt"))
                                                            for L in LAYERS}),
               gptq_baseline_weights=dict(path=hist_w, files={str(L): sha256_file(os.path.join(hist_w, f"L{L}.pt"))
                                                              for L in LAYERS}),
               heldout_caches={k: dict(v, files={str(L): os.path.exists(os.path.join(v["cache"], f"L{L}.pt"))
                                                 for L in LAYERS}) for k, v in SPLITS.items()},
               evaluators=dict(
                   complete_expert="new (this driver): c^2-weighted complete expert MLP output error with frozen BF16 "
                                   "routing, same ops as moe_calib.agq.expert_fp32 / routing_coefficients / dispatch_rows",
                   e3="moe_calib.quant_eval.per_seq_rel_errors (historical E3, oldtest regression)",
                   covariance="diagnostics/covariance_mismatch.py conventions (rel. Frobenius, rel. diag, top-16 "
                              "alignment) re-implemented in moe_calib/hessian_shrinkage.py"),
               historical_inputs_sha256={f: sha256_file(os.path.join(ROOT, f)) for f in HIST_INPUTS},
               calibration_original_ids_max=int(max(es_man["original_chosen"])),
               calibration_disjoint_from_tune=bool(max(es_man["original_chosen"]) < SPLITS["tune"]["rows"][0]),
               manifest_compatible=rows == cm["cache_rows"])
    jdump(inv, p("artifact_inventory.json"))
    proto = dict(
        experiment="Layer-pooled Hessian shrinkage for low-budget MoE GPTQ (feasibility MVP)",
        model="Qwen/Qwen1.5-MoE-A2.7B (partial load: layers 8, 16; BF16)",
        calibration=dict(config=CONFIG, cache=AV.ECACHE, ids_hash=es_man["ids_hash"], seq_len=512,
                         capture="CANONICAL-B1", prefixes="chronological S16 c S32 c S64 c S128 of the frozen order"),
        layers=LAYERS, layers_note="exploratory, outcome-informed choice from earlier AGQ studies; not sampled",
        budgets=dict(primary=list(HS.PRIMARY_BUDGETS), reference=[64, 128]),
        experts=dict(rule=jload("selected_experts.json")["rule"], ids=sel),
        estimators=dict(pooled="H = n_e[(1-a) Hbar_e + a P_e], P_e = Hbar_L tr(Hbar_e)/tr(Hbar_L) (trace-matched)",
                        identity="H = n_e[(1-a) Hbar_e + a tr(Hbar_e)/d I] (trace-matched control)",
                        untraced="H = n_e[(1-a) Hbar_e + a Hbar_L] (covariance diagnostic only, never quantized)",
                        pure_prior="alpha = 1 pooled prior (covariance diagnostic only, never quantized)",
                        alpha0="raw H_e passed unchanged (no normalization round trip)",
                        pooled_hessian="H_L = sum over every MoE-input token of the prefix (forward_pre_hook on the "
                                       "router gate = gate/up input), each token once",
                        down="down_proj always uses its raw routed Hessian at the same budget"),
        alpha_grid=list(HS.ALPHAS), alpha_grid_frozen=True,
        gptq=GPTQ, tuned_damping=dict(values={str(k): v for k, v in TUNED_DAMP.items()},
                                      source="runs_diag/damping_threshold_report.md (best percdamp per prefix, tuning "
                                             "split, seeds 0/1); applied to all three projections; not re-searched"),
        splits=SPLITS,
        stage1=dict(reference="tune split ~/moe_cache rows 2000-2127: routed gu second moment Hbar_ref = H_ref/n_ref "
                              "(finite-sample proxy, not population truth); never used to build the prior",
                    metrics=dict(primary="D_F = ||H - Hbar_ref||_F / (||Hbar_ref||_F + eps), H token-normalized",
                                 secondary=["D_diag relative diagonal mismatch", "top-16 eigenspace alignment"]),
                    budgets_screened=list(HS.PRIMARY_BUDGETS),
                    rule=dict(rules=HS.RULES, c1="mean D_F(pooled, a) <= (1 - 0.03) mean D_F(raw)",
                              c2=">= 8 / 12 experts with D_F(pooled, a) < D_F(raw)",
                              c3="mean D_F(pooled, a) < mean D_F(identity, a)",
                              c4="no layer whose mean D_F(pooled, a) exceeds its mean D_F(raw) by > 1 %",
                              c5="all numerical checks pass",
                              selection="per budget, among passing nonzero alphas, the lowest mean tuning D_F; "
                                        "a budget with no passing alpha is NOT carried into Stage 2",
                              fail="no passing alpha at B16 and B32 -> POOL-PRIOR-NO-GAIN, stop before Stage 2")),
        stage2=dict(
            methods=dict(A="standard GPTQ, raw H, percdamp 1.0", B="pooled shrinkage at alpha*(B), percdamp 1.0",
                         C="identity shrinkage at alpha*(B), percdamp 1.0",
                         D="raw H, tuned percdamp (B16 3.0, B32 2.0) on all projections",
                         E="standard GPTQ at B64 / B128 (quality references, different calibration cost)"),
            primary_metric="J_e = sum_i c_ie^2 ||E_fp(x_i) - E_q(x_i)||^2 / (sum_i c_ie^2 ||E_fp(x_i)||^2 + eps) on the "
                           "confirmation split (~/moe_cache_confirmation rows 0-255), complete expert MLP, frozen BF16 "
                           "routing, fp32 math, BF16-rounded quantized weights (as installed); summary = mean over the 12 "
                           "experts",
            secondary=["unweighted complete-expert output error", "gate / up projection output error",
                       "E3-form projection reconstruction (gate+up+down, ratio of token sums) on confirm",
                       "fraction improved, p90 / worst expert change", "Spearman(delta mismatch, delta J)"],
            criteria=dict(c1=">= 1 % lower mean J than standard GPTQ at B16 or B32",
                          c2="mean J_pool < mean J_identity (same budget)",
                          c3="mean J_pool <= 1.005 x mean J_tuned (same budget)",
                          c4=">= 8 / 12 experts improved vs standard", c5="no expert worse than +5 % vs standard",
                          c6="layer-mean J improves on both L8 and L16",
                          c7="optional six-expert subset replacement does not contradict (pooled <= standard)",
                          c8="correctness checks pass",
                          c9="measured pooled-accumulation time <= 5 % of (replay + 60-expert GPTQ) per layer",
                          note="c1-c6 must hold at the same budget"),
            verdict_precedence="BLOCKED > POOL-PRIOR-NO-GAIN (Stage 1) > PROMISING > REGULARIZATION-ONLY > "
                               "MISMATCH-ONLY > NO-GAIN (moe_calib.hessian_shrinkage.classify)",
            recommendation=dict(PROMISING="A", **{"REGULARIZATION-ONLY": "B", "MISMATCH-ONLY": "C", "NO-GAIN": "B",
                                                   "BLOCKED": "B", "POOL-PRIOR-NO-GAIN": "B"}),
            gap_recovery="R = (E_B^std - E_B^pool)/(E_B^std - E_2B^std), reported only when the gap > 1 % of E_B^std",
            subset="optional: only if Stage 2 improves J; L8/L16 all 60 experts standard GPTQ at the budget vs the same "
                   "with the 6 selected experts' gate/up replaced by pooled weights; exact MoE output on confirm"),
        hard_constraints=["no new calibration data", "GPTQ solver unchanged", "no extra alphas / per-expert alpha",
                          "no damping search", "no AGQ / c^2 Hessians", "historical artifacts untouched",
                          "recommendation A never executed automatically"],
        precision="BF16 model forward; Hessian accumulation fp32; mixing fp64 cast to fp32; evaluation fp32",
        code_sha256=code_hashes(), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(proto, p("protocol.json"))
    print("[protocol] frozen; experts", sel)


# ===================================================================== hessians (Stage 1A)
@torch.no_grad()
def replay_layer(model, L, cache, rows, pr, pooled, digest=True):
    """Profiler.run-equivalent loop over the frozen order (contiguous K-blocks) with an S16 snapshot; also hashes
    layer outputs and records pooled-input == router-input consistency."""
    from moe_calib import model_utils as MU
    n = len(rows)
    blocks = ES.contiguous_blocks(n, K)
    h = hashlib.sha256()
    snap16, consist = None, True
    for i, sid in enumerate(rows):
        pr.cur_seq, pr.cur_block, pr.cur_weight = sid, blocks[i], 1.0
        if pooled is not None:
            pooled.cur_block = blocks[i]
        y = MU.run_layer(model, L, cache[sid:sid + 1].to(MU.GPU, non_blocking=True))
        if digest:
            h.update(y.contiguous().view(torch.int16).cpu().numpy().tobytes())
        if pooled is not None and pooled.keep_input and pr is not None and pr._gate_in is not None:
            consist &= bool(torch.equal(pooled.last_input, pr._gate_in))
        if i == 15:
            snap16 = dict(experts={(e, pj): pr.acc.H[(L, e, pj)][0].cpu().clone() for e in range(E)
                                   for pj in ("gu", "down")} if pr.acc is not None else None,
                          n={e: float(pr.acc.raw_n[(L, e, "gu")][0]) for e in range(E)} if pr.acc is not None else None,
                          pooled=pooled.H[0].cpu().clone() if pooled is not None else None,
                          pooled_n=float(pooled.n[0]) if pooled is not None else None)
    sync()
    return h.hexdigest(), snap16, consist


@torch.no_grad()
def stage_hessians(a):
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    from moe_calib.quant_eval import fp_weights, per_seq_rel_errors
    from moe_calib.gptq import quantize_expert
    os.makedirs(s("hessians"), exist_ok=True)
    rows = manifest_rows()
    sx = selected()["experts"]
    z = np.load(os.path.join(ROOT, "runs_ebss_official_validation", "logs", "capture_routing_nll.npz"))["C"]
    test_ids = list(range(*SPLITS["oldtest"]["rows"]))
    model, t_load = X.load_partial()
    torch.cuda.reset_peak_memory_stats()
    for L in LAYERS:
        outp = p("logs", f"hessians_L{L}.json")
        if os.path.exists(outp):
            print(f"[hessians] L{L} done"); continue
        sel = [e for st in ("low", "median", "high") for e in sx[str(L)][st]]
        li = ALL_LAYERS.index(L)
        cache = load_layer_cache(AV.ECACHE, L)
        dig0 = PQ.layer_digest(model, L)
        blk = MU.moe_block(model, L)
        with MU.layer_on_gpu(model, L):
            # (i) plain forward digest (no hooks)
            h_plain = hashlib.sha256()
            for sid in rows:
                y = MU.run_layer(model, L, cache[sid:sid + 1].to(MU.GPU))
                h_plain.update(y.contiguous().view(torch.int16).cpu().numpy().tobytes())
            # (ii) production Profiler (verify_alignment) + opt-in pooled hook
            pr = Profiler(model, L, K=K, verify_alignment=True)
            pooled = HS.PooledHessian(model.config.hidden_size, K, MU.GPU, keep_input=True)
            pooled.attach(blk)
            pr.attach()
            logits_h = hashlib.sha256()
            lh = blk.gate.register_forward_hook(
                lambda m_, i_, o_: logits_h.update(o_.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()))
            t0 = time.time()
            try:
                dig_hooked, snap16, consist = replay_layer(model, L, cache, rows, pr, pooled)
            finally:
                pr.detach(); pooled.detach(); lh.remove()
            t_rep = time.time() - t0
        # ---- snapshots
        H = {16: {k: v for k, v in snap16["experts"].items()}}
        n = {16: snap16["n"]}
        HL = {16: snap16["pooled"]}
        NL = {16: snap16["pooled_n"]}
        for B, bl in ((32, [0]), (64, [0, 1]), (128, [0, 1, 2, 3])):
            H[B] = {(e, pj): pr.acc.H_sum((L, e, pj), bl).cpu() for e in range(E) for pj in ("gu", "down")}
            n[B] = {e: pr.acc.tokens((L, e, "gu"), bl) for e in range(E)}
            HL[B] = pooled.H_sum(bl).cpu()
            NL[B] = pooled.tokens(bl)
        tailL = pooled.H_sum([2, 3]).cpu()
        val = dict(layer=L, n_alignment_checks=pr.n_alignment_checks, pooled_input_equals_router_input=bool(consist),
                   outputs_identical_with_hooks=dig_hooked == h_plain.hexdigest(), output_digest=dig_hooked,
                   router_logits_digest=logits_h.hexdigest(), layer_weights_unchanged=PQ.layer_digest(model, L) == dig0,
                   pooled_tokens={str(B): NL[B] for B in HS.BUDGETS},
                   pooled_tokens_ok=all(NL[B] == 512 * B for B in HS.BUDGETS),
                   pooled_split_rel=float(torch.linalg.norm(HL[64] + tailL - HL[128]) / torch.linalg.norm(HL[128])),
                   routed_sum_equals_topk_times_tokens=all(abs(sum(n[B].values()) - 4 * NL[B]) < 0.5
                                                           for B in HS.BUDGETS),
                   expert_counts_match_routing_profile=all(int(n[B][e]) == int(z[rows[:B], li * E + e].sum())
                                                           for B in HS.BUDGETS for e in range(E)),
                   d_pooled=list(HL[128].shape), pooled={})
        for B in HS.BUDGETS:
            M = HL[B].double()
            w = torch.linalg.eigvalsh(M.to(MU.GPU)).cpu()
            val["pooled"][str(B)] = dict(finite=bool(torch.isfinite(M).all()),
                                         sym_rel=float(torch.linalg.norm(M - M.T) / torch.linalg.norm(M)),
                                         min_eig_rel=float(w.min() / w.max()), psd_ok=bool(w.min() >= -1e-6 * w.max()),
                                         trace_per_token=float(torch.trace(M)) / NL[B])
        # expert split check + drift statistics vs early-stop (bitwise)
        es = json.load(open(os.path.join(ROOT, "runs_hessian_early_stop_mvp", "logs", f"checkpoints_L{L}.json")))
        val["expert_split_rel_max"] = max(float(torch.linalg.norm(H[64][(e, pj)] + pr.acc.H_sum((L, e, pj), [2, 3]).cpu()
                                                                  - H[128][(e, pj)]) / torch.linalg.norm(H[128][(e, pj)]))
                                          for e in sel for pj in ("gu", "down"))
        drift_eq = True
        for B in (32, 64, 128):
            for e in sel:
                for pj in ("gu", "down"):
                    drift_eq &= ES.trace_drift(H[B][(e, pj)], n[B][e], H[B // 2][(e, pj)], n[B // 2][e]) == \
                        es["stats"][str(B)][str(e)][pj]["trace"]
                    drift_eq &= ES.frob_drift(H[B][(e, pj)], n[B][e], H[B // 2][(e, pj)], n[B // 2][e]) == \
                        es["stats"][str(B)][str(e)][pj]["frob"]
                drift_eq &= int(n[B][e]) == int(es["tokens"][str(B)][str(e)])
        val["drift_stats_equal_early_stop_bitwise"] = bool(drift_eq)
        pr.free(); del pr, pooled
        # ---- standard GPTQ regression: B128 weights == historical; B32/64/128 E3 (oldtest) == early-stop logs
        hist = torch.load(os.path.join(AV.weights_dir("GPTQ", CONFIG), f"L{L}.pt"))
        HO = Profiler(model, L, collect_hessian=False, collect_holdout=True).run(load_layer_cache(AV.OCACHE, L),
                                                                               test_ids).holdout
        w_eq, e3_eq, e3 = True, True, {}
        for B in (32, 64, 128):
            for e in sel:
                Wq = quantize_expert(fp_weights(model, L, e), H[B][(e, "gu")], H[B][(e, "down")], GPTQ["bits"],
                                     GPTQ["group_size"], GPTQ["percdamp"])
                v = float(np.mean(list(per_seq_rel_errors(model, L, e, Wq, HO).values())))
                e3[f"{B}.{e}"] = v
                e3_eq &= v == es["err"][str(B)][str(e)]
                if B == 128:
                    w_eq &= all(torch.equal(Wq[k].to(torch.bfloat16).cpu(), hist[f"{e}.{k}"]) for k in Wq)
        val.update(B128_weights_equal_historical=bool(w_eq), e3_oldtest_equal_early_stop=bool(e3_eq), e3_oldtest=e3)
        del HO, hist
        # ---- save the selected experts' Hessians + pooled snapshots (outside the repo)
        torch.save(dict(layer=L, experts=sel, H={f"{B}.{e}.{pj}": H[B][(e, pj)] for B in HS.BUDGETS for e in sel
                                                 for pj in ("gu", "down")},
                        n={f"{B}.{e}": n[B][e] for B in HS.BUDGETS for e in sel},
                        HL={str(B): HL[B] for B in HS.BUDGETS}, NL={str(B): NL[B] for B in HS.BUDGETS}),
                   s("hessians", f"L{L}.pt"))
        val.update(replay_s=t_rep, peak_gpu_gb=gpu_peak_gb(), peak_rss_gb=rss_gb(),
                   all_tokens={str(B): {str(e): n[B][e] for e in range(E)} for B in HS.BUDGETS},
                   store_sha256=sha256_file(s("hessians", f"L{L}.pt")))
        jdump(val, outp)
        print(f"[hessians] L{L}: align {val['n_alignment_checks']} pooled==router {consist} outputs-identical "
              f"{val['outputs_identical_with_hooks']} pooled-tokens {val['pooled_tokens_ok']} split "
              f"{val['pooled_split_rel']:.1e} counts {val['expert_counts_match_routing_profile']} drift== "
              f"{drift_eq} B128 weights== {w_eq} E3== {e3_eq}", flush=True)
        del H, HL, cache
        torch.cuda.empty_cache()
    vals = {str(L): jload("logs", f"hessians_L{L}.json") for L in LAYERS}
    keys = ["pooled_input_equals_router_input", "outputs_identical_with_hooks", "layer_weights_unchanged",
            "pooled_tokens_ok", "routed_sum_equals_topk_times_tokens", "expert_counts_match_routing_profile",
            "drift_stats_equal_early_stop_bitwise", "B128_weights_equal_historical", "e3_oldtest_equal_early_stop"]
    ok = all(v[k] for v in vals.values() for k in keys) and all(
        v["pooled_split_rel"] < 1e-5 and v["expert_split_rel_max"] < 1e-5 and v["n_alignment_checks"] > 0 and
        all(q["finite"] and q["psd_ok"] and q["sym_rel"] == 0.0 for q in v["pooled"].values()) for v in vals.values())
    jdump(dict(per_layer={L: {k: v for k, v in d.items() if k not in ("all_tokens", "e3_oldtest")}
                          for L, d in vals.items()},
               passed=bool(ok), checks=keys + ["pooled_split_rel < 1e-5", "expert_split_rel_max < 1e-5",
                                                "alignment checks > 0", "pooled finite / symmetric / PSD"]),
          p("pooled_hessian_validation.json"))
    print("[hessians] validation passed:", ok)


# ===================================================================== reference (tune split)
@torch.no_grad()
def stage_reference(a):
    from moe_calib.data_utils import load_layer_cache
    from moe_calib import model_utils as MU
    from moe_calib.profiler import Profiler
    if os.path.exists(s("reference.pt")):
        print("[reference] done"); return
    sx = selected()["experts"]
    ids = list(range(*SPLITS["tune"]["rows"]))
    model, _ = X.load_partial()
    out, meta = {}, {}
    for L in LAYERS:
        sel = [e for st in ("low", "median", "high") for e in sx[str(L)][st]]
        cache = load_layer_cache(SPLITS["tune"]["cache"], L)
        pr = Profiler(model, L, K=1, verify_alignment=True)
        pooled = HS.PooledHessian(model.config.hidden_size, 1, MU.GPU)
        pooled.attach(MU.moe_block(model, L))
        t0 = time.time()
        try:
            pr.run(cache, ids)
        finally:
            pooled.detach()
        for e in sel:
            nr = pr.acc.tokens((L, e, "gu"))
            out[f"{L}.{e}"] = (pr.acc.H_sum((L, e, "gu"), [0]).cpu(), nr)
        out[f"{L}.pooled"] = (pooled.H_sum([0]).cpu(), pooled.tokens([0]))
        meta[str(L)] = dict(tokens={str(e): out[f"{L}.{e}"][1] for e in sel}, pooled_tokens=out[f"{L}.pooled"][1],
                            alignment_checks=pr.n_alignment_checks, seconds=time.time() - t0)
        pr.free(); del pr
    torch.save(out, s("reference.pt"))
    jdump(dict(split="tune", cache=SPLITS["tune"]["cache"], rows=SPLITS["tune"]["rows"], per_layer=meta,
               role="independent finite-sample proxy (not population truth); disjoint from calibration rows",
               calibration_disjoint=jload("artifact_inventory.json")["calibration_disjoint_from_tune"],
               used_for_prior=False), p("logs", "reference_meta.json"))
    print("[reference] done", {L: m["tokens"] for L, m in meta.items()})


# ===================================================================== mismatch (Stage 1B/1C + screen)
@torch.no_grad()
def stage_mismatch(a):
    from moe_calib import model_utils as MU
    if os.path.exists(p("stage1_decision.json")):
        print("[mismatch] stage 1 already decided; not rewritten"); return
    dev = MU.GPU
    ref = torch.load(s("reference.pt"))
    SL = sel_list()
    rec, num = {}, {}
    t0 = time.time()
    for L in LAYERS:
        st = torch.load(s("hessians", f"L{L}.pt"))
        for (LL, e, stratum) in [x for x in SL if x[0] == L]:
            Hr, nr = ref[f"{L}.{e}"]
            R = (Hr.double() / nr).to(dev)
            Uref = HS.top_eigvecs(R)
            for B in HS.BUDGETS:
                He, ne = st["H"][f"{B}.{e}.gu"].to(dev), st["n"][f"{B}.{e}"]
                HL, NL = st["HL"][str(B)].to(dev), st["NL"][str(B)]
                tr0 = float(torch.trace(He.double()))
                r = {}

                def metrics(Hs, tag, check=False):
                    Hn = Hs.double() / ne
                    m = dict(D_F=HS.frob_mismatch(Hn, R), D_diag=HS.diag_mismatch(Hn, R),
                             align16=HS.subspace_alignment(HS.top_eigvecs(Hn), Uref))
                    if check:
                        num[f"{L}.{e}.{B}.{tag}"] = HS.numerics(Hs, tr0)
                    r[tag] = m
                metrics(HS.shrink(He, ne, HL, NL, 0.0), "raw", check=True)
                for al in HS.ALPHAS[1:]:
                    for tg in HS.TARGETS:
                        Hs = HS.shrink(He, ne, HL, NL, al, tg)
                        metrics(Hs, f"{tg}_{al}", check=(tg != "untraced"))
                P = HS.trace_matched_prior(He.double() / ne, HL.double() / NL) * ne
                metrics(P, "pure_prior")
                metrics(HL.double() / NL * ne, "pure_prior_untraced")
                rec[f"{L}.{e}.{B}"] = dict(layer=L, expert=e, stratum=stratum, budget=B, n_e=ne, N=NL, n_ref=nr,
                                           metrics=r)
            print(f"[mismatch] L{L} e{e} ({stratum}) done {time.time() - t0:.0f}s", flush=True)
        del st
    # alpha-zero identity + no mutation
    st = torch.load(s("hessians", "L8.pt"))
    e0 = sel_list()[0][1]
    H0 = st["H"][f"16.{e0}.gu"]
    keep = H0.clone()
    a0 = HS.shrink(H0, st["n"][f"16.{e0}"], st["HL"]["16"], st["NL"]["16"], 0.0)
    for al in HS.ALPHAS[1:]:
        HS.shrink(H0, st["n"][f"16.{e0}"], st["HL"]["16"], st["NL"]["16"], al, "pooled")
    alpha0_ok = bool(torch.equal(a0, H0) and torch.equal(H0, keep))
    num_ok = all(v["finite"] and v["psd_ok"] and v["sym_rel"] < 1e-12 and v["trace_rel_err"] < 1e-6 and
                 v["shape"] == [2048, 2048] and v["dtype"] == "torch.float32" for v in num.values()) and alpha0_ok
    jdump(dict(per_candidate=num, alpha0_returns_raw_bitwise_and_no_mutation=alpha0_ok, passed=bool(num_ok),
               thresholds=dict(sym_rel="< 1e-12", trace_rel_err="< 1e-6", psd="min eig >= -1e-6 max eig")),
          p("logs", "candidate_numerics.json"))
    # covariance reference validation
    rm = jload("logs", "reference_meta.json")
    jdump(dict(reference=rm, split_disjoint=rm["calibration_disjoint"],
               reference_tokens_vs_b128={f"{L}.{e}": dict(n_ref=rec[f"{L}.{e}.128"]["n_ref"],
                                                         n_b128=rec[f"{L}.{e}.128"]["n_e"]) for L, e, _ in SL},
               self_check_D_F_of_reference=0.0,
               note="reference = tune-split routed gu second moment of each selected expert (128 x 512 tokens), a "
                    "finite-sample proxy with sampling noise of its own"), p("covariance_reference_validation.json"))
    jdump(dict(records=rec, n_experts=len(SL), seconds=time.time() - t0), p("covariance_mismatch.json"))
    # ---- screen (B16, B32)
    lay = [L for L, e, _ in SL]
    screens = {}
    for B in HS.PRIMARY_BUDGETS:
        raw = [rec[f"{L}.{e}.{B}"]["metrics"]["raw"]["D_F"] for L, e, _ in SL]
        pool = {al: [rec[f"{L}.{e}.{B}"]["metrics"][f"pooled_{al}"]["D_F"] for L, e, _ in SL] for al in HS.ALPHAS[1:]}
        idn = {al: [rec[f"{L}.{e}.{B}"]["metrics"][f"identity_{al}"]["D_F"] for L, e, _ in SL] for al in HS.ALPHAS[1:]}
        screens[B] = HS.screen_budget(raw, pool, idn, lay)
    label = HS.stage1_label(screens, num_ok)
    jdump(dict(label=label, numerics_ok=bool(num_ok),
               screens={str(B): dict(v, per_alpha={str(k): x for k, x in v["per_alpha"].items()})
                        for B, v in screens.items()},
               eligible_budgets=[B for B in screens if screens[B]["selected_alpha"] is not None],
               decided_before_any_quantization=not os.path.exists(p("quantization_results.json")),
               confirm_results_absent=not glob.glob(p("results", "confirm", "*")),
               created=time.strftime("%Y-%m-%d %H:%M:%S")), p("stage1_decision.json"))
    jdump(dict(rule="per budget, lowest mean tuning D_F among nonzero alphas passing all Stage 1 criteria",
               selected={str(B): screens[B]["selected_alpha"] for B in screens},
               uses_only="calibration prefixes + tune-split reference; no confirmation / test data",
               frozen_at=time.strftime("%Y-%m-%d %H:%M:%S")), p("alpha_selection.json"))
    print(f"[mismatch] Stage 1: {label}; selected", {B: screens[B]["selected_alpha"] for B in screens})
    for B in screens:
        for al, v in screens[B]["per_alpha"].items():
            print(f"   B{B} a={al}: rel {v['rel_improvement']:+.4f} n_imp {v['n_improved']} vs-id "
                  f"{v['pooled_vs_identity_rel']:+.4f} layers {v['layer_rel_change']} pass {v['passes']}")


# ===================================================================== quantize (Stage 2)
def candidates():
    """{name: (B, gu_target, alpha, percdamp)} from the frozen Stage 1 outcome."""
    s1 = jload("stage1_decision.json")
    asel = jload("alpha_selection.json")["selected"]
    out = {f"B{B}_std": (B, "raw", 0.0, GPTQ["percdamp"]) for B in HS.BUDGETS}
    for B in s1["eligible_budgets"]:
        al = asel[str(B)]
        out[f"B{B}_pool"] = (B, "pooled", al, GPTQ["percdamp"])
        out[f"B{B}_identity"] = (B, "identity", al, GPTQ["percdamp"])
        out[f"B{B}_tuned"] = (B, "raw", 0.0, TUNED_DAMP[B])
        out[f"B{B}_alpha0"] = (B, "pooled", 0.0, GPTQ["percdamp"])      # regression: alpha 0 path == std bitwise
    return out


@torch.no_grad()
def stage_quantize(a):
    from moe_calib.quant_eval import fp_weights
    from moe_calib.gptq import gptq_quantize
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    s1 = jload("stage1_decision.json")
    if s1["label"] != "POOL-PRIOR-PROMISING":
        print("[quantize] Stage 1 did not pass; Stage 2 not run"); return
    if os.path.exists(p("quantization_results.json")):
        print("[quantize] done"); return
    os.makedirs(s("weights"), exist_ok=True)
    cands = candidates()
    model, _ = X.load_partial()
    torch.cuda.reset_peak_memory_stats()
    res = dict(candidates={k: dict(budget=v[0], gu_hessian=v[1], alpha=v[2], percdamp=v[3]) for k, v in cands.items()},
               timing={}, checks={})
    for L in LAYERS:
        dig0 = PQ.layer_digest(model, L)
        st = torch.load(s("hessians", f"L{L}.pt"))
        sel = st["experts"]
        W = {}
        for cn, (B, tg, al, pd) in cands.items():
            t_mix = t_solve = 0.0
            W[cn] = {}
            for e in sel:
                He, ne = st["H"][f"{B}.{e}.gu"], st["n"][f"{B}.{e}"]
                assert ne > 0, "RTN fallback would be needed"
                sync(); t0 = time.time()
                Hg = He if tg == "raw" else HS.shrink(He, ne, st["HL"][str(B)], st["NL"][str(B)], al, tg)
                sync(); t_mix += time.time() - t0
                Hd = st["H"][f"{B}.{e}.down"]
                fw = fp_weights(model, L, e)
                sync(); t0 = time.time()
                Wq = {nm: gptq_quantize(fw[nm], (Hg if nm != "down_proj" else Hd).to(MU.GPU), GPTQ["bits"],
                                        GPTQ["group_size"], pd) for nm in ("gate_proj", "up_proj", "down_proj")}
                sync(); t_solve += time.time() - t0
                W[cn][e] = {k: v.to(torch.bfloat16).cpu() for k, v in Wq.items()}
            res["timing"][f"L{L}.{cn}"] = dict(mix_s=t_mix, solve_s=t_solve, n_experts=len(sel))
            torch.save({f"{e}.{k}": v for e in sel for k, v in W[cn][e].items()}, s("weights", f"L{L}_{cn}.pt"))
            print(f"[quantize] L{L} {cn}: solve {t_solve:.1f}s mix {t_mix:.2f}s", flush=True)
        c = {}
        for B in jload("stage1_decision.json")["eligible_budgets"]:
            std, pool, idn = W[f"B{B}_std"], W[f"B{B}_pool"], W[f"B{B}_identity"]
            c[f"B{B}_alpha0_equals_std_bitwise"] = all(torch.equal(W[f"B{B}_alpha0"][e][k], std[e][k])
                                                       for e in sel for k in std[e])
            c[f"B{B}_down_identical_std_pool_identity"] = all(torch.equal(std[e]["down_proj"], x[e]["down_proj"])
                                                              for e in sel for x in (pool, idn))
            c[f"B{B}_gu_differs_pool"] = all(not torch.equal(std[e]["gate_proj"], pool[e]["gate_proj"]) for e in sel)
        hist = torch.load(os.path.join(AV.weights_dir("GPTQ", CONFIG), f"L{L}.pt"))
        c["B128_std_equals_historical"] = all(torch.equal(W["B128_std"][e][k], hist[f"{e}.{k}"]) for e in sel
                                              for k in ("gate_proj", "up_proj", "down_proj"))
        del hist
        c["all_finite"] = all(bool(torch.isfinite(t.float()).all()) for cw in W.values() for ew in cw.values()
                              for t in ew.values())
        c["model_weights_unchanged"] = PQ.layer_digest(model, L) == dig0
        res["checks"][str(L)] = c
        print(f"[quantize] L{L} checks {c}", flush=True)
        del W, st
    res["peak_gpu_gb"], res["peak_rss_gb"] = gpu_peak_gb(), rss_gb()
    res["passed"] = all(all(v.values()) for v in res["checks"].values())
    jdump(res, p("logs", "quantize_meta.json"))
    print("[quantize] checks passed:", res["passed"])


# ===================================================================== evaluate (confirmation split)
@torch.no_grad()
def eval_layer(model, L, split, weights: dict, sel):
    """weights: {cand: {e: {proj: bf16 tensor}}}. Per-sequence numerators / denominators for every metric."""
    import torch.nn.functional as F
    from moe_calib import model_utils as MU
    from moe_calib import agq as AG
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.quant_eval import fp_weights
    GPU = MU.GPU
    ids = list(range(*SPLITS[split]["rows"]))
    S, ne = len(ids), len(sel)
    blk = MU.moe_block(model, L)
    k = MU.top_k(model)
    dig0 = PQ.layer_digest(model, L)
    Wfp = {e: {kk: v.to(GPU) for kk, v in fp_weights(model, L, e).items()} for e in sel}
    Wq = {c: {e: {kk: v.to(GPU, torch.float32) for kk, v in w[e].items()} for e in sel} for c, w in weights.items()}
    cache = load_layer_cache(SPLITS[split]["cache"], L)
    mets = ("J", "U", "gate", "up", "e3")
    num = {c: {m: np.zeros((S, ne)) for m in mets} for c in weights}
    den = {m: np.zeros((S, ne)) for m in mets}
    tokens = np.zeros((S, ne))
    t0 = time.time()
    with MU.layer_on_gpu(model, L):
        for si, sid in enumerate(ids):
            cap = {}
            hk = [blk.register_forward_pre_hook(lambda m_, a_: cap.__setitem__("h", a_[0].detach())),
                  blk.gate.register_forward_hook(lambda m_, i, o: cap.__setitem__("lg", o.detach()))]
            try:
                MU.run_layer(model, L, cache[sid:sid + 1].to(GPU))
            finally:
                for h in hk:
                    h.remove()
            hf = cap["h"].reshape(-1, cap["h"].shape[-1]).float()
            vals, sel_ = AG.routing_coefficients(cap["lg"], k)
            rows = AG.dispatch_rows(sel_, vals, E)
            for j, e in enumerate(sel):
                if e not in rows:
                    continue
                tx, slot, cc = rows[e]
                Xe = hf[tx]
                c2 = cc.double() ** 2
                W0 = Wfp[e]
                g0, u0 = Xe @ W0["gate_proj"].T, Xe @ W0["up_proj"].T
                a0 = F.silu(g0) * u0
                o0 = a0 @ W0["down_proj"].T
                on2 = (o0.double() ** 2).sum(1)
                tokens[si, j] = Xe.shape[0]
                den["J"][si, j] = float((c2 * on2).sum()); den["U"][si, j] = float(on2.sum())
                den["gate"][si, j] = float((g0.double() ** 2).sum()); den["up"][si, j] = float((u0.double() ** 2).sum())
                den["e3"][si, j] = den["gate"][si, j] + den["up"][si, j] + den["U"][si, j]
                for c in weights:
                    Wc = Wq[c][e]
                    g, u = Xe @ Wc["gate_proj"].T, Xe @ Wc["up_proj"].T
                    o = (F.silu(g) * u) @ Wc["down_proj"].T
                    dn2 = ((o0 - o).double() ** 2).sum(1)
                    num[c]["J"][si, j] = float((c2 * dn2).sum()); num[c]["U"][si, j] = float(dn2.sum())
                    eg = float(((g0 - g).double() ** 2).sum()); eu = float(((u0 - u).double() ** 2).sum())
                    num[c]["gate"][si, j], num[c]["up"][si, j] = eg, eu
                    ed = float(((a0 @ (W0["down_proj"] - Wc["down_proj"]).T).double() ** 2).sum())
                    num[c]["e3"][si, j] = eg + eu + ed
    assert PQ.layer_digest(model, L) == dig0, "model weights changed during evaluation"
    del Wq, Wfp, cache
    torch.cuda.empty_cache()
    return num, den, tokens, time.time() - t0


def stage_evaluate(a):
    from moe_calib import model_utils as MU
    if jload("stage1_decision.json")["label"] != "POOL-PRIOR-PROMISING":
        print("[evaluate] Stage 1 did not pass; not run"); return
    split = a.split
    os.makedirs(p("results", split), exist_ok=True)
    cands = [c for c in candidates() if not c.endswith("_alpha0")]
    model, _ = X.load_partial()
    torch.cuda.reset_peak_memory_stats()
    for L in LAYERS:
        outp = p("results", split, f"L{L}.npz")
        if os.path.exists(outp):
            print(f"[evaluate] {split} L{L} done"); continue
        sel = torch.load(s("hessians", f"L{L}.pt"))["experts"]
        W = {}
        for c in cands:
            f = torch.load(s("weights", f"L{L}_{c}.pt"))
            W[c] = {e: {k: f[f"{e}.{k}"] for k in ("gate_proj", "up_proj", "down_proj")} for e in sel}
        num, den, tok, secs = eval_layer(model, L, split, W, sel)
        arr = {f"num_{c}_{m}": v for c, d in num.items() for m, v in d.items()}
        arr.update({f"den_{m}": v for m, v in den.items()})
        np.savez(outp, tokens=tok, experts=np.array(sel), **arr)
        jdump(dict(layer=L, split=split, candidates=cands, experts=sel, seconds=secs, peak_gpu_gb=gpu_peak_gb(),
                   peak_rss_gb=rss_gb()), p("results", split, f"L{L}_meta.json"))
        print(f"[evaluate] {split} L{L}: {secs:.0f}s", flush=True)


# ===================================================================== subset replacement (optional)
@torch.no_grad()
def stage_subset(a):
    """Six-expert subset replacement at budget --budget: all 60 experts standard GPTQ (raw H at that budget) vs the
    same with the 6 selected experts' gate/up replaced by pooled-shrinkage weights (down unchanged). Exact MoE output
    error (fp32, frozen BF16 routing, shared expert included) on the confirmation split."""
    import torch.nn.functional as F
    from moe_calib import model_utils as MU
    from moe_calib import agq as AG
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    from moe_calib.quant_eval import fp_weights
    from moe_calib.gptq import quantize_expert
    B = a.budget
    outp = p("subset_layer_output.json")
    if os.path.exists(outp):
        print("[subset] done"); return
    rows = manifest_rows()
    model, _ = X.load_partial()
    GPU = MU.GPU
    res = dict(budget=B, alpha=jload("alpha_selection.json")["selected"][str(B)], layers={})
    ids = list(range(*SPLITS["confirm"]["rows"]))
    for L in LAYERS:
        st = torch.load(s("hessians", f"L{L}.pt"))
        sel = st["experts"]
        cache = load_layer_cache(AV.ECACHE, L)
        pr = Profiler(model, L, K=K)
        nb = {16: 16, 32: 32, 64: 64, 128: 128}[B]
        pr.run(cache, rows[:nb], blocks=ES.contiguous_blocks(128, K)[:nb])
        bl = [0] if B in (16, 32) else ([0, 1] if B == 64 else [0, 1, 2, 3])
        t0 = time.time()
        Wstd, same = {}, True
        for e in range(E):
            Hg, Hd = pr.acc.H_sum((L, e, "gu"), bl).cpu(), pr.acc.H_sum((L, e, "down"), bl).cpu()
            if e in sel:
                same &= torch.equal(Hg, st["H"][f"{B}.{e}.gu"]) and torch.equal(Hd, st["H"][f"{B}.{e}.down"])
            assert pr.acc.tokens((L, e, "gu"), bl) > 0
            Wq = quantize_expert(fp_weights(model, L, e), Hg, Hd, GPTQ["bits"], GPTQ["group_size"], GPTQ["percdamp"])
            Wstd[e] = {k: v.to(torch.bfloat16).to(GPU, torch.float32) for k, v in Wq.items()}
        pr.free(); del pr
        t_q = time.time() - t0
        fstd = torch.load(s("weights", f"L{L}_B{B}_std.pt"))
        fpool = torch.load(s("weights", f"L{L}_B{B}_pool.pt"))
        std_match = all(torch.equal(Wstd[e][k].to(torch.bfloat16).cpu(), fstd[f"{e}.{k}"]) for e in sel
                        for k in ("gate_proj", "up_proj", "down_proj"))
        Wsub = {e: (dict(Wstd[e], gate_proj=fpool[f"{e}.gate_proj"].to(GPU, torch.float32),
                         up_proj=fpool[f"{e}.up_proj"].to(GPU, torch.float32)) if e in sel else Wstd[e])
                for e in range(E)}
        Wfp = {e: {k: v.to(GPU) for k, v in fp_weights(model, L, e).items()} for e in range(E)}
        blk = MU.moe_block(model, L)
        k = MU.top_k(model)
        dig0 = PQ.layer_digest(model, L)
        tc = load_layer_cache(SPLITS["confirm"]["cache"], L)
        acc = dict(num_std=0.0, num_sub=0.0, den=0.0, diag_sel_std=0.0, diag_all_std=0.0)
        per_seq = dict(std=[], sub=[], den=[])
        with MU.layer_on_gpu(model, L):
            sh = {kk: getattr(blk.shared_expert, kk).weight.float() for kk in AG.PROJ}
            sg = blk.shared_expert_gate.weight.float()
            for sid in ids:
                cap = {}
                hk = [blk.register_forward_pre_hook(lambda m_, a_: cap.__setitem__("h", a_[0].detach())),
                      blk.gate.register_forward_hook(lambda m_, i, o: cap.__setitem__("lg", o.detach()))]
                try:
                    MU.run_layer(model, L, tc[sid:sid + 1].to(GPU))
                finally:
                    for h in hk:
                        h.remove()
                hf = cap["h"].reshape(-1, cap["h"].shape[-1]).float()
                vals, sel_ = AG.routing_coefficients(cap["lg"], k)
                rws = AG.dispatch_rows(sel_, vals, E)
                y = F.sigmoid(hf @ sg.T) * AG.expert_fp32(hf, sh)
                es_, eb_ = torch.zeros_like(hf), torch.zeros_like(hf)
                for e, (tx, slot, cc) in rws.items():
                    Xe = hf[tx]
                    o = AG.expert_fp32(Xe, Wfp[e])
                    y.index_add_(0, tx, o * cc[:, None])
                    ds = (o - AG.expert_fp32(Xe, Wstd[e])) * cc[:, None]
                    es_.index_add_(0, tx, ds)
                    eb_.index_add_(0, tx, (o - AG.expert_fp32(Xe, Wsub[e])) * cc[:, None] if e in sel else ds)
                    dd = float((ds.double() ** 2).sum())
                    acc["diag_all_std"] += dd
                    if e in sel:
                        acc["diag_sel_std"] += dd
                ns_, nb_, d_ = float((es_.double() ** 2).sum()), float((eb_.double() ** 2).sum()), float((y.double() ** 2).sum())
                acc["num_std"] += ns_; acc["num_sub"] += nb_; acc["den"] += d_
                per_seq["std"].append(ns_); per_seq["sub"].append(nb_); per_seq["den"].append(d_)
        assert PQ.layer_digest(model, L) == dig0
        rs, rb = acc["num_std"] / acc["den"], acc["num_sub"] / acc["den"]
        res["layers"][str(L)] = dict(std_rel_err=rs, subset_rel_err=rb, rel_change=rb / rs - 1.0,
                                     selected_share_of_diag_error=acc["diag_sel_std"] / acc["diag_all_std"],
                                     hessians_equal_stored=bool(same), std_weights_equal_stage2=bool(std_match),
                                     quantize_60_s=t_q, per_seq=per_seq)
        print(f"[subset] L{L}: std {rs:.6e} subset {rb:.6e} ({rb / rs - 1:+.4%}); selected share "
              f"{acc['diag_sel_std'] / acc['diag_all_std']:.3f}", flush=True)
        del Wstd, Wsub, Wfp, st
        torch.cuda.empty_cache()
    res["note"] = "six-expert subset replacement; NOT equivalent to pooling all 60 experts"
    jdump(res, outp)


# ===================================================================== timing
@torch.no_grad()
def stage_timing(a):
    """Per layer, 128 calibration sequences pre-materialized: forward only | + pooled hook | + expert Hessians (Profiler,
    production path) | + both. 1 warm-up, 3 interleaved repetitions, synchronized."""
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    if os.path.exists(p("logs", "timing_raw.json")):
        print("[timing] done"); return
    rows = manifest_rows()
    model, _ = X.load_partial()
    conds = ["forward", "pooled", "experts", "experts+pooled"]
    out = {c: {} for c in conds}
    gemm = {}
    for L in LAYERS:
        Xc = load_layer_cache(AV.ECACHE, L)[rows].contiguous()
        blk = MU.moe_block(model, L)
        with MU.layer_on_gpu(model, L):
            def run(c):
                pr = Profiler(model, L, K=K) if "experts" in c else None
                pl = HS.PooledHessian(model.config.hidden_size, K, MU.GPU) if "pooled" in c else None
                if pr: pr.attach()
                if pl: pl.attach(blk)
                sync(); t0 = time.time()
                for i in range(Xc.shape[0]):
                    if pr: pr.cur_seq, pr.cur_block = i, (i * K) // 128
                    if pl: pl.cur_block = (i * K) // 128
                    MU.run_layer(model, L, Xc[i:i + 1].to(MU.GPU, non_blocking=True))
                sync(); dt = time.time() - t0
                if pr: pr.detach(); pr.free()
                if pl: pl.detach()
                del pr, pl
                torch.cuda.empty_cache()
                return dt
            run("experts+pooled")
            for rep in range(3):
                for c in conds[rep:] + conds[:rep]:
                    out[c].setdefault(str(L), []).append(run(c))
            # CUDA-event time of the pooled GEMM alone over the same 128 x 512 tokens
            Xg = Xc.to(MU.GPU)
            Hacc = torch.zeros(2048, 2048, device=MU.GPU)
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            for i in range(Xg.shape[0]):
                xf = Xg[i].float()
                Hacc += xf.T @ xf
            ev1.record(); sync()
            gemm[str(L)] = ev0.elapsed_time(ev1) / 1e3
            del Xg, Hacc
        print(f"[timing] L{L}: " + ", ".join(f"{c} {np.median(out[c][str(L)]):.2f}s" for c in conds), flush=True)
    jdump(dict(conditions=out, pooled_gemm_cuda_event_s=gemm, reps=3,
               note="pooled_gemm uses the cached layer INPUT as a same-shape stand-in for the MoE input (cost only)"),
          p("logs", "timing_raw.json"))


# ===================================================================== tests
def stage_tests(a):
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_hessian_shrinkage.py"] + ([] if not a.full else ["tests"])
    if a.full:
        cmd = [sys.executable, "-m", "pytest", "-q", "tests"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    open(p("logs", "pytest_full.log" if a.full else "pytest_targeted.log"), "w").write(r.stdout + r.stderr)
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-2000:])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--split", default="confirm")
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    dict(protocol=stage_protocol, hessians=stage_hessians, reference=stage_reference, mismatch=stage_mismatch,
         quantize=stage_quantize, evaluate=stage_evaluate, subset=stage_subset, timing=stage_timing,
         tests=stage_tests)[a.stage](a)


if __name__ == "__main__":
    main()
