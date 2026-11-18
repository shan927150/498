"""Legacy metric re-evaluation: do the historically failed calibration strategies (Full / Typical Coverage-LP,
official EBSS, B = 16 EBSS variants) still fail when quantization quality is measured as the ACTUAL aggregated
MoE-layer output error instead of the mean per-expert GPTQ reconstruction error?

Evaluation-only retrospective metric audit. No new calibration data, no new selection, no AGQ / c^2 weighting, no GPTQ
change. Every quantized weight set is standard (q = 0) GPTQ 4-bit g128 percdamp 1.0 act-order, CANONICAL-B1 capture.
The only quantization performed here is Typical Coverage-LP (TC) s0/s1/s2 from its frozen manifests, because TC was
never quantized under CANONICAL-B1; the new TC cache also contains FR s0's rows so that FR s0 re-quantized from the
NEW cache must reproduce the historical weights bitwise before any TC weight is admitted.

Metrics on identical frozen BF16 layer inputs (non-sequential), frozen router / top-4 / shared expert:
  * E_layer (primary): per layer  sum_x ||y - y_hat||^2 / sum_x ||y||^2  (fp32 math, float64 sums, the AGQ-only
    validation's definition); 4-layer aggregate = MEAN of the 4 layer ratios (AGQ convention).
  * E_expert (secondary): the production E3 score (quant_eval.per_seq_rel_errors, mean over held-out sequences per
    expert, mean over the 240 experts), recomputed on the INSTALLED BF16 weights.
  * decomposition ||dy||^2 = D + C with D = sum_k c_k^2 ||dE_k||^2, C = pairwise cross terms (computed independently).

Stages (--stage): audit | protocol | capture_tc | quantize --config {FR_s0_newcache, TC_s0, TC_s1, TC_s2} |
                  evaluate --split {confirm, oldtest} | analyze | report
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import full_depth_pilot as FD            # noqa: E402
import official_ebss_validation as OV    # noqa: E402
from moe_calib import agq as AG          # noqa: E402

OUT = "runs_legacy_metric_reeval"
LAYERS = OV.LAYERS
E = 60
SEEDS = (0, 1, 2)
ECACHE = OV.ECACHE                                                    # historical CANONICAL-B1 calibration cache
TCCACHE = os.path.expanduser(os.environ.get("LMR_TC_CACHE", "~/moe_cache_legacy_metric_reeval"))
WROOT = os.path.join(TCCACHE, "weights")
SPLITS = {
    "confirm": dict(cache=os.path.expanduser("~/moe_cache_confirmation"), rows=[0, 256], role="PRIMARY",
                    note="locked document-disjoint confirmation split (runs_aggregate_constraint); never used for "
                         "selection; previously used once to score expert reconstruction of FR/TR/TC/FC built from "
                         "the batch-8 historical cache, never for MoE-output error"),
    "oldtest": dict(cache=os.path.expanduser("~/moe_cache"), rows=[2128, 2256], role="SECONDARY / regression",
                    note="historical E3 test rows 2128-2255; inspected in many prior studies; exploratory"),
}
TAU = 0.005                                                           # same tau_rel as the AGQ-only validation
N_BOOT, BOOT_SEED = 2000, 20261112


def p(*a):
    return os.path.join(ROOT, OUT, *a)


jdump, sha256_file = FD.jdump, FD.sha256_file


def jload(*a):
    return json.load(open(p(*a)))


# ============================================================================ method registry
def _res(path):
    return json.load(open(os.path.join(ROOT, path)))


def methods_registry():
    """name -> dict(group, strategy, seed, result_json, weights_dir, recorded installed sha, protocol)."""
    reg = {}
    for c in ("FR", "TR", "FC"):
        for s in SEEDS:
            reg[f"{c}_s{s}"] = dict(group="B128", strategy=c, seed=s,
                                    result=f"runs_ebss_official_validation/results/{c}_s{s}.json")
    for s in SEEDS:
        reg[f"TC_s{s}"] = dict(group="B128", strategy="TC", seed=s, result=f"{OUT}/results/TC_s{s}.json")
    reg["EBSS"] = dict(group="B128_EBSS", strategy="EBSS", seed=None,
                       result="runs_ebss_official_validation/results/EBSS.json")
    for s in SEEDS:
        reg[f"C4R16_s{s}"] = dict(group="B16", strategy="C4R16", seed=s,
                                  result=f"runs_ebss_generation_pilot/results/C4R_s{s}.json")
    reg["RS_PO16"] = dict(group="B16", strategy="RS_PO16", seed=None, result="runs_ebss_generation_pilot/results/PO.json")
    reg["RS_EBSS16"] = dict(group="B16", strategy="RS_EBSS16", seed=None,
                            result="runs_ebss_balance_fidelity/results/ALG1_RATE.json")
    reg["PX_PO16"] = dict(group="B16", strategy="PX_PO16", seed=None, result="runs_ebss_prefix_rescue/results/PX_PO.json")
    reg["PX_EBSS16"] = dict(group="B16", strategy="PX_EBSS16", seed=None,
                            result="runs_ebss_prefix_rescue/results/PX_EBSS.json")
    return reg


STRAT_NAMES = dict(FR="Full Random", TR="Typical Random", FC="Full Coverage-LP", TC="Typical Coverage-LP",
                   EBSS="Official EBSS (released file)", C4R16="C4 Random B=16", RS_PO16="random-start PO B=16",
                   RS_EBSS16="random-start EBSS (ALG1_RATE) B=16", PX_PO16="C4-prefix PO B=16",
                   PX_EBSS16="C4-prefix EBSS B=16")


def weight_files_sha(wd):
    return {str(L): sha256_file(os.path.join(wd, f"L{L}.pt")) for L in LAYERS}


# ============================================================================ stage: audit
def stage_audit(a):
    """Inventory every candidate artifact: manifest, sequence hash, policy, GPTQ config, weights, checksums."""
    reg = methods_registry()
    man_ppl = json.load(open(os.path.join(ROOT, "runs_partial_ppl", "selected_set_manifest.json")))
    inv = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), methods={}, not_evaluated={})
    for name, m in reg.items():
        rp = os.path.join(ROOT, m["result"])
        ent = dict(m)
        if not os.path.exists(rp):
            ent.update(status="requires quantization (no CANONICAL-B1 weights exist)", result_exists=False)
            if m["strategy"] == "TC":
                src = man_ppl["configs"][name]
                ent.update(calibration_manifest=src["source"], calibration_ids_hash=src["ids_hash"], K=src["K"],
                           historical_protocol_weights="runs_partial_ppl used the batch-8 historical cache "
                                                       "(ORIGINAL extraction); NOT CANONICAL-B1 -> not comparable",
                           must_quantize=True)
            inv["methods"][name] = ent
            continue
        r = json.load(open(rp))
        wd = r["weights_dir"]
        files_ok = all(os.path.exists(os.path.join(wd, f"L{L}.pt")) for L in LAYERS)
        fsha = weight_files_sha(wd) if files_ok else None
        rec_sha = {L: v["sha256"] for L, v in r["weight_files"].items()}
        ent.update(result_exists=True, weights_dir=wd, weights_exist=files_ok, weight_file_sha256=fsha,
                   weight_files_match_record=fsha == rec_sha, installed_weights_sha256=r["installed_weights_sha256"],
                   policy=r.get("policy"), gptq=r["gptq"], layers=r["layers"],
                   n_quantized_experts=r["n_quantized_experts"], rtn_fallbacks=r["rtn_fallbacks"],
                   calibration_cache=r["calibration_cache"], reconstruction_cache=r["reconstruction_cache"],
                   historical_recon_oldtest=r["recon"], historical_ppl=r.get("ppl"), valid=r["valid"],
                   cache_rows_sha=hashlib.sha256(json.dumps(r.get("cache_rows")).encode()).hexdigest()[:16]
                   if r.get("cache_rows") is not None else None, must_quantize=False)
        if m["strategy"] in ("FR", "TR", "FC"):
            src = man_ppl["configs"][name]
            ent.update(calibration_manifest=src["source"], calibration_ids_hash=src["ids_hash"], K=src["K"])
        inv["methods"][name] = ent
        print(f"[audit] {name}: weights {files_ok} sha-match {ent['weight_files_match_record']} "
              f"policy {ent['policy']} recon {r['recon']:.7f}", flush=True)
    inv["not_evaluated"] = {
        "Coverage-LP-Repr (RC)": "only E3 score JSONs exist (runs_repr_constrained/e3_coverage_lp_repr_*); no saved "
                                 "quantized weights and its rows are not in any CANONICAL-B1 cache -> would need new "
                                 "capture + 3 quantizations; optional comparison skipped (not low cost)",
        "TC historical batch-8 weights": "runs_partial_ppl TC weights used the ORIGINAL batch-8 extraction; not "
                                         "mixed with CANONICAL-B1 weights",
    }
    inv["evaluation_splits"] = {k: dict(v, files={str(L): os.path.exists(os.path.join(v["cache"], f"L{L}.pt"))
                                                  for L in LAYERS}) for k, v in SPLITS.items()}
    jdump(inv, p("artifact_inventory.json"))


# ============================================================================ stage: protocol (frozen before results)
def code_hashes():
    files = ["diagnostics/legacy_metric_reeval.py", "moe_calib/agq.py", "moe_calib/quant_eval.py",
             "moe_calib/profiler.py", "moe_calib/hessian.py", "moe_calib/gptq.py", "moe_calib/partial_model_quant.py",
             "moe_calib/model_utils.py", "e3_allocation.py", "diagnostics/official_ebss_validation.py",
             "diagnostics/eval_partial_ppl.py"]
    return {f: sha256_file(os.path.join(ROOT, f)) for f in files}


def stage_protocol(a):
    if os.path.exists(p("protocol.json")):
        print("[protocol] already frozen; not rewritten"); return
    prot = dict(
        experiment="Legacy metric re-evaluation: expert reconstruction vs actual aggregated MoE-layer output error",
        nature="retrospective, evaluation-only metric audit; not a new calibration method",
        model="Qwen/Qwen1.5-MoE-A2.7B (BF16)", layers=LAYERS,
        gptq=dict(bits=4, group_size=128, percdamp=1.0, act_order=True, hessian_weighting="none (q=0)",
                  capture="CANONICAL-B1 (each calibration sequence forwarded alone, BF16, non-sequential)"),
        methods=dict(
            primary_B128=[f"{c}_s{s}" for c in ("FR", "TR", "FC", "TC") for s in SEEDS],
            secondary_B128_EBSS=["EBSS"],
            exploratory_B16=["C4R16_s0", "C4R16_s1", "C4R16_s2", "RS_PO16", "RS_EBSS16", "PX_PO16", "PX_EBSS16"],
            not_evaluated=["Coverage-LP-Repr (no saved weights; would need new capture + quantization)"]),
        comparisons=dict(
            main="FC_sN vs FR_sN (paired by seed)",
            secondary="TC_sN vs TR_sN (paired by seed)",
            additional="TR_sN vs FR_sN (paired by seed)",
            ebss="EBSS vs FR_s0/s1/s2 and vs TR_s0/s1/s2 (one released set; three comparisons are NOT replications)",
            b16_exploratory="PX_EBSS16, PX_PO16, RS_EBSS16, RS_PO16 vs C4R16_s0 (matched) and vs C4R16 seed range; "
                            "never ranked with B=128"),
        evaluation=dict(
            primary_split="confirm", secondary_split="oldtest", splits=SPLITS,
            inputs="frozen BF16 layer inputs of each split's cache (identical for every method); routing, top-4, "
                   "coefficients and shared expert from the BF16 model, never from a quantized model",
            primary_metric="E_layer per layer = sum_x ||y_BF16 - y_q||^2 / (sum_x ||y_BF16||^2 + 1e-30), fp32 "
                           "expert math, float64 sums, shared expert included in y, frozen routing (the "
                           "AGQ-only validation's LayerEvalAccumulator definition)",
            aggregate="MEAN of the four per-layer ratios (AGQ-only convention); a pooled ratio is stored in json "
                      "for reference only and is not used for any decision",
            relative_difference="Delta = (E^method - E^ref) / E^ref of the 4-layer mean (negative = method better)",
            secondary_metric="E_expert = production E3 score (quant_eval.per_seq_rel_errors on Profiler held-out "
                             "inputs, mean over sequences per expert, mean over 240 experts) on the INSTALLED BF16 "
                             "weights (the weights the layer metric uses); historical E3 (solver fp32 W_q, oldtest) "
                             "is reported alongside",
            native_check="actual Qwen2MoeSparseMoeBlock BF16 forward with installed weights (secondary)",
            decomposition="||dy||^2 = D + C, D = sum_k ||c_k dE_k||^2, C = 2 sum_{a<b} <c_a dE_a, c_b dE_b>; each "
                          "divided by the same layer denominator",
            uncertainty="seed-level paired effects (3 seeds); descriptive paired window bootstrap (2000 reps, "
                        "windows resampled jointly across layers and methods). Experts and tokens are NOT "
                        "treated as independent replicates."),
        decision_rule=dict(
            tau_rel=TAU,
            legacy_direction="from E_expert on the same split: 'penalty' if mean Delta_expert >= +tau and "
                             "Delta_expert > 0 on >= 2/3 seeds; 'gain' mirror; otherwise 'tie'",
            when_legacy_penalty={
                "METRIC-RANKING-REVERSAL (B)": "Delta_layer <= -tau on every paired seed",
                "LEGACY-NEGATIVE-CONFIRMED (A)": "Delta_layer > 0 on >= 2/3 seeds, no seed <= -tau, and mean "
                                                 "Delta_layer >= max(tau, 0.5 * mean Delta_expert)",
                "METRIC-ALIGNED-BUT-SMALL (D)": "Delta_layer > 0 on >= 2/3 seeds, no seed <= -tau, but mean "
                                                "Delta_layer < max(tau, 0.5 * mean Delta_expert)",
                "MIXED-OR-LAYER-SPECIFIC (C)": "otherwise"},
            when_legacy_tie_or_gain={
                "NEW-METRIC-GAIN": "Delta_layer <= -tau on every paired seed",
                "NEW-METRIC-PENALTY": "Delta_layer >= +tau on every paired seed",
                "NO-ROBUST-DIFFERENCE": "otherwise"},
            layer_specific_flag="a layer whose Delta_layer has the opposite sign to the 4-layer mean on every "
                                "paired seed with |Delta| >= tau is reported as a layer-specific reversal",
            evaluation_blocked="EVALUATION-BLOCKED (E) if any correctness gate fails or required weights are missing",
            single_set="EBSS / B16 single sets are classified with the same rule over their comparisons but labelled "
                       "'one calibration set, not replicated'",
            decision_split="classifications on the PRIMARY split decide; secondary-split disagreement is reported",
            final_recommendation={
                "A": "FC-vs-FR is B, or TC-vs-TR is NEW-METRIC-GAIN / B",
                "B": "FC-vs-FR is A or D, TC-vs-TR is not a gain, and EBSS-vs-FR is not a gain",
                "C": "otherwise"}),
        regression=dict(
            fr_s0_weights_sha256="ecfdb234c1cf3b3283d9434739572c5ce890522f8e68a380c5296b07e22776a1",
            fr_s0_recon=0.0046094692935882,
            agq_q0_layer_output_FR_s0_oldtest=0.0037166111573928374,
            note="historical references used only as regression checks after confirming identical universe"),
        hard_constraints=["no new calibration data", "no new selection", "no EBSS regeneration", "no AGQ / c^2",
                          "no GPTQ / damping change", "no full-depth quantization", "no new PPL experiment "
                          "(quant_config's built-in PPL validity check runs for TC and is reported only as a "
                          "byproduct)", "historical artifacts not overwritten"],
        code_sha256=code_hashes(),
        frozen_before_any_new_output_error_result=True,
        created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(prot, p("protocol.json"))
    print("[protocol] frozen")


# ============================================================================ stage: capture_tc
def tc_selection(s):
    import e3_allocation as E3
    man = json.load(open(os.path.join(ROOT, "runs_partial_ppl", "selected_set_manifest.json")))
    ent = man["configs"][f"TC_s{s}"]
    src = os.path.join(ROOT, ent["source"])
    r = json.load(open(src))
    assert sha256_file(src) == ent["source_sha256"], f"{src} changed"
    assert r["chosen"] == ent["chosen"] and E3.warm_hash(r["chosen"]) == ent["ids_hash"]
    assert ent["valid"] and all(ent["checks"].values())
    return dict(chosen=list(map(int, r["chosen"])), K=ent["K"], ids_hash=ent["ids_hash"], source=ent["source"],
                source_sha256=ent["source_sha256"])


def capture_rows():
    fr = OV.selections()["FR_s0"]["chosen"]
    tc = [tc_selection(s)["chosen"] for s in SEEDS]
    return sorted(set(fr).union(*map(set, tc)))


@torch.no_grad()
def stage_capture_tc(a):
    man_path = p("logs", "tc_capture_manifest.json")
    if os.path.exists(man_path) and json.load(open(man_path)).get("complete"):
        print("[capture_tc] done"); return
    rows = capture_rows()
    allt = OV.c4_tokens()
    tokens = allt[rows]
    model, t_load = OV.load_model()
    assert OV.verify_pristine_full(model)
    os.makedirs(TCCACHE, exist_ok=True)
    t0 = time.time()
    stored, C, _, sec, t_tr = OV.forward_b1(model, tokens, list(range(len(rows))), LAYERS, LAYERS, want_nll=False)
    t_total = time.time() - t0
    # regression: every row that is also in the historical CANONICAL-B1 cache must be bitwise identical
    pol = json.load(open(os.path.join(ROOT, OV.OUT, "execution_policy.json")))
    pos = {r: i for i, r in enumerate(pol["rows"]["c4_rows"])}
    shared = [(j, pos[r]) for j, r in enumerate(rows) if r in pos]
    zC = np.load(os.path.join(ROOT, OV.OUT, "logs", "capture_routing_nll.npz"))["C"]
    eq = {}
    for L in LAYERS:
        R = torch.load(os.path.join(ECACHE, f"L{L}.pt"), mmap=True)
        eq[str(L)] = bool(all(torch.equal(stored[L][j], R[i]) for j, i in shared))
    routing_eq = bool(all((C[j].numpy() == zC[i]).all() for j, i in shared))
    files = {}
    for L in LAYERS:
        path = os.path.join(TCCACHE, f"L{L}.pt")
        torch.save(stored[L], path + ".tmp"); os.replace(path + ".tmp", path)
        files[str(L)] = dict(shape=list(stored[L].shape), sha256=sha256_file(path), tensor_sha256=OV.tensor_sha(stored[L]))
    torch.save(dict(c4_rows=rows, tokens=tokens), os.path.join(TCCACHE, "ids.pt"))
    np.savez(p("logs", "tc_capture_routing.npz"), C=C.numpy(), sec=sec)
    man = dict(cache_dir=TCCACHE, rows=rows, n_rows=len(rows), n_shared_with_historical_b1=len(shared),
               shared_rows_bitwise_equal=eq, shared_rows_routing_equal=routing_eq,
               passed=all(eq.values()) and routing_eq and len(shared) > 0, files=files, seconds_total=t_total,
               layer_transfer_s=t_tr, per_row_compute_s_mean=float(sec.mean()), model_load_s=t_load,
               peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, complete=True)
    jdump(man, man_path)
    print(f"[capture_tc] {len(rows)} rows ({len(shared)} shared with historical B1 cache): bitwise {eq} routing "
          f"{routing_eq} ({t_total:.0f} s)", flush=True)
    if not man["passed"]:
        raise SystemExit("TC capture regression FAILED")


# ============================================================================ stage: quantize
def stage_quantize(a):
    name = a.config
    assert name in ("FR_s0_newcache", "TC_s0", "TC_s1", "TC_s2")
    os.makedirs(p("results"), exist_ok=True)
    if os.path.exists(p("results", f"{name}.json")):
        print(f"[quant] {name} done"); return
    cm = jload("logs", "tc_capture_manifest.json")
    assert cm["passed"]
    for L in LAYERS:
        assert sha256_file(os.path.join(TCCACHE, f"L{L}.pt")) == cm["files"][str(L)]["sha256"], "cache changed"
    if name.startswith("TC"):
        reg = jload("baseline_regression.json")
        assert reg["fr_s0_newcache"]["passed"], "FR_s0 new-cache regression has not passed"
        sel = tc_selection(int(name[-1]))
    else:
        sel = OV.selections()["FR_s0"]
    pos = {r: i for i, r in enumerate(cm["rows"])}
    chosen = [pos[r] for r in sel["chosen"]]                       # frozen selection ORDER preserved
    assert len(chosen) == 128 == len(set(chosen))
    model, t_load = OV.load_model()
    assert OV.verify_pristine_full(model)
    r = OV.quant_config(model, name, TCCACHE, chosen, OV.OCACHE, sel["K"], weights_root=WROOT)
    r.update(config=name, policy="CANONICAL-B1", cache_rows=chosen, source_rows=sel["chosen"],
             ids_hash=sel.get("ids_hash"), model_load_s=t_load, hessian_weighting="none")
    jdump(r, p("results", f"{name}.json"))
    print(f"[quant] {name}: recon {r['recon']:.7f} fallbacks {r['rtn_fallbacks']} valid {r['valid']} "
          f"sha {r['installed_weights_sha256'][:12]} ({r['quantize_total_s']:.0f} s)", flush=True)
    if name == "FR_s0_newcache":
        hist = _res("runs_ebss_official_validation/results/FR_s0.json")
        rec = dict(weights_sha256=r["installed_weights_sha256"], weights_sha256_hist=hist["installed_weights_sha256"],
                   weights_exact=r["installed_weights_sha256"] == hist["installed_weights_sha256"],
                   recon=r["recon"], recon_hist=hist["recon"], recon_exact=r["recon"] == hist["recon"],
                   per_expert_exact=[x["mean"] for x in r["per_expert"]] == [x["mean"] for x in hist["per_expert"]],
                   ppl=r["ppl"], ppl_hist=hist["ppl"], ppl_exact=r["ppl"] == hist["ppl"], valid=r["valid"],
                   note="FR s0 re-quantized from the NEW cache (rows re-captured at batch 1) with the CURRENT code")
        rec["passed"] = all(rec[k] for k in ("weights_exact", "recon_exact", "per_expert_exact", "ppl_exact", "valid"))
        br = jload("baseline_regression.json") if os.path.exists(p("baseline_regression.json")) else {}
        br["fr_s0_newcache"] = rec
        jdump(br, p("baseline_regression.json"))
        print(f"[quant] FR_s0 new-cache regression passed={rec['passed']}", flush=True)
        if not rec["passed"]:
            raise SystemExit("FR_s0 regression FAILED")
    if not r["valid"]:
        raise SystemExit(f"{name} failed its validity checks")


# ============================================================================ stage: evaluate
def load_partial():
    os.environ["MOE_LOAD_LAYERS"] = ",".join(map(str, LAYERS))
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    t0 = time.time()
    model, _ = MU.load_model()
    pr = json.load(open(os.path.join(ROOT, FD.OUT, "pristine_digests.json")))
    tg = PQ.target_tensor_digests(model, LAYERS)
    assert tg == {k: pr["pristine_targets"][k] for k in tg}, "expert weights not pristine"
    return model, time.time() - t0


def eval_methods(include_missing=False):
    reg = methods_registry()
    out = {}
    for n, m in reg.items():
        rp = os.path.join(ROOT, m["result"])
        if os.path.exists(rp):
            r = json.load(open(rp))
            out[n] = dict(m, weights_dir=r["weights_dir"], recorded_files={L: v["sha256"] for L, v in
                                                                            r["weight_files"].items()})
        elif include_missing:
            out[n] = dict(m, weights_dir=None)
    return out


E3_CHECK_EXPERTS = (0, 17, 30, 59)          # prespecified, not outcome-based


@torch.no_grad()
def e3_per_seq_fast(W, Wq, HOg, L, e, GPU):
    """quant_eval.per_seq_rel_errors with the held-out inputs already resident on the GPU (same fp16 values, same
    fp32 ops in the same order) -> bitwise identical per-sequence values; verified against the production scorer."""
    D = {k: W[k] - Wq[k].to(GPU, torch.float32) for k in W}
    num, den = {}, {}
    for proj, names in (("gu", ("gate_proj", "up_proj")), ("down", ("down_proj",))):
        for sid, X in HOg.get((L, e, proj), []):
            Xf = X.to(GPU, torch.float32)
            for nm in names:
                num[sid] = num.get(sid, 0.0) + float((Xf @ D[nm].T).pow(2).sum())
                den[sid] = den.get(sid, 0.0) + float((Xf @ W[nm].T).pow(2).sum())
    return {s: num[s] / (den.get(s, 0.0) + 1e-12) for s in num}


@torch.no_grad()
def evaluate_layer(model, L, split, methods, check_methods=()):
    """All methods on one layer of one split. Returns per-method layer/decomposition/native/E3 records."""
    import torch.nn.functional as F
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    from moe_calib.quant_eval import per_seq_rel_errors
    GPU = MU.GPU
    sp = SPLITS[split]
    ids = list(range(*sp["rows"]))
    blk = MU.moe_block(model, L)
    k = MU.top_k(model)
    dig0 = PQ.layer_digest(model, L)
    orig = {e: {kk: w.detach().clone() for kk, w in MU.get_expert_weights(model, L, e).items()} for e in range(E)}
    Wfp = {e: {kk: v.to(GPU, torch.float32) for kk, v in orig[e].items()} for e in range(E)}
    cache = load_layer_cache(sp["cache"], L)
    t0 = time.time()
    # ---- E3 held-out inputs (production Profiler path; pristine weights installed)
    HO = Profiler(model, L, collect_hessian=False, collect_holdout=True).run(cache, ids).holdout
    universe = [e for e in range(E) if HO.get((L, e, "gu"))]
    t_ho = time.time() - t0
    # ---- BF16 reference pass: inputs, router, native y, FP expert outputs (stored), shared output, denominators
    seqs = []
    with MU.layer_on_gpu(model, L):
        sh = {kk: getattr(blk.shared_expert, kk).weight.float() for kk in AG.PROJ}
        sg = blk.shared_expert_gate.weight.float()
        for sid in ids:
            cap = {}
            hk = [blk.register_forward_pre_hook(lambda m_, a_: cap.__setitem__("h", a_[0].detach())),
                  blk.register_forward_hook(lambda m_, i, o: cap.__setitem__("y", o[0].detach())),
                  blk.gate.register_forward_hook(lambda m_, i, o: cap.__setitem__("lg", o.detach()))]
            try:
                MU.run_layer(model, L, cache[sid:sid + 1].to(GPU))
            finally:
                for h in hk:
                    h.remove()
            h = cap["h"].reshape(-1, cap["h"].shape[-1])
            vals, sel = AG.routing_coefficients(cap["lg"], k)
            rows = AG.dispatch_rows(sel, vals, E)
            hf = h.float()
            y = F.sigmoid(hf @ sg.T) * AG.expert_fp32(hf, sh)
            outs = {}
            c2den = 0.0
            for e, (tx, slot, c) in rows.items():
                X = hf[tx]
                o = AG.down_input_fp32(X, Wfp[e]) @ Wfp[e]["down_proj"].T     # identical ops to LayerEvalAccumulator
                y.index_add_(0, tx, o * c[:, None])
                outs[e] = o
                c2den += float((c.double() ** 2 * (o.double() ** 2).sum(1)).sum())
            seqs.append(dict(h=h, lg=cap["lg"], y0=cap["y"].reshape(h.shape), rows=rows, outs=outs,
                             yn2=float((y.double() ** 2).sum()), c2den=c2den,
                             y0n2=float((cap["y"].reshape(h.shape).double() ** 2).sum())))
        t_ref = time.time() - t0 - t_ho
        res = {}
        T, d = seqs[0]["h"].shape
        for name, m in methods.items():
            tm = time.time()
            wd = m["weights_dir"]
            path = os.path.join(wd, f"L{L}.pt")
            fsha = sha256_file(path)
            assert fsha == m["recorded_files"][str(L)], f"{name} L{L} weight file changed"
            files = torch.load(path)
            Wq = {e: {kk: files[f"{e}.{kk}"].to(GPU, torch.float32) for kk in AG.PROJ} for e in range(E)}
            diffs = sum(int((files[f"{e}.{kk}"] != orig[e][kk].cpu()).any()) for e in range(E) for kk in AG.PROJ)
            num_s, den_s, D_s, C_s, tot_s = [], [], [], [], []
            for S in seqs:
                hf = S["h"].float()
                err = torch.zeros_like(hf)
                V = torch.zeros(T, k, d, device=GPU, dtype=torch.float32)
                for e, (tx, slot, c) in S["rows"].items():
                    dl = S["outs"][e] - AG.expert_fp32(hf[tx], Wq[e])
                    err.index_add_(0, tx, dl * c[:, None])
                    V[tx, slot] = dl * c[:, None]
                dec = AG.decompose(V)
                num_s.append(float((err.double() ** 2).sum())); den_s.append(S["yn2"])
                D_s.append(dec["diag"]); C_s.append(dec["cross"]); tot_s.append(dec["total"])
                del V, err
            # native BF16 block forward with installed weights (frozen router: logits must be unchanged)
            for e in range(E):
                MU.set_expert_weights(model, L, e, {kk: files[f"{e}.{kk}"] for kk in AG.PROJ})
            nat_num = 0.0
            for S in seqs:
                yq, lg = blk(S["h"].unsqueeze(0))
                assert torch.equal(lg, S["lg"]), "router output changed under quantization"
                nat_num += float(((yq.reshape(S["h"].shape).double() - S["y0"].double()) ** 2).sum())
            for e in range(E):
                MU.set_expert_weights(model, L, e, orig[e])
            res[name] = dict(weight_file_sha256=fsha, projections_differing_from_bf16=diffs,
                             num=float(np.sum(num_s)), den=float(np.sum(den_s)),
                             moe_layer_output_rel_err=float(np.sum(num_s)) / (float(np.sum(den_s)) + 1e-30),
                             D=float(np.sum(D_s)), C=float(np.sum(C_s)), decomp_total=float(np.sum(tot_s)),
                             c2_weighted_expert_err=float(np.sum(D_s)) / sum(S["c2den"] for S in seqs),
                             native_rel_err=nat_num / sum(S["y0n2"] for S in seqs),
                             per_seq=dict(num=num_s, den=den_s, D=D_s, C=C_s),
                             _Wq_cpu=None, seconds_layer=time.time() - tm)
            del Wq
            torch.cuda.empty_cache()
            res[name]["_files"] = files
            print(f"[evaluate] {split} L{L} {name}: E_layer {res[name]['moe_layer_output_rel_err']:.6e} "
                  f"({time.time() - tm:.0f} s)", flush=True)
        # ---- correctness reference: the validated AGQ LayerEvalAccumulator on the check methods
        chk = {}
        if check_methods:
            Wqc = {n: {e: {kk: res[n]["_files"][f"{e}.{kk}"].to(GPU, torch.float32) for kk in AG.PROJ}
                       for e in range(E)} for n in check_methods}
            acc = AG.LayerEvalAccumulator(list(check_methods), E, model.config.hidden_size)
            for S in seqs:
                hf = S["h"].float()
                vals, sel = AG.routing_coefficients(S["lg"], k)
                shared = F.sigmoid(hf @ sg.T) * AG.expert_fp32(hf, sh)
                acc.add_sequence(hf, vals, sel, Wfp, Wqc, shared)
            s = acc.summary()
            for n in check_methods:
                chk[n] = dict(accumulator=s[n]["moe_layer_output_rel_err"], lean=res[n]["moe_layer_output_rel_err"],
                              equal=s[n]["moe_layer_output_rel_err"] == res[n]["moe_layer_output_rel_err"],
                              abs_diff=abs(s[n]["moe_layer_output_rel_err"] - res[n]["moe_layer_output_rel_err"]))
            del Wqc
    # ---- E3 per method on the installed BF16 weights (pristine weights in the model; per_seq_rel_errors reads them)
    assert PQ.layer_digest(model, L) == dig0, "pristine weights not restored"
    seqs.clear()
    torch.cuda.empty_cache()
    HOg = {key: [(sid, X.to(GPU)) for sid, X in v] for key, v in HO.items()}   # fp16 values unchanged
    Wfe = Wfp                                   # = quant_eval.fp_weights (pristine weight.float()), on the GPU
    e3_check = dict(experts=[], bitwise_equal=True)
    for ni, name in enumerate(methods):
        files = res[name]["_files"]
        tm = time.time()
        M = np.full((E, len(ids)), np.nan)
        for e in universe:
            Wq_e = {kk: files[f"{e}.{kk}"] for kk in AG.PROJ}
            errs = e3_per_seq_fast(Wfe[e], Wq_e, HOg, L, e, GPU)
            if ni == 0 and e in E3_CHECK_EXPERTS:                   # production scorer, bitwise
                ref = per_seq_rel_errors(model, L, e, Wq_e, HO)
                eq = ref == errs
                e3_check["experts"].append(dict(method=name, expert=e, equal=eq))
                e3_check["bitwise_equal"] &= eq
                assert eq, f"fast E3 differs from production scorer at L{L} e{e}"
            for sid, v in errs.items():
                M[e, sid - ids[0]] = v
        res[name].pop("_files")
        res[name]["e3_per_expert_mean"] = [float(np.nanmean(M[e])) if e in universe else None for e in range(E)]
        res[name]["e3_per_expert_p90"] = [float(np.nanquantile(M[e], .9)) if e in universe else None for e in range(E)]
        res[name]["_e3_matrix"] = M
        res[name]["seconds_e3"] = time.time() - tm
        res[name].pop("_Wq_cpu")
        del files
    assert PQ.layer_digest(model, L) == dig0, "pristine weights not restored"
    del HOg, Wfe
    torch.cuda.empty_cache()
    meta = dict(n_seqs=len(ids), tokens=len(ids) * T, universe=universe, holdout_s=t_ho, reference_s=t_ref,
                pristine_restored=True, accumulator_check=chk, e3_fast_vs_production=e3_check,
                decomposition_identity_max_rel=max(abs(res[n]["D"] + res[n]["C"] - res[n]["decomp_total"]) /
                                                   res[n]["decomp_total"] for n in methods),
                decomposition_vs_fp32_err_max_rel=max(abs(res[n]["decomp_total"] - res[n]["num"]) / res[n]["num"]
                                                      for n in methods))
    del seqs, HO, cache, Wfp
    torch.cuda.empty_cache()
    return res, meta


def stage_evaluate(a):
    split = a.split
    os.makedirs(p("results", split), exist_ok=True)
    assert os.path.exists(p("protocol.json")), "protocol must be frozen first"
    methods = eval_methods()
    if a.only:
        methods = {n: methods[n] for n in a.only.split(",")}
    model, t_load = load_partial()
    check = tuple(n for n in ("FR_s0",) if n in methods)
    for L in LAYERS:
        outp = p("results", split, f"L{L}.json")
        done = json.load(open(outp)) if os.path.exists(outp) else dict(methods={})
        todo = {n: m for n, m in methods.items() if n not in done["methods"]}
        if not todo:
            print(f"[evaluate] {split} L{L} done"); continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        res, meta = evaluate_layer(model, L, split, todo, check_methods=tuple(c for c in check if c in todo))
        for n, r in res.items():
            np.save(p("results", split, f"e3_matrix_{n}_L{L}.npy"), r.pop("_e3_matrix"))
            done["methods"][n] = r
        done.setdefault("meta_runs", []).append(dict(meta, methods=list(todo), seconds=time.time() - t0,
                                                     peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
                                                     model_load_s=t_load))
        jdump(done, outp)


# ============================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--config")
    ap.add_argument("--split", default="confirm", choices=list(SPLITS))
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    if a.stage in ("analyze", "report"):
        import report_legacy_metric_reeval as R
        return R.main(a.stage)
    dict(audit=stage_audit, protocol=stage_protocol, capture_tc=stage_capture_tc, quantize=stage_quantize,
         evaluate=stage_evaluate)[a.stage](a)


if __name__ == "__main__":
    main()
