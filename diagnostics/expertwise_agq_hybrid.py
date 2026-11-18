"""Expert-wise GPTQ/AGQ hybrid: small-scale feasibility MVP (exploratory).

For every routed expert of layers 8 and 16 (120 experts), two EXISTING candidate quantizations are available from the
AGQ-only validation (FR s0, C4 Random B = 128 x 512, CANONICAL-B1, GPTQ 4-bit g128 percdamp 1.0 act-order):
    q = 0  standard GPTQ   (historical weights  ~/moe_cache_ebss_official_validation/weights/FR_s0)
    q = 1  AGQ             (historical weights  ~/moe_cache_agq_only/weights/AGQ_FR_s0)
No quantization is performed here. The hybrid installs, per expert, all three projections of the candidate with the lower
c^2-weighted full-expert output error on the TUNING split (~/moe_cache rows 2000-2127). Evaluation is the actual
aggregated MoE-layer output error on the confirmation split (primary) and the old test split (secondary).

Stages (--stage): audit | protocol | regression | score_tune | select | evaluate --split {confirm, oldtest} |
                  tests | report
"""
from __future__ import annotations

import argparse, json, os, resource, subprocess, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import full_depth_pilot as FD            # noqa: E402
import official_ebss_validation as OV    # noqa: E402
from moe_calib import agq as AG          # noqa: E402
from moe_calib import hybrid as HY       # noqa: E402

OUT = "runs_expertwise_agq_hybrid"
LAYERS = [8, 16]
ALL_LAYERS = OV.LAYERS                                   # [1, 8, 16, 23]: layout of historical per-expert arrays
E = 60
CONFIG = "FR_s0"
SPLITS = {
    "tune": dict(cache=os.path.expanduser("~/moe_cache"), rows=[2000, 2128], role="SELECTION-VALIDATION",
                 note="historical tuning split; disjoint from the C4 pool rows 0-1999 the FR s0 calibration set was "
                      "drawn from; the ONLY split used for expert selection and the layer-wise baseline choice"),
    "confirm": dict(cache=os.path.expanduser("~/moe_cache_confirmation"), rows=[0, 256], role="PRIMARY EVALUATION",
                    note="document-disjoint confirmation split; identical windows to runs_legacy_metric_reeval; "
                         "inspected in earlier studies -> not pristine"),
    "oldtest": dict(cache=os.path.expanduser("~/moe_cache"), rows=[2128, 2256], role="SECONDARY EVALUATION",
                    note="historical test rows 2128-2255; inspected in many prior studies"),
}
METHODS = ("GPTQ", "AGQ", "LAYERWISE", "HYBRID")
RULES = dict(tau_rel=0.005, min_sign_agreement=0.65, min_spearman=0.5)


def p(*a):
    return os.path.join(ROOT, OUT, *a)


jdump, sha256_file = FD.jdump, FD.sha256_file


def jload(*a):
    return json.load(open(p(*a)))


def peak_cpu_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def hist(name):
    path = dict(GPTQ="runs_ebss_official_validation/results/FR_s0.json",
                AGQ="runs_agq_only_validation/results/AGQ_FR_s0.json")[name]
    return path, json.load(open(os.path.join(ROOT, path)))


def cand_dirs():
    return {0: hist("GPTQ")[1]["weights_dir"], 1: hist("AGQ")[1]["weights_dir"]}


def expert_hash(files, e):
    return OV.tensor_sha(torch.cat([files[f"{e}.{k}"].reshape(-1) for k in AG.PROJ]))


def load_candidates(L, verify=True):
    """{q: {key: bf16 tensor}} for layer L; file sha verified against the historical record."""
    out = {}
    for q, nm in HY.CANDIDATES.items():
        path = os.path.join(cand_dirs()[q], f"L{L}.pt")
        if verify:
            assert sha256_file(path) == hist(nm)[1]["weight_files"][str(L)]["sha256"], f"{nm} L{L} file changed"
        out[q] = torch.load(path)
    return out


def code_hashes():
    files = ["diagnostics/expertwise_agq_hybrid.py", "moe_calib/hybrid.py", "moe_calib/agq.py", "moe_calib/gptq.py",
             "moe_calib/model_utils.py", "moe_calib/partial_model_quant.py", "tests/test_expertwise_hybrid.py"]
    return {f: sha256_file(os.path.join(ROOT, f)) for f in files if os.path.exists(os.path.join(ROOT, f))}


# ============================================================================ stage: audit
def stage_audit(a):
    inv = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), layers=LAYERS, candidates={})
    cm = json.load(open(os.path.join(ROOT, "runs_agq_only_validation", "calibration_manifest.json")))
    cm_sha = sha256_file(os.path.join(ROOT, "runs_agq_only_validation", "calibration_manifest.json"))
    ev = json.load(open(os.path.join(ROOT, "runs_agq_only_validation", "results", "eval_FR_s0.json")))
    for q, nm in HY.CANDIDATES.items():
        rpath, r = hist(nm)
        ent = dict(q=q, method=nm, result_json=rpath, weights_dir=r["weights_dir"], gptq=r["gptq"],
                   hessian_weighting=r.get("hessian_weighting", "none"), policy=r.get("policy"),
                   calibration_cache=r["calibration_cache"], cache_rows_equal_frozen_manifest=
                   r["cache_rows"] == cm["configs"][CONFIG]["cache_rows"],
                   calibration_manifest="runs_agq_only_validation/calibration_manifest.json",
                   calibration_manifest_sha256=cm_sha, calibration_ids_hash=cm["configs"][CONFIG]["ids_hash"],
                   installed_weights_sha256_all4layers=r["installed_weights_sha256"], rtn_fallbacks=r["rtn_fallbacks"],
                   historical_recon_E3_oldtest_4layers=r["recon"], valid=r["valid"], layers={})
        for L in LAYERS:
            path = os.path.join(r["weights_dir"], f"L{L}.pt")
            exists = os.path.exists(path)
            rec = dict(path=path, exists=exists)
            if exists:
                fsha = sha256_file(path)
                f = torch.load(path)
                keys = sorted(f)
                rec.update(file_sha256=fsha, file_sha256_matches_record=fsha == r["weight_files"][str(L)]["sha256"],
                           n_tensors=len(keys), n_experts=len({k.split(".")[0] for k in keys}),
                           projections=sorted({k.split(".", 1)[1] for k in keys}),
                           all_bf16=all(t.dtype == torch.bfloat16 for t in f.values()),
                           all_finite=all(bool(torch.isfinite(t).all()) for t in f.values()),
                           loadable=True,
                           expert_hashes=[expert_hash(f, e) for e in range(E)],
                           historical_E3_mean_of_60=float(np.mean([x["mean"] for x in r["per_expert"]
                                                                   if x["layer"] == L])),
                           historical_layer_output_oldtest=ev["layers"][str(L)][nm]["moe_layer_output_rel_err"],
                           historical_layer_output_native_oldtest=ev["layers"][str(L)][nm][
                               "moe_layer_output_native_rel_err"])
                rec["structure_ok"] = bool(rec["n_tensors"] == 3 * E and rec["n_experts"] == E and
                                           rec["projections"] == sorted(AG.PROJ))
                del f
            ent["layers"][str(L)] = rec
            print(f"[audit] {nm} L{L}: exists {exists} sha-match {rec.get('file_sha256_matches_record')} "
                  f"structure {rec.get('structure_ok')}", flush=True)
        inv["candidates"][nm] = ent
    lm = json.load(open(os.path.join(ROOT, "runs_legacy_metric_reeval", "results", "confirm", "L8.json")))
    inv["historical_confirm_gptq"] = {
        str(L): json.load(open(os.path.join(ROOT, "runs_legacy_metric_reeval", "results", "confirm", f"L{L}.json")))
        ["methods"]["FR_s0"]["moe_layer_output_rel_err"] for L in LAYERS}
    inv["historical_confirm_agq"] = "not available (AGQ never evaluated on the confirmation split)"
    inv["splits"] = {k: dict(v, files={str(L): os.path.exists(os.path.join(v["cache"], f"L{L}.pt")) for L in LAYERS})
                     for k, v in SPLITS.items()}
    chosen = OV.selections()[CONFIG]["chosen"]
    inv["calibration_rows_disjoint_from_tune"] = bool(max(chosen) < SPLITS["tune"]["rows"][0])
    inv["calibration_rows_max"] = int(max(chosen))
    inv["reproduction_needed"] = not all(
        inv["candidates"][nm]["layers"][str(L)].get("file_sha256_matches_record") and
        inv["candidates"][nm]["layers"][str(L)].get("structure_ok") for nm in HY.CANDIDATES.values() for L in LAYERS)
    del lm
    jdump(inv, p("artifact_inventory.json"))
    print(f"[audit] reproduction needed: {inv['reproduction_needed']}; calibration rows < 2000: "
          f"{inv['calibration_rows_disjoint_from_tune']}")


# ============================================================================ stage: protocol (frozen)
def stage_protocol(a):
    if os.path.exists(p("protocol.json")):
        print("[protocol] already frozen; not rewritten"); return
    inv = jload("artifact_inventory.json")
    assert not inv["reproduction_needed"] and inv["calibration_rows_disjoint_from_tune"]
    prot = dict(
        experiment="Expert-wise GPTQ/AGQ hybrid: small-scale feasibility MVP",
        nature="retrospective exploratory pilot; one calibration seed; outcome-informed layer choice",
        model="Qwen/Qwen1.5-MoE-A2.7B (BF16)", layers=LAYERS,
        layer_choice_caveat="L8 (AGQ slightly worse) and L16 (AGQ best) were chosen WITH knowledge of earlier AGQ "
                            "results; not an independent confirmatory sample",
        calibration=dict(config=CONFIG, manifest="runs_agq_only_validation/calibration_manifest.json",
                         manifest_sha256=inv["candidates"]["GPTQ"]["calibration_manifest_sha256"],
                         policy="CANONICAL-B1", n=128, seqlen=512, new_data=False),
        gptq=dict(bits=4, group_size=128, percdamp=1.0, act_order=True, solver="unchanged moe_calib.gptq",
                  capture="non-sequential BF16"),
        candidates={nm: dict(q=q, weights_dir=inv["candidates"][nm]["weights_dir"],
                             file_sha256={L: inv["candidates"][nm]["layers"][str(L)]["file_sha256"] for L in LAYERS})
                    for q, nm in HY.CANDIDATES.items()},
        splits=SPLITS,
        selection=dict(
            score="J_e^(q) = sum_{i: e in top4(i)} c_{i,e}^2 ||E_e^FP(x_i) - E_e^(q)(x_i)||_2^2 on the tuning split; "
                  "full expert MLP output in fp32 from BF16 weights; c = fp32 softmax top-4 router value of the BF16 "
                  "model (no renormalization); frozen BF16 layer inputs (non-sequential)",
            rule="z_e = 1 (AGQ) iff J_e^(1) < J_e^(0); otherwise (incl. ties) 0 (GPTQ); all three projections of one "
                 "candidate per expert",
            no_threshold=True, no_test_data=True),
        baselines=dict(
            GPTQ="all 60 experts q=0", AGQ="all 60 experts q=1",
            LAYERWISE="per layer, all-AGQ iff its actual aggregated layer-output error on the TUNING split is strictly "
                      "lower than all-GPTQ; else all-GPTQ",
            HYBRID="per-expert z_e from selection_manifest.json"),
        evaluation=dict(
            primary_split="confirm", secondary_split="oldtest",
            metric="E_layer = sum_x ||y_BF16 - y_q||^2 / (sum_x ||y_BF16||^2 + 1e-30), fp32 expert math, float64 "
                   "sums, shared expert included, frozen BF16 router/top-4/coefficients (LayerEvalAccumulator "
                   "definition)",
            aggregate="mean of the two per-layer ratios (2-layer scope; not comparable to 4-layer historical means)",
            native_check="actual BF16 Qwen2MoeSparseMoeBlock forward with installed weights (secondary)",
            decomposition="||dy||^2 = D + C, D = sum_k ||c_k dE_k||^2, C = 2 sum_{a<b} <c_a dE_a, c_b dE_b>",
            paired="per-window ratio differences, fraction of windows improved, window bootstrap of the 2-layer mean "
                   "relative difference (2000 reps, windows resampled jointly across layers)"),
        decision_rule=dict(
            rules=RULES,
            HYBRID_PROMISING="all of: (1) 2-layer mean hybrid error < uniform GPTQ and < uniform AGQ; (2) < layer-wise "
                             "uniform; (3) rel diff vs strongest of GPTQ/AGQ/LAYERWISE <= -tau; (4) per layer, hybrid "
                             "rel diff vs the better uniform method of that layer <= +tau; (5) transfer: pooled "
                             "per-expert val->test sign agreement >= 0.65 AND Spearman of per-expert relative "
                             "improvement >= 0.5 (primary split); (6) correctness passes",
            HYBRID_SELECTION_FAIL="not promising, transfer criterion (5) fails, and hybrid does not improve on either "
                                  "uniform method by >= tau",
            HYBRID_LOCAL_ONLY="not the above; hybrid improves on BOTH uniform methods by >= tau but not on the "
                              "strongest non-hybrid baseline (layer-wise) by >= tau",
            HYBRID_NO_GAIN="otherwise",
            HYBRID_BLOCKED="candidate weights not reproducible or any correctness check fails",
            recommendation=dict(A="HYBRID-PROMISING only",
                                B="hybrid not better than layer-wise by >= tau AND layer-wise beats both uniform "
                                  "methods (2-layer mean)",
                                C="otherwise"),
            decision_split="confirm (primary); oldtest reported, never decisive",
            implementation="moe_calib.hybrid.classify / recommend"),
        hard_constraints=["no new quantization", "no solver / Hessian change", "no new calibration data",
                          "no threshold tuning", "no test data in selection", "no greedy / learned selection",
                          "historical artifacts read-only"],
        code_sha256=code_hashes(), frozen_before_any_selection_or_hybrid_result=True,
        created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(prot, p("protocol.json"))
    print("[protocol] frozen")


# ============================================================================ core evaluator
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


def shared_digest(blk):
    import hashlib
    h = hashlib.sha256()
    for mod in (blk.shared_expert.gate_proj, blk.shared_expert.up_proj, blk.shared_expert.down_proj,
                blk.shared_expert_gate, blk.gate):
        h.update(mod.weight.detach().contiguous().cpu().view(torch.int16).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate_layer(model, L, split, configs: dict, cands: dict, native=True, accumulator_check=False):
    """Evaluate several expert->candidate configurations on one layer of one split with frozen BF16 routing.
    configs = {name: z [60] in {0,1}}; cands = {0: files, 1: files}.
    Returns (per-config records, per-expert per-sequence matrices, meta)."""
    import torch.nn.functional as F
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    GPU = MU.GPU
    sp = SPLITS[split]
    ids = list(range(*sp["rows"]))
    S = len(ids)
    blk = MU.moe_block(model, L)
    k = MU.top_k(model)
    dig0, sdig0 = PQ.layer_digest(model, L), shared_digest(blk)
    orig = {e: {kk: w.detach().clone() for kk, w in MU.get_expert_weights(model, L, e).items()} for e in range(E)}
    Wfp = {e: {kk: v.to(GPU, torch.float32) for kk, v in orig[e].items()} for e in range(E)}
    Wq = {q: {e: {kk: cands[q][f"{e}.{kk}"].to(GPU, torch.float32) for kk in AG.PROJ} for e in range(E)}
          for q in (0, 1)}
    cache = load_layer_cache(sp["cache"], L)
    mats = {nm: np.zeros((S, E)) for nm in ("J0", "J1", "U0", "U1", "Uden", "c2den", "tokens", "csum", "c2sum")}
    rec = {n: dict(num=np.zeros(S), den=np.zeros(S), D=np.zeros(S), C=np.zeros(S), tot=np.zeros(S)) for n in configs}
    acc = AG.LayerEvalAccumulator(["GPTQ", "AGQ"], E, model.config.hidden_size) if accumulator_check else None
    keep = []
    t0 = time.time()
    with MU.layer_on_gpu(model, L):
        sh = {kk: getattr(blk.shared_expert, kk).weight.float() for kk in AG.PROJ}
        sg = blk.shared_expert_gate.weight.float()
        for si, sid in enumerate(ids):
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
            T, d = hf.shape
            shared = F.sigmoid(hf @ sg.T) * AG.expert_fp32(hf, sh)
            y = shared.clone()
            dl = {0: {}, 1: {}}
            for e, (tx, slot, c) in rows.items():
                X = hf[tx]
                o = AG.down_input_fp32(X, Wfp[e]) @ Wfp[e]["down_proj"].T        # identical ops to the accumulator
                y.index_add_(0, tx, o * c[:, None])
                cd = c.double()
                on2 = (o.double() ** 2).sum(1)
                mats["Uden"][si, e] = float(on2.sum())
                mats["c2den"][si, e] = float((cd ** 2 * on2).sum())
                mats["tokens"][si, e] = X.shape[0]
                mats["csum"][si, e] = float(cd.sum())
                mats["c2sum"][si, e] = float((cd ** 2).sum())
                for q in (0, 1):
                    dq = o - AG.expert_fp32(X, Wq[q][e])
                    dl[q][e] = dq
                    dn2 = (dq.double() ** 2).sum(1)
                    mats[f"J{q}"][si, e] = float((cd ** 2 * dn2).sum())
                    mats[f"U{q}"][si, e] = float(dn2.sum())
            yn2 = float((y.double() ** 2).sum())
            for n, z in configs.items():
                err = torch.zeros_like(hf)
                V = torch.zeros(T, k, d, device=GPU, dtype=torch.float32)
                for e, (tx, slot, c) in rows.items():
                    de = dl[int(z[e])][e]
                    err.index_add_(0, tx, de * c[:, None])
                    V[tx, slot] = de * c[:, None]
                dec = AG.decompose(V)
                R = rec[n]
                R["num"][si] = float((err.double() ** 2).sum()); R["den"][si] = yn2
                R["D"][si], R["C"][si], R["tot"][si] = dec["diag"], dec["cross"], dec["total"]
                del V, err
            if acc is not None:
                acc.add_sequence(hf, vals, sel, Wfp, {"GPTQ": Wq[0], "AGQ": Wq[1]}, shared)
            if native:
                keep.append((h, cap["lg"], cap["y"].reshape(h.shape)))
            del dl
        t_lean = time.time() - t0
        # ---- native BF16 block forward with each configuration installed (router must be unchanged)
        nat, install_ok = {}, True
        t1 = time.time()
        if native:
            for n, z in configs.items():
                for e in range(E):
                    MU.set_expert_weights(model, L, e, {kk: cands[int(z[e])][f"{e}.{kk}"] for kk in AG.PROJ})
                inst = MU.get_expert_weights
                for e in range(E):                                    # all 3 projections from the same candidate
                    w = inst(model, L, e)
                    install_ok &= all(torch.equal(w[kk].cpu(), cands[int(z[e])][f"{e}.{kk}"]) for kk in AG.PROJ)
                install_ok &= shared_digest(blk) == sdig0
                num = den = 0.0
                for h, lg0, y0 in keep:
                    yq, lg = blk(h.unsqueeze(0))
                    assert torch.equal(lg, lg0), "router output changed under quantization"
                    num += float(((yq.reshape(h.shape).double() - y0.double()) ** 2).sum())
                    den += float((y0.double() ** 2).sum())
                nat[n] = num / den
            for e in range(E):
                MU.set_expert_weights(model, L, e, orig[e])
        t_native = time.time() - t1
    restored = PQ.layer_digest(model, L) == dig0 and shared_digest(blk) == sdig0
    assert restored, "pristine weights not restored"
    out = {}
    for n, z in configs.items():
        R = rec[n]
        num, den = float(R["num"].sum()), float(R["den"].sum())
        out[n] = dict(z=[int(v) for v in z], n_agq=int(np.sum(z)), num=num, den=den,
                      moe_layer_output_rel_err=num / (den + 1e-30),
                      D=float(R["D"].sum()), C=float(R["C"].sum()), decomp_total=float(R["tot"].sum()),
                      D_rel=float(R["D"].sum()) / (den + 1e-30), C_rel=float(R["C"].sum()) / (den + 1e-30),
                      cross_fraction=float(R["C"].sum() / R["tot"].sum()),
                      native_rel_err=nat.get(n),
                      per_seq=dict(num=R["num"].tolist(), den=R["den"].tolist(), D=R["D"].tolist(),
                                   C=R["C"].tolist()))
    chk = {}
    if acc is not None:
        s = acc.summary()
        for n, q in (("GPTQ", 0), ("AGQ", 1)):
            if n in out:
                chk[n] = dict(accumulator=s[n]["moe_layer_output_rel_err"], lean=out[n]["moe_layer_output_rel_err"],
                              equal=s[n]["moe_layer_output_rel_err"] == out[n]["moe_layer_output_rel_err"],
                              abs_diff=abs(s[n]["moe_layer_output_rel_err"] - out[n]["moe_layer_output_rel_err"]))
    meta = dict(split=split, layer=L, n_seqs=S, tokens=S * 512, lean_s=t_lean, native_s=t_native,
                pristine_restored=bool(restored), install_all_three_from_same_candidate_and_shared_untouched=
                bool(install_ok), accumulator_check=chk,
                decomposition_identity_max_rel=max(abs(out[n]["D"] + out[n]["C"] - out[n]["decomp_total"]) /
                                                   out[n]["decomp_total"] for n in out),
                decomposition_vs_err_max_rel=max(abs(out[n]["decomp_total"] - out[n]["num"]) / out[n]["num"]
                                                 for n in out))
    del Wq, Wfp, keep, cache
    torch.cuda.empty_cache()
    return out, mats, meta


def save_split(split, L, out, mats, meta):
    os.makedirs(p("results", split), exist_ok=True)
    np.savez(p("results", split, f"expert_mats_L{L}.npz"), **mats)
    jdump(dict(configs=out, meta=meta), p("results", split, f"L{L}.json"))


# ============================================================================ stage: regression
def stage_regression(a):
    """Before any selection: (a) weight-level checks; (b) uniform GPTQ / AGQ layer-output error on the old test split
    reproduces the historical AGQ-only validation values (same windows, same definition) and the validated
    LayerEvalAccumulator; native BF16 reproduces the historical native values."""
    from moe_calib import model_utils as MU
    assert os.path.exists(p("protocol.json"))
    inv = jload("artifact_inventory.json")
    ev = json.load(open(os.path.join(ROOT, "runs_agq_only_validation", "results", "eval_FR_s0.json")))
    model, t_load = load_partial()
    res = dict(layers={}, model_load_s=t_load)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for L in LAYERS:
        cands = load_candidates(L)
        wchk = dict(
            gptq_file_sha_match=inv["candidates"]["GPTQ"]["layers"][str(L)]["file_sha256_matches_record"],
            agq_file_sha_match=inv["candidates"]["AGQ"]["layers"][str(L)]["file_sha256_matches_record"],
            expert_hashes_match_audit=all(
                expert_hash(cands[q], e) == inv["candidates"][HY.CANDIDATES[q]]["layers"][str(L)]["expert_hashes"][e]
                for q in (0, 1) for e in range(E)),
            n_experts=[len({k.split(".")[0] for k in cands[q]}) for q in (0, 1)],
            candidates_differ_every_expert=all(any(not torch.equal(cands[0][f"{e}.{kk}"], cands[1][f"{e}.{kk}"])
                                                   for kk in AG.PROJ) for e in range(E)),
            candidates_differ_from_bf16=all(not torch.equal(cands[q][f"{e}.{kk}"],
                                                            MU.get_expert_weights(model, L, e)[kk].detach().cpu())
                                            for q in (0, 1) for e in range(E) for kk in AG.PROJ),
            shapes_match_model=all(cands[q][f"{e}.{kk}"].shape == MU.get_expert_weights(model, L, e)[kk].shape
                                   for q in (0, 1) for e in range(E) for kk in AG.PROJ))
        out, mats, meta = evaluate_layer(model, L, "oldtest", {"GPTQ": np.zeros(E), "AGQ": np.ones(E)}, cands,
                                         native=True, accumulator_check=True)
        lay = {}
        for n in ("GPTQ", "AGQ"):
            h = ev["layers"][str(L)][n]
            lay[n] = dict(lean=out[n]["moe_layer_output_rel_err"], historical=h["moe_layer_output_rel_err"],
                          exact=out[n]["moe_layer_output_rel_err"] == h["moe_layer_output_rel_err"],
                          rel_diff=abs(out[n]["moe_layer_output_rel_err"] / h["moe_layer_output_rel_err"] - 1),
                          native=out[n]["native_rel_err"], native_historical=h["moe_layer_output_native_rel_err"],
                          native_rel_diff=abs(out[n]["native_rel_err"] / h["moe_layer_output_native_rel_err"] - 1),
                          per_seq_exact=[a_ / b_ for a_, b_ in zip(out[n]["per_seq"]["num"], out[n]["per_seq"]["den"])]
                          == h["per_seq_moe_layer_output_rel_err"])
        unw = {q: float(np.nanmean(np.where(mats["Uden"].sum(0) > 0, mats[f"U{q}"].sum(0) / mats["Uden"].sum(0),
                                            np.nan))) for q in (0, 1)}
        hist_unw = {q: float(np.nanmean(ev["layers"][str(L)][HY.CANDIDATES[q]]["expert_unweighted"])) for q in (0, 1)}
        lay["expert_unweighted_full_output_mean60"] = dict(
            GPTQ=unw[0], AGQ=unw[1], GPTQ_hist=hist_unw[0], AGQ_hist=hist_unw[1],
            max_rel_diff=max(abs(unw[q] / hist_unw[q] - 1) for q in (0, 1)))
        lay["weights"] = wchk
        lay["meta"] = meta
        lay["passed"] = bool(all(v for k_, v in wchk.items() if isinstance(v, bool)) and wchk["n_experts"] == [E, E]
                             and all(lay[n]["rel_diff"] <= 1e-9 for n in ("GPTQ", "AGQ"))
                             and all(lay[n]["native_rel_diff"] <= 1e-6 for n in ("GPTQ", "AGQ"))
                             and all(c["abs_diff"] <= 1e-12 for c in meta["accumulator_check"].values())
                             and lay["expert_unweighted_full_output_mean60"]["max_rel_diff"] <= 1e-9
                             and meta["pristine_restored"]
                             and meta["install_all_three_from_same_candidate_and_shared_untouched"])
        res["layers"][str(L)] = lay
        print(f"[regression] L{L}: GPTQ {lay['GPTQ']['lean']:.6e} (hist {lay['GPTQ']['historical']:.6e}, exact "
              f"{lay['GPTQ']['exact']}) AGQ {lay['AGQ']['lean']:.6e} (hist {lay['AGQ']['historical']:.6e}, exact "
              f"{lay['AGQ']['exact']}) passed {lay['passed']}", flush=True)
        del cands
    res["seconds"] = time.time() - t0
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    res["peak_cpu_gb"] = peak_cpu_gb()
    res["historical_E3_reference"] = {nm: {str(L): inv["candidates"][nm]["layers"][str(L)]["historical_E3_mean_of_60"]
                                           for L in LAYERS} for nm in ("GPTQ", "AGQ")}
    res["note_E3"] = ("the historical E3 standard reconstruction (projection-level, first 128 dispatch rows/seq) is "
                      "reproduced via the frozen weight-file hashes (identical weights => identical E3); the "
                      "full-expert unweighted output error is recomputed here and must equal the historical eval")
    res["passed"] = all(v["passed"] for v in res["layers"].values())
    jdump(res, p("candidate_regression.json"))
    print(f"[regression] passed={res['passed']}")
    if not res["passed"]:
        raise SystemExit("candidate regression FAILED -> HYBRID-BLOCKED")


# ============================================================================ stage: score_tune
def stage_score_tune(a):
    assert jload("candidate_regression.json")["passed"]
    model, t_load = load_partial()
    torch.cuda.reset_peak_memory_stats()
    cost = {}
    for L in LAYERS:
        if os.path.exists(p("results", "tune", f"L{L}.json")):
            print(f"[score_tune] L{L} done"); continue
        t0 = time.time()
        cands = load_candidates(L)
        t_load_c = time.time() - t0
        out, mats, meta = evaluate_layer(model, L, "tune", {"GPTQ": np.zeros(E), "AGQ": np.ones(E)}, cands,
                                         native=True)
        meta.update(candidate_load_s=t_load_c, total_s=time.time() - t0)
        save_split("tune", L, out, mats, meta)
        cost[str(L)] = dict(seconds=time.time() - t0, candidate_load_s=t_load_c, lean_s=meta["lean_s"],
                            native_s=meta["native_s"])
        print(f"[score_tune] L{L}: GPTQ {out['GPTQ']['moe_layer_output_rel_err']:.6e} AGQ "
              f"{out['AGQ']['moe_layer_output_rel_err']:.6e} ({time.time() - t0:.0f} s)", flush=True)
    cost.update(peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=peak_cpu_gb(), model_load_s=t_load)
    jdump(cost, p("logs", "cost_score_tune.json"))


# ============================================================================ stage: select (freeze)
def expert_meta():
    """Calibration exposure / affinity statistics (FR s0) and natural exposure deciles."""
    st = json.load(open(os.path.join(ROOT, "runs_agq_only_validation", "affinity_statistics.json")))
    per = {(r["layer"], r["expert"]): r for r in st["configs"][CONFIG]["per_expert"]}
    return per


def stage_select(a):
    if os.path.exists(p("selection_manifest.json")):
        print("[select] manifest already frozen; not rewritten"); return
    for L in LAYERS:
        assert not os.path.exists(p("results", "confirm", f"L{L}.json")), "confirm results exist before selection"
        assert not os.path.exists(p("results", "oldtest", f"L{L}.json")), "oldtest results exist before selection"
    inv = jload("artifact_inventory.json")
    meta = expert_meta()
    t0 = time.time()
    scores, man, lw, stab = [], dict(layers={}), dict(layers={}), dict(layers={})
    for L in LAYERS:
        r = json.load(open(p("results", "tune", f"L{L}.json")))
        M = np.load(p("results", "tune", f"expert_mats_L{L}.npz"))
        J0, J1 = M["J0"].sum(0), M["J1"].sum(0)
        z = HY.select_experts(J0, J1)
        ri = HY.rel_improvement(J0, J1)
        hashes = inv["candidates"]
        ents = []
        tok, cs, c2 = M["tokens"].sum(0), M["csum"].sum(0), M["c2sum"].sum(0)
        for e in range(E):
            m = meta[(L, e)]
            vmean = cs[e] / tok[e] if tok[e] else float("nan")
            vcv = float(np.sqrt(max(c2[e] / tok[e] - vmean ** 2, 0)) / vmean) if tok[e] else float("nan")
            ent = dict(layer=L, expert=e, selected=HY.CANDIDATES[int(z[e])], z=int(z[e]),
                       val_J0=float(J0[e]), val_J1=float(J1[e]), val_rel_improvement_agq=float(ri[e]),
                       val_rel_diff_agq_minus_gptq=float(-ri[e]),
                       val_c2_weighted_rel_err_q0=float(J0[e] / M["c2den"].sum(0)[e]),
                       val_c2_weighted_rel_err_q1=float(J1[e] / M["c2den"].sum(0)[e]),
                       val_unweighted_rel_err_q0=float(M["U0"].sum(0)[e] / M["Uden"].sum(0)[e]),
                       val_unweighted_rel_err_q1=float(M["U1"].sum(0)[e] / M["Uden"].sum(0)[e]),
                       val_tokens=int(tok[e]), val_affinity_mean=float(vmean), val_affinity_cv=vcv,
                       calib_tokens=int(m["n"]), calib_affinity_mean=m["mean"], calib_affinity_cv=m["cv"],
                       natural_p=m["natural_p"], natural_decile=m["natural_decile"],
                       candidate_hash_gptq=hashes["GPTQ"]["layers"][str(L)]["expert_hashes"][e],
                       candidate_hash_agq=hashes["AGQ"]["layers"][str(L)]["expert_hashes"][e])
            ents.append(ent)
        scores += ents
        ch = [{0: x["candidate_hash_gptq"], 1: x["candidate_hash_agq"]} for x in ents]
        man["layers"][str(L)] = dict(
            z=[int(v) for v in z], n_agq=int(z.sum()), n_gptq=int(E - z.sum()),
            installed_config_hash=HY.config_hash(z, ch),
            experts=[{k_: x[k_] for k_ in ("layer", "expert", "selected", "z", "val_J0", "val_J1",
                                           "val_rel_diff_agq_minus_gptq", "candidate_hash_gptq",
                                           "candidate_hash_agq")} for x in ents])
        eg, ea = r["configs"]["GPTQ"]["moe_layer_output_rel_err"], r["configs"]["AGQ"]["moe_layer_output_rel_err"]
        choice = HY.layerwise_choice(eg, ea)
        lw["layers"][str(L)] = dict(val_E_layer_gptq=eg, val_E_layer_agq=ea, rel_agq_vs_gptq=HY.rel(ea, eg),
                                    choice=HY.CANDIDATES[choice], z_all=choice,
                                    installed_config_hash=HY.config_hash(np.full(E, choice), ch),
                                    val_native_gptq=r["configs"]["GPTQ"]["native_rel_err"],
                                    val_native_agq=r["configs"]["AGQ"]["native_rel_err"],
                                    native_choice_agrees=HY.layerwise_choice(r["configs"]["GPTQ"]["native_rel_err"],
                                                                             r["configs"]["AGQ"]["native_rel_err"])
                                    == choice)
        stab["layers"][str(L)] = HY.half_split_stability(M["J0"], M["J1"])
        print(f"[select] L{L}: AGQ chosen for {int(z.sum())}/60 experts; layer-wise -> {HY.CANDIDATES[choice]} "
              f"(val E_layer GPTQ {eg:.6e} AGQ {ea:.6e}); half-split agreement "
              f"{stab['layers'][str(L)]['agreement_halves']:.2f}", flush=True)
    sel_s = time.time() - t0
    Mall = [np.load(p("results", "tune", f"expert_mats_L{L}.npz")) for L in LAYERS]
    stab["pooled"] = HY.half_split_stability(np.concatenate([m["J0"] for m in Mall], 1),
                                             np.concatenate([m["J1"] for m in Mall], 1))
    stab["note"] = "diagnostic only; the frozen primary rule uses the full tuning split and is NOT modified"
    man.update(rule="z_e = 1 (AGQ) iff J_e^(1) < J_e^(0) on tuning rows 2000-2127; ties -> GPTQ", split="tune",
               split_rows=SPLITS["tune"]["rows"], n_agq_total=int(sum(v["n_agq"] for v in man["layers"].values())),
               candidate_dirs={nm: inv["candidates"][nm]["weights_dir"] for nm in ("GPTQ", "AGQ")},
               frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"), test_results_exist_at_freeze=False,
               protocol_sha256=sha256_file(p("protocol.json")))
    jdump(dict(split="tune", rows=SPLITS["tune"]["rows"], experts=scores), p("validation_expert_scores.json"))
    jdump(man, p("selection_manifest.json"))
    lw.update(rule="per layer, all-AGQ iff tuning-split E_layer(AGQ) < E_layer(GPTQ)", split="tune",
              frozen_at=man["frozen_at"])
    jdump(lw, p("layerwise_uniform_selection.json"))
    jdump(stab, p("selection_stability.json"))
    jdump(dict(manifest_sha256=sha256_file(p("selection_manifest.json")),
               layerwise_sha256=sha256_file(p("layerwise_uniform_selection.json")), selection_s=sel_s,
               frozen_at=man["frozen_at"]), p("logs", "selection_freeze.json"))
    print("[select] manifest frozen")


# ============================================================================ stage: evaluate
def frozen_configs(L, cands):
    fr = jload("logs", "selection_freeze.json")
    assert sha256_file(p("selection_manifest.json")) == fr["manifest_sha256"], "selection manifest modified"
    assert sha256_file(p("layerwise_uniform_selection.json")) == fr["layerwise_sha256"], "layer-wise choice modified"
    man, lw = jload("selection_manifest.json"), jload("layerwise_uniform_selection.json")
    ml = man["layers"][str(L)]
    z = np.array(ml["z"], dtype=np.int64)
    # the hybrid is reconstructed exclusively from the manifest: candidate hashes must match the loaded files
    for x in ml["experts"]:
        e = x["expert"]
        assert x["z"] == z[e]
        assert expert_hash(cands[0], e) == x["candidate_hash_gptq"] and expert_hash(cands[1], e) == x["candidate_hash_agq"]
    ch = [{0: x["candidate_hash_gptq"], 1: x["candidate_hash_agq"]} for x in ml["experts"]]
    assert HY.config_hash(z, ch) == ml["installed_config_hash"]
    lc = lw["layers"][str(L)]["z_all"]
    cfg = {"GPTQ": HY.config_vector("GPTQ", E), "AGQ": HY.config_vector("AGQ", E),
           "LAYERWISE": HY.config_vector("LAYERWISE", E, layer_choice=lc), "HYBRID": HY.config_vector("HYBRID", E, z)}
    return cfg, {n: HY.config_hash(v, ch) for n, v in cfg.items()}


def stage_evaluate(a):
    split = a.split
    assert split in ("confirm", "oldtest")
    assert os.path.exists(p("selection_manifest.json")), "selection must be frozen before evaluation"
    model, t_load = load_partial()
    torch.cuda.reset_peak_memory_stats()
    cost = {}
    for L in LAYERS:
        if os.path.exists(p("results", split, f"L{L}.json")):
            print(f"[evaluate] {split} L{L} done"); continue
        t0 = time.time()
        cands = load_candidates(L)
        cfg, hashes = frozen_configs(L, cands)
        out, mats, meta = evaluate_layer(model, L, split, cfg, cands, native=True,
                                         accumulator_check=(split == "confirm"))
        for n in out:
            out[n]["installed_config_hash"] = hashes[n]
        meta["total_s"] = time.time() - t0
        save_split(split, L, out, mats, meta)
        cost[str(L)] = dict(seconds=time.time() - t0, lean_s=meta["lean_s"], native_s=meta["native_s"])
        print(f"[evaluate] {split} L{L}: " + "  ".join(f"{n} {out[n]['moe_layer_output_rel_err']:.6e}" for n in out)
              + f" ({time.time() - t0:.0f} s)", flush=True)
    cost.update(peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=peak_cpu_gb(), model_load_s=t_load)
    jdump(cost, p("logs", f"cost_evaluate_{split}.json"))


# ============================================================================ stage: tests
def stage_tests(a):
    env = dict(os.environ, MOE_DEVICE="cpu")
    env.pop("MOE_LOAD_LAYERS", None)
    for tag, args in (("hybrid", ["tests/test_expertwise_hybrid.py"]), ("all", ["tests"])):
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", *args], cwd=ROOT, env=env, capture_output=True,
                           text=True)
        open(p("logs", f"pytest_{tag}.log"), "w").write(r.stdout + r.stderr)
        print(f"[tests] {tag}: rc={r.returncode} :: {r.stdout.strip().splitlines()[-1] if r.stdout else ''}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--split", default="confirm")
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    if a.stage == "report":
        import report_expertwise_agq_hybrid as R
        return R.main()
    dict(audit=stage_audit, protocol=stage_protocol, regression=stage_regression, score_tune=stage_score_tune,
         select=stage_select, evaluate=stage_evaluate, tests=stage_tests)[a.stage](a)


if __name__ == "__main__":
    main()
