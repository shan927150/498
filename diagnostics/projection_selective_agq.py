"""Projection-selective AGQ: small-scale feasibility MVP (exploratory 2x2 projection ablation).

Four fixed configurations assembled ONLY from the two existing validated candidate weight sets (FR s0, GPTQ 4-bit g128
percdamp 1.0 act-order, CANONICAL-B1), applied to all 60 routed experts of layers 8 and 16:
    GPTQ (A) | GU_AGQ (B: AGQ gate/up + GPTQ down) | DOWN_AGQ (C: GPTQ gate/up + AGQ down) | FULL_AGQ (D)
Complete expert outputs are computed with the actual mixed weights. One configuration per layer is selected on the tuning
split (~/moe_cache rows 2000-2127) and frozen before the confirmation (primary) and old-test (secondary) evaluations.

Stages (--stage): audit | protocol | regression | validate | select | evaluate --split {confirm, oldtest} | tests | report
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import expertwise_agq_hybrid as X        # noqa: E402  (validated loaders / splits / candidate checks)
import full_depth_pilot as FD            # noqa: E402
import official_ebss_validation as OV    # noqa: E402
from moe_calib import agq as AG          # noqa: E402
from moe_calib import projection_select as PS  # noqa: E402

OUT = "runs_projection_selective_agq"
LAYERS = [8, 16]
E = 60
SPLITS = X.SPLITS
HIST_OLDTEST = {8: dict(GPTQ=0.007173198, FULL_AGQ=0.007177917), 16: dict(GPTQ=0.007090592, FULL_AGQ=0.006871301)}
RULES = dict(tau_rel=0.005, interaction_frac=0.25, expert_mixed_frac=0.10, expert_rel=0.01)
SPLIT_FILE = dict(confirm="confirmation_results.json", oldtest="old_test_results.json")


def p(*a):
    return os.path.join(ROOT, OUT, *a)


jdump, sha256_file = FD.jdump, FD.sha256_file
peak_cpu_gb = X.peak_cpu_gb


def jload(*a):
    return json.load(open(p(*a)))


def code_hashes():
    files = ["diagnostics/projection_selective_agq.py", "moe_calib/projection_select.py", "moe_calib/agq.py",
             "moe_calib/gptq.py", "moe_calib/model_utils.py", "diagnostics/expertwise_agq_hybrid.py",
             "tests/test_projection_selective_agq.py"]
    return {f: sha256_file(os.path.join(ROOT, f)) for f in files if os.path.exists(os.path.join(ROOT, f))}


def nontarget_digest(model, L):
    """sha256 over every parameter of decoder layer L that is NOT a routed-expert weight (attention, norms, router,
    shared expert, shared-expert gate)."""
    h = hashlib.sha256()
    for n, w in sorted(model.model.layers[L].named_parameters()):
        if ".experts." in n:
            continue
        h.update(n.encode())
        h.update(w.detach().contiguous().cpu().view(torch.int16).numpy().tobytes())
    return h.hexdigest()


# ============================================================================ stage: audit
def stage_audit(a):
    hy = os.path.join(ROOT, "runs_expertwise_agq_hybrid")
    prev = json.load(open(os.path.join(hy, "artifact_inventory.json")))
    inv = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), layers=LAYERS, candidates={}, configs={})
    for q, nm in X.HY.CANDIDATES.items():
        rpath, r = X.hist(nm)
        ent = dict(q=q, method=nm, result_json=rpath, weights_dir=r["weights_dir"], gptq=r["gptq"],
                   hessian_weighting=r.get("hessian_weighting", "none"), policy=r.get("policy"),
                   cache_rows_sha256=hashlib.sha256(json.dumps(r["cache_rows"]).encode()).hexdigest(),
                   calibration_manifest_sha256=prev["candidates"][nm]["calibration_manifest_sha256"], layers={})
        for L in LAYERS:
            path = os.path.join(r["weights_dir"], f"L{L}.pt")
            rec = dict(path=path, exists=os.path.exists(path))
            if rec["exists"]:
                f = torch.load(path)
                rec.update(file_sha256=sha256_file(path))
                rec["file_sha256_matches_record"] = rec["file_sha256"] == r["weight_files"][str(L)]["sha256"]
                rec["missing"] = [f"{e}.{k}" for e in range(E) for k in PS.PROJ if f"{e}.{k}" not in f]
                rec["n_experts"] = len({k.split(".")[0] for k in f})
                rec["n_tensors"] = len(f)
                rec["shapes"] = {k: list(f[f"0.{k}"].shape) for k in PS.PROJ}
                rec["shapes_uniform"] = all(list(f[f"{e}.{k}"].shape) == rec["shapes"][k] for e in range(E)
                                            for k in PS.PROJ)
                rec["all_bf16"] = all(t.dtype == torch.bfloat16 for t in f.values())
                rec["all_finite"] = all(bool(torch.isfinite(t).all()) for t in f.values())
                rec["tensor_sha256"] = {f"{e}.{k}": OV.tensor_sha(f[f"{e}.{k}"]) for e in range(E) for k in PS.PROJ}
                del f
            rec["valid"] = bool(rec["exists"] and rec.get("file_sha256_matches_record") and not rec.get("missing")
                                and rec.get("n_experts") == E and rec.get("n_tensors") == 3 * E
                                and rec.get("shapes_uniform") and rec.get("all_bf16") and rec.get("all_finite"))
            ent["layers"][str(L)] = rec
            print(f"[audit] {nm} L{L}: valid {rec['valid']}", flush=True)
        inv["candidates"][nm] = ent
    g, a_ = inv["candidates"]["GPTQ"], inv["candidates"]["AGQ"]
    inv["same_calibration_rows"] = g["cache_rows_sha256"] == a_["cache_rows_sha256"]
    inv["same_calibration_manifest"] = g["calibration_manifest_sha256"] == a_["calibration_manifest_sha256"]
    inv["same_gptq_config"] = g["gptq"] == a_["gptq"]
    inv["all_valid"] = bool(all(c["layers"][str(L)]["valid"] for c in inv["candidates"].values() for L in LAYERS)
                            and inv["same_calibration_rows"] and inv["same_calibration_manifest"]
                            and inv["same_gptq_config"])
    jdump(inv, p("artifact_inventory.json"))
    # deterministic configuration manifest (every installed projection -> source candidate + tensor hash)
    cm = dict(configs={c: PS.CONFIGS[c] for c in PS.ORDER}, layers={})
    for L in LAYERS:
        cm["layers"][str(L)] = {}
        for c in PS.ORDER:
            fake = {q: {k: None for k in inv["candidates"][PS.SOURCE[q]]["layers"][str(L)]["tensor_sha256"]}
                    for q in (0, 1)}
            _, man = PS.build_projection_config(fake, c, E)
            for r in man:
                r["tensor_sha256"] = inv["candidates"][r["source"]]["layers"][str(L)]["tensor_sha256"][
                    f"{r['expert']}.{r['projection']}"]
            cm["layers"][str(L)][c] = dict(manifest_sha256=PS.manifest_hash(man), n_projections=len(man),
                                           n_from_gptq=sum(r["q"] == 0 for r in man),
                                           n_from_agq=sum(r["q"] == 1 for r in man), projections=man)
    jdump(cm, p("configuration_manifest.json"))
    print(f"[audit] all valid: {inv['all_valid']}")


# ============================================================================ stage: protocol
def stage_protocol(a):
    if os.path.exists(p("protocol.json")):
        print("[protocol] already frozen"); return
    inv = jload("artifact_inventory.json")
    assert inv["all_valid"]
    prot = dict(
        experiment="Projection-selective AGQ: small-scale feasibility MVP (2x2 GU x down ablation)",
        nature="retrospective exploratory pilot; one calibration seed (FR s0); layers 8/16 chosen with knowledge of "
               "earlier AGQ outcomes; both test splits previously inspected",
        model="Qwen/Qwen1.5-MoE-A2.7B (BF16)", layers=LAYERS,
        candidates={nm: dict(weights_dir=inv["candidates"][nm]["weights_dir"],
                             file_sha256={L: inv["candidates"][nm]["layers"][str(L)]["file_sha256"] for L in LAYERS})
                    for nm in ("GPTQ", "AGQ")},
        gptq=dict(bits=4, group_size=128, percdamp=1.0, act_order=True, new_solves=False),
        configurations=PS.CONFIGS, gate_up_paired=True,
        configuration_manifest_sha256=sha256_file(p("configuration_manifest.json")),
        splits=SPLITS,
        metric="E_layer = sum ||y_BF16 - y_q||^2 / (sum ||y_BF16||^2 + 1e-30); complete expert outputs with the "
               "actual mixed weights in fp32; frozen BF16 inputs, fp32 softmax top-4 c (no renorm), shared expert "
               "included; LayerEvalAccumulator definition",
        aggregate="mean of the L8 and L16 ratios (2-layer scope, not comparable to 4-layer historical means)",
        selection=dict(rule="per layer z* = argmin over {GPTQ, GU_AGQ, DOWN_AGQ, FULL_AGQ} of tuning-split E_layer; "
                            "exact ties -> GPTQ > FULL_AGQ > GU_AGQ > DOWN_AGQ", split="tune",
                       layerwise_uniform_baseline="per layer GPTQ vs FULL_AGQ on tuning E_layer; ties -> GPTQ"),
        regression=dict(split="oldtest", configs=["GPTQ", "FULL_AGQ"], reference=HIST_OLDTEST,
                        reference_full_precision="runs_agq_only_validation/results/eval_FR_s0.json (per window)",
                        gate="A and D must reproduce historical per-window values and totals (rel <= 1e-9) before "
                             "B/C are evaluated anywhere"),
        decision_rule=dict(
            rules=RULES,
            PROMISING="all: (1) a mixed config selected on validation for some layer beats both uniform configs there "
                      "on validation AND on confirmation; (2) 2-layer mean selected vs validation-selected layer-wise "
                      "uniform <= -tau (confirm); (3) per layer selected vs best uniform (confirm) <= +tau; (4) old-test "
                      "selected vs layer-wise 2-layer mean < 0; (5) correctness; (6) cost: 2-layer mean selected vs "
                      "uniform FULL_AGQ (1x preparation cost) <= -tau",
            SELECTION_FAIL="not promising; a mixed config was selected because it beat both uniform configs on "
                           "validation, but on confirmation it does not beat the better uniform config of that layer",
            LOCAL_GAIN="not the above; on confirmation some mixed config beats the best uniform config of some layer by "
                       ">= tau",
            INTERACTION_ONLY="not the above; on a layer where |FULL_AGQ vs GPTQ| >= tau, |I| / |E_A - E_D| >= 0.25 on "
                             "confirmation, OR in some layer >= 10 % of experts have a mixed config whose confirmation "
                             "c^2-weighted expert error is lower than both uniform configs by >= 1 % relative",
            NO_GAIN="otherwise", BLOCKED="candidate invalid or any correctness check fails",
            recommendation=PS.RECOMMEND, decision_split="confirm (primary)", implementation="moe_calib.projection_select"),
        post_hoc="L16 expert 40 attribution is post-hoc mechanism exploration (identified from earlier results); "
                 "never used for selection or decision",
        hard_constraints=["no new quantization / Hessians", "no new data or windows", "no interpolation",
                          "no gate-only / up-only configs", "no expert-wise selection", "historical artifacts read-only"],
        code_sha256=code_hashes(), frozen_before_any_BC_result=True, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(prot, p("protocol.json"))
    print("[protocol] frozen")


# ============================================================================ evaluator
@torch.no_grad()
def evaluate_layer(model, L, split, config_names, cands, native=True, accumulator_check=False):
    """All requested projection configurations on one layer / split with frozen BF16 routing. Expert outputs are the
    complete gated MLP with each configuration's actual mixed weights."""
    import torch.nn.functional as F
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    GPU = MU.GPU
    ids = list(range(*SPLITS[split]["rows"]))
    S = len(ids)
    blk = MU.moe_block(model, L)
    k = MU.top_k(model)
    dig0, nt0 = PQ.layer_digest(model, L), nontarget_digest(model, L)
    orig = {e: {kk: w.detach().clone() for kk, w in MU.get_expert_weights(model, L, e).items()} for e in range(E)}
    Wfp = {e: {kk: v.to(GPU, torch.float32) for kk, v in orig[e].items()} for e in range(E)}
    Wg = {q: {key: t.to(GPU, torch.float32) for key, t in cands[q].items()} for q in (0, 1)}
    Wc = {c: PS.build_projection_config(Wg, c, E)[0] for c in config_names}      # references, no copies
    cache = load_layer_cache(SPLITS[split]["cache"], L)
    mats = {f"J_{c}": np.zeros((S, E)) for c in config_names}
    mats.update({f"U_{c}": np.zeros((S, E)) for c in config_names})
    mats.update({nm: np.zeros((S, E)) for nm in ("Uden", "c2den", "tokens", "csum", "c2sum")})
    rec = {c: dict(num=np.zeros(S), den=np.zeros(S), D=np.zeros(S), C=np.zeros(S), tot=np.zeros(S))
           for c in config_names}
    acc = AG.LayerEvalAccumulator(list(config_names), E, model.config.hidden_size) if accumulator_check else None
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
            err = {c: torch.zeros_like(hf) for c in config_names}
            V = {c: torch.zeros(T, k, d, device=GPU, dtype=torch.float32) for c in config_names}
            for e, (tx, slot, cc) in rows.items():
                Xe = hf[tx]
                o = AG.down_input_fp32(Xe, Wfp[e]) @ Wfp[e]["down_proj"].T          # accumulator's exact ops
                y.index_add_(0, tx, o * cc[:, None])
                cd = cc.double()
                on2 = (o.double() ** 2).sum(1)
                mats["Uden"][si, e] = float(on2.sum()); mats["c2den"][si, e] = float((cd ** 2 * on2).sum())
                mats["tokens"][si, e] = Xe.shape[0]; mats["csum"][si, e] = float(cd.sum())
                mats["c2sum"][si, e] = float((cd ** 2).sum())
                for c in config_names:
                    dq = o - AG.expert_fp32(Xe, Wc[c][e])          # complete expert output, actual mixed weights
                    err[c].index_add_(0, tx, dq * cc[:, None])
                    V[c][tx, slot] = dq * cc[:, None]
                    dn2 = (dq.double() ** 2).sum(1)
                    mats[f"J_{c}"][si, e] = float((cd ** 2 * dn2).sum())
                    mats[f"U_{c}"][si, e] = float(dn2.sum())
            yn2 = float((y.double() ** 2).sum())
            for c in config_names:
                dec = AG.decompose(V[c])
                R = rec[c]
                R["num"][si] = float((err[c].double() ** 2).sum()); R["den"][si] = yn2
                R["D"][si], R["C"][si], R["tot"][si] = dec["diag"], dec["cross"], dec["total"]
            del V, err
            if acc is not None:
                acc.add_sequence(hf, vals, sel, Wfp, Wc, shared)
            if native:
                keep.append((h, cap["lg"], cap["y"].reshape(h.shape)))
        t_lean = time.time() - t0
        nat, install_ok, reinstall_ok, nt_ok = {}, True, None, True
        t1 = time.time()
        if native:
            first = {}
            for c in config_names:
                Wi, _ = PS.build_projection_config(cands, c, E)
                for e in range(E):
                    MU.set_expert_weights(model, L, e, Wi[e])
                for e in range(E):
                    w = MU.get_expert_weights(model, L, e)
                    install_ok &= all(torch.equal(w[kk].cpu(), Wi[e][kk]) for kk in PS.PROJ)
                nt_ok &= nontarget_digest(model, L) == nt0
                num = den = 0.0
                for wi, (h, lg0, y0) in enumerate(keep):
                    yq, lg = blk(h.unsqueeze(0))
                    assert torch.equal(lg, lg0), "router output changed under quantization"
                    if wi < 2:
                        first.setdefault(c, []).append(yq.clone())
                    num += float(((yq.reshape(h.shape).double() - y0.double()) ** 2).sum())
                    den += float((y0.double() ** 2).sum())
                nat[c] = num / den
            # reinstall the first configuration after the others: outputs must be bitwise identical
            c0 = config_names[0]
            Wi, _ = PS.build_projection_config(cands, c0, E)
            for e in range(E):
                MU.set_expert_weights(model, L, e, Wi[e])
            reinstall_ok = all(torch.equal(blk(keep[wi][0].unsqueeze(0))[0], first[c0][wi]) for wi in range(2))
            for e in range(E):
                MU.set_expert_weights(model, L, e, orig[e])
        t_native = time.time() - t1
    restored = PQ.layer_digest(model, L) == dig0 and nontarget_digest(model, L) == nt0
    assert restored, "pristine weights not restored"
    out = {}
    for c in config_names:
        R = rec[c]
        num, den = float(R["num"].sum()), float(R["den"].sum())
        out[c] = dict(num=num, den=den, moe_layer_output_rel_err=num / (den + 1e-30), D=float(R["D"].sum()),
                      C=float(R["C"].sum()), decomp_total=float(R["tot"].sum()), D_rel=float(R["D"].sum()) / den,
                      C_rel=float(R["C"].sum()) / den, cross_fraction=float(R["C"].sum() / R["tot"].sum()),
                      native_rel_err=nat.get(c),
                      per_seq=dict(num=R["num"].tolist(), den=R["den"].tolist(), D=R["D"].tolist(),
                                   C=R["C"].tolist()))
    chk = {}
    if acc is not None:
        s = acc.summary()
        for c in config_names:
            chk[c] = dict(accumulator=s[c]["moe_layer_output_rel_err"], lean=out[c]["moe_layer_output_rel_err"],
                          abs_diff=abs(s[c]["moe_layer_output_rel_err"] - out[c]["moe_layer_output_rel_err"]),
                          per_seq_equal=s[c]["per_seq_moe_layer_output_rel_err"] ==
                          [a_ / (b_ + 1e-30) for a_, b_ in zip(out[c]["per_seq"]["num"], out[c]["per_seq"]["den"])])
    meta = dict(split=split, layer=L, n_seqs=S, lean_s=t_lean, native_s=t_native, pristine_restored=bool(restored),
                install_matches_config=bool(install_ok), nontarget_unchanged_during_install=bool(nt_ok),
                reinstall_bitwise_identical=reinstall_ok, accumulator_check=chk,
                decomposition_identity_max_rel=max(abs(out[c]["D"] + out[c]["C"] - out[c]["decomp_total"]) /
                                                   out[c]["decomp_total"] for c in out),
                decomposition_vs_err_max_rel=max(abs(out[c]["decomp_total"] - out[c]["num"]) / out[c]["num"]
                                                 for c in out))
    del Wg, Wc, Wfp, keep, cache
    torch.cuda.empty_cache()
    return out, mats, meta


def save_split(split, L, out, mats, meta):
    os.makedirs(p("results", split), exist_ok=True)
    np.savez(p("results", split, f"expert_mats_L{L}.npz"), **mats)
    jdump(dict(configs=out, meta=meta), p("results", split, f"L{L}.json"))


def _run(split, configs, accumulator_check):
    model, t_load = X.load_partial()
    torch.cuda.reset_peak_memory_stats()
    cost = dict(model_load_s=t_load, layers={})
    for L in LAYERS:
        if os.path.exists(p("results", split, f"L{L}.json")):
            print(f"[{split}] L{L} done"); continue
        t0 = time.time()
        cands = X.load_candidates(L)
        tl = time.time() - t0
        out, mats, meta = evaluate_layer(model, L, split, configs, cands, native=True,
                                         accumulator_check=accumulator_check)
        meta.update(candidate_load_s=tl, total_s=time.time() - t0)
        save_split(split, L, out, mats, meta)
        cost["layers"][str(L)] = dict(seconds=time.time() - t0, candidate_load_s=tl, lean_s=meta["lean_s"],
                                      native_s=meta["native_s"])
        print(f"[{split}] L{L}: " + "  ".join(f"{c} {out[c]['moe_layer_output_rel_err']:.6e}" for c in out)
              + f" ({time.time() - t0:.0f} s)", flush=True)
    cost.update(peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=peak_cpu_gb())
    jdump(cost, p("logs", f"cost_{split}.json"))


# ============================================================================ stage: regression (A and D only)
def stage_regression(a):
    assert os.path.exists(p("protocol.json"))
    ev = json.load(open(os.path.join(ROOT, "runs_agq_only_validation", "results", "eval_FR_s0.json")))
    os.makedirs(p("logs"), exist_ok=True)
    tmp = "regression_oldtest"
    SPLITS[tmp] = SPLITS["oldtest"]
    try:
        _run(tmp, ("GPTQ", "FULL_AGQ"), accumulator_check=True)
    finally:
        SPLITS.pop(tmp)
    res = dict(layers={}, note="old-test split, configurations A and D only (B/C are not evaluated before this gate)")
    for L in LAYERS:
        r = json.load(open(p("results", tmp, f"L{L}.json")))
        lay = {}
        for c, hn in (("GPTQ", "GPTQ"), ("FULL_AGQ", "AGQ")):
            h = ev["layers"][str(L)][hn]
            o = r["configs"][c]
            lay[c] = dict(value=o["moe_layer_output_rel_err"], historical=h["moe_layer_output_rel_err"],
                          reference_rounded=HIST_OLDTEST[L][c],
                          rel_diff=abs(o["moe_layer_output_rel_err"] / h["moe_layer_output_rel_err"] - 1),
                          per_window_exact=[x / y for x, y in zip(o["per_seq"]["num"], o["per_seq"]["den"])] ==
                          h["per_seq_moe_layer_output_rel_err"],
                          native=o["native_rel_err"], native_historical=h["moe_layer_output_native_rel_err"],
                          native_exact=o["native_rel_err"] == h["moe_layer_output_native_rel_err"])
        lay["meta"] = r["meta"]
        lay["passed"] = bool(all(lay[c]["rel_diff"] <= 1e-9 and lay[c]["per_window_exact"] and lay[c]["native_exact"]
                                 for c in ("GPTQ", "FULL_AGQ"))
                             and all(v["abs_diff"] <= 1e-12 and v["per_seq_equal"]
                                     for v in r["meta"]["accumulator_check"].values())
                             and r["meta"]["pristine_restored"] and r["meta"]["install_matches_config"]
                             and r["meta"]["nontarget_unchanged_during_install"] and r["meta"]["reinstall_bitwise_identical"])
        res["layers"][str(L)] = lay
        print(f"[regression] L{L}: A {lay['GPTQ']['value']:.9f} (hist {lay['GPTQ']['historical']:.9f}) D "
              f"{lay['FULL_AGQ']['value']:.9f} (hist {lay['FULL_AGQ']['historical']:.9f}) passed {lay['passed']}")
    res["inputs_routing_precision_aggregation_match"] = (
        "same cache (~/moe_cache rows 2128-2255), same BF16 layer forward for inputs/router, fp32 expert math, float64 "
        "sums, sum-over-windows ratio, shared expert included: identical to runs_agq_only_validation stage_evaluate")
    res["cost"] = jload("logs", f"cost_{tmp}.json")
    res["passed"] = all(v["passed"] for v in res["layers"].values())
    jdump(res, p("candidate_regression.json"))
    if not res["passed"]:
        raise SystemExit("A/D regression FAILED -> PROJECTION-AGQ-BLOCKED")


# ============================================================================ stage: validate
def stage_validate(a):
    assert jload("candidate_regression.json")["passed"]
    _run("tune", PS.ORDER, accumulator_check=True)
    out = dict(split="tune", rows=SPLITS["tune"]["rows"], layers={})
    for L in LAYERS:
        r = json.load(open(p("results", "tune", f"L{L}.json")))
        Ev = {c: r["configs"][c]["moe_layer_output_rel_err"] for c in PS.ORDER}
        bu = min(Ev["GPTQ"], Ev["FULL_AGQ"])
        out["layers"][str(L)] = dict(
            E_layer=Ev, native={c: r["configs"][c]["native_rel_err"] for c in PS.ORDER},
            rel_vs_gptq={c: PS.rel(Ev[c], Ev["GPTQ"]) for c in PS.ORDER},
            rel_vs_full_agq={c: PS.rel(Ev[c], Ev["FULL_AGQ"]) for c in PS.ORDER},
            rel_vs_best_uniform={c: PS.rel(Ev[c], bu) for c in PS.ORDER},
            frac_windows_improved_vs_best_uniform={
                c: float(np.mean(np.array(r["configs"][c]["per_seq"]["num"]) <
                                 np.array(r["configs"]["GPTQ" if Ev["GPTQ"] <= Ev["FULL_AGQ"] else "FULL_AGQ"]
                                          ["per_seq"]["num"]))) for c in PS.ORDER},
            lowest=min(PS.ORDER, key=lambda c: Ev[c]), meta=r["meta"])
    jdump(out, p("validation_results.json"))


# ============================================================================ stage: select (freeze)
def stage_select(a):
    if os.path.exists(p("selection_manifest.json")):
        print("[select] already frozen"); return
    for s in ("confirm", "oldtest"):
        for L in LAYERS:
            assert not os.path.exists(p("results", s, f"L{L}.json")), "test results exist before selection"
    t0 = time.time()
    val = jload("validation_results.json")
    cm = jload("configuration_manifest.json")
    man, lw = dict(layers={}), dict(layers={})
    for L in LAYERS:
        Ev = val["layers"][str(L)]["E_layer"]
        z, u = PS.select_config(Ev), PS.layerwise_uniform(Ev)
        bu = min(Ev["GPTQ"], Ev["FULL_AGQ"])
        man["layers"][str(L)] = dict(layer=L, validation_E_layer=Ev, selected=z,
                                     rel_margin_vs_best_uniform=PS.rel(Ev[z], bu),
                                     rel_margin_runner_up=PS.rel(sorted(Ev.values())[1], Ev[z]),
                                     selected_config_manifest_sha256=cm["layers"][str(L)][z]["manifest_sha256"],
                                     source_map=PS.CONFIGS[z])
        lw["layers"][str(L)] = dict(choice=u, validation_E_layer={c: Ev[c] for c in PS.UNIFORM},
                                    rel_agq_vs_gptq=PS.rel(Ev["FULL_AGQ"], Ev["GPTQ"]),
                                    config_manifest_sha256=cm["layers"][str(L)][u]["manifest_sha256"])
        print(f"[select] L{L}: selected {z} (margin vs best uniform {PS.rel(Ev[z], bu) * 100:+.3f} %); layer-wise "
              f"uniform {u}", flush=True)
    inv = jload("artifact_inventory.json")
    man.update(rule=jload("protocol.json")["selection"]["rule"], tie_order=list(PS.TIE_ORDER), split="tune",
               validation_results_sha256=sha256_file(p("validation_results.json")),
               validation_cache_files_sha256={str(L): sha256_file(os.path.join(SPLITS["tune"]["cache"], f"L{L}.pt"))
                                              for L in LAYERS},
               candidate_file_sha256={nm: {str(L): inv["candidates"][nm]["layers"][str(L)]["file_sha256"]
                                           for L in LAYERS} for nm in ("GPTQ", "AGQ")},
               configuration_manifest_sha256=sha256_file(p("configuration_manifest.json")),
               protocol_sha256=sha256_file(p("protocol.json")), frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
               test_results_exist_at_freeze=False)
    jdump(man, p("selection_manifest.json"))
    lw.update(rule="per layer FULL_AGQ iff tuning E_layer lower than GPTQ (ties GPTQ)", split="tune",
              frozen_at=man["frozen_at"])
    jdump(lw, p("layerwise_uniform_baseline.json"))
    jdump(dict(selection_manifest_sha256=sha256_file(p("selection_manifest.json")),
               layerwise_sha256=sha256_file(p("layerwise_uniform_baseline.json")), selection_s=time.time() - t0,
               frozen_at=man["frozen_at"]), p("logs", "selection_freeze.json"))


# ============================================================================ stage: evaluate
def stage_evaluate(a):
    split = a.split
    assert split in ("confirm", "oldtest")
    fr = jload("logs", "selection_freeze.json")
    assert sha256_file(p("selection_manifest.json")) == fr["selection_manifest_sha256"], "selection manifest modified"
    assert sha256_file(p("layerwise_uniform_baseline.json")) == fr["layerwise_sha256"]
    assert sha256_file(p("configuration_manifest.json")) == jload("selection_manifest.json")[
        "configuration_manifest_sha256"]
    _run(split, PS.ORDER, accumulator_check=(split == "confirm"))


def stage_tests(a):
    env = dict(os.environ, MOE_DEVICE="cpu")
    env.pop("MOE_LOAD_LAYERS", None)
    for tag, args in (("projection", ["tests/test_projection_selective_agq.py"]), ("all", ["tests"])):
        if tag == "all" and not a.full:
            continue
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", *args], cwd=ROOT, env=env, capture_output=True,
                           text=True)
        open(p("logs", f"pytest_{tag}.log"), "w").write(r.stdout + r.stderr)
        print(f"[tests] {tag}: rc={r.returncode} :: {r.stdout.strip().splitlines()[-1] if r.stdout else ''}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--split", default="confirm")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    if a.stage == "report":
        import report_projection_selective_agq as R
        return R.main()
    dict(audit=stage_audit, protocol=stage_protocol, regression=stage_regression, validate=stage_validate,
         select=stage_select, evaluate=stage_evaluate, tests=stage_tests)[a.stage](a)


if __name__ == "__main__":
    main()
