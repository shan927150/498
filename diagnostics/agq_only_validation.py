"""AGQ-only validation: standard GPTQ vs Affinity-Guided GPTQ (MoEQuant Sec. 4.3, Eq. 18/19) on the SAME frozen
C4 Random (FR) B = 128 x 512 calibration windows (CANONICAL-B1 activation cache), Qwen1.5-MoE-A2.7B, routed experts
of layers 1/8/16/23, GPTQ 4-bit g128 percdamp 1.0 act-order. Only the per-expert Hessian changes:
    H_std = sum_i x_i x_i^T        H_agq = sum_i c_i x_i x_i^T     (c_i = selected top-4 softmax router prob)

Stages (--stage): source_review | inspect | capture_smoke | hessian_oracle | unit_regression | expert_pilot |
                  full_capture | agq_quantize --config FR_sN | evaluate --config FR_sN | diagnostics | report
"""
from __future__ import annotations

import argparse, hashlib, inspect as _inspect, json, os, subprocess, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import eval_partial_ppl as EP            # noqa: E402
import full_depth_pilot as FD            # noqa: E402
import official_ebss_validation as OV    # noqa: E402
from moe_calib import agq as AG          # noqa: E402

OUT = "runs_agq_only_validation"
OCACHE = OV.OCACHE                                           # held-out reconstruction rows 2128-2255 (historical)
ECACHE = OV.ECACHE                                           # CANONICAL-B1 calibration cache (reused, read-only)
WROOT = os.path.expanduser(os.environ.get("AGQ_WEIGHTS_ROOT", "~/moe_cache_agq_only/weights"))
LAYERS = OV.LAYERS
E = 60
SEEDS = (0, 1, 2)
PRIMARY = "FR_s0"
OFFICIAL_SRC = os.path.expanduser("~/ebss_official_src/moequant")
PILOT_LAYER = 8
N_SMOKE_ROWS = 2


def p(*a):
    return os.path.join(ROOT, OUT, *a)


jdump, sha256_file = FD.jdump, FD.sha256_file


def jload(*a):
    return json.load(open(p(*a)))


def code_hashes():
    files = ["moe_calib/agq.py", "moe_calib/hessian.py", "moe_calib/profiler.py", "moe_calib/gptq.py",
             "moe_calib/partial_model_quant.py", "moe_calib/model_utils.py", "e3_allocation.py",
             "diagnostics/agq_only_validation.py", "diagnostics/official_ebss_validation.py",
             "diagnostics/eval_partial_ppl.py", "tests/test_agq.py"]
    return {f: sha256_file(os.path.join(ROOT, f)) for f in files}


def cost_row(r):
    return dict(quantize_total_s=r["quantize_total_s"], hessian_s=r["hessian_s"], gptq_s=r["gptq_s"],
                other_s=r["quantize_total_s"] - r["hessian_s"] - r["gptq_s"],
                peak_gpu_gb=r["peak_gpu_gb"], peak_cpu_gb=r["peak_cpu_gb"])


def hist_result(name):
    return json.load(open(os.path.join(ROOT, OV.OUT, "results", f"{name}.json")))


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


def config_rows(name):
    chosen, K = OV.config_rows(name)
    assert len(chosen) == 128 == len(set(chosen))
    return chosen, K


def natural_exposure():
    prof = np.load(os.path.join(ROOT, "runs_low_budget", "routing_profile_v2.npz"))
    pk = np.array(prof["p"], dtype=np.float64)                 # [240] = layer-major (LAYERS order) x 60
    dec = np.minimum((np.argsort(np.argsort(pk, kind="stable"), kind="stable") * 10) // len(pk), 9)
    return pk, dec


def calib_counts(name):
    z = np.load(os.path.join(ROOT, OV.OUT, "logs", "capture_routing_nll.npz"))
    rows, _ = config_rows(name)
    return z["C"][rows].sum(0)                                  # [240] routed tokens per (layer, expert)


# ===================================================================== stage: source_review
def stage_source_review(a):
    os.makedirs(p("logs"), exist_ok=True)
    must = ["paper_fidelity.md"]
    for f in must:
        assert os.path.exists(p(f)), f"{f} must be written before implementation is used"
    off = {}
    for rel in ("code/fake_quant/gptq_utils_moe.py", "code/fake_quant/gptq_utils.py", "code/fake_quant/main.py",
                "code/fake_quant/quant_layers/quant_layer.py", "asserts/Algorithm_2_AGQ.png"):
        path = os.path.join(OFFICIAL_SRC, rel)
        off[rel] = sha256_file(path) if os.path.exists(path) else None
    src = open(os.path.join(OFFICIAL_SRC, "code/fake_quant/gptq_utils_moe.py")).read()
    facts = dict(
        sqrt_scaling_present="s1 = torch.sqrt(s)" in src and "inp = inp*s1" in src,
        qwen_scores_softmax_topk4_no_renorm=("routing_score = F.softmax(out, dim=1, dtype=torch.float)" in src and
                                             "torch.topk(routing_score, 4, dim=-1)" in src),
        mixtral_renormalizes="routing_score /= routing_score.sum(dim=-1, keepdim=True)" in src,
        shared_expert_sigmoid_weighting="routing_score_shared = F.sigmoid(out)" in src,
        hessian_normalization_2_over_nsamples="math.sqrt(2 / self.nsamples)" in src,
        agq_flag_in_main="args.AGQ_GPTQ" in open(os.path.join(OFFICIAL_SRC, "code/fake_quant/main.py")).read())
    jdump(dict(official_repo=OV.REPO_URL, official_commit=OV.COMMIT, official_files_sha256=off,
               verified_code_facts=facts, paper="arXiv 2505.03804v1 (HTML), Sec. 4.3 Eq. 15-19, Appendix A.1 Table 5",
               paper_fidelity_md_sha256=sha256_file(p("paper_fidelity.md"))), p("logs", "source_review.json"))
    print("[source_review]", facts)
    assert all(facts.values())


# ===================================================================== stage: inspect (+ freeze protocol)
def stage_inspect(a):
    import transformers
    from transformers import AutoConfig
    from transformers.models.qwen2_moe import modeling_qwen2_moe as QM
    snap = "Qwen/Qwen1.5-MoE-A2.7B"
    cfg = AutoConfig.from_pretrained(snap)
    src = _inspect.getsource(QM.Qwen2MoeSparseMoeBlock.forward)
    mlp = _inspect.getsource(QM.Qwen2MoeMLP.forward)
    checks = dict(
        softmax_fp32_over_all_experts="F.softmax(router_logits, dim=1, dtype=torch.float)" in src,
        topk_after_softmax="torch.topk(routing_weights, self.top_k, dim=-1)" in src,
        renorm_only_if_norm_topk_prob="if self.norm_topk_prob:" in src,
        norm_topk_prob=cfg.norm_topk_prob,
        cast_to_hidden_dtype_before_multiply="routing_weights = routing_weights.to(hidden_states.dtype)" in src,
        dispatch_slot_major="expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)" in src
        and "idx, top_x = torch.where(expert_mask[expert_idx])" in src,
        expert_output_times_coefficient="expert_layer(current_state) * routing_weights[top_x, idx, None]" in src,
        shared_expert_sigmoid_gate="F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output" in src,
        swiglu_down_input="self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))" in mlp,
        hidden_act=cfg.hidden_act, num_experts=cfg.num_experts, top_k=cfg.num_experts_per_tok)
    ok = (checks["norm_topk_prob"] is False and all(v for k, v in checks.items()
                                                    if isinstance(v, bool) and k != "norm_topk_prob"))
    rad = dict(
        model=snap, revision=json.load(open(os.path.join(ROOT, OV.OUT, "execution_policy.json")))["model_revision"],
        transformers=transformers.__version__, forward_source_sha256=hashlib.sha256(src.encode()).hexdigest(),
        checks=checks, verified=ok,
        definition=("c_{i,e} = softmax(router_logits_i, fp32)[e] for e in top4(i); NOT renormalized "
                    "(norm_topk_prob=False); the model multiplies E_e(x_i) by bf16(c_{i,e})"),
        used_in_hessian="c_{i,e} in fp32 (pre-cast), as recorded by the official code (save_gate_res_qwen); "
                        "max relative difference to the bf16 multiplier is 2^-9",
        projections="gate_proj and up_proj: H_gu from the expert input rows; down_proj: H_down from "
                    "act(gate x) * up x rows; all three inherit the same c_{i,e}",
        not_used=["router logits", "expert-internal gate_proj activations", "c^2", "renormalized top-k weights"],
        shared_expert="not quantized in this study (historical scope); the official sigmoid-gate shared-expert "
                      "weighting is therefore not applicable",
        row_order="slot-major dispatch (one_hot(sel).permute(2,1,0); where) - reproduced exactly by "
                  "moe_calib.agq.dispatch_rows / Profiler._record_rows")
    jdump(rad, p("router_affinity_definition.json"))
    print("[inspect] routing verified:", ok)
    assert ok
    _freeze()


def _freeze():
    if os.path.exists(p("protocol.json")):
        print("[freeze] protocol already frozen"); return
    frozen = json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))
    man = {}
    for s in SEEDS:
        name = f"FR_s{s}"
        r = hist_result(name)
        rows, K = config_rows(name)
        assert rows == r["cache_rows"] and r["policy"] == "CANONICAL-B1" and r["valid"]
        sel = OV.selections()[name]
        man[name] = dict(seed=s, cache_rows=rows, K=K, ids_hash=sel["ids_hash"], source=sel["source"],
                         source_sha256=sel["source_sha256"], original_chosen=sel["chosen"],
                         historical_gptq_weights_sha256=r["installed_weights_sha256"],
                         historical_weights_dir=r["weights_dir"], historical_ppl=r["ppl"], historical_recon=r["recon"])
    cm = json.load(open(os.path.join(ROOT, OV.OUT, "logs", "capture_manifest.json")))
    jdump(dict(calibration_cache=ECACHE, cache_files=cm["files"], token_sha256=cm["token_sha256"],
               capture_policy="CANONICAL-B1", configs=man), p("calibration_manifest.json"))
    pp = json.load(open(os.path.join(ROOT, EP.OUT, "evaluation_protocol.json")))
    proto = dict(
        experiment="AGQ-only: standard GPTQ vs AGQ-GPTQ on identical C4 Random B=128 windows",
        question="Does router-affinity-weighted Hessian accumulation improve MoE quantization quality over standard "
                 "GPTQ under a strong matched natural-text calibration baseline?",
        model="Qwen/Qwen1.5-MoE-A2.7B @ 1a758c50ecb6350748b9ce0a99d2352fd9fc11c9 (BF16)",
        gptq=frozen["gptq"], layers=LAYERS, n_experts=240, n_projections=720, rtn_fallback="none expected (all "
        "experts have tokens at B=128); any fallback is recorded and invalidates nothing but is reported",
        calibration=dict(source="C4 Random (FR), CANONICAL-B1 cache", primary=PRIMARY,
                         extension=["FR_s1", "FR_s2"],
                         extension_rule="run s1 and s2 if FR_s0 AGQ is valid (correctness + resources), regardless "
                                        "of the direction of the s0 result"),
        conditions=dict(C4R128_GPTQ="historical CANONICAL-B1 GPTQ result (weights sha256 in calibration_manifest); "
                                    "admissible only if the unit_regression stage reproduces it bitwise with the "
                                    "modified code (hessian_weighting='none')",
                        C4R128_AGQ="identical pipeline with hessian_weighting='router_affinity'"),
        agq=dict(hessian="H_e = sum_i c_{i,e} x_i x_i^T via (sqrt(c) x); c not c^2", affinity="see "
                 "router_affinity_definition.json", normalization="none added: production GPTQ uses the raw sum and "
                 "percdamp * mean(diag H) damping, both invariant to a global positive scale",
                 applied_to="all routed gate_proj/up_proj/down_proj of layers 1/8/16/23; shared expert and router "
                            "untouched"),
        evaluation=dict(
            reconstruction_rows=list(frozen["split"]["test"]),
            reconstruction_cache=OCACHE,
            standard_expert_recon="E3 per-expert held-out relative reconstruction (unchanged); config = mean of 240",
            weighted_expert_recon="sum_i c_i ||E(x_i)-Ê(x_i)||^2 / sum_i c_i ||E(x_i)||^2 over ALL routed held-out "
                                  "tokens with held-out routing coefficients; config = mean of 240",
            moe_layer_output="PRIMARY: per layer sum_x ||y_BF16 - y_quant||^2 / sum_x ||y_BF16||^2 over all 65,536 "
                             "held-out tokens, fp32 math, frozen BF16 routing (identical inputs, router and c), shared "
                             "expert included in y; config = mean of the 4 layer values",
            moe_layer_output_native="secondary: same ratio from the actual BF16 Qwen2MoeSparseMoeBlock forward with "
                                    "installed weights",
            ppl_windows_sha256=pp["windows_sha256"], ppl="EP.layerwise_nll, frozen 128 x 512 WikiText-2 windows",
            exposure_deciles="deciles of natural routing frequency p (runs_low_budget/routing_profile_v2.npz)"),
        decision_rule=dict(
            primary_metric="moe_layer_output (mean of 4 layers), relative difference (AGQ-GPTQ)/GPTQ",
            tau_rel=0.005, min_layers_improved=3,
            AGQ_LAYER_QUALITY_GAIN="mean rel diff <= -tau AND >= 3/4 layers improved AND every measured seed improves",
            AGQ_QUALITY_REGRESSION="mean rel diff >= +tau",
            AGQ_SURROGATE_ONLY_GAIN="not the above AND weighted expert recon rel diff <= -tau",
            AGQ_NO_MEASURABLE_GAIN="otherwise",
            AGQ_RUN_BLOCKED="any correctness gate fails (alignment, oracle, unit regression, validity)",
            robustness="a gain is called robust only if every measured seed agrees; one seed is never robust",
            secondary=dict(tau_rel=0.005, ppl_noise=0.014,
                           ppl_noise_source="largest calibration-grouping |dPPL| in update_batch_sensitivity.md / "
                                            "|dPPL| scale in update_ebss_official_validation.md")),
        hard_constraints=["no new calibration data", "no EBSS", "no Hadamard", "solver/damping/act-order/group size/"
                          "bits unchanged", "no extra layers", "no affinity powers or transforms",
                          "historical artifacts not overwritten"],
        code_sha256=code_hashes(), frozen_before_any_quality_result=True,
        created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(proto, p("protocol.json"))
    print("[freeze] protocol.json + calibration_manifest.json written")


# ===================================================================== stage: tests
def stage_tests(a):
    os.makedirs(p("logs"), exist_ok=True)
    env = dict(os.environ, MOE_DEVICE="cpu")
    env.pop("MOE_LOAD_LAYERS", None)
    for tag, args in (("agq", ["tests/test_agq.py"]), ("all", ["tests"])):
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", *args], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        open(p("logs", f"pytest_{tag}.log"), "w").write(r.stdout + r.stderr)
        print(f"[tests] {tag}: rc={r.returncode} :: {r.stdout.strip().splitlines()[-1] if r.stdout else ''}")
        if r.returncode != 0:
            raise SystemExit(f"tests failed ({tag})")


# ===================================================================== stage: capture_smoke
@torch.no_grad()
def stage_capture_smoke(a):
    """Real model, real CANONICAL-B1 calibration rows: (1) Profiler alignment checks on every expert/row,
    (2) the MoE block output is reconstructed BITWISE from captured expert outputs x bf16(c) + shared expert,
    (3) captured c equal the model's own routing weights, (4) unmatched experts receive no rows."""
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    import torch.nn.functional as F
    model, _ = load_partial()
    rows, _ = config_rows(PRIMARY)
    rows = rows[:N_SMOKE_ROWS]
    out = dict(rows=rows, layers={})
    for L in LAYERS:
        cache = load_layer_cache(ECACHE, L)
        pr = Profiler(model, L, K=1, hessian_weighting="router_affinity", collect_affinity=True,
                      verify_alignment=True).run(cache, rows)
        rec = dict(alignment_checks=pr.n_alignment_checks, hessian_finite=True)
        for e in range(E):
            for proj in ("gu", "down"):
                rec["hessian_finite"] &= bool(torch.isfinite(pr.acc.H[(L, e, proj)]).all())
        pr.free()
        # block-level reconstruction
        blk = MU.moe_block(model, L)
        agg_ok, coef_ok, unmatched_ok, sums = True, True, True, []
        with MU.layer_on_gpu(model, L):
            for sid in rows:
                cap = dict(ex={}, out={}, mlp_in=None, mlp_out=None, logits=None, shared=None)
                hooks = [blk.register_forward_pre_hook(lambda m, a_: cap.__setitem__("mlp_in", a_[0].detach())),
                         blk.register_forward_hook(lambda m, i, o: cap.__setitem__("mlp_out", o[0].detach())),
                         blk.gate.register_forward_hook(lambda m, i, o: cap.__setitem__("logits", o.detach())),
                         blk.shared_expert.register_forward_hook(lambda m, i, o: cap.__setitem__("shared", o.detach()))]

                def mk(e):
                    def fn(m, i, o):
                        cap["ex"][e] = i[0].detach(); cap["out"][e] = o.detach()
                    return fn
                hooks += [blk.experts[e].register_forward_hook(mk(e)) for e in range(E)]
                try:
                    MU.run_layer(model, L, cache[sid:sid + 1].to(MU.GPU))
                finally:
                    for h in hooks:
                        h.remove()
                hs = cap["mlp_in"].reshape(-1, cap["mlp_in"].shape[-1])
                vals, sel = AG.routing_coefficients(cap["logits"], MU.top_k(model))
                ref = torch.topk(F.softmax(cap["logits"], dim=1, dtype=torch.float), MU.top_k(model), dim=-1)
                coef_ok &= bool(torch.equal(vals, ref.values) and torch.equal(sel, ref.indices))
                sums.append(vals.sum(-1).cpu())
                rws = AG.dispatch_rows(sel, vals, E)
                final = torch.zeros_like(hs)
                for e in range(E):
                    if e not in rws:
                        unmatched_ok &= (e not in cap["ex"]) or cap["ex"][e].shape[0] == 0
                        continue
                    tx, slot, c = rws[e]
                    agg_ok &= bool(torch.equal(cap["ex"][e], hs[tx]))
                    final.index_add_(0, tx, (cap["out"][e] * c.to(hs.dtype)[:, None]).to(hs.dtype))
                y = final + F.sigmoid(blk.shared_expert_gate(hs)) * cap["shared"]
                agg_ok &= bool(torch.equal(y, cap["mlp_out"].reshape(-1, hs.shape[-1])))
        s = torch.cat(sums)
        rec.update(block_output_reconstructed_bitwise=agg_ok, coefficients_equal_model_routing=coef_ok,
                   unmatched_experts_no_rows=unmatched_ok, coef_sum_mean=float(s.mean()), coef_sum_min=float(s.min()),
                   coef_sum_max=float(s.max()), coef_sum_below_one=bool((s < 1).all()),
                   n_affinity_rows=int(sum(t.numel() for v in pr.affinity.values() for _, t, _, _ in v)),
                   expected_rows=int(len(rows) * 512 * MU.top_k(model)))
        rec["rows_match"] = rec["n_affinity_rows"] == rec["expected_rows"]
        rec["passed"] = bool(rec["alignment_checks"] > 0 and rec["hessian_finite"] and agg_ok and coef_ok
                             and unmatched_ok and rec["rows_match"] and rec["coef_sum_below_one"])
        out["layers"][str(L)] = rec
        print(f"[capture_smoke] L{L}: {rec}", flush=True)
    out["passed"] = all(v["passed"] for v in out["layers"].values())
    jdump(out, p("capture_alignment_results.json"))
    if not out["passed"]:
        raise SystemExit("capture alignment FAILED")


# ===================================================================== stage: hessian_oracle
@torch.no_grad()
def stage_hessian_oracle(a):
    """Production weighted Hessians (Profiler, router_affinity) vs an independent float64 oracle built from the
    captured activation rows and coefficients of the same forward, real model, layer 8, 4 calibration rows."""
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    model, _ = load_partial()
    rows, _ = config_rows(PRIMARY)
    rows = rows[:4]
    L = PILOT_LAYER
    cache = load_layer_cache(ECACHE, L)
    blk = MU.moe_block(model, L)
    X = {e: {"gu": [], "down": []} for e in range(E)}
    C = {e: [] for e in range(E)}
    pr = Profiler(model, L, K=1, hessian_weighting="router_affinity", collect_affinity=True)
    hooks = []
    for e in range(E):
        hooks.append(blk.experts[e].gate_proj.register_forward_pre_hook(
            lambda m, a_, e=e: X[e]["gu"].append(a_[0].detach().double().cpu())))
        hooks.append(blk.experts[e].down_proj.register_forward_pre_hook(
            lambda m, a_, e=e: X[e]["down"].append(a_[0].detach().double().cpu())))
    try:
        pr.run(cache, rows)
    finally:
        for h in hooks:
            h.remove()
    for e in range(E):
        for sid, tx, slot, c in pr.affinity.get((L, e), []):
            C[e].append(c.double())
    res, worst = {}, 0.0
    for e in range(E):
        if not C[e]:
            continue
        c = torch.cat(C[e]).numpy()
        for proj in ("gu", "down"):
            Xe = torch.cat(X[e][proj]).numpy()
            assert Xe.shape[0] == c.shape[0]
            ref = AG.weighted_hessian_reference(Xe, c)
            got = pr.acc.H[(L, e, proj)][0].double().cpu().numpy()
            rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
            unweighted = float(np.linalg.norm(got - Xe.T @ Xe) / np.linalg.norm(Xe.T @ Xe))
            csq = float(np.linalg.norm(got - AG.weighted_hessian_reference(Xe, c ** 2)) / np.linalg.norm(ref))
            res[f"{e}.{proj}"] = dict(rows=int(c.shape[0]), rel_err_vs_oracle=rel, rel_dist_to_unweighted=unweighted,
                                      rel_dist_to_c_squared=csq, symmetric=bool(np.array_equal(got, got.T)))
            worst = max(worst, rel)
    tol = 1e-5
    syn = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_agq.py", "-k",
                          "oracle or chunking or properties or mixed_routing"], cwd=ROOT,
                         env=dict(os.environ, MOE_DEVICE="cpu"), capture_output=True, text=True)
    out = dict(layer=L, rows=rows, tolerance_rel_fro=tol, worst_rel_err=worst, experts=res,
               min_dist_to_c_squared=min(v["rel_dist_to_c_squared"] for v in res.values()),
               min_dist_to_unweighted=min(v["rel_dist_to_unweighted"] for v in res.values()),
               all_symmetric=all(v["symmetric"] for v in res.values()),
               synthetic_tests=syn.stdout.strip().splitlines()[-1] if syn.stdout else "", synthetic_rc=syn.returncode)
    out["passed"] = bool(worst <= tol and out["all_symmetric"] and syn.returncode == 0
                         and out["min_dist_to_c_squared"] > 10 * tol)
    jdump(out, p("hessian_oracle_results.json"))
    print(f"[hessian_oracle] worst rel err {worst:.2e}; passed {out['passed']}")
    if not out["passed"]:
        raise SystemExit("Hessian oracle FAILED")


# ===================================================================== stage: unit_regression
def stage_unit_regression(a):
    """(1) modified code, hessian_weighting='none' -> must reproduce historical FR_s0 weights/PPL/recon bitwise;
    (2) hessian_weighting='unit_affinity' (weighted path, c := 1) -> must reproduce them bitwise too."""
    out_path = p("unit_regression.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    ref = hist_result(PRIMARY)
    rows, K = config_rows(PRIMARY)
    model = None
    for tag, mode in (("GPTQ_rerun", "none"), ("UNIT_affinity", "unit_affinity")):
        if tag in res:
            continue
        if model is None:
            model, t_load = OV.load_model()
            assert OV.verify_pristine_full(model)
        r = OV.quant_config(model, f"{tag}_{PRIMARY}", ECACHE, rows, OCACHE, K, hessian_weighting=mode,
                            weights_root=WROOT)
        r.update(config=PRIMARY, hessian_weighting=mode)
        jdump(r, p("logs", f"result_{tag}_{PRIMARY}.json"))
        res[tag] = dict(mode=mode, weights_sha256=r["installed_weights_sha256"], ppl=r["ppl"], recon=r["recon"],
                        weights_exact=r["installed_weights_sha256"] == ref["installed_weights_sha256"],
                        ppl_exact=r["ppl"] == ref["ppl"], recon_exact=r["recon"] == ref["recon"],
                        per_window_nll_exact=r["per_window_nll"] == ref["per_window_nll"],
                        per_expert_exact=[(x["layer"], x["expert"], x["mean"]) for x in r["per_expert"]] ==
                                         [(x["layer"], x["expert"], x["mean"]) for x in ref["per_expert"]],
                        valid=r["valid"], cost=cost_row(r))
        res[tag]["passed"] = all(res[tag][k] for k in ("weights_exact", "ppl_exact", "recon_exact",
                                                       "per_window_nll_exact", "per_expert_exact", "valid"))
        jdump(res, out_path)
        print(f"[unit_regression] {tag}: {res[tag]}", flush=True)
        # quant_config installs weights in place: reload a pristine model for the next config
        del model; model = None
        torch.cuda.empty_cache()
    res["reference"] = dict(config=PRIMARY, weights_sha256=ref["installed_weights_sha256"], ppl=ref["ppl"],
                            recon=ref["recon"])
    res["passed"] = bool(res["GPTQ_rerun"]["passed"] and res["UNIT_affinity"]["passed"])
    jdump(res, out_path)
    print(f"[unit_regression] passed={res['passed']}")
    if not res["passed"]:
        raise SystemExit("unit-affinity / standard GPTQ regression FAILED")


# ===================================================================== helpers: Hessians per layer
@torch.no_grad()
def layer_hessians(model, L, rows, K, mode, experts=None, collect_affinity=False):
    """Profiler over the frozen calibration rows exactly as E3 (contiguous K blocks, weights 1). Returns
    ({(e, proj): H fp32 on CPU}, profiler seconds, affinity dict)."""
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    cache = load_layer_cache(ECACHE, L)
    pr = (Profiler(model, L, K=K) if mode == "none" else
          Profiler(model, L, K=K, hessian_weighting=mode, collect_affinity=collect_affinity))
    pr.run(cache, rows, weights=[1.0] * len(rows))
    H = {}
    for e in (experts if experts is not None else range(E)):
        for proj in ("gu", "down"):
            H[(e, proj)] = pr.acc.H_sum((L, e, proj), list(range(K))).cpu()
    tokens = {e: pr.acc.tokens((L, e, "gu")) for e in range(E)}
    aff = pr.affinity
    secs = pr.seconds
    pr.free()
    del cache
    return H, secs, aff, tokens


def expert_affinities(aff, L, e):
    lst = aff.get((L, e), [])
    return torch.cat([c for _, _, _, c in lst]).double().numpy() if lst else np.zeros(0)


@torch.no_grad()
def gptq_pair(model, L, e, Hs, Ha, g):
    from moe_calib import gptq as GQ
    from moe_calib.quant_eval import fp_weights
    from moe_calib import model_utils as MU
    W = fp_weights(model, L, e)
    out = {}
    for proj, hk in (("gate_proj", "gu"), ("up_proj", "gu"), ("down_proj", "down")):
        res = {}
        for m, H in (("std", Hs[(e, hk)]), ("agq", Ha[(e, hk)])):
            info = {}
            t0 = time.time()
            q = GQ.gptq_quantize(W[proj], H.to(MU.GPU), g["bits"], g["group_size"], g["percdamp"], info=info)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res[m] = (q, info, time.time() - t0)
        out[proj] = res
    return W, out


# ===================================================================== stage: expert_pilot
@torch.no_grad()
def stage_expert_pilot(a):
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    assert jload("capture_alignment_results.json")["passed"] and jload("hessian_oracle_results.json")["passed"]
    model, _ = load_partial()
    g = jload("protocol.json")["gptq"]
    rows, K = config_rows(PRIMARY)
    L = PILOT_LAYER
    cnt = calib_counts(PRIMARY)[LAYERS.index(L) * E:(LAYERS.index(L) + 1) * E]
    order = np.argsort(-cnt, kind="stable")
    pick = dict(high=[int(order[0]), int(order[1])], medium=[int(order[E // 2 - 1]), int(order[E // 2])],
                low=[int(order[-2]), int(order[-1])])
    experts = [e for v in pick.values() for e in v]
    torch.cuda.reset_peak_memory_stats()
    Hs, ts, _, tok = layer_hessians(model, L, rows, K, "none", experts)
    Ha, ta, aff, _ = layer_hessians(model, L, rows, K, "router_affinity", experts, collect_affinity=True)
    test_ids = list(range(*jload("protocol.json")["evaluation"]["reconstruction_rows"]))
    HO = Profiler(model, L, collect_hessian=False, collect_holdout=True, hessian_weighting="router_affinity",
                  collect_affinity=True).run(load_layer_cache(OCACHE, L), test_ids)
    res = {}
    for grp, es in pick.items():
        for e in es:
            c = expert_affinities(aff, L, e)
            W, q = gptq_pair(model, L, e, Hs, Ha, g)
            ho_c = torch.cat([c_ for _, _, _, c_ in HO.affinity.get((L, e), [])])
            # held-out: first 128 dispatch rows per sequence (E3 convention) with their coefficients
            ho = {}
            for proj, hk in (("gate_proj", "gu"), ("up_proj", "gu"), ("down_proj", "down")):
                num = {"std": 0.0, "agq": 0.0}; wnum = {"std": 0.0, "agq": 0.0}; den = wden = 0.0
                cl = {sid: c_ for sid, _, _, c_ in HO.affinity.get((L, e), [])}
                for sid, X in HO.holdout.get((L, e, hk), []):
                    Xf = X.to(MU.GPU, torch.float32)
                    cc = cl[sid][:Xf.shape[0]].to(MU.GPU).double()
                    t2 = ((Xf @ W[proj].to(MU.GPU).T).double() ** 2).sum(1)
                    den += float(t2.sum()); wden += float((cc * t2).sum())
                    for m in ("std", "agq"):
                        r2 = ((Xf @ (W[proj].to(MU.GPU) - q[proj][m][0]).T).double() ** 2).sum(1)
                        num[m] += float(r2.sum()); wnum[m] += float((cc * r2).sum())
                ho[proj] = dict(std_unweighted=num["std"] / den, agq_unweighted=num["agq"] / den,
                                std_weighted=wnum["std"] / wden, agq_weighted=wnum["agq"] / wden)
            rec = dict(group=grp, tokens=int(tok[e]), affinity=AG.affinity_summary(c),
                       heldout_affinity=AG.affinity_summary(ho_c.double().numpy()), projections={})
            for proj, hk in (("gate_proj", "gu"), ("up_proj", "gu"), ("down_proj", "down")):
                hd = AG.hessian_diagnostics(Hs[(e, hk)].to(MU.GPU), Ha[(e, hk)].to(MU.GPU))
                dS, dA = torch.diag(Hs[(e, hk)]).double(), torch.diag(Ha[(e, hk)]).double()
                spec = {}
                for m, H in (("std", Hs[(e, hk)]), ("agq", Ha[(e, hk)])):
                    Hd = H.double().to(MU.GPU)
                    Hd = Hd + torch.eye(Hd.shape[0], device=Hd.device, dtype=Hd.dtype) * g["percdamp"] * torch.diag(Hd).mean()
                    ev = torch.linalg.eigvalsh(Hd)
                    spec[m] = dict(cond_damped=float(ev.max() / ev.min()), trace=float(torch.trace(H.double())))
                wd = AG.weight_diagnostics(q[proj]["std"][0], q[proj]["agq"][0], W[proj].to(MU.GPU),
                                           q[proj]["std"][1], q[proj]["agq"][1])
                rec["projections"][proj] = dict(
                    hessian=hd, spectral=spec,
                    diag_std=dict(mean=float(dS.mean()), median=float(dS.median()), max=float(dS.max())),
                    diag_agq=dict(mean=float(dA.mean()), median=float(dA.median()), max=float(dA.max())),
                    weights=wd, heldout=ho[proj],
                    gptq_s=dict(std=q[proj]["std"][2], agq=q[proj]["agq"][2]))
            res[str(e)] = rec
            print(f"[pilot] L{L} e{e} ({grp}) tokens {tok[e]} c mean {rec['affinity']['mean']:.3f} "
                  f"N_eff/N {rec['affinity']['n_eff_frac']:.3f} | gate rel-dW "
                  f"{rec['projections']['gate_proj']['weights']['rel_l2_diff']:.3e}", flush=True)
    out = dict(layer=L, selection_rule="FR_s0 calibration routed-token rank within layer 8 (frozen routing; "
               "not AGQ performance): top-2, middle-2, bottom-2", picks=pick, experts=res,
               hessian_s=dict(std=ts, agq=ta), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
    out["nontrivial_change"] = all(r["projections"][pj]["weights"]["frac_weights_changed"] > 0
                                   for r in res.values() for pj in AG.PROJ)
    jdump(out, p("smoke_results.json"))
    print(f"[pilot] nontrivial change: {out['nontrivial_change']}; profiler s std {ts:.1f} agq {ta:.1f}")


# ===================================================================== stage: full_capture
@torch.no_grad()
def stage_full_capture(a):
    """The calibration activations are the existing CANONICAL-B1 cache (verified, not regenerated). Record the
    calibration affinity statistics of every (layer, expert) for FR s0/s1/s2 (router-only replay, batch 1)."""
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    man = jload("calibration_manifest.json")
    for L in LAYERS:
        assert sha256_file(os.path.join(ECACHE, f"L{L}.pt")) == man["cache_files"][str(L)]["sha256"], "cache changed"
    model, _ = load_partial()
    pk, dec = natural_exposure()
    out = dict(cache_verified=True, configs={})
    for s in SEEDS:
        name = f"FR_s{s}"
        rows, K = config_rows(name)
        per = []
        t0 = time.time()
        meta_bytes = 0
        for L in LAYERS:
            cache = load_layer_cache(ECACHE, L)
            pr = Profiler(model, L, collect_hessian=False, hessian_weighting="router_affinity",
                          collect_affinity=True).run(cache, rows, routing_only=True)
            for e in range(E):
                c = expert_affinities(pr.affinity, L, e)
                meta_bytes += c.size * (4 + 4 + 1)                 # fp32 c + int32 token + int8 slot (sparse)
                i = LAYERS.index(L) * E + e
                per.append(dict(layer=L, expert=e, natural_p=float(pk[i]), natural_decile=int(dec[i]),
                                **AG.affinity_summary(c)))
            del cache, pr
        out["configs"][name] = dict(per_expert=per, seconds=time.time() - t0, sparse_metadata_bytes=meta_bytes,
                                    dense_alternative_bytes=128 * 512 * E * 4 * len(LAYERS))
        print(f"[full_capture] {name}: {len(per)} experts, {time.time() - t0:.1f} s", flush=True)
    jdump(out, p("affinity_statistics.json"))


# ===================================================================== stage: agq_quantize
def stage_agq_quantize(a):
    name = a.config
    assert name in [f"FR_s{s}" for s in SEEDS]
    if os.path.exists(p("results", f"AGQ_{name}.json")):
        print(f"[agq] {name} done"); return
    for f in ("capture_alignment_results.json", "hessian_oracle_results.json", "unit_regression.json"):
        assert jload(f)["passed"], f"gate {f} not passed"
    if name != PRIMARY:
        r0 = json.load(open(p("results", f"AGQ_{PRIMARY}.json")))
        assert r0["valid"], "extension seeds require a valid primary AGQ run"
    rows, K = config_rows(name)
    model, t_load = OV.load_model()
    assert OV.verify_pristine_full(model)
    r = OV.quant_config(model, f"AGQ_{name}", ECACHE, rows, OCACHE, K, hessian_weighting="router_affinity",
                        weights_root=WROOT)
    r.update(config=name, hessian_weighting="router_affinity", policy="CANONICAL-B1", cache_rows=rows,
             model_load_s=t_load)
    os.makedirs(p("results"), exist_ok=True)
    jdump(r, p("results", f"AGQ_{name}.json"))
    print(f"[agq] {name}: PPL {r['ppl']:.6f} recon {r['recon']:.7f} fallbacks {r['rtn_fallbacks']} valid {r['valid']} "
          f"({r['quantize_total_s']:.0f} s)", flush=True)
    if not r["valid"]:
        raise SystemExit(f"AGQ {name} failed its correctness checks")


# ===================================================================== stage: evaluate
def weights_dir(method, name):
    if method == "GPTQ":
        return hist_result(name)["weights_dir"]
    return json.load(open(p("results", f"AGQ_{name}.json")))["weights_dir"]


@torch.no_grad()
def stage_evaluate(a):
    """Held-out MoE evaluation for GPTQ vs AGQ of one seed: affinity-weighted / unweighted full-expert error,
    projection-level Eq. 18 losses and the aggregated MoE-layer output error (frozen BF16 routing, fp32 math), plus
    the native BF16 block forward."""
    import torch.nn.functional as F
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    name = a.config
    outp = p("results", f"eval_{name}.json")
    if os.path.exists(outp):
        print(f"[evaluate] {name} done"); return
    model, _ = load_partial()
    test_ids = list(range(*jload("protocol.json")["evaluation"]["reconstruction_rows"]))
    methods = ("GPTQ", "AGQ")
    wd = {m: weights_dir(m, name) for m in methods}
    out = dict(config=name, weights_dirs=wd, layers={})
    t0 = time.time()
    for L in LAYERS:
        blk = MU.moe_block(model, L)
        dig0 = PQ.layer_digest(model, L)
        orig = {e: {k: w.detach().clone() for k, w in MU.get_expert_weights(model, L, e).items()} for e in range(E)}
        files = {m: torch.load(os.path.join(wd[m], f"L{L}.pt")) for m in methods}
        Wfp = {e: {k: v.to(MU.GPU, torch.float32) for k, v in orig[e].items()} for e in range(E)}
        Wq = {m: {e: {k: files[m][f"{e}.{k}"].to(MU.GPU, torch.float32) for k in AG.PROJ} for e in range(E)}
              for m in methods}
        acc = AG.LayerEvalAccumulator(methods, E, model.config.hidden_size)
        native = {m: dict(num=0.0, den=0.0) for m in methods}
        cache = load_layer_cache(OCACHE, L)
        H_in, logits_all, y0_all = [], [], []
        with MU.layer_on_gpu(model, L):
            sh = {k: getattr(blk.shared_expert, k).weight.float() for k in AG.PROJ}
            sg = blk.shared_expert_gate.weight.float()
            for sid in test_ids:
                cap = {}
                hk = [blk.register_forward_pre_hook(lambda m_, a_: cap.__setitem__("h", a_[0].detach())),
                      blk.register_forward_hook(lambda m_, i, o: cap.__setitem__("y", o[0].detach())),
                      blk.gate.register_forward_hook(lambda m_, i, o: cap.__setitem__("lg", o.detach()))]
                try:
                    MU.run_layer(model, L, cache[sid:sid + 1].to(MU.GPU))
                finally:
                    for h in hk:
                        h.remove()
                h = cap["h"].reshape(-1, cap["h"].shape[-1])
                H_in.append(h); logits_all.append(cap["lg"]); y0_all.append(cap["y"].reshape(h.shape))
                vals, sel = AG.routing_coefficients(cap["lg"], MU.top_k(model))
                hf = h.float()
                shared = F.sigmoid(hf @ sg.T) * AG.expert_fp32(hf, sh)
                acc.add_sequence(hf, vals, sel, Wfp, Wq, shared)
            # native BF16 block forward with installed weights (router + inputs unchanged -> identical routing)
            for m in methods:
                for e in range(E):
                    MU.set_expert_weights(model, L, e, {k: files[m][f"{e}.{k}"] for k in AG.PROJ})
                for h, lg0, y0 in zip(H_in, logits_all, y0_all):
                    y, lg = blk(h.unsqueeze(0))
                    assert torch.equal(lg, lg0), "router output changed under quantization"
                    native[m]["num"] += float(((y.reshape(h.shape).double() - y0.double()) ** 2).sum())
                    native[m]["den"] += float((y0.double() ** 2).sum())
            for e in range(E):
                MU.set_expert_weights(model, L, e, orig[e])
        assert PQ.layer_digest(model, L) == dig0, "pristine weights not restored"
        s = acc.summary()
        for m in methods:
            s[m]["moe_layer_output_native_rel_err"] = native[m]["num"] / native[m]["den"]
        out["layers"][str(L)] = s
        print(f"[evaluate] {name} L{L}: MoE out GPTQ {s['GPTQ']['moe_layer_output_rel_err']:.6e} "
              f"AGQ {s['AGQ']['moe_layer_output_rel_err']:.6e} | native GPTQ "
              f"{s['GPTQ']['moe_layer_output_native_rel_err']:.6e} AGQ {s['AGQ']['moe_layer_output_native_rel_err']:.6e}",
              flush=True)
        del cache, H_in, logits_all, y0_all, Wfp, Wq, files
        torch.cuda.empty_cache()
    out["seconds"] = time.time() - t0
    os.makedirs(p("results"), exist_ok=True)
    jdump(out, outp)


# ===================================================================== stage: diagnostics
@torch.no_grad()
def stage_diagnostics(a):
    """Per expert-projection Hessian and weight diagnostics for the primary seed (both methods recomputed with the
    production Profiler and solver; the recomputed BF16 weights must equal the installed files bitwise)."""
    from moe_calib import model_utils as MU
    outp = p("logs", "diagnostics_raw.json")
    if os.path.exists(outp):
        print("[diagnostics] done"); return
    model, _ = load_partial()
    g = jload("protocol.json")["gptq"]
    rows, K = config_rows(PRIMARY)
    files = {m: {L: torch.load(os.path.join(weights_dir(m, PRIMARY), f"L{L}.pt")) for L in LAYERS}
             for m in ("GPTQ", "AGQ")}
    res, match, timing = [], True, dict(profiler_std=0.0, profiler_agq=0.0, gptq_std=0.0, gptq_agq=0.0)
    for L in LAYERS:
        Hs, ts, _, tok = layer_hessians(model, L, rows, K, "none")
        Ha, ta, aff, _ = layer_hessians(model, L, rows, K, "router_affinity", collect_affinity=True)
        timing["profiler_std"] += ts; timing["profiler_agq"] += ta
        for e in range(E):
            c = expert_affinities(aff, L, e)
            W, q = gptq_pair(model, L, e, Hs, Ha, g)
            rec = dict(layer=L, expert=e, tokens=int(tok[e]), affinity=AG.affinity_summary(c), proj={})
            for proj, hk in (("gate_proj", "gu"), ("up_proj", "gu"), ("down_proj", "down")):
                for m, mm in (("GPTQ", "std"), ("AGQ", "agq")):
                    inst = files[m][L][f"{e}.{proj}"]
                    match &= bool(torch.equal(q[proj][mm][0].to(inst.dtype).cpu(), inst))
                timing["gptq_std"] += q[proj]["std"][2]; timing["gptq_agq"] += q[proj]["agq"][2]
                hd = AG.hessian_diagnostics(Hs[(e, hk)].to(MU.GPU), Ha[(e, hk)].to(MU.GPU))
                wdg = AG.weight_diagnostics(q[proj]["std"][0], q[proj]["agq"][0], W[proj].to(MU.GPU),
                                            q[proj]["std"][1], q[proj]["agq"][1])
                bs, ba = files["GPTQ"][L][f"{e}.{proj}"], files["AGQ"][L][f"{e}.{proj}"]
                wdg["frac_bf16_weights_changed"] = float((bs != ba).float().mean())
                rec["proj"][proj] = dict(hessian=hd, weights=wdg)
            res.append(rec)
        print(f"[diagnostics] L{L} done; recomputed == installed so far: {match}", flush=True)
        del Hs, Ha
        torch.cuda.empty_cache()
    jdump(dict(config=PRIMARY, recomputed_weights_equal_installed=match, timing=timing, experts=res), outp)


# ===================================================================== stage: c2_probe (exploratory, post hoc)
@torch.no_grad()
def stage_c2_probe(a):
    """EXPLORATORY mechanism probe, added AFTER the primary results (not part of the frozen decision). Tests whether
    the aggregated MoE-output error behaves like a c^2-weighted expert error: per layer, decompose
    sum_x ||sum_k c_k D_k(x)||^2 = sum_x sum_k c_k^2 ||D_k(x)||^2 (diagonal) + cross terms, and report the c^2-weighted
    full-expert error for GPTQ vs AGQ on the same held-out tokens (FR_s0 weights, frozen BF16 routing)."""
    import torch.nn.functional as F
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    name = a.config
    outp = p("logs", f"c2_probe_{name}.json")
    if os.path.exists(outp):
        print("[c2_probe] done"); return
    model, _ = load_partial()
    test_ids = list(range(*jload("protocol.json")["evaluation"]["reconstruction_rows"]))
    methods = ("GPTQ", "AGQ")
    wd = {m: weights_dir(m, name) for m in methods}
    out = dict(config=name, exploratory=True, layers={})
    for L in LAYERS:
        blk = MU.moe_block(model, L)
        files = {m: torch.load(os.path.join(wd[m], f"L{L}.pt")) for m in methods}
        Wfp = {e: {k: v.to(MU.GPU, torch.float32) for k, v in MU.get_expert_weights(model, L, e).items()} for e in range(E)}
        Wq = {m: {e: {k: files[m][f"{e}.{k}"].to(MU.GPU, torch.float32) for k in AG.PROJ} for e in range(E)} for m in methods}
        acc = {m: dict(total=0.0, diag=0.0, c2num=0.0, c2den=0.0, c1num=0.0, c1den=0.0) for m in methods}
        cache = load_layer_cache(OCACHE, L)
        with MU.layer_on_gpu(model, L):
            for sid in test_ids:
                cap = {}
                hk = [blk.register_forward_pre_hook(lambda m_, a_: cap.__setitem__("h", a_[0].detach())),
                      blk.gate.register_forward_hook(lambda m_, i, o: cap.__setitem__("lg", o.detach()))]
                try:
                    MU.run_layer(model, L, cache[sid:sid + 1].to(MU.GPU))
                finally:
                    for h in hk:
                        h.remove()
                h = cap["h"].reshape(-1, cap["h"].shape[-1]).float()
                vals, sel = AG.routing_coefficients(cap["lg"], MU.top_k(model))
                rows = AG.dispatch_rows(sel, vals, E)
                err = {m: torch.zeros_like(h) for m in methods}
                for e, (tx, slot, c) in rows.items():
                    o = AG.expert_fp32(h[tx], Wfp[e]); on2 = (o.double() ** 2).sum(1); cd = c.double()
                    for m in methods:
                        d = o - AG.expert_fp32(h[tx], Wq[m][e]); dn2 = (d.double() ** 2).sum(1)
                        err[m].index_add_(0, tx, d * c[:, None])
                        A = acc[m]
                        A["diag"] += float((cd ** 2 * dn2).sum())
                        A["c2num"] += float((cd ** 2 * dn2).sum()); A["c2den"] += float((cd ** 2 * on2).sum())
                        A["c1num"] += float((cd * dn2).sum()); A["c1den"] += float((cd * on2).sum())
                for m in methods:
                    acc[m]["total"] += float((err[m].double() ** 2).sum())
        rec = {}
        for m in methods:
            A = acc[m]
            rec[m] = dict(total=A["total"], diagonal_c2_part=A["diag"], cross_part=A["total"] - A["diag"],
                          diag_fraction=A["diag"] / A["total"], c2_weighted_expert_err=A["c2num"] / A["c2den"],
                          c1_weighted_expert_err_pooled=A["c1num"] / A["c1den"])
        rec["rel_diff"] = {k: (rec["AGQ"][k] - rec["GPTQ"][k]) / abs(rec["GPTQ"][k])
                           for k in ("total", "diagonal_c2_part", "cross_part", "c2_weighted_expert_err",
                                     "c1_weighted_expert_err_pooled")}
        out["layers"][str(L)] = rec
        print(f"[c2_probe] L{L}: " + ", ".join(f"{k} {v * 100:+.2f}%" for k, v in rec["rel_diff"].items())
              + f" | diag frac GPTQ {rec['GPTQ']['diag_fraction']:.3f}", flush=True)
        del cache, Wfp, Wq, files
        torch.cuda.empty_cache()
    jdump(out, outp)


# ===================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--config", default=PRIMARY)
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    fn = dict(source_review=stage_source_review, inspect=stage_inspect, tests=stage_tests,
              capture_smoke=stage_capture_smoke, hessian_oracle=stage_hessian_oracle,
              unit_regression=stage_unit_regression, expert_pilot=stage_expert_pilot,
              full_capture=stage_full_capture, agq_quantize=stage_agq_quantize, evaluate=stage_evaluate,
              diagnostics=stage_diagnostics, c2_probe=stage_c2_probe)
    if a.stage == "report":
        import report_agq_only_validation as R
        return R.main()
    fn[a.stage](a)


if __name__ == "__main__":
    main()
