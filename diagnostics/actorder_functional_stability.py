"""Functional impact of GPTQ code disagreement (Experiment 1) and dense act-order stabilization (Experiment 2).

Experiment 1 (no GPTQ solves): held-out activations of the BF16 reference model on the historical WikiText-2 windows
(runs_partial_ppl/windows.pt, 128 x 512) are captured at the actual projection inputs of the four target layers
(gate/up: MLP / expert input; down: the actual down_proj input). Saved quantized weight sets are compared on those
identical activations:
  dense  runs_dense_grouping_scope weights (~/moe_cache_dense_grouping_scope/weights), 5 groupings + ORIGINAL;
  MoE    Family D FR s0 weights (~/moe_cache_multigroup_robustness/weights FR_G* / CTRL_FR_ORIGINAL), ALL 60 routed
         experts of layers 1/8/16/23, reference routing (selected experts + BF16 router probabilities) shared by both
         variants; the shared expert is unchanged and cancels in every difference.
Experiment 2: dense GPTQ re-solved from the SAVED historical Hessians under four act-order policies
(moe_calib.actorder_policy; production solver unchanged), installed as BF16 effective weights and evaluated on the same
windows.

Stages: audit | protocol | capture_dense | capture_moe | func_dense | func_moe | curvature | ao_solve | ao_func | analyze
"""
from __future__ import annotations
import argparse, gc, hashlib, itertools, json, math, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
from moe_calib import functional_diff as FD                                 # noqa: E402
from moe_calib import actorder_policy as AP                                 # noqa: E402

OUT = "runs_actorder_functional_stability"
CACHE = os.path.expanduser(os.environ.get("AFS_CACHE", "~/moe_cache_actorder_functional_stability"))
DCACHE = os.path.expanduser("~/moe_cache_dense_grouping_scope")
MCACHE = os.path.expanduser("~/moe_cache_multigroup_robustness")
ICACHE = os.path.expanduser("~/moe_cache_integer_code_mechanism")
DENSE_RUN, MG_RUN, ICM_RUN = "runs_dense_grouping_scope", "runs_multigroup_robustness", "runs_integer_code_mechanism"
GROUPS = [f"G{s}" for s in (20261111, 20261112, 20261113, 20261114, 20261115)]
CONFIGS = GROUPS + ["ORIGINAL"]
DLAY, MLAY = [1, 9, 19, 27], [1, 8, 16, 23]
PROJ = ("gate_proj", "up_proj", "down_proj")
UNITS = PROJ + ("mlp",)
GPTQ = dict(bits=4, group_size=128, percdamp=1.0)
BUCKET_BITS_PRIMARY = 8
BUCKET_BITS_EXPLORATORY = (6, 10)
POLICY_RUNS = ["natural", "fixed", "bucketed8", "disabled"]                 # primary
POLICY_RUNS_EXPL = ["bucketed6", "bucketed10"]                              # exploratory
NB = 10
CHUNK = 4096

DENSE_PRIMARY = [(g, "ORIGINAL") for g in GROUPS[:4]]                      # 4 single-row perturbations
DENSE_NULL = ("G20261115", "ORIGINAL")
MOE_PRIMARY = [(g, "ORIGINAL") for g in GROUPS]
PAIRS = [(a, "ORIGINAL") for a in GROUPS] + list(itertools.combinations(GROUPS, 2)) + [(c_, "BF16") for c_ in CONFIGS]


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def c(*a):
    return os.path.join(CACHE, *a)


def jdump(o, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(o, f, indent=1, default=float)
    os.replace(tmp, path)


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def tsave(o, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(o, path + ".tmp")
    os.replace(path + ".tmp", path)


def sha256_file(path, bs=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def obj_sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), default=float).encode()).hexdigest()


def pk(a, b):
    return f"{a}__{b}"


def code_files():
    return ["moe_calib/functional_diff.py", "moe_calib/actorder_policy.py", "moe_calib/gptq.py", "moe_calib/dense_gptq.py",
            "moe_calib/code_spatial.py", "moe_calib/model_utils.py", "diagnostics/eval_partial_ppl.py",
            "diagnostics/actorder_functional_stability.py"]


def code_hashes():
    return {f: sha256_file(os.path.join(ROOT, f)) for f in code_files()}


def log_code_state(stage):
    """Code hashes are recorded at every stage (changes after the freeze are reported as deviations, not hidden)."""
    path = p("logs", "code_state.json")
    st = jload(path) or {}
    st.setdefault(stage, []).append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), sha256=code_hashes()))
    jdump(st, path)


def check_protocol():
    pr = jload(p("protocol.json"))
    assert pr, "protocol not frozen"
    body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created")}
    assert obj_sha(body) == pr["protocol_sha256"], "protocol.json modified after freeze"
    return pr


def dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def fp32_strict():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


# ---------------------------------------------------------------- weights access
def dense_pristine():
    import dense_grouping_scope as DS
    from safetensors import safe_open
    with safe_open(os.path.join(DS.snapshot_dir(), "model.safetensors"), framework="pt") as f:
        return {L: {k: f.get_tensor(f"model.layers.{L}.mlp.{k}.weight") for k in PROJ} for L in DLAY}


def dense_w(cfg, L):
    return torch.load(os.path.join(DCACHE, "weights", cfg, f"L{L}.pt"))


def dense_codes(cfg, L):
    return torch.load(os.path.join(DCACHE, "codes", cfg, f"L{L}.pt"))


def dense_H(cfg, L):
    return torch.load(os.path.join(DCACHE, "hessians", cfg, f"L{L}.pt"))


def moe_wdir(cfg):
    return os.path.join(MCACHE, "weights", "CTRL_FR_ORIGINAL" if cfg == "ORIGINAL" else f"FR_{cfg}")


def moe_w(cfg, L):
    """{expert: {proj: bf16}} for one historical config (or the BF16 checkpoint for cfg == 'BF16')."""
    if cfg == "BF16":
        return torch.load(c("moe_bf16", f"L{L}.pt"))
    d = torch.load(os.path.join(moe_wdir(cfg), f"L{L}.pt"))
    out = {}
    for k, v in d.items():
        e, pj = k.split(".")
        out.setdefault(int(e), {})[pj] = v
    return out


def ao_dir(policy, cfg):
    return c("ao", policy, cfg)


def policy_spec(run):
    if run.startswith("bucketed"):
        return "bucketed", int(run[len("bucketed"):])
    return run, None


# ---------------------------------------------------------------- audit
def stage_audit(a):
    log_code_state("audit")
    out = dict(dense={}, moe={}, icm={})
    dinv = json.load(open(os.path.join(ROOT, DENSE_RUN, "artifact_inventory.json")))
    for g in CONFIGS:
        ok = {}
        for L in DLAY:
            f = dinv["configs"][g]["files"][str(L)]
            ok[str(L)] = dict(weights=sha256_file(os.path.join(DCACHE, "weights", g, f"L{L}.pt")) == f["weight_sha256"],
                              codes=sha256_file(os.path.join(DCACHE, "codes", g, f"L{L}.pt")) == f["codes_sha256"],
                              hessians=sha256_file(os.path.join(DCACHE, "hessians", g, f"L{L}.pt")) == f["hessian_sha256"])
        r = jload(os.path.join(ROOT, DENSE_RUN, "results", f"{g}.json"))
        out["dense"][g] = dict(files_match_inventory=ok, all_ok=all(all(v.values()) for v in ok.values()),
                               historical_ppl=r["ppl"], valid=r["valid"], installed_weights_sha256=r["installed_weights_sha256"])
    out["dense_inventory_protocol_sha256"] = dinv.get("protocol_sha256")
    for g in CONFIGS:
        r = jload(os.path.join(ROOT, MG_RUN, "results", "CTRL_FR_ORIGINAL.json" if g == "ORIGINAL" else f"FR_{g}.json"))
        files = {str(L): sha256_file(os.path.join(moe_wdir(g), f"L{L}.pt")) == r["weight_files"][str(L)]["sha256"] for L in MLAY}
        out["moe"][g] = dict(weights_dir=moe_wdir(g), weight_files_match_results=files, all_ok=all(files.values()),
                             historical_ppl=r["ppl"], repeat_eval_identical=r["repeat_eval_per_window_identical"],
                             installed_weights_sha256=r["installed_weights_sha256"], n_quantized_experts=r["n_quantized_experts"])
    icm_par = {g: (jload(os.path.join(ROOT, ICM_RUN, "logs", "moe_replay", f"{g}.json")) or {}).get("all_parity") for g in CONFIGS}
    out["icm"] = dict(moe_codes_replay_parity=icm_par, expert_manifest=os.path.join(ICM_RUN, "moe_expert_manifest.json"),
                      codes_dir=os.path.join(ICACHE, "moe_codes"), hessians_dir=os.path.join(ICACHE, "moe_hessians"))
    # G20261115 == ORIGINAL (dense)
    out["dense_null_equal"] = {str(L): all(torch.equal(dense_w("G20261115", L)[k], dense_w("ORIGINAL", L)[k]) for k in PROJ)
                               for L in DLAY}
    # diag(H) inspection for the bucket precision (Hessians only; no solves, no PPL)
    insp = {}
    for L in DLAY:
        HO, HG = dense_H("ORIGINAL", L), dense_H("G20261111", L)
        for hk in ("gu", "down"):
            d, d2 = AP.solver_diag(HO[hk]).double(), AP.solver_diag(HG[hk]).double()
            s = torch.sort(d, descending=True).values
            rec = dict(n=d.numel(), n_dead=int((torch.diag(HO[hk]) == 0).sum()), max=float(s[0]), min=float(s[-1]),
                       log2_dynamic_range=float(torch.log2(s[0] / s[-1])),
                       median_rel_gap_sorted=float(((s[:-1] - s[1:]) / s[:-1]).median()),
                       median_rel_diag_change_G11_vs_ORIGINAL=float(((d2 - d) / d).abs().median()),
                       max_rel_diag_change_G11_vs_ORIGINAL=float(((d2 - d) / d).abs().max()))
            for b in (4, 6, 8, 10, 12):
                k1, k2 = AP.bucket_key(d, b), AP.bucket_key(d2, b)
                rec[f"bits{b}"] = dict(n_unique_keys=int(torch.unique(k1).numel()),
                                       frac_cols_key_changed_G11=float((k1 != k2).float().mean()))
            insp[f"L{L}.{hk}"] = rec
    out["diag_inspection"] = insp
    out["windows"] = dict(path="runs_partial_ppl/windows.pt",
                          sha256=json.load(open(os.path.join(ROOT, "runs_partial_ppl/evaluation_protocol.json")))["windows_sha256"])
    out["all_ok"] = all(v["all_ok"] for v in out["dense"].values()) and all(v["all_ok"] for v in out["moe"].values()) \
        and all(icm_par.values()) and all(out["dense_null_equal"].values())
    jdump(out, p("logs", "audit.json"))
    print(json.dumps({k: out[k] for k in ("all_ok", "dense_null_equal")}), flush=True)
    for k, v in insp.items():
        print(k, v["log2_dynamic_range"], {b: v[f"bits{b}"] for b in (6, 8, 10)})


# ---------------------------------------------------------------- protocol
def stage_protocol(a):
    au = jload(p("logs", "audit.json"))
    assert au and au["all_ok"], "audit failed"
    if os.path.exists(p("protocol.json")):
        raise SystemExit("protocol already frozen")
    if os.path.isdir(p("results")) and os.listdir(p("results")):
        raise SystemExit("act-order results exist: protocol cannot be created")
    import dense_grouping_scope as DS
    dpr = json.load(open(os.path.join(ROOT, DENSE_RUN, "protocol.json")))
    gman = json.load(open(os.path.join(ROOT, DENSE_RUN, "grouping_manifest.json")))
    cal = json.load(open(os.path.join(ROOT, DENSE_RUN, "calibration_manifest.json")))
    dinv = json.load(open(os.path.join(ROOT, DENSE_RUN, "artifact_inventory.json")))["configs"]
    pr = dict(
        title="Functional impact of GPTQ code disagreement and dense act-order stabilization",
        experiments_order="Experiment 1 (functional, no solves) completes before any Experiment 2 PPL evaluation",
        dense=dict(model=DS.MODEL_ID, revision=DS.REVISION, layers=DLAY, projections=list(PROJ),
                   configs=CONFIGS, historical_run=DENSE_RUN, historical_protocol_sha256=dpr["protocol_sha256"],
                   weights_dir=os.path.join(DCACHE, "weights"), hessians_dir=os.path.join(DCACHE, "hessians"),
                   hessian_sha256={g: {str(L): dinv[g]["files"][str(L)]["hessian_sha256"] for L in DLAY} for g in CONFIGS}),
        moe=dict(model="Qwen/Qwen1.5-MoE-A2.7B", layers=MLAY, experts="all 60 routed experts per layer (240)",
                 configs=CONFIGS, weights={g: moe_wdir(g) for g in CONFIGS},
                 sampled_code_subset=os.path.join(ICM_RUN, "moe_expert_manifest.json") + " (20/240; codes, Hessians, act-order)"),
        quantization=dict(bits=GPTQ["bits"], group_size=GPTQ["group_size"], percdamp=GPTQ["percdamp"],
                          percdamp_semantics="damping = 1.0 x mean(diag H)", blocksize=128, solver="moe_calib.gptq.gptq_quantize (unchanged)",
                          install="fp32 W_q -> .to(bfloat16) (model_utils.set_expert_weights cast)",
                          groups="consecutive group_size columns of the EXECUTION order (solver semantics; no static groups)"),
        calibration=dict(manifest_sha256=obj_sha(cal), n_sequences=cal["n_sequences"], seqlen=cal["seqlen"],
                         total_tokens=cal["total_tokens"], fr_token_sha256=cal["fr_token_sha256"],
                         equivalence="Experiment 2 re-solves from the SAVED historical Hessians (same 128 C4 sequences, "
                                     "same order, same K=4 blocks); Hessian sha256 verified before every solve"),
        groupings={g: dict(sha256=gman["groupings"][g]["sha256"]) for g in CONFIGS},
        evaluation=dict(windows="runs_partial_ppl/windows.pt", windows_sha256=au["windows"]["sha256"], n_windows=128, seqlen=512,
                        evaluator="diagnostics/eval_partial_ppl.layerwise_nll (unchanged), batch 8"),
        experiment1=dict(
            reference_model="BF16 original checkpoint (both models); activations captured once and shared by every variant",
            activations=dict(gate_up="input of mlp.gate_proj (dense) / experts[e].gate_proj (MoE) = post-attention-norm hidden "
                                     "state (MoE: tokens routed to e under BF16 routing)",
                             down="actual input of down_proj in the BF16 forward (silu(gate x) * up x, BF16)",
                             routing="BF16 router: softmax(fp32) -> top-4 -> cast to bf16 (norm_topk_prob=False), as the HF block"),
            capture_validation="per-window NLL of the capture forward == historical BF16 baseline bitwise; MoE gate inputs == "
                               "X[routed tokens] bitwise",
            arithmetic="FP32 (TF32 off) on installed BF16 weights and captured BF16 activations; float64 accumulation",
            E_func="||(W_a - W_b) X||_F / ||W_BF16 X||_F over all (routed) tokens; reference = BF16 original weights",
            E_mlp_dense="||MLP_a(X) - MLP_b(X)||_F / ||MLP_BF16(X)||_F, X = captured MLP input, FP32 SiLU-GLU",
            E_moe_expert="per expert: same as E_mlp on the expert's routed tokens (unweighted expert-level)",
            E_moe_agg="||sum_e w_te (E_a,e(x_t) - E_b,e(x_t))||_F / ||MoE_block_BF16(x)||_F (captured full block output incl. "
                      "shared expert) over all 65,536 tokens; shared reference routing (selected experts + BF16 probabilities); "
                      "secondary denominator: routed-only BF16 output",
            secondary=["rmse", "relative to variant b", "per-token diff p95/p99", "global and per-token cosine",
                       "rel Frobenius weight diff", "BF16 element disagreement", "integer-code disagreement (dense all; MoE "
                       "sampled 20 experts only)", "act-order position disagreement", "Hessian perturbation ||dH||_F/||H||_F"],
            pairs=dict(primary_dense=[pk(*x) for x in DENSE_PRIMARY], null_dense=pk(*DENSE_NULL),
                       primary_moe=[pk(*x) for x in MOE_PRIMARY], secondary="all 10 grouping pairs; each config vs BF16"),
            ratios="R_X = MoE / dense for code, BF16, functional (per projection and MLP/MoE-block) and SD(log PPL); primary pair "
                   "sets as above (exploratory cross-model comparison, not a causal decomposition)",
            classification=dict(
                quantity="R_MLP = mean E_moe_agg over the 5 MoE primary pairs / mean E_mlp_dense over the 4 dense primary pairs "
                         "(both pooled over the 4 layers by squared norms); R_code = 17.86 (runs_integer_code_mechanism, "
                         "per projection gate 51.96 / up 51.57 / down 5.06); R_func_p per projection analogous to R_MLP",
                F1="R_MLP <= 3 AND no projection with R_func_p >= 0.5 x R_code_p",
                F2="R_MLP >= 0.5 x R_code (>= 8.93)",
                F3="otherwise, or projection-level ratios conflict (some R_func_p <= 0.25 R_code_p while another >= 0.5 R_code_p)",
                note="the dense/MoE code ratio is on the 20 sampled MoE experts; functional ratios use all 240 experts"),
            curvature=dict(tensors="dense: all 12, 4 primary pairs; MoE: 20 sampled experts x 3 projections, 5 primary pairs",
                           bins=f"{NB} bins by (i) reference (ORIGINAL) execution position, (ii) calibration diag(H_ORIGINAL) "
                                "quantile, (iii) held-out diag(X^T X) quantile; (i) and (ii) coincide up to reversal by "
                                "construction of act-order",
                           energy="diag approximation sum_j ||dW_:j||^2 Hheld_jj vs within-bin full quadratic form and the "
                                  "full tr(dW Hheld dW^T) (non-additivity reported)",
                           split="quadratic-form split of ||dW X||^2 into code-flip elements, same-code (grid-only) elements, cross")),
        experiment2=dict(
            model="dense only", configs=CONFIGS, primary_policies=POLICY_RUNS, exploratory_policies=POLICY_RUNS_EXPL,
            policies=dict(
                natural="argsort(diag H, descending) inside the solver (historical); reused after bitwise re-solve parity",
                fixed="actorder_override = the ORIGINAL configuration's natural permutation for that projection (computed "
                      "from the ORIGINAL Hessian); grouping Hessians, scales, zero-points, error feedback stay live. Cost: "
                      "requires the reference calibration execution (one extra Hessian pass + argsort) and storage of 12 "
                      "permutations; NOT reference-free",
                bucketed=f"actorder_override = moe_calib.actorder_policy.bucketed_perm(own H, bits): float64 frexp mantissa "
                         f"truncated to bits significant bits (floor), descending key, ties by ascending column index "
                         f"(numpy stable argsort, CPU); primary bits = {BUCKET_BITS_PRIMARY} (frozen after inspecting "
                         f"diag(H) only: logs/audit.json diag_inspection), exploratory {list(BUCKET_BITS_EXPLORATORY)}; "
                         "fixed power-of-two boundaries independent of observed min/max; reference-free",
                disabled="gptq_quantize(actorder=False): original column order, groups = consecutive original columns"),
            validation=dict(A="natural re-solve from saved Hessians == historical BF16 weights bitwise (72 tensors) and natural "
                              "PPL per-window == historical bitwise",
                            B="fixed path with each config's own natural permutation == natural output bitwise (72 tensors)",
                            C="bucketed permutation identical across 2 repeated computations and a clone of H; full bucketed "
                              "solve repeated bitwise for one config",
                            D="disabled: info perm == arange, group == column // 128",
                            E="Hessian files sha256 == historical inventory; tokens == 65,536 per projection",
                            F="G20261115 == ORIGINAL bitwise within each policy",
                            G="only the 12 target tensors change; all other parameters digest-identical; restore to pristine",
                            H="repeat evaluation per-window NLL bitwise identical",
                            stop_rule="any failure -> interpretation BLOCKED"),
            primary_metrics=["SD(PPL), SD(log PPL), CV over the 5 groupings per policy (ddof=1)",
                             "R_SD,m = SD_m(log PPL) / SD_natural(log PPL)",
                             "mean PPL, penalty vs BF16, mean change vs natural, best / worst, max pairwise diff",
                             "window-level paired mean NLL difference vs natural per grouping"],
            secondary_metrics=["code / level / BF16 / rel-Frobenius / act-order position / group / scale / zero disagreement vs "
                               "the policy's ORIGINAL", "E_mlp vs the policy's ORIGINAL on Experiment 1 activations"],
            descriptors=dict(
                code_stabilized="mean code disagreement (4 primary pairs) <= 0.5 x natural",
                functional_stabilized="mean E_mlp (4 primary pairs) <= 0.5 x natural",
                ppl_stabilized="R_SD <= 0.5 (descriptive only: 4 singleton perturbations + 1 exact null)",
                ppl_unchanged="R_SD >= 0.75",
                quality_acceptable="mean PPL - mean natural PPL <= 0.10 x (mean natural PPL - BF16 PPL)",
                hypothesis_C="code_stabilized AND functional_stabilized AND ppl_stabilized AND quality_acceptable",
                hypothesis_D="code_stabilized AND (ppl_unchanged OR NOT quality_acceptable)",
                otherwise="intermediate"),
            statistical_limitation="5 groupings = 4 singleton-row perturbations + 1 exact null; SDs are descriptive"),
        artifacts=dict(cache=CACHE, activations=c("act"), ao_weights=c("ao", "<policy>", "<config>", "L<l>.pt")),
        code_sha256_at_freeze=code_hashes(),
    )
    pr["protocol_sha256"] = obj_sha(pr)
    pr["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    jdump(pr, p("protocol.json"))
    print("protocol", pr["protocol_sha256"][:16], flush=True)


# ---------------------------------------------------------------- activation capture
@torch.no_grad()
def capture_forward(model, windows, targets, hooks_for, on_layer_done):
    """eval_partial_ppl.layerwise_nll with hooks on the target layers (hooks only clone; numerics unchanged)."""
    from moe_calib import model_utils as MU
    import eval_partial_ppl as EP
    N, T = windows.shape
    batch = EP.BATCH
    hs = [MU.embed(model, windows[i:i + batch]) for i in range(0, N, batch)]
    for L in range(MU.num_layers(model)):
        with MU.layer_on_gpu(model, L):
            hk = hooks_for(L) if L in targets else []
            try:
                hs = [MU.run_layer(model, L, h) for h in hs]
            finally:
                for h in hk:
                    h.remove()
            if L in targets:
                on_layer_done(L)
    nll, cnt = [], []
    for bi, h in enumerate(hs):
        for j in range(h.shape[0]):
            logits = model.lm_head(model.model.norm(h[j:j + 1]))
            s, n = EP.nll_from_logits(logits, windows[bi * batch + j:bi * batch + j + 1])
            nll.append(float(s[0])); cnt.append(int(n[0]))
        hs[bi] = None
    return np.array(nll, dtype=np.float64), np.array(cnt, dtype=np.int64)


def stage_capture_dense(a):
    check_protocol(); log_code_state("capture_dense")
    import dense_grouping_scope as DS
    import eval_partial_ppl as EP
    man = jload(p("functional_activation_manifest.json")) or {}
    if man.get("dense", {}).get("complete"):
        print("dense capture done"); return
    win, prot = EP.load_windows()
    model, _ = DS.load_model()
    vc = DS.verify_checkpoint(model)
    assert vc["all_equal"], vc
    buf = {}

    def hooks_for(L):
        m = model.model.layers[L].mlp
        buf[L] = dict(X=[], Y=[], Xd=[])

        def f_mlp(mod, inp, out):
            buf[L]["X"].append(inp[0].reshape(-1, inp[0].shape[-1]).detach().cpu().clone())
            buf[L]["Y"].append(out.reshape(-1, out.shape[-1]).detach().cpu().clone())

        def f_down(mod, inp):
            buf[L]["Xd"].append(inp[0].reshape(-1, inp[0].shape[-1]).detach().cpu().clone())
        return [m.register_forward_hook(f_mlp), m.down_proj.register_forward_pre_hook(f_down)]

    files = {}

    def done(L):
        d = {k: torch.cat(v) for k, v in buf.pop(L).items()}
        path = c("act", "dense", f"L{L}.pt")
        tsave(d, path)
        files[str(L)] = dict(path=path, shapes={k: list(v.shape) for k, v in d.items()}, dtype=str(d["X"].dtype),
                             sha256=sha256_file(path), finite=all(bool(torch.isfinite(v.float()).all()) for v in d.values()))
        print(f"[capture dense] L{L} {files[str(L)]['shapes']}", flush=True)

    t0 = time.time()
    nll, cnt = capture_forward(model, win, set(DLAY), hooks_for, done)
    bf = jload(os.path.join(ROOT, DENSE_RUN, "bf16_baseline.json"))
    man["dense"] = dict(model="Qwen2.5-1.5B BF16 original", windows_sha256=prot["windows_sha256"], n_tokens=int(win.numel()),
                        token_order="window-major (window i, position t) -> row i*512+t; batches of 8 windows",
                        files=files, nll_equals_historical_bf16_bitwise=bool(np.array_equal(nll, np.array(bf["per_window_nll"]))),
                        ppl=math.exp(nll.sum() / cnt.sum()), seconds=time.time() - t0, checkpoint_verified=vc["all_equal"],
                        complete=True)
    assert man["dense"]["nll_equals_historical_bf16_bitwise"], "capture forward differs from historical BF16 -> BLOCKED"
    jdump(man, p("functional_activation_manifest.json"))


def stage_capture_moe(a):
    check_protocol(); log_code_state("capture_moe")
    from moe_calib import model_utils as MU
    import eval_partial_ppl as EP
    man = jload(p("functional_activation_manifest.json")) or {}
    if man.get("moe", {}).get("complete"):
        print("moe capture done"); return
    os.environ.pop("MOE_LOAD_LAYERS", None)
    win, prot = EP.load_windows()
    model, _ = MU.load_model()
    E, K = MU.num_experts(model), MU.top_k(model)
    buf, chk = {}, {}

    def hooks_for(L):
        blk = model.model.layers[L].mlp
        buf[L] = dict(X=[], Y=[], RL=[], tok={e: [] for e in range(E)}, slot={e: [] for e in range(E)},
                      w={e: [] for e in range(E)}, Xd={e: [] for e in range(E)}, nb=0)
        gin = {}
        chk[L] = dict(gate_input_equal=True, n_checked=0)
        hs = []
        for e in range(E):
            ex = blk.experts[e]

            def f_gate(mod, inp, e=e):
                gin[e] = inp[0]

            def f_down(mod, inp, e=e):
                buf[L]["Xd"][e].append(inp[0].detach().cpu().clone())
            hs += [ex.gate_proj.register_forward_pre_hook(f_gate), ex.down_proj.register_forward_pre_hook(f_down)]

        def f_blk(mod, inp, out):
            X = inp[0].reshape(-1, inp[0].shape[-1])
            Y, RL = out
            off = sum(x.shape[0] for x in buf[L]["X"])          # global token index offset of this batch
            rw = torch.softmax(RL, dim=1, dtype=torch.float)
            rw, sel = torch.topk(rw, K, dim=-1)
            rw = rw.to(X.dtype)
            mask = torch.nn.functional.one_hot(sel, num_classes=E).permute(2, 1, 0)
            for e in range(E):
                idx, top_x = torch.where(mask[e])
                ok = torch.equal(gin[e], X[top_x]) if e in gin else top_x.numel() == 0
                chk[L]["gate_input_equal"] &= bool(ok)
                chk[L]["n_checked"] += 1
                buf[L]["tok"][e].append((top_x + off).int().cpu())
                buf[L]["slot"][e].append(idx.to(torch.int8).cpu())
                buf[L]["w"][e].append(rw[top_x, idx].detach().cpu().clone())
            gin.clear()
            buf[L]["X"].append(X.detach().cpu().clone())
            buf[L]["Y"].append(Y.reshape(-1, Y.shape[-1]).detach().cpu().clone())
            buf[L]["RL"].append(RL.detach().cpu().clone())
            buf[L]["nb"] += 1
        hs.append(blk.register_forward_hook(f_blk))
        return hs

    files = {}

    def done(L):
        b = buf.pop(L)
        d = dict(X=torch.cat(b["X"]), Y=torch.cat(b["Y"]), router_logits=torch.cat(b["RL"]),
                 experts={e: dict(tok=torch.cat(b["tok"][e]), slot=torch.cat(b["slot"][e]), w=torch.cat(b["w"][e]),
                                  Xd=torch.cat(b["Xd"][e])) for e in range(E)})
        for e in range(E):
            assert d["experts"][e]["Xd"].shape[0] == d["experts"][e]["tok"].numel()
        path = c("act", "moe", f"L{L}.pt")
        tsave(d, path)
        ntok = {e: int(d["experts"][e]["tok"].numel()) for e in range(E)}
        files[str(L)] = dict(path=path, sha256=sha256_file(path), n_tokens=int(d["X"].shape[0]), routed_pairs=sum(ntok.values()),
                             tokens_per_expert=ntok, gate_input_check=chk[L],
                             finite=bool(torch.isfinite(d["X"].float()).all() and torch.isfinite(d["Y"].float()).all()))
        # pristine BF16 expert weights of this layer (from the loaded checkpoint)
        wb = {e: {k: getattr(model.model.layers[L].mlp.experts[e], k).weight.detach().cpu().clone() for k in PROJ} for e in range(E)}
        tsave(wb, c("moe_bf16", f"L{L}.pt"))
        files[str(L)]["bf16_weights_sha256"] = sha256_file(c("moe_bf16", f"L{L}.pt"))
        print(f"[capture moe] L{L} routed {files[str(L)]['routed_pairs']} gate-check {chk[L]}", flush=True)
        del d, wb
        gc.collect()

    t0 = time.time()
    nll, cnt = capture_forward(model, win, set(MLAY), hooks_for, done)
    bf = jload(os.path.join(ROOT, "runs_partial_ppl", "bf16_baseline.json"))
    man["moe"] = dict(model="Qwen1.5-MoE-A2.7B BF16 original", windows_sha256=prot["windows_sha256"], n_tokens=int(win.numel()),
                      top_k=K, n_experts=E, norm_topk_prob=bool(model.config.norm_topk_prob), files=files,
                      nll_equals_historical_bf16_bitwise=bool(np.array_equal(nll, np.array(bf["runs"][0]["per_window_nll"]))),
                      ppl=math.exp(nll.sum() / cnt.sum()), seconds=time.time() - t0, complete=True)
    jdump(man, p("functional_activation_manifest.json"))
    assert man["moe"]["nll_equals_historical_bf16_bitwise"], "MoE capture forward differs from historical BF16 -> BLOCKED"
    assert all(f["gate_input_check"]["gate_input_equal"] for f in files.values())


# ---------------------------------------------------------------- dense functional differences (shared by Exp 1 and 2)
def code_metrics(ra, rb):
    from moe_calib import code_spatial as CS
    Sa, Za = CS.element_grid(ra)
    Sb, Zb = CS.element_grid(rb)
    cd = ra["q"] != rb["q"]
    return dict(frac_code_changed=float(cd.float().mean()),
                frac_level_changed=float(((ra["q"].float() - Za) != (rb["q"].float() - Zb)).float().mean()),
                frac_scale_changed=float((Sa != Sb).float().mean()), frac_zero_changed=float((Za != Zb).float().mean()),
                frac_group_changed=float((ra["group"] != rb["group"]).float().mean()),
                frac_actorder_pos_changed=float((ra["perm"] != rb["perm"]).float().mean()))


@torch.no_grad()
def dense_functional(wsets, pairs, tag):
    """wsets: {name: {L: {proj: bf16}}} incl. 'BF16'. Per layer / unit / pair: FD accumulator results.
    Projection units use (W_a - W_b) X exactly; the MLP unit uses MLP_a(X) - MLP_b(X) (FP32 SiLU-GLU)."""
    fp32_strict()
    D = dev()
    out = {}
    names = sorted({x for pr in pairs for x in pr} | {"BF16"})
    for L in DLAY:
        act = torch.load(c("act", "dense", f"L{L}.pt"), mmap=True)
        X, Xd = act["X"], act["Xd"]
        W = {n: {k: wsets[n][L][k].to(D) for k in PROJ} for n in names}
        accs = {pk(a, b): {u: FD.acc_new() for u in UNITS} for a, b in pairs}
        dW = {pk(a, b): {k: (W[a][k].float() - W[b][k].float()) for k in PROJ} for a, b in pairs}
        t0 = time.time()
        for i in range(0, X.shape[0], CHUNK):
            x = X[i:i + CHUNK].to(D).float()
            xd = Xd[i:i + CHUNK].to(D).float()
            Yp = {n: {} for n in names}
            M = {}
            for n in names:
                g = x @ W[n]["gate_proj"].float().T
                u = x @ W[n]["up_proj"].float().T
                Yp[n]["gate_proj"], Yp[n]["up_proj"] = g, u
                Yp[n]["down_proj"] = xd @ W[n]["down_proj"].float().T
                M[n] = (torch.nn.functional.silu(g) * u) @ W[n]["down_proj"].float().T
            for a, b in pairs:
                key = pk(a, b)
                for k in PROJ:
                    xin = xd if k == "down_proj" else x
                    FD.acc_add(accs[key][k], xin @ dW[key][k].T, Yp[b][k], Yp["BF16"][k])
                FD.acc_add(accs[key]["mlp"], M[a] - M[b], M[b], M["BF16"])
            del Yp, M, x, xd
        out[str(L)] = {key: {u: FD.acc_final(v) for u, v in d.items()} for key, d in accs.items()}
        # weight-level metrics
        for a, b in pairs:
            key = pk(a, b)
            for k in PROJ:
                out[str(L)][key][k].update(FD.weight_metrics(W[a][k].cpu(), W[b][k].cpu(), W["BF16"][k].cpu()))
        print(f"[func dense {tag}] L{L} {time.time() - t0:.0f}s", flush=True)
        del W, dW, act
        torch.cuda.empty_cache()
    return out


def pool_layers(res, key, unit):
    """Pool a pair's unit over layers by squared norms."""
    num = sum(res[str(L)][key][unit]["sumsq_diff"] for L in res)
    den = sum(res[str(L)][key][unit]["sumsq_ref"] for L in res)
    return math.sqrt(num / den) if den > 0 else float("nan")


def stage_func_dense(a):
    check_protocol(); log_code_state("func_dense")
    assert (jload(p("functional_activation_manifest.json")) or {}).get("dense", {}).get("complete")
    path = p("dense_functional_difference.json")
    if jload(path):
        print("done"); return
    W0 = dense_pristine()
    wsets = {"BF16": W0}
    for g in CONFIGS:
        wsets[g] = {L: dense_w(g, L) for L in DLAY}
    res = dense_functional(wsets, PAIRS, "exp1")
    # codes, act-order, Hessian perturbation per pair (grouping pairs only)
    extra = {}
    for L in DLAY:
        R = {g: dense_codes(g, L) for g in CONFIGS}
        H = {g: dense_H(g, L) for g in CONFIGS}
        for a_, b_ in PAIRS:
            if b_ == "BF16":
                continue
            key = pk(a_, b_)
            for k in PROJ:
                hk = "down" if k == "down_proj" else "gu"
                cm = code_metrics(R[a_][k], R[b_][k])
                cm["hessian_rel_fro"] = float((H[a_][hk] - H[b_][hk]).norm() / H[b_][hk].norm())
                res[str(L)][key][k].update(cm)
            res[str(L)][key]["mlp"]["hessian_rel_fro_gu"] = float((H[a_]["gu"] - H[b_]["gu"]).norm() / H[b_]["gu"].norm())
            res[str(L)][key]["mlp"]["hessian_rel_fro_down"] = float((H[a_]["down"] - H[b_]["down"]).norm() / H[b_]["down"].norm())
        del R, H
    pooled = {pk(a_, b_): {u: pool_layers(res, pk(a_, b_), u) for u in UNITS} for a_, b_ in PAIRS}
    jdump(dict(note="per layer -> pair -> unit; E_func relative to BF16-reference output; FP32 arithmetic", per_layer=res,
               pooled_over_layers=pooled), path)
    for a_, b_ in PAIRS[:5]:
        print(pk(a_, b_), {u: f"{v:.3e}" for u, v in pooled[pk(a_, b_)].items()})


# ---------------------------------------------------------------- MoE functional differences
def stage_func_moe(a):
    check_protocol(); log_code_state("func_moe")
    man = jload(p("functional_activation_manifest.json")) or {}
    assert man.get("moe", {}).get("complete")
    fp32_strict()
    D = dev()
    path = p("logs", "moe_functional_raw.json")
    out = jload(path) or dict(layers={})
    em = json.load(open(os.path.join(ROOT, ICM_RUN, "moe_expert_manifest.json")))
    sampled = {(x["layer"], x["expert"]) for x in em["experts"]}
    names = ["BF16"] + CONFIGS
    for L in MLAY:
        if str(L) in out["layers"]:
            continue
        t0 = time.time()
        act = torch.load(c("act", "moe", f"L{L}.pt"), mmap=True)
        X = act["X"]
        T, Hd = X.shape
        W = {n: moe_w(n, L) for n in names}
        E = len(W["BF16"])
        Xg = X.to(D)
        agg = {n: torch.zeros(T, Hd, device=D, dtype=torch.float32) for n in names}
        lay_acc = {pk(a_, b_): {u: FD.acc_new() for u in PROJ} for a_, b_ in PAIRS}
        per_exp = {}
        for e in range(E):
            ex = act["experts"][e]
            tok = ex["tok"].long().to(D)
            if tok.numel() == 0:
                per_exp[str(e)] = dict(tokens=0)
                continue
            x = Xg[tok].float()
            xd = ex["Xd"].to(D).float()
            w = ex["w"].to(D).float()
            We = {n: {k: W[n][e][k].to(D).float() for k in PROJ} for n in names}
            Yp, Mo = {}, {}
            for n in names:
                g = x @ We[n]["gate_proj"].T
                u = x @ We[n]["up_proj"].T
                Yp[n] = dict(gate_proj=g, up_proj=u, down_proj=xd @ We[n]["down_proj"].T)
                Mo[n] = (torch.nn.functional.silu(g) * u) @ We[n]["down_proj"].T
                agg[n].index_add_(0, tok, Mo[n] * w[:, None])
            rec = dict(tokens=int(tok.numel()), sampled=(L, e) in sampled, pairs={})
            for a_, b_ in PAIRS:
                key = pk(a_, b_)
                r = {}
                for k in PROJ:
                    xin = xd if k == "down_proj" else x
                    Dl = xin @ (We[a_][k] - We[b_][k]).T
                    ea = FD.acc_add(FD.acc_new(), Dl, Yp[b_][k], Yp["BF16"][k], keep_tokens=False)
                    FD.acc_add(lay_acc[key][k], Dl, Yp[b_][k], Yp["BF16"][k], keep_tokens=True)
                    r[k] = FD.acc_final(ea, with_tokens=False)
                    r[k].update(FD.weight_metrics(W[a_][e][k], W[b_][e][k], W["BF16"][e][k]))
                em_ = FD.acc_add(FD.acc_new(), Mo[a_] - Mo[b_], Mo[b_], Mo["BF16"], keep_tokens=False)
                r["mlp"] = FD.acc_final(em_, with_tokens=False)
                rec["pairs"][key] = r
            per_exp[str(e)] = rec
            del Yp, Mo, We, x, xd
        # aggregated MoE output (shared routing; shared expert cancels)
        Yref = act["Y"]
        agg_res = {}
        for a_, b_ in PAIRS:
            acc_full, acc_routed = FD.acc_new(), FD.acc_new()
            for i in range(0, T, 8192):
                dlt = agg[a_][i:i + 8192] - agg[b_][i:i + 8192]
                yr = Yref[i:i + 8192].to(D).float()
                FD.acc_add(acc_full, dlt, agg[b_][i:i + 8192], yr)
                FD.acc_add(acc_routed, dlt, agg[b_][i:i + 8192], agg["BF16"][i:i + 8192])
            agg_res[pk(a_, b_)] = dict(vs_block=FD.acc_final(acc_full), vs_routed=FD.acc_final(acc_routed))
        # BF16-compute noise floor of the block (captured BF16 output vs FP32 recomputation incl. shared expert omitted)
        out["layers"][str(L)] = dict(per_expert=per_exp,
                                     pooled_projection={key: {u: FD.acc_final(v) for u, v in d.items()} for key, d in lay_acc.items()},
                                     aggregate=agg_res, seconds=time.time() - t0)
        jdump(out, path)
        print(f"[func moe] L{L} {time.time() - t0:.0f}s  agg G11-ORIG {agg_res[pk('G20261111', 'ORIGINAL')]['vs_block']['E_func']:.3e}",
              flush=True)
        del W, agg, Xg, act, lay_acc
        gc.collect(); torch.cuda.empty_cache()
    # sampled-expert code metrics (aligned subset)
    if "sampled_codes" not in out:
        sc = {}
        for L in MLAY:
            R = {g: torch.load(os.path.join(ICACHE, "moe_codes", g, f"L{L}.pt")) for g in CONFIGS}
            H = {g: torch.load(os.path.join(ICACHE, "moe_hessians", g, f"L{L}.pt")) for g in CONFIGS}
            for e in R["ORIGINAL"]:
                for a_, b_ in PAIRS:
                    if b_ == "BF16":
                        continue
                    for k in PROJ:
                        hk = "down" if k == "down_proj" else "gu"
                        cm = code_metrics(R[a_][e][k], R[b_][e][k])
                        cm["hessian_rel_fro"] = float((H[a_][e][hk] - H[b_][e][hk]).norm() / H[b_][e][hk].norm())
                        sc[f"{L}|{e}|{pk(a_, b_)}|{k}"] = cm
        out["sampled_codes"] = sc
        jdump(out, path)


# ---------------------------------------------------------------- curvature / functional importance
@torch.no_grad()
def gram(Xs, D):
    G = None
    for x in Xs:
        for i in range(0, x.shape[0], 8192):
            xx = x[i:i + 8192].to(D).double()
            G = xx.T @ xx if G is None else G + xx.T @ xx
    return G


def importance_all(dW, flip, Hheld, perm_ref, Hcal_diag):
    C = dW.shape[1]
    pos = torch.empty(C, dtype=torch.long)
    pos[perm_ref.long()] = torch.arange(C)
    bins = dict(exec_position=(pos * NB) // C,
                calib_diag=FD.quantile_bins(Hcal_diag, NB),
                heldout_diag=FD.quantile_bins(torch.diag(Hheld).cpu(), NB))
    r = {b: FD.importance_profile(dW, flip, Hheld, v, NB) for b, v in bins.items()}
    r["split"] = FD.flip_grid_split(dW, flip, Hheld)
    return r


def stage_curvature(a):
    check_protocol(); log_code_state("curvature")
    fp32_strict()
    D = dev()
    path = p("logs", "curvature_raw.json")
    out = jload(path) or dict(dense={}, moe={})
    for L in DLAY:
        if str(L) in out["dense"]:
            continue
        act = torch.load(c("act", "dense", f"L{L}.pt"), mmap=True)
        Hh = dict(gu=gram([act["X"]], D), down=gram([act["Xd"]], D))
        R = {g: dense_codes(g, L) for g in CONFIGS}
        Wt = {g: dense_w(g, L) for g in CONFIGS}
        HO = dense_H("ORIGINAL", L)
        rl = {}
        for va, rf in DENSE_PRIMARY:
            for k in PROJ:
                hk = "down" if k == "down_proj" else "gu"
                dW = Wt[va][k].float() - Wt[rf][k].float()
                flip = R[va][k]["q"] != R[rf][k]["q"]
                rl[f"{pk(va, rf)}|{k}"] = importance_all(dW, flip, Hh[hk], R[rf][k]["perm"], AP.solver_diag(HO[hk]))
        out["dense"][str(L)] = rl
        jdump(out, path)
        print(f"[curv dense] L{L}", flush=True)
        del Hh, act
        torch.cuda.empty_cache()
    em = json.load(open(os.path.join(ROOT, ICM_RUN, "moe_expert_manifest.json")))
    for L in MLAY:
        if str(L) in out["moe"]:
            continue
        act = torch.load(c("act", "moe", f"L{L}.pt"), mmap=True)
        R = {g: torch.load(os.path.join(ICACHE, "moe_codes", g, f"L{L}.pt")) for g in CONFIGS}
        HO = torch.load(os.path.join(ICACHE, "moe_hessians", "ORIGINAL", f"L{L}.pt"))
        Wt = {g: moe_w(g, L) for g in CONFIGS}
        rl = {}
        for x in [x for x in em["experts"] if x["layer"] == L]:
            e = x["expert"]
            ex = act["experts"][e]
            Hh = dict(gu=gram([act["X"][ex["tok"].long()]], D), down=gram([ex["Xd"]], D))
            for va, rf in MOE_PRIMARY:
                for k in PROJ:
                    hk = "down" if k == "down_proj" else "gu"
                    dW = Wt[va][e][k].float() - Wt[rf][e][k].float()
                    flip = R[va][e][k]["q"] != R[rf][e][k]["q"]
                    rl[f"{e}|{pk(va, rf)}|{k}"] = importance_all(dW, flip, Hh[hk], R[rf][e][k]["perm"], AP.solver_diag(HO[e][hk]))
        out["moe"][str(L)] = rl
        jdump(out, path)
        print(f"[curv moe] L{L}", flush=True)
        del Wt, act
        gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------------------- Experiment 2: act-order policies (dense)
def ao_paths(run, g, L):
    return os.path.join(ao_dir(run, g), f"L{L}.pt")


def stage_ao_solve(a):
    pr = check_protocol(); log_code_state("ao_solve")
    from moe_calib import dense_gptq as DG
    import dense_grouping_scope as DS
    import eval_partial_ppl as EP
    assert jload(p("dense_functional_difference.json")) and jload(p("logs", "moe_functional_raw.json")), \
        "Experiment 1 must complete before Experiment 2"
    D = dev()
    runs = POLICY_RUNS + (POLICY_RUNS_EXPL if a.exploratory else [])
    corr = jload(p("actorder_correctness.json")) or dict(A={}, B={}, C={}, D={}, E={}, F={}, G={}, H={})
    W0 = dense_pristine()
    inv = json.load(open(os.path.join(ROOT, DENSE_RUN, "artifact_inventory.json")))["configs"]
    # ---- E: Hessian identity + reference permutations
    Hs = {}
    for g in CONFIGS:
        for L in DLAY:
            fpth = os.path.join(DCACHE, "hessians", g, f"L{L}.pt")
            corr["E"][f"{g}|L{L}|sha"] = sha256_file(fpth) == inv[g]["files"][str(L)]["hessian_sha256"]
    ref_perm = {}
    for L in DLAY:
        HO, RO = dense_H("ORIGINAL", L), dense_codes("ORIGINAL", L)
        corr["E"][f"L{L}|tokens"] = HO["tokens"]["gu"] == HO["tokens"]["down"] == 65536
        for k in PROJ:
            hk = "down" if k == "down_proj" else "gu"
            pn = AP.natural_perm_cpu(HO[hk].to(D))
            ref_perm[(L, k)] = pn
            corr["E"][f"L{L}|{k}|ref_perm_equals_saved_ORIGINAL_perm"] = bool(torch.equal(pn.int(), RO[k]["perm"]))
    assert all(corr["E"].values()), "calibration / Hessian identity failed -> BLOCKED"
    jdump(corr, p("actorder_correctness.json"))

    def solve_cfg(run, g):
        pol, bb = policy_spec(run)
        res = {}
        for L in DLAY:
            path = ao_paths(run, g, L)
            if os.path.exists(path):
                continue
            H = dense_H(g, L)
            Hg = {hk: H[hk].to(D) for hk in ("gu", "down")}
            wl, cl, meta = {}, {}, {}
            for k in PROJ:
                hk = "down" if k == "down_proj" else "gu"
                info = {}
                t0 = time.time()
                Q = AP.solve(W0[L][k], Hg[hk], pol, GPTQ["bits"], GPTQ["group_size"], GPTQ["percdamp"],
                             ref_perm=ref_perm[(L, k)], bucket_bits=bb, info=info)
                ts = time.time() - t0
                q, gid, ex = DG.integer_codes(Q, info, GPTQ["bits"], GPTQ["group_size"])
                wl[k] = Q.to(torch.bfloat16).cpu()
                cl[k] = dict(q=q, group=gid, perm=info["perm"].int(), scale=info["scale"], zero=info["zero"])
                meta[k] = dict(codes_exact=ex, solve_s=ts, finite=bool(torch.isfinite(Q).all()))
                if pol == "disabled":
                    C = Q.shape[1]
                    corr["D"][f"{run}|{g}|L{L}|{k}"] = bool(torch.equal(info["perm"], torch.arange(C))) and \
                        bool(torch.equal(gid.long(), torch.arange(C) // GPTQ["group_size"]))
                if pol == "natural":
                    corr["A"][f"{g}|L{L}|{k}|weights_bitwise"] = bool(torch.equal(wl[k], dense_w(g, L)[k]))
                    # B: fixed path with this config's own natural permutation == natural output
                    Qb = AP.solve(W0[L][k], Hg[hk], "fixed", GPTQ["bits"], GPTQ["group_size"], GPTQ["percdamp"],
                                  ref_perm=info["perm"])
                    corr["B"][f"{g}|L{L}|{k}"] = bool(torch.equal(Qb, Q))
                if pol == "bucketed":
                    p1, p2 = AP.bucketed_perm(H[hk], bb), AP.bucketed_perm(H[hk].clone(), bb)
                    p3 = AP.bucketed_perm(Hg[hk], bb)
                    corr["C"][f"{run}|{g}|L{L}|{k}|perm_repeat"] = bool(torch.equal(p1, p2) and torch.equal(p1, p3)
                                                                         and torch.equal(p1, info["perm"]))
                    if g == "ORIGINAL" and L == DLAY[0]:
                        Q2 = AP.solve(W0[L][k], Hg[hk], pol, GPTQ["bits"], GPTQ["group_size"], GPTQ["percdamp"], bucket_bits=bb)
                        corr["C"][f"{run}|{g}|L{L}|{k}|solve_repeat_bitwise"] = bool(torch.equal(Q2, Q))
            tsave(dict(weights=wl, codes=cl, meta=meta), path)
            del Hg, H
            torch.cuda.empty_cache()
        jdump(corr, p("actorder_correctness.json"))

    for run in runs:
        for g in CONFIGS:
            solve_cfg(run, g)
        # F: null within policy
        corr["F"][run] = all(torch.equal(torch.load(ao_paths(run, "G20261115", L))["weights"][k],
                                         torch.load(ao_paths(run, "ORIGINAL", L))["weights"][k]) for L in DLAY for k in PROJ)
        jdump(corr, p("actorder_correctness.json"))
        print(f"[solve] {run} done; A {all(corr['A'].values()) if corr['A'] else None} B {all(corr['B'].values()) if corr['B'] else None} "
              f"C {all(corr['C'].values()) if corr['C'] else None} D {all(corr['D'].values()) if corr['D'] else None} F {corr['F'][run]}",
              flush=True)
    for blk in ("A", "B", "C", "D", "E", "F"):
        if corr[blk] and not all(corr[blk].values()):
            raise SystemExit(f"mandatory check {blk} failed -> BLOCKED")

    # ---- PPL evaluation
    win, prot = EP.load_windows()
    assert prot["windows_sha256"] == pr["evaluation"]["windows_sha256"]
    model, _ = DS.load_model()
    vc = DS.verify_checkpoint(model)
    assert vc["all_equal"], vc
    snap = DG.snapshot(model, DLAY)
    tgt0, unt0 = DG.target_digests(model, DLAY), DG.untouched_digests(model, DLAY)
    os.makedirs(p("results"), exist_ok=True)
    bf = jload(os.path.join(ROOT, DENSE_RUN, "bf16_baseline.json"))
    for run in runs:
        for g in CONFIGS:
            rp = p("results", f"{run}__{g}.json")
            if (jload(rp) or {}).get("valid"):
                continue
            assert DG.target_digests(model, DLAY) == tgt0
            for L in DLAY:
                DG.set_weights(model, L, torch.load(ao_paths(run, g, L))["weights"])
            inst_ok = all(torch.equal(DG.get_weights(model, L)[k].detach().cpu(), torch.load(ao_paths(run, g, L))["weights"][k])
                          for L in DLAY for k in PROJ)
            dig = DG.installed_digest(model, DLAY)
            unt_ok = DG.untouched_digests(model, DLAY) == unt0
            n1, k1, s1 = DS.evaluate(model, win)
            n2, _, _ = DS.evaluate(model, win)
            rec = EP.ppl_record(n1, k1, s1)
            dig_after = DG.installed_digest(model, DLAY)
            DG.restore(model, snap)
            eq, tot = DG.matches(model, snap)
            G = dict(installed_equals_solver_bf16=inst_ok, untouched_unchanged=unt_ok, installed_unchanged_during_eval=dig == dig_after,
                     restored=eq == tot and DG.target_digests(model, DLAY) == tgt0)
            Hc = dict(repeat_per_window_bitwise=bool(np.array_equal(n1, n2)))
            rec.update(policy=run, config=g, installed_weights_sha256=dig, G=G, H=Hc, valid=all(G.values()) and all(Hc.values()),
                       protocol_sha256=pr["protocol_sha256"])
            pol, _ = policy_spec(run)
            if pol == "natural":
                h = jload(os.path.join(ROOT, DENSE_RUN, "results", f"{g}.json"))
                rec["historical_ppl"] = h["ppl"]
                rec["A_ppl_per_window_equals_historical"] = bool(np.array_equal(n1, np.array(h["per_window_nll"])))
                rec["A_installed_sha_equals_historical"] = dig == h["installed_weights_sha256"]
                rec["valid"] &= rec["A_ppl_per_window_equals_historical"]
            jdump(rec, rp)
            corr["G"][f"{run}|{g}"] = all(G.values())
            corr["H"][f"{run}|{g}"] = Hc["repeat_per_window_bitwise"]
            if pol == "natural":
                corr["A"][f"{g}|ppl_per_window_bitwise"] = rec["A_ppl_per_window_equals_historical"]
            jdump(corr, p("actorder_correctness.json"))
            print(f"[eval] {run} {g} PPL {rec['ppl']:.6f} valid {rec['valid']} ({s1:.0f}s)", flush=True)
            if not rec["valid"]:
                raise SystemExit(f"{run} {g}: mandatory check failed -> BLOCKED")
    n3, _, _ = DS.evaluate(model, win)
    corr["end"] = dict(bf16_reeval_bitwise=bool(np.array_equal(n3, np.array(bf["per_window_nll"]))),
                       untouched_unchanged=DG.untouched_digests(model, DLAY) == unt0,
                       targets_pristine=DG.target_digests(model, DLAY) == tgt0)
    jdump(corr, p("actorder_correctness.json"))
    print("[end]", corr["end"], flush=True)


def stage_ao_func(a):
    check_protocol(); log_code_state("ao_func")
    runs = POLICY_RUNS + (POLICY_RUNS_EXPL if a.exploratory else [])
    path = p("logs", "ao_functional_raw.json")
    out = jload(path) or {}
    W0 = dense_pristine()
    pairs = [(g, "ORIGINAL") for g in GROUPS] + list(itertools.combinations(GROUPS, 2)) + [(g, "BF16") for g in CONFIGS]
    for run in runs:
        if run in out:
            continue
        wsets = {"BF16": W0}
        codes = {}
        for g in CONFIGS:
            wsets[g], codes[g] = {}, {}
            for L in DLAY:
                d = torch.load(ao_paths(run, g, L))
                wsets[g][L], codes[g][L] = d["weights"], d["codes"]
        res = dense_functional(wsets, pairs, run)
        for L in DLAY:
            for a_, b_ in pairs:
                if b_ == "BF16":
                    continue
                for k in PROJ:
                    res[str(L)][pk(a_, b_)][k].update(code_metrics(codes[a_][L][k], codes[b_][L][k]))
        pooled = {pk(a_, b_): {u: pool_layers(res, pk(a_, b_), u) for u in UNITS} for a_, b_ in pairs}
        # code-grid decomposition vs the policy's ORIGINAL (aligned original coordinates)
        from moe_calib import code_spatial as CS
        dec = {}
        for va in GROUPS:
            for L in DLAY:
                for k in PROJ:
                    dec[f"{va}|L{L}|{k}"] = CS.decompose(codes[va][L][k], codes["ORIGINAL"][L][k], wsets[va][L][k], wsets["ORIGINAL"][L][k])
        out[run] = dict(per_layer=res, pooled_over_layers=pooled, decomposition=dec)
        jdump(out, path)
        print(f"[ao func] {run}: E_mlp G11 {pooled[pk('G20261111', 'ORIGINAL')]['mlp']:.3e}", flush=True)
        del wsets, codes
        gc.collect()


def stage_analyze(a):
    import report_actorder_functional_stability as R
    R.main()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--exploratory", action="store_true")
    a = ap.parse_args(argv)
    dict(audit=stage_audit, protocol=stage_protocol, capture_dense=stage_capture_dense, capture_moe=stage_capture_moe,
         func_dense=stage_func_dense, func_moe=stage_func_moe, curvature=stage_curvature, ao_solve=stage_ao_solve,
         ao_func=stage_ao_func, analyze=stage_analyze)[a.stage](a)


if __name__ == "__main__":
    main()
