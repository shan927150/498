"""Loss-alignment diagnostic MVP B: fixed-input, fixed-routing expert-output and router-weighted MoE-output error of
the effective GPTQ weights (FR / SC / FC, allocation seed 0) on 32 frozen WikiText-2 windows.

The BF16 model's own MoE-block inputs, top-4 expert ids and applied router weights at layers 1/8/16/23 are captured
once; every strategy is then evaluated on exactly those tensors (no quantized sequential forward, no re-routing).
B2 is therefore the DIRECT local effect of the quantized expert weights, not the model-level error.

Stages (python diagnostics/loss_alignment_mvp_b.py --stage S):
  protocol   freeze window ids, weight sources + expected digests, metric definitions   (before any measurement)
  scweights  regenerate SC s0 effective weights with the existing E3 solver (partial load); must reproduce the
             stored A_SC_s0 per-expert errors and installed-weight sha256 bitwise; saved under STORE/weights/SC_s0
  capture    BF16 full-model sequential forward over the 128 PPL windows (the evaluator's own batches of 8, so the
             routing is the PPL run's routing); MoE-block inputs / routing / routed outputs kept for the 32 windows
  eval       partial load; BF16 control + FR, SC, FC installed from files; raw sums per expert / layer / window
  report     tables + figures (+ interpretation.md if present)
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, random, resource, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))

OUT = os.path.join(ROOT, "runs_loss_alignment", "mvp_b")
STORE = os.path.expanduser("~/moe_cache_loss_alignment")
LAYERS = [1, 8, 16, 23]
E, K = 60, 4
SEED = 0
N_SEL, SEL_SEED = 32, 20261116
STRATEGIES = ("FR", "SC", "FC")
OCACHE = os.path.expanduser("~/moe_cache")
WIKI_REF = os.path.expanduser("~/moe_cache_inference_routing/bf16_reference_routing.pt")
MIN_TOKENS = 32                              # experts with fewer routed eval tokens are flagged


def WEIGHTS():
    return {
        "FR": dict(dir=os.path.expanduser("~/moe_cache_batch_sensitivity/weights/FR_ORIGINAL"),
                   expected_sha256_from="runs_partial_ppl/ppl_FR_s0.json", provenance="saved by diagnostics/batch_sensitivity.py "
                   "(FR_ORIGINAL), verified there against runs_partial_ppl"),
        "FC": dict(dir=os.path.expanduser("~/moe_cache_batch_sensitivity/weights/FC_ORIGINAL"),
                   expected_sha256_from="runs_partial_ppl/ppl_FC_s0.json", provenance="saved by diagnostics/batch_sensitivity.py "
                   "(FC_ORIGINAL), verified there against runs_partial_ppl"),
        "SC": dict(dir=os.path.join(STORE, "weights", "SC_s0"),
                   expected_sha256_from="runs_shared_covariance_e3/results/A_SC_s0.json",
                   provenance="regenerated here (stage scweights) with the unchanged E3 solver from the frozen SC s0 selection"),
    }


def p(*a):
    return os.path.join(OUT, *a)


def jload(path):
    with open(path) as f:
        return json.load(f)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tsha(t):
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def env():
    import transformers
    return dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                host=platform.node(), omp_threads=os.environ.get("OMP_NUM_THREADS"),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32)


# ===================================================================== protocol
def stage_protocol(a):
    import eval_partial_ppl as EP
    win, eprot = EP.load_windows()
    sel = sorted(random.Random(SEL_SEED).sample(range(EP.N_WIN), N_SEL))
    exp = {c: jload(os.path.join(ROOT, w["expected_sha256_from"]))["installed_weights_sha256"] for c, w in WEIGHTS().items()}
    sc_sel = jload(os.path.join(ROOT, "runs_shared_covariance_e3", "selections", "seed0.json"))
    prot = dict(
        purpose="diagnostic only: fixed-input / fixed-routing local sensitivity of the effective GPTQ expert weights; "
                "no allocator, no tuning, nothing selected on these windows",
        windows=dict(source="runs_partial_ppl/windows.pt (frozen PPL evaluation windows)",
                     all_windows_sha256=eprot["windows_sha256"], selected_protocol_positions=sel,
                     selection_rule=f"sorted(random.Random({SEL_SEED}).sample(range(128), {N_SEL}))",
                     selected_windows_sha256=EP.tensor_fp(win[sel]), tokens_per_window=EP.SEQLEN,
                     tokens_per_layer=N_SEL * EP.SEQLEN),
        model=dict(name="Qwen/Qwen1.5-MoE-A2.7B", layers=LAYERS, n_experts=E, top_k=K, norm_topk_prob=False,
                   reference="original BF16 weights"),
        strategies={c: dict(allocation_seed=SEED, weights_dir=w["dir"], expected_installed_sha256=exp[c],
                            provenance=w["provenance"]) for c, w in WEIGHTS().items()},
        sc_selection=dict(file="runs_shared_covariance_e3/selections/seed0.json", ids_hash=sc_sel["ids_hash"]),
        capture="BF16 full model, sequential forward of all 128 protocol windows in the PPL evaluator's batches of 8 "
                "(identical activations to the PPL run); forward hooks on the 4 Qwen2MoeSparseMoeBlocks; data kept for "
                "the selected windows only",
        precision=dict(primary="expert MLPs recomputed in FP32 from BF16 inputs and BF16-stored weights (original / "
                               "effective GPTQ); router weights = the BF16-cast values the model multiplies by; all squared-"
                               "error sums accumulated in FP64",
                       secondary="the same metrics with BF16-native expert compute (BF16 matmuls, as the module runs)"),
        metrics=dict(
            B0="per expert: sum_t ||f_Q(x_t) - f_BF16(x_t)||^2 / sum_t ||f_BF16(x_t)||^2 over its routed eval tokens; "
               "reported as the equal-expert mean over 240 experts (and pooled)",
            B1="sum_{L,t} sum_{e in top4} w^2 ||delta_e||^2 / sum_{L,t} ||m_BF16||^2  (uncancelled router-weighted energy)",
            B2="sum_{L,t} ||sum_e w delta_e||^2 / sum_{L,t} ||m_BF16||^2, m_BF16 = sum_e w f_BF16_e  (PRIMARY; routed "
               "experts only, shared expert excluded)",
            B3="cancellation ratio = sum ||sum_e w delta_e||^2 / sum sum_e w^2 ||delta_e||^2  (= B2 / B1)",
            P_seq="E3 projection objective on the SAME WikiText tokens: per expert, mean over windows of the relative "
                  "gate/up/down projection error (down input = reference h); equal-expert mean (bridges R0 and B0)"),
        analysis="relative within-metric change vs FR; paired window bootstrap (10,000) of B2 / B1 / pooled B0 "
                 "differences (descriptive: 32 windows, one allocation seed)",
        min_tokens_flag=MIN_TOKENS, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    if os.path.exists(p("protocol.json")):
        old = jload(p("protocol.json"))
        assert old["windows"] == prot["windows"] and old["strategies"] == prot["strategies"], "protocol changed; refusing"
        print("protocol already frozen; identical"); return
    jdump(prot, p("protocol.json"))
    print(f"[protocol] frozen: windows {sel}")


def protocol():
    pr = jload(p("protocol.json"))
    return pr, hashlib.sha256(open(p("protocol.json"), "rb").read()).hexdigest()


# ===================================================================== SC weights
def stage_scweights(a):
    os.environ["MOE_LOAD_LAYERS"] = ",".join(map(str, LAYERS))
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    import e3_allocation as E3
    pr, _ = protocol()
    out_dir = WEIGHTS()["SC"]["dir"]
    if all(os.path.exists(os.path.join(out_dir, f"L{L}.pt")) for L in LAYERS) and os.path.exists(p("sc_weights.json")):
        print("[scweights] exists"); return
    cfg = jload(os.path.join(ROOT, "configs", "frozen_b4.json"))
    g = cfg["gptq"]
    sel = jload(os.path.join(ROOT, "runs_shared_covariance_e3", "selections", "seed0.json"))
    fz = jload(os.path.join(ROOT, "runs_shared_covariance_e3", "freeze.json"))
    ch = sel["chosen"]
    assert E3.warm_hash(ch) == sel["ids_hash"] == fz["sc_selection_hashes"]["0"] == pr["sc_selection"]["ids_hash"]
    ref = jload(os.path.join(ROOT, "runs_shared_covariance_e3", "results", "A_SC_s0.json"))
    model, _ = MU.load_model()
    snap = PQ.snapshot_experts(model, LAYERS)
    untouched0 = PQ.untouched_digests(model, LAYERS)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    q = PQ.quantize_install(model, OCACHE, LAYERS, ch, list(range(2128, 2256)), 4, g["bits"], g["group_size"], g["percdamp"])
    t_q = time.time() - t0
    now = {(r["layer"], r["expert"]): r["mean"] for r in q["results"]}
    old = {(r["layer"], r["expert"]): r["mean"] for r in ref["per_expert_test"]}
    dig = PQ.installed_digest(model, LAYERS)
    untouched = PQ.untouched_digests(model, LAYERS)
    ck = dict(per_expert_test_bitwise=now == old, installed_sha256_equal=dig == ref["installed_weights_sha256"],
              untouched_unchanged=all(untouched[k] == untouched0[k] for k in untouched0),
              rtn_fallbacks=q["counts"]["n_zero_token_quantization_fallback"], n_quantized=q["counts"]["n_experts_quantized_total"],
              nonfinite=len(PQ.nonfinite_targets(model, LAYERS)))
    ok = ck["per_expert_test_bitwise"] and ck["installed_sha256_equal"] and ck["untouched_unchanged"] and \
        ck["rtn_fallbacks"] == 0 and ck["n_quantized"] == 240 and ck["nonfinite"] == 0
    if not ok:
        jdump(dict(checks=ck, ok=False), p("sc_weights_FAILED.json"))
        raise SystemExit(f"SC s0 regeneration does not reproduce the stored run: {ck}")
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for L in LAYERS:
        f = os.path.join(out_dir, f"L{L}.pt")
        PQ.save_layer(model, L, f)
        files[f"L{L}"] = sha256_file(f)
    PQ.restore_experts(model, snap)
    jdump(dict(checks=ck, ok=True, installed_sha256=dig, files=files, quantize_s=t_q, hessian_s=q["hessian_s"],
               gptq_s=q["gptq_s"], peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=rss_gb(),
               env=env()), p("sc_weights.json"))
    print(f"[scweights] SC s0 reproduced bitwise ({t_q:.0f} s); saved to {out_dir}")


# ===================================================================== capture
def stage_capture(a):
    import eval_partial_ppl as EP
    from moe_calib import model_utils as MU
    from moe_calib import loss_alignment as LA
    pr, psha = protocol()
    if os.path.exists(p("capture_manifest.json")):
        print("[capture] exists"); return
    win, eprot = EP.load_windows()
    assert eprot["windows_sha256"] == pr["windows"]["all_windows_sha256"]
    sel = pr["windows"]["selected_protocol_positions"]
    assert EP.tensor_fp(win[sel]) == pr["windows"]["selected_windows_sha256"]
    model, t_load = EP.load_full_model()
    assert not model.config.norm_topk_prob and MU.top_k(model) == K and MU.num_experts(model) == E
    T = EP.SEQLEN
    keep = {L: dict(X=[], idx=[], w32=[], wa=[], routed=[], pos=[]) for L in LAYERS}
    checks = {L: dict(router_logits_bitwise=True, module_output_bitwise=True, fp32_vs_bf16_expert_rel=[],
                      batches_checked=0) for L in LAYERS}
    state = dict(batch_rows=None)

    def make_hook(L):
        def hook(mod, args, output):
            rows = state["batch_rows"]
            want = [j for j, w_ in enumerate(rows) if w_ in sel]
            if not want:
                return
            hs = args[0]
            B_, T_, d = hs.shape
            flat = hs.reshape(-1, d)
            out, logits = output
            lg = mod.gate(flat)
            ck = checks[L]
            ck["router_logits_bitwise"] &= bool(torch.equal(lg, logits))
            idx, w32, wa = LA.router_topk(logits, K, norm_topk_prob=False, out_dtype=flat.dtype)
            # exact HF re-execution (same modules, same op order) of the routed part, then + shared expert
            final = torch.zeros_like(flat)
            mask = torch.nn.functional.one_hot(idx, num_classes=E).permute(2, 1, 0)
            rel = []
            for e in range(E):
                slot, tx = torch.where(mask[e])
                if tx.numel() == 0:
                    continue
                cur = flat[None, tx].reshape(-1, d)
                y = mod.experts[e](cur)
                Wf = {k: getattr(mod.experts[e], k).weight.float() for k in LA.PROJS}
                y32, _ = LA.mlp_fp32(cur.float(), Wf)
                rel.append((float((y.float() - y32).pow(2).sum()), float(y32.pow(2).sum())))
                final.index_add_(0, tx, (y * wa[tx, slot, None]).to(flat.dtype))
            shared = torch.sigmoid(mod.shared_expert_gate(flat)) * mod.shared_expert(flat)
            ck["module_output_bitwise"] &= bool(torch.equal((final + shared).reshape(B_, T_, d), out))
            n_, d_ = map(sum, zip(*rel))
            ck["fp32_vs_bf16_expert_rel"].append(float(np.sqrt(n_ / d_)))
            ck["batches_checked"] += 1
            for j in want:
                sl = slice(j * T_, (j + 1) * T_)
                keep[L]["X"].append(flat[sl].cpu()); keep[L]["idx"].append(idx[sl].to(torch.int16).cpu())
                keep[L]["w32"].append(w32[sl].cpu()); keep[L]["wa"].append(wa[sl].cpu())
                keep[L]["routed"].append(final[sl].cpu()); keep[L]["pos"].append(rows[j])
        return hook

    hooks = [MU.moe_block(model, L).register_forward_hook(make_hook(L)) for L in LAYERS]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        with torch.no_grad():
            N = win.shape[0]
            hs = [MU.embed(model, win[i:i + EP.BATCH]) for i in range(0, N, EP.BATCH)]
            for L in range(max(LAYERS) + 1):                     # nothing after layer 23 is needed
                with MU.layer_on_gpu(model, L):
                    for bi in range(len(hs)):
                        state["batch_rows"] = list(range(bi * EP.BATCH, min(N, (bi + 1) * EP.BATCH)))
                        hs[bi] = MU.run_layer(model, L, hs[bi])
    finally:
        for h in hooks:
            h.remove()
    t_cap = time.time() - t0
    # cross-check against the stored BF16 reference routing of the PPL windows
    ref = torch.load(WIKI_REF, weights_only=False)
    assert ref["windows_sha256"] == eprot["windows_sha256"]
    os.makedirs(STORE, exist_ok=True)
    man = dict(protocol_sha256=psha, files={}, checks={}, capture_s=t_cap, model_load_s=t_load,
               peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=rss_gb(), env=env())
    for L in LAYERS:
        kp = keep[L]
        assert kp["pos"] == sel, (L, kp["pos"])
        d = {k: torch.cat(kp[k]) for k in ("X", "idx", "w32", "wa", "routed")}
        d["window_positions"] = torch.tensor(sel)
        d["seq_id"] = torch.arange(N_SEL).repeat_interleave(T)
        r_idx = ref["idx"][L][sel].reshape(-1, K).to(torch.int16)
        r_pk = ref["pk"][L][sel].reshape(-1, K)
        ck = checks[L]
        ck.update(idx_equal_stored_reference=bool(torch.equal(d["idx"], r_idx)),
                  w32_equal_stored_reference=bool(torch.equal(d["w32"], r_pk)),
                  fp32_vs_bf16_expert_rel_max=float(max(ck["fp32_vs_bf16_expert_rel"])),
                  weights_sum_le_1=bool((d["w32"].sum(1) <= 1 + 1e-6).all()),
                  slot_desc=bool((d["w32"][:, :-1] >= d["w32"][:, 1:]).all()),
                  ids_in_range=bool(int(d["idx"].min()) >= 0 and int(d["idx"].max()) < E),
                  no_repeat_ids=bool((d["idx"].sort(1).values.diff(dim=1) != 0).all()),
                  finite=bool(torch.isfinite(d["X"].float()).all() and torch.isfinite(d["routed"].float()).all()),
                  tokens=int(d["X"].shape[0]))
        del ck["fp32_vs_bf16_expert_rel"]
        f = os.path.join(STORE, f"capture_L{L}.pt")
        torch.save(d, f)
        man["files"][f"L{L}"] = dict(path=f, sha256=sha256_file(f), tensors={k: tsha(v) for k, v in d.items()})
        man["checks"][f"L{L}"] = ck
        print(f"[capture] L{L}: {json.dumps(ck)}", flush=True)
    ok = all(c["router_logits_bitwise"] and c["module_output_bitwise"] and c["idx_equal_stored_reference"] and
             c["w32_equal_stored_reference"] and c["weights_sum_le_1"] and c["slot_desc"] and c["ids_in_range"] and
             c["no_repeat_ids"] and c["finite"] for c in man["checks"].values())
    man["ok"] = ok
    jdump(man, p("capture_manifest.json" if ok else "capture_manifest_FAILED.json"))
    if not ok:
        raise SystemExit("capture checks failed")
    print(f"[capture] done ({t_cap:.0f} s)")


# ===================================================================== eval
def mlp_bf16(X, W, act=torch.nn.functional.silu):
    """BF16-native expert compute (as the nn.Linear modules run); returns FP32 copies."""
    h = act(X @ W["gate_proj"].T) * (X @ W["up_proj"].T)
    return (h @ W["down_proj"].T).float(), h.float()


def run_strategy(model, caps, W_ref32, W_ref16, LA, MU, bf16_native):
    res = {}
    for L in LAYERS:
        c = caps[L]
        wq = [{k: v.detach() for k, v in MU.get_expert_weights(model, L, e).items()} for e in range(E)]
        if bf16_native:
            Wq = [{k: v.to(MU.GPU, torch.bfloat16) for k, v in w.items()} for w in wq]
            Wr, X, mlp = W_ref16[L], c["X"].to(MU.GPU), mlp_bf16
        else:
            Wq = [{k: v.to(MU.GPU, torch.float32) for k, v in w.items()} for w in wq]
            Wr, X, mlp = W_ref32[L], c["X"].to(MU.GPU, torch.float32), LA.mlp_fp32
        idx = c["idx"].to(MU.GPU).long()
        w = c["wa"].to(MU.GPU).float()
        sid = c["seq_id"].to(MU.GPU)
        r = LA.layer_output_errors(X, idx, w, Wr, Wq, seq_id=sid, n_seq=N_SEL, return_M=True, mlp=mlp)
        routed = c["routed"].to(MU.GPU).float()
        r["M_vs_module_routed_rel"] = float((r.pop("M") - routed).norm() / routed.norm())
        res[L] = r
        del Wq
        torch.cuda.empty_cache()
    return res


def stage_eval(a):
    os.environ["MOE_LOAD_LAYERS"] = ",".join(map(str, LAYERS))
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib import loss_alignment as LA
    pr, psha = protocol()
    man = jload(p("capture_manifest.json"))
    assert man["ok"] and man["protocol_sha256"] == psha
    caps = {}
    for L in LAYERS:
        f = man["files"][f"L{L}"]
        assert sha256_file(f["path"]) == f["sha256"], f"capture L{L} changed"
        caps[L] = torch.load(f["path"])
        assert {k: tsha(v) for k, v in caps[L].items()} == f["tensors"]
    t0 = time.time()
    model, _ = MU.load_model()
    t_load = time.time() - t0
    snap = PQ.snapshot_experts(model, LAYERS)
    untouched0 = PQ.untouched_digests(model, LAYERS)
    pristine_digest = PQ.installed_digest(model, LAYERS)
    W_ref32 = {L: [{k: v.to(MU.GPU, torch.float32) for k, v in snap[(L, e)].items()} for e in range(E)] for L in LAYERS}
    W_ref16 = {L: [{k: v.to(MU.GPU) for k, v in snap[(L, e)].items()} for e in range(E)] for L in LAYERS}
    out = dict(protocol_sha256=psha, capture_manifest_sha256=sha256_file(p("capture_manifest.json")),
               pristine_installed_sha256=pristine_digest, runs={}, env=env(), model_load_s=t_load)
    torch.cuda.reset_peak_memory_stats()
    for c in ("BF16",) + STRATEGIES:
        t0 = time.time()
        eq, tot = PQ.experts_match(model, snap)
        assert eq == tot, f"model not pristine before {c}"
        files = {}
        if c != "BF16":
            wd = pr["strategies"][c]["weights_dir"]
            for L in LAYERS:
                f = os.path.join(wd, f"L{L}.pt")
                files[f"L{L}"] = sha256_file(f)
                PQ.install_layer_file(model, L, f)
        dig = PQ.installed_digest(model, LAYERS)
        expected = pristine_digest if c == "BF16" else pr["strategies"][c]["expected_installed_sha256"]
        untouched = PQ.untouched_digests(model, LAYERS)
        ck = dict(installed_sha256=dig, installed_sha256_matches_expected=dig == expected,
                  untouched_unchanged=all(untouched[k] == untouched0[k] for k in untouched0),
                  n_untouched_checked=len(untouched0), nonfinite_targets=len(PQ.nonfinite_targets(model, LAYERS)),
                  experts_changed=tot - PQ.experts_match(model, snap)[0], weight_files_sha256=files)
        assert ck["installed_sha256_matches_expected"] and ck["untouched_unchanged"] and ck["nonfinite_targets"] == 0, (c, ck)
        assert (ck["experts_changed"] == 0) if c == "BF16" else (ck["experts_changed"] == tot), (c, ck)
        rec = {}
        for mode in ("fp32", "bf16native"):
            r = run_strategy(model, caps, W_ref32, W_ref16, LA, MU, bf16_native=mode == "bf16native")
            rec[mode] = {str(L): jsonable(v) for L, v in r.items()}
        PQ.restore_experts(model, snap)
        eq, tot = PQ.experts_match(model, snap)
        ck["restored_pristine"] = eq == tot
        assert ck["restored_pristine"]
        out["runs"][c] = dict(checks=ck, layers=rec, seconds=time.time() - t0)
        f32 = rec["fp32"]
        print(f"[eval] {c}: B2 " + " ".join(f"L{L}={f32[str(L)]['D_sum'] / f32[str(L)]['M_sum']:.3e}" for L in LAYERS)
              + f" ({time.time() - t0:.0f} s)", flush=True)
    out["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    out["peak_cpu_gb"] = rss_gb()
    jdump(out, p("raw_sums.json"))


def jsonable(r):
    o = {}
    for k, v in r.items():
        if isinstance(v, dict):
            o[k] = jsonable(v)
        elif isinstance(v, np.ndarray):
            o[k] = v.tolist()
        else:
            o[k] = v
    return o


# ===================================================================== report
def sdiv(a, b):
    return a / b if b else float("nan")          # 0/0 only for the BF16 zero-error control


def metrics_from_raw(run, mode="fp32"):
    """Normalized metrics + raw sums for one strategy run."""
    from moe_calib import loss_alignment as LA
    lay = run["layers"][mode]
    per_layer, B0, B0n, B0d, Pseq, ntok = {}, [], 0.0, 0.0, [], []
    D = M = U = V = 0.0
    seq = dict(D=np.zeros(N_SEL), M=np.zeros(N_SEL), U=np.zeros(N_SEL))
    exrows = []
    for L in LAYERS:
        r = lay[str(L)]
        ex = {k: np.asarray(v) for k, v in r["experts"].items()}
        exs = {k: np.asarray(v) for k, v in r["experts_seq"].items()}
        ps = LA.per_seq_projection_mean(exs)
        b0 = ex["out_num"] / ex["out_den"]
        per_layer[L] = dict(B0_mean=float(b0.mean()), B0_pooled=float(ex["out_num"].sum() / ex["out_den"].sum()),
                            P_seq_mean=float(np.nanmean(ps)), B1=r["U_sum"] / r["M_sum"], B2=r["D_sum"] / r["M_sum"],
                            B3=sdiv(r["D_sum"], r["U_sum"]), D_sum=r["D_sum"], M_sum=r["M_sum"], U_sum=r["U_sum"],
                            V_sum=r["V_sum"], n_tokens=r["n_tokens"], n_unrouted=r["n_unrouted_experts"],
                            min_expert_tokens=int(ex["tokens"].min()), finite=r["finite"],
                            M_vs_module_routed_rel=r["M_vs_module_routed_rel"])
        B0 += b0.tolist(); Pseq += ps.tolist(); ntok += ex["tokens"].tolist()
        B0n += ex["out_num"].sum(); B0d += ex["out_den"].sum()
        D += r["D_sum"]; M += r["M_sum"]; U += r["U_sum"]; V += r["V_sum"]
        for k in seq:
            seq[k] += np.asarray(r[f"seq_{k}"])
        for e in range(E):
            exrows.append(dict(layer=L, expert=e, tokens=int(ex["tokens"][e]), out_num=ex["out_num"][e],
                               out_den=ex["out_den"][e], B0=b0[e], proj_num=ex["proj_num"][e], proj_den=ex["proj_den"][e],
                               P_pooled=ex["proj_num"][e] / ex["proj_den"][e], P_seq=ps[e],
                               wout_num=ex["wout_num"][e], wout_den=ex["wout_den"][e]))
    tot = dict(B0_mean=float(np.mean(B0)), B0_pooled=B0n / B0d, B0_num=B0n, B0_den=B0d, P_seq_mean=float(np.nanmean(Pseq)),
               B1=U / M, B2=D / M, B3=sdiv(D, U), B2_layer_mean=float(np.mean([per_layer[L]["B2"] for L in LAYERS])),
               D_sum=D, M_sum=M, U_sum=U, V_sum=V, B1_selfden=U / V, n_experts=len(B0),
               n_experts_below_min_tokens=int(sum(t < MIN_TOKENS for t in ntok)), min_expert_tokens=int(min(ntok)))
    return tot, per_layer, exrows, seq


def stage_report(a):
    from moe_calib import loss_alignment as LA
    raw = jload(p("raw_sums.json"))
    pr, _ = protocol()
    cap = jload(p("capture_manifest.json"))
    res = {}
    for mode in ("fp32", "bf16native"):
        res[mode] = {c: metrics_from_raw(raw["runs"][c], mode) for c in raw["runs"]}
    T = {c: res["fp32"][c][0] for c in raw["runs"]}
    # zero-error control
    bf = T["BF16"]
    zero_ok = bf["D_sum"] == 0 and bf["U_sum"] == 0 and bf["B0_num"] == 0
    # comparisons + paired window bootstrap
    comp = {}
    for c in ("SC", "FC"):
        A, B = res["fp32"][c], res["fp32"]["FR"]
        d = {}
        for m in ("B0_mean", "B0_pooled", "P_seq_mean", "B1", "B2", "B3", "B2_layer_mean"):
            d[m] = (A[0][m] - B[0][m]) / B[0][m]
        sa, sb = A[3], B[3]
        d["B2_boot"] = LA.window_bootstrap_ratio_diff(sa["D"], sa["M"], sb["D"], sb["M"])
        d["B1_boot"] = LA.window_bootstrap_ratio_diff(sa["U"], sa["M"], sb["U"], sb["M"])
        d["per_layer_B2"] = {L: (A[1][L]["B2"] - B[1][L]["B2"]) / B[1][L]["B2"] for L in LAYERS}
        d["per_layer_B0_mean"] = {L: (A[1][L]["B0_mean"] - B[1][L]["B0_mean"]) / B[1][L]["B0_mean"] for L in LAYERS}
        d["per_layer_P_seq"] = {L: (A[1][L]["P_seq_mean"] - B[1][L]["P_seq_mean"]) / B[1][L]["P_seq_mean"] for L in LAYERS}
        # per-expert: does the projection change predict the output change?
        ea = {(r["layer"], r["expert"]): r for r in A[2]}
        eb = {(r["layer"], r["expert"]): r for r in B[2]}
        keys = sorted(ea)
        dP = [(ea[k]["P_seq"] - eb[k]["P_seq"]) / eb[k]["P_seq"] for k in keys]
        dB0 = [(ea[k]["B0"] - eb[k]["B0"]) / eb[k]["B0"] for k in keys]
        from scipy.stats import spearmanr
        d["per_expert_spearman_dPseq_dB0"] = float(spearmanr(dP, dB0).correlation)
        d["experts_B0_improved"] = int(sum(x < 0 for x in dB0))
        d["experts_Pseq_improved"] = int(sum(x < 0 for x in dP))
        bn = {m: (res["bf16native"][c][0][m] - res["bf16native"]["FR"][0][m]) / res["bf16native"]["FR"][0][m]
              for m in ("B0_mean", "B1", "B2")}
        d["bf16native"] = bn
        comp[c] = d
    # MVP A aggregates (s0) and stored NLL / PPL
    A_rows = list(csv.DictReader(open(os.path.join(ROOT, "runs_loss_alignment", "mvp_a", "metrics.csv"))))
    agg = {}
    for c in STRATEGIES:
        r = [x for x in A_rows if x["strategy"] == c and x["seed"] == "0" and x["extraction"] == "ORIGINAL"][0]
        agg[c] = {k: float(r[k]) for k in ("R0", "R1_c4", "R1_wiki", "R2_c4", "R2_wiki", "mean_nll", "ppl") if k in r and r[k] != ""}
    Pc = list(csv.DictReader(open(os.path.join(ROOT, "runs_loss_alignment", "mvp_a", "paired_comparisons.csv"))))
    grp = {c: [x for x in Pc if x["strategy"] == c and x["seed"] == "0"][0] for c in ("SC", "FC")}
    out = dict(protocol=pr["windows"], strategies=pr["strategies"], capture_checks=cap["checks"],
               install_checks={c: raw["runs"][c]["checks"] for c in raw["runs"]},
               zero_error_control_ok=zero_ok, metrics={c: T[c] for c in T},
               metrics_bf16native={c: res["bf16native"][c][0] for c in T},
               per_layer={c: {str(L): v for L, v in res["fp32"][c][1].items()} for c in T},
               comparisons=comp, mvp_a_s0=agg, mvp_a_grouped_s0={c: dict(d_nll_grouped_mean=float(grp[c]["d_nll_grouped_mean"]),
                                                                     d_nll_grouped_sd=float(grp[c]["d_nll_grouped_sd"]))
                                                                 for c in grp},
               runtime=dict(capture_s=cap["capture_s"], capture_peak_gpu_gb=cap["peak_gpu_gb"],
                            capture_peak_cpu_gb=cap["peak_cpu_gb"], eval_s={c: raw["runs"][c]["seconds"] for c in raw["runs"]},
                            eval_peak_gpu_gb=raw["peak_gpu_gb"], eval_peak_cpu_gb=raw["peak_cpu_gb"]),
               env=raw["env"])
    jdump(out, p("metrics.json"))
    with open(p("per_layer.csv"), "w", newline="") as f:
        cols = ["strategy", "precision", "layer", "B0_mean", "B0_pooled", "P_seq_mean", "B1", "B2", "B3", "D_sum", "M_sum",
                "U_sum", "V_sum", "n_tokens", "n_unrouted", "min_expert_tokens", "M_vs_module_routed_rel", "finite"]
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for mode in ("fp32", "bf16native"):
            for c in T:
                for L, v in res[mode][c][1].items():
                    wr.writerow(dict(strategy=c, precision=mode, layer=L, **{k: v[k] for k in cols[3:]}))
    with open(p("per_expert.csv"), "w", newline="") as f:
        cols = ["strategy", "layer", "expert", "tokens", "B0", "out_num", "out_den", "P_seq", "P_pooled", "proj_num",
                "proj_den", "wout_num", "wout_den"]
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for c in T:
            for r in res["fp32"][c][2]:
                wr.writerow(dict(strategy=c, **{k: r[k] for k in cols[1:]}))
    figures(out, res)
    write_summary(out)
    print(f"[report] zero-error control {zero_ok}; SC vs FR B2 {100 * comp['SC']['B2']:+.2f} %, FC vs FR B2 {100 * comp['FC']['B2']:+.2f} %")


COL = {"FR": "#2a78d6", "SC": "#e5a000", "FC": "#9b59b6"}


def figures(out, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = p("figures")
    os.makedirs(fd, exist_ok=True)
    A, M = out["mvp_a_s0"], out["metrics"]
    labels = ["R0\n(C4 test)", "R1 wiki", "R2 wiki", "P_seq\n(wiki tokens)", "B0\nexpert out", "B1\nuncancelled", "B2\nMoE out"]
    def val(c):
        return [A[c]["R0"], A[c]["R1_wiki"], A[c]["R2_wiki"], M[c]["P_seq_mean"], M[c]["B0_mean"], M[c]["B1"], M[c]["B2"]]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))
    for i, c in enumerate(("SC", "FC")):
        rel = [100 * (a - b) / b for a, b in zip(val(c), val("FR"))]
        ax.bar(x + (i - 0.5) * 0.38, rel, width=0.36, color=COL[c], label=f"{c} vs FR")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("relative change vs FR (%)  (< 0 = smaller error)")
    ax.set_title("Seed 0: projection → expert output → router-weighted MoE output (each metric vs its own FR value)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "fig1_metric_chain.png"), dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, m in zip(axs, ("B2", "P_seq_mean")):
        for i, c in enumerate(STRATEGIES):
            ax.bar(np.arange(4) + (i - 1) * 0.27, [out["per_layer"][c][str(L)][m] for L in LAYERS], width=0.25,
                   color=COL[c], label=c)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"L{L}" for L in LAYERS])
        ax.set_title({"B2": "router-weighted MoE output error B2", "P_seq_mean": "projection error P_seq (same tokens)"}[m])
        ax.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "fig2_per_layer.png"), dpi=150)
    plt.close(fig)
    e = {c: {(r["layer"], r["expert"]): r for r in res["fp32"][c][2]} for c in STRATEGIES}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, c in zip(axs, ("SC", "FC")):
        ks = sorted(e[c])
        dP = [100 * (e[c][k]["P_seq"] - e["FR"][k]["P_seq"]) / e["FR"][k]["P_seq"] for k in ks]
        dB = [100 * (e[c][k]["B0"] - e["FR"][k]["B0"]) / e["FR"][k]["B0"] for k in ks]
        ax.scatter(dP, dB, s=10, color=COL[c], alpha=0.7)
        lim = max(map(abs, dP + dB))
        ax.plot([-lim, lim], [-lim, lim], color="gray", lw=0.7, ls="--")
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("Δ projection error P_seq vs FR (%)")
        ax.set_title(f"{c} − FR, 240 experts (ρ = {out['comparisons'][c]['per_expert_spearman_dPseq_dB0']:+.2f})")
        ax.grid(alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("Δ expert output error B0 vs FR (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "fig3_per_expert_projection_vs_output.png"), dpi=150)
    plt.close(fig)


def write_summary(out):
    M, A, C = out["metrics"], out["mvp_a_s0"], out["comparisons"]
    L = []
    w = L.append
    rel = lambda a, b: f"{100 * (a - b) / b:+.2f} %"
    w("# Loss-alignment MVP B — fixed-input, fixed-routing expert-output and MoE-output error (seed 0)\n")
    w(f"FR / SC / FC effective GPTQ weights (allocation seed 0) evaluated on the BF16 model's own MoE-block inputs, "
      f"top-4 routing and applied router weights at layers 1/8/16/23 for {N_SEL} frozen WikiText-2 windows "
      f"({N_SEL * 512} tokens per layer; protocol positions {out['protocol']['selected_protocol_positions']}). "
      "No quantized sequential forward: errors do not propagate and routing never changes, so B2 is the direct local "
      "effect of the expert weights, **not** the model-level error.\n")
    w("## Correctness checks\n")
    cc = out["capture_checks"]
    w("| layer | router logits = module (bitwise) | HF block re-execution = module output (bitwise) | ids = stored PPL-run routing | weights = stored | weights sum ≤ 1 | FP32 vs BF16 expert forward (rel) | tokens |")
    w("|---|---|---|---|---|---|---|---|")
    for k, v in cc.items():
        w(f"| {k} | {v['router_logits_bitwise']} | {v['module_output_bitwise']} | {v['idx_equal_stored_reference']} | "
          f"{v['w32_equal_stored_reference']} | {v['weights_sum_le_1']} | {v['fp32_vs_bf16_expert_rel_max']:.2e} | {v['tokens']} |")
    w("")
    w("| run | installed sha256 = expected | untouched params unchanged | non-finite | experts changed | restored pristine |")
    w("|---|---|---|---|---|---|")
    for c, v in out["install_checks"].items():
        w(f"| {c} | {v['installed_sha256_matches_expected']} (`{v['installed_sha256'][:16]}`) | {v['untouched_unchanged']} "
          f"({v['n_untouched_checked']}) | {v['nonfinite_targets']} | {v['experts_changed']} | {v['restored_pristine']} |")
    w(f"\nBF16 control (original weights through the same code): all error numerators exactly 0: **{out['zero_error_control_ok']}**. "
      f"Routed reference m_BF16 (FP32 recompute) vs the module's BF16 routed sum: rel. deviation "
      + ", ".join(f"L{L} {out['per_layer']['FR'][str(L)]['M_vs_module_routed_rel']:.1e}" for L in LAYERS) +
      f". Min routed eval tokens per expert: {min(M[c]['min_expert_tokens'] for c in STRATEGIES)}; experts below "
      f"{MIN_TOKENS} tokens: {max(M[c]['n_experts_below_min_tokens'] for c in STRATEGIES)}.\n")
    w("## Results (FP32 expert compute; relative changes are within-metric)\n")
    g = out["mvp_a_grouped_s0"]
    w("| metric | definition / data | FR | SC | FC | SC vs FR | FC vs FR |")
    w("|---|---|---|---|---|---|---|")
    for key, lab, d in (("R0", "Projection reconstruction R0", "C4 test, equal-expert"),
                        ("R1_c4", "Frequency-weighted R1", "C4-pool frequency"),
                        ("R1_wiki", "Frequency-weighted R1", "WikiText frequency"),
                        ("R2_c4", "Router-gate-weighted R2", "C4-pool Σw²"),
                        ("R2_wiki", "Router-gate-weighted R2", "WikiText Σw²")):
        if key in A["FR"]:
            w(f"| {lab} | {d} | {A['FR'][key]:.6f} | {A['SC'][key]:.6f} | {A['FC'][key]:.6f} | "
              f"{rel(A['SC'][key], A['FR'][key])} | {rel(A['FC'][key], A['FR'][key])} |")
    for key, lab, d in (("P_seq_mean", "Projection error on the B tokens", "E3 objective, WikiText eval tokens, equal-expert"),
                        ("B0_mean", "Nonlinear expert output error B0", "equal-expert mean of per-expert ratios"),
                        ("B0_pooled", "B0 (pooled)", "Σ num / Σ den over experts"),
                        ("B1", "Uncancelled weighted output error B1", "Σ U / Σ ‖m_BF16‖²"),
                        ("B2", "Router-weighted MoE output error B2 (PRIMARY)", "Σ ‖Σ w δ‖² / Σ ‖m_BF16‖²"),
                        ("B3", "Cancellation ratio B3", "Σ ‖Σ w δ‖² / Σ U")):
        w(f"| {lab} | {d} | {M['FR'][key]:.6e} | {M['SC'][key]:.6e} | {M['FC'][key]:.6e} | "
          f"{rel(M['SC'][key], M['FR'][key])} | {rel(M['FC'][key], M['FR'][key])} |")
    w(f"| Mean NLL, ORIGINAL extraction | 128 windows, sequential | {A['FR']['mean_nll']:.6f} | {A['SC']['mean_nll']:.6f} | "
      f"{A['FC']['mean_nll']:.6f} | {A['SC']['mean_nll'] - A['FR']['mean_nll']:+.6f} | {A['FC']['mean_nll'] - A['FR']['mean_nll']:+.6f} |")
    w(f"| PPL, ORIGINAL extraction | | {A['FR']['ppl']:.4f} | {A['SC']['ppl']:.4f} | {A['FC']['ppl']:.4f} | "
      f"{A['SC']['ppl'] - A['FR']['ppl']:+.4f} | {A['FC']['ppl'] - A['FR']['ppl']:+.4f} |")
    w(f"| ΔNLL, 5-grouping mean ± SD (seed 0) | paired within study | | | | {g['SC']['d_nll_grouped_mean']:+.6f} ± "
      f"{g['SC']['d_nll_grouped_sd']:.6f} | {g['FC']['d_nll_grouped_mean']:+.6f} ± {g['FC']['d_nll_grouped_sd']:.6f} |")
    w("\nNLL differences are absolute (nats/token); all other comparison cells are relative changes of that metric.\n")
    w("Paired window bootstrap (32 windows, 10,000 resamples; descriptive, one allocation seed):\n")
    w("| comparison | B2 change [95 % CI] | B1 change [95 % CI] | experts with lower B0 / lower P_seq (of 240) | per-expert Spearman(ΔP_seq, ΔB0) |")
    w("|---|---|---|---|---|")
    for c in ("SC", "FC"):
        b2, b1 = C[c]["B2_boot"], C[c]["B1_boot"]
        w(f"| {c} vs FR | {100 * b2[0]:+.2f} % [{100 * b2[1]:+.2f}, {100 * b2[2]:+.2f}] | {100 * b1[0]:+.2f} % "
          f"[{100 * b1[1]:+.2f}, {100 * b1[2]:+.2f}] | {C[c]['experts_B0_improved']} / {C[c]['experts_Pseq_improved']} | "
          f"{C[c]['per_expert_spearman_dPseq_dB0']:+.2f} |")
    w("\nPer layer (relative change vs FR):\n")
    w("| comparison | metric | L1 | L8 | L16 | L23 |")
    w("|---|---|---|---|---|---|")
    for c in ("SC", "FC"):
        for m, k in (("P_seq", "per_layer_P_seq"), ("B0 mean", "per_layer_B0_mean"), ("B2", "per_layer_B2")):
            w(f"| {c} vs FR | {m} | " + " | ".join(f"{100 * C[c][k][L]:+.2f} %" for L in LAYERS) + " |")
    w("\nBF16-native expert compute (secondary; includes BF16 rounding in both reference and quantized outputs):\n")
    w("| comparison | B0 mean | B1 | B2 |")
    w("|---|---|---|---|")
    for c in ("SC", "FC"):
        b = C[c]["bf16native"]
        w(f"| {c} vs FR | {100 * b['B0_mean']:+.2f} % | {100 * b['B1']:+.2f} % | {100 * b['B2']:+.2f} % |")
    w("\nRaw sums (FP64; layers pooled) — B2 = D/M, B1 = U/M, B3 = D/U:\n")
    w("| run | D = Σ‖Σ w δ‖² | U = Σ Σ w²‖δ‖² | M = Σ‖m_BF16‖² | B0 Σnum | B0 Σden |")
    w("|---|---|---|---|---|---|")
    for c in ("BF16",) + STRATEGIES:
        m = M[c]
        w(f"| {c} | {m['D_sum']:.6e} | {m['U_sum']:.6e} | {m['M_sum']:.6e} | {m['B0_num']:.6e} | {m['B0_den']:.6e} |")
    w("\nPer-layer and per-expert raw numerators / denominators: `per_layer.csv`, `per_expert.csv`.\n")
    w("![](figures/fig1_metric_chain.png)\n\n![](figures/fig2_per_layer.png)\n\n![](figures/fig3_per_expert_projection_vs_output.png)\n")
    rt = out["runtime"]
    w(f"Runtime: capture {rt['capture_s']:.0f} s (peak GPU {rt['capture_peak_gpu_gb']:.1f} GB, CPU {rt['capture_peak_cpu_gb']:.1f} GB); "
      f"eval " + ", ".join(f"{k} {v:.0f} s" for k, v in rt["eval_s"].items()) +
      f" (peak GPU {rt['eval_peak_gpu_gb']:.1f} GB, CPU {rt['eval_peak_cpu_gb']:.1f} GB). Environment: {json.dumps(out['env'])}.\n")
    interp = p("interpretation.md")
    if os.path.exists(interp):
        w(open(interp).read())
    with open(p("summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["protocol", "scweights", "capture", "eval", "report"])
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    dict(protocol=stage_protocol, scweights=stage_scweights, capture=stage_capture, eval=stage_eval,
         report=stage_report)[a.stage](a)


if __name__ == "__main__":
    main()
