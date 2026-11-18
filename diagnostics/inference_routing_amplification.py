"""Inference-time routing amplification MVP (mechanism identification; no quantization, no new weights).

Question: does freezing every MoE layer's top-4 expert INDICES to the BF16 reference (router logits/probabilities
still computed from the current hidden state) reduce the GPTQ accumulation-order-induced spread of the paired
FC - FR WikiText-2 PPL difference?

Stages (python diagnostics/inference_routing_amplification.py --stage S):
  audit     CPU: hash + load + validate the 14 historical weight sets -> artifact_inventory.json, weight_manifest.json
  protocol  freeze protocol.json (refuses to change an existing one)
  gpu       one model load: BF16 plain/record/repeat/self-forced (reference routing), quantized parity (Tests A/C),
            natural preflight on the 10 primary configs, early-stop check, frozen evaluations, references.
            Resumable per run.
  analyze   CPU analysis of the stored runs -> all JSON outputs, decision.json
Large per-run tensors (routing records, token NLL) go to ~/moe_cache_inference_routing; small JSON to OUT.
"""
from __future__ import annotations
import argparse, gc, hashlib, io, json, math, os, resource, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import eval_partial_ppl as EP                                    # noqa: E402

OUT = "runs_inference_routing_amplification"
CACHE = os.path.expanduser(os.environ.get("IRA_CACHE", "~/moe_cache_inference_routing"))
LAYERS_Q = [1, 8, 16, 23]
ORDERS = [20261121, 20261122, 20261123, 20261124, 20261125]
STRATS = ("FR", "FC")
PARITY_CONFIG = "FR_O20261121"                                   # prespecified (first primary config), not outcome-based
PROJ = ("gate_proj", "up_proj", "down_proj")
SHAPES = {"gate_proj": (1408, 2048), "up_proj": (1408, 2048), "down_proj": (2048, 1408)}
HIST_BF16 = "runs_partial_ppl/bf16_baseline.json"


def p(*a):
    return os.path.join(ROOT, OUT, *a)


def c(*a):
    return os.path.join(CACHE, *a)


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def obj_sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def peak_cpu_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def configs():
    """name -> spec. Primary: 5 orders x {FR, FC}. References: ORIGINAL FP32, FP64 canonical."""
    out = {}
    for g in ORDERS:
        for s in STRATS:
            out[f"{s}_O{g}"] = dict(strategy=s, kind="primary", order=g,
                                    result=f"runs_hessian_order_null/results/{s}_O{g}.json")
    for s in STRATS:
        out[f"{s}_ORIGINAL"] = dict(strategy=s, kind="reference_original", order=None,
                                    result=f"runs_multigroup_robustness/results/CTRL_{s}_ORIGINAL.json",
                                    also=f"runs_hessian_order_null/baseline_regression/{s}.json")
        out[f"{s}_FP64"] = dict(strategy=s, kind="reference_fp64", order=None,
                                result=f"runs_fp64_canonical_reproducibility/results/{s}_s0_FP64_CANONICAL.json")
    return out


PRIMARY = [f"{s}_O{g}" for g in ORDERS for s in STRATS]
REFERENCES = ["FR_ORIGINAL", "FC_ORIGINAL", "FR_FP64", "FC_FP64"]


# ---------------------------------------------------------------- audit
def _bf16_checkpoint_experts():
    from safetensors import safe_open
    from moe_calib import model_utils as MU
    path = MU._snapshot_dir(MU.MODEL_NAME)
    wmap = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for L in LAYERS_Q:
        keys = [k for k in wmap if k.startswith(f"model.layers.{L}.mlp.experts.")]
        files = {}
        for k in keys:
            files.setdefault(wmap[k], []).append(k)
        for f, ks in files.items():
            with safe_open(os.path.join(path, f), framework="pt", device="cpu") as sf:
                for k in ks:
                    e, proj = int(k.split(".")[5]), k.split(".")[6]
                    out[(L, e, proj)] = sf.get_tensor(k)
    return out, os.path.basename(path)


def _read(path):
    with open(path, "rb") as f:
        b = f.read()
    return hashlib.sha256(b).hexdigest(), b


def layer_sha_of(d, E=60):
    h = hashlib.sha256()
    for e in range(E):
        for k in PROJ:
            h.update(d[f"{e}.{k}"].contiguous().view(torch.int16).numpy().tobytes())
    return h.hexdigest()


def stage_audit(a):
    cfgs = configs()
    os.makedirs(p("logs"), exist_ok=True)
    bf16, snap_name = _bf16_checkpoint_experts()
    inv, wman = {}, {}
    expected_sel = jload(os.path.join(ROOT, "runs_hessian_order_null/input_manifest.json"))["selections"]
    for name, spec in cfgs.items():
        t0 = time.time()
        r = jload(os.path.join(ROOT, spec["result"]))
        rec = dict(spec=spec, result_sha256=sha256_file(os.path.join(ROOT, spec["result"])), checks={})
        ck = rec["checks"]
        ck["result_exists"] = r is not None
        wd = r["weights_dir"]
        ck["result_valid_flag"] = bool(r.get("valid", r.get("passed", False)))
        ck["n_quantized_experts_240"] = r.get("n_quantized_experts") == 240
        ck["n_projection_tensors_720"] = r.get("n_projection_tensors") == 720
        ck["rtn_fallbacks_0"] = r.get("rtn_fallbacks") == 0
        ck["strategy"] = r.get("strategy") == spec["strategy"]
        if spec["kind"] == "primary":
            ck["order_seed"] = r.get("order_seed") == spec["order"]
            ck["not_repeat"] = r.get("repeat") is False
        if "gptq" in r:
            ck["gptq_settings"] = r["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True)
        if "ids_hash" in r:
            ck["selection_ids_hash"] = r["ids_hash"] == expected_sel[spec["strategy"]]["ids_hash"]
        if "replay_order" in r:          # calibration rows = same seed-0 selection (order permuted only)
            ck["selection_same_rows_as_seed0"] = sorted(set(r["replay_order"])) == \
                sorted(set(expected_sel[spec["strategy"]]["replay_order"]))
        files, digests = {}, {}
        n_t = n_shape = n_finite = n_bf16 = n_diff_bf16 = 0
        with ThreadPoolExecutor(4) as ex:
            reads = dict(zip(LAYERS_Q, ex.map(_read, [os.path.join(wd, f"L{L}.pt") for L in LAYERS_Q])))
        for L in LAYERS_Q:
            fsha, b = reads[L]
            d = torch.load(io.BytesIO(b))
            del b
            exp = r["weight_files"][str(L)]
            ok_keys = set(d) == {f"{e}.{k}" for e in range(60) for k in PROJ}
            for e in range(60):
                for k in PROJ:
                    t = d[f"{e}.{k}"]
                    n_t += 1
                    n_shape += tuple(t.shape) == SHAPES[k]
                    n_bf16 += t.dtype == torch.bfloat16
                    n_finite += bool(torch.isfinite(t.float()).all())
                    n_diff_bf16 += not torch.equal(t, bf16[(L, e, k)])
            lsha = layer_sha_of(d)
            digests[L] = d
            files[str(L)] = dict(path=os.path.join(wd, f"L{L}.pt"), sha256=fsha, expected_sha256=exp["sha256"],
                                 sha256_match=fsha == exp["sha256"], layer_sha256=lsha,
                                 expected_layer_sha256=exp["layer_sha256"], layer_sha256_match=lsha == exp["layer_sha256"],
                                 keys_complete=ok_keys)
        h = hashlib.sha256()
        for L in LAYERS_Q:
            for e in range(60):
                for k in PROJ:
                    h.update(digests[L][f"{e}.{k}"].contiguous().view(torch.int16).numpy().tobytes())
        inst = h.hexdigest()
        del digests
        gc.collect()
        ck["file_sha256_all_match"] = all(f["sha256_match"] for f in files.values())
        ck["layer_sha256_all_match"] = all(f["layer_sha256_match"] for f in files.values())
        ck["keys_complete_240_experts"] = all(f["keys_complete"] for f in files.values())
        ck["installed_weights_sha256_match"] = inst == r["installed_weights_sha256"]
        ck["n_tensors_720"] = n_t == 720
        ck["shapes_ok_720"] = n_shape == 720
        ck["dtype_bf16_720"] = n_bf16 == 720
        ck["finite_720"] = n_finite == 720
        ck["all_720_differ_from_bf16_checkpoint"] = n_diff_bf16 == 720
        if spec.get("also"):
            r2 = jload(os.path.join(ROOT, spec["also"]))
            ck["order_null_O0_regression_identical_weights"] = (
                r2["installed_weights_sha256"] == r["installed_weights_sha256"] and r2.get("passed") is True
                and all(r2["weight_files"][str(L)]["sha256"] == r["weight_files"][str(L)]["sha256"] for L in LAYERS_Q))
        rec.update(weights_dir=wd, files=files, installed_weights_sha256=inst,
                   historical=dict(ppl=r["ppl"], mean_nll=r["mean_nll"], total_tokens=r["total_tokens"],
                                   per_window_nll=r["per_window_nll"]),
                   valid=all(ck.values()), audit_s=time.time() - t0)
        inv[name] = {k: v for k, v in rec.items() if k != "historical"}
        wman[name] = dict(strategy=spec["strategy"], kind=spec["kind"], order=spec["order"], weights_dir=wd,
                          installed_weights_sha256=inst, files={L: dict(sha256=f["sha256"], layer_sha256=f["layer_sha256"])
                                                                for L, f in files.items()},
                          historical_result=spec["result"], historical_ppl=r["ppl"],
                          historical_per_window_nll=r["per_window_nll"])
        print(f"[audit] {name}: valid={rec['valid']} ({rec['audit_s']:.0f} s) "
              f"{[k for k, v in ck.items() if not v]}", flush=True)
    out = dict(checkpoint=dict(model="Qwen/Qwen1.5-MoE-A2.7B", snapshot=snap_name), configs=inv,
               all_valid=all(v["valid"] for v in inv.values()), n_configs=len(inv),
               missing=[k for k, v in inv.items() if not v["checks"].get("result_exists")],
               regenerated_weights=0, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(out, p("artifact_inventory.json"))
    jdump(dict(configs=wman, manifest_sha256=obj_sha(wman)), p("weight_manifest.json"))
    print(f"[audit] all valid: {out['all_valid']}")


# ---------------------------------------------------------------- protocol
def code_hashes():
    fs = ["moe_calib/inference_routing_override.py", "moe_calib/frozen_routing.py", "moe_calib/model_utils.py",
          "moe_calib/partial_model_quant.py", "diagnostics/eval_partial_ppl.py",
          "diagnostics/inference_routing_amplification.py", "tests/test_inference_routing_override.py"]
    return {f: sha256_file(os.path.join(ROOT, f)) for f in fs}


def stage_protocol(a):
    import transformers
    inv = jload(p("artifact_inventory.json"))
    wm = jload(p("weight_manifest.json"))
    assert inv and inv["all_valid"], "artifact audit must pass before freezing"
    _, eprot = EP.load_windows()
    body = dict(
        study="Inference-time routing amplification MVP (mechanism identification)",
        model=dict(name="Qwen/Qwen1.5-MoE-A2.7B", snapshot=inv["checkpoint"]["snapshot"],
                   transformers=transformers.__version__, torch=torch.__version__,
                   moe_block="Qwen2MoeSparseMoeBlock: softmax(fp32) -> topk(4) -> norm_topk_prob=False -> cast BF16 -> "
                             "expert loop 0..59 with index_add_ -> + sigmoid(shared_gate) * shared_expert",
                   n_moe_layers=24, n_experts=60, top_k=4),
        quantized=dict(layers=LAYERS_Q, experts_per_layer=60, gptq=dict(bits=4, group_size=128, percdamp=1.0, act_order=True),
                       fake_quant="BF16 effective weights; everything else BF16 and unchanged"),
        weight_manifest_sha256=wm["manifest_sha256"],
        weight_sets={k: dict(kind=v["kind"], strategy=v["strategy"], order=v["order"],
                             installed_weights_sha256=v["installed_weights_sha256"]) for k, v in wm["configs"].items()},
        primary_configs=PRIMARY, primary_orders=ORDERS, reference_configs=REFERENCES,
        references_excluded_from_primary_sd=True,
        evaluation=dict(windows="runs_partial_ppl/windows.pt", windows_sha256=eprot["windows_sha256"], n_windows=128,
                        seqlen=512, scored_targets=65408, batch=EP.BATCH,
                        forward="historical sequential 24-layer layer-wise offload forward (EP.layerwise_nll call order)",
                        loss="EP.nll_from_logits semantics; PPL = exp(sum NLL / 65408)"),
        routing_recording=dict(layers="all 24 MoE layers", stored="ordered top-4 indices (int16, original slot order), "
                               "top-4 probabilities, 5th probability (margin p4 - p5)",
                               mechanism="forward hook on mlp.gate recomputing softmax+topk on the router logits; "
                                         "block forward unmodified",
                               position_map="batch bi row r -> window bi*8 + r//512, position r%512"),
        intervention=dict(name="BF16-frozen top-k support",
                          rule="selected_experts <- BF16 reference ordered indices (same window/position/layer) in every "
                               "one of the 24 MoE layers; routing weights = CURRENT softmax probs gathered at the forced "
                               "indices; no renormalisation (norm_topk_prob=False); logits not modified",
                          implementation="moe_calib.frozen_routing.make_forced_forward (verbatim class forward with only "
                                         "selected_experts substituted), installed by moe_calib.inference_routing_override",
                          not_frozen=["router logits", "router probabilities", "hidden states", "attention", "expert outputs"]),
        parity_tests=dict(
            A_record_only="BF16 and PARITY_CONFIG: recorder-attached forward vs historical EP.layerwise_nll: identical "
                          "per-window NLL (bitwise), PPL; recorder vs no-recorder per-layer hidden-state hashes",
            B_bf16_self_routing="BF16 forced with its own recorded indices vs natural BF16: bitwise per-layer hidden "
                                "states, final hidden states, per-window and per-token NLL",
            C_quant_self_routing="PARITY_CONFIG forced with its own natural indices vs its natural run: same recorded "
                                 "router selections/probs, bitwise per-layer and final hidden states, per-token NLL, PPL",
            parity_config=PARITY_CONFIG, tolerance="exact (bitwise); no loosening after results",
            failure="ROUTING-INTERVENTION-BLOCKED; stop"),
        primary_metric=dict(definition="R_SD = SD_g(Delta_frozen_g) / SD_g(Delta_natural_g), sample SD (ddof=1) over "
                                       "the 5 orders, Delta_g = PPL(FC_O_g) - PPL(FR_O_g) (negative = FC lower PPL)",
                            natural_values="from the new natural evaluator runs (required to reproduce the historical "
                                           "per-window NLL bitwise); historical SD 0.00287 is not hardcoded"),
        decision_rules={"ROUTING-INTERVENTION-BLOCKED": "any mandatory parity / correctness check fails",
                        "ROUTING-NO-FLIPS": "zero top-4 SET disagreements vs BF16 across all 10 primary configs and "
                                            "all 24 layers (natural routing) -> stop before frozen evaluation",
                        "ROUTING-AMPLIFICATION-SUPPORTED": "R_SD <= 0.5", "ROUTING-PARTIAL-CONTRIBUTION": "0.5 < R_SD < 0.9",
                        "ROUTING-NO-SUBSTANTIAL-REDUCTION": "0.9 <= R_SD <= 1.1", "ROUTING-VARIABILITY-INCREASED": "R_SD > 1.1",
                        "note": "practical screening criteria on 5 deterministic orders of one seed, not hypothesis tests"},
        secondary=["SD of FR and FC PPL under both regimes", "mean / range / sign stability of Delta",
                   "delta_g = Delta_natural_g - Delta_frozen_g", "fraction of windows improved/worsened by freezing",
                   "per-layer set and order-only disagreement, first divergence layer, per-window distribution",
                   "p4-p5 margin distributions (BF16 and quantized) for changed vs stable sets",
                   "token-level NLL concentration vs ORIGINAL (Analysis A), local and causal-prefix routing "
                   "association (B, C), expert counts and forced-probability distributions"],
        token_alignment="token NLL j (0..510) is the loss of predicting token j+1 from the logit at position j; the "
                        "'local' routing event of target j is at input position j (all 24 layers); its causal prefix "
                        "is positions 0..j-1",
        early_stop="if all primary configs show zero top-4 set disagreement vs BF16 in all layers, stop (ROUTING-NO-FLIPS)",
        out_of_scope="no requantisation, no new orders/seeds, no router/shared-expert changes, no threshold tuning",
        code_sha256_at_freeze=code_hashes())
    body["protocol_sha256"] = obj_sha(body)
    if os.path.exists(p("protocol.json")):
        old = jload(p("protocol.json"))
        ob = {k: v for k, v in old.items() if k not in ("protocol_sha256", "created", "code_sha256_at_freeze")}
        nb = {k: v for k, v in body.items() if k not in ("protocol_sha256", "created", "code_sha256_at_freeze")}
        assert obj_sha(ob) == obj_sha(nb), "protocol already frozen with different conditions; refusing"
        print("protocol already frozen"); return
    body["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    jdump(body, p("protocol.json"))
    print(f"[protocol] frozen {body['protocol_sha256'][:16]}")


def check_protocol():
    pr = jload(p("protocol.json"))
    assert pr, "protocol not frozen"
    b = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created")}
    assert obj_sha(b) == pr["protocol_sha256"], "protocol.json modified"
    return pr


# ---------------------------------------------------------------- GPU helpers
class Timer:
    def __init__(self):
        self.t = {}

    def add(self, k, v):
        self.t.setdefault(k, []).append(v)


def run_file(name, mode):
    return c("runs", f"{name}_{mode}.pt")


def res_file(name, mode):
    return p("results", f"{name}_{mode}.json")


def save_run(name, mode, sess, r, extra):
    os.makedirs(c("runs"), exist_ok=True)
    os.makedirs(p("results"), exist_ok=True)
    d = dict(token_nll=torch.from_numpy(r["token_nll"]), dispatch=sess.dispatch.clone() if sess else None)
    if sess is not None and sess.record:
        d.update(idx=sess.idx, pk=sess.pk, p_next=sess.p_next, p_forced=sess.p_forced)
    tmp = run_file(name, mode) + ".tmp"
    torch.save(d, tmp)
    os.replace(tmp, run_file(name, mode))
    rec = EP.ppl_record(r["nll"], r["cnt"], extra.pop("eval_s"))
    rec.update(config=name, mode=mode, final_hidden_sha256=r["final_sha"], layer_hidden_sha256=r["layer_sha"],
               run_file=run_file(name, mode), run_file_sha256=sha256_file(run_file(name, mode)), **extra)
    jdump(rec, res_file(name, mode))
    return rec


def gate_digest(model):
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    return [PQ._digest(MU.moe_block(model, L).gate.weight.view(torch.int16)) for L in range(MU.num_layers(model))]


def install(model, name, wm, snap):
    """Install a historical weight set from its files (sha-verified), verify digests. Returns seconds."""
    from moe_calib import partial_model_quant as PQ
    t0 = time.time()
    eq, tot = PQ.experts_match(model, snap)
    assert eq == tot, f"model not pristine before {name}"
    ent = wm["configs"][name]
    for L in LAYERS_Q:
        path = os.path.join(ent["weights_dir"], f"L{L}.pt")
        fsha, b = _read(path)
        assert fsha == ent["files"][str(L)]["sha256"], f"{name} L{L} file sha changed"
        d = torch.load(io.BytesIO(b))
        del b
        from moe_calib import model_utils as MU
        for e in range(60):
            MU.set_expert_weights(model, L, e, {k: d[f"{e}.{k}"] for k in PROJ})
        del d
        assert PQ.layer_digest(model, L) == ent["files"][str(L)]["layer_sha256"]
    assert PQ.installed_digest(model, LAYERS_Q) == ent["installed_weights_sha256"]
    eq, _ = PQ.experts_match(model, snap)
    assert eq == 0, f"{name}: {eq} projections still equal BF16 after install"
    return time.time() - t0


def restore(model, snap):
    from moe_calib import partial_model_quant as PQ
    PQ.restore_experts(model, snap)
    eq, tot = PQ.experts_match(model, snap)
    assert eq == tot


def evaluate(model, win, mode, ref=None, record=True, layer_hashes=False, plain=False):
    """plain=True: the historical EP.layerwise_nll itself (no session). Otherwise IRO.layerwise_eval with a session."""
    from moe_calib import inference_routing_override as IRO
    from moe_calib import frozen_routing as FRT
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    if plain:
        FRT.assert_no_override(model)
        nll, cnt = EP.layerwise_nll(model, win)
        return None, dict(nll=nll, cnt=cnt), time.time() - t0, torch.cuda.max_memory_allocated() / 1e9
    if mode is None:                                   # same evaluator, no session / hooks at all
        FRT.assert_no_override(model)
        r = IRO.layerwise_eval(model, win, EP.BATCH, None, layer_hashes=layer_hashes)
        return None, r, time.time() - t0, torch.cuda.max_memory_allocated() / 1e9
    sess = IRO.new_session(model, win, EP.BATCH, mode=mode, ref_idx=ref, record=record)
    with IRO.attach(model, sess):
        r = IRO.layerwise_eval(model, win, EP.BATCH, sess, layer_hashes=layer_hashes)
    if mode == "force":
        assert (sess.n_forced_calls == len(range(0, win.shape[0], EP.BATCH))).all(), sess.n_forced_calls
    assert (sess.n_gate_calls == len(range(0, win.shape[0], EP.BATCH))).all()
    return sess, r, time.time() - t0, torch.cuda.max_memory_allocated() / 1e9


def nll_equal(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return dict(bitwise=bool(np.array_equal(a, b)), max_abs_diff=float(np.max(np.abs(a - b))))


def ppl_of(nll):
    return math.exp(float(np.sum(nll)) / 65408)


# ---------------------------------------------------------------- GPU stage
def stage_gpu(a):
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib import inference_routing_override as IRO
    pr = check_protocol()
    wm = jload(p("weight_manifest.json"))
    assert wm["manifest_sha256"] == pr["weight_manifest_sha256"]
    win, eprot = EP.load_windows()
    assert eprot["windows_sha256"] == pr["evaluation"]["windows_sha256"]
    os.makedirs(p("logs"), exist_ok=True)
    os.makedirs(c("runs"), exist_ok=True)
    cost = jload(p("logs", "gpu_cost_raw.json")) or dict(runs={}, loads=[])
    t0 = time.time()
    model, _ = MU.load_model()
    cost["loads"].append(dict(model_load_s=time.time() - t0, at=time.strftime("%H:%M:%S")))
    snap = PQ.snapshot_experts(model, LAYERS_Q)
    t1 = time.time()
    untouched0 = PQ.untouched_digests(model, LAYERS_Q)
    cost["untouched_digest_s"] = time.time() - t1
    gates0 = gate_digest(model)
    hb = jload(os.path.join(ROOT, HIST_BF16))
    val = jload(p("routing_override_validation.json")) or {}

    def log_cost(key, s, peak):
        cost["runs"][key] = dict(eval_s=s, peak_gpu_gb=peak, peak_cpu_gb=peak_cpu_gb())
        jdump(cost, p("logs", "gpu_cost_raw.json"))

    # ---------- BF16 reference
    if not os.path.exists(c("bf16_reference_routing.pt")):
        _, rp, s, pk = evaluate(model, win, None, plain=True)
        log_cost("BF16_plain", s, pk)
        s1, r1, s, pk = evaluate(model, win, "natural", layer_hashes=True)
        log_cost("BF16_record_1", s, pk)
        s2, r2, s, pk = evaluate(model, win, "natural")
        log_cost("BF16_record_2", s, pk)
        _, r0, s, pk = evaluate(model, win, None, layer_hashes=True)   # hooks off, same evaluator
        log_cost("BF16_nohook_hashes", s, pk)
        rin, rrep = IRO.validate_indices(s1.idx, 60)
        ref = dict(idx=s1.idx.clone(), pk=s1.pk.clone(), p_next=s1.p_next.clone(), token_nll=torch.from_numpy(r1["token_nll"]),
                   dispatch=s1.dispatch.clone(), windows_sha256=eprot["windows_sha256"],
                   layout="[layer 0..23, window 0..127 (protocol order), position 0..511, slot 0..3]")
        torch.save(ref, c("bf16_reference_routing.pt"))
        fsha = sha256_file(c("bf16_reference_routing.pt"))
        hist = hb["runs"][0]["per_window_nll"]
        checks = dict(
            exactly_4_per_token=tuple(s1.idx.shape) == (24, 128, 512, 4), indices_in_range=rin, no_repeated_ids=rrep,
            slot_order_is_descending_prob=bool((s1.pk[..., :-1] >= s1.pk[..., 1:]).all()),
            p4_ge_p5=bool((s1.pk[..., -1] >= s1.p_next).all()),
            probs_finite=bool(torch.isfinite(s1.pk).all() and torch.isfinite(s1.p_next).all()),
            all_24_layers_recorded=bool((s1.n_gate_calls == 16).all()),
            dispatch_equals_index_counts=bool(torch.equal(s1.dispatch, IRO.expert_counts(s1.idx, 60))),
            record_vs_plain_per_window_bitwise=nll_equal(r1["nll"], rp["nll"])["bitwise"],
            record_vs_nohook_layer_hashes_equal=r1["layer_sha"] == r0["layer_sha"] and r1["final_sha"] == r0["final_sha"],
            plain_vs_historical_bf16_per_window_bitwise=nll_equal(rp["nll"], hist)["bitwise"],
            ppl_equals_historical=ppl_of(rp["nll"]) == hb["ppl"],
            repeat_recording_identical_indices=bool(torch.equal(s1.idx, s2.idx)),
            repeat_recording_identical_probs=bool(torch.equal(s1.pk, s2.pk) and torch.equal(s1.p_next, s2.p_next)),
            repeat_per_window_bitwise=nll_equal(r1["nll"], r2["nll"])["bitwise"],
            token_nll_sums_equal_window_nll=bool(np.array_equal(r1["token_nll"].sum(1), r1["nll"])))
        margin = (s1.pk[..., 3] - s1.p_next)
        man = dict(file=c("bf16_reference_routing.pt"), file_sha256=fsha, idx_sha256=hashlib.sha256(
                       s1.idx.numpy().tobytes()).hexdigest(), shape=list(s1.idx.shape), dtype="int16",
                   storage_bytes=os.path.getsize(c("bf16_reference_routing.pt")),
                   ppl=ppl_of(r1["nll"]), historical_ppl=hb["ppl"], per_window_nll=r1["nll"].tolist(),
                   margin_quantiles={str(q): float(torch.quantile(margin.flatten()[::7].double(), q))
                                     for q in (0.001, 0.01, 0.05, 0.1, 0.5)},
                   frac_margin_zero=float((margin == 0).double().mean()),
                   expert_counts=IRO.expert_counts(s1.idx, 60).tolist(), checks=checks, valid=all(checks.values()),
                   protocol_sha256=pr["protocol_sha256"], created=time.strftime("%Y-%m-%d %H:%M:%S"))
        jdump(man, p("bf16_routing_manifest.json"))
        print(f"[bf16] reference routing recorded, PPL {man['ppl']:.6f}, checks {checks}", flush=True)
        val["A_bf16"] = dict(record_vs_plain=nll_equal(r1["nll"], rp["nll"]),
                             layer_hashes_equal=checks["record_vs_nohook_layer_hashes_equal"],
                             ppl_record=ppl_of(r1["nll"]), ppl_plain=ppl_of(rp["nll"]),
                             passed=checks["record_vs_plain_per_window_bitwise"] and checks["record_vs_nohook_layer_hashes_equal"])
        # Test B: BF16 forced with its own reference
        sb, rb, s, pk = evaluate(model, win, "force", ref=s1.idx.clone(), layer_hashes=True)
        log_cost("BF16_self_forced", s, pk)
        save_run("BF16", "frozen", sb, rb, dict(eval_s=s, peak_gpu_gb=pk))
        val["B_bf16_self_routing"] = dict(
            per_window=nll_equal(rb["nll"], r1["nll"]), token_nll_bitwise=bool(np.array_equal(rb["token_nll"], r1["token_nll"])),
            layer_hashes_equal=rb["layer_sha"] == r1["layer_sha"], final_hidden_equal=rb["final_sha"] == r1["final_sha"],
            natural_choice_under_forcing_equal=bool(torch.equal(sb.idx, s1.idx)),
            forced_probs_equal_natural_topk=bool(torch.equal(sb.p_forced, s1.pk)),
            dispatch_equal=bool(torch.equal(sb.dispatch, s1.dispatch)), forced_calls_per_layer=sb.n_forced_calls.tolist())
        val["B_bf16_self_routing"]["passed"] = all(v for k, v in val["B_bf16_self_routing"].items()
                                                   if isinstance(v, bool)) and val["B_bf16_self_routing"]["per_window"]["bitwise"]
        jdump(val, p("routing_override_validation.json"))
        del s1, s2, sb, r0, r1, r2, rb
        gc.collect()
    ref = torch.load(c("bf16_reference_routing.pt"))
    bman = jload(p("bf16_routing_manifest.json"))
    assert sha256_file(c("bf16_reference_routing.pt")) == bman["file_sha256"] and bman["valid"], "BF16 reference invalid"
    ref_idx = ref["idx"]
    if not (val.get("A_bf16", {}).get("passed") and val.get("B_bf16_self_routing", {}).get("passed")):
        raise SystemExit("BF16 parity failed -> ROUTING-INTERVENTION-BLOCKED")

    def run_cfg(name, modes, extra_parity=False):
        todo = [m for m in modes if not (jload(res_file(name, m)) or {}).get("valid")]
        if not todo and not extra_parity:
            return
        t_inst = install(model, name, wm, snap)
        cost["runs"][f"{name}_install"] = dict(install_s=t_inst)
        hist = wm["configs"][name]["historical_per_window_nll"]
        common = dict(installed_weights_sha256=wm["configs"][name]["installed_weights_sha256"],
                      protocol_sha256=pr["protocol_sha256"], strategy=wm["configs"][name]["strategy"],
                      kind=wm["configs"][name]["kind"], order=wm["configs"][name]["order"])
        nat_sess = nat_r = None
        if extra_parity:
            _, rp, s, pk = evaluate(model, win, None, plain=True)
            log_cost(f"{name}_plain", s, pk)
            sn, rn, s, pk = evaluate(model, win, "natural", layer_hashes=True)
            log_cost(f"{name}_record_hashes", s, pk)
            _, r0, s, pk = evaluate(model, win, None, layer_hashes=True)
            log_cost(f"{name}_nohook_hashes", s, pk)
            sc, rc, s, pk = evaluate(model, win, "force", ref=sn.idx.clone(), layer_hashes=True)
            log_cost(f"{name}_self_forced", s, pk)
            val["A_quant"] = dict(config=name, record_vs_plain=nll_equal(rn["nll"], rp["nll"]),
                                  plain_vs_historical=nll_equal(rp["nll"], hist),
                                  layer_hashes_equal=rn["layer_sha"] == r0["layer_sha"] and rn["final_sha"] == r0["final_sha"])
            val["A_quant"]["passed"] = (val["A_quant"]["record_vs_plain"]["bitwise"] and val["A_quant"]["layer_hashes_equal"]
                                        and val["A_quant"]["plain_vs_historical"]["bitwise"])
            val["C_quant_self_routing"] = dict(
                config=name, per_window=nll_equal(rc["nll"], rn["nll"]),
                token_nll_bitwise=bool(np.array_equal(rc["token_nll"], rn["token_nll"])),
                layer_hashes_equal=rc["layer_sha"] == rn["layer_sha"], final_hidden_equal=rc["final_sha"] == rn["final_sha"],
                selections_equal=bool(torch.equal(sc.idx, sn.idx)), router_probs_equal=bool(torch.equal(sc.pk, sn.pk)),
                forced_probs_equal_natural_topk=bool(torch.equal(sc.p_forced, sn.pk)),
                dispatch_equal=bool(torch.equal(sc.dispatch, sn.dispatch)), ppl_equal=ppl_of(rc["nll"]) == ppl_of(rn["nll"]),
                forced_calls_per_layer=sc.n_forced_calls.tolist())
            v = val["C_quant_self_routing"]
            v["passed"] = all(x for x in v.values() if isinstance(x, bool)) and v["per_window"]["bitwise"]
            jdump(val, p("routing_override_validation.json"))
            print(f"[parity] A_quant {val['A_quant']['passed']}  C {v['passed']}", flush=True)
            if not (val["A_quant"]["passed"] and v["passed"]):
                restore(model, snap)
                raise SystemExit("quantized parity failed -> ROUTING-INTERVENTION-BLOCKED")
            nat_sess, nat_r = sn, rn
            del sc, rc, r0, rp
        for m in todo:
            if m == "natural":
                if nat_sess is None:
                    nat_sess, nat_r, s, pk = evaluate(model, win, "natural")
                else:
                    s, pk = cost["runs"][f"{name}_record_hashes"]["eval_s"], cost["runs"][f"{name}_record_hashes"]["peak_gpu_gb"]
                log_cost(f"{name}_natural", s, pk)
                reg = nll_equal(nat_r["nll"], hist)
                rec = save_run(name, "natural", nat_sess, nat_r, dict(eval_s=s, peak_gpu_gb=pk, historical_regression=reg,
                                                                       valid=True, **common))
            else:
                fs, fr, s, pk = evaluate(model, win, "force", ref=ref_idx)
                log_cost(f"{name}_frozen", s, pk)
                ck = dict(dispatch_equals_bf16=bool(torch.equal(fs.dispatch, ref["dispatch"])))
                rec = save_run(name, "frozen", fs, fr, dict(eval_s=s, peak_gpu_gb=pk, checks=ck, valid=all(ck.values()),
                                                             **common))
                del fs, fr
            print(f"[{m}] {name}: PPL {rec['ppl']:.6f} ({rec['eval_s']:.0f} s)"
                  + (f"  hist bitwise {rec['historical_regression']['bitwise']}" if m == "natural" else ""), flush=True)
        assert PQ.installed_digest(model, LAYERS_Q) == wm["configs"][name]["installed_weights_sha256"], "weights changed"
        assert gate_digest(model) == gates0, "router weights changed"
        restore(model, snap)
        nat_sess = nat_r = None
        gc.collect()

    # ---------- quantized parity (Tests A, C) + its natural run
    if not (val.get("A_quant", {}).get("passed") and val.get("C_quant_self_routing", {}).get("passed")):
        run_cfg(PARITY_CONFIG, ["natural"], extra_parity=True)
    # ---------- natural preflight on the 10 primary configs
    for name in PRIMARY:
        run_cfg(name, ["natural"])
    n_flip = 0
    for name in PRIMARY:
        d = torch.load(run_file(name, "natural"))
        sc_, _ = IRO.set_disagreement(d["idx"], ref_idx)
        n_flip += int(sc_.sum())
    pre = dict(total_set_disagreements_primary=n_flip, early_stop=n_flip == 0)
    jdump(pre, p("logs", "preflight_flip_count.json"))
    print(f"[preflight] total top-4 set disagreements over primary configs: {n_flip}", flush=True)
    if n_flip == 0:
        print("[preflight] ROUTING-NO-FLIPS: stopping before frozen evaluation"); return
    # ---------- frozen primary, then references
    for name in PRIMARY:
        run_cfg(name, ["frozen"])
    for name in REFERENCES:
        run_cfg(name, ["natural", "frozen"])
    # ---------- end-of-process invariants
    t1 = time.time()
    untouched1 = PQ.untouched_digests(model, LAYERS_Q)
    eq, tot = PQ.experts_match(model, snap)
    _, rend, s, pk = evaluate(model, win, None, plain=True)
    cost["end_checks"] = dict(untouched_params_unchanged=untouched1 == untouched0, n_untouched=len(untouched0),
                              targets_pristine=eq == tot, bf16_end_per_window_bitwise=nll_equal(rend["nll"],
                              hb["runs"][0]["per_window_nll"])["bitwise"], gates_unchanged=gate_digest(model) == gates0,
                              s=time.time() - t1)
    cost["peak_cpu_gb"] = peak_cpu_gb()
    jdump(cost, p("logs", "gpu_cost_raw.json"))
    print(f"[end] {cost['end_checks']}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["audit", "protocol", "gpu", "analyze"])
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    os.makedirs(p("figures"), exist_ok=True)
    if a.stage == "analyze":
        import report_inference_routing_amplification as RIR
        return RIR.analyze()
    dict(audit=stage_audit, protocol=stage_protocol, gpu=stage_gpu)[a.stage](a)


if __name__ == "__main__":
    main()
