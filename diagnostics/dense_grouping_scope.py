"""Dense-model control for GPTQ calibration-grouping reproducibility (Qwen2.5-1.5B).

The ONLY intended variable is the calibration batch grouping, reproduced exactly from the MoE Family D studies:
  * groupings G20261111..15 = moe_calib.multigroup_robustness.grouping_batches(union, seed) of the SAME 217-row
    union (sorted FR s0 u FC s0 calibration ids of ~/moe_cache), 27 x 8 + 1, verified against the frozen
    runs_multigroup_robustness/grouping_manifests/*.json;
  * the grouping acts on the pristine BF16 batched forward that builds the per-sequence layer-input cache
    (diagnostics/batch_sensitivity.forward_plan, unchanged);
  * Hessians: FR s0's 128 chosen rows in their frozen order, one sequence at a time, K = 4 blocks, summed
    (moe_calib.dense_gptq.DenseProfiler = Profiler semantics with the unchanged BlockHessianAccumulator);
  * GPTQ: moe_calib.gptq.quantize_expert, 4-bit, g128, percdamp 1.0 (x mean diag H), act-order;
  * BF16 fake-quantized effective weights; evaluation runs_partial_ppl/windows.pt with EP.layerwise_nll, batch 8.
The token ids are the historical ones (Qwen1.5 and Qwen2.5 share the base vocabulary; verified in `audit`).
ORIGINAL (cache_inputs.py consecutive batches of 8) is a descriptive reference only, as in Family D.

Stages: audit | protocol | gpu | repeat | hessian | weights | analyze
"""
from __future__ import annotations
import argparse, gc, glob, hashlib, json, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))

OUT = "runs_dense_grouping_scope"
CACHE = os.path.expanduser(os.environ.get("DGS_CACHE", "~/moe_cache_dense_grouping_scope"))
MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MOE_LAYERS, MOE_DEPTH = (1, 8, 16, 23), 24
GROUP_SEEDS = (20261111, 20261112, 20261113, 20261114, 20261115)
GROUPS = [f"G{s}" for s in GROUP_SEEDS]
REF = "G20261111"                                   # prespecified reference grouping (first seed)
CONFIGS = GROUPS + ["ORIGINAL"]                     # ORIGINAL: descriptive reference, never in the 5-grouping SD
B = 8
GPTQ = dict(bits=4, group_size=128, percdamp=1.0, act_order=True)
PROJ = ("gate_proj", "up_proj", "down_proj")


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


def sha256_file(path, bs=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def obj_sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), default=float).encode()).hexdigest()


def tensor_sha(t):
    t = t.detach().contiguous().cpu()
    return hashlib.sha256((t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()).hexdigest()


def snapshot_dir():
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt"],
                             local_files_only=True)


def dense_layers():
    from moe_calib import dense_gptq as DG
    cfg = json.load(open(os.path.join(snapshot_dir(), "config.json")))
    return DG.map_layers(MOE_LAYERS, MOE_DEPTH, cfg["num_hidden_layers"]), cfg


def code_hashes():
    fs = ["moe_calib/dense_gptq.py", "moe_calib/gptq.py", "moe_calib/hessian.py", "moe_calib/model_utils.py",
          "moe_calib/multigroup_robustness.py", "moe_calib/batch_sensitivity.py", "diagnostics/batch_sensitivity.py",
          "diagnostics/eval_partial_ppl.py", "diagnostics/dense_grouping_scope.py"]
    return {f: sha256_file(os.path.join(ROOT, f)) for f in fs}


def selections():
    """FR s0 selection + 217-row union, re-verified by the historical loaders (no re-selection)."""
    import full_depth_pilot as FD
    from moe_calib import multigroup_robustness as MG
    _, sel = FD.load_selections()
    union = sorted(set(sel["FR"]["chosen"]) | set(sel["FC"]["chosen"]))
    pr = json.load(open(os.path.join(ROOT, "runs_multigroup_robustness/protocol.json")))
    assert len(union) == MG.N_UNION and union == pr["union_calibration_ids"]
    return sel["FR"], union


# ---------------------------------------------------------------- audit (Phase 1, CPU)
def stage_audit(a):
    from transformers import AutoTokenizer
    from moe_calib import multigroup_robustness as MG, model_utils as MU, batch_sensitivity as BS
    import eval_partial_ppl as EP, batch_sensitivity as DB
    fr, union = selections()
    out = dict(code_paths={
        "1_selection": "FR s0 = e3_allocation 'random' B=128 (runs_typicality_matched/e3_random_B128_b4_d1_s0_warm-*.json), "
                       "loaded/verified by diagnostics/full_depth_pilot.load_selections",
        "2_grouping": "moe_calib/multigroup_robustness.grouping_batches: sorted 217-row union (FR s0 u FC s0) -> "
                      "numpy default_rng(seed).permutation -> consecutive batches of 8 (27 x 8 + 1); the batches change "
                      "ONLY the pristine BF16 batched forward that builds the layer-input cache "
                      "(diagnostics/batch_sensitivity.forward_plan, used by diagnostics/multigroup_robustness.stage_cache)",
        "3_hessian": "moe_calib/profiler.Profiler.run: replay layer L on cached inputs, batch 1, chosen order; "
                     "moe_calib/hessian.BlockHessianAccumulator.add (fp32 X^T X, K=4 contiguous blocks), summed "
                     "(moe_calib/quant_eval.quantize_from_acc); non-sequential (pristine inputs at every layer)",
        "4_gptq": "moe_calib/gptq.gptq_quantize via quantize_expert (gate/up share H_gu; down H_down); lazy-batch 128",
        "5_actorder": "argsort(diag H, descending) after dead-column fix, before damping",
        "6_groupwise": "asymmetric 4-bit min/max per 128 permuted columns, fitted at each group start (_qparams)",
        "7_bf16": "fp32 dequantized W_q -> p.copy_(W_q.to(bf16)) (model_utils.set_expert_weights)",
        "8_install_eval": "partial_model_quant.quantize_install -> e3_allocation.quantize_and_score(write_back=True); "
                          "PPL: diagnostics/eval_partial_ppl.layerwise_nll, runs_partial_ppl/windows.pt, batch 8",
        "percdamp_semantics": "percdamp=1.0 is passed verbatim: damping = 1.0 x mean(diag H) (configs/frozen_b4.json, "
                              "selected on the tuning split)"})
    # groupings: regenerate and compare with the frozen manifests
    gv = {}
    for s in GROUP_SEEDS:
        m = json.load(open(os.path.join(ROOT, f"runs_multigroup_robustness/grouping_manifests/G{s}.json")))
        bt = MG.grouping_batches(union, s)
        gv[f"G{s}"] = dict(regenerated_equals_frozen=bt == m["batches"], manifest_sha256=m["manifest_sha256"],
                           sizes=[len(b) for b in bt][-2:], n_batches=len(bt),
                           singleton_row=bt[-1][0], singleton_row_in_FR=bt[-1][0] in set(fr["chosen"]),
                           validation=MG.validate_grouping(bt, union))
    out["groupings"] = gv
    # historical seed-0 FR grouping PPL (single strategy, single seed; NOT the pooled 3-seed SD)
    ppl = np.array([json.load(open(os.path.join(ROOT, f"runs_multigroup_robustness/results/FR_{g}.json")))["ppl"]
                    for g in GROUPS])
    out["moe_FR_s0"] = dict(ppl=dict(zip(GROUPS, ppl.tolist())), mean=float(ppl.mean()), sd=float(ppl.std(ddof=1)),
                            sd_log=float(np.log(ppl).std(ddof=1)), cv=float(ppl.std(ddof=1) / ppl.mean()),
                            min=float(ppl.min()), max=float(ppl.max()), range=float(ppl.max() - ppl.min()),
                            original_ppl=json.load(open(os.path.join(ROOT, "runs_multigroup_robustness/results/CTRL_FR_ORIGINAL.json")))["ppl"],
                            bf16_ppl=json.load(open(os.path.join(ROOT, "runs_partial_ppl/bf16_baseline.json")))["ppl"],
                            note="sample SD over the 5 seed-0 groupings (ddof=1); the 0.0052 figure is the 3-seed pooled within-seed SD")
    # tokenizer equivalence -> identical token ids
    t_moe = AutoTokenizer.from_pretrained(MU._snapshot_dir("Qwen/Qwen1.5-MoE-A2.7B"))
    t_d = AutoTokenizer.from_pretrained(snapshot_dir())
    v0, v1 = t_moe.get_vocab(), t_d.get_vocab()
    shared = [k for k in v0 if v1.get(k) == v0[k]]
    alltok = DB.all_tokens()
    win, prot = EP.load_windows()
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = t_d("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    import random
    n_full = enc.shape[0] // EP.SEQLEN
    idx = sorted(random.Random(EP.WIN_SEED).sample(range(n_full), EP.N_WIN))
    win_d = torch.stack([enc[i * EP.SEQLEN:(i + 1) * EP.SEQLEN] for i in idx])
    sp = snapshot_dir()
    out["tokenization"] = dict(
        moe_vocab=len(v0), dense_vocab=len(v1), n_moe_entries_identical_in_dense=len(shared),
        all_moe_entries_identical=len(shared) == len(v0), dense_only_entries=sorted(set(v1) - set(v0)),
        merges_sha256=dict(moe=sha256_file(os.path.join(MU._snapshot_dir("Qwen/Qwen1.5-MoE-A2.7B"), "merges.txt")),
                           dense=sha256_file(os.path.join(sp, "merges.txt"))),
        calibration_max_token_id=int(alltok.max()), eval_max_token_id=int(win.max()),
        dense_retokenized_windows_equal_historical=bool(torch.equal(win_d, win)),
        decision="identical token ids reused: calibration ~/moe_cache/ids.pt rows, evaluation runs_partial_ppl/windows.pt")
    # dense model structure
    layers, cfg = dense_layers()
    from safetensors import safe_open
    n_par, shapes = 0, {}
    with safe_open(os.path.join(sp, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            shp = sl.get_shape()
            n_par += int(np.prod(shp))
            if k.startswith(f"model.layers.{layers[0]}.mlp."):
                shapes[k.split(".mlp.")[1]] = shp
    nq = len(layers) * 3 * cfg["hidden_size"] * cfg["intermediate_size"]
    moe_cfg = json.load(open(os.path.join(MU._snapshot_dir("Qwen/Qwen1.5-MoE-A2.7B"), "config.json")))
    moe_q = 4 * 60 * 3 * moe_cfg["hidden_size"] * moe_cfg["moe_intermediate_size"]
    out["dense_model"] = dict(model=MODEL_ID, revision=REVISION, depth=cfg["num_hidden_layers"], hidden=cfg["hidden_size"],
                              intermediate=cfg["intermediate_size"], tie_word_embeddings=cfg["tie_word_embeddings"],
                              layer_mapping=dict(zip(map(str, MOE_LAYERS), layers)),
                              mapping_rule="round(L * (28-1) / (24-1)) (relative depth, endpoints to endpoints)",
                              mlp_shapes=shapes, n_quantized_tensors=len(layers) * 3, n_quantized_params=nq,
                              n_total_params=n_par, frac_quantized=nq / n_par,
                              checkpoint_sha256=sha256_file(os.path.join(sp, "model.safetensors")))
    out["moe_model"] = dict(n_quantized_tensors=720, n_quantized_params=moe_q,
                            hidden=moe_cfg["hidden_size"], moe_intermediate=moe_cfg["moe_intermediate_size"])
    out["calibration"] = dict(n_union=len(union), n_fr=len(fr["chosen"]), seqlen=int(alltok.shape[1]),
                              fr_tokens=len(fr["chosen"]) * int(alltok.shape[1]), K=fr["K"], fr_ids_hash=fr["ids_hash"],
                              source="C4 en/c4-train.00000-of-01024 (cache_inputs.py, data_seed 0)")
    jdump(out, p("logs", "audit.json"))
    print(json.dumps({k: out[k] for k in ("groupings", "moe_FR_s0")}, indent=1, default=float)[:3000])
    print({k: v for k, v in out["tokenization"].items() if k != "dense_only_entries"})
    print({k: v for k, v in out["dense_model"].items() if k != "mlp_shapes"}, out["dense_model"]["mlp_shapes"])


# ---------------------------------------------------------------- protocol (frozen before any grouping result)
def stage_protocol(a):
    from moe_calib import multigroup_robustness as MG, batch_sensitivity as BS
    import eval_partial_ppl as EP, batch_sensitivity as DB
    au = jload(p("logs", "audit.json"))
    assert au and all(v["regenerated_equals_frozen"] for v in au["groupings"].values())
    assert au["tokenization"]["all_moe_entries_identical"] and au["tokenization"]["dense_retokenized_windows_equal_historical"]
    if glob.glob(p("results", "*_G*.json")):
        raise SystemExit("grouping results already exist: protocol cannot be (re)created")
    fr, union = selections()
    alltok = DB.all_tokens()
    win, prot = EP.load_windows()
    layers = au["dense_model"]["layer_mapping"]
    fr_rows = MG.remap(fr["chosen"], union)
    gm = {}
    for s in GROUP_SEEDS:
        bt = MG.grouping_batches(union, s)
        flat = [r for b in bt for r in b]
        gm[f"G{s}"] = dict(seed=s, batches=bt, n_batches=len(bt), batch_size_pattern=[len(b) for b in bt],
                           historical_manifest_sha256=au["groupings"][f"G{s}"]["manifest_sha256"],
                           multiset_equals_union=sorted(flat) == union,
                           fr_rows_batch_position={str(r): [bi for bi, b in enumerate(bt) if r in b][0] for r in fr["chosen"]})
    ob = BS.original_batches(union, B, int(alltok.shape[0]))
    gm["ORIGINAL"] = dict(batches=ob, n_batches=len(ob), batch_size_pattern=[len(b) for b in ob],
                          construction="cache_inputs.py: consecutive batches of 8 original rows; each union row in its "
                                       "original batch with its original batch-mates (BS.original_batches)",
                          role="descriptive reference only (as Family D); not one of the 5 groupings")
    for g in gm.values():
        g["sha256"] = obj_sha(g["batches"])
    jdump(dict(groupings=gm, note="batch membership and within-batch order = list order; executed in list order"),
          p("grouping_manifest.json"))
    cal = dict(source=au["calibration"]["source"], token_tensor="~/moe_cache/ids.pt (pool || holdout), row = sequence id",
               ids_pt_sha256=sha256_file(os.path.expanduser("~/moe_cache/ids.pt")),
               union_ids=union, union_token_sha256=tensor_sha(alltok[union]),
               fr_chosen_ordered=fr["chosen"], fr_ids_hash=fr["ids_hash"], fr_source=fr["source"],
               fr_source_sha256=fr["source_sha256"], fr_union_rows=fr_rows,
               fr_token_sha256=tensor_sha(alltok[fr["chosen"]]),
               n_sequences=len(fr["chosen"]), seqlen=int(alltok.shape[1]), total_tokens=len(fr["chosen"]) * int(alltok.shape[1]),
               K_blocks=fr["K"], block_rule="Profiler.run default: block of i-th chosen = (i*K)//n",
               tokenizer_note="Qwen1.5-MoE and Qwen2.5 share every base-vocabulary id (audit); token ids reused verbatim",
               held_out="calibration = C4 train shard; evaluation = WikiText-2 test (disjoint sources)")
    jdump(cal, p("calibration_manifest.json"))
    body = dict(
        study="Dense-model control for GPTQ calibration-grouping reproducibility",
        model=dict(id=MODEL_ID, revision=REVISION, checkpoint_sha256=au["dense_model"]["checkpoint_sha256"],
                   config={k: au["dense_model"][k] for k in ("depth", "hidden", "intermediate", "tie_word_embeddings")},
                   dtype="bfloat16", attn_implementation="eager", offload="model_utils layer-wise (one decoder layer on GPU)"),
        quantization=dict(GPTQ, solver="moe_calib.gptq.quantize_expert (unchanged)", blocksize=128,
                          percdamp_semantics="damping = 1.0 x mean(diag H)", effective_weights="BF16 fake-quantized",
                          targets="mlp.gate_proj/up_proj/down_proj of the mapped layers; everything else BF16",
                          layer_mapping=layers, n_quantized_tensors=au["dense_model"]["n_quantized_tensors"],
                          n_quantized_params=au["dense_model"]["n_quantized_params"],
                          frac_quantized=au["dense_model"]["frac_quantized"]),
        calibration=dict(manifest_sha256=obj_sha(cal), n=cal["n_sequences"], seqlen=cal["seqlen"], tokens=cal["total_tokens"],
                         strategy="FR s0 (random, B=128), the historical single-strategy comparison"),
        groupings={g: dict(sha256=v["sha256"], n_batches=v["n_batches"]) for g, v in gm.items()},
        reference_grouping=REF,
        evaluation=dict(windows="runs_partial_ppl/windows.pt", windows_sha256=prot["windows_sha256"], n_windows=128,
                        seqlen=512, batch=EP.BATCH, evaluator="diagnostics/eval_partial_ppl.layerwise_nll (unchanged)",
                        loss="FP32 logits -> FP64 token-summed NLL", repeats=2),
        metrics=dict(
            primary="over the 5 groupings (ddof=1): SD(PPL), SD(log PPL), CV(PPL) = SD/mean",
            secondary=["mean, min, max, max pairwise PPL difference", "PPL - PPL(G20261111)",
                       "per-window paired NLL differences (10 pairs)",
                       "weights: GRA.weight_diff (identical definition to Family D) vs G20261111, 10 pairs, vs ORIGINAL",
                       "integer-code disagreement (solver info capture), separately from BF16 ULP",
                       "Hessians: frozen_routing.hessian_metrics (rel_fro_trace_normalized = historical Family D metric), "
                       "act-order positions / group membership, diag-rank changes, damped-H conditioning"]),
        comparison=dict(
            moe_reference="historical Qwen1.5-MoE FR s0 over the same 5 groupings (runs_multigroup_robustness; "
                          "weights: runs_grouping_routing_amplification/logs/weight_differences_raw.json FR_s0 pairs)",
            ratios="R_PPL = SD_dense(log PPL) / SD_moe(log PPL); R_W = mean pairwise frac BF16 elements changed "
                   "(dense / moe FR s0)",
            rule={"A-SUBSTANTIAL-DENSE-SENSITIVITY": "R_W >= 0.5 and R_PPL >= 0.5",
                  "B-SMALL-DENSE-SENSITIVITY": "R_W < 0.1 and R_PPL < 0.1",
                  "C-WEIGHT-SENSITIVE-PPL-STABLE": "R_W >= 0.5 and R_PPL < 0.5",
                  "D-WEIGHT-STABLE-PPL-SENSITIVE": "R_W < 0.1 and R_PPL >= 0.5",
                  "E-INTERMEDIATE": "otherwise",
                  "BLOCKED": "any mandatory check A-F fails"},
            caveats="screening ratios, not tests; 5 groupings, 1 seed; F(4,4) 95% factor on an SD ratio is ~3.1x; "
                    "architecture, size, matrix geometry, calibration statistics and baseline PPL are not matched"),
        checks=dict(
            A="BF16 dense checkpoint evaluated (PPL + per-window NLL); all parameters equal the checkpoint file",
            B="every evaluation repeated in-process: per-window NLL bitwise identical; grouping G20261111 re-quantized "
              "in a fresh process from its cache: installed weights bitwise identical",
            C="every grouping: identical union multiset, FR token sha, 128 x 512 tokens per projection Hessian, same layers; "
              "reversed batch execution order reproduces each cache bitwise",
            D="every config: 12 target tensors expected shape, finite, all differ from BF16 original, non-targets unchanged, "
              "installed == bf16(solver output)",
            E="each config starts from pristine targets (digest == checkpoint) - never from another grouping's weights",
            F="after each config targets restored bitwise; non-targets unchanged; BF16 re-evaluation bitwise at end",
            tolerance="exact"),
        storage=dict(cache_dir=CACHE, weights="weights/<config>/L<l>.pt", codes="codes/<config>/L<l>.pt",
                     hessians="hessians/<config>/L<l>.pt"),
        code_sha256=code_hashes())
    body["protocol_sha256"] = obj_sha(body)
    body["created"] = time.strftime("%F %T")
    old = jload(p("protocol.json"))
    if old:
        assert old["protocol_sha256"] == body["protocol_sha256"], "protocol changed after freezing"
        print("protocol already frozen and consistent"); return
    jdump(body, p("protocol.json"))
    print(f"protocol frozen sha256 {body['protocol_sha256'][:16]}")


def check_protocol():
    pr = jload(p("protocol.json"))
    body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created")}
    assert obj_sha(body) == pr["protocol_sha256"], "protocol.json modified"
    assert code_hashes() == pr["code_sha256"], "code changed since the protocol was frozen"
    return pr


# ---------------------------------------------------------------- model
def load_model():
    os.environ.pop("MOE_LOAD_LAYERS", None)
    from moe_calib import model_utils as MU
    t0 = time.time()
    model, tok = MU.load_model(snapshot_dir())
    return model, time.time() - t0


@torch.no_grad()
def verify_checkpoint(model):
    """Every parameter equals the checkpoint tensor bitwise (tied lm_head = embed_tokens)."""
    from safetensors import safe_open
    params = dict(model.named_parameters())
    ok, n = True, 0
    with safe_open(os.path.join(snapshot_dir(), "model.safetensors"), framework="pt") as f:
        keys = set(f.keys())
        for k in keys:
            ok &= torch.equal(params[k].detach().cpu(), f.get_tensor(k).to(params[k].dtype)); n += 1
    missing = [k for k in params if k not in keys and k != "lm_head.weight"]
    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    return dict(all_equal=bool(ok), n_checked=n, missing=missing, lm_head_tied=bool(tied))


def evaluate(model, win):
    import eval_partial_ppl as EP
    t0 = time.time()
    nll, cnt = EP.layerwise_nll(model, win)
    return nll, cnt, time.time() - t0


# ---------------------------------------------------------------- GPU stage
def stage_gpu(a):
    from moe_calib import dense_gptq as DG, model_utils as MU, multigroup_robustness as MG
    import eval_partial_ppl as EP, batch_sensitivity as DB
    pr = check_protocol()
    gman = jload(p("grouping_manifest.json"))["groupings"]
    cal = jload(p("calibration_manifest.json"))
    assert obj_sha(cal) == pr["calibration"]["manifest_sha256"]
    for g, v in gman.items():
        assert obj_sha(v["batches"]) == pr["groupings"][g]["sha256"] == v["sha256"]
    LAY = [int(x) for x in pr["quantization"]["layer_mapping"].values()]
    win, prot = EP.load_windows()
    assert prot["windows_sha256"] == pr["evaluation"]["windows_sha256"]
    union, fr_rows = cal["union_ids"], cal["fr_union_rows"]
    alltok = DB.all_tokens()
    assert tensor_sha(alltok[union]) == cal["union_token_sha256"] and tensor_sha(alltok[cal["fr_chosen_ordered"]]) == cal["fr_token_sha256"]
    assert [union[i] for i in fr_rows] == cal["fr_chosen_ordered"]
    corr = jload(p("logs", "gpu_checks.json")) or {}
    cost = jload(p("logs", "gpu_cost_raw.json")) or dict(loads=[], configs={})
    model, t_load = load_model()
    cost["loads"].append(t_load)
    torch.cuda.reset_peak_memory_stats()
    vc = verify_checkpoint(model)
    assert vc["all_equal"] and not vc["missing"], vc
    corr["checkpoint"] = vc
    snap = DG.snapshot(model, LAY)
    tgt0 = DG.target_digests(model, LAY)
    unt0 = DG.untouched_digests(model, LAY)
    corr["n_untouched"] = len(unt0)

    def save():
        jdump(corr, p("logs", "gpu_checks.json")); jdump(cost, p("logs", "gpu_cost_raw.json"))

    # ---- A: BF16 reference (+ B repeat)
    if not jload(p("bf16_baseline.json")):
        n1, k1, s1 = evaluate(model, win)
        n2, k2, s2 = evaluate(model, win)
        rec = EP.ppl_record(n1, k1, s1)
        rec.update(model=MODEL_ID, revision=REVISION, repeat_per_window_bitwise=bool(np.array_equal(n1, n2)),
                   repeat_eval_s=s2, windows_sha256=prot["windows_sha256"], moe_bf16_ppl_historical=9.237089824789335)
        jdump(rec, p("bf16_baseline.json"))
        print(f"[bf16] PPL {rec['ppl']:.6f} repeat bitwise {rec['repeat_per_window_bitwise']}", flush=True)
        assert rec["repeat_per_window_bitwise"], "BF16 evaluation not deterministic -> BLOCKED"
    bf = jload(p("bf16_baseline.json"))

    # ---- C: grouping caches (+ reversed execution order control)
    cm = jload(p("cache_manifest.json")) or {}
    for g in CONFIGS:
        if cm.get(g, {}).get("complete"):
            continue
        bt = gman[g]["batches"]
        d = c("cache", g); os.makedirs(d, exist_ok=True)
        keep, files = {}, {}

        def on_layer(L, X):
            if L in LAY:
                path = os.path.join(d, f"L{L}.pt")
                torch.save(X.clone(), path + ".tmp"); os.replace(path + ".tmp", path)
                keep[L] = X.clone()
                files[str(L)] = dict(shape=list(X.shape), sha256=sha256_file(path), tensor_sha256=tensor_sha(X),
                                     finite=bool(torch.isfinite(X.float()).all()))
        t0 = time.time()
        tsha = DB.forward_plan(model, alltok, bt, union, on_layer)
        tb = time.time() - t0
        neg = {}

        def cmp(L, X):
            if L in LAY:
                neg[str(L)] = bool(torch.equal(X, keep[L]))
        DB.forward_plan(model, alltok, bt[::-1], union, cmp)
        cm[g] = dict(files=files, token_sha256=tsha, token_sha_equals_union=tsha == cal["union_token_sha256"],
                     reversed_order_bitwise=neg, reversed_ok=all(neg.values()) and len(neg) == len(LAY),
                     build_s=tb, complete=True)
        jdump(cm, p("cache_manifest.json"))
        print(f"[cache] {g}: {tb:.0f} s, reversed-order bitwise {cm[g]['reversed_ok']}", flush=True)
        if not (cm[g]["reversed_ok"] and cm[g]["token_sha_equals_union"]):
            raise SystemExit(f"{g}: cache determinism / token control failed -> BLOCKED")
        del keep
    # activation drift per layer vs the reference grouping and vs ORIGINAL (all 217 rows and FR's 128 rows)
    if "activation_drift" not in corr:
        ad = {}
        for L in LAY:
            X = {g: torch.load(c("cache", g, f"L{L}.pt")) for g in CONFIGS}
            for refn in (REF, "ORIGINAL"):
                for g in CONFIGS:
                    if g == refn:
                        continue
                    A, R = X[g], X[refn]
                    rows_eq = torch.tensor([torch.equal(A[i], R[i]) for i in range(A.shape[0])])
                    D = (A.float() - R.float())
                    ad[f"L{L}|{g}_vs_{refn}"] = dict(
                        frac_rows_bitwise=float(rows_eq.float().mean()),
                        frac_fr_rows_bitwise=float(rows_eq[fr_rows].float().mean()),
                        n_rows_differ=int((~rows_eq).sum()), rows_differ=[union[i] for i in (~rows_eq).nonzero().flatten().tolist()][:20],
                        frac_elements_equal=float((A == R).float().mean()),
                        rel_l2=float(D.norm() / R.float().norm()),
                        rel_l2_fr_rows=float(D[fr_rows].norm() / R[fr_rows].float().norm()))
            del X
        corr["activation_drift"] = ad
        save()

    # ---- per configuration: E (pristine start), Hessians + GPTQ + install, D, B, F
    os.makedirs(p("results"), exist_ok=True)
    for g in CONFIGS:
        if (jload(p("results", f"{g}.json")) or {}).get("valid"):
            continue
        assert DG.target_digests(model, LAY) == tgt0, "targets not pristine before config (E)"
        t_all = time.time()
        torch.cuda.reset_peak_memory_stats()
        infos_all, hdiag, t_h, t_q = {}, {}, 0.0, 0.0
        for L in LAY:
            for sub in ("weights", "codes", "hessians"):
                os.makedirs(c(sub, g), exist_ok=True)
            cache = torch.load(c("cache", g, f"L{L}.pt"), mmap=True)
            assert sha256_file(c("cache", g, f"L{L}.pt")) == cm[g]["files"][str(L)]["sha256"]
            prof = DG.DenseProfiler(model, L, cal["K_blocks"]).run(cache, fr_rows)
            t_h += prof.seconds
            Hg, Hd = prof.H("gu"), prof.H("down")
            ntok = dict(gu=prof.tokens("gu"), down=prof.tokens("down"))
            tq = time.time()
            Wq, infos = DG.quantize_layer(model, L, Hg, Hd, GPTQ["bits"], GPTQ["group_size"], GPTQ["percdamp"])
            if g == REF and L == LAY[0]:                   # info capture never alters numerics (regression)
                Wq0, _ = DG.quantize_layer(model, L, Hg, Hd, GPTQ["bits"], GPTQ["group_size"], GPTQ["percdamp"], with_info=False)
                corr["info_path_bitwise_default"] = all(torch.equal(Wq[k], Wq0[k]) for k in PROJ)
                del Wq0
            t_q += time.time() - tq
            codes, exact = {}, {}
            for k in PROJ:
                q, gid, ex = DG.integer_codes(Wq[k], infos[k], GPTQ["bits"], GPTQ["group_size"])
                codes[k] = dict(q=q, group=gid, perm=infos[k]["perm"].int(), scale=infos[k]["scale"], zero=infos[k]["zero"])
                exact[k] = ex
            orig = {k: w.detach().cpu().clone() for k, w in DG.get_weights(model, L).items()}
            DG.set_weights(model, L, Wq)
            inst = {k: w.detach().cpu().clone() for k, w in DG.get_weights(model, L).items()}
            torch.save(inst, c("weights", g, f"L{L}.pt"))
            torch.save(codes, c("codes", g, f"L{L}.pt"))
            torch.save(dict(gu=Hg.cpu(), down=Hd.cpu(), tokens=ntok), c("hessians", g, f"L{L}.pt"))
            hdiag[str(L)] = dict(
                tokens=ntok, codes_reproduce_solver_bitwise=exact,
                shapes={k: list(inst[k].shape) for k in PROJ},
                shapes_ok=all(list(inst[k].shape) == list(orig[k].shape) for k in PROJ),
                finite=all(bool(torch.isfinite(inst[k]).all()) for k in PROJ),
                differs_from_bf16=all(not torch.equal(inst[k], orig[k]) for k in PROJ),
                installed_equals_bf16_of_solver=all(torch.equal(inst[k], Wq[k].to(torch.bfloat16).cpu()) for k in PROJ),
                frac_elements_unchanged_vs_bf16={k: float((inst[k] == orig[k]).float().mean()) for k in PROJ},
                weight_sha256=sha256_file(c("weights", g, f"L{L}.pt")), codes_sha256=sha256_file(c("codes", g, f"L{L}.pt")),
                hessian_sha256=sha256_file(c("hessians", g, f"L{L}.pt")))
            del prof, Hg, Hd, Wq, infos, cache, codes
            torch.cuda.empty_cache()
        unt_ok = DG.untouched_digests(model, LAY) == unt0
        dig = DG.installed_digest(model, LAY)
        n1, k1, s1 = evaluate(model, win)
        n2, k2, s2 = evaluate(model, win)
        rec = EP.ppl_record(n1, k1, s1)
        D = dict(all_shapes_ok=all(v["shapes_ok"] for v in hdiag.values()), all_finite=all(v["finite"] for v in hdiag.values()),
                 all_targets_quantized=all(v["differs_from_bf16"] for v in hdiag.values()),
                 installed_equals_solver=all(v["installed_equals_bf16_of_solver"] for v in hdiag.values()),
                 codes_exact=all(all(v["codes_reproduce_solver_bitwise"].values()) for v in hdiag.values()),
                 untouched_unchanged=unt_ok,
                 tokens_per_projection_ok=all(v["tokens"]["gu"] == v["tokens"]["down"] == cal["total_tokens"] for v in hdiag.values()))
        Bc = dict(repeat_per_window_bitwise=bool(np.array_equal(n1, n2)))
        installed_unchanged = DG.installed_digest(model, LAY) == dig
        DG.restore(model, snap)
        eq, tot = DG.matches(model, snap)
        F = dict(installed_unchanged_during_eval=installed_unchanged, targets_restored=eq == tot,
                 target_digest_pristine=DG.target_digests(model, LAY) == tgt0,
                 untouched_unchanged=DG.untouched_digests(model, LAY) == unt0)
        valid = all(D.values()) and all(Bc.values()) and all(F.values())
        rec.update(config=g, role="grouping" if g in GROUPS else "reference", installed_weights_sha256=dig,
                   layers=LAY, per_layer=hdiag, D=D, B=Bc, F=F, E=dict(started_pristine=True),
                   hessian_s=t_h, gptq_s=t_q, repeat_eval_s=s2, total_s=time.time() - t_all,
                   peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, protocol_sha256=pr["protocol_sha256"], valid=valid)
        jdump(rec, p("results", f"{g}.json"))
        cost["configs"][g] = dict(hessian_s=t_h, gptq_s=t_q, eval_s=s1, total_s=rec["total_s"], peak_gpu_gb=rec["peak_gpu_gb"])
        save()
        print(f"[{g}] PPL {rec['ppl']:.6f} (BF16 {bf['ppl']:.6f}) D {all(D.values())} B {Bc} F {all(F.values())} "
              f"H {t_h:.0f}s Q {t_q:.0f}s", flush=True)
        if not valid:
            raise SystemExit(f"{g}: integrity failed -> BLOCKED")
        gc.collect()
    # end: BF16 bitwise re-evaluation
    n3, _, _ = evaluate(model, win)
    corr["end"] = dict(bf16_reeval_bitwise=bool(np.array_equal(n3, np.array(bf["per_window_nll"]))),
                       target_digest_pristine=DG.target_digests(model, LAY) == tgt0,
                       untouched_unchanged=DG.untouched_digests(model, LAY) == unt0)
    import resource
    cost["peak_cpu_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    save()
    print(f"[end] {corr['end']}", flush=True)


# ---------------------------------------------------------------- B (fresh-process quantization repeat)
def stage_repeat(a):
    from moe_calib import dense_gptq as DG
    pr = check_protocol()
    cal = jload(p("calibration_manifest.json"))
    LAY = [int(x) for x in pr["quantization"]["layer_mapping"].values()]
    ref = jload(p("results", f"{REF}.json"))
    assert ref and ref["valid"]
    model, _ = load_model()
    out = {}
    for L in LAY:
        cache = torch.load(c("cache", REF, f"L{L}.pt"), mmap=True)
        prof = DG.DenseProfiler(model, L, cal["K_blocks"]).run(cache, cal["fr_union_rows"])
        H = torch.load(c("hessians", REF, f"L{L}.pt"))
        Wq, _ = DG.quantize_layer(model, L, prof.H("gu"), prof.H("down"), GPTQ["bits"], GPTQ["group_size"],
                                  GPTQ["percdamp"], with_info=False)
        W = torch.load(c("weights", REF, f"L{L}.pt"))
        out[str(L)] = dict(hessians_bitwise=bool(torch.equal(prof.H("gu").cpu(), H["gu"]) and torch.equal(prof.H("down").cpu(), H["down"])),
                           weights_bitwise=all(torch.equal(Wq[k].to(torch.bfloat16).cpu(), W[k]) for k in PROJ))
        del prof, Wq, cache
        torch.cuda.empty_cache()
    out["passed"] = all(v["hessians_bitwise"] and v["weights_bitwise"] for v in out.values() if isinstance(v, dict))
    jdump(out, p("logs", "repeat_control.json"))
    print(out)


# ---------------------------------------------------------------- comparison pairs (Hessians, weights, codes)
def comparison_pairs():
    """(variant, reference): 4 x vs the reference grouping, the 10 unordered grouping pairs (i < j), 5 x vs ORIGINAL."""
    pr = [(g, REF) for g in GROUPS if g != REF] + [(GROUPS[i], GROUPS[j]) for i in range(1, 5) for j in range(i + 1, 5)] + \
         [(g, "ORIGINAL") for g in GROUPS]
    assert len(pr) == 4 + 6 + 5
    return pr


def ten_pairs(pairs):
    """The 10 unordered grouping pairs within comparison_pairs()."""
    return [(a, b) for a, b in pairs if b != "ORIGINAL"]


# ---------------------------------------------------------------- Hessian diagnostics
def stage_hessian(a):
    from moe_calib import frozen_routing as FRT
    from moe_calib.batch_sensitivity import actorder_perm
    pr = check_protocol()
    LAY = [int(x) for x in pr["quantization"]["layer_mapping"].values()]
    out = dict(pairs={}, conditioning={}, per_config={})
    pairs = comparison_pairs()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for L in LAY:
        H = {g: torch.load(c("hessians", g, f"L{L}.pt")) for g in CONFIGS}
        for pj in ("gu", "down"):
            perm = {g: actorder_perm(H[g][pj]) for g in CONFIGS}
            for a_, b_ in pairs:
                m = FRT.hessian_metrics(H[a_][pj], H[b_][pj])
                pa, pb = perm[a_], perm[b_]
                n = len(pa)
                pos_a, pos_b = torch.empty(n, dtype=torch.long), torch.empty(n, dtype=torch.long)
                pos_a[pa] = torch.arange(n); pos_b[pb] = torch.arange(n)
                m.update(frac_actorder_positions_changed=float((pa != pb).float().mean()),
                         frac_columns_group_changed=float(((pos_a // 128) != (pos_b // 128)).float().mean()),
                         mean_abs_diag_rank_shift=float((pos_a - pos_b).abs().float().mean()),
                         max_abs_diag_rank_shift=int((pos_a - pos_b).abs().max()),
                         frac_elements_bitwise_equal=float((H[a_][pj] == H[b_][pj]).float().mean()))
                out["pairs"][f"L{L}|{pj}|{a_}__{b_}"] = m
            for g in CONFIGS:                               # conditioning of H and of the solver's damped H
                Hm = H[g][pj].to(dev, torch.float64)
                dg = torch.diagonal(Hm).clone()
                lam = GPTQ["percdamp"] * float(dg.mean())
                ev0 = torch.linalg.eigvalsh(Hm)              # damped spectrum = ev0 + lam (H + lam I)
                out["conditioning"][f"L{L}|{pj}|{g}"] = dict(
                    cond_damped=float((ev0[-1] + lam) / (ev0[0] + lam)), damping=lam,
                    cond_undamped=float(ev0[-1] / ev0[0]) if float(ev0[0]) > 0 else float("inf"),
                    min_eig_undamped=float(ev0[0]), max_eig=float(ev0[-1]), trace=float(dg.sum()),
                    n_dead_columns=int((dg == 0).sum()), diag_cv=float(dg.std() / dg.mean()),
                    eff_rank_frac=float((ev0.clamp_min(0).sum() ** 2 / (ev0.clamp_min(0) ** 2).sum()) / len(ev0)))
                del Hm, ev0
            torch.cuda.empty_cache()
        # Cholesky-factor comparison (solver preprocessing on the reference permutation of each config, gu only)
        from moe_calib.hessian_order_null import solver_preprocess
        for pj in ("gu", "down"):
            Fac = {}
            for g in GROUPS:
                Hd, U = solver_preprocess(H[g][pj].to(dev), actorder_perm(H[g][pj]).to(dev), GPTQ["percdamp"])
                Fac[g] = torch.diagonal(U).double().cpu()
                del Hd, U
            for g in GROUPS:
                if g != REF:
                    out["per_config"][f"L{L}|{pj}|{g}_vs_{REF}|hinv_chol_diag_rel"] = float(
                        (Fac[g] - Fac[REF]).norm() / Fac[REF].norm())
        del H
        gc.collect()
        print(f"[hessian] L{L} done", flush=True)
    jdump(out, p("logs", "hessian_raw.json"))


# ---------------------------------------------------------------- weight + integer-code differences (CPU)
def stage_weights(a):
    import grouping_routing_amplification as GRA
    pr = check_protocol()
    LAY = [int(x) for x in pr["quantization"]["layer_mapping"].values()]
    Wb = {}

    def load_w(g):
        out = {}
        for L in LAY:
            d = torch.load(c("weights", g, f"L{L}.pt"))
            for k in PROJ:
                out[(L, 0, k)] = d[k]
        return out

    def load_codes(g):
        return {L: torch.load(c("codes", g, f"L{L}.pt")) for L in LAY}

    W = {g: load_w(g) for g in CONFIGS}
    C = {g: load_codes(g) for g in CONFIGS}
    pairs = comparison_pairs()
    out = dict(weights={}, codes={})
    for a_, b_ in pairs:
        key = f"{a_}__{b_}"
        if key in out["weights"]:
            continue
        out["weights"][key] = GRA.weight_diff(W[a_], W[b_])
        cd = dict(per_layer_proj={})
        n = nc = nb = nb_c = nc_nb = ngc = 0
        for L in LAY:
            for k in PROJ:
                qa, qb = C[a_][L][k]["q"], C[b_][L][k]["q"]
                ba, bb = W[a_][(L, 0, k)], W[b_][(L, 0, k)]
                code_d = qa != qb
                bf_d = ba != bb
                grp_d = (C[a_][L][k]["group"] != C[b_][L][k]["group"])
                n += qa.numel(); nc += int(code_d.sum()); nb += int(bf_d.sum())
                nb_c += int((bf_d & code_d).sum()); nc_nb += int((code_d & ~bf_d).sum())
                ngc += int(grp_d.sum()) * qa.shape[0]
                cd["per_layer_proj"][f"{L}.{k}"] = dict(frac_codes_changed=float(code_d.float().mean()),
                                                        frac_bf16_changed=float(bf_d.float().mean()),
                                                        frac_columns_group_changed=float(grp_d.float().mean()),
                                                        mean_abs_code_shift_changed=float((qa.int() - qb.int())[code_d].abs().float().mean()) if code_d.any() else 0.0)
        cd.update(frac_codes_changed=nc / n, frac_bf16_changed=nb / n,
                  frac_bf16_changed_with_code_change=nb_c / max(nb, 1),
                  frac_bf16_changed_same_code=(nb - nb_c) / max(nb, 1),
                  frac_code_changed_same_bf16=nc_nb / max(nc, 1),
                  frac_elements_in_group_changed_columns=ngc / n)
        out["codes"][key] = cd
        print(f"[weights] {key}: bf16 {out['weights'][key]['frac_changed']:.4f} codes {cd['frac_codes_changed']:.4f}", flush=True)
    out["note"] = ("weights: GRA.weight_diff, the identical function used for Family D (keys (layer, 0, proj)); codes: "
                   "integer q in original column order recovered from the solver info capture (bitwise-verified)")
    jdump(out, p("logs", "weight_differences_raw.json"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["audit", "protocol", "gpu", "repeat", "hessian", "weights", "analyze"])
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    os.makedirs(p("figures"), exist_ok=True)
    if a.stage == "analyze":
        import report_dense_grouping_scope as R
        return R.analyze()
    dict(audit=stage_audit, protocol=stage_protocol, gpu=stage_gpu, repeat=stage_repeat, hessian=stage_hessian,
         weights=stage_weights)[a.stage](a)


if __name__ == "__main__":
    main()
