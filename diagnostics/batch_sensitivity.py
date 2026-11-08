"""Controlled calibration batch-composition sensitivity in MoE GPTQ (four layers, seed 0).

Only ONE thing varies: which calibration sequences share a BF16 forward batch while the
layer-input cache used for the GPTQ Hessians is built.
  ORIGINAL  cache_inputs.py batches (8 consecutive original rows from row 0) = ~/moe_cache,
            reproduced bitwise for all 24 layers by the full-depth pilot v2 cache
  COMPACT   the 345 needed rows (pilot compact order) regrouped into batches of 8 (43 x 8 + 1),
            i.e. the failed v1 construction, regenerated and checked against the v1 file hashes
  SHUFFLED  the same 345 rows permuted once (numpy default_rng(20261110)), same batch pattern
Everything else is frozen: selections (FR s0 / FC s0), GPTQ (4-bit, g128, percdamp 1.0,
act-order), layers 1/8/16/23, reconstruction rows (always ~/moe_cache test rows 2128-2255),
WikiText-2 windows and evaluation batching.

Stages (--stage): preflight | cache | forward | quant --strategy FR|FC --variant V |
                  regression | weights | report
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import eval_partial_ppl as EP           # noqa: E402
import full_depth_pilot as FD           # noqa: E402
from moe_calib import batch_sensitivity as BS   # noqa: E402

OUT = "runs_batch_sensitivity"
OCACHE = os.path.expanduser("~/moe_cache")
V2 = os.path.expanduser("~/moe_cache_full_depth_pilot_v2")          # ORIGINAL composition, all 24 layers
V1 = os.path.expanduser("~/moe_cache_full_depth_pilot")             # failed v1 evidence (read only)
BCACHE = os.path.expanduser(os.environ.get("BATCH_SENS_CACHE", "~/moe_cache_batch_sensitivity"))
VDIR = {"COMPACT": os.path.join(BCACHE, "compact"), "SHUFFLED": os.path.join(BCACHE, "shuffled")}
LAYERS = [1, 8, 16, 23]
DIAG_EXPERTS = [0, 30]
B = 8
STRATS = {"FR": "Full Random s0", "FC": "Full Coverage-LP s0"}
N_LAYERS = 24


def p(*a):
    return os.path.join(ROOT, OUT, *a)


jdump, sha256_file = FD.jdump, FD.sha256_file


def cfg_name(c, v):
    return f"{c}_{v}"


def frozen():
    cfg = json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))
    pilot = json.load(open(os.path.join(ROOT, FD.OUT, "protocol.json")))
    sel = json.load(open(os.path.join(ROOT, FD.OUT, "selected_sets.json")))
    return cfg, pilot, sel


def tensor_sha(t):
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def env_record():
    import transformers
    return dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                transformers=transformers.__version__, python=platform.python_version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                note="determinism settings are recorded, not changed (changing them would change ORIGINAL numerics)")


def all_tokens():
    ids = torch.load(os.path.join(OCACHE, "ids.pt"))
    return torch.cat([ids["pool"], ids["holdout"]])


def plans(rows, n_total):
    orig = BS.original_batches(rows, B, n_total)
    return {"ORIGINAL": orig,
            "ORIGINAL_REVERSED": orig[::-1],                       # negative control: execution order only
            "COMPACT": BS.compact_batches(rows, B),
            "SHUFFLED": BS.shuffled_batches(rows, B, BS.SHUFFLE_SEED)}


CONSTRUCTION = {
    "ORIGINAL": "cache_inputs.py: batches of 8 consecutive original rows from row 0; every needed row in its "
                "original batch (= ~/moe_cache; = full-depth pilot v2 cache for all 24 layers)",
    "ORIGINAL_REVERSED": "negative control: the ORIGINAL batches (same members, same within-batch order) executed "
                         "in reversed batch order",
    "COMPACT": "failed v1 construction: the 345 needed rows in pilot compact order (sorted FR u FC ids, then test "
               "rows 2128-2255) in consecutive batches of 8; final batch of 1",
    "SHUFFLED": f"the same 345 rows permuted once with numpy default_rng({BS.SHUFFLE_SEED}).permutation, then "
                "consecutive batches of 8; final batch of 1 (same pattern as COMPACT)",
}


# ---------------------------------------------------------------- preflight
def stage_preflight(a):
    cfg, pilot, sel = frozen()
    os.makedirs(p("batch_manifests"), exist_ok=True)
    man, sel_live = FD.load_selections()                         # verifies source hashes vs partial-PPL manifest
    rows = pilot["compact_rows"]
    test_ids = list(range(*cfg["split"]["test"]))
    assert rows == FD.compact_rows(sel_live, test_ids) and len(rows) == 345
    warm = json.load(open(os.path.join(ROOT, "runs_typicality_matched", "warm_starts.json")))["seeds"]["0"]["ids"]
    checks = {}
    for c in STRATS:
        ch = sel_live[c]["chosen"]
        checks[c] = dict(n=len(ch), unique=len(set(ch)), ids_hash=sel_live[c]["ids_hash"],
                         ids_hash_matches_pilot=sel_live[c]["ids_hash"] == sel[c]["ids_hash"],
                         chosen_matches_pilot=ch == sel[c]["chosen"], warm_start_is_frozen=ch[:32] == warm,
                         all_in_pool=all(0 <= x < cfg["split"]["pool"][1] for x in ch),
                         no_tune_or_test_leakage=not (set(ch) & set(range(cfg["split"]["tune"][0], cfg["split"]["test"][1]))))
        assert checks[c]["n"] == 128 == checks[c]["unique"] and all(v for k, v in checks[c].items()
                                                                   if isinstance(v, bool)), checks[c]
    union = sorted(set(sel_live["FR"]["chosen"]) | set(sel_live["FC"]["chosen"]))
    n_total = int(all_tokens().shape[0])
    pl = plans(rows, n_total)
    for v, bt in pl.items():
        jdump(BS.plan_manifest(v, bt, rows, CONSTRUCTION[v]), p("batch_manifests", f"{v}.json"))
    assert BS.batch_sizes(pl["COMPACT"]) == BS.batch_sizes(pl["SHUFFLED"])
    win, prot = EP.load_windows()
    v1m = json.load(open(os.path.join(ROOT, FD.OUT, "logs", "cache_manifest_v1.json")))
    v2m = json.load(open(os.path.join(ROOT, FD.OUT, "cache_manifest.json")))
    assert v1m["rows"] == rows == v2m["rows"]
    mem = dict(line.split(":")[0:2] for line in open("/proc/meminfo") if ":" in line)
    gpu_free, gpu_total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    base = json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))
    ref = {c: json.load(open(os.path.join(ROOT, EP.OUT, f"ppl_{c}_s0.json"))) for c in STRATS}
    v1reg = json.load(open(os.path.join(ROOT, FD.OUT, "logs", "regression_attempt1_FAILED_v1cache.json")))
    proto = dict(
        experiment="Controlled calibration batch-composition sensitivity in MoE GPTQ (numerical reproducibility)",
        model=pilot["model"], model_snapshot=pilot["model_snapshot"], gptq=cfg["gptq"], B=128, seed=0,
        quant_layers=LAYERS, n_quantized_experts=240, projections=["gate_proj", "up_proj", "down_proj"],
        calibration="non-sequential: Hessians of layer L from the variant's cached BF16 inputs of layer L "
                    "(E3 Profiler: one sequence at a time)",
        reconstruction="E3 held-out relative expert reconstruction on ~/moe_cache test rows 2128-2255 for EVERY "
                       "variant (eval_cache_dir); never the variant cache",
        evaluation="frozen runs_partial_ppl windows (128 x 512, 65,408 targets), EP.layerwise_nll batch 8",
        windows_sha256=prot["windows_sha256"], bf16_ppl=base["ppl"],
        selections=checks, union_calibration_ids=union, n_union=len(union), cache_rows=rows,
        cache_row_semantics="index i of every variant cache = original row id rows[i] (aligned by sequence id)",
        variants=CONSTRUCTION, shuffle_seed=BS.SHUFFLE_SEED, batch_size=B,
        batch_size_patterns={v: f"{len(bt)} batches, sizes {sorted(set(BS.batch_sizes(bt)))}, last {len(bt[-1])}"
                             for v, bt in pl.items()},
        references=dict(partial_ppl={c: dict(ppl=ref[c]["ppl"], installed_weights_sha256=ref[c]["installed_weights_sha256"],
                                             recon=ref[c]["recon_recomputed"]) for c in STRATS},
                        v1_compact_regression={c: dict(ppl=v1reg["configs"][c]["pilot_cache"]["ppl"],
                                                       installed_sha256=v1reg["configs"][c]["pilot_cache"]["installed_sha256"])
                                               for c in STRATS}),
        diagnostics=dict(activation_layers=list(range(N_LAYERS)), routing_layers=list(range(N_LAYERS)),
                         routing_rows="the 217 calibration rows (FR u FC), replayed one sequence at a time on each "
                                      "variant's cached layer input (the E3 Profiler path)",
                         hessian_layers=LAYERS, hessian_experts=DIAG_EXPERTS, hessian_projections=["gu", "down"],
                         hessian_decomposition="gu only: common routed tokens vs membership change, from the "
                                               "replayed MoE-block inputs"),
        decision_rule=BS.DECISION_RULE, environment=env_record(),
        resources=dict(host_mem_available_gb=int(mem["MemAvailable"].split()[0]) / 1e6, gpu_free_gb=gpu_free / 1e9,
                       gpu_total_gb=gpu_total / 1e9, disk_free_tb=shutil.disk_usage(os.path.expanduser("~")).free / 1e12,
                       estimates=dict(new_caches_gb=2 * 17.4, weights_gb=6 * 4.15)),
        hard_constraints="no new allocation, no re-selection, B=128 seed 0, 4 layers, frozen GPTQ / precision / windows / "
                         "evaluation batching, exactly one shuffle chosen before any result, no extra seeds",
        created=time.strftime("%Y-%m-%d %H:%M:%S"))
    if os.path.exists(p("protocol.json")):
        old = json.load(open(p("protocol.json")))
        for k in ("decision_rule", "cache_rows", "variants", "selections", "diagnostics", "gptq"):
            assert old[k] == proto[k], f"protocol field {k} changed"
        print("protocol already frozen; consistent"); return
    jdump(proto, p("protocol.json"))
    print(f"preflight ok: 345 rows; union {len(union)}; plans {proto['batch_size_patterns']}; "
          f"GPU free {gpu_free / 1e9:.1f} GB")


# ---------------------------------------------------------------- shared model checks
def load_model():
    model, t = FD.load_model_full()
    return model, t


def model_config_record(model):
    from moe_calib import model_utils as MU
    args, kw = MU.layer_kwargs(model, 512)
    desc = {}
    for k, v in list(kw.items()) + [(f"arg{i}", x) for i, x in enumerate(args)]:
        if torch.is_tensor(v):
            desc[k] = dict(shape=list(v.shape), dtype=str(v.dtype), sha256=tensor_sha(v))
        elif isinstance(v, (tuple, list)) and all(torch.is_tensor(x) for x in v):
            desc[k] = [dict(shape=list(x.shape), dtype=str(x.dtype), sha256=tensor_sha(x)) for x in v]
        else:
            desc[k] = repr(v)
    drop = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.p > 0]
    return dict(layer_kwargs=desc, training=bool(model.training), dropout_modules_p_gt_0=drop,
                attention_dropout=float(getattr(model.config, "attention_dropout", 0.0)),
                attn_implementation=model.config._attn_implementation, dtype=str(MU.DTYPE),
                param_dtypes=sorted({str(q.dtype) for q in model.parameters()}),
                use_cache=bool(model.config.use_cache), norm_topk_prob=bool(model.config.norm_topk_prob),
                kwargs_note="the decoder-layer kwargs (causal mask, positions, rotary) are captured once for batch 1 "
                            "and broadcast over the batch: identical for every batch size and composition")


def verify_pristine_full(model):
    """Every parameter equals the checkpoint state recorded by the full-depth pilot preflight."""
    from moe_calib import partial_model_quant as PQ
    pr = json.load(open(os.path.join(ROOT, FD.OUT, "pristine_digests.json")))
    layers = list(range(N_LAYERS))
    return (PQ.target_tensor_digests(model, layers) == pr["pristine_targets"] and
            PQ.untouched_digests(model, layers) == pr["pristine_untouched"])


# ---------------------------------------------------------------- cache
@torch.no_grad()
def forward_plan(model, alltok, batches, rows, on_layer):
    """Build per-sequence layer inputs for `rows` by forwarding `batches` (list order = execution order).
    on_layer(L, X) gets the inputs of layer L reassembled in `rows` order."""
    from moe_calib import model_utils as MU
    idx = BS.reassembly_index(batches, rows)
    tok = BS.reassemble([alltok[b] for b in batches], idx)
    assert torch.equal(tok, alltok[rows]), "token ids changed by the batch plan"
    hs = [MU.embed(model, alltok[b]).cpu() for b in batches]
    n = MU.num_layers(model)
    for L in range(n):
        on_layer(L, BS.reassemble(hs, idx))
        if L < n - 1:
            with MU.layer_on_gpu(model, L):
                hs = [MU.run_layer(model, L, h.to(MU.GPU)).cpu() for h in hs]
    return tensor_sha(tok)


def stage_cache(a):
    proto = json.load(open(p("protocol.json")))
    rows = proto["cache_rows"]
    out_path = p("cache_validation.json")
    val = json.load(open(out_path)) if os.path.exists(out_path) else {}
    alltok = all_tokens()
    pl = {v: json.load(open(p("batch_manifests", f"{v}.json")))["batches"] for v in
          ("ORIGINAL", "ORIGINAL_REVERSED", "COMPACT", "SHUFFLED")}
    assert pl == {k: v for k, v in plans(rows, int(alltok.shape[0])).items()}, "batch manifests changed"
    model, t_load = load_model()
    t0 = time.time()
    val["model_pristine_all_parameters"] = verify_pristine_full(model)
    val["model_config"] = model_config_record(model)
    val["environment"] = env_record()
    assert val["model_pristine_all_parameters"], "model weights differ from the checkpoint state"
    assert not val["model_config"]["training"] and not val["model_config"]["dropout_modules_p_gt_0"]
    v1m = json.load(open(os.path.join(ROOT, FD.OUT, "logs", "cache_manifest_v1.json")))
    v2m = json.load(open(os.path.join(ROOT, FD.OUT, "cache_manifest.json")))

    # 0. ORIGINAL reference = pilot v2 cache: verify file integrity
    if "ORIGINAL" not in val:
        ok = {L: sha256_file(os.path.join(V2, f"L{L}.pt")) == v2m["files"][str(L)]["sha256"] for L in range(N_LAYERS)}
        val["ORIGINAL"] = dict(cache_dir=V2, files_match_pilot_manifest=all(ok.values()),
                               bitwise_equal_to_moe_cache_at=v2m["comparison_with_original_cache"],
                               file_sha256={str(L): v2m["files"][str(L)]["sha256"] for L in range(N_LAYERS)})
        assert all(ok.values()); jdump(val, out_path)

    # 1. negative control: ORIGINAL batches in reversed execution order -> must equal v2 bitwise
    if "negative_control" not in val:
        res = {}

        def cmp(L, X):
            R = torch.load(os.path.join(V2, f"L{L}.pt"), mmap=True)
            res[str(L)] = dict(bitwise_equal=bool(torch.equal(X, R)),
                               max_abs=float((X.float() - R.float()).abs().max()))
        t1 = time.time()
        tsha = forward_plan(model, alltok, pl["ORIGINAL_REVERSED"], rows, cmp)
        val["negative_control"] = dict(plan="ORIGINAL_REVERSED", token_sha256=tsha, per_layer=res,
                                       passed=all(r["bitwise_equal"] for r in res.values()), seconds=time.time() - t1)
        jdump(val, out_path)
        print(f"[cache] negative control (reversed execution order): passed={val['negative_control']['passed']}",
              flush=True)
        if not val["negative_control"]["passed"]:
            raise SystemExit("negative control FAILED: hidden statefulness / nondeterminism; investigate first")

    # 2./3. COMPACT and SHUFFLED caches
    for v in ("COMPACT", "SHUFFLED"):
        if val.get(v, {}).get("complete"):
            continue
        d = VDIR[v]
        os.makedirs(d, exist_ok=True)
        files = {}

        def save(L, X):
            path = os.path.join(d, f"L{L}.pt")
            torch.save(X.clone(), path + ".tmp"); os.replace(path + ".tmp", path)
            files[str(L)] = dict(shape=list(X.shape), dtype=str(X.dtype), sha256=sha256_file(path),
                                 tensor_sha256=tensor_sha(X), nan=int(torch.isnan(X.float()).sum()),
                                 inf=int(torch.isinf(X.float()).sum()))
            if v == "COMPACT":                                        # provenance: tensor-level v1 equality
                files[str(L)]["tensor_equal_v1"] = bool(torch.equal(X, torch.load(os.path.join(V1, f"L{L}.pt"), mmap=True)))
        t1 = time.time()
        tsha = forward_plan(model, alltok, pl[v], rows, save)
        ent = dict(cache_dir=d, token_sha256=tsha, files=files, seconds=time.time() - t1,
                   batch_size_pattern=BS.batch_sizes(pl[v]), complete=True)
        if v == "COMPACT":
            eq = {L: files[str(L)]["sha256"] == v1m["files"][str(L)]["sha256"] or files[str(L)]["tensor_equal_v1"]
                  for L in range(N_LAYERS)}
            ent["v1_reproduced_bitwise_per_layer"] = eq
            ent["v1_reproduced_bitwise_all_layers"] = all(eq.values())
        val[v] = ent
        val.setdefault("token_sha256", {})[v] = tsha
        jdump(val, out_path)
        print(f"[cache] {v}: built in {ent['seconds']:.0f} s" +
              (f"; v1 reproduced bitwise at all 24 layers: {ent['v1_reproduced_bitwise_all_layers']}"
               if v == "COMPACT" else ""), flush=True)
    val["token_sha256"]["ORIGINAL"] = tensor_sha(alltok[rows])
    val["token_sha256"]["ORIGINAL_REVERSED"] = val["negative_control"]["token_sha256"]
    val["semantics"] = dict(
        token_ids_identical=len(set(val["token_sha256"].values())) == 1,
        position_ids_and_mask_identical="by construction: one set of layer kwargs (batch-1 causal mask, positions, "
                                        "rotary) broadcast over every batch; recorded in model_config.layer_kwargs",
        no_padding="every row is a full 512-token chunk (no pad tokens, no attention mask beyond causal)",
        same_model_weights=val["model_pristine_all_parameters"], same_dtype=val["model_config"]["dtype"],
        eval_mode=not val["model_config"]["training"],
        batch_pattern_compact_equals_shuffled=val["COMPACT"]["batch_size_pattern"] == val["SHUFFLED"]["batch_size_pattern"])
    val["valid"] = bool(val["semantics"]["token_ids_identical"] and val["semantics"]["same_model_weights"]
                        and val["semantics"]["eval_mode"] and val["semantics"]["batch_pattern_compact_equals_shuffled"]
                        and val["negative_control"]["passed"])
    val["seconds_total"], val["model_load_s"] = time.time() - t0, t_load
    jdump(val, out_path)
    print(f"[cache] semantics valid: {val['valid']}")


# ---------------------------------------------------------------- forward diagnostics
def variant_cache(v, L):
    d = V2 if v == "ORIGINAL" else VDIR[v]
    return torch.load(os.path.join(d, f"L{L}.pt"), mmap=True)


@torch.no_grad()
def replay(model, L, X, capture_input):
    """E3 Profiler path: one sequence at a time on cached layer inputs. Returns router logits
    [n, T, E] (bf16) and, optionally, the MoE-block inputs [n, T, d] (bf16)."""
    from moe_calib import model_utils as MU
    blk = MU.moe_block(model, L)
    logits, inp = [], []
    h1 = blk.gate.register_forward_hook(lambda m, i, o: logits.append(o.detach().cpu()))
    h2 = blk.register_forward_pre_hook(lambda m, i: inp.append(i[0].detach().cpu())) if capture_input else None
    try:
        for i in range(X.shape[0]):
            MU.run_layer(model, L, X[i:i + 1].to(MU.GPU))
    finally:
        h1.remove()
        if h2:
            h2.remove()
    T = X.shape[1]
    lg = torch.stack([x.reshape(T, -1) for x in logits])
    return lg, (torch.stack([x.reshape(T, -1) for x in inp]) if capture_input else None)


def stage_forward(a):
    from moe_calib import model_utils as MU
    from moe_calib.profiler import Profiler
    proto = json.load(open(p("protocol.json")))
    val = json.load(open(p("cache_validation.json")))
    assert val["valid"], "cache semantics invalid"
    rows = proto["cache_rows"]
    union = proto["union_calibration_ids"]
    ui = FD.remap(union, rows)
    assert ui == list(range(len(union)))
    sel = json.load(open(os.path.join(ROOT, FD.OUT, "selected_sets.json")))
    pos = {r: i for i, r in enumerate(union)}
    srows = {c: [pos[r] for r in sel[c]["chosen"]] for c in STRATS}          # index into union rows
    model, t_load = load_model()
    E, k = MU.num_experts(model), MU.top_k(model)
    drift, routing, hess = {}, {}, {}
    t0 = time.time()
    for L in range(N_LAYERS):
        X = {v: variant_cache(v, L)[:len(union)].clone() for v in BS.VARIANTS}
        Xall = {v: variant_cache(v, L) for v in BS.VARIANTS}
        drift[str(L)] = {v: dict(calibration_rows=BS.activation_drift(X[v], X["ORIGINAL"]),
                                 all_345_rows=BS.activation_drift(Xall[v], Xall["ORIGINAL"]))
                         for v in ("COMPACT", "SHUFFLED")}
        diag = L in LAYERS
        lg, inp = {}, {}
        with MU.layer_on_gpu(model, L):
            for v in BS.VARIANTS:
                lg[v], inp[v] = replay(model, L, X[v], diag)
        T = X["ORIGINAL"].shape[1]
        probs, sels = {}, {}
        for v in BS.VARIANTS:                  # the model's exact op, on the GPU, one sequence ([T, E]) at a time:
            ps, ss = [], []                    # BF16 logits tie often; CPU topk breaks ties differently
            for i in range(lg[v].shape[0]):
                pr_, s_ = BS.topk_route(lg[v][i].to(MU.GPU), k)
                ps.append(pr_.cpu()); ss.append(s_.cpu())
            probs[v], sels[v] = torch.cat(ps), torch.cat(ss)
        routing[str(L)] = {v: BS.routing_compare(sels[v], sels["ORIGINAL"], probs[v], probs["ORIGINAL"], E)
                           for v in ("COMPACT", "SHUFFLED")}
        for v in ("COMPACT", "SHUFFLED"):
            per_seq = (~(torch.zeros(len(union) * T, E, dtype=torch.bool).scatter_(1, sels[v], True) ==
                         torch.zeros(len(union) * T, E, dtype=torch.bool).scatter_(1, sels["ORIGINAL"], True)).all(1))
            routing[str(L)][v]["n_sequences_with_set_change"] = int(per_seq.reshape(len(union), T).any(1).sum())
        if diag:
            hess[str(L)] = hessian_diag(model, L, X, inp, sels, srows, T, E)
        del X, Xall, lg, inp
        print(f"[forward] L{L:2d}: drift C {drift[str(L)]['COMPACT']['calibration_rows']['relative_l2']:.2e} "
              f"S {drift[str(L)]['SHUFFLED']['calibration_rows']['relative_l2']:.2e}; set changed C "
              f"{routing[str(L)]['COMPACT']['frac_any_expert_changed']:.4f} S "
              f"{routing[str(L)]['SHUFFLED']['frac_any_expert_changed']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    first = {v: next((int(L) for L in map(str, range(N_LAYERS))
                      if drift[L][v]["calibration_rows"]["frac_elements_exactly_equal"] < 1.0), None)
             for v in ("COMPACT", "SHUFFLED")}
    jdump(dict(rows="217 calibration rows (FR u FC) and all 345 cache rows", reference="ORIGINAL", per_layer=drift,
               first_layer_with_differences=first,
               note="layer L = input of decoder layer L; layer 0 input = token embeddings"), p("activation_drift.json"))
    jdump(dict(rows="217 calibration rows", path="E3 Profiler replay (one sequence at a time) of each variant's "
               "cached input; selection = softmax(float) + topk(4), the model's op", per_layer=routing),
          p("routing_disagreement.json"))
    jdump(dict(layers=LAYERS, experts=DIAG_EXPERTS, per_layer=hess, seconds=time.time() - t0, model_load_s=t_load),
          p("hessian_sensitivity.json"))


def hessian_diag(model, L, X, inp, sels, srows, T, E):
    """Existing E3 Profiler Hessians for the prespecified subset + routing-membership decomposition (gu)."""
    from moe_calib.profiler import Profiler
    sel = json.load(open(os.path.join(ROOT, FD.OUT, "selected_sets.json")))
    out = {}
    for c in STRATS:
        idx = srows[c]
        tok_idx = (torch.tensor(idx)[:, None] * T + torch.arange(T)[None]).flatten()
        H = {}
        for v in BS.VARIANTS:
            pf = Profiler(model, L, K=sel[c]["K"]).run(X[v], idx)
            H[v] = {(e, pj): (pf.acc.H[(L, e, pj)].sum(0).cpu(), float(pf.acc.raw_n[(L, e, pj)].sum()))
                    for e in DIAG_EXPERTS for pj in ("gu", "down")}
            pf.free(); del pf
        res = {}
        for e in DIAG_EXPERTS:
            mo = BS.expert_token_mask(sels["ORIGINAL"][tok_idx], e)
            Xo = inp["ORIGINAL"].reshape(-1, inp["ORIGINAL"].shape[-1])[tok_idx]
            Ho_full = H["ORIGINAL"][(e, "gu")][0].double()
            check = float((BS.gram(Xo[mo]).double() - Ho_full).norm() / Ho_full.norm())
            for v in ("COMPACT", "SHUFFLED"):
                mv = BS.expert_token_mask(sels[v][tok_idx], e)
                Xv = inp[v].reshape(-1, inp[v].shape[-1])[tok_idx]
                common, only_o, only_v = mo & mv, mo & ~mv, mv & ~mo
                memb = dict(tokens_original=int(mo.sum()), tokens_variant=int(mv.sum()), tokens_common=int(common.sum()),
                            tokens_only_original=int(only_o.sum()), tokens_only_variant=int(only_v.sum()),
                            membership_changed=bool((only_o | only_v).any()))
                act = (BS.gram(Xv[common]).double() - BS.gram(Xo[common]).double()).norm() / Ho_full.norm()
                mem = (BS.gram(Xv[only_v]).double() - BS.gram(Xo[only_o]).double()).norm() / Ho_full.norm()
                for pj in ("gu", "down"):
                    (Hv, nv), (Ho, no) = H[v][(e, pj)], H["ORIGINAL"][(e, pj)]
                    r = dict(BS.hessian_compare(Hv, Ho), profiler_tokens_variant=nv, profiler_tokens_original=no, **memb)
                    if pj == "gu":
                        r.update(decomp_rel_fro_common_tokens_activation=float(act),
                                 decomp_rel_fro_membership_change=float(mem),
                                 decomp_gram_vs_profiler_rel=check)
                    r["routing_matches_profiler"] = bool(nv == r["tokens_variant"] and no == r["tokens_original"])
                    res[f"{v}|e{e}|{pj}"] = r
        out[c] = res
    return out


# ---------------------------------------------------------------- quantization (one config per process)
def wdir(c, v):
    return os.path.join(BCACHE, "weights", cfg_name(c, v))


def stage_quant(a):
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    c, v = a.strategy, a.variant
    name = cfg_name(c, v)
    os.makedirs(p("results"), exist_ok=True)
    if os.path.exists(p("results", f"{name}.json")):
        print(f"[quant] {name} done"); return
    cfg, pilot, sel = frozen()
    proto = json.load(open(p("protocol.json")))
    val = json.load(open(p("cache_validation.json")))
    assert val["valid"]
    if v != "ORIGINAL":
        reg = json.load(open(p("regression_checks.json")))
        assert reg["passed"], "ORIGINAL regression has not passed"
        for L in LAYERS:                                              # cache integrity
            assert sha256_file(os.path.join(VDIR[v], f"L{L}.pt")) == val[v]["files"][str(L)]["sha256"]
    rows = proto["cache_rows"]
    test_ids = list(range(*cfg["split"]["test"]))
    if v == "ORIGINAL":
        cdir, chosen, ecache = OCACHE, sel[c]["chosen"], None            # frozen E3 cache, original ids
    else:
        cdir, chosen, ecache = VDIR[v], FD.remap(sel[c]["chosen"], rows), OCACHE
    win, prot = EP.load_windows()
    assert prot["windows_sha256"] == proto["windows_sha256"]
    model, t_load = load_model()
    pr = json.load(open(os.path.join(ROOT, FD.OUT, "pristine_digests.json")))
    t0 = time.time()
    tg0 = PQ.target_tensor_digests(model, LAYERS)
    assert tg0 == {k: pr["pristine_targets"][k] for k in tg0}, "targets not pristine"
    u0 = PQ.untouched_digests(model, LAYERS)
    t_verify = time.time() - t0
    orig = {(L, e): {k: w.detach().cpu().clone() for k, w in MU.get_expert_weights(model, L, e).items()}
            for L, e in EP.CHECK_EXPERTS}
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    g = cfg["gptq"]
    q = PQ.quantize_install(model, cdir, LAYERS, chosen, test_ids, sel[c]["K"], g["bits"], g["group_size"],
                            g["percdamp"], capture=EP.CHECK_EXPERTS, eval_cache_dir=ecache)
    t_q = time.time() - t1
    checks = []
    for L, e in EP.CHECK_EXPERTS:
        Xc = PQ.expert_inputs(model, L, e, load_layer_cache(OCACHE, L)[test_ids[:EP.N_CHECK_SEQS]])
        ck = PQ.expert_output_check(model, L, e, q["captured"][(L, e)], orig[(L, e)], Xc)
        ck["tolerance"] = EP.INSTALL_TOL(ck)
        ck["pass"] = bool(ck["installed_equals_bf16_of_e3_wq"] and ck["installed_differs_from_bf16_original"]
                          and ck["installed_vs_e3_rel"] <= ck["tolerance"])
        checks.append(ck)
    after = PQ.target_tensor_digests(model, LAYERS)
    unchanged = [kk for kk in after if after[kk] == tg0[kk]]
    nonfin = PQ.nonfinite_targets(model, LAYERS)
    untouched_ok = PQ.untouched_digests(model, LAYERS) == u0
    dig = PQ.installed_digest(model, LAYERS)
    os.makedirs(wdir(c, v), exist_ok=True)
    files = {}
    for L in LAYERS:
        path = os.path.join(wdir(c, v), f"L{L}.pt")
        PQ.save_layer(model, L, path)
        files[str(L)] = dict(sha256=sha256_file(path), layer_sha256=PQ.layer_digest(model, L))
    evals = []
    for _ in range(2):
        t2 = time.time()
        nll, cnt = EP.layerwise_nll(model, win)
        evals.append(EP.ppl_record(nll, cnt, time.time() - t2))
    rec = evals[0]
    ok = (q["counts"]["n_zero_token_quantization_fallback"] == 0 and all(ck["pass"] for ck in checks) and
          not unchanged and not nonfin and untouched_ok and bool(np.isfinite(rec["per_window_nll"]).all()))
    out = dict(strategy=c, variant=v, name=name, ids_hash=sel[c]["ids_hash"], calibration_cache=cdir,
               reconstruction_cache=ecache or OCACHE, gptq=g, layers=LAYERS,
               n_quantized_experts=q["counts"]["n_experts_quantized_total"],
               rtn_fallbacks=q["counts"]["n_zero_token_quantization_fallback"],
               n_experts_scored=q["counts"]["n_experts_test_observed"],
               recon=float(np.mean([r["mean"] for r in q["results"]])),
               per_expert=[dict(layer=r["layer"], expert=r["expert"], tokens=r["tokens"], mean=r["mean"])
                           for r in q["results"]],
               projections_unchanged_from_bf16=len(unchanged), nonfinite_weights=len(nonfin),
               untouched_params_unchanged=untouched_ok, n_untouched_params=len(u0), expert_checks=checks,
               installed_weights_sha256=dig, weight_files=files, weights_dir=wdir(c, v),
               repeat_eval_abs_diff_ppl=abs(evals[0]["ppl"] - evals[1]["ppl"]),
               model_load_s=t_load, pristine_verify_s=t_verify, quantize_s=t_q, hessian_s=q["hessian_s"],
               gptq_s=q["gptq_s"], peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=FD.peak_cpu_gb(),
               valid=bool(ok), **rec)
    jdump(out, p("results", f"{name}.json"))
    print(f"[quant] {name}: PPL {rec['ppl']:.6f} recon {out['recon']:.7f} sha {dig[:16]} valid {ok} "
          f"({t_q:.0f} s)", flush=True)
    if not ok:
        raise SystemExit(f"{name} failed its correctness checks")


def stage_regression(a):
    proto = json.load(open(p("protocol.json")))
    ref = proto["references"]["partial_ppl"]
    out = dict(configs={})
    for c in STRATS:
        r = json.load(open(p("results", f"{cfg_name(c, 'ORIGINAL')}.json")))
        out["configs"][c] = dict(ppl=r["ppl"], ppl_reference=ref[c]["ppl"], ppl_exact=r["ppl"] == ref[c]["ppl"],
                                 sha256=r["installed_weights_sha256"], sha256_reference=ref[c]["installed_weights_sha256"],
                                 weights_exact=r["installed_weights_sha256"] == ref[c]["installed_weights_sha256"],
                                 recon=r["recon"], recon_reference=ref[c]["recon"], recon_exact=r["recon"] == ref[c]["recon"])
    out["passed"] = all(x["ppl_exact"] and x["weights_exact"] and x["recon_exact"] for x in out["configs"].values())
    jdump(out, p("regression_checks.json"))
    print(f"[regression] ORIGINAL reproduces partial-PPL FR/FC exactly: {out['passed']}")
    if not out["passed"]:
        raise SystemExit("ORIGINAL regression FAILED")


# ---------------------------------------------------------------- weight comparison
def stage_weights(a):
    out = {}
    for c in STRATS:
        ro = json.load(open(p("results", f"{cfg_name(c, 'ORIGINAL')}.json")))
        for v in ("COMPACT", "SHUFFLED"):
            rv = json.load(open(p("results", f"{cfg_name(c, v)}.json")))
            per_layer, per_expert, tot = {}, [], dict(n=0, diff=0, d2=0.0, o2=0.0, mx=0.0)
            for L in LAYERS:
                fo, fv = os.path.join(ro["weights_dir"], f"L{L}.pt"), os.path.join(rv["weights_dir"], f"L{L}.pt")
                assert sha256_file(fo) == ro["weight_files"][str(L)]["sha256"]
                assert sha256_file(fv) == rv["weight_files"][str(L)]["sha256"]
                Wo, Wv = torch.load(fo), torch.load(fv)
                lt = dict(n=0, diff=0, d2=0.0, o2=0.0, mx=0.0)
                for key in sorted(Wo, key=lambda s: (int(s.split(".")[0]), s)):
                    o, w = Wo[key].float(), Wv[key].float()
                    cm = BS.weight_compare(w, o)
                    e, proj = key.split(".")
                    per_expert.append(dict(layer=L, expert=int(e), proj=proj, **cm))
                    for acc in (lt, tot):
                        acc["n"] += o.numel(); acc["diff"] += int((o != w).sum())
                        acc["d2"] += float(((w - o) ** 2).sum()); acc["o2"] += float((o ** 2).sum())
                        acc["mx"] = max(acc["mx"], cm["max_abs"])
                per_layer[str(L)] = dict(frac_differ=lt["diff"] / lt["n"], rel_l2=(lt["d2"] / lt["o2"]) ** .5, max_abs=lt["mx"])
            rel = np.array([x["rel_l2"] for x in per_expert])
            out[f"{c}|{v}"] = dict(frac_differ=tot["diff"] / tot["n"], rel_l2=(tot["d2"] / tot["o2"]) ** .5,
                                   max_abs=tot["mx"], per_layer=per_layer,
                                   per_expert_rel_l2_quantiles={q: float(np.quantile(rel, q)) for q in (0, .1, .5, .9, 1)},
                                   n_projections_identical=int(sum(x["frac_differ"] == 0 for x in per_expert)),
                                   per_expert=per_expert)
            print(f"[weights] {c} {v}: {out[f'{c}|{v}']['frac_differ']:.3f} of weights differ, rel L2 "
                  f"{out[f'{c}|{v}']['rel_l2']:.3e}", flush=True)
    reg = json.load(open(p("regression_checks.json")))
    jdump(dict(comparisons=out, scales_zero_points="not exposed by the frozen solver (gptq_quantize returns only the "
               "dequantized matrix); the act-order permutation / group assignment is compared on the Hessian subset",
               solver_determinism=dict(identical_cache_gives_identical_weights=reg["passed"],
                                       evidence="ORIGINAL weights in this study (fresh process) are bitwise equal to the "
                                                "partial-PPL run's installed weights (sha256)")),
          p("weight_sensitivity.json"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["preflight", "cache", "forward", "quant", "regression", "weights", "report"])
    ap.add_argument("--strategy", choices=list(STRATS))
    ap.add_argument("--variant", choices=list(BS.VARIANTS))
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True); os.makedirs(p("figures"), exist_ok=True)
    if a.stage == "report":
        import report_batch_sensitivity as R
        R.main(); return
    dict(preflight=stage_preflight, cache=stage_cache, forward=stage_forward, quant=stage_quant,
         regression=stage_regression, weights=stage_weights)[a.stage](a)


if __name__ == "__main__":
    main()
