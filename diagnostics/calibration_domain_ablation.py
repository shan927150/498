"""C4 vs WikiText-2-train calibration: 2 x 2 domain-matching ablation (source {C4, WT} x selection {FR, SC}).

Controlled dataset ablation. Nothing new is optimized: the WT arm re-uses the EXISTING E3 GPTQ solver, the EXISTING
shared covariance-aware selector (moe_calib/shared_covariance_selection + the sketch pass of
diagnostics/shared_covariance_e3, same SETTINGS), the EXISTING warm-start / FR / grouping conventions
(moe_calib/calibration_domain re-states them; tests check they reproduce the stored C4 selections), the frozen
partial-PPL windows and installer checks (diagnostics/eval_partial_ppl via shared_covariance_e3.install_and_ppl) and the
loss-alignment MVP B fixed-input metrics (P_seq, B2) on the frozen 32-window BF16 capture.

Stages (--stage):
  audit      CPU  artifact-availability audit -> artifact_audit.md / audit.json
  pool       CPU  WikiText-2 TRAIN candidate pool (2,000 x 512) + leakage checks -> dataset_manifest.json
  protocol   CPU  freeze everything (before any new GPU result) -> protocol.json
  cache      GPU  full model: BF16 layer inputs of the WT pool, ORIGINAL batching (consecutive batches of 8)
  routing    GPU  partial: WT pool routing matrix (E3.routing_profile, model top-k) -> STORE/routing_profile_v2.npz
  sketch     GPU  partial: SC sketches + full-dim pool Hessians from the WT pool (shared_covariance_e3.stage_sketch)
  select     GPU  WT warm start, WT-FR, WT-SC (twice, determinism) -> selection_manifest.json; freeze
  gmanifest  CPU  WT grouping manifests (3 prespecified grouping seeds)
  gcache     GPU  full model: WT grouping caches (+ reversed-order bitwise control)
  quant      GPU  full model: --configs SRC:STRAT:EXT ... (EXT = ORIG or G<seed>); quantize, install, checks, NLL,
                  C4-test reconstruction, WikiText P_seq / B2, restore; optional --bf16 baseline check first
  report     CPU  -> diagnostics/report_calibration_domain_ablation.py
"""
from __future__ import annotations

import argparse, hashlib, json, os, random, resource, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
from moe_calib import calibration_domain as CD        # noqa: E402

OUT = "runs_calibration_domain_ablation"
STORE = os.path.expanduser(os.environ.get("CDA_STORE", "~/moe_cache_calibration_domain"))
WCACHE = os.path.join(STORE, "wt_pool")                 # WT pool layer caches (rows 0..1999 = pool ids)
OCACHE = os.path.expanduser("~/moe_cache")
LAYERS = [1, 8, 16, 23]
E, K, B, SEQLEN, POOL_N = 60, 4, 128, 512, 2000
SEL_SEED = 0
WT_POOL_SEED = 20261117                                  # prespecified; independent of every other seed
GSEEDS = [20261111, 20261112, 20261113]                  # first three of the Stage B grouping seeds
SOURCES, STRATS = ("C4", "WT"), ("FR", "SC")
CAPTURE_MANIFEST = "runs_loss_alignment/mvp_b/capture_manifest.json"
FROZEN_CODE = ("moe_calib/gptq.py", "moe_calib/quant_eval.py", "moe_calib/profiler.py", "moe_calib/hessian.py",
               "moe_calib/model_utils.py", "e3_allocation.py", "moe_calib/partial_model_quant.py",
               "diagnostics/eval_partial_ppl.py", "moe_calib/shared_covariance_selection.py",
               "moe_calib/expertwise_calibration.py", "diagnostics/shared_covariance_e3.py",
               "moe_calib/multigroup_robustness.py", "diagnostics/batch_sensitivity.py",
               "moe_calib/loss_alignment.py", "configs/frozen_b4.json")
EXP_CODE = ("moe_calib/calibration_domain.py", "diagnostics/calibration_domain_ablation.py")


# ===================================================================== io
def p(*a):
    return os.path.join(ROOT, OUT, *a)


def st(*a):
    return os.path.join(STORE, *a)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def jload(path):
    return json.load(open(path if os.path.isabs(path) else os.path.join(ROOT, path)))


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def obj_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def hashes(files):
    return {f: sha256_file(os.path.join(ROOT, f)) for f in files}


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def gpu_peak():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def env():
    import platform, transformers
    return dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32, omp_threads=os.environ.get("OMP_NUM_THREADS"))


def ids_hash(ids):
    import e3_allocation as E3
    return E3.warm_hash(ids)


def sce():
    """diagnostics/shared_covariance_e3 (C4 SC study): selections, settings, installer, sketch pass."""
    import shared_covariance_e3 as SCE
    return SCE


# ===================================================================== C4 arm (existing, validated)
def c4_selection(c):
    SCE = sce()
    s = SCE.selection(c, SEL_SEED)                     # FR: re-validated baseline; SC: frozen SC E3 selection
    if c == "SC":
        assert s["ids_hash"] == SCE.jload(SCE.p("freeze.json"))["sc_selection_hashes"][str(SEL_SEED)]
    return s


def c4_grouping(g):
    """SC E3 Stage B grouping cache of allocation seed 0 (union FR u TR u TC u SC), with integrity checks."""
    SCE = sce()
    gp = SCE.gprotocol()
    union = gp["seeds"][str(SEL_SEED)]["union"]
    ent = SCE.jload(SCE.p("grouping_robustness", f"cache_manifest_s{SEL_SEED}.json"))["entries"][SCE.gname(g)]
    assert ent["complete"]
    return dict(dir=SCE.gcache_dir(SEL_SEED, g), union=union, files=ent["files"], manifest_sha256=ent["manifest_sha256"])


# ===================================================================== audit
def stage_audit(a):
    SCE = sce()
    rows, ok = [], True

    def add(item, path, status, note=""):
        rows.append(dict(item=item, path=path, status=status, note=note))

    for c in STRATS:
        s = c4_selection(c)
        add(f"C4-{c} s0 selection", "runs_partial_ppl/selected_set_manifest.json" if c == "FR"
            else "runs_shared_covariance_e3/selections/seed0.json", "reusable", f"ids_hash {s['ids_hash']}")
        r = jload(f"runs_shared_covariance_e3/results/A_{c}_s0.json")
        add(f"C4-{c} s0 ORIGINAL: test recon / NLL / installed sha", f"runs_shared_covariance_e3/results/A_{c}_s0.json",
            "reusable", f"mean_nll {r['mean_nll']:.6f}, sha {r['installed_weights_sha256'][:12]}, install_ok {r['install_ok']}"
            + (", FR reproduced the stored partial-PPL run bitwise" if c == "FR" else ""))
        for g in GSEEDS:
            f = f"runs_shared_covariance_e3/grouping_robustness/results/{c}_s0_{SCE.gname(g)}.json"
            r = jload(f)
            add(f"C4-{c} s0 {SCE.gname(g)} grouping result", f, "reusable (NLL, recon, sha); B2 needs the weights",
                f"mean_nll {r['mean_nll']:.6f}, sha {r['installed_weights_sha256'][:12]}")
    for g in GSEEDS:
        cg = c4_grouping(g)
        present = all(os.path.exists(os.path.join(cg["dir"], f"L{L}.pt")) for L in LAYERS)
        ok &= present
        add(f"C4 s0 grouping cache {SCE.gname(g)}", cg["dir"], "present" if present else "MISSING",
            f"{len(cg['union'])} rows (union FR u TR u TC u SC), sha256 recorded in cache_manifest_s0.json")
    for c, d in (("FR", os.path.expanduser("~/moe_cache_batch_sensitivity/weights/FR_ORIGINAL")),
                 ("SC", os.path.expanduser("~/moe_cache_loss_alignment/weights/SC_s0"))):
        add(f"C4-{c} s0 ORIGINAL effective weights", d, "present" if os.path.isdir(d) else "MISSING",
            "B2 / P_seq already measured in runs_loss_alignment/mvp_b (raw_sums.json)")
    man = jload(CAPTURE_MANIFEST)
    cap_ok = man["ok"] and all(os.path.exists(f["path"]) for f in man["files"].values())
    ok &= cap_ok
    add("WikiText BF16 reference activations (32 frozen test windows, 4 layers)", os.path.dirname(man["files"]["L1"]["path"]),
        "reusable" if cap_ok else "MISSING", "loss-alignment MVP B capture; sha256-checked at load")
    add("Frozen WikiText-2 test windows (128 x 512)", "runs_partial_ppl/windows.pt", "reusable",
        jload("runs_partial_ppl/evaluation_protocol.json")["windows_sha256"][:16])
    add("BF16 partial-PPL baseline", "runs_partial_ppl/bf16_baseline.json", "reusable (re-checked in the first quant run)",
        f"PPL {jload('runs_partial_ppl/bf16_baseline.json')['ppl']:.6f}")
    for item in ("WikiText-train 2,000-sequence candidate pool", "WT pool BF16 layer-input caches (L1/8/16/23)",
                 "WT pool routing profile", "WT SC sketches + full-pool references", "WT warm start / WT-FR / WT-SC selections",
                 "WT grouping caches (3)", "WT quantized configurations"):
        add(item, STORE, "MISSING -> generated by this experiment")
    new_runs = ["WT-FR ORIG", "WT-SC ORIG"] + [f"WT-{c} G{g}" for g in GSEEDS for c in STRATS] + \
               [f"C4-{c} G{g} (re-run of a stored Stage B config, needed for B2; bitwise-checked)" for g in GSEEDS for c in STRATS] + \
               ["C4-FR ORIG (validation re-run: NLL, sha and MVP B B2 must reproduce bitwise)"]
    jdump(dict(rows=rows, required_new_gptq_runs=new_runs, all_required_inputs_present=bool(ok),
               created=time.strftime("%F %T")), p("audit.json"))
    L = ["# Artifact audit — calibration-domain ablation\n",
         "| item | location | status | note |", "|---|---|---|---|"]
    L += [f"| {r['item']} | `{r['path']}` | {r['status']} | {r['note']} |" for r in rows]
    L += ["", "## Required new GPTQ runs", ""] + [f"- {x}" for x in new_runs]
    L += ["", "C4 Stage B configurations are re-run (not only reused) because their effective weights were not saved and "
              "B2 / P_seq need them; each re-run must reproduce the stored installed-weight sha256, reconstruction and "
              "per-window NLL bitwise, otherwise the stored values are not used.",
          "", "Compatibility notes:",
          "- C4 grouping caches batch the union of FOUR seed-0 strategies (FR, TR, TC, SC; 374 rows); the WT caches batch "
          "the union of WT-FR and WT-SC. Same construction rule and grouping seeds; groupings are matched by protocol "
          "and seed, not by identical activations (C4 and WT rows are different texts anyway).",
          "- ORIGINAL extraction (consecutive batches of 8 over the pool cache rows) is reported separately and never "
          "averaged with the grouping replicates."]
    open(p("artifact_audit.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ===================================================================== pool
def stage_pool(a):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import datasets, transformers
    import eval_partial_ppl as EP
    from moe_calib import model_utils as MU
    out = p("dataset_manifest.json")
    if os.path.exists(out) and os.path.exists(st("wt_pool", "ids.pt")):
        print("[pool] done"); return
    path = MU._snapshot_dir(MU.MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(path)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    enc = tok(text, return_tensors="pt").input_ids[0]
    n_full = CD.partition_windows(enc.shape[0], SEQLEN)
    idx = CD.sample_pool(n_full, POOL_N, WT_POOL_SEED)
    pool = CD.windows_from_ids(enc, idx, SEQLEN)
    # held-out test windows (frozen) and validation split, for leakage checks only
    win, eprot = EP.load_windows()
    test_rows = len(load_dataset("wikitext", "wikitext-2-raw-v1", split="test"))
    val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    venc = tok("\n\n".join(val["text"]), return_tensors="pt").input_ids[0]
    vwin = CD.windows_from_ids(venc, range(venc.shape[0] // SEQLEN), SEQLEN)
    leak_test = CD.ngram_overlap(pool.numpy(), win.numpy(), 32)
    leak_val = CD.ngram_overlap(pool.numpy(), vwin.numpy(), 32)
    c4 = torch.load(os.path.join(OCACHE, "ids.pt"))["pool"]
    cache_files = [f["filename"] for f in getattr(ds, "cache_files", [])]
    man = dict(
        dataset=dict(hf_id="wikitext", config="wikitext-2-raw-v1", split="train", n_rows=len(ds),
                     datasets_version=datasets.__version__, fingerprint=getattr(ds, "_fingerprint", None),
                     cache_revision=[c for c in cache_files[0].split(os.sep) if len(c) == 40][0] if cache_files else None,
                     text_sha256=hashlib.sha256(text.encode()).hexdigest(), test_split_rows=test_rows),
        tokenizer=dict(name=MU.MODEL_NAME, snapshot=os.path.basename(path), cls=type(tok).__name__,
                       transformers_version=transformers.__version__,
                       same_snapshot_as_ppl_protocol=os.path.basename(path) == eprot["tokenizer"]["snapshot"]),
        preprocessing='"\\n\\n".join(train["text"]); one tokenizer call on the whole text; no special tokens added '
                      '(identical to the frozen WikiText-2 test-window protocol, applied to the train split)',
        packing=f"non-overlapping {SEQLEN}-token windows i*{SEQLEN}..(i+1)*{SEQLEN}; tail discarded",
        total_tokens=int(enc.shape[0]), n_complete_windows=n_full, pool_size=POOL_N, sequence_length=SEQLEN,
        sampling=f"sorted(random.Random({WT_POOL_SEED}).sample(range({n_full}), {POOL_N})); pool row r = window idx[r]",
        sampling_seed=WT_POOL_SEED, window_indices=idx, token_offsets=[i * SEQLEN for i in idx],
        pool_token_sha256=CD.token_sha(pool),
        leakage=dict(split_boundary="calibration, routing profiling, SC sketches and selection read ONLY these "
                                    "train-split windows; test windows are used for NLL / B2 evaluation only",
                     vs_frozen_test_windows=leak_test, vs_validation_windows=leak_val,
                     test_windows_sha256=eprot["windows_sha256"]),
        c4_pool=dict(rows=int(c4.shape[0]), seqlen=int(c4.shape[1]), token_sha256=CD.token_sha(c4),
                     construction="data_utils.get_splits: one random 512-token window per C4 document (shard 00000), "
                                  "data_seed 0"),
        equal_counts_and_lengths=bool(c4.shape[0] == pool.shape[0] == POOL_N and c4.shape[1] == pool.shape[1] == SEQLEN),
        created=time.strftime("%F %T"))
    assert man["equal_counts_and_lengths"], "C4 and WT pools differ in size / length"
    assert leak_test["exact_window_matches"] == 0, "a test window is in the calibration pool"
    os.makedirs(WCACHE, exist_ok=True)
    torch.save({"pool": pool, "window_indices": torch.tensor(idx)}, st("wt_pool", "ids.pt"))
    man["ids_file_sha256"] = sha256_file(st("wt_pool", "ids.pt"))
    jdump(man, out)
    print(f"[pool] {enc.shape[0]} tokens, {n_full} windows, pool {POOL_N}; test 32-gram overlap "
          f"{leak_test['fraction']:.2e} ({leak_test['test_ngrams_in_pool']}), exact {leak_test['exact_window_matches']}; "
          f"val 32-gram overlap {leak_val['fraction']:.2e}")


def wt_tokens():
    d = torch.load(st("wt_pool", "ids.pt"))
    assert CD.token_sha(d["pool"]) == jload(p("dataset_manifest.json"))["pool_token_sha256"]
    return d["pool"]


# ===================================================================== protocol
def stage_protocol(a):
    SCE = sce()
    if os.path.exists(p("protocol.json")):
        protocol(); print("[protocol] already frozen"); return
    dm = jload(p("dataset_manifest.json"))
    eprot = jload("runs_partial_ppl/evaluation_protocol.json")
    # prespecified reference variability: within-seed SD over the 5 Stage B groupings of the paired C4 SC-FR dNLL
    ref = {}
    for s in (0, 1, 2):
        d = [jload(f"runs_shared_covariance_e3/grouping_robustness/results/SC_s{s}_G{g}.json")["mean_nll"] -
             jload(f"runs_shared_covariance_e3/grouping_robustness/results/FR_s{s}_G{g}.json")["mean_nll"]
             for g in SCE.GSEEDS]
        ref[str(s)] = float(np.std(d, ddof=1))
    ref_sd = float(np.sqrt(np.mean(np.square(list(ref.values())))))
    body = dict(
        experiment="C4 vs WikiText-2-train calibration: 2x2 domain-matching ablation (source x {FR, SC})",
        model="Qwen/Qwen1.5-MoE-A2.7B", layers=LAYERS, quantized="routed-expert gate/up/down of layers 1/8/16/23 only",
        gptq=SCE.SETTINGS["gptq"], calibration="non-sequential GPTQ on cached BF16 layer inputs (E3 solver, K=4)",
        budget=B, pool_size=POOL_N, seqlen=SEQLEN, selection_seed=SEL_SEED,
        sources=dict(
            C4=dict(pool="~/moe_cache rows 0-1999 (existing)", FR="runs_partial_ppl FR s0 (matched warm start)",
                    SC="runs_shared_covariance_e3/selections/seed0.json (frozen)"),
            WT=dict(pool=f"WikiText-2 train, dataset_manifest.json (pool sha {dm['pool_token_sha256'][:16]})",
                    warm_start="same convention as C4: typical_half (1000 lowest routing-L1 d(s) on the WT pool routing) "
                               "& random_half (random.Random(20261106).sample(ids, 1000)); W = random.Random(0).sample(sorted"
                               "(T & H), 32)",
                    FR="W + random.Random(0).sample([WT pool ids not in W], 96) (e3_allocation random + fixed warm start)",
                    SC="shared_covariance_selection.select_shared(seed 0) from W with SETTINGS of the C4 SC study; sketches, "
                       "routing and pool references recomputed from the WT pool ONLY")),
        sc_settings=SCE.SETTINGS,
        extraction=dict(
            ORIGINAL="pool cache built with consecutive batches of 8 pool rows (cache_inputs.py convention); reported "
                     "separately, never averaged with grouping replicates",
            groupings=dict(seeds=GSEEDS, rule="sorted(union of the source's selections) -> numpy default_rng(g)."
                                                "permutation -> consecutive batches of 8 (multigroup_robustness.grouping_batches)",
                           C4="reuse SC E3 Stage B seed-0 caches (union FR u TR u TC u SC); configs re-run for B2 and "
                              "required to reproduce the stored results bitwise",
                           WT="union WT-FR u WT-SC")),
        evaluation=dict(
            primary="token-weighted mean NLL on the frozen 128 x 512 WikiText-2 test windows (sha "
                    f"{eprot['windows_sha256'][:16]}), sequential partial-model forward (eval_partial_ppl.layerwise_nll)",
            secondary=["PPL = exp(mean NLL)",
                       "P_seq and B2 (pooled, layer-equal, per layer) on the loss-alignment 32-window BF16 capture, FP32 "
                       "expert compute (loss_alignment.layer_output_errors)",
                       "C4 held-out test reconstruction (E3 test rows 2128-2255 of ~/moe_cache), out-of-domain for WT"]),
        analysis=dict(
            per_grouping="dSC_C4 = NLL(C4,SC)-NLL(C4,FR); dSC_WT = NLL(WT,SC)-NLL(WT,FR); dDomain_FR = NLL(WT,FR)-"
                         "NLL(C4,FR); dDomain_SC; interaction = dSC_WT - dSC_C4 (negative favours SC under WT)",
            summary="mean, SD, min, max, sign counts over the 3 groupings; exploratory, no significance claims",
            reference_sd=dict(value=ref_sd, per_seed=ref,
                              definition="root-mean-square over allocation seeds 0-2 of the within-seed SD (ddof 1) of the "
                                         "C4 SC-FR paired dNLL over the 5 SC E3 Stage B groupings"),
            label_rule="CONSISTENT-{NEGATIVE,POSITIVE} iff the same sign in 3/3 groupings AND |mean| > reference_sd; "
                       "otherwise NOT-DISTINGUISHABLE (applied to dSC_C4, dSC_WT, dDomain_FR, dDomain_SC, interaction)"),
        frozen_code_sha256=hashes(FROZEN_CODE))
    body["protocol_sha256"] = obj_sha(body)
    body["created"] = time.strftime("%F %T")
    jdump(body, p("protocol.json"))
    print(f"[protocol] frozen {body['protocol_sha256'][:16]}; reference SD {ref_sd:.6f} ({ref})")


def protocol():
    pr = jload(p("protocol.json"))
    body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created")}
    assert obj_sha(body) == pr["protocol_sha256"], "protocol.json modified"
    assert hashes(FROZEN_CODE) == pr["frozen_code_sha256"], "frozen solver / selector / evaluation code changed"
    return pr


# ===================================================================== WT pool cache (ORIGINAL batching)
@torch.no_grad()
def stage_cache(a):
    from moe_calib import model_utils as MU
    protocol()
    if os.path.exists(st("wt_pool", "meta.json")):
        print("[cache] done"); return
    ids = wt_tokens()
    os.environ.pop("MOE_LOAD_LAYERS", None)
    t0 = time.time()
    model, _ = MU.load_model()
    t_load = time.time() - t0
    N, T = ids.shape
    Bt = 8                                                   # cache_inputs.py default
    t0 = time.time()
    hs = torch.empty(N, T, MU.expert_dims(model)[0], dtype=MU.DTYPE)
    for i in range(0, N, Bt):
        hs[i:i + Bt] = MU.embed(model, ids[i:i + Bt]).cpu()
    files = {}
    last = max(LAYERS)
    for L in range(last + 1):
        if L in LAYERS:
            f = os.path.join(WCACHE, f"L{L}.pt")
            torch.save(hs, f + ".tmp"); os.replace(f + ".tmp", f)
            files[str(L)] = dict(sha256=sha256_file(f), bytes=os.path.getsize(f), finite=bool(torch.isfinite(hs.float()).all()))
            print(f"[cache] saved L{L}", flush=True)
        if L == last:
            break
        with MU.layer_on_gpu(model, L):
            for i in range(0, N, Bt):
                hs[i:i + Bt] = MU.run_layer(model, L, hs[i:i + Bt].to(MU.GPU)).cpu()
    el = time.time() - t0
    meta = dict(pool_size=N, holdout_size=0, seqlen=T, layers=LAYERS, model=MU.MODEL_NAME, num_layers=MU.num_layers(model),
                source="WikiText-2 train (dataset_manifest.json)", batching="consecutive batches of 8 rows from row 0",
                cache_seconds=el, model_load_s=t_load, files=files, token_sha256=CD.token_sha(ids),
                peak_gpu_gb=gpu_peak(), peak_cpu_gb=rss_gb(), env=env())
    jdump(meta, st("wt_pool", "meta.json"))
    print(f"[cache] done {el:.0f}s")


def check_wcache():
    meta = jload(st("wt_pool", "meta.json"))
    assert meta["token_sha256"] == jload(p("dataset_manifest.json"))["pool_token_sha256"]
    return meta


# ===================================================================== routing
def stage_routing(a):
    protocol()
    out = st("routing_profile_v2.npz")
    if os.path.exists(p("wt_routing.json")):
        print("[routing] done"); return
    check_wcache()
    SCE = sce()
    model, t_load = SCE.load_partial()
    import e3_allocation as E3
    t0 = time.time()
    C, pu = E3.routing_profile(model, WCACHE, LAYERS, list(range(POOL_N)))
    t = time.time() - t0
    np.savez(out, C=C, p=pu, layers=np.array(LAYERS), pool_ids=np.array(list(range(POOL_N))), E=E, rule="model_topk")
    tok = C.reshape(POOL_N, 4, E)
    jdump(dict(file=out, sha256=sha256_file(out), seconds=t, model_load_s=t_load,
               tokens_per_seq_per_layer_equal_topk_x_T=bool(np.all(tok.sum(2) == K * SEQLEN)),
               pool_tokens_per_expert=dict(min=int(tok.sum(0).min()), max=int(tok.sum(0).max())),
               experts_without_pool_tokens=int((tok.sum(0) == 0).sum()), created=time.strftime("%F %T")), p("wt_routing.json"))
    print(f"[routing] {t:.0f}s; min pool tokens/expert {int(tok.sum(0).min())}")


def routing():
    r = jload(p("wt_routing.json"))
    assert sha256_file(r["file"]) == r["sha256"]
    z = np.load(r["file"])
    assert z["layers"].tolist() == LAYERS and z["pool_ids"].tolist() == list(range(POOL_N)) and str(z["rule"]) == "model_topk"
    return z


# ===================================================================== sketch (existing SC sketch pass, WT inputs)
def patch_sce():
    """Point the UNCHANGED shared_covariance_e3 sketch / selection code at the WT pool and this experiment's dirs."""
    SCE = sce()
    SCE.OUT = os.path.join(OUT, "wt_sc")
    SCE.STORE = st("sketch")
    SCE.OCACHE = WCACHE
    SCE.ROUTING_NPZ = jload(p("wt_routing.json"))["file"]
    SCE.protocol = protocol
    return SCE


def stage_sketch(a):
    protocol()
    check_wcache(); routing()
    SCE = patch_sce()
    SCE.stage_sketch(a)
    for L in LAYERS:
        assert jload(SCE.p("profiling", f"sketch_L{L}.json"))["passed"]


# ===================================================================== selections
def stage_select(a):
    out = p("selection_manifest.json")
    if os.path.exists(out):
        print("[select] done"); return
    protocol()
    c4 = {c: c4_selection(c) for c in STRATS}                # before patch_sce (reads the C4 SC study's dirs)
    SCE = patch_sce()
    from moe_calib import shared_covariance_selection as SCS
    assert all(jload(SCE.p("profiling", f"sketch_L{L}.json"))["passed"] for L in LAYERS)
    z = routing()
    C = z["C"]
    ids = list(range(POOL_N))
    typ, hal, scores = CD.typicality_halves(C, ids, 4, E)
    W = CD.warm_start(typ, hal, SEL_SEED)
    warm = W["ids"]
    fr = CD.fr_selection(ids, warm, B, SEL_SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    M, n, w = SCE.load_stacked(dev)
    S = SCE.SETTINGS
    sc, info = SCS.select_shared(SCS.StackedObjective(M, n, w), warm, B, SEL_SEED, S["greedy_sample_frac"],
                                 S["swap_max_passes"], S["swap_rel_tol"])
    t_sel = time.time() - t0
    sc2, info2 = SCS.select_shared(SCS.StackedObjective(M, n, w), warm, B, SEL_SEED, S["greedy_sample_frac"],
                                   S["swap_max_passes"], S["swap_rel_tol"])
    for ch in (fr, sc):
        SCS.validate_shared(ch, warm, B, POOL_N, ())
        assert all(CD.validate_selection(ch, warm, B, ids).values())
    obj = SCS.StackedObjective(M, n, w)
    sets = dict(WARM=warm, FR=fr, SC=sc)
    J = {}
    for c, ch in sets.items():
        d = obj.per_term(ch).cpu().numpy().reshape(4, E, 2)
        J[c] = dict(objective=float(d.mean()), gu=float(d[..., 0].mean()), down=float(d[..., 1].mean()),
                    per_layer={str(L): float(d[i].mean()) for i, L in enumerate(LAYERS)})
    ref = CD.TY.pool_reference(C, 4, E)
    rows = {}
    for c in ("FR", "SC"):
        X = C[sets[c]].reshape(B, 4, E)
        tot = X.sum(0)
        agg = CD.TY.pool_reference(C[sets[c]], 4, E)
        rows[c] = dict(tokens_per_expert=dict(median=float(np.median(tot)), p05=float(np.quantile(tot, .05)),
                                              min=float(tot.min()), mean=float(tot.mean())),
                       gini_by_layer={str(L): SCS.gini(tot[i]) for i, L in enumerate(LAYERS)},
                       set_routing_l1_by_layer={str(L): float(np.abs(agg[i] - ref[i]).sum()) for i, L in enumerate(LAYERS)},
                       set_js=CD.TY.selected_set_js(C[sets[c]], ref, 4, E),
                       mean_seq_atypicality_d=float(np.nanmean(scores["d"][sets[c]])),
                       post_warm_d=float(np.nanmean(scores["d"][sets[c][32:]])),
                       distinct_sources_per_expert_mean=float((X > 0).sum(0).mean()))
    dm = jload(p("dataset_manifest.json"))
    res = dict(
        C4=dict(FR=dict(chosen=c4["FR"]["chosen"], ids_hash=c4["FR"]["ids_hash"], source="runs_partial_ppl (validated)"),
                SC=dict(chosen=c4["SC"]["chosen"], ids_hash=c4["SC"]["ids_hash"], source="runs_shared_covariance_e3/selections/seed0.json"),
                warm=c4["FR"]["chosen"][:32], warm_shared=c4["FR"]["chosen"][:32] == c4["SC"]["chosen"][:32]),
        WT=dict(warm=dict(ids=warm, hash=ids_hash(warm), eligible_size=len(W["eligible"]), eligible_hash=ids_hash(W["eligible"])),
                FR=dict(chosen=fr, ids_hash=ids_hash(fr)), SC=dict(chosen=sc, ids_hash=ids_hash(sc)),
                sc_optimizer={k: v for k, v in info.items() if k != "greedy_order"},
                sc_deterministic_rerun_identical=bool(sc == sc2 and info["trace"] == info2["trace"]),
                sc_objective_monotone=bool(all(y <= x for x, y in zip(info["trace"], info["trace"][1:]))),
                sketch_objective=J, routing=rows, overlap_FR_SC=SCS.overlap(fr, sc),
                pool=dict(token_sha256=dm["pool_token_sha256"], routing_sha256=jload(p("wt_routing.json"))["sha256"],
                          reference="WT pool only (rows 0-1999 of the WT cache); no test / validation activations"),
                cost=dict(select_s=t_sel, peak_gpu_gb=gpu_peak(), device=dev)),
        note="C4 and WT ids index different pools (different texts): they are not paired across sources",
        created=time.strftime("%F %T"))
    assert res["WT"]["sc_deterministic_rerun_identical"] and res["WT"]["sc_objective_monotone"] and res["C4"]["warm_shared"]
    jdump(res, out)
    jdump(dict(experiment_code_sha256=hashes(EXP_CODE), frozen_code_sha256=hashes(FROZEN_CODE),
               selection_manifest_sha256=sha256_file(out),
               selection_hashes={f"{s}-{c}": res[s][c]["ids_hash"] for s in SOURCES for c in STRATS},
               note="frozen after WT selection, BEFORE any WT quantization / evaluation", created=time.strftime("%F %T")),
          p("freeze.json"))
    print(f"[select] WT J warm {J['WARM']['objective']:.4f} FR {J['FR']['objective']:.4f} SC {J['SC']['objective']:.4f}; "
          f"swaps {info['accepted']}; overlap FR/SC {res['WT']['overlap_FR_SC']}; {t_sel:.0f}s")


def check_frozen():
    pr = protocol()
    fz = jload(p("freeze.json"))
    assert hashes(EXP_CODE) == fz["experiment_code_sha256"], "experiment code changed after freeze"
    assert sha256_file(p("selection_manifest.json")) == fz["selection_manifest_sha256"]
    return pr, jload(p("selection_manifest.json"))


# ===================================================================== WT groupings
def gname(g):
    return f"G{int(g)}"


def stage_gmanifest(a):
    from moe_calib import multigroup_robustness as MG
    _, sel = check_frozen()
    path = p("grouping", "protocol.json")
    union = sorted(set(sel["WT"]["FR"]["chosen"]) | set(sel["WT"]["SC"]["chosen"]))
    gr = {}
    for g in GSEEDS:
        bt = MG.grouping_batches(union, g)
        v = MG.validate_grouping(bt, union)
        assert all(v.values()), v
        m = MG.manifest(g, bt, union, {c: sel["WT"][c]["chosen"] for c in STRATS})
        m.update(source="WT", construction="sorted(WT-FR u WT-SC) -> numpy.random.default_rng(g).permutation -> batches of 8")
        m["manifest_sha256"] = MG.obj_sha(MG.strip_sha(m))
        m["validation"] = v
        jdump(m, p("grouping", "manifests", f"WT_{gname(g)}.json"))
        gr[gname(g)] = dict(batches=bt, manifest_sha256=m["manifest_sha256"])
    body = dict(union=union, n_union=len(union), groupings=gr, grouping_seeds=GSEEDS,
                selection_hashes={c: sel["WT"][c]["ids_hash"] for c in STRATS})
    if os.path.exists(path):
        assert jload(path)["body_sha256"] == obj_sha(body), "WT grouping protocol changed"
        print("[gmanifest] already frozen"); return
    jdump(dict(body, body_sha256=obj_sha(body), created=time.strftime("%F %T")), path)
    print(f"[gmanifest] WT union {len(union)}")


def gprotocol():
    pr = jload(p("grouping", "protocol.json"))
    assert obj_sha({k: v for k, v in pr.items() if k not in ("body_sha256", "created")}) == pr["body_sha256"]
    return pr


def stage_gcache(a):
    import batch_sensitivity as DB
    check_frozen()
    gp = gprotocol()
    union = gp["union"]
    manp = p("grouping", "cache_manifest.json")
    man = jload(manp) if os.path.exists(manp) else dict(entries={})
    todo = [g for g in GSEEDS if not man["entries"].get(gname(g), {}).get("complete")]
    if not todo:
        print("[gcache] done"); return
    alltok = wt_tokens()
    model, t_load = DB.load_model()
    tsha_ref = DB.tensor_sha(alltok[union])
    for g in todo:
        bt = gp["groupings"][gname(g)]["batches"]
        d = st("grouping", gname(g))
        os.makedirs(d, exist_ok=True)
        files, keep = {}, {}

        def save(L, X):
            if L not in LAYERS:
                return
            f = os.path.join(d, f"L{L}.pt")
            torch.save(X.clone(), f + ".tmp"); os.replace(f + ".tmp", f)
            keep[L] = X.clone()
            files[str(L)] = dict(shape=list(X.shape), sha256=sha256_file(f), bytes=os.path.getsize(f),
                                 finite=bool(torch.isfinite(X.float()).all()))
        t0 = time.time()
        tsha = DB.forward_plan(model, alltok, bt, union, save)
        t_b = time.time() - t0
        neg = {}

        def cmp(L, X):
            if L in LAYERS:
                neg[str(L)] = bool(torch.equal(X, keep[L]))
        tsha_r = DB.forward_plan(model, alltok, bt[::-1], union, cmp)
        ent = dict(cache_dir=d, n_rows=len(union), files=files, generation_s=t_b, reversed_order_bitwise=neg,
                   token_ids_identical=tsha == tsha_ref == tsha_r, manifest_sha256=gp["groupings"][gname(g)]["manifest_sha256"])
        ent["complete"] = bool(len(neg) == 4 and all(neg.values()) and ent["token_ids_identical"]
                               and all(f["finite"] for f in files.values()))
        man["entries"][gname(g)] = ent
        man["model_load_s"] = t_load
        jdump(man, manp)
        print(f"[gcache] WT {gname(g)}: {t_b:.0f}s reversed bitwise {all(neg.values())}", flush=True)
        if not ent["complete"]:
            raise SystemExit("grouping cache control failed")
        del keep


# ===================================================================== quantize + evaluate
def load_captures():
    man = jload(CAPTURE_MANIFEST)
    assert man["ok"]
    caps = {}
    for L in LAYERS:
        f = man["files"][f"L{L}"]
        assert sha256_file(f["path"]) == f["sha256"], f"capture L{L} changed"
        caps[L] = torch.load(f["path"])
    return caps, sha256_file(os.path.join(ROOT, CAPTURE_MANIFEST))


@torch.no_grad()
def b2_eval(model, snap, caps):
    """Loss-alignment MVP B fixed-input metrics (FP32 mode) of the currently installed weights; raw sums."""
    from moe_calib import model_utils as MU
    from moe_calib import loss_alignment as LA
    import loss_alignment_mvp_b as LB
    lay = {}
    for L in LAYERS:
        c = caps[L]
        Wr = [{k: v.to(MU.GPU, torch.float32) for k, v in snap[(L, e)].items()} for e in range(E)]
        Wq = [{k: v.detach().to(MU.GPU, torch.float32) for k, v in MU.get_expert_weights(model, L, e).items()} for e in range(E)]
        r = LA.layer_output_errors(c["X"].to(MU.GPU, torch.float32), c["idx"].to(MU.GPU).long(), c["wa"].to(MU.GPU).float(),
                                   Wr, Wq, seq_id=c["seq_id"].to(MU.GPU), n_seq=LB.N_SEL, return_M=True, mlp=LA.mlp_fp32)
        routed = c["routed"].to(MU.GPU).float()
        r["M_vs_module_routed_rel"] = float((r.pop("M") - routed).norm() / routed.norm())
        lay[str(L)] = LB.jsonable(r)
        del Wr, Wq
        torch.cuda.empty_cache()
    return {"layers": {"fp32": lay}}


def config_spec(cfg, sel):
    src, c, ext = cfg.split(":")
    chosen = sel[src][c]["chosen"]
    if ext == "ORIG":
        cache = OCACHE if src == "C4" else WCACHE
        return dict(cache=cache, chosen=chosen, eval_cache_dir=None if src == "C4" else OCACHE, files=None)
    g = int(ext[1:])
    if src == "C4":
        cg = c4_grouping(g)
        d, union, files = cg["dir"], cg["union"], cg["files"]
    else:
        ent = jload(p("grouping", "cache_manifest.json"))["entries"][gname(g)]
        assert ent["complete"]
        d, union, files = ent["cache_dir"], gprotocol()["union"], ent["files"]
    from moe_calib import multigroup_robustness as MG
    rc = MG.remap(chosen, union)
    assert [union[i] for i in rc] == chosen
    return dict(cache=d, chosen=rc, eval_cache_dir=OCACHE, files=files)


def stored_reference(cfg):
    src, c, ext = cfg.split(":")
    if src != "C4":
        return None
    if ext == "ORIG":
        return jload(f"runs_shared_covariance_e3/results/A_{c}_s0.json")
    return jload(f"runs_shared_covariance_e3/grouping_robustness/results/{c}_s0_{ext}.json")


def stage_quant(a):
    import eval_partial_ppl as EP
    from moe_calib import partial_model_quant as PQ
    import loss_alignment_mvp_b as LB
    pr, sel = check_frozen()
    SCE = sce()
    todo = [c for c in a.configs if not os.path.exists(p("results", f"{c.replace(':', '_')}.json"))]
    if not todo and not a.bf16:
        print("[quant] done"); return
    win, eprot = EP.load_windows()
    caps, capsha = load_captures()
    model, t_load = SCE.load_full()
    snap = PQ.snapshot_experts(model, LAYERS)
    untouched0 = PQ.untouched_digests(model, LAYERS)
    test_ids = list(range(2128, 2256))
    if a.bf16 and not os.path.exists(p("results", "BF16.json")):
        t0 = time.time()
        nll, cnt = EP.layerwise_nll(model, win)
        rec = EP.ppl_record(nll, cnt, time.time() - t0)
        ref = jload("runs_partial_ppl/bf16_baseline.json")
        b2 = b2_eval(model, snap, caps)
        tot, _, _, _ = LB.metrics_from_raw(b2)
        rec.update(reproduces_stored_bf16_ppl=rec["ppl"] == ref["ppl"], stored_ppl=ref["ppl"], b2_zero=tot["D_sum"] == 0.0,
                   installed_weights_sha256=PQ.installed_digest(model, LAYERS))
        jdump(rec, p("results", "BF16.json"))
        print(f"[quant] BF16 PPL {rec['ppl']:.6f} reproduces stored {rec['reproduces_stored_bf16_ppl']}; B2 zero {rec['b2_zero']}")
        assert rec["reproduces_stored_bf16_ppl"] and rec["b2_zero"]
    for cfg in todo:
        src, c, ext = cfg.split(":")
        sp = config_spec(cfg, sel)
        if sp["files"]:
            for L in LAYERS:
                assert sha256_file(os.path.join(sp["cache"], f"L{L}.pt")) == sp["files"][str(L)]["sha256"], "cache changed"
        holder, orig_restore = {}, PQ.restore_experts

        def restore_with_b2(m, s):                          # measured on the installed weights, before restore
            holder["raw"] = b2_eval(m, s, caps)
            return orig_restore(m, s)
        PQ.restore_experts = restore_with_b2
        t0 = time.time()
        try:
            r = SCE.install_and_ppl(model, snap, untouched0, sp["cache"], sp["chosen"], K, test_ids, win,
                                    eval_cache_dir=sp["eval_cache_dir"])
        finally:
            PQ.restore_experts = orig_restore
        tot, per_layer, _, _ = LB.metrics_from_raw(holder["raw"])
        r.update(config=cfg, source=src, strategy=c, extraction=ext, ids_hash=sel[src][c]["ids_hash"],
                 calibration_cache=sp["cache"], windows_sha256=eprot["windows_sha256"], capture_manifest_sha256=capsha,
                 b2=tot, b2_per_layer={str(L): v for L, v in per_layer.items()}, b2_raw=holder["raw"],
                 wall_s=time.time() - t0, model_load_s=t_load, peak_cpu_gb=rss_gb(), env=env())
        ref = stored_reference(cfg)
        if ref is not None:
            rp = {(x["layer"], x["expert"]): x["mean"] for x in r["per_expert_test"]}
            rr = {(x["layer"], x["expert"]): x["mean"] for x in ref["per_expert_test"]}
            r["reproduction"] = dict(installed_weights_sha256_equal=r["installed_weights_sha256"] == ref["installed_weights_sha256"],
                                     mean_nll_equal=r["mean_nll"] == ref["mean_nll"],
                                     per_window_nll_equal=r["per_window_nll"] == ref["per_window_nll"],
                                     per_expert_test_bitwise=rp == rr)
            if cfg == "C4:FR:ORIG":                       # B2 must equal the loss-alignment MVP B measurement
                mb = jload("runs_loss_alignment/mvp_b/raw_sums.json")["runs"]["FR"]["layers"]["fp32"]
                r["reproduction"]["mvp_b_D_M_U_bitwise"] = all(
                    mb[str(L)][k] == holder["raw"]["layers"]["fp32"][str(L)][k] for L in LAYERS for k in ("D_sum", "M_sum", "U_sum"))
            r["reproduction"]["all"] = all(r["reproduction"].values())
        jdump(r, p("results", f"{cfg.replace(':', '_')}.json"))
        print(f"[quant] {cfg}: NLL {r['mean_nll']:.6f} PPL {r['ppl']:.5f} C4-recon {r['recon_test']:.7f} "
              f"B2 {tot['B2']:.5e} P_seq {tot['P_seq_mean']:.5e} ok {r['install_ok']}"
              + (f" reproduced {r['reproduction']}" if ref is not None else "") + f" ({r['wall_s']:.0f}s)", flush=True)
        if not r["install_ok"]:
            raise SystemExit(f"{cfg}: installation checks failed")
        if ref is not None and not r["reproduction"]["all"]:
            print(f"[quant] WARNING {cfg} does not reproduce the stored result; reported, stored value NOT reused", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["audit", "pool", "protocol", "cache", "routing", "sketch", "select",
                                                       "gmanifest", "gcache", "quant", "report"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--configs", nargs="+", default=[])
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args(argv)
    if a.stage == "report":
        import report_calibration_domain_ablation as R
        R.main(); return
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
