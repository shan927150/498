"""Held-out evaluation validation (C4 validation) and dense 3-bit GPTQ execution sensitivity.

Experiment 1: re-evaluate EXISTING 4-bit weight sets (MoE Family D 30 configs + seed-0 ORIGINAL references; dense
  grouping 5 + ORIGINAL; dense dose-response batch-1 k in {1,4,16,64} x 8 replicates x {natural, fixed}; optional
  5 x 5 reference-order matrix) on a newly frozen, never-used C4-validation evaluation set.
Experiment 2: dense 3-bit GPTQ from the SAVED historical grouping Hessians (runs_dense_grouping_scope), unchanged
  solver (moe_calib.gptq via moe_calib.dense_gptq.quantize_layer) with bits=3; evaluated on WikiText-2 + held-out.

Nothing historical is modified: historical weights / Hessians are only read (sha256-verified). New artifacts go to
runs_heldout_bitwidth_validation/ and ~/moe_cache_heldout_bitwidth/.

Stages (python diagnostics/heldout_bitwidth_validation.py --stage S):
  select    freeze the C4-validation evaluation windows (content-only rules; no model) + overlap audit
  audit     historical artifact inventory (hashes, shapes, finiteness) + evaluation-usage audit
  protocol  freeze protocol.json (hash) before any new PPL result
  dense     dense GPU stage: WikiText parity, BF16 + dense4 + dose + refmatrix on held-out (resumable)
  dense3    dense 3-bit solves from saved Hessians + correctness + both datasets (resumable)
  moe       MoE GPU stage: WikiText parity subset, 30 Family D configs + s0 ORIGINAL refs on held-out (resumable)
  analyze   all analysis JSONs, figures, summary inputs (report_heldout_bitwidth_validation.py)
"""
from __future__ import annotations
import argparse, gc, glob, gzip, hashlib, io, json, math, os, random, resource, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))

OUT = "runs_heldout_bitwidth_validation"
CACHE = os.path.expanduser(os.environ.get("HBV_CACHE", "~/moe_cache_heldout_bitwidth"))
DCACHE = os.path.expanduser("~/moe_cache_dense_grouping_scope")
DOSE_CACHE = os.path.expanduser("~/moe_cache_dense_dose_response")
C4_REPO, C4_REV = "allenai/c4", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"   # same revision as the calibration shard
C4_VAL_FILE = "en/c4-validation.00000-of-00008.json.gz"
C4_TRAIN_FILE = "en/c4-train.00000-of-01024.json.gz"
SEL_SEED = 20261117
SEQLEN, N_WIN, BATCH = 512, 128, 8
NGRAM = 13
GROUPS = ["G20261111", "G20261112", "G20261113", "G20261114", "G20261115"]
DENSE_CONFIGS = GROUPS + ["ORIGINAL"]
DENSE_LAYERS = [1, 9, 19, 27]
PROJ = ("gate_proj", "up_proj", "down_proj")
MOE_SEEDS = (0, 1, 2)
MOE_STRATS = ("FR", "FC")
MOE_CONFIGS = [f"{st}_s{s}_{g}" for s in MOE_SEEDS for g in GROUPS for st in MOE_STRATS]
MOE_REFS = ["FR_ORIGINAL", "FC_ORIGINAL"]                          # seed-0 ORIGINAL extraction (Family A weights)
MOE_PARITY = ["FR_s0_G20261111", "FC_s2_G20261111", "FR_ORIGINAL"]  # prespecified WikiText parity subset
DENSE_PARITY = DENSE_CONFIGS                                       # all 6 dense configs (cheap)
DOSE_K = (1, 4, 16, 64)
DOSE_REPS = 8
DOSE_PARITY = ["b1_k001_r0", "b1_k064_r7"]                         # prespecified WikiText parity subset (both policies)
GPTQ4 = dict(bits=4, group_size=128, percdamp=1.0, act_order=True)
GPTQ3 = dict(bits=3, group_size=128, percdamp=1.0, act_order=True)
REPEAT_SOLVE_CONFIG = "G20261111"                                  # prespecified deterministic re-solve check


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


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def obj_sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), default=float).encode()).hexdigest()


def tensor_sha(t):
    t = t.detach().contiguous().cpu()
    return hashlib.sha256((t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()).hexdigest()


def tensor_fp_int64(t):
    """eval_partial_ppl.tensor_fp convention (int64 bytes)."""
    return hashlib.sha256(t.contiguous().numpy().astype(np.int64).tobytes()).hexdigest()


def peak_cpu_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def code_files():
    return ["diagnostics/heldout_bitwidth_validation.py", "moe_calib/gptq.py", "moe_calib/dense_gptq.py",
            "moe_calib/actorder_policy.py", "moe_calib/model_utils.py", "moe_calib/partial_model_quant.py",
            "diagnostics/eval_partial_ppl.py", "diagnostics/dense_grouping_scope.py",
            "diagnostics/inference_routing_amplification.py"]


def code_hashes():
    return {f: sha256_file(os.path.join(ROOT, f)) for f in code_files()}


def log_code_state(stage):
    path = p("logs", "code_state.json")
    d = jload(path) or {}
    d[f"{stage}@{time.strftime('%F %T')}"] = code_hashes()
    jdump(d, path)


def cost_add(stage, key, val):
    path = p("logs", "cost_raw.json")
    d = jload(path) or {}
    d.setdefault(stage, {})[key] = val
    jdump(d, path)


# ====================================================================== data helpers
def c4_local(fname):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(C4_REPO, fname, repo_type="dataset", revision=C4_REV, local_files_only=True)


def read_c4(path):
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def moe_tokenizer():
    from transformers import AutoTokenizer
    from moe_calib import model_utils as MU
    path = MU._snapshot_dir("Qwen/Qwen1.5-MoE-A2.7B")
    return AutoTokenizer.from_pretrained(path), path


def dense_tokenizer():
    from transformers import AutoTokenizer
    import dense_grouping_scope as DS
    path = DS.snapshot_dir()
    return AutoTokenizer.from_pretrained(path), path


_MULT = np.uint64(0x9E3779B97F4A7C15)


def ngram_hashes(seq, n=NGRAM):
    """64-bit polynomial hashes of all n-grams of a 1-D int sequence (wrapping uint64 arithmetic)."""
    a = np.asarray(seq, dtype=np.uint64)
    if a.shape[0] < n:
        return np.zeros(0, dtype=np.uint64)
    pw = np.array([pow(int(_MULT), n - 1 - i, 1 << 64) for i in range(n)], dtype=np.uint64)
    win = np.lib.stride_tricks.sliding_window_view(a, n)
    with np.errstate(over="ignore"):
        return (win * pw).sum(axis=1, dtype=np.uint64)


def token_sources():
    """Every historical token tensor that was used for calibration, tuning, selection or evaluation (best effort:
    all ids.pt files of the experiment caches + the WikiText-2 evaluation windows + the EBSS official tokens)."""
    home = os.path.expanduser("~")
    files = sorted(glob.glob(os.path.join(home, "moe_cache*", "ids.pt")) +
                   glob.glob(os.path.join(home, "moe_cache*", "*", "ids.pt")))
    files += [os.path.join(ROOT, "runs_partial_ppl/windows.pt"),
              os.path.join(ROOT, "runs_ebss_official_validation/official_data/tokens_all.pt"),
              os.path.join(ROOT, "runs_ebss_official_validation/official_data/ebss_windows.pt")]
    files += sorted(glob.glob(os.path.join(ROOT, "runs_ebss_generation_pilot/logs/generated_*.pt")) +
                    glob.glob(os.path.join(ROOT, "runs_ebss_balance_fidelity/logs/generated_*.pt")))
    return [f for f in files if os.path.exists(f)]


def _int_tensors(o, out, prefix=""):
    if torch.is_tensor(o):
        if not o.is_floating_point() and o.dtype != torch.bool:
            out.append((prefix, o))
    elif isinstance(o, dict):
        for k, v in o.items():
            _int_tensors(v, out, f"{prefix}.{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            _int_tensors(v, out, f"{prefix}[{i}]")


def load_source_rows(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    ts = []
    _int_tensors(o, ts)
    rows = []
    for name, t in ts:
        if t.dim() == 1:
            rows.append(t.long().numpy())
        elif t.dim() == 2:
            rows.extend(list(t.long().numpy()))
    return rows


# ====================================================================== select (no model)
def stage_select(a):
    if os.path.exists(p("evaluation_manifest.json")):
        man = jload(p("evaluation_manifest.json"))
        win = torch.load(c("heldout_windows.pt"))
        assert tensor_fp_int64(win) == man["windows_sha256"], "frozen windows changed"
        print("evaluation set already frozen; identical windows"); return
    import datasets, transformers
    t0 = time.time()
    tok, tok_path = moe_tokenizer()
    vpath, tpath = c4_local(C4_VAL_FILE), c4_local(C4_TRAIN_FILE)
    val = read_c4(vpath)
    print(f"[select] C4 validation shard: {len(val)} docs", flush=True)
    # ---- calibration / historical token sources -> 13-gram hash set
    srcs = token_sources()
    hs, src_info = [], {}
    full_windows = set()
    for f in srcs:
        rows = load_source_rows(f)
        n_tok = 0
        for r in rows:
            hs.append(ngram_hashes(r)); n_tok += len(r)
            if len(r) == SEQLEN:
                full_windows.add(r.astype(np.int64).tobytes())
        src_info[os.path.relpath(f, os.path.expanduser("~"))] = dict(sha256=sha256_file(f), n_rows=len(rows), n_tokens=n_tok)
        print(f"[select] source {f}: {len(rows)} rows", flush=True)
    hist_hash = np.unique(np.concatenate(hs)) if hs else np.zeros(0, np.uint64)
    del hs
    # ---- C4 train shard (calibration source) text / URL index
    train = read_c4(tpath)
    train_text = {sha256_bytes(d["text"].encode()) for d in train}
    train_url = {d["url"] for d in train}
    n_train = len(train)
    del train
    gc.collect()
    print(f"[select] C4 train shard 00000: {n_train} docs indexed; history n-grams {hist_hash.size}", flush=True)
    # ---- selection: one random 512-token window per document (data_utils.get_splits convention), seeded order
    rng = random.Random(SEL_SEED)
    order = list(range(len(val)))
    rng.shuffle(order)
    chosen, excl = [], dict(too_short=0, text_in_c4_train=0, url_in_c4_train=0, ngram_overlap_history=0,
                            exact_window_in_history=0, ngram_overlap_selected=0)
    sel_hash = np.zeros(0, np.uint64)
    examined = 0
    for idx in order:
        if len(chosen) >= N_WIN:
            break
        examined += 1
        d = val[idx]
        ids = tok(d["text"], return_tensors="pt").input_ids[0]
        if ids.shape[0] <= SEQLEN:
            excl["too_short"] += 1
            continue
        st = rng.randint(0, ids.shape[0] - SEQLEN - 1)
        w = ids[st:st + SEQLEN]
        reason = None
        th = sha256_bytes(d["text"].encode())
        wh = ngram_hashes(w.numpy())
        if th in train_text:
            reason = "text_in_c4_train"
        elif d["url"] in train_url:
            reason = "url_in_c4_train"
        elif w.numpy().astype(np.int64).tobytes() in full_windows:
            reason = "exact_window_in_history"
        elif np.isin(wh, hist_hash).any():
            reason = "ngram_overlap_history"
        elif sel_hash.size and np.isin(wh, sel_hash).any():
            reason = "ngram_overlap_selected"
        if reason:
            excl[reason] += 1
            continue
        chosen.append(dict(row=idx, url=d["url"], timestamp=d.get("timestamp"), text_sha256=th,
                           n_doc_tokens=int(ids.shape[0]), start=st, window=w.clone()))
        sel_hash = np.union1d(sel_hash, wh)
    assert len(chosen) == N_WIN, f"only {len(chosen)} windows"
    chosen.sort(key=lambda r: r["row"])                     # fixed ordering: ascending validation-shard row
    win = torch.stack([r["window"] for r in chosen]).long()
    # ---- dense tokenizer parity on the selected documents
    dtok, dtok_path = dense_tokenizer()
    dense_equal = []
    for r in chosen:
        ids_d = dtok(val[r["row"]]["text"], return_tensors="pt").input_ids[0]
        dense_equal.append(bool(torch.equal(ids_d[r["start"]:r["start"] + SEQLEN], r["window"])))
    os.makedirs(CACHE, exist_ok=True)
    torch.save(win, c("heldout_windows.pt"))
    torch.save(win, p("heldout_windows.pt"))
    wsha = tensor_fp_int64(win)
    per_window = [dict(window=i, c4_validation_row=r["row"], url=r["url"], c4_timestamp=r["timestamp"],
                       text_sha256=r["text_sha256"], n_doc_tokens=r["n_doc_tokens"], token_start=r["start"],
                       window_token_sha256=tensor_fp_int64(r["window"]),
                       dense_tokenizer_identical=dense_equal[i]) for i, r in enumerate(chosen)]
    man = dict(
        dataset=dict(source="C4 (allenai/c4) English, VALIDATION split, shard en/c4-validation.00000-of-00008.json.gz",
                     hf_repo=C4_REPO, revision=C4_REV, file=C4_VAL_FILE, file_sha256=sha256_file(vpath),
                     n_documents_in_shard=len(val), datasets_version=datasets.__version__,
                     split_provenance="official C4 validation split file (distinct file family from c4-train.*); "
                                      "historical calibration used ONLY c4-train.00000-of-01024 (data_utils.get_splits, "
                                      "build_confirmation_split) at the same repo revision",
                     downloaded=time.strftime("%F", time.localtime(os.path.getmtime(os.path.dirname(vpath))))),
        selection=dict(seed=SEL_SEED, rule="random.Random(seed): shuffle validation-shard row indices; for each document "
                                           "in that order tokenize the full text (no special tokens added); skip if <= 512 "
                                           "tokens; draw start = rng.randint(0, n - 513) (data_utils.get_splits convention); "
                                           "window = ids[start:start+512]; content-only exclusion rules (below) applied "
                                           "BEFORE any model evaluation; stop at 128; order windows by ascending shard row",
                       exclusion_rules=dict(text_in_c4_train="document text sha256 equals a C4 train-shard-00000 document",
                                            url_in_c4_train="document URL appears in C4 train shard 00000",
                                            exact_window_in_history="512-token window equals any historical token row",
                                            ngram_overlap_history=f"window shares any {NGRAM}-token n-gram with any "
                                                                  "historical calibration/tuning/evaluation token row",
                                            ngram_overlap_selected=f"window shares any {NGRAM}-gram with an already "
                                                                   "selected window (near-duplicate guard)"),
                       documents_examined=examined, exclusions=excl, no_model_performance_inspected=True),
        tokenizer=dict(name="Qwen/Qwen1.5-MoE-A2.7B", snapshot=os.path.basename(tok_path), cls=type(tok).__name__,
                       transformers_version=transformers.__version__,
                       adds_special_tokens=tok("a").input_ids != tok("a", add_special_tokens=False).input_ids,
                       dense_tokenizer=dict(name="Qwen/Qwen2.5-1.5B", snapshot=os.path.basename(dtok_path),
                                            all_windows_identical=all(dense_equal), n_identical=sum(dense_equal)),
                       max_token_id=int(win.max())),
        sequence_construction=dict(context_length=SEQLEN, n_windows=N_WIN, one_window_per_document=True,
                                   scored_targets_per_window=SEQLEN - 1, scored_targets_total=N_WIN * (SEQLEN - 1),
                                   scoring="identical to historical: logits[:, :-1] vs input_ids[:, 1:]; FP32 CE, FP64 "
                                           "accumulation; PPL = exp(total NLL / 65408); batch 8; fixed window order"),
        windows_file=os.path.relpath(c("heldout_windows.pt"), os.path.expanduser("~")),
        windows_copy=os.path.join(OUT, "heldout_windows.pt"),
        windows_sha256=wsha, windows_file_sha256=sha256_file(c("heldout_windows.pt")),
        selected_document_rows=[r["row"] for r in chosen], per_window=per_window,
        selection_timestamp=time.strftime("%F %T"), selection_seconds=time.time() - t0)
    jdump(man, p("evaluation_manifest.json"))
    # ---- overlap audit (post-selection verification, independent recomputation)
    wt = torch.load(os.path.join(ROOT, "runs_partial_ppl/windows.pt"))
    win_np = win.numpy()
    all_h = np.concatenate([ngram_hashes(r) for r in win_np])
    within = []
    for i in range(N_WIN):
        hi = ngram_hashes(win_np[i])
        for j in range(i + 1, N_WIN):
            if np.isin(hi, ngram_hashes(win_np[j])).any():
                within.append([i, j])
    ov = dict(
        historical_token_sources=src_info, n_historical_ngrams=int(hist_hash.size), ngram=NGRAM,
        selected_windows_sharing_any_ngram_with_history=int(sum(np.isin(ngram_hashes(r), hist_hash).any() for r in win_np)),
        selected_windows_exactly_in_history=int(sum(r.astype(np.int64).tobytes() in full_windows for r in win_np)),
        selected_docs_text_in_c4_train_shard=int(sum(r["text_sha256"] in train_text for r in chosen)),
        selected_docs_url_in_c4_train_shard=int(sum(r["url"] in train_url for r in chosen)),
        c4_train_shard=dict(file=C4_TRAIN_FILE, file_sha256=sha256_file(tpath), n_documents=n_train),
        wikitext2_eval_windows=dict(sha256=tensor_fp_int64(wt), shared_ngrams_with_heldout=int(np.isin(all_h, np.unique(
            np.concatenate([ngram_hashes(r) for r in wt.numpy()]))).sum())),
        within_heldout_near_duplicate_pairs=within, n_unique_documents=len({r["row"] for r in chosen}),
        n_unique_urls=len({r["url"] for r in chosen}), n_unique_text_sha=len({r["text_sha256"] for r in chosen}),
        usage_audit=dict(
            code_search="grep of all *.py: load_dataset calls are C4 data_files={'train': c4-train.00000-of-01024} "
                        "(moe_calib/data_utils.py, diagnostics/build_confirmation_split.py) and wikitext-2-raw-v1 "
                        "(test: evaluation; train/validation: calibration_domain_ablation); no C4 validation access",
            hf_cache="before this study the local HF cache held only en/c4-train.00000-of-01024.json.gz for allenai/c4; "
                     "c4-validation.00000-of-00008 was first downloaded by this study (symlink mtime)",
            reports="no update_*.md / runs_*/ summary reports a C4-validation evaluation or selection"),
        passed=True)
    ov["passed"] = (ov["selected_windows_sharing_any_ngram_with_history"] == 0 and ov["selected_windows_exactly_in_history"] == 0
                    and ov["selected_docs_text_in_c4_train_shard"] == 0 and ov["selected_docs_url_in_c4_train_shard"] == 0
                    and not within and ov["n_unique_documents"] == N_WIN and all(dense_equal))
    jdump(ov, p("evaluation_overlap_audit.json"))
    print(f"[select] frozen {N_WIN} windows sha {wsha[:16]}; examined {examined}; excl {excl}; overlap passed {ov['passed']}")


def load_heldout():
    man = jload(p("evaluation_manifest.json"))
    win = torch.load(c("heldout_windows.pt"))
    assert tensor_fp_int64(win) == man["windows_sha256"] and tuple(win.shape) == (N_WIN, SEQLEN), "held-out windows changed"
    return win, man


# ====================================================================== audit (CPU, read-only)
def dose_names():
    return ["b1_k000"] + [f"b1_k{k:03d}_r{r}" for k in DOSE_K for r in range(DOSE_REPS)]


def moe_manifest():
    """Weight manifest for the 30 Family D configs (runs_grouping_routing_amplification) + the seed-0 ORIGINAL pair
    (runs_inference_routing_amplification), read-only copies of the validated historical manifests."""
    gra = jload(os.path.join(ROOT, "runs_grouping_routing_amplification/weight_manifest.json"))
    fa = jload(os.path.join(ROOT, "runs_inference_routing_amplification/weight_manifest.json"))
    from inference_routing_amplification import obj_sha as fa_obj_sha
    assert fa_obj_sha(gra["configs"]) == gra["manifest_sha256"] and fa_obj_sha(fa["configs"]) == fa["manifest_sha256"]
    cfg = {n: gra["configs"][n] for n in MOE_CONFIGS}
    for n in MOE_REFS:
        cfg[n] = fa["configs"][n]
    return dict(configs=cfg, sources=dict(gra_manifest_sha256=gra["manifest_sha256"], fa_manifest_sha256=fa["manifest_sha256"]))


def stage_audit(a):
    t0 = time.time()
    inv = dict(created=time.strftime("%F %T"))
    # ---- evaluation-set provenance (WikiText windows untouched)
    import eval_partial_ppl as EP
    win, prot = EP.load_windows()
    inv["wikitext2_windows"] = dict(file="runs_partial_ppl/windows.pt", sha256=prot["windows_sha256"],
                                    file_sha256=sha256_file(os.path.join(ROOT, "runs_partial_ppl/windows.pt")),
                                    n_windows=int(win.shape[0]), seqlen=int(win.shape[1]), unchanged=True)
    # ---- dense grouping weights / codes / Hessians
    dinv = jload(os.path.join(ROOT, "runs_dense_grouping_scope/artifact_inventory.json"))
    dense = {}
    for g in DENSE_CONFIGS:
        res = jload(os.path.join(ROOT, f"runs_dense_grouping_scope/results/{g}.json"))
        ent = dict(historical_result=f"runs_dense_grouping_scope/results/{g}.json", historical_valid=bool(res["valid"]),
                   historical_ppl=res["ppl"], installed_weights_sha256=res["installed_weights_sha256"],
                   inventory_installed_sha256=dinv["configs"][g]["installed_weights_sha256"], files={})
        ok = ent["installed_weights_sha256"] == ent["inventory_installed_sha256"] and ent["historical_valid"]
        h = hashlib.sha256()
        for L in DENSE_LAYERS:
            exp = dinv["configs"][g]["files"][str(L)]
            fw, fc, fh = (os.path.join(DCACHE, s, g, f"L{L}.pt") for s in ("weights", "codes", "hessians"))
            sw, sc, sh = sha256_file(fw), sha256_file(fc), sha256_file(fh)
            W = torch.load(fw); H = torch.load(fh); C = torch.load(fc)
            for k in PROJ:
                h.update(W[k].contiguous().view(torch.int16).numpy().tobytes())
            shp = {k: list(W[k].shape) for k in PROJ}
            fl = dict(weight_sha256_match=sw == exp["weight_sha256"], codes_sha256_match=sc == exp["codes_sha256"],
                      hessian_sha256_match=sh == exp["hessian_sha256"],
                      shapes_ok=shp == {"gate_proj": [8960, 1536], "up_proj": [8960, 1536], "down_proj": [1536, 8960]},
                      dtype_bf16=all(W[k].dtype == torch.bfloat16 for k in PROJ),
                      finite=all(bool(torch.isfinite(W[k].float()).all()) for k in PROJ),
                      hessian_shapes_ok=list(H["gu"].shape) == [1536, 1536] and list(H["down"].shape) == [8960, 8960],
                      hessian_finite=bool(torch.isfinite(H["gu"]).all() and torch.isfinite(H["down"]).all()),
                      hessian_dtype_fp32=H["gu"].dtype == torch.float32 and H["down"].dtype == torch.float32,
                      hessian_tokens_65536=H["tokens"]["gu"] == H["tokens"]["down"] == 65536,
                      codes_4bit_range=all(int(C[k]["q"].max()) <= 15 for k in PROJ))
            ent["files"][str(L)] = dict(weight_sha256=sw, codes_sha256=sc, hessian_sha256=sh, **fl)
            ok &= all(fl.values())
            del W, H, C
        ent["recomputed_installed_sha256"] = h.hexdigest()
        ent["installed_sha_recomputed_match"] = h.hexdigest() == ent["installed_weights_sha256"]
        ent["valid"] = bool(ok and ent["installed_sha_recomputed_match"])
        dense[g] = ent
        print(f"[audit] dense {g}: valid {ent['valid']}", flush=True)
    nullc = {}
    for L in DENSE_LAYERS:
        a_ = torch.load(os.path.join(DCACHE, "hessians", "G20261115", f"L{L}.pt"))
        b_ = torch.load(os.path.join(DCACHE, "hessians", "ORIGINAL", f"L{L}.pt"))
        nullc[str(L)] = bool(torch.equal(a_["gu"], b_["gu"]) and torch.equal(a_["down"], b_["down"]))
    inv["dense_grouping"] = dict(configs=dense, null_G20261115_equals_ORIGINAL_hessians=nullc,
                                 all_valid=all(v["valid"] for v in dense.values()) and all(nullc.values()),
                                 cache=DCACHE, historical_bf16=jload(os.path.join(ROOT, "runs_dense_grouping_scope/bf16_baseline.json"))["ppl"])
    # ---- dose-response weights
    dinv2 = jload(os.path.join(ROOT, "runs_dense_dose_response/artifact_inventory.json"))
    dose = {}
    for n in dose_names():
        res = jload(os.path.join(ROOT, f"runs_dense_dose_response/results/{n}.json"))
        e = dict(config=res["config"], valid=bool(res["valid"]), policies={})
        for pol in ("natural", "fixed"):
            files = [os.path.join(DOSE_CACHE, "cfg", n, f"Q_{pol}_L{L}.pt") for L in DENSE_LAYERS]
            e["policies"][pol] = dict(files_present=all(os.path.exists(f) for f in files),
                                      file_bytes={os.path.basename(f): os.path.getsize(f) for f in files if os.path.exists(f)},
                                      inventory_installed_sha256=dinv2["weight_hashes"][f"{n}|{pol}"],
                                      result_installed_sha256=res["policies"][pol]["installed_weights_sha256"],
                                      historical_ppl=res["policies"][pol]["ppl"])
            pe = e["policies"][pol]
            pe["sha_consistent"] = pe["inventory_installed_sha256"] == pe["result_installed_sha256"]
            e["valid"] &= pe["files_present"] and pe["sha_consistent"]
        dose[n] = e
    inv["dose_response"] = dict(configs=dose, all_valid=all(v["valid"] for v in dose.values()), cache=DOSE_CACHE,
                                note="per-tensor weight sha256 (results meta) and installed digest are re-verified at load "
                                     "time in the GPU stage")
    # ---- reference matrix
    rraw = jload(os.path.join(ROOT, "runs_dense_dose_response/logs/refmatrix_raw.json"))
    rm = {}
    for i in GROUPS:
        for j in GROUPS:
            f = os.path.join(DOSE_CACHE, "refmatrix", f"{i}__{j}.pt")
            rm[f"{i}|{j}"] = dict(file_present=os.path.exists(f), file_bytes=os.path.getsize(f) if os.path.exists(f) else None,
                                  historical_ppl=rraw[f"{i}|{j}"]["ppl"])
    inv["reference_matrix"] = dict(configs=rm, all_present=all(v["file_present"] for v in rm.values()),
                                   note="no stored weight hash: integrity = WikiText-2 per-window NLL bitwise equal to "
                                        "refmatrix_raw.json for all 25 + diagonal equals historical natural weights")
    # ---- MoE Family D
    wm = moe_manifest()
    moe = {}
    for n, ent in wm["configs"].items():
        hr = jload(os.path.join(ROOT, ent["historical_result"]))
        files_present = all(os.path.exists(os.path.join(ent["weights_dir"], f"L{L}.pt")) for L in (1, 8, 16, 23))
        moe[n] = dict(weights_dir=ent["weights_dir"], installed_weights_sha256=ent["installed_weights_sha256"],
                      historical_result=ent["historical_result"], historical_ppl=float(ent["historical_ppl"]),
                      historical_result_sha_match=hr.get("installed_weights_sha256") == ent["installed_weights_sha256"],
                      historical_result_valid=bool(hr.get("valid", True)), files_present=files_present,
                      file_bytes={L: os.path.getsize(os.path.join(ent["weights_dir"], f"L{L}.pt")) for L in (1, 8, 16, 23)}
                      if files_present else None)
        moe[n]["valid"] = moe[n]["historical_result_sha_match"] and files_present and moe[n]["historical_result_valid"]
    shas = [v["installed_weights_sha256"] for v in moe.values()]
    inv["moe_family_d"] = dict(
        configs=moe, n=len(moe), duplicates=len(shas) != len(set(shas)), all_valid=all(v["valid"] for v in moe.values()),
        manifest_sources=wm["sources"],
        missing_references=dict(s1_ORIGINAL="weights never stored (runs_partial_ppl quantized in-process)",
                                s2_ORIGINAL="weights never stored"),
        note="per-file sha256, layer digests, tensor counts / shapes / dtype / finiteness and the combined installed digest "
             "are re-verified at install time in the GPU stage (single read of the ~4 GB per set)")
    jdump(wm, p("logs", "moe_weight_manifest.json"))
    inv["all_valid"] = all([inv["dense_grouping"]["all_valid"], inv["dose_response"]["all_valid"],
                            inv["reference_matrix"]["all_present"], inv["moe_family_d"]["all_valid"],
                            not inv["moe_family_d"]["duplicates"]])
    inv["audit_s"] = time.time() - t0
    jdump(inv, p("artifact_inventory.json"))
    print(f"[audit] all valid {inv['all_valid']} ({inv['audit_s']:.0f} s)")


# ====================================================================== protocol
def stage_protocol(a):
    inv, man, ov = jload(p("artifact_inventory.json")), jload(p("evaluation_manifest.json")), jload(p("evaluation_overlap_audit.json"))
    assert inv and inv["all_valid"] and man and ov and ov["passed"], "audit / selection must pass first"
    if glob.glob(p("results", "**", "*.json"), recursive=True):
        raise SystemExit("results already exist: protocol cannot be (re)created")
    body = dict(
        study="Held-out evaluation validation (C4 validation) and dense 3-bit GPTQ execution sensitivity",
        datasets=dict(
            A_wikitext2=dict(windows="runs_partial_ppl/windows.pt", sha256=inv["wikitext2_windows"]["sha256"],
                             role="historical evaluation set (adaptively used); untouched; parity + 3-bit evaluation"),
            B_heldout=dict(windows=man["windows_file"], sha256=man["windows_sha256"], source=man["dataset"]["source"],
                           revision=C4_REV, selection_seed=SEL_SEED, role="new independent evaluation set; frozen before any "
                                                                          "model evaluation on it")),
        evaluator=dict(fn="diagnostics/eval_partial_ppl.layerwise_nll (unchanged)", n_windows=N_WIN, seqlen=SEQLEN,
                       scored_targets=N_WIN * (SEQLEN - 1), batch=BATCH, attn="eager", dtype="bfloat16",
                       ppl="exp(total NLL / 65408); log PPL = total NLL / 65408 (same aggregate)"),
        included=dict(
            moe=dict(configs=MOE_CONFIGS, references=MOE_REFS, model="Qwen/Qwen1.5-MoE-A2.7B", gptq=GPTQ4,
                     inference_routing="natural (historical); frozen routing NOT run",
                     structure="3 allocation seeds x 5 groupings x {FR, FC}; FR/FC paired by (seed, grouping); groupings "
                               "nested in seeds; seed-0 ORIGINAL pair descriptive only (seed-1/2 ORIGINAL weights never "
                               "stored -> no held-out ORIGINAL reference for those seeds)"),
            dense4=dict(configs=DENSE_CONFIGS, model="Qwen/Qwen2.5-1.5B", layers=DENSE_LAYERS, gptq=GPTQ4,
                        statistics_over=GROUPS, reference="ORIGINAL (descriptive; bitwise = G20261115)"),
            dose=dict(configs=dose_names(), policies=["natural", "fixed"], family="b1 (batch-1) k in {1,4,16,64} x 8 "
                      "replicates + k=0", primary_policy="natural", replicate_ids="preserved from runs_dense_dose_response"),
            refmatrix=dict(optional=True, configs=[f"{i}|{j}" for i in GROUPS for j in GROUPS],
                           note="pi_i (reference permutation) x H_j; held-out used only to re-evaluate, never to select"),
            dense3=dict(configs=DENSE_CONFIGS, gptq=GPTQ3, hessians="runs_dense_grouping_scope saved Hessians (sha256-"
                        "verified)", solver="moe_calib.dense_gptq.quantize_layer -> moe_calib.gptq.quantize_expert, bits=3 "
                        "(maxq 7, same asymmetric min/max grid, g128, percdamp 1.0, natural act-order, lazy-batch 128)",
                        datasets=["A_wikitext2", "B_heldout"])),
        metrics=dict(
            moe_primary="Delta[s,g] = PPL(FC,s,g) - PPL(FR,s,g) (positive = FR better); pooled within-seed SD(Delta) = "
                        "sqrt(mean_s SD_s^2), ddof 1 (moe_calib.cross_seed_grouping.variance_summary); R_dataset = "
                        "SD_heldout / SD_WikiText2 (historical 0.006872, recomputed from stored per-window NLL)",
            moe_secondary=["seed means, seed SDs, FR / FC pooled within-seed PPL SDs", "counts of positive / negative Delta",
                           "R1 reversals (sign opposite to sign of M = mean of seed means, within each dataset)",
                           "R2 (seed 0 only on held-out: sign vs seed-0 ORIGINAL Delta on the SAME dataset)",
                           "cross-dataset sign agreement of the 15 Delta (descriptive)",
                           "hierarchical bootstrap of mean Delta (CS.hierarchical_bootstrap, n 20000, seed 20261120)",
                           "historical decision rule CS.decide on each dataset",
                           "paired mean-NLL Delta (nats/token) and its pooled within-seed SD; ratio of NLL-scale SDs",
                           "window-level paired bootstrap CI per pair (full_depth_pilot.paired_bootstrap, conditional on "
                           "weights)", "rank of the seed-0 ORIGINAL FR / FC PPL among the 5 seed-0 grouping variants"],
            dense_primary="over the 5 groupings (ddof 1): S = SD(PPL), SD(log PPL); Q = mean PPL - PPL_BF16 (same dataset); "
                          "R = S / Q; R_log = SD(log PPL) / (mean log PPL - log PPL_BF16); max pairwise diff; per-window "
                          "NLL differences",
            dose="per dose k: s_k = SD_r(PPL) over 8 replicates (ddof 1); Q0 = PPL(k=0 = ORIGINAL) - PPL_BF16 on the same "
                 "dataset; R_PPL,k = s_k / Q0; replicate IDs paired across datasets; chi-square 95% CI for s_k",
            refmatrix="mean PPL per reference permutation over 5 Hessians; within-reference SD; between-reference range; "
                      "rank of pi_ORIGINAL (=pi_G20261115)",
            bitwidth="Q_b, S_b, R_b = S_b/Q_b, R_b_log on each dataset; R_relative = R3/R4; S3/S4; Q3/Q4; per-grouping "
                     "deviations from the bit-specific mean and their Pearson / Spearman correlation (n = 5, descriptive)",
            unstable_ratio_rule="a penalty-normalized ratio is reported as UNDEFINED if Q <= 0 and as UNSTABLE if Q < 2 x "
                                "its window-bootstrap SE (Q not distinguishable from 0)"),
        uncertainty=dict(
            sd_ci="chi-square (n-1 df) 95% CI for single SDs; F(n1-1, n2-1) 95% CI for SD ratios with independent "
                  "samples; for paired 3-bit vs 4-bit ratios a grouping-paired bootstrap (resample the 5 groupings "
                  "jointly across bit-widths, 20000 draws, seed 20261117) is reported as a sensitivity interval",
            window_bootstrap="joint window resampling (2000 draws, seed 20261117) of all configurations of one dataset "
                             "-> conditional-on-weights uncertainty of Q, S, R (evaluation-text uncertainty only)",
            moe="nested seed / grouping structure respected (pooled within-seed SD; hierarchical bootstrap; no flat-15 "
                "inference)",
            power="n ~ ((z_.975 + z_.8) s / delta)^2 for delta in {0.0025, 0.005, 0.01}; paired-design s = measured "
                  "pooled within-seed SD(Delta) (MoE, both datasets); independent-arm illustration s = sqrt(2) x "
                  "single-method SD (dense 4/3-bit, MoE FR/FC) with the independence assumption stated"),
        historical_comparisons=dict(
            moe_wikitext="recomputed from the stored historical natural per-window NLL (bitwise re-verified on the parity "
                         "subset); expected pooled SD(Delta) 0.006872",
            dense4_wikitext="runs_dense_grouping_scope results (re-evaluated bitwise for all 6): SD 0.00503, SD log 3.99e-4, "
                            "Q 0.1007 (ORIGINAL-based) / 0.0954 (5-grouping mean based)",
            dose_wikitext="runs_dense_dose_response results (stored per-window NLL)"),
        decision_criteria=dict(
            note="No equivalence thresholds; outcomes are classified descriptively with exact ratios and intervals. "
                 "Classification labels (A1/A2/A3, B1-B4) are assigned by these prespecified rules:",
            A="A3 if held-out MoE pooled SD(Delta) CI (chi-square) upper bound < 0.5 x WikiText point AND dense4 held-out S "
              "chi-square CI upper bound < 0.5 x WikiText point; A1 if both ratios (held-out / WikiText, MoE R_dataset and "
              "dense S ratio and dense R ratio) lie inside the corresponding F-type 95% band around 1 (i.e. not "
              "distinguishable from equal); A2 otherwise",
            B="on each dataset compute S3/S4 and Q3/Q4 and the paired-bootstrap interval of R3/R4: B1 if the R3/R4 "
              "interval contains 1; B2 if its upper bound < 1; B3 if its lower bound > 1; B4 if the two datasets give "
              "different labels",
            BLOCKED="any mandatory integrity / parity check fails -> results reported as blocked for that component"),
        correctness=dict(
            all="target tensor counts/shapes/dtype/finiteness; weight hashes vs historical manifests; pristine start and "
                "bitwise restore; non-targets unchanged; repeated evaluation bitwise (every dense4, dense3, MoE config; "
                "dose subset k000 + parity configs)",
            parity="WikiText-2 per-window NLL bitwise equal to history: BF16 (both models), dense4 all 6, dose "
                   f"{DOSE_PARITY} (both policies), refmatrix all 25, MoE {MOE_PARITY}",
            dense3=["codes in [0, 7]", "scale*(q-zero) reproduces the FP32 solver output bitwise",
                    "installed BF16 == bf16(solver output)", "deterministic in-process + fresh-process re-solve of "
                    f"{REPEAT_SOLVE_CONFIG}", "bits=4 re-solve reproduces all 6 historical weight sets and codes bitwise",
                    "G20261115 and ORIGINAL 3-bit weights / codes / per-window NLL bitwise identical",
                    "only the 12 target tensors change"],
            moe_non_target="full non-target digest (14 B parameters) at start / after parity / at end; per config: gate "
                           "digests + quantized-layer non-expert parameters + embeddings/head/final norm (restore only "
                           "writes targets, so an end-of-run full-digest match bounds every intermediate config)"),
        constraints=["no historical artifact modified", "no historical 4-bit GPTQ re-run except the bits=4 parity re-solve "
                     "(compared, never used as a result)", "no calibration Hessian regenerated", "held-out results never "
                     "used to choose numerical settings", "production GPTQ defaults unchanged"],
        code_sha256=code_hashes())
    body["protocol_sha256"] = obj_sha(body)
    old = jload(p("protocol.json"))
    if old:
        assert old["protocol_sha256"] == body["protocol_sha256"], "protocol changed after freezing"
        print("protocol already frozen and consistent"); return
    body["created"] = time.strftime("%F %T")
    jdump(body, p("protocol.json"))
    print(f"[protocol] frozen sha256 {body['protocol_sha256'][:16]}")


def check_protocol(require_code=True):
    pr = jload(p("protocol.json"))
    body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created")}
    assert obj_sha(body) == pr["protocol_sha256"], "protocol.json modified"
    if require_code:
        now = code_hashes()
        changed = [f for f in now if now[f] != pr["code_sha256"].get(f)]
        assert not changed, f"code changed since protocol freeze: {changed}"
    return pr


# ====================================================================== dense GPU helpers
def dense_eval(model, win):
    import eval_partial_ppl as EP
    t0 = time.time()
    nll, cnt = EP.layerwise_nll(model, win)
    return nll, cnt, time.time() - t0


def rec_of(nll, cnt, s):
    import eval_partial_ppl as EP
    return EP.ppl_record(nll, cnt, s)


class DenseRig:
    def __init__(self):
        from moe_calib import dense_gptq as DG
        import dense_grouping_scope as DS
        self.DG, self.DS = DG, DS
        t0 = time.time()
        self.model, _ = DS.load_model()
        self.load_s = time.time() - t0
        vc = DS.verify_checkpoint(self.model)
        assert vc["all_equal"] and not vc["missing"], vc
        self.snap = DG.snapshot(self.model, DENSE_LAYERS)
        self.tgt0 = DG.target_digests(self.model, DENSE_LAYERS)
        self.unt0 = DG.untouched_digests(self.model, DENSE_LAYERS)

    def pristine(self):
        return self.DG.target_digests(self.model, DENSE_LAYERS) == self.tgt0

    def run(self, wq, sets, repeat=(), expect_digest=None):
        """Install wq {L: {proj: bf16}}, evaluate on each (name, windows) in `sets` (repeat for names in `repeat`),
        verify and restore. Returns (records by dataset, checks)."""
        DG, m = self.DG, self.model
        assert self.pristine(), "targets not pristine before install"
        for L in DENSE_LAYERS:
            DG.set_weights(m, L, wq[L])
        ck = dict(installed_equals_given=all(torch.equal(DG.get_weights(m, L)[k].detach().cpu(), wq[L][k])
                                             for L in DENSE_LAYERS for k in PROJ),
                  n_target_tensors_12=sum(len(wq[L]) for L in DENSE_LAYERS) == 12,
                  all_targets_differ_from_bf16=all(not torch.equal(wq[L][k], self.snap[L][k]) for L in DENSE_LAYERS for k in PROJ),
                  finite=all(bool(torch.isfinite(wq[L][k].float()).all()) for L in DENSE_LAYERS for k in PROJ),
                  shapes_ok=all(list(wq[L][k].shape) == list(self.snap[L][k].shape) for L in DENSE_LAYERS for k in PROJ),
                  untouched_unchanged=DG.untouched_digests(m, DENSE_LAYERS) == self.unt0)
        dig = DG.installed_digest(m, DENSE_LAYERS)
        ck["installed_digest"] = dig
        if expect_digest is not None:
            ck["installed_digest_matches_history"] = dig == expect_digest
        out = {}
        for nm, win in sets:
            n1, k1, s1 = dense_eval(m, win)
            r = rec_of(n1, k1, s1)
            if nm in repeat:
                n2, _, _ = dense_eval(m, win)
                r["repeat_per_window_bitwise"] = bool(np.array_equal(n1, n2))
            out[nm] = r
        ck["installed_unchanged_during_eval"] = DG.installed_digest(m, DENSE_LAYERS) == dig
        DG.restore(m, self.snap)
        eq, tot = DG.matches(m, self.snap)
        ck["restored"] = eq == tot and self.pristine()
        return out, ck


def ok_checks(ck):
    return all(v for k, v in ck.items() if isinstance(v, bool))


def load_dense_hist_weights(g):
    return {L: torch.load(os.path.join(DCACHE, "weights", g, f"L{L}.pt")) for L in DENSE_LAYERS}


# ====================================================================== dense GPU stage (4-bit, held-out)
def stage_dense(a):
    pr = check_protocol(); log_code_state("dense")
    import eval_partial_ppl as EP
    wt, _ = EP.load_windows()
    ho, _ = load_heldout()
    rig = DenseRig()
    cost_add("dense", "model_load_s", rig.load_s)
    R = lambda *x: p("results", "dense", *x)
    os.makedirs(R(), exist_ok=True)
    hb = jload(os.path.join(ROOT, "runs_dense_grouping_scope/bf16_baseline.json"))
    # ---- BF16
    if not (jload(R("BF16.json")) or {}).get("valid"):
        t0 = time.time()
        n0, k0, s0 = dense_eval(rig.model, wt)
        n1, k1, s1 = dense_eval(rig.model, ho)
        n2, _, _ = dense_eval(rig.model, ho)
        rec = dict(config="BF16", wikitext=rec_of(n0, k0, s0), heldout=rec_of(n1, k1, s1))
        rec["heldout"]["repeat_per_window_bitwise"] = bool(np.array_equal(n1, n2))
        rec["checks"] = dict(wikitext_parity_bitwise=bool(np.array_equal(n0, np.array(hb["per_window_nll"]))),
                             heldout_repeat_bitwise=rec["heldout"]["repeat_per_window_bitwise"],
                             heldout_tokens=int(k1.sum()) == N_WIN * (SEQLEN - 1))
        rec["valid"] = ok_checks(rec["checks"])
        jdump(rec, R("BF16.json"))
        cost_add("dense", "BF16", time.time() - t0)
        print(f"[dense BF16] WT {rec['wikitext']['ppl']:.6f} parity {rec['checks']['wikitext_parity_bitwise']}  "
              f"HO {rec['heldout']['ppl']:.6f} repeat {rec['heldout']['repeat_per_window_bitwise']}", flush=True)
        if not rec["valid"]:
            raise SystemExit("dense BF16 parity / determinism failed -> BLOCKED")
    # ---- dense 4-bit grouping configs
    for g in DENSE_CONFIGS:
        if (jload(R(f"dense4_{g}.json")) or {}).get("valid"):
            continue
        t0 = time.time()
        h = jload(os.path.join(ROOT, f"runs_dense_grouping_scope/results/{g}.json"))
        out, ck = rig.run(load_dense_hist_weights(g), [("wikitext", wt), ("heldout", ho)], repeat=("heldout",),
                          expect_digest=h["installed_weights_sha256"])
        ck["wikitext_parity_bitwise"] = bool(np.array_equal(np.array(out["wikitext"]["per_window_nll"]), np.array(h["per_window_nll"])))
        ck["heldout_repeat_bitwise"] = out["heldout"]["repeat_per_window_bitwise"]
        rec = dict(config=g, bits=4, **out, checks=ck, valid=ok_checks(ck), seconds=time.time() - t0)
        jdump(rec, R(f"dense4_{g}.json"))
        cost_add("dense", f"dense4_{g}", rec["seconds"])
        print(f"[dense4 {g}] WT {out['wikitext']['ppl']:.6f} (parity {ck['wikitext_parity_bitwise']})  HO {out['heldout']['ppl']:.6f} "
              f"valid {rec['valid']}", flush=True)
        if not rec["valid"]:
            raise SystemExit(f"dense4 {g} integrity failed -> BLOCKED")
    # ---- dose-response (batch-1 family), both policies
    winv = jload(os.path.join(ROOT, "runs_dense_dose_response/artifact_inventory.json"))["weight_hashes"]
    for n in dose_names():
        if (jload(R(f"dose_{n}.json")) or {}).get("valid"):
            continue
        t0 = time.time()
        hres = jload(os.path.join(ROOT, f"runs_dense_dose_response/results/{n}.json"))
        rec = dict(config=n, spec=hres["config"], policies={})
        for pol in ("natural", "fixed"):
            wq, tsha = {}, True
            for L in DENSE_LAYERS:
                d = torch.load(os.path.join(DOSE_CACHE, "cfg", n, f"Q_{pol}_L{L}.pt"))
                wq[L] = d["weights"]
                for k in PROJ:
                    tsha &= tensor_sha(wq[L][k]) == hres["policies"][pol]["meta"][str(L)][k]["weight_sha256"]
                del d
            parity = n in DOSE_PARITY
            sets = ([("wikitext", wt)] if parity else []) + [("heldout", ho)]
            rep = ("heldout",) if (parity or n == "b1_k000") else ()
            out, ck = rig.run(wq, sets, repeat=rep, expect_digest=winv[f"{n}|{pol}"])
            ck["tensor_sha256_match_history"] = bool(tsha)
            if parity:
                ck["wikitext_parity_bitwise"] = bool(np.array_equal(np.array(out["wikitext"]["per_window_nll"]),
                                                                    np.array(hres["policies"][pol]["per_window_nll"])))
            if "repeat_per_window_bitwise" in out["heldout"]:
                ck["heldout_repeat_bitwise"] = out["heldout"]["repeat_per_window_bitwise"]
            rec["policies"][pol] = dict(**out, checks=ck, valid=ok_checks(ck),
                                        historical_wikitext_ppl=hres["policies"][pol]["ppl"],
                                        historical_wikitext_per_window_nll=hres["policies"][pol]["per_window_nll"])
        rec["valid"] = all(v["valid"] for v in rec["policies"].values())
        rec["seconds"] = time.time() - t0
        jdump(rec, R(f"dose_{n}.json"))
        cost_add("dense", f"dose_{n}", rec["seconds"])
        print(f"[dose {n}] HO nat {rec['policies']['natural']['heldout']['ppl']:.6f} fix {rec['policies']['fixed']['heldout']['ppl']:.6f} "
              f"valid {rec['valid']}", flush=True)
        if not rec["valid"]:
            raise SystemExit(f"dose {n} integrity failed -> BLOCKED")
    # ---- optional: 5 x 5 reference matrix
    if not a.skip_optional:
        rraw = jload(os.path.join(ROOT, "runs_dense_dose_response/logs/refmatrix_raw.json"))
        for i in GROUPS:
            for j in GROUPS:
                nm = f"ref_{i}__{j}"
                if (jload(R(f"{nm}.json")) or {}).get("valid"):
                    continue
                t0 = time.time()
                d = torch.load(os.path.join(DOSE_CACHE, "refmatrix", f"{i}__{j}.pt"))
                wq = d["weights"]
                del d
                extra = {}
                if i == j:
                    hw = load_dense_hist_weights(j)
                    extra["diag_equals_historical_natural"] = all(torch.equal(wq[L][k], hw[L][k]) for L in DENSE_LAYERS for k in PROJ)
                out, ck = rig.run(wq, [("wikitext", wt), ("heldout", ho)])
                ck.update(extra)
                ck["wikitext_parity_bitwise"] = bool(np.array_equal(np.array(out["wikitext"]["per_window_nll"]),
                                                                    np.array(rraw[f"{i}|{j}"]["per_window_nll"])))
                rec = dict(config=nm, reference_perm=i, hessian=j, **out, checks=ck, valid=ok_checks(ck), seconds=time.time() - t0)
                jdump(rec, R(f"{nm}.json"))
                cost_add("dense", nm, rec["seconds"])
                print(f"[ref pi={i} H={j}] HO {out['heldout']['ppl']:.6f} parity {ck['wikitext_parity_bitwise']} valid {rec['valid']}", flush=True)
                if not rec["valid"]:
                    raise SystemExit(f"refmatrix {nm} integrity failed -> BLOCKED")
    # ---- end checks
    bf = jload(R("BF16.json"))
    n0, _, _ = dense_eval(rig.model, wt)
    n1, _, _ = dense_eval(rig.model, ho)
    end = dict(bf16_wikitext_bitwise=bool(np.array_equal(n0, np.array(bf["wikitext"]["per_window_nll"]))),
               bf16_heldout_bitwise=bool(np.array_equal(n1, np.array(bf["heldout"]["per_window_nll"]))),
               targets_pristine=rig.pristine(), untouched_unchanged=rig.DG.untouched_digests(rig.model, DENSE_LAYERS) == rig.unt0,
               peak_cpu_gb=peak_cpu_gb(), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
               at=time.strftime("%F %T"))
    jdump(end, p("logs", "dense_end_checks.json"))
    print(f"[dense end] {end}", flush=True)


# ====================================================================== dense 3-bit stage
def d3dir(*x):
    return c("dense3", *x)


def solve_layer(rig, L, H, bits):
    """Unchanged solver via dense_gptq.quantize_layer (pristine FP weights of the model) + integer codes."""
    DG = rig.DG
    assert rig.pristine()
    Wq, infos = DG.quantize_layer(rig.model, L, H["gu"], H["down"], bits, 128, 1.0)
    codes, meta = {}, {}
    for k in PROJ:
        q, gid, ex = DG.integer_codes(Wq[k], infos[k], bits, 128)
        sc, zr = infos[k]["scale"].float(), infos[k]["zero"].float()
        codes[k] = dict(q=q, group=gid, perm=infos[k]["perm"].int(), scale=infos[k]["scale"], zero=infos[k]["zero"])
        meta[k] = dict(codes_exact=bool(ex), code_min=int(q.min()), code_max=int(q.max()),
                       codes_in_range=bool(int(q.min()) >= 0 and int(q.max()) <= 2 ** bits - 1),
                       zero_in_range=bool(float(zr.min()) >= 0 and float(zr.max()) <= 2 ** bits - 1),
                       scale_positive=bool((sc > 0).all()), finite=bool(torch.isfinite(Wq[k]).all()),
                       n_levels_used=int(torch.unique(q).numel()))
    return Wq, codes, meta


def stage_dense3(a):
    pr = check_protocol(); log_code_state("dense3")
    import eval_partial_ppl as EP
    wt, _ = EP.load_windows()
    ho, _ = load_heldout()
    inv = jload(p("artifact_inventory.json"))["dense_grouping"]["configs"]
    rig = DenseRig()
    cost_add("dense3", "model_load_s", rig.load_s)
    R = lambda *x: p("results", "dense3", *x)
    os.makedirs(R(), exist_ok=True)
    corr = jload(p("logs", "dense3_checks_raw.json")) or {}
    for g in DENSE_CONFIGS:
        if (jload(R(f"dense3_{g}.json")) or {}).get("valid"):
            continue
        t0 = time.time()
        os.makedirs(d3dir(g), exist_ok=True)
        wq3, per_layer, tq = {}, {}, 0.0
        p4 = {}
        for L in DENSE_LAYERS:
            fh = os.path.join(DCACHE, "hessians", g, f"L{L}.pt")
            hsha = sha256_file(fh)
            assert hsha == inv[g]["files"][str(L)]["hessian_sha256"], f"{g} L{L} Hessian sha mismatch"
            H = torch.load(fh)
            tq0 = time.time()
            Wq, codes, meta = solve_layer(rig, L, H, 3)
            tq += time.time() - tq0
            bf = {k: Wq[k].to(torch.bfloat16).cpu() for k in PROJ}
            torch.save(dict(weights=bf, codes=codes, meta=meta, hessian_sha256=hsha, bits=3), d3dir(g, f"L{L}.pt"))
            wq3[L] = bf
            # bits=4 parity re-solve (never used as a result)
            W4, C4, M4 = solve_layer(rig, L, H, 4)
            hw = torch.load(os.path.join(DCACHE, "weights", g, f"L{L}.pt"))
            hc = torch.load(os.path.join(DCACHE, "codes", g, f"L{L}.pt"))
            p4[str(L)] = dict(weights_bitwise=all(torch.equal(W4[k].to(torch.bfloat16).cpu(), hw[k]) for k in PROJ),
                              codes_bitwise=all(torch.equal(C4[k]["q"], hc[k]["q"]) for k in PROJ),
                              perm_bitwise=all(torch.equal(C4[k]["perm"], hc[k]["perm"]) for k in PROJ))
            rep = None
            if g == REPEAT_SOLVE_CONFIG:
                W3b, C3b, _ = solve_layer(rig, L, H, 3)
                rep = dict(fp32_bitwise=all(torch.equal(W3b[k], Wq[k]) for k in PROJ),
                           codes_bitwise=all(torch.equal(C3b[k]["q"], codes[k]["q"]) for k in PROJ))
                del W3b, C3b
            per_layer[str(L)] = dict(meta=meta, hessian_sha256=hsha, file_sha256=sha256_file(d3dir(g, f"L{L}.pt")),
                                     weight_sha256={k: tensor_sha(bf[k]) for k in PROJ},
                                     bf16_cast_equals_solver=all(torch.equal(bf[k], Wq[k].to(torch.bfloat16).cpu()) for k in PROJ),
                                     bits4_parity=p4[str(L)], repeat_solve=rep,
                                     frac_bf16_equal_to_4bit_historical={k: float((bf[k] == hw[k]).float().mean()) for k in PROJ})
            del Wq, W4, C4, H, hw, hc
            torch.cuda.empty_cache()
        out, ck = rig.run(wq3, [("wikitext", wt), ("heldout", ho)], repeat=("wikitext", "heldout"))
        ck.update(codes_exact=all(m["codes_exact"] for v in per_layer.values() for m in v["meta"].values()),
                  codes_in_range=all(m["codes_in_range"] for v in per_layer.values() for m in v["meta"].values()),
                  zero_in_range=all(m["zero_in_range"] for v in per_layer.values() for m in v["meta"].values()),
                  scales_positive=all(m["scale_positive"] for v in per_layer.values() for m in v["meta"].values()),
                  solver_finite=all(m["finite"] for v in per_layer.values() for m in v["meta"].values()),
                  bf16_cast_equals_solver=all(v["bf16_cast_equals_solver"] for v in per_layer.values()),
                  bits4_parity=all(all(x.values()) for x in p4.values()),
                  wikitext_repeat_bitwise=out["wikitext"]["repeat_per_window_bitwise"],
                  heldout_repeat_bitwise=out["heldout"]["repeat_per_window_bitwise"])
        if g == REPEAT_SOLVE_CONFIG:
            ck["repeat_solve_bitwise"] = all(v["repeat_solve"]["fp32_bitwise"] and v["repeat_solve"]["codes_bitwise"]
                                             for v in per_layer.values())
        rec = dict(config=g, bits=3, gptq=GPTQ3, **out, per_layer=per_layer, checks=ck, valid=ok_checks(ck),
                   gptq_s=tq, seconds=time.time() - t0, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
        jdump(rec, R(f"dense3_{g}.json"))
        cost_add("dense3", g, dict(gptq3_s=tq, total_s=rec["seconds"]))
        print(f"[dense3 {g}] WT {out['wikitext']['ppl']:.6f} HO {out['heldout']['ppl']:.6f} valid {rec['valid']} "
              f"bits4 parity {ck['bits4_parity']} ({rec['seconds']:.0f} s)", flush=True)
        if not rec["valid"]:
            raise SystemExit(f"dense3 {g} integrity failed -> BLOCKED (diagnose before interpreting)")
    # ---- null-control parity: G20261115 == ORIGINAL at 3 bits
    A, B = jload(R("dense3_G20261115.json")), jload(R("dense3_ORIGINAL.json"))
    nul = {}
    for L in DENSE_LAYERS:
        x, y = torch.load(d3dir("G20261115", f"L{L}.pt")), torch.load(d3dir("ORIGINAL", f"L{L}.pt"))
        nul[str(L)] = dict(weights=all(torch.equal(x["weights"][k], y["weights"][k]) for k in PROJ),
                           codes=all(torch.equal(x["codes"][k]["q"], y["codes"][k]["q"]) for k in PROJ),
                           scale_zero=all(torch.equal(x["codes"][k]["scale"], y["codes"][k]["scale"]) and
                                          torch.equal(x["codes"][k]["zero"], y["codes"][k]["zero"]) for k in PROJ))
    corr["null_G20261115_vs_ORIGINAL"] = dict(per_layer=nul,
                                              per_window_wikitext=A["wikitext"]["per_window_nll"] == B["wikitext"]["per_window_nll"],
                                              per_window_heldout=A["heldout"]["per_window_nll"] == B["heldout"]["per_window_nll"],
                                              installed_digest=A["checks"]["installed_digest"] == B["checks"]["installed_digest"])
    corr["null_G20261115_vs_ORIGINAL"]["passed"] = all(all(v.values()) for v in nul.values()) and all(
        v for k, v in corr["null_G20261115_vs_ORIGINAL"].items() if isinstance(v, bool))
    n0, _, _ = dense_eval(rig.model, wt)
    bf = jload(p("results", "dense", "BF16.json"))
    corr["end"] = dict(bf16_wikitext_bitwise=bool(np.array_equal(n0, np.array(bf["wikitext"]["per_window_nll"]))) if bf else None,
                       targets_pristine=rig.pristine(),
                       untouched_unchanged=rig.DG.untouched_digests(rig.model, DENSE_LAYERS) == rig.unt0,
                       peak_cpu_gb=peak_cpu_gb(), at=time.strftime("%F %T"))
    jdump(corr, p("logs", "dense3_checks_raw.json"))
    print(f"[dense3] null control {corr['null_G20261115_vs_ORIGINAL']['passed']}; end {corr['end']}", flush=True)


def stage_dense3_repeat(a):
    """Fresh-process deterministic re-solve of the prespecified config at 3 bits vs the saved artifacts."""
    check_protocol(); log_code_state("dense3_repeat")
    inv = jload(p("artifact_inventory.json"))["dense_grouping"]["configs"]
    rig = DenseRig()
    out = {}
    for L in DENSE_LAYERS:
        fh = os.path.join(DCACHE, "hessians", REPEAT_SOLVE_CONFIG, f"L{L}.pt")
        assert sha256_file(fh) == inv[REPEAT_SOLVE_CONFIG]["files"][str(L)]["hessian_sha256"]
        Wq, codes, _ = solve_layer(rig, L, torch.load(fh), 3)
        sv = torch.load(d3dir(REPEAT_SOLVE_CONFIG, f"L{L}.pt"))
        out[str(L)] = dict(weights_bitwise=all(torch.equal(Wq[k].to(torch.bfloat16).cpu(), sv["weights"][k]) for k in PROJ),
                           codes_bitwise=all(torch.equal(codes[k]["q"], sv["codes"][k]["q"]) for k in PROJ))
    out["passed"] = all(v["weights_bitwise"] and v["codes_bitwise"] for v in out.values() if isinstance(v, dict))
    out["config"] = REPEAT_SOLVE_CONFIG
    jdump(out, p("logs", "dense3_fresh_process_repeat.json"))
    print(out)


# ====================================================================== MoE GPU stage
LAYERS_Q = (1, 8, 16, 23)
MOE_SHAPES = {"gate_proj": (1408, 2048), "up_proj": (1408, 2048), "down_proj": (2048, 1408)}


def moe_light_digest(model):
    """Per-config non-target check: router gates (all 24 layers), non-expert parameters of the quantized layers,
    embeddings, final norm, lm_head."""
    from moe_calib import model_utils as MU
    h = hashlib.sha1()
    for L in range(MU.num_layers(model)):
        h.update(MU.moe_block(model, L).gate.weight.detach().contiguous().cpu().view(torch.int16).numpy().tobytes())
    for L in LAYERS_Q:
        for n, prm in model.model.layers[L].named_parameters():
            if ".experts." not in n and not prm.is_meta:
                t = prm.detach().contiguous().cpu()
                h.update((t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes())
    for t in (model.model.embed_tokens.weight, model.model.norm.weight, model.lm_head.weight):
        h.update(t.detach().contiguous().cpu().view(torch.int16).numpy().tobytes())
    return h.hexdigest()


def moe_install(model, ent, snap):
    """Install a historical weight set: file sha256, keys, tensor counts / shapes / dtype / finiteness, layer and
    combined digests, all 720 projections differ from BF16."""
    from moe_calib import model_utils as MU, partial_model_quant as PQ
    t0 = time.time()
    eq, tot = PQ.experts_match(model, snap)
    assert eq == tot, "model not pristine before install"
    ck = dict(file_sha256=True, keys=True, n_tensors=0, shapes=True, dtype_bf16=True, finite=True, layer_sha256=True)
    for L in LAYERS_Q:
        path = os.path.join(ent["weights_dir"], f"L{L}.pt")
        with open(path, "rb") as f:
            b = f.read()
        ck["file_sha256"] &= sha256_bytes(b) == ent["files"][str(L)]["sha256"]
        d = torch.load(io.BytesIO(b))
        del b
        ck["keys"] &= set(d) == {f"{e}.{k}" for e in range(60) for k in PROJ}
        for e in range(60):
            W = {k: d[f"{e}.{k}"] for k in PROJ}
            for k, t in W.items():
                ck["n_tensors"] += 1
                ck["shapes"] &= tuple(t.shape) == MOE_SHAPES[k]
                ck["dtype_bf16"] &= t.dtype == torch.bfloat16
                ck["finite"] &= bool(torch.isfinite(t.float()).all())
            MU.set_expert_weights(model, L, e, W)
        del d
        ck["layer_sha256"] &= PQ.layer_digest(model, L) == ent["files"][str(L)]["layer_sha256"]
    ck["n_tensors_720"] = ck.pop("n_tensors") == 720
    ck["installed_digest"] = PQ.installed_digest(model, LAYERS_Q) == ent["installed_weights_sha256"]
    eq, _ = PQ.experts_match(model, snap)
    ck["all_720_differ_from_bf16"] = eq == 0
    ck["install_s"] = time.time() - t0
    return ck


def stage_moe(a):
    pr = check_protocol(); log_code_state("moe")
    import eval_partial_ppl as EP
    from moe_calib import model_utils as MU, partial_model_quant as PQ
    wt, _ = EP.load_windows()
    ho, _ = load_heldout()
    wm = jload(p("logs", "moe_weight_manifest.json"))
    R = lambda *x: p("results", "moe", *x)
    os.makedirs(R(), exist_ok=True)
    t0 = time.time()
    model, _ = MU.load_model()
    cost_add("moe", f"model_load_s@{time.strftime('%T')}", time.time() - t0)
    snap = PQ.snapshot_experts(model, LAYERS_Q)
    t1 = time.time()
    unt0 = PQ.untouched_digests(model, LAYERS_Q)
    cost_add("moe", "untouched_digest_s", time.time() - t1)
    light0 = moe_light_digest(model)
    hb = jload(os.path.join(ROOT, "runs_partial_ppl/bf16_baseline.json"))
    ev = lambda w: dense_eval(model, w)
    if not (jload(R("BF16.json")) or {}).get("valid"):
        t0 = time.time()
        n0, k0, s0 = ev(wt)
        n1, k1, s1 = ev(ho)
        n2, _, _ = ev(ho)
        rec = dict(config="BF16", wikitext=rec_of(n0, k0, s0), heldout=rec_of(n1, k1, s1))
        rec["checks"] = dict(wikitext_parity_bitwise=bool(np.array_equal(n0, np.array(hb["runs"][0]["per_window_nll"]))),
                             heldout_repeat_bitwise=bool(np.array_equal(n1, n2)))
        rec["valid"] = ok_checks(rec["checks"])
        jdump(rec, R("BF16.json"))
        cost_add("moe", "BF16", time.time() - t0)
        print(f"[moe BF16] WT {rec['wikitext']['ppl']:.6f} parity {rec['checks']['wikitext_parity_bitwise']} HO "
              f"{rec['heldout']['ppl']:.6f} repeat {rec['checks']['heldout_repeat_bitwise']}", flush=True)
        if not rec["valid"]:
            raise SystemExit("MoE BF16 parity failed -> BLOCKED")
    order = MOE_PARITY + [n for n in MOE_CONFIGS + MOE_REFS if n not in MOE_PARITY]
    full_checks = jload(p("logs", "moe_full_untouched_checks.json")) or {}
    for n in order:
        if (jload(R(f"{n}.json")) or {}).get("valid"):
            continue
        t0 = time.time()
        ent = wm["configs"][n]
        ick = moe_install(model, ent, snap)
        out = {}
        if n in MOE_PARITY:
            nw, kw, sw = ev(wt)
            out["wikitext"] = rec_of(nw, kw, sw)
        n1, k1, s1 = ev(ho)
        n2, _, _ = ev(ho)
        out["heldout"] = rec_of(n1, k1, s1)
        ck = dict(**{k: v for k, v in ick.items() if k != "install_s"},
                  heldout_repeat_bitwise=bool(np.array_equal(n1, n2)),
                  installed_unchanged_during_eval=PQ.installed_digest(model, LAYERS_Q) == ent["installed_weights_sha256"],
                  light_non_target_unchanged=moe_light_digest(model) == light0)
        if n in MOE_PARITY:
            ck["wikitext_parity_bitwise"] = bool(np.array_equal(nw, np.array(ent["historical_per_window_nll"], dtype=np.float64)))
        PQ.restore_experts(model, snap)
        eq, tot = PQ.experts_match(model, snap)
        ck["restored_pristine"] = eq == tot
        rec = dict(config=n, strategy=ent["strategy"], allocation_seed=ent.get("allocation_seed"),
                   grouping=ent.get("grouping"), kind=ent.get("kind"), installed_weights_sha256=ent["installed_weights_sha256"],
                   historical_wikitext_ppl=float(ent["historical_ppl"]),
                   historical_wikitext_per_window_nll=ent["historical_per_window_nll"], **out, checks=ck,
                   valid=ok_checks(ck), install_s=ick["install_s"], seconds=time.time() - t0, peak_cpu_gb=peak_cpu_gb())
        jdump(rec, R(f"{n}.json"))
        cost_add("moe", n, dict(install_s=ick["install_s"], total_s=rec["seconds"], peak_cpu_gb=peak_cpu_gb()))
        print(f"[moe {n}] HO {out['heldout']['ppl']:.6f} (hist WT {float(ent['historical_ppl']):.6f}"
              + (f", WT parity {ck['wikitext_parity_bitwise']}" if n in MOE_PARITY else "") + f") valid {rec['valid']} "
              f"{rec['seconds']:.0f} s", flush=True)
        if not rec["valid"]:
            raise SystemExit(f"MoE {n} integrity failed -> BLOCKED")
        if n == MOE_PARITY[-1] and "after_parity" not in full_checks:
            full_checks["after_parity"] = PQ.untouched_digests(model, LAYERS_Q) == unt0
            jdump(full_checks, p("logs", "moe_full_untouched_checks.json"))
        gc.collect()
    bf = jload(R("BF16.json"))
    n0, _, _ = ev(wt)
    n1, _, _ = ev(ho)
    eq, tot = PQ.experts_match(model, snap)
    full_checks[f"end@{time.strftime('%F %T')}"] = dict(
        untouched_unchanged=PQ.untouched_digests(model, LAYERS_Q) == unt0, n_untouched=len(unt0), targets_pristine=eq == tot,
        bf16_wikitext_bitwise=bool(np.array_equal(n0, np.array(bf["wikitext"]["per_window_nll"]))),
        bf16_heldout_bitwise=bool(np.array_equal(n1, np.array(bf["heldout"]["per_window_nll"]))),
        light_non_target_unchanged=moe_light_digest(model) == light0, peak_cpu_gb=peak_cpu_gb(),
        peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
    jdump(full_checks, p("logs", "moe_full_untouched_checks.json"))
    print(f"[moe end] {full_checks}", flush=True)


# ====================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "audit", "protocol", "dense", "dense3", "dense3_repeat",
                                                       "moe", "analyze"])
    ap.add_argument("--skip-optional", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    os.makedirs(p("figures"), exist_ok=True)
    if a.stage == "analyze":
        import report_heldout_bitwidth_validation as R
        return R.analyze()
    dict(select=stage_select, audit=stage_audit, protocol=stage_protocol, dense=stage_dense, dense3=stage_dense3,
         dense3_repeat=stage_dense3_repeat, moe=stage_moe)[a.stage](a)


if __name__ == "__main__":
    main()
