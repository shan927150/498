"""Official MoEQuant EBSS calibration-quality validation (Qwen1.5-MoE-A2.7B, routed experts of layers
1/8/16/23, frozen GPTQ 4-bit g128 percdamp 1.0 act-order, B = 128 x 512).

Compares Full Random (FR), Typical Random (TR), Full Coverage-LP (FC; negative control), seeds 0/1/2,
with ONE frozen 128-window set built from the released `EBSS_qwen.jsonl`. All ten calibration sets go
through one matched, source-independent extraction policy (CANONICAL-B1: every calibration sequence is
forwarded ALONE, batch size 1, BF16) because the historical C4 caches were built in batches of 8
consecutive rows, which the EBSS records cannot share. Reconstruction (E3 held-out rows 2128-2255 of
~/moe_cache) and WikiText-2 evaluation (frozen 128 x 512 windows) are unchanged.

Stages (--stage): preflight | download | inspect_data | compatibility | freeze_selection |
                  baseline_regression | capture | capture_repeat | quantize --config NAME [--repeat] |
                  reconstruction | ppl | report
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, platform, shutil, subprocess, sys, time, urllib.request

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import eval_partial_ppl as EP           # noqa: E402
import full_depth_pilot as FD           # noqa: E402
from moe_calib import ebss_official_data as ED   # noqa: E402

OUT = "runs_ebss_official_validation"
OCACHE = os.path.expanduser("~/moe_cache")
ECACHE = os.path.expanduser(os.environ.get("EBSS_VALIDATION_CACHE", "~/moe_cache_ebss_official_validation"))
SRC = os.path.expanduser("~/ebss_official_src")
REPO_URL = "https://github.com/chenzx921020/MoEQuant"
COMMIT = "d77ba1b7f31f4131a90798f5a6df5bf5d3b414b0"
REL_PATH = "code/EBSS_data/EBSS_qwen.jsonl"
RAW_URL = f"https://raw.githubusercontent.com/chenzx921020/MoEQuant/{COMMIT}/{REL_PATH}"
LAYERS = [1, 8, 16, 23]
N_LAYERS = 24
STRATS = ("FR", "TR", "FC")
SEEDS = (0, 1, 2)
CONFIGS = [f"{c}_s{s}" for c in STRATS for s in SEEDS] + ["EBSS"]
QUANT_ORDER = ["FR_s0", "EBSS", "TR_s0", "FC_s0", "FR_s1", "TR_s1", "FC_s1", "FR_s2", "TR_s2", "FC_s2"]
BF16_REF = 9.237090
REPEAT_CONFIG = "FR_s0"
N_REPEAT_ROWS = 8                    # per source, capture repeatability control


def p(*a):
    return os.path.join(ROOT, OUT, *a)


jdump, sha256_file = FD.jdump, FD.sha256_file


def tensor_sha(t):
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def env_record():
    import transformers
    return dict(torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__,
                python=platform.python_version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


def code_hashes():
    files = ["moe_calib/ebss_official_data.py", "diagnostics/official_ebss_validation.py", "moe_calib/gptq.py",
             "moe_calib/profiler.py", "moe_calib/hessian.py", "moe_calib/partial_model_quant.py",
             "moe_calib/model_utils.py", "e3_allocation.py", "diagnostics/eval_partial_ppl.py"]
    return {f: sha256_file(os.path.join(ROOT, f)) for f in files}


def frozen_cfg():
    return json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")


def raw_path():
    return p("official_data", "EBSS_qwen.jsonl")


def c4_tokens():
    ids = torch.load(os.path.join(OCACHE, "ids.pt"))
    return torch.cat([ids["pool"], ids["holdout"]])


def selections():
    """The 9 frozen C4 selections, re-verified against their source files (partial-PPL manifest)."""
    import e3_allocation as E3
    man = json.load(open(os.path.join(ROOT, EP.OUT, "selected_set_manifest.json")))
    out = {}
    for c in STRATS:
        for s in SEEDS:
            ent = man["configs"][f"{c}_s{s}"]
            src = os.path.join(ROOT, ent["source"])
            r = json.load(open(src))
            assert sha256_file(src) == ent["source_sha256"], f"{src} changed"
            assert r["chosen"] == ent["chosen"] and E3.warm_hash(r["chosen"]) == ent["ids_hash"]
            assert ent["valid"] and all(ent["checks"].values())
            out[f"{c}_s{s}"] = dict(strategy=c, seed=s, name=ent["name"], source=ent["source"],
                                    source_sha256=ent["source_sha256"], ids_hash=ent["ids_hash"], K=ent["K"],
                                    chosen=list(map(int, r["chosen"])), checks=ent["checks"])
    return out


# ================================================================ stage: preflight
def stage_preflight(a):
    os.makedirs(p("logs"), exist_ok=True)
    cfg = frozen_cfg()
    sel = selections()
    win, prot = EP.load_windows()
    mem = dict(line.split(":")[0:2] for line in open("/proc/meminfo") if ":" in line)
    gpu_free, gpu_total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    rec = dict(frozen_config=cfg, selections_verified={k: v["ids_hash"] for k, v in sel.items()},
               windows_sha256=prot["windows_sha256"], environment=env_record(),
               host_mem_available_gb=int(mem["MemAvailable"].split()[0]) / 1e6, gpu_free_gb=gpu_free / 1e9,
               gpu_total_gb=gpu_total / 1e9, disk_free_tb=shutil.disk_usage(os.path.expanduser("~")).free / 1e12,
               existing_outputs_untouched="this study writes only to runs_ebss_official_validation/ and "
                                          f"{ECACHE}")
    jdump(rec, p("logs", "preflight.json"))
    print(f"[preflight] 9 C4 selections verified; GPU free {gpu_free / 1e9:.1f} GB; host avail "
          f"{rec['host_mem_available_gb']:.0f} GB")


# ================================================================ stage: download
def stage_download(a):
    os.makedirs(p("official_data"), exist_ok=True)
    t0 = time.time()
    if not os.path.exists(raw_path()):
        with urllib.request.urlopen(RAW_URL, timeout=60) as r:
            raw = r.read()
        open(raw_path() + ".tmp", "wb").write(raw)
        os.replace(raw_path() + ".tmp", raw_path())
    t_dl = time.time() - t0
    raw = open(raw_path(), "rb").read()
    sha = ED.sha256_bytes(raw)
    clone = os.path.join(SRC, "moequant")
    G = ["git", "-c", f"safe.directory={clone}", "-C", clone]
    head = subprocess.run(G + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    clone_sha = sha256_file(os.path.join(clone, REL_PATH))
    hist = subprocess.run(G + ["log", "--format=%H|%ad|%an|%s", "--date=iso", "--", REL_PATH],
                          capture_output=True, text=True).stdout.strip().splitlines()
    blob = subprocess.run(G + ["rev-parse", f"{COMMIT}:{REL_PATH}"], capture_output=True, text=True).stdout.strip()
    git_blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    prov = dict(
        repository=REPO_URL, commit=COMMIT, clone_head=head, clone_head_equals_commit=head == COMMIT,
        path=REL_PATH, download_url=RAW_URL, local_copy=os.path.relpath(raw_path(), ROOT), bytes=len(raw),
        sha256=sha, clone_file_sha256=clone_sha, download_equals_clone=sha == clone_sha,
        git_blob_sha1_in_commit=blob, git_blob_sha1_of_download=git_blob, git_blob_match=blob == git_blob,
        file_history=[dict(zip(("commit", "date", "author", "subject"), h.split("|", 3))) for h in hist],
        download_s=t_dl,
        official_consumption=dict(
            loader="code/fake_quant/main.py (--EBSS_calib --calib_path): reads the FIRST args.nsamples (default 128) "
                   "JSONL lines, DataLoader(batch_size=1, shuffle=True)",
            tokenization="code/fake_quant/gptq_utils.py gptq_fwrd: the branch that handles text records "
                         "(`if args.llmqat_calib`) tokenizes with truncation at model.seqlen = 2048, padding=True, "
                         "then F.pad(..., value 0) to 2048 tokens and calls model(input_ids) with no attention mask; "
                         "`--llmqat_calib` is not defined in the released argument parser (utils.py), so the released "
                         "EBSS path is not runnable as published",
            seqlen=2048, nsamples=128,
            readme_example="uses --calib_path ./gen_data/EBSS_mixtral_8x7b.json (not the released EBSS_data file)",
            supported_qwen_paths=["/ssd/model/qwen15_moe_2.7b_chat", "/ssd/model/qwen1.5_moe_a2.7b"],
            qwen_model_code_dir="code/qwen_moe_14b_chat/ (configuration_qwen2_moe.py, modeling_qwen2_moe.py)"),
        authors_descriptions=dict(
            rebuttal_tables="Tables_for_R2/R3/R4*.md: 'MoEQuant generates 128 sequences of 512 length' (time-cost "
                            "tables); Table R4 'snapshot of generated samples by Qwen-MoE-14B' begins '! ### GMAT "
                            "Club Daily Prep ...' (record 0 of EBSS_qwen.jsonl begins '! A A. 1 B. 2 C. 3 D. 4 ... "
                            "### GMAT Club Daily Prep ...'); EBSS generation time 'Qwen-MoE-14B 18 mins' (A800, "
                            "Table R6)",
            paper="ICML 2025 (PMLR 267:8245-8260; text read from arXiv 2505.03804 HTML, the PMLR PDF could not be "
                  "parsed): Sec. 4.2 EBSS = beam-like search keeping w branches scored by cumulative log-probability "
                  "and the per-layer SD of expert usage (tau = 1.2, w = 4); 'Qwen-MoE-14B' cites "
                  "huggingface.co/Qwen/Qwen1.5-MoE-A2.7B; the number and length of EBSS sequences are not stated; "
                  "reported Qwen results combine EBSS with AGQ and QuaRot-style Hadamard rotation; no EBSS "
                  "generation time in the paper"),
        generating_checkpoint="UNDOCUMENTED for this artifact: no metadata in the file; the paper evaluates "
                              "Qwen1.5-MoE-A2.7B (base) and also a Chat variant elsewhere; the repo ships a "
                              "'qwen_moe_14b_chat' model directory and lists both base and chat local paths",
        generation_procedure="consistent with one self-generated continuation per start token: records 0-93 "
                             "begin exactly with the text of vocabulary token id i (single-byte tokens); later "
                             "records begin with a lone byte that decodes to U+FFFD or merges with the next bytes, "
                             "so the seed token cannot be verified from text for them (official_dataset_audit.json); "
                             "length up to ~2048 tokens; search / sampling settings for this file are not documented",
        raw_file_unchanged="the local copy is the byte-exact official blob; it is never edited")
    jdump(prov, p("source_provenance.json"))
    assert prov["download_equals_clone"] and prov["git_blob_match"], "artifact integrity failed"
    print(f"[download] {len(raw)} bytes sha256 {sha[:16]}; blob match {prov['git_blob_match']}")


# ================================================================ stage: inspect_data
def stage_inspect(a):
    raw = open(raw_path(), "rb").read()
    recs, bad = ED.parse_jsonl(raw)
    tok = tokenizer()
    t0 = time.time()
    toks = [tok(r["text"], add_special_tokens=False).input_ids for r in recs]
    t_tok = time.time() - t0
    n = np.array([len(x) for x in toks])
    texts = [r["text"] for r in recs]
    cnt = collections.Counter(texts)
    first_ids = [x[0] if x else None for x in toks]
    special = {s: sum(s in t for t in texts) for s in ("<|endoftext|>", "<|im_start|>", "<|im_end|>")}
    special_ids = set(tok.all_special_ids)
    n_special_tok = int(sum(sum(1 for i in x if i in special_ids) for x in toks))
    roundtrip = [i for i, (x, t) in enumerate(zip(toks, texts)) if tok.decode(x) != t]
    long_ids = [i for i in range(len(toks)) if n[i] >= ED.SEQLEN]
    first512 = [toks[i][:ED.SEQLEN] for i in long_ids]
    q = lambda a: {k: float(np.quantile(a, v)) for k, v in (("min", 0), ("p5", .05), ("p25", .25), ("median", .5),
                                                             ("p75", .75), ("p95", .95), ("max", 1))}
    cjk = sum(any("一" <= ch <= "鿿" for ch in t) for t in texts)
    audit = dict(
        n_lines=len(raw.decode("utf-8").splitlines()), n_records=len(recs), parse_errors=bad,
        keys=dict(collections.Counter(tuple(sorted(r.keys())) for r in recs).most_common()) and
        {"|".join(k): v for k, v in collections.Counter(tuple(sorted(r.keys())) for r in recs).items()},
        content_format="raw text in key 'text'; no token ids, no source / tokenizer / model metadata",
        tokenizer="Qwen/Qwen1.5-MoE-A2.7B snapshot 1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
        tokenization_s=t_tok,
        token_counts=dict(quantiles=q(n), mean=float(n.mean()), total=int(n.sum()),
                          histogram=dict(zip(["[0,64)", "[64,128)", "[128,256)", "[256,512)", "[512,1024)",
                                              "[1024,2048)", "[2048,2049)", ">=2049"],
                                             np.histogram(n, [0, 64, 128, 256, 512, 1024, 2048, 2049, 10 ** 6])[0]
                                             .tolist()))),
        n_records_ge_512_tokens=len(long_ids), frac_records_with_complete_512_window=len(long_ids) / len(recs),
        n_records_ge_2048_tokens=int((n >= 2048).sum()),
        first_128_records=dict(n_ge_512=int((n[:128] >= 512).sum()), median_tokens=float(np.median(n[:128]))),
        n_unique_texts=len(cnt), exact_duplicate_records=len(texts) - len(cnt),
        n_duplicate_groups=sum(1 for v in cnt.values() if v > 1),
        exact_duplicate_first512_windows=len(first512) - len({ED.window_hash(w) for w in first512}),
        record_i_first_token_equals_i=dict(n=sum(1 for i, f in enumerate(first_ids) if f == i),
                                           of=len(first_ids),
                                           first_mismatch=next((i for i, f in enumerate(first_ids) if f != i), None),
                                           note="records 0-93 start with token id i; from record 94 the seed would be "
                                                "a lone byte token (U+FFFD / merged on decode), so re-tokenization "
                                                "cannot recover it"),
        records_starting_with_replacement_char=sum(t.startswith("\ufffd") for t in texts),
        special_strings_in_text=special, special_token_ids_in_tokenization=n_special_tok,
        decode_roundtrip_exact=len(recs) - len(roundtrip), decode_roundtrip_mismatch_records=roundtrip,
        records_with_cjk_chars=cjk,
        text_diagnostics_first512_of_long_records=ED.text_diagnostics(first512),
        near_duplicates_first512_of_long_records=ED.near_duplicate_stats(first512),
        sample_for_format_validation=[dict(record=i, n_tokens=int(n[i]), head=texts[i][:160]) for i in (0, 1, 5, 127, 300)],
        note="statistics describe the artifact only; no record is altered or excluded on subjective quality grounds")
    jdump(audit, p("official_dataset_audit.json"))
    torch.save(dict(tokens=[torch.tensor(x, dtype=torch.long) for x in toks]), p("official_data", "tokens_all.pt"))
    print(f"[inspect] {len(recs)} records; {len(long_ids)} >= 512 tokens; {len(cnt)} unique; "
          f"distinct-1 (first 512) median {audit['text_diagnostics_first512_of_long_records']['distinct1_quantiles']['median']:.3f}")


# ================================================================ stage: compatibility
def _official_config_defaults():
    import ast
    path = os.path.join(SRC, "moequant", "code", "qwen_moe_14b_chat", "configuration_qwen2_moe.py")
    tree = ast.parse(open(path).read())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            args = node.args.args[1:]
            defs = node.args.defaults
            for arg, d in zip(args[len(args) - len(defs):], defs):
                try:
                    out[arg.arg] = ast.literal_eval(d)
                except Exception:
                    pass
    return out, sha256_file(path)


def stage_compatibility(a):
    from transformers import AutoConfig
    tok = tokenizer()
    cfg = AutoConfig.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
    toks = torch.load(p("official_data", "tokens_all.pt"))["tokens"]
    max_id = int(max(int(t.max()) for t in toks))
    off, off_sha = _official_config_defaults()
    keys = ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "moe_intermediate_size", "shared_expert_intermediate_size", "num_experts", "num_experts_per_tok",
            "norm_topk_prob", "max_position_embeddings", "rope_theta"]
    arch = {k: dict(ours=getattr(cfg, k, None), official_code_default=off.get(k)) for k in keys}
    for k in arch:
        arch[k]["equal"] = arch[k]["ours"] == arch[k]["official_code_default"]
    audit = json.load(open(p("official_dataset_audit.json")))
    comp = dict(
        our_model=dict(name="Qwen/Qwen1.5-MoE-A2.7B", variant="base", snapshot="1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"),
        technical=dict(
            data_is_text=True, stored_token_ids=False,
            tokenizer_class=type(tok).__name__, tokenizer_len=len(tok), model_vocab_size=cfg.vocab_size,
            max_token_id_in_data=max_id, all_ids_in_model_vocab=max_id < cfg.vocab_size,
            decode_roundtrip_exact=f"{audit['decode_roundtrip_exact']}/{audit['n_records']}",
            chat_template_markers_in_text=audit["special_strings_in_text"],
            architecture_vs_official_qwen_code=arch, official_config_file_sha256=off_sha,
            architecture_family_compatible=all(v["equal"] for k, v in arch.items()
                                               if k not in ("max_position_embeddings", "rope_theta")),
            tokenizer_note="Qwen1.5 base and chat checkpoints share the Qwen2 BPE vocabulary (151,646 tokens + "
                           "padding to 151,936); the record-start pattern (record i starts with token id i) is "
                           "reproduced exactly by our tokenizer, which confirms the vocabulary/id mapping",
            usable_without_model_or_quantizer_change=True),
        scientific_provenance=dict(
            generating_checkpoint="undocumented (base Qwen1.5-MoE-A2.7B, its Chat variant, or another Qwen1.5-MoE "
                                  "build; the artifact carries no metadata and no generation script is released)",
            evidence_for_base=["the paper's Qwen model cites huggingface.co/Qwen/Qwen1.5-MoE-A2.7B (base)",
                               "rebuttal Table R4 attributes a sample matching record 0 to 'Qwen-MoE-14B' (no '-Chat')",
                               "no chat-template markers in any record"],
            evidence_against_or_unclear=["the repo's Qwen model code directory is named qwen_moe_14b_chat",
                                         "the code's supported paths include a chat checkpoint",
                                         "authors describe 128 x 512-token sequences, the file holds 600 records "
                                         "of up to ~2048 tokens (generation settings for this file unknown)",
                                         "the paper's Qwen results always combine EBSS with AGQ and rotation"],
            source_distribution="self-generated by a model (no external corpus); seeding by vocabulary token id",
            classification="UNDOCUMENTED"),
        experiment_label="cross-checkpoint (provenance-undocumented) EBSS calibration TRANSFER evaluation; not an "
                         "exact reproduction of model-matched EBSS",
        gate=dict(technical_compatibility=True, proceed_to_gpu=True))
    assert comp["technical"]["all_ids_in_model_vocab"] and comp["technical"]["architecture_family_compatible"]
    jdump(comp, p("compatibility.json"))
    print(f"[compat] technical OK (max id {max_id} < {cfg.vocab_size}); provenance UNDOCUMENTED")


# ================================================================ stage: freeze_selection
def leakage_reference():
    """Frozen evaluation data: held-out C4 rows (tune 2000-2127 + test 2128-2255) and WikiText windows."""
    cfg = frozen_cfg()
    allt = c4_tokens()
    held = allt[cfg["split"]["tune"][0]:cfg["split"]["test"][1]]
    win, prot = EP.load_windows()
    return held, win


def stage_freeze(a):
    if os.path.exists(p("protocol.json")):
        print("[freeze] protocol already frozen; not re-freezing"); return
    toks = [t.tolist() for t in torch.load(p("official_data", "tokens_all.pt"))["tokens"]]
    held, win = leakage_reference()
    ref = [w.tolist() for w in held] + [w.tolist() for w in win]
    idx = ED.build_ngram_index(ref, ED.LEAK_NGRAM)
    ext = ED.extract_windows(toks, leak_index=idx)
    leak_report = dict(n_eligible_scanned=ext["last_record_used"], skipped_leakage=ext["skipped_leakage"],
                       skipped_duplicate=ext["skipped_duplicate"])
    # verify leakage-free (collision-free re-check) and near-duplicate (13-gram) contacts, reported only
    near13 = ED.build_ngram_index(ref, ED.NEAR_NGRAM)
    contacts = [i for i, w in enumerate(ext["windows"]) if ED.shares_ngram(w, near13, ED.NEAR_NGRAM)]
    verified = {i: dict(heldout_c4=any(ED.exact_ngram_overlap(ext["windows"][i], r.tolist(), ED.NEAR_NGRAM) for r in held),
                        wikitext=any(ED.exact_ngram_overlap(ext["windows"][i], r.tolist(), ED.NEAR_NGRAM) for r in win))
                for i in contacts}
    exact_eq = sum(1 for w in ext["windows"] for r in ref if w == r)
    if ext["status"] != "OK":
        jdump(dict(status="DATA-BLOCKED", extraction=leak_report), p("ebss_selection_manifest.json"))
        raise SystemExit("DATA-BLOCKED: fewer than 128 valid windows")
    W = torch.tensor(ext["windows"], dtype=torch.long)
    torch.save(W, p("official_data", "ebss_windows.pt"))
    raw_sha = sha256_file(raw_path())
    comp = json.load(open(p("compatibility.json")))
    man = dict(
        source=dict(repository=REPO_URL, commit=COMMIT, path=REL_PATH, raw_sha256=raw_sha),
        model="Qwen/Qwen1.5-MoE-A2.7B", model_snapshot="1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
        tokenizer_snapshot="1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
        extraction_policy=ED.EXTRACTION_POLICY, status="OK", B=ED.B, seqlen=ED.SEQLEN,
        total_valid_tokens=int(W.numel()),
        record_ids=[r for r, _ in ext["selected"]], window_offsets=[o for _, o in ext["selected"]],
        n_independent_parent_records=len({r for r, _ in ext["selected"]}),
        records_scanned=ext["last_record_used"] + 1, n_short_records_in_scanned_prefix=ext["n_short"],
        skipped_duplicate_records=ext["skipped_duplicate"], skipped_leakage_records=ext["skipped_leakage"],
        window_sha256=[ED.window_hash(w) for w in ext["windows"]], windows_tensor_sha256=tensor_sha(W),
        ordered_set_sha256=ED.ids_hash(ext["windows"]),
        duplicates=dict(exact_duplicate_windows_in_set=len(ext["windows"]) - len(set(map(ED.window_hash, ext["windows"])))),
        leakage=dict(exact_window_equal_to_eval_data=exact_eq,
                     exclusion_ngram=ED.LEAK_NGRAM, windows_excluded_for_leakage=len(ext["skipped_leakage"]),
                     near_duplicate_ngram=ED.NEAR_NGRAM, windows_with_13gram_contact=len(contacts),
                     verified_13gram_contacts=verified,
                     reference="held-out C4 rows 2000-2255 (tune + test) and the 128 frozen WikiText-2 windows"),
        provenance_classification=comp["scientific_provenance"]["classification"],
        experiment_label=comp["experiment_label"],
        official_loader_difference="the official loader would take the first 128 lines (only "
                                   f"{json.load(open(p('official_dataset_audit.json')))['first_128_records']['n_ge_512']} "
                                   "of them have 512 tokens) and zero-pad / truncate them to 2048 tokens; B = 128 x 512 "
                                   "without padding cannot be met that way, so the frozen policy keeps record "
                                   "boundaries and takes the first 128 eligible records in file order",
        windows_file=os.path.relpath(p("official_data", "ebss_windows.pt"), ROOT),
        frozen_before_any_quality_result=True, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(man, p("ebss_selection_manifest.json"))

    sel = selections()
    cfg = frozen_cfg()
    base = dict(source="runs_partial_ppl/selected_set_manifest.json (re-verified against each source file)",
                configs={k: dict(v, chosen=v["chosen"]) for k, v in sel.items()},
                historical_context={k: dict(
                    recon_partial_ppl=json.load(open(os.path.join(ROOT, EP.OUT, f"ppl_{k}.json")))["recon_recomputed"],
                    ppl_partial_ppl=json.load(open(os.path.join(ROOT, EP.OUT, f"ppl_{k}.json")))["ppl"],
                    extraction="historical ORIGINAL policy (batches of 8 consecutive original rows)")
                    for k in sel},
                reuse="NOT reused for the primary comparison: historical quantizations used the ORIGINAL batch-of-8 "
                      "extraction, which EBSS cannot share. All 9 are re-quantized under CANONICAL-B1. Historical "
                      "values are context only, never pooled.")
    jdump(base, p("baseline_manifest.json"))

    union = sorted(set().union(*[set(v["chosen"]) for v in sel.values()]))
    rows = dict(c4_rows=union, n_c4=len(union), n_ebss=ED.B, n_total=len(union) + ED.B,
                row_semantics="cache row i < n_c4: C4 original row id c4_rows[i]; row n_c4 + j: EBSS window j "
                              "(manifest order)")
    policy = dict(
        name="CANONICAL-B1",
        description="every calibration sequence is forwarded ALONE (batch size 1) through decoder layers 0..23 "
                    "with one layer on the GPU at a time; layer inputs of 1/8/16/23 are stored per sequence",
        batch_size=1, padding="none (all windows are exactly 512 real tokens; no attention mask beyond causal)",
        ordering="C4 union rows ascending by original id, then the 128 EBSS windows in manifest order; at batch "
                 "size 1 the order cannot affect any activation (checked by capture_repeat in a fresh process, "
                 "reversed order)",
        precision="BF16 weights and activations; FP32 router softmax", attention="eager",
        layer_kwargs="captured once at batch 1 (moe_calib.model_utils.layer_kwargs) - same as all prior studies",
        model_revision="Qwen/Qwen1.5-MoE-A2.7B @ 1a758c50ecb6350748b9ce0a99d2352fd9fc11c9, all parameters checked "
                       "against the full-depth pilot pristine digests",
        kernel_settings=env_record(), source_independent=True, filler_free=True,
        applies_to="calibration activations of all 10 configurations (FR/TR/FC s0-2 and EBSS)",
        unchanged=dict(reconstruction_activations="~/moe_cache test rows 2128-2255 (historical ORIGINAL cache), "
                                                  "identical for every configuration",
                       ppl_windows="runs_partial_ppl/windows.pt, EP.layerwise_nll batch 8"),
        historical_policy="cache_inputs.py: batches of 8 consecutive original rows (ORIGINAL); its results are "
                          "context only",
        routing_definition="the model's op on the batch-1 layer input: softmax(router logits, float) -> top-4 "
                           "(GPU), counted per (sequence, layer, expert) = E3 Profiler routed-token counts",
        rows=rows)
    jdump(policy, p("execution_policy.json"))

    win, prot = EP.load_windows()
    proto = dict(
        experiment="Official MoEQuant EBSS calibration-quality validation (cross-checkpoint transfer, provenance "
                   "undocumented)",
        central_question="Does the released EBSS_qwen.jsonl yield lower held-out expert reconstruction error and "
                         "partial-model PPL than C4 Full Random and Typical Random under the identical GPTQ pipeline?",
        model="Qwen/Qwen1.5-MoE-A2.7B", gptq=cfg["gptq"], layers=LAYERS, B=128, seqlen=512,
        n_experts=240, n_projections=720, projections=["gate_proj", "up_proj", "down_proj"],
        calibration="non-sequential; Hessians from CANONICAL-B1 layer inputs via the unchanged E3 Profiler (K = 4)",
        configs=CONFIGS, quantization_order=QUANT_ORDER, repeat_config=REPEAT_CONFIG,
        reconstruction="E3 per-expert held-out relative reconstruction on ~/moe_cache rows 2128-2255 "
                       "(eval_cache_dir); config score = mean over the 240 experts (unchanged aggregation)",
        exposure_deciles="deciles of each expert's natural routing frequency p (runs_low_budget/routing_profile_v2.npz, "
                         "C4 pool), fixed and method-independent",
        exposure_thresholds=dict(absolute=ED.EXPOSURE_ABS_THRESHOLDS, relative_to_uniform=ED.EXPOSURE_REL_THRESHOLDS),
        typicality="existing d(s) (moe_calib.typicality) against the C4-pool reference of routing_profile_v2; atypical "
                   "= d(s) > d*, d* = largest d in the frozen typical half (pool_typical_half.json)",
        ppl=dict(windows_sha256=prot["windows_sha256"], n_windows=128, targets=65408, bf16_reference=BF16_REF),
        decision_rule=ED.DECISION_RULE, tau_relative_gain_pct=ED.TAU_REL_GAIN,
        provenance_classification=comp["scientific_provenance"]["classification"],
        extraction_policy_ref="execution_policy.json", ebss_manifest_sha256=sha256_file(p("ebss_selection_manifest.json")),
        baseline_manifest_sha256=sha256_file(p("baseline_manifest.json")),
        execution_policy_sha256=sha256_file(p("execution_policy.json")),
        code_sha256=code_hashes(),
        hard_constraints="no GPTQ / solver / bit / group / damping / checkpoint change; no AGQ, rotation or new "
                         "allocator; no EBSS regeneration or C4 mixing; no tuning on held-out results; one EBSS set; "
                         "no numerical-sensitivity sweep",
        frozen_before_any_new_quality_result=True, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    jdump(proto, p("protocol.json"))
    print(f"[freeze] EBSS: {len(ext['selected'])} windows from records 0..{ext['last_record_used']}; skipped dup "
          f"{len(ext['skipped_duplicate'])} leak {len(ext['skipped_leakage'])}; 13-gram contacts {len(contacts)}; "
          f"C4 union {len(union)} rows")


# ================================================================ model helpers
def load_model():
    model, t = FD.load_model_full()
    return model, t


def verify_pristine_full(model):
    from moe_calib import partial_model_quant as PQ
    pr = json.load(open(os.path.join(ROOT, FD.OUT, "pristine_digests.json")))
    layers = list(range(N_LAYERS))
    return (PQ.target_tensor_digests(model, layers) == pr["pristine_targets"] and
            PQ.untouched_digests(model, layers) == pr["pristine_untouched"])


def cache_tokens():
    pol = json.load(open(p("execution_policy.json")))
    allt = c4_tokens()
    W = torch.load(p("official_data", "ebss_windows.pt"))
    return torch.cat([allt[pol["rows"]["c4_rows"]], W]), pol


@torch.no_grad()
def forward_b1(model, tokens, order, store_layers, route_layers, want_nll):
    """CANONICAL-B1: each sequence alone. Returns {L: [n, T, d] bf16} in `order` positions, routing counts
    [n, len(route_layers) * E], per-row NLL (511 targets) and per-row compute seconds."""
    from moe_calib import model_utils as MU
    n, T = len(order), tokens.shape[1]
    E, k = MU.num_experts(model), MU.top_k(model)
    hs = [MU.embed(model, tokens[i:i + 1]).cpu() for i in order]
    stored = {L: torch.empty(n, T, hs[0].shape[-1], dtype=hs[0].dtype) for L in store_layers}
    C = torch.zeros(n, len(route_layers) * E, dtype=torch.long)
    sec = np.zeros(n)
    t_transfer = 0.0
    last = N_LAYERS - 1 if want_nll else max(list(store_layers) + list(route_layers))
    for L in range(last + 1):
        if L in stored:
            for j in range(n):
                stored[L][j] = hs[j][0]
        if L == last and not want_nll and L not in route_layers:
            break
        tt = time.time()
        with MU.layer_on_gpu(model, L):
            torch.cuda.synchronize()
            t_transfer += time.time() - tt
            logits = []
            hook = (MU.moe_block(model, L).gate.register_forward_hook(lambda m, i, o: logits.append(o.detach()))
                    if L in route_layers else None)
            try:
                for j in range(n):
                    t0 = time.time()
                    hs[j] = MU.run_layer(model, L, hs[j].to(MU.GPU)).cpu()
                    torch.cuda.synchronize()
                    sec[j] += time.time() - t0
                    if hook is not None:
                        lg = logits.pop().reshape(T, E)
                        sel = torch.topk(torch.softmax(lg, dim=-1, dtype=torch.float), k, dim=-1).indices
                        li = route_layers.index(L)
                        C[j, li * E:(li + 1) * E] = torch.bincount(sel.flatten(), minlength=E).cpu()
            finally:
                if hook is not None:
                    hook.remove()
    nll = np.full(n, np.nan)
    if want_nll:
        for j in range(n):
            logits = model.lm_head(model.model.norm(hs[j].to(MU.GPU)))
            s, c = EP.nll_from_logits(logits, tokens[order[j]:order[j] + 1])
            nll[j] = float(s[0]) / int(c[0])
    return stored, C, nll, sec, t_transfer


# ================================================================ stage: baseline_regression
def quant_config(model, name, cdir, chosen, ecache, K, tag=""):
    """One configuration: quantize + install the 240 experts, integrity checks, reconstruction (E3), PPL x2."""
    from moe_calib import model_utils as MU
    from moe_calib import partial_model_quant as PQ
    from moe_calib.data_utils import load_layer_cache
    cfg = frozen_cfg()
    test_ids = list(range(*cfg["split"]["test"]))
    win, prot = EP.load_windows()
    pr = json.load(open(os.path.join(ROOT, FD.OUT, "pristine_digests.json")))
    tg0 = PQ.target_tensor_digests(model, LAYERS)
    assert tg0 == {k: pr["pristine_targets"][k] for k in tg0}, "targets not pristine"
    u0 = PQ.untouched_digests(model, LAYERS)
    router_names = [n for n in u0 if any(f"layers.{L}.mlp.gate." in n for L in LAYERS)]
    orig = {(L, e): {kk: w.detach().cpu().clone() for kk, w in MU.get_expert_weights(model, L, e).items()}
            for L, e in EP.CHECK_EXPERTS}
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    g = cfg["gptq"]
    q = PQ.quantize_install(model, cdir, LAYERS, chosen, test_ids, K, g["bits"], g["group_size"], g["percdamp"],
                            capture=EP.CHECK_EXPERTS, eval_cache_dir=ecache)
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
    u1 = PQ.untouched_digests(model, LAYERS)
    untouched_ok = u1 == u0
    routers_ok = all(u1[n] == u0[n] for n in router_names)
    dig = PQ.installed_digest(model, LAYERS)
    wd = os.path.join(ECACHE, "weights", name + tag)
    os.makedirs(wd, exist_ok=True)
    files = {}
    for L in LAYERS:
        path = os.path.join(wd, f"L{L}.pt")
        PQ.save_layer(model, L, path)
        files[str(L)] = dict(sha256=sha256_file(path), layer_sha256=PQ.layer_digest(model, L))
    evals = []
    for _ in range(2):
        t2 = time.time()
        nll, cnt = EP.layerwise_nll(model, win)
        evals.append(EP.ppl_record(nll, cnt, time.time() - t2))
    rec = evals[0]
    out = dict(name=name + tag, calibration_cache=cdir, reconstruction_cache=ecache or cdir, gptq=g, layers=LAYERS,
               n_quantized_experts=q["counts"]["n_experts_quantized_total"], n_projections=3 * q["counts"]["n_experts_quantized_total"],
               rtn_fallbacks=q["counts"]["n_zero_token_quantization_fallback"],
               n_experts_scored=q["counts"]["n_experts_test_observed"],
               recon=float(np.mean([r["mean"] for r in q["results"]])),
               per_expert=[dict(layer=r["layer"], expert=r["expert"], tokens=r["tokens"], mean=r["mean"],
                                p90=r["p90"], p99=r["p99"], fallback=r["fallback"]) for r in q["results"]],
               projections_unchanged_from_bf16=len(unchanged), nonfinite_weights=len(nonfin),
               untouched_params_unchanged=untouched_ok, n_untouched_params=len(u0),
               router_params_checked=len(router_names), routers_unchanged=routers_ok, expert_checks=checks,
               installed_weights_sha256=dig, weight_files=files, weights_dir=wd,
               repeat_eval_abs_diff_ppl=abs(evals[0]["ppl"] - evals[1]["ppl"]),
               quantize_total_s=t_q, hessian_s=q["hessian_s"], gptq_s=q["gptq_s"],
               peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=FD.peak_cpu_gb(), **rec)
    out["valid"] = bool(out["n_quantized_experts"] == 240 and all(ck["pass"] for ck in checks) and not unchanged
                        and not nonfin and untouched_ok and routers_ok and out["repeat_eval_abs_diff_ppl"] == 0
                        and bool(np.isfinite(rec["per_window_nll"]).all()) and out["n_experts_scored"] == 240)
    return out


def stage_baseline_regression(a):
    out_path = p("logs", "baseline_regression.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    model, t_load = load_model()
    if "model" not in res:
        res["model"] = dict(pristine_all_parameters=verify_pristine_full(model), load_s=t_load, env=env_record())
        assert res["model"]["pristine_all_parameters"]
        jdump(res, out_path)
    win, prot = EP.load_windows()
    # 1. BF16 PPL under the current environment
    if "bf16" not in res:
        t0 = time.time()
        nll, cnt = EP.layerwise_nll(model, win)
        r = EP.ppl_record(nll, cnt, time.time() - t0)
        res["bf16"] = dict(ppl=r["ppl"], total_tokens=r["total_tokens"], eval_s=r["eval_s"],
                           historical=json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))["ppl"],
                           per_window_nll=r["per_window_nll"])
        res["bf16"]["exact_match_historical"] = res["bf16"]["ppl"] == res["bf16"]["historical"]
        jdump(res, out_path)
        print(f"[regression] BF16 PPL {r['ppl']:.6f} (historical {res['bf16']['historical']:.6f})", flush=True)
    # 2. extraction-implementation regression: forward_b1-style code with the ORIGINAL batch-8 plan for two
    #    original batches must reproduce ~/moe_cache rows bitwise (validates the forward/storage path)
    if "extraction_impl" not in res:
        from moe_calib import model_utils as MU
        allt = c4_tokens()
        rows = list(range(0, 16))
        hs = [MU.embed(model, allt[i:i + 8]).cpu() for i in (0, 8)]
        ok = {}
        with torch.no_grad():
            for L in range(max(LAYERS) + 1):
                if L in LAYERS:
                    X = torch.cat(hs)
                    R = torch.load(os.path.join(OCACHE, f"L{L}.pt"), mmap=True)[rows]
                    ok[str(L)] = bool(torch.equal(X, R))
                if L == max(LAYERS):
                    break
                with MU.layer_on_gpu(model, L):
                    hs = [MU.run_layer(model, L, h.to(MU.GPU)).cpu() for h in hs]
        res["extraction_impl"] = dict(rows=rows, plan="ORIGINAL batches [0..7], [8..15]", bitwise_equal=ok,
                                      passed=all(ok.values()))
        jdump(res, out_path)
        print(f"[regression] extraction implementation reproduces ~/moe_cache: {res['extraction_impl']}", flush=True)
    # 3. quantization-pipeline regression: FR s0 on the historical cache reproduces partial-PPL exactly
    if "quant_pipeline" not in res:
        sel = selections()["FR_s0"]
        r = quant_config(model, "FR_s0_historical_cache", OCACHE, sel["chosen"], None, sel["K"])
        ref = json.load(open(os.path.join(ROOT, EP.OUT, "ppl_FR_s0.json")))
        res["quant_pipeline"] = dict(
            config="FR_s0 on ~/moe_cache (historical ORIGINAL extraction)", ppl=r["ppl"], ppl_reference=ref["ppl"],
            ppl_exact=r["ppl"] == ref["ppl"], sha256=r["installed_weights_sha256"],
            sha256_reference=ref["installed_weights_sha256"],
            weights_exact=r["installed_weights_sha256"] == ref["installed_weights_sha256"], recon=r["recon"],
            recon_reference=ref["recon_recomputed"], recon_exact=r["recon"] == ref["recon_recomputed"],
            per_window_nll_exact=r["per_window_nll"] == ref["per_window_nll"], valid=r["valid"],
            quantize_total_s=r["quantize_total_s"])
        res["quant_pipeline"]["passed"] = all(res["quant_pipeline"][k] for k in
                                              ("ppl_exact", "weights_exact", "recon_exact", "per_window_nll_exact", "valid"))
        jdump(r, p("logs", "regression_FR_s0_historical_cache.json"))
        jdump(res, out_path)
    res["passed"] = bool(res["model"]["pristine_all_parameters"] and res["extraction_impl"]["passed"]
                         and res["quant_pipeline"]["passed"] and res["bf16"]["ppl"] == BF16_REF_EXACT())
    jdump(res, out_path)
    print(f"[regression] passed={res['passed']} (quant pipeline {res['quant_pipeline']['passed']})")
    if not res["passed"]:
        raise SystemExit("baseline regression FAILED")


def BF16_REF_EXACT():
    return json.load(open(os.path.join(ROOT, EP.OUT, "bf16_baseline.json")))["ppl"]


# ================================================================ stage: capture
def stage_capture(a):
    man_path = p("logs", "capture_manifest.json")
    if os.path.exists(man_path) and json.load(open(man_path)).get("complete"):
        print("[capture] done"); return
    reg = json.load(open(p("logs", "baseline_regression.json")))
    assert reg.get("passed"), "baseline regression has not passed"
    tokens, pol = cache_tokens()
    N = tokens.shape[0]
    model, t_load = load_model()
    assert verify_pristine_full(model)
    os.makedirs(ECACHE, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    stored, C, nll, sec, t_tr = forward_b1(model, tokens, list(range(N)), LAYERS, LAYERS, want_nll=True)
    t_total = time.time() - t0
    torch.save(dict(c4_rows=pol["rows"]["c4_rows"], tokens=tokens), os.path.join(ECACHE, "ids.pt"))
    files = {}
    for L in LAYERS:
        path = os.path.join(ECACHE, f"L{L}.pt")
        torch.save(stored[L], path + ".tmp"); os.replace(path + ".tmp", path)
        X = stored[L]
        files[str(L)] = dict(shape=list(X.shape), sha256=sha256_file(path), tensor_sha256=tensor_sha(X),
                             nonfinite=int((~torch.isfinite(X.float())).sum()))
    np.savez(p("logs", "capture_routing_nll.npz"), C=C.numpy(), nll=nll, sec=sec)
    meta = dict(pool_size=0, holdout_size=0, seqlen=512, layers=LAYERS, policy="CANONICAL-B1", n_rows=N)
    json.dump(meta, open(os.path.join(ECACHE, "meta.json"), "w"), indent=1)
    man = dict(cache_dir=ECACHE, n_rows=N, n_c4=pol["rows"]["n_c4"], n_ebss=pol["rows"]["n_ebss"],
               token_sha256=tensor_sha(tokens), files=files, seconds_total=t_total,
               layer_transfer_s=t_tr, per_row_compute_s_mean=float(sec.mean()),
               per_row_compute_s_c4_mean=float(sec[:pol["rows"]["n_c4"]].mean()),
               per_row_compute_s_ebss_mean=float(sec[pol["rows"]["n_c4"]:].mean()),
               layers_forwarded="0..23 (23 and lm_head only for the calibration-NLL diagnostic)",
               peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_cpu_gb=FD.peak_cpu_gb(),
               model_load_s=t_load, routing_and_nll="logs/capture_routing_nll.npz",
               bytes_on_disk=sum(os.path.getsize(os.path.join(ECACHE, f"L{L}.pt")) for L in LAYERS), complete=True)
    jdump(man, man_path)
    print(f"[capture] {N} rows at batch 1 in {t_total:.0f} s ({sec.mean() * 1e3:.1f} ms/row compute)")


def stage_capture_repeat(a):
    """Repeatability + order-independence control: a fresh process re-extracts 8 C4 + 8 EBSS rows in
    REVERSED order at batch 1; inputs of layers 1/8/16/23 and routing counts must equal the cache bitwise."""
    out = p("logs", "capture_repeat.json")
    if os.path.exists(out):
        print("[capture_repeat] done"); return
    tokens, pol = cache_tokens()
    nc = pol["rows"]["n_c4"]
    rows = list(range(N_REPEAT_ROWS)) + list(range(nc, nc + N_REPEAT_ROWS))
    order = rows[::-1]
    model, _ = load_model()
    stored, C, nll, sec, _ = forward_b1(model, tokens, order, LAYERS, LAYERS, want_nll=False)
    z = np.load(p("logs", "capture_routing_nll.npz"))
    res = {}
    for L in LAYERS:
        R = torch.load(os.path.join(ECACHE, f"L{L}.pt"), mmap=True)[order]
        res[str(L)] = bool(torch.equal(stored[L], R))
    routing_eq = bool((C.numpy() == z["C"][order]).all())
    rec = dict(rows=order, bitwise_equal=res, routing_equal=routing_eq, passed=all(res.values()) and routing_eq)
    jdump(rec, out)
    print(f"[capture_repeat] passed={rec['passed']}")
    if not rec["passed"]:
        raise SystemExit("capture repeatability FAILED")


# ================================================================ stage: quantize
def config_rows(name):
    pol = json.load(open(p("execution_policy.json")))
    pos = {r: i for i, r in enumerate(pol["rows"]["c4_rows"])}
    if name == "EBSS":
        return [pol["rows"]["n_c4"] + j for j in range(ED.B)], 4
    sel = selections()[name]
    return [pos[r] for r in sel["chosen"]], sel["K"]


def stage_quantize(a):
    name = a.config
    assert name in CONFIGS
    tag = "_repeat" if a.repeat else ""
    os.makedirs(p("results"), exist_ok=True)
    if os.path.exists(p("results", f"{name}{tag}.json")):
        print(f"[quant] {name}{tag} done"); return
    assert json.load(open(p("logs", "baseline_regression.json")))["passed"]
    assert json.load(open(p("logs", "capture_repeat.json")))["passed"]
    cm = json.load(open(p("logs", "capture_manifest.json")))
    for L in LAYERS:
        assert sha256_file(os.path.join(ECACHE, f"L{L}.pt")) == cm["files"][str(L)]["sha256"], "cache changed"
    chosen, K = config_rows(name)
    assert len(chosen) == 128 == len(set(chosen))
    model, t_load = load_model()
    assert verify_pristine_full(model)
    r = quant_config(model, name, ECACHE, chosen, OCACHE, K, tag)
    r.update(config=name, policy="CANONICAL-B1", cache_rows=chosen, model_load_s=t_load)
    jdump(r, p("results", f"{name}{tag}.json"))
    print(f"[quant] {name}{tag}: PPL {r['ppl']:.6f} recon {r['recon']:.7f} fallbacks {r['rtn_fallbacks']} "
          f"valid {r['valid']} ({r['quantize_total_s']:.0f} s)", flush=True)
    if not r["valid"]:
        raise SystemExit(f"{name} failed its correctness checks")


# ================================================================ aggregation stages
def stage_reconstruction(a):
    out = {}
    for c in CONFIGS:
        r = json.load(open(p("results", f"{c}.json")))
        out[c] = dict(recon=r["recon"], per_expert=r["per_expert"], rtn_fallbacks=r["rtn_fallbacks"],
                      n_experts_scored=r["n_experts_scored"], valid=r["valid"])
    jdump(dict(metric="E3 held-out relative expert reconstruction (mean over 240 experts); lower is better",
               evaluation_rows="~/moe_cache 2128-2255", configs=out), p("logs", "reconstruction_raw.json"))
    print("[reconstruction] aggregated", {c: round(v["recon"], 7) for c, v in out.items()})


def stage_ppl(a):
    out = {}
    for c in CONFIGS:
        r = json.load(open(p("results", f"{c}.json")))
        out[c] = dict(ppl=r["ppl"], total_nll=r["total_nll"], total_tokens=r["total_tokens"],
                      per_window_nll=r["per_window_nll"], repeat_eval_abs_diff_ppl=r["repeat_eval_abs_diff_ppl"])
    jdump(out, p("logs", "ppl_raw.json"))
    print("[ppl] aggregated", {c: round(v["ppl"], 6) for c, v in out.items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--config")
    ap.add_argument("--repeat", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    fn = dict(preflight=stage_preflight, download=stage_download, inspect_data=stage_inspect,
              compatibility=stage_compatibility, freeze_selection=stage_freeze,
              baseline_regression=stage_baseline_regression, capture=stage_capture,
              capture_repeat=stage_capture_repeat, quantize=stage_quantize, reconstruction=stage_reconstruction,
              ppl=stage_ppl)
    if a.stage == "report":
        import report_official_ebss as R
        return R.main()
    fn[a.stage](a)


if __name__ == "__main__":
    main()
