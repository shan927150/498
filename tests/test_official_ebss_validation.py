"""Official EBSS calibration-quality validation: helpers (CPU, synthetic) and frozen artifacts.

Artifact tests skip when the corresponding output has not been produced yet."""
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
from moe_calib import ebss_official_data as ED   # noqa: E402

OUT = os.path.join(ROOT, "runs_ebss_official_validation")


def _j(*a):
    path = os.path.join(OUT, *a)
    if not os.path.exists(path):
        pytest.skip(f"{path} not produced yet")
    return json.load(open(path))


def _drv():
    import official_ebss_validation as DRV
    return DRV


# 1. official dataset parsing
def test_parse_jsonl_strict():
    raw = b'{"text": "a b"}\n{"text": "c"}\n'
    recs, bad = ED.parse_jsonl(raw)
    assert [r["text"] for r in recs] == ["a b", "c"] and bad == []
    recs, bad = ED.parse_jsonl(b'{"text": 1}\nnot json\n\n{"text": "ok"}\n')
    assert len(recs) == 1 and len(bad) == 3


def test_official_file_parses_completely():
    a = _j("official_dataset_audit.json")
    assert a["n_records"] == a["n_lines"] == 600 and a["parse_errors"] == []
    assert a["keys"] == {"text": 600}


# 2. dataset integrity hashes
def test_dataset_integrity_hash():
    prov = _j("source_provenance.json")
    raw = open(os.path.join(OUT, "official_data", "EBSS_qwen.jsonl"), "rb").read()
    assert ED.sha256_bytes(raw) == prov["sha256"] == \
        "691c7a50618afbec7217168349835fe005eae869a7c5d98321b375c288253de9"
    assert prov["git_blob_match"] and prov["download_equals_clone"] and prov["commit"].startswith("d77ba1b7")


# 3. tokenizer handling
def test_tokenizer_handling_matches_manifest():
    pytest.importorskip("transformers")
    m = _j("ebss_selection_manifest.json")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", local_files_only=True)
    except Exception:
        pytest.skip("tokenizer not cached")
    recs, _ = ED.parse_jsonl(open(os.path.join(OUT, "official_data", "EBSS_qwen.jsonl"), "rb").read())
    W = torch.load(os.path.join(OUT, "official_data", "ebss_windows.pt"))
    for j in (0, 64, 127):
        rid = m["record_ids"][j]
        ids = tok(recs[rid]["text"], add_special_tokens=False).input_ids
        assert ids[:512] == W[j].tolist()
    assert tok("x", add_special_tokens=True).input_ids == tok("x", add_special_tokens=False).input_ids


# 4. model compatibility checks
def test_model_compatibility_record():
    c = _j("compatibility.json")
    t = c["technical"]
    assert t["all_ids_in_model_vocab"] and t["max_token_id_in_data"] < t["model_vocab_size"]
    for k in ("hidden_size", "num_hidden_layers", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
              "vocab_size"):
        assert t["architecture_vs_official_qwen_code"][k]["equal"]


# 5-7. exact windows, exact budget, deterministic extraction
def _synthetic_tokens(seed=0):
    g = np.random.default_rng(seed)
    out = []
    for i in range(300):
        n = int(g.choice([100, 600, 2048]))
        out.append(g.integers(0, 1000, n).tolist())
    return out


def test_exact_512_windows_and_budget():
    ext = ED.extract_windows(_synthetic_tokens())
    assert ext["status"] == "OK" and len(ext["windows"]) == 128
    assert all(len(w) == 512 for w in ext["windows"])
    assert all(o == 0 for _, o in ext["selected"])


def test_extraction_deterministic_and_file_ordered():
    a, b = ED.extract_windows(_synthetic_tokens()), ED.extract_windows(_synthetic_tokens())
    assert a["selected"] == b["selected"] and a["windows"] == b["windows"]
    rids = [r for r, _ in a["selected"]]
    assert rids == sorted(rids) and len(set(rids)) == 128


def test_frozen_manifest_budget():
    m = _j("ebss_selection_manifest.json")
    W = torch.load(os.path.join(OUT, "official_data", "ebss_windows.pt"))
    assert tuple(W.shape) == (128, 512) and m["total_valid_tokens"] == 65536
    assert m["n_independent_parent_records"] == 128 == len(set(m["record_ids"]))
    assert [ED.window_hash(w.tolist()) for w in W] == m["window_sha256"]


# 8. duplicate detection
def test_duplicate_windows_skipped():
    toks = _synthetic_tokens()
    first_long = next(i for i, t in enumerate(toks) if len(t) >= 512)
    toks.insert(first_long + 1, list(toks[first_long][:600]))
    ext = ED.extract_windows(toks)
    assert first_long + 1 in ext["skipped_duplicate"]
    assert len({ED.window_hash(w) for w in ext["windows"]}) == 128


# 9. evaluation leakage detection
def test_leakage_detection_excludes_window():
    toks = _synthetic_tokens()
    target = next(i for i, t in enumerate(toks) if len(t) >= 512)
    ref = [list(range(5000, 5040)) + toks[target][100:140]]
    idx = ED.build_ngram_index(ref, ED.LEAK_NGRAM)
    ext = ED.extract_windows(toks, leak_index=idx)
    assert target in ext["skipped_leakage"]
    assert ED.exact_ngram_overlap(toks[target][:512], ref[0], ED.LEAK_NGRAM)
    assert not ED.exact_ngram_overlap(list(range(40)), list(range(100, 140)), 13)


# 10. calibration sample provenance
def test_selection_provenance_recorded():
    m = _j("ebss_selection_manifest.json")
    assert m["source"]["raw_sha256"] == _j("source_provenance.json")["sha256"]
    assert m["window_offsets"] == [0] * 128 and m["record_ids"] == sorted(m["record_ids"])
    assert m["leakage"]["exact_window_equal_to_eval_data"] == 0


# 11. FR/TR/FC baseline manifests
def test_baseline_manifests():
    b = _j("baseline_manifest.json")
    assert sorted(b["configs"]) == sorted(f"{c}_s{s}" for c in ("FR", "TR", "FC") for s in (0, 1, 2))
    for k, v in b["configs"].items():
        assert len(v["chosen"]) == 128 == len(set(v["chosen"])) and all(0 <= x < 2000 for x in v["chosen"])
        assert v["strategy"] == k.split("_")[0]


# 12. unknown source checkpoint handling
def test_unknown_checkpoint_is_transfer_limited():
    q = dict(quality_class="OFFICIAL-EBSS-QUALITY-GAIN")
    assert ED.decide(True, True, "UNDOCUMENTED", q) == "OFFICIAL-EBSS-TRANSFER-LIMITED"
    assert ED.decide(True, True, "DIFFERENT-CHECKPOINT", q) == "OFFICIAL-EBSS-TRANSFER-LIMITED"
    assert ED.decide(True, True, "MATCHED-DOCUMENTED", q) == "OFFICIAL-EBSS-QUALITY-GAIN"
    assert _j("compatibility.json")["scientific_provenance"]["classification"] in ED.PROVENANCE_CLASSES


# 13. insufficient EBSS data
def test_insufficient_data_is_blocked():
    toks = [[1] * 600 for _ in range(10)] + [[2] * 100 for _ in range(200)]
    ext = ED.extract_windows(toks)
    assert ext["status"] == "DATA-BLOCKED" and len(ext["windows"]) == 1      # 10 identical windows -> 1
    assert ED.decide(False, True, "MATCHED-DOCUMENTED", {}) == "DATA-BLOCKED"


# 14. frozen calibration execution policy
def test_execution_policy_frozen():
    pol = _j("execution_policy.json")
    proto = _j("protocol.json")
    assert pol["name"] == "CANONICAL-B1" and pol["batch_size"] == 1
    assert pol["rows"]["n_ebss"] == 128 and pol["rows"]["n_total"] == pol["rows"]["n_c4"] + 128
    import hashlib
    assert proto["execution_policy_sha256"] == hashlib.sha256(
        open(os.path.join(OUT, "execution_policy.json"), "rb").read()).hexdigest()


def test_forward_b1_is_batch_independent_on_tiny_model(monkeypatch):
    from moe_calib import model_utils as MU
    DRV = _drv()
    monkeypatch.setattr(DRV, "N_LAYERS", 3)
    model, _ = MU.load_model()
    tok = torch.randint(0, 64, (5, 16), generator=torch.Generator().manual_seed(0))
    a = DRV.forward_b1(model, tok, [0, 1, 2, 3, 4], [1, 2], [1, 2], want_nll=False)
    b = DRV.forward_b1(model, tok, [4, 3, 2, 1, 0], [1, 2], [1, 2], want_nll=False)
    for L in (1, 2):
        assert torch.equal(a[0][L], b[0][L].flip(0))
    assert torch.equal(a[1], b[1].flip(0))
    assert (a[1].reshape(5, 2, -1).sum(-1) == 16 * 2).all()        # top-2 slots of 16 tokens per layer
    # batch-1 rows equal a stand-alone forward of the same row
    h = MU.embed(model, tok[2:3])
    for L in range(1):
        h = MU.run_layer(model, L, h)
    assert torch.equal(a[0][1][2], h[0])


# 15-16. exact quantized layer scope and counts
def test_layer_scope_and_counts():
    proto = _j("protocol.json")
    assert proto["layers"] == [1, 8, 16, 23] and proto["n_experts"] == 240 and proto["n_projections"] == 720
    assert proto["gptq"] == {"bits": 4, "group_size": 128, "percdamp": 1.0, "act_order": True}
    for c in proto["configs"]:
        path = os.path.join(OUT, "results", f"{c}.json")
        if os.path.exists(path):
            r = json.load(open(path))
            assert r["layers"] == [1, 8, 16, 23] and r["n_quantized_experts"] == 240 and r["n_projections"] == 720
            assert r["n_experts_scored"] == 240


# 17-20. no RTN fallback (C4), finite weights, routers unchanged, effective-weight installation
def _results():
    d = os.path.join(OUT, "results")
    if not os.path.isdir(d) or not os.listdir(d):
        pytest.skip("no quantization results yet")
    return {f[:-5]: json.load(open(os.path.join(d, f))) for f in os.listdir(d) if f.endswith(".json")}


def test_no_rtn_fallback_in_c4_baselines():
    for k, r in _results().items():
        if not k.startswith("EBSS"):
            assert r["rtn_fallbacks"] == 0


def test_no_nan_inf():
    for r in _results().values():
        assert r["nonfinite_weights"] == 0 and np.isfinite(r["per_window_nll"]).all() and np.isfinite(r["recon"])


def test_router_weights_unchanged():
    for r in _results().values():
        assert r["routers_unchanged"] and r["router_params_checked"] == 4 * 1 and r["untouched_params_unchanged"]


def test_effective_weight_installation():
    for r in _results().values():
        assert r["projections_unchanged_from_bf16"] == 0
        for ck in r["expert_checks"]:
            assert ck["installed_equals_bf16_of_e3_wq"] and ck["installed_differs_from_bf16_original"] and ck["pass"]


# 21-22. fixed reconstruction and PPL evaluation
def test_fixed_reconstruction_evaluation():
    for r in _results().values():
        assert r["reconstruction_cache"].endswith("moe_cache")
        assert sorted((e["layer"], e["expert"]) for e in r["per_expert"]) == \
            sorted((L, e) for L in (1, 8, 16, 23) for e in range(60))


def test_fixed_ppl_evaluation():
    proto = _j("protocol.json")
    import eval_partial_ppl as EP
    assert proto["ppl"]["windows_sha256"] == json.load(open(os.path.join(ROOT, EP.OUT, "evaluation_protocol.json")))[
        "windows_sha256"]
    for r in _results().values():
        assert r["total_tokens"] == 65408 and len(r["per_window_nll"]) == 128 and r["repeat_eval_abs_diff_ppl"] == 0


# 23. relative-error calculations
def test_relative_penalty():
    assert ED.relative_penalty(0.99, 1.0) == pytest.approx(-1.0)
    assert ED.relative_penalty(0.0050, 0.0046) == pytest.approx((0.0050 / 0.0046 - 1) * 100)
    q = ED.quality_class(0.98, [1.0, 1.0, 1.0], [0.99, 1.0, 1.0], 9.0, [9.1] * 3, [9.2] * 3)
    assert q["R"] and q["P"] and q["quality_class"] == "OFFICIAL-EBSS-QUALITY-GAIN"
    assert q["mean_rel_penalty_vs_FR"] == pytest.approx(-2.0)


# 24. no artificial EBSS seed replication
def test_single_ebss_set_no_seed_replication():
    proto = _j("protocol.json")
    assert [c for c in proto["configs"] if c.startswith("EBSS")] == ["EBSS"]
    assert "subdivision" in ED.EXTRACTION_POLICY and "one EBSS set" in ED.EXTRACTION_POLICY["subdivision"]
    s = os.path.join(OUT, "summary.json")
    if os.path.exists(s):
        assert json.load(open(s))["ebss"].get("sd") is None


# 25. timing accounting
def test_timing_accounting():
    t = ED.measured_time(1.0, 2.0, 3.0, 4.0)
    assert t["T_measured_s"] == 10.0 and set(t) == {"tokenization_s", "activation_s", "hessian_s", "gptq_s",
                                                     "T_measured_s"}
    with pytest.raises(AssertionError):
        ED.measured_time(None, 1, 1, 1)


# 26. decision classification
def test_decision_classification():
    fr, tr = [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]
    assert ED.quality_class(0.997, fr, tr, 8.0, [9] * 3, [9] * 3)["quality_class"] == "OFFICIAL-EBSS-PPL-GAIN-ONLY"
    assert ED.quality_class(0.99, fr, tr, 10.0, [9] * 3, [9] * 3)["quality_class"] == \
        "OFFICIAL-EBSS-RECONSTRUCTION-GAIN-ONLY"
    assert ED.quality_class(1.01, fr, tr, 10.0, [9] * 3, [9] * 3)["quality_class"] == "OFFICIAL-EBSS-NO-CONSISTENT-GAIN"
    # lower than all but one seed -> no R
    assert not ED.quality_class(0.99, [1.0, 1.0, 0.98], tr, 10.0, [9] * 3, [9] * 3)["R"]
    assert ED.decide(True, False, "MATCHED-DOCUMENTED", {}) == "RUN-BLOCKED"


def test_exposure_stats():
    s = ED.exposure_stats([0, 10, 100, 1000], uniform_mean=100)
    assert s["n_zero"] == 1 and s["min"] == 0 and s["n_below_32"] == 2 and s["n_below_0.1x_uniform"] == 1
    assert ED.gini([5, 5, 5]) == pytest.approx(0.0) and ED.gini([0, 0, 0, 10]) == pytest.approx(0.75)
    assert ED.routing_l1([1, 1], [0.5, 0.5]) == 0.0


def test_text_diagnostics():
    d = ED.text_diagnostics([[1] * 512, list(range(512))])
    assert d["distinct1_quantiles"]["min"] == pytest.approx(1 / 512) and d["distinct1_quantiles"]["max"] == 1.0
    assert d["longest_identical_token_run"]["max"] == 512 and d["exact_duplicate_windows"] == 0


# 27. backward compatibility with previous experiments
def test_previous_artifacts_untouched():
    import hashlib
    man = json.load(open(os.path.join(ROOT, "runs_partial_ppl", "selected_set_manifest.json")))
    for k, v in man["configs"].items():
        src = os.path.join(ROOT, v["source"])
        assert hashlib.sha256(open(src, "rb").read()).hexdigest() == v["source_sha256"]
    assert json.load(open(os.path.join(ROOT, "runs_partial_ppl", "bf16_baseline.json")))["ppl"] == pytest.approx(9.237090, abs=1e-6)
    reg = os.path.join(OUT, "logs", "baseline_regression.json")
    if os.path.exists(reg) and "quant_pipeline" in json.load(open(reg)):
        assert json.load(open(reg))["quant_pipeline"]["passed"]
