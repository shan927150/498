"""CPU tests for partial-model GPTQ installation and the PPL evaluator (tiny Qwen2-MoE)."""
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

from conftest import E, LAYERS, tiny_model
from moe_calib import model_utils as MU
from moe_calib import partial_model_quant as PQ
from moe_calib.data_utils import load_meta, split_ids
from moe_calib.gptq import gptq_quantize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import eval_partial_ppl as EP      # noqa: E402

GS = 16


def _install(model, cache_dir, chosen, capture=()):
    test = split_ids(load_meta(cache_dir))["test"]
    return PQ.quantize_install(model, cache_dir, LAYERS, chosen, test, 4, 4, GS, 1.0, capture=capture)


# ------------------------------------------------------------- module mapping / installation
def test_expert_module_mapping():
    model, _ = tiny_model()
    names = PQ.target_names(model, LAYERS)
    assert len(names) == len(LAYERS) * E * 3
    params = dict(model.named_parameters())
    for L in LAYERS:
        for e in range(E):
            for k, w in MU.get_expert_weights(model, L, e).items():
                assert params[f"model.layers.{L}.mlp.experts.{e}.{k}.weight"] is w
    assert not any(".shared_expert." in n or ".mlp.gate." in n or "self_attn" in n for n in names)


def test_install_replaces_exactly_targets_and_preserves_rest(cache_dir):
    model, _ = tiny_model()
    snap = PQ.snapshot_experts(model, LAYERS)
    u0 = PQ.untouched_digests(model, LAYERS)
    q = _install(model, cache_dir, list(range(0, 12)), capture=[(LAYERS[0], 1), (LAYERS[1], 5)])
    assert q["counts"]["n_experts_quantized_total"] == len(LAYERS) * E
    assert q["counts"]["n_zero_token_quantization_fallback"] == 0          # no unintended RTN
    assert PQ.experts_match(model, snap)[0] == 0                           # every target changed
    assert PQ.untouched_digests(model, LAYERS) == u0                       # attention/router/shared/other untouched
    for (L, e), Wq in q["captured"].items():
        for k, w in MU.get_expert_weights(model, L, e).items():
            assert torch.equal(w.detach(), Wq[k].to(torch.bfloat16))       # installed = BF16 cast of E3's W_q
            assert w.shape == snap[(L, e)][k].shape                        # orientation preserved


def test_rtn_fallback_is_reported_not_hidden(cache_dir):
    model, _ = tiny_model()
    q = _install(model, cache_dir, [])
    assert q["counts"]["n_zero_token_quantization_fallback"] == len(LAYERS) * E


def test_act_order_and_group_mapping():
    torch.manual_seed(0)
    W = torch.randn(8, 64)
    X = torch.randn(256, 64) * torch.linspace(0.1, 3, 64)
    H = X.T @ X
    Q = gptq_quantize(W, H, 4, GS, 1.0, actorder=True)
    perm = torch.argsort(torch.diag(H), descending=True)
    for r in range(8):
        for g in range(0, 64, GS):
            assert len(torch.unique(Q[r, perm[g:g + GS]])) <= 16             # one scale/zero per permuted group
    Q2 = gptq_quantize(W, H, 4, GS, 1.0, actorder=False)
    for r in range(8):
        for g in range(0, 64, GS):
            assert len(torch.unique(Q2[r, g:g + GS])) <= 16
    assert not torch.equal(Q, Q2)


def test_expert_output_agrees_with_e3(cache_dir):
    model, _ = tiny_model()
    snap = PQ.snapshot_experts(model, LAYERS)
    L, e = LAYERS[0], 1
    q = _install(model, cache_dir, list(range(0, 20)), capture=[(L, e)])
    from moe_calib.data_utils import load_layer_cache
    test = split_ids(load_meta(cache_dir))["test"]
    X = PQ.expert_inputs(model, L, e, load_layer_cache(cache_dir, L)[test[:6]])
    assert X.shape[0] > 0
    ck = PQ.expert_output_check(model, L, e, q["captured"][(L, e)], snap[(L, e)], X)
    assert ck["installed_equals_bf16_of_e3_wq"] and ck["installed_differs_from_bf16_original"]
    assert ck["quant_effect_rel"] > 0 and ck["ratio_to_quant_effect"] < 0.5


def test_restore_prevents_cross_run_contamination(cache_dir):
    model, _ = tiny_model()
    snap = PQ.snapshot_experts(model, LAYERS)
    _install(model, cache_dir, list(range(0, 10)))
    PQ.restore_experts(model, snap)
    eq, tot = PQ.experts_match(model, snap)
    assert eq == tot
    _install(model, cache_dir, list(range(10, 20)))
    d_reused = PQ.installed_digest(model, LAYERS)
    fresh, _ = tiny_model()
    _install(fresh, cache_dir, list(range(10, 20)))
    assert PQ.installed_digest(fresh, LAYERS) == d_reused


# ------------------------------------------------------------- loss / PPL
def test_target_shift_and_counting():
    V, ids = 5, torch.tensor([[1, 2, 3, 4]])
    logits = torch.full((1, 4, V), -30.0)
    for t in range(3):
        logits[0, t, ids[0, t + 1]] = 30.0                                  # position t predicts token t+1
    s, n = EP.nll_from_logits(logits, ids)
    assert int(n[0]) == 3 and float(s[0]) < 1e-10
    bad = torch.full((1, 4, V), -30.0)
    for t in range(4):
        bad[0, t, ids[0, t]] = 30.0                                         # unshifted: predicts current token
    assert float(EP.nll_from_logits(bad, ids)[0][0]) > 100


def test_padding_excluded():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 6, 7, generator=g)
    ids = torch.randint(0, 7, (2, 6), generator=g)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    s, n = EP.nll_from_logits(logits, ids, mask)
    assert n.tolist() == [3, 5]
    s0, _ = EP.nll_from_logits(logits[:1, :4], ids[:1, :4])
    assert float(s[0]) == pytest.approx(float(s0[0]))
    ids2 = ids.clone(); ids2[0, 4:] = 0
    assert float(EP.nll_from_logits(logits, ids2, mask)[0][0]) == pytest.approx(float(s[0]))


def test_token_weighted_accumulation():
    nll, cnt = np.array([10.0, 1.0]), np.array([10, 2])
    r = EP.ppl_record(nll, cnt, 0.0)
    assert r["ppl"] == pytest.approx(math.exp(11 / 12))
    assert r["ppl"] != pytest.approx(np.mean([math.exp(1.0), math.exp(0.5)]))


def test_layerwise_forward_matches_hf_and_is_deterministic():
    model, _ = tiny_model()
    g = torch.Generator().manual_seed(3)
    win = torch.randint(0, 64, (4, 16), generator=g)
    nll, cnt = EP.layerwise_nll(model, win, batch=2)
    nll2, _ = EP.layerwise_nll(model, win, batch=2)
    assert np.array_equal(nll, nll2) and cnt.tolist() == [15] * 4
    with torch.no_grad():
        hf = [float(model(input_ids=win[i:i + 1], labels=win[i:i + 1]).loss) for i in range(4)]
    assert np.allclose(nll / cnt, hf, rtol=2e-3, atol=2e-3)


def test_quantized_layers_propagate_in_sequential_forward(cache_dir):
    model, _ = tiny_model()
    g = torch.Generator().manual_seed(4)
    win = torch.randint(0, 64, (2, 16), generator=g)
    base, _ = EP.layerwise_nll(model, win, batch=2)
    _install(model, cache_dir, list(range(0, 20)))
    q, _ = EP.layerwise_nll(model, win, batch=2)
    assert not np.array_equal(base, q)


# ------------------------------------------------------------- frozen inputs
def test_frozen_gptq_config():
    cfg = json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))
    assert cfg["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True)
    assert cfg["layers"] == [1, 8, 16, 23] and cfg["split"]["seqlen"] == 512


def test_window_fingerprint_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(EP, "ROOT", str(tmp_path))
    os.makedirs(tmp_path / EP.OUT)
    win = torch.randint(0, 100, (EP.N_WIN, EP.SEQLEN))
    torch.save(win, tmp_path / EP.OUT / "windows.pt")
    json.dump(dict(windows_sha256=EP.tensor_fp(win)), open(tmp_path / EP.OUT / "evaluation_protocol.json", "w"))
    EP.load_windows()
    win[0, 0] += 1
    torch.save(win, tmp_path / EP.OUT / "windows.pt")
    with pytest.raises(AssertionError):
        EP.load_windows()


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "runs_typicality_matched", "warm_starts.json")),
                    reason="matched selections absent")
def test_selected_sets_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(EP, "OUT", str(tmp_path))                         # absolute -> os.path.join keeps it
    EP.stage_manifest(None)
    man = json.load(open(tmp_path / "selected_set_manifest.json"))
    assert len(man["configs"]) == 12 and all(v["valid"] for v in man["configs"].values())
    assert all(v["checks"]["confirmation_run_same_ids"] for v in man["configs"].values())
    assert man["configs"]["FR_s0"]["ids_hash"] == "ec355fb8d866e4ff"


def test_install_tolerance_is_bf16_justified():
    # smoke attempt 1, layer 1 expert 0: weights exact; deviation explained by BF16 storage + arithmetic
    ck = dict(weight_rounding_only_rel=0.00107, bf16_compute_noise_rel=0.00263, installed_vs_e3_rel=0.00340)
    assert ck["installed_vs_e3_rel"] <= EP.INSTALL_TOL(ck)
    ck["installed_vs_e3_rel"] = 0.02                       # a real installation error (e.g. wrong expert) fails
    assert ck["installed_vs_e3_rel"] > EP.INSTALL_TOL(ck)
