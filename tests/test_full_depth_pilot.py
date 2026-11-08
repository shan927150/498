"""CPU tests for the full-depth routed-expert GPTQ pilot (tiny Qwen2-MoE: 3 MoE layers x 8 experts)."""
import json
import os
import sys

import numpy as np
import pytest
import torch

from conftest import E, LAYERS, tiny_model
from moe_calib import model_utils as MU
from moe_calib import partial_model_quant as PQ
from moe_calib.data_utils import load_layer_cache, load_meta, split_ids

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import eval_partial_ppl as EP       # noqa: E402
import full_depth_pilot as FD       # noqa: E402

GS = 16


def _q(model, cdir, layers, chosen, ev, capture=()):
    return PQ.quantize_install(model, cdir, layers, chosen, ev, 4, 4, GS, 1.0, capture=capture)


def _full_cache(tmp_path, n_layers=3, N=40, T=16, D=64):
    """Cache for EVERY layer (the conftest cache only has layers 0 and 2)."""
    d = tmp_path / "full"
    d.mkdir()
    g = torch.Generator().manual_seed(7)
    for L in range(n_layers):
        torch.save(torch.randn(N, T, D, generator=g).to(torch.bfloat16), d / f"L{L}.pt")
    return str(d)


def _compact_cache(src, tmp_path, rows, n_layers=3):
    d = tmp_path / "compact"
    d.mkdir()
    for L in range(n_layers):
        torch.save(load_layer_cache(src, L)[rows].clone(), d / f"L{L}.pt")
    return str(d)


# ------------------------------------------------------------- enumeration / mapping
def test_resolve_all_layers_and_backward_compat():
    model, _ = tiny_model()
    assert PQ.resolve_layers("all", model) == [0, 1, 2]
    assert PQ.resolve_layers("0,2", model) == LAYERS
    assert PQ.resolve_layers([2, 0], model) == [0, 2]
    with pytest.raises(ValueError):
        PQ.resolve_layers("5", model)


def test_target_digests_cover_every_projection():
    model, _ = tiny_model()
    tg = PQ.target_tensor_digests(model, [0, 1, 2])
    assert len(tg) == 3 * E * 3
    assert {k.split(".")[2] for k in tg} == {"gate_proj", "up_proj", "down_proj"}
    assert {int(k.split(".")[1]) for k in tg} == set(range(E))
    assert not set(PQ.untouched_digests(model, [0, 1, 2])) & PQ.target_names(model, [0, 1, 2])


def test_nonfinite_detection():
    model, _ = tiny_model()
    assert PQ.nonfinite_targets(model, [0, 1, 2]) == []
    with torch.no_grad():
        MU.get_expert_weights(model, 1, 3)["up_proj"][0, 0] = float("nan")
    assert PQ.nonfinite_targets(model, [0, 1, 2]) == ["1.3.up_proj"]


# ------------------------------------------------------------- calibration semantics
def test_compact_cache_remap_gives_identical_weights(tmp_path):
    """The compact cache (only needed rows, remapped ids) yields bitwise the same GPTQ weights."""
    full = _full_cache(tmp_path)
    chosen, ev = [3, 17, 5, 22, 9, 11, 30, 2], [34, 35, 36, 37]
    rows = sorted(set(chosen)) + ev
    comp = _compact_cache(full, tmp_path, rows)
    a, _ = tiny_model(); b, _ = tiny_model()
    ra = _q(a, full, [0, 1, 2], chosen, ev)
    rb = _q(b, comp, [0, 1, 2], FD.remap(chosen, rows), FD.remap(ev, rows))
    assert PQ.installed_digest(a, [0, 1, 2]) == PQ.installed_digest(b, [0, 1, 2])
    assert [r["mean"] for r in ra["results"]] == [r["mean"] for r in rb["results"]]


def test_layer_by_layer_equals_all_at_once_non_sequential(tmp_path):
    """Quantizing layer 0 first never changes layer 2's calibration (cached BF16 inputs)."""
    full = _full_cache(tmp_path)
    chosen, ev = list(range(0, 12)), [36, 37]
    a, _ = tiny_model(); b, _ = tiny_model()
    _q(a, full, [0, 1, 2], chosen, ev)
    for L in (2, 0, 1):                                  # any order, one layer at a time
        _q(b, full, [L], chosen, ev)
    assert PQ.installed_digest(a, [0, 1, 2]) == PQ.installed_digest(b, [0, 1, 2])


def test_no_skipped_experts_and_no_hidden_rtn(tmp_path):
    full = _full_cache(tmp_path)
    model, _ = tiny_model()
    pristine = PQ.target_tensor_digests(model, [0, 1, 2])
    q = _q(model, full, [0, 1, 2], list(range(0, 20)), [36])
    assert q["counts"]["n_experts_quantized_total"] == 3 * E
    assert q["counts"]["n_zero_token_quantization_fallback"] == 0
    after = PQ.target_tensor_digests(model, [0, 1, 2])
    assert all(after[k] != pristine[k] for k in after)


# ------------------------------------------------------------- installation / contamination / resume
def test_install_preserves_untouched_and_fresh_load_is_pristine(tmp_path):
    full = _full_cache(tmp_path)
    model, _ = tiny_model()
    u0 = PQ.untouched_digests(model, [0, 1, 2])
    t0 = PQ.target_tensor_digests(model, [0, 1, 2])
    q = _q(model, full, [0, 1, 2], list(range(0, 12)), [36], capture=[(1, 2)])
    assert PQ.untouched_digests(model, [0, 1, 2]) == u0
    for k, w in MU.get_expert_weights(model, 1, 2).items():
        assert torch.equal(w.detach(), q["captured"][(1, 2)][k].to(torch.bfloat16))
    fresh, _ = tiny_model()                               # second strategy: fresh model, never the used one
    assert PQ.target_tensor_digests(fresh, [0, 1, 2]) == t0


def test_layer_artifact_roundtrip_and_corruption(tmp_path):
    full = _full_cache(tmp_path)
    model, _ = tiny_model()
    _q(model, full, [1], list(range(0, 12)), [36])
    path = str(tmp_path / "L1.pt")
    PQ.save_layer(model, 1, path)
    assert not os.path.exists(path + ".tmp")
    dig, sha = PQ.layer_digest(model, 1), FD.sha256_file(path)
    fresh, _ = tiny_model()
    PQ.install_layer_file(fresh, 1, path)
    assert PQ.layer_digest(fresh, 1) == dig
    with open(path, "r+b") as f:                          # corrupt one byte -> file hash no longer matches
        f.seek(200); b = f.read(1); f.seek(200); f.write(bytes([b[0] ^ 1]))
    assert FD.sha256_file(path) != sha


def test_layerwise_inference_full_depth_matches_hf(tmp_path):
    full = _full_cache(tmp_path)
    model, _ = tiny_model()
    _q(model, full, [0, 1, 2], list(range(0, 12)), [36])
    win = torch.randint(0, 64, (3, 16), generator=torch.Generator().manual_seed(5))
    nll, cnt = EP.layerwise_nll(model, win, batch=3)
    with torch.no_grad():
        hf = [float(model(input_ids=win[i:i + 1], labels=win[i:i + 1]).loss) for i in range(3)]
    assert np.allclose(nll / cnt, hf, rtol=2e-3, atol=2e-3) and np.isfinite(nll).all()


# ------------------------------------------------------------- statistics / decision
def test_paired_bootstrap_joint_and_token_weighted():
    tok = np.array([511] * 4 + [100] * 4, float)
    a = np.array([2.0] * 8) * tok
    b = a - 0.01 * tok                                     # a worse by exactly 0.01 nats/token everywhere
    r = FD.paired_bootstrap(a, b, tok, n_boot=500)
    assert r["point"] == pytest.approx(0.01) and r["ci95"][0] == pytest.approx(0.01) and r["ci95"][1] == pytest.approx(0.01)
    r2 = FD.paired_bootstrap(a, b, tok, n_boot=500)
    assert r2 == r                                          # fixed seed
    noisy = b + np.random.default_rng(0).normal(0, 3, 8) * tok
    assert FD.paired_bootstrap(a, noisy, tok, n_boot=500)["ci95"][0] < 0.01


def test_decision_rule_cases():
    b = 9.237
    assert FD.decide(b, 9.40, 9.45, 0.01, [0.001, 0.01], True) == "FULL-DEPTH-PENALTY-AMPLIFIED"
    assert FD.decide(b, 9.40, 9.415, 0.01, [0.0005, 0.003], True) == "FULL-DEPTH-PENALTY-PERSISTS"
    assert FD.decide(b, 9.40, 9.41, 0.01, [-0.0005, 0.003], True) == "FULL-DEPTH-PENALTY-NOT-OBSERVED"
    assert FD.decide(b, 9.40, 9.39, 0.01, [-0.003, -0.001], True) == "FULL-DEPTH-PENALTY-NOT-OBSERVED"
    assert FD.decide(b, 9.40, 9.45, 0.01, [0.001, 0.01], False) == "FULL-DEPTH-UNSTABLE"
    assert FD.decide(b, 25.0, 25.5, 0.01, [0.001, 0.01], True) == "FULL-DEPTH-UNSTABLE"


# ------------------------------------------------------------- frozen selections / windows
@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "runs_partial_ppl", "selected_set_manifest.json")),
                    reason="partial-PPL manifest absent")
def test_frozen_seed0_selections_and_compact_rows():
    man, sel = FD.load_selections()
    for c in ("FR", "FC"):
        ch = sel[c]["chosen"]
        assert len(ch) == 128 == len(set(ch)) and all(x < 2000 for x in ch)
        assert ch[:32] == sel[c]["warm_start"] == json.load(open(os.path.join(
            ROOT, "runs_typicality_matched", "warm_starts.json")))["seeds"]["0"]["ids"]
    assert sel["FR"]["ids_hash"] == "ec355fb8d866e4ff"
    rows = FD.compact_rows(sel, list(range(2128, 2256)))
    assert len(rows) == len(set(rows)) and rows[-128:] == list(range(2128, 2256))
    assert set(sel["FR"]["chosen"]) | set(sel["FC"]["chosen"]) == set(rows[:-128])
    win, prot = EP.load_windows()
    assert tuple(win.shape) == (128, 512) and prot["scored_targets_total"] == 65408
