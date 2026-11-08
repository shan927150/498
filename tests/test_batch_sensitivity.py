"""CPU tests for the calibration batch-composition sensitivity study (tiny Qwen2-MoE)."""
import json
import os
import sys

import numpy as np
import pytest
import torch

from conftest import E, tiny_model
from moe_calib import batch_sensitivity as BS
from moe_calib import model_utils as MU
from moe_calib import partial_model_quant as PQ

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import batch_sensitivity as DB          # noqa: E402
import eval_partial_ppl as EP           # noqa: E402
import full_depth_pilot as FD           # noqa: E402

ROWS = sorted([3, 17, 5, 22, 9, 11, 30, 2, 41, 7, 19]) + [50, 51, 52, 53, 54, 55]   # union + "test" rows
PROTO = os.path.join(ROOT, DB.OUT, "protocol.json")


# ------------------------------------------------------------- 1-6 batch plans / reassembly
def test_sequence_ids_preserved_by_every_plan():
    pl = DB.plans(ROWS, 64)
    for v, bt in pl.items():
        flat = [r for b in bt for r in b]
        assert len(flat) == len(set(flat)) and set(ROWS) <= set(flat)
        if v in ("COMPACT", "SHUFFLED"):
            assert sorted(flat) == sorted(ROWS)


def test_original_grouping_is_cache_inputs_batches():
    bt = BS.original_batches(ROWS, 8, 64)
    for b in bt:
        g = b[0] // 8
        assert b == list(range(g * 8, g * 8 + 8))                  # consecutive rows from a multiple of 8
    assert {r // 8 for r in ROWS} == {b[0] // 8 for b in bt}
    assert BS.original_batches([61], 8, 62) == [[56, 57, 58, 59, 60, 61]]   # truncated last original batch


def test_compact_grouping_deterministic_and_v1_pattern():
    assert BS.compact_batches(ROWS, 8) == BS.compact_batches(ROWS, 8)
    assert BS.compact_batches(ROWS, 8)[0] == ROWS[:8]
    assert BS.batch_sizes(BS.compact_batches(list(range(345)), 8)) == [8] * 43 + [1]


def test_shuffled_grouping_deterministic_fixed_seed():
    a, b = BS.shuffled_batches(ROWS, 8), BS.shuffled_batches(ROWS, 8, 20261110)
    assert a == b and BS.SHUFFLE_SEED == 20261110
    assert sorted(r for x in a for r in x) == sorted(ROWS)
    assert a != BS.compact_batches(ROWS, 8)
    assert BS.shuffled_batches(ROWS, 8, 1) != a


def test_last_batch_treatment_identical():
    for n in (17, 345, 16):
        r = list(range(n))
        assert BS.batch_sizes(BS.compact_batches(r, 8)) == BS.batch_sizes(BS.shuffled_batches(r, 8))
    assert BS.batch_sizes(BS.shuffled_batches(list(range(345)), 8))[-1] == 1


def test_reassembly_by_original_sequence_id():
    bt = BS.shuffled_batches(ROWS, 4)
    data = {r: torch.full((3,), float(r)) for r in ROWS}
    tens = [torch.stack([data[r] for r in b]) for b in bt]
    X = BS.reassemble(tens, BS.reassembly_index(bt, ROWS))
    assert X[:, 0].tolist() == [float(r) for r in ROWS]
    with pytest.raises(ValueError):
        BS.reassembly_index(bt, ROWS + [99])


# ------------------------------------------------------------- 7-10 forward semantics
def _forward(model, plan, rows, alltok):
    got = {}
    DB.forward_plan(model, alltok, plan, rows, lambda L, X: got.__setitem__(L, X.clone()))
    return got


def test_positions_mask_and_no_cross_sequence_leakage():
    """In FP64 every batch plan gives the same per-sequence layer inputs (to 1e-12): no positions,
    mask or cross-sequence interaction depends on batch composition; only rounding can."""
    model, _ = tiny_model()
    model = model.double()
    alltok = torch.randint(0, 64, (64, 16), generator=torch.Generator().manual_seed(3))
    ref = None
    for v, plan in DB.plans(ROWS, 64).items():
        got = _forward(model, plan, ROWS, alltok)
        if ref is None:
            ref = got
        for L in got:
            assert torch.allclose(got[L], ref[L], atol=1e-12, rtol=0), (v, L)
    single = _forward(model, [[r] for r in ROWS], ROWS, alltok)           # batch of 1 each
    assert all(torch.allclose(single[L], ref[L], atol=1e-12, rtol=0) for L in ref)


def test_token_ids_checked_and_bf16_reorder_negative_control():
    model, _ = tiny_model()
    alltok = torch.randint(0, 64, (64, 16), generator=torch.Generator().manual_seed(4))
    pl = DB.plans(ROWS, 64)
    a = _forward(model, pl["ORIGINAL"], ROWS, alltok)
    b = _forward(model, pl["ORIGINAL_REVERSED"], ROWS, alltok)
    assert all(torch.equal(a[L], b[L]) for L in a)                       # execution order alone: bitwise
    assert all(a[L].dtype == torch.bfloat16 for L in a)


def test_model_config_record():
    model, _ = tiny_model()
    rec = DB.model_config_record(model)
    assert rec["training"] is False and rec["dropout_modules_p_gt_0"] == []
    assert rec["attn_implementation"] == "eager" and rec["param_dtypes"] == ["torch.bfloat16"]
    pe = rec["layer_kwargs"]["position_embeddings"]
    assert all(x["shape"][0] == 1 for x in pe)                           # batch-1 kwargs, broadcast


# ------------------------------------------------------------- 11-14 routing / drift / Hessian metrics
def test_routing_alignment_and_set_vs_order():
    o = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3], [8, 9, 10, 11]])
    v = torch.tensor([[0, 1, 2, 3], [5, 4, 6, 7], [0, 1, 2, 9], [8, 9, 10, 11]])
    r = BS.routing_compare(v, o, n_experts=12)
    assert r["frac_ordered_identical"] == 0.5 and r["frac_set_identical"] == 0.75
    assert r["frac_reordered_same_set"] == 0.25 and r["frac_any_expert_changed"] == 0.25
    assert r["mean_topk_overlap"] == pytest.approx((4 + 4 + 3 + 4) / 16)
    assert r["per_expert_count_abs_diff"][3] == 1 and r["per_expert_count_abs_diff"][9] == 1
    assert r["routing_distribution_l1"] == pytest.approx(2 / 16)
    # alignment: permuting tokens of one side breaks agreement (tokens are compared by position)
    assert BS.routing_compare(v.flip(0), o, n_experts=12)["frac_set_identical"] < 0.75


def test_topk_route_matches_model_op():
    lg = torch.randn(50, 8, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    probs, sel = BS.topk_route(lg, 2)
    ref = torch.topk(torch.softmax(lg, dim=1, dtype=torch.float), 2, dim=-1).indices
    assert torch.equal(sel, ref)
    assert BS.expert_token_mask(sel, int(sel[0, 0]))[0]


def test_activation_drift_metrics():
    R = torch.ones(2, 3, 4)
    X = R.clone(); X[1, 2, 0] = 3.0
    d = BS.activation_drift(X, R)
    assert d["relative_l2"] == pytest.approx(2 / R.norm().item())
    assert d["max_abs"] == 2 and d["mean_abs"] == pytest.approx(2 / 24)
    assert d["frac_elements_exactly_equal"] == pytest.approx(23 / 24)
    assert d["n_sequences_with_differences"] == 1 and d["nan_count"] == 0 and d["inf_count"] == 0
    assert d["token_rel_l2_p50"] == 0 and d["token_rel_l2_p99"] > 0


def test_hessian_compare_metrics_and_actorder():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(300, 256, generator=g)
    H = BS.gram(X)
    same = BS.hessian_compare(H, H)
    assert same["rel_fro"] == 0 and same["trace_ratio"] == 1 and same["actorder_perm_identical"]
    s = BS.hessian_compare(2 * H, H)
    assert s["rel_fro"] == pytest.approx(1) and s["trace_ratio"] == pytest.approx(2) and s["actorder_perm_identical"]
    X2 = X.clone(); X2[:, 0] *= 0.01                                    # demote column 0 in act-order
    c = BS.hessian_compare(BS.gram(X2), H)
    assert not c["actorder_perm_identical"] and c["frac_columns_group_changed"] > 0
    assert torch.equal(BS.actorder_perm(H), torch.argsort(torch.diag(H), descending=True))


# ------------------------------------------------------------- 15-18 GPTQ / contamination / reconstruction
def test_gptq_settings_unchanged():
    cfg = json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))
    assert cfg["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True)
    if os.path.exists(PROTO):
        pr = json.load(open(PROTO))
        assert pr["gptq"] == cfg["gptq"] and pr["quant_layers"] == [1, 8, 16, 23] and pr["B"] == 128 and pr["seed"] == 0


def _cache(tmp_path, name, seed, N=60, T=16, D=64):
    d = tmp_path / name
    d.mkdir()
    g = torch.Generator().manual_seed(seed)
    for L in range(3):
        torch.save(torch.randn(N, T, D, generator=g).to(torch.bfloat16), d / f"L{L}.pt")
    return str(d)


def _q(model, cdir, chosen, ev, ecache=None):
    return PQ.quantize_install(model, cdir, [0, 2], chosen, ev, 4, 4, 16, 1.0, eval_cache_dir=ecache)


def test_fingerprint_reproducible_and_no_contamination(tmp_path):
    c = _cache(tmp_path, "c", 1)
    a, _ = tiny_model(); b, _ = tiny_model()
    t0 = PQ.target_tensor_digests(b, [0, 2])
    _q(a, c, list(range(12)), [50, 51])
    fresh, _ = tiny_model()
    assert PQ.target_tensor_digests(fresh, [0, 2]) == t0                 # next config starts pristine
    _q(b, c, list(range(12)), [50, 51])
    assert PQ.installed_digest(a, [0, 2]) == PQ.installed_digest(b, [0, 2])


def test_reconstruction_set_fixed_across_calibration_caches(tmp_path):
    """Calibration from a perturbed cache, reconstruction always from the fixed eval cache:
    the evaluation universe / activations do not follow the calibration variant."""
    ref = _cache(tmp_path, "ref", 1)
    alt = tmp_path / "alt"; alt.mkdir()
    for L in range(3):                                                    # perturb EVERY row incl. eval rows
        X = torch.load(os.path.join(ref, f"L{L}.pt"))
        noise = 0.05 * torch.randn(X.shape, generator=torch.Generator().manual_seed(L))
        torch.save((X.float() + noise).to(torch.bfloat16), alt / f"L{L}.pt")
    m1, _ = tiny_model(); m2, _ = tiny_model(); m3, _ = tiny_model()
    r_ref = _q(m1, ref, list(range(12)), [50, 51])
    r_alt_fixed = _q(m2, str(alt), list(range(12)), [50, 51], ecache=ref)
    r_alt_own = _q(m3, str(alt), list(range(12)), [50, 51])
    assert r_alt_fixed["counts"]["eval_universe"] == r_ref["counts"]["eval_universe"]
    assert PQ.installed_digest(m2, [0, 2]) == PQ.installed_digest(m3, [0, 2])     # weights: calibration only
    assert PQ.installed_digest(m2, [0, 2]) != PQ.installed_digest(m1, [0, 2])
    assert [x["mean"] for x in r_alt_fixed["results"]] != [x["mean"] for x in r_alt_own["results"]]


# ------------------------------------------------------------- 19-22 evaluation / statistics / regression
@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, EP.OUT, "windows.pt")), reason="windows absent")
def test_frozen_ppl_windows():
    win, prot = EP.load_windows()
    assert tuple(win.shape) == (128, 512) and prot["scored_targets_total"] == 65408
    if os.path.exists(PROTO):
        assert json.load(open(PROTO))["windows_sha256"] == prot["windows_sha256"]


def test_deltas_interaction_and_decision():
    ppl = {("FR", "ORIGINAL"): 9.2679, ("FC", "ORIGINAL"): 9.2779, ("FR", "COMPACT"): 9.2720,
           ("FC", "COMPACT"): 9.2680, ("FR", "SHUFFLED"): 9.2690, ("FC", "SHUFFLED"): 9.2770}
    s = BS.strategy_deltas(ppl)
    assert s["delta"]["ORIGINAL"] == pytest.approx(0.0100) and s["delta"]["COMPACT"] == pytest.approx(-0.0040)
    assert s["interaction"]["COMPACT"] == pytest.approx(-0.0140) and s["interaction"]["ORIGINAL"] == 0
    assert s["strategy_range"]["FC"] == pytest.approx(0.0099) and s["n_fc_worse"] == 2 and s["ranking_reverses"]
    assert BS.decide(ppl, 9.237) == "BATCH-SENSITIVITY-DOMINANT"
    stable = {k: v + (0.003 if k == ("FC", "COMPACT") else 0) for k, v in ppl.items() if k[1] == "ORIGINAL"}
    stable.update({("FR", "COMPACT"): 9.2679 + 0.001, ("FC", "COMPACT"): 9.2779 + 0.003,
                   ("FR", "SHUFFLED"): 9.2679 - 0.001, ("FC", "SHUFFLED"): 9.2779})
    assert BS.decide(stable, 9.237) == "BATCH-SENSITIVITY-MODERATE"
    tiny = {k: (9.2679 if k[0] == "FR" else 9.2779) + (0.0002 if k[1] != "ORIGINAL" else 0)
            for k in ppl}
    assert BS.decide(tiny, 9.237) == "BATCH-SENSITIVITY-NEGLIGIBLE"
    assert BS.decide(ppl, 9.237, valid=False) == "BATCH-VARIANT-INVALID"
    assert BS.decide(ppl, 9.237, blocked=True) == "RUN-BLOCKED"


def test_paired_bootstrap_window_level():
    tok = np.array([511.0] * 6)
    a = np.array([2.0, 2.1, 1.9, 2.0, 2.2, 1.8]) * tok
    r = FD.paired_bootstrap(a, a - 0.002 * tok, tok, n_boot=300)
    assert r["point"] == pytest.approx(0.002) and r["ci95"] == pytest.approx([0.002, 0.002])
    assert FD.paired_bootstrap(a, a - 0.002 * tok, tok, n_boot=300) == r


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, EP.OUT, "ppl_FR_s0.json")), reason="partial PPL absent")
def test_original_regression_reference_and_result():
    ref = {c: json.load(open(os.path.join(ROOT, EP.OUT, f"ppl_{c}_s0.json"))) for c in ("FR", "FC")}
    assert ref["FR"]["ppl"] == pytest.approx(9.267851, abs=1e-6) and ref["FC"]["ppl"] == pytest.approx(9.277908, abs=1e-6)
    if os.path.exists(PROTO):
        pr = json.load(open(PROTO))["references"]["partial_ppl"]
        assert all(pr[c]["installed_weights_sha256"] == ref[c]["installed_weights_sha256"] for c in ref)
    reg = os.path.join(ROOT, DB.OUT, "regression_checks.json")
    if os.path.exists(reg):
        assert json.load(open(reg))["passed"]
