"""CPU tests for the multi-grouping robustness study (synthetic cases + tiny Qwen2-MoE)."""
import json
import os
import sys

import numpy as np
import pytest
import torch

from conftest import tiny_model
from moe_calib import batch_sensitivity as BS
from moe_calib import multigroup_robustness as MG
from moe_calib import partial_model_quant as PQ

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import batch_sensitivity as DB          # noqa: E402
import eval_partial_ppl as EP           # noqa: E402
import full_depth_pilot as FD           # noqa: E402
import multigroup_robustness as MR      # noqa: E402

PROTO = os.path.join(ROOT, MR.OUT, "protocol.json")
UNION217 = sorted(np.random.default_rng(5).choice(2000, 217, replace=False).tolist())
HAVE_SEL = os.path.exists(os.path.join(ROOT, EP.OUT, "selected_set_manifest.json"))


# ------------------------------------------------------------- groupings
def test_five_fixed_grouping_seeds():
    assert MG.GROUPING_SEEDS == (20261111, 20261112, 20261113, 20261114, 20261115)
    assert MR.GROUPS == [f"G{s}" for s in MG.GROUPING_SEEDS]
    bts = [MG.grouping_batches(UNION217, s) for s in MG.GROUPING_SEEDS]
    assert all(MG.grouping_batches(UNION217, s) == b for s, b in zip(MG.GROUPING_SEEDS, bts))   # deterministic
    assert len({json.dumps(b) for b in bts}) == 5                                               # all distinct


def test_grouping_is_rng_permutation_of_sorted_union():
    s = 20261113
    shuffled = list(reversed(UNION217))                       # input order must not matter: union is sorted first
    bt = MG.grouping_batches(shuffled, s)
    perm = np.random.default_rng(s).permutation(217)
    assert [r for b in bt for r in b] == [UNION217[i] for i in perm]          # within-batch order preserved


def test_27_full_batches_plus_singleton_no_duplicates():
    for s in MG.GROUPING_SEEDS:
        bt = MG.grouping_batches(UNION217, s)
        assert [len(b) for b in bt] == MG.EXPECTED_SIZES == [8] * 27 + [1]
        v = MG.validate_grouping(bt, UNION217, forbidden=range(2000, 2256))
        assert all(v.values()), v
    bad = MG.grouping_batches(UNION217, 1)
    bad[0][0] = bad[1][0]                                                     # duplicate -> detected
    v = MG.validate_grouping(bad, UNION217)
    assert not v["no_duplicates"] and not v["exact_union"]


def test_no_test_row_inclusion_detected():
    bt = MG.grouping_batches(UNION217, MG.GROUPING_SEEDS[0])
    assert MG.validate_grouping(bt, UNION217, forbidden=range(2128, 2256))["no_forbidden_rows"]
    leak = [list(b) for b in bt]
    leak[-1].append(2130)
    assert not MG.validate_grouping(leak, UNION217, forbidden=range(2128, 2256))["no_forbidden_rows"]


def test_same_grouping_paired_for_fr_and_fc():
    fr, fc = UNION217[:128], UNION217[-128:]
    bt = MG.grouping_batches(UNION217, MG.GROUPING_SEEDS[1])
    m = MG.manifest(MG.GROUPING_SEEDS[1], bt, UNION217, {"FR": fr, "FC": fc})
    assert m["batches"] == bt                                                  # one grouping for both
    for c, ids in (("FR", fr), ("FC", fc)):
        per = m["selected_members_per_batch"][c]
        assert [r for b in per for r in b] == [r for b in bt for r in b if r in set(ids)]
        assert sorted(r for b in per for r in b) == sorted(ids)
    assert m["manifest_sha256"] == MG.obj_sha(MG.strip_sha(m))


def test_reassembly_by_sequence_id_and_last_batch():
    bt = MG.grouping_batches(UNION217, MG.GROUPING_SEEDS[2])
    tens = [torch.tensor([float(r) for r in b])[:, None] for b in bt]
    X = BS.reassemble(tens, BS.reassembly_index(bt, UNION217))
    assert X[:, 0].tolist() == [float(r) for r in UNION217]
    last = bt[-1][0]
    assert X[UNION217.index(last), 0] == float(last)                          # the singleton batch lands correctly


def test_cache_index_remap():
    chosen = [UNION217[i] for i in (5, 200, 0, 77)]                          # frozen (unsorted) order kept
    idx = MG.remap(chosen, UNION217)
    assert idx == [5, 200, 0, 77] and [UNION217[i] for i in idx] == chosen
    with pytest.raises(KeyError):
        MG.remap([2300], UNION217)


# ------------------------------------------------------------- forward semantics (tiny model)
def _fwd(model, plan, rows, alltok):
    got = {}
    DB.forward_plan(model, alltok, plan, rows, lambda L, X: got.__setitem__(L, X.clone()))
    return got


def test_tokens_positions_attention_preserved_fp64():
    """FP64: every grouping (and batch-1 execution) gives the same per-sequence layer inputs to 1e-12 -> token ids,
    positions and causal per-sequence attention do not depend on grouping; only BF16 rounding can."""
    model, _ = tiny_model()
    model = model.double()
    rows = sorted([3, 17, 5, 22, 9, 11, 30, 2, 41, 7, 19, 44, 60, 1, 33, 28, 50])          # 17 rows -> 2 x 8 + 1
    alltok = torch.randint(0, 64, (64, 16), generator=torch.Generator().manual_seed(3))
    ref = _fwd(model, [[r] for r in rows], rows, alltok)
    for s in MG.GROUPING_SEEDS:
        bt = MG.grouping_batches(rows, s)
        assert [len(b) for b in bt] == [8, 8, 1]
        got = _fwd(model, bt, rows, alltok)
        assert all(torch.allclose(got[L], ref[L], atol=1e-12, rtol=0) for L in ref)
    assert torch.equal(ref[0], model.model.embed_tokens(alltok[rows]))               # embeddings = token ids


def test_execution_order_negative_control_bf16():
    model, _ = tiny_model()
    rows = sorted([3, 17, 5, 22, 9, 11, 30, 2, 41, 7, 19, 44, 60, 1, 33, 28, 50])
    alltok = torch.randint(0, 64, (64, 16), generator=torch.Generator().manual_seed(4))
    bt = MG.grouping_batches(rows, MG.GROUPING_SEEDS[0])
    a, b = _fwd(model, bt, rows, alltok), _fwd(model, bt[::-1], rows, alltok)
    assert all(torch.equal(a[L], b[L]) and a[L].dtype == torch.bfloat16 for L in a)


def test_routing_diagnostic_alignment():
    o = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3]])
    v = torch.tensor([[1, 0, 2, 3], [4, 5, 6, 8], [0, 1, 2, 3]])
    r = BS.routing_compare(v, o, n_experts=9)
    assert r["frac_any_expert_changed"] == pytest.approx(1 / 3) and r["frac_reordered_same_set"] == pytest.approx(1 / 3)
    # token rows (seq i, position t) -> i*T + t: the per-strategy subset index used by the diagnostics
    T, idx = 2, [1, 0]
    ti = (torch.tensor(idx)[:, None] * T + torch.arange(T)[None]).flatten()
    assert ti.tolist() == [2, 3, 0, 1]


# ------------------------------------------------------------- GPTQ / installation / contamination
def test_frozen_gptq_settings_and_protocol():
    cfg = json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))
    assert cfg["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True)
    assert MR.LAYERS == [1, 8, 16, 23]
    if os.path.exists(PROTO):
        pr = json.load(open(PROTO))
        assert pr["gptq"] == cfg["gptq"] and pr["quant_layers"] == [1, 8, 16, 23] and pr["B"] == 128
        assert pr["allocation_seed"] == 0 and pr["n_union"] == 217 and len(pr["groupings"]) == 5
        assert MG.obj_sha({k: v for k, v in pr.items() if k not in ("protocol_sha256", "created", "environment")}) \
            == pr["protocol_sha256"]
        assert pr["selections"]["FR"]["ids_hash"] == "ec355fb8d866e4ff"
        assert pr["selections"]["FC"]["ids_hash"] == "0353229b6d3090db"
        for g, x in pr["groupings"].items():
            assert x["batches"] == MG.grouping_batches(pr["union_calibration_ids"], x["seed"])
            assert not set(r for b in x["batches"] for r in b) & set(range(2000, 2256))


@pytest.mark.skipif(not HAVE_SEL, reason="frozen selections absent")
def test_frozen_selection_hashes():
    cfg, sel, checks, union = MR.selections()
    assert checks["FR"]["ids_hash"] == "ec355fb8d866e4ff" and checks["FC"]["ids_hash"] == "0353229b6d3090db"
    assert len(union) == 217 and all(checks[c]["ordered_warm_start_is_frozen"] for c in checks)


def _cache(tmp_path, name, seed, N=40, T=16, D=64, noise=None):
    d = tmp_path / name
    d.mkdir()
    g = torch.Generator().manual_seed(seed)
    for L in range(3):
        X = torch.randn(N, T, D, generator=g)
        if noise is not None:
            X = X + noise * torch.randn(X.shape, generator=torch.Generator().manual_seed(100 + L))
        torch.save(X.to(torch.bfloat16), d / f"L{L}.pt")
    return str(d)


def _q(model, cdir, chosen, ev, ecache=None):
    return PQ.quantize_install(model, cdir, [0, 2], chosen, ev, 4, 4, 16, 1.0, eval_cache_dir=ecache)


def test_weight_installation_and_no_cross_run_contamination(tmp_path):
    c = _cache(tmp_path, "c", 1)
    m1, _ = tiny_model(); m2, _ = tiny_model()
    t0 = PQ.target_tensor_digests(m1, [0, 2]); u0 = PQ.untouched_digests(m1, [0, 2])
    q = _q(m1, c, list(range(12)), [30, 31])
    after = PQ.target_tensor_digests(m1, [0, 2])
    assert all(after[k] != t0[k] for k in after) and PQ.untouched_digests(m1, [0, 2]) == u0
    assert q["counts"]["n_experts_quantized_total"] == 2 * 8
    fresh, _ = tiny_model()
    assert PQ.target_tensor_digests(fresh, [0, 2]) == t0                      # next run starts pristine
    _q(m2, c, list(range(12)), [30, 31])
    assert PQ.installed_digest(m1, [0, 2]) == PQ.installed_digest(m2, [0, 2])  # repeat: identical weights


def test_grouping_cache_changes_weights_not_reconstruction_rows(tmp_path):
    ref = _cache(tmp_path, "ref", 1)
    alt = _cache(tmp_path, "alt", 1, noise=0.02)
    m1, _ = tiny_model(); m2, _ = tiny_model()
    r1 = _q(m1, alt, list(range(12)), [30, 31], ecache=ref)
    r2 = _q(m2, ref, list(range(12)), [30, 31])
    assert r1["counts"]["eval_universe"] == r2["counts"]["eval_universe"]
    assert PQ.installed_digest(m1, [0, 2]) != PQ.installed_digest(m2, [0, 2])


# ------------------------------------------------------------- evaluation / statistics / decision
@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, EP.OUT, "windows.pt")), reason="windows absent")
def test_frozen_evaluation_windows():
    win, prot = EP.load_windows()
    assert tuple(win.shape) == (128, 512) and prot["scored_targets_total"] == 65408 and EP.BATCH == 8
    if os.path.exists(PROTO):
        assert json.load(open(PROTO))["evaluation"]["windows_sha256"] == prot["windows_sha256"]


def _ppl(deltas, fr=None):
    fr = fr or [9.27] * len(deltas)
    gs = [f"G{i}" for i in range(len(deltas))]
    ppl = {("FR", g): f for g, f in zip(gs, fr)}
    ppl.update({("FC", g): f + d for g, f, d in zip(gs, fr, deltas)})
    return ppl, gs


def test_paired_differences_and_grouping_stats():
    ppl, gs = _ppl([0.01, -0.004, 0.002, 0.006, -0.012], fr=[9.268, 9.272, 9.270, 9.269, 9.278])
    s = MG.grouping_stats(ppl, gs, delta_original=0.010057, tau=9.237e-4)
    d = np.array([0.01, -0.004, 0.002, 0.006, -0.012])
    assert s["delta"]["G1"] == pytest.approx(-0.004) and s["mean_delta"] == pytest.approx(d.mean())
    assert s["median_delta"] == pytest.approx(0.002) and s["sd_delta"] == pytest.approx(d.std(ddof=1))
    assert s["min_delta"] == pytest.approx(-0.012) and s["delta_range"] == pytest.approx(0.022)
    assert s["n_fc_worse"] == 3 and s["n_fc_better"] == 2 and s["n_sign_reversals"] == 2
    assert s["n_sign_reversals_beyond_tau"] == 2
    assert s["interaction"]["G4"] == pytest.approx(-0.012 - 0.010057)
    assert s["strategy_range"]["FR"] == pytest.approx(0.010)
    assert s["sd_over_abs_delta_original"] == pytest.approx(d.std(ddof=1) / 0.010057)


def test_decision_rule():
    b = 9.237090
    assert MG.decide(*_ppl([0.01, -0.004, 0.002, 0.006, -0.012]), b) == "GROUPING-DEPENDENT-RANKING"
    # no reversal but interaction >= 0.5 |Delta_O|
    assert MG.decide(*_ppl([0.0101, 0.004, 0.009, 0.010, 0.011]), b) == "GROUPING-DEPENDENT-RANKING"
    # reversal within tau only, large interaction anyway -> dependent
    assert MG.decide(*_ppl([0.0101, -0.0005, 0.009, 0.010, 0.011]), b) == "GROUPING-DEPENDENT-RANKING"
    # stable sign, moderate change (> max(tau, 0.1 |D|) = 1.0e-3 but < 0.5 |D|, SD < 0.5 |D|)
    assert MG.decide(*_ppl([0.0101, 0.0080, 0.0110, 0.0095, 0.0120]), b) == "RANKING-STABLE-BUT-SENSITIVE"
    # strategy range alone above floor
    assert MG.decide(*_ppl([0.0101] * 5, fr=[9.268, 9.270, 9.268, 9.268, 9.268]), b) == "RANKING-STABLE-BUT-SENSITIVE"
    assert MG.decide(*_ppl([0.0101, 0.0100, 0.0102, 0.0099, 0.0101], fr=[9.268, 9.2681, 9.268, 9.2679, 9.268]), b) \
        == "RANKING-ROBUST-WITHIN-TESTED-GROUPINGS"
    ppl, gs = _ppl([0.01] * 5)
    assert MG.decide(ppl, gs, b, valid=False) == "INVALID-COMPARISON"
    assert MG.decide(ppl, gs, b, blocked=True) == "RUN-BLOCKED"
    assert MG.decide(ppl, gs, b, valid=False, blocked=True) == "INVALID-COMPARISON"
    assert MG.DELTA_ORIGINAL == pytest.approx(0.010057, abs=1e-6)


def test_bootstrap_paired_and_interpretation():
    tok = np.array([511.0] * 8)
    a = np.array([2.0, 2.1, 1.9, 2.0, 2.2, 1.8, 2.05, 1.95]) * tok
    r = FD.paired_bootstrap(a + 0.001 * tok, a, tok, n_boot=500)
    assert r["point"] == pytest.approx(0.001) and r["ci95"] == pytest.approx([0.001, 0.001])   # paired: shared noise cancels
    s = MG.bootstrap_summary({"G0": [0.0002, 0.002], "G1": [-0.002, -0.0004], "G2": [-0.001, 0.001]}, {})
    assert s["n_ci_excluding_zero_positive"] == 1 and s["n_ci_excluding_zero_negative"] == 1
    assert s["contradictory_conditional_conclusions"] and "CONDITIONAL" in s["note"]
    assert "does not describe calibration-batching uncertainty" in s["note"]
