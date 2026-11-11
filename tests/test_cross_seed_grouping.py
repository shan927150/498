"""CPU tests for the cross-seed x calibration-grouping robustness study."""
import json
import os
import sys

import numpy as np
import pytest
import torch

from conftest import tiny_model
from moe_calib import batch_sensitivity as BS
from moe_calib import cross_seed_grouping as CS
from moe_calib import multigroup_robustness as MG
from moe_calib import partial_model_quant as PQ

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import batch_sensitivity as DB          # noqa: E402
import cross_seed_grouping as XG        # noqa: E402
import eval_partial_ppl as EP           # noqa: E402

PROTO = os.path.join(ROOT, XG.OUT, "protocol.json")
HAVE_SEL = os.path.exists(os.path.join(ROOT, EP.OUT, "selected_set_manifest.json"))
HASHES = {0: ("ec355fb8d866e4ff", "0353229b6d3090db"), 1: ("e36d91e6b151320b", "325e8efc5aa125e7"),
          2: ("cf1f867af56f29d1", "f63a4a6122b79aab")}


# ------------------------------------------------------------- selections / unions
@pytest.mark.skipif(not HAVE_SEL, reason="frozen selections absent")
@pytest.mark.parametrize("s", [0, 1, 2])
def test_frozen_selections_per_seed(s):
    cfg, sel, checks, union = XG.selections(s)
    assert (sel["FR"]["ids_hash"], sel["FC"]["ids_hash"]) == HASHES[s]
    assert all(checks[c]["n"] == 128 == checks[c]["unique"] and checks[c]["ordered_warm_start_is_frozen"] for c in checks)
    assert sel["FR"]["chosen"][:32] == sel["FC"]["chosen"][:32]
    assert union == sorted(set(sel["FR"]["chosen"]) | set(sel["FC"]["chosen"]))
    assert len(union) == {0: 217, 1: 220, 2: 222}[s]
    assert not set(union) & set(range(cfg["split"]["tune"][0], cfg["split"]["test"][1]))


def test_union_construction_and_variable_sizes():
    assert CS.union_of([5, 3, 9], [9, 1]) == [1, 3, 5, 9]
    for n, last in ((217, 1), (220, 4), (222, 6), (224, 8)):
        u = list(range(n))
        for g in MG.GROUPING_SEEDS:
            bt = MG.grouping_batches(u, g)
            assert [len(b) for b in bt] == CS.expected_sizes(n) and len(bt[-1]) == last
            assert all(MG.validate_grouping(bt, u, forbidden=range(2000, 2256)).values())


def test_five_grouping_seeds_deterministic_and_nested():
    assert XG.GSEEDS == [20261111, 20261112, 20261113, 20261114, 20261115]
    u1, u2 = list(range(0, 440, 2)), list(range(1, 445, 2))
    for g in XG.GSEEDS:
        assert MG.grouping_batches(u1, g) == MG.grouping_batches(list(reversed(u1)), g)    # deterministic, sorted first
        assert MG.grouping_batches(u1, g) != MG.grouping_batches(u2, g)                    # different unions -> nested


def test_pairing_no_duplicates_no_leakage():
    u = sorted(np.random.default_rng(1).choice(2000, 220, replace=False).tolist())
    fr, fc = u[:128], u[32:32] + u[:32] + u[-96:]
    bt = MG.grouping_batches(u, 20261112)
    m = MG.manifest(20261112, bt, u, {"FR": fr, "FC": fc})
    flat = [r for b in m["batches"] for r in b]
    assert len(flat) == len(set(flat)) == 220
    for c, ids in (("FR", fr), ("FC", fc)):
        assert sorted(r for b in m["selected_members_per_batch"][c] for r in b) == sorted(set(ids))
    assert not MG.validate_grouping(bt + [[2130]], u, forbidden=range(2128, 2256))["no_forbidden_rows"]
    assert not MG.validate_grouping(bt[:-1] + [bt[-1] + [bt[0][0]]], u)["no_duplicates"]


def test_reassembly_final_batch_and_cache_index():
    u = list(range(100, 322))                                           # 222 rows -> last batch of 6
    bt = MG.grouping_batches(u, 20261115)
    X = BS.reassemble([torch.tensor([float(r) for r in b])[:, None] for b in bt], BS.reassembly_index(bt, u))
    assert X[:, 0].tolist() == [float(r) for r in u]
    chosen = [u[7], u[221], u[0]]
    assert MG.remap(chosen, u) == [7, 221, 0]


def test_execution_order_and_token_preservation_tiny_model():
    model, _ = tiny_model()
    rows = sorted([3, 17, 5, 22, 9, 11, 30, 2, 41, 7, 19, 44, 60, 1, 33, 28, 50, 12, 13, 14])     # 20 -> 8, 8, 4
    alltok = torch.randint(0, 64, (64, 16), generator=torch.Generator().manual_seed(4))
    bt = MG.grouping_batches(rows, 20261111)
    got, rev = {}, {}
    tsha = DB.forward_plan(model, alltok, bt, rows, lambda L, X: got.__setitem__(L, X.clone()))
    tsha_r = DB.forward_plan(model, alltok, bt[::-1], rows, lambda L, X: rev.__setitem__(L, X.clone()))
    assert tsha == tsha_r == DB.tensor_sha(alltok[rows])
    assert all(torch.equal(got[L], rev[L]) for L in got)


def test_cache_fingerprint_sensitive():
    X = torch.randn(3, 4, 5).to(torch.bfloat16)
    Y = X.clone(); Y[0, 0, 0] += 1
    assert DB.tensor_sha(X) == DB.tensor_sha(X.clone()) != DB.tensor_sha(Y)


# ------------------------------------------------------------- GPTQ / installation
def test_gptq_settings_and_counts():
    cfg = json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))
    assert cfg["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True)
    assert XG.LAYERS == [1, 8, 16, 23] and 4 * 60 == 240 and 240 * 3 == 720
    if os.path.exists(PROTO):
        pr = json.load(open(PROTO))
        assert pr["gptq"] == cfg["gptq"] and pr["n_quantized_experts"] == 240 and pr["n_projection_tensors"] == 720
        assert pr["allocation_seeds"] == [0, 1, 2] and len(pr["groupings"]) == 15
        body = {k: v for k, v in pr.items() if k not in ("protocol_sha256", "created")}
        assert MG.obj_sha(body) == pr["protocol_sha256"]
        for x in pr["groupings"].values():
            union = pr["seeds"][str(x["allocation_seed"])]
            assert x["batch_size_pattern"] == CS.expected_sizes(union["n_union"])


def _cache(tmp_path, name, seed, N=40, T=16, D=64):
    d = tmp_path / name
    d.mkdir()
    g = torch.Generator().manual_seed(seed)
    for L in range(3):
        torch.save(torch.randn(N, T, D, generator=g).to(torch.bfloat16), d / f"L{L}.pt")
    return str(d)


def test_no_cross_run_contamination(tmp_path):
    c1, c2 = _cache(tmp_path, "a", 1), _cache(tmp_path, "b", 2)
    m1, _ = tiny_model()
    t0 = PQ.target_tensor_digests(m1, [0, 2])
    PQ.quantize_install(m1, c1, [0, 2], list(range(12)), [30, 31], 4, 4, 16, 1.0)
    d1 = PQ.installed_digest(m1, [0, 2])
    m2, _ = tiny_model()
    assert PQ.target_tensor_digests(m2, [0, 2]) == t0                    # fresh model is pristine
    PQ.quantize_install(m2, c2, [0, 2], list(range(12)), [30, 31], 4, 4, 16, 1.0, eval_cache_dir=c1)
    m3, _ = tiny_model()
    PQ.quantize_install(m3, c2, [0, 2], list(range(12)), [30, 31], 4, 4, 16, 1.0, eval_cache_dir=c1)
    assert PQ.installed_digest(m2, [0, 2]) == PQ.installed_digest(m3, [0, 2]) != d1


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, EP.OUT, "windows.pt")), reason="windows absent")
def test_frozen_windows():
    win, prot = EP.load_windows()
    assert tuple(win.shape) == (128, 512) and prot["scored_targets_total"] == 65408 and EP.BATCH == 8
    if os.path.exists(PROTO):
        assert json.load(open(PROTO))["evaluation"]["windows_sha256"] == prot["windows_sha256"]


# ------------------------------------------------------------- statistics
def _ppl(D):
    ppl = {}
    for s, row in D.items():
        for g, d in row.items():
            ppl[("FR", s, g)] = 9.27 + 0.001 * s
            ppl[("FC", s, g)] = 9.27 + 0.001 * s + d
    return ppl


D3 = {0: dict(zip(range(5), [-0.004069, 0.007601, -0.005889, -0.008394, 0.001091])),
      1: dict(zip(range(5), [0.002, 0.004, -0.001, 0.006, 0.003])),
      2: dict(zip(range(5), [0.010, -0.002, 0.005, 0.001, 0.0]))}


def test_paired_differences_and_unpaired_rejected():
    ppl = _ppl(D3)
    d = CS.paired_deltas(ppl)
    assert d[0][1] == pytest.approx(0.007601) and set(d) == {0, 1, 2}
    del ppl[("FC", 2, 4)]
    with pytest.raises(KeyError):
        CS.paired_deltas(ppl)


def test_seed_means_sds_pooled_between():
    v = CS.variance_summary(D3)
    a = {s: np.array(list(D3[s].values())) for s in D3}
    for s in D3:
        assert v["mean_delta"][s] == pytest.approx(a[s].mean()) and v["sd_grouping"][s] == pytest.approx(a[s].std(ddof=1))
    assert v["mean_delta"][0] == pytest.approx(-0.001932, abs=1e-6) and v["sd_grouping"][0] == pytest.approx(0.006364, abs=1e-6)
    assert v["pooled_within_sd"] == pytest.approx(np.sqrt(np.mean([a[s].var(ddof=1) for s in a])))
    # balanced: equals the usual pooled within-group variance sum SS / sum (k-1)
    ss = sum(((a[s] - a[s].mean()) ** 2).sum() for s in a)
    assert v["pooled_within_sd"] == pytest.approx(np.sqrt(ss / (3 * 4)))
    m = np.array([a[s].mean() for s in a])
    assert v["sd_between_seed_means"] == pytest.approx(m.std(ddof=1))
    assert v["overall_mean_delta"] == pytest.approx(m.mean())
    assert v["between_seed_variance_component"] == pytest.approx(max(0, m.var(ddof=1) - v["pooled_within_sd"] ** 2 / 5))
    assert v["n_fc_worse"][1] == 4 and v["n_fc_better"][1] == 1 and v["n_fc_worse_total"] == 9


def test_flat_pseudoreplication_rejected():
    flat = [x for s in D3 for x in D3[s].values()]
    with pytest.raises(TypeError):
        CS.variance_summary(flat)
    with pytest.raises(TypeError):
        CS.hierarchical_bootstrap(flat)
    with pytest.raises(ValueError):
        CS.variance_summary({0: {0: 1.0, 1: 2.0}, 1: {0: 1.0}})              # unbalanced


def test_hierarchical_bootstrap():
    b = CS.hierarchical_bootstrap(D3, n_boot=4000, seed=1)
    assert b["point"] == pytest.approx(CS.variance_summary(D3)["overall_mean_delta"])
    assert b["ci95"][0] <= b["point"] <= b["ci95"][1]
    assert b == CS.hierarchical_bootstrap(D3, n_boot=4000, seed=1)                # fixed seed
    assert b["n_distinct_seed_resamples"] <= 10                                    # C(3+3-1, 3) multisets
    const = {s: {g: 0.003 for g in range(5)} for s in range(3)}
    assert CS.hierarchical_bootstrap(const, n_boot=500)["ci95"] == pytest.approx([0.003, 0.003])
    # seed-level resampling: one seed far away -> wider interval than within-seed noise alone would give
    far = {0: {g: 0.0 for g in range(5)}, 1: {g: 0.0 for g in range(5)}, 2: {g: 0.03 for g in range(5)}}
    ci = CS.hierarchical_bootstrap(far, n_boot=4000)["ci95"]
    assert ci[0] == pytest.approx(0.0) and ci[1] == pytest.approx(0.03)


def test_decision_rule():
    pos = {s: dict(zip(range(5), [0.010, 0.012, 0.008, 0.011, 0.009])) for s in range(3)}
    assert CS.decide(pos)[0] == "FC-PPL-PENALTY-ROBUST"
    neg = {s: {g: -v for g, v in pos[s].items()} for s in range(3)}
    assert CS.decide(neg)[0] == "FC-PPL-ADVANTAGE-EXPLORATORY"
    conflict = {0: pos[0], 1: neg[1], 2: dict(zip(range(5), [0.001, -0.001, 0.0005, -0.0005, 0.0]))}
    cat, det = CS.decide(conflict)
    assert cat == "ALLOCATION-SEED-DEPENDENT" and det["confident_positive_seeds"] == [0]
    noisy = {s: dict(zip(range(5), [-0.004, 0.0076, -0.0059, -0.0084, 0.0011])) for s in range(3)}
    assert CS.decide(noisy)[0] == "GROUPING-VARIATION-DOMINATES"
    # consistent but small sign majority not met -> ambiguous fallback
    amb = {s: dict(zip(range(5), [0.012, 0.011, -0.001, -0.001, -0.001])) for s in range(3)}
    cat, det = CS.decide(amb)
    assert cat == "GROUPING-VARIATION-DOMINATES" and det["ambiguous"]
    assert CS.decide(pos, blocked=True)[0] == "RUN-BLOCKED"


def test_bootstrap_levels_documented():
    assert "CONDITIONAL" in CS.UNCERTAINTY_LEVELS["window_bootstrap"]
    assert "not combined" in CS.UNCERTAINTY_LEVELS["combination"]


# ------------------------------------------------------------- backward compatibility / reconstruction matching
def test_seed0_backward_compatibility():
    mr = os.path.join(ROOT, "runs_multigroup_robustness", "protocol.json")
    if not os.path.exists(mr):
        pytest.skip("seed-0 study absent")
    pr = json.load(open(mr))
    for g in XG.GSEEDS:
        assert pr["groupings"][f"G{g}"]["batches"] == MG.grouping_batches(pr["union_calibration_ids"], g)
    res = json.load(open(os.path.join(ROOT, "runs_multigroup_robustness", "ppl_results.json")))
    assert res["delta"]["G20261112"] == pytest.approx(0.007601, abs=1e-6)
    rec = json.load(open(os.path.join(ROOT, "runs_multigroup_robustness", "reconstruction_results.json")))
    assert rec["evaluator"].startswith("E3 held-out") and "2128-2255" in rec["evaluator"]
    if os.path.exists(PROTO):
        assert "2128-2255" in json.load(open(PROTO))["reconstruction"]
