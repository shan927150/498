"""CPU tests for the calibration-domain ablation helpers (moe_calib/calibration_domain.py)."""
import json
import os
import random
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_calib import calibration_domain as CD          # noqa: E402
from moe_calib import multigroup_robustness as MG       # noqa: E402

ROUT = os.path.join(ROOT, "runs_typicality_matched", "routing_profile_v2.npz")
HAVE = os.path.exists(ROUT) and os.path.exists(os.path.join(ROOT, "runs_partial_ppl", "selected_set_manifest.json"))


def test_partition_and_pool_sampling():
    assert CD.partition_windows(512 * 7 + 511, 512) == 7
    a = CD.sample_pool(5000, 2000, 20261117)
    assert a == CD.sample_pool(5000, 2000, 20261117) and len(set(a)) == 2000 and a == sorted(a)
    assert a != CD.sample_pool(5000, 2000, 1)
    with pytest.raises(ValueError):
        CD.sample_pool(1999, 2000, 0)


def test_windows_are_non_overlapping_slices():
    enc = np.arange(10 * 4 + 3)
    w = CD.windows_from_ids(enc, [0, 2, 9], 4)
    assert w.tolist() == [[0, 1, 2, 3], [8, 9, 10, 11], [36, 37, 38, 39]]


def test_ngram_overlap_detects_leakage():
    rng = np.random.default_rng(0)
    pool = rng.integers(0, 50000, size=(20, 64))
    clean = rng.integers(0, 50000, size=(4, 64))
    r = CD.ngram_overlap(pool, clean, 8)
    assert r["exact_window_matches"] == 0 and r["test_ngrams_in_pool"] == 0
    leak = np.concatenate([pool[3:4], clean[:1]])
    r = CD.ngram_overlap(pool, leak, 8)
    assert r["exact_window_matches"] == 1 and r["test_ngrams_in_pool"] == 64 - 8 + 1


@pytest.mark.skipif(not HAVE, reason="stored C4 artifacts not available")
def test_warm_start_and_fr_rules_reproduce_stored_c4_selections():
    z = np.load(ROUT)
    ids = z["pool_ids"].tolist()
    typ, hal, _ = CD.typicality_halves(z["C"], ids, 4, 60)
    pools = json.load(open(os.path.join(ROOT, "runs_typicality", "pools.json")))["pools"]
    assert typ == pools["typical_half"]["ids"] and hal == pools["random_half"]["ids"]
    ws = json.load(open(os.path.join(ROOT, "runs_typicality_matched", "warm_starts.json")))
    man = json.load(open(os.path.join(ROOT, "runs_partial_ppl", "selected_set_manifest.json")))
    for s in (0, 1, 2):
        W = CD.warm_start(typ, hal, s)["ids"]
        assert W == ws["seeds"][str(s)]["ids"]
        fr = CD.fr_selection(ids, W, 128, s)
        assert fr == man["configs"][f"FR_s{s}"]["chosen"]
        assert all(CD.validate_selection(fr, W, 128, ids).values())


def test_validate_selection_flags():
    v = CD.validate_selection([1, 2, 2], [1], 3, range(5))
    assert not v["n_unique_B"]
    v = CD.validate_selection([2, 1, 3], [1], 3, range(5))
    assert not v["warm_prefix"]
    v = CD.validate_selection([1, 2, 9], [1], 3, range(5))
    assert not v["in_pool"]


def test_grouping_rule_is_shared_and_paired():
    union = sorted(random.Random(3).sample(range(2000), 230))
    a, b = MG.grouping_batches(union, 20261111), MG.grouping_batches(union, 20261111)
    assert a == b and all(MG.validate_grouping(a, union).values())
    assert a != MG.grouping_batches(union, 20261112)


def test_effects_and_interaction_signs():
    nll = {("C4", "FR"): 2.0, ("C4", "SC"): 1.9, ("WT", "FR"): 1.5, ("WT", "SC"): 1.2}
    e = CD.effects(nll)
    assert e["delta_SC_C4"] == pytest.approx(-0.1) and e["delta_SC_WT"] == pytest.approx(-0.3)
    assert e["delta_domain_FR"] == pytest.approx(-0.5) and e["delta_domain_SC"] == pytest.approx(-0.7)
    assert e["interaction"] == pytest.approx(-0.2)                  # SC helps more under WT
    # interaction == domain effect difference (algebraic identity)
    assert e["interaction"] == pytest.approx(e["delta_domain_SC"] - e["delta_domain_FR"])


def test_summary_and_label_rule():
    s = CD.summarize([-0.002, -0.003, -0.001])
    assert s["n_negative"] == 3 and s["mean"] == pytest.approx(-0.002) and s["sd"] == pytest.approx(0.001)
    assert CD.label(s, 0.0015) == "CONSISTENT-NEGATIVE"
    assert CD.label(s, 0.0025) == "NOT-DISTINGUISHABLE"             # same sign but below the reference SD
    assert CD.label(CD.summarize([-0.01, 0.01, -0.02]), 1e-6) == "NOT-DISTINGUISHABLE"
    assert CD.label(CD.summarize([0.01, 0.02, 0.03]), 0.001) == "CONSISTENT-POSITIVE"
