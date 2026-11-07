"""CPU tests for the activation-aware swap pilot."""
import json
import os
import random
import sys

import numpy as np
import pytest
import torch

from conftest import E, LAYERS, tiny_model
from moe_calib import activation_swap as AS
from moe_calib.data_utils import load_layer_cache, load_meta, split_ids
from moe_calib.profiler import Profiler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))


# ------------------------------------------------------------- extraction (tiny model)

def test_routed_token_extraction_matches_existing_hessian_and_gate(cache_dir):
    model, _ = tiny_model()
    L = LAYERS[0]
    experts = [1, 3, 6]
    ids = [0, 3, 7, 11]
    cache = load_layer_cache(cache_dir, L)
    P = AS.DiagProfiler(model, L, experts).run(cache, ids)
    d = model.config.hidden_size
    n, g = P.arrays(ids, d)
    for i, s in enumerate(ids):                         # per sequence vs the existing accumulator
        H = Profiler(model, L, K=1).run(cache, [s])
        for k, e in enumerate(experts):
            assert n[i, k] == H.acc.tokens((L, e, "gu"))
            assert n[i, k] == P.seq_counts.get(s, {}).get((L, e), 0)      # gate-hook routing count
            np.testing.assert_allclose(g[i, k], H.acc.H_sum((L, e, "gu"), [0]).diagonal().double().numpy(),
                                       rtol=1e-5, atol=1e-6)
    # only routed tokens: totals are far below all tokens entering the layer
    T = cache.shape[1]
    assert n.sum() < len(ids) * T * len(experts)
    assert g.dtype == np.float64


def test_expert_id_mapping_and_top_experts_tiebreak():
    C = np.zeros((5, 2 * 4))
    C[:, 4:] = [[3, 3, 1, 1]] * 5                       # layer index 1: experts 0 and 1 tie
    top, tot = AS.top_experts(C, 1, 4, 3)
    assert top == [0, 1, 2] and tot[0] == tot[1] == 15


# ------------------------------------------------------------- statistics

def test_token_weighted_moments_not_averaged_variances():
    n = np.array([[1.0], [3.0]])
    g = np.array([[[2.0, 0.0]], [[6.0, 3.0]]])
    h, N = AS.set_moments(n, g, [0, 1])
    np.testing.assert_allclose(h, [[8 / 4, 3 / 4]])       # sum g / sum n, not mean of per-seq means
    assert not np.allclose(h, [[(2 / 1 + 6 / 3) / 2, (0 + 1) / 2]])
    with pytest.raises(ValueError):
        AS.set_moments(np.array([[0.0]]), np.zeros((1, 1, 2)), [0])


def test_incremental_swap_equals_full_recompute():
    rng = np.random.default_rng(0)
    n = rng.integers(1, 50, size=(30, 4)).astype(float)
    g = rng.gamma(2.0, size=(30, 4, 16)) * n[:, :, None]
    href = rng.gamma(2.0, size=(4, 16))
    S0 = list(range(10))
    Ns, Gs = n[S0].sum(0), g[S0].sum(0)
    for o, i in ((3, 20), (0, 29), (9, 15)):
        D_inc, De_inc = AS.swap_mismatch(Ns, Gs, n[o], g[o], n[i], g[i], href)
        S = [i if s == o else s for s in S0]
        D_full, De_full = AS.mismatch(AS.set_moments(n, g, S)[0], href)
        assert D_inc == pytest.approx(D_full, rel=1e-12)
        np.testing.assert_allclose(De_inc, De_full, rtol=1e-12)


def test_synthetic_swaps_with_known_opposite_effects():
    """h_ref = 1 everywhere. S0 holds one sequence with h = 3 (too large). Swapping
    it for an h = 1 sequence must improve matching; swapping a matching sequence
    for an h = 5 one must worsen it."""
    d = 8
    href = np.ones((1, d))
    seqs = {"a": 1.0, "b": 1.0, "hi": 3.0, "good": 1.0, "bad": 5.0}
    keys = list(seqs)
    n = np.full((len(keys), 1), 10.0)
    g = np.stack([np.full((1, d), 10.0 * seqs[k]) for k in keys])
    S0 = [0, 1, 2]                                       # a, b, hi
    D0, _ = AS.mismatch(AS.set_moments(n, g, S0)[0], href)
    Ns, Gs = n[S0].sum(0), g[S0].sum(0)
    D_good, _ = AS.swap_mismatch(Ns, Gs, n[2], g[2], n[3], g[3], href)   # hi -> good
    D_bad, _ = AS.swap_mismatch(Ns, Gs, n[0], g[0], n[4], g[4], href)    # a -> bad
    assert D0 - D_good > 0 and D0 - D_bad < 0
    assert D_good == pytest.approx(0.0)


# ------------------------------------------------------------- swaps / proposals

def test_swapped_set_exact_B_unique_one_out_one_in():
    S0 = list(range(100, 228))
    S = AS.swapped(S0, 150, 999)
    assert len(S) == 128 == len(set(S)) and set(S0) - set(S) == {150} and set(S) - set(S0) == {999}
    assert S.index(999) == S0.index(150)                  # in place: other positions unchanged
    with pytest.raises(ValueError):
        AS.swapped(S0, 150, 101)                          # incoming already in the set
    with pytest.raises(ValueError):
        AS.swapped(S0, 5, 999)                            # outgoing not in the set


def test_pick_unique_deterministic_and_prefers_unused_ids():
    ranked = [dict(out=1, **{"in": 10}), dict(out=1, **{"in": 11}), dict(out=2, **{"in": 10}),
              dict(out=3, **{"in": 12}), dict(out=4, **{"in": 13})]
    a, ua = AS.pick_unique(ranked, 3, set(), set())
    b, ub = AS.pick_unique(ranked, 3, set(), set())
    assert a == b and ua == 3
    assert [(p["out"], p["in"]) for p in a] == [(1, 10), (3, 12), (4, 13)]
    c, uc = AS.pick_unique(ranked[:3], 3, set(), set())   # not enough unique -> fill, recorded
    assert uc == 1 and len(c) == 3


# ------------------------------------------------------------- frozen setup / isolation

def test_frozen_config_reference_isolation_and_s0_if_present():
    import activation_swap as DR
    if not os.path.exists(os.path.join(ROOT, "runs_typicality_matched", "warm_starts.json")):
        pytest.skip("matched runs not present")
    a = type("A", (), dict(frozen=os.path.join(ROOT, "configs/frozen_b4.json"),
                           cache_dir=os.path.expanduser("~/moe_cache"),
                           matched_dir=os.path.join(ROOT, "runs_typicality_matched")))()
    if not os.path.exists(os.path.join(a.cache_dir, "meta.json")):
        pytest.skip("activation cache not present")
    cfg, sp, S0, _ = DR.frozen_and_s0(a)
    assert len(S0) == 128 == len(set(S0)) and all(0 <= s < 2000 for s in S0)
    assert sp["tune"] == list(range(2000, 2128)) and sp["test"] == list(range(2128, 2256))
    assert not set(sp["tune"]) & set(sp["test"]) and not set(S0) & set(sp["tune"])
    assert cfg["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True)


def test_frozen_proposals_reproducible_if_present():
    p = os.path.join(ROOT, "runs_activation_swap", "proposals.json")
    if not os.path.exists(p):
        pytest.skip("proposals not frozen yet")
    pr = json.load(open(p))
    assert len(pr["nine"]) == 9 and len({x["set_hash"] for x in pr["nine"]}) == 9
    for x in pr["nine"]:
        S = AS.swapped(pr["S0"], x["out"], x["in"])
        assert AS.ids_hash(S) == x["set_hash"] and len(S) == 128 == len(set(S))
    assert [x["predicted_gain"] > 0 for x in pr["nine"] if x["group"] == "beneficial"] == [True] * 3
    assert [x["predicted_gain"] < 0 for x in pr["nine"] if x["group"] == "harmful"] == [True] * 3
