"""CPU tests for shared covariance-aware selection (full E3): stacked objective, warm-start optimizer, MVP parity,
validation and the frozen decision rule. Synthetic sketches only (no model)."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_calib import expertwise_calibration as EW          # noqa: E402
from moe_calib import shared_covariance_selection as SCS    # noqa: E402


def _synthetic(N=70, n_layers=2, E=3, k=5, seed=0, zero_frac=0.25):
    """Stacked terms ordered (layer, expert, proj); returns M [T,N,V] fp64, n [T,N], w [T] plus the MVP-style dicts
    for a 'flattened expert' view (key = layer*E + e)."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    iu, w = EW.svech_index(k)
    Ms, ns, Md, nd = [], [], {}, {}
    for L in range(n_layers):
        for e in range(E):
            cnt = rng.integers(1, 10, size=N)
            cnt[rng.random(N) < zero_frac] = 0
            key = L * E + e
            nd[key] = torch.as_tensor(cnt, dtype=torch.float64)
            for p in EW.PROJS:
                rows = []
                for s in range(N):
                    X = torch.randn(int(cnt[s]), k, generator=g, dtype=torch.float64) * (1 + 0.4 * (s % 7)) * (1 + key)
                    rows.append(EW.svech(X.T @ X, iu, w))
                Md[(key, p)] = torch.stack(rows)
                Ms.append(Md[(key, p)])
                ns.append(nd[key])
    T = len(Ms)
    return torch.stack(Ms), torch.stack(ns), torch.full((T,), 1.0 / T, dtype=torch.float64), Md, nd, \
        list(range(n_layers * E))


def test_stacked_objective_equals_mvp_objective():
    M, n, w, Md, nd, keys = _synthetic()
    obj = SCS.StackedObjective(M, n, w)
    ref = EW.build_terms(Md, nd, keys, shared=True)              # MVP: weight 1/(2|E|) per term = 1/T
    for sel in ([], [4], [1, 5, 9, 33, 60], list(range(0, 70, 3))):
        assert obj.exact(sel) == pytest.approx(ref.exact(sel), rel=1e-12, abs=1e-14)
    assert obj.exact([]) == pytest.approx(1.0, abs=1e-9)
    assert obj.exact(list(range(70))) == pytest.approx(0.0, abs=1e-9)


def test_closed_form_scores_match_exact():
    M, n, w, *_ = _synthetic(seed=1)
    obj = SCS.StackedObjective(M, n, w)
    for sel in ([], [3], [2, 8, 15, 40, 61]):
        st = obj.state(sel)
        sc = obj.add_scores(st)
        for c in (0, 7, 30, 69):
            if c not in sel:
                assert float(sc[c]) == pytest.approx(obj.exact(sel + [c]), rel=1e-9, abs=1e-12)
        for i in sel[:2]:
            sw = obj.swap_scores(st, i)
            for c in (1, 22, 68):
                if c not in sel:
                    assert float(sw[c]) == pytest.approx(obj.exact([x for x in sel if x != i] + [c]), rel=1e-9)


def test_fp32_storage_close_to_fp64():
    M, n, w, *_ = _synthetic(seed=2)
    a = SCS.StackedObjective(M, n, w)
    b = SCS.StackedObjective(M.float(), n, w)
    sel = [0, 3, 9, 12, 50]
    assert b.exact(sel) == pytest.approx(a.exact(sel), rel=1e-6)
    assert torch.allclose(b.add_scores(b.state(sel)), a.add_scores(a.state(sel)), rtol=1e-5)


def test_reproduces_mvp_optimizer_without_warm_start():
    """With W = [] and fp64 storage the stacked optimizer must make the MVP's choices (same RNG stream)."""
    M, n, w, Md, nd, keys = _synthetic(N=60, seed=3)
    B, seed = 10, 4
    obj = SCS.StackedObjective(M, n, w)
    mine, info = SCS.select_shared(obj, [], B, seed, sample_frac=0.2, max_passes=4)
    mvp_obj = EW.build_terms(Md, nd, keys, shared=True)
    mvp, mvp_info = EW.covariance_select(mvp_obj, B, EW._rng(seed, EW.COND_CODE["SC"]), 0.2, 4)
    assert sorted(mine) == mvp
    assert info["final_objective"] == pytest.approx(mvp_info["final_objective"], rel=1e-12)


def test_warm_start_fixed_monotone_deterministic():
    M, n, w, *_ = _synthetic(N=80, seed=5)
    obj = SCS.StackedObjective(M, n, w)
    warm = [70, 3, 41, 12]
    a, ia = SCS.select_shared(obj, warm, 14, seed=0, sample_frac=0.25)
    b, ib = SCS.select_shared(SCS.StackedObjective(M, n, w), warm, 14, seed=0, sample_frac=0.25)
    c, _ = SCS.select_shared(obj, warm, 14, seed=1, sample_frac=0.25)
    assert a == b and a != c
    assert a[:4] == warm and len(set(a)) == 14
    tr = ia["trace"]
    assert all(y <= x for x, y in zip(tr, tr[1:])) and ia["final_objective"] <= ia["greedy_objective"]
    assert ia["accepted"] == len(tr) - 1
    SCS.validate_shared(a, warm, 14, 80, forbidden=[80, 81])


def test_swap_never_removes_warm_slots():
    M, n, w, *_ = _synthetic(N=50, seed=6)
    obj = SCS.StackedObjective(M, n, w)
    warm = list(range(6))                     # deliberately arbitrary
    g, _ = SCS.greedy_from_warm(obj, warm, 12, np.random.default_rng(0), sample_frac=1.0)
    sel, tr, info = SCS.swap_refine_fixed(obj, g, 6, np.random.default_rng(1), max_passes=5)
    assert sel[:6] == warm and len(set(sel)) == 12


def test_validate_shared_failures():
    warm = [5, 1]
    SCS.validate_shared([5, 1, 2, 3], warm, 4, 10)
    with pytest.raises(AssertionError):
        SCS.validate_shared([5, 1, 2, 2], warm, 4, 10)
    with pytest.raises(AssertionError):
        SCS.validate_shared([1, 5, 2, 3], warm, 4, 10)          # warm order lost
    with pytest.raises(AssertionError):
        SCS.validate_shared([5, 1, 2, 12], warm, 4, 10)          # outside pool
    with pytest.raises(AssertionError):
        SCS.validate_shared([5, 1, 2, 3], warm, 4, 10, forbidden=[3])
    with pytest.raises(AssertionError):
        SCS.validate_shared([5, 1, 2], warm, 4, 10)


def test_zero_pool_token_term_rejected():
    M, n, w, *_ = _synthetic(seed=7)
    n = n.clone()
    n[0] = 0
    with pytest.raises(ValueError):
        SCS.StackedObjective(M, n, w)


def test_gini_overlap_relimp():
    assert SCS.gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert SCS.gini([0, 0, 0, 4]) == pytest.approx(0.75)
    o = SCS.overlap([1, 2, 3, 4], [3, 4, 5, 6])
    assert o["intersection"] == 2 and o["jaccard"] == pytest.approx(2 / 6)
    assert SCS.relative_improvement(0.99, 1.0) == pytest.approx(0.01)


@pytest.mark.parametrize("r_seed,r_layer,r_conf,grouping,label", [
    ([0.001, 0.002, 0.003], {1: .01, 8: .01, 16: .01, 23: .01}, [.01] * 3, None, "NO-GAIN"),
    ([0.006, 0.007, 0.008], {1: .01, 8: .01, 16: .01, 23: -.01}, [.01] * 3, None, "SUPPORTED"),
    ([0.006, -0.001, 0.02], {1: .01, 8: .01, 16: .01, 23: .01}, [.01] * 3, None, "INCONCLUSIVE"),
    ([0.006, 0.007, 0.008], {1: .01, 8: -.01, 16: .01, 23: -.01}, [.01] * 3, None, "INCONCLUSIVE"),
    ([0.006, 0.007, 0.008], {1: .01, 8: .01, 16: .01, 23: .01}, [-.01] * 3, None, "INCONCLUSIVE"),
    ([0.006, 0.007, 0.008], {1: .01, 8: .01, 16: .01, 23: .01}, [.01] * 3,
     dict(seed_means=[1, 2, 3], ci95=[0.5, 3]), "ROBUST-SUPPORTED"),
    ([0.006, 0.007, 0.008], {1: .01, 8: .01, 16: .01, 23: .01}, [.01] * 3,
     dict(seed_means=[1, 2, 3], ci95=[-0.5, 3]), "SUPPORTED"),
    ([0.006, 0.007, 0.008], {1: .01, 8: .01, 16: .01, 23: .01}, [.01] * 3,
     dict(seed_means=[1, -2, 3], ci95=[-0.5, 3]), "INCONCLUSIVE"),
    ([-0.01, -0.02, -0.01], {1: -.01, 8: -.01, 16: -.01, 23: -.01}, [-.01] * 3, None, "NO-GAIN"),
])
def test_decision_rule(r_seed, r_layer, r_conf, grouping, label):
    assert SCS.decide(r_seed, r_layer, r_conf, grouping)[0] == label
