"""Tests for the EXPERIMENTAL affinity exponent q in {0, 1, 2} (H_q = X^T diag(c^q) X) and the c^2 pilot helpers."""
import inspect
import os
import sys

import numpy as np
import pytest
import torch

from conftest import E, LAYERS, tiny_model
from moe_calib import agq as AG
from moe_calib import gptq as GQ
from moe_calib import model_utils as MU
from moe_calib.hessian import BlockHessianAccumulator
from moe_calib.profiler import Profiler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
L0 = LAYERS[0]


def _prod(X, w=None, chunks=None):
    acc = BlockHessianAccumulator([0], 1, X.shape[1], X.shape[1], K=1, device="cpu", store_device="cpu")
    chunks = chunks or [X.shape[0]]
    i = 0
    for n in chunks:
        acc.add((0, 0, "gu"), X[i:i + n], 0, row_weight=None if w is None else w[i:i + n])
        i += n
    return acc.H_sum((0, 0, "gu"), [0]).double().numpy()


def _weights(c, q):
    return None if q == 0 else (c if q == 1 else c.pow(q))       # exactly the Profiler's rule


CASES = {
    "unit": lambda g: (torch.randn(30, 12, generator=g), torch.ones(30)),
    "constant": lambda g: (torch.randn(30, 12, generator=g), torch.full((30,), 0.2)),
    "unequal": lambda g: (torch.randn(30, 12, generator=g), torch.rand(30, generator=g)),
    "very_small": lambda g: (torch.randn(30, 12, generator=g), torch.rand(30, generator=g) * 1e-3 + 1e-4),
    "zero_rows": lambda g: (torch.randn(30, 12, generator=g), torch.rand(30, generator=g) * (torch.arange(30) % 4 != 0)),
    "down_dim": lambda g: (torch.randn(30, 7, generator=g), torch.rand(30, generator=g)),
}


@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("name", list(CASES))
def test_oracle_q(name, q):
    X, c = CASES[name](torch.Generator().manual_seed(5))
    ref = AG.weighted_hessian_reference_q(X.numpy(), c.numpy(), q)
    got = _prod(X, _weights(c, q), chunks=[11, 19])
    assert np.allclose(got, ref, rtol=2e-5, atol=2e-5 * np.abs(ref).max() + 1e-30)
    if q == 2 and name in ("unequal", "very_small"):
        c4 = AG.weighted_hessian_reference_q(X.numpy(), c.numpy(), 4)
        c1 = AG.weighted_hessian_reference_q(X.numpy(), c.numpy(), 1)
        assert not np.allclose(got, c4, rtol=1e-2) and not np.allclose(got, c1, rtol=1e-2)


def test_q_permutation_and_dispatch_order():
    g = torch.Generator().manual_seed(9)
    X, c = torch.randn(40, 10, generator=g), torch.rand(40, generator=g)
    perm = torch.randperm(40, generator=g)
    for q in (1, 2):
        assert np.allclose(_prod(X[perm], _weights(c[perm], q)), _prod(X, _weights(c, q)), rtol=1e-5, atol=1e-6)
        assert not np.allclose(_prod(X, _weights(c[perm], q)), _prod(X, _weights(c, q)), rtol=1e-3)


def test_q2_constant_invariance_and_small_c_no_dead_columns():
    g = torch.Generator().manual_seed(4)
    X = torch.randn(300, 32, generator=g); W = torch.randn(8, 32, generator=g)
    H0 = torch.from_numpy(_prod(X)).float()
    H2 = torch.from_numpy(_prod(X, torch.full((300,), 0.5) ** 2)).float()     # c = 0.5 -> c^2 = 0.25: exact scale
    assert torch.equal(GQ.gptq_quantize(W, H0, group_size=16, percdamp=1.0), GQ.gptq_quantize(W, H2, group_size=16, percdamp=1.0))
    small = torch.rand(300, generator=g) * 0.03 + 0.005                          # realistic c (0.005-0.035) -> c^2 ~1e-5
    Hs = torch.from_numpy(_prod(X, small ** 2)).float()
    info = {}
    q = GQ.gptq_quantize(W, Hs, group_size=16, percdamp=1.0, info=info)
    assert info["n_dead"] == 0 and torch.isfinite(q).all() and (torch.diag(Hs) > 0).all()


def test_profiler_power_semantics():
    tm, _ = tiny_model()
    with pytest.raises(ValueError):
        Profiler(tm, L0, hessian_weighting="router_affinity", affinity_power=3)
    with pytest.raises(ValueError):
        Profiler(tm, L0, affinity_power=2)                                        # q = 0 is hessian_weighting="none"
    assert inspect.signature(Profiler).parameters["affinity_power"].default == 1


def test_profiler_q_paths(cache_dir):
    from moe_calib.data_utils import load_layer_cache
    tm, _ = tiny_model()
    cache = load_layer_cache(cache_dir, L0)
    ids = [4, 1, 6]                                                             # multiple sequences
    base = Profiler(tm, L0, K=1, hessian_weighting="router_affinity", collect_affinity=True).run(cache, ids)
    p1 = Profiler(tm, L0, K=1, hessian_weighting="router_affinity", affinity_power=1).run(cache, ids)
    p2 = Profiler(tm, L0, K=1, hessian_weighting="router_affinity", affinity_power=2, verify_alignment=True).run(cache, ids)
    p0 = Profiler(tm, L0, K=1).run(cache, ids)
    blk = MU.moe_block(tm, L0)
    X = {e: {"gu": [], "down": []} for e in range(E)}
    hooks = []
    for e in range(E):
        hooks.append(blk.experts[e].gate_proj.register_forward_pre_hook(lambda m, a, e=e: X[e]["gu"].append(a[0].double())))
        hooks.append(blk.experts[e].down_proj.register_forward_pre_hook(lambda m, a, e=e: X[e]["down"].append(a[0].double())))
    try:
        Profiler(tm, L0, K=1, collect_hessian=False).run(cache, ids)
    finally:
        for h in hooks:
            h.remove()
    for e in range(E):
        for proj in ("gu", "down"):                                             # all three projection types (gu = gate/up)
            k = (L0, e, proj)
            assert torch.equal(p1.acc.H[k], base.acc.H[k])                       # q = 1 is the existing AGQ path, bitwise
            lst = base.affinity.get((L0, e), [])
            if not lst:                                                          # empty expert: zero Hessian everywhere
                assert not p2.acc.H[k].any() and not p0.acc.H[k].any()
                continue
            c = torch.cat([c_ for *_, c_ in lst]).double().numpy()
            Xe = torch.cat(X[e][proj]).numpy()
            ref = AG.weighted_hessian_reference_q(Xe, c, 2)
            assert np.allclose(p2.acc.H[k][0].double().numpy(), ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())
            assert torch.equal(p0.acc.raw_n[k], p2.acc.raw_n[k])               # same calibration tokens in every q


def test_effective_sample_size_q():
    c = np.array([.5, .1, .1, .1])
    assert AG.effective_sample_size_q(c, 0) == pytest.approx(4)
    assert AG.effective_sample_size_q(c, 1) == pytest.approx(AG.effective_sample_size(c))
    assert AG.effective_sample_size_q(c, 2) < AG.effective_sample_size_q(c, 1) < 4


def test_decomposition_reconstructs_total():
    g = torch.Generator().manual_seed(3)
    T, d, k, Ex = 9, 6, 4, 5
    sel = torch.stack([torch.randperm(Ex, generator=g)[:k] for _ in range(T)])
    vals = torch.rand(T, k, generator=g)
    rows = AG.dispatch_rows(sel, vals, Ex)
    deltas = {e: torch.randn(len(tx), d, generator=g) for e, (tx, _, _) in rows.items()}
    V = AG.slot_contributions(T, d, k, rows, deltas, "cpu")
    dc = AG.decompose(V)
    dy = torch.zeros(T, d)
    D = 0.0
    for e, (tx, slot, c) in rows.items():
        dy.index_add_(0, tx, deltas[e] * c[:, None])
        D += float(((deltas[e] * c[:, None]).double() ** 2).sum())
    assert dc["total"] == pytest.approx(float((dy.double() ** 2).sum()), rel=1e-6)
    assert dc["diag"] == pytest.approx(D, rel=1e-9)
    assert dc["total"] == pytest.approx(dc["diag"] + dc["cross"], rel=1e-9)


def test_classify_c2():
    R = dict(surrogate_gain_rel=0.01, max_layer_regression_rel=0.005)
    assert AG.classify_c2(R, -0.02, {8: -0.001, 16: 0.002}, True, True)["verdict"] == "C2-PILOT-PROMISING"
    assert AG.classify_c2(R, -0.02, {8: 0.001, 16: 0.002}, True, True)["verdict"] == "C2-SURROGATE-ONLY"
    assert AG.classify_c2(R, -0.02, {8: -0.01, 16: 0.006}, True, True)["verdict"] == "C2-REGRESSION"
    assert AG.classify_c2(R, -0.005, {8: -0.001, 16: 0.0}, True, True)["verdict"] == "C2-NO-GAIN"
    assert AG.classify_c2(R, -0.02, {8: -0.01, 16: -0.01}, False, True)["verdict"] == "C2-BLOCKED"
    assert AG.classify_c2(R, -0.02, {8: -0.01, 16: -0.01}, True, True, unstable=True)["verdict"] == "C2-REGRESSION"
    assert AG.classify_c2(R, -0.02, {8: -0.01, 16: -0.01}, True, False)["verdict"] == "C2-SURROGATE-ONLY"


def test_expert_selection_rule():
    import agq_c2_pilot as CP
    if not os.path.exists(os.path.join(ROOT, "runs_ebss_official_validation", "logs", "capture_routing_nll.npz")):
        pytest.skip("routing counts absent")
    sel = CP.select_experts()
    for L, s in sel.items():
        cnt = {int(k): v for k, v in s["counts"].items()}
        order = sorted(cnt, key=lambda e: (cnt[e], e))
        assert s["low"] == order[:2] and s["high"] == order[-2:] and s["median"] == order[29:31]
        assert len(set(s["low"] + s["median"] + s["high"])) == 6
