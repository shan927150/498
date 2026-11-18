"""CPU tests for FP64-canonical Hessian accumulation, the loaded-Hessian accumulator and the synthesis statistics."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_calib import fp64_canonical as FC   # noqa: E402
from moe_calib import profiler as PR          # noqa: E402
from moe_calib.hessian import BlockHessianAccumulator   # noqa: E402
from moe_calib.hessian_order_null import Fp64Accumulator   # noqa: E402
from moe_calib.gptq import gptq_quantize      # noqa: E402


def _contribs(n=24, t=40, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(t, d, generator=g).to(torch.bfloat16) * 30 for _ in range(n)]


def _acc(X, order, blocks, d=32):
    a = Fp64Accumulator([(1, 0, "gu")], dict(gu=d, down=d), 4, "cpu")
    for i in order:
        a.add((1, 0, "gu"), X[i], blocks[i])
    return a.sums((1, 0, "gu"))


def test_fp64_order_invariant_fp32_not_necessarily():
    X = _contribs()
    blocks = [(i * 4) // len(X) for i in range(len(X))]
    o0 = list(range(len(X)))
    o1 = list(np.random.default_rng(1).permutation(len(X)))
    h32a, h64a = _acc(X, o0, blocks)
    h32b, h64b = _acc(X, o1, blocks)
    assert torch.equal(h64a, h64b) and torch.equal(h64a.float(), h64b.float())
    assert h64a.dtype == torch.float64 and torch.equal(h64a, h64a.T)
    assert float((h32a.double() - h64a).norm() / h64a.norm()) < 1e-6


def test_canonical_cast_once():
    Hb = torch.randn(4, 8, 8, dtype=torch.float64)
    H64, H32 = FC.canonical_cast(Hb)
    assert torch.equal(H64, Hb.sum(0)) and H32.dtype == torch.float32 and torch.equal(H32, H64.float())
    with pytest.raises(AssertionError):
        FC.canonical_cast(Hb.float())


def test_loaded_accumulator_counts_and_serves():
    store = {"1.0.gu": torch.eye(4) * 2, "1.0.down": torch.eye(3)}
    with FC.serve_hessians(store):
        assert PR.BlockHessianAccumulator is FC.LoadedHessianAccumulator
        a = PR.BlockHessianAccumulator([1], 1, 4, 3, K=4, device="cpu")
        a.add((1, 0, "gu"), torch.ones(5, 4), 2)
        a.add((1, 0, "gu"), torch.ones(0, 4), 1)
        assert a.tokens((1, 0, "gu")) == 5 and a.raw_n[(1, 0, "gu")].tolist() == [0, 0, 5, 0]
        assert torch.equal(a.H_sum((1, 0, "gu"), [0, 1, 2, 3]), store["1.0.gu"])
        with pytest.raises(ValueError):
            a.H_sum((1, 0, "gu"), [0])
        with pytest.raises(ValueError):
            a.add((1, 0, "gu"), torch.ones(2, 4), 0, weight=2.0)
    assert PR.BlockHessianAccumulator is BlockHessianAccumulator and FC.LoadedHessianAccumulator.STORE is None
    with pytest.raises(RuntimeError):
        FC.LoadedHessianAccumulator([1], 1, 4, 3)


def test_serve_restores_on_error():
    with pytest.raises(KeyError):
        with FC.serve_hessians({}):
            raise KeyError
    assert PR.BlockHessianAccumulator is BlockHessianAccumulator


def test_identical_hessian_gives_identical_weights():
    X = torch.randn(300, 128, generator=torch.Generator().manual_seed(3))
    H = (X.T @ X).double().float()
    W = torch.randn(16, 128, generator=torch.Generator().manual_seed(4))
    perm = torch.argsort(torch.diag(H), descending=True)
    a = gptq_quantize(W, H.clone(), 4, 128, 1.0, actorder_override=perm)
    b = gptq_quantize(W, H.clone(), 4, 128, 1.0, actorder_override=perm)
    assert torch.equal(a, b)


def test_hierarchical_bootstrap_and_pooled_sd():
    d = {0: [1.0, 1.0, 1.0], 1: [1.0, 1.0, 1.0]}
    assert FC.hierarchical_bootstrap(d, n_boot=200) == (1.0, 1.0)
    d2 = {0: [0.0, 2.0], 1: [10.0, 12.0]}
    assert FC.pooled_within_sd(d2) == pytest.approx(np.sqrt(2.0))
    lo, hi = FC.hierarchical_bootstrap(d2, n_boot=2000)
    assert 0.0 <= lo < hi <= 12.0
    assert FC.ppl_from_nll(np.log(9.0) * 10, 10) == pytest.approx(9.0)


def test_classifiers():
    assert FC.classify_canonical(False, True, True) == "CANONICAL-VALIDATED"
    assert FC.classify_canonical(False, False, True) == "CANONICAL-NOT-ORDER-INVARIANT"
    assert FC.classify_canonical(False, True, False) == "CANONICAL-NOT-ORDER-INVARIANT"
    assert FC.classify_canonical(True, True, True) == "CANONICAL-BLOCKED"
    assert FC.classify_execution(True, 0.5, True) == "EXECUTION-VARIABILITY-MATERIAL"
    assert FC.classify_execution(False, 2.5, True) == "EXECUTION-VARIABILITY-MATERIAL"
    assert FC.classify_execution(False, 1.0, True) == "EXECUTION-VARIABILITY-LIMITED"
    assert FC.classify_execution(True, 3.0, False) == "EXECUTION-VARIABILITY-LIMITED"
