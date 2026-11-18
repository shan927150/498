"""CPU tests for layer-pooled Hessian shrinkage: estimators, pooled hook, metrics, frozen screen / verdict rules."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_calib import hessian_shrinkage as HS   # noqa: E402
from moe_calib.gptq import gptq_quantize        # noqa: E402


def _data(d=16, n_e=40, N=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    XL = torch.randn(N, d, generator=g) * torch.linspace(0.5, 2.0, d)
    Xe = XL[:n_e] * 1.3
    return Xe.T @ Xe, float(n_e), XL.T @ XL, float(N)


def test_alpha0_returns_raw_bitwise_and_no_mutation():
    He, ne, HL, N = _data()
    keep = He.clone()
    for tg in HS.TARGETS:
        out = HS.shrink(He, ne, HL, N, 0.0, tg)
        assert torch.equal(out, He) and out is not He
        HS.shrink(He, ne, HL, N, 0.3, tg)
    assert torch.equal(He, keep)


@pytest.mark.parametrize("tg", ["pooled", "identity"])
@pytest.mark.parametrize("a", [0.1, 0.3, 0.5, 1.0])
def test_trace_preserved_symmetric_psd(tg, a):
    He, ne, HL, N = _data()
    H = HS.shrink(He, ne, HL, N, a, tg)
    nm = HS.numerics(H, float(torch.trace(He.double())))
    assert nm["finite"] and nm["psd_ok"] and nm["sym_rel"] < 1e-6 and nm["trace_rel_err"] < 1e-6
    assert H.dtype == torch.float32 and H.shape == He.shape


def test_formula_matches_definition():
    He, ne, HL, N = _data()
    a = 0.3
    Hb, Lb = He.double() / ne, HL.double() / N
    P = Lb * torch.trace(Hb) / torch.trace(Lb)
    ref = ne * ((1 - a) * Hb + a * P)
    assert torch.allclose(HS.shrink(He, ne, HL, N, a, "pooled").double(), ref, rtol=1e-6)
    I = torch.eye(He.shape[0], dtype=torch.float64) * torch.trace(Hb) / He.shape[0]
    assert torch.allclose(HS.shrink(He, ne, HL, N, a, "identity").double(), ne * ((1 - a) * Hb + a * I), rtol=1e-6)
    assert torch.allclose(HS.shrink(He, ne, HL, N, a, "untraced").double(), ne * ((1 - a) * Hb + a * Lb), rtol=1e-6)


def test_invalid_inputs():
    He, ne, HL, N = _data()
    with pytest.raises(ValueError):
        HS.shrink(He, ne, HL, N, 1.5)
    with pytest.raises(ValueError):
        HS.shrink(He, ne, HL, N, 0.3, "other")
    with pytest.raises(ValueError):
        HS.shrink(He, 0.0, HL, N, 0.3)


def test_alpha0_gptq_bitwise():
    He, ne, HL, N = _data(d=128, n_e=300, N=900)
    W = torch.randn(8, 128, generator=torch.Generator().manual_seed(1))
    a = gptq_quantize(W, He, 4, 128, 1.0)
    b = gptq_quantize(W, HS.shrink(He, ne, HL, N, 0.0, "pooled"), 4, 128, 1.0)
    assert torch.equal(a, b)


def test_pooled_hook_counts_each_token_once():
    class Blk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(8, 6, bias=False)

        def forward(self, x):
            return self.gate(x.reshape(-1, 8))
    blk = Blk()
    ph = HS.PooledHessian(8, 2, "cpu", keep_input=True).attach(blk)
    xs = [torch.randn(1, 5, 8) for _ in range(4)]
    for i, x in enumerate(xs):
        ph.cur_block = i // 2
        blk(x)
    ph.detach()
    Xall = torch.cat([x.reshape(-1, 8) for x in xs])
    assert ph.tokens([0, 1]) == 20 and ph.tokens([0]) == 10
    assert torch.allclose(ph.H_sum([0, 1]), Xall.T @ Xall, atol=1e-5)
    assert torch.allclose(ph.H_sum([0]) + ph.H_sum([1]), ph.H_sum([0, 1]))
    assert torch.equal(ph.last_input, xs[-1].reshape(-1, 8))
    blk(xs[0])                                   # detached: no further accumulation
    assert ph.tokens([0, 1]) == 20


def test_mismatch_metrics():
    R = torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.5]))
    assert HS.frob_mismatch(R, R) == 0.0 and HS.diag_mismatch(R, R) == 0.0
    assert HS.frob_mismatch(2 * R, R) == pytest.approx(1.0)
    U = HS.top_eigvecs(R, 2)
    assert HS.subspace_alignment(U, U) == pytest.approx(1.0)
    V = HS.top_eigvecs(torch.diag(torch.tensor([0.5, 1.0, 2.0, 4.0])), 2)
    assert HS.subspace_alignment(U, V) == pytest.approx(0.0, abs=1e-12)


def test_select_experts_by_exposure_with_ties():
    c = np.arange(60)[::-1].copy()
    c[10] = c[11]                                # tie -> lower id first
    sel = HS.select_experts(c)
    assert sel["low"] == [59, 58] and sel["high"] == [1, 0]
    assert len({e for v in sel.values() for e in v}) == 6
    c2 = np.full(60, 5); sel2 = HS.select_experts(c2)
    assert sel2 == dict(low=[0, 1], median=[29, 30], high=[58, 59])


def _screen(raw, pool, idn):
    lay = [8] * 6 + [16] * 6
    return HS.screen_budget(raw, {0.1: pool}, {0.1: idn}, lay)


def test_screen_pass_and_select_lowest():
    raw = np.ones(12)
    lay = [8] * 6 + [16] * 6
    sc = HS.screen_budget(raw, {0.1: raw * 0.96, 0.3: raw * 0.9, 0.5: raw * 0.95},
                          {0.1: raw, 0.3: raw, 0.5: raw}, lay)
    assert sc["passing_alphas"] == [0.1, 0.3, 0.5] and sc["selected_alpha"] == 0.3
    assert HS.stage1_label({16: sc, 32: sc}, True) == "POOL-PRIOR-PROMISING"
    assert HS.stage1_label({16: sc}, False) == "POOL-PRIOR-BLOCKED"


def test_screen_failures():
    raw = np.ones(12)
    assert not _screen(raw, raw * 0.98, raw)["per_alpha"][0.1]["criteria"]["c1_rel_improvement"]
    assert not _screen(raw, raw * 0.9, raw * 0.8)["per_alpha"][0.1]["criteria"]["c3_beats_identity"]
    pool = raw.copy(); pool[:6] = 1.02; pool[6:] = 0.8                     # L8 regresses +2 %
    v = _screen(raw, pool, raw)["per_alpha"][0.1]
    assert v["criteria"]["c1_rel_improvement"] and not v["criteria"]["c4_no_layer_regression"]
    pool = raw.copy(); pool[:5] = 0.5                                     # only 5 experts improve
    assert not _screen(raw, pool, raw)["per_alpha"][0.1]["criteria"]["c2_experts_improved"]
    sc = _screen(raw, raw * 1.1, raw)
    assert sc["selected_alpha"] is None and HS.stage1_label({16: sc, 32: sc}, True) == "POOL-PRIOR-NO-GAIN"


def _J(std, pool, idn, tuned):
    return dict(std=np.full(12, std) if np.isscalar(std) else std, pool=np.full(12, pool) if np.isscalar(pool) else pool,
                identity=np.full(12, idn), tuned=np.full(12, tuned))


LAY = [8] * 6 + [16] * 6


def test_classify_promising_and_regularization_only():
    ok = HS.budget_criteria(_J(1.0, 0.97, 0.99, 0.975), LAY)
    assert all(ok["criteria"].values())
    assert HS.classify("POOL-PRIOR-PROMISING", {16: ok}, True, True, True)["verdict"] == "POOL-SHRINKAGE-PROMISING"
    assert HS.classify("POOL-PRIOR-PROMISING", {16: ok}, False, True, True)["verdict"] != "POOL-SHRINKAGE-PROMISING"
    reg = HS.budget_criteria(_J(1.0, 0.97, 0.96, 0.975), LAY)                # identity better
    v = HS.classify("POOL-PRIOR-PROMISING", {16: reg}, None, True, True)
    assert v == dict(verdict="POOL-SHRINKAGE-REGULARIZATION-ONLY", recommendation="B")
    tun = HS.budget_criteria(_J(1.0, 0.97, 0.99, 0.95), LAY)                 # tuned damping much better
    assert HS.classify("POOL-PRIOR-PROMISING", {16: tun}, None, True, True)["verdict"] == \
        "POOL-SHRINKAGE-REGULARIZATION-ONLY"


def test_classify_mismatch_only_no_gain_blocked():
    small = HS.budget_criteria(_J(1.0, 0.995, 0.999, 0.99), LAY)
    assert HS.classify("POOL-PRIOR-PROMISING", {16: small, 32: small}, None, True, True) == \
        dict(verdict="POOL-SHRINKAGE-MISMATCH-ONLY", recommendation="C")
    pool = np.full(12, 0.95); pool[0] = 1.2                                  # tail regression
    tail = HS.budget_criteria(_J(1.0, pool, 0.99, 0.99), LAY)
    assert tail["criteria"]["c1_min_improvement"] and not tail["criteria"]["c5_no_tail_regression"]
    assert HS.classify("POOL-PRIOR-PROMISING", {16: tail}, None, True, True)["verdict"] == "POOL-SHRINKAGE-NO-GAIN"
    assert HS.classify("POOL-PRIOR-NO-GAIN", {}, None, True, True)["verdict"] == "POOL-PRIOR-NO-GAIN"
    assert HS.classify("POOL-PRIOR-PROMISING", {16: small}, None, False, True)["verdict"] == "POOL-SHRINKAGE-BLOCKED"


def test_gap_recovery():
    assert HS.gap_recovery(1.0, 0.98, 0.9) == pytest.approx(0.2)
    assert HS.gap_recovery(1.0, 0.98, 0.995) is None
