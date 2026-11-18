"""CPU tests for projection-selective AGQ (moe_calib/projection_select.py + the mixed-weight evaluation path)."""
import numpy as np
import pytest
import torch

from moe_calib import agq as AG
from moe_calib import projection_select as PS

RULES = dict(tau_rel=0.005, interaction_frac=0.25, expert_mixed_frac=0.10, expert_rel=0.01)
N = 4


def fake_cands(n=N, d=8, h=6):
    g = torch.Generator().manual_seed(0)
    c = {}
    for q in (0, 1):
        c[q] = {}
        for e in range(n):
            c[q][f"{e}.gate_proj"] = torch.randn(h, d, generator=g)
            c[q][f"{e}.up_proj"] = torch.randn(h, d, generator=g)
            c[q][f"{e}.down_proj"] = torch.randn(d, h, generator=g)
    return c


@pytest.mark.parametrize("name,src", [("GPTQ", (0, 0, 0)), ("GU_AGQ", (1, 1, 0)), ("DOWN_AGQ", (0, 0, 1)),
                                      ("FULL_AGQ", (1, 1, 1))])
def test_config_sources(name, src):
    c = fake_cands()
    W, man = PS.build_projection_config(c, name, N)
    assert len(W) == N and len(man) == 3 * N
    for e in range(N):
        assert set(W[e]) == set(PS.PROJ)
        for k, q in zip(PS.PROJ, src):
            assert W[e][k] is c[q][f"{e}.{k}"]                      # the candidate tensor itself, unmodified
    assert all(r["q"] == dict(zip(PS.PROJ, src))[r["projection"]] for r in man)


def test_missing_tensor_raises_no_bf16_fallback():
    c = fake_cands()
    del c[1]["2.down_proj"]
    PS.build_projection_config(c, "GU_AGQ", N)                     # does not need AGQ down
    with pytest.raises(KeyError):
        PS.build_projection_config(c, "DOWN_AGQ", N)
    with pytest.raises(ValueError):
        PS.build_projection_config(c, "GATE_ONLY", N)


def test_candidates_unchanged_and_manifest_deterministic():
    c = fake_cands()
    before = {q: {k: v.clone() for k, v in c[q].items()} for q in (0, 1)}
    hs = [PS.manifest_hash(PS.build_projection_config(c, n, N)[1]) for n in PS.ORDER]
    hs2 = [PS.manifest_hash(PS.build_projection_config(c, n, N)[1]) for n in PS.ORDER]
    assert hs == hs2 and len(set(hs)) == 4
    assert all(torch.equal(before[q][k], c[q][k]) for q in (0, 1) for k in c[q])


def test_mixed_expert_output_uses_actual_mixed_weights():
    c = fake_cands()
    x = torch.randn(5, 8)
    W, _ = PS.build_projection_config(c, "GU_AGQ", N)
    ref = c[0]["1.down_proj"] @ (torch.nn.functional.silu(c[1]["1.gate_proj"] @ x.T) * (c[1]["1.up_proj"] @ x.T))
    assert torch.allclose(AG.expert_fp32(x, W[1]), ref.T, atol=1e-5)
    W, _ = PS.build_projection_config(c, "DOWN_AGQ", N)
    ref = c[1]["1.down_proj"] @ (torch.nn.functional.silu(c[0]["1.gate_proj"] @ x.T) * (c[0]["1.up_proj"] @ x.T))
    assert torch.allclose(AG.expert_fp32(x, W[1]), ref.T, atol=1e-5)


def test_mixed_output_not_sum_of_projection_errors():
    """The full-expert error of a mixed config differs from summing per-projection errors (nonlinear gate)."""
    c = fake_cands()
    x = torch.randn(64, 8)
    Wfp = {k: c[0][f"0.{k}"] + 0.1 for k in PS.PROJ}
    full = lambda W: float(((AG.expert_fp32(x, Wfp) - AG.expert_fp32(x, W)) ** 2).sum())
    WB = PS.build_projection_config(c, "GU_AGQ", N)[0][0]
    WA = PS.build_projection_config(c, "GPTQ", N)[0][0]
    WD = PS.build_projection_config(c, "FULL_AGQ", N)[0][0]
    WC = PS.build_projection_config(c, "DOWN_AGQ", N)[0][0]
    I = full(WD) - full(WB) - full(WC) + full(WA)
    assert abs(I) > 1e-6


def test_select_tie_order():
    e = dict(GPTQ=1.0, GU_AGQ=1.0, DOWN_AGQ=1.0, FULL_AGQ=1.0)
    assert PS.select_config(e) == "GPTQ"
    assert PS.select_config(dict(e, GPTQ=1.1)) == "FULL_AGQ"
    assert PS.select_config(dict(e, GPTQ=1.1, FULL_AGQ=1.1)) == "GU_AGQ"
    assert PS.select_config(dict(GPTQ=1.1, FULL_AGQ=1.1, GU_AGQ=1.1, DOWN_AGQ=1.0)) == "DOWN_AGQ"
    assert PS.layerwise_uniform(dict(GPTQ=1.0, FULL_AGQ=1.0)) == "GPTQ"
    assert PS.layerwise_uniform(dict(GPTQ=1.0, FULL_AGQ=0.9)) == "FULL_AGQ"


def test_interactions_additive_and_sign():
    E = dict(GPTQ=10.0, GU_AGQ=9.0, DOWN_AGQ=8.0, FULL_AGQ=7.0)
    r = PS.interactions(E)
    assert r["interaction_I"] == 0 and r["interaction_sign"] == "additive"
    assert r["G_GU_given_down_gptq"] == 1 and r["G_D_given_gu_gptq"] == 2
    assert r["shapley_share_of_full_gain"]["GU"] == pytest.approx(1 / 3)
    assert PS.interactions(dict(E, FULL_AGQ=7.5))["interaction_sign"] == "antagonistic"
    assert PS.interactions(dict(E, FULL_AGQ=6.5))["interaction_sign"] == "synergistic"


def _d(a, b, c, d):
    return dict(GPTQ=a, GU_AGQ=b, DOWN_AGQ=c, FULL_AGQ=d)


def test_classify_promising_and_no_gain():
    val = {8: _d(1.0, 0.98, 1.0, 1.0), 16: _d(1.0, 0.95, 0.99, 0.97)}
    conf = {8: _d(1.0, 0.98, 1.0, 1.0), 16: _d(1.0, 0.95, 0.99, 0.97)}
    sel = {L: PS.select_config(val[L]) for L in val}
    lw = {L: PS.layerwise_uniform(val[L]) for L in val}
    v = PS.classify(RULES, val, conf, conf, sel, lw, True, {8: 0.0, 16: 0.0})
    assert v["verdict"] == "PROJECTION-AGQ-PROMISING" and PS.RECOMMEND[v["verdict"]] == "A"
    flat = {8: _d(1.0, 1.0001, 1.0002, 1.0001), 16: _d(1.0, 0.985, 0.985, 0.97)}   # additive (I = 0)
    sel = {L: PS.select_config(flat[L]) for L in flat}
    lw = {L: PS.layerwise_uniform(flat[L]) for L in flat}
    v = PS.classify(RULES, flat, flat, flat, sel, lw, True, {8: 0.0, 16: 0.0})
    assert v["verdict"] == "PROJECTION-AGQ-NO-GAIN" and PS.RECOMMEND[v["verdict"]] == "B"


def test_classify_selection_fail_and_blocked():
    val = {8: _d(1.0, 0.99, 1.0, 1.0), 16: _d(1.0, 0.99, 1.0, 0.98)}
    conf = {8: _d(1.0, 1.01, 1.0, 1.0), 16: _d(1.0, 0.99, 1.0, 0.98)}
    sel = {L: PS.select_config(val[L]) for L in val}
    lw = {L: PS.layerwise_uniform(val[L]) for L in val}
    assert PS.classify(RULES, val, conf, conf, sel, lw, True, {8: 0, 16: 0})["verdict"] == "PROJECTION-AGQ-SELECTION-FAIL"
    assert PS.classify(RULES, val, conf, conf, sel, lw, False, {})["verdict"] == "PROJECTION-AGQ-BLOCKED"


def test_classify_interaction_only():
    flat = {8: _d(1.0, 1.0001, 1.0002, 1.0001), 16: _d(1.0, 0.99, 0.99, 0.97)}   # I = -0.01 -> |I|/gain = 0.33
    sel = {L: PS.select_config(flat[L]) for L in flat}
    lw = {L: PS.layerwise_uniform(flat[L]) for L in flat}
    v = PS.classify(RULES, flat, flat, flat, sel, lw, True, {8: 0.0, 16: 0.0})
    assert v["verdict"] == "PROJECTION-AGQ-INTERACTION-ONLY" and PS.RECOMMEND[v["verdict"]] == "C"


def test_decomposition_identity_mixed():
    torch.manual_seed(1)
    T, d, k, n = 16, 8, 4, 6
    vals, sel = AG.routing_coefficients(torch.randn(T, n), k)
    rows = AG.dispatch_rows(sel, vals, n)
    V = torch.zeros(T, k, d)
    err = torch.zeros(T, d)
    for e, (tx, slot, c) in rows.items():
        dl = torch.randn(len(tx), d)
        V[tx, slot] = dl * c[:, None]
        err.index_add_(0, tx, dl * c[:, None])
    dec = AG.decompose(V)
    assert dec["diag"] + dec["cross"] == pytest.approx(float((err.double() ** 2).sum()), rel=1e-6)
