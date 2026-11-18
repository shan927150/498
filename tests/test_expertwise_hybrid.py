"""Unit tests for the expert-wise GPTQ/AGQ hybrid selection and decision helpers (moe_calib/hybrid.py) and for the
mixed-candidate error decomposition used by diagnostics/expertwise_agq_hybrid.py."""
import numpy as np
import pytest
import torch

from moe_calib import agq as AG
from moe_calib import hybrid as HY

RULES = dict(tau_rel=0.005, min_sign_agreement=0.65, min_spearman=0.5)


def test_select_strict_and_ties_to_gptq():
    J0 = np.array([1.0, 2.0, 3.0, 1.0, np.nan])
    J1 = np.array([0.5, 2.0, 3.5, 0.999999, 0.1])
    assert HY.select_experts(J0, J1).tolist() == [1, 0, 0, 1, 0]


def test_rel_improvement_sign():
    r = HY.rel_improvement([2.0, 1.0, 0.0], [1.0, 1.5, 0.0])
    assert r[0] == pytest.approx(0.5) and r[1] == pytest.approx(-0.5) and np.isnan(r[2])


def test_layerwise_choice():
    assert HY.layerwise_choice(1.0, 0.9) == 1
    assert HY.layerwise_choice(1.0, 1.0) == 0
    assert HY.layerwise_choice(1.0, 1.1) == 0


def test_config_vectors_and_hash():
    n = 5
    ch = [{0: f"g{e}", 1: f"a{e}"} for e in range(n)]
    z = np.array([0, 1, 0, 1, 1])
    assert HY.config_vector("GPTQ", n).tolist() == [0] * n
    assert HY.config_vector("AGQ", n).tolist() == [1] * n
    assert HY.config_vector("LAYERWISE", n, layer_choice=1).tolist() == [1] * n
    assert HY.config_vector("HYBRID", n, z).tolist() == z.tolist()
    with pytest.raises(AssertionError):
        HY.config_vector("HYBRID", n, [0, 2, 0, 0, 0])
    assert HY.config_hash(z, ch) != HY.config_hash(np.zeros(n), ch)
    assert HY.config_hash(z, ch) == HY.config_hash(z.copy(), ch)


def test_half_split_stability_perfect():
    rng = np.random.default_rng(0)
    J0 = rng.uniform(1, 2, size=(8, 10))
    J1 = J0 * np.where(np.arange(10) % 2 == 0, 0.5, 1.5)          # consistent preference in every sequence
    s = HY.half_split_stability(J0, J1)
    assert s["agreement_halves"] == 1.0 and s["n_agq_full"] == 5


def test_transfer_metrics_regret():
    vJ0, vJ1 = np.array([1.0, 1.0, 1.0]), np.array([0.9, 1.1, 0.95])  # z = [1, 0, 1]
    tJ0, tJ1 = np.array([1.0, 1.0, 1.0]), np.array([0.8, 0.9, 1.2])   # test best = [1, 1, 0]
    z = HY.select_experts(vJ0, vJ1)
    t = HY.transfer_metrics(vJ0, vJ1, tJ0, tJ1, z)
    assert t["sign_agreement"] == pytest.approx(1 / 3)
    assert t["false_selections"] == 2
    assert t["regret_sum"] == pytest.approx(0.1 + 0.2)
    assert t["realized_gain_vs_gptq"] == pytest.approx(0.2 - 0.2)


def _layers(g, a, l, h):
    return {8: dict(GPTQ=g[0], AGQ=a[0], LAYERWISE=l[0], HYBRID=h[0]),
            16: dict(GPTQ=g[1], AGQ=a[1], LAYERWISE=l[1], HYBRID=h[1])}


GOOD = dict(sign_agreement=0.8, spearman_rel_improvement=0.8)
BAD = dict(sign_agreement=0.5, spearman_rel_improvement=0.1)


def test_classify_promising():
    v = HY.classify(RULES, _layers((1.0, 1.0), (1.002, 0.97), (1.0, 0.97), (0.98, 0.95)), GOOD, True)
    assert v["verdict"] == "HYBRID-PROMISING" and HY.recommend(v, 0.005) == "A"


def test_classify_local_only_recommend_b():
    v = HY.classify(RULES, _layers((1.0, 1.0), (1.02, 0.97), (1.0, 0.97), (0.999, 0.968)), GOOD, True)
    assert v["verdict"] == "HYBRID-LOCAL-ONLY" and HY.recommend(v, 0.005) == "B"


def test_classify_no_gain():
    v = HY.classify(RULES, _layers((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.999, 0.999)), GOOD, True)
    assert v["verdict"] == "HYBRID-NO-GAIN" and HY.recommend(v, 0.005) == "C"


def test_classify_selection_fail():
    v = HY.classify(RULES, _layers((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.001, 1.0)), BAD, True)
    assert v["verdict"] == "HYBRID-SELECTION-FAIL"


def test_classify_layer_regression_blocks_promising():
    v = HY.classify(RULES, _layers((1.0, 1.0), (1.0, 0.90), (1.0, 0.90), (1.02, 0.85)), GOOD, True)
    assert not v["criteria"]["4_no_layer_regression_vs_best_uniform"] and v["verdict"] != "HYBRID-PROMISING"


def test_classify_blocked():
    assert HY.classify(RULES, {}, GOOD, False)["verdict"] == "HYBRID-BLOCKED"


def test_mixed_candidate_decomposition_identity():
    """D + C reproduces ||sum_k c_k dE_k||^2 when each expert's delta comes from a different candidate."""
    torch.manual_seed(0)
    T, d, k, n = 32, 16, 4, 8
    logits = torch.randn(T, n)
    vals, sel = AG.routing_coefficients(logits, k)
    rows = AG.dispatch_rows(sel, vals, n)
    dl = {q: {e: torch.randn(len(rows[e][0]), d) for e in rows} for q in (0, 1)}
    z = np.array([e % 2 for e in range(n)])
    err = torch.zeros(T, d)
    V = torch.zeros(T, k, d)
    for e, (tx, slot, c) in rows.items():
        de = dl[int(z[e])][e]
        err.index_add_(0, tx, de * c[:, None])
        V[tx, slot] = de * c[:, None]
    dec = AG.decompose(V)
    tot = float((err.double() ** 2).sum())
    assert dec["diag"] + dec["cross"] == pytest.approx(tot, rel=1e-6)
    assert dec["total"] == pytest.approx(tot, rel=1e-6)
    # diagonal equals the sum of per-expert c^2-weighted squared errors (the selection score summed over experts)
    J = sum(float(((rows[e][2].double() ** 2) * (dl[int(z[e])][e].double() ** 2).sum(1)).sum()) for e in rows)
    assert dec["diag"] == pytest.approx(J, rel=1e-6)
