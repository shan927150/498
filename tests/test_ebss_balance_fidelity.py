"""Balance-strength fidelity study: normalisation variants of the EBSS balance term.
Production (moe_calib.ebss_generator) is compared with the independent oracle
runs_ebss_balance_fidelity/normalization_oracle.py (NumPy only) and with the frozen pre-study generator."""
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
from moe_calib import ebss_generator as G                                  # noqa: E402

OUT = os.path.join(ROOT, "runs_ebss_balance_fidelity")
TAU = G.TAU_REF


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


NO = _load(os.path.join(OUT, "normalization_oracle.py"), "normalization_oracle")
OLD = _load(os.path.join(OUT, "logs", "ebss_generator_pre_fidelity.py"), "ebss_generator_pre_fidelity")


def F():
    import ebss_balance_fidelity as m
    return m


def vsig(C, n, v, k=4):
    vv = G.BALANCE_VARIANTS[v]
    return float(G.balance_sigma(torch.as_tensor(C), n, vv["balance_std_mode"], vv["usage_mode"], k))


# 1, 3 four definitions vs independent oracle and a hand value
@pytest.mark.parametrize("v", list(G.BALANCE_VARIANTS))
def test_four_definitions_match_oracle(v):
    rng = np.random.default_rng(1)
    for _ in range(10):
        C = rng.integers(0, 30, size=(24, 60)); n = int(rng.integers(8, 512))
        assert abs(vsig(C, n, v) - NO.sigma_variant(C.tolist(), n, v)) <= 1e-15


def test_hand_value_layer():
    C = [[6, 2, 2, 1, 1, 0]]
    u = np.array([1, 1 / 3, 1 / 3, 1 / 6, 1 / 6, 0]); ss = ((u - u.mean()) ** 2).sum()
    assert abs(vsig(C, 6, "EQ4_RATE", 2) - math.sqrt(ss / 5)) < 1e-15
    assert abs(vsig(C, 6, "ALG1_RATE", 2) - math.sqrt(6 * ss / 5)) < 1e-15


# 2 analytic scaling, E read from the tensor
@pytest.mark.parametrize("E,k", [(60, 4), (8, 2), (64, 6)])
def test_analytic_scaling(E, k):
    C = torch.randint(0, 40, (5, E)); n = 37
    base = vsig(C, n, "EQ4_RATE", k)
    for v, r in dict(EQ4_SHARE=1 / k, ALG1_SHARE=math.sqrt(E) / k, ALG1_RATE=math.sqrt(E)).items():
        assert abs(vsig(C, n, v, k) / base - r) <= 1e-12 * r


def test_strength_table_matches_protocol():
    d = json.load(open(os.path.join(OUT, "normalization_definitions.json")))["variants"]
    for v, r in dict(EQ4_RATE=1, EQ4_SHARE=.25, ALG1_SHARE=1.9364916731037085, ALG1_RATE=7.745966692414834).items():
        assert abs(d[v]["relative_strength"] - r) < 1e-12


def test_heterogeneous_experts_oracle():
    a = NO.sigma([[4, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]], 2, "algorithm1", "per_token", 2)
    l0 = math.sqrt(4 / 3 * sum((x - .5) ** 2 for x in [2, 0, 0, 0]))
    l1 = math.sqrt(8 / 7 * sum((x - .25) ** 2 for x in [.5] * 4 + [0] * 4))
    assert abs(a - (l0 + l1) / 2) < 1e-15


# 4, 23 exact default regression / backward compatibility
@pytest.mark.parametrize("ub", [True, False])
def test_default_bitwise_vs_frozen_generator(ub):
    kw = dict(seqlen=12, w=4, tau=TAU, use_balance=ub)
    a = OLD.generate(OLD.SyntheticBackend(seed=4), [0, 1], **kw)
    b = G.generate(G.SyntheticBackend(seed=4), [0, 1], **kw)
    c = G.generate(G.SyntheticBackend(seed=4), [0, 1], **kw, **G.BALANCE_VARIANTS["EQ4_RATE"])
    for f in ("tokens", "cum_logp", "token_logp", "counts", "sigma", "final_score"):
        assert torch.equal(getattr(a, f), getattr(b, f)) and torch.equal(getattr(a, f), getattr(c, f))


def test_default_balance_sigma_bitwise_vs_frozen():
    C = torch.randint(0, 50, (7, 24, 60))
    assert torch.equal(G.balance_sigma(C, 77), OLD.balance_sigma(C, 77))


def test_unknown_mode_rejected_and_share_needs_k():
    with pytest.raises(ValueError):
        G.balance_sigma(torch.ones(2, 4), 3, "foo")
    with pytest.raises(ValueError):
        G.balance_sigma(torch.ones(2, 4), 3, "eq4", "assignment_share")


# 5 positive penalty / uniform zero
@pytest.mark.parametrize("v", list(G.BALANCE_VARIANTS))
def test_uniform_zero_nonuniform_positive(v):
    assert vsig(torch.full((24, 60), 4), 60, v) == 0.0
    assert vsig(torch.tensor([[5, 3, 0, 0]] * 3), 2, v) > 0


# 6, 9 deferred parent sigma & cross-parent ranking through generate()
@pytest.mark.parametrize("v", list(G.BALANCE_VARIANTS))
def test_deferred_parent_sigma_cross_parent(v):
    import ebss_score_audit as A
    ok, info = F().deferred_variant(A, v)
    assert ok, info


# 7, 10, 12 minimisation, global pruning, stable ties: variant only shifts per-parent sigma
def test_global_minimisation_and_ties_unchanged():
    S = torch.tensor([[0.3, 0.1, 0.1], [0.1, 0.5, 0.2]], dtype=torch.float64)
    pp, tt, _ = G.select_best(S, 3)
    assert list(zip(pp.tolist(), tt.tolist())) == [(0, 1), (0, 2), (1, 0)]


# 8 PO invariance under every variant's options
@pytest.mark.parametrize("v", list(G.BALANCE_VARIANTS))
def test_probability_only_invariant(v):
    a = G.generate(G.SyntheticBackend(seed=9), [3], seqlen=10, w=4, tau=TAU, use_balance=False)
    b = G.generate(G.SyntheticBackend(seed=9), [3], seqlen=10, w=4, tau=TAU, use_balance=False, **G.BALANCE_VARIANTS[v])
    assert torch.equal(a.tokens, b.tokens) and torch.equal(a.cum_logp, b.cum_logp)


# 11 candidate-score decomposition: EBSS - PO = sigma_v / tau per parent
@pytest.mark.parametrize("v", list(G.BALANCE_VARIANTS))
def test_score_decomposition(v):
    cum = torch.tensor([-1.0, -2.5], dtype=torch.float64)
    lp = torch.log_softmax(torch.randn(2, 9, dtype=torch.float64), -1)
    sig = torch.tensor([vsig(torch.tensor([[3, 1, 0, 0]]), 2, v), vsig(torch.tensor([[1, 1, 1, 1]]), 2, v)],
                       dtype=torch.float64)
    d = G.candidate_scores(cum, lp, sig, 5, TAU, True) - G.candidate_scores(cum, lp, sig, 5, TAU, False)
    assert torch.allclose(d, (sig / TAU)[:, None].expand_as(d), atol=1e-15, rtol=0)


# 13 configuration serialisation
@pytest.mark.parametrize("v", list(G.BALANCE_VARIANTS))
def test_config_serialised(v):
    r = G.generate(G.SyntheticBackend(seed=2), [1], seqlen=6, w=4, tau=TAU, **G.BALANCE_VARIANTS[v])
    assert {kk: r.config[kk] for kk in ("balance_std_mode", "usage_mode")} == G.BALANCE_VARIANTS[v]
    json.dumps(r.config)


# 14 frozen start tokens
def test_start_tokens_frozen():
    prot = json.load(open(os.path.join(OUT, "protocol.json")))
    man = json.load(open(os.path.join(ROOT, "runs_ebss_generation_pilot", "initialization_manifest.json")))
    assert prot["short_generation"]["start_ids"] == man["pilot_start_ids"][:4] == G.sample_first_tokens(4)


# 15, 16, 17, 18 deterministic, valid lengths, routing-count integrity, per-layer stats
@pytest.mark.parametrize("v", list(G.BALANCE_VARIANTS))
def test_deterministic_lengths_counts(v):
    kw = dict(seqlen=11, w=4, tau=TAU, **G.BALANCE_VARIANTS[v])
    a = G.generate(G.SyntheticBackend(seed=6, L=3, E=8, k=2), [0, 1, 2], **kw)
    b = G.generate(G.SyntheticBackend(seed=6, L=3, E=8, k=2), [0, 1, 2], **kw)
    assert torch.equal(a.tokens, b.tokens) and a.tokens.shape == (3, 11)
    assert torch.equal(a.counts, torch.stack([G.recount(a.hist_experts[i], 8) for i in range(3)]))
    assert (a.counts.sum(-1) == 11 * 2).all()                          # every layer: tokens x k


def test_set_exposure_per_layer():
    C = np.tile(np.arange(60), (24, 1)) + 1
    se = G.set_exposure(C)
    assert len(se["per_layer"]) == 24 and abs(se["per_layer"][0]["cv"] - np.std(C[0]) / np.mean(C[0])) < 1e-12


# 19 canonical cross-variant metrics use one definition
def test_canonical_metric_independent_of_variant():
    import diagnostics_ebss_pilot as DG
    sel = np.random.default_rng(0).integers(0, 60, size=(2, 16, 24, 4))
    a = DG.exposure_block(DG.counts_from_sel(sel))
    assert abs(a["paper_sigma"] - NO.sigma_variant(DG.counts_from_sel(sel).tolist(), 32, "EQ4_RATE")) < 1e-12


# 20, 22 frozen gate and classification
def test_gate_thresholds_frozen():
    g = json.load(open(os.path.join(OUT, "protocol.json")))["scientific_gate"]
    assert "0.05" in g["G1_beam"] and "0.10" in g["G2_balance"] and "0.25" in g["G5_routing"]


def test_classification_rule():
    import report_ebss_balance_fidelity as R
    assert R.primary_label(regression_ok=False, any_material=True) == "BALANCE-NORMALIZATION-RUN-BLOCKED"
    assert R.primary_label(True, True) == "BALANCE-NORMALIZATION-MATERIAL"
    assert R.primary_label(True, False) == "BALANCE-NORMALIZATION-IMMATERIAL"
    c4 = [1.0, 1.01, 0.99]
    assert R.quality_label({"X": 0.98}, 1.0, c4) == "BALANCE-VARIANT-QUALITY-GAIN"
    assert R.quality_label({"X": 1.2}, 1.0, c4) == "BALANCE-VARIANT-QUALITY-NO-GAIN"
    assert R.quality_label({"X": 1.0 * 0.99}, 1.0, [0.5, 0.5, 0.5]) == "BALANCE-VARIANT-QUALITY-MIXED"


# 21 GPTQ pipeline invariance
def test_gptq_source_unchanged():
    import hashlib
    h = hashlib.sha256(open(os.path.join(ROOT, "moe_calib", "gptq.py"), "rb").read()).hexdigest()
    assert h.startswith("58e40d1c")


# pool reduction is exact for any per-parent sigma
def test_reduced_pool_reproduces_production():
    for cond in ("EQ4_RATE", "ALG1_RATE", "PROBABILITY_ONLY"):
        pr = F().PoolRecorder(4)
        G.generate(G.SyntheticBackend(V=50, L=3, E=8, k=4, seed=8), [0, 1], seqlen=14, w=4, tau=TAU, audit=pr,
                   **F().gen_kwargs(cond))
        path = os.path.join(OUT, "logs", f"_test_pool_{cond}.pt")
        F().save_pools(pr, path)
        try:
            rows = F().counterfactual_pools(path)
        finally:
            os.remove(path)
        assert all(r["res"][cond]["kept"] == r["prod"] for r in rows)
