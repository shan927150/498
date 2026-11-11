"""EBSS score/sign audit regression tests (oracle: runs_ebss_score_audit/independent_oracle.py, NumPy only).

Every test compares PRODUCTION code (moe_calib.ebss_generator) with the independent oracle, and the key tests also
show that deliberately wrong variants would fail, so a passing result is not vacuous."""
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import ebss_score_audit as A                                   # noqa: E402
from moe_calib import ebss_generator as G                       # noqa: E402

O = A.O
TAU = G.TAU_REF


def test_oracle_is_independent_of_production():
    src = open(os.path.join(ROOT, "runs_ebss_score_audit", "independent_oracle.py")).read()
    assert "ebss_generator" not in src.split('"""', 2)[2] and "import torch" not in src


def test_logp_and_nll_representations_identical():
    for L, pv, l, s in [(0.0, math.log(0.6), 1, 0.2), (-3.7, math.log(1e-30), 9, 0.0), (-250.0, -0.01, 511, 0.04)]:
        assert abs(O.score_logp(L, pv, l, s, TAU) - O.score_nll(-L, pv, l, s, TAU)) <= 1e-12


def test_high_probability_candidate_wins():
    S = A.prod_scores([-0.9], [[math.log(0.8), math.log(0.2)]], [0.1], 3)
    assert S[0, 0] < S[0, 1]
    assert A.prod_select(S, 1)[0][:2] == (0, 0)


def test_balance_monotonicity():
    S = A.prod_scores([-0.9, -0.9], [[math.log(0.5)], [math.log(0.5)]], [0.1, 0.3], 3)
    assert S[0, 0] < S[1, 0]


def test_cross_parent_sign_trap():
    """Parent A cum logP -0.2, parent B -2.0, same next token p = 0.5, same sigma: A must win.
    The wrong formula -(NLL + logp)/(l+1) prefers B -- the test would catch it."""
    pv, l, s = math.log(0.5), 4, 0.1
    S = A.prod_scores([-0.2, -2.0], [[pv], [pv]], [s, s], l)
    assert S[0, 0] < S[1, 0]
    assert abs(S[0, 0] - ((0.2 - pv) / 5 + s / TAU)) < 1e-15
    wrong = [O.wrong_score_nll_sign(N, pv, l, s, TAU) for N in (0.2, 2.0)]
    assert wrong[1] < wrong[0]


def test_cross_parent_trap_through_generate():
    assert A.trap_end_to_end()["passed"]


def test_tradeoff_threshold_tau_star_2():
    t = A.tradeoff()
    assert t["passed"]
    winners = {r["tau"]: r["production_winner"] for r in t["rows"]}
    assert winners[1.3] == "B" and winners[2.0] == "A" and winners[3.0] == "A"


def test_beam_width_global_ranking_and_tie_breaking():
    b = A.beam_width_ranking()
    assert b["passed"]
    ebss_w4 = [c for c in b["cases"] if c["w"] == 4 and c["use_balance"]][0]
    for v in ("maximise", "probability_only", "balance_after_pruning", "wrong_parent_sigma(rolled)",
              "wrong_parent_history(rolled)", "reversed_sigma"):
        assert ebss_w4["wrong_variant_differs_from_correct"][v], v


def test_deferred_sigma_uses_parent_prefix():
    assert A.deferred_sigma_end_to_end()["passed"]


def test_sigma_uniform_lower_than_concentrated():
    uni = np.full((2, 6), 2); conc = np.array([[12, 0, 0, 0, 0, 0], [6, 6, 0, 0, 0, 0]])
    assert O.sigma_eq4(uni, 6) == 0.0 < O.sigma_eq4(conc, 6)
    pc = G.balance_sigma(torch.tensor(conc), 6).item()
    assert abs(pc - O.sigma_eq4(conc, 6)) < 1e-15


def test_metamorphic_invariants():
    m = A.metamorphic(n_sets=120)
    assert m["passed"], m["failing_examples"][:3]
    assert all(v > 0 for v in m["wrong_variant_detection"]["sets_where_variant_changes_kept_set"].values())


@pytest.mark.parametrize("ub", [True, False])
def test_generate_trace_matches_oracle(ub):
    s = A.synthetic_trace(seed=3, ub=ub, N=2, T=10)
    assert s["passed"], s["mismatches"]
    assert s["final"]["pick_matches"]


def test_audit_hook_is_read_only():
    kw = dict(seqlen=9, w=4, tau=TAU, use_balance=True)
    r1 = G.generate(G.SyntheticBackend(seed=5), [1, 2], **kw)
    r2 = G.generate(G.SyntheticBackend(seed=5), [1, 2], audit=lambda r: None, **kw)
    assert torch.equal(r1.tokens, r2.tokens) and torch.equal(r1.cum_logp, r2.cum_logp)
