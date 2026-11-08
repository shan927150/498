"""CPU tests for the extreme low-budget (B = 4 / 8) scarcity study."""
import json
import os
import random
import sys

import numpy as np
import pytest

import e3_allocation as E3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import extreme_low_budget as X      # noqa: E402
import low_budget as LB             # noqa: E402

OUT = os.path.join(ROOT, "runs_extreme_low_budget")
SEL = os.path.join(OUT, "matched_selection.json")


# ------------------------------------------------------------- schedule / compatibility
def test_uneven_schedules_b4_b8():
    assert X.schedule(4) == (1, [1, 2])
    assert X.schedule(8) == (2, [3, 3])


def test_default_k_cannot_express_b4():
    """The documented limitation: with K = 4 the warm floor swallows the whole B = 4 budget."""
    assert E3.round_schedule(4, 4, 2) == (4, [])
    assert E3.round_schedule(8, 4, 2)[0] == 4


@pytest.mark.parametrize("B", [16, 32, 64, 128])
def test_k_eff_backward_compatible(B):
    assert E3.round_schedule(B, X.k_eff(B), 2) == E3.round_schedule(B, 4, 2)


# ------------------------------------------------------------- metrics
def test_tail_metrics_known_vector():
    n = np.array([0, 10, 40, 100, 200, 300, 1000, 50, 60, 70], float)
    m = X.tail_metrics(n)
    assert m["min"] == 0 and m["max"] == 1000 and m["zero"] == 1
    assert m["below_32"] == 2 and m["below_64"] == 5 and m["below_128"] == 7 and m["below_256"] == 8
    assert m["frac_below_128"] == pytest.approx(0.7)
    assert m["p10"] == pytest.approx(np.quantile(n, .1)) and m["p5"] == pytest.approx(np.quantile(n, .05))


def test_gini_and_source_diversity():
    assert LB.gini(np.ones(6)) == pytest.approx(0)
    assert LB.gini([0, 0, 0, 4]) == pytest.approx(0.75)
    Xs = np.array([[10, 0, 5], [10, 0, 5], [0, 7, 0]], float)
    st, n = X.set_tail(np.tile(Xs, (1, 1)), [1], 3)
    eff = LB.eff_sources(Xs)
    assert eff[0] == pytest.approx(2.0) and eff[1] == pytest.approx(1.0) and eff[2] == pytest.approx(2.0)
    assert st["all"]["unique_sources_median"] == 2 and st["all"]["unique_sources_min"] == 1


def test_routing_token_accounting():
    g = np.random.default_rng(3)
    C = g.poisson(5, size=(30, 2 * 5)).astype(float)
    ids = [3, 7, 11]
    st, n = X.set_tail(C[ids], [1, 2], 5)
    assert np.array_equal(n, C[ids].sum(0))
    assert st["per_layer"]["2"]["min"] == C[ids][:, 5:].sum(0).min()


def test_probability_and_wilson():
    p = X.prob([True] * 3 + [False] * 7)
    assert p["p"] == pytest.approx(0.3) and p["k"] == 3 and p["n"] == 10
    lo, hi = p["ci95"]
    assert 0 < lo < 0.3 < hi < 1
    assert X.prob([False] * 50)["ci95"][0] == 0.0
    f = X.scarcity_flags(X.tail_metrics(np.array([0, 50, 100, 200, 300, 400, 500, 600, 700, 800], float)))
    assert f["any_zero"] and f["any_below_64"] and f["frac_below_128_ge_10pct"]


def test_monte_carlo_reproducible_and_independent():
    pool = list(range(100, 300))
    a, b = X.mc_sets(pool, 4, 11, 50), X.mc_sets(pool, 4, 11, 50)
    assert a == b
    assert X.mc_sets(pool, 4, 12, 50) != a
    assert all(len(s) == 4 == len(set(s)) and set(s) <= set(pool) for s in a)
    assert len(set(X.MC_SEEDS.values())) == len(X.MC_SEEDS)


# ------------------------------------------------------------- synthetic scenarios
def test_random_can_produce_zero_exposure():
    C = np.zeros((40, 6)); C[:, :5] = 5; C[0, 5] = 3          # only row 0 reaches expert 5
    sets = X.mc_sets(list(range(40)), 4, 0, 200)
    zero = [X.tail_metrics(C[s].sum(0))["zero"] > 0 for s in sets]
    assert X.prob(zero)["p"] > 0.8


def test_coverage_eliminates_zero_exposure_at_b4():
    C = np.zeros((40, 6)); C[:, :4] = 5; C[30:, 4] = 3; C[35:, 5] = 3
    ids = list(range(40))
    ch, w, traj = E3.select_lp_utility("coverage_lp", C, ids, [1], 6, 4, X.k_eff(4), 2, random.Random(0), None,
                                       warm_start=[0])
    assert len(ch) == 4 == len(set(ch)) and ch[0] == 0
    assert traj[0]["n_zero_exposure"] == 2 and traj[0]["n_exploration_targets_reached"] == 2
    assert (C[ch].sum(0) > 0).all()


def test_low_budget_lp_cannot_cover_every_expert():
    """5 zero-exposure experts, each reachable by one distinct row; only 3 post-warm slots."""
    C = np.zeros((20, 7)); C[:, :2] = 10
    for e, r in zip(range(2, 7), range(10, 15)):
        C[r, e] = 4
    ids = list(range(20))
    ch, w, traj = E3.select_lp_utility("coverage_lp", C, ids, [1], 7, 4, X.k_eff(4), 2, random.Random(0), None,
                                       warm_start=[0])
    assert len(ch) == 4 == len(set(ch))
    assert traj[0]["n_exploration_targets_unreached"] == 4
    assert X.tail_metrics(C[ch].sum(0))["zero"] == 2           # reported, not hidden


def test_lower_gini_with_worse_minimum():
    a, b = np.array([2, 10, 10, 10, 10], float), np.array([5, 5, 5, 20, 20], float)
    ma, mb = X.tail_metrics(a), X.tail_metrics(b)
    assert ma["gini"] < mb["gini"] and ma["min"] < mb["min"]


def test_balanced_selection_can_be_atypical():
    from moe_calib import typicality as TY
    C = np.zeros((40, 4)); C[:30] = [8, 8, 8, 0]; C[30:] = [0, 0, 0, 24]
    ref = TY.pool_reference(C, 1, 4)
    sc = TY.sequence_scores(C, 1, 4, ref)
    frow = {s: s for s in range(40)}
    bal = LB.set_stats([0, 1, 2, 30], C, frow, [7], 4, ref, sc)
    typ = LB.set_stats([0, 1, 2, 3], C, frow, [7], 4, ref, sc)
    assert bal["per_layer"]["7"]["gini"] < typ["per_layer"]["7"]["gini"]
    assert bal["representativeness"]["d_post_warm"] > typ["representativeness"]["d_post_warm"]


def _fake(minv, p10, b128, gini, frac=0.4, zero=0):
    return dict(tail=dict(all=dict(min=minv, p5=p10, p10=p10, below_64=0, below_128=b128, below_256=0, zero=zero,
                                   gini_layer_mean=gini, effective_sources_median=1.0, frac_below_128=frac)),
                repr=dict(d_post_warm=0, set_js=0, set_entropy_mean=0, seq_entropy_mean=0,
                          aggregate_l1_by_layer={"1": 0}))


def test_decision_gate_cases():
    def stats(cov):
        s = {}
        for B in X.BUDGETS:
            for sd in X.SEEDS:
                s[f"FR_B{B}_s{sd}"] = _fake(50, 100, 100, 0.12)
                s[f"TR_B{B}_s{sd}"] = _fake(50, 100, 100, 0.12)
                s[f"FC_B{B}_s{sd}"] = cov
                s[f"TC_B{B}_s{sd}"] = _fake(50, 100, 100, 0.12)
        return s
    assert X.decide(stats(_fake(55, 115, 40, 0.10)), True)[0] == "EXTREME-SCARCITY-PROMISING"
    assert X.decide(stats(_fake(40, 115, 40, 0.10)), True)[0] == "EXTREME-SCARCITY-PARTIAL"     # min worse
    assert X.decide(stats(_fake(40, 90, 120, 0.15)), True)[0] == "EXTREME-SCARCITY-NO-OPPORTUNITY"
    assert X.decide(stats(_fake(55, 115, 40, 0.10)), False)[0] == "CPU-RUN-BLOCKED"


# ------------------------------------------------------------- produced selections
@pytest.fixture(scope="module")
def sel():
    if not os.path.exists(SEL):
        pytest.skip("matched selections not produced")
    return json.load(open(SEL))


def test_selections_sizes_prefixes_pools(sel):
    ws = json.load(open(os.path.join(ROOT, "runs_typicality_matched", "warm_starts.json")))
    typ = set(json.load(open(os.path.join(ROOT, "runs_typicality", "pool_typical_half.json")))["ids"])
    for B in X.BUDGETS:
        for s in X.SEEDS:
            W = ws["seeds"][str(s)]["ids"][:B // 4]
            assert sel["warm_starts"][f"B{B}_s{s}"]["ids"] == W and set(W) <= typ
            for c in X.STRATS:
                ch = sel["sets"][f"{c}_B{B}_s{s}"]["chosen"]
                assert len(ch) == B == len(set(ch)) and ch[:B // 4] == W
                assert all(0 <= x < 2000 for x in ch)
                if X.STRATS[c][1] == "typical_half":
                    assert set(ch) <= typ


def test_selections_reproducible_and_atypicality_consistent(sel):
    from moe_calib import typicality as TY
    C, layers, E, full, typ, hal, ws, pools = LB.load_inputs(os.path.join(ROOT, "runs_typicality_matched"),
                                                              os.path.join(ROOT, "runs_typicality"))
    frow = {s: i for i, s in enumerate(full)}
    sc = TY.sequence_scores(C, len(layers), E, TY.pool_reference(C, len(layers), E))
    for c in X.STRATS:
        ch, _ = X.select(c, 4, 1, C, full, typ, layers, E, ws)
        assert ch == sel["sets"][f"{c}_B4_s1"]["chosen"]
        st = sel["stats"][f"{c}_B8_s2"]
        ch8 = sel["sets"][f"{c}_B8_s2"]["chosen"]
        assert st["repr"]["d_post_warm"] == pytest.approx(sc["d"][[frow[s] for s in ch8[2:]]].mean())
        assert st["tail"]["all"]["min"] == C[[frow[s] for s in ch8]].sum(0).min()


def test_b16_backward_compatible_with_low_budget_sweep():
    path = os.path.join(ROOT, "runs_low_budget", "selected_sets.json")
    if not os.path.exists(path):
        pytest.skip("low-budget selections absent")
    old = json.load(open(path))
    C, layers, E, full, typ, hal, ws, pools = LB.load_inputs(os.path.join(ROOT, "runs_typicality_matched"),
                                                              os.path.join(ROOT, "runs_typicality"))
    for c in ("FC", "TC"):
        wf = {"seeds": {"0": {"ids": ws["seeds"]["0"]["ids"][:4]}}}
        ch_lb, _ = LB.select(c, 16, 0, C, full, typ, layers, E, wf)
        ch_x, _ = X.select(c, 16, 0, C, full, typ, layers, E, ws)
        assert ch_lb == ch_x == old["sets"][f"{c}_B16_s0"]["chosen"]
