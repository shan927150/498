"""CPU tests for the low-budget allocation sweep."""
import json
import os
import random
import sys

import numpy as np
import pytest

import e3_allocation as E3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import low_budget as LB            # noqa: E402
import report_low_budget as RLB    # noqa: E402

SEL = os.path.join(ROOT, "runs_low_budget", "selected_sets.json")


# ------------------------------------------------------------- schedule / warm starts

@pytest.mark.parametrize("B,warm,rounds", [(16, 4, [6, 6]), (32, 8, [12, 12]), (64, 16, [24, 24]), (128, 32, [48, 48])])
def test_budget_schedule(B, warm, rounds):
    w, sched = E3.round_schedule(B, 4, 2)
    assert w == warm and [s[2] for s in sched] == rounds and w + sum(rounds) == B


def test_nested_warm_prefixes_and_hashes():
    ws = dict(pools={"full": "f"}, pool_dir="x", eligible={}, seeds={
        "0": dict(ids=list(range(100, 132)), hash=E3.warm_hash(list(range(100, 132))))})
    prev = []
    for B in (16, 32, 64, 128):
        wf = LB.warm_file(ws, B)
        ids = wf["seeds"]["0"]["ids"]
        assert ids == list(range(100, 100 + B // 4)) and ids[:len(prev)] == prev
        assert wf["seeds"]["0"]["hash"] == E3.warm_hash(ids) and wf["warm_size"] == B // 4
        prev = ids


def test_controller_accepts_prefix_warm_start_and_rejects_wrong_length():
    g = np.random.default_rng(0)
    C = g.poisson(4.0, size=(60, 2 * 6)).astype(float)
    ids = list(range(1000, 1060))
    ch, w, traj = E3.select_lp_utility("coverage_lp", C, ids, [1, 2], 6, 16, 4, 2, random.Random(0), None,
                                       warm_start=[1000, 1001, 1002, 1003])
    assert len(ch) == 16 == len(set(ch)) and ch[:4] == [1000, 1001, 1002, 1003]
    assert [t["round_budget"] for t in traj] == [6, 6] and [t["regime_m"] for t in traj] == [4, 10]
    with pytest.raises(ValueError, match="schedule needs"):
        E3.select_lp_utility("coverage_lp", C, ids, [1, 2], 6, 16, 4, 2, random.Random(0), None,
                             warm_start=[1000, 1001, 1002])


# ------------------------------------------------------------- zero exposure / lower tail

def test_zero_exposure_experts_trigger_existing_exploration():
    """Warm start never routes to experts 4 and 5; the existing exploration cover
    reaches them; the 1/n_e utility is never evaluated at n_e = 0."""
    C = np.zeros((40, 6))
    C[:, :4] = 5
    C[30:, 4] = 3                       # only rows 30.. reach expert 4
    C[35:, 5] = 3                       # only rows 35.. reach expert 5
    ids = list(range(40))
    ch, w, traj = E3.select_lp_utility("coverage_lp", C, ids, [1], 6, 16, 4, 2, random.Random(0), None,
                                       warm_start=[0, 1, 2, 3])
    t0 = traj[0]
    assert t0["n_zero_exposure"] == 2 and t0["n_exploration_targets"] == 2
    assert t0["n_exploration_targets_reached"] == 2 and t0["n_exploration_sequences"] >= 1
    n = C[ch].sum(0)
    assert (n > 0).all()


def test_coverage_improves_lower_tail_synthetic():
    """Rows are either 'common' (experts 0-3) or 'rare' (expert 5); coverage takes rare rows."""
    g = np.random.default_rng(1)
    C = np.zeros((80, 6))
    C[:, :4] = g.poisson(10, size=(80, 4))
    C[60:, 5] = 20
    C[:, 4] = 5
    ids = list(range(80))
    warm = [0, 1, 2, 60]
    cov, _, _ = E3.select_lp_utility("coverage_lp", C, ids, [1], 6, 16, 4, 2, random.Random(0), None, warm_start=warm)
    rnd = warm + random.Random(0).sample([s for s in ids if s not in warm], 12)
    assert C[cov].sum(0)[5] > C[rnd].sum(0)[5]
    assert LB.gini(C[cov].sum(0)) < LB.gini(C[rnd].sum(0))


def test_balance_better_but_separate_metric_worse():
    """Coverage lowers Gini yet a separate (atypicality) metric gets worse: the two
    are distinct quantities and the stats module reports both."""
    C = np.zeros((40, 4))
    C[:30] = [8, 8, 8, 0]               # typical rows: never expert 3
    C[30:] = [0, 0, 0, 24]              # atypical rows: only expert 3
    frow = {s: s for s in range(40)}
    from moe_calib import typicality as TY
    ref = TY.pool_reference(C, 1, 4)
    sc = TY.sequence_scores(C, 1, 4, ref)
    bal = LB.set_stats(list(range(0, 6)) + [30, 31], C, frow, [7], 4, ref, sc)
    typ = LB.set_stats(list(range(0, 8)), C, frow, [7], 4, ref, sc)
    assert bal["per_layer"]["7"]["gini"] < typ["per_layer"]["7"]["gini"]
    assert bal["representativeness"]["d_mean"] > typ["representativeness"]["d_mean"]


# ------------------------------------------------------------- statistics

def test_exposure_stats_and_gini():
    X = np.array([[0, 1, 10, 100], [0, 1, 10, 100], [0, 0, 0, 312]], float)
    st, n = LB.exposure_stats(X)
    assert list(n) == [0, 2, 20, 512]
    assert st["zero"] == 1 and st["below_64"] == 3 and st["below_512"] == 3 and st["min"] == 0
    assert st["distinct_sources_min"] == 0 and st["median"] == 11
    assert LB.gini(np.ones(5)) == pytest.approx(0) and LB.gini([0, 0, 0, 1]) == pytest.approx(0.75)
    q = np.array([[1.0], [1.0]])
    assert LB.eff_sources(q)[0] == pytest.approx(2.0)


def test_paired_gain_and_doubling_return():
    E = {("FC", 16, 0): 1.1, ("FR", 16, 0): 1.0, ("FR", 32, 0): 0.9}
    assert RLB.coverage_gain(E, "FC", "FR", 16, 0) == pytest.approx(0.1)
    assert RLB.doubling_return(E, "FR", 16, 0) == pytest.approx(0.1)


def test_decision_rule_cases():
    flat = {fam: {B: {L: 0.0 for L in "1 8 16 23".split()} for B in (16, 32, 64, 128)} for fam in ("full", "typical")}
    p95 = {fam: {B: 0.0 for B in (16, 32, 64, 128)} for fam in ("full", "typical")}
    G = {"full": {16: [-.01, -.008, -.012], 32: [.01] * 3, 64: [.02] * 3, 128: [.025] * 3},
         "typical": {B: [.001] * 3 for B in (16, 32, 64, 128)}}
    assert RLB.decide(G, flat, p95)[0] == "LOW-BUDGET-COVERAGE-PROMISING"
    G["full"][16] = [-.01, .002, -.012]                      # one seed positive -> not promising
    assert RLB.decide(G, flat, p95)[0] == "LOW-BUDGET-COVERAGE-PARTIAL"
    G["full"] = {16: [.03] * 3, 32: [.02] * 3, 64: [.025] * 3, 128: [.025] * 3}
    assert RLB.decide(G, flat, p95)[0] == "LOW-BUDGET-COVERAGE-NO-GAIN"


# ------------------------------------------------------------- prepared selections

@pytest.fixture(scope="module")
def sel():
    if not os.path.exists(SEL):
        pytest.skip("selections not prepared")
    return json.load(open(SEL))


def test_prepared_selections_valid_and_warm_identical(sel):
    ws = json.load(open(os.path.join(ROOT, "runs_typicality_matched", "warm_starts.json")))
    typ = set(json.load(open(os.path.join(ROOT, "runs_typicality", "pool_typical_half.json")))["ids"])
    for B in (16, 32, 64, 128):
        for s in (0, 1, 2):
            W = ws["seeds"][str(s)]["ids"][:B // 4]
            for c in LB.STRATS:
                ch = sel["sets"][f"{c}_B{B}_s{s}"]["chosen"]
                assert len(ch) == B == len(set(ch)) and ch[:B // 4] == W
                assert all(0 <= x < 2000 for x in ch)              # no tune/test leakage
                if LB.STRATS[c][1] == "typical_half":
                    assert set(ch) <= typ


def test_prepared_selections_reproducible_and_b128_backward_compatible(sel):
    C, layers, E, full, typ, hal, ws, pools = LB.load_inputs(os.path.join(ROOT, "runs_typicality_matched"),
                                                              os.path.join(ROOT, "runs_typicality"))
    wf = LB.warm_file(ws, 16)
    for c in LB.STRATS:
        ch, _ = LB.select(c, 16, 1, C, full, typ, layers, E, wf)
        assert ch == sel["sets"][f"{c}_B16_s1"]["chosen"]
    import report_typicality_matched as RTM
    old = json.load(open(RTM.run_path(os.path.join(ROOT, "runs_typicality_matched"), "full", "coverage_lp", 0, ws)))
    assert old["chosen"] == sel["sets"]["FC_B128_s0"]["chosen"]
