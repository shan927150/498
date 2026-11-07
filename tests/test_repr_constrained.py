"""CPU tests for the representativeness-constrained Coverage-LP (coverage_lp_repr)."""
import json
import os
import random
import sys

import numpy as np
import pytest

import e3_allocation as E3
from conftest import LAYERS, tiny_model
from moe_calib import alloc_models as AM
from moe_calib import model_utils as MU
from moe_calib import repr_constraint as RC
from moe_calib import typicality as TY
from moe_calib.data_utils import load_meta, split_ids
from test_allocation_minimal import frozen_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import typicality_pools as TP          # noqa: E402
import typicality_warm_starts as TW    # noqa: E402


# ------------------------------------------------------------- synthetic data

def synth(S=80, layers=2, E=4, seed=0, n_atyp=20):
    """Typical rows route ~uniformly; the first n_atyp rows concentrate on the
    rare expert 3 of each layer (high coverage value, high d)."""
    g = np.random.default_rng(seed)
    C = np.zeros((S, layers * E))
    for li in range(layers):
        base = g.multinomial(64, [0.32, 0.32, 0.32, 0.04], size=S).astype(float)
        base[:n_atyp] = g.multinomial(64, [0.1, 0.1, 0.1, 0.7], size=n_atyp)
        C[:, li * E:(li + 1) * E] = base
    return C


def d_manual(C, L, E):
    ref = np.stack([C[:, l * E:(l + 1) * E].sum(0) / C[:, l * E:(l + 1) * E].sum() for l in range(L)])
    out = []
    for s in range(C.shape[0]):
        out.append(np.mean([np.abs(C[s, l * E:(l + 1) * E] / C[s, l * E:(l + 1) * E].sum() - ref[l]).sum()
                            for l in range(L)]))
    return np.array(out)


def run(strategy, C, ids, B=24, K=4, warm=None, d=None, tau=None, seed=0):
    return E3.select_lp_utility(strategy, C, ids, [1, 2], 4, B, K, 2, random.Random(seed), None,
                                warm_start=warm, repr_d=d, repr_tau=tau)


# ------------------------------------------------------------- typicality

def test_typicality_matches_definition_and_frozen_reference():
    C = synth()
    d, ref = RC.pool_typicality(C, 2, 4)
    np.testing.assert_allclose(d, d_manual(C, 2, 4), rtol=0, atol=1e-12)
    np.testing.assert_allclose(ref, TY.pool_reference(C, 2, 4))
    # identical to the existing typicality module (same function as the Typical-half pools)
    np.testing.assert_array_equal(d, TY.sequence_scores(C, 2, 4, ref)["d"])
    tau, rule = RC.resolve_tau("median", d)
    assert tau == float(np.median(d)) and "median" in rule
    bad = C.copy()
    bad[0, :4] = 0
    with pytest.raises(ValueError, match="malformed"):
        RC.pool_typicality(bad, 2, 4)


def test_typicality_matches_stored_artifact_if_present():
    sc = os.path.join(ROOT, "runs_typicality", "typicality_scores.csv")
    rp = os.path.join(ROOT, "runs_typicality", "routing_profile_v2.npz")
    if not (os.path.exists(sc) and os.path.exists(rp)):
        pytest.skip("real typicality artifacts not present")
    import csv
    z = np.load(rp)
    d, _ = RC.pool_typicality(z["C"], len(z["layers"]), int(z["E"]))
    rows = list(csv.DictReader(open(sc)))
    assert [int(r["id"]) for r in rows] == z["pool_ids"].tolist()
    assert np.max(np.abs(np.array([float(r["d"]) for r in rows]) - d)) <= 1e-9


# ------------------------------------------------------------- budget accounting

def test_budget_pacing_reserve_and_cumulative_accounting():
    rb = RC.ReprBudget(0.5, 4)
    d_cand = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 0.4])
    cap, info = rb.round_cap(d_cand, 2)
    assert info["paced_cap"] == pytest.approx(1.0)                   # 2.0 * 2 / 4
    assert info["reserve_future"] == pytest.approx(0.3 + 0.4)        # sorted positions [2, 4)
    assert cap == pytest.approx(1.0)
    rb.commit([0.2, 0.6])                                           # under-spend 0.2 -> carried forward
    assert rb.spent == pytest.approx(0.8) and rb.n_taken == 2
    cap2, info2 = rb.round_cap(np.array([0.1, 0.3, 0.9, 1.0]), 2)
    assert info2["k_future"] == 0 and cap2 == pytest.approx(1.2)     # last round gets the rest exactly
    rb.commit([0.3, 0.9])
    fin = rb.final_check()
    assert fin["satisfied"] and fin["mean_post_warm_d"] == pytest.approx(0.5)
    assert fin["per_round_spent"] == pytest.approx([0.8, 1.2]) and fin["tightness"] == pytest.approx(1.0)
    with pytest.raises(RC.ReprInfeasible):
        rb.commit([0.01])


def test_infeasible_constraint_is_detected_never_relaxed():
    rb = RC.ReprBudget(0.05, 4)
    with pytest.raises(RC.ReprInfeasible, match="cap"):
        rb.round_cap(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 2)
    C = synth()
    d, _ = RC.pool_typicality(C, 2, 4)
    with pytest.raises(RC.ReprInfeasible):
        run("coverage_lp_repr", C, list(range(80)), d=d, tau=float(d.min()) / 2)
    with pytest.raises(RC.ReprInfeasible):
        E3.solve_lp_constrained(C[:5], np.ones(8), np.full(8, 10.0), 2, d[:5], 0.0)


# ------------------------------------------------------------- rounding

def test_fractional_feasible_but_naive_rounding_infeasible_is_repaired():
    """Budget 1, cap 0.5: the LP mixes the high-coverage atypical row A (d=.9)
    with the typical row B (d=.1) at z = (.5, .5); naive top-z rounding takes A
    (d = .9 > .5); constraint-aware rounding takes B."""
    Cc = np.array([[0, 0, 0, 10.0], [5.0, 0, 0, 1.0], [0, 5.0, 0, 0]])
    value, cap = np.array([0.1, 0.1, 0.1, 1.0]), np.array([10.0, 10.0, 10.0, 10.0])
    d = np.array([0.9, 0.1, 0.2])
    z, info = E3.solve_lp_constrained(Cc, value, cap, 1, d, 0.5)
    assert d @ z <= 0.5 + 1e-9 and 0 < z[0] < 1
    take, rows, log = RC.select_round_repr(Cc, [10, 11, 12], value, cap, np.zeros(4, bool), 1, d, 0.5,
                                           random.Random(0), E3.solve_lp_constrained)
    assert log["naive_rounding_violates"] and log["naive_rounding_d_sum"] == pytest.approx(0.9)
    assert take == [11] and log["n_rounding_repairs"] == 1
    assert log["round_d_sum"] <= 0.5 and log["rounding_objective_loss"] > 0


def test_rounding_deterministic_unique_exact_budget_and_feasible():
    g = np.random.default_rng(5)
    Cc = g.poisson(3.0, size=(50, 8)).astype(float)
    d = g.uniform(0.1, 0.9, 50)
    value, cap = np.ones(8) / 30, np.full(8, 30.0)
    cap_d = RC.minsum(d, 10) + 0.4
    outs = [RC.select_round_repr(Cc, list(range(200, 250)), value, cap, np.zeros(8, bool), 10, d, cap_d,
                                 random.Random(1), E3.solve_lp_constrained) for _ in range(2)]
    assert outs[0][0] == outs[1][0]
    take, rows, log = outs[0]
    assert len(take) == 10 == len(set(take)) and log["round_d_sum"] <= cap_d + RC.TOL
    assert (log["n_exploration_sequences"] + log["n_lp_positive_value_selected"]
            + log["n_zero_marginal_random_fill"] + log["n_random_fallback_sequences"]) == 10


def test_exploration_and_fill_respect_constraint_with_zero_count_experts():
    """Warm start never touches expert 7 -> exploration target; the only rows
    that reach it are atypical; the cover must not break the cap."""
    g = np.random.default_rng(2)
    Cc = g.poisson(3.0, size=(30, 8)).astype(float)
    Cc[:, 7] = 0
    Cc[:3, 7] = 5
    d = np.full(30, 0.2)
    d[:3] = 0.95
    targets = np.zeros(8, bool)
    targets[7] = True
    take, rows, log = RC.select_round_repr(Cc, list(range(30)), np.zeros(8), np.zeros(8), targets, 4,
                                           d, 0.2 * 4, random.Random(0), E3.solve_lp_constrained)
    assert not set(rows) & {0, 1, 2} and log["n_exploration_skipped_for_constraint"] >= 1
    assert log["used_random_utility_fallback"] and log["round_d_sum"] <= 0.8 + RC.TOL
    take2, rows2, log2 = RC.select_round_repr(Cc, list(range(30)), np.zeros(8), np.zeros(8), targets, 4,
                                              d, 0.95 + 0.6, random.Random(0), E3.solve_lp_constrained)
    assert log2["n_exploration_sequences"] == 1 and rows2[0] in (0, 1, 2)


# ------------------------------------------------------------- controller

def test_unconstrained_selects_atypical_constrained_more_representative():
    C = synth()
    ids = list(range(1000, 1080))                        # non-zero-based original ids
    d, _ = RC.pool_typicality(C, 2, 4)
    tau = float(np.median(d))
    warm = [1040, 1041, 1042, 1043, 1044, 1045]
    a, _, _ = run("coverage_lp", C, ids, warm=warm)
    b, _, tb = run("coverage_lp_repr", C, ids, warm=warm, d=d, tau=tau)
    row = {s: i for i, s in enumerate(ids)}
    da = np.mean([d[row[s]] for s in a[6:]])
    db = np.mean([d[row[s]] for s in b[6:]])
    assert da > tau                                      # unconstrained coverage goes atypical
    assert db <= tau + RC.TOL and db < da                # constrained: representative on average
    fin = tb[-1]["repr_final"]
    assert fin["satisfied"] and fin["spent"] == pytest.approx(sum(d[row[s]] for s in b[6:]))
    # 5: not hard Typical-half filtering -> some post-warm picks above the median
    assert any(d[row[s]] > tau for s in b[6:])
    assert a[:6] == b[:6] == warm and len(set(b)) == 24


def test_round_to_round_budget_tracking_logged():
    C = synth(seed=3)
    ids = list(range(80))
    d, _ = RC.pool_typicality(C, 2, 4)
    tau = float(np.median(d))
    ch, w, traj = run("coverage_lp_repr", C, ids, warm=list(range(30, 36)), d=d, tau=tau)
    spent = 0.0
    for t in traj:
        assert t["round_d_sum"] <= t["round_d_cap"] + RC.TOL
        assert t["remaining_before"] == pytest.approx(18 * tau - spent)
        spent += t["round_d_sum"]
        assert t["cumulative_spent"] == pytest.approx(spent)
        for k in ("naive_rounding_violates", "n_rounding_repairs", "rounding_objective_loss",
                  "paced_cap", "reserve_future", "lp_fractional_d_sum"):
            assert k in t
    assert spent == pytest.approx(sum(d[s] for s in ch[6:])) and spent <= 18 * tau + RC.TOL


def test_reproducible_and_coverage_lp_backward_compatible():
    C = synth(seed=1)
    ids = list(range(80))
    d, _ = RC.pool_typicality(C, 2, 4)
    warm = list(range(40, 46))
    x = run("coverage_lp_repr", C, ids, warm=warm, d=d, tau=float(np.median(d)))
    y = run("coverage_lp_repr", C, ids, warm=warm, d=d, tau=float(np.median(d)))
    assert x[0] == y[0]
    # coverage_lp itself is untouched: no constraint fields, identical to a direct replay
    a, _, ta = run("coverage_lp", C, ids, warm=warm)
    b, _, _ = run("coverage_lp", C, ids, warm=warm)
    assert a == b and "repr_final" not in ta[-1] and "round_d_sum" not in ta[0]
    with pytest.raises(ValueError, match="repr_d"):
        run("coverage_lp_repr", C, ids, warm=warm)
    assert "coverage_lp_repr" not in AM.LP_UTILITY_STRATEGIES
    assert AM.CONSTRAINED_LP_STRATEGIES["coverage_lp_repr"] == "coverage_lp"


# ------------------------------------------------------------- end to end (tiny model)

@pytest.fixture(scope="module")
def e2e(cache_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("repr")
    pool_dir, out = root / "pools", root / "out"
    sp = split_ids(load_meta(cache_dir))
    cfg_p = root / "frozen.json"
    cfg_p.write_text(json.dumps(frozen_for(LAYERS, sp["definition"])))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MU, "load_model", lambda *a, **k: tiny_model())
        E3.main(["--cache-dir", cache_dir, "--outdir", str(pool_dir), "--B", "12", "--strategy",
                 "random", "--frozen-config", str(cfg_p), "--routing-cache"])
        TP.build(str(pool_dir / "routing_profile_v2.npz"), str(cfg_p), str(pool_dir), half=20,
                 pool_seed=1, min_size=12)
        ws = TW.build(str(pool_dir), str(cfg_p), str(out), seeds=(0, 1), warm=4)
        common = ["--cache-dir", cache_dir, "--outdir", str(out), "--B", "12",
                  "--frozen-config", str(cfg_p), "--routing-cache",
                  "--warm-start-file", str(out / "warm_starts.json")]
        res = {}
        for seed in (0, 1):
            for st in ("random", "coverage_lp", "coverage_lp_repr"):
                extra = (["--typicality-scores", str(pool_dir / "typicality_scores.csv")]
                         if st == "coverage_lp_repr" else [])
                res[(st, seed)] = json.load(open(E3.main(common + ["--strategy", st, "--seed", str(seed)] + extra)))
        res["again"] = json.load(open(E3.main(common + ["--strategy", "coverage_lp_repr", "--seed", "0"])))
    pools = {k: v["ids"] for k, v in json.load(open(pool_dir / "pools.json"))["pools"].items()}
    return dict(res=res, ws=ws, pools=pools, split=sp, common=common, cfg_p=cfg_p, pool_dir=pool_dir,
                cache_dir=cache_dir)


def test_e2e_B_warm_unique_split_isolation_and_constraint(e2e):
    r, sp = e2e["res"], e2e["split"]
    held = set(sp["tune"]) | set(sp["test"])
    for seed in (0, 1):
        want = e2e["ws"]["seeds"][str(seed)]["ids"]
        x = r[("coverage_lp_repr", seed)]
        ch = x["chosen"]
        assert len(ch) == 12 == len(set(ch)) and ch[:4] == want and x["warm_start"] == want
        assert all(r[(st, seed)]["chosen"][:4] == want for st in ("random", "coverage_lp"))
        assert set(ch) <= set(e2e["pools"]["full"]) and not set(ch) & held
        rc = x["summary"]["repr_constraint"]
        assert rc["satisfied"] and rc["n_post"] == 8 and rc["tau_spec"] == "median"
        assert rc["scores_max_abs_diff"] <= 1e-9 and rc["mean_post_warm_d"] <= rc["tau"] + RC.TOL
        z = np.load(e2e["pool_dir"] / "routing_profile_v2.npz")
        d, _ = RC.pool_typicality(z["C"], len(LAYERS), int(z["E"]))
        row = {int(s): i for i, s in enumerate(z["pool_ids"])}
        assert rc["tau"] == float(np.median(d))
        assert np.mean([d[row[s]] for s in ch[4:]]) <= rc["tau"] + RC.TOL
    assert r["again"]["chosen"] == r[("coverage_lp_repr", 0)]["chosen"]


def test_e2e_frozen_gptq_and_eval_universe_identical(e2e):
    r = e2e["res"]
    univ = {json.dumps(v["eval_universe"]) for v in r.values()}
    assert len(univ) == 1
    for v in r.values():
        s = v["summary"]
        assert (s["percdamp"], s["bits"], s["eval_split"]) == (1.0, 4, "test")
        assert s["routing_counts_match_hessian_tokens"]
    base = e2e["common"] + ["--strategy", "coverage_lp_repr", "--seed", "0"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MU, "load_model", lambda *a, **k: tiny_model())
        with pytest.raises(ValueError, match="frozen"):
            E3.main(base + ["--percdamp", "0.01"])
        with pytest.raises(ValueError, match="full candidate pool"):
            E3.main(base + ["--candidate-subset-file", str(e2e["pool_dir"] / "pool_typical_half.json")])


def test_e2e_typicality_scores_mismatch_rejected(e2e, tmp_path):
    src = (e2e["pool_dir"] / "typicality_scores.csv").read_text().splitlines()
    hdr, first = src[0], src[1].split(",")
    first[1] = str(float(first[1]) + 0.01)
    bad = tmp_path / "bad.csv"
    bad.write_text("\n".join([hdr, ",".join(first)] + src[2:]) + "\n")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MU, "load_model", lambda *a, **k: tiny_model())
        with pytest.raises(ValueError, match="differs"):
            E3.main(e2e["common"] + ["--strategy", "coverage_lp_repr", "--seed", "0",
                                     "--typicality-scores", str(bad)])
