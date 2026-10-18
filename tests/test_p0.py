"""Tests for the P0 research-validity fixes (+ fixed blocks, E1 run identity)
and the follow-up edge cases (low-count labels, support, fallbacks, budgets)."""
import json
import os
import random
import shutil

import numpy as np
import pytest
import torch

import e1_label_sufficiency as E1
import e2_signals as E2
import e3_allocation as E3
import support_diagnostic as SD
from conftest import E, LAYERS, POOL, HOLDOUT, TUNE, tiny_model
from moe_calib import model_utils as MU
from moe_calib.data_utils import load_meta, split_ids
from moe_calib.gptq import gptq_quantize
from moe_calib.hessian import assign_blocks, block_of
from moe_calib.profiler import Profiler
from moe_calib.quant_eval import fp_weights, quantize_from_acc_or_fallback
from moe_calib.runid import e1_run_dir, load_e1_runs
from moe_calib.utility import (SIGNALS, allocation_terms, exploration_cover,
                               predicted_delta)

CHAR_SEEDS = (0, 1, 2)

STRATEGIES = [["--strategy", "random"],
              ["--strategy", "random_plus", "--delta-seqs", "2"],
              ["--strategy", "coverage"],
              ["--strategy", "tlp"],
              ["--strategy", "tlp", "--signal", "count"],
              ["--strategy", "tlp", "--weighting", "usage"],
              ["--strategy", "tlp", "--allocation", "relse_precision"],
              ["--strategy", "oracle"]]


@pytest.fixture(scope="module")
def pipeline(cache_dir, tmp_path_factory):
    """E1 (2 characterization seeds) -> E2 -> E3 for every strategy, recording
    which sequence ids each Profiler.run touched."""
    out = str(tmp_path_factory.mktemp("runs"))
    calls = []
    orig_run = Profiler.run

    def rec(self, cache, seq_ids, *a, **k):
        calls.append((self.collect_holdout, list(seq_ids)))
        return orig_run(self, cache, seq_ids, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MU, "load_model", lambda *a, **k: tiny_model())
        mp.setattr(Profiler, "run", rec)
        common = ["--cache-dir", cache_dir, "--outdir", out]
        for seed in CHAR_SEEDS:
            E1.main(common + ["--n0", "4", "8", "--reps", "2", "--min-paired-seqs", "3",
                              "--char-seed", str(seed)])
        n_e12 = len(calls)
        art = E2.main(common + ["--min-char-seeds", "3"])
        n_e2 = len(calls)
        e3 = {}
        for strat in STRATEGIES:
            path = E3.main(common + ["--B", "12"] + strat)
            e3[" ".join(strat)] = json.load(open(path))
    return dict(out=out, calls=calls, n_e1=n_e12, n_e2=n_e2, art=art, e3=e3,
                cache_dir=cache_dir)


# 1 -----------------------------------------------------------------------
def test_splits_disjoint(cache_dir):
    sp = split_ids(load_meta(cache_dir))
    pool, tune, test = map(set, (sp["pool"], sp["tune"], sp["test"]))
    assert not (pool & tune) and not (pool & test) and not (tune & test)
    assert len(pool) == POOL and len(tune) == TUNE and len(test) == HOLDOUT - TUNE
    assert sp["definition"]["tune"] == [POOL, POOL + TUNE]
    with pytest.raises(AssertionError):
        split_ids(load_meta(cache_dir), tune_size=HOLDOUT)


# 2 -----------------------------------------------------------------------
def test_e1_uses_tune_e3_uses_test(pipeline):
    sp = split_ids(load_meta(pipeline["cache_dir"]))
    calls = pipeline["calls"]
    e1_ho = [ids for ho, ids in calls[:pipeline["n_e1"]] if ho]
    e3_ho = [ids for ho, ids in calls[pipeline["n_e2"]:] if ho]
    assert e1_ho and all(ids == sp["tune"] for ids in e1_ho)
    assert e3_ho and all(ids == sp["test"] for ids in e3_ho)
    # no calibration/characterization replay ever touches either holdout split
    assert all(set(ids) <= set(sp["pool"]) for ho, ids in calls if not ho)
    for res in pipeline["e3"].values():
        assert res["summary"]["eval_split"] == "test"
        assert res["summary"]["split"] == pipeline["art"]["split"] == sp["definition"]


# 3 -----------------------------------------------------------------------
def test_e3_auto_loads_e2_model(pipeline, cache_dir, tmp_path):
    art = json.load(open(os.path.join(pipeline["out"], "e2_model_b4.json")))
    assert art["selected_signal"] in SIGNALS
    assert art["selection_criterion"] == "leave_one_char_seed_out_mse"
    assert art["characterization_seeds"] == list(CHAR_SEEDS) and art["preliminary"] is False
    s = pipeline["e3"]["--strategy tlp"]["summary"]
    assert s["signal"] == art["selected_signal"]
    assert s["signal_model"].endswith("e2_model_b4.json")
    with pytest.raises(FileNotFoundError):
        E3.main(["--strategy", "tlp", "--cache-dir", cache_dir, "--outdir", str(tmp_path)])
    # a preliminary (one-seed) model is refused unless explicitly allowed
    json.dump({**art, "preliminary": True}, open(tmp_path / "e2_model_b4.json", "w"))
    with pytest.raises(RuntimeError, match="PRELIMINARY"):
        E3.main(["--strategy", "tlp", "--cache-dir", cache_dir, "--outdir", str(tmp_path)])


# 4 -----------------------------------------------------------------------
def test_signal_changes_utilities(pipeline):
    rng = np.random.default_rng(0)
    J = 20
    stats = {"count": rng.integers(1, 500, J).astype(float),
             **{s: rng.random(J) for s in SIGNALS if s != "count"}}
    art = pipeline["art"]
    dh = {s: predicted_delta(art, s, stats)[0] for s in SIGNALS}
    for s in SIGNALS:
        if s != "count":
            assert not np.allclose(dh[s], dh["count"]), s
    assert not np.allclose(dh["drift"], dh["relse"])
    v_d = allocation_terms(stats["count"], dh["drift"], np.ones(J))[0]
    v_r = allocation_terms(stats["count"], dh["relse"], np.ones(J))[0]
    assert not np.allclose(v_d, v_r)


# 5 -----------------------------------------------------------------------
def test_allocation_objective_synthetic():
    C = np.array([[10, 0], [0, 10], [5, 5]], float)
    # value ranks experts by predicted benefit per token
    v, cap = allocation_terms(np.array([10., 10.]), np.array([0.5, 0.0]), np.ones(2))
    assert np.allclose(v, [0.05, 0.0]) and np.allclose(cap, [10, 10])
    assert np.argmax(E3.solve_lp(C, v, cap, 1)) == 0
    # capacity = n saturates: with n=4 each, the 5/5 sequence covers both
    v, cap = allocation_terms(np.array([4., 4.]), np.array([0.5, 0.5]), np.ones(2))
    assert np.argmax(E3.solve_lp(C, v, cap, 1)) == 2
    # weighting is a separate switch: uniform vs usage can flip the choice
    n, dh = np.array([10., 10.]), np.array([0.4, 0.5])
    vu, cap = allocation_terms(n, dh, np.ones(2))
    vp, _ = allocation_terms(n, dh, np.array([0.9, 0.1]))
    assert np.argmax(E3.solve_lp(C[:2], vu, cap, 1)) == 1
    assert np.argmax(E3.solve_lp(C[:2], vp, cap, 1)) == 0
    # negative / undefined benefit -> zero value; zero exposure -> no capacity
    v, cap = allocation_terms(np.array([10., 0.]), np.array([-0.3, np.nan]), np.ones(2))
    assert np.allclose(v, 0) and cap[1] == 0
    # exploration covers the zero-exposure expert with the fewest sequences
    picks, reached = exploration_cover(C, np.array([False, True]), 2)
    assert picks == [1] and reached.tolist() == [False, True]


# 6 -----------------------------------------------------------------------
def test_blocks_fixed_across_rounds(cache_dir):
    r1 = [5, 42, 7]
    r2 = [3, 100, 42, 7, 5]
    b1 = dict(zip(r1, assign_blocks(r1, 4, 0)))
    b2 = dict(zip(r2, assign_blocks(r2, 4, 0)))
    assert all(b1[s] == b2[s] for s in r1)
    assert block_of(42, 4, 0) == b1[42] == b2[42]
    # statistics depend on the chosen SET, not on selection order
    model, _ = tiny_model()
    chosen = list(range(0, 24, 2))
    a = E3.expert_stats(model, cache_dir, LAYERS, chosen, {}, 4, 0)
    b = E3.expert_stats(model, cache_dir, LAYERS, chosen[::-1], {}, 4, 0)
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-5, equal_nan=True)


# 7 -----------------------------------------------------------------------
def test_char_seed_changes_draws():
    pool = list(range(200))
    d0 = E1.make_draws(pool, [8, 16], 2, 0)
    assert d0 == E1.make_draws(pool, [8, 16], 2, 0)
    d1 = E1.make_draws(pool, [8, 16], 2, 1)
    assert d0[8]["base"] != d1[8]["base"] and d0[16]["base"] != d1[16]["base"]


# 8 -----------------------------------------------------------------------
def test_incompatible_e1_resume_rejected(tmp_path):
    fp = dict(model="m", data_seed=0, char_seed=0, layers=[1], n0=[8], reps=1, K=4,
              bits=4, group_size=128, min_tokens=64, split={"tune": [0, 1]})
    d = e1_run_dir(str(tmp_path), fp)
    assert e1_run_dir(str(tmp_path), dict(fp)) == d            # same config resumes
    for k, v in [("min_tokens", 32), ("K", 8), ("reps", 2), ("split", {"tune": [0, 2]})]:
        assert e1_run_dir(str(tmp_path), {**fp, k: v}) != d    # changed -> new dir
    with open(os.path.join(d, "fingerprint.json"), "w") as f:  # stale/tampered dir
        json.dump({**fp, "min_tokens": 1}, f)
    with pytest.raises(RuntimeError):
        e1_run_dir(str(tmp_path), fp)
    # E2 refuses to pool runs that differ in anything but char_seed
    for i, extra in enumerate([{}, {"min_tokens": 32}]):
        rd = e1_run_dir(str(tmp_path / "pool"), {**fp, "char_seed": i, **extra})
        open(os.path.join(rd, "labels.csv"), "w").close()
        json.dump({}, open(os.path.join(rd, "draws.json"), "w"))
    with pytest.raises(RuntimeError):
        load_e1_runs(str(tmp_path / "pool"), 4)


# 9 -----------------------------------------------------------------------
def test_experts_distinct_across_layers(cache_dir):
    model, _ = tiny_model()
    ids = list(range(POOL))
    C, p = E3.routing_profile(model, cache_dir, LAYERS, ids)
    C2, p2 = E3.routing_profile(model, cache_dir, [LAYERS[1]], ids)
    assert C.shape[1] == len(LAYERS) * E
    np.testing.assert_array_equal(C[:, E:], C2)               # layer 2 block intact
    assert not np.array_equal(C[:, :E], C[:, E:])             # not merged with layer 0
    assert np.allclose(p[:E].sum(), 1) and np.allclose(p[E:].sum(), 1)
    st = E3.expert_stats(model, cache_dir, LAYERS, ids[:10], {}, 4, 0)
    np.testing.assert_allclose(st["count"], C[:10].sum(0))   # j = li*E + e everywhere


# 10 ----------------------------------------------------------------------
def test_every_strategy_scores_same_experts(pipeline):
    total = len(LAYERS) * E
    keys = set()
    for name, res in pipeline["e3"].items():
        s = res["summary"]
        assert s["n_experts_quantized_total"] == total, name
        assert s["n_experts_test_observed"] == len(res["eval_universe"]), name
        keys.add(tuple(sorted((r["layer"], r["expert"]) for r in res["per_expert"])))
    assert len(keys) == 1
    for name in ("--strategy tlp", "--strategy oracle"):
        traj = pipeline["e3"][name]["trajectory"]
        assert traj and all("n_zero_exposure" in t for t in traj)


# 11 ----------------------------------------------------------------------
def _rtn(W, bits, gs):
    maxq = 2 ** bits - 1
    out = torch.empty_like(W)
    for c in range(0, W.shape[1], gs):
        g = W[:, c:c + gs]
        wmax, wmin = g.max(1).values.clamp(min=0), g.min(1).values.clamp(max=0)
        scale = ((wmax - wmin).clamp(min=1e-10) / maxq)[:, None]
        zero = torch.round(-wmin[:, None] / scale)
        out[:, c:c + gs] = scale * (torch.clamp(torch.round(g / scale + zero), 0, maxq) - zero)
    return out


def test_zero_token_fallback_is_fixed_rtn(cache_dir):
    meta = load_meta(cache_dir)
    test = split_ids(meta)["test"]
    model, _ = tiny_model()
    fp_before = {(L, e): fp_weights(model, L, e) for L in LAYERS for e in range(E)}
    res, counts, orig, _, _ = E3.quantize_and_score(
        model, cache_dir, LAYERS, [], {}, test, 4, 4, 32, write_back=True)
    assert counts["n_zero_token_quantization_fallback"] == len(LAYERS) * E
    assert counts["n_experts_quantized_total"] == len(LAYERS) * E
    assert counts["n_experts_test_observed"] == len(res)
    assert all(r["fallback"] for r in res)
    for (L, e), W in fp_before.items():
        now = fp_weights(model, L, e)
        for k in W:
            ref = _rtn(W[k], 4, 32).to(torch.bfloat16).float()
            assert not torch.equal(now[k], W[k]), "zero-token expert left in bf16"
            torch.testing.assert_close(now[k], ref, rtol=0, atol=1e-6)
    # identical for any strategy: the rule only reads the FP weights
    model2, _ = tiny_model()
    acc = Profiler(model2, LAYERS[0], K=1).acc        # empty accumulator
    Wq, fb = quantize_from_acc_or_fallback(model2, acc, LAYERS[0], 3, 4, 32)
    assert fb
    for k, W in fp_weights(model2, LAYERS[0], 3).items():
        torch.testing.assert_close(Wq[k], gptq_quantize(W, torch.eye(W.shape[1]), 4, 32,
                                                        actorder=False))
        torch.testing.assert_close(Wq[k], _rtn(W, 4, 32), rtol=0, atol=1e-6)


# ===================== follow-up: research-validity edge cases =====================

def test_low_count_observations_retained(pipeline):
    # a 5-token expert with enough paired tuning sequences is labelled
    e0 = {s: 0.10 for s in range(4)}
    e1 = {s: 0.08 + 0.001 * s for s in range(4)}
    lab, why = E1.label_expert(e0, e1, tokens=5, min_calib_tokens=1, min_paired_seqs=3)
    assert why is None and lab["n_seqs"] == 4 and np.isclose(lab["rel_delta"], 0.185)
    # in the synthetic pipeline every count is << 256 and labels are still produced
    import csv
    for d in [r for r in os.listdir(pipeline["out"]) if r.startswith("e1_b4_s")]:
        rows = list(csv.DictReader(open(os.path.join(pipeline["out"], d, "labels.csv"))))
        assert rows and max(float(r["tokens_n0"]) for r in rows) < 256
        fp = json.load(open(os.path.join(pipeline["out"], d, "fingerprint.json")))
        assert fp["min_calib_tokens"] == 1 and fp["min_paired_seqs"] == 3
    sup = pipeline["art"]["support_by_starting_budget"]
    assert set(sup) == {"n0=4", "n0=8"}
    for v in sup.values():
        assert v["n_fitted"] > 0 and v["count_min"] <= v["count_median"] <= v["count_max"]
        assert {"count_p5", "count_p25", "count_p75", "count_p95",
                "n_excluded", "excluded_by_reason"} <= set(v)


def test_insufficient_paired_eval_excluded_with_reason():
    e0 = {0: 0.1, 1: 0.1}
    e1 = {0: 0.09, 1: 0.09}
    assert E1.label_expert(e0, e1, 50, 1, 3) == (None, "insufficient_paired_eval_seqs")
    assert E1.label_expert({}, {}, 0, 1, 3) == (None, "zero_calib_tokens")
    assert E1.label_expert(e0, e1, 5, 10, 1) == (None, "below_min_calib_tokens")
    bad = {0: float("nan"), 1: 0.1, 2: 0.1}
    assert E1.label_expert(bad, bad, 50, 1, 3) == (None, "nonfinite_error")


def test_no_benefit_extrapolation_outside_support():
    entry = lambda s: dict(signal=s, orientation=1.0, coef=dict(beta0=1.0, beta_count=-0.1,
                           beta_signal=0.5), norm=dict(mean=0.0, std=1.0),
                           n_obs=10, count_range=[10.0, 100.0])
    model = {"predictors": {"count": entry("count"), "relse": entry("relse")}}
    n = np.array([0., 5., 10., 50., 100., 500.])
    stats = {"count": n, "relse": np.full(6, 0.2)}
    dh, info = predicted_delta(model, "relse", stats)
    assert info["below"].tolist() == [False, True, False, False, False, False]
    assert info["above"].tolist() == [False, False, False, False, False, True]
    assert np.isnan(dh[[0, 1, 5]]).all() and np.isfinite(dh[2:5]).all()
    v, cap = allocation_terms(n, dh, np.ones(6))
    assert v[[0, 1, 5]].tolist() == [0, 0, 0] and cap[[0, 1, 5]].tolist() == [0, 0, 0]


def test_all_zero_utility_uses_seeded_random_fallback():
    rng0 = np.random.default_rng(0)
    Cc = rng0.integers(0, 5, size=(30, 6)).astype(float)
    cand = list(range(100, 130))
    zeros = np.zeros(6)
    no_targets = np.zeros(6, bool)
    take, w, log = E3.select_round(Cc, cand, zeros, np.full(6, 10.), no_targets, 5,
                                   random.Random(7), False)
    take2, _, _ = E3.select_round(Cc, cand, zeros, np.full(6, 10.), no_targets, 5,
                                  random.Random(7), False)
    assert take == take2 == random.Random(7).sample(cand, 5)
    assert take != cand[:5] and w == {}
    assert log["used_random_utility_fallback"] and log["n_random_fallback_sequences"] == 5
    assert log["n_positive_utility"] == 0
    # with a positive utility the LP is used instead
    v = np.zeros(6); v[2] = 0.01
    _, _, log = E3.select_round(Cc + 1, cand, v, np.full(6, 10.), no_targets, 5,
                                random.Random(7), False)
    assert not log["used_random_utility_fallback"] and log["n_positive_utility"] == 1


def test_exploration_never_exceeds_budget(pipeline):
    Cc = np.eye(10)                                 # each sequence reaches one target
    targets = np.ones(10, bool)
    picks, reached = exploration_cover(Cc, targets, 3)
    assert len(picks) == 3 and reached.sum() == 3
    take, _, log = E3.select_round(Cc, list(range(10)), np.zeros(10), np.zeros(10),
                                   targets, 3, random.Random(0), False)
    assert len(take) == 3 and log["n_exploration_sequences"] == 3
    assert log["n_exploration_targets_unreached"] == 7
    assert log["exploration_fraction_of_round"] == 1.0
    for name in ("--strategy tlp", "--strategy tlp --signal count", "--strategy oracle"):
        res = pipeline["e3"][name]
        assert len(res["chosen"]) == res["summary"]["B"] == 12
        for t in res["trajectory"]:
            assert t["n_exploration_sequences"] <= t["n_taken"] <= t["round_budget"]
            for k in ("n_zero_exposure", "n_out_of_support", "n_exploration_targets_reached",
                      "n_exploration_targets_unreached", "exploration_fraction_of_round",
                      "n_below_fit_count_range", "n_above_fit_count_range",
                      "used_random_utility_fallback", "n_random_fallback_sequences",
                      "n_lp_positive_value_selected", "n_zero_marginal_random_fill"):
                assert k in t, (name, k)


def test_eval_universe_fixed_across_strategies(pipeline, cache_dir, tmp_path):
    unis = {json.dumps(r["eval_universe"]) for r in pipeline["e3"].values()}
    files = {r["summary"]["eval_universe_file"] for r in pipeline["e3"].values()}
    assert len(unis) == 1 and len(files) == 1
    uni = json.loads(unis.pop())
    # built from the test split alone: same as a calibration-free replay
    model, _ = tiny_model()
    test = split_ids(load_meta(cache_dir))["test"]
    _, counts, _, _, _ = E3.quantize_and_score(model, cache_dir, LAYERS, [], {}, test, 4, 4, 128)
    assert [list(u) for u in counts["eval_universe"]] == uni
    # a different universe for the same split/layers is rejected
    d = str(tmp_path)
    E3.check_eval_universe(d, {"s": 1}, LAYERS, [(0, 1)])
    with pytest.raises(AssertionError):
        E3.check_eval_universe(d, {"s": 1}, LAYERS, [(0, 2)])


def test_all_experts_quantized_even_outside_eval_universe(cache_dir):
    model, _ = tiny_model()
    before = {(L, e): fp_weights(model, L, e) for L in LAYERS for e in range(E)}
    chosen = list(range(0, 10))
    res, counts, orig, _, _ = E3.quantize_and_score(
        model, cache_dir, LAYERS, chosen, {}, [], 4, 4, 32, write_back=True)
    assert counts["n_experts_test_observed"] == 0 and res == []
    assert counts["n_experts_quantized_total"] == len(LAYERS) * E
    assert len(orig) == len(LAYERS) * E
    for (L, e), W in before.items():
        now = fp_weights(model, L, e)
        assert all(not torch.equal(now[k], W[k]) for k in W), (L, e)


def test_e2_loads_multiple_char_seeds(pipeline, tmp_path):
    import csv
    art = pipeline["art"]
    assert art["characterization_seeds"] == list(CHAR_SEEDS)
    assert len(art["e1_runs"]) == 3
    assert art["selection_criterion"] == "leave_one_char_seed_out_mse"
    rows = list(csv.DictReader(open(os.path.join(pipeline["out"], "e2_signals_b4.csv"))))
    assert {int(r["characterization_seed"]) for r in rows} == set(CHAR_SEEDS)
    with pytest.raises(RuntimeError, match="need >= 4"):
        E2.main(["--cache-dir", pipeline["cache_dir"], "--outdir", pipeline["out"],
                 "--min-char-seeds", "4"])


# ===================== pre-experiment: zero-marginal fill, support diagnostic =====================

def test_zero_marginal_slots_are_seeded_random():
    # expert 0 has positive utility but capacity 5: ONE of seq 0/1 saturates it,
    # every other slot of the 4-sequence round has zero marginal value
    Cc = np.array([[5, 1], [5, 0]] + [[0, k] for k in range(1, 9)], float)
    cand = list(range(200, 210))
    value, cap = np.array([0.1, 0.0]), np.array([5.0, 0.0])
    take, w, log = E3.select_round(Cc, cand, value, cap, np.zeros(2, bool), 4,
                                   random.Random(7), False)
    kept = take[0]
    assert kept in (200, 201) and len(take) == 4 and w == {}
    assert take[1:] == random.Random(7).sample([c for c in cand if c != kept], 3)
    assert log["n_lp_positive_value_selected"] == 1
    assert log["n_zero_marginal_random_fill"] == 3
    assert log["n_random_fallback_sequences"] == 0 and not log["used_random_utility_fallback"]
    # the marginal check itself: a second saturating sequence adds nothing
    z = np.array([1.0, 1.0, 1.0])
    C3 = np.array([[5, 0], [5, 0], [0, 9]], float)
    assert E3.positive_marginal_selection(z, C3, value, cap, 3) == [0]


def test_support_coverage_known_counts():
    pool = list(range(20))
    layers, E = [3, 7], 2
    C = np.ones((20, 4))
    C[:, 0] = 0                                      # layer 3 / expert 0 never routed
    # K=1, rounds=2, B=8: warm min(8, max(1, 2)) = 2, per round 3 -> checkpoints 2, 5
    rows, rep = SD.support_coverage(C, pool, layers, E, lo=4, hi=6, budgets=[8, 40],
                                    seeds=[0, 1], K=1, rounds=2)
    r = rep["B=8"]
    assert r["schedule"]["decision_checkpoints"] == [2, 5]
    cp0, cp1 = (c["overall"] for c in r["checkpoints"])
    assert cp0["fraction_zero"] == 0.25 and cp0["fraction_below_support_nonzero"] == 0.75
    assert cp1["fraction_inside_support"] == 0.75 and cp1["fraction_zero"] == 0.25
    fin = r["FINAL_STATE_ONLY"]
    assert fin["decision_chosen_size"] == 8 and fin["overall"]["fraction_above_support"] == 0.75
    assert r["checkpoints"][1]["by_layer"]["3"]["fraction_zero"] == 0.5
    assert r["checkpoints"][1]["by_layer"]["7"]["fraction_inside_support"] == 1.0
    assert set(r["checkpoints"][0]["by_seed"]) == {"0", "1"}
    assert "skipped" in rep["B=40"]                                # B > pool
    assert len(rows) == 3 * 2 * 2                                  # (2 checkpoints + final) x seeds x layers
    assert {x["stage"] for x in rows} == {"decision", "FINAL_STATE_ONLY"}


def test_decision_checkpoints_match_e3_schedule(pipeline):
    # the proposal's example schedule (K=4, rounds=2)
    expected = {64: [16, 40], 128: [32, 80], 256: [64, 160], 512: [128, 320]}
    C = np.ones((600, 2))
    _, rep = SD.support_coverage(C, list(range(600)), [0], 2, 1, 10, list(expected),
                                 [0], K=4, rounds=2)
    for B, cps in expected.items():
        warm, sched = E3.round_schedule(B, 4, 2)
        assert [size for _, size, _ in sched] == cps
        assert rep[f"B={B}"]["schedule"]["decision_checkpoints"] == cps
        assert sum(b for _, _, b in sched) + warm == B
    # ... and they are exactly where the real E3 loop evaluated the predictor
    out = SD.main(["--cache-dir", pipeline["cache_dir"], "--outdir", pipeline["out"],
                   "--budgets", "12"])
    cps = out["report"]["B=12"]["schedule"]["decision_checkpoints"]
    pool = split_ids(load_meta(pipeline["cache_dir"]))["pool"]
    for name in ("--strategy tlp", "--strategy oracle"):
        res = pipeline["e3"][name]
        assert [t["n_selected_before"] for t in res["trajectory"]] == cps
        warm = out["report"]["B=12"]["schedule"]["warm_start"]
        assert res["chosen"][:warm] == random.Random(0).sample(pool, warm)


def test_support_diagnostic_cli(pipeline):
    out = SD.main(["--cache-dir", pipeline["cache_dir"], "--outdir", pipeline["out"],
                   "--budgets", "4", "12", "64"])
    art = pipeline["art"]
    assert out["support"] == art["predictors"][art["selected_signal"]]["count_range"]
    for cp in out["report"]["B=12"]["checkpoints"] + [out["report"]["B=12"]["FINAL_STATE_ONLY"]]:
        r = cp["overall"]
        assert r["n_experts"] == 3 * len(LAYERS) * E
        assert abs(r["fraction_zero"] + r["fraction_below_support_nonzero"]
                   + r["fraction_inside_support"] + r["fraction_above_support"] - 1) < 1e-9
    assert "skipped" in out["report"]["B=64"]
    base = os.path.join(pipeline["out"], "support_diagnostic_b4")
    header = open(base + ".csv").readline().strip().split(",")
    assert header[:4] == ["target_final_B", "stage", "round", "decision_chosen_size"]
    assert os.path.exists(os.path.join(pipeline["out"], "routing_profile.npz"))
