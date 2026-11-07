"""CPU tests for the minimal layer-aware allocation experiment (11/06):
exported P2c models, regime interpolation / support, the shared LP controller
for coverage_lp / global_exposure / layer_budget / layer_aware, and the
frozen-config / artifact guards."""
import copy
import json
import os
import random
import sys

import numpy as np
import pytest

import e3_allocation as E3
from conftest import E, LAYERS, POOL, TUNE, tiny_model
from moe_calib import alloc_models as AM
from moe_calib import gptq as G
from moe_calib import model_utils as MU
from moe_calib.data_utils import load_meta, split_ids

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import p2c_layer_exposure as P2C       # noqa: E402

LP_STRATS = list(AM.LP_UTILITY_STRATEGIES)
ALL_STRATS = ["random", "coverage", *LP_STRATS]


def synth_table(layers=(1, 8, 16, 23), budgets=(16, 32, 64, 128), n_exp=60, per_seq=34,
                seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in (0, 1, 2):
        for L in layers:
            for b in budgets:
                n = rng.lognormal(np.log(per_seq * b), 0.4, n_exp)
                base = (0.08 if L == layers[0] else 0.05) * (budgets[0] / b) ** (0.6 if L == layers[0] else 0.3)
                y = base * (n / n.mean()) ** -0.3 + 0.002 * rng.normal(size=n_exp)
                rows += [(s, L, e, b, n[e], y[e]) for e in range(n_exp)]
    a = np.array(rows, float)
    return dict(seed=a[:, 0], layer=a[:, 1], expert=a[:, 2], budget=a[:, 3], count=a[:, 4], y=a[:, 5])


def frozen_for(layers, split, model="tiny", percdamp=1.0):
    return dict(name="frozen_test", model=model,
                gptq=dict(bits=4, group_size=128, percdamp=percdamp, act_order=True),
                layers=list(layers), e1=dict(n0=[2, 4, 16], char_seeds=[0, 1, 2], reps=2, K=4),
                split=split)


def artifact_for(d, cfg):
    return AM.fit_export(d, dict(frozen=dict(name=cfg["name"], model=cfg["model"], gptq=cfg["gptq"],
                                             split=cfg["split"], layers=list(cfg["layers"]))))


# ---------------------------------------------------------------- regression

@pytest.mark.parametrize("name", ["M1", "Dlb", "M4"])
def test_export_matches_p2c_fit(name):
    d = synth_table()
    art = artifact_for(d, frozen_for([1, 8, 16, 23], {}))
    ref = P2C.fit(d, name)(d)
    got = np.empty_like(ref)
    for L in np.unique(d["layer"]):
        for b in np.unique(d["budget"]):
            m = (d["layer"] == L) & (d["budget"] == b)
            got[m] = AM.predict_at_budget(art["models"][name], L, d["count"][m], b)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_real_artifact_matches_p2c_on_characterized_rows():
    art_p, csv_p = os.path.join(ROOT, "runs/allocation_models_b4_d1.json"), \
        os.path.join(ROOT, "runs/e2_signals_b4_d1.csv")
    if not (os.path.exists(art_p) and os.path.exists(csv_p)):
        pytest.skip("real P2 table / artifact not present")
    art = json.load(open(art_p))
    d, _ = AM.load_table(csv_p)
    assert AM.file_sha256(csv_p) == art["training_table"]["sha256"]
    for st, name in AM.STRATEGY_MODEL.items():
        ref = P2C.fit(d, name)(d)
        for L in np.unique(d["layer"]):
            for b in np.unique(d["budget"]):
                m = (d["layer"] == L) & (d["budget"] == b)
                pred, status = AM.predict_marginal_benefit(art, name, L, d["count"][m], b)
                assert (status == "ok").all()          # every training row is in support
                np.testing.assert_allclose(pred, ref[m], rtol=1e-9)


def test_training_only_standardization_and_determinism():
    d = synth_table()
    tr = {k: v[d["seed"] != 2] for k, v in d.items()}
    e = AM.fit_model(tr, "M4")
    assert e["lc_mean"] == pytest.approx(np.log(tr["count"]).mean())
    assert e["lc_mean"] != pytest.approx(np.log(d["count"]).mean())
    a1, a2 = artifact_for(d, frozen_for([1], {})), artifact_for(d, frozen_for([1], {}))
    assert a1["fingerprint"] == a2["fingerprint"]
    x = AM.predict_at_budget(a1["models"]["M4"], 8, [3000.0, 4000.0], 128)
    assert np.array_equal(x, AM.predict_at_budget(a2["models"]["M4"], 8, [3000.0, 4000.0], 128))


def test_layer_indexing_is_layer_major():
    d = synth_table()
    art = artifact_for(d, frozen_for([1, 8, 16, 23], {}))
    layers, Ee = [1, 8, 16, 23], 5
    n = np.tile(np.linspace(1000, 1400, Ee), 4)
    dh, st = AM.compute_expert_utility("layer_aware", n, layers, Ee, 32, art)
    for li, L in enumerate(layers):
        ref, _ = AM.predict_marginal_benefit(art, "M4", L, n[li * Ee:(li + 1) * Ee], 32)
        np.testing.assert_array_equal(dh[li * Ee:(li + 1) * Ee], ref)
    # the first layer of the synthetic table has a different benefit level
    assert np.nanmean(dh[:Ee]) > np.nanmean(dh[Ee:2 * Ee])


def test_intermediate_budget_interpolation():
    d = synth_table()
    art = artifact_for(d, frozen_for([1, 8, 16, 23], {}))
    assert AM.bracket(32, art["budgets"]) == (32.0, 32.0, 0.0)
    lo, hi, w = AM.bracket(80, art["budgets"])
    assert (lo, hi) == (64.0, 128.0) and w == pytest.approx(np.log(80 / 64) / np.log(2))
    n = np.array([2900.0, 3500.0])
    f64 = AM.predict_at_budget(art["models"]["Dlb"], 8, n, 64)
    f128 = AM.predict_at_budget(art["models"]["Dlb"], 8, n, 128)
    p80, st = AM.predict_marginal_benefit(art, "Dlb", 8, n, 80)
    assert (st == "ok").all()
    np.testing.assert_allclose(p80, (1 - w) * f64 + w * f128)
    p_geo, _ = AM.predict_marginal_benefit(art, "Dlb", 8, n, np.sqrt(64 * 128))
    np.testing.assert_allclose(p_geo, 0.5 * (f64 + f128))
    # the count is held fixed: interpolation is not a function of the count shift
    p64, _ = AM.predict_marginal_benefit(art, "Dlb", 8, n, 64)
    np.testing.assert_allclose(p64, f64)


def test_no_silent_extrapolation():
    d = synth_table()
    art = artifact_for(d, frozen_for([1, 8, 16, 23], {}))
    n = np.array([1000.0, 1100.0])
    for m in (8, 15, 129, 256):
        p, st = AM.predict_marginal_benefit(art, "M4", 1, n, m)
        assert np.isnan(p).all() and (st == "budget_out_of_range").all()
    lo, hi = art["count_support"]["1"]["32"]
    p, st = AM.predict_marginal_benefit(art, "M4", 1, np.array([0, -3, np.nan, lo - 1, lo, hi, hi + 1]), 32)
    assert st.tolist() == ["zero", "invalid", "invalid", "below", "ok", "ok", "above"]
    assert np.isnan(p[[0, 1, 2, 3, 6]]).all() and np.isfinite(p[[4, 5]]).all()
    # interpolated regime: support is the hull lo(L, 64) .. hi(L, 128)
    s = art["count_support"]["1"]
    _, st = AM.predict_marginal_benefit(art, "M4", 1, np.array([s["64"][0] - 1, s["64"][0], s["128"][1],
                                                                s["128"][1] + 1]), 80)
    assert st.tolist() == ["below", "ok", "ok", "above"]
    with pytest.raises(ValueError):
        AM.predict_at_budget(art["models"]["M4"], 5, n, 32)          # uncharacterized layer
    with pytest.raises(ValueError):
        AM.predict_at_budget(art["models"]["M4"], 1, n, 48)          # not a characterized budget


def test_artifact_guards():
    d = synth_table()
    cfg = frozen_for([1, 8, 16, 23], {"pool": [0, 2000]})
    art = artifact_for(d, cfg)
    AM.check_artifact(art, cfg, [1, 8, 16, 23], "layer_aware")
    bad = copy.deepcopy(art)
    bad["models"]["M4"]["beta"][0] += 1e-3                              # edited coefficients
    with pytest.raises(ValueError, match="fingerprint"):
        AM.check_artifact(bad, cfg, [1, 8], "layer_aware")
    other = frozen_for([1, 8, 16, 23], {"pool": [0, 2000]}, percdamp=0.01)
    with pytest.raises(ValueError, match="different configuration"):
        AM.check_artifact(art, other, [1, 8], "layer_aware")
    with pytest.raises(ValueError, match="not characterized"):
        AM.check_artifact(art, cfg, [1, 5], "layer_aware")
    with pytest.raises(ValueError):
        AM.compute_expert_utility("layer_aware", np.ones(4), [1], 4, 32, None)


# ------------------------------------------------------- controlled strategy

def test_coverage_lp_uses_inverse_count():
    n = np.array([0., 5., 10., 40.])
    dh, st = AM.compute_expert_utility("coverage_lp", n, [0], 4, 4, None)
    v, cap = E3.allocation_terms(n, dh, np.ones(4))
    np.testing.assert_allclose(v, [0, 1 / 5, 1 / 10, 1 / 40])
    np.testing.assert_allclose(cap, [0, 5, 10, 40])
    assert st.tolist() == ["zero", "ok", "ok", "ok"]


def test_layer_budget_values_differ_across_layers():
    d = synth_table()
    art = artifact_for(d, frozen_for([1, 8, 16, 23], {}))
    n = np.full(8, 1100.0)                                  # same count in two layers
    dh, _ = AM.compute_expert_utility("layer_budget", n, [1, 23], 4, 32, art)
    assert np.isfinite(dh).all() and abs(dh[0] - dh[4]) > 1e-3


def test_cross_layer_scaling_changes_selection_with_same_within_layer_order():
    """Within each layer, learned and coverage utilities order experts
    identically (both decreasing in n), but cross-layer magnitudes differ, so
    the LP picks a different sequence."""
    n = np.array([10., 20., 10., 20.])                      # layer a: e0,e1  layer b: e2,e3
    cap = n.copy()
    Cc = np.array([[10., 0, 0, 0],                          # seq A: layer-a expert only
                   [0, 0, 10., 10.]])                       # seq B: two layer-b experts
    v_cov = 1 / n
    v_learn = np.array([0.1, 0.1, 0.01, 0.01]) / n
    for v in (v_cov, v_learn):                              # same within-layer order
        assert v[0] > v[1] and v[2] > v[3]
    nt = np.zeros(4, bool)
    take_cov, _, _ = E3.select_round(Cc, ["A", "B"], v_cov, cap, nt, 1, random.Random(0), False)
    take_lrn, _, _ = E3.select_round(Cc, ["A", "B"], v_learn, cap, nt, 1, random.Random(0), False)
    assert take_cov == ["B"] and take_lrn == ["A"]


def test_lp_warm_start_equals_coverage_warm_start_at_B128():
    """coverage samples range(P) and the LP strategies sample pool_ids, both
    B // 4 = 32 draws from the same seeded rng -> identical warm starts."""
    P = 2000
    rng0 = np.random.default_rng(0)
    C = rng0.poisson(1.0, size=(P, 6)).astype(np.float32)
    pool = list(range(P))
    for seed in (0, 1, 2):
        warm, _ = E3.round_schedule(128, 4, 2)
        assert warm == 32
        cov = E3.select_coverage(C, pool, 128, random.Random(seed))
        assert cov[:32] == random.Random(seed).sample(pool, warm)


# ------------------------------------------------------------- E3 end to end

@pytest.fixture(scope="module")
def e3_runs(cache_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("alloc")
    meta = load_meta(cache_dir)
    sp = split_ids(meta)
    cfg = frozen_for(LAYERS, sp["definition"])
    cfg_p = out / "frozen.json"
    cfg_p.write_text(json.dumps(cfg))
    # tiny model: ~4 routed tokens / expert / sequence; regimes 2, 4, 16 sequences
    d = synth_table(layers=tuple(LAYERS), budgets=(2, 4, 16), n_exp=E, per_seq=4)
    art_p = out / "models.json"
    art_p.write_text(json.dumps(artifact_for(d, cfg)))
    calls = []
    orig = G.gptq_quantize

    def rec(W, H, bits=4, group_size=128, percdamp=0.01, actorder=True, blocksize=128):
        calls.append((bits, group_size, percdamp, actorder))
        return orig(W, H, bits, group_size, percdamp, actorder, blocksize)

    common = ["--cache-dir", cache_dir, "--outdir", str(out), "--B", "12",
              "--frozen-config", str(cfg_p), "--alloc-models", str(art_p), "--routing-cache"]
    res, gcalls = {}, {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MU, "load_model", lambda *a, **k: tiny_model())
        mp.setattr(G, "gptq_quantize", rec)
        for s in ALL_STRATS:
            calls.clear()
            res[s] = json.load(open(E3.main(common + ["--strategy", s, "--seed", "0"])))
            gcalls[s] = list(calls)
        res["layer_aware_rerun"] = json.load(open(E3.main(common + ["--strategy", "layer_aware",
                                                                    "--seed", "0"])))
        res["layer_aware_s1"] = json.load(open(E3.main(common + ["--strategy", "layer_aware",
                                                                 "--seed", "1"])))
    return dict(res=res, gcalls=gcalls, split=sp, common=common, cfg=cfg, cfg_p=cfg_p,
                art_p=art_p, out=out)


def test_exactly_B_unique_from_pool_only(e3_runs):
    sp = e3_runs["split"]
    pool, held = set(sp["pool"]), set(sp["tune"]) | set(sp["test"])
    for s in ALL_STRATS:
        ch = e3_runs["res"][s]["chosen"]
        assert len(ch) == 12 and len(set(ch)) == 12, s
        assert set(ch) <= pool and not set(ch) & held, s
        assert e3_runs["res"][s]["summary"]["n_unique_selected"] == 12


def test_lp_strategies_share_warm_start_and_schedule(e3_runs):
    r = e3_runs["res"]
    warm = r["coverage_lp"]["warm_start"]
    assert len(warm) == 4
    for s in LP_STRATS:
        assert r[s]["warm_start"] == warm and r[s]["chosen"][:4] == warm, s
        assert [t["regime_m"] for t in r[s]["trajectory"]] == [4, 8]
        assert [t["round_budget"] for t in r[s]["trajectory"]] == [4, 4]
    # regime 8 lies between characterized budgets 4 and 16 -> interpolated
    br = r["layer_aware"]["trajectory"][1]["regime_bracket"]
    assert (br["b_lo"], br["b_hi"]) == (4.0, 16.0) and 0 < br["w_hi"] < 1


def test_same_seed_same_selection(e3_runs):
    r = e3_runs["res"]
    assert r["layer_aware"]["chosen"] == r["layer_aware_rerun"]["chosen"]
    assert r["layer_aware"]["summary"]["recon_mean"] == r["layer_aware_rerun"]["summary"]["recon_mean"]
    assert r["layer_aware_s1"]["warm_start"] != r["layer_aware"]["warm_start"]


def test_frozen_gptq_enforced_and_solver_unchanged(e3_runs):
    for s, calls in e3_runs["gcalls"].items():
        assert calls, s
        assert {c[:3] for c in calls} == {(4, 128, 1.0)}, s   # frozen bits / group / damping
        assert all(c[3] is True for c in calls), s            # act-order on
    for s in ALL_STRATS:
        sm = e3_runs["res"][s]["summary"]
        assert sm["percdamp"] == 1.0 and sm["bits"] == 4 and sm["eval_split"] == "test"
        assert sm["routing_counts_match_hessian_tokens"], s   # routing cache order == Hessian experts


def test_trajectory_logs_support_and_fallbacks(e3_runs):
    for s in LP_STRATS:
        r = e3_runs["res"][s]
        for t in r["trajectory"]:
            fr = t["status_fractions"]["all"]
            assert sum(fr.values()) == pytest.approx(1.0)
            for k in ("n_lp_positive_value_selected", "n_zero_marginal_random_fill",
                      "n_random_fallback_sequences", "n_exploration_sequences", "per_layer",
                      "round_objective"):
                assert k in t
            assert t["n_exploration_sequences"] + t["n_lp_positive_value_selected"] \
                + t["n_zero_marginal_random_fill"] + t["n_random_fallback_sequences"] == t["n_taken"]
        if s != "coverage_lp":
            assert r["summary"]["alloc_models"]["model"] == AM.STRATEGY_MODEL[s]


def test_cached_routing_profile_expert_order(e3_runs, cache_dir):
    from moe_calib.profiler import Profiler
    from moe_calib.data_utils import load_layer_cache
    z = np.load(e3_runs["out"] / E3.ROUTING_CACHE)
    model, _ = tiny_model()
    ids = list(range(6))
    for li, L in enumerate(LAYERS):
        prof = Profiler(model, L, collect_hessian=False).run(load_layer_cache(cache_dir, L), ids,
                                                             routing_only=True)
        for s in ids:
            for (_, e), c in prof.seq_counts[s].items():
                assert z["C"][s, li * E + e] == c


def test_missing_or_mismatched_artifact_fails(e3_runs, tmp_path):
    base = [a for a in e3_runs["common"]]
    i = base.index("--alloc-models")
    no_art = base[:i] + base[i + 2:]
    with pytest.raises(FileNotFoundError):
        E3.main(no_art + ["--strategy", "layer_aware"])
    i = no_art.index("--frozen-config")
    with pytest.raises(ValueError, match="frozen-config"):
        E3.main(no_art[:i] + no_art[i + 2:] + ["--strategy", "coverage_lp"])
    with pytest.raises(ValueError, match="percdamp"):
        E3.main(base + ["--strategy", "coverage_lp", "--percdamp", "0.5"])
    with pytest.raises(ValueError, match="layers"):
        E3.main(base + ["--strategy", "coverage_lp", "--layers", str(LAYERS[0])])
    other = dict(e3_runs["cfg"], gptq=dict(e3_runs["cfg"]["gptq"], percdamp=0.01))
    p = tmp_path / "other.json"
    p.write_text(json.dumps(other))
    j = base.index("--frozen-config")
    swapped = base[:j] + ["--frozen-config", str(p)] + base[j + 2:]
    with pytest.raises(ValueError, match="different configuration"):
        E3.main(swapped + ["--strategy", "layer_aware"])
