"""CPU tests for the frozen-routing robustness study (tiny Qwen2-MoE from conftest + synthetic metric inputs)."""
import inspect
import json
import os
import sys

import numpy as np
import pytest
import torch

from conftest import E, LAYERS, tiny_model
from moe_calib import frozen_routing as FRT
from moe_calib import frozen_routing_robustness as RB
from moe_calib import model_utils as MU
from moe_calib import partial_model_quant as PQ
from moe_calib.data_utils import load_layer_cache, load_meta, split_ids
from moe_calib.profiler import Profiler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
GS = 16
L0 = LAYERS[0]


def _table(sids, k, seed=0, T=16):
    g = torch.Generator().manual_seed(seed)
    return {s: torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)]) for s in sids}


def _forced(model, cache, rows, table, r2s, K=2):
    fp = FRT.ForcedProfiler(model, L0, table, r2s, expect_natural=False, K=K).run(cache, rows, weights=[1.0] * len(rows))
    return {(e, p): (fp.acc.H_sum((L0, e, p), list(range(K))), fp.acc.tokens((L0, e, p))) for e in range(E)
            for p in ("gu", "down")}, fp


def _alt_cache(cache, seed=5, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    return (cache.float() + scale * torch.randn(cache.shape, generator=g)).to(torch.bfloat16)


@pytest.fixture
def setup(cache_dir):
    model, _ = tiny_model()
    return model, load_layer_cache(cache_dir, L0), list(range(12))


# ---------------------------------------------------------------- 1, 2: frozen inputs
def test_exact_grouping_seeds():
    from moe_calib import multigroup_robustness as MG
    assert RB.GROUPING_SEEDS == MG.GROUPING_SEEDS == (20261111, 20261112, 20261113, 20261114, 20261115)
    assert RB.GROUPINGS == tuple(MG.gname(s) for s in MG.GROUPING_SEEDS)
    assert RB.REUSED_AO == ("G20261111",) and RB.NEW_AO == ("G20261112", "G20261113", "G20261114", "G20261115")
    d = os.path.join(ROOT, "runs_multigroup_robustness", "grouping_manifests")
    if os.path.isdir(d):
        for g, s in zip(RB.GROUPINGS, RB.GROUPING_SEEDS):
            assert json.load(open(os.path.join(d, f"{g}.json")))["seed"] == s


def test_exact_selection_hashes():
    assert RB.SELECTION_HASHES == {"FR": "ec355fb8d866e4ff", "FC": "0353229b6d3090db"}
    path = os.path.join(ROOT, "runs_frozen_routing", "protocol.json")
    if os.path.exists(path):
        pr = json.load(open(path))
        assert {c: v["ids_hash"] for c, v in pr["selections"].items()} == RB.SELECTION_HASHES
        assert pr["n_union"] == 217 == len(set(pr["union_calibration_ids"]))


# ---------------------------------------------------------------- 3-7: routing reuse and Hessians
def test_original_sequence_id_alignment(setup):
    """A grouping cache stores the sequences in another row order; with row_to_sid the forced Hessians are
    identical to the identity-ordered cache (alignment by sequence id, never by row index)."""
    model, cache, rows = setup
    table = _table(rows, MU.top_k(model), seed=3)
    perm = rows[::-1]                                   # row i of the permuted cache holds sequence perm[i]
    pcache = cache.clone(); pcache[rows] = cache[perm]
    a, _ = _forced(model, cache, rows, table, {r: r for r in rows})
    prow = [perm.index(s) for s in rows]               # chosen sequence ids remapped to rows (MG.remap)
    b, _ = _forced(model, pcache, prow, table, {i: perm[i] for i in rows})
    wrong, _ = _forced(model, pcache, prow, table, {r: r for r in rows})        # row-index alignment: wrong
    assert all(torch.equal(a[k][0], b[k][0]) for k in a)
    assert any(not torch.equal(a[k][0], wrong[k][0]) for k in a)


def test_routing_table_reuse_paths():
    import frozen_routing_robustness as XR
    assert XR.ao_result_path("FR", "G20261111").endswith(os.path.join("runs_frozen_routing", "results", "FR_AO.json"))
    assert XR.ao_hessian_path("FC", "G20261111").endswith(os.path.join("hessians", "FC_AO.pt"))
    for g in RB.NEW_AO:
        assert XR.ao_result_path("FR", g).endswith(os.path.join(XR.OUT, "results", f"FR_AO_{g}.json"))
        assert XR.ao_weights_dir("FR", g).startswith(XR.RCACHE) and XR.ao_weights_dir("FR", g) != XR.ao_weights_dir("FR", "G20261111")
    src = inspect.getsource(XR.stage_capture) + inspect.getsource(XR.stage_regression)
    assert 'FRD.load_tables("O")' in src and "capture_routing" not in src      # R_O reused, never recomputed


def test_forced_assignments_match_R_O(setup):
    model, cache, rows = setup
    table = _table(rows, MU.top_k(model), seed=4)
    seen = {}
    fp = FRT.ForcedProfiler(model, L0, table, {r: r for r in rows}, expect_natural=False, K=1)
    orig = fp._provider
    fp._provider = lambda nat: seen.__setitem__(fp.cur_seq, orig(nat).clone()) or seen[fp.cur_seq]
    fp.run(cache, rows)
    assert all(torch.equal(seen[r].cpu(), table[r]) for r in rows)
    assert fp.stats()["n_set_diff"] > 0                  # the table differs from the natural routing


def test_ao_token_counts_identical_across_groupings(setup):
    model, cache, rows = setup
    k = MU.top_k(model)
    table = _table(rows, k, seed=6)
    counts, nat = [], []
    for s in (5, 6, 7):                                  # three "groupings" = three different activation caches
        alt = _alt_cache(cache, seed=s)
        h, _ = _forced(model, alt, rows, table, {r: r for r in rows})
        counts.append({kk: v[1] for kk, v in h.items()})
        pf = Profiler(model, L0, K=2).run(alt, rows, weights=[1.0] * len(rows))
        nat.append({(e, p): pf.acc.tokens((L0, e, p)) for e in range(E) for p in ("gu", "down")})
    assert RB.ao_counts_invariant(dict(enumerate(counts)))
    assert all(counts[0][(e, "gu")] == int(FRT.expert_token_counts(table, rows, E)[e]) for e in range(E))
    assert not RB.ao_counts_invariant(dict(enumerate(nat)))       # natural routing varies with the activations


def test_gate_up_and_down_hessians_use_X_g_with_R_O(setup):
    model, cache, rows = setup
    alt = _alt_cache(cache)
    table = _table(rows, MU.top_k(model), seed=7)
    got, _ = _forced(model, alt, rows, table, {r: r for r in rows}, K=1)
    blk = MU.moe_block(model, L0)
    X = {}
    cur = {}
    h = blk.register_forward_pre_hook(lambda m, a: X.__setitem__(cur["r"], a[0].reshape(-1, a[0].shape[-1]).clone()))
    for r in rows:
        cur["r"] = r
        MU.run_layer(model, L0, alt[r:r + 1])
    h.remove()
    for e in range(E):
        ex = blk.experts[e]
        Hgu = Hd = 0
        for r in rows:
            top_x = torch.where((table[r] == e).any(-1))[0]
            xs = X[r][top_x]
            z = ex.act_fn(ex.gate_proj(xs)) * ex.up_proj(xs)
            Hgu = Hgu + xs.float().T @ xs.float()
            Hd = Hd + z.float().T @ z.float()
        assert torch.allclose(got[(e, "gu")][0], Hgu, rtol=1e-5, atol=1e-4)
        assert torch.allclose(got[(e, "down")][0], Hd, rtol=1e-5, atol=1e-4)


# ---------------------------------------------------------------- 8-13: E3 integration
def _e3(model, cache_dir, chosen, table, r2s=None):
    import e3_allocation as E3
    test = split_ids(load_meta(cache_dir))["test"]
    with FRT.forced_e3(E3, {L0: table}, r2s or {r: r for r in chosen}, False) as st:
        q = PQ.quantize_install(model, cache_dir, [L0], chosen, test, 2, 4, GS, 1.0)
    assert E3.Profiler is Profiler
    return q, st


def _full_cover_table(chosen, k):
    t = _table(chosen, k, seed=8)
    t[chosen[0]][:E // k] = torch.arange(E).reshape(-1, k)   # every expert receives tokens
    return t


def test_no_forced_routing_at_inference(cache_dir):
    model, _ = tiny_model()
    chosen = list(range(10))
    x = load_layer_cache(cache_dir, L0)[:1].clone()
    _e3(model, cache_dir, chosen, _full_cover_table(chosen, MU.top_k(model)))
    FRT.assert_no_override(model)
    blk = MU.moe_block(model, L0)
    assert "forward" not in blk.__dict__
    out = MU.run_layer(model, L0, x)
    assert torch.isfinite(out.float()).all()


def test_all_experts_and_projections_quantized_no_rtn_fallback(cache_dir):
    model, _ = tiny_model()
    chosen = list(range(10))
    tg0 = PQ.target_tensor_digests(model, [L0])
    q, _ = _e3(model, cache_dir, chosen, _full_cover_table(chosen, MU.top_k(model)))
    after = PQ.target_tensor_digests(model, [L0])
    assert q["counts"]["n_experts_quantized_total"] == E and len(after) == 3 * E
    assert q["counts"]["n_zero_token_quantization_fallback"] == 0
    assert all(after[k] != tg0[k] for k in after) and not PQ.nonfinite_targets(model, [L0])
    assert 4 * 60 == 240 and 240 * 3 == 720         # the real study's counts (checked per config in run_config)


def test_frozen_gptq_configuration(cache_dir):
    import e3_allocation as E3
    model, _ = tiny_model()
    chosen = list(range(10))
    calls, orig = [], E3.quantize_from_acc_or_fallback

    def spy(m, acc, L, e, *a, **k):
        calls.append(a)
        return orig(m, acc, L, e, *a, **k)
    E3.quantize_from_acc_or_fallback = spy
    try:
        _e3(model, cache_dir, chosen, _full_cover_table(chosen, MU.top_k(model)))
    finally:
        E3.quantize_from_acc_or_fallback = orig
    assert calls and all(a == (4, GS, 1.0) for a in calls)
    cfg = json.load(open(os.path.join(ROOT, "configs", "frozen_b4.json")))
    assert cfg["gptq"] == dict(bits=4, group_size=128, percdamp=1.0, act_order=True)


def test_reconstruction_evaluation_consistent_across_groupings(cache_dir, tmp_path):
    """Two different activation caches with the same forced table: the held-out reconstruction universe (natural
    routing on the fixed eval cache) is identical; only the Hessians / weights differ."""
    chosen = list(range(10))
    alt_dir = tmp_path / "alt"
    alt_dir.mkdir()
    for f in os.listdir(cache_dir):
        src = os.path.join(cache_dir, f)
        if f.startswith("L"):
            torch.save(_alt_cache(torch.load(src), seed=3), alt_dir / f)
        else:
            (alt_dir / f).write_bytes(open(src, "rb").read())
    import e3_allocation as E3
    test = split_ids(load_meta(cache_dir))["test"]
    res = []
    for d in (cache_dir, str(alt_dir)):
        model, _ = tiny_model()
        table = _full_cover_table(chosen, MU.top_k(model))
        with FRT.forced_e3(E3, {L0: table}, {r: r for r in chosen}, False):
            q = PQ.quantize_install(model, d, [L0], chosen, test, 2, 4, GS, 1.0, eval_cache_dir=cache_dir)
        res.append((q, PQ.installed_digest(model, [L0])))
    assert res[0][0]["counts"]["eval_universe"] == res[1][0]["counts"]["eval_universe"]
    assert [r["tokens"] for r in res[0][0]["results"]] == [r["tokens"] for r in res[1][0]["results"]]
    assert res[0][1] != res[1][1]


def test_artifact_matching_names():
    import frozen_routing_robustness as XR
    new = [XR.ao_name(c, g) for g in RB.NEW_AO for c in RB.STRATS]
    assert len(new) == 8 == len(set(new))
    assert XR.aa_result_path("FC", "G20261113").endswith(os.path.join("runs_multigroup_robustness", "results", "FC_G20261113.json"))
    assert XR.oo_result_path("FR").endswith("CTRL_FR_ORIGINAL.json")
    assert len({XR.ao_hessian_path(c, g) for c in RB.STRATS for g in RB.GROUPINGS}) == 10


# ---------------------------------------------------------------- 14-20: metrics and decision
AO = {"FR": [9.266, 9.267, 9.265, 9.268, 9.266], "FC": [9.278, 9.279, 9.277, 9.278, 9.280]}
AA = {"FR": [9.278, 9.268, 9.262, 9.272, 9.270], "FC": [9.274, 9.284, 9.270, 9.276, 9.288]}


def test_per_strategy_grouping_sd():
    s = RB.spread(AO["FR"], AA["FR"])
    assert s["SD_AO"] == pytest.approx(np.std(AO["FR"], ddof=1)) and s["SD_AA"] == pytest.approx(np.std(AA["FR"], ddof=1))
    assert s["Range_AO"] == pytest.approx(0.003) and s["Range_AA"] == pytest.approx(0.016)


def test_paired_delta_sd_and_pairing():
    rk = RB.ranking({"AO": AO["FR"], "AA": AA["FR"]}, {"AO": AO["FC"], "AA": AA["FC"]}, 0.010057)
    d_ao = [b - a for a, b in zip(AO["FR"], AO["FC"])]
    assert rk["AO"]["Delta"] == pytest.approx(d_ao) and rk["AO"]["SD"] == pytest.approx(np.std(d_ao, ddof=1))
    # pairing is by grouping position: shuffling one strategy changes the result
    rk2 = RB.ranking({"AO": AO["FR"][::-1], "AA": AA["FR"]}, {"AO": AO["FC"], "AA": AA["FC"]}, 0.010057)
    assert rk2["AO"]["Delta"] != rk["AO"]["Delta"]
    assert rk["RankingInteraction"] == pytest.approx([a - b for a, b in zip(rk["AO"]["Delta"], rk["AA"]["Delta"])])
    e = RB.intervention_effects(AO["FR"], AA["FR"])
    assert e["per_grouping"] == pytest.approx([a - b for a, b in zip(AO["FR"], AA["FR"])])


def test_sd_and_range_ratios_and_zero_denominator():
    s = RB.spread(AO["FC"], AA["FC"])
    assert s["SD_ratio"] == pytest.approx(s["SD_AO"] / s["SD_AA"]) and s["Range_ratio"] == pytest.approx(3 / 18)
    z = RB.spread([1.0, 1.1, 1.2], [2.0, 2.0, 2.0])
    assert z["SD_ratio"] is None and z["Range_ratio"] is None and z["SD_AA"] == 0


def test_distance_to_OO():
    d = RB.distances([9.266, 9.270], [9.278, 9.268], 9.267)
    assert d["Dist_AO"] == pytest.approx([0.001, 0.003]) and d["Dist_AA"] == pytest.approx([0.011, 0.001])
    assert d["n_AO_closer"] == 1 and d["n_AO_farther"] == 1
    assert d["mean_Dist_AO"] == pytest.approx(0.002) and d["median_Dist_AA"] == pytest.approx(0.006)


def test_ranking_reversal_counts():
    fr = {"AO": [1.0, 1.0, 1.0, 1.0, 1.0], "AA": [1.0] * 5}
    fc = {"AO": [1.1, 1.2, 0.9, 1.05, 1.0], "AA": [0.9, 0.8, 1.1, 0.95, 0.99]}
    rk = RB.ranking(fr, fc, 0.01)
    assert rk["AO"]["n_positive"] == 3 and rk["AO"]["n_negative"] == 1 and rk["AO"]["n_zero"] == 1
    assert rk["AO"]["n_preserved_vs_OO"] == 3 and rk["AO"]["n_reversals_vs_OO"] == 2      # a tie is not preserved
    assert rk["AA"]["n_preserved_vs_OO"] == 1 and rk["AA"]["n_reversals_vs_OO"] == 4


def _crit(**kw):
    base = dict(sd_ratio_FR=False, sd_ratio_FC=False, sd_delta_ratio=False, closer_FR=False, closer_FC=False,
                ao_ranking_preserved=False, ao_ranking_preserved_more_than_aa=False)
    base.update(kw)
    return base


def test_decision_classification():
    full = _crit(sd_ratio_FR=True, sd_ratio_FC=True, sd_delta_ratio=True, closer_FR=True, closer_FC=True,
                 ao_ranking_preserved=True)
    assert RB.decide(full)[0] == "FROZEN-ROUTING-ROBUSTLY-STABILIZES-PPL"
    assert RB.decide(full, blocked=True)[0] == "RUN-BLOCKED"
    assert RB.decide(_crit(sd_ratio_FR=True)) == ("FROZEN-ROUTING-PARTIALLY-STABILIZES", ["a_sd_ratio_FR"])
    assert RB.decide(_crit(closer_FR=True, closer_FC=True))[1] == ["d_closer_both"]
    assert RB.decide(_crit(closer_FR=True))[0] == "FROZEN-ROUTING-NOT-STABLE"
    assert RB.decide(_crit(ao_ranking_preserved=True))[0] == "FROZEN-ROUTING-NOT-STABLE"   # not more than AA
    assert RB.decide(_crit(ao_ranking_preserved=True, ao_ranking_preserved_more_than_aa=True))[1] == ["e_ranking"]
    assert RB.decide(_crit())[0] == "FROZEN-ROUTING-NOT-STABLE"
    # criteria from metrics: an undefined ratio never satisfies <= 0.5
    sp = {c: RB.spread(AO[c], AA[c]) for c in RB.STRATS}
    sp["FC"]["SD_ratio"] = None
    rk = RB.ranking({k: (AO if k == "AO" else AA)["FR"] for k in ("AO", "AA")},
                    {k: (AO if k == "AO" else AA)["FC"] for k in ("AO", "AA")}, 0.010057)
    dist = {c: RB.distances(AO[c], AA[c], 9.267851 if c == "FR" else 9.277908) for c in RB.STRATS}
    cr = RB.criteria(sp, rk, dist)
    assert cr["sd_ratio_FC"] is False and cr["sd_ratio_FR"] is True
    assert RB.DECISION_RULE["thresholds"] == dict(ratio_max=0.5, min_groupings=4, n_groupings=5)


# ---------------------------------------------------------------- 21: backward compatibility
def test_G20261111_backward_compatibility():
    import frozen_routing as FRD
    sig = inspect.signature(FRD.run_config)
    assert sig.parameters["grouping"].default == FRD.GROUPING == "G20261111"
    assert all(sig.parameters[k].default is None for k in ("result_path", "wd", "hpath"))
    assert inspect.signature(FRD.act_source).parameters["grouping"].default == "G20261111"
    for c, ref in (("FR", 9.265860908067534), ("FC", None)):
        path = os.path.join(ROOT, "runs_frozen_routing", "results", f"{c}_AO.json")
        if os.path.exists(path):
            r = json.load(open(path))
            assert r["valid"] and r["condition"] == "AO" and (ref is None or r["ppl"] == ref)
            assert round(r["ppl"], 6) == {"FR": 9.265861, "FC": 9.277881}[c]
