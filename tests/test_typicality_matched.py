"""CPU tests for the matched-warm-start typicality experiment: shared warm
starts, E3 --warm-start-file for random / coverage_lp, pool restriction."""
import json
import os
import random
import sys

import numpy as np
import pytest

import e3_allocation as E3
from conftest import E, LAYERS, tiny_model
from moe_calib import alloc_models as AM
from moe_calib import model_utils as MU
from moe_calib.data_utils import load_meta, split_ids
from test_allocation_minimal import frozen_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import typicality_pools as TP          # noqa: E402
import typicality_warm_starts as TW    # noqa: E402

POOLS = {"full": None, "typical_half": "pool_typical_half.json", "random_half": "pool_random_half.json"}
STRATS = ("random", "coverage_lp")


@pytest.fixture(scope="module")
def setup(cache_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("matched")
    pool_dir, out = root / "pools", root / "matched"
    sp = split_ids(load_meta(cache_dir))
    cfg_p = root / "frozen.json"
    cfg_p.write_text(json.dumps(frozen_for(LAYERS, sp["definition"])))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MU, "load_model", lambda *a, **k: tiny_model())
        # one plain run builds the full-pool routing cache
        E3.main(["--cache-dir", cache_dir, "--outdir", str(pool_dir), "--B", "12", "--strategy",
                 "random", "--frozen-config", str(cfg_p), "--routing-cache"])
        TP.build(str(pool_dir / "routing_profile_v2.npz"), str(cfg_p), str(pool_dir), half=20,
                 pool_seed=1, min_size=12)
        ws = TW.build(str(pool_dir), str(cfg_p), str(out), seeds=(0, 1), warm=4)
        common = ["--cache-dir", cache_dir, "--outdir", str(out), "--B", "12",
                  "--frozen-config", str(cfg_p), "--routing-cache"]
        res = {}
        for seed in (0, 1):
            for pool, f in POOLS.items():
                for st in STRATS:
                    args = common + ["--strategy", st, "--seed", str(seed),
                                     "--warm-start-file", str(out / "warm_starts.json")]
                    if f:
                        args += ["--candidate-subset-file", str(pool_dir / f)]
                    res[(pool, st, seed)] = json.load(open(E3.main(args)))
        res["again"] = json.load(open(E3.main(
            common + ["--strategy", "coverage_lp", "--seed", "0", "--warm-start-file",
                      str(out / "warm_starts.json"), "--candidate-subset-file",
                      str(pool_dir / POOLS["typical_half"])])))
        res["plain_random"] = json.load(open(E3.main(common + ["--strategy", "random", "--seed", "0"])))
    pools = {k: v["ids"] for k, v in json.load(open(pool_dir / "pools.json"))["pools"].items()}
    return dict(res=res, ws=ws, pools=pools, split=sp, common=common, out=out, pool_dir=pool_dir)


def test_warm_start_construction_deterministic_and_in_intersection(setup, tmp_path):
    ws, pools = setup["ws"], setup["pools"]
    elig = sorted(set(pools["typical_half"]) & set(pools["random_half"]))
    assert ws["eligible"]["ids"] == elig
    for s in ("0", "1"):
        ids = ws["seeds"][s]["ids"]
        assert ids == random.Random(int(s)).sample(elig, 4) and len(set(ids)) == 4
        assert all(i in pools["full"] and i in pools["typical_half"] and i in pools["random_half"]
                   for i in ids)
        assert not set(ids) & (set(setup["split"]["tune"]) | set(setup["split"]["test"]))
    again = TW.build(str(setup["pool_dir"]), str(setup["pool_dir"].parent / "frozen.json"),
                     str(tmp_path), seeds=(0, 1), warm=4)
    assert again["seeds"] == ws["seeds"]
    with pytest.raises(SystemExit, match="not feasible"):
        TW.build(str(setup["pool_dir"]), str(setup["pool_dir"].parent / "frozen.json"),
                 str(tmp_path), seeds=(0,), warm=len(elig) + 1)


def test_identical_ordered_warm_start_across_six_conditions(setup):
    r = setup["res"]
    for seed in (0, 1):
        want = setup["ws"]["seeds"][str(seed)]["ids"]
        for pool in POOLS:
            for st in STRATS:
                x = r[(pool, st, seed)]
                assert x["warm_start"] == want and x["chosen"][:4] == want, (pool, st, seed)
                assert x["summary"]["warm_start_fixed"]["hash"] == setup["ws"]["seeds"][str(seed)]["hash"]


def test_selection_size_uniqueness_and_pool_restriction(setup):
    r, pools = setup["res"], setup["pools"]
    held = set(setup["split"]["tune"]) | set(setup["split"]["test"])
    for seed in (0, 1):
        for pool in POOLS:
            for st in STRATS:
                ch = r[(pool, st, seed)]["chosen"]
                assert len(ch) == 12 and len(set(ch)) == 12
                assert set(ch) <= set(pools[pool]) and not set(ch) & held
                assert not set(ch[4:]) & set(ch[:4])            # warm start excluded afterwards


def test_gptq_and_eval_universe_identical(setup):
    r = setup["res"]
    univ = {json.dumps(v["eval_universe"]) for k, v in r.items() if isinstance(k, tuple)}
    assert len(univ) == 1
    for k, v in r.items():
        s = v["summary"]
        assert (s["percdamp"], s["bits"], s["eval_split"]) == (1.0, 4, "test")
        assert s["routing_counts_match_hessian_tokens"]


def test_reproducible_and_backward_compatible(setup):
    r = setup["res"]
    assert r["again"]["chosen"] == r[("typical_half", "coverage_lp", 0)]["chosen"]
    sp = setup["split"]
    assert r["plain_random"]["chosen"] == random.Random(0).sample(sp["pool"], 12)
    assert r["plain_random"]["summary"]["warm_start_fixed"] is None


def test_lp_counts_initialised_from_warm_start_and_rows_map(monkeypatch):
    rng0 = np.random.default_rng(3)
    ids = list(range(100, 160))                          # non-zero-based original ids
    C = rng0.poisson(2.0, size=(60, 2 * 5)).astype(float)
    warm = [130, 101, 155, 142]
    seen = []
    orig = AM.compute_expert_utility

    def spy(strategy, n, layers, E_, m, art):
        seen.append((n.copy(), m))
        return orig(strategy, n, layers, E_, m, art)

    monkeypatch.setattr(AM, "compute_expert_utility", spy)
    ch, w, traj = E3.select_lp_utility("coverage_lp", C, ids, [1, 2], 5, 12, 4, 2,
                                       random.Random(0), None, warm_start=warm)
    np.testing.assert_array_equal(seen[0][0], C[[i - 100 for i in warm]].sum(0))
    assert seen[0][1] == 4 and ch[:4] == warm and w == warm and len(set(ch)) == 12
    with pytest.raises(ValueError, match="schedule needs"):
        E3.select_lp_utility("coverage_lp", C, ids, [1, 2], 5, 12, 4, 2, random.Random(0), None,
                             warm_start=warm[:3])


def test_shared_warm_start_different_pools_diverge():
    """Same warm start, different eligible pools -> different remaining picks."""
    rng0 = np.random.default_rng(0)
    C = rng0.poisson(3.0, size=(40, 2 * 4)).astype(float)
    C[20:, 0] += 10                                      # rows 20.. strongly cover expert 0
    full, half = list(range(40)), list(range(20)) + [20]
    warm = [0, 1, 2, 3]
    a, _, _ = E3.select_lp_utility("coverage_lp", C, full, [1, 2], 4, 12, 4, 2, random.Random(0),
                                   None, warm_start=warm)
    b, _, _ = E3.select_lp_utility("coverage_lp", C[half], half, [1, 2], 4, 12, 4, 2,
                                   random.Random(0), None, warm_start=warm)
    assert a[:4] == b[:4] == warm and a[4:] != b[4:] and set(b) <= set(half)


def test_invalid_warm_start_files_fail(setup, tmp_path):
    base = setup["common"] + ["--seed", "0"]
    wf = setup["out"] / "warm_starts.json"
    d = json.load(open(wf))
    bad = dict(d, seeds={"0": dict(d["seeds"]["0"], hash="0")})
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="hash"):
        E3.main(base + ["--strategy", "random", "--warm-start-file", str(p)])
    other = dict(d, pools={"full": d["pools"]["full"], "typical_half": "deadbeef"})
    p2 = tmp_path / "other.json"
    p2.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="not built for candidate pool"):
        E3.main(base + ["--strategy", "random", "--warm-start-file", str(p2),
                        "--candidate-subset-file", str(setup["pool_dir"] / POOLS["typical_half"])])
    with pytest.raises(ValueError, match="no entry for seed"):
        E3.main(setup["common"] + ["--seed", "7", "--strategy", "random", "--warm-start-file", str(wf)])
    with pytest.raises(ValueError, match="not supported"):
        E3.main(base + ["--strategy", "coverage", "--warm-start-file", str(wf)])
