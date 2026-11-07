"""CPU tests for the routing-typicality experiment: typicality scores, pool
construction, and E3's --candidate-subset-file restriction."""
import json

import numpy as np
import pytest

import e3_allocation as E3
from conftest import E, LAYERS, tiny_model
from moe_calib import model_utils as MU
from moe_calib import typicality as TY
from moe_calib.data_utils import load_meta, split_ids
from test_allocation_minimal import frozen_for


# --------------------------------------------------------------- scores

def test_normalization_l1_and_pool_reference():
    # 2 layers x 3 experts; rows = sequences
    C = np.array([[2, 2, 0, 1, 1, 2],
                  [0, 4, 0, 4, 0, 0]], float)
    ref = TY.pool_reference(C, 2, 3)
    np.testing.assert_allclose(ref, [[2 / 8, 6 / 8, 0], [5 / 8, 1 / 8, 2 / 8]])
    sc = TY.sequence_scores(C, 2, 3)
    q0 = np.array([[.5, .5, 0], [.25, .25, .5]])
    d0 = np.abs(q0 - ref).sum(1)
    np.testing.assert_allclose(sc["d_layer"][0], d0)
    assert sc["d"][0] == pytest.approx(d0.mean())
    np.testing.assert_allclose(sc["entropy"][1], 0.0)             # one expert per layer
    # subset scored against the ORIGINAL reference, not recentered
    sub = TY.sequence_scores(C[:1], 2, 3, ref)
    assert sub["d"][0] == pytest.approx(sc["d"][0])
    assert TY.sequence_scores(C[:1], 2, 3)["d"][0] == pytest.approx(0.0)


def test_layer_ordering_is_layer_major():
    rng = np.random.default_rng(0)
    C = rng.poisson(3, size=(30, 3 * 4)).astype(float)
    sc = TY.sequence_scores(C, 3, 4)
    perm = np.concatenate([np.arange(8, 12), np.arange(0, 4), np.arange(4, 8)])   # layers 2,0,1
    sp = TY.sequence_scores(C[:, perm], 3, 4)
    np.testing.assert_allclose(sp["d_layer"], sc["d_layer"][:, [2, 0, 1]])
    np.testing.assert_allclose(sp["d"], sc["d"])


def test_ties_zero_counts_and_determinism():
    C = np.array([[1, 1, 1, 1],
                  [1, 1, 1, 1],          # tie with row 0
                  [4, 0, 4, 0],
                  [0, 0, 2, 2],          # zero routed tokens in layer 0 -> malformed
                  [2, 0, 0, 2]], float)
    sc = TY.sequence_scores(C, 2, 2)
    assert sc["malformed"].tolist() == [False, False, False, True, False]
    assert np.isnan(sc["d"][3])
    ids = [10, 11, 12, 13, 14]
    assert TY.typical_half(ids, sc["d"], 1) == [10]               # tie -> lowest original id
    assert 13 not in TY.typical_half(ids, sc["d"], 4)
    with pytest.raises(ValueError):
        TY.typical_half(ids, sc["d"], 5)                          # malformed never imputed
    r1, r2 = TY.random_half(range(2000), 1000, 7), TY.random_half(range(2000), 1000, 7)
    assert r1 == r2 and len(set(r1)) == 1000 and r1 != TY.random_half(range(2000), 1000, 8)


# ------------------------------------------------------------- E3 subset

def write_subset(path, name, ids):
    path.write_text(json.dumps(dict(name=name, ids=ids, fingerprint=E3.subset_fingerprint(name, ids))))
    return str(path)


@pytest.fixture(scope="module")
def runs(cache_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("typ")
    sp = split_ids(load_meta(cache_dir))
    cfg_p = out / "frozen.json"
    cfg_p.write_text(json.dumps(frozen_for(LAYERS, sp["definition"])))
    sub_ids = sp["pool"][::2]                                     # 20 of 40 candidates
    sub = write_subset(out / "pool_even.json", "even", sub_ids)
    full = write_subset(out / "pool_fullcopy.json", "fullcopy", list(sp["pool"]))
    common = ["--cache-dir", cache_dir, "--outdir", str(out), "--B", "12",
              "--frozen-config", str(cfg_p), "--routing-cache"]
    res = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MU, "load_model", lambda *a, **k: tiny_model())
        load = lambda args: json.load(open(E3.main(common + args)))
        res["random"] = load(["--strategy", "random", "--candidate-subset-file", sub])
        res["cov"] = load(["--strategy", "coverage_lp", "--candidate-subset-file", sub])
        res["cov_again"] = load(["--strategy", "coverage_lp", "--candidate-subset-file", sub])
        res["cov_noflag"] = load(["--strategy", "coverage_lp"])
        res["cov_fullcopy"] = load(["--strategy", "coverage_lp", "--candidate-subset-file", full])
    return dict(res=res, sub_ids=sub_ids, split=sp, common=common, out=out)


def test_selection_stays_in_subset(runs):
    sub = set(runs["sub_ids"])
    held = set(runs["split"]["tune"]) | set(runs["split"]["test"])
    for k in ("random", "cov", "cov_again"):
        r = runs["res"][k]
        ch = r["chosen"]
        assert len(ch) == 12 and len(set(ch)) == 12 and set(ch) <= sub and not set(ch) & held, k
        assert r["summary"]["candidate_subset"]["name"] == "even"
    cov = runs["res"]["cov"]
    assert set(cov["warm_start"]) <= sub
    assert cov["summary"]["routing_counts_match_hessian_tokens"]


def test_lp_rows_map_to_original_ids(runs):
    """tokens_added per layer in round 0 must equal the FULL-pool routing rows
    of the sequences that round actually took (original ids)."""
    z = np.load(runs["out"] / E3.ROUTING_CACHE)
    C, row = z["C"], {int(s): i for i, s in enumerate(z["pool_ids"])}
    r = runs["res"]["cov"]
    t0 = r["trajectory"][0]
    take = r["chosen"][4:4 + t0["n_taken"]]
    for li, L in enumerate(LAYERS):
        want = C[[row[s] for s in take]][:, li * E:(li + 1) * E].sum()
        assert t0["per_layer"][str(L)]["tokens_added"] == pytest.approx(want)


def test_reproducible_and_backward_compatible(runs):
    r = runs["res"]
    assert r["cov"]["chosen"] == r["cov_again"]["chosen"]
    # a subset listing the whole pool reproduces the no-flag run exactly
    assert r["cov_fullcopy"]["chosen"] == r["cov_noflag"]["chosen"]
    assert r["cov_fullcopy"]["summary"]["recon_mean"] == r["cov_noflag"]["summary"]["recon_mean"]
    assert r["cov_noflag"]["summary"]["candidate_subset"] is None
    for k in r:
        assert r[k]["summary"]["percdamp"] == 1.0 and r[k]["summary"]["bits"] == 4


def test_invalid_subsets_fail(runs, tmp_path):
    sp = runs["split"]
    base = runs["common"] + ["--strategy", "coverage_lp"]
    leak = write_subset(tmp_path / "leak.json", "leak", sp["pool"][:20] + sp["test"][:1])
    with pytest.raises(ValueError, match="outside the calibration pool"):
        E3.main(base + ["--candidate-subset-file", leak])
    bad = tmp_path / "badfp.json"
    bad.write_text(json.dumps(dict(name="x", ids=sp["pool"][:20], fingerprint="0")))
    with pytest.raises(ValueError, match="fingerprint"):
        E3.main(base + ["--candidate-subset-file", str(bad)])
    small = write_subset(tmp_path / "small.json", "small", sp["pool"][:5])
    with pytest.raises(ValueError, match="< B"):
        E3.main(base + ["--candidate-subset-file", small])
    ok = write_subset(tmp_path / "ok.json", "ok", sp["pool"][:20])
    nocache = [a for a in base if a != "--routing-cache"]
    with pytest.raises(ValueError, match="routing-cache"):
        E3.main(nocache + ["--candidate-subset-file", ok])
    with pytest.raises(ValueError, match="not supported"):
        E3.main(runs["common"] + ["--strategy", "tlp", "--candidate-subset-file", ok])
