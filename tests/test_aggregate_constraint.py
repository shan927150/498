"""CPU tests for the aggregate-shift-constrained Coverage-LP (coverage_lp_aggregate)."""
import json
import os
import random
import sys

import numpy as np
import pytest

import e3_allocation as E3
from moe_calib import aggregate_constraint as AG
from moe_calib import alloc_models as AM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))

NL, E, T = 2, 4, 8          # 2 layers x 4 experts, 8 routed tokens per sequence per layer


def synth(S=60, seed=0, n_skew=20):
    """Every row routes exactly T tokens per layer. Rows < n_skew put most
    tokens on the rare expert 3 (coverage wants them; they shift the aggregate)."""
    g = np.random.default_rng(seed)
    C = np.zeros((S, NL * E))
    for l in range(NL):
        C[:, l * E:(l + 1) * E] = g.multinomial(T, [0.32, 0.32, 0.32, 0.04], size=S)
        C[:n_skew, l * E:(l + 1) * E] = g.multinomial(T, [0.1, 0.1, 0.1, 0.7], size=n_skew)
    return C


def spec_for(C, B, eps):
    return AG.AggSpec(AG.pool_reference_flat(C, NL, E), np.full(NL, eps) if np.isscalar(eps) else eps,
                      B, AG.token_invariant(C, NL, E), NL, E)


def D_manual(C, rows, B):
    out = []
    for l in range(NL):
        X = C[:, l * E:(l + 1) * E]
        ref = X.sum(0) / X.sum()
        sel = X[rows].sum(0) / X[rows].sum()
        out.append(np.abs(sel - ref).sum())
    return np.array(out)


# ------------------------------------------------------------- basics

def test_reference_l1_and_fixed_token_normalisation():
    C = synth()
    ref = AG.pool_reference_flat(C, NL, E)
    for l in range(NL):
        X = C[:, l * E:(l + 1) * E]
        np.testing.assert_allclose(ref[l * E:(l + 1) * E], X.sum(0) / X.sum())
    rows = [1, 5, 9, 30, 40, 41]
    sp = spec_for(C, len(rows), 1.0)
    assert AG.token_invariant(C, NL, E) == T
    # (sum C z)/(B*T) equals the normalised selected distribution because each row routes T tokens
    np.testing.assert_allclose(sp.D(C[rows].sum(0)), D_manual(C, rows, len(rows)), atol=1e-14)
    bad = C.copy()
    bad[0, 0] += 1
    with pytest.raises(ValueError, match="not constant"):
        AG.token_invariant(bad, NL, E)


def test_real_routing_cache_invariant_if_present():
    rp = os.path.join(ROOT, "runs_typicality_matched", "routing_profile_v2.npz")
    if not os.path.exists(rp):
        pytest.skip("real routing cache not present")
    z = np.load(rp)
    assert AG.token_invariant(z["C"], 4, 60) == 2048.0


def test_epsilon_deterministic_and_median_of_pooled_draws():
    C = synth(S=60)
    ids = list(range(100, 160))
    warm = {"0": [100, 101, 102], "1": [110, 111, 112]}
    a = AG.build_epsilon(C, ids, warm, 8, n_draws=50, eps_seed=7, n_layers=NL, E=E)
    b = AG.build_epsilon(C, ids, warm, 8, n_draws=50, eps_seed=7, n_layers=NL, E=E)
    np.testing.assert_array_equal(a["eps"], b["eps"])
    assert a["draws"] == b["draws"]
    allD = np.concatenate([a["D"]["0"], a["D"]["1"]])
    np.testing.assert_array_equal(a["eps"], np.median(allD, 0))
    row = {s: i for i, s in enumerate(ids)}
    for k, W in warm.items():
        for d, draw in zip(a["D"][k][:5], a["draws"][k][:5]):
            assert not set(draw) & {row[s] for s in W} and len(set(draw)) == 5
            np.testing.assert_allclose(d, D_manual(C, [row[s] for s in W] + list(draw), 8), atol=1e-14)
    c = AG.build_epsilon(C, ids, warm, 8, n_draws=50, eps_seed=8, n_layers=NL, E=E)
    assert c["draws"] != a["draws"]


# ------------------------------------------------------------- LP / MILP

def test_auxiliary_lp_equals_exact_l1_at_integer_points():
    """With z fixed to an integer selection, the minimal sum of t equals D_L."""
    C = synth()
    rows = [2, 7, 25, 33, 44, 50]
    sp = spec_for(C, 6, D_manual(C, rows, 6) + 1e-9)     # bound = exact D (tight)
    ok, _, _ = AG.completion_feasible(C[rows], np.zeros(NL * E), 6, sp)
    assert ok
    sp2 = spec_for(C, 6, D_manual(C, rows, 6) - 1e-6)
    ok2, _, _ = AG.completion_feasible(C[rows], np.zeros(NL * E), 6, sp2)
    assert not ok2


def test_warm_start_counts_enter_the_constraint():
    C = synth()
    W = list(range(0, 4))                                # skewed warm start
    sp = spec_for(C, 8, 0.25)
    rest = list(range(4, 60))
    ok, f, _ = AG.completion_feasible(C[rest], C[W].sum(0), 4, sp)
    if ok:
        assert sp.ok(C[W].sum(0) + C[rest][f].sum(0))
    # ignoring the warm start would wrongly certify a completion that is infeasible with it
    typical = list(range(30, 60))
    okw, _, _ = AG.completion_feasible(C[typical], C[W].sum(0), 4, spec_for(C, 8, 0.05))
    ok0, _, _ = AG.completion_feasible(C[typical], np.zeros(NL * E), 4, spec_for(C, 4, 0.30))
    assert ok0 and not okw


def test_fractional_feasible_but_naive_rounding_infeasible_is_repaired():
    """Hand-made: the LP mixes a skewed row with a typical one; naive rounding
    takes the skewed row and breaks the bound; the MILP repair restores it."""
    C = synth(S=40, seed=3, n_skew=10)
    sp = spec_for(C, 6, 0.30)
    warm = [20, 21]
    n_fix = C[warm].sum(0)
    cand = [i for i in range(40) if i not in warm]
    Cc = C[cand]
    value = 1.0 / np.maximum(n_fix, 1)
    cap = n_fix.copy()
    z, f, info = AG.lp_round(Cc, value, cap, 2, 2, n_fix, sp)
    assert info["lp_status"] == 0 and abs(z.sum() - 2) < 1e-6
    take, rows, log = AG.select_round_aggregate(Cc, cand, value, cap, np.zeros(NL * E, bool), 2, 2,
                                                n_fix, sp, random.Random(0), E3.positive_marginal_selection)
    assert len(take) == 2 == len(set(take))
    ok, comp, _ = AG.completion_feasible(np.delete(Cc, rows, 0), n_fix + Cc[rows].sum(0), 2, sp)
    assert ok
    if not log["naive_rounding_feasible"]:
        assert log["repaired"] and log["completion_certificate"]


def test_naive_rounding_violation_detected_and_repaired_constructed(monkeypatch):
    """Deterministic construction: the fractional point z = (.5 skewed, .25, .25
    typical) satisfies the bound (L1 is convex), but rounding to the top-z row
    (the skewed one) violates it; the MILP repair returns a feasible row."""
    typ = [2, 2, 2, 2]
    rows = [typ * NL] * 20 + [[1, 3, 2, 2] * NL, [3, 1, 2, 2] * NL] + [[0, 0, 0, 8] * NL]
    C = np.array(rows, float)                           # every row routes T = 8 tokens per layer
    ids = list(range(len(C)))
    warm = list(range(4))
    n_fix = C[warm].sum(0)
    cand = ids[4:]
    Cc = C[cand]
    iA, i1, i2 = len(cand) - 1, 0, 1                     # skewed row and two plain typical rows
    z = np.zeros(len(cand)); z[iA], z[i1], z[i2] = 0.5, 0.25, 0.25
    B = 5
    ref = AG.pool_reference_flat(C, NL, E)
    D = lambda counts: AG.aggregate_l1(counts, ref, B * T, NL, E)
    frac = D(n_fix + z @ Cc)
    eps = frac + 1e-6
    sp = AG.AggSpec(ref, eps, B, T, NL, E)
    assert not sp.ok(n_fix + Cc[iA]) and sp.ok(n_fix + Cc[i1])      # naive infeasible, others fine
    monkeypatch.setattr(AG, "lp_round", lambda *a, **k: (z, np.zeros(0), dict(lp_status=0, lp_objective=1.0, lp_s=0.0)))
    value = np.zeros(NL * E); value[[3, 7]] = 1.0
    cap = np.full(NL * E, 100.0)
    take, rr, log = AG.select_round_aggregate(Cc, cand, value, cap, np.zeros(NL * E, bool), 1, 0, n_fix, sp,
                                              random.Random(0), E3.positive_marginal_selection)
    assert log["naive_rounding_feasible"] is False and log["repaired"] is True
    assert rr != [iA] and sp.ok(n_fix + Cc[rr].sum(0)) and log["rounding_objective_loss"] > 0


def test_infeasible_bounds_raise_never_relaxed():
    C = synth()
    ids = list(range(60))
    sp = spec_for(C, 12, 1e-6)
    with pytest.raises(AG.AggInfeasible):
        E3.select_lp_utility("coverage_lp_aggregate", C, ids, [1, 2], E, 12, 4, 2, random.Random(0), None,
                             warm_start=[0, 1, 2, 3], agg_spec=sp)


# ------------------------------------------------------------- controller

def run_ag(C, ids, warm, B, eps, seed=0):
    sp = spec_for(C, B, eps)
    return E3.select_lp_utility("coverage_lp_aggregate", C, ids, [1, 2], E, B, 4, 2, random.Random(seed),
                                None, warm_start=warm, agg_spec=sp), sp


def test_controller_final_bounds_all_layers_unique_B_and_id_mapping():
    C = synth(S=80, seed=4, n_skew=25)
    ids = list(range(500, 580))                          # non-zero-based original ids
    warm = [540, 541, 542, 543]
    B = 12
    unc, _, _ = E3.select_lp_utility("coverage_lp", C, ids, [1, 2], E, B, 4, 2, random.Random(0), None,
                                     warm_start=warm)
    row = {s: i for i, s in enumerate(ids)}
    D_unc = D_manual(C, [row[s] for s in unc], B)
    eps = np.minimum(D_unc * 0.6, 0.3)
    (ch, w, traj), sp = run_ag(C, ids, warm, B, eps)
    assert len(ch) == B == len(set(ch)) and ch[:4] == warm and set(ch) <= set(ids)
    D = D_manual(C, [row[s] for s in ch], B)
    assert np.all(D <= eps + 1e-12)                      # every layer, warm start included
    np.testing.assert_allclose(traj[-1]["aggregate_final"]["D"], D, atol=1e-12)
    assert np.all(D_unc > eps)                           # unconstrained coverage violates the bound
    for t in traj:
        for k in ("lp_objective", "naive_objective", "final_objective", "naive_rounding_feasible",
                  "repaired", "n_rounding_repairs", "certificate_s", "round_select_s", "lp_s"):
            assert k in t
    assert traj[0]["k_future"] == traj[0]["round_budget"] and traj[1]["k_future"] == 0


def test_controller_deterministic_and_coverage_lp_backward_compatible():
    C = synth(S=70, seed=5)
    ids = list(range(70))
    warm = [0, 41, 42, 43]                               # row 0 covers the rare experts
    a, _ = run_ag(C, ids, warm, 12, 0.25)
    b, _ = run_ag(C, ids, warm, 12, 0.25)
    assert a[0] == b[0]
    u1 = E3.select_lp_utility("coverage_lp", C, ids, [1, 2], E, 12, 4, 2, random.Random(0), None, warm_start=warm)
    u2 = E3.select_lp_utility("coverage_lp", C, ids, [1, 2], E, 12, 4, 2, random.Random(0), None, warm_start=warm)
    assert u1[0] == u2[0] and "aggregate_final" not in u1[2][-1]
    assert "coverage_lp_aggregate" not in AM.LP_UTILITY_STRATEGIES
    assert AM.AGGREGATE_LP_STRATEGIES["coverage_lp_aggregate"] == "coverage_lp"
    with pytest.raises(ValueError, match="AggSpec"):
        E3.select_lp_utility("coverage_lp_aggregate", C, ids, [1, 2], E, 12, 4, 2, random.Random(0), None,
                             warm_start=warm)


def test_multi_round_first_round_keeps_integer_completion():
    """After round 0 the remaining pool must still admit an integer completion
    (the certificate recorded in the log is such a completion)."""
    C = synth(S=70, seed=6, n_skew=25)
    ids = list(range(70))
    (ch, w, traj), sp = run_ag(C, ids, [30, 31, 32, 33], 12, 0.22)
    cert = traj[0]["completion_certificate"]
    first = ch[:4 + traj[0]["round_budget"]]
    assert len(cert) == traj[0]["k_future"] and not set(cert) & set(first)
    assert sp.ok(C[first].sum(0) + C[cert].sum(0))


# ------------------------------------------------------------- locked protocol / split

def test_driver_never_relocks_and_detects_tampering(tmp_path):
    import aggregate_constraint as DR
    p = tmp_path / "locked_protocol.json"
    p.write_text(json.dumps({"x": 1}))
    (tmp_path / "locked_protocol.json.sha256").write_text(DR.sha_file(str(p)) + "\n")
    proto, h = DR.load_protocol(str(tmp_path))
    assert proto == {"x": 1}
    p.write_text(json.dumps({"x": 2}))
    with pytest.raises(SystemExit, match="modified after locking"):
        DR.load_protocol(str(tmp_path))
    with pytest.raises(SystemExit, match="already locked"):
        DR.lock(type("A", (), dict(outdir=str(tmp_path), frozen="configs/frozen_b4.json"))())


def test_confirmation_split_artifacts_if_present():
    """Exact reproduction + document disjointness, as recorded by the replay."""
    d = os.path.expanduser("~/moe_cache_confirmation/provenance.json")
    if not os.path.exists(d):
        pytest.skip("confirmation split not built")
    prov = json.load(open(d))
    assert prov["reproduces_existing_ids"] and prov["disjoint_ok"]
    assert prov["confirmation_size"] == 256 and not prov["document_index_overlap"]
    assert not prov["document_text_overlap"] and not prov["windows_sharing_a_32token_shingle_with_old"]
    import torch
    import build_confirmation_split as BC
    new = torch.load(os.path.expanduser("~/moe_cache_confirmation/ids.pt"))["confirmation"]
    assert BC.fingerprint(new) == prov["confirmation_fingerprint"] and new.shape == (256, 512)
    docs = {r: set(v) for r, v in prov["old_role_docs"].items()}
    newd = {p["doc"] for p in prov["confirmation"]}
    assert all(not (newd & v) for v in docs.values())


def test_selections_artifact_isolation_if_present():
    p = os.path.join(ROOT, "runs_aggregate_constraint", "selections.json")
    if not os.path.exists(p):
        pytest.skip("selections not prepared")
    sel = json.load(open(p))
    ws = json.load(open(os.path.join(ROOT, "runs_typicality_matched", "warm_starts.json")))
    for sd, runs in sel["seeds"].items():
        for k, r in runs.items():
            ch = r["chosen"]
            assert len(ch) == 128 == len(set(ch)) and ch[:32] == ws["seeds"][sd]["ids"]
            assert all(0 <= s < 2000 for s in ch)        # pool only: no tune/test/confirmation rows
        assert runs["AG"]["satisfies_aggregate_bounds"]


def test_zero_exposure_targets_are_refused_explicitly():
    C = synth(S=70, seed=5)
    typ = [i for i in range(20, 70) if C[i, 3] == 0 and C[i, 7] == 0][:4]
    with pytest.raises(AG.AggUnresolved, match="exploration targets"):
        run_ag(C, list(range(70)), typ, 12, 0.25)
