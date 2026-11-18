"""CPU tests for the expert-wise calibration MVP (synthetic routing / activations + the tiny Qwen2-MoE fixture)."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_calib import expertwise_calibration as EW   # noqa: E402
from moe_calib import model_utils as MU             # noqa: E402


# ------------------------------------------------------------------ synthetic sketches
def _synthetic(N=60, experts=(0, 1, 2), k=6, seed=0, zero_frac=0.3):
    """Per-sequence svech sketches from random 'activations' with expert-specific, sequence-varying scales; some
    (seq, expert) pairs get zero routed tokens."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    iu, w = EW.svech_index(k)
    M, n = {}, {}
    for e in experts:
        cnt = rng.integers(1, 12, size=N)
        cnt[rng.random(N) < zero_frac] = 0
        n[e] = torch.as_tensor(cnt, dtype=torch.float64)
        for p in EW.PROJS:
            rows = []
            for s in range(N):
                X = torch.randn(int(cnt[s]), k, generator=g, dtype=torch.float64) * (1 + 0.5 * (s % 5)) * (1 + e)
                rows.append(EW.svech(X.T @ X, iu, w))
            M[(e, p)] = torch.stack(rows)
    return M, n, list(experts)


def test_svech_preserves_frobenius_and_linearity():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(7, 7, generator=g, dtype=torch.float64)
    A, B = A + A.T, torch.eye(7, dtype=torch.float64) * 3
    assert torch.isclose(torch.linalg.norm(EW.svech(A)), torch.linalg.norm(A))
    assert torch.allclose(EW.svech(A + 2 * B), EW.svech(A) + 2 * EW.svech(B))


def test_select_experts_covers_frequency_range_and_breaks_ties_by_id():
    counts = np.array([5, 1, 9, 1, 7, 3, 3, 8, 2, 6])
    ex = EW.select_experts(counts, 4)
    order = [1, 3, 8, 5, 6, 0, 9, 4, 7, 2]           # ascending count, ties -> lower id
    assert ex == [order[0], order[3], order[6], order[9]]
    assert ex[0] == 1 and ex[-1] == 2
    assert EW.select_experts(counts, 4) == ex          # deterministic


def test_projection_matrices_fixed_seed():
    a = EW.projection_matrices({"gu": 32, "down": 16}, k=8, seed=3)
    b = EW.projection_matrices({"gu": 32, "down": 16}, k=8, seed=3)
    c = EW.projection_matrices({"gu": 32, "down": 16}, k=8, seed=4)
    assert all(torch.equal(a[p], b[p]) for p in EW.PROJS)
    assert not torch.equal(a["gu"], c["gu"]) and a["gu"].shape == (32, 8) and a["down"].dtype == torch.float32


def test_objective_matches_direct_definition_and_empty_rule():
    M, n, ex = _synthetic()
    obj = EW.build_terms(M, n, ex, shared=True)
    sel = [3, 7, 11, 20, 41]
    direct = 0.0
    for e in ex:
        for p in EW.PROJS:
            R = M[(e, p)].sum(0) / n[e].sum()
            ns = n[e][sel].sum()
            cur = M[(e, p)][sel].sum(0) / ns if ns > 0 else torch.zeros_like(R)
            direct += float(torch.linalg.norm(cur - R) / (torch.linalg.norm(R) + EW.EPS)) / (2 * len(ex))
    assert obj.exact(sel) == pytest.approx(direct, rel=1e-12)
    assert obj.exact([]) == pytest.approx(1.0, abs=1e-9)               # empty set -> every term ~1
    z = int(torch.nonzero(n[ex[0]] == 0)[0])
    one = EW.build_terms(M, n, [ex[0]], shared=False)[ex[0]]
    assert one.exact([z]) == pytest.approx(1.0, abs=1e-9)                # zero routed tokens -> H(S) := 0
    full = list(range(len(n[ex[0]])))
    assert obj.exact(full) == pytest.approx(0.0, abs=1e-9)               # whole pool reproduces the reference


@pytest.mark.parametrize("shared", [True, False])
def test_closed_form_add_and_swap_scores_equal_exact(shared):
    M, n, ex = _synthetic(seed=1)
    obj = EW.build_terms(M, n, ex, shared=shared)
    obj = obj if shared else obj[ex[1]]
    for sel in ([], [5], [2, 9, 17, 30, 44, 51]):
        st = obj.state(sel)
        sc = obj.add_scores(st)
        for c in (0, 1, 13, 59):
            if c not in sel:
                assert float(sc[c]) == pytest.approx(obj.exact(sel + [c]), rel=1e-9, abs=1e-12)
        for i in sel[:2]:
            sw = obj.swap_scores(st, i)
            for c in (0, 22, 58):
                if c not in sel:
                    trial = [x for x in sel if x != i] + [c]
                    assert float(sw[c]) == pytest.approx(obj.exact(trial), rel=1e-9, abs=1e-12)


def test_selection_deterministic_valid_and_monotone():
    M, n, ex = _synthetic(N=80, seed=2)
    B = 12
    a = EW.select_all(M, n, ex, B, seed=5, sample_frac=0.25, max_passes=4)
    b = EW.select_all(M, n, ex, B, seed=5, sample_frac=0.25, max_passes=4)
    c = EW.select_all(M, n, ex, B, seed=6, sample_frac=0.25, max_passes=4)
    assert a["selection"] == b["selection"]
    assert a["selection"]["SR"] != c["selection"]["SR"]
    EW.validate_selection(a["selection"], ex, B, 80)
    for info in [a["optimizer"]["SC"], *a["optimizer"]["EC"].values()]:
        tr = info["trace"]
        assert all(y <= x for x, y in zip(tr, tr[1:]))
        assert info["final_objective"] <= info["greedy_objective"]
    # expert-wise sets are distinct per expert; shared sets identical
    assert len({tuple(a["selection"]["EC"][e]) for e in ex}) > 1
    assert len({tuple(a["selection"]["SR"][e]) for e in ex}) == 1


def test_covariance_selection_beats_random_on_its_objective():
    M, n, ex = _synthetic(N=100, seed=3)
    r = EW.select_all(M, n, ex, 15, seed=0, sample_frac=0.2)
    sel = r["selection"]
    sc = EW.build_terms(M, n, ex, shared=True)
    assert sc.exact(sel["SC"][ex[0]]) < sc.exact(sel["SR"][ex[0]])
    for e in ex:
        assert EW.expert_mismatch(M, n, e, sel["EC"][e]) <= EW.expert_mismatch(M, n, e, sel["SC"][e]) + 1e-12 or \
            EW.expert_mismatch(M, n, e, sel["EC"][e]) < EW.expert_mismatch(M, n, e, sel["ER"][e])


def test_swap_rejects_non_improving_and_trace_monotone():
    M, n, ex = _synthetic(N=40, seed=4)
    obj = EW.build_terms(M, n, ex, shared=True)
    full_opt = EW.greedy_select(obj, 10, np.random.default_rng(0), sample_frac=1.0)
    sel, tr, info = EW.swap_refine(obj, full_opt, np.random.default_rng(0), max_passes=6)
    assert len(sel) == 10 == len(set(sel)) and info["accepted"] == len(tr) - 1
    assert all(y < x for x, y in zip(tr, tr[1:]))
    sel2, tr2, info2 = EW.swap_refine(obj, sel, np.random.default_rng(1), max_passes=1)
    assert tr2[-1] <= tr[-1] + 1e-15


def test_validate_selection_catches_duplicates_and_size():
    ex = [0, 1]
    good = {c: {e: list(range(4)) for e in ex} for c in EW.CONDITIONS}
    EW.validate_selection(good, ex, 4, 10)
    bad = {c: {e: list(v) for e, v in d.items()} for c, d in good.items()}
    bad["EC"][1] = [0, 0, 1, 2]
    with pytest.raises(AssertionError):
        EW.validate_selection(bad, ex, 4, 10)
    bad = {c: {e: list(v) for e, v in d.items()} for c, d in good.items()}
    bad["SC"][1] = [0, 1, 2, 5]
    with pytest.raises(AssertionError):
        EW.validate_selection(bad, ex, 4, 10)
    with pytest.raises(AssertionError):
        EW.validate_selection(good, [0, 1, 2], 4, 10)


# ------------------------------------------------------------------ token matching
def test_match_tokens_exact_quota_unique_and_deterministic():
    counts = {3: 5, 9: 0, 11: 7, 20: 2}
    a = EW.match_tokens(counts, 8, np.random.default_rng(1))
    b = EW.match_tokens(counts, 8, np.random.default_rng(1))
    assert sum(len(v) for v in a.values()) == 8
    assert all(torch.equal(a[s], b[s]) for s in a) and a.keys() == b.keys()
    for s, rows in a.items():
        assert counts[s] > 0 and len(set(rows.tolist())) == len(rows) and int(rows.max()) < counts[s]
        assert torch.equal(rows, torch.sort(rows).values)
    full = EW.match_tokens(counts, 14, np.random.default_rng(0))
    assert {s: len(v) for s, v in full.items()} == {3: 5, 11: 7, 20: 2}
    with pytest.raises(ValueError):
        EW.match_tokens(counts, 15, np.random.default_rng(0))


def test_matching_plan_identical_quota_and_aligned_rows():
    M, n, ex = _synthetic(N=50, seed=5, zero_frac=0.1)
    sel = EW.select_all(M, n, ex, 10, seed=1, sample_frac=0.3)["selection"]
    plan = EW.matching_plan(sel, lambda e: n[e].long().numpy(), ex, seed=1)
    for e in ex:
        assert plan["quota"][e] == min(plan["counts"][c][e] for c in EW.CONDITIONS)
        for c in EW.CONDITIONS:
            rows = plan["plan"][c][e]
            assert sum(len(v) for v in rows.values()) == plan["quota"][e]
            assert set(rows) <= set(sel[c][e])
    # gate/up and down use the same row indices -> sampled (x_gu, x_down) pairs stay paired
    g = torch.Generator().manual_seed(0)
    Xg = torch.randn(9, 4, generator=g)
    Xd = Xg[:, :2] * 10                                      # down row i is a function of gu row i
    rows = EW.match_tokens({0: 9}, 4, np.random.default_rng(0))[0]
    assert torch.equal(Xg[rows][:, :2] * 10, Xd[rows])


def test_source_diversity_and_overlap():
    d = EW.source_diversity({1: [0, 1], 2: [0, 1], 3: []})
    assert d["distinct"] == 2 and d["inv_simpson"] == pytest.approx(2.0) and d["exp_entropy"] == pytest.approx(2.0)
    sel = {"SR": {0: [1, 2], 1: [1, 2]}, "SC": {0: [2, 3], 1: [2, 3]},
           "EC": {0: [2, 3], 1: [4, 5]}, "ER": {0: [1, 6], 1: [7, 8]}}
    ov = EW.overlap_stats(sel, [0, 1])
    assert ov["SC"]["union"] == 2 and ov["EC"]["union"] == 4 and ov["ER"]["union"] == 4
    assert ov["EC"]["mean_jaccard_vs_SC"] == pytest.approx(0.5) and ov["EC"]["mean_pairwise_jaccard"] == 0.0


def test_mismatch_helpers_consistent():
    M, n, ex = _synthetic(seed=6)
    e, sel = ex[0], [1, 4, 9, 16]
    ref = {p: M[(e, p)].sum(0) / n[e].sum() for p in EW.PROJS}
    A = {p: M[(e, p)][sel].sum(0) for p in EW.PROJS}
    assert EW.mismatch_from_sums(A, float(n[e][sel].sum()), ref) == pytest.approx(
        EW.expert_mismatch(M, n, e, sel), rel=1e-12)
    H = torch.eye(3) * 2
    assert EW.rel_frob(H, 2.0, H, 2.0) == 0.0 and EW.rel_frob(H, 0.0, H, 1.0) == pytest.approx(1.0)


# ------------------------------------------------------------------ capture on the tiny model
def test_capture_natural_routing_alignment_and_counts(cache_dir):
    from conftest import tiny_model, LAYERS
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.profiler import Profiler
    model, _ = tiny_model()
    L = LAYERS[1]
    cache = load_layer_cache(cache_dir, L)
    experts = [0, 3, 5]
    seqs = list(range(10))
    got = {}

    def cb(s, e, Xg, Xd):
        assert Xg.shape[0] == Xd.shape[0]
        got[(s, e)] = Xg.shape[0]

    plain = [MU.run_layer(model, L, cache[s:s + 1]) for s in seqs]
    info = EW.replay(model, L, cache, seqs, experts, cb)
    hooked = [MU.run_layer(model, L, cache[s:s + 1]) for s in seqs]   # hooks removed, model unchanged
    assert all(torch.equal(a, b) for a, b in zip(plain, hooked))
    assert info["n_alignment_checks"] == len(seqs) * len(experts)
    prof = Profiler(model, L, collect_hessian=False).run(cache, seqs, routing_only=True)
    for s in seqs:
        for e in experts:
            assert got[(s, e)] == prof.seq_counts.get(s, {}).get((L, e), 0)


def test_capture_detects_misalignment(cache_dir):
    from conftest import tiny_model, LAYERS
    from moe_calib.data_utils import load_layer_cache
    model, _ = tiny_model()
    L = LAYERS[0]
    cache = load_layer_cache(cache_dir, L)
    blk = MU.moe_block(model, L)
    # perturb the gate/up input of expert 1 AFTER the capture hook would see it -> down rows no longer match
    h = blk.experts[1].up_proj.register_forward_pre_hook(lambda m, a: (a[0] * 2,))
    try:
        with pytest.raises(RuntimeError, match="not aligned"):
            EW.replay(model, L, cache, list(range(6)), [1], lambda *a: None)
    finally:
        h.remove()


def test_end_to_end_matched_hessians_quantize_on_tiny_model(cache_dir):
    """Sketch -> select -> match -> matched Hessians -> unchanged GPTQ -> held-out error, on the tiny model."""
    from conftest import tiny_model, LAYERS, POOL, HOLDOUT
    from moe_calib.data_utils import load_layer_cache
    from moe_calib.gptq import quantize_expert
    from moe_calib.profiler import Profiler
    from moe_calib.quant_eval import fp_weights, per_seq_rel_errors
    model, _ = tiny_model()
    L = LAYERS[1]
    cache = load_layer_cache(cache_dir, L)
    blk = MU.moe_block(model, L)
    experts = [0, 2, 6]
    dims = {"gu": model.config.hidden_size, "down": model.config.moe_intermediate_size}
    Rp = EW.projection_matrices(dims, k=8, seed=1)
    iu, w = EW.svech_index(8)
    Ms = {(e, p): torch.zeros(POOL, len(w), dtype=torch.float64) for e in experts for p in EW.PROJS}
    ns = {e: torch.zeros(POOL, dtype=torch.float64) for e in experts}

    def sk(s, e, Xg, Xd):
        ns[e][s] = Xg.shape[0]
        for p, X in (("gu", Xg), ("down", Xd)):
            Ms[(e, p)][s] = EW.svech(EW.sketch_outer(X, Rp[p]), iu, w)
    EW.replay(model, L, cache, list(range(POOL)), experts, sk)
    B = 12
    r1 = EW.select_all(Ms, ns, experts, B, seed=0, sample_frac=0.5)
    r2 = EW.select_all(Ms, ns, experts, B, seed=0, sample_frac=0.5)
    assert r1["selection"] == r2["selection"]
    sel = r1["selection"]
    EW.validate_selection(sel, experts, B, POOL)
    plan = EW.matching_plan(sel, lambda e: ns[e].long().numpy(), experts, seed=0)
    H = {(c, e, p): torch.zeros(dims[p], dims[p]) for c in EW.CONDITIONS for e in experts for p in EW.PROJS}
    cnt = {(c, e): 0 for c in EW.CONDITIONS for e in experts}

    def acc(s, e, Xg, Xd):
        for c in EW.CONDITIONS:
            rows = plan["plan"][c][e].get(s)
            if rows is None:
                continue
            cnt[(c, e)] += len(rows)
            for p, X in (("gu", Xg), ("down", Xd)):
                Xi = X[rows].float()
                H[(c, e, p)] += Xi.T @ Xi
    union = sorted({s for c in EW.CONDITIONS for e in experts for s in sel[c][e]})
    EW.replay(model, L, cache, union, experts, acc)
    for e in experts:
        assert len({cnt[(c, e)] for c in EW.CONDITIONS}) == 1 and cnt[("SR", e)] == plan["quota"][e]
    test_ids = list(range(POOL + HOLDOUT // 2, POOL + HOLDOUT))
    HO = Profiler(model, L, collect_hessian=False, collect_holdout=True).run(cache, test_ids).holdout
    before = {k: v.clone() for k, v in MU.get_expert_weights(model, L, experts[0]).items()}
    for c in EW.CONDITIONS:
        for e in experts:
            if plan["quota"][e] == 0:
                continue
            Wq = quantize_expert(fp_weights(model, L, e), H[(c, e, "gu")], H[(c, e, "down")], 4, 32, 1.0)
            errs = per_seq_rel_errors(model, L, e, Wq, HO)
            assert errs and all(np.isfinite(list(errs.values())))
    assert all(torch.equal(before[k], v) for k, v in MU.get_expert_weights(model, L, experts[0]).items())
    assert blk is MU.moe_block(model, L)
