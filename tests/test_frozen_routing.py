"""CPU tests for the frozen-routing factorial ablation (tiny Qwen2-MoE from conftest).

The GPU top-k semantics are checked in the integration stage (`capture`): the captured routing must reproduce the
UNMODIFIED Profiler's per-sequence routed-token counts on the GPU; here the same comparison runs on CPU."""
import copy
import hashlib
import os
import sys

import pytest
import torch

from conftest import E, LAYERS, tiny_model
from moe_calib import frozen_routing as FRT
from moe_calib import model_utils as MU
from moe_calib import partial_model_quant as PQ
from moe_calib.data_utils import load_layer_cache, load_meta, split_ids
from moe_calib.profiler import Profiler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = 16
L0 = LAYERS[0]


def _sha(t):
    return hashlib.sha256(t.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


def _plain_hessians(model, cache, rows, K=2):
    pf = Profiler(model, L0, K=K).run(cache, rows, weights=[1.0] * len(rows))
    return {(e, p): (pf.acc.H_sum((L0, e, p), list(range(K))), pf.acc.tokens((L0, e, p))) for e in range(E)
            for p in ("gu", "down")}, pf


def _forced_hessians(model, cache, rows, table, r2s, natural=False, K=2):
    fp = FRT.ForcedProfiler(model, L0, table, r2s, expect_natural=natural, K=K)
    fp.run(cache, rows, weights=[1.0] * len(rows))
    return {(e, p): (fp.acc.H_sum((L0, e, p), list(range(K))), fp.acc.tokens((L0, e, p))) for e in range(E)
            for p in ("gu", "down")}, fp


def _natural_table(model, cache, rows):
    sel, prob = FRT.capture_routing(model, L0, cache, rows, rows)
    return sel, prob


def _block_inputs(model, cache, rows):
    """MoE-block input rows of the natural replay (the activation values the experts see)."""
    blk = MU.moe_block(model, L0)
    got = {}
    cur = {}
    h = blk.register_forward_pre_hook(lambda m, a: got.__setitem__(cur["r"], a[0].reshape(-1, a[0].shape[-1]).clone()))
    try:
        for r in rows:
            cur["r"] = r
            MU.run_layer(model, L0, cache[r:r + 1])
    finally:
        h.remove()
    return got


@pytest.fixture
def setup(cache_dir):
    model, _ = tiny_model()
    cache = load_layer_cache(cache_dir, L0)
    rows = list(range(0, 12))
    return model, cache, rows


# ---------------------------------------------------------------- forward equivalence / capture
def test_forced_forward_natural_is_bitwise_original(setup):
    model, cache, rows = setup
    blk = MU.moe_block(model, L0)
    x = torch.randn(1, 16, blk.gate.in_features).to(torch.bfloat16)
    ref = blk(x)
    with FRT.block_override(model, L0, lambda s: s):
        got = blk(x)
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
    assert "forward" not in blk.__dict__                               # removed afterwards


def test_capture_is_ordered_topk_and_matches_unmodified_profiler(setup):
    model, cache, rows = setup
    sel, prob = _natural_table(model, cache, rows)
    pf = Profiler(model, L0, collect_hessian=False).run(cache, rows, routing_only=True)
    got = FRT.expert_seq_counts(sel, E)
    for r in rows:
        assert got[r] == {e: c for (_, e), c in pf.seq_counts[r].items()}
        assert sel[r].shape == (16, MU.top_k(model))
        assert bool((prob[r][:, :-1] >= prob[r][:, 1:]).all())       # ordered: descending probability


def test_natural_forced_hessians_bitwise_equal_unmodified_profiler(setup):
    model, cache, rows = setup
    sel, _ = _natural_table(model, cache, rows)
    ref, _ = _plain_hessians(model, cache, rows)
    got, fp = _forced_hessians(model, cache, rows, sel, {r: r for r in rows}, natural=True)
    for k in ref:
        assert torch.equal(ref[k][0], got[k][0]) and ref[k][1] == got[k][1], k
    assert fp.stats()["n_ordered_diff"] == 0


def test_expect_natural_raises_on_mismatch(setup):
    model, cache, rows = setup
    sel, _ = _natural_table(model, cache, rows)
    bad = {r: s.clone() for r, s in sel.items()}
    bad[rows[0]][0] = bad[rows[0]][0].flip(0)
    with pytest.raises(AssertionError):
        _forced_hessians(model, cache, rows, bad, {r: r for r in rows}, natural=True)
    assert "forward" not in MU.moe_block(model, L0).__dict__          # override removed even on failure


# ---------------------------------------------------------------- forced membership / Hessian semantics
def _random_table(rows, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {r: torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(16)]) for r in rows}


def test_forced_membership_and_manual_hessians(setup):
    model, cache, rows = setup
    k = MU.top_k(model)
    table = _random_table(rows, k)
    got, fp = _forced_hessians(model, cache, rows, table, {r: r for r in rows}, K=1)
    X = _block_inputs(model, cache, rows)
    blk = MU.moe_block(model, L0)
    for e in range(E):
        ex = blk.experts[e]
        Hgu = torch.zeros(X[rows[0]].shape[1], X[rows[0]].shape[1])
        Hd = None
        n = 0
        for r in rows:
            idx, top_x = torch.where(torch.nn.functional.one_hot(table[r], E).permute(2, 1, 0)[e])
            xs = X[r][top_x]
            n += xs.shape[0]
            Hgu += xs.float().T @ xs.float()
            z = ex.act_fn(ex.gate_proj(xs)) * ex.up_proj(xs)                   # ORIGINAL BF16 expert weights
            Hd = z.float().T @ z.float() if Hd is None else Hd + z.float().T @ z.float()
        assert got[(e, "gu")][1] == n == int(FRT.expert_token_counts(table, rows, E)[e])
        assert torch.allclose(got[(e, "gu")][0], Hgu, rtol=1e-5, atol=1e-4)
        assert torch.allclose(got[(e, "down")][0], Hd, rtol=1e-5, atol=1e-4)
    assert fp.stats()["n_set_diff"] > 0


def test_router_weights_do_not_enter_hessians(setup):
    """Same forced assignments, different router (gate) weights -> identical Hessians: the routing weights only scale
    the expert output, exactly as in the original block and Profiler."""
    model, cache, rows = setup
    table = _random_table(rows, MU.top_k(model), seed=3)
    a, _ = _forced_hessians(model, cache, rows, table, {r: r for r in rows})
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        MU.moe_block(m2, L0).gate.weight.mul_(-3.0)
    b, _ = _forced_hessians(m2, cache, rows, table, {r: r for r in rows})
    for kk in a:
        assert torch.equal(a[kk][0], b[kk][0])


def test_alignment_by_sequence_id_not_row(setup, tmp_path):
    """A cache stored in a different row order + row->sid map gives bitwise the same Hessians (positions aligned)."""
    model, cache, rows = setup
    sel, _ = _natural_table(model, cache, rows)
    perm = rows[::-1]
    pc = cache[perm].clone()
    ref, _ = _forced_hessians(model, cache, rows, sel, {r: r for r in rows}, natural=True)
    chosen_rows = [perm.index(r) for r in rows]                          # same chosen ORDER, permuted storage
    got, _ = _forced_hessians(model, pc, chosen_rows, sel, dict(enumerate(perm)), natural=True)
    for kk in ref:
        assert torch.equal(ref[kk][0], got[kk][0])


def test_no_cross_sequence_leakage(setup):
    model, cache, rows = setup
    table = _random_table(rows + [20, 21], MU.top_k(model), seed=5)
    a, _ = _forced_hessians(model, cache, rows, table, {r: r for r in rows})
    t2 = dict(table)
    t2[20], t2[21] = table[21], table[20]                              # unchosen sequences: must not matter
    b, _ = _forced_hessians(model, cache, rows, t2, {r: r for r in rows})
    for kk in a:
        assert torch.equal(a[kk][0], b[kk][0])
    t3 = dict(table)
    t3[rows[0]], t3[rows[1]] = table[rows[1]], table[rows[0]]           # swapping chosen sequences' routing: changes
    c, _ = _forced_hessians(model, cache, rows, t3, {r: r for r in rows})
    assert any(not torch.equal(a[kk][0], c[kk][0]) for kk in a)


def test_cache_router_and_inference_untouched(setup):
    model, cache, rows = setup
    before_cache = _sha(cache[rows].clone())
    gate_before = MU.moe_block(model, L0).gate.weight.clone()
    blk = MU.moe_block(model, L0)
    x = cache[rows[0]:rows[0] + 1].clone()
    out_before = MU.run_layer(model, L0, x)
    _forced_hessians(model, cache, rows, _random_table(rows, MU.top_k(model)), {r: r for r in rows})
    assert _sha(cache[rows].clone()) == before_cache
    assert torch.equal(gate_before, blk.gate.weight)
    FRT.assert_no_override(model, [L0])
    assert torch.equal(out_before, MU.run_layer(model, L0, x))          # inference uses the class forward


# ---------------------------------------------------------------- E3 integration (reconstruction universe, GPTQ)
def _e3(model, cache_dir, chosen, table=None, natural=False, cap=None):
    import e3_allocation as E3
    test = split_ids(load_meta(cache_dir))["test"]
    if table is None:
        return PQ.quantize_install(model, cache_dir, [L0], chosen, test, 2, 4, GS, 1.0)
    full = {L0: table}
    with FRT.forced_e3(E3, full, {r: r for r in chosen}, natural, on_hessian=cap):
        q = PQ.quantize_install(model, cache_dir, [L0], chosen, test, 2, 4, GS, 1.0)
    assert E3.Profiler is Profiler                                    # restored
    return q


def test_natural_forced_e3_weight_hash_and_recon_regression(cache_dir):
    chosen = list(range(10))
    m1, _ = tiny_model()
    q1 = _e3(m1, cache_dir, chosen)
    m2, _ = tiny_model()
    sel, _ = FRT.capture_routing(m2, L0, load_layer_cache(cache_dir, L0), chosen, chosen)
    seen = []
    q2 = _e3(m2, cache_dir, chosen, sel, natural=True, cap=lambda L, e, acc: seen.append((L, e)))
    assert PQ.installed_digest(m1, [L0]) == PQ.installed_digest(m2, [L0])
    assert [r["mean"] for r in q1["results"]] == [r["mean"] for r in q2["results"]]
    assert q1["counts"]["eval_universe"] == q2["counts"]["eval_universe"]
    assert len(seen) == E


def test_forced_e3_reconstruction_universe_uses_natural_routing(cache_dir):
    chosen = list(range(10))
    m1, _ = tiny_model()
    q1 = _e3(m1, cache_dir, chosen)
    m2, _ = tiny_model()
    q2 = _e3(m2, cache_dir, chosen, _random_table(chosen, MU.top_k(m2), seed=9))
    assert q1["counts"]["eval_universe"] == q2["counts"]["eval_universe"]     # same held-out activations / universe
    assert PQ.installed_digest(m1, [L0]) != PQ.installed_digest(m2, [L0])     # but different Hessians -> weights


def test_gptq_settings_preserved_and_no_cross_run_contamination(cache_dir):
    import e3_allocation as E3
    chosen = list(range(10))
    model, _ = tiny_model()
    snap = PQ.snapshot_experts(model, [L0])
    table = _random_table(chosen, MU.top_k(model), seed=11)
    calls = []
    orig = E3.quantize_from_acc_or_fallback

    def spy(model_, acc, L, e, *a, **k):
        calls.append((a, k))
        return orig(model_, acc, L, e, *a, **k)
    E3.quantize_from_acc_or_fallback = spy
    try:
        _e3(model, cache_dir, chosen, table)
    finally:
        E3.quantize_from_acc_or_fallback = orig
    assert all(a == (4, GS, 1.0) for a, _ in calls)                  # bits, group size, percdamp unchanged
    d1 = PQ.installed_digest(model, [L0])
    PQ.restore_experts(model, snap)
    assert PQ.experts_match(model, snap) == (E * 3, E * 3)
    _e3(model, cache_dir, chosen, table)
    assert PQ.installed_digest(model, [L0]) == d1


# ---------------------------------------------------------------- metrics
def test_actorder_perm_matches_solver_and_group_change():
    H = torch.diag(torch.tensor([3., 0., 5., 0.5]))
    p = FRT.actorder_perm(H)
    assert p.tolist() == [2, 0, 1, 3]                                # dead column -> diag 1 (as in gptq.py)
    m = FRT.perm_metrics(torch.tensor([0, 1, 2, 3]), torch.tensor([1, 0, 2, 3]), group_size=2)
    assert m["frac_positions_changed"] == 0.5 and m["frac_columns_changed_group"] == 0.0
    m = FRT.perm_metrics(torch.tensor([0, 2, 1, 3]), torch.tensor([0, 1, 2, 3]), group_size=2)
    assert m["frac_columns_changed_group"] == 0.5


def test_hessian_metrics_scale_invariance():
    A = torch.randn(6, 6, dtype=torch.float64)
    H = A @ A.T
    m = FRT.hessian_metrics(3 * H, H, 30, 10)
    assert m["rel_fro_trace_normalized"] < 1e-12 and abs(m["trace_ratio"] - 3) < 1e-12
    assert abs(m["rel_fro_raw"] - 2) < 1e-12 and m["rel_fro_mean_normalized"] < 1e-12


def test_weight_diff():
    a = {"x": torch.tensor([1., 2., 3.]).to(torch.bfloat16), "y": torch.ones(2).to(torch.bfloat16)}
    b = {"x": torch.tensor([1., 2., 4.]).to(torch.bfloat16), "y": torch.ones(2).to(torch.bfloat16)}
    d = FRT.weight_diff(a, b)
    assert d["n_differ"] == 1 and d["n_tensors_identical"] == 1 and d["frac_differ"] == 0.2


# ---------------------------------------------------------------- factorial contrasts + synthetic example
def test_factorial_contrasts_and_interaction():
    c = FRT.factorial_contrasts(10.0, 11.0, 12.5, 14.0)
    assert c["A_effect_R0"] == 1.0 and c["R_effect_X0"] == 2.5 and c["A_effect_R1"] == 1.5 and c["R_effect_X1"] == 3.0
    assert c["interaction"] == 0.5 and c["natural_effect"] == 4.0
    assert c["A_effect_R1"] - c["A_effect_R0"] == c["interaction"] == c["R_effect_X1"] - c["R_effect_X0"]


def test_synthetic_factorial_activation_vs_routing(setup):
    """Activation change with fixed routing keeps membership (counts) but changes values; routing change with fixed
    activations changes membership; both change -> AA; the Hessian effects interact (non-additive)."""
    model, cache, rows = setup
    sel_o, _ = _natural_table(model, cache, rows)
    cache_a = (cache[:16].float() + 0.3 * torch.randn(16, *cache.shape[1:], generator=torch.Generator().manual_seed(2))
               ).to(torch.bfloat16)
    sel_a, _ = FRT.capture_routing(model, L0, cache_a, rows, rows)
    assert any(not torch.equal(sel_o[r], sel_a[r]) for r in rows)
    r2s = {r: r for r in rows}
    H = {"OO": _forced_hessians(model, cache, rows, sel_o, r2s)[0], "AO": _forced_hessians(model, cache_a, rows, sel_o, r2s)[0],
         "OA": _forced_hessians(model, cache, rows, sel_a, r2s)[0], "AA": _forced_hessians(model, cache_a, rows, sel_a, r2s)[0]}
    nat_aa, _ = _forced_hessians(model, cache_a, rows, sel_a, r2s, natural=True)
    for kk in H["AA"]:
        assert torch.equal(H["AA"][kk][0], nat_aa[kk][0])
        assert H["AO"][kk][1] == H["OO"][kk][1] and H["OA"][kk][1] == H["AA"][kk][1]
    tot = {c: sum(float(H[c][(e, "gu")][0].double().norm()) for e in range(E)) for c in H}
    fc = FRT.factorial_contrasts(tot["OO"], tot["AO"], tot["OA"], tot["AA"])
    assert abs(fc["interaction"]) > 0


# ---------------------------------------------------------------- frozen decision rule
def _f(**kw):
    base = dict(local_suppression=False, local_routing_dominant=False, ppl_suppression=False,
                activation_dominant=False, interaction_large=False)
    base.update(kw)
    return base


def test_flags_and_decide():
    tau = FRT.TAU
    y = dict(OO=9.0, AO=9.0005, OA=9.009, AA=9.010)
    f = FRT.flags(dict(AO=0.01, OA=0.05, AA=0.06), dict(AO=0.01, OA=0.08, AA=0.09), y)
    assert f["dY"]["AO"] == 0.0                                       # below TAU -> 0
    assert f["local_suppression"] and f["local_routing_dominant"] and f["ppl_suppression"] and not f["activation_dominant"]
    d = dict(OO=0.010, AO=0.009, OA=-0.003, AA=-0.004)
    assert FRT.ranking_preserved(d)
    assert FRT.decide({"FR": f, "FC": f}, d) == "FROZEN-ROUTING-SUPPRESSES-SENSITIVITY"
    assert FRT.decide({"FR": f, "FC": f}, d, blocked=True) == "RUN-BLOCKED"
    act = _f(activation_dominant=True)
    assert FRT.decide({"FR": act, "FC": act}, d) == "ACTIVATION-VALUES-DOMINATE"
    loc = _f(local_suppression=True)
    assert FRT.decide({"FR": loc, "FC": loc}, d) == "LOCAL-ROUTING-EFFECT-WITHOUT-PPL-EXPLANATION"
    sup_int = _f(local_suppression=True, ppl_suppression=True, interaction_large=True)
    assert FRT.decide({"FR": sup_int, "FC": sup_int}, d) == "ROUTING-AND-ACTIVATION-INTERACT"
    assert FRT.decide({"FR": _f(), "FC": _f()}, d) == "ROUTING-AND-ACTIVATION-INTERACT"
    sup = _f(local_suppression=True, ppl_suppression=True)
    assert FRT.decide({"FR": sup, "FC": sup}, dict(OO=0.01, AO=-0.004, OA=0, AA=-0.004)) == \
        "LOCAL-ROUTING-EFFECT-WITHOUT-PPL-EXPLANATION"                 # ranking not preserved
    assert abs(tau - 0.000923709) < 1e-9


def test_conditions_and_diag_subset_prespecified():
    assert FRT.CONDITIONS == {"OO": ("O", "O"), "AO": ("A", "O"), "OA": ("O", "A"), "AA": ("A", "A")}
    assert len(FRT.DIAG_KEYS) == 16
    sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
    import eval_partial_ppl as EP
    assert [(L, e) for L in FRT.DIAG_LAYERS for e in FRT.DIAG_EXPERTS] == EP.CHECK_EXPERTS
