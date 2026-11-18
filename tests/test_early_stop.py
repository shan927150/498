"""Hessian early-stopping MVP tests (tiny CPU Qwen2-MoE from conftest + synthetic data + frozen artifacts if present)."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest
import torch

from conftest import E, LAYERS, tiny_model
from moe_calib import early_stop as ES
from moe_calib import gptq as GQ
from moe_calib import model_utils as MU
from moe_calib.profiler import Profiler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
L0 = LAYERS[0]


def test_chronological_prefixes():                                                    # 1
    order = [7, 3, 9, 1, 0, 5, 2, 8]
    pre = ES.prefixes(order, (2, 4, 8))
    assert pre[2] == [7, 3] and pre[4] == [7, 3, 9, 1] and pre[8] == order
    with pytest.raises(AssertionError):
        ES.prefixes([1, 1, 2, 3], (2, 4))
    # production contiguous blocks are prefixes, not hashed
    b = ES.contiguous_blocks(128, 4)
    assert b[:32] == [0] * 32 and b[32:64] == [1] * 32 and b[127] == 3


def _snapshots(tm, cache, order, K=4):
    """Profiler replay with a mid-block S(n/8) snapshot, exactly as the MVP stage does."""
    n = len(order)
    blocks = ES.contiguous_blocks(n, K)
    pr = Profiler(tm, L0, K=K)
    q = n // 8
    pr.run(cache, order[:q], blocks=blocks[:q])
    snap = {e: pr.acc.H[(L0, e, "gu")][0].clone() for e in range(E)}
    nq = {e: float(pr.acc.raw_n[(L0, e, "gu")][0]) for e in range(E)}
    pr.run(cache, order[q:], blocks=blocks[q:])
    return pr, snap, nq


def test_cumulative_snapshots_and_counts(cache_dir):                                 # 2, 3, 12
    from moe_calib.data_utils import load_layer_cache
    tm, _ = tiny_model()
    cache = load_layer_cache(cache_dir, L0)
    order = [5, 2, 9, 0, 7, 3, 1, 8, 4, 6, 11, 10, 13, 12, 15, 14]
    pr, snap, nq = _snapshots(tm, cache, order)
    full = Profiler(tm, L0, K=4).run(cache, order)                                   # production single call
    pre8 = Profiler(tm, L0, K=1).run(cache, order[:8])
    pre2 = Profiler(tm, L0, K=1).run(cache, order[:2])
    for e in range(E):
        k = (L0, e, "gu")
        assert torch.equal(pr.acc.H_sum(k, [0, 1, 2, 3]), full.acc.H_sum(k, [0, 1, 2, 3]))   # split replay == production
        assert torch.allclose(pr.acc.H_sum(k, [0, 1]), pre8.acc.H_sum(k, [0]), rtol=1e-5, atol=1e-4)
        assert torch.allclose(snap[e], pre2.acc.H_sum(k, [0]), rtol=1e-5, atol=1e-4)
        assert nq[e] == pre2.acc.tokens(k)
        assert pr.acc.tokens(k, [0, 1]) == pre8.acc.tokens(k)
        tot = pr.acc.H_sum(k, [0, 1]) + pr.acc.H_sum(k, [2, 3])
        assert torch.allclose(tot, pr.acc.H_sum(k, [0, 1, 2, 3]), rtol=1e-5, atol=1e-4)
        assert pr.acc.tokens((L0, e, "down")) == pr.acc.tokens(k)


def test_drift_definitions():                                                         # 5, 6, 7
    g = torch.Generator().manual_seed(0)
    X = torch.randn(50, 6, generator=g)
    Ha, Hb = X[:25].T @ X[:25], X.T @ X
    tr = ES.trace_drift(Hb, 50, Ha, 25)
    assert tr == pytest.approx(abs(float(torch.trace(Hb)) / 50 - float(torch.trace(Ha)) / 25) /
                               abs(float(torch.trace(Hb)) / 50), rel=1e-6)
    fr = ES.frob_drift(Hb, 50, Ha, 25)
    A, B = Hb.double() / 50, Ha.double() / 25
    assert fr == pytest.approx(float(torch.linalg.norm(A - B) / torch.linalg.norm(A)), rel=1e-9)
    assert ES.frob_drift(2 * Hb, 100, Hb, 50) == pytest.approx(0, abs=1e-12)       # scale-normalized
    assert np.isnan(ES.trace_drift(Hb, 50, Ha, 0)) and np.isnan(ES.frob_drift(Hb, 0, Ha, 25))
    s = {"gu": dict(trace=np.nan, frob=np.nan), "down": dict(trace=0.0, frob=0.0), "n": 0}
    assert not ES.rule_ok(s, "frob", dict(tau_frob=1.0))                             # zero exposure never stops


def test_stop_assignment_no_leakage():                                               # 4, 13
    keys = ["a", "b", "c"]
    mk = lambda f: {"gu": dict(trace=f, frob=f), "down": dict(trace=f, frob=f), "n": 100}
    st = {32: {"a": mk(0.01), "b": mk(0.5), "c": mk(0.5)}, 64: {"a": mk(0.9), "b": mk(0.01), "c": mk(0.5)}}
    thr = {32: dict(tau_frob=0.1, tau_trace=0.1, n_min=0), 64: dict(tau_frob=0.1, tau_trace=0.1, n_min=0)}
    stop = ES.assign_stop_budgets(st, "frob", thr)
    assert stop == {"a": 32, "b": 64, "c": 128}               # 'a' stays stopped at 32 although B64 stats look bad
    # decisions at B=32 only see st[32]: altering st[64] never changes B=32 stops
    st2 = {32: st[32], 64: {k: mk(0.0) for k in keys}}
    assert ES.assign_stop_budgets(st2, "frob", thr)["a"] == 32


def test_false_safe_and_workload():                                                  # 14
    err_full = np.array([1.0, 1.0, 1.0, 1.0])
    err_stop = np.array([1.005, 1.02, 1.0, 1.3])
    fs = ES.false_safe(err_stop, err_full, [True, True, False, False], 0.01)
    assert fs == dict(n_stopped=2, n_false=1, rate=0.5)
    tok = {32: {"a": 10, "b": 20}, 64: {"a": 20, "b": 40}, 128: {"a": 40, "b": 80}}
    w = ES.workload(tok, {"a": 32, "b": 128}, 2, 1)
    assert w["work_fraction_done"] == pytest.approx((10 + 80) / 120)
    assert w["token_expert_pairs_done"] == 90 and w["mean_effective_budget"] == 80


def test_projection_is_labelled():                                                   # 15
    r = ES.project_time(dict(forward_s=10, hook_overhead_s=2, accumulation_s=4, solve_s=50), 0.5)
    assert r["kind"].startswith("PROJECTED") and r["replay_s"] == 14 and r["total_s"] == 64


def test_classify_precedence():
    S = lambda a, b: dict(any_rule_passes=a, any_passing_rule_beats_frontier=b)
    assert ES.classify(S(False, False), True) == "EARLY-STOP-SIGNAL-FAIL"
    assert ES.classify(S(True, False), True) == "EARLY-STOP-NO-ADVANTAGE"
    assert ES.classify(S(True, True), False) == "EARLY-STOP-SYSTEMS-FAIL"
    assert ES.classify(S(True, True), True) == "EARLY-STOP-PROMISING"
    assert ES.classify(S(True, True), True, blocked=True) == "EARLY-STOP-BLOCKED"
    assert ES.frontier_interp([(0.75, 0.1), (0.5, 0.04), (0, 0)], 0.25) == pytest.approx(0.02)


def test_masked_replay_forward_identical_and_skips(cache_dir):                       # 8, 9, 10
    import hessian_early_stop_mvp as HM
    from moe_calib.data_utils import load_layer_cache
    tm, _ = tiny_model()
    X = load_layer_cache(cache_dir, L0)[:8].contiguous()
    old = HM.STOP_AFTER
    HM.STOP_AFTER = 4
    try:
        outs, res = {}, {}
        for c in ("forward", "noacc", "full", "stop50", "stop75"):
            r = HM.replay(tm, L0, X, c, out_hash=True)
            r, pr = (r, None) if isinstance(r, dict) else r
            outs[c] = r["digest"]; res[c] = (r, pr)
        assert len(set(outs.values())) == 1                                          # forward never changes
        full, pf = res["full"]; s50, p50 = res["stop50"]
        assert full["rows_skipped"] == 0 and s50["rows_skipped"] > 0
        assert full["rows_done"] == s50["rows_done"] + s50["rows_skipped"]
        for e in range(E):
            k = (L0, e, "gu")
            if HM.MASKS["stop50"](e):
                assert pf.acc.H_sum(k, [0, 1, 2, 3]) is not None
                assert torch.equal(p50.acc.raw_n[k][2:], torch.zeros(2, dtype=torch.float64))  # no updates after stop
            else:
                assert torch.equal(p50.acc.H[k], pf.acc.H[k])                       # active experts untouched
        assert res["noacc"][0]["rows_done"] == 0
    finally:
        HM.STOP_AFTER = old


def test_gptq_unchanged():                                                            # 11, 16
    assert hashlib.sha256(open(os.path.join(ROOT, "moe_calib/gptq.py"), "rb").read()).hexdigest() == \
        "58e40d1c0e9f2d4fd39d22e4a12b83c9828763a37be25eea544f9db03d6fd39d"


def test_frozen_artifacts_if_present():                                               # 1, 16
    path = os.path.join(ROOT, "runs_hessian_early_stop_mvp", "checkpoint_manifest.json")
    if not os.path.exists(path):
        pytest.skip("not frozen")
    m = json.load(open(path))
    hist = json.load(open(os.path.join(ROOT, "runs_ebss_official_validation", "results", "FR_s0.json")))
    assert m["cache_rows"] == hist["cache_rows"]
    assert m["prefixes"]["64"] == m["cache_rows"][:64]
    v = os.path.join(ROOT, "runs_hessian_early_stop_mvp", "hessian_checkpoint_validation.json")
    if os.path.exists(v):
        assert json.load(open(v))["passed"]
