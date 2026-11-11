"""CPU tests for the FP32 Hessian accumulation-order null experiment (tiny Qwen2-MoE from conftest, synthetic
matrices, and the frozen artifacts when present)."""
import inspect
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import E, LAYERS, tiny_model
from moe_calib import actorder_ablation as AA
from moe_calib import frozen_routing as FRT
from moe_calib import gptq as GQ
from moe_calib import hessian_order_null as HN
from moe_calib import model_utils as MU
from moe_calib import partial_model_quant as PQ
from moe_calib.data_utils import load_layer_cache, load_meta, split_ids
from moe_calib.hessian import BlockHessianAccumulator
from moe_calib.profiler import Profiler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
OUT = os.path.join(ROOT, "runs_hessian_order_null")
GS = 16
L0 = LAYERS[0]
K = 2
ROWS = list(range(12))
CHOSEN = [7, 2, 11, 0, 5, 9, 1, 4, 10, 3, 8, 6]          # an unsorted replay order (like the allocation's `chosen`)


def _manifest():
    path = os.path.join(OUT, "input_manifest.json")
    if not os.path.exists(path):
        pytest.skip("input manifest not present")
    return json.load(open(path))


@pytest.fixture
def setup(cache_dir):
    model, _ = tiny_model()
    cache = load_layer_cache(cache_dir, L0)
    sel, _ = FRT.capture_routing(model, L0, cache, ROWS, ROWS)
    return model, cache, sel


def _ordered(model, cache, sel, seed, record=None, fp_keys=(), chosen=CHOSEN):
    order, blocks, _ = HN.order_spec(chosen, seed, K)
    pf = HN.OrderedForcedProfiler(model, L0, sel, {s: s for s in chosen}, True, chosen, order, blocks,
                                  record=record, fp_keys=fp_keys, K=K)
    pf.run(cache, chosen, weights=[1.0] * len(chosen))
    return pf


def _H(pf):
    return {(e, p): pf.acc.H_sum((L0, e, p), list(range(K))) for e in range(E) for p in ("gu", "down")}


def _e3(model, cache_dir, chosen, seed=None, perms=None, log=None, ordered=True):
    import e3_allocation as E3
    import actorder_ablation as XA
    test = split_ids(load_meta(cache_dir))["test"]
    cache = load_layer_cache(cache_dir, L0)
    sel, _ = FRT.capture_routing(model, L0, cache, list(range(cache.shape[0])), list(range(cache.shape[0])))
    order, blocks, _ = HN.order_spec(chosen, seed, K)
    orig = E3.quantize_from_acc_or_fallback
    hs, infos_all = {}, {}

    def qw(m, acc, L, e, bits, gs, pd):
        infos = {}
        hs[(L, e)] = (FRT.tensor_sha(acc.H_sum((L, e, "gu"), list(range(acc.K)))), acc.tokens((L, e, "gu")))
        if perms is not None and acc.tokens((L, e, "gu")) > 0:
            out = orig(m, acc, L, e, bits, gs, pd, perms=perms[(L, e)], infos=infos)
        else:
            out = orig(m, acc, L, e, bits, gs, pd)
        infos_all[(L, e)] = infos
        return out
    E3.quantize_from_acc_or_fallback = qw
    try:
        with XA.install_spy(log if log is not None else []):
            if ordered:
                with HN.ordered_forced_e3(E3, {L0: sel}, {s: s for s in chosen}, chosen, order, blocks):
                    q = PQ.quantize_install(model, cache_dir, [L0], chosen, test, K, 4, GS, 1.0)
            else:
                q = PQ.quantize_install(model, cache_dir, [L0], chosen, test, K, 4, GS, 1.0)
    finally:
        E3.quantize_from_acc_or_fallback = orig
    return q, hs, infos_all


# ---------------------------------------------------------------- 1-3 frozen inputs
def test_exact_frozen_selections():
    import hessian_order_null as XH
    assert XH.SEL_PREFIX == {"FR": "ec355fb8d866e4ff", "FC": "0353229b6d3090db"}
    man = _manifest()
    for c, v in man["selections"].items():
        assert v["ids_hash"] == XH.SEL_PREFIX[c] and v["n"] == v["unique"] == 128 == len(set(v["replay_order"]))
        assert v["selection_sha256"] == HN.selection_sha(v["replay_order"]) and v["K"] == 4


def test_original_activation_cache_references():
    import multigroup_robustness as MRD
    man = _manifest()
    mpr = json.load(open(os.path.join(ROOT, MRD.OUT, "protocol.json")))
    assert man["original_cache_files"] == {L: mpr["references"]["original_cache_files"][L] for L in man["original_cache_files"]}
    assert man["original_cache_dir"].endswith("moe_cache")


def test_original_routing_assignments():
    man = _manifest()
    r = json.load(open(os.path.join(ROOT, "runs_frozen_routing", "routing_original.json")))
    assert man["routing_original"]["file_sha256"] == r["file_sha256"]


# ---------------------------------------------------------------- 4-7 orders
def test_five_fixed_seeds():
    assert HN.ORDER_SEEDS == (20261121, 20261122, 20261123, 20261124, 20261125)
    assert set(HN.FP64_SEEDS) == {20261121, 20261122} <= set(HN.ORDER_SEEDS)


def test_deterministic_permutation_generation():
    for s in HN.ORDER_SEEDS:
        assert HN.index_permutation(s) == HN.index_permutation(s) == np.random.default_rng(s).permutation(128).tolist()
    assert len({tuple(HN.index_permutation(s)) for s in HN.ORDER_SEEDS}) == 5


def test_every_sequence_exactly_once_and_original_blocks_kept():
    chosen = list(np.random.default_rng(3).permutation(2000)[:128])
    base = HN.positional_blocks(128, 4)
    blk = dict(zip(chosen, base))
    for s in (None,) + HN.ORDER_SEEDS:
        order, blocks, perm = HN.order_spec(chosen, s, 4)
        assert sorted(order) == sorted(int(x) for x in chosen) and len(set(order)) == 128
        assert blocks == [blk[x] for x in order]                      # each sequence keeps its ORIGINAL block
        assert [sum(b == k for b in blocks) for k in range(4)] == [32] * 4
    order, blocks, perm = HN.order_spec(chosen, None, 4)
    assert order == [int(x) for x in chosen] and blocks == base     # identity = the Profiler's default
    wb = HN.within_block_orders(*HN.order_spec(chosen, 20261121, 4)[:2], 4)
    ob = HN.within_block_orders(order, base, 4)
    assert all(sorted(wb[k]) == sorted(ob[k]) and wb[k] != ob[k] for k in range(4))


def test_same_positional_permutation_policy_for_fr_and_fc():
    fr = list(range(1000, 1128))
    fc = list(range(5000, 5128))[::-1]
    for s in HN.ORDER_SEEDS:
        _, _, p1 = HN.order_spec(fr, s, 4)
        o2, _, p2 = HN.order_spec(fc, s, 4)
        assert p1 == p2 == HN.index_permutation(s) and o2 == [fc[j] for j in p2]


# ---------------------------------------------------------------- 8-13 per-sequence contributions, membership
def test_token_order_within_sequence_preserved(setup):
    model, cache, sel = setup
    keys = [(L0, e, p) for e in range(E) for p in ("gu", "down")]
    r0, r1 = {}, {}
    _ordered(model, cache, sel, None, r0, keys)
    _ordered(model, cache, sel, 20261121, r1, keys)
    # sha256 of each sequence's routed input X (rows in token / slot order) identical in both replay orders
    assert {k: v[0] for k, v in r0["contrib"].items()} == {k: v[0] for k, v in r1["contrib"].items()}
    # and equal to the unmodified Profiler's hook input for the same sequence
    got = {}
    h = MU.moe_block(model, L0).experts[0].gate_proj.register_forward_pre_hook(
        lambda m, a: got.__setitem__(cur["s"], FRT.tensor_sha(a[0].reshape(-1, a[0].shape[-1]))))
    cur = {}
    try:
        for s in CHOSEN:
            cur["s"] = s
            MU.run_layer(model, L0, cache[s:s + 1])
    finally:
        h.remove()
    for (key, s), v in r0["contrib"].items():
        if key == (L0, 0, "gu"):
            assert got[s] == v[0]


def test_per_sequence_contribution_alignment(setup):
    model, cache, sel = setup
    keys = [(L0, e, p) for e in range(E) for p in ("gu", "down")]
    r0, r1 = {}, {}
    _ordered(model, cache, sel, None, r0, keys)
    _ordered(model, cache, sel, 20261122, r1, keys)
    assert r0["contrib"] == r1["contrib"] and len(r0["contrib"]) > 0
    # the recorded contribution is exactly the accumulator's increment
    acc = BlockHessianAccumulator([0], 1, 8, 8, 1, device="cpu", store_device="cpu")
    X = torch.randn(13, 8).to(torch.bfloat16)
    acc.add((0, 0, "gu"), X, 0)
    Xf = X.float()
    assert torch.equal(acc.H[(0, 0, "gu")][0], Xf.T @ Xf)


def test_identical_expert_routed_token_counts(setup):
    model, cache, sel = setup
    a = _ordered(model, cache, sel, None)
    b = _ordered(model, cache, sel, 20261123)
    for e in range(E):
        for p in ("gu", "down"):
            assert a.acc.tokens((L0, e, p)) == b.acc.tokens((L0, e, p))
            assert torch.equal(a.acc.raw_n[(L0, e, p)], b.acc.raw_n[(L0, e, p)])
    exp = FRT.expert_token_counts(sel, CHOSEN, E)
    assert [a.acc.tokens((L0, e, "gu")) for e in range(E)] == exp.double().tolist()


def test_identical_block_and_topk_slot_membership(setup):
    model, cache, sel = setup
    r0, r1 = {}, {}
    _ordered(model, cache, sel, None, r0)
    _ordered(model, cache, sel, 20261124, r1)
    m0 = {k: sorted(v) for k, v in r0["members"].items()}
    m1 = {k: sorted(v) for k, v in r1["members"].items()}
    assert m0 == m1
    assert any(r0["members"][k] != r1["members"][k] for k in r0["members"])       # but the addition order differs
    # forced routing equals the natural top-k for every token (expect_natural) -> identical slot assignments
    bad = {s: v.clone() for s, v in sel.items()}
    bad[CHOSEN[0]][0] = bad[CHOSEN[0]][0].flip(0) if bad[CHOSEN[0]].shape[1] > 1 else bad[CHOSEN[0]][0]
    with pytest.raises(AssertionError):
        _ordered(model, cache, bad, 20261124)


def test_hessian_accumulation_dtype(setup):
    model, cache, sel = setup
    pf = _ordered(model, cache, sel, 20261121)
    assert all(v.dtype == torch.float32 for v in pf.acc.H.values())


def test_original_order_baseline_bitwise(setup):
    model, cache, sel = setup
    plain = Profiler(model, L0, K=K).run(cache, CHOSEN, weights=[1.0] * len(CHOSEN))
    forced = FRT.ForcedProfiler(model, L0, sel, {s: s for s in CHOSEN}, True, K=K).run(cache, CHOSEN, weights=[1.0] * len(CHOSEN))
    ordered = _ordered(model, cache, sel, None)
    for e in range(E):
        for p in ("gu", "down"):
            k = (L0, e, p)
            assert torch.equal(plain.acc.H[k], ordered.acc.H[k]) and torch.equal(forced.acc.H[k], ordered.acc.H[k])
    with pytest.raises(AssertionError):                       # the caller must pass the original replay order
        o, b, _ = HN.order_spec(CHOSEN, None, K)
        HN.OrderedForcedProfiler(model, L0, sel, {s: s for s in CHOSEN}, True, CHOSEN, o, b, K=K).run(cache, sorted(CHOSEN))


# ---------------------------------------------------------------- 14-16 FP32 / FP64 accumulation
def test_fp32_accumulation_order_perturbation_synthetic():
    g = torch.Generator().manual_seed(0)
    C = [(lambda X: X.T @ X)(torch.randn(50, 32, generator=g) * (1 + 10 * torch.rand(1, generator=g))) for _ in range(32)]
    def acc(order, dt):
        H = torch.zeros(32, 32, dtype=dt)
        for i in order:
            H += C[i].to(dt)
        return H
    perm = HN.index_permutation(20261121, 32)
    a32, b32 = acc(range(32), torch.float32), acc(perm, torch.float32)
    a64, b64 = acc(range(32), torch.float64), acc(perm, torch.float64)
    assert not torch.equal(a32, b32)                          # FP32 addition is not associative
    assert float((a32.double() - b32.double()).norm() / a32.double().norm()) < 1e-6
    d32 = float((a32.double() - b32.double()).norm() / a32.double().norm())
    d64 = float((a64 - b64).norm() / a64.norm())
    assert d64 < d32 * 1e-6
    m = HN.hdiff(b32, a32)
    assert m["n_diff"] > 0 and not m["bitwise_identical"] and m["max_ulp"] >= 1 and m["nonfinite"] == 0
    assert HN.hdiff(a32, a32)["n_diff"] == 0 and HN.hdiff(a32, a32)["max_ulp"] == 0


def test_fp64_accumulation_diagnostic(setup):
    model, cache, sel = setup
    keys = [(L0, e, p) for e in range(E) for p in ("gu", "down")]
    d = MU.expert_dims(model)
    for seed in (None, 20261121):
        ref = _ordered(model, cache, sel, seed)
        order, blocks, _ = HN.order_spec(CHOSEN, seed, K)
        pf = HN.OrderedForcedProfiler(model, L0, sel, {s: s for s in CHOSEN}, True, CHOSEN, order, blocks, K=K,
                                      collect_hessian=False)
        pf.acc = HN.Fp64Accumulator(keys, dict(gu=d[0], down=d[1]), K, "cpu")
        pf.run(cache, CHOSEN, weights=[1.0] * len(CHOSEN))
        for k in keys:
            H32, H64 = pf.acc.sums(k)
            assert torch.equal(H32, ref.acc.H_sum(k, list(range(K))))       # identical FP32 arithmetic
            assert H64.dtype == torch.float64 and torch.allclose(H64, H32.double(), rtol=1e-5, atol=1e-5)
    # ordered_int is monotone with unit steps
    x = torch.tensor([1.0, -1.0, 3.5e-3, -2.0e4], dtype=torch.float32)
    nx = torch.nextafter(x, torch.full_like(x, float("inf")))
    assert ((HN.ordered_int(nx) - HN.ordered_int(x)) == 1).all()
    assert int(HN.ordered_int(torch.tensor([0.0])) - HN.ordered_int(torch.tensor([-0.0]))) == 0


def test_hessian_normalization_unchanged(setup):
    model, cache, sel = setup
    pf = _ordered(model, cache, sel, 20261125)
    # unnormalized sum of x x^T over routed tokens (no division by token count, unit weights)
    X = []
    h = MU.moe_block(model, L0).experts[1].gate_proj.register_forward_pre_hook(lambda m, a: X.append(a[0].reshape(-1, a[0].shape[-1]).float()))
    try:
        for s in CHOSEN:
            MU.run_layer(model, L0, cache[s:s + 1])
    finally:
        h.remove()
    Xc = torch.cat(X)
    assert torch.allclose(pf.acc.H_sum((L0, 1, "gu"), list(range(K))), Xc.T @ Xc, rtol=1e-4, atol=1e-3)
    assert float(pf.acc.n[(L0, 1, "gu")].sum()) == Xc.shape[0]
    with pytest.raises(AssertionError):
        o, b, _ = HN.order_spec(CHOSEN, None, K)
        HN.OrderedForcedProfiler(model, L0, sel, {s: s for s in CHOSEN}, True, CHOSEN, o, b, K=K).run(
            cache, CHOSEN, weights=[2.0] * len(CHOSEN))


# ---------------------------------------------------------------- 17-22 GPTQ / installation / reconstruction
def _problem(rows=24, cols=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(rows, cols, generator=g)
    X = torch.randn(4 * cols, cols, generator=g) * torch.linspace(0.2, 3, cols)
    return W, X.T @ X


def test_frozen_actorder_permutation(cache_dir):
    model, _ = tiny_model()
    q, hs, infos = _e3(model, cache_dir, CHOSEN, None, perms=None)
    # capture natural perms, then freeze a foreign one: applied perm == frozen
    g = torch.Generator().manual_seed(5)
    live = [e for e in range(E) if hs[(L0, e)][1] > 0]
    d = MU.expert_dims(model)
    frozen = {(L0, e): {"gate_proj": torch.randperm(d[0], generator=g)} for e in range(E)}
    for e in range(E):
        frozen[(L0, e)]["up_proj"] = frozen[(L0, e)]["gate_proj"]
        frozen[(L0, e)]["down_proj"] = torch.randperm(d[1], generator=g)
    m2, _ = tiny_model()
    _, hs2, inf2 = _e3(m2, cache_dir, CHOSEN, 20261121, perms=frozen)
    assert all(torch.equal(inf2[(L0, e)][k]["perm"], frozen[(L0, e)][k]) for e in live for k in AA.PROJ)
    man_path = os.path.join(ROOT, "runs_actorder_ablation", "original_actorder_manifest.json")
    if os.path.exists(man_path):
        import actorder_ablation as XA
        P = XA.load_original_perms("FR")                      # verifies all permutation hashes
        assert len(P) == 720


def test_no_quantizer_grid_freezing_and_exact_codes():
    W, H = _problem()
    _, H2 = _problem(seed=1)
    perm = torch.argsort(torch.diag(H), descending=True)
    i1, i2 = {}, {}
    Q1 = GQ.gptq_quantize(W, H, 4, GS, 1.0, actorder_override=perm, info=i1)
    Q2 = GQ.gptq_quantize(W, H2, 4, GS, 1.0, actorder_override=perm, info=i2)
    assert torch.equal(i1["perm"], i2["perm"]) and not torch.equal(i1["scale"], i2["scale"])
    c1, ex1 = HN.codes_from(Q1, perm, i1["scale"], i1["zero"], GS)
    assert ex1 and c1.dtype == torch.uint8 and int(c1.max()) <= 15
    idx = torch.arange(W.shape[1]) // GS
    g = AA.group_of(perm, GS)
    S, Z = i1["scale"][g].T, i1["zero"][g].T
    assert torch.equal(S * (c1.float() - Z), Q1)
    assert "zero" not in inspect.signature(GQ.gptq_quantize).parameters and "scale" not in inspect.signature(GQ.gptq_quantize).parameters
    del idx


def _original_gptq():
    try:
        src = subprocess.run(["git", "show", "HEAD:moe_calib/gptq.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip("git HEAD solver unavailable")
    ns = {}
    exec(compile(src, "gptq_head.py", "exec"), ns)
    return ns["gptq_quantize"]


def test_gptq_default_behavior_unchanged():
    head = _original_gptq()
    for seed in range(3):
        W, H = _problem(seed=seed)
        assert torch.equal(GQ.gptq_quantize(W, H, 4, GS, 1.0), head(W, H, 4, GS, 1.0))
        perm = torch.argsort(torch.diag(H), descending=True)
        assert torch.equal(GQ.gptq_quantize(W, H, 4, GS, 1.0, actorder_override=perm), head(W, H, 4, GS, 1.0))


def test_installed_weight_integrity(cache_dir):
    model, _ = tiny_model()
    log = []
    _e3(model, cache_dir, CHOSEN, 20261122, log=log)
    assert len(log) == 3 * E and all(x[1] and x[2] and x[4] for x in log)


def test_no_inference_time_routing_override(cache_dir):
    import e3_allocation as E3
    model, _ = tiny_model()
    p0, q0 = E3.Profiler, E3.quantize_from_acc_or_fallback
    r0 = MU.moe_block(model, L0).gate.weight.clone()
    _e3(model, cache_dir, CHOSEN, 20261123)
    FRT.assert_no_override(model)
    assert E3.Profiler is p0 and E3.quantize_from_acc_or_fallback is q0
    assert torch.equal(MU.moe_block(model, L0).gate.weight, r0)


def test_frozen_reconstruction_universe_and_identity_order_regression(cache_dir):
    m0, _ = tiny_model()
    q0, h0, _ = _e3(m0, cache_dir, CHOSEN, ordered=False)            # the unmodified E3 path
    m1, _ = tiny_model()
    q1, h1, _ = _e3(m1, cache_dir, CHOSEN, None)                     # new pathway, original order
    assert h1 == h0 and PQ.installed_digest(m1, [L0]) == PQ.installed_digest(m0, [L0])
    assert [r["mean"] for r in q1["results"]] == [r["mean"] for r in q0["results"]]
    m2, _ = tiny_model()
    q2, h2, _ = _e3(m2, cache_dir, CHOSEN, 20261121)
    assert q2["counts"]["eval_universe"] == q0["counts"]["eval_universe"]
    assert {k: v[1] for k, v in h2.items()} == {k: v[1] for k, v in h0.items()}     # same token counts


# ---------------------------------------------------------------- 23-28 evaluation, metrics, decision
def test_frozen_ppl_windows():
    path = os.path.join(ROOT, "runs_partial_ppl", "evaluation_protocol.json")
    if not os.path.exists(path):
        pytest.skip("evaluation protocol not present")
    pr = json.load(open(path))
    assert pr["scored_targets_total"] == 65408 and pr["n_windows"] == 128 and pr["context_length"] == 512
    import eval_partial_ppl as EP
    assert EP.BATCH == 8 and EP.SEQLEN == 512 and EP.N_WIN == 128


def test_paired_strategy_differences():
    fr, fc = [9.26, 9.27, 9.265, 9.268, 9.262], [9.27, 9.275, 9.26, 9.28, 9.271]
    r = HN.paired(fr, fc, 9.267851, 9.277908)
    assert np.allclose(r["delta"], np.array(fc) - np.array(fr))
    assert r["sign_reversals"] == 1 and abs(r["delta_oo"] - 0.010057) < 1e-9
    assert r["max_abs_dev_fr"] == pytest.approx(max(abs(x - 9.267851) for x in fr))


def test_sd_computation():
    x = [1.0, 2.0, 4.0]
    assert HN.sd(x) == pytest.approx(np.std(x, ddof=1))
    assert np.isnan(HN.sd([1.0]))


def test_materiality_threshold():
    assert HN.TAU_PPL == pytest.approx(0.000923709, abs=1e-9)
    assert HN.decide(False, True, False, [HN.TAU_PPL]) == "ORDER-SENSITIVE-WEIGHTS-ONLY"   # <= tau is not material
    assert HN.decide(False, True, False, [HN.TAU_PPL * 1.0001]) == "ORDER-SENSITIVE-PPL"


def test_early_stop_behavior(monkeypatch, tmp_path):
    import hessian_order_null as XH
    # isolate from the study's real results/ (once they exist, stage_quant returns "done" before the gate)
    monkeypatch.setattr(XH, "p", lambda *a: os.path.join(str(tmp_path), *a))
    monkeypatch.setattr(XH, "protocol", lambda: {})
    monkeypatch.setattr(XH, "load_json", lambda path: dict(checks_passed=True, early_stop=True, repeat_config=None)
                        if path.endswith("hessian_differences.json") else None)
    a = SimpleNamespace(strategy="FR", order=20261121, repeat=False)
    with pytest.raises(SystemExit, match="NULL-ORDER-INVARIANT"):
        XH.stage_quant(a)
    monkeypatch.setattr(XH, "load_json", lambda path: dict(checks_passed=False, early_stop=False) if path.endswith("hessian_differences.json") else None)
    with pytest.raises(AssertionError):
        XH.stage_quant(a)
    assert HN.decide(False, False, True, []) == "NULL-ORDER-INVARIANT"
    sh = open(os.path.join(ROOT, "scripts", "run_hessian_order_null.sh")).read() if os.path.exists(
        os.path.join(ROOT, "scripts", "run_hessian_order_null.sh")) else ""
    assert sh == "" or "early_stop" in sh


def test_decision_classification():
    assert HN.decide(True, True, False, [1.0]) == "RUN-BLOCKED"
    assert HN.decide(True, False, True, []) == "RUN-BLOCKED"
    assert HN.decide(False, True, True, [0.0] * 10) == "ORDER-SENSITIVE-HESSIAN-ONLY"
    assert HN.decide(False, True, False, [0.0005] * 10) == "ORDER-SENSITIVE-WEIGHTS-ONLY"
    assert HN.decide(False, True, False, [0.0005] * 9 + [0.002]) == "ORDER-SENSITIVE-PPL"
    hd = {(c, s): False for c in ("FR", "FC") for s in HN.ORDER_SEEDS}
    assert HN.repeat_config(hd) is None
    hd[("FC", 20261122)] = True
    assert HN.repeat_config(hd) == ("FC", 20261122)
    hd[("FR", 20261125)] = True
    assert HN.repeat_config(hd) == ("FR", 20261125)
    assert len(HN.DECISION_RULE["precedence"]) == 5


# ---------------------------------------------------------------- 29-30 contamination, backward compatibility
def test_no_cross_run_weight_contamination(cache_dir):
    import hessian_order_null as XH
    m1, _ = tiny_model()
    _e3(m1, cache_dir, CHOSEN, 20261124)
    d1 = PQ.installed_digest(m1, [L0])
    m2, _ = tiny_model()
    _e3(m2, cache_dir, CHOSEN, 20261124)
    assert PQ.installed_digest(m2, [L0]) == d1                        # fresh model + same order -> identical
    names = {XH.cname(c, s, r) for c in ("FR", "FC") for s in (None,) + HN.ORDER_SEEDS for r in (False, True)}
    assert len(names) == 24                                           # distinct output dirs per configuration
    assert XH.HCACHE.rstrip("/").endswith("moe_cache_hessian_order_null") and XH.OUT == "runs_hessian_order_null"


def test_backward_compatibility_with_original_gptq_and_aof():
    from moe_calib import quant_eval as QE
    sig = inspect.signature(GQ.gptq_quantize).parameters
    assert (sig["bits"].default, sig["group_size"].default, sig["percdamp"].default, sig["actorder"].default,
            sig["actorder_override"].default) == (4, 128, 0.01, True, None)
    assert inspect.signature(Profiler.run).parameters["blocks"].default is None
    assert HN.positional_blocks(128, 4) == [(i * 4) // 128 for i in range(128)]
    for f in (QE.quantize_from_H, QE.quantize_from_acc, QE.quantize_from_acc_or_fallback):
        assert inspect.signature(f).parameters["perms"].default is None
    if os.path.exists(os.path.join(ROOT, "runs_actorder_ablation", "protocol.json")):
        import actorder_ablation as XA
        XA.protocol()                                                 # act-order study code / protocol unchanged
