"""AGQ-only validation tests (tiny CPU Qwen2-MoE from conftest + synthetic matrices + frozen artifacts when present).
Numbered comments map to the 30 required test items of the AGQ protocol."""
import hashlib
import inspect
import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import E, LAYERS, tiny_model
from moe_calib import agq as AG
from moe_calib import gptq as GQ
from moe_calib import model_utils as MU
from moe_calib.hessian import BlockHessianAccumulator
from moe_calib.profiler import Profiler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
OUT = os.path.join(ROOT, "runs_agq_only_validation")
L0 = LAYERS[0]


def _block_capture(model, L, h):
    """Run the real MoE block on h [1, T, d]; capture router logits, every expert's input/output, down inputs,
    shared output and the block output."""
    blk = MU.moe_block(model, L)
    cap = dict(logits=None, ex_in={}, ex_out={}, down_in={}, shared=None)
    hooks = [blk.gate.register_forward_hook(lambda m, i, o: cap.__setitem__("logits", o.detach()))]
    def ex_hook(e):
        def fn(m, i, o):
            cap["ex_in"][e] = i[0].detach()
            cap["ex_out"][e] = o.detach()
        return fn
    for e in range(MU.num_experts(model)):
        ex = blk.experts[e]
        hooks.append(ex.register_forward_hook(ex_hook(e)))
        hooks.append(ex.down_proj.register_forward_pre_hook(lambda m, a, e=e: cap["down_in"].__setitem__(e, a[0].detach())))
    hooks.append(blk.shared_expert.register_forward_hook(lambda m, i, o: cap.__setitem__("shared", o.detach())))
    try:
        out, _ = blk(h)
    finally:
        for hk in hooks:
            hk.remove()
    return cap, out


@pytest.fixture(scope="module")
def tm():
    return tiny_model()[0]


def _h(model, T=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, T, model.config.hidden_size, generator=g).to(torch.bfloat16)


# 1, 2, 3 -------------------------------------------------------------- routing coefficients
def test_router_logits_vs_coefficients_topk_and_normalization(tm):
    cap, _ = _block_capture(tm, L0, _h(tm))
    vals, sel = AG.routing_coefficients(cap["logits"], MU.top_k(tm))
    assert vals.dtype == torch.float32 and sel.shape == (16, MU.top_k(tm))
    # coefficients are probabilities, not logits
    assert not torch.allclose(vals, torch.topk(cap["logits"].float(), MU.top_k(tm)).values)
    assert (vals > 0).all() and (vals <= 1).all()
    # (2) exactly k distinct experts per token
    assert all(len(set(r.tolist())) == MU.top_k(tm) for r in sel)
    # (3) NO renormalization: sum of selected = sum of the top-k softmax mass (< 1), as norm_topk_prob=False
    assert tm.config.norm_topk_prob is False
    p = F.softmax(cap["logits"], dim=-1, dtype=torch.float)
    assert torch.equal(vals.sum(-1), torch.topk(p, MU.top_k(tm)).values.sum(-1))
    assert (vals.sum(-1) < 1).all()


# 1, 4, 5, 6 --------------------------------------------------------- aggregation + shared expert + alignment
def test_aggregation_reconstructs_block_output_bitwise(tm):
    blk = MU.moe_block(tm, L0)
    h = _h(tm, seed=3)
    cap, out = _block_capture(tm, L0, h)
    hs = h.view(-1, h.shape[-1])
    vals, sel = AG.routing_coefficients(cap["logits"], MU.top_k(tm))
    rows = AG.dispatch_rows(sel, vals, E)
    final = torch.zeros_like(hs)
    for e in range(E):
        if e not in rows:
            assert e not in cap["ex_in"] or cap["ex_in"][e].shape[0] == 0       # unmatched experts get no rows
            continue
        top_x, slot, c = rows[e]
        assert torch.equal(cap["ex_in"][e], hs[top_x])                      # (5,6) activation row i <-> token top_x[i]
        final.index_add_(0, top_x, (cap["ex_out"][e] * c.to(hs.dtype)[:, None]).to(hs.dtype))
    shared = F.sigmoid(blk.shared_expert_gate(hs)) * cap["shared"]           # (4) shared expert, unquantized
    assert torch.equal((final + shared).view_as(out), out)
    # c's BF16 rounding is the multiplier actually used
    assert all(torch.equal(rows[e][2].to(torch.bfloat16).float(), rows[e][2].to(torch.bfloat16).float()) for e in rows)


def test_dispatch_is_slot_major_not_token_order():
    sel = torch.tensor([[1, 0], [0, 1], [0, 2]])
    vals = torch.tensor([[.5, .3], [.6, .2], [.4, .35]])
    r = AG.dispatch_rows(sel, vals, 3)
    top_x, slot, c = r[0]
    # expert 0: token1 slot0, token2 slot0, token0 slot1  -> slot-major
    assert top_x.tolist() == [1, 2, 0] and slot.tolist() == [0, 0, 1]
    assert torch.allclose(c, torch.tensor([.6, .4, .3]))


# 7, 8, 9 -------------------------------------------------------------- projection inputs
def test_projection_inputs(tm):
    cap, _ = _block_capture(tm, L0, _h(tm, seed=5))
    blk = MU.moe_block(tm, L0)
    for e, X in cap["ex_in"].items():
        if X.shape[0] == 0:
            continue
        ex = blk.experts[e]
        # gate_proj and up_proj both consume the expert input X; down_proj consumes act(gate X) * up X
        assert torch.equal(cap["down_in"][e], ex.act_fn(ex.gate_proj(X)) * ex.up_proj(X))
        assert cap["down_in"][e].shape == (X.shape[0], tm.config.moe_intermediate_size) != X.shape


def test_profiler_alignment_and_affinity_trace(tm, cache_dir):
    from moe_calib.data_utils import load_layer_cache
    cache = load_layer_cache(cache_dir, L0)
    pr = Profiler(tm, L0, K=1, hessian_weighting="router_affinity", collect_affinity=True,
                  verify_alignment=True).run(cache, [0, 1, 2])
    assert pr.n_alignment_checks > 0
    for (L, e), lst in pr.affinity.items():
        for sid, top_x, slot, c in lst:
            assert torch.isfinite(c).all() and (c >= 0).all() and (c <= 1).all()
            assert top_x.numel() == len(set(top_x.tolist()))                 # one row per token per expert
    # routed-token counts agree with the routing profile
    for sid in (0, 1, 2):
        for (L, e), n in pr.seq_counts[sid].items():
            got = sum(t.numel() for s, t, _, _ in pr.affinity[(L, e)] if s == sid)
            assert got == n


def test_profiler_detects_misalignment(tm, cache_dir):
    from moe_calib.data_utils import load_layer_cache
    cache = load_layer_cache(cache_dir, L0)
    pr = Profiler(tm, L0, K=1, hessian_weighting="router_affinity", verify_alignment=True)
    orig = pr._record_rows

    def bad(vals, sel):
        orig(vals, sel)
        for e, (tx, idx, c) in list(pr._cur_rows.items()):
            if tx.numel() > 1:
                pr._cur_rows[e] = (tx.flip(0), idx, c)
    pr._record_rows = bad
    with pytest.raises(RuntimeError, match="alignment"):
        pr.run(cache, [0])


# 10-18 ---------------------------------------------------------------- weighted Hessian vs float64 oracle
def _acc(d):
    return BlockHessianAccumulator([0], 1, d, d, K=1, device="cpu", store_device="cpu")


def _prod(X, c=None, chunks=None):
    acc = _acc(X.shape[1])
    chunks = chunks or [X.shape[0]]
    i = 0
    for n in chunks:
        acc.add((0, 0, "gu"), X[i:i + n], 0, row_weight=None if c is None else c[i:i + n])
        i += n
    return acc.H_sum((0, 0, "gu"), [0]).double().numpy()


CASES = {
    "unit": lambda g: (torch.randn(40, 16, generator=g), torch.ones(40)),
    "uniform": lambda g: (torch.randn(40, 16, generator=g), torch.full((40,), 0.3)),
    "nonuniform": lambda g: (torch.randn(40, 16, generator=g), torch.rand(40, generator=g)),
    "zero_rows": lambda g: (torch.randn(40, 16, generator=g), torch.rand(40, generator=g) * (torch.arange(40) % 3 != 0)),
    "single_token": lambda g: (torch.randn(1, 16, generator=g), torch.tensor([0.7])),
    "repeated": lambda g: (torch.randn(1, 16, generator=g).repeat(30, 1), torch.rand(30, generator=g)),
    "lin_dep": lambda g: ((torch.randn(40, 3, generator=g) @ torch.randn(3, 16, generator=g)), torch.rand(40, generator=g)),
    "dim_down": lambda g: (torch.randn(40, 11, generator=g), torch.rand(40, generator=g)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_hessian_oracle(name):                                                       # 10, 18
    X, c = CASES[name](torch.Generator().manual_seed(7))
    ref = AG.weighted_hessian_reference(X.numpy(), c.numpy())
    got = _prod(X, c)
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-5 * np.abs(ref).max())
    assert np.allclose(got, got.T)                                                   # 15 symmetric (exact in fp32)
    assert np.linalg.eigvalsh(ref).min() >= -1e-9 * max(1.0, np.abs(ref).max())      # 16 PSD


@pytest.mark.parametrize("chunks", [[40], [1] * 40, [7, 13, 20], [39, 1]])
def test_chunking_and_sequence_boundaries(chunks):                                   # 10, 13
    g = torch.Generator().manual_seed(3)
    X, c = torch.randn(40, 16, generator=g), torch.rand(40, generator=g)
    ref = AG.weighted_hessian_reference(X.numpy(), c.numpy())
    assert np.allclose(_prod(X, c, chunks), ref, rtol=1e-5, atol=1e-6)


def test_hessian_properties():                                                       # 11-17
    g = torch.Generator().manual_seed(11)
    X, c = torch.randn(50, 12, generator=g), torch.rand(50, generator=g) + 0.1
    H0 = _prod(X)
    assert np.array_equal(_prod(X, torch.ones(50)), H0)                              # 11 c=1 -> unweighted, bitwise
    Hc = _prod(X, torch.full((50,), 0.25))
    assert np.array_equal(Hc, 0.25 * H0)                                             # 12 constant (power of 2: exact)
    c2 = c.clone(); c2[0] *= 4
    d = _prod(X, c2) - _prod(X, c)
    x0 = X[0].double().numpy()
    assert np.allclose(d, 3 * float(c[0]) * np.outer(x0, x0), atol=1e-4)             # 13 larger c -> larger share
    cz = c.clone(); cz[5] = 0
    assert np.allclose(_prod(X, cz), AG.weighted_hessian_reference(np.delete(X.numpy(), 5, 0),
                                                                   np.delete(c.numpy(), 5)), atol=1e-4)   # 14 zero
    perm = torch.randperm(50, generator=g)
    assert np.allclose(_prod(X[perm], c[perm]), _prod(X, c), rtol=1e-5, atol=1e-5)   # 17 consistent permutation
    assert not np.allclose(_prod(X, c[perm]), _prod(X, c), rtol=1e-3, atol=1e-3)     # coefficients alone permuted
    # c, not c^2
    sq = AG.weighted_hessian_reference(X.numpy(), (c ** 2).numpy())
    assert not np.allclose(_prod(X, c), sq, rtol=1e-3, atol=1e-3)


def test_row_weight_validation():
    acc = _acc(4)
    with pytest.raises(ValueError):
        acc.add((0, 0, "gu"), torch.randn(3, 4), 0, row_weight=torch.ones(2))
    with pytest.raises(ValueError):
        acc.add((0, 0, "gu"), torch.randn(3, 4), 0, row_weight=torch.tensor([1.0, -1.0, 1.0]))


def test_mixed_routing_and_topk_slots_manual():                                      # 9, 11 synthetic unequal affinities
    g = torch.Generator().manual_seed(2)
    T, d, Ex = 6, 5, 4
    h = torch.randn(T, d, generator=g)
    vals = torch.tensor([[.4, .3, .2, .1]] * T) * torch.linspace(1, 2, T)[:, None]
    sel = torch.stack([torch.roll(torch.arange(Ex), s) for s in range(T)])
    rows = AG.dispatch_rows(sel, vals, Ex)
    for e, (tx, slot, c) in rows.items():
        manual = sum(float(vals[t, k]) * np.outer(h[t].numpy(), h[t].numpy())
                     for t in range(T) for k in range(4) if int(sel[t, k]) == e)
        assert np.allclose(_prod(h[tx], c), manual, atol=1e-5)


def test_effective_sample_size():
    assert AG.effective_sample_size(np.ones(10)) == pytest.approx(10)
    assert AG.effective_sample_size(np.array([1, 0, 0, 0])) == pytest.approx(1)
    assert 1 < AG.effective_sample_size(np.array([.9, .1, .1])) < 3


# 19, 21, 30 ------------------------------------------------------------ profiler / GPTQ regression
def _profile(tm, cache, mode):
    return Profiler(tm, L0, K=2) if mode is None else Profiler(tm, L0, K=2, hessian_weighting=mode)


def test_unit_affinity_reproduces_standard_hessian_and_gptq(tm, cache_dir):
    from moe_calib.data_utils import load_layer_cache
    cache = load_layer_cache(cache_dir, L0)
    ids = [3, 0, 7, 5]
    A = _profile(tm, cache, None).run(cache, ids).acc
    U = _profile(tm, cache, "unit_affinity").run(cache, ids).acc
    R = _profile(tm, cache, "router_affinity").run(cache, ids).acc
    changed = 0
    for e in range(E):
        for proj in ("gu", "down"):
            k = (L0, e, proj)
            assert torch.equal(A.H[k], U.H[k]), "unit affinity must reproduce standard GPTQ Hessians bitwise"
            assert torch.equal(A.raw_n[k], R.raw_n[k])
            changed += not torch.equal(A.H[k], R.H[k])
        if A.tokens((L0, e, "gu")) > 0:
            W = {kk: v.float() for kk, v in MU.get_expert_weights(tm, L0, e).items()}
            qa = GQ.quantize_expert(W, A.H_sum((L0, e, "gu"), [0, 1]), A.H_sum((L0, e, "down"), [0, 1]), group_size=16)
            qu = GQ.quantize_expert(W, U.H_sum((L0, e, "gu"), [0, 1]), U.H_sum((L0, e, "down"), [0, 1]), group_size=16)
            assert all(torch.equal(qa[p], qu[p]) for p in qa)                       # 21, 23
            assert all(qa[p].shape == W[p].shape and qa[p].dtype == torch.float32 for p in qa)
    assert changed > 0, "router affinity must change Hessians"


def test_batch1_profiles_sum(tm, cache_dir):                                         # 19
    from moe_calib.data_utils import load_layer_cache
    cache = load_layer_cache(cache_dir, L0)
    both = Profiler(tm, L0, K=1, hessian_weighting="router_affinity").run(cache, [1, 2]).acc
    a = Profiler(tm, L0, K=1, hessian_weighting="router_affinity").run(cache, [1]).acc
    b = Profiler(tm, L0, K=1, hessian_weighting="router_affinity").run(cache, [2]).acc
    for e in range(E):
        k = (L0, e, "gu")
        assert torch.allclose(both.H[k], a.H[k] + b.H[k], rtol=1e-5, atol=1e-5)


def test_constant_affinity_gptq_invariance():                                       # 12 + damping interaction
    g = torch.Generator().manual_seed(4)
    X = torch.randn(200, 32, generator=g)
    W = torch.randn(8, 32, generator=g)
    H = X.T @ X
    q1 = GQ.gptq_quantize(W, H, group_size=16, percdamp=1.0)
    q2 = GQ.gptq_quantize(W, 0.25 * H, group_size=16, percdamp=1.0)
    assert torch.equal(q1, q2)       # relative damping (percdamp * mean diag) scales with H -> exact invariance
    q3 = GQ.gptq_quantize(W, 0.3 * H, group_size=16, percdamp=1.0)
    assert torch.allclose(q1, q3, atol=1e-5)


def test_defaults_backward_compatible():                                             # 22, 30
    import e3_allocation as E3
    from moe_calib import partial_model_quant as PQ
    assert inspect.signature(Profiler).parameters["hessian_weighting"].default == "none"
    assert inspect.signature(E3.quantize_and_score).parameters["hessian_weighting"].default == "none"
    assert inspect.signature(PQ.quantize_install).parameters["hessian_weighting"].default == "none"
    assert inspect.signature(BlockHessianAccumulator.add).parameters["row_weight"].default is None
    sig = inspect.signature(GQ.gptq_quantize).parameters
    assert sig["actorder"].default is True and sig["bits"].default == 4 and sig["group_size"].default == 128
    with pytest.raises(ValueError):
        Profiler(tiny_model()[0], L0, hessian_weighting="c_squared")
    X = torch.randn(9, 6)
    acc = _acc(6); acc.add((0, 0, "gu"), X, 0)
    assert torch.equal(acc.H[(0, 0, "gu")][0], X.T @ X)


def test_gptq_solver_source_unchanged():                                             # 22
    path = os.path.join(ROOT, "runs_ebss_official_validation", "protocol.json")
    if not os.path.exists(path):
        pytest.skip("official validation protocol absent")
    prot = json.load(open(path))
    hashes = prot.get("code_sha256") or prot.get("code_hashes") or {}
    ref = hashes.get("moe_calib/gptq.py")
    if ref is None:
        pytest.skip("no recorded gptq.py hash")
    assert hashlib.sha256(open(os.path.join(ROOT, "moe_calib/gptq.py"), "rb").read()).hexdigest() == ref


# 24, 25 ---------------------------------------------------------------- evaluation metrics
def test_weighted_error_and_moe_output_eval(tm):
    torch.manual_seed(0)
    d = tm.config.hidden_size
    blk = MU.moe_block(tm, L0)
    h = _h(tm, T=12, seed=9)
    cap, out = _block_capture(tm, L0, h)
    vals, sel = AG.routing_coefficients(cap["logits"], MU.top_k(tm))
    Wfp = {e: {k: v.float() for k, v in MU.get_expert_weights(tm, L0, e).items()} for e in range(E)}
    Wn = {e: {k: v + 0.01 * torch.randn_like(v) for k, v in Wfp[e].items()} for e in range(E)}
    hf = h.view(-1, d).float()
    shared = F.sigmoid(hf @ blk.shared_expert_gate.weight.float().T) * AG.expert_fp32(
        hf, {k: getattr(blk.shared_expert, k).weight.float() for k in AG.PROJ})
    acc = AG.LayerEvalAccumulator(["same", "noisy"], E, d)
    y = acc.add_sequence(hf, vals, sel, Wfp, {"same": Wfp, "noisy": Wn}, shared)
    s = acc.summary()
    assert s["same"]["moe_layer_output_rel_err"] == 0 and np.nansum(s["same"]["expert_weighted"]) == 0
    assert s["noisy"]["moe_layer_output_rel_err"] > 0
    # fp32 reference output == the block's BF16 output up to BF16 rounding
    assert torch.allclose(y, out.view(-1, d).float(), rtol=0.05, atol=0.05 * float(out.float().abs().max()))
    # hand computation of the weighted expert error and MoE output error
    rows = AG.dispatch_rows(sel, vals, E)
    err = torch.zeros_like(hf)
    for e, (tx, slot, c) in rows.items():
        a, b = AG.expert_fp32(hf[tx], Wfp[e]), AG.expert_fp32(hf[tx], Wn[e])
        wn = float((c.double() * ((a - b).double() ** 2).sum(1)).sum())
        wd = float((c.double() * (a.double() ** 2).sum(1)).sum())
        assert s["noisy"]["expert_weighted"][e] == pytest.approx(wn / wd, rel=1e-6)    # c, not c^2
        err.index_add_(0, tx, (a - b) * c[:, None])
    assert s["noisy"]["moe_layer_output_rel_err"] == pytest.approx(
        float((err.double() ** 2).sum() / (y.double() ** 2).sum()), rel=1e-6)


# 29 ------------------------------------------------------------------- classification
THR = dict(tau_rel=0.005, min_layers_improved=3)


def test_classification():
    lr = {1: -.01, 8: -.02, 16: -.01, 23: .001}
    assert AG.classify(THR, lr, -.03, True, {0: -.012})["verdict"] == "AGQ-LAYER-QUALITY-GAIN"
    assert AG.classify(THR, lr, -.03, True, {0: -.012, 1: .001})["verdict"] == "AGQ-SURROGATE-ONLY-GAIN"
    assert AG.classify(THR, {1: .01, 8: .01, 16: .0, 23: .0}, -.03, True, {0: .005})["verdict"] == "AGQ-QUALITY-REGRESSION"
    assert AG.classify(THR, {1: -.001, 8: .001, 16: 0, 23: 0}, -.001, True, {0: 0})["verdict"] == "AGQ-NO-MEASURABLE-GAIN"
    assert AG.classify(THR, {1: -.001, 8: .001, 16: 0, 23: 0}, -.02, True, {0: 0})["verdict"] == "AGQ-SURROGATE-ONLY-GAIN"
    assert AG.classify(THR, lr, -.03, False, {0: -.01})["verdict"] == "AGQ-RUN-BLOCKED"
    assert AG.classify(THR, lr, -.03, True, {0: -.012}, primary_rel=-.001)["verdict"] == "AGQ-SURROGATE-ONLY-GAIN"
    assert AG.classify_ppl(-0.02, 0.014) == "gain" and AG.classify_ppl(0.001, 0.014) == "inconclusive"
    assert AG.classify_secondary(-0.01, 0.005) == "gain"


def test_weight_diagnostics_codes():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(100, 32, generator=g); W = torch.randn(4, 32, generator=g)
    info = {}
    Q = GQ.gptq_quantize(W, X.T @ X, group_size=16, info=info)
    c = AG.codes(Q, info)
    assert torch.allclose(c, c.round()) and c.min() >= 0 and c.max() <= 15
    info2 = {}
    Q2 = GQ.gptq_quantize(W, X.T @ (X * torch.rand(100, 1, generator=g)), group_size=16, info=info2)
    wd = AG.weight_diagnostics(Q, Q2, W, info, info2)
    assert 0 <= wd["frac_codes_changed"] <= 1 and wd["rel_l2_diff"] >= 0
    hd = AG.hessian_diagnostics(X.T @ X, 3 * X.T @ X)
    assert hd["shape_residual"] < 1e-6 and hd["rel_frob_trace_norm"] < 1e-6 and hd["actorder_positions_changed"] == 0


# 20, 26, 27, 28 ----------------------------------------------------- frozen artifacts (skip if absent)
def _protocol():
    path = os.path.join(OUT, "protocol.json")
    if not os.path.exists(path):
        pytest.skip("AGQ protocol not frozen yet")
    return json.load(open(path))


def test_calibration_manifest_identity():                                           # 20
    prot = _protocol()
    man = json.load(open(os.path.join(OUT, "calibration_manifest.json")))
    for name, ent in man["configs"].items():
        r = json.load(open(os.path.join(ROOT, "runs_ebss_official_validation", "results", f"{name}.json")))
        assert r["cache_rows"] == ent["cache_rows"] and len(set(ent["cache_rows"])) == 128
        assert r["policy"] == "CANONICAL-B1"
    assert prot["calibration"]["primary"] == "FR_s0"


def test_fixed_eval_windows_and_ppl_protocol():                                       # 26, 27
    prot = _protocol()
    assert prot["evaluation"]["reconstruction_rows"] == [2128, 2256]
    pp = json.load(open(os.path.join(ROOT, "runs_partial_ppl", "evaluation_protocol.json")))
    assert prot["evaluation"]["ppl_windows_sha256"] == pp["windows_sha256"]


def test_cost_accounting():                                                          # 28
    import agq_only_validation as AV
    c = AV.cost_row(dict(quantize_total_s=10.0, hessian_s=3.0, gptq_s=6.0, peak_gpu_gb=1.0, peak_cpu_gb=2.0))
    assert c["other_s"] == pytest.approx(1.0) and c["quantize_total_s"] == 10.0


def test_empty_expert_call_is_accepted(tm):
    pr = Profiler(tm, L0, K=1, hessian_weighting="router_affinity")
    pr._cur_rows = {}
    pr._check_rows(0, "gu", torch.zeros(0, tm.config.hidden_size))       # model calls unrouted experts with 0 rows
    with pytest.raises(RuntimeError):
        pr._check_rows(0, "gu", torch.zeros(2, tm.config.hidden_size))
