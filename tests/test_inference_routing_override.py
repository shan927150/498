"""CPU tests for the opt-in inference-time routing recorder / top-k override (tiny Qwen2-MoE from conftest)."""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diagnostics"))
from conftest import E, tiny_model                              # noqa: E402
from moe_calib import model_utils as MU                         # noqa: E402
from moe_calib import inference_routing_override as IRO         # noqa: E402
from moe_calib import frozen_routing as FRT                     # noqa: E402
from moe_calib import partial_model_quant as PQ                 # noqa: E402
import eval_partial_ppl as EP                                   # noqa: E402

N, T, BATCH = 6, 16, 4                                          # uneven last batch on purpose


@pytest.fixture
def mw():
    model, _ = tiny_model()
    g = torch.Generator().manual_seed(3)
    win = torch.randint(0, model.config.vocab_size, (N, T), generator=g)
    return model, win


def _digests(model):
    return {n: PQ._digest(p.view(torch.int16) if p.dtype == torch.bfloat16 else p) for n, p in model.named_parameters()}


def _record(model, win):
    s = IRO.new_session(model, win, BATCH)
    with IRO.attach(model, s):
        r = IRO.layerwise_eval(model, win, BATCH, s, layer_hashes=True)
    return s, r


def test_token_nll_sums_bitwise_equal_historical_scorer():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 9, 50, generator=g).to(torch.bfloat16)
    ids = torch.randint(0, 50, (3, 9), generator=g)
    s0, n0 = EP.nll_from_logits(logits, ids)
    s1, n1, t = IRO.nll_tokens_from_logits(logits, ids)
    assert torch.equal(s0, s1) and torch.equal(n0, n1) and t.shape == (3, 8) and t.dtype == torch.float64
    assert torch.equal(t.sum(1), s0)


def test_plain_evaluator_equals_historical_layerwise_nll(mw):
    model, win = mw
    nll0, cnt0 = EP.layerwise_nll(model, win, batch=BATCH)
    r = IRO.layerwise_eval(model, win, BATCH)
    assert np.array_equal(nll0, r["nll"]) and np.array_equal(cnt0, r["cnt"])
    assert np.array_equal(r["token_nll"].sum(1), r["nll"])          # token NLL sums reproduce per-window NLL


def test_record_only_does_not_change_outputs_and_records_class_topk(mw):
    model, win = mw
    d0 = _digests(model)
    base = IRO.layerwise_eval(model, win, BATCH, layer_hashes=True)
    s, r = _record(model, win)
    assert np.array_equal(base["nll"], r["nll"]) and base["layer_sha"] == r["layer_sha"]
    assert base["final_sha"] == r["final_sha"]
    assert _digests(model) == d0
    FRT.assert_no_override(model)
    # recorded indices equal the block's own topk on the same block input
    inp = {}
    h = MU.moe_block(model, 1).register_forward_pre_hook(lambda m, a: inp.setdefault(len(inp), a[0].clone()))
    IRO.layerwise_eval(model, win, BATCH)
    h.remove()
    for bi, x in inp.items():
        blk = MU.moe_block(model, 1)
        w, i = torch.topk(F.softmax(blk.gate(x.view(-1, x.shape[-1])), dim=1, dtype=torch.float), blk.top_k, dim=-1)
        w0, B = bi * BATCH, x.shape[0]
        assert torch.equal(s.idx[1, w0:w0 + B].reshape(-1, blk.top_k).long(), i)
        assert torch.equal(s.pk[1, w0:w0 + B].reshape(-1, blk.top_k), w)
    assert (s.pk[..., -1] >= s.p_next).all()
    ok, norep = IRO.validate_indices(s.idx, E)
    assert ok and norep
    # dispatch counts = counts of the recorded indices
    assert torch.equal(s.dispatch, IRO.expert_counts(s.idx, E))


def test_self_forcing_is_bitwise_identical(mw):
    model, win = mw
    s, r = _record(model, win)
    f = IRO.new_session(model, win, BATCH, mode="force", ref_idx=s.idx.clone())
    with IRO.attach(model, f):
        rf = IRO.layerwise_eval(model, win, BATCH, f, layer_hashes=True)
    assert (f.n_forced_calls > 0).all()                              # the forced path really ran in every layer
    assert np.array_equal(r["nll"], rf["nll"]) and np.array_equal(r["token_nll"], rf["token_nll"])
    assert r["layer_sha"] == rf["layer_sha"] and r["final_sha"] == rf["final_sha"]
    assert torch.equal(f.idx, s.idx) and torch.equal(f.pk, s.pk)     # router computed normally
    assert torch.equal(f.p_forced, s.pk)                             # gathered probs = natural topk values
    FRT.assert_no_override(model)


def _random_ref(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = MU.top_k(model)
    return torch.stack([torch.stack([torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])
                                     for _ in range(N)]) for _ in range(MU.num_layers(model))]).to(torch.int16)


def test_forced_indices_use_current_probs_and_dispatch_matches_reference(mw):
    model, win = mw
    ref = _random_ref(model)
    f = IRO.new_session(model, win, BATCH, mode="force", ref_idx=ref)
    inp = {}
    hk = MU.moe_block(model, 2).register_forward_pre_hook(lambda m, a: inp.setdefault(len(inp), a[0].clone()))
    outs = {}
    ho = MU.moe_block(model, 2).register_forward_hook(lambda m, a, o: outs.setdefault(len(outs), o[0].clone()))
    with IRO.attach(model, f):
        IRO.layerwise_eval(model, win, BATCH, f)
    hk.remove(); ho.remove()
    assert torch.equal(f.dispatch, IRO.expert_counts(ref, E))
    blk = MU.moe_block(model, 2)
    for bi, x in inp.items():                                        # manual forced MoE output
        xf = x.view(-1, x.shape[-1])
        probs = F.softmax(blk.gate(xf), dim=1, dtype=torch.float)
        sel = ref[2, bi * BATCH:bi * BATCH + x.shape[0]].reshape(-1, blk.top_k).long()
        w = probs.gather(1, sel)
        assert torch.equal(f.p_forced[2, bi * BATCH:bi * BATCH + x.shape[0]].reshape(-1, blk.top_k), w)
        w = w.to(xf.dtype)
        y = torch.zeros_like(xf)
        for e in range(E):                                           # original dispatch order
            mask = F.one_hot(sel, E).permute(2, 1, 0)[e]
            idx, tx = torch.where(mask)
            y.index_add_(0, tx, (blk.experts[e](xf[tx]) * w[tx, idx, None]).to(xf.dtype))
        y = y + F.sigmoid(blk.shared_expert_gate(xf)) * blk.shared_expert(xf)
        assert torch.equal(y.view_as(outs[bi]), outs[bi])


def test_forcing_differs_when_reference_differs(mw):
    model, win = mw
    _, r = _record(model, win)
    f = IRO.new_session(model, win, BATCH, mode="force", ref_idx=_random_ref(model, 1))
    with IRO.attach(model, f):
        rf = IRO.layerwise_eval(model, win, BATCH, f)
    assert not np.array_equal(r["nll"], rf["nll"])


def test_detach_restores_class_forward_and_parameters(mw):
    model, win = mw
    d0 = _digests(model)
    f = IRO.new_session(model, win, BATCH, mode="force", ref_idx=_random_ref(model))
    with IRO.attach(model, f):
        with pytest.raises(AssertionError):
            FRT.assert_no_override(model)
        IRO.layerwise_eval(model, win, BATCH, f)
    FRT.assert_no_override(model)
    for L in range(MU.num_layers(model)):
        b = MU.moe_block(model, L)
        assert not b.gate._forward_hooks and not any(x._forward_pre_hooks for x in b.experts)
    assert _digests(model) == d0


def test_session_rejects_bad_reference_shape(mw):
    model, win = mw
    with pytest.raises(AssertionError):
        IRO.new_session(model, win, BATCH, mode="force", ref_idx=torch.zeros(1, 2, 3, 2, dtype=torch.int16))


def test_set_disagreement_semantics():
    a = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 3, 4], [2, 1, 3, 4], [1, 2, 3, 5]])
    sc, oo = IRO.set_disagreement(a, b)
    assert sc.tolist() == [False, False, True] and oo.tolist() == [False, True, False]
    assert IRO.n_membership_changes(a, b).tolist() == [0, 0, 1]
    assert IRO.validate_indices(torch.tensor([[0, 1, 1, 2]]), 60) == (True, False)
    assert IRO.validate_indices(torch.tensor([[0, 1, 60, 2]]), 60) == (False, True)
