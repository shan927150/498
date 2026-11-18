"""CPU tests for moe_calib.loss_alignment (synthetic data only, no model)."""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_calib import loss_alignment as LA   # noqa: E402

torch.set_default_dtype(torch.float32)


# ------------------------------------------------------------------ MVP A
def test_r0_equals_macro_mean():
    rng = np.random.default_rng(0)
    E = rng.uniform(1e-3, 1e-2, size=(4, 60))
    assert LA.weighted_reconstruction(E, LA.uniform_weights(4, 60)) == pytest.approx(E.mean(), rel=1e-14)


def test_weighted_aggregation_by_hand():
    E = np.array([[1.0, 3.0], [2.0, 6.0]])
    w = LA.normalize_rows([[1, 3], [1, 1]])
    # layer 0: 0.25*1 + 0.75*3 = 2.5 ; layer 1: 4.0 ; mean over layers 3.25
    assert LA.weighted_reconstruction(E, w) == pytest.approx(3.25)
    assert np.allclose(LA.per_layer_weighted(E, w), [2.5, 4.0])


def test_normalization_and_validation():
    w = LA.normalize_rows(np.arange(1, 13).reshape(3, 4))
    assert np.allclose(w.sum(1), 1.0)
    with pytest.raises(ValueError):
        LA.normalize_rows([[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        LA.normalize_rows([[-1, 2]])
    with pytest.raises(ValueError):
        LA.weighted_reconstruction(np.ones((2, 2)), np.ones((2, 2)))          # rows sum to 2


def test_layer_weighting_is_equal_even_if_counts_differ():
    """A layer with 10x more routed tokens must NOT get 10x the weight (rows are normalized per layer)."""
    E = np.array([[1.0, 1.0], [5.0, 5.0]])
    w = LA.normalize_rows([[10, 10], [1, 1]])
    assert LA.weighted_reconstruction(E, w) == pytest.approx(3.0)


def test_expert_matrix_alignment_and_errors():
    per = [dict(layer=L, expert=e, mean=L * 100 + e) for L in (8, 1) for e in range(3)]
    M = LA.expert_matrix(per, [1, 8], 3)
    assert M[0, 2] == 102 and M[1, 0] == 800
    with pytest.raises(ValueError):
        LA.expert_matrix(per[:-1], [1, 8], 3)
    with pytest.raises(ValueError):
        LA.expert_matrix(per + per[:1], [1, 8], 3)


def test_routing_stats_index_alignment():
    idx = torch.tensor([[0, 2], [2, 1], [2, 0]])
    w = torch.tensor([[0.5, 0.25], [0.5, 0.1], [0.2, 0.3]])
    cnt, s1, s2 = LA.routing_stats(idx, w, 4)
    assert cnt.tolist() == [2, 1, 3, 0]
    assert s1 == pytest.approx([0.8, 0.1, 0.95, 0.0])
    assert s2 == pytest.approx([0.25 + 0.09, 0.01, 0.0625 + 0.25 + 0.04, 0.0])
    with pytest.raises(ValueError):
        LA.routing_stats(torch.tensor([[0, 5]]), torch.tensor([[1.0, 1.0]]), 4)


def test_router_topk_matches_hf_semantics():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(50, 60, generator=g).to(torch.bfloat16)
    idx, w32, wa = LA.router_topk(logits, 4, norm_topk_prob=False)
    p = F.softmax(logits, dim=1, dtype=torch.float)
    ref_w, ref_i = torch.topk(p, 4, dim=-1)
    assert torch.equal(idx, ref_i) and torch.equal(w32, ref_w) and torch.equal(wa, ref_w.to(torch.bfloat16))
    assert (w32.sum(1) <= 1 + 1e-6).all() and (w32[:, :-1] >= w32[:, 1:]).all()
    _, wn, _ = LA.router_topk(logits, 4, norm_topk_prob=True)
    assert torch.allclose(wn.sum(1), torch.ones(50))


# ------------------------------------------------------------------ MVP B
def _experts(E=5, d=16, f=12, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return [{"gate_proj": torch.randn(f, d, generator=g) * scale, "up_proj": torch.randn(f, d, generator=g) * scale,
             "down_proj": torch.randn(d, f, generator=g) * scale} for _ in range(E)]


def _routing(T=40, E=5, k=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, E, generator=g)
    idx, w, _ = LA.router_topk(logits, k, out_dtype=torch.float32)
    X = torch.randn(T, 16, generator=g)
    return X, idx, w


def test_zero_error_when_weights_identical():
    W = _experts()
    X, idx, w = _routing()
    r = LA.layer_output_errors(X, idx, w, W, W)
    assert r["D_sum"] == 0 and r["U_sum"] == 0 and r["experts"]["out_num"].sum() == 0
    assert r["experts"]["proj_num"].sum() == 0 and r["M_sum"] > 0 and r["finite"]


def test_reference_aggregate_matches_dense_moe():
    W = _experts()
    X, idx, w = _routing()
    r = LA.layer_output_errors(X, idx, w, W, W, return_M=True)
    dense = torch.zeros(X.shape[0], 16)
    for t in range(X.shape[0]):
        for j in range(idx.shape[1]):
            e = int(idx[t, j])
            dense[t] += w[t, j] * LA.mlp_fp32(X[t:t + 1], W[e])[0][0]
    assert torch.allclose(r["M"], dense, atol=1e-5, rtol=1e-5)


def test_metrics_against_brute_force():
    Wr, Wq = _experts(seed=0), _experts(seed=0)
    g = torch.Generator().manual_seed(9)
    for e in range(len(Wq)):
        for k in Wq[e]:
            Wq[e][k] = Wq[e][k] + 0.05 * torch.randn(Wq[e][k].shape, generator=g)
    X, idx, w = _routing()
    seq = torch.arange(X.shape[0]) // 10
    r = LA.layer_output_errors(X, idx, w, Wr, Wq, seq_id=seq, n_seq=4)
    D = U = Mm = 0.0
    out_num = np.zeros(5)
    for t in range(X.shape[0]):
        dm, mm, u = torch.zeros(16), torch.zeros(16), 0.0
        for j in range(idx.shape[1]):
            e = int(idx[t, j])
            fr = LA.mlp_fp32(X[t:t + 1], Wr[e])[0][0]
            fq = LA.mlp_fp32(X[t:t + 1], Wq[e])[0][0]
            dm += w[t, j] * (fq - fr)
            mm += w[t, j] * fr
            u += float(w[t, j]) ** 2 * float((fq - fr).pow(2).sum())
            out_num[e] += float((fq - fr).pow(2).sum())
        D += float(dm.pow(2).sum()); Mm += float(mm.pow(2).sum()); U += u
    assert r["D_sum"] == pytest.approx(D, rel=1e-4) and r["M_sum"] == pytest.approx(Mm, rel=1e-4)
    assert r["U_sum"] == pytest.approx(U, rel=1e-4)
    assert np.allclose(r["experts"]["out_num"], out_num, rtol=1e-4)
    assert r["seq_D"].sum() == pytest.approx(r["D_sum"], rel=1e-9)
    s = LA.summarize_layer(r)
    assert s["B3"] == pytest.approx(D / U, rel=1e-4) and s["B2"] == pytest.approx(D / Mm, rel=1e-4)


def _two_expert_case(sign):
    """Every token routes to experts 0 and 1 with equal weight; expert 1's down-projection error is +/- expert 0's
    with identical gate/up, so the two output perturbations are exactly equal (sign=+1) or opposite (sign=-1)."""
    d, f = 8, 6
    g = torch.Generator().manual_seed(3)
    base = {"gate_proj": torch.randn(f, d, generator=g), "up_proj": torch.randn(f, d, generator=g),
            "down_proj": torch.randn(d, f, generator=g)}
    Wr = [dict(base), dict(base)]
    err = 0.1 * torch.randn(d, f, generator=g)
    Wq = [dict(base, down_proj=base["down_proj"] + err), dict(base, down_proj=base["down_proj"] + sign * err)]
    T = 20
    X = torch.randn(T, d, generator=g)
    idx = torch.tensor([[0, 1]] * T)
    w = torch.full((T, 2), 0.5)
    return LA.summarize_layer(LA.layer_output_errors(X, idx, w, Wr, Wq))


def test_constructive_and_destructive_interference():
    con, des = _two_expert_case(+1), _two_expert_case(-1)
    # equal perturbations: ||0.5 d + 0.5 d||^2 = ||d||^2 = 2 * (0.25 + 0.25) ||d||^2 -> ratio 2
    assert con["B3"] == pytest.approx(2.0, rel=1e-5)
    # opposite perturbations cancel exactly
    assert des["B3"] == pytest.approx(0.0, abs=1e-9)
    assert con["B1"] == pytest.approx(des["B1"], rel=1e-6)          # uncancelled energy is sign-blind


def test_projection_error_matches_e3_definition():
    Wr, Wq = _experts(E=1, seed=4), _experts(E=1, seed=5)
    X = torch.randn(7, 16)
    _, h = LA.mlp_fp32(X, Wr[0])
    num, den = LA.projection_errors(X, h, Wr[0], Wq[0])
    n_ref = sum(float((inp @ (Wr[0][k] - Wq[0][k]).T).pow(2).sum()) for k, inp in
                (("gate_proj", X), ("up_proj", X), ("down_proj", h)))
    assert float(num.sum()) == pytest.approx(n_ref, rel=1e-5)
    ex_seq = dict(proj_num=np.array([[1.0, 4.0, 0.0]]), proj_den=np.array([[10.0, 10.0, 0.0]]),
                  tokens=np.array([[2, 3, 0]]))
    assert LA.per_seq_projection_mean(ex_seq)[0] == pytest.approx(0.25)    # mean(0.1, 0.4), empty window skipped


def test_unrouted_expert_reported_not_crashing():
    W = _experts(E=4)
    X = torch.randn(10, 16)
    idx = torch.tensor([[0, 1]] * 10)
    r = LA.layer_output_errors(X, idx, torch.full((10, 2), 0.5), W, W)
    assert r["n_unrouted_experts"] == 2


def test_window_bootstrap_ratio():
    a = np.array([1.0, 2.0, 3.0]); d = np.array([10.0, 10.0, 10.0])
    pt, lo, hi = LA.window_bootstrap_ratio_diff(a * 0.9, d, a, d, n_boot=500)
    assert pt == pytest.approx(-0.1) and lo == pytest.approx(-0.1) and hi == pytest.approx(-0.1)


def test_custom_mlp_hook_and_bf16_inputs():
    """A BF16-native forward can be passed in; outputs are accumulated in FP32 and zero error stays exactly zero."""
    W = [{k: v.to(torch.bfloat16) for k, v in w.items()} for w in _experts()]
    X, idx, w = _routing()

    def mlp_bf16(X_, W_, act=F.silu):
        h = act(X_ @ W_["gate_proj"].T) * (X_ @ W_["up_proj"].T)
        return (h @ W_["down_proj"].T).float(), h.float()

    r = LA.layer_output_errors(X.to(torch.bfloat16), idx, w, W, W, mlp=mlp_bf16)
    assert r["D_sum"] == 0 and r["M_sum"] > 0 and r["experts"]["proj_num"].sum() == 0
