"""CPU tests for moe_calib.actorder_policy (opt-in act-order policies) and moe_calib.functional_diff."""
import os

os.environ.setdefault("MOE_DEVICE", "cpu")

import torch

from moe_calib import actorder_policy as AP
from moe_calib import functional_diff as FD
from moe_calib.gptq import gptq_quantize


def _problem(seed=0, r=24, c=256, n=600):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(r, c, generator=g)
    X = torch.randn(n, c, generator=g) * torch.linspace(0.05, 4, c)
    return W, X.T @ X, X


def test_natural_policy_is_default_solver_bitwise():
    W, H, _ = _problem()
    assert torch.equal(AP.solve(W, H, "natural", 4, 64, 1.0), gptq_quantize(W, H, 4, 64, 1.0))


def test_fixed_with_own_natural_perm_equals_natural():
    W, H, _ = _problem(1)
    info = {}
    Qn = AP.solve(W, H, "natural", 4, 64, 1.0, info=info)
    Qf = AP.solve(W, H, "fixed", 4, 64, 1.0, ref_perm=info["perm"])
    assert torch.equal(Qn, Qf)
    assert torch.equal(AP.natural_perm_cpu(H), info["perm"])


def test_disabled_uses_original_order():
    W, H, _ = _problem(2)
    info = {}
    Q = AP.solve(W, H, "disabled", 4, 64, 1.0, info=info)
    assert torch.equal(info["perm"], torch.arange(W.shape[1]))
    Qi = gptq_quantize(W, H, 4, 64, 1.0, actorder_override=torch.arange(W.shape[1]))
    assert torch.equal(Q, Qi)


def test_bucket_key_and_ties():
    d = torch.tensor([1.0, 1.001, 3.0, 0.5, 1.0039, 1.0040, 0.0], dtype=torch.float32)
    k = AP.bucket_key(d, 8)
    assert k[0] == k[1] == 1.0 and k[2] == 3.0 and k[3] == 0.5
    H = torch.diag(d)
    perm = AP.bucketed_perm(H, 8)
    # dead column (diag 0) -> 1.0, so it ties with columns 0, 1 and sorts after them by index
    assert perm.tolist()[:1] == [2]
    ones = [i for i in perm.tolist() if float(k[i] if d[i] != 0 else 1.0) == 1.0]
    assert ones == sorted(ones)
    assert torch.equal(AP.bucketed_perm(H, 8), AP.bucketed_perm(H.clone(), 8))
    # boundaries do not depend on the other columns
    d2 = d.clone(); d2[2] = 1e6
    assert torch.equal(AP.bucket_key(d2, 8)[[0, 1, 3]], k[[0, 1, 3]])


def test_bucketed_solver_deterministic_and_valid():
    W, H, _ = _problem(3)
    info = {}
    Q1 = AP.solve(W, H, "bucketed", 4, 64, 1.0, bucket_bits=4, info=info)
    Q2 = AP.solve(W, H, "bucketed", 4, 64, 1.0, bucket_bits=4)
    assert torch.equal(Q1, Q2) and torch.equal(info["perm"], AP.bucketed_perm(H, 4))


def test_accumulator_matches_direct():
    g = torch.Generator().manual_seed(4)
    Ya, Yb, Yr = (torch.randn(50, 7, generator=g) for _ in range(3))
    acc = FD.acc_new()
    FD.acc_add(acc, Ya[:20] - Yb[:20], Yb[:20], Yr[:20])
    FD.acc_add(acc, Ya[20:] - Yb[20:], Yb[20:], Yr[20:])
    r = FD.acc_final(acc)
    assert abs(r["E_func"] - float((Ya - Yb).norm() / Yr.norm())) < 1e-5
    cos = float((Ya * Yb).sum() / (Ya.norm() * Yb.norm()))
    assert abs(r["cosine"] - cos) < 1e-5
    assert abs(r["rmse"] - float((Ya - Yb).pow(2).mean().sqrt())) < 1e-5


def test_importance_profile_and_split():
    W, H, X = _problem(5, r=8, c=64, n=200)
    Hh = X.double().T @ X.double()
    dW = torch.randn(8, 64, dtype=torch.float32)
    flip = torch.rand(8, 64) < 0.3
    bins = FD.quantile_bins(torch.diag(Hh), 4)
    r = FD.importance_profile(dW, flip, Hh, bins, 4)
    full = float((X.double() @ dW.double().T).pow(2).sum())
    assert abs(r["full_energy"] - full) / full < 1e-9
    assert abs(sum(b["diag_share"] for b in r["bins"]) - 1) < 1e-9
    assert sum(b["flips"] for b in r["bins"]) == int(flip.sum())
    s = FD.flip_grid_split(dW, flip, Hh)
    assert abs(s["share_flip"] + s["share_grid"] + s["share_cross"] - 1) < 1e-9
    assert abs(s["energy_total"] - full) / full < 1e-9
