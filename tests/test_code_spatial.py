"""CPU tests for moe_calib.code_spatial (alignment, decomposition, spatial statistics, conditioned null)."""
import os

os.environ.setdefault("MOE_DEVICE", "cpu")

import torch

from moe_calib import code_spatial as CS
from moe_calib import dense_gptq as DG
from moe_calib.gptq import gptq_quantize


def _rec(W, H, gs=32):
    info = {}
    Q = gptq_quantize(W, H, 4, gs, 1.0, info=info)
    q, gid, exact = DG.integer_codes(Q, info, 4, gs)
    return Q, dict(q=q, group=gid, perm=info["perm"].int(), scale=info["scale"], zero=info["zero"]), exact


def test_dequant_reproduces_solver_bitwise_and_validate():
    g = torch.Generator().manual_seed(0)
    W = torch.randn(24, 96, generator=g)
    X = torch.randn(400, 96, generator=g) * torch.linspace(0.1, 3, 96)
    Q, rec, exact = _rec(W, X.T @ X)
    assert exact and torch.equal(CS.dequant(rec), Q)
    v = CS.validate_record(rec, Q.to(torch.bfloat16), 32)
    assert all(v.values()), v


def test_decompose_identity_and_grid_only():
    g = torch.Generator().manual_seed(1)
    W = torch.randn(16, 64, generator=g)
    X = torch.randn(300, 64, generator=g)
    Q, rec, _ = _rec(W, X.T @ X)
    Wb = Q.to(torch.bfloat16)
    d0 = CS.decompose(rec, rec, Wb, Wb)
    assert d0["cat_A"] == 1.0 and d0["frac_code_changed"] == 0.0
    r2 = dict(rec, scale=rec["scale"] * 1.5)                       # same codes, rescaled grid
    W2 = CS.dequant(r2).to(torch.bfloat16)
    d = CS.decompose(r2, rec, W2, Wb)
    assert d["frac_code_changed"] == 0.0 and d["cat_B"] > 0 and d["B_with_grid_change"] == 1.0
    assert d["bf16_changed_grid_only_a_on_b"] == 1.0


def test_null_preserves_counts_and_matches_analytic():
    g = torch.Generator().manual_seed(2)
    M = torch.rand(64, 512, generator=g) < 0.05
    M[:8, :40] = True                                               # some clustering
    gen = torch.Generator().manual_seed(3)
    N = CS.null_masks(M, gen)
    S, Sn = M.view(64, 4, 128).sum(-1), N.view(64, 4, 128).sum(-1)
    assert torch.equal(S, Sn)
    an = CS.analytic_pairs(M)
    sims = [CS.stats(CS.null_masks(M, gen), full=False)["pairs_1"] for _ in range(300)]
    m = sum(sims) / len(sims)
    assert abs(m - an["segnull_pairs_1"]) < 0.05 * an["segnull_pairs_1"] + 2
    obs = CS.stats(M)
    assert obs["max_run"] >= 40 and CS.enrichment(obs, 1) > 1.5


def test_exec_order_and_runs():
    M = torch.zeros(2, 256, dtype=torch.bool)
    M[0, [3, 4, 5, 200]] = True
    s = CS.stats(M)
    assert s["flips"] == 4 and s["pairs_1"] == 2 and s["runs"] == 2 and s["max_run"] == 3
    perm = torch.arange(255, -1, -1)
    Me = CS.to_exec(M, perm)
    assert Me[0, 255 - 3] and Me[0, 255 - 200]
