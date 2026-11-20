"""CPU tests for diagnostics/theorem1_check.py (Theorem-1 closed form of GPTQ's error).

Small sizes: d=24, rows=5, group_size=4, blocksize=8, random correlated data."""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import theorem1_check as T1                                       # noqa: E402
from moe_calib.gptq import _qparams, gptq_quantize                # noqa: E402

D, ROWS, GS, BS, BITS = 24, 5, 4, 8, 4
NUS = [0.001, 0.01, 0.1, 1.0, 10.0]


def data(seed=0, n=40, dead=None):
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(D, D, generator=g, dtype=torch.float64) * 0.4 + torch.eye(D, dtype=torch.float64)
    X = torch.randn(n, D, generator=g, dtype=torch.float64) @ mix
    X = X * torch.linspace(0.3, 3.0, D, dtype=torch.float64)          # unequal scales -> non-trivial act-order
    if dead is not None:
        X[:, dead] = 0.0
    H = (X.T @ X).float()
    W = torch.randn(ROWS, D, generator=g) * 0.05
    Xs = torch.randn(3 * n, D, generator=g, dtype=torch.float64) @ mix
    S = Xs.T @ Xs / Xs.shape[0]
    return W, H, S


class _F64Torch:
    """torch proxy whose float32 is float64: runs the UNMODIFIED gptq.py code path (lazy batches, groups,
    dead columns, act-order) in fp64 so algebraic identities can be checked below fp32 round-off."""
    float32 = torch.float64

    def __getattr__(self, k):
        return getattr(torch, k)

    @staticmethod
    def eye(*a, **k):
        k.setdefault("dtype", torch.float64)
        return torch.eye(*a, **k)


def solver(W, H, nu, info, fp64):
    if not fp64:
        return gptq_quantize(W, H, BITS, GS, nu, True, blocksize=BS, info=info)
    import moe_calib.gptq as G
    orig = G.torch
    G.torch = _F64Torch()
    try:
        return gptq_quantize(W, H, BITS, GS, nu, True, blocksize=BS, info=info)
    finally:
        G.torch = orig


def ref_gptq(W, H, percdamp, dtype=torch.float32):
    """Column-by-column GPTQ (no lazy batches), copying gptq.py's dead-column fix, act-order, damping,
    Hinv and group fitting at group starts. Records Delta (working weight - dequantized) per column."""
    W = W.to(dtype).clone()
    H = H.to(dtype).clone()
    cols = W.shape[1]
    maxq = 2 ** BITS - 1
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    perm = torch.argsort(torch.diag(H), descending=True)
    W, H = W[:, perm], H[perm][:, perm]
    H += torch.eye(cols, dtype=dtype) * (percdamp * torch.mean(torch.diag(H)))
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    Delta = torch.zeros_like(W)
    for c in range(cols):
        if c % GS == 0:
            scale, zero = _qparams(W[:, c:c + GS], maxq)
        w = W[:, c]
        q = torch.clamp(torch.round(w / scale + zero), 0, maxq)
        dq = scale * (q - zero)
        Q[:, c] = dq
        Delta[:, c] = w - dq
        err = (w - dq) / Hinv[c, c]
        W[:, c:] -= err.unsqueeze(1) * Hinv[c, c:].unsqueeze(0)
    return Q[:, torch.argsort(perm)].double(), Delta.double(), perm, dead


@pytest.mark.parametrize("nu", NUS)
@pytest.mark.parametrize("dead", [None, [5]])
def test_cholesky_form_equals_nested_ridge(nu, dead):
    _, H, _ = data(1, dead=dead)
    Hp, perm, dmask = T1.solver_coords(H)
    M1 = T1.m_matrix(Hp, nu)
    M2 = T1.m_ridge(Hp, nu)
    assert torch.allclose(torch.diag(M1), torch.ones(D, dtype=torch.float64))
    assert torch.triu(M1, 1).abs().max() == 0
    assert (M1 - M2).abs().max() <= 1e-10 * max(1.0, float(M2.abs().max()))


@pytest.mark.parametrize("fp64", [True, False])
@pytest.mark.parametrize("nu", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("dead", [None, [5]])
@pytest.mark.parametrize("seed", [0, 3])
def test_gptq_matches_reference_and_u_equals_M_delta(nu, dead, seed, fp64):
    """fp64=True: the frozen solver's own code run in fp64 -> identities to the stated tolerances
    (Q <= 1e-5, u = M Delta <= 1e-6 relative). fp64=False: the production fp32 run; u = W - Q then carries
    fp32 cancellation (~eps*|W|/|u|), so u = M Delta is checked at 1e-5 relative there."""
    W, H, S = data(seed, dead=dead)
    inf = {}
    Q = solver(W, H, nu, inf, fp64)
    Qr, Delta, perm, dmask = ref_gptq(W, H, nu, torch.float64 if fp64 else torch.float32)
    tol_u = 1e-6 if fp64 else 1e-5
    assert torch.equal(inf["perm"], perm)
    assert inf["n_dead"] == (0 if dead is None else len(dead))
    # frozen lazy-batch solver == column-by-column reference
    assert (Q.double() - Qr).abs().max() <= 1e-5
    # u = M Delta in permuted coordinates, u from the solver's own output
    Hp, perm2, _ = T1.solver_coords(H, inf["perm"])
    M = T1.m_matrix(Hp, nu)
    Wfix = W.double().clone()
    Wfix[:, dmask] = 0.0
    u = (Wfix - Q.double())[:, perm]
    pred = Delta @ M.T                         # rows: (M Delta_r)^T
    assert torch.linalg.norm(u - pred) <= tol_u * torch.linalg.norm(u)
    # col_err2 sums to ||Delta||_F^2 (and matches per column)
    assert abs(float(inf["col_err2"].sum()) - float(Delta.pow(2).sum())) <= 1e-5 * float(Delta.pow(2).sum())
    assert torch.allclose(inf["col_err2"], Delta.pow(2).sum(0), rtol=1e-4, atol=1e-12)
    # residual reconstruction used by the GPU diagnostic
    D_rec = T1.residuals_from_u(M, u)
    assert torch.linalg.norm(D_rec - Delta) <= 10 * tol_u * torch.linalg.norm(Delta)
    # e_meas identity: sum_r u_r^T S u_r == sum_r Delta_r^T (M^T S M) Delta_r
    Sp = S[perm][:, perm]
    e1 = T1.quad_err(u, Sp)
    e2 = float(((Delta @ (M.T @ Sp @ M)) * Delta).sum())
    assert abs(e1 - e2) <= 1e-6 * abs(e2)
    # basis invariance: e_meas in original coordinates
    assert abs(T1.quad_err(Wfix - Q.double(), S) - e1) <= 1e-9 * abs(e1)


@pytest.mark.parametrize("dead", [None, [5]])
def test_dead_column_decoupled(dead):
    _, H, _ = data(2, dead=dead)
    Hp, perm, dmask = T1.solver_coords(H)
    M = T1.m_matrix(Hp, 0.01)
    if dead is not None:
        t = int((perm == dead[0]).nonzero())
        e = torch.zeros(D, dtype=torch.float64)
        e[t] = 1.0
        assert torch.equal(M[:, t], e) and torch.equal(M[t, :], e)


def test_b2_cell_consistent_on_cpu():
    W, H, S = data(4)
    rec, Q = T1.b2_cell({"down_proj": W}, H, S, "down", 0.05, gs=GS)
    assert abs(rec["R_ideal"] - rec["trMSM"] / rec["trS"]) <= 1e-12
    assert rec["delta_recon_relerr"] <= 1e-4
    assert 0.0 <= rec["clamp_frac_W"] <= 1.0 and 0.0 <= rec["clamp_frac_work"] <= 1.0
    assert abs(rec["e_meas"] - rec["e_true"]) <= 1e-12 * rec["e_true"]


def test_r_ideal_tends_to_one():
    _, H, S = data(5)
    Hp, perm, _ = T1.solver_coords(H)
    Sp = S[perm][:, perm]
    rs = [T1.r_ideal(T1.m_matrix(Hp, nu), Sp) for nu in (1e2, 1e4, 1e6, 1e8)]
    assert abs(rs[-1] - 1.0) <= 1e-6
    assert all(abs(b - 1) <= abs(a - 1) + 1e-12 for a, b in zip(rs, rs[1:]))


def test_cholesky_failure_is_reported_not_jittered():
    Hp = torch.zeros(D, D, dtype=torch.float64)
    Hp[0, 0] = -1.0
    assert T1.m_matrix(Hp, 0.0) is None


def test_parabola_and_george_saab():
    import numpy as np
    xs = np.log([0.01, 0.1, 1.0])
    ys = [(x - np.log(0.3)) ** 2 for x in xs]
    assert abs(np.exp(T1.parabola_min(xs, ys, 1)) - 0.3) < 1e-9
    assert T1.parabola_min(xs, ys, 0) is None
    assert abs(T1.george_saab(2048, 2048) - 1.0) < 1e-12
    assert abs(T1.george_saab(8 * 2048, 2048) - 0.5) < 1e-12
