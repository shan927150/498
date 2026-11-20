"""CPU tests for diagnostics/h5_rules.py: unit-CV identity (R4), vectorized R_ideal interpolation, law fitting."""
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import h5_rules as H5                                               # noqa: E402
import theorem1_check as T1                                         # noqa: E402

D = 12


def blocks_data(seed=0, dead=None, empty=None, per=7):
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(D, D, generator=g, dtype=torch.float64) * 0.5 + torch.eye(D, dtype=torch.float64)
    Xb = []
    for b in range(H5.N_BLOCKS):
        k = 0 if b == empty else per + b
        X = torch.randn(k, D, generator=g, dtype=torch.float64) @ mix
        X = X * torch.linspace(0.5, 2.0, D, dtype=torch.float64)
        if dead is not None:
            X[:, dead] = 0.0
        Xb.append(X)
    return Xb


def ridge_M(A):
    """Independent nested-ridge construction from an arbitrary damped A: M[>t, t] = -A[>t,>t]^-1 A[>t, t]."""
    d = A.shape[0]
    M = torch.eye(d, dtype=A.dtype)
    for t in range(d - 1):
        M[t + 1:, t] = -torch.linalg.solve(A[t + 1:, t + 1:], A[t + 1:, t])
    return M


def brute_cv(Xb, nu):
    """Refit leaving each block out: A_-f = sum_{b != f} X_b^T X_b + n lambda I (full-prefix lambda, perm, dead fix),
    then sum over the left-out block's tokens of ||M_-f^T x||^2, divided by n."""
    X = torch.cat(Xb)
    n = X.shape[0]
    G = X.T @ X
    Hp, perm, dead = T1.solver_coords(G / n)
    lam = nu * float(torch.mean(torch.diag(Hp)))
    tot = 0.0
    for f, Xf in enumerate(Xb):
        if Xf.shape[0] == 0:
            continue
        Gm = sum((Xb[b].T @ Xb[b] for b in range(len(Xb)) if b != f), torch.zeros(D, D, dtype=torch.float64))
        Gm = Gm[perm][:, perm]
        Gm[dead[perm], dead[perm]] = float(n)                      # dead-column fix of the full prefix (Hhat = 1)
        A = Gm + n * lam * torch.eye(D, dtype=torch.float64)
        M = ridge_M(A)
        Y = Xf[:, perm] @ M
        tot += float(Y.pow(2).sum())
    return tot / n


@pytest.mark.parametrize("dead,empty", [(None, None), ([4], None), (None, 3), ([4], 5)])
@pytest.mark.parametrize("nu", [0.01, 0.3, 3.0, 50.0])
def test_cv_equals_brute_force_refit(dead, empty, nu):
    Xb = blocks_data(1, dead, empty)
    blocks = [(X.T @ X, X.shape[0]) for X in Xb if X.shape[0] > 0]
    G = sum(Gb for Gb, _ in blocks)
    n = sum(nb for _, nb in blocks)
    _, perm, _ = T1.solver_coords((G / n).float())
    perm_ref = T1.solver_coords(G / n)[1]
    assert torch.equal(perm, perm_ref)
    out = H5.cv_curve(blocks, perm, [nu])
    ref = brute_cv(Xb, nu)
    assert abs(out["cv"][0] - ref) <= 1e-10 * abs(ref)
    assert len(blocks) == (H5.N_BLOCKS if empty is None else H5.N_BLOCKS - 1)


def test_cv_curve_selects_interior_and_refines():
    Xb = blocks_data(2, per=3)                                     # n < d per fold -> needs damping
    blocks = [(X.T @ X, X.shape[0]) for X in Xb]
    G = sum(Gb for Gb, _ in blocks)
    perm = T1.solver_coords(G / sum(nb for _, nb in blocks))[1]
    out = H5.cv_curve(blocks, perm, H5.NU_CV)
    assert "nu" in out and out["cv_star"] <= min(v for v in out["cv"] if v is not None) + 1e-15


def test_m_from_A_matches_m_matrix():
    Xb = blocks_data(3)
    X = torch.cat(Xb)
    Hp = T1.solver_coords(X.T @ X)[0]
    for nu in (0.01, 1.0):
        assert torch.equal(T1.m_matrix(Hp, nu), T1.m_from_A(T1.damped(Hp, nu)))


def test_block_ids_contiguous():
    order = list(range(100, 132))
    b = H5.block_ids(order, 16)
    assert len(b) == 8 and all(len(x) == 2 for x in b) and sum(b, []) == order[:16]


def test_curve_interp_matches_scipy():
    from scipy.interpolate import PchipInterpolator
    rng = np.random.default_rng(0)
    R = 1 + (H5.LX[None, :] - rng.normal(0, 1, (5, 1))) ** 2 * 0.01 + rng.normal(0, 1e-3, (5, len(H5.LX)))
    I = H5.CurveInterp(R)
    nu = np.exp(rng.uniform(math.log(0.005), math.log(200), 5))
    got = I(nu)
    for i in range(5):
        ref = PchipInterpolator(H5.LX, R[i])(np.log(np.clip(nu[i], *H5.NU_CLIP)))
        assert abs(got[i] - ref) <= 1e-12
    # exact at grid points
    assert np.allclose(I(np.full(5, H5.NU_E1[7])), R[:, 7], rtol=0, atol=1e-14)


def test_fit_law_recovers_constant():
    rng = np.random.default_rng(1)
    n = rng.uniform(200, 20000, 300)
    c_true, d = 3.4, 1408
    nu_star = c_true * (d / n) ** (2 / 3)
    R = 1 + 0.02 * (H5.LX[None, :] - np.log(nu_star)[:, None]) ** 2
    cg = dict(R=R, R_star=np.ones(len(n)), nu_star=nu_star, tokens=n, ntilde=n, interp=H5.CurveInterp(R),
              cluster=np.arange(len(n)))
    p = H5.fit_law(cg, "down", "expert", False)
    assert abs(p["c"] / c_true - 1) < 0.02
    pf = H5.fit_law(cg, "down", "expert", True)
    assert abs(pf["b"] - 2 / 3) < 0.03 and abs(pf["c"] / c_true - 1) < 0.05
