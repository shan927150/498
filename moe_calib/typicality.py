"""Routing typicality of candidate sequences (11/06 follow-up; CPU only).

From the full-pool routing matrix C (rows = ORIGINAL pool ids in cache order,
columns layer-major j = li * E + e, as in routing_profile_v2.npz):

  p(s, L, .)    = C[s, L, .] / sum_e C[s, L, e]            per-sequence routing
  p(pool, L, .) = sum_s C[s, L, .] / sum_s,e C[s, L, e]     ORIGINAL pool reference
  d(s, L)       = || p(s, L, .) - p(pool, L, .) ||_1
  d(s)          = mean_L d(s, L)        (equal layer weights; fixed before any run)

Secondary (never used to build pools): JS divergence to the pool and routing
entropy, both averaged over layers. A row with zero routed tokens in any
layer is malformed: its scores are NaN and it is excluded from the typical
pool (counted in the diagnostics), never silently imputed.
"""
from __future__ import annotations
import random

import numpy as np


def _layer_blocks(C, n_layers, E):
    C = np.asarray(C, float)
    assert C.shape[1] == n_layers * E, f"C has {C.shape[1]} columns, expected {n_layers} x {E}"
    return [C[:, li * E:(li + 1) * E] for li in range(n_layers)]


def pool_reference(C, n_layers, E):
    """[n_layers, E] routing distribution of the whole matrix (the original pool)."""
    out = []
    for X in _layer_blocks(C, n_layers, E):
        tot = X.sum()
        if tot <= 0:
            raise ValueError("a layer has no routed tokens in the reference pool")
        out.append(X.sum(0) / tot)
    return np.stack(out)


def _kl(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(a > 0, a * np.log(a / b), 0.0)
    return t.sum(-1)


def sequence_scores(C, n_layers, E, ref=None):
    """Per-sequence d(s, L), d(s), JS and entropy. `ref` defaults to the
    distribution of C itself; pass the ORIGINAL pool reference when scoring a
    subset. Returns a dict of arrays (rows aligned with C)."""
    ref = pool_reference(C, n_layers, E) if ref is None else np.asarray(ref, float)
    blocks = _layer_blocks(C, n_layers, E)
    S = blocks[0].shape[0]
    dL = np.full((S, n_layers), np.nan)
    js = np.full((S, n_layers), np.nan)
    ent = np.full((S, n_layers), np.nan)
    malformed = np.zeros(S, bool)
    for li, X in enumerate(blocks):
        tot = X.sum(1)
        ok = tot > 0
        malformed |= ~ok
        q = np.zeros_like(X)
        q[ok] = X[ok] / tot[ok, None]
        r = ref[li][None, :]
        m = 0.5 * (q + r)
        dL[ok, li] = np.abs(q[ok] - r).sum(1)
        js[ok, li] = (0.5 * _kl(q, m) + 0.5 * _kl(np.broadcast_to(r, q.shape), m))[ok]
        with np.errstate(divide="ignore", invalid="ignore"):
            ent[ok, li] = -np.where(q > 0, q * np.log(q), 0.0).sum(1)[ok]
    d = dL.mean(1)
    d[malformed] = np.nan
    return dict(d=d, d_layer=dL, js=js.mean(1), entropy=ent.mean(1), malformed=malformed)


def typical_half(ids, d, k):
    """The k ids with the smallest d; ties broken by ORIGINAL id; NaN excluded."""
    ids = np.asarray(ids)
    ok = np.isfinite(d)
    order = sorted((float(d[i]), int(ids[i])) for i in np.flatnonzero(ok))
    if len(order) < k:
        raise ValueError(f"only {len(order)} well-formed sequences for a pool of {k}")
    return sorted(s for _, s in order[:k])


def random_half(ids, k, seed):
    """k ids uniformly at random with an independent pool-construction seed."""
    return sorted(random.Random(seed).sample(sorted(int(s) for s in ids), k))


def selected_set_js(C_rows, ref, n_layers, E):
    """JS divergence (nats) between a selected set's aggregate routing and the
    reference, averaged over layers."""
    agg = pool_reference(C_rows, n_layers, E)
    m = 0.5 * (agg + ref)
    return float(np.mean(0.5 * _kl(agg, m) + 0.5 * _kl(ref, m)))
