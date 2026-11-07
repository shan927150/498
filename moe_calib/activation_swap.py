"""Activation-aware single-sequence swap pilot (11/07; exploratory, layer 16).

Expert-specific diagonal second moments of the gate/up-projection INPUT,
accumulated per (sequence, expert) from the tokens actually routed to that
expert (Qwen2MoeSparseMoeBlock feeds expert e only hidden_states[top_x]; the
existing Profiler pre-hook on gate_proj sees exactly those rows):

    n[s,e]   = routed tokens of sequence s at expert e
    g[s,e,j] = sum over those tokens of x_j^2          (FP32 accumulation)
    h(S)[e]  = sum_{s in S} g[s,e] / sum_{s in S} n[s,e]   (token-weighted)

Mismatch (prespecified):
    D_e(S) = ||h(S)[e] - h_ref[e]||_2^2 / (||h_ref[e]||_2^2 + 1e-12),  D(S) = mean_e D_e(S)
    predicted_gain(swap) = D(S0) - D(S0 - out + in)        (> 0 = better matching)
Swaps are evaluated by additive updates of the sufficient statistics
(G - g[out] + g[in], N - n[out] + n[in]).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from .profiler import Profiler


class DiagProfiler(Profiler):
    """Profiler that, for the chosen experts of one layer, accumulates per
    sequence the routed-token count and the per-coordinate sum of squares of
    the gate/up input. No Hessians, no token storage."""

    def __init__(self, model, layer, experts):
        super().__init__(model, layer, collect_hessian=False)
        self.experts = [int(e) for e in experts]
        self.diag: dict[tuple[int, int], torch.Tensor] = {}
        self.ntok: dict[tuple[int, int], int] = {}

    def _expert_hook(self, e, proj):
        if proj != "gu" or e not in self.experts:
            return lambda module, args: None

        def fn(module, args):
            X = args[0].reshape(-1, args[0].shape[-1])
            if X.shape[0] == 0:
                return
            k = (self.cur_seq, e)
            sq = X.to(torch.float32).pow(2).sum(0).double().cpu()
            self.diag[k] = self.diag.get(k, 0) + sq
            self.ntok[k] = self.ntok.get(k, 0) + int(X.shape[0])
        return fn

    def arrays(self, seq_ids, d):
        """(n [S, E'], g [S, E', d]) in float64, rows in `seq_ids` order."""
        n = np.zeros((len(seq_ids), len(self.experts)))
        g = np.zeros((len(seq_ids), len(self.experts), d))
        for i, s in enumerate(seq_ids):
            for k, e in enumerate(self.experts):
                if (s, e) in self.ntok:
                    n[i, k] = self.ntok[(s, e)]
                    g[i, k] = self.diag[(s, e)].numpy()
        return n, g


def set_moments(n, g, rows):
    """h(S) [E', d] from per-sequence sums (token-weighted)."""
    N = n[rows].sum(0)
    G = g[rows].sum(0)
    if np.any(N <= 0):
        raise ValueError("an expert has no routed tokens in the set")
    return G / N[:, None], N


def mismatch(h, href):
    """(D, D_e) for h, href [E', d]."""
    De = ((h - href) ** 2).sum(1) / ((href ** 2).sum(1) + 1e-12)
    return float(De.mean()), De


def swap_mismatch(Nsum, Gsum, n_out, g_out, n_in, g_in, href):
    """D after removing one sequence and adding one, via additive updates."""
    N = Nsum - n_out + n_in
    G = Gsum - g_out + g_in
    if np.any(N <= 0):
        raise ValueError("swap leaves an expert without routed tokens")
    return mismatch(G / N[:, None], href)


def top_experts(C, layer_index, E, k):
    """k most-routed experts of one layer over the ORIGINAL pool (ties -> lower id)."""
    tot = np.asarray(C, float)[:, layer_index * E:(layer_index + 1) * E].sum(0)
    order = sorted(range(E), key=lambda e: (-tot[e], e))
    return order[:k], tot


def ids_hash(ids):
    return hashlib.sha256(json.dumps([int(s) for s in ids]).encode()).hexdigest()[:16]


def swapped(S0, out, inn):
    """S0 with `out` replaced IN PLACE by `inn` (order otherwise unchanged, so the
    Profiler's contiguous block assignment differs only at that position)."""
    if out not in S0 or inn in S0:
        raise ValueError("invalid swap")
    S = [inn if s == out else s for s in S0]
    assert len(S) == len(set(S)) == len(S0) and len(set(S) ^ set(S0)) == 2
    return S


def pick_unique(ranked, k, used_out, used_in):
    """First k proposals of `ranked` whose out/in ids are unused (preferred);
    if fewer exist, fill from the rest in rank order (recorded by caller)."""
    take = []
    for p in ranked:
        if len(take) >= k:
            break
        if p["out"] not in used_out and p["in"] not in used_in:
            take.append(p)
            used_out.add(p["out"]); used_in.add(p["in"])
    unique = len(take)
    for p in ranked:
        if len(take) >= k:
            break
        if p not in take:
            take.append(p)
    return take, unique
