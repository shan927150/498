"""Independent reference oracle for the MoEQuant EBSS candidate score (audit of moe_calib/ebss_generator.py).

Written from the paper equations, NOT from the production code:
  Eq. 11  L(S) = sum_i log P(s_i | s_<i)                       (cumulative log-probability, <= 0)
  Eq. 13  J(S||v) = -(L(S) + log P(v|S)) / (|S| + 1) + sigma(M, S) / tau      (minimised)
          equivalently with N(S) = -L(S) >= 0:  J = (N(S) - log P(v|S)) / (|S| + 1) + sigma / tau
  Eq. 3/4 sigma = (1/L) sum_l sqrt( 1/(E-1) sum_e (u_l^e - mean_l u)^2 )
  Eq. 14  keep the w best (smallest J) expansions over ALL parents x tokens
  Alg. 1  final sentence: best of -L(c)/|c| + sigma(c)/tau over the surviving branches

Rules: pure Python / NumPy float64 only; never imports torch or moe_calib.ebss_generator; cumulative values are
re-summed from per-token lists with math.fsum; ranking uses np.lexsort / sorted(). Deliberately WRONG variants
(names start with `wrong_`) exist only to prove that the audit tests can tell them apart from the correct score.
"""
from __future__ import annotations

import math

import numpy as np


# ------------------------------------------------------------------ probabilities
def log_softmax64(logits) -> np.ndarray:
    """float64 log-softmax over the last axis (max-shifted log-sum-exp)."""
    x = np.asarray(logits, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def cumulative_logp(token_logps) -> float:
    """Eq. 11 from an explicit per-token list (exact fsum)."""
    return math.fsum(float(v) for v in token_logps)


# ------------------------------------------------------------------ expert balance
def counts_from_routing(routing, n_experts: int) -> np.ndarray:
    """routing [T, L, k] routed expert ids of every prefix token -> counts [L, E] (each selected expert +1)."""
    r = np.asarray(routing, dtype=np.int64)
    T, L, k = r.shape
    c = np.zeros((L, n_experts), dtype=np.int64)
    for t in range(T):
        for l in range(L):
            for e in r[t, l]:
                c[l, e] += 1
    return c


def sigma_eq4(counts, n_tokens: int) -> float:
    """Eq. 3/4 with u = count / n_tokens, explicit loops, ddof = 1, mean over layers."""
    c = np.asarray(counts, dtype=np.float64)
    L, E = c.shape
    per_layer = []
    for l in range(L):
        u = [c[l, e] / n_tokens for e in range(E)]
        mu = math.fsum(u) / E
        per_layer.append(math.sqrt(math.fsum((x - mu) ** 2 for x in u) / (E - 1)))
    return math.fsum(per_layer) / L


# ------------------------------------------------------------------ scores (scalar, explicit)
def score_logp(L_S: float, p_v: float, prefix_len: int, sigma: float, tau: float, use_balance=True) -> float:
    s = -(L_S + p_v) / (prefix_len + 1)
    return s + sigma / tau if use_balance else s


def score_nll(N_S: float, p_v: float, prefix_len: int, sigma: float, tau: float, use_balance=True) -> float:
    s = (N_S - p_v) / (prefix_len + 1)
    return s + sigma / tau if use_balance else s


def wrong_score_nll_sign(N_S: float, p_v: float, prefix_len: int, sigma: float, tau: float, use_balance=True):
    """The suspected bug: positive cumulative NLL plugged into the log-probability formula."""
    s = -(N_S + p_v) / (prefix_len + 1)
    return s + sigma / tau if use_balance else s


def final_score(L_c: float, length: int, sigma: float, tau: float, use_balance=True) -> float:
    s = -L_c / length
    return s + sigma / tau if use_balance else s


# ------------------------------------------------------------------ vectorised candidate pool
VARIANTS = ("correct", "wrong_nll_sign", "wrong_reversed_sigma", "wrong_tau_multiplies")


def candidate_matrix(L_parent, logp, sigma_parent, prefix_len, tau, use_balance=True, variant="correct"):
    """Full [B, V] score matrix.  L_parent [B] cumulative logP, logp [B, V] next-token logP, sigma_parent [B]."""
    Lp = np.asarray(L_parent, np.float64)[:, None]
    lp = np.asarray(logp, np.float64)
    sg = np.asarray(sigma_parent, np.float64)[:, None]
    if variant == "wrong_nll_sign":
        prob = -((-Lp) + lp) / (prefix_len + 1)
    else:
        prob = (-Lp - lp) / (prefix_len + 1)               # written in NLL form on purpose (independent algebra)
    if not use_balance:
        return prob
    if variant == "wrong_reversed_sigma":
        return prob - sg / tau
    if variant == "wrong_tau_multiplies":
        return prob + sg * tau
    return prob + sg / tau


def rank_global(scores, w: int, maximise=False):
    """Global top-w over all (parent, token): smallest score first, ties by (parent, token)."""
    s = np.asarray(scores, np.float64)
    B, V = s.shape
    par = np.repeat(np.arange(B), V)
    tok = np.tile(np.arange(V), B)
    key = -s.reshape(-1) if maximise else s.reshape(-1)
    order = np.lexsort((tok, par, key))[:w]
    return [(int(par[i]), int(tok[i]), float(s.reshape(-1)[i])) for i in order]


def wrong_rank_per_parent(scores, w: int):
    """Deliberately wrong: keep the best ceil(w/B) of EACH parent (local instead of global pruning)."""
    s = np.asarray(scores, np.float64)
    B, V = s.shape
    per = -(-w // B)
    out = []
    for b in range(B):
        for t in sorted(range(V), key=lambda t: (s[b, t], t))[:per]:
            out.append((b, t, float(s[b, t])))
    return sorted(out, key=lambda x: (x[2], x[0], x[1]))[:w]


def final_select(L_c, sigma_c, length, tau, use_balance=True) -> int:
    vals = [final_score(float(a), length, float(b), tau, use_balance) for a, b in zip(L_c, sigma_c)]
    return min(range(len(vals)), key=lambda i: (vals[i], i))


def kept_gap(scores, kept, w):
    """Smallest score difference between the w-th kept and the best rejected candidate (tie diagnostics)."""
    s = np.asarray(scores, np.float64).reshape(-1)
    V = np.asarray(scores).shape[1]
    kept_idx = {p * V + t for p, t, _ in kept}
    worst_kept = max(s[i] for i in kept_idx)
    mask = np.ones_like(s, bool); mask[list(kept_idx)] = False
    return float(s[mask].min() - worst_kept) if mask.any() else float("inf")
