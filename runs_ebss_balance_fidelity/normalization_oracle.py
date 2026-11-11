"""Independent CPU float64 oracle for the four MoEQuant balance normalisations (balance-strength fidelity study).

Written from the source equations, not from moe_calib.ebss_generator (never imported here; no torch):
  usage   rate   u_e = c_e / n                      share  u_e = c_e / (k n)
  std     Eq. 4  sigma_l = sqrt( 1/(E-1) * sum_e (u_e - mean)^2 )
          Alg.1  sigma_l = sqrt( E/(E-1) * sum_e (u_e - mean)^2 )      (line 30)
  sigma = (1/L) sum_l sigma_l   (Eq. 3; every layer uses its own E_l)
  score J(S||v) = -(L(S) + log P(v|S))/(|S|+1) + sigma(S)/tau, minimised; ties by (parent, token).
"""
from __future__ import annotations

import math

import numpy as np

VARIANTS = {"EQ4_RATE": ("eq4", "per_token"), "EQ4_SHARE": ("eq4", "assignment_share"),
            "ALG1_SHARE": ("algorithm1", "assignment_share"), "ALG1_RATE": ("algorithm1", "per_token")}


def layer_sigma(counts_row, n_tokens, std_mode, usage_mode, k):
    E = len(counts_row)
    denom = n_tokens * k if usage_mode == "assignment_share" else n_tokens
    u = [float(c) / denom for c in counts_row]
    mu = math.fsum(u) / E
    ss = math.fsum((x - mu) ** 2 for x in u)
    return math.sqrt(ss / (E - 1)) if std_mode == "eq4" else math.sqrt(E * ss / (E - 1))


def sigma(counts, n_tokens, std_mode="eq4", usage_mode="per_token", k=4):
    """counts: list of per-layer count vectors (layers may differ in expert count)."""
    per = [layer_sigma(c, n_tokens, std_mode, usage_mode, k) for c in counts]
    return math.fsum(per) / len(per)


def sigma_variant(counts, n_tokens, variant, k=4):
    s, u = VARIANTS[variant]
    return sigma(counts, n_tokens, s, u, k)


def analytic_ratio(variant, E, k=4):
    """sigma_variant / sigma_EQ4_RATE for a single layer with E experts."""
    s, u = VARIANTS[variant]
    return (math.sqrt(E) if s == "algorithm1" else 1.0) * (1.0 / k if u == "assignment_share" else 1.0)


def scores(L_parent, logp, sigma_parent, prefix_len, tau, use_balance=True):
    Lp = np.asarray(L_parent, np.float64)[:, None]
    s = (-Lp - np.asarray(logp, np.float64)) / (prefix_len + 1)
    return s + np.asarray(sigma_parent, np.float64)[:, None] / tau if use_balance else s


def rank(pool_scores, parents, tokens, w):
    """Global top-w smallest over an explicit candidate list; ties by (parent, token)."""
    order = np.lexsort((np.asarray(tokens), np.asarray(parents), np.asarray(pool_scores, np.float64)))
    return order[:w], order
