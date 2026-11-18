"""CPU tests for the Family D frozen-routing driver helpers (weight differences, design, nested statistics)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diagnostics"))
import grouping_routing_amplification as G                      # noqa: E402
from moe_calib import cross_seed_grouping as CS                # noqa: E402


def _w(seed):
    g = torch.Generator().manual_seed(seed)
    return {(L, e, k): torch.randn(4, 8, generator=g).to(torch.bfloat16)
            for L in G.LAYERS_Q for e in range(2) for k in G.PROJ}


def test_design_is_15_pairs_nested_in_3_seeds():
    assert len(G.CONFIGS) == 30 and len(set(G.CONFIGS)) == 30
    assert set(G.PARITY_CONFIGS) <= set(G.CONFIGS)
    assert G.result_src("FR", 0, "G20261111") == "runs_multigroup_robustness/results/FR_G20261111.json"
    assert G.result_src("FC", 2, "G20261115") == "runs_cross_seed_grouping/results/FC_s2_G20261115.json"


def test_weight_diff_identical_is_zero():
    A = _w(0)
    d = G.weight_diff(A, {k: v.clone() for k, v in A.items()})
    assert d["frac_changed"] == 0 and d["rel_frobenius"] == 0 and d["mean_abs_diff"] == 0


def test_weight_diff_counts_and_ulps():
    A = _w(0)
    B = {k: v.clone() for k, v in A.items()}
    key = sorted(B)[0]
    t = B[key].view(torch.int16).clone()
    t[0, 0] += 1                                                    # 1 ULP (same sign)
    t[0, 1] += 2                                                    # 2 ULP
    t[0, 2] += 20                                                   # >= 8 ULP
    B[key] = t.view(torch.bfloat16)
    d = G.weight_diff(A, B)
    n = sum(v.numel() for v in A.values())
    assert d["n_changed"] == 3 and abs(d["frac_changed"] - 3 / n) < 1e-15
    u = d["ulp_share_of_changed"]
    assert abs(u["1"] - 1 / 3) < 1e-12 and abs(u["2"] - 1 / 3) < 1e-12 and abs(u[">=8"] - 1 / 3) < 1e-12
    assert d["rel_frobenius"] > 0


def test_nested_pooled_sd_definition_matches_historical():
    rng = np.random.default_rng(0)
    D = {s: {g: float(rng.normal()) for g in G.GROUPINGS} for s in G.SEEDS}
    v = CS.variance_summary(D)
    pooled = np.sqrt(np.mean([np.std(list(D[s].values()), ddof=1) ** 2 for s in G.SEEDS]))
    assert abs(v["pooled_within_sd"] - pooled) < 1e-15
