"""CPU tests for the legacy metric re-evaluation helpers (classifier, categories, fast E3, decomposition)."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import legacy_metric_reeval as LM            # noqa: E402
import report_legacy_metric_reeval as R      # noqa: E402
from moe_calib import agq as AG              # noqa: E402


def rows(de, dl, per_layer=None):
    out = []
    for i, (a, b) in enumerate(zip(de, dl)):
        pl = per_layer[i] if per_layer else {L: b for L in LM.LAYERS}
        out.append(dict(d_expert=a, d_layer=b, d_layer_per_layer=pl))
    return out


def test_reversal():
    assert R.classify(rows([.02, .02, .03], [-.01, -.006, -.02]))["verdict"] == "METRIC-RANKING-REVERSAL"


def test_confirmed_vs_small():
    assert R.classify(rows([.025] * 3, [.02, .015, .018]))["verdict"] == "LEGACY-NEGATIVE-CONFIRMED"
    assert R.classify(rows([.025] * 3, [.003, .004, .002]))["verdict"] == "METRIC-ALIGNED-BUT-SMALL"
    # 2/3 positive with no robust reversal still counts as "most"
    assert R.classify(rows([.025] * 3, [.02, .02, -.001]))["verdict"] == "LEGACY-NEGATIVE-CONFIRMED"


def test_mixed():
    assert R.classify(rows([.025] * 3, [.02, -.01, .01]))["verdict"] == "MIXED-OR-LAYER-SPECIFIC"
    assert R.classify(rows([.025] * 3, [-.001, -.002, .001]))["verdict"] == "MIXED-OR-LAYER-SPECIFIC"


def test_tie_branch():
    assert R.classify(rows([.001, -.001, .002], [-.01, -.006, -.02]))["verdict"] == "NEW-METRIC-GAIN"
    assert R.classify(rows([.001, -.001, .002], [.01, .006, .02]))["verdict"] == "NEW-METRIC-PENALTY"
    assert R.classify(rows([.001, -.001, .002], [.01, -.006, .02]))["verdict"] == "NO-ROBUST-DIFFERENCE"
    assert R.classify(rows([-.01] * 3, [.0, .0, .0]))["legacy_direction"] == "gain"


def test_layer_specific_flag():
    pl = [{1: -.02, 8: .03, 16: .01, 23: .01}] * 3
    c = R.classify(rows([.025] * 3, [.02] * 3, pl))
    assert c["layer_specific_reversals"] == [1]


def test_categories():
    assert R.category(.01, .01) == "worsens both"
    assert R.category(-.01, -.01) == "improves both"
    assert R.category(.01, -.01) == "worsens expert recon, improves MoE output"
    assert R.category(-.01, .01) == "improves expert recon, worsens MoE output"


def test_fast_e3_matches_production_formula():
    g = torch.Generator().manual_seed(0)
    W = {k: torch.randn(6, 5, generator=g) for k in ("gate_proj", "up_proj")}
    W["down_proj"] = torch.randn(5, 6, generator=g)
    Wq = {k: (v + 0.05 * torch.randn(v.shape, generator=g)).to(torch.bfloat16) for k, v in W.items()}
    HO = {(1, 0, "gu"): [(7, torch.randn(4, 5, generator=g).half()), (9, torch.randn(3, 5, generator=g).half())],
          (1, 0, "down"): [(7, torch.randn(4, 6, generator=g).half()), (9, torch.randn(3, 6, generator=g).half())]}
    fast = LM.e3_per_seq_fast(W, Wq, HO, 1, 0, "cpu")
    # independent reference of quant_eval.per_seq_rel_errors' formula
    ref = {}
    for sid in (7, 9):
        num = den = 0.0
        for proj, names in (("gu", ("gate_proj", "up_proj")), ("down", ("down_proj",))):
            X = dict(HO[(1, 0, proj)])[sid].float()
            for nm in names:
                num += float((X @ (W[nm] - Wq[nm].float()).T).pow(2).sum())
                den += float((X @ W[nm].T).pow(2).sum())
        ref[sid] = num / (den + 1e-12)
    assert set(fast) == {7, 9}
    for s in fast:
        assert fast[s] == pytest.approx(ref[s], rel=1e-6)


def test_decomposition_identity():
    g = torch.Generator().manual_seed(1)
    V = torch.randn(10, 4, 8, generator=g)
    d = AG.decompose(V)
    assert d["diag"] + d["cross"] == pytest.approx(d["total"], rel=1e-12)
    assert d["total"] == pytest.approx(float((V.double().sum(1) ** 2).sum()), rel=1e-12)


def test_registry_groups():
    reg = LM.methods_registry()
    assert sum(v["group"] == "B128" for v in reg.values()) == 12
    assert reg["EBSS"]["group"] == "B128_EBSS"
    assert all(reg[n]["group"] == "B16" for n in R.B16)
