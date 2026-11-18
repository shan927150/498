"""CPU tests for the Fixed GU-AGQ robustness study (decision rule, fixed configuration, attribution identities)."""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
import fixed_gu_agq_validation as FG          # noqa: E402
from moe_calib import projection_select as PS  # noqa: E402

R = FG.RULES


def lay(v):
    return {1: v[0], 8: v[1], 16: v[2], 23: v[3]}


def test_fixed_gu_config_sources():
    assert PS.CONFIGS["GU_AGQ"] == dict(gate_proj=1, up_proj=1, down_proj=0)
    g = torch.Generator().manual_seed(0)
    c = {q: {f"{e}.{k}": torch.randn(3, 3, generator=g) for e in range(60) for k in PS.PROJ} for q in (0, 1)}
    W, man = PS.build_projection_config(c, "GU_AGQ", 60)
    assert len(W) == 60 and len(man) == 180
    for e in range(60):
        assert W[e]["gate_proj"] is c[1][f"{e}.gate_proj"] and W[e]["up_proj"] is c[1][f"{e}.up_proj"]
        assert W[e]["down_proj"] is c[0][f"{e}.down_proj"]
    assert PS.manifest_hash(man) == PS.manifest_hash(PS.build_projection_config(c, "GU_AGQ", 60)[1])


def test_config_sets():
    assert FG.PRIMARY_CONFIGS == ("GPTQ", "FULL_AGQ", "GU_AGQ") and FG.DIAG_CONFIGS == ("DOWN_AGQ",)
    assert FG.LAYERS == [1, 8, 16, 23] and FG.SEEDS == (0, 1, 2)


def test_shapley_sums_to_full_gain():
    E = dict(GPTQ=10.0, GU_AGQ=9.1, DOWN_AGQ=10.3, FULL_AGQ=9.5)
    it = PS.interactions(E)
    sh = it["shapley_share_of_full_gain"]
    assert sh["GU"] + sh["DOWN"] == pytest.approx(1.0)
    phi_gu = 0.5 * ((E["GPTQ"] - E["GU_AGQ"]) + (E["DOWN_AGQ"] - E["FULL_AGQ"]))
    phi_d = 0.5 * ((E["GPTQ"] - E["DOWN_AGQ"]) + (E["GU_AGQ"] - E["FULL_AGQ"]))
    assert phi_gu + phi_d == pytest.approx(E["GPTQ"] - E["FULL_AGQ"])


def test_classify_promising():
    v = FG.classify(R, {0: -0.01, 1: -0.006, 2: -0.008}, {0: -0.004, 1: -0.002, 2: -0.001},
                    lay([-0.01, -0.01, 0.001, -0.005]), True, 1.0)
    assert v["verdict"] == "FIXED-GU-PROMISING" and v["recommendation"] == "A"


def test_classify_cost_blocks_promising():
    v = FG.classify(R, {0: -0.01, 1: -0.006, 2: -0.008}, {0: -0.004, 1: -0.002, 2: -0.001},
                    lay([-0.01, -0.01, 0.001, -0.005]), True, 2.0)
    assert v["verdict"] != "FIXED-GU-PROMISING"


def test_classify_small_gain():
    v = FG.classify(R, {0: -0.002, 1: -0.001, 2: -0.0015}, {0: -0.001, 1: -0.001, 2: 0.0},
                    lay([-0.002, -0.003, 0.0008, -0.001]), True, 1.0)
    assert v["verdict"] == "FIXED-GU-SMALL-GAIN" and v["recommendation"] == "C"


def test_classify_unstable():
    v = FG.classify(R, {0: -0.006, 1: 0.002, 2: -0.001}, {0: 0, 1: 0, 2: 0}, lay([0, 0, 0, 0]), True, 1.0)
    assert v["verdict"] == "FIXED-GU-UNSTABLE" and v["recommendation"] == "B"


def test_classify_layer_specific_and_no_gain():
    v = FG.classify(R, {0: -0.001, 1: 0.001, 2: 0.0005}, {0: 0.0, 1: 0.0, 2: 0.0},
                    lay([-0.01, 0.0, 0.008, 0.0]), True, 1.0)
    assert v["verdict"] == "FIXED-GU-LAYER-SPECIFIC" and v["recommendation"] == "B"
    v = FG.classify(R, {0: 0.001, 1: 0.002, 2: 0.0015}, {0: 0.001, 1: 0.001, 2: 0.001},
                    lay([0.001, 0.002, 0.001, 0.002]), True, 1.0)
    assert v["verdict"] == "FIXED-GU-NO-GAIN"


def test_classify_blocked():
    assert FG.classify(R, {0: -1}, {0: -1}, lay([0, 0, 0, 0]), False, 1.0)["verdict"] == "FIXED-GU-BLOCKED"
