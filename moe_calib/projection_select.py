"""Projection-selective AGQ: pure configuration / selection / decision helpers (no model code).

Two existing candidate quantizations per routed expert: q = 0 (standard GPTQ) and q = 1 (AGQ). A projection configuration
assembles, for EVERY expert of a layer, gate/up (as a pair) and down from one of the two candidates:

    GPTQ      gate/up q0, down q0          (A)
    GU_AGQ    gate/up q1, down q0          (B)
    DOWN_AGQ  gate/up q0, down q1          (C)
    FULL_AGQ  gate/up q1, down q1          (D)

No weights are created, interpolated or modified; every installed tensor is a candidate tensor, bit for bit.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

PROJ = ("gate_proj", "up_proj", "down_proj")
CONFIGS = {                         # name -> {projection: candidate q}
    "GPTQ": dict(gate_proj=0, up_proj=0, down_proj=0),
    "GU_AGQ": dict(gate_proj=1, up_proj=1, down_proj=0),
    "DOWN_AGQ": dict(gate_proj=0, up_proj=0, down_proj=1),
    "FULL_AGQ": dict(gate_proj=1, up_proj=1, down_proj=1),
}
ORDER = ("GPTQ", "GU_AGQ", "DOWN_AGQ", "FULL_AGQ")
TIE_ORDER = ("GPTQ", "FULL_AGQ", "GU_AGQ", "DOWN_AGQ")      # frozen tie-break preference
UNIFORM = ("GPTQ", "FULL_AGQ")
MIXED = ("GU_AGQ", "DOWN_AGQ")
SOURCE = {0: "GPTQ", 1: "AGQ"}


def build_projection_config(cands: dict, config_name: str, n_experts: int) -> tuple[dict, list]:
    """cands = {0: {"e.proj": tensor}, 1: {...}} for ONE layer. Returns (weights {e: {proj: tensor}}, manifest rows).
    Raises KeyError on any missing tensor (never falls back to BF16)."""
    if config_name not in CONFIGS:
        raise ValueError(f"unknown configuration {config_name}")
    src = CONFIGS[config_name]
    W, man = {}, []
    for e in range(n_experts):
        W[e] = {}
        for k in PROJ:
            key = f"{e}.{k}"
            q = src[k]
            if key not in cands[q]:
                raise KeyError(f"candidate {SOURCE[q]} lacks {key}")
            W[e][k] = cands[q][key]
            man.append(dict(expert=e, projection=k, source=SOURCE[q], q=q))
    return W, man


def manifest_hash(man: list) -> str:
    return hashlib.sha256(json.dumps(man, sort_keys=True).encode()).hexdigest()


def select_config(errors: dict) -> str:
    """argmin of validation E_layer over the four configurations; exact ties resolved by TIE_ORDER."""
    best = min(errors[c] for c in ORDER)
    return next(c for c in TIE_ORDER if errors[c] == best)


def layerwise_uniform(errors: dict) -> str:
    """Baseline E: GPTQ vs FULL_AGQ on validation, ties -> GPTQ."""
    return "FULL_AGQ" if errors["FULL_AGQ"] < errors["GPTQ"] else "GPTQ"


def rel(a: float, b: float) -> float:
    return (a - b) / b


def interactions(E: dict) -> dict:
    """2x2 GU x down analysis on any error metric (lower = better). Positive gains = improvement."""
    A, B, C, D = E["GPTQ"], E["GU_AGQ"], E["DOWN_AGQ"], E["FULL_AGQ"]
    I = D - B - C + A
    full = A - D
    out = dict(E=dict(E), G_GU_given_down_gptq=A - B, G_D_given_gu_gptq=A - C, G_GU_given_down_agq=C - D,
               G_D_given_gu_agq=B - D, interaction_I=I, full_agq_gain=full,
               rel=dict(G_GU_given_down_gptq=(A - B) / A, G_D_given_gu_gptq=(A - C) / A,
                        G_GU_given_down_agq=(C - D) / A, G_D_given_gu_agq=(B - D) / A, interaction_I=I / A,
                        full_agq_gain=full / A))
    # share of the full-AGQ gain attributable to each group (Shapley over the 2 groups) and to the interaction
    if full != 0:
        sh_gu = 0.5 * ((A - B) + (C - D))
        sh_d = 0.5 * ((A - C) + (B - D))
        out["shapley_share_of_full_gain"] = dict(GU=sh_gu / full, DOWN=sh_d / full)
        out["interaction_over_full_gain"] = I / full
    # I > 0: combining both is WORSE than additive (antagonistic); I < 0: better than additive (synergistic)
    out["interaction_sign"] = "additive" if I == 0 else ("antagonistic" if I > 0 else "synergistic")
    return out


def classify(rules: dict, val: dict, conf: dict, old: dict, selected: dict, layerwise: dict, correctness_ok: bool,
             expert_mixed_frac: dict) -> dict:
    """Frozen decision rule (protocol.json). val/conf/old = {layer: {config: E_layer}}; selected/layerwise =
    {layer: config}; expert_mixed_frac = {layer: fraction of experts whose confirm c^2 error under some mixed config is
    lower than under both uniform configs by >= rules['expert_rel']}."""
    if not correctness_ok:
        return dict(verdict="PROJECTION-AGQ-BLOCKED", criteria={})
    tau = rules["tau_rel"]
    Ls = sorted(conf)
    m = lambda d, pick: float(np.mean([d[L][pick[L]] for L in Ls]))
    mean_sel, mean_lw = m(conf, selected), m(conf, layerwise)
    mean_agq = float(np.mean([conf[L]["FULL_AGQ"] for L in Ls]))
    best_uni = {L: min(conf[L]["GPTQ"], conf[L]["FULL_AGQ"]) for L in Ls}
    r_lw, r_agq = rel(mean_sel, mean_lw), rel(mean_sel, mean_agq)
    r_old = rel(m(old, selected), m(old, layerwise))
    mixed_sel = [L for L in Ls if selected[L] in MIXED]
    val_beats = {L: selected[L] in MIXED and val[L][selected[L]] < min(val[L]["GPTQ"], val[L]["FULL_AGQ"]) for L in Ls}
    retains = {L: val_beats[L] and conf[L][selected[L]] < best_uni[L] for L in Ls}
    per_layer = {str(L): rel(conf[L][selected[L]], best_uni[L]) for L in Ls}
    crit = {
        "1_mixed_selected_beats_both_uniform_val_and_confirm": any(retains.values()),
        "2_selected_vs_layerwise_mean_le_minus_tau": r_lw <= -tau,
        "3_no_layer_regression_gt_tau_vs_best_uniform": max(per_layer.values()) <= tau,
        "4_oldtest_directionally_consistent": r_old < 0,
        "5_correctness": True,
        "6_gain_vs_uniform_agq_le_minus_tau": r_agq <= -tau,
    }
    mixed_conf_gain = {str(L): min(rel(conf[L][c], best_uni[L]) for c in MIXED) for L in Ls}
    big_effect = [L for L in Ls if abs(rel(conf[L]["FULL_AGQ"], conf[L]["GPTQ"])) >= tau]
    inter_frac = {str(L): abs(interactions(conf[L])["interaction_over_full_gain"]) for L in big_effect}
    interaction_material = (any(v >= rules["interaction_frac"] for v in inter_frac.values())
                            or any(v >= rules["expert_mixed_frac"] for v in expert_mixed_frac.values()))
    if all(crit.values()):
        v = "PROJECTION-AGQ-PROMISING"
    elif any(val_beats.values()) and not any(retains.values()):
        v = "PROJECTION-AGQ-SELECTION-FAIL"
    elif any(g <= -tau for g in mixed_conf_gain.values()):
        v = "PROJECTION-AGQ-LOCAL-GAIN"
    elif interaction_material:
        v = "PROJECTION-AGQ-INTERACTION-ONLY"
    else:
        v = "PROJECTION-AGQ-NO-GAIN"
    return dict(verdict=v, criteria=crit, mean_selected=mean_sel, mean_layerwise=mean_lw, mean_full_agq=mean_agq,
                rel_vs_layerwise=r_lw, rel_vs_full_agq=r_agq, oldtest_rel_vs_layerwise=r_old,
                per_layer_rel_vs_best_uniform=per_layer, mixed_selected_layers=mixed_sel,
                val_mixed_beats_uniform=val_beats, confirm_retains=retains,
                best_mixed_rel_vs_best_uniform_confirm=mixed_conf_gain,
                interaction_over_full_gain_confirm=inter_frac, expert_mixed_frac=expert_mixed_frac,
                interaction_material=bool(interaction_material))


RECOMMEND = {"PROJECTION-AGQ-PROMISING": "A", "PROJECTION-AGQ-NO-GAIN": "B", "PROJECTION-AGQ-SELECTION-FAIL": "B",
             "PROJECTION-AGQ-BLOCKED": "B", "PROJECTION-AGQ-LOCAL-GAIN": "C", "PROJECTION-AGQ-INTERACTION-ONLY": "C"}
