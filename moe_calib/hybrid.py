"""Expert-wise GPTQ/AGQ hybrid: pure selection / decision helpers (no model code).

Each routed expert e has two existing candidate quantizations: q = 0 (standard GPTQ, unweighted Hessian) and q = 1
(AGQ, router-affinity-weighted Hessian). The hybrid installs, for every expert, ALL three projections of exactly one
candidate z_e in {0, 1}, chosen on a held-out tuning split by the diagonal c^2-weighted full-expert output error

    J_e^(q) = sum_{i: e in topk(i)} c_{i,e}^2 || E_e^FP(x_i) - E_e^(q)(x_i) ||^2 .

Frozen rule: z_e = 1 iff J_e^(1) < J_e^(0) (ties -> GPTQ).
"""
from __future__ import annotations

import hashlib

import numpy as np

CANDIDATES = {0: "GPTQ", 1: "AGQ"}


# ----------------------------------------------------------------------------- selection
def select_experts(J0, J1) -> np.ndarray:
    """Frozen rule: AGQ (1) only if strictly lower c^2-weighted validation error; ties and missing -> GPTQ (0)."""
    J0, J1 = np.asarray(J0, dtype=np.float64), np.asarray(J1, dtype=np.float64)
    assert J0.shape == J1.shape
    z = (J1 < J0) & np.isfinite(J0) & np.isfinite(J1)
    return z.astype(np.int64)


def rel_improvement(J0, J1) -> np.ndarray:
    """(J0 - J1) / J0: positive = AGQ better. NaN where J0 == 0."""
    J0, J1 = np.asarray(J0, dtype=np.float64), np.asarray(J1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(J0 > 0, (J0 - J1) / J0, np.nan)


def layerwise_choice(e_gptq: float, e_agq: float) -> int:
    """Validation-selected layer-wise uniform baseline: all-AGQ iff strictly lower actual layer-output error."""
    return int(e_agq < e_gptq)


def config_vector(kind: str, n: int, z_hybrid=None, layer_choice: int | None = None) -> np.ndarray:
    if kind == "GPTQ":
        return np.zeros(n, dtype=np.int64)
    if kind == "AGQ":
        return np.ones(n, dtype=np.int64)
    if kind == "LAYERWISE":
        assert layer_choice in (0, 1)
        return np.full(n, layer_choice, dtype=np.int64)
    if kind == "HYBRID":
        z = np.asarray(z_hybrid, dtype=np.int64)
        assert z.shape == (n,) and set(np.unique(z)) <= {0, 1}
        return z
    raise ValueError(kind)


def config_hash(z, cand_hashes: list[dict]) -> str:
    """Identity of an installed configuration: sha256 over the chosen candidate hash of every expert, in order.
    cand_hashes[e] = {0: sha, 1: sha}."""
    h = hashlib.sha256()
    for e, q in enumerate(np.asarray(z, dtype=np.int64)):
        h.update(cand_hashes[e][int(q)].encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------- stability / transfer
def half_split_stability(Jseq0: np.ndarray, Jseq1: np.ndarray) -> dict:
    """Jseq*: [n_seq, n_experts] per-sequence c^2-weighted numerators on the tuning split. Selections are recomputed on
    the two non-overlapping contiguous halves and compared (diagnostic only)."""
    n = Jseq0.shape[0]
    a, b = slice(0, n // 2), slice(n // 2, n)
    za = select_experts(Jseq0[a].sum(0), Jseq1[a].sum(0))
    zb = select_experts(Jseq0[b].sum(0), Jseq1[b].sum(0))
    zf = select_experts(Jseq0.sum(0), Jseq1.sum(0))
    ra = rel_improvement(Jseq0[a].sum(0), Jseq1[a].sum(0))
    rb = rel_improvement(Jseq0[b].sum(0), Jseq1[b].sum(0))
    ok = np.isfinite(ra) & np.isfinite(rb)
    return dict(n_seq_half=[n // 2, n - n // 2], agreement_halves=float((za == zb).mean()),
                agreement_half_a_vs_full=float((za == zf).mean()), agreement_half_b_vs_full=float((zb == zf).mean()),
                n_agq_half_a=int(za.sum()), n_agq_half_b=int(zb.sum()), n_agq_full=int(zf.sum()),
                pearson_rel_improvement_halves=_corr(ra[ok], rb[ok]),
                spearman_rel_improvement_halves=_corr(_rank(ra[ok]), _rank(rb[ok])))


def _rank(x):
    x = np.asarray(x, dtype=np.float64)
    r = np.empty_like(x)
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def _corr(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def transfer_metrics(val_J0, val_J1, test_J0, test_J1, z) -> dict:
    """Validation -> test generalization of per-expert preferences (evaluation only, never used to reselect)."""
    val_J0, val_J1, test_J0, test_J1 = (np.asarray(x, dtype=np.float64) for x in (val_J0, val_J1, test_J0, test_J1))
    z = np.asarray(z, dtype=np.int64)
    zt = select_experts(test_J0, test_J1)
    rv, rt = rel_improvement(val_J0, val_J1), rel_improvement(test_J0, test_J1)
    ok = np.isfinite(rv) & np.isfinite(rt)
    chosen = np.where(z == 1, test_J1, test_J0)
    best = np.minimum(test_J0, test_J1)
    regret = chosen - best
    false_sel = (regret > 0)
    # margin terciles on |validation relative improvement|
    m = np.abs(rv)
    terc = {}
    if ok.sum() >= 3:
        qs = np.quantile(m[ok], [1 / 3, 2 / 3])
        bins = np.digitize(m, qs)
        for t, nm in enumerate(("small", "medium", "large")):
            sel = ok & (bins == t)
            terc[nm] = dict(n=int(sel.sum()), sign_agreement=float((z[sel] == zt[sel]).mean()) if sel.any() else None,
                            false_selections=int(false_sel[sel].sum()),
                            regret_share=float(regret[sel].sum() / regret.sum()) if regret.sum() > 0 else 0.0,
                            abs_val_margin_max=float(m[sel].max()) if sel.any() else None)
    return dict(
        n=int(len(z)), sign_agreement=float((z == zt).mean()),
        n_val_agq=int(z.sum()), n_test_agq=int(zt.sum()),
        pearson_rel_improvement=_corr(rv[ok], rt[ok]), spearman_rel_improvement=_corr(_rank(rv[ok]), _rank(rt[ok])),
        false_selections=int(false_sel.sum()),
        regret_sum=float(regret.sum()), oracle_gain_vs_gptq=float((test_J0 - best).sum()),
        realized_gain_vs_gptq=float((test_J0 - chosen).sum()),
        regret_rel_to_test_J0=float(regret.sum() / test_J0.sum()),
        realized_rel_gain_vs_gptq=float((test_J0 - chosen).sum() / test_J0.sum()),
        oracle_rel_gain_vs_gptq=float((test_J0 - best).sum() / test_J0.sum()),
        realized_rel_gain_vs_agq=float((test_J1 - chosen).sum() / test_J1.sum()),
        margin_terciles=terc)


# ----------------------------------------------------------------------------- decision
def rel(a: float, b: float) -> float:
    """(a - b) / b; negative = a lower (better) error than b."""
    return (a - b) / b


def classify(rules: dict, layer_err: dict, transfer: dict, correctness_ok: bool) -> dict:
    """Frozen decision rule.
    layer_err = {layer: {"GPTQ": e, "AGQ": e, "LAYERWISE": e, "HYBRID": e}} on the PRIMARY split.
    transfer  = {"sign_agreement": float, "spearman_rel_improvement": float} pooled over all experts (primary split).
    """
    if not correctness_ok:
        return dict(verdict="HYBRID-BLOCKED", criteria={})
    tau = rules["tau_rel"]
    Ls = sorted(layer_err)
    mean = {m: float(np.mean([layer_err[L][m] for L in Ls])) for m in ("GPTQ", "AGQ", "LAYERWISE", "HYBRID")}
    r_g, r_a, r_l = rel(mean["HYBRID"], mean["GPTQ"]), rel(mean["HYBRID"], mean["AGQ"]), rel(mean["HYBRID"],
                                                                                              mean["LAYERWISE"])
    strongest = min(("GPTQ", "AGQ", "LAYERWISE"), key=lambda m: mean[m])
    r_s = rel(mean["HYBRID"], mean[strongest])
    per_layer = {str(L): rel(layer_err[L]["HYBRID"], min(layer_err[L]["GPTQ"], layer_err[L]["AGQ"])) for L in Ls}
    transfer_ok = bool(transfer["sign_agreement"] >= rules["min_sign_agreement"]
                       and transfer["spearman_rel_improvement"] >= rules["min_spearman"])
    crit = {
        "1_beats_both_uniform": bool(r_g < 0 and r_a < 0),
        "2_beats_layerwise": bool(r_l < 0),
        "3_margin_vs_strongest_nonhybrid": bool(r_s <= -tau),
        "4_no_layer_regression_vs_best_uniform": bool(max(per_layer.values()) <= tau),
        "5_preferences_transfer": transfer_ok,
        "6_correctness": True,
    }
    if all(crit.values()):
        v = "HYBRID-PROMISING"
    elif not transfer_ok and min(r_g, r_a) > -tau:
        v = "HYBRID-SELECTION-FAIL"
    elif r_g <= -tau and r_a <= -tau and not crit["3_margin_vs_strongest_nonhybrid"]:
        v = "HYBRID-LOCAL-ONLY"
    else:
        v = "HYBRID-NO-GAIN"
    return dict(verdict=v, criteria=crit, mean=mean, rel_vs_gptq=r_g, rel_vs_agq=r_a, rel_vs_layerwise=r_l,
                strongest_nonhybrid=strongest, rel_vs_strongest=r_s, per_layer_rel_vs_best_uniform=per_layer,
                transfer_ok=transfer_ok)


def recommend(verdict: dict, tau: float) -> str:
    """A only if HYBRID-PROMISING. B if layer-wise selection matches or exceeds expert-wise selection (hybrid not better
    than layer-wise by >= tau) AND layer-wise beats both uniform methods on the 2-layer mean. Otherwise C."""
    v = verdict["verdict"]
    if v == "HYBRID-PROMISING":
        return "A"
    if v == "HYBRID-BLOCKED":
        return "C"
    m = verdict["mean"]
    if verdict["rel_vs_layerwise"] > -tau and m["LAYERWISE"] < min(m["GPTQ"], m["AGQ"]):
        return "B"
    return "C"
