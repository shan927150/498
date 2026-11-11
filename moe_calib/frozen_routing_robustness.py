"""Frozen-routing robustness across the five prespecified calibration batch groupings (helpers).

AO_g = GPTQ(X_g, R_O): calibration Hessians built from the BF16 layer inputs of grouping g with the ORIGINAL expert
assignments R_O (moe_calib.frozen_routing, unchanged); AA_g = GPTQ(X_g, R_g) is the natural multigroup result and
OO = GPTQ(X_O, R_O) the single ORIGINAL reference per strategy. This module holds the grouping list, the
robustness / ranking metrics and the frozen decision rule; it has no model or GPU dependency.
"""
from __future__ import annotations
import hashlib
import json

import numpy as np

GROUPING_SEEDS = (20261111, 20261112, 20261113, 20261114, 20261115)          # = moe_calib.multigroup_robustness
GROUPINGS = tuple(f"G{s}" for s in GROUPING_SEEDS)
REUSED_AO = ("G20261111",)                                                    # factorial-study AO results
NEW_AO = tuple(g for g in GROUPINGS if g not in REUSED_AO)
STRATS = ("FR", "FC")
SELECTION_HASHES = {"FR": "ec355fb8d866e4ff", "FC": "0353229b6d3090db"}
ZERO_SD = 1e-12                       # a denominator SD at or below this is "effectively zero": ratio undefined
RATIO_MAX = 0.5
MIN_GROUPINGS = 4                     # "at least 4 of 5"


def sd(x):
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1)) if x.size > 1 else float("nan")


def rng(x):
    x = np.asarray(x, float)
    return float(x.max() - x.min())


def ratio(num, den):
    """num / den, or None when the denominator is effectively zero (raw values are always reported alongside)."""
    return None if not np.isfinite(den) or abs(den) <= ZERO_SD else float(num / den)


def spread(ao, aa):
    """Per-strategy grouping sensitivity of AO vs AA (lists over the same grouping order)."""
    s_ao, s_aa, r_ao, r_aa = sd(ao), sd(aa), rng(ao), rng(aa)
    return dict(SD_AO=s_ao, SD_AA=s_aa, SD_ratio=ratio(s_ao, s_aa), Range_AO=r_ao, Range_AA=r_aa,
                Range_ratio=ratio(r_ao, r_aa), mean_AO=float(np.mean(ao)), mean_AA=float(np.mean(aa)))


def distances(ao, aa, oo):
    d_ao = [abs(x - oo) for x in ao]
    d_aa = [abs(x - oo) for x in aa]
    return dict(Dist_AO=d_ao, Dist_AA=d_aa, mean_Dist_AO=float(np.mean(d_ao)), mean_Dist_AA=float(np.mean(d_aa)),
                median_Dist_AO=float(np.median(d_ao)), median_Dist_AA=float(np.median(d_aa)),
                n_AO_closer=int(sum(a < b for a, b in zip(d_ao, d_aa))),
                n_AO_farther=int(sum(a > b for a, b in zip(d_ao, d_aa))),
                n_tied=int(sum(a == b for a, b in zip(d_ao, d_aa))),
                mean_signed_AO_minus_OO=float(np.mean(ao) - oo), mean_signed_AA_minus_OO=float(np.mean(aa) - oo))


def ranking(fr, fc, delta_oo):
    """fr / fc: {"AO": [...], "AA": [...]} PPL over the same grouping order. Delta = FC - FR (positive: FC worse)."""
    out = {}
    for k in ("AO", "AA"):
        d = [b - a for a, b in zip(fr[k], fc[k])]
        out[k] = dict(Delta=d, mean=float(np.mean(d)), SD=sd(d), range=rng(d),
                      n_positive=int(sum(x > 0 for x in d)), n_negative=int(sum(x < 0 for x in d)),
                      n_zero=int(sum(x == 0 for x in d)),
                      n_reversals_vs_OO=int(sum(np.sign(x) != np.sign(delta_oo) for x in d)),
                      n_preserved_vs_OO=int(sum(np.sign(x) == np.sign(delta_oo) and x != 0 for x in d)),
                      abs_dev_from_OO=[abs(x - delta_oo) for x in d],
                      mean_abs_dev_from_OO=float(np.mean([abs(x - delta_oo) for x in d])))
    out["Delta_OO"] = delta_oo
    out["SD_Delta_ratio"] = ratio(out["AO"]["SD"], out["AA"]["SD"])
    out["Range_Delta_ratio"] = ratio(out["AO"]["range"], out["AA"]["range"])
    out["n_AO_closer_to_Delta_OO"] = int(sum(a < b for a, b in zip(out["AO"]["abs_dev_from_OO"], out["AA"]["abs_dev_from_OO"])))
    out["RankingInteraction"] = [a - b for a, b in zip(out["AO"]["Delta"], out["AA"]["Delta"])]
    return out


def intervention_effects(ao, aa):
    """Paired (same grouping) InterventionEffect_g = PPL_AO,g - PPL_AA,g."""
    e = [a - b for a, b in zip(ao, aa)]
    return dict(per_grouping=e, mean=float(np.mean(e)), SD=sd(e), mean_abs=float(np.mean(np.abs(e))))


# ---------------------------------------------------------------- frozen decision rule
DECISION_RULE = {
    "definitions": {
        "SD_ratio[c]": "std_g(PPL_AO,g, ddof=1) / std_g(PPL_AA,g, ddof=1), five groupings, strategy c in {FR, FC}",
        "SD_Delta_ratio": "std_g(Delta_AO,g) / std_g(Delta_AA,g), Delta = PPL_FC - PPL_FR (same grouping)",
        "closer[c]": "number of groupings with |PPL_AO,g - PPL_OO| < |PPL_AA,g - PPL_OO| (strategy c)",
        "preserved[k]": "number of groupings with sign(Delta_k,g) == sign(Delta_OO) (Delta_OO = +0.010057), k in {AO, AA}",
        "undefined_ratio": f"a denominator SD <= {ZERO_SD} makes the ratio undefined; an undefined ratio never "
                           "satisfies a <= 0.5 criterion (raw SDs reported)",
    },
    "precedence": [
        "RUN-BLOCKED: a required artifact is missing / fails its fingerprint, G20261111 AO does not regress exactly, "
        "a routing-alignment or token-count invariant fails, or any AO configuration fails a correctness check",
        "FROZEN-ROUTING-ROBUSTLY-STABILIZES-PPL: SD_ratio[FR] <= 0.5 AND SD_ratio[FC] <= 0.5 AND SD_Delta_ratio <= 0.5 "
        "AND closer[FR] >= 4 AND closer[FC] >= 4 AND preserved[AO] >= 4",
        "FROZEN-ROUTING-PARTIALLY-STABILIZES: at least one of (a) SD_ratio[FR] <= 0.5, (b) SD_ratio[FC] <= 0.5, "
        "(c) SD_Delta_ratio <= 0.5, (d) closer[FR] >= 4 AND closer[FC] >= 4, (e) preserved[AO] >= 4 AND "
        "preserved[AO] > preserved[AA]; the satisfied items are reported",
        "FROZEN-ROUTING-NOT-STABLE: otherwise",
    ],
    "thresholds": dict(ratio_max=RATIO_MAX, min_groupings=MIN_GROUPINGS, n_groupings=len(GROUPINGS)),
    "note": "exploratory practical criteria, not hypothesis tests; frozen before any new AO quantization or PPL",
}


def criteria(sp, rk, dist):
    """sp[c] = spread(...), dist[c] = distances(...), rk = ranking(...)."""
    le = lambda r: r is not None and r <= RATIO_MAX
    return {"sd_ratio_FR": le(sp["FR"]["SD_ratio"]), "sd_ratio_FC": le(sp["FC"]["SD_ratio"]),
            "sd_delta_ratio": le(rk["SD_Delta_ratio"]),
            "closer_FR": dist["FR"]["n_AO_closer"] >= MIN_GROUPINGS, "closer_FC": dist["FC"]["n_AO_closer"] >= MIN_GROUPINGS,
            "ao_ranking_preserved": rk["AO"]["n_preserved_vs_OO"] >= MIN_GROUPINGS,
            "ao_ranking_preserved_more_than_aa": rk["AO"]["n_preserved_vs_OO"] > rk["AA"]["n_preserved_vs_OO"]}


def decide(cr, blocked=False):
    if blocked:
        return "RUN-BLOCKED", []
    robust = all(cr[k] for k in ("sd_ratio_FR", "sd_ratio_FC", "sd_delta_ratio", "closer_FR", "closer_FC",
                                 "ao_ranking_preserved"))
    if robust:
        return "FROZEN-ROUTING-ROBUSTLY-STABILIZES-PPL", []
    partial = {"a_sd_ratio_FR": cr["sd_ratio_FR"], "b_sd_ratio_FC": cr["sd_ratio_FC"], "c_sd_delta_ratio": cr["sd_delta_ratio"],
               "d_closer_both": cr["closer_FR"] and cr["closer_FC"],
               "e_ranking": cr["ao_ranking_preserved"] and cr["ao_ranking_preserved_more_than_aa"]}
    hit = [k for k, v in partial.items() if v]
    if hit:
        return "FROZEN-ROUTING-PARTIALLY-STABILIZES", hit
    return "FROZEN-ROUTING-NOT-STABLE", []


def ao_counts_invariant(counts_by_grouping):
    """{g: {key: tokens}} -> True iff every grouping has the identical routed-token count vector."""
    vals = list(counts_by_grouping.values())
    return all(v == vals[0] for v in vals[1:])


def obj_sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()
