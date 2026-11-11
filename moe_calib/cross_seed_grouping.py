"""Cross-seed x calibration-grouping robustness: pure helpers (no model, no I/O).

Design: allocation seed s in {0,1,2} (outer unit) x grouping seed g in GROUPING_SEEDS (nested within s:
the same RNG seed permutes a DIFFERENT union U_s) x strategy {FR, FC} (paired within (s, g)).
Grouping policy is exactly moe_calib.multigroup_robustness.grouping_batches.
"""
from __future__ import annotations

import numpy as np

from .multigroup_robustness import GROUPING_SEEDS, BATCH, grouping_batches, validate_grouping, obj_sha  # noqa: F401

ALLOC_SEEDS = (0, 1, 2)
N_GROUPINGS = len(GROUPING_SEEDS)
HBOOT_N, HBOOT_SEED = 20000, 20261120


def union_of(fr, fc):
    return sorted(set(map(int, fr)) | set(map(int, fc)))


def expected_sizes(n, B=BATCH):
    return [B] * (n // B) + ([n % B] if n % B else [])


def cname(s, g):
    return f"s{int(s)}_G{int(g)}"


# ---------------------------------------------------------------- paired differences / variance summaries
def paired_deltas(ppl):
    """ppl[(strategy, s, g)] -> {s: {g: PPL(FC,s,g) - PPL(FR,s,g)}}; pairs ONLY within the same (s, g)."""
    keys = sorted({(s, g) for (_, s, g) in ppl})
    out = {}
    for s, g in keys:
        if ("FR", s, g) not in ppl or ("FC", s, g) not in ppl:
            raise KeyError(f"unpaired configuration (s={s}, g={g})")
        out.setdefault(s, {})[g] = ppl[("FC", s, g)] - ppl[("FR", s, g)]
    return out


def _nested(deltas):
    """Validate the nested structure: dict seed -> dict grouping -> value, >=2 groupings per seed, balanced.
    A flat list (pseudo-replicated 15 observations) is rejected."""
    if not isinstance(deltas, dict) or not all(isinstance(v, dict) for v in deltas.values()):
        raise TypeError("deltas must be nested {allocation_seed: {grouping: Delta}}; flat pooling is not allowed")
    sizes = {len(v) for v in deltas.values()}
    if len(sizes) != 1 or min(sizes) < 2:
        raise ValueError(f"unbalanced or degenerate nested design: groupings per seed {sizes}")
    return {s: np.array([deltas[s][g] for g in sorted(deltas[s])], float) for s in sorted(deltas)}


def variance_summary(deltas):
    D = _nested(deltas)
    S = sorted(D)
    k = len(next(iter(D.values())))
    means = {s: float(D[s].mean()) for s in S}
    sds = {s: float(D[s].std(ddof=1)) for s in S}
    allv = np.concatenate([D[s] for s in S])
    m = np.array([means[s] for s in S])
    pooled = float(np.sqrt(np.mean([sds[s] ** 2 for s in S])))
    sd_between = float(m.std(ddof=1)) if len(S) > 1 else float("nan")
    var_comp = max(0.0, sd_between ** 2 - pooled ** 2 / k) if len(S) > 1 else float("nan")
    return dict(
        n_seeds=len(S), groupings_per_seed=k, mean_delta=means, sd_grouping=sds,
        se_grouping_mean={s: sds[s] / np.sqrt(k) for s in S},
        median_delta_per_seed={s: float(np.median(D[s])) for s in S},
        n_fc_worse={s: int((D[s] > 0).sum()) for s in S}, n_fc_better={s: int((D[s] < 0).sum()) for s in S},
        overall_mean_delta=float(m.mean()), overall_median_delta=float(np.median(allv)),
        n_fc_worse_total=int((allv > 0).sum()), n_fc_better_total=int((allv < 0).sum()),
        min_delta=float(allv.min()), max_delta=float(allv.max()),
        pooled_within_sd=pooled, sd_between_seed_means=sd_between,
        expected_sd_of_seed_means_from_grouping_only=pooled / np.sqrt(k),
        between_seed_variance_component=var_comp, between_seed_sd_component=float(np.sqrt(var_comp)) if var_comp == var_comp else float("nan"),
        note="seed means include finite-grouping averaging error (expected SD pooled/sqrt(k)); the variance component "
             "max(0, Var(seed means) - pooled^2/k) is a descriptive, highly uncertain estimate with 3 seeds")


def hierarchical_bootstrap(deltas, n_boot=HBOOT_N, seed=HBOOT_SEED):
    """Resample seeds with replacement; within each sampled seed resample its k groupings with replacement;
    statistic = mean over sampled seeds of the within-seed means (= OverallMeanDelta definition)."""
    D = _nested(deltas)
    S = sorted(D)
    A = np.stack([D[s] for s in S])                        # [n_seeds, k]
    n, k = A.shape
    rng = np.random.default_rng(seed)
    si = rng.integers(0, n, size=(n_boot, n))
    gi = rng.integers(0, k, size=(n_boot, n, k))
    stat = A[si[..., None], gi].mean(axis=2).mean(axis=1)
    return dict(point=float(A.mean(1).mean()), ci95=[float(np.quantile(stat, .025)), float(np.quantile(stat, .975))],
                frac_boot_positive=float((stat > 0).mean()), n_boot=n_boot, seed=seed, n_seeds=n, k=k,
                n_distinct_seed_resamples=int(len({tuple(sorted(r)) for r in si.tolist()})),
                note=f"only {n} allocation seeds: the outer resampling has very few distinct configurations "
                     f"({len({tuple(sorted(r)) for r in si.tolist()})}); the interval is unstable and descriptive, not "
                     "population-level inference")


# ---------------------------------------------------------------- frozen decision rule
DECISION_RULE = {
    "quantities": "Delta[s,g] = PPL(FC,s,g) - PPL(FR,s,g) for s in {0,1,2}, g in 5 grouping seeds (all 15, none "
                  "excluded); m_s = mean_g Delta[s,g]; sd_s = SD_g (ddof 1); se_s = sd_s/sqrt(5); W = pooled within-seed "
                  "SD = sqrt(mean_s sd_s^2); M = mean_s m_s; SB = SD_s(m_s) (ddof 1); EXP = W/sqrt(5)",
    "RUN-BLOCKED": "any of the 30 configurations missing or invalid, a cache / determinism / reuse-integrity check "
                   "fails, or the seed-0 reused results do not match their frozen artifacts",
    "ALLOCATION-SEED-DEPENDENT": "(a) conflicting confident directions: some m_s > 2 se_s AND some m_t < -2 se_t; OR "
                                 "(b) material scale disagreement: SB >= 2 EXP (seed means spread at least twice what "
                                 "grouping averaging alone predicts) AND max_s |m_s| >= W",
    "GROUPING-VARIATION-DOMINATES": "(checked after seed dependence) |M| < 0.5 W AND at least 1/3 of the 15 Delta[s,g] "
                                    "have sign opposite to sign(M)",
    "FC-PPL-PENALTY-ROBUST": "all m_s > 0 AND in every seed >= 3/5 groupings FC worse AND M >= 0.5 W",
    "FC-PPL-ADVANTAGE-EXPLORATORY": "all m_s < 0 AND in every seed >= 3/5 groupings FC better AND -M >= 0.5 W",
    "fallback": "otherwise GROUPING-VARIATION-DOMINATES with ambiguous=True (explicitly reported)",
    "precedence": "RUN-BLOCKED > ALLOCATION-SEED-DEPENDENT > GROUPING-VARIATION-DOMINATES > FC-PPL-PENALTY-ROBUST / "
                  "FC-PPL-ADVANTAGE-EXPLORATORY > fallback",
}


def decide(deltas, blocked=False):
    """Returns (category, details)."""
    if blocked:
        return "RUN-BLOCKED", dict(reason="blocked")
    v = variance_summary(deltas)
    D = _nested(deltas)
    S = sorted(D)
    m, se, W, M = v["mean_delta"], v["se_grouping_mean"], v["pooled_within_sd"], v["overall_mean_delta"]
    conf_pos = [s for s in S if m[s] > 2 * se[s]]
    conf_neg = [s for s in S if m[s] < -2 * se[s]]
    scale = v["sd_between_seed_means"] >= 2 * v["expected_sd_of_seed_means_from_grouping_only"] and \
        max(abs(m[s]) for s in S) >= W
    allv = np.concatenate([D[s] for s in S])
    opp = float((np.sign(allv) == -np.sign(M)).mean()) if M != 0 else 1.0
    det = dict(confident_positive_seeds=conf_pos, confident_negative_seeds=conf_neg, scale_disagreement=bool(scale),
               frac_opposite_sign_to_overall=opp, abs_M_over_W=abs(M) / W if W > 0 else float("inf"), ambiguous=False)
    if (conf_pos and conf_neg) or scale:
        return "ALLOCATION-SEED-DEPENDENT", det
    if abs(M) < 0.5 * W and opp >= 1 / 3:
        return "GROUPING-VARIATION-DOMINATES", det
    k = v["groupings_per_seed"]
    need = int(np.ceil(0.6 * k))
    if all(m[s] > 0 for s in S) and all(v["n_fc_worse"][s] >= need for s in S) and M >= 0.5 * W:
        return "FC-PPL-PENALTY-ROBUST", det
    if all(m[s] < 0 for s in S) and all(v["n_fc_better"][s] >= need for s in S) and -M >= 0.5 * W:
        return "FC-PPL-ADVANTAGE-EXPLORATORY", det
    det["ambiguous"] = True
    return "GROUPING-VARIATION-DOMINATES", det


UNCERTAINTY_LEVELS = {
    "window_bootstrap": "evaluation-text sampling uncertainty CONDITIONAL on one fixed pair of quantized models (s, g)",
    "grouping_replication": "sensitivity to calibration execution conditions (BF16 batch membership) within a seed",
    "allocation_seeds": "variation in the selected calibration sets (only 3 seeds: outer unit)",
    "combination": "not combined into a single interval; the hierarchical bootstrap covers seeds and groupings only, "
                   "with the evaluation windows held fixed",
}
