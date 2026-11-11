# Matched-warm-start typicality experiment (B = 128, seeds [0, 1, 2])

Prespecified decision: **MATCHED-TYPICALITY-SUPPORTED**.

Every run of seed s starts from the same ordered 32 sequences W_s (drawn from typical ∩ random half, 479 candidates). Warm-start hashes: s0 71f4c1599ef18971, s1 1d2b3965567279e4, s2 ba99ad010c44de11. Exact ordered warm-start identity across the 6 conditions: s0 yes, s1 yes, s2 yes.

## Reconstruction (mean expert relative error, test split; lower is better)

| pool | strategy | mean | sd | seed 0 / 1 / 2 | expert median | p90 | p95 |
|---|---|---|---|---|---|---|---|
| full | random | 0.004615 | 9.4e-06 | 0.004610 / 0.004626 / 0.004610 | 0.004803 | 0.006000 | 0.006216 |
| full | coverage_lp | 0.004730 | 4.5e-06 | 0.004727 / 0.004727 / 0.004735 | 0.004910 | 0.006136 | 0.006373 |
| typical_half | random | 0.004606 | 6.0e-06 | 0.004612 / 0.004600 / 0.004605 | 0.004792 | 0.006000 | 0.006195 |
| typical_half | coverage_lp | 0.004613 | 3.4e-06 | 0.004616 / 0.004609 / 0.004614 | 0.004798 | 0.005996 | 0.006216 |
| random_half | random | 0.004615 | 7.6e-06 | 0.004610 / 0.004611 / 0.004624 | 0.004806 | 0.005994 | 0.006220 |
| random_half | coverage_lp | 0.004697 | 9.3e-06 | 0.004698 / 0.004706 / 0.004687 | 0.004887 | 0.006096 | 0.006342 |

## Paired interactions (absolute ×1e-6; negative = typical eligibility helps)

| quantity | seed 0 | seed 1 | seed 2 | mean | sd | seeds < 0 |
|---|---|---|---|---|---|---|
| P_F = E(F,C) − E(F,R) | +117.5 | +101.3 | +125.0 | +114.6 | 12.1 | 0/3 |
| P_T = E(T,C) − E(T,R) | +4.1 | +9.4 | +9.6 | +7.7 | 3.1 | 0/3 |
| P_H = E(H,C) − E(H,R) | +88.7 | +94.6 | +63.8 | +82.3 | 16.4 | 0/3 |
| **I = P_T − P_F** | -113.4 | -91.9 | -115.4 | -106.9 | 13.0 | 3/3 |
| **J = P_T − P_H** | -84.6 | -85.2 | -54.2 | -74.6 | 17.7 | 3/3 |
| **D = E(T,C) − E(F,C)** | -111.1 | -118.1 | -120.8 | -116.6 | 5.0 | 3/3 |
| D relative to E(F,C) | -2.35% | -2.50% | -2.55% | -2.47% | | 3/3 |
| relative penalty P_X / E(X,R): F / T / H | | | | +2.48% / +0.17% / +1.78% | | |

## Relative changes ((A − B) / B; negative = A better)

| A vs B | seed 0 / 1 / 2 | mean | seeds A better |
|---|---|---|---|
| TC vs FC | -2.35% / -2.50% / -2.55% | -2.47% | 3/3 |
| TR vs FR | +0.05% / -0.57% / -0.12% | -0.21% | 2/3 |
| TC vs TR | +0.09% / +0.21% / +0.21% | +0.17% | 0/3 |
| FC vs FR | +2.55% / +2.19% / +2.71% | +2.48% | 0/3 |
| HC vs FC | -0.61% / -0.46% / -1.01% | -0.69% | 3/3 |
| TC vs HC | -1.75% / -2.05% / -1.56% | -1.79% | 3/3 |
| HC vs HR | +1.92% / +2.05% / +1.38% | +1.78% | 0/3 |
| TC vs FR | +0.14% / -0.36% / +0.09% | -0.04% | 1/3 |

## Per-layer error (seed mean)

| pool | strategy | L1 | L8 | L16 | L23 |
|---|---|---|---|---|---|
| full | random | 0.003618 | 0.004682 | 0.005147 | 0.005015 |
| full | coverage_lp | 0.003711 | 0.004784 | 0.005287 | 0.005137 |
| typical_half | random | 0.003602 | 0.004675 | 0.005145 | 0.005001 |
| typical_half | coverage_lp | 0.003618 | 0.004684 | 0.005149 | 0.005002 |
| random_half | random | 0.003616 | 0.004676 | 0.005147 | 0.005020 |
| random_half | coverage_lp | 0.003696 | 0.004753 | 0.005242 | 0.005098 |

## Routing, exposure and source diversity (seed mean; typicality vs the ORIGINAL pool)

| pool | strategy | d(s) all 128 | d(s) post-warm | entropy | set JS | set L1 per layer | median tokens | Gini | eff. sources median / mean |
|---|---|---|---|---|---|---|---|---|---|
| full | random | 0.3242 | 0.3357 | 3.9913 | 0.00023 | 0.022, 0.032, 0.032, 0.042 | 4307 | 0.054 | 118.5 / 116.5 |
| full | coverage_lp | 0.4092 | 0.4490 | 3.9386 | 0.00154 | 0.055, 0.096, 0.104, 0.100 | 4345 | 0.043 | 112.0 / 110.5 |
| typical_half | random | 0.2891 | 0.2888 | 4.0123 | 0.00045 | 0.032, 0.046, 0.048, 0.056 | 4310 | 0.065 | 120.9 / 119.1 |
| typical_half | coverage_lp | 0.2953 | 0.2972 | 4.0131 | 0.00030 | 0.024, 0.033, 0.030, 0.052 | 4330 | 0.046 | 120.4 / 118.7 |
| random_half | random | 0.3289 | 0.3418 | 3.9872 | 0.00022 | 0.021, 0.036, 0.030, 0.039 | 4296 | 0.055 | 118.3 / 116.1 |
| random_half | coverage_lp | 0.3819 | 0.4126 | 3.9596 | 0.00131 | 0.048, 0.084, 0.095, 0.096 | 4358 | 0.039 | 114.4 / 112.7 |

## Typical vs Full Coverage-LP within seed (T-C − F-C)

| seed | Δ error | Δ d post-warm | Δ set JS | Δ eff. sources (median) | Δ Gini | per-layer Δ error (rel.) |
|---|---|---|---|---|---|---|
| 0 | -111.1e-6 | -0.1495 | -0.00138 | +8.1 | +0.007 | L1 -2.66%, L8 -2.09%, L16 -2.46%, L23 -2.25% |
| 1 | -118.1e-6 | -0.1521 | -0.00108 | +8.3 | +0.003 | L1 -2.44%, L8 -2.11%, L16 -2.58%, L23 -2.81% |
| 2 | -120.8e-6 | -0.1540 | -0.00126 | +8.6 | -0.001 | L1 -2.48%, L8 -2.08%, L16 -2.78%, L23 -2.81% |

## Selected-set overlap (fraction of 128; 0.25 = warm start only)

| sets | seed 0 / 1 / 2 |
|---|---|
| TC vs FC | 0.289 / 0.258 / 0.289 |
| HC vs FC | 0.430 / 0.453 / 0.359 |
| TC vs TR | 0.336 / 0.320 / 0.297 |
| TR vs FR | 0.312 / 0.297 / 0.281 |
| TC vs HC | 0.312 / 0.281 / 0.320 |

## Unmatched (earlier) vs matched warm start — not pooled

| experiment | F-C | T-C | F-R | T-R | P_F | P_T | I (×1e-6) | J (×1e-6) | T-C vs F-C | warm start |
|---|---|---|---|---|---|---|---|---|---|---|
| unmatched (`runs_typicality/`) | 0.004677 | 0.004608 | 0.004616 | 0.004608 | +61.1 | +0.4 | -60.7 | -65.6 | -1.47% | different per pool (T-C/F-C overlap 0.055/0.078/0.070) |
| matched (this run) | 0.004730 | 0.004613 | 0.004615 | 0.004606 | +114.6 | +7.7 | -106.9 | -74.6 | -2.47% | identical ordered W_s in all 6 conditions |

Matched D / unmatched D = 1.68.

## Controller and cost (seed mean)

| pool | strategy | LP-selected / fill+fallback (3 seeds) | selection s | Hessian s | GPTQ s | routing profile (full-forward equiv.) s | peak GPU GB |
|---|---|---|---|---|---|---|---|
| full | random | – | 0.0 | 66 | 284 | 0 | 7.8 |
| full | coverage_lp | 288 / 0 | 1.1 | 66 | 281 | 357 | 7.8 |
| typical_half | random | – | 0.0 | 62 | 285 | 0 | 7.8 |
| typical_half | coverage_lp | 288 / 0 | 0.4 | 67 | 284 | 357 | 7.8 |
| random_half | random | – | 0.0 | 64 | 278 | 0 | 7.8 |
| random_half | coverage_lp | 288 / 0 | 0.5 | 70 | 275 | 357 | 7.8 |

Notes: estimand is conditional on a shared warm start drawn from typical ∩ random half. Three seeds = three paired replicates; no significance claims. Random from a filtered pool also relies on the full-pool routing profile to define the pool (not charged in its cost field). Parallel runs (2 at a time) inflate timings. Equal-budget comparison only.
