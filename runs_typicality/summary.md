# Routing-typicality experiment (B = 128, seeds [0, 1, 2])

Prespecified reading: **TYPICALITY-SUPPORTED**.

## Reconstruction (mean expert relative error, test split; lower is better)

| pool | strategy | mean | sd | seed 0 / 1 / 2 | expert median | p90 | p95 |
|---|---|---|---|---|---|---|---|
| full | random | 0.004616 | 9.9e-06 | 0.004606 / 0.004616 / 0.004626 | 0.004809 | 0.006009 | 0.006220 |
| full | coverage_lp | 0.004677 | 9.2e-06 | 0.004669 / 0.004687 / 0.004676 | 0.004866 | 0.006083 | 0.006305 |
| typical_half | random | 0.004608 | 8.8e-06 | 0.004602 / 0.004604 / 0.004618 | 0.004795 | 0.006008 | 0.006197 |
| typical_half | coverage_lp | 0.004608 | 9.3e-06 | 0.004600 / 0.004607 / 0.004619 | 0.004794 | 0.005993 | 0.006197 |
| random_half | random | 0.004622 | 1.2e-05 | 0.004609 / 0.004633 / 0.004622 | 0.004816 | 0.006011 | 0.006230 |
| random_half | coverage_lp | 0.004687 | 5.7e-06 | 0.004681 / 0.004692 / 0.004689 | 0.004875 | 0.006100 | 0.006326 |

## Coverage penalty and interaction (absolute error ×1e-6; negative = coverage better / penalty smaller)

| quantity | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| P_F = E(F,C) − E(F,R) | +62.6 | +70.9 | +49.8 | +61.1 |
| P_T = E(T,C) − E(T,R) | -2.1 | +2.9 | +0.3 | +0.4 |
| P_H = E(H,C) − E(H,R) | +72.0 | +59.5 | +66.3 | +66.0 |
| **I = P_T − P_F** | -64.7 | -68.1 | -49.5 | -60.7 |
| **P_T − P_H** | -74.1 | -56.7 | -66.0 | -65.6 |
| P_H − P_F | +9.4 | -11.4 | +16.6 | +4.8 |
| relative penalty P_X / E(X,R): F / T / H (mean) | | | | +1.32% / +0.01% / +1.43% |

## Relative error changes ((A − B) / B; negative = A better)

| A vs B | seed 0 / 1 / 2 | mean | seeds A better |
|---|---|---|---|
| Typical Random vs Full Random | -0.08% / -0.26% / -0.17% | -0.17% | 3/3 |
| Typical Cov-LP vs Full Cov-LP | -1.47% / -1.71% / -1.22% | -1.47% | 3/3 |
| Typical Cov-LP vs Typical Random | -0.05% / +0.06% / +0.01% | +0.01% | 1/3 |
| Typical Cov-LP vs RandHalf Cov-LP | -1.73% / -1.83% / -1.49% | -1.69% | 3/3 |
| RandHalf Cov-LP vs Full Cov-LP | +0.27% / +0.12% / +0.28% | +0.22% | 0/3 |
| RandHalf Random vs Full Random | +0.07% / +0.37% / -0.08% | +0.12% | 1/3 |
| Typical Random vs RandHalf Random | -0.15% / -0.63% / -0.09% | -0.29% | 3/3 |
| Typical Cov-LP vs Full Random | -0.13% / -0.20% / -0.16% | -0.16% | 3/3 |

## Per-layer error (seed mean)

| pool | strategy | L1 | L8 | L16 | L23 |
|---|---|---|---|---|---|
| full | random | 0.003619 | 0.004676 | 0.005151 | 0.005017 |
| full | coverage_lp | 0.003657 | 0.004734 | 0.005229 | 0.005088 |
| typical_half | random | 0.003603 | 0.004676 | 0.005147 | 0.005006 |
| typical_half | coverage_lp | 0.003616 | 0.004675 | 0.005143 | 0.004999 |
| random_half | random | 0.003626 | 0.004683 | 0.005152 | 0.005026 |
| random_half | coverage_lp | 0.003682 | 0.004741 | 0.005229 | 0.005097 |

## Selected-set routing and exposure (seed mean; typicality always vs the ORIGINAL pool)

| pool | strategy | d(s) of 128 chosen | d(s) after warm start | entropy | set JS vs pool | median expert tokens | Gini of expert tokens | low-exposure experts | median sources / expert | median effective sources / expert |
|---|---|---|---|---|---|---|---|---|---|---|
| full | random | 0.3345 | 0.3303 | 3.9845 | 0.00024 | 4296 | 0.053 | 0.0 | 128.0 | 117.8 |
| full | coverage_lp | 0.3863 | 0.3994 | 3.9525 | 0.00070 | 4350 | 0.035 | 0.0 | 128.0 | 113.8 |
| typical_half | random | 0.2868 | 0.2853 | 4.0132 | 0.00048 | 4302 | 0.065 | 0.0 | 128.0 | 121.0 |
| typical_half | coverage_lp | 0.2961 | 0.2974 | 4.0121 | 0.00031 | 4332 | 0.045 | 0.0 | 128.0 | 120.5 |
| random_half | random | 0.3402 | 0.3437 | 3.9832 | 0.00023 | 4321 | 0.048 | 0.0 | 128.0 | 117.6 |
| random_half | coverage_lp | 0.3796 | 0.3952 | 3.9603 | 0.00074 | 4349 | 0.037 | 0.0 | 128.0 | 114.6 |

Low exposure: n_e < 0.25 x layer tokens / 60, summed over the 4 layers. Sources = selected sequences routing >= 1 token to the expert; effective sources = exp(entropy of the per-sequence shares of the expert's routed tokens).

## Selected-set overlap (fraction of 128)

| sets | seed 0 / 1 / 2 |
|---|---|
| TC vs FC | 0.055 / 0.078 / 0.070 |
| HC vs FC | 0.094 / 0.156 / 0.133 |
| TC vs TR | 0.328 / 0.328 / 0.359 |
| TC vs HC | 0.055 / 0.039 / 0.062 |
| TR vs FR | 0.094 / 0.141 / 0.109 |

## Mechanism checks (per seed)

| check | per seed | all |
|---|---|---|
| 1_typical_filter_lowers_selected_atypicality (TR<FR, TC<FC, d of 128 chosen) | yes / yes / yes / yes / yes / yes | yes |
| 2_coverage_lp_from_T_picks_more_typical_sequences (post-warm d, TC<FC) | yes / yes / yes | yes |
| 3_set_JS_to_original_pool_decreases (TC<FC) | yes / yes / yes | yes |
| 4_reconstruction_improves (TC<FC) | yes / yes / yes | yes |
| 5_coverage_penalty_shrinks (I<0) | yes / yes / yes | yes |
| 5b_more_than_random_half (P_T-P_H<0) | yes / yes / yes | yes |
| 6_consistent_across_seeds | | yes |

## Cost (seed mean)

| pool | strategy | selection s | Hessian s | GPTQ s | routing profile (full-forward equiv.) s | peak GPU GB |
|---|---|---|---|---|---|---|
| full | random | 0.0 | 31 | 203 | 0 | 7.8 |
| full | coverage_lp | 1.2 | 55 | 256 | 357 | 7.8 |
| typical_half | random | 0.0 | 67 | 285 | 0 | 7.8 |
| typical_half | coverage_lp | 0.4 | 68 | 285 | 357 | 7.8 |
| random_half | random | 0.0 | 68 | 282 | 0 | 7.8 |
| random_half | coverage_lp | 0.5 | 69 | 282 | 357 | 7.8 |

Notes: full-pool runs are reused from `runs_allocation_minimal/` after validation. Random from a filtered pool also needs the full-pool routing profile (to define the pool) — not charged in its E3 cost field. Seed timings for parallel runs include contention. Equal-budget comparison only; no equal-compute claim.
