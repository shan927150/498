# Minimal allocation experiment (B = 128, seeds [0, 1, 2])

Decision rule output: **ALLOCATION-NO-GAIN** (prespecified rule in `diagnostics/report_minimal_allocation.py`; 0.5 % is a practical effect size, not a test).

## Reconstruction (expert mean relative error, test split; lower is better)

| strategy | mean | sd (seeds) | per seed | vs random | vs coverage | vs coverage_lp | expert p90 | expert p95 | zero-exposure experts |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.004616 | 9.88e-06 | 0.004606, 0.004616, 0.004626 | – | -1.43% | -1.31% | 0.006009 | 0.006220 | 0.0 |
| coverage | 0.004683 | 5.35e-06 | 0.004677, 0.004683, 0.004688 | +1.45% | – | +0.13% | 0.006082 | 0.006315 | 0.0 |
| coverage_lp | 0.004677 | 9.19e-06 | 0.004669, 0.004687, 0.004676 | +1.32% | -0.13% | – | 0.006083 | 0.006305 | 0.0 |
| global_exposure | 0.004666 | 2.43e-05 | 0.004639, 0.004675, 0.004685 | +1.09% | -0.36% | -0.23% | 0.006067 | 0.006293 | 0.0 |
| layer_budget | 0.004700 | 1.26e-05 | 0.004706, 0.004686, 0.004709 | +1.82% | +0.36% | +0.49% | 0.006110 | 0.006337 | 0.0 |
| layer_aware | 0.004678 | 2.68e-05 | 0.004654, 0.004674, 0.004707 | +1.35% | -0.10% | +0.02% | 0.006097 | 0.006307 | 0.0 |

## Paired per-seed differences (strategy − reference; negative = better)

| comparison | Δ seed 0 / 1 / 2 (×1e-6) | rel. per seed | mean rel. | seeds better | per-layer rel. | p95 rel. | expert win fraction |
|---|---|---|---|---|---|---|---|
| global_exposure vs coverage_lp | -29.74 / -12.28 / +9.67 | -0.64% / -0.26% / +0.21% | -0.23% | 2/3 | L1 -0.34%, L8 -0.13%, L16 -0.25%, L23 -0.23% | -0.18% | 0.93 / 0.76 / 0.26 |
| layer_budget vs coverage_lp | +36.84 / -1.35 / +33.25 | +0.79% / -0.03% / +0.71% | +0.49% | 1/3 | L1 +0.51%, L8 +0.42%, L16 +0.52%, L23 +0.51% | +0.51% | 0.06 / 0.50 / 0.08 |
| layer_aware vs coverage_lp | -14.80 / -13.24 / +31.27 | -0.32% / -0.28% / +0.67% | +0.02% | 2/3 | L1 -0.08%, L8 +0.01%, L16 +0.07%, L23 +0.06% | +0.04% | 0.78 / 0.75 / 0.07 |
| coverage_lp vs coverage | -8.81 / +3.72 / -12.56 | -0.19% / +0.08% / -0.27% | -0.13% | 2/3 | L1 -0.37%, L8 -0.18%, L16 +0.04%, L23 -0.06% | -0.17% | 0.68 / 0.45 / 0.75 |
| layer_aware vs layer_budget | -51.64 / -11.89 / -1.98 | -1.10% / -0.25% / -0.04% | -0.46% | 3/3 | L1 -0.59%, L8 -0.41%, L16 -0.45%, L23 -0.45% | -0.47% | 1.00 / 0.79 / 0.55 |
| coverage vs random | +71.41 / +67.20 / +62.34 | +1.55% / +1.46% / +1.35% | +1.45% | 0/3 | L1 +1.43%, L8 +1.42%, L16 +1.46%, L23 +1.49% | +1.53% | 0.00 / 0.00 / 0.00 |
| coverage_lp vs random | +62.60 / +70.93 / +49.78 | +1.36% / +1.54% / +1.08% | +1.32% | 0/3 | L1 +1.05%, L8 +1.23%, L16 +1.50%, L23 +1.42% | +1.36% | 0.01 / 0.00 / 0.04 |
| global_exposure vs random | +32.86 / +58.65 / +59.45 | +0.71% / +1.27% / +1.29% | +1.09% | 0/3 | L1 +0.70%, L8 +1.10%, L16 +1.25%, L23 +1.19% | +1.18% | 0.07 / 0.03 / 0.02 |
| layer_budget vs random | +99.44 / +69.58 / +83.03 | +2.16% / +1.51% / +1.79% | +1.82% | 0/3 | L1 +1.57%, L8 +1.66%, L16 +2.03%, L23 +1.94% | +1.88% | 0.00 / 0.00 / 0.00 |
| layer_aware vs random | +47.80 / +57.69 / +81.05 | +1.04% / +1.25% / +1.75% | +1.35% | 0/3 | L1 +0.97%, L8 +1.24%, L16 +1.57%, L23 +1.48% | +1.40% | 0.04 / 0.02 / 0.00 |

## Per-layer reconstruction (seed mean)

| strategy | L1 | L8 | L16 | L23 |
|---|---|---|---|---|
| random | 0.003619 | 0.004676 | 0.005151 | 0.005017 |
| coverage | 0.003671 | 0.004743 | 0.005227 | 0.005092 |
| coverage_lp | 0.003657 | 0.004734 | 0.005229 | 0.005088 |
| global_exposure | 0.003645 | 0.004728 | 0.005216 | 0.005077 |
| layer_budget | 0.003676 | 0.004754 | 0.005256 | 0.005114 |
| layer_aware | 0.003654 | 0.004734 | 0.005232 | 0.005091 |

## Selection overlap and routing vs coverage_lp

| strategy | overlap (of 128) per seed | overlap excl. warm start | identical sets | per-layer L1 of routed-count distribution vs coverage_lp |
|---|---|---|---|---|
| random | 0.273 / 0.281 / 0.305 | 0.031 / 0.042 / 0.073 | none | L1 0.036, L8 0.072, L16 0.068, L23 0.088 |
| coverage | 0.430 / 0.453 / 0.383 | 0.240 / 0.271 / 0.177 | none | L1 0.031, L8 0.039, L16 0.040, L23 0.050 |
| global_exposure | 0.453 / 0.492 / 0.453 | 0.271 / 0.323 / 0.271 | none | L1 0.027, L8 0.042, L16 0.039, L23 0.054 |
| layer_budget | 0.383 / 0.453 / 0.391 | 0.177 / 0.271 / 0.188 | none | L1 0.032, L8 0.052, L16 0.048, L23 0.055 |
| layer_aware | 0.438 / 0.461 / 0.398 | 0.250 / 0.281 / 0.198 | none | L1 0.027, L8 0.037, L16 0.035, L23 0.046 |

## Controller diagnostics (per round; seeds pooled)

| strategy | seed | round | m | bracket | LP / explore / fill / fallback | non-ok expert statuses | objective by layer | mean δ̂ by layer |
|---|---|---|---|---|---|---|---|---|
| coverage_lp | 0 | 0 | 32 | – | 48 / 0 / 0 / 0 | none | L1 60.000, L8 59.896, L16 59.999, L23 59.924 | L1 1.0000, L8 1.0000, L16 1.0000, L23 1.0000 |
| coverage_lp | 0 | 1 | 80 | – | 48 / 0 / 0 / 0 | none | L1 36.234, L8 36.997, L16 36.559, L23 37.330 | L1 1.0000, L8 1.0000, L16 1.0000, L23 1.0000 |
| coverage_lp | 1 | 0 | 32 | – | 48 / 0 / 0 / 0 | none | L1 60.000, L8 59.905, L16 59.987, L23 59.798 | L1 1.0000, L8 1.0000, L16 1.0000, L23 1.0000 |
| coverage_lp | 1 | 1 | 80 | – | 48 / 0 / 0 / 0 | none | L1 36.240, L8 36.813, L16 36.526, L23 37.468 | L1 1.0000, L8 1.0000, L16 1.0000, L23 1.0000 |
| coverage_lp | 2 | 0 | 32 | – | 48 / 0 / 0 / 0 | none | L1 60.000, L8 59.967, L16 60.000, L23 59.891 | L1 1.0000, L8 1.0000, L16 1.0000, L23 1.0000 |
| coverage_lp | 2 | 1 | 80 | – | 48 / 0 / 0 / 0 | none | L1 36.251, L8 37.222, L16 36.883, L23 37.041 | L1 1.0000, L8 1.0000, L16 1.0000, L23 1.0000 |
| coverage_lp | all | | | | LP-selected fraction of adaptive picks: 1.000 | | | |
| global_exposure | 0 | 0 | 32 | 32–32 (w=0.00) | 48 / 0 / 0 / 0 | none | L1 3.427, L8 3.410, L16 3.417, L23 3.396 | L1 0.0571, L8 0.0569, L16 0.0569, L23 0.0568 |
| global_exposure | 0 | 1 | 80 | 64–128 (w=0.32) | 48 / 0 / 0 / 0 | none | L1 1.513, L8 1.559, L16 1.544, L23 1.579 | L1 0.0417, L8 0.0419, L16 0.0419, L23 0.0420 |
| global_exposure | 1 | 0 | 32 | 32–32 (w=0.00) | 48 / 0 / 0 / 0 | above 0.4% | L1 3.426, L8 3.366, L16 3.413, L23 3.393 | L1 0.0571, L8 0.0571, L16 0.0569, L23 0.0568 |
| global_exposure | 1 | 1 | 80 | 64–128 (w=0.32) | 48 / 0 / 0 / 0 | none | L1 1.522, L8 1.582, L16 1.584, L23 1.616 | L1 0.0418, L8 0.0420, L16 0.0421, L23 0.0422 |
| global_exposure | 2 | 0 | 32 | 32–32 (w=0.00) | 48 / 0 / 0 / 0 | above 0.8% | L1 3.426, L8 3.370, L16 3.370, L23 3.407 | L1 0.0571, L8 0.0572, L16 0.0571, L23 0.0568 |
| global_exposure | 2 | 1 | 80 | 64–128 (w=0.32) | 48 / 0 / 0 / 0 | none | L1 1.533, L8 1.570, L16 1.576, L23 1.619 | L1 0.0419, L8 0.0420, L16 0.0421, L23 0.0422 |
| global_exposure | all | | | | LP-selected fraction of adaptive picks: 1.000 | | | |
| layer_budget | 0 | 0 | 32 | 32–32 (w=0.00) | 48 / 0 / 0 / 0 | none | L1 3.713, L8 3.363, L16 3.444, L23 3.119 | L1 0.0619, L8 0.0562, L16 0.0575, L23 0.0520 |
| layer_budget | 0 | 1 | 80 | 64–128 (w=0.32) | 48 / 0 / 0 / 0 | none | L1 1.462, L8 1.639, L16 1.639, L23 1.506 | L1 0.0402, L8 0.0439, L16 0.0433, L23 0.0396 |
| layer_budget | 1 | 0 | 32 | 32–32 (w=0.00) | 48 / 0 / 0 / 0 | above 0.4% | L1 3.713, L8 3.318, L16 3.447, L23 3.108 | L1 0.0619, L8 0.0563, L16 0.0575, L23 0.0520 |
| layer_budget | 1 | 1 | 80 | 64–128 (w=0.32) | 48 / 0 / 0 / 0 | none | L1 1.477, L8 1.654, L16 1.602, L23 1.510 | L1 0.0403, L8 0.0440, L16 0.0432, L23 0.0397 |
| layer_budget | 2 | 0 | 32 | 32–32 (w=0.00) | 48 / 0 / 0 / 0 | above 0.8% | L1 3.713, L8 3.317, L16 3.395, L23 3.108 | L1 0.0619, L8 0.0563, L16 0.0576, L23 0.0520 |
| layer_budget | 2 | 1 | 80 | 64–128 (w=0.32) | 48 / 0 / 0 / 0 | none | L1 1.461, L8 1.619, L16 1.603, L23 1.514 | L1 0.0402, L8 0.0438, L16 0.0432, L23 0.0397 |
| layer_budget | all | | | | LP-selected fraction of adaptive picks: 1.000 | | | |
| layer_aware | 0 | 0 | 32 | 32–32 (w=0.00) | 48 / 0 / 0 / 0 | none | L1 3.727, L8 3.374, L16 3.460, L23 3.084 | L1 0.0622, L8 0.0563, L16 0.0577, L23 0.0514 |
| layer_aware | 0 | 1 | 80 | 64–128 (w=0.32) | 48 / 0 / 0 / 0 | none | L1 1.440, L8 1.610, L16 1.575, L23 1.477 | L1 0.0395, L8 0.0436, L16 0.0429, L23 0.0394 |
| layer_aware | 1 | 0 | 32 | 32–32 (w=0.00) | 48 / 0 / 0 / 0 | above 0.4% | L1 3.729, L8 3.326, L16 3.458, L23 3.076 | L1 0.0622, L8 0.0564, L16 0.0577, L23 0.0514 |
| layer_aware | 1 | 1 | 80 | 64–128 (w=0.32) | 48 / 0 / 0 / 0 | none | L1 1.445, L8 1.657, L16 1.608, L23 1.500 | L1 0.0395, L8 0.0438, L16 0.0430, L23 0.0395 |
| layer_aware | 2 | 0 | 32 | 32–32 (w=0.00) | 48 / 0 / 0 / 0 | above 0.8% | L1 3.729, L8 3.324, L16 3.407, L23 3.083 | L1 0.0622, L8 0.0563, L16 0.0578, L23 0.0514 |
| layer_aware | 2 | 1 | 80 | 64–128 (w=0.32) | 48 / 0 / 0 / 0 | none | L1 1.443, L8 1.616, L16 1.594, L23 1.481 | L1 0.0395, L8 0.0437, L16 0.0430, L23 0.0394 |
| layer_aware | all | | | | LP-selected fraction of adaptive picks: 1.000 | | | |

## Mechanism vs random (per expert, seed mean; descriptive)

| strategy | routing JS vs pool | experts with more tokens than random | P(worse \| more tokens) | P(worse \| fewer tokens) | corr(log token ratio, log error ratio) |
|---|---|---|---|---|---|
| random | 0.0002 | – | – | – | – |
| coverage | 0.0007 | 0.55 | 1.00 | 1.00 | +0.18 |
| coverage_lp | 0.0007 | 0.55 | 0.99 | 0.98 | -0.04 |
| global_exposure | 0.0006 | 0.53 | 0.96 | 0.96 | -0.08 |
| layer_budget | 0.0009 | 0.53 | 0.99 | 1.00 | +0.10 |
| layer_aware | 0.0006 | 0.54 | 0.99 | 0.97 | +0.04 |

## Atypicality of the sequences picked after the warm start (routing cache; seed mean)

Pool mean: per-sequence L1 to pool routing 0.337, routing entropy 3.983 nats.

| strategy | mean L1(sequence routing, pool routing) | mean per-sequence routing entropy |
|---|---|---|
| random | 0.330 | 3.988 |
| coverage | 0.415 | 3.935 |
| coverage_lp | 0.399 | 3.945 |
| global_exposure | 0.402 | 3.942 |
| layer_budget | 0.414 | 3.934 |
| layer_aware | 0.404 | 3.941 |

## Compute (seed mean)

| strategy | selection s | Hessian replay s | GPTQ s | routing (full-forward equiv.) s | peak GPU GB | equiv. extra random seqs | est. random error at equal compute |
|---|---|---|---|---|---|---|---|
| random | 0.0 | 31 | 203 | 0 | 7.8 | 0 | 0.004616 (B≈128, interpolated, NOT run) |
| coverage | 0.0 | 49 | 258 | 357 | 7.8 | 2000 | – |
| coverage_lp | 1.2 | 55 | 256 | 357 | 7.8 | 2007 | – |
| global_exposure | 1.2 | 50 | 257 | 357 | 7.8 | 2007 | – |
| layer_budget | 1.2 | 56 | 254 | 357 | 7.8 | 2007 | – |
| layer_aware | 1.2 | 27 | 202 | 357 | 7.8 | 2007 | – |

Notes: three seeds = three paired replicates; no confidence intervals are claimed. Expert win fractions are descriptive (experts are not independent replicates). Equal-compute errors are interpolations of the existing random headroom curve, not runs.
