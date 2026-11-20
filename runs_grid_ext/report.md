# Damping-grid extension report

**Verdict: `EXT-INTERIOR-CONFIRMED`**  (measured down gate: PASS)

Records: E1 40/40, E2 280/280. Failed cells 0, flat cells 0. E2 anchor (ν=5 vs Part B): 2016 comparisons, max rel 0.

## Q1 budget-level ν* (E1, expert-mean R_ideal, 42-point grid)

| group | prefix | median n/d | ν* [95% CI] | interior | old Part B argmin pred/meas | Part C power law |
|---|---|---|---|---|---|---|
| gu | 16 | 0.26 | 2.37 [2.26, 2.49] | yes | 2.0 / 2.0 | 2.16 (n<d) |
| gu | 32 | 0.53 | 1.56 [1.48, 1.63] | yes | 1.5 / 1.5 | 1.41 (n<d) |
| gu | 64 | 1.06 | 1.01 [0.96, 1.06] | yes | 1.0 / 1.0 | 0.93 (n>d) |
| gu | 128 | 2.12 | 0.63 [0.60, 0.66] | yes | 0.5 / 0.5 | 0.57 (n>d) |
| gu | 256 | 4.23 | 0.37 [0.35, 0.39] | yes | 0.3 / 0.3 | 0.34 (n>d) |
| down | 16 | 0.38 | 6.44 [5.64, 7.39] | yes | 5.0 / 5.0 | 5.48 (n<d) |
| down | 32 | 0.77 | 4.12 [3.68, 4.56] | yes | 5.0 / 5.0 | 3.08 (n<d) |
| down | 64 | 1.54 | 2.65 [2.40, 2.92] | yes | 3.0 / 3.0 | 1.89 (n>d) |
| down | 128 | 3.08 | 1.64 [1.48, 1.80] | yes | 1.5 / 1.5 | 1.12 (n>d) |
| down | 256 | 6.15 | 1.00 [0.91, 1.11] | yes | 1.0 / 1.0 | 0.66 (n>d) |

## Q2 measured confirmation (B2, 18-point merged grid)

| group | prefix | cells | median Spearman | argmin agree ±1 | median argmin pred / meas | frac meas argmin at 50 |
|---|---|---|---|---|---|---|
| gu | 16 | 96 | 0.998 | 100.0% | 2.5 / 2.5 | 0.0% |
| gu | 32 | 96 | 0.998 | 100.0% | 1.5 / 1.5 | 0.0% |
| gu | all | 192 | 0.998 | 100.0% | 2 / 2 | 0.0% |
| down | 16 | 96 | 0.998 | 96.9% | 10 / 10 | 3.1% |
| down | 32 | 96 | 0.996 | 97.9% | 5 / 5 | 1.0% |
| down | 64 | 96 | 0.994 | 96.9% | 3 / 3 | 1.0% |
| down | 128 | 96 | 0.994 | 96.9% | 2 / 2 | 0.0% |
| down | 256 | 96 | 0.990 | 97.9% | 1 / 1 | 0.0% |
| down | all | 480 | 0.996 | 97.3% | 3 / 3 | 1.0% |

Clamp fraction (down, mean / max over B2 cells), original W and working weights:

| ν | clamp W mean | clamp W max | clamp work mean | clamp work max |
|---|---|---|---|---|
| 0.01 | 1.39e-03 | 2.14e-03 | 5.33e-04 | 8.40e-04 |
| 0.03 | 1.25e-03 | 1.83e-03 | 4.64e-04 | 6.93e-04 |
| 0.05 | 1.16e-03 | 1.61e-03 | 4.29e-04 | 6.31e-04 |
| 0.1 | 1.02e-03 | 1.36e-03 | 3.75e-04 | 5.30e-04 |
| 0.2 | 8.57e-04 | 1.12e-03 | 3.15e-04 | 4.49e-04 |
| 0.3 | 7.60e-04 | 1.02e-03 | 2.80e-04 | 3.96e-04 |
| 0.5 | 6.38e-04 | 8.41e-04 | 2.35e-04 | 3.29e-04 |
| 1.0 | 4.82e-04 | 6.68e-04 | 1.78e-04 | 2.54e-04 |
| 1.5 | 3.97e-04 | 5.28e-04 | 1.47e-04 | 2.15e-04 |
| 2.0 | 3.44e-04 | 4.68e-04 | 1.28e-04 | 1.89e-04 |
| 3.0 | 2.76e-04 | 3.85e-04 | 1.02e-04 | 1.53e-04 |
| 5.0 | 2.03e-04 | 3.00e-04 | 7.58e-05 | 1.16e-04 |
| 7.0 | 1.64e-04 | 2.45e-04 | 6.14e-05 | 9.54e-05 |
| 10.0 | 1.29e-04 | 2.06e-04 | 4.87e-05 | 7.91e-05 |
| 15.0 | 9.89e-05 | 1.57e-04 | 3.69e-05 | 6.28e-05 |
| 20.0 | 8.27e-05 | 1.35e-04 | 3.04e-05 | 5.48e-05 |
| 30.0 | 6.67e-05 | 1.02e-04 | 2.38e-05 | 4.40e-05 |
| 50.0 | 5.47e-05 | 8.08e-05 | 1.83e-05 | 3.23e-05 |

## Q3 practical size (excess of the old ν over ν*)

| group | prefix | pred excess ν=5 (budget / cell median) | pred excess ν=1 (budget / cell median) | meas argmin (B2 mean) | meas excess ν=5 (mean / cell median) | meas excess ν=1 (mean / cell median) |
|---|---|---|---|---|---|---|
| gu | 16 | +1.13% / +0.98% | +1.84% / +1.99% | 2 (18-pt) | +1.02% / +0.84% | +1.76% / +1.80% |
| gu | 32 | +3.19% / +2.75% | +0.60% / +0.85% | 1.5 (18-pt) | +3.02% / +2.79% | +0.61% / +0.81% |
| gu | 64 | +6.27% / +5.61% | +0.00% / +0.16% | 1 (12-pt) | +6.03% / +5.84% | +0.00% / +0.10% |
| gu | 128 | +10.46% / +9.46% | +0.63% / +0.42% | 0.5 (12-pt) | +9.92% / +9.61% | +0.40% / +0.33% |
| gu | 256 | +15.00% / +13.83% | +2.35% / +1.96% | 0.3 (12-pt) | +14.35% / +14.12% | +2.07% / +2.03% |
| down | 16 | +0.08% / +0.46% | +6.75% / +7.88% | 7 (18-pt) | +0.01% / +0.58% | +6.28% / +7.86% |
| down | 32 | +0.06% / +0.31% | +4.50% / +5.24% | 5 (18-pt) | +0.00% / +0.34% | +4.26% / +5.66% |
| down | 64 | +0.59% / +0.44% | +2.12% / +2.52% | 3 (18-pt) | +0.44% / +0.25% | +2.10% / +2.92% |
| down | 128 | +1.79% / +1.38% | +0.49% / +0.85% | 1.5 (18-pt) | +1.42% / +0.93% | +0.55% / +1.29% |
| down | 256 | +3.42% / +2.85% | +0.00% / +0.40% | 1 (18-pt) | +2.74% / +2.08% | +0.00% / +0.44% |

## Q4 ν*_down / ν*_gu

| prefix | budget-level | per-cell median [IQR] |
|---|---|---|
| 16 | 2.72 | 3.51 [2.43, 4.96] (n=478) |
| 32 | 2.64 | 3.09 [2.24, 4.53] (n=478) |
| 64 | 2.62 | 3.00 [2.13, 4.36] (n=478) |
| 128 | 2.62 | 2.86 [2.04, 4.07] (n=478) |
| 256 | 2.71 | 2.87 [2.07, 4.09] (n=478) |

All matched cells: 3.05 [2.16, 4.44] (n=2390, 10 excluded for an edge ν*). Trend: slope of log ratio on log tokens = -0.076 [-0.095, -0.056] (cluster bootstrap); budget-level slope -0.002.

## Q5 per-cell ν* distribution

| group | prefix | n | median | IQR | p5 / p95 | at 1e2 edge | at 1e-2 edge |
|---|---|---|---|---|---|---|---|
| gu | 16 | 480 | 2.46 | 1.99–3.1 | 0.93 / 4.07 | 0.0% | 0.0% |
| gu | 32 | 480 | 1.66 | 1.29–2 | 0.604 / 2.57 | 0.0% | 0.0% |
| gu | 64 | 480 | 1.09 | 0.848–1.29 | 0.428 / 1.63 | 0.0% | 0.0% |
| gu | 128 | 480 | 0.682 | 0.512–0.81 | 0.277 / 1.03 | 0.0% | 0.0% |
| gu | 256 | 480 | 0.405 | 0.3–0.491 | 0.164 / 0.636 | 0.0% | 0.0% |
| down | 16 | 480 | 8.29 | 4.85–14.1 | 2.29 / 34.6 | 0.4% | 0.0% |
| down | 32 | 480 | 4.92 | 2.87–8.19 | 1.45 / 17.8 | 0.0% | 0.4% |
| down | 64 | 480 | 3.02 | 1.83–5.1 | 0.825 / 10.6 | 0.0% | 0.4% |
| down | 128 | 480 | 1.78 | 1.11–2.9 | 0.51 / 6.04 | 0.0% | 0.4% |
| down | 256 | 480 | 1.08 | 0.644–1.73 | 0.299 / 3.62 | 0.0% | 0.4% |

Edge experts (layer, expert, group: cells (seed, prefix, side)):

- L1 e6 down: s1/B16/hi
- L1 e42 down (T1 outlier): s0/B32/lo, s0/B64/lo, s0/B128/lo, s0/B256/lo, s1/B32/lo, s1/B64/lo, s1/B128/lo, s1/B256/lo
- L8 e51 down: s1/B16/hi

T1 outlier experts, down ν* (seed 0, seed 1) vs layer median:

- L1 e0 B16: 0.0393, 0.0362 (layer median 5.72)
- L1 e0 B32: 0.0329, 0.0312 (layer median 3.23)
- L1 e0 B64: 0.026, 0.0243 (layer median 1.84)
- L1 e0 B128: 0.0169, 0.0174 (layer median 1.12)
- L1 e0 B256: 0.013, 0.0118 (layer median 0.624)
- L1 e33 B16: 3.31, 3.86 (layer median 5.72)
- L1 e33 B32: 1.5, 1.96 (layer median 3.23)
- L1 e33 B64: 0.942, 1.26 (layer median 1.84)
- L1 e33 B128: 0.569, 0.678 (layer median 1.12)
- L1 e33 B256: 0.344, 0.393 (layer median 0.624)
- L1 e42 B16: 0.0133, 0.0123 (layer median 5.72)
- L1 e42 B32: 0.01, 0.01 (layer median 3.23)
- L1 e42 B64: 0.01, 0.01 (layer median 1.84)
- L1 e42 B128: 0.01, 0.01 (layer median 1.12)
- L1 e42 B256: 0.01, 0.01 (layer median 0.624)
- L23 e25 B16: 2.54, 2.3 (layer median 7.05)
- L23 e25 B32: 1.45, 1.45 (layer median 4.5)
- L23 e25 B64: 0.69, 0.838 (layer median 2.83)
- L23 e25 B128: 0.351, 0.359 (layer median 1.68)
- L23 e25 B256: 0.145, 0.143 (layer median 1.09)
- L23 e26 B16: 1.84, 2.06 (layer median 7.05)
- L23 e26 B32: 1.11, 1.45 (layer median 4.5)
- L23 e26 B64: 0.695, 0.945 (layer median 2.83)
- L23 e26 B128: 0.455, 0.588 (layer median 1.68)
- L23 e26 B256: 0.322, 0.379 (layer median 1.09)
- L23 e57 B16: 0.212, 0.235 (layer median 7.05)
- L23 e57 B32: 0.145, 0.233 (layer median 4.5)
- L23 e57 B64: 0.143, 0.17 (layer median 2.83)
- L23 e57 B128: 0.116, 0.128 (layer median 1.68)
- L23 e57 B256: 0.0842, 0.0968 (layer median 1.09)

Figures: `figures/mean_curves.png`, `figures/nu_star_vs_ratio.png`, `figures/nu_star_hist.png`.
