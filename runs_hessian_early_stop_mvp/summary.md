# Hessian early-stopping MVP

**Verdict (frozen rule): `EARLY-STOP-SIGNAL-FAIL`**

## Measured timing (4 layers, B=128)

forward 9.06 s | hook overhead 7.22 s | accumulation 8.04 s (events: conversion 1.65, GEMM 5.33, update 5.40) | replay 24.32 s | GPTQ solve 184.4 s

| Condition | rows skipped | replay (s) | replay reduction | replay+solve (s) | e2e reduction | peak GPU (GB) |
|---|---|---|---|---|---|---|
| full | 0.0 % | 24.32 | +0.0 % | 208.5 | +0.00 % | 8.72 |
| stop50 | 25.3 % | 22.28 | +8.4 % | 204.0 | +2.17 % | 8.72 |
| stop75 | 37.6 % | 21.24 | +12.7 % | 205.0 | +1.70 % | 8.72 |

## Quality-cost (offline; times are PROJECTED from the measured cost model)

| Strategy | mean err | Δ vs B128 | p90 Δ | false-safe >1 % | work skipped | projected total (s) |
|---|---|---|---|---|---|---|
| fixed_B32 | 0.005061 | +9.79 % | +9.16 % | 240/240 | 75.0 % | 202.7 |
| fixed_B64 | 0.004814 | +4.43 % | +4.13 % | 239/240 | 50.0 % | 204.7 |
| fixed_B128 | 0.004609 | +0.00 % | +0.00 % | 0/0 | 0.0 % | 208.7 |
| trace_q25 | 0.004638 | +0.62 % | +0.61 % | 18/18 | 5.0 % | 208.3 |
| frob_q25 | 0.004622 | +0.26 % | +0.00 % | 11/11 | 3.2 % | 208.5 |
| frob_exposure_q25 | 0.004617 | +0.16 % | +0.00 % | 7/7 | 2.1 % | 208.6 |
| exposure_q25 | 0.004953 | +7.46 % | +5.42 % | 200/200 | 61.8 % | 203.8 |
| trace_q50 (primary) | 0.004718 | +2.36 % | +0.73 % | 74/74 | 20.3 % | 207.1 |
| frob_q50 (primary) | 0.004631 | +0.47 % | +0.00 % | 20/20 | 5.7 % | 208.3 |
| frob_exposure_q50 (primary) | 0.004620 | +0.22 % | +0.00 % | 9/9 | 2.8 % | 208.5 |
| exposure_q50 | 0.004834 | +4.87 % | +2.07 % | 139/139 | 43.5 % | 205.2 |
| trace_q75 | 0.004884 | +5.96 % | +4.54 % | 174/175 | 49.9 % | 204.7 |
| frob_q75 | 0.004830 | +4.78 % | +2.70 % | 145/145 | 43.6 % | 205.2 |
| frob_exposure_q75 | 0.004685 | +1.64 % | +0.66 % | 52/52 | 16.9 % | 207.4 |
| exposure_q75 | 0.004720 | +2.41 % | +1.03 % | 72/72 | 23.2 % | 206.9 |
