# P0 — extended damping study + per-projection threshold test

seeds [np.int64(0), np.int64(1)]; layers [np.int64(1), np.int64(8), np.int64(16), np.int64(23)]; nested prefixes [16, 32, 64, 128, 256]; tuning split only.

## 1. Damping grid (all seeds x layers)

| percdamp | mean err (avg over prefixes) | % experts better than RTN | 16->32 improved % | 32->64 improved % | 64->128 improved % | 128->256 improved % |
|---|---|---|---|---|---|---|
| 0.01 | 0.00775 | 44.5% | 10% | 100% | 100% | 100% |
| 0.1 | 0.00576 | 76.3% | 93% | 100% | 100% | 100% |
| 0.2 | 0.00533 | 93.5% | 100% | 100% | 100% | 100% |
| 0.3 | 0.00514 | 98.8% | 100% | 100% | 100% | 100% |
| 0.5 | 0.00497 | 100.0% | 100% | 100% | 100% | 100% |
| 1.0 | 0.00487 | 100.0% | 100% | 100% | 100% | 100% |
| 1.5 | 0.00487 | 100.0% | 100% | 100% | 100% | 100% |
| 2.0 | 0.00490 | 100.0% | 100% | 100% | 100% | 100% |
| 3.0 | 0.00496 | 100.0% | 100% | 100% | 100% | 100% |
| 5.0 | 0.00508 | 100.0% | 100% | 100% | 100% | 100% |

**Best overall: percdamp = 1.0** — interior to the grid, safe to freeze.

Best percdamp per prefix (does the optimum depend on budget?):

| prefix | best | its err | RTN err |
|---|---|---|---|
| 16 | 3.0 | 0.00526 | 0.00679 |
| 32 | 2.0 | 0.00503 | 0.00679 |
| 64 | 1.0 | 0.00482 | 0.00679 |
| 128 | 1.0 | 0.00460 | 0.00679 |
| 256 | 0.5 | 0.00439 | 0.00679 |

## 2. Per-projection error vs routed tokens / input dim

GPTQ/RTN error ratio (mean over expert-prefix points). Double-descent prediction: peak near tokens/d_in ~ 1 for BOTH groups; damping flattens it.

### percdamp = 0.01

| tokens / d_in | gate+up ratio (n) | down ratio (n) |
|---|---|---|
| [0, 0.25) | 1.272 (172) | – (2) |
| [0.25, 0.5) | 1.329 (433) | 2.101 (460) |
| [0.5, 0.75) | 1.389 (353) | 2.430 (194) |
| [0.75, 1.0) | 1.274 (113) | 2.559 (296) |
| [1.0, 1.25) | 1.125 (345) | – (14) |
| [1.25, 1.5) | 1.021 (24) | 1.821 (173) |
| [1.5, 2.0) | 0.864 (103) | 1.638 (297) |
| [2.0, 3.0) | 0.788 (377) | 1.267 (171) |
| [3.0, 5.0) | 0.666 (457) | 1.162 (319) |
| [5.0, inf) | 0.634 (23) | 1.010 (474) |

Peak bin: gate+up [0.5, 0.75), down [0.75, 1.0).

### percdamp = 1.0

| tokens / d_in | gate+up ratio (n) | down ratio (n) |
|---|---|---|
| [0, 0.25) | 0.776 (172) | – (2) |
| [0.25, 0.5) | 0.744 (433) | 1.014 (460) |
| [0.5, 0.75) | 0.712 (353) | 0.974 (194) |
| [0.75, 1.0) | 0.691 (113) | 0.981 (296) |
| [1.0, 1.25) | 0.672 (345) | – (14) |
| [1.25, 1.5) | 0.662 (24) | 0.943 (173) |
| [1.5, 2.0) | 0.657 (103) | 0.941 (297) |
| [2.0, 3.0) | 0.640 (377) | 0.911 (171) |
| [3.0, 5.0) | 0.624 (457) | 0.902 (319) |
| [5.0, inf) | 0.598 (23) | 0.885 (474) |

Peak bin: gate+up [0, 0.25), down [0.25, 0.5).

## 3. Within a fixed prefix (percdamp = 0.01): token quintile -> GPTQ/RTN ratio

Rising with tokens below the threshold and falling above it supports a peak in tokens rather than in the number of sequences.

| prefix | group | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|---|
| 16 | gu | 1.263 | 1.289 | 1.261 | 1.262 | 1.376 |
| 16 | down | 1.920 | 1.996 | 2.079 | 2.177 | 2.392 |
| 32 | gu | 1.395 | 1.405 | 1.365 | 1.393 | 1.400 |
| 32 | down | 2.359 | 2.524 | 2.517 | 2.622 | 2.508 |
| 64 | gu | 1.271 | 1.227 | 1.120 | 1.085 | 1.072 |
| 64 | down | 1.845 | 1.797 | 1.705 | 1.643 | 1.520 |

## 4. Effective rank of H vs routed tokens (median over experts)

| prefix | median tokens | gate+up erank / 2048 | down erank / 1408 |
|---|---|---|---|
| 16 | 537 | 0.084 | 0.161 |
| 32 | 1083 | 0.132 | 0.267 |
| 64 | 2170 | 0.181 | 0.385 |
| 128 | 4336 | 0.225 | 0.503 |
| 256 | 8660 | 0.254 | 0.582 |
