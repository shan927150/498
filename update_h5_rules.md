# H5: deployable damping rules, and what sets ν* (Part K)

**Frozen H5 verdict: `H5-PASS-R4`.** Rules are checked in order of simplicity, and the first one that passes is the
label:

| rule | overall oracle capture (measured `all`) [95 % CI] | gain vs R0 at B16 [CI] | gain vs R0 at B32 [CI] | pass? |
|---|---|---|---|---|
| R2: group budget law c_g·(d/ñ)^{2/3} | 0.71 [0.67, 0.75] | +2.53 % [+2.33, +2.70] | +1.15 % [+1.04, +1.26] | no (capture < 0.80) |
| R3: per-expert law c'_g·(d/n_e)^{2/3} | 0.71 [0.67, 0.75] | +2.54 % [+2.34, +2.71] | +1.14 % [+1.02, +1.24] | no (capture < 0.80) |
| **R4: per-expert unit CV (8 blocks)** | **0.96 [0.92, 0.98]** | **+2.70 % [+2.52, +2.86]** | **+1.43 % [+1.27, +1.57]** | **yes** |

**Frozen K verdict: `K-NEITHER`, a near miss on both sides.**
- **Tails alone.** Heavy tails along the top eigendirections (κ_top) shrink the down/gate+up gap in log ν* by
  47 %, just under the 50 % bar. Within-down R² rises by only +0.06, against a bar of 0.10.
- **Spectrum alone.** Spectrum flatness (erank/d) raises within-down R² by +0.23 but shrinks the gap by only 39 %.
- **Both together (M3).** The gap shrinks by 75 % and within-down R² goes from 0.43 to 0.71.
- **Reading.** Down's larger damping needs both tails and spectrum; neither explains it alone.

**Practical recommendation.** Choose ν_gu and ν_down per expert by 8-block unit cross-validation of the calibration
data (R4). It is the simplest rule that passed. It captures 96 % of the oracle's gain over ν = 1.0 using
calibration data only, against about 71 % for any fitted power law.

Protocol, split, rule definitions, fitting objective, grids, CV folds, metrics, κ definitions, regression specs and
both verdict rules were frozen in `runs_h5/protocol.json` (sha256 `84a0745b…`) before any fit or measurement.
There are no deviations. One sanctioned refactor and some descriptive additions are listed at the end and in
`runs_h5/deviations.json`.

## Setup

**Split** (out of sample, tuning split only):

| role | data |
|---|---|
| fit set | seed 0, layers {1, 8}, all 60 experts, prefixes 16–256 |
| eval set (primary) | seed 1, layers {16, 23}, all 60 experts, prefixes 16–256 |
| swapped fold (secondary) | fit on seed 1 {16, 23}, scored with R_ideal on seed 0 {1, 8} |

**Rules:**
- **R0:** ν = 1.0, the frozen baseline.
- **R1:** one shared ν per prefix, from fit-set measured `all`. This is current practice.
- **R1g:** one ν per (prefix, group), from the fit-set expert-mean R_ideal curve.
- **R2 / R2f:** group budget law, using ñ = the layer's median routed tokens. R2f frees the exponent.
- **R3 / R3f:** per-expert law, using n_e = the expert's routed tokens. R3f frees the exponent.
- **R4:** Chen & Yeh unit CV. 8 contiguous sequence blocks, full-prefix λ kept when a fold is left out,
  15-point log grid plus parabola refinement.
- **R5:** the oracle, per-(expert, group) argmin of R_ideal using the eval-set Σ.

**Fitting and evaluation:**
- The laws minimize the fit-set excess R_ideal(ν_rule)/R_ideal(ν*) − 1. R_ideal comes from PCHIP interpolation of
  the 42-point curves in `runs_grid_ext/raw_e1.jsonl`, which were reused, not recomputed.
- The eval set was quantized with every rule's (ν_gu, ν_down), with gate+up and down damped separately and GPTQ
  run once per distinct ν per group. Each result was scored with `tune_err_groups`.
- The eval took 1.84 GPU-h, of which R4's CV was 0.70 h. Part K took 2 min.

## Correctness checks (all passed)

| Check | Result |
|---|---|
| `tests/test_h5_rules.py` (21 tests) | unit CV = brute-force leave-one-block-out refit (independent nested-ridge construction) to ≤ 1e-10, including a dead column and an empty block; vectorized PCHIP = scipy; law fit recovers a planted c and b |
| `tests/test_theorem1.py` after the `m_from_A` refactor | 40 / 40 pass |
| Part B R_ideal at the 12 grid ν, 60 experts × 2 groups (seed 1, L16, B16) | bitwise identical (max rel 0.0) |
| Per-group quantizer with ν_gu = ν_down = 1.0 vs `low_data_diag.quant` (3 experts) | bitwise identical |
| R0 measured `tune_err_groups` (all/gu/down) vs `damping_threshold_raw.jsonl` at ν = 1.0, all 600 eval cells | ≤ 1e-6 relative |
| R4 fold Grams: Σ block tokens = Profiler token count, every cell | exact |
| R4 cells: Cholesky failures / empty folds / CV argmin at grid edge | 0 / 0 / 0 % |
| Part K cells dropped (< 64 tuning rows) | 0 of 480 |
| `moe_calib/gptq.py` sha256 | `58e40d1c…` (unchanged) |

## 1. Fitted constants (fit set; 95 % cluster-bootstrap CI over (layer, expert))

| fold (fit data) | group | R2 c | R2f c, b [CI of b] | R3 c' | R3f c', b' [CI of b'] | fit excess R2 / R3 |
|---|---|---|---|---|---|---|
| primary (s0, L1/L8) | gu | **0.85** [0.76, 0.94] | 0.86, 0.72 [0.72, 0.75] | **0.84** [0.75, 0.93] | 0.85, 0.73 [0.72, 0.75] | 1.02 % / 0.99 % |
| primary (s0, L1/L8) | down | **2.12** [1.17, 4.12] | 2.04, 0.59 [0.49, 0.75] | **2.13** [1.19, 4.09] | 2.09, 0.63 [0.54, 0.76] | 4.31 % / 4.19 % |
| swapped (s1, L16/L23) | gu | 1.08 [1.03, 1.12] | 1.06, 0.60 [0.59, 0.61] | 1.09 [1.05, 1.13] | 1.08, 0.61 [0.59, 0.62] | 0.15 % / 0.12 % |
| swapped (s1, L16/L23) | down | 2.83 [2.34, 3.37] | 2.75, 0.61 [0.56, 0.67] | 2.86 [2.35, 3.38] | 2.81, 0.63 [0.57, 0.70] | 1.00 % / 0.96 % |

| prefix | R1 shared ν (bootstrap 2.5–97.5 %) | R1g gu [CI] | R1g down [CI] | R1g swapped gu / down |
|---|---|---|---|---|
| 16 | 3.0 (2.0–3.0) | 2.32 [2.13, 2.54] | 7.39 [5.91, 9.18] | 2.40 / 5.68 |
| 32 | 1.5 (1.5–2.0) | 1.47 [1.35, 1.60] | 4.43 [3.64, 5.27] | 1.62 / 3.74 |
| 64 | 1.0 (1.0–1.0) | 0.93 [0.85, 1.02] | 2.81 [2.38, 3.35] | 1.10 / 2.49 |
| 128 | 0.5 (0.5–0.5) | 0.56 [0.51, 0.61] | 1.73 [1.46, 2.04] | 0.71 / 1.56 |
| 256 | 0.3 (0.3–0.5) | 0.32 [0.29, 0.35] | 1.04 [0.89, 1.22] | 0.44 / 0.97 |

**The hypothesized constants did not hold up as constants.**
- **In-sample value.** The task stated c_gu ≈ 1.0 and c_down ≈ 3.4, from the grid extension's budget-level fit
  over all four layers.
- **Fitted values.** Fitted on layers {1, 8} by excess error, c_gu = 0.85 and c_down = 2.1. Fitted on layers
  {16, 23}, they are 1.08 and 2.8.
- **Transfer between folds.**
  - gate+up: the two folds' CIs do not overlap ([0.76, 0.94] vs [1.03, 1.12]).
  - down: the layer-{1, 8} CI is very wide, [1.2, 4.1]. Layer 1's low-ν outlier experts (L1 e0 / e42) plausibly
    pull the excess objective toward a smaller c (not tested separately).
- **Conclusion.** c depends on the layer, and that is what limits R2 and R3 out of sample.

**Exponents.** The free exponents straddle 2/3 and don't support any single value:
- **gate+up:** 0.72–0.73 on {1, 8} and 0.60–0.61 on {16, 23}. Both CIs exclude 2/3, in opposite directions.
- **down:** 0.59–0.63 on both folds. Both CIs include 2/3.

Freeing the exponent never improves the measured capture (R2f ≈ R2, R3f ≈ R3).

## 2. Eval set, measured `all` (seed 1, layers 16/23; gain vs R0 [95 % CI] / oracle capture)

| rule | B16 | B32 | B64 | B128 | B256 | overall capture [CI] |
|---|---|---|---|---|---|---|
| R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| R1 | +2.44 % [+2.25, +2.62] / 0.87 | +1.00 % [+0.92, +1.07] / 0.66 | +0.00 % / 0.00 | **−0.60 %** [−0.74, −0.45] / −1.25 | +0.40 % [+0.17, +0.62] / 0.30 | 0.51 [0.47, 0.55] |
| R1g | +2.50 % [+2.22, +2.74] / 0.89 | +1.22 % [+1.02, +1.38] / 0.81 | +0.23 % [+0.12, +0.33] / 0.41 | +0.09 % [+0.01, +0.16] / 0.18 | +0.94 % [+0.79, +1.08] / 0.71 | 0.76 [0.67, 0.81] |
| R2 | +2.53 % [+2.33, +2.70] / 0.90 | +1.15 % [+1.04, +1.26] / 0.77 | +0.08 % [+0.04, +0.13] / 0.15 | −0.04 % [−0.13, +0.05] / −0.09 | +0.93 % [+0.77, +1.10] / 0.70 | 0.71 [0.67, 0.75] |
| R2f | +2.54 % [+2.35, +2.71] / 0.91 | +1.18 % [+1.07, +1.28] / 0.79 | +0.09 % [+0.04, +0.13] / 0.16 | −0.07 % [−0.16, +0.02] / −0.15 | +0.89 % [+0.72, +1.05] / 0.67 | 0.71 [0.67, 0.75] |
| R3 | +2.54 % [+2.34, +2.71] / 0.91 | +1.14 % [+1.02, +1.24] / 0.76 | +0.07 % [+0.02, +0.11] / 0.12 | −0.03 % [−0.11, +0.06] / −0.05 | +0.94 % [+0.77, +1.10] / 0.70 | 0.71 [0.67, 0.75] |
| R3f | +2.57 % [+2.37, +2.75] / 0.92 | +1.18 % [+1.06, +1.29] / 0.79 | +0.07 % [+0.02, +0.11] / 0.12 | −0.09 % [−0.18, −0.00] / −0.20 | +0.83 % [+0.66, +1.01] / 0.63 | 0.70 [0.65, 0.74] |
| **R4** | **+2.70 %** [+2.52, +2.86] / 0.96 | **+1.43 %** [+1.27, +1.57] / 0.96 | **+0.50 %** [+0.39, +0.59] / 0.88 | **+0.45 %** [+0.40, +0.49] / 0.93 | **+1.31 %** [+1.20, +1.43] / 0.98 | **0.96 [0.92, 0.98]** |
| R5 (oracle) | +2.80 % [+2.64, +2.95] | +1.50 % [+1.38, +1.62] | +0.56 % [+0.49, +0.64] | +0.48 % [+0.42, +0.53] | +1.33 % [+1.21, +1.45] | 1.00 |

E_R0 per prefix: 0.00588, 0.00558, 0.00532, 0.00507, 0.00491. The oracle's headroom over R0 is small in the middle
(0.6 % at B64, 0.5 % at B128), so per-prefix captures there are noisy. The overall capture is gain-weighted.

**Where the fitted rules lose:**
- **B16 and B32.** All of them get 77–92 % of the oracle.
- **B64 and B128.** They get almost nothing, because ν = 1.0 is already near the group-level optimum there and
  their remaining error is per-expert.
- **B256.** About 70 %.

R4 tracks the oracle at every budget. R1, current practice, is *worse* than ν = 1.0 at B128 (−0.60 %), because its
single shared ν = 0.5 under-damps down.

## 3. Per group

| rule | gate+up overall capture [CI] | down overall capture [CI] | down B16 gain | down B256 gain |
|---|---|---|---|---|
| R1 | 0.74 [0.70, 0.77] | 0.17 [0.13, 0.21] | +5.12 % | −2.94 % |
| R1g | 0.79 [0.75, 0.82] | 0.90 [0.85, 0.98] | +5.71 % | +0.02 % |
| R2 | 0.70 [0.66, 0.74] | 0.76 [0.71, 0.86] | +5.60 % | −0.53 % |
| R2f | 0.70 [0.65, 0.74] | 0.76 [0.71, 0.86] | +5.47 % | −0.37 % |
| R3 | 0.70 [0.65, 0.74] | 0.77 [0.71, 0.86] | +5.62 % | −0.51 % |
| R3f | 0.68 [0.63, 0.72] | 0.77 [0.71, 0.87] | +5.56 % | −0.43 % |
| **R4** | **0.97 [0.96, 0.97]** | **0.90 [0.69, 0.99]** | +5.79 % | +0.15 % |
| R5 | 1.00 | 1.00 | +6.07 % | +0.28 % |

The full per-prefix tables are in `runs_h5/report.md` §3a/§3b.
- **Down's gains are 2–4× larger than gate+up's at low budget.** At B16 the oracle gains +6.1 % on down against
  +2.0 % on gate+up.
- **R1 is poor for down** (capture 0.17) because it shares one ν across groups.
- **R4 for down has a wide CI** ([0.69, 0.99]): CV with 8 folds is noisier per cell for down's dispersed ν*.

## 4. R4 − R3 (paired, % of E_R0; negative means R4 is better)

| metric | B16 | B32 | B64 | B128 | B256 | overall |
|---|---|---|---|---|---|---|
| all | −0.16 [−0.23, −0.10] | −0.29 [−0.35, −0.24] | −0.43 [−0.51, −0.35] | −0.47 [−0.54, −0.40] | −0.37 [−0.44, −0.31] | **−0.34 [−0.39, −0.29]** |
| gu | −0.05 [−0.09, −0.02] | −0.23 [−0.28, −0.19] | −0.38 [−0.45, −0.32] | −0.41 [−0.48, −0.34] | −0.29 [−0.35, −0.24] | −0.27 [−0.32, −0.22] |
| down | −0.17 [−0.48, +0.37] | −0.21 [−0.72, +0.74] | −0.34 [−0.97, +0.81] | −0.62 [−1.07, +0.15] | −0.66 [−1.09, +0.01] | −0.39 [−0.85, +0.42] |

R4 beats R3 at every budget on `all` and on gate+up, with CIs excluding 0. On down alone the point estimates favour
R4, but every CI includes 0.

## 5. Swapped fold (scored with R_ideal; overall oracle capture)

| rule | primary gu | primary down | primary pooled | swapped gu | swapped down | swapped pooled |
|---|---|---|---|---|---|---|
| R1 | 0.75 | 0.20 | 0.31 | – | – | – |
| R1g | 0.81 | 0.77 | 0.78 | 0.58 | 0.71 | 0.68 |
| R2 | 0.72 | 0.69 | 0.70 | 0.60 | 0.69 | 0.67 |
| R2f | 0.72 | 0.69 | 0.69 | 0.59 | 0.69 | 0.66 |
| R3 | 0.73 | 0.70 | 0.71 | 0.60 | 0.70 | 0.68 |
| R3f | 0.71 | 0.70 | 0.70 | 0.60 | 0.70 | 0.67 |
| R4 | 0.97 | 0.96 | 0.96 | – | – | – |

- **R_ideal scoring agrees with measurement.** In the primary fold, R_ideal-scored capture matches the measured
  capture (R2 0.70 vs 0.71, R4 0.96 vs 0.96).
- **Swapped direction.** Every fitted rule stays at 0.66–0.68 pooled when fitting on {16, 23} and scoring on
  {1, 8}. So the shortfall of the fitted rules is not specific to one fold.
- **Cause.** c moves between layers, and no per-group constant captures the per-expert dispersion.

## 6. Part K: tails or spectrum?

Regression of log ν* over 4,790 non-edge `raw_e1` cells, with a cluster bootstrap over (layer, expert).
κ_best = κ_top, the eigenvalue-weighted kurtosis along the top-16 eigendirections of Σ.

| model | 1[down] coefficient [95 % CI] | R² | R² down | R² gu | other coefficients |
|---|---|---|---|---|---|
| M0: log(d/n) + 1[down] | +1.38 [+1.29, +1.47] | 0.639 | 0.433 | 0.689 | log(d/n) +0.71 |
| M1, κ_coord | +1.16 [+0.35, +1.98] | 0.639 | 0.434 | 0.687 | log κ_coord +0.09 |
| M1, κ_dir | +1.43 [+1.21, +1.65] | 0.639 | 0.433 | 0.690 | log κ_dir −0.09 |
| **M1, κ_top (best)** | **+0.73** [+0.54, +0.95] | 0.671 | 0.494 | 0.693 | log κ_top +0.26 |
| **M2: + log(erank/d)** | **+0.85** [+0.75, +0.95] | 0.777 | 0.667 | 0.774 | log(erank/d) +0.73 |
| M3: + log κ_top + log(erank/d) | +0.35 [+0.20, +0.52] | 0.797 | 0.711 | 0.767 | log κ_top +0.21, log(erank/d) +0.70 |

**Conditions for each label:**
- **K-TAILS (M1 vs M0):** the group coefficient shrinks by 47.0 % (needs ≥ 50 %), and within-down R² rises by
  +0.062 (needs ≥ 0.10). Not met.
- **K-SPECTRUM (M2 vs M0):** the group coefficient shrinks by 38.8 % (needs ≥ 50 %), and within-down R² rises by
  +0.234. Not met. → **`K-NEITHER`**.

**Group medians:**

| statistic | gate+up | down |
|---|---|---|
| κ_coord | 3.1 | 34.7 |
| κ_dir | 3.0 | 4.4 |
| κ_top | 2.3 | 29.4 |
| erank/d | 0.22 | 0.50 |
| participation ratio/d | 0.02 | 0.24 |

**Interpretation (descriptive):**
- **Two differences between the groups.** Down's inputs (the SwiGLU products) are extremely heavy-tailed along a
  few coordinates and along the top eigendirections, but close to Gaussian along random directions. Down's spectrum
  is also much flatter.
- **Effective rank does most of the work.** It is the stronger single explanation of which down experts want more
  damping (within-down R² from 0.43 to 0.67). Flatter spectra want more damping, with an elasticity of about 0.7.
- **Top-direction tails explain more of the gap between groups** (47 % against 39 %).
- **Combined.** Together they remove three quarters of the group gap: e^{1.38} ≈ 4.0× falls to e^{0.35} ≈ 1.4×.
- **Coordinate and random-direction kurtosis carry no signal** once n is controlled.

Neither mechanism alone meets the frozen bar, so the down/gate+up difference is best described as tails *and*
spectrum.

## Figures (`runs_h5/figures/`)

- `gain_by_rule.png`: measured gain vs R0 per prefix for every rule, with 95 % CIs.
- `kappa_vs_scaled_nu.png`: κ_top against ν*·(n_e/d)^{2/3} per (layer, expert, group), coloured by group.
- `group_coef.png`: the 1[down] coefficient across M0–M3 and the other κ variants.

## Deviations and changes

- **Deviations from the frozen protocol:** none.
- **Sanctioned refactor (numerics unchanged).** In `diagnostics/theorem1_check.py`, `m_matrix` now returns
  `m_from_A(damped(Hp, ν))`, and `m_from_A` holds the unchanged Cholesky chain. R4 needs this because its A₋f uses
  the full-prefix λ. The tests still pass (40 / 40), and Part B's R_ideal reproduces bitwise.
- **Descriptive additions:**
  - overall gain vs R0 with CI;
  - R_ideal-scored capture for the primary fold (in the protocol);
  - group medians of every κ and spectrum statistic;
  - `tests/test_h5_rules.py`.
- **Untouched:** `configs/`, `runs_diag/`, `runs_theorem1_check/`, `runs_grid_ext/` and the test and
  confirmation splits.

## New files

- `diagnostics/h5_rules.py` (`protocol | fit | eval | kappa`)
- `diagnostics/report_h5_rules.py`
- `tests/test_h5_rules.py`
- `update_h5_rules.md`
- `runs_h5/`:
  - `protocol.json` (frozen) and `deviations.json`
  - `fit.json`
  - `raw_eval.jsonl`, `raw_kappa.jsonl`, `events.jsonl`
  - `report.json` and `report.md`
  - `figures/gain_by_rule.png`, `figures/kappa_vs_scaled_nu.png`, `figures/group_coef.png`
  - `run_all.sh` and `logs/{fit,eval,kappa}.log`
- Modified: `diagnostics/theorem1_check.py` (`m_from_A`; untracked file from an earlier task)
