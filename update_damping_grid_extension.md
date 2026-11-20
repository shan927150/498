# Damping-grid extension: the true optimum for down

**Frozen verdict: `EXT-INTERIOR-CONFIRMED`.** For every prefix and both groups, the budget-level ν* of R_ideal is
interior on a 42-point grid from 1e-2 to 1e2, and the measured gate for down passes:

| check | result | gate |
|---|---|---|
| budget-level ν* interior, all 10 (prefix, group) | 10 / 10 | all interior → pass |
| down, B2 cells, 18-point merged measured grid (ν 0.01 … 50) | median Spearman 0.996, argmin agreement 97.3 % (480 cells) | ≥ 0.8 and ≥ 70 % → PASS |

**Answer to the main question:**
- **Where down's optimum sits.** The ν = 5.0 at the top of the old Part B grid was an edge, but only just. The true
  budget-level optimum for down is ν* = 6.4 at B16 and 4.1 at B32.
- **What the edge cost.** The old grid's best point (5.0) is within 0.1 % of the optimum in both cases, predicted
  and measured.
- **The down/gate+up ratio.** The "down wants about 2.5×" headline becomes **≈ 2.7× at budget level**. It is
  constant across budgets (2.62–2.72 at every prefix). Per expert, the median is **3.0×**, falling slightly
  with more data.

Protocol, grids, cell definitions, the refinement rule and the verdict rule were frozen in `runs_grid_ext/protocol.json`
(sha256 `6270e27d…`) before any measurement. There are no deviations. One sanctioned refactor (R1) and some
descriptive additions are listed at the end and in `runs_grid_ext/deviations.json`.

## What was run

**E1: R_ideal only, no quantization.**
- **Cells:** all 60 experts × {gate+up, down} × prefixes {16, 32, 64, 128, 256} × layers {1, 8, 16, 23} × seeds
  {0, 1} = 2,400 cells per group.
- **ν grid:** 42 points, the union of 33 log-spaced values from 1e-2 to 1e2 and the 12 Part B points.
- **Per-cell ν*:** the grid argmin, refined by a parabola in log ν (`theorem1_check.sweep_curve`).
- **Budget-level ν*:** the parabola vertex of the expert-mean curve, the same quantity T1 reported.
- **Confidence intervals:** cluster bootstrap over (layer, expert), 2,000 resamples.

**E2: actual GPTQ on the B2 experts** (12 per layer, the set after deviation D1).
- **ν values:** {5, 7, 10, 15, 20, 30, 50}.
- **Prefixes:** all five for down; 16 and 32 for gate+up (the control).
- **Merged curves:** combined with the stored Part B points into 18-point measured curves per cell.

All math is imported from `diagnostics/theorem1_check.py`: Ĥ accumulation, Σ, M, R_ideal, `b2_cell` and
`tune_err_groups`.

Wall clock was 1.53 GPU-h: E1 0.85 h, E2 0.68 h. The estimate after the first (seed, layer) was 1.52 h.

## Correctness checks (all passed)

| Check | Result |
|---|---|
| `tests/test_theorem1.py` after the refactor | 40 / 40 pass |
| E1: R_ideal at the 12 Part B points vs `runs_theorem1_check/raw_grid.jsonl` (seed 0, L1, 60 experts × 2 groups × 5 prefixes) | max relative difference **0.0** (bitwise) |
| E2 anchor ν = 5.0: `tune_err_groups`, e_meas and R_ideal vs stored Part B, every B2 cell | 2,016 comparisons, max relative difference **0.0** |
| B2 expert sets recomputed vs `runs_theorem1_check/grid_events.jsonl` | identical on all 4 layers |
| Cholesky failures / flat curves | 0 / 0 |
| `moe_calib/gptq.py` sha256 | `58e40d1c…` (unchanged) |

## Q1: budget-level ν* (E1, expert-mean R_ideal)

| group | prefix | median n/d | **ν* [95 % CI]** | interior | old Part B argmin (pred / meas) | Part C power law at median n/d |
|---|---|---|---|---|---|---|
| gate+up | 16 | 0.26 | **2.37** [2.26, 2.49] | yes | 2.0 / 2.0 | 2.16 |
| gate+up | 32 | 0.53 | **1.56** [1.48, 1.63] | yes | 1.5 / 1.5 | 1.41 |
| gate+up | 64 | 1.06 | **1.01** [0.96, 1.06] | yes | 1.0 / 1.0 | 0.93 |
| gate+up | 128 | 2.12 | **0.63** [0.60, 0.66] | yes | 0.5 / 0.5 | 0.57 |
| gate+up | 256 | 4.23 | **0.37** [0.35, 0.39] | yes | 0.3 / 0.3 | 0.34 |
| down | 16 | 0.38 | **6.44** [5.64, 7.39] | yes | 5.0 / 5.0 *(edge)* | 5.48 |
| down | 32 | 0.77 | **4.12** [3.68, 4.56] | yes | 5.0 / 5.0 *(edge)* | 3.08 |
| down | 64 | 1.54 | **2.65** [2.40, 2.92] | yes | 3.0 / 3.0 | 1.89 |
| down | 128 | 3.08 | **1.64** [1.48, 1.80] | yes | 1.5 / 1.5 | 1.12 |
| down | 256 | 6.15 | **1.00** [0.91, 1.11] | yes | 1.0 / 1.0 | 0.66 |

- **B16 for down was a real edge.** The optimum is 6.4, above the old grid maximum, and its CI excludes 5.0.
- **B32 for down was not.** The optimum is 4.1, and the 5.0 argmin was simply the nearest grid point (the grid
  jumps 3 → 5).
- **Everywhere else** the refined ν* sits between the old argmin and its neighbour.
- **Part C power law, gate+up.** It predicts within 10 % at every budget.
- **Part C power law, down at n < d (B16, B32).** It is about 15–25 % low.
- **Part C power law, down at n > d.** It is about 1.4–1.5× low: 1.89 / 1.12 / 0.66 against 2.65 / 1.64 / 1.00.
  Part C's fit used only 3 experts per layer and random sequence subsets, so the down exponent from Part C should
  not be used to set ν at large budgets. The slope matches; the intercept does not.
- **George–Saab** is far off for down at every budget (see `figures/nu_star_vs_ratio.png`).

## Q2: measured confirmation (B2 experts, 18-point merged grid)

| group | prefix | cells | median Spearman | argmin agree ±1 step | median argmin pred / meas |
|---|---|---|---|---|---|
| down | 16 | 96 | 0.998 | 96.9 % | 10 / 10 |
| down | 32 | 96 | 0.996 | 97.9 % | 5 / 5 |
| down | 64 | 96 | 0.994 | 96.9 % | 3 / 3 |
| down | 128 | 96 | 0.994 | 96.9 % | 2 / 2 |
| down | 256 | 96 | 0.990 | 97.9 % | 1 / 1 |
| **down, all** | | **480** | **0.996** | **97.3 %** | 3 / 3 → **PASS** |
| gate+up (not gated) | 16, 32 | 192 | 0.998 | 100.0 % | 2 / 2 |

- **Matching levels.** The measured B2 means lie on the predicted curves all the way out to ν = 50 (open circles
  in `figures/mean_curves.png`).
- **The 50 edge.** At B16, 3 of 96 down cells (3.1 %) still have their measured argmin at 50. The predicted
  argmin sits at the same edge for the same share (1.0 % over all prefixes for both).
- **Clamping does not grow with ν; it shrinks.** For down, the mean clamp fraction on the original weights falls
  monotonically from 1.4e-3 at ν = 0.01, to 2.0e-4 at ν = 5, to 5.5e-5 at ν = 50. The maximum per cell is 8.1e-5 at
  ν = 50. On the working weights it falls from 5.3e-4 to 1.8e-5. Stronger damping means smaller error-feedback
  updates, so fewer weights are pushed off the fitted grid.

## Q3: practical size. Was the old value good enough?

Excess error of a fixed ν over the optimum. "Pred" values come from the E1 expert-mean curve (60 experts × 8).
"Meas" values come from the B2 mean measured curve (12 experts × 8; on the 18-point grid, or the 12-point grid
for gate+up at prefixes ≥ 64).

| group | prefix | ν = 5 → ν*: pred / meas | ν = 1 → ν*: pred / meas |
|---|---|---|---|
| down | 16 | **+0.08 % / +0.01 %** | +6.75 % / +6.28 % |
| down | 32 | **+0.06 % / +0.00 %** | +4.50 % / +4.26 % |
| down | 64 | +0.59 % / +0.44 % | +2.12 % / +2.10 % |
| down | 128 | +1.79 % / +1.42 % | +0.49 % / +0.55 % |
| down | 256 | +3.42 % / +2.74 % | +0.00 % / +0.00 % |
| gate+up | 16 | +1.13 % / +1.02 % | +1.84 % / +1.76 % |
| gate+up | 32 | +3.19 % / +3.02 % | +0.60 % / +0.61 % |
| gate+up | 64 | +6.27 % / +6.03 % | +0.00 % / +0.00 % |
| gate+up | 128 | +10.5 % / +9.9 % | +0.63 % / +0.40 % |
| gate+up | 256 | +15.0 % / +14.4 % | +2.35 % / +2.07 % |

The per-cell medians agree (see `runs_grid_ext/report.md`).

**Was the old Part B choice good enough?** Yes:
- **Down at B16 and B32.** The 5.0 pick is within 0.1 % of the true optimum, predicted and measured. The curve is
  flat near its minimum (the expert-mean R varies by 0.84 % over the whole range ν = 3–15 at B16). The grid edge cost nothing in
  practice.
- **Every old Part B argmin** (2 / 1.5 / 1 / 0.5 / 0.3 and 5 / 5 / 3 / 1.5 / 1) is within 0.15 % of the error at
  its refined ν* (predicted, expert-mean curve).

**What is not good enough** is a single ν shared by both groups:
- **ν = 1 for down at low budget** costs 6.8 % (B16), 4.5 % (B32) and 2.1 % (B64).
- **ν = 5 for gate+up** costs 3–15 % at B32 and above.

## Q4: the ratio ν*_down / ν*_gu

| prefix | budget level | per-expert median [IQR], matched (seed, layer, prefix, expert) |
|---|---|---|
| 16 | 2.72 | 3.51 [2.43, 4.96] (n = 478) |
| 32 | 2.64 | 3.09 [2.24, 4.53] (n = 478) |
| 64 | 2.62 | 3.00 [2.13, 4.36] (n = 478) |
| 128 | 2.62 | 2.86 [2.04, 4.07] (n = 478) |
| 256 | 2.71 | 2.87 [2.07, 4.09] (n = 478) |
| all | 2.62–2.72 | **3.05** [2.16, 4.44] (n = 2,390; 10 excluded for an edge ν*) |

- **Budget level.** The ratio is **constant at ≈ 2.7×**. The slope of log ratio against log tokens over the five
  prefixes is −0.002.
- **Per expert.** The median is about 3×, and it trends down slightly with data: the slope against log tokens is
  −0.076 [−0.095, −0.056], from 3.5× at B16 to 2.9× at B256.
- **Why the two differ.** Down's per-expert ν* is much more dispersed (Q5), and the argmin of the mean curve is
  pulled toward the experts whose curves are steepest.
- **Replacement figure.** "About 2.5×" (T1, read off the coarse grid as 2.5–3.3×) becomes **2.7× for a per-group
  budget rule** (e.g. ν_down = 2.7·ν_gu) and **3.0× per expert (median)**.

## Q5: per-expert spread

| group | prefix | median ν* | IQR | p5 / p95 | at 1e2 edge | at 1e-2 edge |
|---|---|---|---|---|---|---|
| gate+up | 16 | 2.46 | 1.99–3.10 | 0.93 / 4.07 | 0 % | 0 % |
| gate+up | 32 | 1.66 | 1.29–2.00 | 0.60 / 2.57 | 0 % | 0 % |
| gate+up | 64 | 1.09 | 0.85–1.29 | 0.43 / 1.63 | 0 % | 0 % |
| gate+up | 128 | 0.68 | 0.51–0.81 | 0.28 / 1.03 | 0 % | 0 % |
| gate+up | 256 | 0.41 | 0.30–0.49 | 0.16 / 0.64 | 0 % | 0 % |
| down | 16 | 8.29 | 4.85–14.1 | 2.29 / 34.6 | 0.4 % | 0 % |
| down | 32 | 4.92 | 2.87–8.19 | 1.45 / 17.8 | 0 % | 0.4 % |
| down | 64 | 3.02 | 1.83–5.10 | 0.83 / 10.6 | 0 % | 0.4 % |
| down | 128 | 1.78 | 1.11–2.90 | 0.51 / 6.04 | 0 % | 0.4 % |
| down | 256 | 1.08 | 0.64–1.73 | 0.30 / 3.62 | 0 % | 0.4 % |

**Spread.** Down's per-expert ν* spans about 15× between p5 and p95, against about 4× for gate+up. The
per-expert median sits above the budget-level ν* (e.g. 8.3 against 6.4 at B16) because the mean curve is
dominated by low-ν experts with steep curves.

**Edge cells.** Only two cells are at the 1e2 edge: L1 e6 and L8 e51, down, both seed 1 at B16. Neither is a T1
outlier.

**The T1 outlier experts sit at the opposite end.** They want much *less* damping than their layer, not more.

| expert (down) | ν* at B16 → B256 (seeds 0 / 1) | layer median B16 → B256 |
|---|---|---|
| L1 e42 | 0.013 → **0.01 (low edge from B32 on, both seeds)** | 5.7 → 0.62 |
| L1 e0 | 0.039 → 0.013 | 5.7 → 0.62 |
| L23 e57 | 0.21–0.24 → 0.08–0.10 | 7.1 → 1.09 |
| L23 e25 | 2.3–2.5 → 0.14 | 7.1 → 1.09 |
| L23 e26 | 1.8–2.1 → 0.32–0.38 | 7.1 → 1.09 |
| L1 e33 | 3.3–3.9 → 0.34–0.39 | 5.7 → 0.62 |

All six are below their layer median at every prefix; L1 e42, L1 e0 and L23 e57 are 1–2 orders of magnitude below.
These are the experts where GPTQ's rounding variance differs from RTN's (the R3 term in T2). Their Ĥ is dominated by
a few directions in which almost no damping is wanted, and any global damping rule over-damps them. They are
a small minority (about 0.4 % of down cells at the low edge) and do not move the budget-level ν*.

## Figures (`runs_grid_ext/figures/`)

- `mean_curves.png`: expert-mean R_ideal vs ν per prefix, one panel per group. Measured B2 means are overlaid as
  open circles, ν* as triangles, and the old grid maximum ν = 5 is marked.
- `nu_star_vs_ratio.png`: budget-level ν* (with 95 % CI) against median n/d for both groups, with the Part C
  power laws and George–Saab.
- `nu_star_hist.png`: histograms of per-cell ν* by prefix, one panel per group.

## Deviations and changes

- **Deviations from the frozen protocol:** none.
- **R1 (refactor sanctioned by §0, numerics unchanged).** In `diagnostics/theorem1_check.py`:
  - the per-layer setup moved into `layer_context()`;
  - the nested-prefix Ĥ accumulation moved into the generator `prefix_hessians()`;
  - `sweep_curve()` gained optional `nus` / `curve` / `gs` arguments, whose defaults reproduce Part C.

  The tests still pass (40 / 40), and both invariants reproduce Part B bitwise.
- **Descriptive additions:**
  - the E1 invariant against Part B's stored R_ideal (in the protocol, beyond the task text);
  - Q5 also reports the 1e-2 edge;
  - Q2 also reports Spearman against e_meas/e_rtn (median 0.998 for down).
- **Untouched:** `runs_theorem1_check/`, `configs/` and every other existing output (no file newer than this
  protocol), and the test and confirmation splits.

## New files

- `diagnostics/damping_grid_extension.py` (`protocol | run`)
- `diagnostics/report_damping_grid_extension.py`
- `update_damping_grid_extension.md`
- `runs_grid_ext/`:
  - `protocol.json` (frozen) and `deviations.json`
  - `raw_e1.jsonl`, `raw_e2.jsonl`, `events.jsonl`
  - `report.json` and `report.md`
  - `figures/mean_curves.png`, `figures/nu_star_vs_ratio.png`, `figures/nu_star_hist.png`
  - `run_all.sh` and `logs/run.log`
- Modified: `diagnostics/theorem1_check.py` (R1, untracked file from the previous task)
