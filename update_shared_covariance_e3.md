# Shared covariance-aware calibration (SC): full E3 validation

**Answer: SUPPORTED, with limits.** The MVP found that choosing one shared 128-sequence calibration set to match per-expert activation covariances (SC) beats random selection. That result carries over to the full E3 setting: all 240 routed experts in layers 1/8/16/23, a matched 32-sequence warm start, and no token matching.
- **SC vs Full Random (FR): ROBUST-SUPPORTED under the frozen rule.**
  - Test reconstruction: **+0.68 %** (+0.67 / +0.89 / +0.49 % per seed).
  - Confirmation split: **+0.73 %**.
  - 227–234 of 240 experts and 4/4 layers improve per seed.
  - Calibration grouping: 15/15 configurations favor SC; hierarchical 95 % CI **[+0.50 %, +0.89 %]**.
- **SC vs Typical Coverage-LP (TC): ROBUST-SUPPORTED.** +0.64 % test; every seed is ≥ 0.53 %.
- **SC vs Typical Random (TR): NO-GAIN under the frozen rule.** SC is better on every seed, every layer and 15/15 groupings, but the mean is **+0.47 %**, below the 0.5 % materiality threshold.
- **Perplexity: no gain established.**
  - Stage A ΔPPL vs FR is mixed in sign (+0.008 / −0.010 / −0.003).
  - Stage B's mean is −0.0041 (CI [−0.0072, −0.0011]), but the grouping SD of 0.0049 is larger than |mean|.
- **Compute: not compute-efficient.** SC costs about 632 s of extra profiling, roughly 3,500 random-sequence forwards. Its gain equals about 15 extra random sequences, an extrapolated estimate (see Results).

Isolated-expert reconstruction is the primary metric. 4-layer partial-model WikiText-2 PPL is secondary.

Pretrained weights, routing, the GPTQ implementation and the frozen evaluation configuration are unchanged. No earlier result was modified: the stored FR/TR/TC results were reused only after reruns reproduced them bitwise.

## Setup

- **Model and quantization:**
  - Qwen1.5-MoE-A2.7B, non-sequential partial-model expert GPTQ on layers 1/8/16/23 (240 routed experts).
  - 4-bit, g128, act-order, percdamp 1.0 (`configs/frozen_b4.json`). Attention and shared experts stay BF16.
- **Budget and baselines:**
  - B = 128 shared sequences from the 2,000-sequence C4 pool (rows 0–1999), with allocation seeds 0/1/2.
  - All four strategies share the same 32-sequence warm start per seed.
  - FR/TR/TC come from `runs_partial_ppl/selected_set_manifest.json`.
- **SC objective** (`moe_calib/shared_covariance_selection.py`):
  - Per (layer, expert, Hessian group ∈ {gate+up, down}), a 64-d random-projection sketch of the per-sequence second moment X^T X.
  - J = mean over the 480 terms of the relative Frobenius mismatch between the selected-set covariance and the pool covariance (token-weighted).
  - Optimization: greedy from the warm start, then swap refinement that never removes warm slots. Closed-form add/swap scores in fp64 over fp32 storage.
  - Same objective as the MVP's SC condition, stacked over all experts.
- **Splits:**
  - Selection sees pool rows only.
  - Primary metric: E3 relative reconstruction error on the test split.
  - Secondary metrics: the confirmation split (`~/moe_cache_confirmation` rows 0–255) and partial-model PPL.
- **Freezes:**
  - Protocol sha `8ccc0ce6eaf496cb`, frozen before any sketch.
  - Selections and code hashes in `freeze.json`, written before any SC reconstruction.
  - Decision rule `decide()`, threshold 0.5 %.
- **Stage B:**
  - Per seed, the union of the four strategies' sequence ids (374 / 373 / 390), with 5 calibration groupings (20261111–15, batches of 8).
  - All four strategies are quantized from the same grouping cache: 60 configurations.
  - Analysis is nested (seed > grouping) with a hierarchical bootstrap.

## Correctness checks (all passed)

- 18 unit tests, including exact parity with the MVP optimizer when the warm start is empty. All 17 MVP tests still pass.
- **Sketch pass** (4 layers × 2,000 sequences):
  - The identity check against the projected full Hessian is ≤ 3.3e-6.
  - Per-expert token counts match the routing cache.
  - Sketches are bitwise-equal to the MVP's on its L16 experts, and the MVP scores are reproduced to 1.5e-9.
- **Selections:**
  - Deterministic and monotone in J.
  - Valid: warm start preserved, 128 unique ids, all within the pool.
- **Baselines:** FR s0, TR s1 and TC s2 were rerun through this pipeline. Test/confirmation errors, installed-weight sha256 and PPL all reproduced bitwise, so the stored baselines were reused.
- **Stage B:**
  - All 15 grouping caches passed the reversed-order bitwise and token checks.
  - All 60 configurations have `install_ok`.
  - The fresh-process control (SC s0 G20261111) reproduced weights, reconstruction, PPL and per-window NLL bitwise.
- **Thread count:** `OMP_NUM_THREADS=8` was set during the run after an oversubscription slowdown. It changes runtime only; the bitwise controls confirm the numerics are unaffected.

## Results

Test reconstruction error (3-seed mean, ORIGINAL extraction):

| strategy | test error | confirmation error | PPL | test improvement vs FR |
|---|---|---|---|---|
| FR | 0.004615 | 0.004580 | 9.2694 | — |
| TR | 0.004606 | 0.004559 | 9.2683 | +0.21 % |
| TC | 0.004613 | 0.004562 | 9.2725 | +0.04 % |
| SC | 0.004584 | 0.004546 | 9.2678 | +0.68 % |

BF16 PPL is 9.2371.

Paired comparisons (positive = SC better):

| comparison | test per seed | confirmation mean | Stage B advantage (CI) | Stage B ΔPPL (CI), \|M\|/W | label |
|---|---|---|---|---|---|
| SC vs FR | +0.67 / +0.89 / +0.49 % | +0.73 % | +0.686 % [+0.50, +0.89] | −0.0041 [−0.0072, −0.0011], 0.85 | ROBUST-SUPPORTED |
| SC vs TR | +0.72 / +0.33 / +0.38 % | +0.28 % | +0.475 % [+0.32, +0.72] | −0.0001 [−0.0073, +0.0043], 0.05 | NO-GAIN |
| SC vs TC | +0.81 / +0.53 / +0.58 % | +0.34 % | +0.639 % [+0.53, +0.81] | −0.0020 [−0.0100, +0.0053], 0.31 | ROBUST-SUPPORTED |

**Mechanism (descriptive):**
- SC lowers the held-out full-dimensional covariance mismatch from 0.443 (FR) to 0.364, for 98–100 % of experts.
- Across the 9 within-seed deltas vs FR, test error tracks this mismatch: Spearman +0.82 (down projection +0.90). It does not track token count (+0.07), Gini (+0.20) or routing atypicality (−0.53).
- Token exposure is essentially the same for all strategies (median ≈ 4,320 tokens/expert), so the gain is not a token-quantity effect.

**Compute:**
- SC's extra cost is the sketch pass (617 s), selection (~15 s per seed) and 8 GB of sketches. That is ≈ 3,539 sequence-forward equivalents at 0.178 s/sequence.
- For comparison, `runs_low_budget` shows a 4.2 % FR improvement from B = 64 to B = 128.
- Log-linear extrapolation of those two points puts SC's error at an FR budget of about B ≈ 143. This is an estimate, not a measurement.

Full tables, per-layer and quantile breakdowns, the Stage B nested statistics, figures and the 7-question interpretation are in `runs_shared_covariance_e3/summary.md`.

## Interpretation

Covariance-aware shared selection gives a small (~0.5–0.9 %), very consistent reduction in held-out expert reconstruction error at full E3 scale.
- It is consistent across seeds, layers, nearly all experts and calibration groupings.
- It is tied to how well the selected set reproduces each expert's activation covariance, not to token counts.

This contrasts with the routing-exposure allocators in `update_low_budget_sweep.md`, where Coverage-LP never beat its matched Random baseline.

The effect is too small to show up reliably in 4-layer PPL. Most of its value over TR could also be had by sampling from the typical half.

## Limitations

- One model, 4 of 24 layers, a single budget (B = 128), 4-bit g128 GPTQ.
- PPL is partial-model (4 quantized layers). Full-depth behavior was not tested.
- The SC vs TR margin is below the materiality threshold, and the confirmation-split margin over TR/TC is small (+0.28 / +0.34 %).
- Diagnostic associations are across 12 sets that differ in more than one respect; they are not causal.
- The compute-equivalence figure (B ≈ 143) extrapolates two FR points from a separate run.

## Reproduction

```bash
source ~/venvs/moe/bin/activate
scripts/run_shared_covariance_e3.sh all     # tests preflight protocol sketch mvpcheck select freeze stageA diag stageB report
python diagnostics/report_shared_covariance_e3.py
pytest -q tests/test_shared_covariance_e3.py
```

Run only one full-model process at a time, because of the 72 GB host RAM cap. Stage A takes about 11 min per config; Stage B about 5.5 min per config (~6 h total).

## Artifacts

- Code:
  - `moe_calib/shared_covariance_selection.py`
  - `diagnostics/shared_covariance_e3.py`
  - `diagnostics/report_shared_covariance_e3.py`
  - `scripts/run_shared_covariance_e3.sh`
  - `tests/test_shared_covariance_e3.py`
- Outputs in `runs_shared_covariance_e3/`:
  - `protocol.json`, `preflight.json`, `freeze.json`
  - `selections/`, `results/`, `diagnostics/`, `grouping_robustness/`, `profiling/`
  - `summary.md` / `summary.json`, `interpretation.md`
  - `figures/` (fig1–fig4)
  - `logs/`
- Sketches and grouping caches: `~/moe_cache_shared_covariance_e3/` (not in the repo).
