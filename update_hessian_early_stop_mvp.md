# Per-expert Hessian early stopping: feasibility MVP

**Verdict (rules frozen in `runs_hessian_early_stop_mvp/protocol.json` before any measurement): `EARLY-STOP-SIGNAL-FAIL`.**
The systems gate fails as well.

**Answer:** per-expert Hessian early stopping is **not** a worthwhile optimization for this pipeline. There are two
reasons, and either one alone is enough:

1. **No systems headroom.**
   - Hessian accumulation is 8.0 s of a 208.5 s replay + solve pipeline (3.9 %).
   - The GPTQ solve takes 184 s and dominates.
   - Stopping half of the experts at B = 64 saves 8.4 % of replay time and about 2 % end to end.
2. **Nothing to detect.**
   - Stopping at B = 64 makes **239 of 240** experts worse by more than 1 %. The median is +4.5 %; at B = 32 it is
     +10 %.
   - The penalty is almost uniform across experts and exposure deciles.
   - So there is no subset of "already converged" experts for a calibration statistic to find. Every rule's false-safe
     rate is about 100 %.

**Recommendation C:** stop systems development and finalize CS498.

## Setup

- **Data:** Qwen1.5-MoE-A2.7B; FR s0 (C4 Random, the frozen 128 × 512 windows in the manifest order, CANONICAL-B1 cache).
- **Scope:** routed experts of layers 1/8/16/23, 240 experts.
- **Quantization:** GPTQ 4-bit, g128, percdamp 1.0, act-order, solver unchanged.
- **Evaluation:** held-out E3 reconstruction on rows 2128–2255.
- **Checkpoints:** true chronological prefixes S16 ⊂ S32 ⊂ S64 ⊂ S128.
  - Production's final Profiler uses contiguous blocks in `chosen` order: block 0 = S32 and blocks 0–1 = S64. These are
    not the hashed statistical blocks.
  - S16 is a mid-block CPU snapshot.
- **Validation (`hessian_checkpoint_validation.json`):**
  - H128 = H64 + H64:128 to within 8e-8 relative.
  - Routed-token counts equal the routing profile at every prefix.
  - B = 128 reproduces the historical FR s0 weights and per-expert reconstruction **bitwise**.

## Experiment 1: measured systems upper bound (`timing_breakdown.json`, `fixed_mask_benchmark.json`)

The measurement uses the production Profiler path on 4 layers × 128 sequences: one warm-up, then 3 interleaved repeats
with CUDA synchronization. Layer outputs are bitwise identical in every condition on all 4 layers. The masks are fixed
by expert ID: stop50 is even IDs, stop75 is e % 4 ≠ 3, and masked experts skip conversion + GEMM + update after
sequence 64.

| Metric (4 layers) | Full | Stop 50 % | Stop 75 % |
|---|---|---|---|
| expert IDs stopped / accumulation rows actually skipped | 0 / 0 % | 50 % / **25.3 %** | 75 % / **37.9 %** |
| model forward (forward-only condition) | 9.06 s | same | same |
| activation-hook overhead (hooks, no accumulation) | 7.22 s | same | same |
| Hessian accumulation (full − no-acc) | 8.04 s | | |
| ↳ CUDA-event GPU time: fp32 conversion / GEMM / update | 1.65 / 5.33 / 5.40 s | | |
| total Hessian replay, median [range] | 24.32 [23.99–24.37] s | 22.28 [22.12–22.33] s | 21.24 [21.05–21.24] s |
| Hessian replay reduction (Speedup_H) | — | **8.4 % (1.09×)** | 12.7 % (1.15×) |
| GPTQ solve (720 projections) | 184.4 s | 181.9 s | 183.4 s |
| replay + solve (measured once) | 208.5 s | 204.0 s | 205.0 s |
| end-to-end reduction | — | 2.2 % | 1.7 % |
| peak GPU | 8.72 GB | 8.72 GB | 8.72 GB |
| peak CPU RSS (process high-water mark, partial 4-layer model) | 10.45 GB | 10.45 GB | 10.45 GB |

The RSS figure is a monotone process maximum. It cannot resolve differences between conditions; it shows no increase.

How to read these numbers:
- **Rows skipped vs IDs stopped.** Stopping 50 % of IDs after half the data skips only 25 % of accumulation rows. The
  absolute replay saving (2.0 s) matches 25 % of 8.0 s.
- **End-to-end differences are noise.** They are dominated by run-to-run spread in the solve (±1–2 s). Stop75 saved
  *less* end to end than stop50.
- **Upper bound.** Skipping *all* accumulation would save at most 33 % of replay and **3.9 %** of replay + solve.
  Against the historical `quantize_total` (347 s, which adds E3 scoring and cache I/O) it would be about 2.3 %.
  Upstream CANONICAL-B1 activation capture (about 60 s for 128 rows) is unaffected.
- **Memory.** Peak memory is unchanged, because the Hessian buffers are preallocated.
- **Where the overhead is.** Python hook dispatch (7.2 s) costs about as much as all of the accumulation.
- **CUDA events.** The event times add up to more than the 8 s wall difference because GPU work overlaps the CPU-side
  hook dispatch.

**Systems gate: fails.** It needed ≥ 15 % replay reduction and ≥ 3 % end-to-end reduction on stop50; measured were
8.4 % and 2.2 %.

## Experiment 2: offline stopping simulation

### Fixed budgets (`reconstruction_by_budget.json`)

| | mean error | vs B128 | p90 | experts ≤ 1 % worse than B128 |
|---|---|---|---|---|
| B32 | 0.0050608 | +9.79 % | 0.006541 | 0 / 240 (minimum +2.1 %, median +10.0 %) |
| B64 | 0.0048138 | +4.43 % | 0.006239 | 1 / 240 (minimum +0.69 %, median +4.5 %, maximum +5.9 %) |
| B128 | 0.0046095 | — | 0.005992 | — |

- **By layer:** the B64 penalty is +4.5 % (L1), +4.7 % (L8), +4.6 % (L16) and +3.9 % (L23).
- **By natural-exposure decile:** it ranges from +4.2 % to +4.7 %, rarest first.
- **Interpretation:** at percdamp 1.0, every expert keeps benefitting from the second 64 sequences by almost the same
  amount.

### Calibration-only statistics (`expert_statistics.json`, `false_safe_stop_analysis.json`)

- **Statistics computed:** trace drift and Frobenius drift of token-normalized prefix Hessians (B vs B/2), separately
  for gu and down, plus the routed-token count n_B. relSE was not used; it would need more than the 1–4 contiguous
  blocks available.
- **Prediction of the B → 128 degradation (Spearman, 240 experts, not independent):**
  - Frobenius drift: B32 gu 0.07, down 0.17; B64 gu 0.33, down 0.27.
  - Trace drift: |ρ| ≤ 0.10.
  - Exposure: −0.30 to −0.32. Experts with more tokens degrade slightly less.
- **Separation:** none. Among experts flagged "stable" by the median-threshold rules, the truly-safe rate is 0–5 %,
  against a base rate of 0–0.4 %.

### Stopping policies (thresholds = calibration-only quantiles, fixed a priori; primary = median; no held-out data used)

| Strategy | mean error | Δ vs B128 | p90 error | false-safe > 1 % (> 2 %) | Hessian work skipped | stops at 32 / 64 / 128 | time |
|---|---|---|---|---|---|---|---|
| Fixed B32 | 0.0050608 | +9.79 % | 0.006541 | 240/240 (100 %) | 75.0 % | 240 / – / – | projected 202.7 s |
| Fixed B64 | 0.0048138 | +4.43 % | 0.006239 | 239/240 (99.6 %) | 50.0 % | – / 240 / – | projected 204.7 s |
| Fixed B128 | 0.0046095 | 0 | 0.005992 | — | 0 | – / – / 240 | **measured** 208.5 s |
| Trace early stop (primary) | 0.0047182 | +2.36 % | 0.006036 | 74/74 (100 %) | 20.3 % | 43 / 31 / 166 | projected 207.1 s |
| Frobenius early stop (primary) | 0.0046312 | +0.47 % | 0.005992 | 20/20 (100 %) | 5.7 % | 15 / 5 / 220 | projected 208.3 s |
| Drift + exposure (primary) | 0.0046197 | +0.22 % | 0.005992 | 9/9 (100 %) | 2.8 % | 7 / 2 / 231 | projected 208.5 s |
| Exposure only (baseline) | 0.0048341 | +4.87 % | 0.006116 | 139/139 (100 %) | 43.5 % | 120 / 19 / 101 | projected 205.2 s |

- **Measured vs projected.** "Measured" times are real runs. "Projected" times use the measured cost model:
  forward + hook overhead + work fraction × 8.04 s accumulation + 184.4 s solve. They are not measured runs.
- **Grid results.** The exploratory 25 % / 75 % quantile grid behaves the same way (`stopping_policy_results.json`).
  Every rule has a false-safe rate of 99–100 %, and none reaches the scientific gate:
  - ≥ 25 % work skipped,
  - ≤ 0.5 % mean degradation,
  - ≤ 1 % p90 degradation,
  - ≤ 10 % false-safe.
- **Frontier.** The Frobenius rules sit marginally below the fixed-budget frontier, but only where they skip 3–6 % of
  the work, worth about 0.3 s. Every rule that skips a meaningful amount lies on or above the frontier.

## Answers

1. **Actual timing:** accumulation is 8.0 s of 208.5 s (3.9 %). The GPTQ solve is 184 s, forward 9 s, hook overhead
   7 s.
2. **Maximum realistic speedup:** about 1.09–1.15× on Hessian replay, and ≤ 3.9 % end to end even with zero
   accumulation. The measured end-to-end changes (−2.2 % / −1.7 %) are inside solve-time noise.
3. **Calibration-only statistics:** drift declines with budget, but it barely ranks which experts lose quality
   (|ρ| ≤ 0.33).
4. **Held-out degradation:** +4.4 % (B64) and +9.8 % (B32) on average, and almost every expert is affected.
5. **False-safe rate:** about 100 % for every rule and threshold.
6. **Against fixed budgets:** no meaningful advantage. Adaptive rules either stop almost nothing or track the fixed
   B64 / B32 frontier.
7. **Engineering and hardware:**
   - A real controller would need per-expert active masks inside the Profiler hooks plus an asynchronous schedule. It
     would save at most about 8 s per 4-layer quantization on this 20 GB MIG slice.
   - The memory peak would not change unless buffers were released.
   - The larger, cheaper systems lever is the 184 s GPTQ solve, or the 7 s of Python hook dispatch. Neither is in
     this project's scope.

## Recommendation

**C: stop systems development and finalize CS498.**
- The potential saving is below 4 % of quantization time.
- The calibration-sufficiency statistics cannot identify experts that are safe to freeze, because at this damping no
  expert is saturated by B = 64.
- This is consistent with the project's earlier findings: the sufficiency signals characterize Hessian drift but do not
  translate into useful allocation or stopping decisions.

Recommendation A (implementing a real adaptive controller) was not executed.

## Artifacts

- **Run outputs:** `runs_hessian_early_stop_mvp/` (protocol, manifests, validation, timing, benchmark, statistics,
  reconstruction, policies, false-safe analysis, quality–cost, decision, summary, figures 1–5, logs).
- **Code:** `moe_calib/early_stop.py`, `diagnostics/hessian_early_stop_mvp.py`,
  `diagnostics/report_hessian_early_stop_mvp.py`, `tests/test_early_stop.py`.
- **Production files:** none modified.
