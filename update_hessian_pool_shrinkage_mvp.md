# Layer-pooled Hessian shrinkage for low-budget MoE GPTQ: feasibility MVP

**Frozen verdict: `POOL-SHRINKAGE-MISMATCH-ONLY` → recommendation C.** Report it as a covariance-estimation finding and
do not continue algorithm development.

**Answer to the main question:** no. Shrinking expert gate/up Hessians toward a trace-matched layer-pooled second moment
does reduce Frobenius mismatch against an independent reference. It does not reduce the calibration data GPTQ needs
beyond what damping already achieves:
- **Small gain.** The reconstruction gain is +0.63 % at B16 and +0.50 % at B32, below the frozen 1 % bar.
- **Loses to identity.** It is smaller than an equally strong, trace-matched identity shrinkage (+1.26 % / +0.57 %).
- **Far behind tuned damping.** Historical budget-specific damping gives +4.81 % / +2.73 %.

**Why the label is MISMATCH-ONLY.** Under the frozen precedence, a budget is classified REGULARIZATION-ONLY only if
criterion C1 (≥ 1 % gain) holds there and C2 or C3 fails. Here C1 fails at both budgets, so the frozen label is
MISMATCH-ONLY → C. C2 and C3 fail too, so the practical conclusion is also B: keep standard GPTQ with tuned damping.

Rules, alpha grid, expert selection and thresholds were frozen in `runs_hessian_pool_shrinkage_mvp/protocol.json` before
any covariance or quality measurement. Nothing was changed afterwards. Recommendation A was not executed.

## Setup

- **Model and calibration:**
  - Qwen1.5-MoE-A2.7B.
  - FR s0: C4 Random, 128 × 512 windows in the frozen manifest order, CANONICAL-B1 cache.
  - No new data.
- **Chronological prefixes:** S16 ⊂ S32 ⊂ S64 ⊂ S128. They are accumulated exactly as in `runs_hessian_early_stop_mvp`:
  contiguous K = 4 blocks, with S16 as a mid-block snapshot.
- **Layers 8 and 16:** exploratory choices informed by earlier AGQ outcomes, not sampled.
- **Experts:** 12, selected by **B32 routed-token exposure only**. Within each layer, sorted ascending (ties to the
  lower ID), the strata are ranks {0,1} low, {29,30} median and {58,59} high.
  - **L8:** {3, 42 | 37, 47 | 58, 55}.
  - **L16:** {1, 32 | 15, 57 | 39, 9}.
  - **Exposure range:** routing is nearly balanced. At B32 the selected experts have 798–1,613 tokens, and every
    expert's gate/up Hessian is rank-deficient (n_e < d = 2048) at B16 and B32.
- **Pooled Hessian:**
  - H_L = Σ x xᵀ over every MoE-input token, each counted once.
  - It is captured by an opt-in `forward_pre_hook` on the router gate (`moe_calib/hessian_shrinkage.PooledHessian`).
    The hook input is verified bitwise equal to the router input, which the Profiler in turn verifies is equal to every
    routed expert's gate_proj rows.
- **Estimators** (mixed in fp64, cast to fp32, then passed to the unchanged `gptq_quantize`):

  | Estimator | Formula |
  |---|---|
  | Pooled | n_e[(1−α)H̄_e + αP_e], with P_e = H̄_L tr(H̄_e)/tr(H̄_L) |
  | Identity | n_e[(1−α)H̄_e + α tr(H̄_e)/d · I] |
  | α = 0 | the raw H_e object itself |

  The down projection always uses its raw Hessian.
- **GPTQ:** 4-bit, g128, act-order, percdamp 1.0, solver unchanged.
- **Tuned damping control:** percdamp 3.0 at B16 and 2.0 at B32 on all projections, from
  `runs_diag/damping_threshold_report.md`.

## Stage 1A: pooled Hessian validation (`pooled_hessian_validation.json`)

Every check passed on both layers:

| Check | Result |
|---|---|
| Pooled-hook input vs router input | equal, 128/128 sequences |
| Profiler gate_proj-row alignment checks | 7,676 (L8) / 7,679 (L16) |
| Pooled token counts | 512·B exactly |
| Σ_e n_e | = 4·N |
| Expert counts vs routing profile | equal |
| H_L¹²⁸ − (H_L⁶⁴ + H_L^{64:128}) | 3.7e-8 relative |
| Pooled matrices | finite, exactly symmetric, PSD |
| Layer outputs with / without hooks | bitwise identical |

Regression against historical artifacts, for the 12 experts:
- **Drift statistics:** the trace and Frobenius drift statistics equal the early-stop study's bitwise.
- **E3 on old-test:** standard B32/B64/B128 E3 equals the early-stop logs bitwise.
- **Weights:** standard B128 weights equal the historical FR s0 weights bitwise.

## Stage 1C: covariance mismatch (`covariance_mismatch.json`)

The reference is each expert's routed gate/up second moment on the tune split (`~/moe_cache` rows 2000–2127; 3.2k–5.4k
tokens per expert). It is disjoint from the calibration rows and never used to build the prior. It is a finite-sample
proxy, not the population covariance.

| Mean D_F (12 experts) | raw | pooled α=.1/.3/.5 | identity α=.1/.3/.5 | untraced α=.3 | pure prior α=1 |
|---|---|---|---|---|---|
| B16 | 0.580 | 0.547 / 0.518 / 0.537 | 0.531 / **0.504** / 0.575 | 0.517 | 0.748 |
| B32 | 0.448 | 0.424 / **0.418** / 0.464 | 0.414 / 0.431 / 0.542 | 0.418 | 0.711 |
| B64 (ref.) | 0.368 | 0.353 / 0.368 / 0.432 | 0.350 / 0.403 / 0.536 | 0.368 | 0.698 |
| B128 (ref.) | 0.302 | 0.294 / 0.328 / 0.408 | 0.294 / 0.375 / 0.526 | 0.328 | 0.689 |

How to read the table:
- **Frobenius gain shrinks with budget.** Pooling reduces Frobenius mismatch by up to 10.7 % at B16 (α = 0.3, 12/12
  experts) and 6.6 % at B32. The best α falls as the budget grows, consistent with a variance–bias trade-off.
- **The pooled prior is biased.** On its own (α = 1) it is much further from the expert reference (0.69–0.75) than even
  the raw B16 estimate. Pooling also roughly doubles the relative diagonal mismatch: 0.088 → 0.179 at B16, α = 0.3,
  against 0.267 for identity. Top-16 eigenspace alignment is unchanged (0.361 → 0.364).
- **Identity is as good or better.** It matches or beats pooled at α = 0.1 everywhere and at α = 0.3 on B16. Pooled
  wins only where identity over-shrinks (α = 0.5 at B16, α = 0.3 at B32).
- **By stratum:** low-exposure experts have the highest raw mismatch and the largest absolute reduction (B16: 0.669 →
  0.596).

**Frozen screen** (`stage1_decision.json`): `POOL-PRIOR-PROMISING`.
- **B16:** α = 0.5 is the only alpha passing all criteria: −7.3 %, 8/12 experts, 6.6 % better than identity, no layer
  regression.
- **B32:** α = 0.3 passes: −6.6 %, 11/12, 2.9 % better than identity.
- **Selection consequence:** the rule picks the lowest-D_F *passing* alpha. At B16 the best pooled alpha, 0.3, fails C3
  (identity is better), so α\* = 0.5 there. This was frozen before any GPTQ result.

## Stage 2: quantization quality (`quantization_results.json`)

J is computed on the confirmation split (256 × 512 tokens). It is the c²-weighted complete expert MLP output error,
with frozen BF16 routing, fp32 math and BF16-rounded quantized weights. The summary is the mean over the 12 experts, and
the 95 % CIs come from a paired sequence bootstrap.

| Method | Calibration | Hessian | percdamp | Mean J | vs standard, same budget |
|---|---|---|---|---|---|
| Standard GPTQ | B16 | routed | 1.0 | 0.017278 | — |
| Identity shrinkage | B16 | expert → I, α = 0.5 | 1.0 | 0.017061 | +1.26 % |
| **Layer-pooled shrinkage** | B16 | expert → pooled, α = 0.5 | 1.0 | 0.017169 | **+0.63 %** [0.42, 0.85] |
| Tuned damping | B16 | routed | 3.0 | 0.016448 | +4.81 % |
| Standard GPTQ | B32 | routed | 1.0 | 0.016469 | — |
| Identity shrinkage | B32 | expert → I, α = 0.3 | 1.0 | 0.016375 | +0.57 % |
| **Layer-pooled shrinkage** | B32 | expert → pooled, α = 0.3 | 1.0 | 0.016386 | **+0.50 %** [0.32, 0.71] |
| Tuned damping | B32 | routed | 2.0 | 0.016019 | +2.73 % |
| Standard GPTQ | B64 | routed | 1.0 | 0.015710 | (reference) |
| Standard GPTQ | B128 | routed | 1.0 | 0.015080 | (reference) |

**Pooled against the controls:**
- **Against identity:** −0.64 % at B16 (CI [−0.89, −0.36]) and −0.07 % at B32 (CI [−0.26, +0.14]).
- **Against tuned damping:** −4.4 % at B16 and −2.3 % at B32.

**Robustness:**
- 9/12 experts improve at both budgets.
- Worst expert: +1.8 % (B16) and +0.65 % (B32).
- Both layers improve: L8 +0.83 % / +0.62 %, L16 +0.40 % / +0.36 %.
- By stratum: low-exposure experts gain most (+1.30 % / +0.86 %). Median experts gain least (−0.20 % / +0.11 %).

**Frozen criteria (both budgets):** C1, C2 and C3 fail; C4, C5 and C6 pass. C7 passes: the subset replacement does not
contradict. C8 (correctness) and C9 (cost) pass.

**Secondary metrics (B16, pooled / identity / tuned vs standard):**

| Metric | Pooled | Identity | Tuned |
|---|---|---|---|
| Gate output | +3.1 % | +2.4 % | +2.4 % |
| Up output | +3.2 % | +2.6 % | +2.6 % |
| E3-form projection sum | +2.8 % | +2.2 % | +2.9 % |
| Unweighted complete expert | +1.4 % | +1.3 % | +4.5 % |

- **Projection-level vs complete-expert outputs.** Pooling *does* beat identity on gate/up projection outputs. That
  advantage does not carry through to complete expert outputs once the routing weights are applied.
- **Mismatch vs quality.** Spearman(ΔD_F, ΔJ) across the 12 experts is +0.83 (B16) and +0.63 (B32). Experts whose
  mismatch fell more also gained more in J, but the size of the J effect is small.

**Post-hoc observation (descriptive, not a criterion): where tuned damping's advantage comes from.**
- **Identity shrinkage is algebraically damping.** Identity shrinkage at α with percdamp 1.0 equals the raw H with
  percdamp (1+α)/(1−α), up to a global scale. At B16, α = 0.5 gives exactly 3.0, the tuned value.
- **The gate/up weights match.** The identity and tuned gate/up weights agree except in ≤ 0.1 % of elements
  (fp rounding).
- **So the gap is the down projection.** Tuned damping's extra +3.5 % J at B16 comes almost entirely from damping
  down_proj (d = 1408).
- **Pooling cannot reach that gap.** By design, it touches only gate/up.

## Gap recovery (`quality_cost_comparison.json`)

R = share of the doubling gain recovered:

| Budget step | doubling gain | pooled | identity | tuned |
|---|---|---|---|---|
| B16 → B32 | 4.7 % | 13 % | 27 % | 103 % |
| B32 → B64 | 4.6 % | 11 % | 12 % | 59 % |

## Six-expert subset replacement (`subset_layer_output.json`, B16)

- **Setup:** all 60 experts use standard B16 GPTQ, against the same layer with the 6 selected experts' gate/up replaced
  by pooled weights.
- **Result:** the exact MoE output error on the confirmation split changes by −0.077 % (L8) and −0.009 % (L16).
- **Scale:** the six experts carry 9.8 % / 10.6 % of the baseline diagonal error.
- **Limits:** the result is consistent with the expert-level effect. It is **not** equivalent to pooling all 60
  experts.

## Cost (`timing_breakdown.json`)

Per layer, 128 sequences, medians of 3 interleaved repetitions:

| Component | L8 | L16 |
|---|---|---|
| Forward | 2.42 s | 2.43 s |
| Expert Hessians and hooks | 3.91 s | 3.79 s |
| Pooled accumulation (extra) | 0.11 s | 0.13 s |
| Pooled GEMM alone (CUDA events) | 0.134 s | 0.134 s |
| Mixing per expert (fp64, incl. host transfer) | 0.09–0.11 s | 0.07–0.11 s |
| 60-expert GPTQ solve (projected from 6 measured) | ≈ 44 s | ≈ 41 s |

- **Deployment overhead:** one pooled 2048² Hessian per layer plus one mix per expert, solved once. This is 0.22–0.27 %
  of replay + solve, so C9 passes. In FLOPs, pooled accumulation is 17 % of expert-Hessian accumulation, which is itself
  a small part of the pipeline.
- **Cost against benefit:**
  - Pooling costs about as much as 2–3 extra calibration sequences of replay, but recovers only 11–13 % of the gain
    from doubling the data.
  - Tuned damping is free and recovers 59–103 %.
- **MVP cost:**
  - Quantization: 10 candidate sets × 12 experts. Evaluation: 18 s per layer.
  - Peak GPU 7.9 GB, peak RSS 15.7 GB.

## Correctness (`correctness_results.json`): 21/21 pass

Chronological prefixes; pooled input = actual MoE input; every token counted once; natural routing; calibration and
tune disjoint; α = 0 reproduces H and GPTQ weights bitwise; trace preserved (pooled and identity); finite and symmetric;
PSD; solver unchanged (sha256 + git diff); down unchanged across A/B/C; tuned damping uses the documented values;
model weights never modified; no test data in alpha selection (decided before any quantization); outputs and routing
unchanged by the hooks; no RTN fallback; peak GPU < 20 GB; historical inputs unchanged (sha256); tests pass (20 new,
full suite 618 passed).

## Answers

1. **Closer to the reference?** Yes, under Frobenius loss: −10.7 % at B16 and −6.6 % at B32. The diagonal gets worse,
   and the pure prior is strongly biased.
2. **Does trace matching matter?** No. Untraced D_F is within 0.001; expert traces per token are close to the layer's.
3. **Does pooled beat equally strong identity shrinkage?** Only on Frobenius mismatch at the selected alphas, and on
   gate/up projection errors. Not on complete-expert J: −0.64 % at B16, −0.07 % at B32.
4. **Where does it help most?** At B16 (0.63 %), shrinking to 0.50 % at B32. At B64/B128 the best mismatch α is already
   0.1, with ≤ 4 % gain.
5. **Reconstruction or only mismatch?** Mostly mismatch. J improves by 0.5–0.6 %, below the 1 % bar.
6. **Larger gains for low-exposure experts?** Yes, modestly: +1.30 % / +0.86 % against +0.75 % / +0.52 % (high) and
   −0.20 % / +0.11 % (median). With 2 experts per stratum this is not conclusive.
7. **Recovers a meaningful share of doubling?** No: 11–13 %.
8. **Competes with tuned damping?** No: 2.3–4.4 % worse. Most of tuned damping's advantage comes from the down
   projection.
9. **Generalizes across L8 and L16?** The direction does: both are positive. L16's gain is about half of L8's.
10. **Genuine subset metric?** It improves, but negligibly: −0.08 % / −0.01 %, with the six experts carrying about 10 %
    of the error.
11. **Extra cost?** About 0.12 s per layer, 0.2–0.3 % of the pipeline.
12. **Worth a multi-seed study?** No. The frozen promising criteria fail on effect size and on both controls.

## Recommendation

**C: report the covariance-estimation finding without continuing algorithm development.**
- **What holds:** layer pooling lowers Frobenius mismatch of rank-deficient low-budget expert Hessians, and the GPTQ
  effect tracks it in direction.
- **Why it is not worth pursuing:** the effect is small, and an equally strong identity target or the existing tuned
  damping does as well or better.
- **Production default:** stays standard GPTQ with tuned damping.

Recommendation A was not executed.

## Artifacts

- **Run outputs:** `runs_hessian_pool_shrinkage_mvp/` (all required JSONs, `summary.md`, figures 1–5, logs, confirm npz).
- **Hessians and weights** (outside the repo): `~/moe_cache_hessian_pool_shrinkage/`.
- **Code:**
  - `moe_calib/hessian_shrinkage.py`
  - `diagnostics/hessian_pool_shrinkage_mvp.py`
  - `diagnostics/report_hessian_pool_shrinkage_mvp.py`
  - `tests/test_hessian_shrinkage.py`
- **Production files** (GPTQ solver, Profiler, Hessian accumulator): not modified.
