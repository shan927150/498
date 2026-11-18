# Functional impact of GPTQ code disagreement and act-order stabilization

Protocol `d6202dc2b91f5cd3`, frozen 2026-11-15 04:42 before any functional or PPL result. There were no amendments. The
hashed code was identical at every stage (`logs/code_state.json`). Experiment 1 finished before any Experiment 2
evaluation.

## Outcomes

| question | result |
|---|---|
| **Experiment 1** (prespecified F1 / F2 / F3) | **`F3-MIXED`**: R_MLP = 3.14 (MoE block vs dense MLP). Projection ratios conflict: functional / code ratio is 0.36 for gate, 0.18 for up and 1.98 for down |
| **Experiment 2: fixed reference act-order** | **INTERMEDIATE**: code ×0.15, functional ×0.49, R_SD = 0.508 (threshold 0.5), quality acceptable (+0.0076 PPL) |
| **Experiment 2: bucketed act-order, 8 bits** (reference-free, primary) | **INTERMEDIATE, effectively no effect**: code ×0.97, functional ×1.08, R_SD = 1.09; mean PPL −0.008 |
| **Experiment 2: disabled act-order** | **`D-CODE-STABILITY-ONLY`**: code ×0.14 but functional ×0.86. R_SD = 0.37, but the quantization penalty grows 2.2× (+0.117 PPL) |

Interpretation is not blocked:
- Every mandatory check A–H passes.
- The final BF16 re-evaluation is bitwise identical, the non-target parameters are untouched, and the targets are
  restored to pristine (`actorder_correctness.json`).

**Headline.**
- Integer-code disagreement is a poor proxy for functional disagreement across projections and policies, and a poor
  proxy for PPL spread.
- The ~18× MoE/dense code gap becomes a 3–14× functional gap, depending on the denominator and layer pooling.
- MoE grouping variants differ by about 0.8× their own quantization error; dense variants by 0.06–0.37×.
- Fixing the act-order permutation halves the functional disagreement and the PPL SD at a small quality cost.
- The prespecified reference-free alternative (8-bit bucketed) does not stabilize anything.
- The five-grouping SD(log PPL) is too noisy to rank policies by itself. Exploratory bucketed variants with **unchanged**
  code and functional disagreement show R_SD from 0.27 to 1.09.

---

## Experiment 1: functional sensitivity of saved weights (no GPTQ solves)

### Activation capture (`functional_activation_manifest.json`)
- **Reference and windows.** BF16 original checkpoints on the historical windows (`runs_partial_ppl/windows.pt`,
  128 × 512 = 65,536 tokens).
- **Bitwise capture.** Hooks capture the *actual* projection inputs: the MLP/expert input for gate/up and the real
  `down_proj` input. The capture forward reproduces the historical BF16 per-window NLL **bitwise** in both models (dense
  PPL 12.523451, MoE 9.237090).
- **MoE routing.**
  - Routing is recomputed exactly as in the HF block (softmax fp32 → top-4 → bf16, `norm_topk_prob=False`).
  - Every one of the 3,840 expert calls has a gate input equal to `X[routed tokens]` bitwise.
  - There are 262,144 routed (token, expert) pairs per layer.
- **Coverage.** All 240 MoE experts are included in the functional analysis. Code metrics exist only for the 20
  sampled experts, and are reported only on that aligned subset.
- **Arithmetic.**
  - FP32 with TF32 off, on installed BF16 weights and BF16 activations; squared norms accumulate in float64.
  - Projection error is computed as `(W_a − W_b) X` directly.
  - E is relative to the **BF16-original** output.
  - The MoE aggregate uses shared reference routing; the shared expert cancels.

### Functional error, primary comparisons (pooled over layers by squared norms)

| | gate | up | down | full MLP / MoE block | MoE routed-only | quantization error vs BF16 (MLP / block / routed) |
|---|---|---|---|---|---|---|
| dense (4 single-row perturbations vs ORIGINAL) | 0.32 % | 0.71 % | 0.61 % | **0.68 %** | — | 4.49 % |
| MoE (5 groupings vs ORIGINAL, 240 experts) | 6.06 % | 6.73 % | 6.08 % | **2.14 %** | 9.87 % | 2.60 % / 12.0 % |

- Output cosine similarity stays at ≥ 0.9985 (dense MLP) and 0.994–0.99994 (MoE block) per layer.
- **Null.** The dense G20261115 vs ORIGINAL difference is exactly 0 in every unit.
- **Per-expert MLP error** (unweighted, 240 experts): median 11.0 % (p10 6.5 %, p90 13.6 %, max 17.4 %).
  - By token tertile: low 11.7 %, mid 11.0 %, high 10.5 %.
  - Spearman ρ between tokens and error is −0.22. Rare experts are only mildly more sensitive.

### Dense vs MoE ratios (`functional_vs_code_comparison.json`, Fig. 2; different quantities, exploratory)

| quantity | MoE / dense |
|---|---|
| integer-code disagreement (20 sampled experts) | **17.86** (gate 52.0, up 51.6, down 5.06) |
| BF16 element disagreement (all experts) | 7.80 (7.62 on the sampled experts) |
| functional, gate / up / down | **18.95 / 9.43 / 10.04** |
| functional, MLP vs MoE block (primary R_MLP) | **3.14** |
| functional, MLP vs MoE routed-only output | 14.49 |
| SD(log PPL), 5 groupings | **1.20** |

**Classification `F3-MIXED`.**
- R_MLP = 3.14 falls between the F1 bound (≤ 3) and the F2 bound (≥ 8.93).
- The projections conflict:
  - **gate and up:** the code ratio overstates the functional ratio by 2.7× and 5.5×;
  - **down:** the functional ratio (10.0) is *twice* the code ratio (5.1).

**Post-hoc sensitivity** (`logs/posthoc_layer_matched_ratios.json`; not in the classification).
- The pooled R_MLP is dominated by the layers with large output norms.
- **Depth-matched layer pairs:**

  | layer pair (dense–MoE) | R_block | R_routed |
  |---|---|---|
  | 1–1 | 7.0 | 9.5 |
  | 9–8 | 1.6 | 2.2 |
  | 19–16 | 2.2 | 2.9 |
  | 27–23 | 2.6 | 13.1 |

- **Mean of per-layer errors:** R_block = 1.99, R_routed = 3.49.
- Grouping-induced difference / own quantization error, per layer:
  - **dense:** 0.06 / 0.37 / 0.24 / 0.14;
  - **MoE:** 0.42 / 0.87 / 0.79 / 0.83, the same for block and routed denominators.
- **What this implies.** MoE grouping variants are nearly decorrelated quantization draws. If two errors of equal norm
  have correlation ρ, their difference is √(2(1 − ρ)) times each norm. A ratio of 0.82 gives ρ ≈ 0.66; the dense
  model's 0.15 gives ρ ≈ 0.99.

### Correlations (descriptive; units share configurations, so they are not independent)
- **Independent configurations:**
  - dense: 4 single-row perturbations plus ORIGINAL;
  - MoE: 6 configurations.
- **Dense, layer × projection × 4 primary pairs (n = 48).** Spearman ρ between E_func and:
  - code 0.74;
  - BF16 0.83;
  - relative Frobenius difference 0.80;
  - act-order position change 0.48;
  - Hessian perturbation 0.54.
- **Dense, within one projection:** gate code ρ = 0.55 (n = 16), down ρ = 0.99.
- **MoE sampled experts × projection × 5 pairs (n = 300):** code 0.68, BF16 0.70, relative Frobenius 0.67, act-order
  0.71, Hessian perturbation 0.79.
- **MoE, all experts:** BF16 vs E has ρ = 0.73 / 0.86 / 0.91 for gate / up / down.
- **Both models pooled (n = 348):** code ρ = 0.79, with log-Pearson 0.86. The models fall on **different lines**, and
  down_proj sits on a separate line in each (Fig. 1).
- **Pair level vs PPL:** functional error does **not** predict |Δ log PPL| between two variants (dense ρ = −0.13,
  n = 10; MoE ρ = −0.21, n = 15).

### Curvature and functional importance (`logs/curvature_raw.json`, Fig. 5)

Held-out quadratic forms tr(ΔW H_heldout ΔWᵀ) were computed for dense (12 tensors × 4 pairs) and MoE (20 experts × 3
projections × 5 pairs).

**Code flips, not grid refits, carry the functional change.** Median shares of the energy:

| | flipped elements | same-code (grid) changes | cross term |
|---|---|---|---|
| dense gate / up | 98.6 % | 1.7 % | −0.4 % |
| dense down | 94.9 % | 6.4 % | −1.4 % |
| MoE gate / up | 85 % | 20 % | −6 % |
| MoE down | 89 % | 13 % | −1 % |

- Per element, a flip carries 370–390× (dense gate/up), 74× (dense down), 6× (MoE gate/up) and 18× (MoE down) the energy
  of a grid-only change.

**Low-impact concentration**, measured in the low-curvature half of held-out diag(H) columns:

| | share of flips | share of energy |
|---|---|---|
| dense gate / up | 73 % | 73 % (block-exact) |
| MoE gate / up | 56 % | 64 % |
| **dense down** | **58 %** | **40 %** |
| **MoE down** | **55 %** | **34 %** |

- Flips are **proportionate** to impact for gate/up.
- Down_proj is the only projection where a disproportionate share of flips falls in low-impact directions.

**Caveats.**
- Execution position and calibration diag(H) are the same ordering by construction.
- The diagonal approximation overstates the full form by 6–21 % (median), so column contributions are not additive.

---

## Experiment 2: dense act-order stabilization

### Policies and validation
- Every configuration is re-solved from the **saved historical Hessians**: the same 128 C4 sequences, 65,536 tokens and
  identical sha256.
- The production solver is unchanged. Policies use only the existing `actorder_override` and `actorder=False` switches
  (`moe_calib/actorder_policy.py`).
- **Groups:** consecutive 128 columns of the execution order (solver semantics, no static groups). Group assignments
  are saved for every policy.

**Mandatory checks (`actorder_correctness.json`):**
- **A:** the natural re-solve equals the historical weights bitwise (72/72), and natural PPL per window equals the
  historical values bitwise (6/6).
- **B:** the fixed path with each configuration's own natural permutation is bitwise equal to natural (72/72).
- **C:** bucketed permutations are identical across repeated, cloned and GPU-sourced computations, and the full
  repeated solve is bitwise.
- **D:** disabled gives perm = arange and group = column // 128.
- **E:** Hessian sha256 and token counts match, and the reference permutation equals the saved ORIGINAL permutation.
- **F:** G20261115 equals ORIGINAL bitwise within every policy.
- **G:** only the 12 target tensors change.
- **H:** repeat evaluation is bitwise identical.

### Results (5 groupings per policy; `actorder_ppl_results.json`, `actorder_stability_comparison.json`, `actorder_code_difference.json`, `actorder_quality_tradeoff.json`)

| policy | code disagr. | E_MLP | SD(PPL) | SD(log PPL) | R_SD | mean PPL | penalty vs BF16 | Δ mean vs natural | quant error E vs BF16 |
|---|---|---|---|---|---|---|---|---|---|
| natural | 1.18 % | 0.68 % | 0.00503 | 3.99e-4 | 1.00 | 12.6188 | 0.0954 | — | 4.49 % |
| fixed (reference order) | **0.18 %** | **0.34 %** | 0.00256 | 2.03e-4 | **0.51** | 12.6265 | 0.1030 | +0.0076 | 4.49 % |
| bucketed, 8 bits (primary) | 1.14 % | 0.74 % | 0.00548 | 4.35e-4 | 1.09 | 12.6110 | 0.0875 | −0.0079 | 4.48 % |
| disabled (desc_act=False) | **0.16 %** | 0.59 % | 0.00189 | 1.48e-4 | 0.37 | **12.7361** | **0.2126** | **+0.1173** | **7.12 %** |
| bucketed, 6 bits (exploratory) | 1.10 % | 0.76 % | 0.00137 | 1.09e-4 | 0.27 | 12.6215 | 0.0981 | +0.0027 | 4.50 % |
| bucketed, 10 bits (exploratory) | 1.17 % | 0.74 % | 0.00241 | 1.91e-4 | 0.48 | 12.6148 | 0.0914 | −0.0040 | 4.49 % |

Column notes:
- Code disagreement and E_MLP are means over the 4 primary pairs, against the policy's own ORIGINAL.
- BF16 PPL is 12.5235.

**Grid and order changes** (4 pairs vs the policy ORIGINAL):

| policy | act-order position changed | group reassigned | zero-point changed | scale changed | level (q − zero) changed | rel. Frobenius |
|---|---|---|---|---|---|---|
| natural | 13.3 % | 0.153 % | 0.72 % | 93 % | 0.62 % | 0.021 |
| fixed | 0 | 0 | 0.08 % | 90 % | 0.11 % | 0.0086 |
| bucketed8 | 10.8 % | 0.156 % | 0.71 % | 92 % | 0.60 % | 0.021 |
| disabled | 0 | 0 | 0.07 % | 89 % | 0.10 % | 0.0088 |

**Window-paired quality vs natural**, in nats/token over 128 windows with a bootstrap 95 % CI:
- **fixed (G20261111):** +0.00092 [0.00047, 0.00134]. Worse on 66 % of windows; per grouping it ranges from 0 to +0.0013.
- **disabled:** about +0.0090 [0.0063, 0.0118] for every grouping. Worse on 70 % of windows.
- **bucketed8 (ORIGINAL):** −0.0013 [−0.0022, −0.0005]. A small improvement.

**Critical limitation.** The five groupings are four single-row perturbations plus an exact null. SDs are descriptive.
For two independent 5-sample SDs, the F(4,4) 95 % band for the SD ratio is about [0.32, 3.1]. Every observed R_SD
(0.27–1.09) lies essentially inside it.

The exploratory bucketed rows show this directly:
- Their code and functional disagreement is unchanged (×0.94–0.99, ×1.08–1.12).
- Yet R_SD is 0.27 at 6 bits, 1.09 at 8 bits and 0.48 at 10 bits.
- R_SD therefore cannot by itself separate policies at this sample size.

**Why bucketing at 8 bits failed.**
- Bucket keys change for only 0.2–1.1 % of columns, but each change moves a column across a bucket boundary. Ties are
  then ordered by column index.
- Act-order positions still differ on 10.8 % of columns, and group reassignment (0.156 %) is unchanged.
- The code flips and the functional difference persist. Coarsening the sort key does not remove the near-tie instability
  that matters, which is group-boundary crossing.

---

## Answers

1. **Is integer-code disagreement a good predictor of functional disagreement?**
   - **Within a projection type of one model:** moderately to very well. Dense down has ρ = 0.99; dense gate 0.55; MoE
     0.68–0.80.
   - **Across projections, models or policies:** no.
     - The models lie on different lines, and down_proj on a separate line.
     - Disabled act-order cuts codes 86 % but functional error only 14 %.
     - BF16 disagreement and the Hessian perturbation are at least as predictive as codes.
   - Neither code nor functional disagreement predicts pairwise |Δ log PPL|.
2. **Does the ~18× code gap correspond to a comparably large functional gap?** Partly, depending on projection and
   denominator:
   - gate 19× (vs 52× code), up 9.4× (vs 52×), down 10× (vs 5.1×);
   - full MLP vs MoE block 3.1× (routed-only 14.5×);
   - depth-matched layers: 1.6–7.0× (block) and 2.2–13× (routed).
   The block-level gap is close to the 1.2× SD(log PPL) ratio. The routed-expert gap is not.
3. **Are code flips concentrated in low-impact directions?**
   - **gate/up:** no, flips are proportionate to held-out curvature impact.
   - **down_proj:** partly. 55–58 % of flips fall in the low-curvature half of columns, which carries 34–40 % of the
     energy.
   - Flips (not grid refits) carry 85–99 % of the functional difference in both models.
   - Many *BF16* changes, by contrast, are low-impact same-code grid refits.
4. **Does fixing act-order reduce PPL variability?** Descriptively, SD(log PPL) halves (R_SD 0.508; 0.54 over the 4
   non-null groupings), together with code ×0.15 and functional ×0.49. It just misses the prespecified 0.5 threshold,
   and with 4 perturbations it is not distinguishable from noise.
5. **Does reference-free bucketed act-order achieve a similar effect?** No.
   - At the prespecified 8 bits it reproduces neither the code reduction (×0.97) nor the functional reduction (×1.08),
     and R_SD = 1.09.
   - The 6- and 10-bit exploratory variants also leave code and functional disagreement unchanged.
6. **What quality cost does each policy introduce?**
   - **fixed:** +0.0076 PPL (+8 % of the natural quantization penalty; window-paired CI excludes 0 for G20261111).
   - **bucketed8:** −0.0079, a slight gain.
   - **disabled:** +0.117 PPL, so the penalty grows 2.2× and the quantization error rises from 4.5 % to 7.1 %.
7. **Is disabled act-order a reasonable practical alternative?** No.
   - It buys code stability (×0.14) and a lower SD at a large quality cost.
   - Its functional disagreement barely falls (×0.86).
   - Hypothesis D holds for this policy.
8. **Does reduced code disagreement reliably imply reduced functional and PPL variability?** No.
   - **Fixed and disabled:** the same code reduction (×0.15 / ×0.14) gives different functional reductions (×0.49 /
     ×0.86).
   - **Bucketed variants:** no code or functional change, yet R_SD varies 0.27–1.09.
9. **Practical mitigation vs diagnostic conclusion.**
   - **Practical (conditional):** fixed reference act-order. It halves functional disagreement and SD at a small quality
     cost. It needs a canonical reference calibration run and stored permutations, so it is not reference-free, and its
     PPL benefit is not statistically established.
   - **Diagnostic only:**
     - code-flip counts as a stability metric;
     - bucketed act-order as a substitute;
     - disabled act-order;
     - every R_SD ranking among policies.
   - **Diagnostic finding:** activation-weighted functional disagreement relative to the model's own quantization error
     separates MoE (≈ 0.8) from dense (0.06–0.37) more informatively than code counts.
10. **Next test to separate functional insensitivity from dose-response saturation.**
    - **Design:** a dense dose-response over the number of perturbed calibration rows (1, 4, 16, 64, 128), under natural
      vs fixed act-order.
    - **Replicates:** ≥ 8 independent perturbations per dose, so that SD(log PPL) has usable precision.
    - **Measure at each dose:** E_MLP, E relative to the quantization error, and SD(log PPL).
    - **Interpretation:** saturation predicts E/quant-error → about 0.8 (the MoE value) while SD(log PPL) plateaus.
      Functional insensitivity predicts SD rising roughly in proportion to E.
    - **Optional MoE arm:** a fixed-order MoE arm on the sampled experts would test whether the 0.8 decorrelation is
      act-order driven.

## Conclusions by dimension

- **Code-level stability.**
  - Fixed and disabled act-order eliminate act-order and group reassignment, leaving 14–15 % of natural code flips.
  - Bucketed act-order (any precision tested) does not reduce code flips.
  - MoE code disagreement is 18× dense on the sampled experts.
- **Functional stability.**
  - Fixed act-order halves dense MLP disagreement; disabled act-order barely helps; bucketed does not.
  - MoE grouping variants differ by about 0.8× their own quantization error, against 0.06–0.37× in dense.
  - The functional MoE/dense gap (3–14×) is smaller than the code gap for gate/up and larger for down.
- **PPL stability.** Descriptive R_SD values: fixed 0.51, disabled 0.37, bucketed 0.27–1.09. None is distinguishable
  from five-grouping SD noise, and dense and MoE SD(log PPL) differ only 1.2×.
- **Quantization quality.**
  - fixed +0.0076 PPL (acceptable by the prespecified rule);
  - bucketed8 −0.008;
  - disabled +0.117 (unacceptable, 2.2× penalty).

## Deviations and limitations

- **Protocol and code.** No protocol amendment. The hashed code did not change after the freeze.
- **Changes after results (report module, formatting and assembly only):** the `artifact_inventory.json` builder, and
  the post-hoc layer-matched ratio script (`diagnostics/actorder_functional_posthoc.py`, labelled post-hoc).
- **Pooling dependence.** The prespecified R_MLP pools layers by squared norms and is dominated by large-norm layers.
  The layer-matched sensitivity analysis is reported but does not change the classification.
- **MoE scope.**
  - Code, act-order and Hessian metrics cover only the 20 sampled experts.
  - The MoE aggregate uses shared reference routing (a diagnostic intervention, not deployment routing).
  - The MoE denominator includes the unquantized shared expert; the routed-only alternative is reported alongside.
- **Experiment 2 scope.**
  - Dense only, one calibration seed, 4 single-row perturbations plus a null.
  - The optional static-group analysis (§15) was not run.
- **Artifacts.**
  - Activations: 11 GB.
  - Policy weights and codes: 17 GB.
  - MoE BF16 expert weights: 3.9 GB.
  - Location: `~/moe_cache_actorder_functional_stability/`, inventoried with sha256 in `artifact_inventory.json`.

## Files

- **Code:**
  - `moe_calib/functional_diff.py`
  - `moe_calib/actorder_policy.py`
  - `diagnostics/actorder_functional_stability.py`
  - `diagnostics/report_actorder_functional_stability.py`
  - `diagnostics/actorder_functional_posthoc.py`
  - `tests/test_actorder_functional.py` (7 tests)
- **Figures:**
  - Fig. 1: code vs functional disagreement;
  - Fig. 2: dense/MoE ratios;
  - Fig. 3: act-order stability;
  - Fig. 4: quality–stability Pareto;
  - Fig. 5: curvature and importance;
  - Fig. 6: grouping-level PPL.
