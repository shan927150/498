# Dense GPTQ numerical-perturbation dose response and act-order stabilization

Protocol `runs_dense_dose_response/protocol.json` (hash `417ce99114c71957`) was frozen on 2026-11-15 at 06:50, before
any dose result. A read-only copy is in `logs/protocol_readonly_copy.json`. The strong-path pilot criteria (hash
`c2a4e336`) were frozen before the pilot, and the pilot measured activations only. Section 6 lists the deviations.

## Outcomes

| question | answer | key numbers |
|---|---|---|
| Does R_func rise monotonically with dose? | **Yes** | ρ(D_H) 0.88 / 0.90, ρ(k) 0.98–0.99; dose means never decrease |
| Dose-response form | **Concave power law** | R_func ∝ D_H^**0.246** [0.218, 0.300] (natural), D_H^0.256 (fixed); R² 0.81 (0.89 with strong path) |
| What drives it? (post-hoc) | **Number of perturbed rows k, not ‖ΔH‖** | R_func ∝ k^0.23, R² 0.988; D_H adds nothing given k |
| Does PPL spread follow R_func? | **Flat, then rises** | batch-1 R_func 0.15 → 0.44 with R_PPL flat at 0.03–0.05; strong path R_func 0.665 gives R_PPL **0.156** |
| Saturation? | **Not established** | solver: top-dose D_H ratio only 1.4× (needs 2×); evaluation: M3 wins by < 2 AIC |
| Reaches the MoE range? | **Nearly** | strong path k = 128 R_func 0.752 = 91 % of MoE 0.823; extrapolation needed 1.09× |
| Fixed act-order mitigation | **Halves function at every dose; PPL benefit inconsistent** | R_mitigation 0.45–0.51, 56/56 pairs; PPL-SD ratio 0.33–1.13; +0.0046 PPL |
| Reference-permutation sensitivity | **Substantial** | mean PPL range 0.0096 over 5 references, vs within-reference SD 0.0026; ORIGINAL is the worst |

**Headline.**
- **The MoE point lies on the dense curve, descriptively.** The dense strong path reaches R_func 0.665 at R_PPL
  0.156 [0.10, 0.32]. The MoE sits at 0.823 and 0.144. This is consistent with MoE grouping sensitivity being the same
  GPTQ mechanism driven by a larger, broader calibration perturbation. It is a descriptive consistency, not a proof.
- **The solver response is extremely concave.** One perturbed row already gives R_func ≈ 0.15, a third of the
  128-row effect. Within the batch-1 family the four historical singletons span 50× in D_H at the same R_func.
- **Integer codes exaggerate both dose dependence and mitigation.** Code disagreement rises from 1 % to 37 % (slope 0.43
  in D_H vs 0.25 for function). Fixed act-order cuts codes by 81–92 % but function by only about 50 %.
- **Fixed act-order swaps execution sensitivity for reference sensitivity.** The previously reported fixed-order
  quality cost (+0.0076) comes mostly from the ORIGINAL permutation happening to be the worst of five references.

**Validation** (all mandatory checks pass; `actorder_correctness.json`, `activation_cache_validation.json`):
- **Cache splicing.** Batch-1 activations reproduce every historical singleton row bitwise. Each spliced cache equals
  the historical grouping cache on all 217 rows. Physical mixed executions at k = 4 and 16 equal the spliced caches
  bitwise.
- **k = 0.** Natural and fixed both reproduce the historical ORIGINAL Hessians, weights and per-window NLL bitwise.
- **Historical singletons.** The four k = 1 configs (rows 491, 1024, 1667, 1064) reproduce Hessians, act-order
  permutations, integer codes, weights and PPL bitwise.
- **Integrity.** All 126 models are valid. Only the 12 target tensors change, repeat evaluation is bitwise, and the
  targets are restored to pristine. BF16 re-evaluates bitwise at the end of both GPU sessions.
- **Reference matrix.** The diagonal equals historical natural, and the ORIGINAL-reference row equals the previous
  fixed-order run (9/9 checks).
- **Functional pipeline.** It reproduces the G20261111 R_func of `runs_actorder_functional_stability` exactly.

## 1. Design

- **Model and target.** Qwen2.5-1.5B, 4 layers (L1, L9, L19, L27) × gate/up/down = 12 tensors. The calibration set
  (128 × 512 WikiText-2) and the production solver default are unchanged.
- **Dose.** k calibration rows are executed on a perturbed numerical path; the remaining rows use the batch-8 reference
  activations. Replicate subsets are nested.
- **Families:**
  - **batch-1 (`b1`, primary):** k = 1, 4, 16, 64 × 8 replicates, plus k = 0 and the k = 128 endpoint;
  - **optional:** k = 32 and 96 × 8;
  - **strong path (`sp`, SDPA attention):** k = 64 × 8 plus the k = 128 endpoint.
- **Policies.** Every configuration is solved under natural act-order and under fixed act-order with the ORIGINAL
  reference permutation. That makes 63 configurations and 126 solved and evaluated models.
- **Reference matrix.** A separate 5 × 5 crossed design: Q_ij = GPTQ(H_j, π_i) over the 5 historical Hessians and 5
  historical permutations, 25 models.
- **Metrics:**
  - **D_H:** pooled ‖H − H₀‖_F / ‖H₀‖_F;
  - **R_func:** the full-MLP output difference from ORIGINAL divided by ORIGINAL's own quantization error, on held-out
    activations;
  - **s_k:** the SD of PPL over 8 replicates, with R_PPL = s_k / Q₀;
  - **Q₀:** PPL₀ − PPL_BF16 = 12.62415 − 12.52345 = **0.1007**.
- **Replicates.** Deterministic reruns are not counted as replicates. Each replicate uses a different row subset.

## 2. Strong-path pilot

Activation difference vs batch 8 (median relative L2 over target layers):

| path | difference | vs batch 1 |
|---|---|---|
| batch 1 | 0.0048 | 1× |
| batch 2 | 0.0047 | ≈ 1× |
| batch 4 | 0.0055 | 1.15× |
| batch 16 | 0 (bitwise) | — |
| BF16 reduced-precision reduction off | 0 (bitwise) | — |
| **SDPA attention** | **0.051** | **10.6×** |

- **Selection.** SDPA was selected under the frozen rule: deterministic, well-defined per row, and at least 2× the
  batch-1 difference.
- **Post-pilot caveat** (`logs/strong_path_equivalence_check.json`). SDPA moves the BF16 model's own PPL from 12.523 to
  12.246 (−2.2 %).
  - It is a large, systematic numerical perturbation, not rounding noise.
  - Its results are labelled `sp` and never pooled with `b1`.

## 3. Dose response

| family / k | D_H | natural code | natural R_func | fixed R_func | natural s_k (R_PPL) | fixed s_k (R_PPL) |
|---|---|---|---|---|---|---|
| b1 k=1 | 4.7e-5 | 1.07 % | **0.150** | 0.073 | 0.0033 (0.033) | 0.0019 (0.019) |
| b1 k=4 | 1.0e-4 | 1.81 % | **0.209** | 0.104 | 0.0046 (0.046) | 0.0033 (0.033) |
| b1 k=16 | 3.7e-4 | 3.24 % | **0.300** | 0.149 | 0.0048 (0.048) | 0.0032 (0.032) |
| b1 k=32 (opt.) | 5.8e-4 | 4.32 % | 0.338 | 0.170 | 0.0053 (0.052) | 0.0031 (0.031) |
| b1 k=64 | 1.0e-3 | 5.74 % | **0.399** | 0.199 | 0.0041 (0.041) | 0.0035 (0.035) |
| b1 k=96 (opt.) | 1.3e-3 | 6.77 % | 0.437 | 0.221 | 0.0038 (0.037) | 0.0042 (0.042) |
| b1 k=128 (endpoint) | 1.8e-3 | 7.42 % | **0.461** | 0.234 | — | — |
| sp k=64 | 8.2e-3 | 29.4 % | **0.665** | 0.299 | **0.0157 (0.156)** | 0.0052 (0.052) |
| sp k=128 (endpoint) | 1.4e-2 | 37.0 % | **0.752** | 0.353 | — | — |
| MoE reference | — | — | 0.823 | — | R_PPL 0.144 | — |

Full CIs, mean-PPL shifts and act-order position changes are in `summary.md` and `dose_response_models.json`.

**Perturbation → solver.**
- D_H rises roughly as k^0.75.
- The power-law support criterion (R² ≥ 0.8) is met.
- The local slope between the two largest batch-1 doses is 0.16. D_H grows only 1.4× there, so solver saturation cannot
  be claimed.
- **Post-hoc** (`logs/posthoc_k_vs_DH.json`):
  - R_func ∝ k^0.23 with R² 0.988 (natural) and 0.985 (fixed).
  - Adding log D_H gives a coefficient of 0.004.
  - Within a dose, D_H varies up to 10× with little change in R_func.
  - Across families, amplitude does matter: at k = 64, sp gives 0.665 vs b1 0.399.
- **Per layer.** L9 > L19 ≈ L27 > L1 at every dose; L9 and L19 down_proj carry the largest projection error.

**Functional → PPL.**
- **Batch-1 family.** R_PPL is flat and all χ² CIs overlap.
  - Natural only: the constant model is best by AIC, with √X +1.2, saturating +1.6 and proportional +8.1.
  - Both policies: √X is best, saturating +0.6, constant +5.6, proportional +15.7.
- **All families.** The saturating model is best but only by 1.3–1.6 AIC, below the prespecified 2.
- **Shape.** The data show an insensitive regime at low R_func followed by a rise. This is not saturation at high R_func.
- **Mean PPL.** Natural configurations average 0.002–0.008 below ORIGINAL in the batch-1 family, because the ORIGINAL
  draw sits high. The strong path lowers the natural mean by about 0.028.

## 4. Act-order mitigation and reference sensitivity

**Mitigation** (`actorder_mitigation.json`):
- **Function.** R_mitigation (fixed / natural RMS R_func) is 0.48–0.51 at every batch-1 dose and 0.45 on the strong
  path. Every bootstrap CI excludes 1, and fixed is lower in all 56 replicate pairs.
- **Codes.** The code ratio is 0.13–0.19 (strong path 0.08).
- **PPL SD.** The fixed / natural SD(PPL) ratio is 0.59 / 0.72 / 0.67 / 0.59 / 0.86 / **1.13** at k = 1 / 4 / 16 / 32 /
  64 / 96, and 0.33 on the strong path. It is not consistently below 1.
- **Closeness to ORIGINAL.** The pooled prespecified PPL benefit is positive (CI [1.7e-7, 3.3e-7]): fixed configurations
  sit closer to ORIGINAL's PPL. This partly holds by construction.
- **Quality cost.** +0.0046 PPL [0.0031, 0.0059] vs natural, with the ORIGINAL reference.

**Reference matrix** (`reference_order_matrix.json`):

| reference π | mean PPL over H | SD over H | R_func vs own reference |
|---|---|---|---|
| G20261111 | 12.6189 | 0.0025 | 0.074 |
| G20261112 | **12.6168** (best) | 0.0036 | 0.075 |
| G20261113 | 12.6230 | 0.0012 | 0.074 |
| G20261114 | 12.6187 | 0.0033 | 0.075 |
| G20261115 = ORIGINAL | **12.6265** (worst) | 0.0026 | 0.075 |

- **Between references:** the mean PPL ranges 0.0096. Within a reference, the mean SD is 0.0026; across references at a
  fixed Hessian, it is 0.0043.
- **Reference choice decides the cost.** With π_G20261112 as reference, fixed act-order would average below the natural
  mean (12.6188).
- **Functional disagreement** relative to the chosen reference is the same for every reference.

## 5. Conclusions

- **Mechanism.**
  - Real execution-path differences perturb the Hessian monotonically.
  - The GPTQ solver responds to how many rows are perturbed, very concavely.
  - Function grows as about k^0.23. PPL spread stays near 0.04 Q₀ until R_func is well above 0.44.
- **Dense vs MoE.**
  - The dense strong path reaches the MoE regime in both R_func and R_PPL. This supports, without proving, the reading
    that MoE grouping sensitivity is the same GPTQ mechanism under a larger perturbation, with nearly every row and
    layer perturbed.
  - Caveats:
    - The architectures, quantized fractions and Q₀ (0.031 MoE vs 0.101 dense) differ.
    - D_H definitions differ.
    - The strong path is a systematic perturbation.
- **Practical.**
  - Fixed reference act-order is a dependable *functional* stabilizer (about ×0.5).
  - It is not a dependable PPL-variance stabilizer.
  - Its quality depends on which reference permutation is chosen, by up to 0.01 PPL.
  - Any use should select or vary the reference, not take ORIGINAL by default.
- **Diagnostic.** Integer-code disagreement should not be used as a dose or mitigation metric.
- **Replicate requirements.**
  - Execution SD(PPL) is about 0.003–0.005 at every batch-1 dose; a single-row difference already gives about 70 % of
    the spread.
  - GPTQ method comparisons at the 0.01-PPL scale need at least 5–8 execution replicates per arm.
  - Variance comparisons need far more.
  - Fixed-reference designs must also vary the reference.

**Recommended next steps.**
- **Fill the evaluation gap.** Add an intermediate-amplitude perturbation family between b1 (D_H ≤ 1.8e-3) and sp
  (8e-3), with more replicates (≥ 16), to separate threshold, √X and saturating PPL models.
- **Find a cleaner strong path.** Use a PPL-equivalent strong perturbation, adding a pilot criterion on BF16 PPL, so
  the high-R_func points are not confounded by a systematic shift.
- **Test reference selection.** Check whether a selected reference permutation (for example, the best of N on held-out
  data) generalizes, or whether the 0.0096 range is itself noise.

## 6. Deviations

- **Before the freeze.** The audit's reference-permutation argsort first ran on the CPU, where down_proj near-ties sort
  differently. It was moved to the solver's GPU device.
- **Pilot criteria gap.** The criteria lacked a PPL-equivalence requirement. SDPA's −2.2 % BF16 PPL shift was found
  afterwards and is recorded as a post-pilot diagnostic.
- **After the freeze.** `functional_one` in `diagnostics/dense_dose_response.py` gained an optional `wref` argument,
  used only by the `func` and `refmatrix` stages. The `gpu` stage ran on frozen code. The change is logged in
  `logs/code_state.json`.
- **Post-hoc analyses**, outside every criterion: the k-vs-D_H regression and the SDPA equivalence check.
- **MoE reference.** The R_func denominator uses the mean quantization error over the 6 MoE configs.
- **Statistics.** Eight replicates per dose give χ² SD intervals spanning about a factor of 3. The k = 128 endpoints are
  single models.

## 7. Files

- **Code:**
  - `diagnostics/dense_dose_response.py` (stages: audit, pilot_criteria, pilot, protocol, caches, gpu, refmatrix, func,
    analyze)
  - `diagnostics/report_dense_dose_response.py`
  - `diagnostics/dense_dose_posthoc.py`
  - reuses `moe_calib/actorder_policy.py` and `moe_calib/functional_diff.py`
- **Outputs:** `runs_dense_dose_response/`
  - `summary.md`: answers to the 12 questions;
  - the required JSONs:
    - `protocol.json`
    - `calibration_manifest.json`
    - `perturbation_manifest.json`
    - `strong_path_pilot.json`
    - `activation_cache_validation.json`
    - `actorder_correctness.json`
    - `hessian_dose_results.json`
    - `dose_quantization_results.json`
    - `dose_functional_results.json`
    - `dose_ppl_results.json`
    - `dose_variability.json`
    - `dose_response_models.json`
    - `actorder_mitigation.json`
    - `reference_order_matrix.json`
    - `cost_results.json`
    - `artifact_inventory.json`
  - `figures/fig1`–`fig7`;
  - `results/`;
  - `logs/`.
- **Cost.** GPU stage 1.76 h; reference matrix 15 min; functional stage 60 min.
- **Cache.** `~/moe_cache_dense_dose_response/`, 165 GB, mostly per-configuration Hessians. It can be deleted if no
  rerun is planned.
