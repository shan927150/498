# Dense GPTQ numerical-perturbation dose response and functional stability (Qwen2.5-1.5B)

## Protocol and correctness

**Protocol.** `protocol.json` (hash `417ce99114c71957`, frozen 2026-11-15 06:50) was fixed before any dose result. A
read-only copy is in `logs/protocol_readonly_copy.json`. The strong-path pilot criteria were frozen and hashed
(`c2a4e336…`) before the pilot ran. The pilot measures activations only, never PPL.

**Cache validation** (`activation_cache_validation.json`):
- The batch-size-1 activations reproduce every historical singleton row bitwise, and each spliced (hybrid) cache equals
  the historical grouping cache on all 217 rows.
- Physical mixed executions match the spliced caches bitwise on all 217 rows (non-calibration fillers fill the gaps).
  This was checked for replicate 0 at k = 4 and k = 16.

**Mandatory parity** (`actorder_correctness.json`; all pass):
- **k = 0:** both policies reproduce the historical ORIGINAL Hessians, weights and per-window NLL bitwise. Fixed equals
  natural.
- **Historical singletons:** the four k = 1 singleton configs (rows 491, 1024, 1667, 1064) reproduce the historical
  Hessians, act-order permutations, integer codes, weights and per-window PPL bitwise.
- **Integrity:** every configuration is valid. Only the 12 target tensors change, repeat evaluation is bitwise, and the
  model is restored to pristine. At the end, BF16 re-evaluates bitwise.
- **Reference matrix:** the diagonal equals the historical natural results, and the ORIGINAL-reference row equals the
  previous fixed-order run (9/9 checks).

**Design.**
- **Primary family (batch-1):** 4 doses (k = 1, 4, 16, 64) × 8 nested replicate subsets, plus k = 0 and the k = 128
  endpoint.
- **Optional doses:** k = 32 and 96 × 8.
- **Strong path (SDPA):** k = 64 × 8 plus the k = 128 endpoint.
- **Policies:** every configuration is solved under natural and fixed reference act-order.
- **Size:** 63 configurations, 126 solved and evaluated models.
- **Reference matrix:** a separate 5 × 5 crossed design of 25 models.

## Strong path (`strong_path_pilot.json`)

Pilot activation differences vs batch 8 (median relative L2 over target layers):

| path | difference vs batch 8 | vs batch 1 |
|---|---|---|
| batch 1 | 0.0048 | 1× |
| batch 2 | 0.0047 | ≈ 1× |
| batch 4 | 0.0055 | 1.15× |
| batch 16 | 0 (bitwise) | — |
| `allow_bf16_reduced_precision_reduction=False` | 0 (bitwise) | — |
| **SDPA attention** | **0.051** | **10.6×** |

- SDPA was **selected under the frozen rule**: it is deterministic, well-defined per row, and at least 2× the batch-1
  difference.
- **Caveat, found after the freeze** (`logs/strong_path_equivalence_check.json`): SDPA moves the BF16 model's own PPL
  from 12.523 to 12.246 (−2.2 %). Its median per-token difference at the layer-1 input is about 10 %.
  - It is the same attention computation on a different deterministic kernel path. The likely cause is that eager
    attention rounds QKᵀ to BF16 while SDPA keeps FP32 accumulation (a hypothesis, not verified).
  - It is therefore a **large, systematic** numerical perturbation, not rounding-scale noise.
  - Its results are labelled `sp` and never pooled with the batch-1 (`b1`) family.

## Main results

The quantization penalty is Q₀ = PPL₀ − PPL_BF16 = 12.62415 − 12.52345 = **0.1007**.

| family / k | D_H (pooled, mean) | act-order pos. changed | natural: code | natural: R_func RMS [95 % CI] | fixed: code | fixed: R_func RMS | natural: s_k (R_PPL [χ² CI]) | fixed: s_k (R_PPL) | natural: mean PPL − PPL₀ | fixed: mean PPL − PPL₀ |
|---|---|---|---|---|---|---|---|---|---|---|
| b1 k=1 | 4.7e-5 | 17.6 % | 1.07 % | **0.150** [0.144, 0.156] | 0.19 % | 0.073 | 0.0033 (0.033 [0.021, 0.066]) | 0.0019 (0.019) | −0.0019 | +0.0002 |
| b1 k=4 | 1.0e-4 | 32 % | 1.81 % | **0.209** [0.203, 0.218] | 0.34 % | 0.104 | 0.0046 (0.046) | 0.0033 (0.033) | −0.0042 | −0.0007 |
| b1 k=16 | 3.7e-4 | 48 % | 3.24 % | **0.300** [0.287, 0.315] | 0.55 % | 0.149 | 0.0048 (0.048) | 0.0032 (0.032) | −0.0073 | +0.0007 |
| b1 k=32 (opt.) | 5.8e-4 | 55 % | 4.32 % | 0.338 | 0.66 % | 0.170 | 0.0053 (0.052) | 0.0031 (0.031) | −0.0066 | −0.0006 |
| b1 k=64 | 1.0e-3 | 61 % | 5.74 % | **0.399** [0.394, 0.404] | 0.81 % | 0.199 | 0.0041 (0.041 [0.027, 0.083]) | 0.0035 (0.035) | −0.0052 | −0.0016 |
| b1 k=96 (opt.) | 1.3e-3 | 65 % | 6.77 % | 0.437 | 0.91 % | 0.221 | 0.0038 (0.037) | 0.0042 (0.042) | −0.0084 | −0.0036 |
| b1 k=128 (endpoint) | 1.8e-3 | — | 7.42 % | **0.461** (single) | 0.98 % | 0.234 | — (one PPL: 12.6184) | — (12.6132) | −0.0057 | −0.0109 |
| sp k=64 | 8.2e-3 | 92.5 % | 29.4 % | **0.665** [0.661, 0.669] | 2.26 % | 0.299 | **0.0157 (0.156 [0.103, 0.318])** | 0.0052 (0.052) | −0.0281 | −0.0076 |
| sp k=128 (endpoint) | 1.4e-2 | — | 37.0 % | **0.752** (single) | 2.85 % | 0.353 | — (12.5990) | — (12.6242) | −0.0252 | +0.0000 |

How to read the table:
- **D_H** is the pooled ‖H − H₀‖_F / ‖H₀‖_F over 4 layers × (gu, down). The historical trace-normalized metric is in
  `hessian_dose_results.json`; the two are not interchangeable.
- **R_func** is the full-MLP output difference from ORIGINAL divided by ORIGINAL's own quantization error, pooled over
  the 4 layers by squared norms. It uses the held-out activations of `runs_actorder_functional_stability`. This
  pipeline reproduces that run's G20261111 value exactly (relative difference 0).
- **s_k** is the SD of PPL over 8 replicates (ddof = 1), and R_PPL = s_k / Q₀.

Other references:
- The historical dense 5-grouping set has R_func ≈ 0.15 and R_PPL = 0.050.
- **MoE reference** (same definitions): R_func = 0.823 (block including the shared expert); R_PPL = 0.144 with the Q₀
  denominator (0.121 with the mean-penalty denominator).

### Perturbation → solver (Fig. 1, Fig. 2, `dose_response_models.json`)
- **D_H rises with k** without cancellation: roughly ∝ k^0.75 between k = 4 and 128.
- **R_func increases monotonically with dose** under both policies:
  - Spearman ρ with D_H is 0.88 (natural) and 0.90 (fixed); with k it is 0.98–0.99.
  - Dose means never decrease.
- **Power law in D_H.** R_func = a·D_H^b with b = **0.246** [0.218, 0.300] (natural) and **0.256** [0.229, 0.310]
  (fixed). R² is 0.81 for the batch-1 family alone and 0.89 with the strong path. The prespecified support criterion
  (R² ≥ 0.8) is met.
- **No solver saturation** (prespecified test). The local slope between the two largest batch-1 doses is 0.16, but D_H
  increases only 1.4× there, below the required 2×.
- **Post-hoc: k, not D_H, governs R_func** (`logs/posthoc_k_vs_DH.json`).
  - R_func ∝ **k^0.23** with R² = 0.988 (natural) and 0.985 (fixed). With k in the model, log D_H adds nothing
    (coefficient 0.004, R² unchanged).
  - Within a dose, D_H varies up to 10× while R_func barely moves; within-dose Spearman ranges from −0.62 to +0.47.
  - The four historical singletons span 50× in D_H (1.3e-5 to 6.5e-4) yet have R_func 0.143–0.157.
  - Interpretation: within the batch-1 family the solver responds to **how many calibration rows** are perturbed, not to
    the Frobenius size of the Hessian change. Any nonzero perturbation of a row flips a near-random set of borderline
    decisions.
  - Across families, amplitude does matter: at k = 64, the strong path gives R_func 0.665 vs 0.399 for batch 1.
- **Codes scale faster than function.** Over D_H, the log-log slope is 0.43 for integer-code disagreement vs 0.25 for
  R_func; relative Frobenius gives 0.20.
- **Per layer and projection (Fig. 7).** The middle layers dominate at every dose, with R_func_layer ordered
  L9 > L19 ≈ L27 > L1. At k = 1 the four layers give 0.35 / 0.24 / 0.14 / 0.06; at k = 96 they give
  0.59 / 0.43 / 0.45 / 0.15. Projection-level error is largest for L9 and L19 down_proj at every dose.

### Functional → PPL (Fig. 3)
- **Batch-1 family.** R_func grows 3× (0.15 → 0.44) but R_PPL stays **flat at about 0.03–0.05**. Every χ² CI overlaps
  (each spans about a factor of 3).
  - Natural policy: the constant model M0 has the best AIC, with M2 (√X) at +1.2, M3 (saturating) at +1.6 and M1
    (proportional) at +8.1.
  - Both policies together: M2 is best, M3 +0.6, M0 +5.6, M1 +15.7.
- **Strong path.** k = 64 natural reaches R_func 0.665 and R_PPL **0.156** [0.10, 0.32].
  - Across all families and policies, M3 has the best AIC, but only by 1.3 over M1 and 1.6 over M2.
  - The prespecified evaluation-saturation criterion (M3 better by ≥ 2 AIC and sufficient range) is **not met**.
- **Mean PPL shifts with dose.** Natural configurations average 0.002–0.008 PPL *below* ORIGINAL in the batch-1
  family. The ORIGINAL draw happens to sit high (consistent with the reference matrix below). The strong path lowers
  natural mean PPL by about 0.028.

### Act-order mitigation (Fig. 4, `actorder_mitigation.json`)
- **R_mitigation is constant** (RMS R_func, fixed / natural). It is 0.48–0.51 at every batch-1 dose (k = 1…96) and 0.45
  on the strong path. Every bootstrap CI excludes 1, and in all 56 replicate pairs fixed has the lower R_func.
- **Codes fall more than function:** the code ratio is 0.13–0.19 (strong path 0.08).
- **The PPL-SD ratio does not track it.** Fixed / natural SD(PPL) is 0.59, 0.72, 0.67, 0.59, 0.86 and **1.13** at
  k = 1, 4, 16, 32, 64, 96, and 0.33 on the strong path. It is not consistently below 1.
- **Pooled pre-specified PPL benefit.** (Δlog PPL_nat)² − (Δlog PPL_fix)², averaged over replicates × doses, is positive
  (CI [1.7e-7, 3.3e-7]): fixed configurations sit closer to ORIGINAL's PPL. This partly holds by construction, because
  they share ORIGINAL's permutation.
- **Quality cost.** Fixed costs +0.0046 PPL [0.0031, 0.0059] vs natural on average.

### Reference-permutation sensitivity (Fig. 6, `reference_order_matrix.json`)

Q_ij = GPTQ(H_j, π_i) for 5 historical Hessians × 5 historical permutations (G20261115 = ORIGINAL):

| reference π_i | mean PPL over H_j | SD over H_j | R_func vs own reference |
|---|---|---|---|
| G20261111 | 12.6189 | 0.0025 | 0.074 |
| G20261112 | **12.6168** | 0.0036 | 0.075 |
| G20261113 | 12.6230 | 0.0012 | 0.074 |
| G20261114 | 12.6187 | 0.0033 | 0.075 |
| G20261115 = ORIGINAL | **12.6265** (worst) | 0.0026 | 0.075 |

- **Between references:** the means range 0.0096 (SD of means 0.0039).
- **Within a reference:** the mean SD is 0.0026, while the mean SD across references at a fixed Hessian is 0.0043.
- **The ORIGINAL permutation was the worst-quality choice of the five.** The previous "+0.0076 PPL cost of fixed
  act-order" is largely a **reference-selection effect**. With π_G20261112 as reference, fixed act-order would have
  averaged *below* the natural mean (12.6188).
- Functional disagreement relative to the chosen reference is the same for every reference (0.074–0.075).

## Answers

1. **Does functional sensitivity rise monotonically with Hessian perturbation?** Yes, over configurations (ρ ≈ 0.9) and
   dose means. But within the batch-1 family the number of perturbed rows explains R_func almost completely
   (R² 0.99), and the Frobenius D_H explains it worse and adds nothing given k.
2. **Exponent.** R_func ∝ D_H^0.25 (95 % CI about 0.22–0.31), or equivalently ∝ k^0.23. The response is strongly
   concave: a single perturbed row already produces a third of the k = 128 functional difference.
3. **Codes vs function.** Code disagreement grows faster than function (slope 0.43 vs 0.25 in D_H) and rises from 1 %
   to 37 % across the range. Fixed act-order cuts codes 81–92 % but function only about 50 %. The two metrics diverge
   increasingly with dose.
4. **Does PPL variability rise with functional disagreement?** Not detectably within the batch-1 range (R_func
   0.15–0.44; R_PPL 0.03–0.05; constant model preferred). It is clearly larger at the strong path's R_func = 0.67
   (R_PPL 0.156). The overall relationship is consistent with a threshold or convex rise, but the data cannot
   distinguish the candidate forms.
5. **Convincing saturation?** No.
   - **Solver:** the top-dose D_H ratio is only 1.4×.
   - **Evaluation:** the saturating model wins by < 2 AIC.
   - The batch-1 PPL plateau is a flat region at *low* R_func (an insensitive regime), not saturation at high R_func.
6. **Does the dense experiment reach the MoE range (0.82)?** Nearly, with the strong path. Batch-1 reaches at most 0.46
   (k = 128); the strong path reaches 0.665 (k = 64) and 0.752 (k = 128), 91 % of the MoE value. The prespecified
   dynamic-range criterion (≥ 0.41) is met.
7. **Is the MoE point consistent with the dense relationship?** Descriptively yes.
   - The MoE point (R_func 0.82, R_PPL 0.144) lies next to the dense strong-path dose (0.665, 0.156 [0.10, 0.32]).
   - The dense power law reaches 0.82 at D_H ≈ 2e-2, the same order as the MoE trace-normalized Hessian perturbations
     (about 0.004–0.014).
   - Caveats: the architectures, quantized fractions, Q₀ (0.031 vs 0.101) and D_H definitions differ, and the
     strong-path perturbation is systematic. The MoE point is not a replicate of the dense process.
8. **Extrapolation needed.** 1.09× beyond the strong-path endpoint, or 1.78× beyond the batch-1 endpoint.
9. **Does fixed act-order consistently reduce functional instability?** Yes: about 50 % at every dose and in both
   perturbation families (R_mitigation 0.45–0.51; CI < 1 everywhere; 56/56 paired replicates).
10. **Does that carry over to PPL?** Not reliably.
    - The PPL-SD ratio ranges 0.33–1.13 across doses, and the χ² CIs are wide.
    - Mean PPL is 0.0046 higher than natural on average with the ORIGINAL reference.
    - The reference matrix shows the size and sign of that cost depend on which permutation is chosen.
11. **Sensitivity to the reference permutation.** Substantial. Across the five references the mean PPL ranges 0.0096,
    which is comparable to or larger than the within-reference grouping SD (0.0026) and the natural 5-grouping SD
    (0.0050).
    - Fixing act-order removes about half of the execution-induced functional variation. It adds a between-reference
      quality lottery of similar size.
    - This supports reference-selection sensitivity without proving a variance transfer.
12. **Replicate requirements.** Execution SD(PPL) is about 0.003–0.005 (R_PPL about 0.03–0.05) at *every* batch-1
    dose. So even a single-row execution difference produces about 70 % of the PPL spread seen at larger batch-1 doses (0.0033 vs about 0.0045).
    - With 8 replicates, a dose SD is uncertain by about a factor of 3, and SD ratios between methods cannot be ranked.
    - A paired design with 8 replicates resolves mean differences of about 0.003–0.005 PPL.
    - GPTQ method comparisons at the 0.01-PPL scale therefore need at least 5–8 execution replicates per arm, and
      variance comparisons need far more.
    - Fixed-reference designs must also vary the reference permutation.

## Conclusions

- **Perturbation → solver.** Real numerical execution changes (batch 1, SDPA) perturb the Hessian monotonically in k.
  The solver's response is extremely concave and is governed by how many rows are perturbed, not by the Frobenius size
  of the Hessian change. One row already yields R_func ≈ 0.15.
- **Solver → function.** R_func ∝ D_H^0.25 ≈ k^0.23, with middle layers (L9, L19) and their down_proj dominating.
  Integer-code disagreement exaggerates the dose dependence and the mitigation effect.
- **Function → PPL.** PPL spread is flat (≈ 0.04 Q₀) over R_func 0.15–0.44 and much larger at 0.67 (strong path). This
  is an insensitive low-R_func regime followed by a rise. Saturation is not established.
- **Act-order stabilization.**
  - Fixed reference act-order consistently halves functional disagreement at every dose.
  - PPL-SD reduction is inconsistent across doses.
  - It trades execution sensitivity for reference-selection sensitivity: mean PPL depends on the chosen permutation by
    up to 0.01.
- **Reference-selection bias.** The historical ORIGINAL permutation was the worst of five. The previously reported fixed
  act-order quality cost is mostly this choice, not an intrinsic cost.
- **Dense vs MoE.** The dense strong path reaches 91 % of the MoE functional ratio, with a PPL spread (R_PPL 0.156)
  matching the MoE's (0.144). The data are consistent with MoE grouping sensitivity being the same GPTQ mechanism driven
  by a broader and larger calibration perturbation, where nearly every row and all layers are perturbed.
  - This is a descriptive consistency, not a proof.
  - The model architectures, the quantized fraction (10.7 % dense vs routed experts) and the perturbation sources
    (kernel path vs routing changes) differ.

## Deviations and limitations

- **Audit fix before the freeze.** The audit's reference-permutation check first ran argsort on the CPU, where
  down_proj near-ties sort differently. It was rerun on the solver's device (GPU) and passes.
- **Pilot criteria gap.** The criteria did not require PPL equivalence. SDPA was selected under the frozen rule and then
  found to shift the BF16 PPL by −2.2 %. This is recorded as a post-pilot diagnostic, and the family is labelled
  separately.
- **Code changed after the freeze.**
  - **What changed:** `diagnostics/dense_dose_response.py` gained an optional `wref` argument in `functional_one`, so
    each reference-matrix model could also be compared against its own reference's model.
  - **What it touched:** only the `func` and `refmatrix` stages; the `refmatrix` logic itself is unchanged. The `gpu`
    stage ran on the frozen code. All drift is logged in `logs/code_state.json`.
- **Post-hoc analyses** (labelled as such, not in any criterion): the k-vs-D_H regression and the SDPA equivalence
  check.
- **MoE reference.** The R_func denominator uses the mean quantization error over the 6 MoE configs; the dense
  denominator uses ORIGINAL only.
- **Not run:** a 5 × 5 functional matrix of each permutation's effect on its own Hessian beyond R_func vs ORIGINAL and
  vs the own reference.
- **Statistics.** Eight replicates give wide SD intervals; replicate trajectories are the bootstrap unit. k = 128
  endpoints have no SD.
- **Cost.**
  - GPU stage 1.76 h (median 88 s per configuration); reference matrix 15 min; functional stage 60 min.
  - The cache is 165 GB (`~/moe_cache_dense_dose_response/`), mostly per-configuration Hessians.

## Files

- **Code:**
  - `diagnostics/dense_dose_response.py`
  - `diagnostics/report_dense_dose_response.py`
  - `diagnostics/dense_dose_posthoc.py`
  - reuses `moe_calib/actorder_policy.py` and `moe_calib/functional_diff.py`
- **Deliverables:** all required JSONs in `runs_dense_dose_response/`, plus figures 1–7 and `logs/`.
- **Raw results:** per-window NLL and installed-weight hashes for every model are in `dose_ppl_results.json` and
  `results/`.
