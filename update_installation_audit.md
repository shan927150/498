# Installation audit: the FC AO G20261114 BF16 tolerance exceedance

**Artifacts:** `runs_actorder_ablation/installation_audit.json` (stage `audit` of `scripts/run_actorder_ablation.sh`).
**Decision: AUDIT-CONFIRMED-BF16-REPRESENTATION.**

The frozen-routing robustness study's **RUN-BLOCKED** decision is **unchanged**, and so are the frozen tolerance and its failed result. This audit only establishes the correctness basis for the act-order ablation.

## What failed

In FC AO G20261114, the installation-output check for layer 1 / expert 0 gave `installed_vs_e3_rel` = 0.005595, above its tolerance of 0.005202 (1.075×). The check is computed as follows.

The check runs on the frozen validation inputs X: the routed MoE-block inputs of expert (L, e) for the first 4 test sequences of `~/moe_cache`.
- **`installed_vs_e3_rel`** = ‖y_inst − y_e3‖ / ‖y_e3‖, where:
  - y_e3 is the FP32 expert MLP with E3's FP32 solver weights W_q;
  - y_inst is the installed BF16 module on BF16 inputs.
- **tolerance** = `weight_rounding_only_rel` + 1.5 × `bf16_compute_noise_rel`, where:
  - `weight_rounding_only_rel` is the FP32 MLP with BF16-cast W_q, compared against y_e3;
  - `bf16_compute_noise_rel` is BF16 versus FP32 execution measured with the *original* weights.

## The solver and installation path (as implemented)

**The solver:** `moe_calib/gptq.py::gptq_quantize`.
1. **Dead columns:** a column with diag(H) = 0 gets H_jj = 1 and W[:, j] = 0.
2. **Act-order:** perm = argsort(diag H, descending); W and H are permuted.
3. **Damping:** H += percdamp · mean(diag H) · I.
4. **Inverse factor:** the upper Cholesky factor of H⁻¹.
5. **Column loop** in lazy 128-column blocks:
   - At every permuted column index divisible by 128, the asymmetric per-row scale and zero are fitted from the *current, error-feedback-updated* group columns (min/max, 4-bit).
   - Each column is rounded.
   - The error (w − dq)/Hinv_jj is propagated to the remaining columns.
6. **Un-permute:** Q = Q[:, argsort(perm)]. The result is FP32 dequantized weights.

**The call chain:**
- `quantize_expert`: gate and up share H_gu; down uses H_down.
- E3 `quantize_from_acc_or_fallback`.
- `MU.set_expert_weights`: `param.copy_(w.to(bf16))`, which rounds to nearest.

## Audit design

- **Cases:**
  - Every configuration that stores full Hessians and installed weights for the 8 prespecified check experts: 24 configurations, 192 expert cases. These are the factorial OO/AO/OA/AA, the four new AO_g and the four AA_g, for FR and FC.
  - The historical worst case: partial-PPL FC s2, layer 1 / expert 0, at 0.954× tolerance. Its weights were never stored, so I validated it by reproducing its stored check floats.
- **Recomputing the solver weights:** for each case, W_q (FP32) was recomputed from the stored Hessian with the unchanged solver and compared bitwise against the stored installed weights.
- **Decomposition:** done on identical, exactly-BF16 inputs:
  - weight representation: Y_cast − Y_solver;
  - BF16 arithmetic: Y_bf16 − Y_cast;
  - total.

  This was done per projection (gate and up on X; down on the actual BF16 intermediate) and for the whole expert.
- **Independent installation reference:** F.linear(gate), F.linear(up), act, mul, F.linear(down), all in BF16, built from bfloat16(W_solver) rather than read back from the module.

## Findings

| check | result |
|---|---|
| validation inputs exactly representable in BF16 | 193/193 |
| recomputed solver W_q → BF16 == stored installed weights (bitwise) | 192/192 (historical: weights not stored) |
| installed parameter == bfloat16(W_solver) (bitwise) | 193/193 |
| \|W_bf16 − W\| ≤ 2⁻⁸\|W\| elementwise (correct rounding) | 193/193 |
| installed module output == independent BF16 reference (bitwise) | 193/193 |
| stored check floats reproduced exactly (same inputs, same references) | 193/193 |
| frozen-tolerance failures | 1 (FC AO G20261114 L1 e0; next highest 0.954) |

**Decomposition of the failing case** (relative to ‖y_solver‖):
- weight representation 0.00125;
- BF16 arithmetic 0.00494;
- total 0.00559.

The arithmetic deviation is **1.88×** the original-weight BF16 noise (0.00263) that the 1.5× margin was calibrated on. By the triangle inequality, total ≤ representation + arithmetic. A tolerance exceedance therefore *requires* arithmetic > 1.5 × original-weight noise, and that is exactly what happened.

**Why this expert:**
- Per projection the arithmetic deviation is ordinary: 0.0017 (gate), 0.0017 (up), 0.0020 (down).
- It grows through the BF16 SiLU·mul intermediate.
- Five tokens with large outputs (max |Δ| = 1.55) carry **97 %** of the squared error.

**Across all 193 cases:**
- The quantized-to-original arithmetic ratio ranges from 0.58 to 1.88 (median 1.01).
- The old check's ratio correlates 0.94 with this arithmetic ratio, and −0.43 with the quantization effect.

In other words, the old tolerance mostly measures how BF16 arithmetic noise happens to differ between the quantized and the original weights on a handful of outlier tokens. It does not measure installation correctness.

## Negative controls (on the failing expert)

| corruption | frozen tolerance check | new integrity check |
|---|---|---|
| expert 30's weights installed in expert 0 | **passes** (0.996×) | detected (I1, I2 fail) |
| one element changed by one BF16 ulp | unchanged (1.075×) | detected (I1, I2 fail) |
| truncating instead of round-to-nearest cast | **passes** (0.75×) | detected (I1, I3 fail) |

## Precision-aware installation integrity (new; act-order study only)

The protocol was frozen in `runs_actorder_ablation/protocol.json` before any AOF configuration ran.

| check | definition | justification |
|---|---|---|
| I1 | every one of the 720 installed tensors is bitwise equal to bfloat16(W_solver), checked at write-back | the only correct installation of an FP32 solver output into a BF16 model |
| I2 | for the 8 check experts, installed module output == independent BF16 reference (bitwise) | catches wrong placement, wrong expert or a wrong execution path; 193/193 empirically |
| I3 | \|W_bf16 − W_solver\| ≤ 2⁻⁸\|W_solver\| elementwise | the IEEE round-to-nearest bound for bfloat16 (8 significand bits) |
| I4 | finite weights and outputs | |

**What these checks replace and how strict they are:**
- The relative FP32-versus-BF16 output deviation is still computed and reported, but it no longer gates. It measures the BF16 execution environment, which is identical for every configuration and every condition.
- All four checks are exact. Nothing was widened: I did not loosen any tolerance.
- The frozen tolerance check is still computed for every new configuration and reported next to the new one.
