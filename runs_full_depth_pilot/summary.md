# Full-depth routed-expert GPTQ pilot — summary (seed 0)

Routed experts (gate/up/down) of **all 24 MoE layers** (1,440 experts, 4,320 projections) GPTQ-quantized (4-bit, g128, percdamp 1.0, act-order; fake-quantized weights in BF16); attention, routers, shared experts, embeddings, norms, lm_head BF16. Not fully 4-bit LLM quantization. Calibration non-sequential on BF16 inputs; evaluation a sequential 24-layer forward on the frozen 128 × 512 WikiText-2 windows.

**Decision (rule frozen in protocol.json): FULL-DEPTH-PENALTY-NOT-OBSERVED**

| configuration | PPL | mean NLL | Δ vs BF16 | rel. degradation | mean expert recon |
|---|---|---|---|---|---|
| BF16 | 9.237090 | 2.223227 | — | — | — |
| Full Random s0, 4 layers | 9.267851 | 2.226552 | +0.0308 | +0.333 % | 0.004610 |
| Full Coverage-LP s0, 4 layers | 9.277908 | 2.227636 | +0.0408 | +0.442 % | 0.004727 |
| Full Random s0, 24 layers | 9.364730 | 2.236951 | +0.1276 | +1.382 % | 0.005061 (layer mean) |
| Full Coverage-LP s0, 24 layers | 9.370674 | 2.237585 | +0.1336 | +1.446 % | 0.005192 (layer mean) |

## Coverage penalty

- Δ_4 = PPL(FC4) − PPL(FR4) = +0.010057 (+0.109 %); ΔNLL +0.001085; paired window bootstrap 95 % CI of ΔNLL [+0.000202, +0.002004]
- Δ_24 = PPL(FC24) − PPL(FR24) = +0.005944 (+0.063 %); ΔNLL +0.000634; 95 % CI of ΔNLL [-0.001578, +0.002778]
- Δ_24 − Δ_4 = -0.004113; Δ_24 / Δ_4 = 0.59; τ = 0.000924
- Windows where FC24 is worse: 69/128; per-window ΔNLL median +0.00086, 5–95 % [-0.02159, +0.01907]; correlation with the 4-layer per-window differences +0.23
- Expert reconstruction (test rows): FC worse than FR in 24/24 layers; layer-mean relative difference +2.61 %

## Calibration-numerics sensitivity (four layers; from the failed v1-cache regression)

Same selections, solver and settings; BF16 calibration inputs differing only by batch-composition rounding:
- Full Random s0: PPL 9.267851 → 9.272002 (+0.004150); recon 0.0046095 → 0.0046097
- Full Coverage-LP s0: PPL 9.277908 → 9.268026 (-0.009882); recon 0.0047270 → 0.0047271
- Δ_4: +0.010057 → -0.003975. PPL shifts up to 0.0099 from numerically equivalent calibration inputs — the same order as Δ_4; the window bootstrap does not capture this source of variation.

## Correctness

- Full Random s0: 24 layers, 1440 experts, 4320 projections; unchanged from BF16 0; non-finite weights 0; untouched params unchanged True; RTN fallbacks 0; finite NLL True; repeated evaluation |ΔPPL| 0.0e+00; expert checks pass True (max deviation / tolerance 0.56); weights sha256 98fb44e0a716a16f
- Full Coverage-LP s0: 24 layers, 1440 experts, 4320 projections; unchanged from BF16 0; non-finite weights 0; untouched params unchanged True; RTN fallbacks 0; finite NLL True; repeated evaluation |ΔPPL| 0.0e+00; expert checks pass True (max deviation / tolerance 0.56); weights sha256 f6358bbf581668bb
- Four-layer regression: original cache reproduces the partial-PPL weights and PPL exactly: True; new cache within tolerance: True (Δ_4 with new cache +0.010057)
  - cache L1: 345/345 rows bitwise equal to ~/moe_cache; max abs diff 0; max rel. (Frobenius) 0.00e+00
  - cache L8: 345/345 rows bitwise equal to ~/moe_cache; max abs diff 0; max rel. (Frobenius) 0.00e+00
  - cache L16: 345/345 rows bitwise equal to ~/moe_cache; max abs diff 0; max rel. (Frobenius) 0.00e+00
  - cache L23: 345/345 rows bitwise equal to ~/moe_cache; max abs diff 0; max rel. (Frobenius) 0.00e+00

## Cost

| strategy | quantize (min) | Hessian (min) | GPTQ (min) | PPL eval (s) | peak GPU GB | peak CPU GB | weight artifacts GB |
|---|---|---|---|---|---|---|---|
| Full Random s0 | 31.8 | 3.0 | 20.8 | 23 | 8.4 | 37.8 | 24.9 |
| Full Coverage-LP s0 | 32.3 | 3.1 | 21.2 | 23 | 8.4 | 37.2 | 24.9 |

BF16 calibration cache: 345 rows × 24 layers, 17.4 GB, 6.2 min. Full Coverage-LP additionally needs the ~357 s routing profile.

Figures: `figures/fig1_ppl_by_depth`, `fig2_coverage_penalty_by_depth`, `fig3_window_nll_differences`, `fig4_runtime_memory` (PDF + PNG).
