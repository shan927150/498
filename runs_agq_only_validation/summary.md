# AGQ-only validation: standard GPTQ vs AGQ-GPTQ on matched C4 Random B=128

**Primary verdict (frozen rule): `AGQ-LAYER-QUALITY-GAIN`**

Seeds measured: FR_s0, FR_s1, FR_s2. Gates: source_review=True, routing_definition=True, capture_alignment=True, hessian_oracle=True, unit_regression=True, diagnostics_recompute_equals_installed=True, AGQ_FR_s0_valid=True, AGQ_FR_s1_valid=True, AGQ_FR_s2_valid=True

## FR_s0

| Metric | GPTQ | AGQ | Relative difference |
|---|---|---|---|
| Standard expert reconstruction | 0.00460947 | 0.00468294 | +1.594 % |
| Weighted expert reconstruction | 0.0147503 | 0.0147639 | +0.092 % |
| MoE layer output reconstruction | 0.00371661 | 0.00366068 | -1.505 % |
| MoE layer output (native BF16 block) | 0.00373401 | 0.0036773 | -1.519 % |
| Partial-model WikiText-2 PPL | 9.269 | 9.27556 | +0.071 % |
| Quantization wall time (s) | 346.817 | 347.377 | +0.162 % |
| Peak GPU memory (GB) | 8.41772 | 8.41781 | +0.001 % |

Per-layer MoE output error (fp32, frozen routing): L1 -1.831 %, L8 +0.066 %, L16 -3.093 %, L23 -1.209 %
Experts improved (standard / weighted): 3/240 / 91/240

## FR_s1

| Metric | GPTQ | AGQ | Relative difference |
|---|---|---|---|
| Standard expert reconstruction | 0.00462593 | 0.00470299 | +1.666 % |
| Weighted expert reconstruction | 0.0147835 | 0.0148202 | +0.248 % |
| MoE layer output reconstruction | 0.00371337 | 0.00367506 | -1.032 % |
| MoE layer output (native BF16 block) | 0.00373077 | 0.00369194 | -1.041 % |
| Partial-model WikiText-2 PPL | 9.27257 | 9.25761 | -0.161 % |
| Quantization wall time (s) | 346.817 | 1194.78 | +244.498 % |
| Peak GPU memory (GB) | 8.41772 | 8.41781 | +0.001 % |

Per-layer MoE output error (fp32, frozen routing): L1 -2.110 %, L8 +0.227 %, L16 -2.269 %, L23 -0.862 %
Experts improved (standard / weighted): 3/240 / 80/240

## FR_s2

| Metric | GPTQ | AGQ | Relative difference |
|---|---|---|---|
| Standard expert reconstruction | 0.0046104 | 0.00468796 | +1.682 % |
| Weighted expert reconstruction | 0.0147498 | 0.01478 | +0.204 % |
| MoE layer output reconstruction | 0.00371125 | 0.00366823 | -1.159 % |
| MoE layer output (native BF16 block) | 0.00372857 | 0.0036854 | -1.158 % |
| Partial-model WikiText-2 PPL | 9.27469 | 9.26917 | -0.060 % |
| Quantization wall time (s) | 346.817 | 679.608 | +95.956 % |
| Peak GPU memory (GB) | 8.41772 | 8.41781 | +0.001 % |

Per-layer MoE output error (fp32, frozen routing): L1 -4.443 %, L8 +0.217 %, L16 -2.416 %, L23 -1.131 %
Experts improved (standard / weighted): 2/240 / 77/240

