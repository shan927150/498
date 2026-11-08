# Partial-model WikiText-2 perplexity — summary

**PARTIAL model:** only the 240 routed experts (gate/up/down) of layers 1/8/16/23 are GPTQ-quantized (4-bit, g128, percdamp 1.0, act-order; fake-quantized weights in BF16); attention, routers, shared experts, embeddings, norms, lm_head and the other 20 layers are BF16. Not a fully 4-bit model.

**Decision (rule frozen in evaluation_protocol.json): PPL-RECON-ALIGNED**

Valid evaluations: 12 / 12

BF16 PPL: **9.2371** (128 windows × 511 scored targets = 65,408 targets)

## PPL by strategy and seed

| strategy | s0 | s1 | s2 | mean | SD | Δ vs BF16 | rel. degradation | recon (test) mean | recon (confirmation) mean |
|---|---|---|---|---|---|---|---|---|---|
| Full Random | 9.2679 | 9.2747 | 9.2656 | 9.2694 | 0.0047 | +0.0323 | +0.350 % | 0.004615 | 0.004580 |
| Typical Random | 9.2680 | 9.2747 | 9.2623 | 9.2683 | 0.0062 | +0.0312 | +0.338 % | 0.004606 | 0.004559 |
| Full Coverage-LP | 9.2779 | 9.2774 | 9.2787 | 9.2780 | 0.0007 | +0.0409 | +0.443 % | 0.004730 | 0.004708 |
| Typical Coverage-LP | 9.2725 | 9.2775 | 9.2675 | 9.2725 | 0.0050 | +0.0354 | +0.383 % | 0.004613 | 0.004562 |

## Paired differences (same seed; > 0 = first method worse)

| pair | ΔPPL s0 / s1 / s2 | mean | rel. ΔPPL | share of the baseline's degradation | 95 % window bootstrap per seed | Δrecon (rel.) s0 / s1 / s2 |
|---|---|---|---|---|---|---|
| Full Coverage-LP − Full Random | +0.0101 / +0.0027 / +0.0131 | +0.0086 | +0.109 % / +0.029 % / +0.141 % | +32.7 % / +7.2 % / +45.8 % | [+0.0015, +0.0186]; [-0.0065, +0.0116]; [+0.0039, +0.0215] | +2.55 % / +2.19 % / +2.71 % |
| Typical Coverage-LP − Typical Random | +0.0045 / +0.0028 / +0.0052 | +0.0042 | +0.049 % / +0.030 % / +0.057 % | +14.5 % / +7.4 % / +20.8 % | [-0.0035, +0.0126]; [-0.0058, +0.0114]; [-0.0031, +0.0140] | +0.09 % / +0.21 % / +0.21 % |
| Typical Random − Full Random | +0.0002 / +0.0000 / -0.0033 | -0.0010 | +0.002 % / +0.000 % / -0.036 % | +0.6 % / +0.0 % / -11.7 % | [-0.0077, +0.0078]; [-0.0098, +0.0095]; [-0.0127, +0.0048] | +0.05 % / -0.57 % / -0.12 % |
| Full Coverage-LP − Typical Random | +0.0099 / +0.0027 / +0.0164 | +0.0097 | +0.107 % / +0.029 % / +0.177 % | +31.9 % / +7.2 % / +65.1 % | [+0.0009, +0.0191]; [-0.0061, +0.0121]; [+0.0071, +0.0260] | +2.50 % / +2.77 % / +2.83 % |

## Reconstruction vs PPL (12 method × seed configurations; descriptive, not independent replicates)

- Spearman / Pearson with old-test reconstruction: +0.517 / +0.659
- Spearman / Pearson with locked-confirmation reconstruction: +0.615 / +0.674
- Ranking of means (best → worst): PPL ['TR', 'FR', 'TC', 'FC']; recon test ['TR', 'TC', 'FR', 'FC']; recon confirmation ['TR', 'TC', 'FR', 'FC']

## Numerical reliability

- BF16 repeated in one process: |ΔPPL| = 0.00e+00; across processes: 0.00e+00; after all 13 quantize/restore cycles: 0.0
- Full Random s0 re-quantized and re-evaluated: |ΔPPL| = 0.0; identical installed weights: True
- τ = max(3 × max numerical difference, 1e-4 × PPL_BF16) = 0.00092
- Independent HF forward (accelerate offload, labels=) window 0: mean NLL 1.763398 vs layer-wise 1.764284 (rel. diff 5.0e-04)
- Independent HF forward (accelerate offload, labels=) window 1: mean NLL 2.385790 vs layer-wise 2.383898 (rel. diff 7.9e-04)

## Installation checks (all 12 configurations)

- E3 per-expert reconstruction recomputed from the installed run vs the frozen run: max rel. diff 0.0e+00
- 8 check experts × 12: installed BF16 weights bitwise = BF16 cast of E3's W_q: True; installed-module output vs E3 FP32 output: max rel. 4.84e-03 = at most 15.3 % of the quantization perturbation itself (median 0.126); of which weight BF16 rounding alone ≤ 3.0e-03; BF16-compute noise with original weights ≤ 3.7e-03; largest deviation / amended tolerance = 0.95 (tolerance = weight-rounding deviation + 1.5 × BF16-compute deviation; amended after smoke attempt 1, before any quantized PPL — logs/amendment_install_tolerance.json)
- Untouched parameters/buffers changed: 0 (of 3964 checked per run); RTN fallbacks: 0; distinct installed weight sets: 12 / 12

## Compute (mean per run)

| strategy | Hessian replay s | GPTQ s | quantize wall s (incl. E3 scoring and checks) | PPL eval s | peak GPU GB | peak CPU GB (process) | routing profile s |
|---|---|---|---|---|---|---|---|
| Full Random | 32 | 206 | 268 | 28 | 8.4 | 68.3 | 0 |
| Typical Random | 33 | 204 | 268 | 27 | 8.4 | 68.4 | 357 |
| Full Coverage-LP | 32 | 204 | 264 | 22 | 8.4 | 68.4 | 357 |
| Typical Coverage-LP | 31 | 206 | 266 | 22 | 8.4 | 68.5 | 357 |

BF16: model load 8 s, PPL eval 44 s, peak GPU 3.3 GB, process peak CPU 40.0 GB. Packed 4-bit artifact for 240 experts would be ≈ 1.10 GB (not produced; weights are fake-quantized).

Figures: `figures/fig1_ppl_by_strategy`, `fig2_ppl_degradation`, `fig3_recon_vs_ppl`, `fig4_paired_differences` (PDF + PNG).
