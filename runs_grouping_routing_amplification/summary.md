# Family D grouping: frozen-routing causal intervention

**Verdict: WEAK-COLLAPSE (B)** - R_SD = 0.915 (pooled within-seed SD of FC-FR, frozen / natural); R1 reversals 6/15 -> 8/15; protocol sha256 ca444093eb0bf1a7.

Freezing every quantized variant's routing to the BF16 reference barely changes the grouping-induced spread of the FR-FC difference: pooled within-seed SD(Δ) goes 0.006872 -> 0.006286 (R_SD 0.915, an 8.5 % reduction; seed-0-only ratio 1.057, i.e. no reduction), R1 reversals go 6/15 -> 8/15, and 3/15 comparisons change sign. Unlike Family A (accumulation order, R_SD 0.359), the calibration-grouping effect is therefore **not** explained by inference-time routing amplification; it persists when routing is held fixed, so it travels through the quantized expert weights themselves. This fits the weight audit: grouping changes ~55 % of BF16 effective weight elements (rel. Frobenius ~0.09), about 7,000x more elements and ~100x larger rel. Frobenius than an accumulation-order change (0.0074 %, 8.4e-04).

## 1. Artifact and parity audit

- 30/30 weight sets valid; duplicates: False; missing: none; historical Δ table reproduced from stored results for 15/15 pairs: True.
- References: seed-0 ORIGINAL reused from Family A; seed-1/2 ORIGINAL-extraction weights were never saved (runs_partial_ppl quantized in-process) -> frozen ORIGINAL Δ unavailable for seeds 1/2.
- Parity (all exact):
  - A_natural_parity_all_bitwise: True
  - B_record_only: True
  - C_self_forcing: True
  - D_bf16_self_forcing: True
  - E_frozen_dispatch: True
  - F_restoration: True
- End-of-process: non-target params unchanged True, BF16 bitwise True, no hooks True; GPTQ solves 0.

## 2. Primary paired PPL table

| comparison | FR nat | FC nat | FR frozen | FC frozen | Δ nat | Δ frozen | δ_g | CI nat | CI frozen |
|---|---|---|---|---|---|---|---|---|---|
| s0_G20261111 | 9.278182 | 9.274113 | 9.274681 | 9.275402 | -0.004069 | +0.000721 | -0.004790 | – | – |
| s0_G20261112 | 9.268159 | 9.275760 | 9.267316 | 9.276877 | +0.007601 | +0.009561 | -0.001960 | – | FR |
| s0_G20261113 | 9.271035 | 9.265146 | 9.272412 | 9.266228 | -0.005889 | -0.006183 | +0.000295 | – | – |
| s0_G20261114 | 9.278102 | 9.269707 | 9.276955 | 9.269850 | -0.008394 | -0.007106 | -0.001289 | FC | FC |
| s0_G20261115 | 9.272717 | 9.273808 | 9.271482 | 9.272560 | +0.001091 | +0.001078 | +0.000013 | – | – |
| s1_G20261111 | 9.266099 | 9.265572 | 9.268815 | 9.267153 | -0.000527 | -0.001662 | +0.001135 | – | – |
| s1_G20261112 | 9.267534 | 9.263857 | 9.265103 | 9.266850 | -0.003677 | +0.001747 | -0.005424 | – | – |
| s1_G20261113 | 9.265318 | 9.268279 | 9.262063 | 9.270033 | +0.002961 | +0.007970 | -0.005009 | – | FR |
| s1_G20261114 | 9.267026 | 9.275531 | 9.268506 | 9.277161 | +0.008505 | +0.008655 | -0.000150 | FR | FR |
| s1_G20261115 | 9.273307 | 9.269481 | 9.272534 | 9.266885 | -0.003826 | -0.005650 | +0.001824 | – | – |
| s2_G20261111 | 9.262634 | 9.272568 | 9.263155 | 9.270934 | +0.009934 | +0.007779 | +0.002155 | FR | FR |
| s2_G20261112 | 9.268870 | 9.269566 | 9.266980 | 9.266631 | +0.000696 | -0.000349 | +0.001045 | – | – |
| s2_G20261113 | 9.282398 | 9.268849 | 9.276153 | 9.267183 | -0.013549 | -0.008970 | -0.004578 | FC | FC |
| s2_G20261114 | 9.271457 | 9.266417 | 9.267137 | 9.265977 | -0.005041 | -0.001160 | -0.003880 | – | – |
| s2_G20261115 | 9.272849 | 9.268318 | 9.271092 | 9.269454 | -0.004531 | -0.001638 | -0.002893 | – | – |

seed-0 ORIGINAL (reference, not in SDs): Δ natural +0.010057, frozen +0.008739.

## 3. Variability and mean preservation

| statistic | natural | frozen |
|---|---|---|
| pooled within-seed SD(Δ) | 0.006872 (historical 0.006872) | 0.006286 |
| seed 0 SD(Δ) | 0.006364 | 0.006728 (ratio 1.057) |
| seed 1 SD(Δ) | 0.005176 | 0.006158 (ratio 1.190) |
| seed 2 SD(Δ) | 0.008624 | 0.005945 (ratio 0.689) |
| flat SD over 15 (descriptive) | 0.006522 | 0.005986 |
| pooled within-seed SD PPL FR | 0.005198 | 0.004214 |
| pooled within-seed SD PPL FC | 0.003812 | 0.003754 |
| mean Δ (mean of seed means) | -0.001248 | +0.000319 |
| seed means | -0.00193, +0.00069, -0.00250 | -0.00039, +0.00221, -0.00087 |
| range Δ | -0.013549 .. +0.009934 | -0.008970 .. +0.009561 |
| FR better / FC better | 6/9 | 7/8 |
| hierarchical bootstrap 95% | [-0.00478, +0.00215] | [-0.00283, +0.00361] |
| historical decision rule | GROUPING-VARIATION-DOMINATES | GROUPING-VARIATION-DOMINATES |

**R_SD = 0.915** (8.5 % reduction); seed-0-only ratio 1.057; change in mean Δ +0.001567; δ_g mean -0.001567, SD 0.002686.

## 4. Ranking reversals and window CIs

- R1 (historical decision rule: sign opposite to overall mean M): natural 6/15 (M -0.001248), frozen 8/15 (M +0.000319); natural R1 reversals still opposite natural M after freezing: ['s0_G20261112', 's0_G20261115', 's1_G20261113', 's1_G20261114', 's2_G20261111'].
- R2 (vs same seed's ORIGINAL Δ sign, fp64 mechanism table): natural 9/15, frozen 8/15 (seed 0 fully frozen: 3/5 -> 2/5; seeds 1/2 frozen vs natural ORIGINAL sign, mixed-regime). Remaining: ['s0_G20261113', 's0_G20261114', 's1_G20261111', 's1_G20261115', 's2_G20261113', 's2_G20261114', 's2_G20261115'].
- Comparisons whose sign flips under freezing: 3/15.
- Window CIs excluding zero: natural FR 2, FC 2; frozen FR 4, FC 2; mean width 0.001904 -> 0.001593 (mean-NLL units). Window CIs condition on one weight pair; they measure text-sampling uncertainty, not grouping variability.

## 5. Routing divergence

- vs BF16: mean 8.92 % of (layer, token) sets differ; layers 0-1 identical in all 30: True.
- Family D seed-0 grouping vs ORIGINAL: 6.09 % (Family A order vs ORIGINAL 3.67 %).
- Family D grouping vs grouping within seed: 6.10 % (Family A order vs order 3.68 %); order-only changes 6.87 % (excluded).
- BF16 exact-tie rate (reference file) 4.88 %; share of D s0-vs-ORIGINAL flips on exact ORIGINAL ties 27.4 % (Family A 35.1 %); flip rate on exact ties 34.0 % vs 26.3 %.
- Margin-decile flip rate (1st..10th), D: 27.5, 18.8, 8.3, 3.8, 1.7, 0.7, 0.2, 0.1, 0.0, 0.0; A: 20.0, 11.3, 3.4, 1.3, 0.6, 0.2, 0.1, 0.0, 0.0, 0.0 (%).
- Descriptive: Spearman(|δ_g|, BF16 disagreement) = 0.37857142857142856.

## 6. Weight differences

- Seed-0 grouping vs ORIGINAL changed BF16 elements: FR_s0_G20261111 54.1 %, FR_s0_G20261112 54.1 %, FR_s0_G20261113 54.2 %, FR_s0_G20261114 54.0 %, FR_s0_G20261115 54.0 %, FC_s0_G20261111 55.1 %, FC_s0_G20261112 55.2 %, FC_s0_G20261113 55.2 %, FC_s0_G20261114 55.1 %, FC_s0_G20261115 55.2 %
- Within-seed grouping pairs: mean 54.8 % (range 53.8-55.8 %), mean rel. Frobenius 0.0898.
- Family A (same function): FR_O20261121 0.0072 % / rel Frob 8.16e-04, FC_O20261121 0.0076 % / rel Frob 8.56e-04
- ULP distance of changed elements (within-seed mean): 1: 34.7 %, 2: 11.5 %, 3-7: 14.8 %, >=8: 37.7 %, sign_change: 1.3 %
- Descriptive over 60 within-seed pairs: Spearman(rel Frob, |ΔPPL| frozen) = -0.2230619616560156, natural = -0.18205056960266744; mean |ΔPPL| between two variants 0.00515 natural -> 0.00469 frozen.

## 7. Family A vs Family D

| metric | Family A (accumulation order) | Family D (grouping) |
|---|---|---|
| BF16 weight elements changed vs ORIGINAL | 0.0074 % | 54.6 % (seed 0) |
| rel. Frobenius vs ORIGINAL | 8.36e-04 | 0.0897 |
| natural paired SD | 0.002870 (5 orders, 1 seed) | 0.006872 pooled within-seed; seed 0 0.006364 |
| frozen paired SD | 0.001032 | 0.006286; seed 0 0.006728 |
| R_SD | 0.359 | 0.915 (seed 0 only 1.057) |
| sign reversals vs ORIGINAL sign (natural) | 0/5 | 9/15 (seed 0: 3/5) |
| sign reversals vs ORIGINAL sign (frozen) | 0/5 | 8/15 (seed 0: 2/5) |
| routing disagreement vs ORIGINAL | 3.67 % | 6.09 % (seed 0) |
| routing disagreement variant vs variant | 3.68 % | 6.10 % |
| mean Δ natural -> frozen | +0.00803 -> +0.00822 | -0.00125 -> +0.00032 |

## 8. Answers and limitations

**Does frozen routing collapse grouping variability?** No (WEAK-COLLAPSE by the prespecified rule). The SD reduction is small and inconsistent across seeds (per-seed frozen/natural ratios 1.057, 1.190, 0.689), and reversals do not decrease.

**Does the FR-vs-FC conclusion change under frozen routing?** No. The mean Δ stays near zero (natural -0.00125, frozen +0.00032), both hierarchical-bootstrap 95 % intervals contain zero, and FR-better/FC-better counts stay balanced (6/9 natural, 7/8 frozen). Under grouping variation the FR/FC ranking remains undetermined whether or not routing is fixed. The seed-0 ORIGINAL pair (+0.0101 natural, +0.0087 frozen) is one draw from this spread, not a stable effect.

**Where does the variability come from?** Mainly from the weights. Grouping perturbs routing more than accumulation order does (6.10 % vs 3.68 % of (layer, token) expert sets differ between variants, concentrated on low-margin and exact-tie tokens, as in Family A), but removing that divergence leaves most of the PPL spread in place. Pairwise weight-difference magnitude does not predict |ΔPPL| (Spearman -0.18 natural, -0.22 frozen over 60 pairs). Rel. Frobenius varies little across pairs, so this test has little leverage and is descriptive only.

**Limitations.**
- 3 allocation seeds x 5 groupings, with groupings nested in seeds. With n=5 per seed the SD estimates are noisy, and R_SD is a descriptive screening ratio, not a significance test.
- Frozen routing forces the BF16 expert sets but keeps each model's own gate probabilities (p_forced differs from BF16 in ~92 % of entries), so it removes selection divergence only, not gate-weight divergence. It is also an intervention the deployed model never runs.
- Only 4 quantized layers (1, 8, 16, 23), 4-bit GPTQ g128 with act-order, one model (Qwen1.5-MoE-A2.7B), and one WikiText-2 partial-model PPL protocol. Effect sizes are ~1e-3 to 1e-2 PPL on ~9.27, where window CIs exclude zero for only 2-4 of 15 per strategy.
- Seed-1/2 ORIGINAL-extraction weights were never stored, so R2 for those seeds compares frozen Δ against a natural-only reference (mixed regime). Seed-0 R2 is fully frozen (3/5 -> 2/5).

## Cost

median natural eval 21.2 s, frozen 21.3 s, install 28 s; peak GPU 3.32 GB; storage 3.43 GB; GPTQ solves 0.

