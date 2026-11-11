# Official MoEQuant EBSS calibration-quality validation

**Decision: OFFICIAL-EBSS-TRANSFER-LIMITED** (provenance UNDOCUMENTED; quality class ignoring provenance: OFFICIAL-EBSS-NO-CONSISTENT-GAIN)

## Quality (CANONICAL-B1 extraction for every configuration)

BF16 PPL 9.237090.

| config | reconstruction | PPL | ΔPPL vs BF16 |
|---|---|---|---|
| FR_s0 | 0.0046095 | 9.269004 | +0.031914 |
| FR_s1 | 0.0046259 | 9.272567 | +0.035478 |
| FR_s2 | 0.0046104 | 9.274693 | +0.037603 |
| TR_s0 | 0.0046120 | 9.271137 | +0.034047 |
| TR_s1 | 0.0046000 | 9.267561 | +0.030471 |
| TR_s2 | 0.0046051 | 9.259243 | +0.022153 |
| FC_s0 | 0.0047270 | 9.270335 | +0.033246 |
| FC_s1 | 0.0047277 | 9.267461 | +0.030371 |
| FC_s2 | 0.0047352 | 9.274605 | +0.037515 |
| EBSS | 0.0059707 | 9.273237 | +0.036147 |

## EBSS relative reconstruction penalty (%; < 0 = EBSS better) and ΔPPL (EBSS − baseline)

| baseline | s0 | s1 | s2 | mean | ΔPPL s0 / s1 / s2 | frac experts improved |
|---|---|---|---|---|---|---|
| FR | +29.530 | +29.069 | +29.504 | +29.368 | +0.00423 / +0.00067 / -0.00146 | 0.00 / 0.00 / 0.00 |
| TR | +29.460 | +29.797 | +29.653 | +29.636 | +0.00210 / +0.00568 / +0.01399 | 0.00 / 0.00 / 0.00 |
| FC | +26.311 | +26.291 | +26.092 | +26.231 | +0.00290 / +0.00578 / -0.00137 | 0.00 / 0.00 / 0.00 |

## Calibration diagnostics (means over layers 1/8/16/23)

| config | min exp. | p10 exp. | CV | Gini | routing L1 | atypical | model PPL | distinct-1 | distinct-2 | near-dup windows |
|---|---|---|---|---|---|---|---|---|---|---|
| FR_s0 | 3363 | 3904 | 0.100 | 0.055 | 0.031 | 0.35 | 11.30 | 0.502 | 0.867 | 0 |
| FR_s1 | 3420 | 3863 | 0.106 | 0.058 | 0.035 | 0.35 | 10.93 | 0.505 | 0.868 | 0 |
| FR_s2 | 3429 | 3932 | 0.091 | 0.050 | 0.031 | 0.40 | 11.30 | 0.510 | 0.873 | 0 |
| TR_s0 | 3314 | 3751 | 0.120 | 0.066 | 0.048 | 0.02 | 12.24 | 0.528 | 0.899 | 0 |
| TR_s1 | 3350 | 3804 | 0.116 | 0.063 | 0.043 | 0.01 | 12.21 | 0.532 | 0.899 | 0 |
| TR_s2 | 3280 | 3793 | 0.124 | 0.068 | 0.045 | 0.02 | 12.46 | 0.529 | 0.901 | 0 |
| FC_s0 | 3682 | 3980 | 0.074 | 0.041 | 0.093 | 0.65 | 8.86 | 0.462 | 0.788 | 0 |
| FC_s1 | 3570 | 3981 | 0.079 | 0.042 | 0.084 | 0.66 | 9.30 | 0.473 | 0.805 | 0 |
| FC_s2 | 3626 | 3949 | 0.081 | 0.045 | 0.089 | 0.65 | 9.29 | 0.479 | 0.810 | 0 |
| EBSS | 2167 | 2871 | 0.331 | 0.173 | 0.253 | 0.96 | 1.26 | 0.134 | 0.254 | 2 |

## Correctness

- all_configs_valid: True
- c4_rtn_fallbacks: 0
- ebss_rtn_fallbacks: 0
- quant_pipeline_regression: True
- extraction_impl_regression: True
- bf16_ppl_exact: True
- capture_repeatability: True
- quant_repeat_bitwise: True
- profiler_count_crosscheck_mismatches: 0

## Efficiency (seconds; EBSS generation cost unknown and excluded)

| config | tokenization | activation | Hessian | GPTQ | T_measured |
|---|---|---|---|---|---|
| FR_s0 | 0.0 | 62.5 | 29.2 | 259.0 | 350.7 |
| FR_s1 | 0.0 | 61.1 | 28.8 | 257.7 | 347.6 |
| FR_s2 | 0.0 | 61.2 | 28.9 | 265.5 | 355.6 |
| TR_s0 | 0.0 | 61.2 | 29.3 | 270.1 | 360.6 |
| TR_s1 | 0.0 | 61.1 | 29.2 | 269.7 | 360.0 |
| TR_s2 | 0.0 | 61.1 | 28.8 | 271.0 | 360.9 |
| FC_s0 | 0.0 | 61.3 | 28.7 | 269.6 | 359.6 |
| FC_s1 | 0.0 | 61.1 | 29.6 | 263.5 | 354.2 |
| FC_s2 | 0.0 | 61.2 | 28.7 | 255.0 | 344.9 |
| EBSS | 0.3 | 59.3 | 27.2 | 251.4 | 338.2 |

See `update_ebss_official_validation.md` for the interpretation.
