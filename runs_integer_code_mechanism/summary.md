# GPTQ integer-code sensitivity: spatial structure and dense/MoE comparison

**Spatial classification (dense): MIXED-SPATIAL-STRUCTURE**. **Dense/MoE code comparison: B-MUCH-LARGER-MOE-CODE-DISAGREEMENT** (R_q = 17.86). Protocol sha256 1e59c47e286b861d.

**Interpretation.**

*Spatial (dense, Experiment A).* The prespecified rule returns MIXED-SPATIAL-STRUCTURE. That comes from boundary flags (60 % of primary cells), not adjacency clustering, which meets the clustered-cell criterion in only 4 % of cells. The underlying pattern is cleaner than the label:
1. Code flips are not adjacent-clustered within a quantization group. In both execution orders, lag-d flip-pair counts are 0.99–1.02x the null that preserves each (row, 128-group) flip count, at every lag from 1 to 64; cross-group lag-128 pairs are also about 1x. Where E1 exceeds its null mean, the excess comes from segment-edge enrichment of the marginals (first/last 8 positions of a group, 1.3–2.7x in execution order for down_proj and L1/L9 gate/up), not from extra adjacent pairs.
2. Flips are strongly concentrated by (row, group) cell. Relative to a row-only null, adjacent pairs are about 8x (up to 36x) in execution order. Columns that changed quantization-group assignment flip at 15–230x the rate of the others.
3. Post-hoc, not prespecified: flip rate rises steeply along GPTQ execution order. Dense gate/up go from 0.02 % in the first decile of columns to about 0.7 % later; down_proj from 0.27 % to about 4 %. MoE goes from about 4 % to about 30 %, falling again in the last decile (logs/posthoc_execution_profile.json, Fig. 7). Whole-group code shifts account for only 0.1–1.7 % of flips.

*Codes (Experiment C).* The MoE replay reproduces all 360 sampled historical tensors bitwise. On those 20/240 experts, 21.1 % of MoE integer codes change (5 groupings vs ORIGINAL), against 1.18 % for dense (4 non-null single-row perturbations vs ORIGINAL), so R_q = 17.9 (gate 52, up 52, down 5.1): B-MUCH-LARGER-MOE-CODE-DISAGREEMENT. Grid refitting does explain much of the BF16 figure: 66 % of changed MoE BF16 elements keep their code (87 % dense), and in each case a's codes placed on b's grid reproduce b's BF16 weight exactly. The code gap itself is real, about 18x, not a BF16-metric artifact. It is only somewhat larger than the BF16 gap (7.6x).

## Validation

- Dense alignment: 72 tensors, all checks pass: True (valid permutation, group = position // 128, codes in [0, 15], scale*(q-zero) -> BF16 bitwise = saved weights); FP32 solver parity verified at production time.
- Dense null G20261115 vs ORIGINAL: code flips 0.0, BF16 changes 0.0.
- MoE replay (unchanged solver, read-only info capture, 20 / 240 routed experts, 6 configs): bitwise = historical BF16 for all 360 tensors: True; codes exact; FP32 dequant == solver output.

## Dense spatial structure (Experiment A)

Primary cells (4 pairs x 12 tensors x 2 execution orders = 96): clustered fraction by order ref_exec 0.04, var_exec 0.04; by projection gate_proj 0.00, up_proj 0.00, down_proj 0.12; boundary-flagged 0.60; cells with < 200 flips 0.00.

| pair | L.proj | order | flips | p | raw E1 | null E1 mean [99.5 %] | E1 / null | lag-1 pairs / seg-null | lag-8 / seg-null | lag-64 / seg-null | max run | edge first / null |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G20261111__ORIGINAL | L19.down_proj | ref_exec | 541557 | 0.0394 | 12.029 | 11.575 [11.589] | 1.039 | 0.998 | 0.999 | 1.000 | 128 | 1.31 |
| G20261111__ORIGINAL | L19.down_proj | var_exec | 541557 | 0.0394 | 11.961 | 11.573 [11.585] | 1.034 | 0.999 | 0.999 | 1.000 | 128 | 1.32 |
| G20261111__ORIGINAL | L19.gate_proj | ref_exec | 75938 | 0.0055 | 107.589 | 90.874 [91.075] | 1.184 | 1.002 | 1.001 | 0.997 | 128 | 2.11 |
| G20261111__ORIGINAL | L19.gate_proj | var_exec | 75938 | 0.0055 | 98.770 | 90.833 [91.049] | 1.087 | 1.002 | 1.001 | 0.999 | 128 | 2.11 |
| G20261111__ORIGINAL | L19.up_proj | ref_exec | 72584 | 0.0053 | 110.755 | 92.897 [93.081] | 1.192 | 1.001 | 1.002 | 1.001 | 128 | 2.16 |
| G20261111__ORIGINAL | L19.up_proj | var_exec | 72584 | 0.0053 | 101.231 | 92.865 [93.056] | 1.090 | 1.001 | 1.002 | 1.001 | 128 | 2.16 |
| G20261111__ORIGINAL | L1.down_proj | ref_exec | 51961 | 0.0038 | 203.156 | 155.075 [155.440] | 1.310 | 1.002 | 1.003 | 0.999 | 127 | 3.09 |
| G20261111__ORIGINAL | L1.down_proj | var_exec | 51961 | 0.0038 | 198.339 | 155.126 [155.437] | 1.279 | 1.001 | 1.002 | 1.000 | 128 | 3.06 |
| G20261111__ORIGINAL | L1.gate_proj | ref_exec | 3651 | 0.0003 | 915.489 | 915.008 [927.327] | 1.001 | 1.002 | 0.997 | 0.997 | 121 | 1.11 |
| G20261111__ORIGINAL | L1.gate_proj | var_exec | 3651 | 0.0003 | 914.449 | 915.437 [928.186] | 0.999 | 1.000 | 1.002 | 1.002 | 121 | 1.10 |
| G20261111__ORIGINAL | L1.up_proj | ref_exec | 4799 | 0.0003 | 1262.370 | 1266.414 [1273.506] | 0.997 | 1.000 | 1.001 | 0.998 | 128 | 1.02 |
| G20261111__ORIGINAL | L1.up_proj | var_exec | 4799 | 0.0003 | 1262.370 | 1266.555 [1274.244] | 0.997 | 1.000 | 1.004 | 1.000 | 128 | 1.02 |
| G20261111__ORIGINAL | L27.down_proj | ref_exec | 127301 | 0.0092 | 58.180 | 53.489 [53.566] | 1.088 | 0.999 | 1.001 | 1.000 | 128 | 1.93 |
| G20261111__ORIGINAL | L27.down_proj | var_exec | 127301 | 0.0092 | 57.146 | 53.505 [53.582] | 1.068 | 1.000 | 1.001 | 1.001 | 128 | 1.79 |
| G20261111__ORIGINAL | L27.gate_proj | ref_exec | 22790 | 0.0017 | 217.318 | 217.283 [218.223] | 1.000 | 1.001 | 0.999 | 0.999 | 128 | 0.90 |
| G20261111__ORIGINAL | L27.gate_proj | var_exec | 22790 | 0.0017 | 217.423 | 217.269 [218.172] | 1.001 | 1.001 | 0.999 | 0.999 | 128 | 0.90 |
| G20261111__ORIGINAL | L27.up_proj | ref_exec | 20618 | 0.0015 | 221.022 | 220.405 [221.434] | 1.003 | 1.004 | 1.000 | 0.998 | 128 | 0.92 |
| G20261111__ORIGINAL | L27.up_proj | var_exec | 20618 | 0.0015 | 220.489 | 220.441 [221.586] | 1.000 | 1.001 | 1.001 | 0.997 | 128 | 0.91 |
| G20261111__ORIGINAL | L9.down_proj | ref_exec | 659771 | 0.0479 | 9.926 | 9.554 [9.564] | 1.039 | 0.999 | 0.999 | 1.000 | 128 | 1.35 |
| G20261111__ORIGINAL | L9.down_proj | var_exec | 659771 | 0.0479 | 9.991 | 9.555 [9.566] | 1.046 | 0.999 | 0.999 | 1.000 | 128 | 1.36 |
| G20261111__ORIGINAL | L9.gate_proj | ref_exec | 301004 | 0.0219 | 22.791 | 21.022 [21.047] | 1.084 | 1.000 | 1.000 | 1.001 | 128 | 1.54 |
| G20261111__ORIGINAL | L9.gate_proj | var_exec | 301004 | 0.0219 | 22.799 | 21.024 [21.047] | 1.084 | 1.001 | 1.001 | 1.000 | 128 | 1.54 |
| G20261111__ORIGINAL | L9.up_proj | ref_exec | 298846 | 0.0217 | 22.919 | 21.113 [21.137] | 1.086 | 1.000 | 1.000 | 1.000 | 128 | 1.55 |
| G20261111__ORIGINAL | L9.up_proj | var_exec | 298846 | 0.0217 | 22.916 | 21.120 [21.145] | 1.085 | 0.999 | 1.000 | 1.000 | 128 | 1.56 |

Aggregated over the 4 primary pairs (median over pairs):

| L.proj | order | p | raw E1 | E1 / null mean | lag-1 excess vs seg-null | vs row-null | lag-128 cross pairs / null | edge first / null | edge last / null | group-changed cols flip rate / same |
|---|---|---|---|---|---|---|---|---|---|---|
| L1.gate_proj | original | 0.0057 | 10.834 | 0.980 | 0.984 | 0.980 | 1.042 | 1.82 | 0.78 | 145.32 |
| L1.gate_proj | ref_exec | 0.0057 | 102.510 | 1.167 | 1.001 | 7.685 | 0.993 | 2.09 | 2.09 | 145.32 |
| L1.gate_proj | var_exec | 0.0057 | 102.483 | 1.176 | 1.001 | 7.684 | 0.992 | 2.08 | 2.09 | 145.32 |
| L1.up_proj | original | 0.0061 | 10.686 | 0.992 | 0.996 | 0.994 | 1.037 | 1.72 | 0.77 | 135.24 |
| L1.up_proj | ref_exec | 0.0061 | 100.109 | 1.156 | 1.002 | 7.776 | 0.993 | 2.00 | 2.03 | 135.24 |
| L1.up_proj | var_exec | 0.0061 | 100.162 | 1.165 | 1.002 | 7.781 | 0.993 | 2.00 | 2.03 | 135.24 |
| L1.down_proj | original | 0.0045 | 3.620 | 1.034 | 1.020 | 1.044 | 0.954 | 1.08 | 1.10 | 230.63 |
| L1.down_proj | ref_exec | 0.0045 | 174.857 | 1.255 | 1.001 | 36.194 | 1.030 | 2.73 | 2.72 | 230.63 |
| L1.down_proj | var_exec | 0.0045 | 171.434 | 1.236 | 1.001 | 36.189 | 1.028 | 2.73 | 2.72 | 230.63 |
| L9.gate_proj | original | 0.0102 | 7.165 | 1.031 | 1.035 | 1.039 | 0.975 | 0.93 | 0.87 | 67.67 |
| L9.gate_proj | ref_exec | 0.0102 | 56.867 | 1.100 | 1.001 | 7.271 | 0.994 | 1.63 | 1.65 | 67.67 |
| L9.gate_proj | var_exec | 0.0102 | 55.974 | 1.100 | 1.001 | 7.270 | 0.992 | 1.62 | 1.66 | 67.67 |
| L9.up_proj | original | 0.0102 | 6.980 | 1.028 | 1.031 | 1.034 | 0.974 | 0.94 | 0.90 | 67.31 |
| L9.up_proj | ref_exec | 0.0102 | 56.165 | 1.103 | 1.000 | 7.323 | 1.027 | 1.65 | 1.69 | 67.31 |
| L9.up_proj | var_exec | 0.0102 | 55.312 | 1.101 | 1.001 | 7.326 | 1.013 | 1.65 | 1.68 | 67.31 |
| L9.down_proj | original | 0.0488 | 1.161 | 0.982 | 0.983 | 0.982 | 0.994 | 0.94 | 1.02 | 15.87 |
| L9.down_proj | ref_exec | 0.0488 | 9.799 | 1.039 | 0.999 | 7.982 | 1.072 | 1.34 | 1.35 | 15.87 |
| L9.down_proj | var_exec | 0.0488 | 9.844 | 1.041 | 0.999 | 7.991 | 1.079 | 1.35 | 1.38 | 15.87 |
| L19.gate_proj | original | 0.0009 | 46.988 | 0.983 | 0.984 | 0.978 | 0.855 | 0.95 | 1.07 | 152.96 |
| L19.gate_proj | ref_exec | 0.0009 | 447.315 | 1.002 | 1.002 | 8.472 | 0.994 | 0.86 | 1.07 | 152.96 |
| L19.gate_proj | var_exec | 0.0009 | 446.837 | 1.001 | 1.002 | 8.462 | 1.003 | 0.86 | 1.07 | 152.96 |
| L19.up_proj | original | 0.0008 | 52.209 | 0.961 | 0.965 | 0.957 | 0.858 | 0.97 | 1.03 | 160.36 |
| L19.up_proj | ref_exec | 0.0008 | 468.333 | 1.001 | 1.002 | 8.393 | 0.994 | 0.85 | 1.06 | 160.36 |
| L19.up_proj | var_exec | 0.0008 | 467.726 | 1.000 | 1.002 | 8.388 | 0.984 | 0.85 | 1.06 | 160.36 |
| L19.down_proj | original | 0.0329 | 1.312 | 1.013 | 1.014 | 1.014 | 1.026 | 0.94 | 1.02 | 23.53 |
| L19.down_proj | ref_exec | 0.0329 | 14.328 | 1.044 | 1.004 | 10.567 | 1.021 | 1.36 | 1.37 | 23.53 |
| L19.down_proj | var_exec | 0.0329 | 14.363 | 1.038 | 1.002 | 10.569 | 1.014 | 1.36 | 1.39 | 23.53 |
| L27.gate_proj | original | 0.0012 | 48.659 | 1.069 | 1.068 | 1.065 | 1.008 | 1.13 | 1.09 | nan |
| L27.gate_proj | ref_exec | 0.0012 | 335.185 | 0.999 | 1.000 | 7.898 | 0.976 | 0.97 | 0.98 | nan |
| L27.gate_proj | var_exec | 0.0012 | 335.329 | 0.999 | 1.000 | 7.901 | 1.007 | 0.97 | 0.97 | nan |
| L27.up_proj | original | 0.0011 | 45.910 | 1.048 | 1.047 | 1.038 | 1.030 | 1.09 | 1.07 | nan |
| L27.up_proj | ref_exec | 0.0011 | 346.454 | 1.002 | 1.003 | 7.951 | 1.009 | 0.97 | 0.97 | nan |
| L27.up_proj | var_exec | 0.0011 | 346.161 | 1.000 | 1.001 | 7.950 | 0.988 | 0.97 | 0.97 | nan |
| L27.down_proj | original | 0.0141 | 1.886 | 1.018 | 1.020 | 1.021 | 1.001 | 1.06 | 1.01 | 62.87 |
| L27.down_proj | ref_exec | 0.0141 | 43.835 | 1.103 | 1.003 | 21.055 | 0.991 | 2.03 | 1.95 | 62.87 |
| L27.down_proj | var_exec | 0.0141 | 44.025 | 1.099 | 1.000 | 21.175 | 1.064 | 1.85 | 2.09 | 62.87 |
- original: lag-1 pairs / segment-null over 48 primary cells: median 1.0058, range 0.9117-1.2316; vs row-only null median 1.01
- original: lag-2 pairs / segment-null over 48 primary cells: median 1.0058, range 0.8798-1.2226
- original: lag-8 pairs / segment-null over 48 primary cells: median 0.9908, range 0.9139-1.8201
- original: lag-64 pairs / segment-null over 48 primary cells: median 1.0514, range 0.7141-1.2863
- ref_exec: lag-1 pairs / segment-null over 48 primary cells: median 1.0011, range 0.9959-1.0113; vs row-only null median 8.27
- ref_exec: lag-2 pairs / segment-null over 48 primary cells: median 1.0006, range 0.9941-1.0053
- ref_exec: lag-8 pairs / segment-null over 48 primary cells: median 1.0006, range 0.9920-1.0039
- ref_exec: lag-64 pairs / segment-null over 48 primary cells: median 0.9998, range 0.9964-1.0040
- var_exec: lag-1 pairs / segment-null over 48 primary cells: median 1.0010, range 0.9966-1.0219; vs row-only null median 8.27
- var_exec: lag-2 pairs / segment-null over 48 primary cells: median 1.0006, range 0.9957-1.0145
- var_exec: lag-8 pairs / segment-null over 48 primary cells: median 1.0009, range 0.9920-1.0040
- var_exec: lag-64 pairs / segment-null over 48 primary cells: median 0.9999, range 0.9948-1.0053

**MoE secondary spatial (20 sampled experts x 3 projections, 5 groupings vs ORIGINAL, 200 permutations):**

| order | lag-1 pairs / seg-null (median, range) | vs row-only null | E1 / null mean | edge first / null |
|---|---|---|---|---|
| original | 0.996 (0.937-1.074) | 1.00 | nan | nan |
| ref_exec | 1.038 (0.997-1.073) | 2.58 | 1.057 | 1.54 |
| var_exec | 1.038 (0.997-1.086) | 2.58 | 1.052 | 1.56 |

MoE cells meeting the dense clustered-cell criterion: 0/600.


Row-flip-rate strata (ref exec, median over primary cells): lag-1 pairs / segment-null: low 1.009, mid 0.998, high 1.001.

## Integer codes and quantization grid (Experiments A + C)

| set | frac_bf16_changed | frac_code_changed | frac_level_changed | cat_B | cat_C | cat_D | frac_bf16_changed_same_code | frac_code_changed_same_bf16 | code_changed_same_grid | frac_scale_changed | frac_zero_changed | frac_group_changed | bf16_changed_grid_only_a_on_b | D_same_grid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense 4 vs ORIGINAL | 0.0693 | 0.0118 | 0.0062 | 0.0602 | 0.0027 | 0.0092 | 0.8680 | 0.2249 | 0.0000 | 0.9266 | 0.0072 | 0.0015 | 0.8680 | 0.0007 |
| dense 6 grouping pairs | 0.0814 | 0.0149 | 0.0076 | 0.0698 | 0.0033 | 0.0116 | 0.8573 | 0.2189 | 0.0000 | 0.9319 | 0.0096 | 0.0022 | 0.8573 | 0.0006 |
| dense null | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | nan | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | nan |
| dense gate_proj | 0.0268 | 0.0048 | 0.0025 | 0.0232 | 0.0012 | 0.0037 | 0.8635 | 0.2410 | 0.0000 | 0.9036 | 0.0031 | 0.0008 | 0.8635 | 0.0022 |
| dense up_proj | 0.0264 | 0.0049 | 0.0025 | 0.0227 | 0.0012 | 0.0037 | 0.8603 | 0.2431 | 0.0000 | 0.9023 | 0.0032 | 0.0008 | 0.8603 | 0.0023 |
| dense down_proj | 0.1546 | 0.0257 | 0.0137 | 0.1346 | 0.0056 | 0.0201 | 0.8700 | 0.2184 | 0.0000 | 0.9738 | 0.0154 | 0.0030 | 0.8700 | 0.0001 |
| MoE 5 vs ORIGINAL | 0.5283 | 0.2108 | 0.0918 | 0.3468 | 0.0293 | 0.1816 | 0.6564 | 0.1389 | 0.0003 | 0.9309 | 0.1833 | 0.0842 | 0.6564 | 0.0015 |
| MoE 10 grouping pairs | 0.5274 | 0.2112 | 0.0919 | 0.3455 | 0.0293 | 0.1819 | 0.6551 | 0.1388 | 0.0003 | 0.9306 | 0.1839 | 0.0844 | 0.6551 | 0.0015 |
| MoE gate_proj | 0.5800 | 0.2507 | 0.1079 | 0.3634 | 0.0342 | 0.2166 | 0.6266 | 0.1362 | 0.0002 | 0.9399 | 0.2217 | 0.1066 | 0.6266 | 0.0009 |
| MoE up_proj | 0.5807 | 0.2516 | 0.1081 | 0.3635 | 0.0344 | 0.2172 | 0.6260 | 0.1367 | 0.0002 | 0.9401 | 0.2227 | 0.1066 | 0.6260 | 0.0009 |
| MoE down_proj | 0.4243 | 0.1302 | 0.0594 | 0.3134 | 0.0193 | 0.1109 | 0.7386 | 0.1483 | 0.0004 | 0.9126 | 0.1056 | 0.0395 | 0.7386 | 0.0038 |
| MoE tokens low | 0.5915 | 0.2652 | 0.1131 | 0.3625 | 0.0361 | 0.2291 | 0.6128 | 0.1363 | 0.0004 | 0.9315 | 0.2374 | 0.1189 | 0.6128 | 0.0018 |
| MoE tokens mid | 0.5295 | 0.2182 | 0.0943 | 0.3413 | 0.0300 | 0.1882 | 0.6446 | 0.1377 | 0.0002 | 0.9317 | 0.1912 | 0.0869 | 0.6446 | 0.0012 |
| MoE tokens high | 0.4641 | 0.1501 | 0.0683 | 0.3358 | 0.0218 | 0.1284 | 0.7234 | 0.1450 | 0.0002 | 0.9294 | 0.1225 | 0.0472 | 0.7234 | 0.0013 |
| MoE L1 | 0.3961 | 0.1158 | 0.0548 | 0.2975 | 0.0172 | 0.0986 | 0.7511 | 0.1484 | 0.0001 | 0.9289 | 0.0894 | 0.0313 | 0.7511 | 0.0009 |
| MoE L8 | 0.6210 | 0.2919 | 0.1232 | 0.3686 | 0.0395 | 0.2524 | 0.5935 | 0.1353 | 0.0003 | 0.9337 | 0.2641 | 0.1360 | 0.5935 | 0.0012 |
| MoE L16 | 0.5381 | 0.2100 | 0.0916 | 0.3573 | 0.0292 | 0.1808 | 0.6640 | 0.1392 | 0.0003 | 0.9294 | 0.1820 | 0.0806 | 0.6640 | 0.0016 |
| MoE L23 | 0.5581 | 0.2257 | 0.0975 | 0.3636 | 0.0312 | 0.1945 | 0.6516 | 0.1383 | 0.0004 | 0.9314 | 0.1978 | 0.0890 | 0.6516 | 0.0020 |

Dense by layer x projection (4 vs ORIGINAL): L1.gate_proj code 0.0052 / BF16 0.0245; L1.up_proj code 0.0057 / BF16 0.0246; L1.down_proj code 0.0069 / BF16 0.0105; L9.gate_proj code 0.0109 / BF16 0.0629; L9.up_proj code 0.0109 / BF16 0.0620; L9.down_proj code 0.0484 / BF16 0.3121; L19.gate_proj code 0.0020 / BF16 0.0105; L19.up_proj code 0.0019 / BF16 0.0100; L19.down_proj code 0.0339 / BF16 0.2351; L27.gate_proj code 0.0012 / BF16 0.0095; L27.up_proj code 0.0011 / BF16 0.0092; L27.down_proj code 0.0137 / BF16 0.0609.

MoE per expert x projection code-flip rate: min 0.0233, median 0.2063, max 0.4280 (token tertile edges [4212.0, 4699.0]).

## Dense vs MoE

| metric | dense | MoE Family D (sampled) |
|---|---|---|
| model | Qwen2.5-1.5B (12 MLP tensors, all elements) | Qwen1.5-MoE-A2.7B (20 / 240 routed experts, 60 tensors) |
| comparisons | 4 single-row perturbations vs ORIGINAL | 5 groupings vs ORIGINAL |
| bf16_changed | 0.06931 | 0.5283 |
| integer_code_changed | 0.01181 | 0.2108 |
| zero_relative_level_changed | 0.006232 | 0.09179 |
| same_code_given_bf16_changed | 0.868 | 0.6564 |
| same_bf16_given_code_changed | 0.2249 | 0.1389 |
| code_changed_on_identical_grid | 6.281e-06 | 0.00027 |
| scale_changed_elements | 0.9266 | 0.9309 |
| zero_changed_elements | 0.007232 | 0.1833 |
| group_assignment_changed_elements | 0.001526 | 0.08424 |
| actorder_position_changed_columns | 0.1328 | 0.8676 |
| bf16_diff_explained_by_grid_a_on_b | 0.868 | 0.6564 |
| rel_frobenius_sampled | None | 0.08787 |
| ge8ulp_share_sampled | None | 0.374 |
| historical_whole_model_bf16_changed | 0.05545 | 0.5406 |
| historical_hessian_rel_fro_tn | 4.659e-05 | AO 0.0039 / AA 0.0141 (16 diagnostic Hessians) |
| historical_sd_ppl_5_groupings | 0.00503 | 0.004422 |

R_q = 17.86 (by projection gate_proj 51.96, up_proj 51.57, down_proj 5.06); R_BF16 = 7.62. Outcome B-MUCH-LARGER-MOE-CODE-DISAGREEMENT.

## Fixed-act-order replay (optional controlled check)

Parity (natural permutation supplied as override reproduces saved weights): 84/84.

- dense gate_proj: code flips natural 0.0048 -> fixed act-order 0.0004; BF16 0.0268 -> 0.0030; group changes 0.0008 -> 0.0000; scale changed 0.9036 -> 0.8989
- dense up_proj: code flips natural 0.0049 -> fixed act-order 0.0004; BF16 0.0264 -> 0.0029; group changes 0.0008 -> 0.0000; scale changed 0.9023 -> 0.8970
- dense down_proj: code flips natural 0.0257 -> fixed act-order 0.0047; BF16 0.1546 -> 0.0389; group changes 0.0030 -> 0.0000; scale changed 0.9738 -> 0.9099
- dense fixed-act-order spatial (ORIGINAL order, 48 cells, NaN = cells with < 2 flips excluded): median E1 / null 0.999; median lag-1 pairs / seg-null 0.9999; median flip rate 0.00043
- MoE gate_proj: code flips natural 0.2507 -> fixed act-order 0.0385; BF16 0.5800 -> 0.2959; group changes 0.1066 -> 0.0000; scale changed 0.9399 -> 0.9374
- MoE up_proj: code flips natural 0.2516 -> fixed act-order 0.0384; BF16 0.5807 -> 0.2953; group changes 0.1066 -> 0.0000; scale changed 0.9401 -> 0.9374
- MoE down_proj: code flips natural 0.1302 -> fixed act-order 0.0297; BF16 0.4243 -> 0.2387; group changes 0.0395 -> 0.0000; scale changed 0.9126 -> 0.9087

## Answers

1. **Are dense code changes spatially clustered beyond a suitable permutation null?** Not at the adjacency level. Lag-1 to lag-64 pair counts are 1.000–1.001x (median; range 0.996–1.022) the (row, group)-conditioned null in both execution orders, and no row-flip-rate stratum shows excess (low/mid/high 1.009 / 0.998 / 1.001). The flips are, however, highly concentrated in particular (row, group) cells (about 8x vs a row-only null) and in columns that changed group. The prespecified label is MIXED-SPATIAL-STRUCTURE because of boundary concentration.
2. **Does it depend on the execution order?** The (row, group) concentration and the edge effects appear only in GPTQ execution order, where a group's 128 columns are contiguous. In original column order the same masks look nearly unstructured: row-null excess about 1.0 and lag ratios 0.9–1.2 with no consistent sign. Reference and variant act-orders give nearly identical statistics, since only 13 % of dense column positions differ.
3. **Do flips concentrate at groups or block boundaries?**
   - Yes at groups: the first and last 8 positions of a 128-group carry 1.3–2.7x the null-expected flips in execution order (gate/up of L1 and L9 and all down_proj; not L19/L27 gate/up).
   - Columns whose group assignment changed flip 15–230x more often.
   - Lazy-batch blocks coincide with groups (blocksize = group size = 128), so the two boundaries cannot be separated.
4. **Is the pattern consistent with error feedback, independent rounding changes, or a mixture?** A mixture.
   - Against local propagation: within a group, flip positions are exchangeable given their count, with no runs beyond chance.
   - Between groups: the steep rise along execution order (post-hoc) and the concentration in changed-group columns fit accumulated downstream divergence (error feedback through H⁻¹ plus regrouping or grid refitting). This is not independent uniform rounding.
   - Confound: execution position is the descending rank of diag(H), so 'late' also means 'small curvature'.
5. **What causal claims are not yet supported?**
   - That error feedback causes any flip.
   - That the execution-order gradient is due to order rather than diag(H) magnitude.
   - That edge enrichment reflects solver dynamics rather than group-start grid fitting.
   - That MoE spatial results generalize: they are a secondary analysis with 200 permutations, show a small within-group lag-1 excess (1.04x in execution order), and 0/600 cells meet the clustered criterion.
   - That fixed-grid rounding sensitivity was measured: code changes on an identical grid are about 0 in both models (dense 6e-6, MoE 3e-4 of elements) because the grid itself refits almost everywhere.
6. **What fraction of MoE Family D codes change?** 21.1 % of elements in the 20 sampled experts (5 groupings vs ORIGINAL; 21.1 % for the 10 grouping pairs).
   - By projection: gate 25.1 %, up 25.2 %, down 13.0 %.
   - By layer: L1 11.6 %, L8 29.2 %, L16 21.0 %, L23 22.6 %.
   - By token tertile: low 26.5 %, mid 21.8 %, high 15.0 %.
   - Per expert x projection: 2.3–42.8 % (median 20.6 %).
   - Zero-relative level changes (q − zero, excluding pure zero-point relabels): 9.2 %.
   This is a 20/240 sample, not a whole-model rate.
7. **How does it compare with the dense 1.36 %?** That historical figure averaged 10 pairs including duplicate G20261115 = ORIGINAL comparisons. With matched definitions, dense is 1.18 % (4 vs ORIGINAL) or 1.49 % (6 grouping pairs).
   - MoE is 18x higher (gate/up about 52x, down 5x).
   - Level changes: 9.2 % vs 0.62 % (15x).
   - This is not attributed to architecture: the MoE Hessian perturbation is 55–200x larger, and MoE experts see about 2 tokens per gu input dimension against 43 for dense. Every calibration sequence is perturbed in the MoE, against one in dense.
8. **Does grid refitting explain much of the ~55 % MoE BF16 disagreement?** About two thirds of it.
   - 52.8 % of sampled MoE elements change BF16. Of those, 65.6 % keep their integer code; the scale changed and is verified to reproduce the difference exactly. That is about 35 % of all elements.
   - The remaining 18.2 % of elements change both code and BF16.
   - Scales differ on about 93 % of elements in both models: the median relative difference of changed scales is 0.47 % (MoE) and below 1e-4 (dense).
   - Zero-points differ on 18.3 % (MoE) vs 0.7 % (dense); group assignments on 8.4 % vs 0.15 %; act-order positions on 87 % vs 13 % of columns.
   - Counterfactual shares are order-dependent and not additive.
9. **Is down_proj especially sensitive at the code level?**
   - In the dense model, yes: down 2.57 % vs gate/up 0.48 % (5x). It peaks in mid layers (L9 4.8 %, L19 3.4 % vs L1 0.7 %, L27 1.4 %), matching its BF16 pattern.
   - In the MoE, no: down_proj is the least code-sensitive projection (13 % vs 25 %).
   - Dense down_proj has 8960 inputs (7.3 tokens per dimension); MoE down_proj has 1408 (3.1).
10. **Which mechanism should the dose-response experiment test directly?** Act-order permutation instability as the amplifier.
   - In the optional fixed-act-order replay, the variant Hessian is solved with the reference permutation. Parity was 84/84 (natural permutation as override reproduces the saved weights).
   - This removes 85–92 % of code flips: dense gate/up 0.48 → 0.04 %, down 2.57 → 0.47 %; MoE gate/up 25 → 3.9 %, down 13 → 3.0 %.
   - It also removes all group reassignments, while about 90 % of scales still change. The dense fixed-order residual flips are spatially exchangeable (lag-1 pairs / null 1.000).
   - The next experiment should measure code flips per unit trace-normalized Hessian perturbation under natural vs fixed act-order in both models, at matched perturbation sizes.
   - A direct feedback test with frozen permutation and frozen grid would need a new opt-in solver option (fixed scale/zero); it was not run here.

## Limitations

- **MoE coverage.** 20/240 experts (8 prespecified check experts plus 12 token-count strata), seed-0 FR only; whole-MoE code rates are not claimed. MoE spatial statistics are secondary (200 permutations).
- **Dense design.** The dense primary comparisons are 4 single-row perturbations from one calibration seed; the 6 grouping pairs are combinations of them, not independent.
- **What the null conditions on.** It preserves (row, segment) counts. It cannot detect structure at the (row, group) level itself, which is described separately: row-only-null excess, changed-group columns and the post-hoc execution profile.
- **Interpretation of E1.** E1 ratios are sensitive to edge marginals; lag-pair / analytic-null ratios are the cleaner adjacency measure.
- **Post-hoc analysis.** The execution-order profile was not prespecified and is confounded with diag(H) rank.
- **Fixed-act-order replay.** It changes the variant's computation (grid and error-feedback trajectory still differ), so reductions there are not a clean attribution.
- **Protocol amendment.** One amendment before any result: a one-line bug in the dense_align stage (logs/protocol_amendment.json). The report module was changed after results only for output formatting: NaN-aware medians, extra summary lines and one figure key index.

## Files

- Code: `moe_calib/code_spatial.py`, `diagnostics/integer_code_mechanism.py`, `diagnostics/report_integer_code_mechanism.py`, `tests/test_code_spatial.py`.
- Outputs: `runs_integer_code_mechanism/` (required JSONs, `figures/fig1`-`fig6`, `logs/`).
- MoE codes / Hessians of the sampled experts: `/home/kaixuan6/moe_cache_integer_code_mechanism/moe_codes|moe_hessians/<config>/L<l>.pt`.

