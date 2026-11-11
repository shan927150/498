# C4-prefix diversity rescue (modified EBSS; B=16, P=64 + G=448)

**Primary: `PREFIX-RESCUE-PARTIAL`**  |  **Diversity: `NO-DIVERSITY-IMPROVEMENT`**

## Diversity by region (per-sequence means)

| set | cont. distinct-2 | cont. repetition | full distinct-2 | full repetition | cont. NLL |
|---|---|---|---|---|---|
| C4R_s0 | 0.855 | 0.500 | 0.849 | 0.517 | 2.217 |
| PX_PO | 0.200 | 0.872 | 0.273 | 0.826 | 0.193 |
| PX_EBSS | 0.278 | 0.838 | 0.349 | 0.793 | 0.215 |
| RS_PO | 0.357 | 0.790 | 0.376 | 0.778 | 0.140 |
| RS_EBSS | 0.421 | 0.744 | 0.421 | 0.748 | 0.149 |
| SPLICE_C4prefix_RS_PO | — | — | 0.433 | 0.728 | — |
| SPLICE_C4prefix_RS_EBSS | — | — | 0.489 | 0.690 | — |

## Routing (CANONICAL-B1, 24 layers)

| set | CV | min | p10 | routing L1 | atypical |
|---|---|---|---|---|---|
| C4R_s0 | 0.1755 | 354.4 | 437.5 | 0.110 | 0.44 |
| PX_PO | 0.3178 | 240.9 | 351.0 | 0.260 | 1.00 |
| PX_EBSS | 0.3054 | 246.0 | 357.9 | 0.255 | 1.00 |
| RS_PO | 0.2608 | 299.7 | 390.8 | 0.210 | 0.88 |
| RS_EBSS | 0.1663 | 330.2 | 436.5 | 0.142 | 0.94 |
| C4R_s1 | 0.1562 | 379.0 | 452.3 | 0.085 | 0.38 |
| C4R_s2 | 0.1316 | 388.0 | 461.8 | 0.078 | 0.38 |

## B=16 GPTQ (layers 1/8/16/23)

| config | recon | vs C4 mean | PPL |
|---|---|---|---|
| C4 Random s0 (= matched natural) | 0.0053818 | -0.08 % | 9.267487 |
| C4 Random s1 | 0.0053950 | +0.17 % | 9.268871 |
| C4 Random s2 | 0.0053815 | -0.09 % | 9.275386 |
| Random-start PO | 0.0061027 | +13.30 % | 9.268311 |
| Random-start ALG1_RATE | 0.0061072 | +13.39 % | 9.274363 |
| C4-prefix PO | 0.0057175 | +6.15 % | 9.272269 |
| C4-prefix ALG1_RATE | 0.0056954 | +5.74 % | 9.266021 |

Δ_PO = -3.852e-04, Δ_EBSS = -4.118e-04, Δ_balance|prefix = -2.209e-05; gap recovery G_PO = 0.538, G_EBSS = 0.571

See `../update_ebss_prefix_rescue.md` for the interpretation.
