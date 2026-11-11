# Model-matched EBSS generation pilot (16 x 512, w = 4, tau = 1.3)

**Decision: MODEL-MATCHED-EBSS-BALANCING-SUPPORTED** (balancing class ignoring GPTQ: MODEL-MATCHED-EBSS-BALANCING-SUPPORTED)

## Main generation comparison (CANONICAL-B1 replay routing, all 24 MoE layers)

| metric | Probability-only | EBSS | EBSS - PO | C4 Random B16 (mean of 3) |
|---|---|---|---|---|
| paper sigma (Eq. 3/4) | 0.01753 | 0.01667 | -0.00086 | 0.01038 |
| load CV (mean of layers) | 0.2608 | 0.2480 | -0.0128 | 0.1544 |
| Gini | 0.1385 | 0.1329 | -0.0056 | 0.0847 |
| min expert exposure | 299.7 | 294.0 | -5.6 | 373.8 |
| p10 expert exposure | 390.8 | 402.2 | +11.4 | 450.5 |
| zero-exposure experts / layer | 0.00 | 0.00 | +0.00 | 0.00 |
| routing L1 vs C4 (L1/8/16/23) | 0.210 | 0.201 | -0.009 | 0.091 |
| frac routing-atypical | 0.88 | 1.00 | +0.12 | 0.40 |
| model PPL on the set | 1.206 | 1.198 | -0.008 | 11.025 |
| distinct-1 | 0.222 | 0.220 | -0.002 | 0.510 |
| distinct-2 | 0.376 | 0.385 | +0.009 | 0.871 |
| repetition rate | 0.778 | 0.780 | +0.002 | 0.490 |
| generation wall-s (= GPU-s) | 1264 | 1338 | +74 | - |

Intervention: {"steps_with_choice": 8160, "steps_kept_set_differs_from_probability_ranking": 362, "frac": 0.04436274509803922, "steps_kept_parents_differ": 362}
Generation-time vs replay routing agreement: {'PROBABILITY_ONLY': 0.8721, 'EBSS': 0.8792}

## B=16 GPTQ (layers 1/8/16/23, frozen GPTQ)

| config | recon | RTN fallbacks | PPL |
|---|---|---|---|
| C4R_s0 | 0.0053818 | 0 | 9.267487 |
| C4R_s1 | 0.0053950 | 0 | 9.268871 |
| C4R_s2 | 0.0053815 | 0 | 9.275386 |
| PO | 0.0061027 | 0 | 9.268311 |
| EBSS | 0.0061308 | 0 | 9.271518 |

- EBSS_vs_PO: +0.46 %, experts improved 0.29
- EBSS_vs_C4R_mean: +13.83 %, experts improved 0.00
- PO_vs_C4R_mean: +13.30 %, experts improved 0.00

See `update_ebss_generation_pilot.md` for the interpretation.
