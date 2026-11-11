# EBSS balance-strength fidelity experiment

**Primary: BALANCE-NORMALIZATION-MATERIAL**  |  **Quality: BALANCE-VARIANT-QUALITY-MIXED**

## Counterfactual pruning on the EQ4_RATE 4 x 128 candidate pools

| variant | strength | D vs EQ4_RATE | D vs PO | median balance/probability spread |
|---|---|---|---|---|
| PROBABILITY_ONLY | 0 | 0.089 | 0.000 | 0.000 |
| EQ4_RATE | 1 | 0.000 | 0.089 | 0.046 |
| EQ4_SHARE | 0.25 | 0.063 | 0.028 | 0.012 |
| ALG1_SHARE | 1.936 | 0.048 | 0.125 | 0.090 |
| ALG1_RATE | 7.746 | 0.282 | 0.315 | 0.359 |

## Short generation 4 x 128 (canonical EQ4_RATE metrics, CANONICAL-B1 routing)

| condition | CV | min | p10 | distinct-2 | repetition | longest rep. n-gram | routing L1 | atypical | PPL |
|---|---|---|---|---|---|---|---|---|---|
| PROBABILITY_ONLY | 0.5058 | 7.3 | 16.2 | 0.598 | 0.572 | 35.5 | 0.384 | 1.00 | 1.621 |
| EQ4_RATE | 0.4060 | 8.3 | 18.7 | 0.547 | 0.604 | 42.8 | 0.331 | 1.00 | 1.579 |
| EQ4_SHARE | 0.5057 | 6.8 | 16.6 | 0.569 | 0.574 | 37.2 | 0.383 | 1.00 | 1.659 |
| ALG1_SHARE | 0.4375 | 8.2 | 17.2 | 0.608 | 0.594 | 34.2 | 0.354 | 1.00 | 1.624 |
| ALG1_RATE | 0.3935 | 7.7 | 19.0 | 0.689 | 0.516 | 9.0 | 0.331 | 1.00 | 1.596 |
| C4R_s0_first128 | 0.2202 | 77.2 | 100.4 | 0.932 | 0.327 | 3.8 | 0.148 | 1.00 | 15.629 |
| C4R_s0_first128_first4 | 0.3368 | 11.5 | 21.4 | 0.963 | 0.287 | 2.5 | 0.222 | 1.00 | 19.654 |

Gate: qualifying {'EQ4_SHARE': True, 'ALG1_SHARE': True, 'ALG1_RATE': True}; B=16 run: ['ALG1_SHARE', 'ALG1_RATE']; excluded {'EQ4_SHARE': 'qualified but lower fidelity priority than the two Algorithm-1 variants (max two new variants)'}

## B=16 x 512

| condition | CV | min | p10 | distinct-2 | repetition | routing L1 | atypical | PPL |
|---|---|---|---|---|---|---|---|---|
| PROBABILITY_ONLY | 0.2608 | 299.7 | 390.8 | 0.376 | 0.778 | 0.210 | 0.88 | 1.206 |
| EQ4_RATE | 0.2480 | 294.0 | 402.2 | 0.385 | 0.780 | 0.201 | 1.00 | 1.198 |
| C4R_s0 | 0.1755 | 354.4 | 437.5 | 0.849 | 0.517 | 0.110 | 0.44 | 10.250 |
| C4R_s1 | 0.1562 | 379.0 | 452.3 | 0.889 | 0.478 | 0.085 | 0.38 | 10.437 |
| C4R_s2 | 0.1316 | 388.0 | 461.8 | 0.875 | 0.477 | 0.078 | 0.38 | 12.388 |
| ALG1_SHARE | 0.1863 | 327.8 | 423.0 | 0.413 | 0.762 | 0.158 | 1.00 | 1.223 |
| ALG1_RATE | 0.1663 | 330.2 | 436.5 | 0.421 | 0.748 | 0.142 | 0.94 | 1.222 |

## B=16 GPTQ (layers 1/8/16/23)

| config | recon | PPL |
|---|---|---|
| C4R_s0 | 0.0053818 | 9.267487 |
| C4R_s1 | 0.0053950 | 9.268871 |
| C4R_s2 | 0.0053815 | 9.275386 |
| PROBABILITY_ONLY | 0.0061027 | 9.268311 |
| EQ4_RATE | 0.0061308 | 9.271518 |
| ALG1_SHARE | 0.0060830 | 9.276911 |
| ALG1_RATE | 0.0061072 | 9.274363 |
- ALG1_SHARE: vs PO -0.32 %, vs EQ4_RATE -0.78 %, vs C4 mean +12.94 %; experts improved vs EQ4_RATE 0.75
- ALG1_RATE: vs PO +0.07 %, vs EQ4_RATE -0.38 %, vs C4 mean +13.39 %; experts improved vs EQ4_RATE 0.64

See `../update_ebss_balance_fidelity.md` for the interpretation.
