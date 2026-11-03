# Exp2 — budget headroom (random selection, frozen GPTQ, test split)


| B | seeds | expert recon mean | p90 | rel. change vs previous B | PPL mean ± sd |
|---|---|---|---|---|---|
| 0 (RTN) | 1 | 0.00680 | 0.00752 |  | – |
| 32 | 3 | 0.00511 | 0.00604 | -24.85% | – |
| 64 | 3 | 0.00483 | 0.00575 | -5.36% | – |
| 128 | 3 | 0.00462 | 0.00551 | -4.50% | – |
| 256 | 3 | 0.00448 | 0.00537 | -2.89% | – |
| 512 | 3 | 0.00439 | 0.00525 | -2.04% | – |

Reading: compare each step's PPL change with the seed sd at the same B. If a doubling is within ~2 sd, model-level gains from allocation (a fraction of a doubling) will not be detectable; report E3 on expert reconstruction instead.
