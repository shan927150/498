# Calibration batch-composition sensitivity — summary (four layers, seed 0)

**Decision (rule frozen in protocol.json): BATCH-SENSITIVITY-DOMINANT**

Routed experts of layers 1/8/16/23 GPTQ-quantized (4-bit, g128, percdamp 1.0, act-order; fake-quantized BF16 weights), everything else BF16. Only the grouping of calibration sequences into BF16 forward batches during activation caching differs between variants.

| strategy | cache variant | PPL | mean NLL | Δ vs BF16 | change vs ORIGINAL | mean recon | recon change |
|---|---|---|---|---|---|---|---|
| BF16 | — | 9.237090 | — | — | — | — | — |
| Full Random s0 | ORIGINAL | 9.267851 | 2.226552 | +0.0308 (+0.333 %) | +0.000000 | 0.0046095 | +0.000 % |
| Full Random s0 | COMPACT | 9.272002 | 2.226999 | +0.0349 (+0.378 %) | +0.004150 | 0.0046093 | -0.006 % |
| Full Random s0 | SHUFFLED | 9.278272 | 2.227675 | +0.0412 (+0.446 %) | +0.010421 | 0.0046090 | -0.012 % |
| Full Coverage-LP s0 | ORIGINAL | 9.277908 | 2.227636 | +0.0408 (+0.442 %) | +0.000000 | 0.0047270 | +0.000 % |
| Full Coverage-LP s0 | COMPACT | 9.268026 | 2.226570 | +0.0309 (+0.335 %) | -0.009882 | 0.0047269 | -0.003 % |
| Full Coverage-LP s0 | SHUFFLED | 9.265841 | 2.226335 | +0.0288 (+0.311 %) | -0.012067 | 0.0047267 | -0.007 % |

| cache variant | FR PPL | FC PPL | FC − FR | ranking | interaction Δ(v) − Δ(ORIGINAL) | window-bootstrap 95 % CI of ΔNLL |
|---|---|---|---|---|---|---|
| ORIGINAL | 9.267851 | 9.277908 | +0.010057 | FC worse | +0.000000 | [+0.000202, +0.002004] |
| COMPACT | 9.272002 | 9.268026 | -0.003975 | FR worse | -0.014032 | [-0.001324, +0.000461] |
| SHUFFLED | 9.278272 | 9.265841 | -0.012431 | FR worse | -0.022488 | [-0.002273, -0.000376] |

## Numerical sensitivity (observed ranges over three deterministic groupings; not variance estimates)

- PPL range across cache variants: FR 0.010421, FC 0.012067 (= 1.04× and 1.20× |Δ ORIGINAL| = 0.010057)
- Largest change from ORIGINAL: 0.012067 (1.20× |Δ ORIGINAL|); largest |interaction| 0.022488 (2.24×)
- Range of FC − FR across variants: 0.022488; FC worse in 1/3 variants; ranking reverses: True
- Correlation of per-window FC − FR differences with ORIGINAL's: COMPACT +0.38, SHUFFLED +0.23
- the window bootstrap measures evaluation-text variability for a FIXED pair of quantized models; it says nothing about calibration-batch sensitivity

## Reconstruction (same 240 experts, same ~/moe_cache test rows for every variant)

- FR|COMPACT: per-expert |relative change| median 0.071 %, p90 0.165 %, max 0.286 %; worse for 48 % of experts
- FR|SHUFFLED: per-expert |relative change| median 0.064 %, p90 0.154 %, max 0.296 %; worse for 47 % of experts
- FC|COMPACT: per-expert |relative change| median 0.062 %, p90 0.154 %, max 0.306 %; worse for 48 % of experts
- FC|SHUFFLED: per-expert |relative change| median 0.072 %, p90 0.154 %, max 0.273 %; worse for 46 % of experts

## Activation drift (217 calibration rows, aligned by sequence id and position)

First layer input with any difference: {'COMPACT': 1, 'SHUFFLED': 1}

| layer | COMPACT rel. L2 | COMPACT % elements equal | SHUFFLED rel. L2 | SHUFFLED % elements equal | p99 token rel. L2 (C / S) |
|---|---|---|---|---|---|
| 0 | 0.00e+00 | 100.0 | 0.00e+00 | 100.0 | 0.00e+00 / 0.00e+00 |
| 1 | 4.32e-05 | 100.0 | 1.80e-04 | 99.9 | 1.65e-04 / 2.14e-04 |
| 2 | 1.73e-03 | 70.1 | 1.57e-03 | 69.7 | 8.15e-03 / 8.37e-03 |
| 4 | 6.10e-03 | 20.2 | 6.00e-03 | 20.2 | 6.54e-02 / 6.48e-02 |
| 8 | 5.68e-03 | 13.0 | 5.68e-03 | 13.0 | 7.99e-02 / 7.98e-02 |
| 12 | 9.71e-03 | 10.6 | 9.73e-03 | 10.7 | 8.67e-02 / 8.63e-02 |
| 16 | 1.48e-02 | 11.1 | 1.47e-02 | 11.2 | 7.37e-02 / 7.33e-02 |
| 20 | 1.86e-02 | 11.1 | 1.86e-02 | 11.2 | 7.15e-02 / 7.03e-02 |
| 23 | 2.15e-02 | 10.5 | 2.14e-02 | 10.5 | 7.41e-02 / 7.37e-02 |

NaN/Inf elements in alternative caches: 0

## Routing disagreement (E3 profiler replay, 217 calibration rows = 111,104 tokens per layer)

| layer | C: top-4 set changed | C: same set, reordered | C: mean overlap | S: top-4 set changed | S: same set, reordered | S: routing-distribution L1 |
|---|---|---|---|---|---|---|
| 0 | 0.000 % | 0.000 % | 1.00000 | 0.000 % | 0.000 % | 0.00e+00 |
| 1 | 0.603 % | 0.458 % | 0.99849 | 0.571 % | 0.482 % | 5.13e-04 |
| 2 | 3.184 % | 2.841 % | 0.99201 | 3.190 % | 2.745 % | 9.50e-04 |
| 4 | 4.647 % | 4.311 % | 0.98835 | 4.697 % | 4.369 % | 1.23e-03 |
| 8 | 3.963 % | 4.644 % | 0.99006 | 3.920 % | 4.746 % | 1.26e-03 |
| 12 | 3.989 % | 5.084 % | 0.98998 | 3.957 % | 5.066 % | 1.29e-03 |
| 16 | 3.875 % | 4.624 % | 0.99024 | 3.850 % | 4.548 % | 1.09e-03 |
| 20 | 5.722 % | 6.323 % | 0.98561 | 5.570 % | 6.221 % | 1.66e-03 |
| 23 | 6.674 % | 7.218 % | 0.98319 | 6.674 % | 7.170 % | 1.68e-03 |

## Hessian sensitivity (existing E3 Profiler Hessians; layers 1/8/16/23 × experts 0, 30)
Replay routing equals the Profiler's routed-token counts for 64/64 entries; max relative difference between the replay Gram matrix and the Profiler gu Hessian 1.9e-07; membership term larger than the common-token activation term in 32/32 gu comparisons; act-order permutation identical in 0/64.


| layer | strategy | variant | expert | proj | tokens O / V | membership change (−/+) | rel. Frob | diag rel | trace ratio | act-order perm identical | cols changing group | gu decomposition: common-token / membership |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FR | COMPACT | 0 | gu | 4539 / 4535 | 6 / 2 | 3.39e-03 | 8.98e-04 | 0.999129 | False | 4.2 % | 1.08e-04 / 3.38e-03 |
| 1 | FR | COMPACT | 0 | down | 4539 / 4535 | 6 / 2 | 5.34e-05 | 4.55e-05 | 0.999917 | False | 1.1 % | — |
| 1 | FR | SHUFFLED | 0 | gu | 4539 / 4537 | 5 / 3 | 3.27e-03 | 4.67e-04 | 0.999501 | False | 4.6 % | 1.27e-04 / 3.27e-03 |
| 1 | FR | SHUFFLED | 0 | down | 4539 / 4537 | 5 / 3 | 5.59e-05 | 4.87e-05 | 0.999922 | False | 0.9 % | — |
| 1 | FR | COMPACT | 30 | gu | 4719 / 4719 | 1 / 1 | 1.45e-03 | 2.49e-04 | 0.999970 | False | 1.6 % | 1.29e-04 / 1.45e-03 |
| 1 | FR | COMPACT | 30 | down | 4719 / 4719 | 1 / 1 | 1.53e-03 | 5.06e-04 | 0.999989 | False | 0.3 % | — |
| 1 | FR | SHUFFLED | 30 | gu | 4719 / 4717 | 3 / 1 | 2.37e-03 | 5.40e-04 | 0.999480 | False | 2.3 % | 1.59e-04 / 2.36e-03 |
| 1 | FR | SHUFFLED | 30 | down | 4719 / 4717 | 3 / 1 | 2.50e-03 | 6.59e-04 | 0.999824 | False | 0.3 % | — |
| 1 | FC | COMPACT | 0 | gu | 4542 / 4543 | 5 / 6 | 4.58e-03 | 1.07e-03 | 1.000089 | False | 4.4 % | 1.23e-04 / 4.58e-03 |
| 1 | FC | COMPACT | 0 | down | 4542 / 4543 | 5 / 6 | 1.42e-05 | 2.42e-06 | 1.000005 | False | 1.1 % | — |
| 1 | FC | SHUFFLED | 0 | gu | 4542 / 4542 | 5 / 5 | 3.94e-03 | 5.24e-04 | 1.000175 | False | 3.5 % | 1.21e-04 / 3.94e-03 |
| 1 | FC | SHUFFLED | 0 | down | 4542 / 4542 | 5 / 5 | 4.78e-05 | 4.60e-05 | 1.000045 | False | 0.9 % | — |
| 1 | FC | COMPACT | 30 | gu | 4244 / 4246 | 1 / 3 | 2.34e-03 | 6.69e-04 | 1.000356 | False | 2.1 % | 1.44e-04 / 2.33e-03 |
| 1 | FC | COMPACT | 30 | down | 4244 / 4246 | 1 / 3 | 1.94e-03 | 6.48e-04 | 1.000131 | False | 0.3 % | — |
| 1 | FC | SHUFFLED | 30 | gu | 4244 / 4245 | 1 / 2 | 2.05e-03 | 2.34e-04 | 1.000184 | False | 1.6 % | 1.55e-04 / 2.05e-03 |
| 1 | FC | SHUFFLED | 30 | down | 4244 / 4245 | 1 / 2 | 1.92e-03 | 7.21e-04 | 1.000112 | False | 0.1 % | — |
| 8 | FR | COMPACT | 0 | gu | 4659 / 4671 | 26 / 38 | 1.46e-02 | 3.69e-03 | 1.002588 | False | 13.3 % | 4.28e-03 / 1.39e-02 |
| 8 | FR | COMPACT | 0 | down | 4659 / 4671 | 26 / 38 | 3.20e-02 | 7.78e-03 | 1.002146 | False | 4.3 % | — |
| 8 | FR | SHUFFLED | 0 | gu | 4659 / 4662 | 29 / 32 | 1.40e-02 | 2.54e-03 | 1.000453 | False | 12.9 % | 4.22e-03 / 1.33e-02 |
| 8 | FR | SHUFFLED | 0 | down | 4659 / 4662 | 29 / 32 | 3.45e-02 | 9.37e-03 | 1.000979 | False | 4.5 % | — |
| 8 | FR | COMPACT | 30 | gu | 3968 / 3985 | 82 / 99 | 2.93e-02 | 8.81e-03 | 1.004735 | False | 21.0 % | 5.62e-03 / 2.87e-02 |
| 8 | FR | COMPACT | 30 | down | 3968 / 3985 | 82 / 99 | 4.53e-02 | 1.22e-02 | 1.001971 | False | 8.0 % | — |
| 8 | FR | SHUFFLED | 30 | gu | 3968 / 3968 | 76 / 76 | 2.63e-02 | 4.52e-03 | 1.000197 | False | 20.3 % | 5.49e-03 / 2.57e-02 |
| 8 | FR | SHUFFLED | 30 | down | 3968 / 3968 | 76 / 76 | 4.11e-02 | 1.15e-02 | 0.999508 | False | 7.2 % | — |
| 8 | FC | COMPACT | 0 | gu | 4064 / 4069 | 47 / 52 | 2.29e-02 | 2.75e-03 | 1.001359 | False | 17.3 % | 5.24e-03 / 2.23e-02 |
| 8 | FC | COMPACT | 0 | down | 4064 / 4069 | 47 / 52 | 4.25e-02 | 1.04e-02 | 1.000725 | False | 6.0 % | — |
| 8 | FC | SHUFFLED | 0 | gu | 4064 / 4065 | 53 / 54 | 2.30e-02 | 2.08e-03 | 1.000038 | False | 18.1 % | 5.27e-03 / 2.24e-02 |
| 8 | FC | SHUFFLED | 0 | down | 4064 / 4065 | 53 / 54 | 4.25e-02 | 1.06e-02 | 1.000172 | False | 6.0 % | — |
| 8 | FC | COMPACT | 30 | gu | 4704 / 4702 | 76 / 74 | 2.36e-02 | 8.05e-03 | 1.000557 | False | 14.7 % | 5.55e-03 / 2.30e-02 |
| 8 | FC | COMPACT | 30 | down | 4704 / 4702 | 76 / 74 | 1.59e-02 | 6.50e-03 | 1.002905 | False | 4.8 % | — |
| 8 | FC | SHUFFLED | 30 | gu | 4704 / 4698 | 73 / 67 | 2.20e-02 | 4.55e-03 | 0.999130 | False | 16.5 % | 5.51e-03 / 2.13e-02 |
| 8 | FC | SHUFFLED | 30 | down | 4704 / 4698 | 73 / 67 | 1.43e-02 | 3.36e-03 | 1.000823 | False | 5.4 % | — |
| 16 | FR | COMPACT | 0 | gu | 4909 / 4905 | 21 / 17 | 7.66e-03 | 5.88e-04 | 0.999131 | False | 8.4 % | 2.96e-03 / 7.08e-03 |
| 16 | FR | COMPACT | 0 | down | 4909 / 4905 | 21 / 17 | 2.38e-02 | 6.01e-03 | 0.999423 | False | 2.8 % | — |
| 16 | FR | SHUFFLED | 0 | gu | 4909 / 4906 | 23 / 20 | 8.06e-03 | 6.23e-04 | 0.999396 | False | 9.0 % | 2.92e-03 / 7.54e-03 |
| 16 | FR | SHUFFLED | 0 | down | 4909 / 4906 | 23 / 20 | 2.51e-02 | 6.39e-03 | 1.000261 | False | 2.5 % | — |
| 16 | FR | COMPACT | 30 | gu | 5439 / 5442 | 6 / 9 | 6.23e-03 | 1.19e-03 | 1.000551 | False | 5.8 % | 3.39e-03 / 5.22e-03 |
| 16 | FR | COMPACT | 30 | down | 5439 / 5442 | 6 / 9 | 1.49e-02 | 4.54e-03 | 1.000299 | False | 2.6 % | — |
| 16 | FR | SHUFFLED | 30 | gu | 5439 / 5437 | 7 / 5 | 5.97e-03 | 9.47e-04 | 0.999278 | False | 5.9 % | 3.30e-03 / 4.98e-03 |
| 16 | FR | SHUFFLED | 30 | down | 5439 / 5437 | 7 / 5 | 1.45e-02 | 4.03e-03 | 0.999148 | False | 3.1 % | — |
| 16 | FC | COMPACT | 0 | gu | 4316 / 4303 | 33 / 20 | 1.22e-02 | 1.85e-03 | 0.996851 | False | 12.2 % | 4.00e-03 / 1.15e-02 |
| 16 | FC | COMPACT | 0 | down | 4316 / 4303 | 33 / 20 | 2.87e-02 | 6.26e-03 | 0.998339 | False | 3.7 % | — |
| 16 | FC | SHUFFLED | 0 | gu | 4316 / 4300 | 33 / 17 | 1.21e-02 | 1.99e-03 | 0.995877 | False | 10.9 % | 4.05e-03 / 1.14e-02 |
| 16 | FC | SHUFFLED | 0 | down | 4316 / 4300 | 33 / 17 | 2.85e-02 | 7.28e-03 | 0.997406 | False | 4.3 % | — |
| 16 | FC | COMPACT | 30 | gu | 4482 / 4478 | 13 / 9 | 1.11e-02 | 2.23e-03 | 0.998903 | False | 8.7 % | 4.41e-03 / 1.01e-02 |
| 16 | FC | COMPACT | 30 | down | 4482 / 4478 | 13 / 9 | 1.80e-02 | 5.08e-03 | 0.999992 | False | 3.1 % | — |
| 16 | FC | SHUFFLED | 30 | gu | 4482 / 4475 | 17 / 10 | 1.11e-02 | 2.26e-03 | 0.998108 | False | 8.8 % | 4.35e-03 / 1.01e-02 |
| 16 | FC | SHUFFLED | 30 | down | 4482 / 4475 | 17 / 10 | 1.82e-02 | 4.67e-03 | 0.999154 | False | 2.7 % | — |
| 23 | FR | COMPACT | 0 | gu | 4570 / 4584 | 116 / 130 | 1.45e-02 | 4.28e-03 | 1.003350 | False | 14.2 % | 2.37e-03 / 1.43e-02 |
| 23 | FR | COMPACT | 0 | down | 4570 / 4584 | 116 / 130 | 4.11e-02 | 1.20e-02 | 1.004639 | False | 9.9 % | — |
| 23 | FR | SHUFFLED | 0 | gu | 4570 / 4583 | 127 / 140 | 1.52e-02 | 3.87e-03 | 1.003314 | False | 14.9 % | 2.34e-03 / 1.50e-02 |
| 23 | FR | SHUFFLED | 0 | down | 4570 / 4583 | 127 / 140 | 3.86e-02 | 1.20e-02 | 1.003632 | False | 11.4 % | — |
| 23 | FR | COMPACT | 30 | gu | 4013 / 4018 | 37 / 42 | 1.12e-02 | 1.67e-03 | 1.001218 | False | 9.9 % | 3.07e-03 / 1.07e-02 |
| 23 | FR | COMPACT | 30 | down | 4013 / 4018 | 37 / 42 | 2.58e-02 | 7.34e-03 | 1.000158 | False | 3.7 % | — |
| 23 | FR | SHUFFLED | 30 | gu | 4013 / 4019 | 43 / 49 | 1.11e-02 | 1.95e-03 | 1.001475 | False | 10.9 % | 3.03e-03 / 1.07e-02 |
| 23 | FR | SHUFFLED | 30 | down | 4013 / 4019 | 43 / 49 | 2.43e-02 | 6.98e-03 | 1.001302 | False | 3.1 % | — |
| 23 | FC | COMPACT | 0 | gu | 4574 / 4581 | 125 / 132 | 1.98e-02 | 5.54e-03 | 1.001449 | False | 14.7 % | 3.80e-03 / 1.94e-02 |
| 23 | FC | COMPACT | 0 | down | 4574 / 4581 | 125 / 132 | 4.65e-02 | 1.07e-02 | 1.001505 | False | 8.1 % | — |
| 23 | FC | SHUFFLED | 0 | gu | 4574 / 4576 | 121 / 123 | 1.93e-02 | 3.55e-03 | 1.000949 | False | 15.2 % | 3.70e-03 / 1.89e-02 |
| 23 | FC | SHUFFLED | 0 | down | 4574 / 4576 | 121 / 123 | 5.26e-02 | 1.07e-02 | 1.003136 | False | 8.7 % | — |
| 23 | FC | COMPACT | 30 | gu | 5202 / 5214 | 52 / 64 | 1.24e-02 | 2.82e-03 | 1.002531 | False | 9.4 % | 4.37e-03 / 1.17e-02 |
| 23 | FC | COMPACT | 30 | down | 5202 / 5214 | 52 / 64 | 1.61e-02 | 7.05e-03 | 1.002557 | False | 2.8 % | — |
| 23 | FC | SHUFFLED | 30 | gu | 5202 / 5210 | 54 / 62 | 1.20e-02 | 2.72e-03 | 1.001755 | False | 8.6 % | 4.36e-03 / 1.12e-02 |
| 23 | FC | SHUFFLED | 30 | down | 5202 / 5210 | 54 / 62 | 1.51e-02 | 6.52e-03 | 1.000263 | False | 3.1 % | — |

## Effective GPTQ weights vs ORIGINAL

| strategy | variant | % weights differ | rel. L2 | max abs | per-projection rel. L2 median [min, max] | identical projections |
|---|---|---|---|---|---|---|
| FR | COMPACT | 53.99 | 8.889e-02 | 0.083 | 8.943e-02 [2.156e-02, 1.137e-01] | 0/720 |
| FR | SHUFFLED | 53.96 | 8.881e-02 | 0.083 | 8.950e-02 [1.641e-02, 1.125e-01] | 0/720 |
| FC | COMPACT | 55.09 | 9.039e-02 | 0.0825 | 9.164e-02 [1.971e-02, 1.095e-01] | 0/720 |
| FC | SHUFFLED | 55.00 | 9.029e-02 | 0.0825 | 9.055e-02 [1.557e-02, 1.107e-01] | 0/720 |

not exposed by the frozen solver (gptq_quantize returns only the dequantized matrix); the act-order permutation / group assignment is compared on the Hessian subset

## Correctness and provenance

- ORIGINAL regression (fresh processes): weights, PPL and reconstruction bitwise equal to the partial-PPL run: True
- Negative control (ORIGINAL batches in reversed execution order): 24/24 layers bitwise equal; passed True
- COMPACT regenerated: cache bitwise equal to the v1 cache at all 24 layers: True; GPTQ weights equal to v1 regression: {'FR': True, 'FC': True}; PPL equal: {'FR': True, 'FC': True}
- Input semantics: token ids identical True; same model weights True; eval mode True; dtype torch.bfloat16; batch pattern COMPACT = SHUFFLED True; by construction: one set of layer kwargs (batch-1 causal mask, positions, rotary) broadcast over every batch; recorded in model_config.layer_kwargs
- FR|ORIGINAL: 240 experts, RTN fallbacks 0, unchanged 0, non-finite 0, untouched ok True, expert checks pass True (max 0.68 of tolerance), repeat eval |ΔPPL| 0.0e+00
- FR|COMPACT: 240 experts, RTN fallbacks 0, unchanged 0, non-finite 0, untouched ok True, expert checks pass True (max 0.81 of tolerance), repeat eval |ΔPPL| 0.0e+00
- FR|SHUFFLED: 240 experts, RTN fallbacks 0, unchanged 0, non-finite 0, untouched ok True, expert checks pass True (max 0.60 of tolerance), repeat eval |ΔPPL| 0.0e+00
- FC|ORIGINAL: 240 experts, RTN fallbacks 0, unchanged 0, non-finite 0, untouched ok True, expert checks pass True (max 0.90 of tolerance), repeat eval |ΔPPL| 0.0e+00
- FC|COMPACT: 240 experts, RTN fallbacks 0, unchanged 0, non-finite 0, untouched ok True, expert checks pass True (max 0.73 of tolerance), repeat eval |ΔPPL| 0.0e+00
- FC|SHUFFLED: 240 experts, RTN fallbacks 0, unchanged 0, non-finite 0, untouched ok True, expert checks pass True (max 0.62 of tolerance), repeat eval |ΔPPL| 0.0e+00

| config | quantize (s) | PPL eval (s) | peak GPU GB | peak CPU GB |
|---|---|---|---|---|
| FR|ORIGINAL | 290 | 53 | 8.4 | 43.8 |
| FR|COMPACT | 270 | 46 | 8.4 | 43.5 |
| FR|SHUFFLED | 276 | 42 | 8.4 | 43.1 |
| FC|ORIGINAL | 277 | 41 | 8.4 | 42.4 |
| FC|COMPACT | 269 | 40 | 8.4 | 43.3 |
| FC|SHUFFLED | 268 | 43 | 8.4 | 42.8 |

Figures: `figures/fig1_activation_drift`, `fig2_routing_disagreement`, `fig3_hessian_weight_sensitivity`, `fig4_ppl_by_batching`, `fig5_coverage_penalty_by_batching`, `fig6_recon_vs_ppl_change` (PDF + PNG).
