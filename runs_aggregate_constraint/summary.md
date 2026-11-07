# Aggregate-shift-constrained Coverage-LP — locked confirmation summary

Protocol `locked_protocol.json` sha256 `1abbe55aa734c3c5…`, commit `7ec7aa8139dc`. Mean relative expert reconstruction error on the **locked confirmation split** (256 new, document-disjoint C4 windows); 240 routed experts, layers 1/8/16/23; partial-model non-sequential expert GPTQ (attention and shared experts BF16). Lower is better.

**Decision (rule frozen in the protocol): AGGREGATE-ALLOCATION-PARTIAL**

Valid runs: 15 / 15; evaluation universe 240 experts (identical for all runs).

eps_L (median of 3,000 random completions of the fixed warm starts): L1 0.02131, L8 0.03392, L16 0.02892, L23 0.04031

## Reconstruction (confirmation split)

| strategy | mean | SD | s0 | s1 | s2 | expert median | p90 | p95 | L1 | L8 | L16 | L23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Typical Random | 0.004559 | 0.000008 | 0.004566 | 0.004561 | 0.004550 | 0.004747 | 0.005916 | 0.006168 | 0.003572 | 0.004621 | 0.005087 | 0.004958 |
| Typical Coverage-LP | 0.004562 | 0.000003 | 0.004559 | 0.004565 | 0.004561 | 0.004751 | 0.005922 | 0.006175 | 0.003582 | 0.004627 | 0.005090 | 0.004949 |
| Full Random | 0.004580 | 0.000011 | 0.004573 | 0.004592 | 0.004575 | 0.004768 | 0.005939 | 0.006188 | 0.003591 | 0.004639 | 0.005103 | 0.004987 |
| **Aggregate Coverage-LP** | 0.004629 | 0.000017 | 0.004647 | 0.004614 | 0.004626 | 0.004815 | 0.006005 | 0.006267 | 0.003628 | 0.004679 | 0.005168 | 0.005042 |
| Full Coverage-LP | 0.004708 | 0.000006 | 0.004713 | 0.004702 | 0.004709 | 0.004888 | 0.006103 | 0.006366 | 0.003690 | 0.004760 | 0.005264 | 0.005118 |

## Paired differences: Aggregate Coverage-LP − baseline

| baseline | Δ_s (rel.) s0 / s1 / s2 | mean | seeds AG better | per-layer mean (L1/L8/L16/L23) | p95 rel |
|---|---|---|---|---|---|
| Typical Random | +1.77 % / +1.16 % / +1.67 % | +1.53 % | 0/3 | +1.56 % / +1.26 % / +1.60 % / +1.70 % | +1.61 % |
| Typical Coverage-LP | +1.93 % / +1.07 % / +1.44 % | +1.48 % | 0/3 | +1.27 % / +1.13 % / +1.54 % / +1.89 % | +1.49 % |
| Full Random | +1.63 % / +0.48 % / +1.12 % | +1.08 % | 0/3 | +1.02 % / +0.87 % / +1.27 % / +1.11 % | +1.27 % |
| Full Coverage-LP | -1.40 % / -1.86 % / -1.75 % | -1.67 % | 3/3 | -1.70 % / -1.69 % / -1.82 % / -1.49 % | -1.55 % |

## Selection: routing shift, typicality, exposure (mean over seeds)

| strategy | aggregate L1 (L1/L8/L16/L23) | mean L1 | within all eps | set JS | mean d(s) | post-warm d | Gini | median tokens | median eff. sources |
|---|---|---|---|---|---|---|---|---|---|
| Typical Random | 0.0322 / 0.0464 / 0.0477 / 0.0558 | 0.0455 | no | 0.00045 | 0.2891 | 0.2888 | 0.0654 | 4310 | 120.9 |
| Typical Coverage-LP | 0.0237 / 0.0330 / 0.0301 / 0.0524 | 0.0348 | no | 0.00030 | 0.2953 | 0.2972 | 0.0457 | 4330 | 120.4 |
| Full Random | 0.0221 / 0.0319 / 0.0324 / 0.0423 | 0.0322 | no | 0.00023 | 0.3242 | 0.3357 | 0.0543 | 4307 | 118.5 |
| Aggregate Coverage-LP | 0.0205 / 0.0333 / 0.0277 / 0.0393 | 0.0302 | yes | 0.00023 | 0.3666 | 0.3921 | 0.0333 | 4340 | 115.2 |
| Full Coverage-LP | 0.0553 / 0.0957 / 0.1035 / 0.1003 | 0.0887 | no | 0.00154 | 0.4092 | 0.4490 | 0.0427 | 4345 | 112.0 |

Post-warm overlap of AG with: Full Random 0.042; Typical Random 0.024; Typical Coverage-LP 0.017; Full Coverage-LP 0.118

## Controller (coverage_lp_aggregate)

| seed | round | naive rounding feasible | repaired | swaps | LP obj | naive obj | final obj | loss vs naive | MILP status / gap |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | True | False | 0 | 240.00 | 239.90 | 239.90 | +0.00 % | — |
| 0 | 1 | False | True | 32 | 147.16 | 147.33 | 146.87 | +0.31 % | 4 / 1.99e-03 |
| 1 | 0 | True | False | 0 | 240.00 | 239.85 | 239.85 | +0.00 % | — |
| 1 | 1 | False | True | 5 | 146.50 | 146.39 | 146.25 | +0.10 % | 4 / 1.71e-03 |
| 2 | 0 | True | False | 0 | 240.00 | 239.88 | 239.88 | +0.00 % | — |
| 2 | 1 | False | True | 28 | 146.78 | 146.77 | 146.51 | +0.18 % | 4 / 1.87e-03 |

## Compute

- Routing profile of the 2,000 candidates: ~357 s full-model-forward equivalent (needed by TR, TC, FC, AG; not by FR).
- eps construction 0.2 s; AG selection per seed 368 s, 297 s, 351 s (LP 5.7, 5.8, 6.0 s; integer certificates 8.5, 7.5, 8.1 s; repair MILP 349, 278, 328 s).
- Hessian replay / GPTQ per run (s): Full Random 53 / 334; Typical Random 53 / 333; Typical Coverage-LP 51 / 333; Full Coverage-LP 51 / 331; Aggregate Coverage-LP 53 / 330; peak GPU 7.8 GB.
- Equal selected-sequence budget (B = 128) is not equal compute: every routing-based strategy pays the ~357 s profile (≈ 2,000 sequences of forward passes) plus selection time.

Figures: `figures/confirmation_recon.{pdf,png}`, `figures/shift_vs_error.{pdf,png}`, `figures/per_layer_paired.{pdf,png}`.
