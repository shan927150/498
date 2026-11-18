# Inference-time routing amplification MVP - summary

Decision: **ROUTING-AMPLIFICATION-SUPPORTED** (conclusion A); R_SD = 0.3594; protocol sha256 79849e02531c1cb5...

## Parity

| test | result |
|---|---|
| A record-only, BF16 | per-window bitwise True, layer hashes True |
| A record-only, FR_O20261121 | per-window bitwise True, layer hashes True, vs historical True |
| B BF16 self-routing | per-window True, per-token True, 24 layer hashes True, final True |
| C quantized self-routing (FR_O20261121) | per-window True, per-token True, layer hashes True, selections True, router probs True, PPL True |

Historical regression: all natural evaluations reproduce historical per-window NLL bitwise: True.

## Routing divergence vs BF16 (natural inference)

| config | % tokens set changed (all layers) | % tokens any layer changed | first layer with change | L0/L1 identical | max-layer % |
|---|---|---|---|---|---|
| FR_O20261121 | 8.947 | 82.37 | 2 | True | L2: 14.156 |
| FC_O20261121 | 8.777 | 81.98 | 2 | True | L2: 13.864 |
| FR_O20261122 | 8.936 | 82.51 | 2 | True | L2: 14.171 |
| FC_O20261122 | 8.767 | 81.88 | 2 | True | L2: 13.892 |
| FR_O20261123 | 8.936 | 82.48 | 2 | True | L2: 14.197 |
| FC_O20261123 | 8.780 | 82.01 | 2 | True | L2: 13.849 |
| FR_O20261124 | 8.950 | 82.40 | 2 | True | L2: 14.122 |
| FC_O20261124 | 8.782 | 82.14 | 2 | True | L2: 13.805 |
| FR_O20261125 | 8.949 | 82.52 | 2 | True | L2: 14.160 |
| FC_O20261125 | 8.773 | 82.00 | 2 | True | L2: 13.901 |
| FR_ORIGINAL | 8.955 | 82.46 | 2 | True | L2: 14.197 |
| FC_ORIGINAL | 8.790 | 82.11 | 2 | True | L2: 13.867 |
| FR_FP64 | 8.944 | 82.50 | 2 | True | L2: 14.195 |
| FC_FP64 | 8.750 | 81.94 | 2 | True | L2: 13.832 |

## PPL (natural vs BF16-frozen routing)

| order | FR nat | FR frozen | FC nat | FC frozen | Delta nat | Delta frozen | delta_g |
|---|---|---|---|---|---|---|---|
| 20261121 | 9.267791 | 9.268410 | 9.275571 | 9.278260 | +0.007779 | +0.009850 | -0.002070 |
| 20261122 | 9.267245 | 9.269271 | 9.279571 | 9.276344 | +0.012326 | +0.007073 | +0.005254 |
| 20261123 | 9.270495 | 9.270318 | 9.277078 | 9.278128 | +0.006583 | +0.007810 | -0.001226 |
| 20261124 | 9.271444 | 9.269121 | 9.276058 | 9.277549 | +0.004614 | +0.008428 | -0.003814 |
| 20261125 | 9.269601 | 9.269958 | 9.278450 | 9.277910 | +0.008849 | +0.007952 | +0.000897 |

SD(Delta) natural 0.002870 (historical 0.002870), frozen 0.001032, **R_SD = 0.3594**. SD FR 0.001776 -> 0.000746; SD FC 0.001662 -> 0.000772. Mean Delta +0.008030 -> +0.008222; positive Delta 5/5 natural, 5/5 frozen.

## References (descriptive; not in the 5-order SD)

| reference | FR nat | FR frozen | FC nat | FC frozen | Delta nat | Delta frozen |
|---|---|---|---|---|---|---|
| ORIGINAL | 9.267851 | 9.270037 | 9.277908 | 9.278776 | +0.010057 | +0.008739 |
| FP64 | 9.271421 | 9.268393 | 9.278551 | 9.277924 | +0.007129 | +0.009531 |

## Token-level NLL vs ORIGINAL (mean over the 10 primary configs)

- natural: mean |dNLL| 3.19e-02, median 1.34e-02, p99 1.98e-01, max 1.606; top 1% tokens carry 9.0% and top 10% carry 42.0% of total |dNLL|
- frozen: mean |dNLL| 2.43e-02, median 9.20e-03, p99 1.26e-01, max 0.481; top 1% tokens carry 6.7% and top 10% carry 41.0% of total |dNLL|

## Correctness

| check | pass |
|---|---|
| 01_historical_weight_hashes_match | True |
| 02_no_quantized_weights_regenerated | True |
| 03_all_240_experts_intended_weights | True |
| 04_non_target_params_unchanged | True |
| 05_windows_match_historical_hash | True |
| 06_bf16_ppl_matches_historical | True |
| 07_record_only_matches_normal | True |
| 08_bf16_self_routing_parity | True |
| 09_quant_self_routing_parity | True |
| 10_router_logits_computed_normally | True |
| 11_router_probs_computed_normally | True |
| 12_only_selected_indices_overridden | True |
| 13_current_probs_used_for_forced | True |
| 14_topk_normalisation_preserved | True |
| 15_dispatch_and_accumulation_order_preserved | True |
| 16_all_24_layers_covered | True |
| 17_reference_aligned_to_windows_positions | True |
| 18_no_repeated_indices | True |
| 19_valid_expert_ids | True |
| 20_hooks_change_no_parameters | True |
| 21_natural_ppl_reproduces_order_null | True |
| 22_original_fp64_reproduce_historical | True |
| 23_token_nll_sums_reproduce_window_nll | True |
| 24_window_nll_sums_reproduce_ppl | True |
| 25_natural_frozen_identical_weights | True |
| 26_no_old_artifacts_overwritten | True |
| 27_targeted_tests_pass | True |
| 28_full_test_suite_passes | True |
| frozen_dispatch_equals_bf16 | True |
| bf16_end_of_process_bitwise | True |
| no_missing_runs | True |

## Cost

natural eval 35.1 s, frozen 53.2 s, historical plain 51.7 s; peak GPU 3.32 GB; peak CPU 50.3 GB; storage 1.67 GB; GPTQ solves 0.

