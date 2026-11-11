# Cross-seed x calibration-grouping robustness (FR vs FC, four-layer GPTQ)

**Decision (rule frozen in protocol.json, sha256 `404a0274a7309c3b`): GROUPING-VARIATION-DOMINATES**

Decision details: {'confident_positive_seeds': [], 'confident_negative_seeds': [], 'scale_disagreement': False, 'frac_opposite_sign_to_overall': 0.4, 'abs_M_over_W': 0.18154771826718377, 'ambiguous': False}

| seed | grouping | FR PPL | FC PPL | FC − FR |
|---|---|---|---|---|
| 0 | 20261111 | 9.278182 | 9.274113 | -0.004069 |
| 0 | 20261112 | 9.268159 | 9.275760 | +0.007601 |
| 0 | 20261113 | 9.271035 | 9.265146 | -0.005889 |
| 0 | 20261114 | 9.278102 | 9.269707 | -0.008394 |
| 0 | 20261115 | 9.272717 | 9.273808 | +0.001091 |
| 1 | 20261111 | 9.266099 | 9.265572 | -0.000527 |
| 1 | 20261112 | 9.267534 | 9.263857 | -0.003677 |
| 1 | 20261113 | 9.265318 | 9.268279 | +0.002961 |
| 1 | 20261114 | 9.267026 | 9.275531 | +0.008505 |
| 1 | 20261115 | 9.273307 | 9.269481 | -0.003826 |
| 2 | 20261111 | 9.262634 | 9.272568 | +0.009934 |
| 2 | 20261112 | 9.268870 | 9.269566 | +0.000696 |
| 2 | 20261113 | 9.282398 | 9.268849 | -0.013549 |
| 2 | 20261114 | 9.271457 | 9.266417 | -0.005041 |
| 2 | 20261115 | 9.272849 | 9.268318 | -0.004531 |

| seed | mean Δ | SD (ddof 1) | FC worse / better | ORIGINAL-extraction Δ |
|---|---|---|---|---|
| 0 | -0.001932 | 0.006364 | 2/3 | +0.010057 |
| 1 | +0.000687 | 0.005176 | 2/3 | +0.002706 |
| 2 | -0.002498 | 0.008624 | 2/3 | +0.013070 |

- overall mean Δ -0.001248; overall median -0.003677; min -0.013549, max +0.009934; FC worse 6/15
- pooled within-seed SD 0.006872; SD of seed means 0.001699 (grouping-only expectation 0.003073); between-seed component SD ≈ 0.000000 (descriptive)
- hierarchical bootstrap 95% interval [-0.004777, +0.002155] (10 distinct seed resamples; unstable, descriptive)
- checks: {'seed0_reuse_valid': True, 'new_caches_valid': True, 'repeat_quant_control': True, 'repeat_ppl_identical': True, 'selection_hashes_match': True, 'manifests_match': True, 'windows_tokens': True}

