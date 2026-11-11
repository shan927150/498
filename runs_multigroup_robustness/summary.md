# Multi-grouping robustness: FR s0 vs FC s0 (four-layer GPTQ)

**Decision (rule frozen in protocol.json, sha256 `f97271a3a1e0d42c`): GROUPING-DEPENDENT-RANKING**

BF16 PPL 9.237090; tau = 1e-4 x BF16 = 0.000924; Delta_ORIGINAL = +0.010057.

| grouping | FR PPL | FC PPL | Delta_g | interaction | FC rank | window-bootstrap 95% CI (NLL diff) |
|---|---|---|---|---|---|---|
| G20261111 | 9.278182 | 9.274113 | -0.004069 | -0.014126 | FC better | [-0.00129, +0.00047] |
| G20261112 | 9.268159 | 9.275760 | +0.007601 | -0.002455 | FC worse | [-0.00002, +0.00166] |
| G20261113 | 9.271035 | 9.265146 | -0.005889 | -0.015946 | FC better | [-0.00152, +0.00024] |
| G20261114 | 9.278102 | 9.269707 | -0.008394 | -0.018451 | FC better | [-0.00174, -0.00004] |
| G20261115 | 9.272717 | 9.273808 | +0.001091 | -0.008966 | FC worse | [-0.00083, +0.00110] |

- mean Delta -0.001932, median -0.004069, SD 0.006364, min -0.008394, max +0.007601, range 0.015996
- FC worse in 2/5, better in 3/5; sign reversals vs ORIGINAL 3/5 (3 beyond tau)
- Range_FR 0.010023, Range_FC 0.010614; max |I_g| 0.018451; SD/|Delta_O| 0.63
- reconstruction FC-FR gap: G20261111 +2.54%, G20261112 +2.56%, G20261113 +2.55%, G20261114 +2.56%, G20261115 +2.55%, ORIGINAL +2.55%
- controls: {'original_reproduced': True, 'execution_order': True, 'repeat_ppl_identical': True, 'cache_semantics_valid': True}

## Historical comparison (different policies; not pooled)

| grouping | sample | policy | FR | FC | Delta |
|---|---|---|---|---|---|
| ORIGINAL | historical reference | cache_inputs.py, 176 x 8 consecutive original rows | 9.267851 | 9.277908 | +0.010057 |
| COMPACT | historical reference | 345 rows (217 union + 128 old test), sorted, 43 x 8 + 1 | 9.272002 | 9.268026 | -0.003975 |
| SHUFFLED | historical reference | 345 rows, default_rng(20261110), 43 x 8 + 1 | 9.278272 | 9.265841 | -0.012431 |
| G20261111 | primary randomized sample | 217 calibration rows only, default_rng(20261111), 27 x 8 + 1 | 9.278182 | 9.274113 | -0.004069 |
| G20261112 | primary randomized sample | 217 calibration rows only, default_rng(20261112), 27 x 8 + 1 | 9.268159 | 9.275760 | +0.007601 |
| G20261113 | primary randomized sample | 217 calibration rows only, default_rng(20261113), 27 x 8 + 1 | 9.271035 | 9.265146 | -0.005889 |
| G20261114 | primary randomized sample | 217 calibration rows only, default_rng(20261114), 27 x 8 + 1 | 9.278102 | 9.269707 | -0.008394 |
| G20261115 | primary randomized sample | 217 calibration rows only, default_rng(20261115), 27 x 8 + 1 | 9.272717 | 9.273808 | +0.001091 |
