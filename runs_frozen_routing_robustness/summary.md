# Frozen-routing robustness across five calibration groupings (FR s0 / FC s0, four layers)

**Decision (rule frozen in protocol.json, sha256 `21ba43d82773934e`): RUN-BLOCKED**

- blocker: FC_AO_G20261114 installation check L1 e0: installed_vs_e3_rel 0.005595 > tolerance 0.005202 (1.075x); installed == BF16(E3 Wq): True
- conditional (NOT the decision; failed configuration accepted as is): FROZEN-ROUTING-PARTIALLY-STABILIZES ['b_sd_ratio_FC', 'e_ranking']

All numbers below include the blocked configuration (descriptive only).

| grouping | FR OO | FR AA_g | FR AO_g | FC OO | FC AA_g | FC AO_g | Δ_AA,g | Δ_AO,g |
|---|---|---|---|---|---|---|---|---|
| G20261111 | 9.267851 | 9.278182 | 9.265861 | 9.277908 | 9.274113 | 9.277881 | -0.004069 | +0.012020 |
| G20261112 | 9.267851 | 9.268159 | 9.270916 | 9.277908 | 9.275760 | 9.274432 | +0.007601 | +0.003516 |
| G20261113 | 9.267851 | 9.271035 | 9.276043 | 9.277908 | 9.265146 | 9.273312 | -0.005889 | -0.002731 |
| G20261114 | 9.267851 | 9.278102 | 9.273279 | 9.277908 | 9.269707 | 9.274137 | -0.008394 | +0.000858 |
| G20261115 | 9.267851 | 9.272717 | 9.268226 | 9.277908 | 9.273808 | 9.274729 | +0.001091 | +0.006503 |

| quantity | AA | AO | ratio AO/AA |
|---|---|---|---|
| FR SD | 0.004422 | 0.004020 | 0.909 |
| FR range | 0.010023 | 0.010182 | 1.016 |
| FR mean |dist to OO| | 0.005788 | 0.003810 | AO closer in 3/5 |
| FC SD | 0.004292 | 0.001749 | 0.408 |
| FC range | 0.010614 | 0.004568 | 0.430 |
| FC mean |dist to OO| | 0.006201 | 0.003010 | AO closer in 4/5 |
| Δ SD | 0.006364 | 0.005612 | 0.882 |
| Δ mean | -0.001932 | +0.004033 | Δ_OO +0.010057 |
| ranking preserved vs OO | 2/5 | 4/5 | |

criteria: sd_ratio_FR=False, sd_ratio_FC=True, sd_delta_ratio=False, closer_FR=False, closer_FC=True, ao_ranking_preserved=True, ao_ranking_preserved_more_than_aa=True

| strategy | grouping | Hessian median AO vs OO | AA vs OO | cols changing group AO / AA | weights differ AO / AA | rel L2 AO / AA | recon AO / AA rel vs OO |
|---|---|---|---|---|---|---|---|
| FR | G20261111 | 0.0039 | 0.0141 | 2.4% / 7.3% | 39.2% / 54.1% | 0.0694 / 0.0889 | -0.0098% / -0.0053% |
| FR | G20261112 | 0.0038 | 0.0144 | 2.4% / 7.3% | 39.3% / 54.1% | 0.0695 / 0.0889 | -0.0034% / -0.0081% |
| FR | G20261113 | 0.0038 | 0.0140 | 2.4% / 7.3% | 39.0% / 54.2% | 0.0692 / 0.0890 | -0.0020% / -0.0077% |
| FR | G20261114 | 0.0038 | 0.0137 | 2.4% / 7.2% | 39.2% / 54.0% | 0.0693 / 0.0888 | -0.0044% / -0.0088% |
| FR | G20261115 | 0.0039 | 0.0147 | 2.4% / 7.3% | 39.1% / 54.0% | 0.0693 / 0.0888 | -0.0054% / -0.0049% |
| FC | G20261111 | 0.0049 | 0.0158 | 2.6% / 7.6% | 40.2% / 55.1% | 0.0711 / 0.0904 | -0.0096% / -0.0097% |
| FC | G20261112 | 0.0049 | 0.0150 | 2.6% / 7.6% | 40.0% / 55.2% | 0.0710 / 0.0905 | -0.0019% / +0.0013% |
| FC | G20261113 | 0.0049 | 0.0143 | 2.6% / 7.6% | 40.2% / 55.2% | 0.0711 / 0.0906 | -0.0004% / -0.0053% |
| FC | G20261114 | 0.0050 | 0.0147 | 2.6% / 7.6% | 40.3% / 55.1% | 0.0712 / 0.0905 | -0.0054% / -0.0021% |
| FC | G20261115 | 0.0049 | 0.0155 | 2.6% / 7.6% | 40.5% / 55.2% | 0.0713 / 0.0904 | -0.0051% / +0.0001% |
