# GPTQ act-order ablation after frozen-routing calibration (FR s0 / FC s0, four layers, five groupings)

**Decision (rule frozen in protocol.json, sha256 `6fd73236771cdcc0`): ACTORDER-CONTRIBUTES-TO-WEIGHT-SENSITIVITY-ONLY**

- installation audit: AUDIT-CONFIRMED-BF16-REPRESENTATION; earlier study decision unchanged: RUN-BLOCKED
- criteria: W=True, P=False, D=True, W_none=False, P_none=False

| grouping | FR AO | FR AOF | FC AO | FC AOF | Δ_AO | Δ_AOF |
|---|---|---|---|---|---|---|
| OO | 9.267851 | | 9.277908 | | +0.010057 | |
| G20261111 | 9.265861 | 9.266665 | 9.277881 | 9.272543 | +0.012020 | +0.005878 |
| G20261112 | 9.270916 | 9.268927 | 9.274432 | 9.277243 | +0.003516 | +0.008316 |
| G20261113 | 9.276043 | 9.265309 | 9.273312 | 9.271201 | -0.002731 | +0.005892 |
| G20261114 | 9.273279 | 9.264406 | 9.274137 | 9.273505 | +0.000858 | +0.009099 |
| G20261115 | 9.268226 | 9.270099 | 9.274729 | 9.275548 | +0.006503 | +0.005449 |

| quantity | AO | AOF | ratio |
|---|---|---|---|
| FR SD | 0.004020 | 0.002397 | 0.596 |
| FR range | 0.010182 | 0.005693 | 0.559 |
| FR mean \|PPL−OO\| | 0.003810 | 0.002099 | AOF closer 4/5 |
| FR weight rel L2 vs OO (mean) | 0.0694 | 0.0405 | reduction +0.416 |
| FR elements ≠ OO (mean) | 39.2% | 16.9% | |
| FC SD | 0.001749 | 0.002404 | 1.375 |
| FC range | 0.004568 | 0.006042 | 1.323 |
| FC mean \|PPL−OO\| | 0.003010 | 0.003900 | AOF closer 2/5 |
| FC weight rel L2 vs OO (mean) | 0.0711 | 0.0419 | reduction +0.410 |
| FC elements ≠ OO (mean) | 40.2% | 17.9% | |
| Δ SD | 0.005612 | 0.001658 | 0.296 |
| ranking preserved vs OO | 4/5 | 5/5 | (AA 2/5) |
