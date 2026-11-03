# Exp1 — within-layer ranking check (b4_d1)

Validation: expert-grouped 5-fold.

Layer-aware minus coverage capture: pooled across layers +3.12 pp (wins 17/20 cells), within layers +0.00 pp (wins 0/20).

| rule | pooled capture | within-layer capture | within-layer Spearman vs realized | within-layer rank agreement with coverage |
|---|---|---|---|---|
| coverage | 0.9314 | 0.9613 | 0.762 | 1.000 |
| global | 0.9314 | 0.9596 | 0.762 | 1.000 |
| layer_aware | 0.9627 | 0.9613 | 0.762 | 1.000 |
| oracle | 1.0000 | 1.0000 | 1.000 | 0.762 |

Reading: if the within-layer gain is ~0 and rank agreement with coverage is ~1, the P2c advantage came from cross-layer comparisons that sequence selection cannot act on; E3 can then only profit through cross-layer weighting inside the LP.
