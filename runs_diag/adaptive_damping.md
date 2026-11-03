# Exp5 — adaptive damping vs frozen percdamp = 1.0

Grid [0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]; seeds [0, 1]; layers [1, 8, 16, 23]; each rule chosen on the other seed. Gain = reduction of mean tuning error vs the frozen value.

| sequences | frozen err | per budget | per layer+budget | per token bin | per-expert oracle | doubling the data | best value (all seeds) |
|---|---|---|---|---|---|---|---|
| 16 | 0.00539 | +2.36% | +2.48% | +2.32% | +2.68% | +5.73% | 3.0 |
| 32 | 0.00508 | +0.95% | +1.13% | +0.95% | +1.34% | +5.20% | 2.0 |
| 64 | 0.00482 | +0.00% | +0.16% | +0.05% | +0.44% | +4.54% | 1.0 |
| 128 | 0.00460 | +0.00% | +0.42% | +0.05% | +0.60% | +3.11% | 1.0 |
| 256 | 0.00445 | +1.50% | +1.71% | +1.50% | +1.84% | – | 0.5 |

All per-budget optima are interior.
