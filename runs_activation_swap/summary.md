# Activation-aware calibration swap pilot — summary

Layer 16, 8 experts, allocation seed 0, B = 128, frozen 4-bit GPTQ (g128, percdamp 1.0, act-order). Reconstruction on the OLD test split (rows 2128–2255; exploratory, previously inspected). Reference statistics from the tuning split (rows 2000–2127; previously used). **Exploratory pilot; no significance claims.**

**Decision (rule frozen in proposals.json): ACTIVATION-SIGNAL-INCONCLUSIVE**

- Experts (8 most-routed in layer 16 over the original pool): [30, 53, 9, 42, 24, 50, 16, 0]
- Baseline S0 = matched Full Random seed 0 (hash ec355fb8d866e4ff); D(S0) = 0.00165815; E0 = 0.004688 (8-expert mean; per-expert values match the stored full run to 0.0e+00).
- Proposals: 1024 (16 out × 64 in); predicted gain > 0: 522, < 0: 502.
- Profiling: 192 sequences in 8.5 s + reference (128 tuning seqs) 5.3 s on one layer; statistics file 25.3 MB. GPTQ evaluations: 10 (Hessian replay ≈ 7.1 s, GPTQ of 8 experts ≈ 8.7 s each; peak GPU 7.8 GB).

## Nine swaps

| swap | out → in | predicted gain | actual gain (×1e-6) | rel. change | experts improved | Δ tokens (8 experts) | Δ mean d(s) | Δ agg. L1 (L16) |
|---|---|---|---|---|---|---|---|---|
| beneficial0 | 988 → 1654 | +2.386e-04 | +3.58 | -0.076 % | 5/8 | -132 | +0.00220 | -0.00083 |
| beneficial1 | 674 → 1714 | +1.533e-04 | -0.13 | +0.003 % | 4/8 | -48 | +0.00073 | -0.00021 |
| beneficial2 | 887 → 32 | +1.342e-04 | +1.79 | -0.038 % | 5/8 | -66 | +0.00107 | -0.00171 |
| harmful0 | 1008 → 1981 | -1.403e-04 | +1.83 | -0.039 % | 6/8 | -82 | -0.00079 | +0.00161 |
| harmful1 | 872 → 432 | -1.276e-04 | +1.12 | -0.024 % | 5/8 | +48 | +0.00021 | +0.00153 |
| harmful2 | 797 → 1932 | -1.060e-04 | +2.01 | -0.043 % | 5/8 | +144 | -0.00126 | +0.00115 |
| random0 | 1463 → 920 | -1.536e-06 | +0.02 | -0.001 % | 4/8 | +260 | -0.00390 | +0.00186 |
| random1 | 517 → 1676 | +8.942e-05 | +2.71 | -0.058 % | 6/8 | -24 | +0.00035 | -0.00127 |
| random2 | 1284 → 235 | +2.328e-05 | +3.96 | -0.085 % | 7/8 | +27 | +0.00081 | -0.00084 |

**Spearman(predicted, actual) = +0.117** (n = 9).

| group | mean actual gain (×1e-6) | swaps improving |
|---|---|---|
| beneficial | +1.75 | 2/3 |
| harmful | +1.65 | 3/3 |
| random | +2.23 | 3/3 |

Random-group SD of actual gain: 2.01 ×1e-6 (noise reference).

## Per-expert

| expert | S0 error | Spearman (per-expert predicted vs actual) | share of mean |Δ error| |
|---|---|---|---|
| 30 | 0.004369 | +0.05 | 5.8 % |
| 53 | 0.004725 | -0.37 | 11.6 % |
| 9 | 0.004376 | -0.02 | 17.4 % |
| 42 | 0.003939 | +0.55 | 22.1 % |
| 24 | 0.004817 | -0.02 | 12.8 % |
| 50 | 0.004867 | -0.20 | 8.7 % |
| 16 | 0.005629 | -0.63 | 12.9 % |
| 0 | 0.004781 | +0.15 | 8.8 % |

Figures: `figures/predicted_vs_actual.{pdf,png}`, `figures/gain_by_group.{pdf,png}`.
