# Extreme low-budget (B = 4 / 8) scarcity study — summary (CPU, routing counts only; no GPTQ)

**Decision (prespecified gate): EXTREME-SCARCITY-NO-OPPORTUNITY**

All 24 matched selections valid: True. Schedules: B = 4: warm 1, rounds [1, 2]; B = 8: warm 2, rounds [3, 3]. Controller unchanged; K passed = {'4': 1, '8': 2} (see module docstring).

## Random Monte Carlo (1,000 sets each) — headline

| pool | B | median min tokens [q05, q95] | median p10 | P(any zero) | P(any < 64) | P(any < 128) | P(≥ 10 % < 128) |
|---|---|---|---|---|---|---|---|
| full | 4 | 52 [28, 70] | 98 | 0.000 | 0.833 | 1.000 | 1.000 |
| full | 8 | 139 [93, 171] | 214 | 0.000 | 0.004 | 0.301 | 0.000 |
| typical_half | 4 | 57 [36, 74] | 100 | 0.000 | 0.728 | 1.000 | 1.000 |
| typical_half | 8 | 149 [108, 173] | 215 | 0.000 | 0.001 | 0.159 | 0.000 |

## Matched selections (all 240 experts)

| B | seed | strategy | min | p5 | p10 | median | zero | < 64 | < 128 | < 256 | Gini (layer mean) | eff. sources (median) | post-warm d(s) | set JS | agg. L1 (L1/L8/L16/L23) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0 | Full Random | 59 | 91 | 100 | 131 | 0 | 1 | 105 | 239 | 0.1287 | 3.80 | 0.345 | 0.00547 | 0.103 / 0.148 / 0.138 / 0.228 |
| 4 | 0 | Full Coverage-LP | 34 | 71 | 78 | 130 | 0 | 8 | 108 | 235 | 0.1929 | 3.91 | 0.346 | 0.01434 | 0.149 / 0.296 / 0.267 / 0.289 |
| 4 | 0 | Typical Random | 53 | 91 | 98 | 133 | 0 | 1 | 98 | 239 | 0.1294 | 3.87 | 0.290 | 0.00471 | 0.109 / 0.167 / 0.142 / 0.172 |
| 4 | 0 | Typical Coverage-LP | 51 | 71 | 88 | 129 | 0 | 8 | 116 | 236 | 0.1814 | 3.92 | 0.308 | 0.01126 | 0.159 / 0.262 / 0.243 / 0.262 |
| 4 | 1 | Full Random | 41 | 99 | 104 | 136 | 0 | 1 | 83 | 239 | 0.1054 | 3.85 | 0.299 | 0.00410 | 0.100 / 0.146 / 0.108 / 0.176 |
| 4 | 1 | Full Coverage-LP | 31 | 71 | 85 | 126 | 0 | 5 | 123 | 232 | 0.1897 | 3.90 | 0.317 | 0.01284 | 0.159 / 0.270 / 0.209 / 0.307 |
| 4 | 1 | Typical Random | 51 | 83 | 92 | 131 | 0 | 4 | 106 | 238 | 0.1556 | 3.90 | 0.290 | 0.00709 | 0.115 / 0.206 / 0.156 / 0.233 |
| 4 | 1 | Typical Coverage-LP | 34 | 76 | 86 | 132 | 0 | 5 | 110 | 237 | 0.1682 | 3.90 | 0.300 | 0.01034 | 0.146 / 0.234 / 0.172 / 0.310 |
| 4 | 2 | Full Random | 48 | 89 | 101 | 136 | 0 | 3 | 92 | 240 | 0.1156 | 3.85 | 0.316 | 0.00560 | 0.100 / 0.154 / 0.141 / 0.198 |
| 4 | 2 | Full Coverage-LP | 41 | 79 | 94 | 135 | 0 | 6 | 106 | 238 | 0.1535 | 3.93 | 0.270 | 0.00825 | 0.132 / 0.250 / 0.191 / 0.210 |
| 4 | 2 | Typical Random | 63 | 98 | 110 | 135 | 0 | 1 | 87 | 240 | 0.0955 | 3.86 | 0.297 | 0.00365 | 0.088 / 0.139 / 0.111 / 0.167 |
| 4 | 2 | Typical Coverage-LP | 45 | 84 | 95 | 131 | 0 | 3 | 109 | 238 | 0.1495 | 3.90 | 0.310 | 0.01006 | 0.116 / 0.223 / 0.230 / 0.271 |
| 8 | 0 | Full Random | 126 | 194 | 208 | 264 | 0 | 0 | 1 | 95 | 0.1102 | 7.61 | 0.331 | 0.00365 | 0.081 / 0.143 / 0.144 / 0.158 |
| 8 | 0 | Full Coverage-LP | 124 | 193 | 209 | 272 | 0 | 0 | 1 | 94 | 0.1042 | 7.46 | 0.385 | 0.00381 | 0.097 / 0.139 / 0.129 / 0.172 |
| 8 | 0 | Typical Random | 166 | 201 | 213 | 270 | 0 | 0 | 0 | 87 | 0.0973 | 7.64 | 0.296 | 0.00259 | 0.079 / 0.131 / 0.095 / 0.123 |
| 8 | 0 | Typical Coverage-LP | 104 | 213 | 230 | 272 | 0 | 0 | 1 | 73 | 0.0784 | 7.67 | 0.285 | 0.00301 | 0.071 / 0.108 / 0.099 / 0.144 |
| 8 | 1 | Full Random | 114 | 205 | 228 | 274 | 0 | 0 | 1 | 76 | 0.0770 | 7.58 | 0.317 | 0.00206 | 0.061 / 0.090 / 0.069 / 0.130 |
| 8 | 1 | Full Coverage-LP | 99 | 189 | 218 | 270 | 0 | 0 | 3 | 78 | 0.0997 | 7.47 | 0.381 | 0.00452 | 0.088 / 0.142 / 0.126 / 0.168 |
| 8 | 1 | Typical Random | 113 | 184 | 206 | 262 | 0 | 0 | 1 | 101 | 0.1222 | 7.68 | 0.302 | 0.00409 | 0.095 / 0.155 / 0.103 / 0.179 |
| 8 | 1 | Typical Coverage-LP | 123 | 201 | 223 | 274 | 0 | 0 | 1 | 73 | 0.0868 | 7.60 | 0.309 | 0.00326 | 0.082 / 0.126 / 0.113 / 0.146 |
| 8 | 2 | Full Random | 156 | 194 | 210 | 268 | 0 | 0 | 0 | 92 | 0.1062 | 7.59 | 0.369 | 0.00521 | 0.104 / 0.170 / 0.172 / 0.180 |
| 8 | 2 | Full Coverage-LP | 157 | 206 | 221 | 273 | 0 | 0 | 0 | 85 | 0.0879 | 7.54 | 0.335 | 0.00306 | 0.071 / 0.107 / 0.078 / 0.162 |
| 8 | 2 | Typical Random | 150 | 192 | 210 | 270 | 0 | 0 | 0 | 88 | 0.1036 | 7.71 | 0.298 | 0.00345 | 0.091 / 0.136 / 0.093 / 0.170 |
| 8 | 2 | Typical Coverage-LP | 165 | 211 | 220 | 274 | 0 | 0 | 0 | 75 | 0.0810 | 7.65 | 0.301 | 0.00233 | 0.061 / 0.093 / 0.082 / 0.132 |

## Paired Coverage − Random (same seed, same warm start)

| B | family | seed | Δmin | Δp5 | Δp10 (rel) | Δzero | Δ<64 | Δ<128 (rel. reduction) | Δ<256 | ΔGini | Δeff. sources | Δpost-warm d(s) | Δset JS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | full | 0 | -25 | -20 | -22 (-22.1 %) | +0 | +7 | +3 (-2.9 %) | -4 | +0.0642 | +0.10 | +0.000 | +0.00887 |
| 4 | full | 1 | -10 | -28 | -19 (-18.4 %) | +0 | +4 | +40 (-48.2 %) | -7 | +0.0843 | +0.05 | +0.018 | +0.00874 |
| 4 | full | 2 | -7 | -10 | -7 (-6.9 %) | +0 | +3 | +14 (-15.2 %) | -2 | +0.0379 | +0.08 | -0.046 | +0.00266 |
| 4 | typical | 0 | -2 | -20 | -10 (-10.3 %) | +0 | +7 | +18 (-18.4 %) | -3 | +0.0520 | +0.05 | +0.018 | +0.00655 |
| 4 | typical | 1 | -17 | -7 | -6 (-6.5 %) | +0 | +1 | +4 (-3.8 %) | -1 | +0.0126 | -0.00 | +0.011 | +0.00325 |
| 4 | typical | 2 | -18 | -14 | -15 (-13.7 %) | +0 | +2 | +22 (-25.3 %) | -2 | +0.0540 | +0.04 | +0.014 | +0.00642 |
| 8 | full | 0 | -2 | -1 | +1 (+0.4 %) | +0 | +0 | +0 (+0.0 %) | -1 | -0.0060 | -0.15 | +0.053 | +0.00016 |
| 8 | full | 1 | -15 | -16 | -10 (-4.3 %) | +0 | +0 | +2 (-200.0 %) | +2 | +0.0227 | -0.11 | +0.064 | +0.00246 |
| 8 | full | 2 | +1 | +12 | +11 (+5.2 %) | +0 | +0 | +0 (—) | -7 | -0.0183 | -0.05 | -0.034 | -0.00215 |
| 8 | typical | 0 | -62 | +12 | +17 (+7.9 %) | +0 | +0 | +1 (—) | -14 | -0.0189 | +0.03 | -0.012 | +0.00042 |
| 8 | typical | 1 | +10 | +17 | +17 (+8.2 %) | +0 | +0 | +0 (+0.0 %) | -28 | -0.0354 | -0.08 | +0.006 | -0.00083 |
| 8 | typical | 2 | +15 | +19 | +10 (+4.8 %) | +0 | +0 | +0 (—) | -13 | -0.0225 | -0.06 | +0.003 | -0.00111 |
| 16 (ref.) | full | 0 | +47 | +28 | +40 (+9.3 %) | +0 | +0 | +0 (—) | +0 | -0.0302 | -0.54 | +0.080 | -0.00039 |
| 16 (ref.) | full | 1 | -67 | +11 | +12 (+2.6 %) | +0 | +0 | +0 (—) | +0 | -0.0174 | -0.79 | +0.110 | +0.00076 |
| 16 (ref.) | full | 2 | -229 | +21 | +21 (+4.6 %) | +0 | +0 | +0 (—) | +2 | -0.0087 | -0.09 | +0.033 | +0.00148 |
| 16 (ref.) | typical | 0 | +52 | +17 | +25 (+5.5 %) | +0 | +0 | +0 (—) | +0 | -0.0217 | -0.09 | +0.015 | -0.00084 |
| 16 (ref.) | typical | 1 | -121 | +42 | +42 (+9.8 %) | +0 | +0 | +0 (—) | +1 | -0.0305 | -0.04 | -0.002 | -0.00059 |
| 16 (ref.) | typical | 2 | -10 | +26 | +32 (+7.2 %) | +0 | +0 | +0 (—) | +0 | -0.0217 | -0.10 | +0.011 | -0.00015 |

## Zero-exposure handling (Coverage-LP trajectories)

| B | seed | strategy | round | budget | zero-exposure experts before | targets reached / unreached | exploration seqs | LP positive | random fill |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0 | Full Coverage-LP | 0 | 1 | 0 | 0 / 0 | 0 | 1 | 0 |
| 4 | 0 | Full Coverage-LP | 1 | 2 | 0 | 0 / 0 | 0 | 2 | 0 |
| 4 | 0 | Typical Coverage-LP | 0 | 1 | 0 | 0 / 0 | 0 | 1 | 0 |
| 4 | 0 | Typical Coverage-LP | 1 | 2 | 0 | 0 / 0 | 0 | 2 | 0 |
| 4 | 1 | Full Coverage-LP | 0 | 1 | 0 | 0 / 0 | 0 | 1 | 0 |
| 4 | 1 | Full Coverage-LP | 1 | 2 | 0 | 0 / 0 | 0 | 2 | 0 |
| 4 | 1 | Typical Coverage-LP | 0 | 1 | 0 | 0 / 0 | 0 | 1 | 0 |
| 4 | 1 | Typical Coverage-LP | 1 | 2 | 0 | 0 / 0 | 0 | 2 | 0 |
| 4 | 2 | Full Coverage-LP | 0 | 1 | 0 | 0 / 0 | 0 | 1 | 0 |
| 4 | 2 | Full Coverage-LP | 1 | 2 | 0 | 0 / 0 | 0 | 2 | 0 |
| 4 | 2 | Typical Coverage-LP | 0 | 1 | 0 | 0 / 0 | 0 | 1 | 0 |
| 4 | 2 | Typical Coverage-LP | 1 | 2 | 0 | 0 / 0 | 0 | 2 | 0 |
| 8 | 0 | Full Coverage-LP | 0 | 3 | 0 | 0 / 0 | 0 | 3 | 0 |
| 8 | 0 | Full Coverage-LP | 1 | 3 | 0 | 0 / 0 | 0 | 3 | 0 |
| 8 | 0 | Typical Coverage-LP | 0 | 3 | 0 | 0 / 0 | 0 | 3 | 0 |
| 8 | 0 | Typical Coverage-LP | 1 | 3 | 0 | 0 / 0 | 0 | 3 | 0 |
| 8 | 1 | Full Coverage-LP | 0 | 3 | 0 | 0 / 0 | 0 | 3 | 0 |
| 8 | 1 | Full Coverage-LP | 1 | 3 | 0 | 0 / 0 | 0 | 3 | 0 |
| 8 | 1 | Typical Coverage-LP | 0 | 3 | 0 | 0 / 0 | 0 | 3 | 0 |
| 8 | 1 | Typical Coverage-LP | 1 | 3 | 0 | 0 / 0 | 0 | 3 | 0 |
| 8 | 2 | Full Coverage-LP | 0 | 3 | 0 | 0 / 0 | 0 | 3 | 0 |
| 8 | 2 | Full Coverage-LP | 1 | 3 | 0 | 0 / 0 | 0 | 3 | 0 |
| 8 | 2 | Typical Coverage-LP | 0 | 3 | 0 | 0 / 0 | 0 | 3 | 0 |
| 8 | 2 | Typical Coverage-LP | 1 | 3 | 0 | 0 / 0 | 0 | 3 | 0 |

## Selection overlap (Jaccard of post-warm picks)

| B | seed | FC vs FR | TC vs TR | FC vs TC | FR vs TR |
|---|---|---|---|---|---|
| 4 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 2 | 0.00 | 0.00 | 0.00 | 0.20 |
| 8 | 0 | 0.00 | 0.00 | 0.00 | 0.09 |
| 8 | 1 | 0.00 | 0.00 | 0.00 | 0.09 |
| 8 | 2 | 0.00 | 0.00 | 0.00 | 0.00 |

## Prespecified gate

1. Full Random scarce (≥ 10 % of experts < 128 tokens, or ≥ 1 zero) in ≥ 2/3 matched seeds; 2. Coverage (full or typical) raises p10 by ≥ 10 % or cuts experts < 128 by ≥ 50 % in ≥ 2/3 seeds; 3. its minimum is not worse in ≥ 2/3 seeds; 4. valid selections.

| B | FR scarce by seed | c1 | family | lower tail improved by seed | c2 | min worse by seed | c3 | Gini better by seed |
|---|---|---|---|---|---|---|---|---|
| 4 | Y/Y/Y | True | full | n/n/n | False | Y/Y/Y | False | n/n/n |
| 4 | Y/Y/Y | True | typical | n/n/n | False | Y/Y/Y | False | n/n/n |
| 8 | n/n/n | False | full | n/n/n | False | Y/Y/n | False | Y/n/Y |
| 8 | n/n/n | False | typical | n/n/n | False | Y/n/n | True | Y/Y/Y |

## Compute feasibility of a targeted GPTQ comparison (estimates from the B = 16–128 runs)

Hessian replay fit: 1.8 s + 0.497 s × B; GPTQ ≈ 288 s per run (independent of B); routing profile ≈ 357 s forward-equivalent (once; needed by FC, TC and TR); LP selection ≤ 1.4 s; peak GPU ≈ 7.8 GB.

| B | Hessian (extrap.) s | GPTQ s | per run s | calibration forward-equiv s | profile / calibration forwards | 12-run comparison GPU s |
|---|---|---|---|---|---|---|
| 4 | 3.8 | 288 | 292 | 0.71 | 500× | 3499 |
| 8 | 5.8 | 288 | 294 | 1.43 | 250× | 3522 |

Figures (`figures/`, PDF + PNG): fig1_exposure_by_budget, fig2_scarcity_probability, fig3_coverage_lower_tail, fig4_balance_vs_atypicality, fig5_source_diversity. None of them measures quantization quality.
