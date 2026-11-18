# FP64 canonical GPTQ and numerical reproducibility: summary

**Canonicalization verdict: `CANONICAL-VALIDATED`. Execution-variability verdict: `EXECUTION-VARIABILITY-MATERIAL`.**
All 24 correctness checks pass.

## FP64 canonical vs FP32 accumulation order (FR s0 / FC s0, ORIGINAL activations, R_O, P_O)

| Accumulation policy | FR PPL | FC PPL | FC − FR |
|---|---|---|---|
| Original FP32 | 9.267851 | 9.277908 | +0.010057 |
| FP32 order 20261121 | 9.267791 | 9.275571 | +0.007779 |
| FP32 order 20261122 | 9.267245 | 9.279571 | +0.012326 |
| FP32 order 20261123 | 9.270495 | 9.277078 | +0.006583 |
| FP32 order 20261124 | 9.271444 | 9.276058 | +0.004614 |
| FP32 order 20261125 | 9.269601 | 9.278450 | +0.008849 |
| **FP64 canonical** | **9.271421** | **9.278551** | **+0.007129** |

- **Hessians:** bitwise order-invariant for the 3 tested orders, both in FP64 and after the single cast to FP32. This
  holds for 480/480 Hessians per strategy.
- **Determinism:** the FR repeat is bitwise identical in weights, reconstruction, PPL and per-window NLL.
- **Against the order-null runs:** Δ_FP64 lies inside the FP32 order range, and the FR/FC ranking is unchanged.
  - It is −0.0029 from the ORIGINAL FP32 Δ and −0.0009 from the order mean (−0.31 order-SD).
  - The canonical FR (9.271421) sits at the top of the order range.
- **Against ORIGINAL FP32:**
  - **Effective weights:** 0.0070 % (FR) / 0.0067 % (FC) of elements differ, rel. L2 8.1e-4.
  - **Integer codes:** 3–9 × 10⁻⁶ of codes differ. About 91 % of the changed BF16 elements keep their integer code.
  - **Reconstruction:** relative change below 1e-6.
  - **PPL:** changes by 0.0036 (FR) and 0.0006 (FC), which is above deterministic repeatability (0).

## Execution variability (existing studies, not pooled)

| Intervention | Activations | Routing | Act-order | Elements ≠ OO (FR) | FR SD | FC SD | Δ SD |
|---|---|---|---|---|---|---|---|
| FP32 accumulation order (n=5) | fixed | fixed | fixed | 0.0067–0.0072 % | 0.00178 | 0.00166 | 0.00287 |
| AOF (n=5) | varied | fixed | fixed | 16.9 % | 0.00240 | 0.00240 | 0.00166 |
| AO (n=5) | varied | fixed | varied | 39.2 % | 0.00402 | 0.00175 | 0.00561 |
| Natural grouping, seed 0 (n=5) | varied | varied | varied | 54.1 % | 0.00442 | 0.00429 | 0.00636 |
| Natural grouping, 3 seeds × 5 | varied | varied | varied | — | — | — | pooled within-seed 0.00687 |

- **Grouping changes the sign within every seed.** The 15 matched FC−FR differences range from −0.0135 to +0.0099,
  with FC worse in 6/15.
  - The mean is −0.00125, with hierarchical interval [−0.00478, +0.00216].
  - FC reconstruction is nevertheless worse in 15/15 configurations (+2.19 to +2.71 %).
- **Fixed-model window intervals point both ways.** Window-bootstrap intervals exclude zero in favour of FR in 2
  configurations and FC in 2.

See `update_fp64_canonical_reproducibility.md`.
