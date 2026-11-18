# Legacy metric re-evaluation — summary

Full report: `update_legacy_metric_reeval.md`. Decision rule frozen in `protocol.json` before any output-error result.

- **Primary verdict:** FC vs FR = **LEGACY-NEGATIVE-CONFIRMED** on the confirmation split (primary) and on old-test.
- **Recommendation: B.** Retain C4 Random and strengthen the negative-results report.
- **Correctness:** all gates pass.
  - FR s0 from the new B1 cache reproduces the historical weights bitwise.
  - The AGQ q=0 layer output (0.0037166) is reproduced to 5e-16.
  - Expert reconstruction on the installed BF16 weights is +0.03 % vs the historical value, with per-expert Spearman 1.0.

| comparison (confirmation split) | Δ expert recon (s0/s1/s2) | Δ MoE output (s0/s1/s2) | class |
|---|---|---|---|
| FC − FR | +3.08 / +2.39 / +2.92 % | +2.20 / +1.73 / +1.99 % | LEGACY-NEGATIVE-CONFIRMED |
| TC − TR | −0.15 / +0.08 / +0.22 % | −0.48 / −0.02 / +0.19 % | NO-ROBUST-DIFFERENCE |
| TR − FR | −0.13 / −0.68 / −0.53 % | −0.16 / −0.30 / −0.53 % | NO-ROBUST-DIFFERENCE |
| EBSS − FR (one set) | +30.5 / +29.9 / +30.4 % | +22.3 / +22.3 / +22.3 % | LEGACY-NEGATIVE-CONFIRMED |

- **Direction is preserved and magnitude shrinks:** MoE-output penalties are 0.55–0.74× the expert-recon penalties
  (Spearman across configurations 0.98).
- **Differences are diagonal (c²-weighted):** cross-expert terms are about 0 except at L1, where they are negative and
  seed-dependent.
- **No AGQ-like reversal for any calibration strategy.**
