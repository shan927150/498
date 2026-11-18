# Decomposing GPTQ execution variability and evaluation-sampling uncertainty

**Scope.** CPU-only re-analysis of the stored per-window NLL in `runs_heldout_bitwidth_validation/results/`. No model
forwards, GPU, GPTQ solves or new data were used, and no historical artifact was modified.

**Protocol.** `protocol.json` (sha256 `9dcad051…`) was frozen on 2026-11-18 at 05:03, after the audit and before any
variance or reliability result existed. The protocol hash is re-checked at the start of every stage.

**Code:**
- `diagnostics/evaluation_uncertainty.py` (stages audit / protocol / synth / analyze);
- `diagnostics/report_evaluation_uncertainty.py` (figures);
- `diagnostics/evaluation_uncertainty_posthoc.py` (labelled post-hoc checks).

## Verdicts

| setting | verdict (prespecified rule) | one-line reading |
|---|---|---|
| **MoE Family D** (FC − FR, 15 pairs) | **D-INCONCLUSIVE** | WT2: effects beyond sampling (null p = 0.0005; split-half r 0.64, null p < 0.002), but the conservative V_corr interval touches 0. C4HO: dispersion is consistent with sampling (80 %). |
| **Dense 4-bit** (5 groupings) | **D-INCONCLUSIVE** | Same pattern. WT2: null p = 0.002, split-half r 0.75 (p = 0.006), CI lower bound < 0. C4HO: V_obs is smaller than the sampling expectation (V_corr < 0). |
| **Dense 3-bit** (5 groupings) | **C-MIXED** | WT2: resolved (V_corr CI [0.9, 10.4]×10⁻⁷, sampling fraction 9 %, split-half r 0.90). C4HO: nothing resolved (V_corr < 0, r −0.62). |
| Dose, natural act-order (secondary) | k = 1: **B**; k = 4: **D**; k = 16: **D**; k = 64: **B** | k = 4/16 look like dense 4-bit on WT2. k = 1/64 are not separable from sampling on either dataset. |
| Dose, fixed act-order (exploratory) | k = 1/16: **B**; k = 4/64: **D** | |

**Why D rather than A or C for MoE and dense 4-bit.** The frozen rule calls a dataset "resolved" only if the pivot
95 % interval of V_corr excludes 0 **and** the common-effect null test gives p < 0.05. For MoE and dense 4-bit on WT2
the null test, the split-half reliability and its null all pass, but the pivot interval's lower bound is slightly
negative:
- MoE: −0.7×10⁻⁷ (V_corr 4.1×10⁻⁷);
- dense 4-bit: −0.3×10⁻⁷ (V_corr 1.25×10⁻⁷).

In synthetic test A the pivot interval covered zero in 60/60 runs, so it is conservative. Read substantively, both
settings show the dense 3-bit pattern: a WT2 signal and a C4HO signal that is not resolved. The prespecified verdict
nevertheless stays D, and this note is an interpretation, not a re-classification.

**No setting meets A.** Distribution-dependent rankings were never shown with both datasets individually reliable.

## 0. Audit (`artifact_inventory.json`, `alignment_validation.json`, `window_structure_audit.json`)

- **Inventory.** All 112 required configurations are present on both datasets; none is missing:
  - MoE: 30 configs, plus the ORIGINAL pair and BF16;
  - dense 4-bit: 6 configs;
  - dense 3-bit: 6 configs;
  - dose: k = 0, 1, 4, 16 and 64 × 8 replicates, under natural and fixed act-order.
- **Storage format.** Per-window NLL is stored as a **sum over 511 scored targets**; every window has 511 targets
  (65,408 in total).
  - Window mean NLL = sum / 511.
  - The token-weighted mean NLL equals the equal-weight mean of window means.
  - For every configuration and dataset, Σ reproduces total NLL, mean NLL and exp(mean) = PPL.
- **Agreement with history:**
  - WT2 arrays equal the historical per-window arrays (`runs_cross_seed_grouping`, embedded historical arrays);
  - model-level PPL equals `runs_grouping_routing_amplification` and `runs_dense_grouping_scope`.
- **Exact nulls.**
  - G20261115 = ORIGINAL bitwise (4 and 3 bit, both datasets).
  - Dose k = 0 natural = dense-4 ORIGINAL bitwise.
  - The k = 0 WT2 array comes from the embedded historical array, because k = 0 was not re-evaluated on WT2.
- **C4HO structure.**
  - 128 distinct documents (rows, URLs and text hashes all distinct), one window each, so **document = resampling
    unit**.
  - Lag-1 autocorrelation of BF16 window NLL is 0.08.
- **WT2 structure.**
  - 128 of 584 consecutive 512-token windows of the concatenated test stream, sorted in stream order.
  - They fall in **49 of 62 articles**: 1–10 windows per article, median 2.
  - 31 windows are directly contiguous with the previous one; 14 straddle an article boundary (assigned to the
    majority article).
  - Lag-1 autocorrelation of BF16 window NLL is 0.37, but only −0.04 for the paired FC − FR effect.
  - **Primary unit = article cluster.** Sensitivity: non-overlapping contiguous blocks b = 4, 8 and 16 (32, 16 and 8
    units; 128 is divisible by all three, so nothing is dropped), plus a naive window bootstrap for comparison.

## 1. Synthetic validation (`statistical_validation.json`, all passed)

| test | result |
|---|---|
| **A** no true differences, correlated noise | mean V_corr = −0.37×10⁻⁸ ± 0.63×10⁻⁸ (V_samp 1.24×10⁻⁷), so no positive bias. 61 % of estimates are negative. The pivot CI covered 0 in 60/60 runs (conservative). |
| **B** stable differences | V_corr / true = 0.95 (n = 128) and 1.00 (n = 1024). Split-half r rises from 0.77 to 0.96. |
| **C** domain-dependent | Within-domain r rises with n (0.41 → 0.97). Cross-domain r stays at the true effect correlation (the two random effect vectors happened to correlate at 0.67, so cross ≈ 0.63). |
| **D** AR(1) φ = 0.8 windows | Naive window SE is 0.32× the true SD; block-16 SE is 0.78×. The naive bootstrap badly under-covers serially correlated text. |
| **E** nested 3 × 5 with seed offsets | Within-seed V_corr / true = 0.99. Flattening the 15 values inflates it 40×. |
| **F** duplicate reference | The null contrast is exactly 0 in every draw and half. Sign denominators exclude it. |

## 2. Variance components (mean-NLL scale, nats²; `variance_components.json`, `variance_component_uncertainty.json`, Fig. 1)

Primary schemes are WT2 = article cluster and C4HO = document. The intervals are nested-bootstrap pivot 95 % intervals
(Σ re-estimated in every outer draw). p is from the common-effect null test.

| | MoE WT2 | MoE C4HO | dense-4 WT2 | dense-4 C4HO | dense-3 WT2 | dense-3 C4HO |
|---|---|---|---|---|---|---|
| V_obs (×10⁻⁸) | 54.9 | 8.72 | 15.9 | 0.88 | 72.7 | 1.03 |
| V_samp = tr(PΣ)/df (×10⁻⁸) | 13.7 | 7.02 | 3.36 | 2.27 | 6.37 | 4.26 |
| **V_corr** raw (×10⁻⁸) | **41.2** | **1.7** | **12.5** | **−1.39** | **66.3** | **−3.22** |
| V_corr 95 % (×10⁻⁸) | [−7.1, 74.5] | [−11.2, 10.5] | [−3.5, 25.0] | [−6.6, 1.3] | **[8.8, 103.8]** | [−13.4, 1.3] |
| constrained (≥ 0) | 41.2 | 1.7 | 12.5 | 0 | 66.3 | 0 |
| corrected SD (nats) | 6.4×10⁻⁴ (≤ 8.6×10⁻⁴) | 1.3×10⁻⁴ (≤ 3.2×10⁻⁴) | 3.5×10⁻⁴ (≤ 5.0×10⁻⁴) | n/a (≤ 1.2×10⁻⁴) | 8.1×10⁻⁴ [3.0, 10.2]×10⁻⁴ | n/a (≤ 1.1×10⁻⁴) |
| null p | **0.0005** | 0.26 | **0.002** | 0.82 | **< 0.0005** | 0.92 |
| **sampling fraction** V_samp/V_obs | **0.25** | **0.80** | **0.21** | **2.6** | **0.09** | **4.1** |

**Units.** df = 12 for MoE (within-seed blocks) and 4 for dense. On the PPL scale the MoE corrected SD is about
0.006 PPL on WT2 (cf. the historical 0.0069) and about 0.0015 PPL on C4HO.

**Negative estimates.** These are raw method-of-moments values. A negative V_corr means the observed spread is smaller
than the sampling expectation. It does **not** show that the true variance is 0; it shows that the data cannot resolve
a positive variance on C4HO at 128 documents.

**WT2 block sensitivity** (V_corr ×10⁻⁸ for block 4 / 8 / 16 and naive window):

| set | block 4 | block 8 | block 16 | naive window | primary (article) |
|---|---:|---:|---:|---:|---:|
| MoE | 40.9 | 41.0 | 43.8 | 41.3 | 41.2 |
| dense-4 | 12.9 | 12.5 | 13.2 | 12.8 | 12.5 |
| dense-3 | 65.0 | 66.2 | 67.9 | 65.7 | 66.3 |

- **The correction barely depends on the resampling scheme.** V_samp changes by ≤ 20 % across schemes; b = 16 has only
  8 units and is unstable.
- **The naive window bootstrap is anticonservative only for MoE.** It is the only scheme where the MoE V_corr interval
  excludes 0 ([+0.65, 6.9]×10⁻⁷).

## 3. Split-half reliability (`split_half_reliability.json`, `split_half_null_comparison.json`, Figs. 2, 4, 7)

**Construction.**
- 1,000 partitions per dataset.
- WT2: a balanced random article split (sensitivity: blocks 4/8/16 and naive, plus one chronological split).
- C4HO: 64/64 documents.
- MoE uses the 15-dimensional within-seed-centred effects.

**Common-effect null.** 500 bootstrap datasets from the effect-free population × 40 partitions, compared on the median
Pearson.

| | MoE WT2 | MoE C4HO | dense-4 WT2 | dense-4 C4HO | dense-3 WT2 | dense-3 C4HO |
|---|---|---|---|---|---|---|
| Pearson median [5 %, 95 %] | **0.64** [0.41, 0.82] | 0.14 [−0.18, 0.49] | **0.75** [0.36, 0.95] | −0.43 [−0.80, 0.38] | **0.90** [0.69, 0.99] | −0.62 [−0.90, 0.18] |
| frac > 0 / > 0.5 | 1.00 / 0.84 | 0.74 / 0.03 | 1.00 / 0.80 | 0.15 / 0.02 | 1.00 / 1.00 | 0.07 / 0.01 |
| Spearman median | 0.63 (within-seed ranks) | −0.07 | 0.80 | −0.30 | 0.80 | −0.60 |
| pair-order agreement | 0.76 (30 within-seed pairs) | 0.48 | 0.82 | 0.39 | 0.83 | 0.29 |
| sign agreement (all non-null) | 0.64 (15) | 0.38 | 0.78 (4) | 0.37 | 0.88 (4) | 0.25 |
| sign agreement, \|effect\| > 1 SE | 0.89 (8 contrasts) | 0.86 (3) | 0.96 (3) | — (0) | 0.88 (4) | — (0) |
| top-variant agreement | 0.80 | 0.34 | 0.22 | 0.01 | 0.54 | 0.01 |
| null p (median r) | **< 0.002** | 0.26 | **0.006** | 0.80 | **< 0.002** | 0.90 |

**Robustness across WT2 split schemes.** The medians stay within 0.61–0.69 (MoE), 0.72–0.84 (dense-4) and 0.88–0.95
(dense-3) for every WT2 split, including the chronological one. MoE per-seed WT2 medians are 0.68 / 0.56 / 0.82.

**Why C4HO correlations are negative.** Negative C4HO split-half medians are the expected signature of V_obs < V_samp.
The halves' deviations from a nearly flat full-sample vector are anti-correlated. They are not evidence of reversed
effects (null p 0.80–0.90).

**Dose (natural)**, WT2 split-half median r with null p in parentheses:

| | k = 1 | k = 4 | k = 16 | k = 64 |
|---|---|---|---|---|
| WT2 | 0.36 (0.12) | 0.61 (0.012) | 0.59 (0.038) | 0.31 (0.20) |
| C4HO | −0.32 | 0.07 | −0.01 | −0.26 |

None of the C4HO values is distinguishable from the null.

## 4. Within- vs cross-dataset (`cross_dataset_reliability.json`, Figs. 3, 5)

**How within and cross are compared.**
- **Within-dataset values are stepped up to full length.** Split-half medians use half the data, so they are converted
  to n = 128 with the Spearman–Brown formula 2r/(1+r). That puts them on the same footing as the full-vs-full
  cross-dataset correlation.
- **λ = V_corr / V_obs** is each dataset's reliability. If both datasets measured identical true effects, the expected
  cross-dataset correlation would be √(λ_WT2 · λ_C4HO). That expectation is computed only when both λ are positive.

| | WT2 within (stepped up) | C4HO within (stepped up) | cross r (95 %) | cross Spearman | sign agreement | λ_WT2 / λ_C4HO | expected r if identical |
|---|---|---|---|---|---|---|---|
| MoE | 0.78 | 0.25 | **−0.21** (−0.56, 0.27) | −0.07 | 7/15 | 0.75 / 0.20 | 0.38 |
| dense-4 | 0.86 | < 0 | −0.32 (−0.88, 0.76) | −0.30 | 3/4 | 0.79 / < 0 | n/a |
| dense-3 | 0.95 | < 0 | −0.84 (−0.94, 0.64) | −0.70 | 1/4 | 0.91 / < 0 | n/a |
| dose k = 4 / 16 (natural) | 0.76 / 0.74 | 0.13 / < 0 | −0.17 / −0.12 | | 4/8 / 5/8 | | |

**Disattenuated correlations: none reported.** The rule requires both datasets' V_corr lower bounds to be > 0, and
that holds for no setting.

**For MoE, the cross-dataset correlation's upper bound (0.27) is below the attenuated expectation (0.38).** That
expectation is itself very uncertain, because λ_C4HO = 0.20 rests on an unresolved V_corr. So the result is suggestive
at most.

## 5. MoE window CIs (`moe_window_ci_analysis.json`, Figs. 5, 6)

| | WT2 article (primary) | WT2 block4 / 8 / 16 | WT2 naive window | C4HO |
|---|---|---|---|---|
| pair CIs excluding 0 (of 15) | **3** | 3 / 2 / 4 | 4 (= historical "4/15") | **0** |
| median CI width relative to naive | 1.01 | 1.01 / 0.99 / 0.86 | 1 | |

**Sign agreement and contradictions.**
- 8/15 point estimates change sign between datasets.
- **0/15 pairs have intervals that exclude 0 in opposite directions.**
- The three WT2-significant pairs are s1_G20261114, s2_G20261111 and s2_G20261113. All three have C4HO intervals
  that cover 0.

**Execution dispersion vs per-pair window SE.**
- WT2: within-seed SD of the 15 paired effects is 7.4×10⁻⁴ (corrected 6.4×10⁻⁴), against a median per-pair window SE
  of 4.8×10⁻⁴ (ratio 1.55).
- C4HO: 3.0×10⁻⁴ (corrected 1.3×10⁻⁴) against 3.3×10⁻⁴ (ratio 0.88).

**Execution-averaged FC − FR** (the grand mean of the 15 pairs):

| | grand mean | eval-only SE | between-execution SE | combined SE |
|---|---|---|---|---|
| WT2 | −1.3×10⁻⁴ | 2.1×10⁻⁴ | 1.7×10⁻⁴ | 2.7×10⁻⁴ |
| C4HO | +0.3×10⁻⁴ | 1.4×10⁻⁴ | 0.3×10⁻⁴ | 1.4×10⁻⁴ |

Both grand means are far inside their uncertainty.

## 6. Post-hoc checks (labelled; `posthoc_exploratory.json`, not prespecified, verdicts unchanged)

- **Is the WT2 corrected variance larger than the C4HO one?** Difference pivot 95 %:
  - MoE [−1.0, 7.5]×10⁻⁷;
  - dense-4 [−0.25, 2.7]×10⁻⁷;
  - **dense-3 [1.2, 10.8]×10⁻⁷, which excludes 0.**

  The observed variance ratio WT2/C4HO is 6.3 (MoE), 18 (dense-4) and 70 (dense-3). The sampling variance ratio is only
  1.5–2.0, so the windows are not noisier on WT2; the variants differ more there.
- **Article influence (leave-one-article-out, WT2).** The WT2 signal is not carried by one article.

  | set | min V_corr (share of full) | max sampling fraction | min split-half r |
  |---|---|---|---|
  | MoE | 3.2×10⁻⁷ (77 %) | 0.33 | 0.54 |
  | dense-4 | 1.1×10⁻⁷ (86 %) | 0.24 | 0.69 |
  | dense-3 | 5.5×10⁻⁷ (82 %) | 0.10 | 0.87 |

- **Concentration.** On WT2 the top 10 windows carry 31–45 % of the centred between-variant sum of squares, and 63–72 %
  of windows contribute positively. The signal is broad.

## 7. Answers

1. **How much of the observed across-execution NLL variance is attributable to finite evaluation sampling?**
   - On WT2, about a quarter or less: MoE 25 %, dense-4 21 %, dense-3 9 %, dose k = 4/16 29–32 %, dose k = 1/64
     54–62 %.
   - On C4HO, all of it: MoE 80 % (V_corr unresolved); dense-4 260 % and dense-3 410 % (V_obs below the sampling
     expectation).
   - Intervals are wide; see the `sampling_fraction_ci` field.
2. **Does the corrected variance remain positive on WT2?** Its point estimate is positive in every setting.
   - Dense-3: resolved (CI excludes 0).
   - MoE, dense-4 and dose k = 4/16: the null tests reject a common-effect world (p 0.0005–0.016), but the conservative
     pivot interval touches 0.
   - Dose k = 1/64: not resolved.
3. **Does it remain positive on C4HO?** No setting resolves a positive variance on C4HO. MoE's raw estimate is
   +1.7×10⁻⁸ (CI ±1.1×10⁻⁷); dense-4 and dense-3 estimates are negative.
4. **Are MoE FR/FC grouping effects reliable within each dataset?** On WT2, yes: split-half r 0.64 (stepped up to full
   length 0.78), above the null (p < 0.002), and robust to every split scheme and to dropping any article. On C4HO, no
   (r 0.14, null p 0.26).
5. **Are dense grouping effects reliable within each dataset?** On WT2, yes (4-bit r 0.75, 3-bit r 0.90, both above
   the null). On C4HO, no.
6. **Does dense 3-bit behave differently from dense 4-bit?** Same direction, larger WT2 signal. At 3 bits the WT2
   between-grouping variance is larger (V_obs 4.6×, V_corr 5.3× the 4-bit value), and it is the only setting resolved under the strict rule. C4HO remains
   flat at both bit-widths.
7. **Is low-dose execution variability distinguishable from sampling?** On WT2, k = 4 and 16 are distinguishable
   (null p ≤ 0.016, split-half r ≈ 0.6), but k = 1 and k = 64 are not. On C4HO no dose level is. The earlier finding
   that one perturbed row already gives most of the PPL spread is therefore not resolved at the k = 1 level by these
   128-window sets.
8. **Are split-half correlations higher within a dataset than between datasets?** WT2: yes, decisively (0.78–0.95
   stepped up vs cross r −0.84 to −0.21). C4HO: no, its own within-dataset reliability is ≈ 0 or negative. So the
   comparison that would demonstrate distribution-dependent rankings fails on the C4HO side.
9. **Is there evidence of genuine evaluation-distribution-dependent rankings?** Not established.
   - Rankings are reliable on WT2 and do not reappear on C4HO.
   - On C4HO there is no reliable ranking to compare against, so "different ranking" cannot be told apart from
     "effects too small to rank".
   - What is supported is **distribution-dependent magnitude**: much larger between-execution differences on WT2 than
     on C4HO. This is formally significant only for dense-3 (post hoc).
10. **Can evaluation sampling alone plausibly explain the observed C4HO dispersion?** Yes, for every setting (null p
    0.26–0.97).
11. **How much does WT2 block correlation change the results?** Very little.
    - V_samp changes by ≤ 20 % across article, b = 4, b = 8, b = 16 and naive schemes.
    - Per-pair CI widths change by +1 % (article, b = 4) to −14 % (b = 16, 8 units, unstable).
    - Paired FC − FR effects have essentially no serial correlation (lag-1 −0.04), even though raw window NLL does
      (0.37).
    - The naive window bootstrap is the only scheme under which the MoE WT2 V_corr interval excludes 0.
12. **How many MoE comparisons have window CIs excluding zero, and how often do opposite directions occur?**
    - WT2: 3/15 (article), 2–4/15 across schemes; naive 4/15, as reported historically.
    - C4HO: 0/15.
    - Opposite significant directions: 0. The 8/15 sign flips are all between point estimates with at least one CI
      covering 0.
13. **Is the historical criticism of window-only CIs still supported?**
    - **Supported on WT2.** The across-execution SD is 1.55× the per-pair window SE, and 75 % of the between-grouping
      variance is not sampling. A window CI for one fixed-weight FR/FC comparison would understate the uncertainty of an
      execution-averaged claim; 3–4 pairs are individually "significant" in directions that are not stable across
      executions.
    - **Weaker on C4HO.** Window CIs (median SE 3.3×10⁻⁴) are already as wide as the between-execution spread (corrected
      SD ≈ 1.3×10⁻⁴), so they cover the grouping effects.
    - **Both statements hold.** A fixed-weight window CI never estimates execution variability, but whether that gap is
      practically important depends on the evaluation distribution. It cannot be claimed universally from WT2 alone.
14. **Should the paper emphasise distribution-dependent rankings, insufficient evaluation power, or both?**
    - **Both, with different scopes.**
      - On WT2 there are real, reproducible execution-variant differences, and window-only CIs understate the
        uncertainty of method rankings.
      - On C4HO, 128 documents cannot resolve execution-variant differences at all; the FR/FC ranking is undetermined
        because of evaluation power.
    - **Do not claim** that rankings reverse across distributions. The cross-dataset sign flips are fully compatible
      with C4HO sampling noise.
    - **Do claim** that the magnitude of execution sensitivity depends on the evaluation distribution, with the formal
      support limited to dense-3.
15. **Is a fresh WT2 evaluation subset needed to separate domain shift from sampling uncertainty?** Yes.
    - **What it settles.** The WT2 within-dataset reliability is established on the same 128 windows, by splitting
      them. A fresh WT2 subset would test whether the WT2 rankings replicate on new Wikipedia text; the remaining 456
      unused test windows, or WT2 validation, would serve.
    - **Larger C4 set alongside.** A larger C4HO set (≥ 4–8× more documents, since V_samp scales as 1/n) is also needed.
      It would show whether C4 effects are merely small or truly absent.
    - **Without both,** "WT2-specific effects" and "WT2 effects that generalise but are invisible at C4's power" cannot
      be distinguished.

## Deviations and notes

- **Pre-freeze code fix.** In the pre-freeze smoke test, the first split-half null kept resampled copies of an article
  in one half. That induced finite-population anti-correlation (null median ≈ −0.36 on pure noise). It was replaced
  **before the freeze** by treating each draw as a split unit, which makes the null halves independent. On pure noise
  the null is then centred at 0 and p is roughly uniform. The protocol text describes the final construction.
- **The pivot interval is conservative** (synthetic coverage 60/60 at zero). The prespecified "resolved" rule
  therefore leans to D. The rule was kept as frozen.
- **k = 0 dose on WT2 uses the embedded historical array.** It is bitwise equal to dense-4 ORIGINAL.
- **Not separately identified:**
  - seed-level MoE variance (the analysis is within seed by design);
  - the reference matrix (not part of this protocol).
- **Partitions are not independent evidence.** Split-half partitions and the 15 MoE / 5 dense comparisons are not
  independent; all split-half distributions are sensitivity diagnostics. Significance comes only from the
  resampling-based nulls.
- **Units.** Everything is on the mean-NLL scale. PPL differences appear only in `moe_window_ci_analysis.json`
  (`ppl_diff`, with their own intervals).

## Files

- **Required outputs:**
  - `protocol.json` (read-only); its hash is in `logs/protocol_sha256.txt`.
  - `artifact_inventory.json`, `alignment_validation.json`, `window_structure_audit.json`
  - `joint_bootstrap_covariance.json` (θ, Σ, SE and correlation per set × dataset × scheme)
  - `variance_components.json`, `variance_component_uncertainty.json`
  - `split_half_reliability.json`, `split_half_null_comparison.json`, `cross_dataset_reliability.json`
  - `moe_window_ci_analysis.json`, `dense_grouping_uncertainty.json`, `dense_dose_uncertainty.json`
  - `statistical_validation.json`, `decision.json`
- **Post-hoc:** `posthoc_exploratory.json`.
- **Figures:** `figures/fig1`–`fig7` (PNG and PDF).
- **Logs:** `logs/` holds the run log, the per-set intermediate parts and the bootstrap draws.
