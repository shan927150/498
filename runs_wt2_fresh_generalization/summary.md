# Final WikiText-2 fresh evaluation: article-level generalization of GPTQ execution effects

**Protocol.** `protocol.json` (sha256 `cd559666…`, read-only copy in `logs/`) was frozen on 2026-11-18 at 05:45. That was
after the cohort and overlap audit and **before any model saw the fresh text**.

**Reuse, no new quantization.** All 44 evaluated weight sets are historical and hash-verified: MoE Family D 30 plus the
seed-0 ORIGINAL pair; dense 4-bit 6; dense 3-bit 6. There were no GPTQ solves and no recalibration.

**Evaluator.** Unchanged: `eval_partial_ppl.layerwise_nll`, 512-token windows, batch 8, natural MoE routing.

## Classifications (prespecified rules, `decision.json`)

| model | article generalization (original → fresh articles) | execution effect on fresh articles | secondary: same-article fresh windows |
|---|---|---|---|
| **MoE Family D** | **ARTICLE-GENERALIZATION-INCONCLUSIVE** | **EXECUTION-EFFECT-BELOW-EVALUATION-RESOLUTION** | transfer SUPPORTED (r 0.77 [0.39, 0.84]); execution effect INCONCLUSIVE (null p < 0.0005, split-half 0.70; V_corr CI lower bound −1×10⁻⁹) |
| **Dense 4-bit** | **ARTICLE-GENERALIZATION-INCONCLUSIVE** | **EXECUTION-EFFECT-BELOW-EVALUATION-RESOLUTION** | transfer SUPPORTED (r 0.85 [0.36, 0.97]); **EXECUTION-EFFECT-RESOLVED** |
| **Dense 3-bit** | **ARTICLE-GENERALIZATION-INCONCLUSIVE** | **EXECUTION-EFFECT-BELOW-EVALUATION-RESOLUTION** | transfer SUPPORTED (r 0.92 [0.64, 0.98]); **EXECUTION-EFFECT-RESOLVED** |

### What the INCONCLUSIVE labels mean

- **The cohort is tiny.** The fresh-article cohort uses all 10 unseen WikiText-2 test articles: 35 windows, 17,885
  targets, 27 % of the historical budget.
- **The protocol said in advance it could not resolve the effect.** The prespecified power simulation (written before
  any fresh result, using original-WT2 noise) predicted 10 % power to resolve an original-magnitude MoE variance, with an
  expected split-half r of 0.18.
- **What was observed on fresh articles:**
  - **MoE:** corrected variance **4.2×10⁻⁷, the same as the original 4.1×10⁻⁷**, but not resolvable (null p 0.065).
  - **MoE ranking transfer:** within-seed effects correlate with the original ranking at **r = 0.66 (95 % 0.17–0.74)**,
    exactly the attenuated expectation under identical true effects (0.64).
  - **Dense 3-bit:** transfer r = 0.95 (0.25–0.99).
  - **Dense 4-bit:** does not transfer (r −0.28, CI −0.83 to 0.66); its fresh-article dispersion is below sampling noise.
- **Why the rule gives INCONCLUSIVE rather than SUPPORTED.** The rule withholds SUPPORTED whenever the fresh-article
  execution effect is itself below resolution. The descriptive evidence, however, favours transfer for MoE and dense
  3-bit.

## 1. Historical significance audit (`historical_significance_audit.json`, Fig. 7)

| Seed | Grouping | Δ mean NLL (FC − FR) | Article CI 95 % | Significant? | Favoured | naive window | p_BH | p_maxT |
|---|---|---:|---|---|---|---|---:|---:|
| 1 | G20261114 | +9.17×10⁻⁴ | [+0.51, +19.0]×10⁻⁴ | **yes** | **FR** | yes | 0.18 | 0.41 |
| 2 | G20261111 | +10.72×10⁻⁴ | [+0.55, +19.6]×10⁻⁴ | **yes** | **FR** | yes | 0.18 | 0.28 |
| 2 | G20261113 | −14.61×10⁻⁴ | [−26.2, −4.5]×10⁻⁴ | **yes** | **FC** | yes | 0.075 | 0.11 |
| 0 | G20261114 | −9.05×10⁻⁴ | [−18.4, +0.34]×10⁻⁴ | no | FC | **yes** (naive only) | 0.18 | 0.46 |
| other 11 | | | all cover 0 | no | | no | ≥ 0.18 | ≥ 0.37 |

- **The three significant comparisons point in opposite directions.** Two favour FR and one favours FC.
- **The naive window bootstrap adds one more**, s0_G20261114, which favours FC. The opposite directions do not depend on
  the naive bootstrap; they already exist under the article-cluster bootstrap.
- **Multiplicity:** **0/15 survive BH-FDR (q < 0.05) or a max-|t| familywise correction**, under both schemes.
- **The stored marginal CIs are reproduced exactly** (same seeds and draws).
- **Recommended abstract wording** (`abstract_wording_recommendation.md`): say that the opposite conclusions are
  *marginally* significant and do not survive a familywise correction.

## 2. WikiText-2 article structure (`historical_article_manifest.json`, `article_overlap_audit.json`)

- **The test split has 60 articles, not 62.** The header regex used before also matched two formula lines inside
  "Constant k filter". The corrected rule (a title line preceded by an empty line) changes nothing for the historical
  windows: the cluster partition is identical (checked), and the **49-article** majority count is confirmed.
- **Historical coverage.**
  - 50 articles have at least one token in a historical window; 14 historical windows cross an article boundary.
  - **10 articles were never represented:** Hed PE, Dvorak technique, New York State Route 31B, Head VI, Constant k
    filter, Shackleton (crater), Attalea (palm), Operation USA, Typhoon Krosa (2013) and Tatwine.
- **The unused grid is verified at 456 windows** of the historical i·512 grid (584 − 128):
  - **A** (single never-represented article): 30 windows in 8 articles;
  - **B** (single previously represented article, no token overlap): 381;
  - **C** (crosses an article boundary): 45, excluded.
- **Evaluated cohorts use article-aligned windows.** These are complete 512-token slices of the same tokenized stream,
  starting at the article (or unused-segment) start, so that short unseen articles are not lost to the grid:
  - **Cohort A (fresh articles):** 35 windows, 10 articles (1–8 windows each), 17,885 scored targets; 2,779 article-tail
    tokens are not evaluated.
  - **Cohort B (fresh windows, same articles):** 381 windows, 48 articles, 194,691 targets.
- **Isolation.** Token overlap is 0 for A and B against history and 0 between A and B. Article overlap of A with history
  is 0.
- **Prior use.**
  - WT2 test was never used for calibration.
  - Early `e3_allocation.py --ppl` smoke runs evaluated the first 81,920 stream tokens. That touches cohort-A articles
    7, 13 and 14 (11 windows), so they are flagged; a prespecified sensitivity analysis excludes them.
  - WT2 validation was used by the domain ablation, so the optional cohort C was omitted.

## 3. Evaluator correctness (`evaluator_correctness.json`)

All checks passed:
- **Parity:** bitwise parity on the original 128 windows for:
  - BF16 (both models);
  - dense4 G20261111 and ORIGINAL;
  - dense3 G20261111;
  - MoE FR_s0_G20261111 and FC_s2_G20261111.
- **Repeats:** BF16 and the prespecified repeat configs reproduce bitwise.
- **Installation checks:**
  - weight-file, layer and installed-digest hashes match history (dense 3-bit tensor sha256 included);
  - all 3,964 MoE non-target parameters are unchanged;
  - every config is restored to pristine.
- **Exact nulls:** **G20261115 = ORIGINAL bitwise on both fresh cohorts at 4 and 3 bits.**
- **Token accounting:** Σ per-window NLL reproduces every mean NLL and PPL, with 511 targets per window.

**Software.** torch 2.11.0+cu128, transformers 4.51.3, A100 MIG 2g.20gb.

## 4. Results

### BF16 and quantized PPL per cohort

| cohort | MoE BF16 | dense BF16 | dense4 mean (SD) | dense3 mean (SD) |
|---|---|---|---|---|
| original WT2 (49 articles) | 9.237 | 12.523 | 12.6188 (0.0050) | 13.1375 (0.0112) |
| fresh articles (10) | 10.018 | 12.960 | 13.0820 (0.0035) | 13.5069 (0.0099) |
| fresh windows (48) | 9.030 | 12.369 | 12.4774 (0.0043) | 12.9284 (0.0076) |
| C4-val (128) | 11.379 | 16.585 | 16.6733 (0.0016) | 17.0560 (0.0017) |

### Variance components (×10⁻⁸ nats², article clusters; `fresh_variance_components.json`, Figs. 2 and 6)

| | V_obs | V_samp | **V_corr [pivot 95 %]** | null p | split-half r (null p) |
|---|---:|---:|---|---:|---|
| MoE original | 54.9 | 13.6 | 41.3 [−4.9, 75.1] | < 0.0005 | 0.63 (0.002) |
| MoE **fresh articles** | 77.4 | 35.0 | **42.4 [−23.4, 87.3]** | 0.065 | 0.36 (0.18) |
| MoE fresh windows | 26.2 | 5.25 | **20.9 [−0.1, 35.2]** | < 0.0005 | **0.70 (< 0.002)** |
| MoE C4 | 8.7 | 7.1 | 1.7 [−11.8, 11.0] | 0.25 | 0.15 (0.26) |
| dense4 original | 15.9 | 3.4 | 12.5 [−4.5, 24.6] | 0.005 | 0.75 (0.006) |
| dense4 fresh articles | 7.1 | 10.3 | −3.2 [−36.7, 10.2] | 0.57 | −0.18 (0.60) |
| dense4 fresh windows | 11.7 | 1.15 | **10.6 [3.5, 16.2]** | < 0.0005 | **0.87 (< 0.002)** |
| dense3 original | 72.7 | 6.1 | **66.6 [11.7, 105]** | < 0.0005 | 0.90 (< 0.002) |
| dense3 fresh articles | 54.1 | 26.4 | 27.7 [−124, 88] | 0.135 | 0.46 (0.28) |
| dense3 fresh windows | 34.4 | 3.3 | **31.1 [5.1, 50.5]** | < 0.0005 | **0.89 (< 0.002)** |

**Sensitivity (fresh articles).**
- **Window bootstrap:** gives the same picture (MoE null p 0.069).
- **Excluding the early-dev articles (7 left):** MoE V_corr 5.1×10⁻⁷ (p 0.09); dense V_corr is negative.
- **Synthetic check, 10 clusters:** null-test size 3.3 % (`statistical_validation.json`).

### Ranking transfer (centred effects; MoE within-seed; `cross_cohort_rank_transfer.json`, Figs. 1 and 4)

| | MoE r [95 %] (expected if identical) | dense4 r | dense3 r |
|---|---|---|---|
| original vs **fresh articles** | **0.66 [0.17, 0.74]** (0.64) | −0.28 [−0.83, 0.66] | **0.95 [0.25, 0.99]** |
| original vs fresh windows | **0.77 [0.39, 0.84]** (0.78) | **0.85 [0.36, 0.97]** (0.84) | **0.92 [0.64, 0.98]** (0.91) |
| fresh articles vs fresh windows | 0.63 [0.04, 0.78] (0.66) | 0.11 | 0.99 [0.19, 0.99] |
| original vs C4 (reference) | −0.21 [−0.54, 0.28] | −0.32 | −0.84 |
| original vs fresh articles without early-dev | 0.52 [0.04, 0.67] | −0.02 | 0.81 [−0.28, 0.98] |

**Supporting MoE detail.**
- Spearman rank correlation is 0.40 (fresh articles) and 0.67 (fresh windows).
- FR/FC sign agreement is 9/15 and 11/15.
- Pairwise within-seed order agreement is 0.67 and 0.77.

**Dense ORIGINAL rank among 5 (1 = best).**

| bit-width | original | fresh articles | fresh windows | C4 |
|---|---|---|---|---|
| 4-bit | 5 | 2 | 5 | 4 |
| 3-bit | 1 | 1 | 1 | 4 |

### MoE window CIs and the FR/FC conclusion (`fresh_window_ci_analysis.json`, Fig. 5)

**Pair CIs excluding 0, by cohort:**

| cohort | pairs excluding 0 | directions | BH-FDR q < 0.05 | max-\|t\| familywise |
|---|---|---|---:|---:|
| original | 3/15 | opposite (2 FR, 1 FC) | 0 | 0 |
| fresh articles | 4/15 | all FR, all seed 2 | 2 | 2 |
| fresh windows | 4/15 | all FC | 3 | 3 |
| C4 | 0/15 | — | 0 | 0 |

**Fresh-article effects are broad-based.** s2_G20261111 is +4.2×10⁻³ and FR-favoured in 9/10 fresh articles. Percentile
CIs with 10 clusters may still under-cover.

**Historical significant pairs in the fresh cohorts:**
- **s2_G20261113 (FC)** replicates significantly on fresh windows (FC; familywise p 0.009).
- **s2_G20261111 (FR)** replicates significantly on fresh articles (FR).
- **s1_G20261114 (FR)** keeps its FR sign but is not significant.
- No historical significant pair is significantly reversed.

**The same weights reach opposite significant conclusions on two WT2 samples.** For s2_G20261115:
- fresh articles: FR, [+0.17, +3.5]×10⁻³ (marginal; familywise p 0.42);
- fresh windows: FC, [−1.5, −0.45]×10⁻³ (familywise p 0.002).

**Execution-averaged FC − FR** (grand mean of the 15 pairs):

| cohort | grand mean | eval-only 95 % | seed-level t (2 df) | eval + execution |
|---|---|---|---|---|
| original | −1.3×10⁻⁴ | [−5.7, +2.6]×10⁻⁴ | [−5.9, +3.2]×10⁻⁴ | not ≠ 0 |
| fresh articles | **+6.7×10⁻⁴** | **[+0.4, +14.1]×10⁻⁴ (FR)** | [−23, +36]×10⁻⁴ | not ≠ 0 |
| fresh windows | **−2.5×10⁻⁴** | **[−4.8, −0.15]×10⁻⁴ (FC)** | [−8.6, +3.5]×10⁻⁴ | not ≠ 0 |
| C4 | +0.3×10⁻⁴ | [−2.4, +3.1]×10⁻⁴ | [−1.9, +2.5]×10⁻⁴ | not ≠ 0 |

**An eval-only CI points to opposite winners on the two fresh WT2 cohorts. Once execution uncertainty is included, the
FR/FC ranking is undetermined in every cohort.**

## 5. Answers

1. **Which three historical MoE comparisons were significant under the article-cluster bootstrap?**
   s1_G20261114, s2_G20261111 and s2_G20261113.
2. **Did their signs actually disagree?** Yes, marginally: two favour FR and one favours FC. None survives BH or max-|t|.
3. **How many truly unseen WT2 articles are available?** **10** (of 60; the earlier "62" double-counted two formula
   lines).
4. **How many unused windows belong to those articles?** 30 grid windows (8 articles). The evaluated cohort uses 35
   article-aligned windows covering all 10 articles (17,885 targets).
5. **How many unused windows come from previously represented articles?** 381 grid windows. The evaluated cohort B has
   381 aligned windows in 48 articles. A further 45 unused grid windows cross an article boundary and were excluded.
6. **Does MoE Family D execution sensitivity remain distinguishable from evaluation-sampling noise on fresh articles?**
   Not at this size. The corrected variance has the original magnitude (4.2 vs 4.1×10⁻⁷), but null p is 0.065 with
   10 articles, which is the prespecified 10 % power regime. On fresh windows from the same articles it is clearly
   distinguishable from sampling (null p < 0.0005, split-half 0.70), though the strict pivot interval just touches 0.
7. **Does the within-seed ranking transfer from the original evaluation?**
   - The descriptive evidence says yes: r = 0.66 [0.17, 0.74] to unseen articles and 0.77 [0.39, 0.84] to unseen
     passages. Both equal the attenuated expectation for identical true effects.
   - The formal article-level label is INCONCLUSIVE, because the fresh-article effect is itself below resolution.
8. **Do dense 4-bit and 3-bit grouping effects transfer?**
   - To fresh windows: yes, for both (r 0.85 and 0.92; variance resolved).
   - To fresh articles: dense 3-bit yes descriptively (r 0.95, CI > 0); dense 4-bit no (its fresh-article dispersion
     is below sampling noise).
   - ORIGINAL keeps its position at 3 bits (best in every WT2 cohort). At 4 bits it is worst in original and fresh
     windows.
9. **Does the original WT2 split-half reliability generalize to new articles?**
   - It reproduces on new passages: 0.70 / 0.87 / 0.89 vs the original 0.63 / 0.75 / 0.90.
   - It cannot be resolved on 10 new articles: 0.36 / −0.18 / 0.46, null p 0.18–0.60, as predicted by the power
     simulation.
10. **Are same-article fresh windows more consistent with the original sample than unseen articles?** Only slightly for
    MoE (0.77 vs 0.66), and both match their sampling-attenuated expectations. So there is **no evidence of
    article-specific effects** beyond sampling. Dense 4-bit's gap reflects the unresolved fresh-article cohort.
11. **Does the FR/FC method ranking remain undetermined after execution uncertainty is included?** Yes, in all four
    cohorts. Eval-only intervals even favour FR on fresh articles and FC on fresh windows.
12. **Does the article-cluster bootstrap still reveal significant individual FR/FC comparisons?**
    - Yes: 4/15 on fresh articles (all FR) and 4/15 on fresh windows (all FC); 2 and 3 survive familywise correction.
    - Within each fresh cohort the significant directions agree.
    - Across the two fresh cohorts, the same weight set (s2_G20261115) is significant in opposite directions.
13. **Which paper claims are now supported by independent article-level evidence?**
    - **Reproducibility:** bitwise evaluator reproducibility and the exact G20261115 = ORIGINAL null hold on all new text.
    - **Same-article generalization:** execution-induced, reproducible NLL differences generalize to unseen passages of
      the evaluated articles. They are resolved for dense 4- and 3-bit; for MoE the null test rejects and split-half
      r is 0.70.
    - **Ranking transfer to unseen articles:** the within-seed MoE ranking and the dense 3-bit ranking carry over at
      the strength expected from sampling attenuation. This is descriptive: CI > 0, but the rule-level label is
      inconclusive.
    - **Reversal of fixed-weight FR/FC conclusions:** this is supported, with familywise-significant individual
      comparisons whose direction depends on the evaluation sample. The execution-averaged FR/FC difference is not
      distinguishable from zero anywhere.
14. **Which conclusions remain conditional on the original 49 articles?**
    - Resolution of execution variance on genuinely new articles (power-limited).
    - Dense 4-bit grouping effects on new articles.
    - Exact magnitudes, such as the MoE corrected SD of about 0.006 PPL.
    - The original "opposite significant conclusions" wording, which is marginal only.
15. **Is further evaluation necessary, or is the evidence sufficient to begin final paper writing?** Sufficient. All
    unseen WT2 test articles are now used, so more article-level power would require a different corpus (WikiText-103
    articles outside WT2), which is optional. The fresh results agree with the original conclusions wherever power
    allows, contradict none of them, and sharpen the wording; see `abstract_wording_recommendation.md`. Begin writing.

## Deviations and notes

- **Article rule corrected before the protocol freeze** (62 → 60). The historical clusters are identical.
- **Cohorts use article-aligned windows instead of the historical i·512 grid.** This was decided before the freeze, so
  that all 10 unseen articles are included. The grid inventory is reported for verification.
- **Synthetic validation.** It used original-WT2 data only. It was run after the freeze and before any MoE fresh result
  existed, while dense results were being produced; it does not read them.
- **Fixed MoE parity subset.** The original-WT2 MoE arrays of non-parity configs are the verified historical arrays, as
  in runs_evaluation_uncertainty. Parity configs were re-evaluated bitwise.
- **Not run:**
  - MoE seed-1/2 ORIGINAL references (never saved);
  - the optional reference matrix;
  - the optional WT2-validation cohort (prior use).
- **Cost.**
  - Dense stage: about 7 min.
  - MoE stage: 1 h 37 min (median about 3 min per config; peak 56.9 GB CPU, 4.4 GB GPU).
  - Analysis: 3 min.
  - See `cost_results.json`.

## Files

- **Code:**
  - `diagnostics/wt2_fresh_generalization.py` (stages hist / cohorts / protocol / dense / moe)
  - `diagnostics/wt2_fresh_analysis.py` (synth / analyze)
  - `diagnostics/report_wt2_fresh_generalization.py` (figures)
- **Outputs:**
  - every required JSON, plus `abstract_wording_recommendation.md`;
  - `figures/fig1`–`fig7`;
  - `results/{dense,moe}/*.json` (raw per-window NLL with token counts and weight hashes);
  - `logs/`;
  - `windows_*.pt` (token tensors; git-ignored).
