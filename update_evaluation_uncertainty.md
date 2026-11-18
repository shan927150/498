# Execution variability vs evaluation-sampling uncertainty

Protocol `runs_evaluation_uncertainty/protocol.json` (sha256 `9dcad051…`) was frozen on 2026-11-18 at 05:03. That was
after the artifact audit and before any variance or reliability result existed. The full report is
`runs_evaluation_uncertainty/summary.md`; this note condenses it.

The analysis is CPU-only and re-analyses the stored per-window NLL of `runs_heldout_bitwidth_validation/results/`. No
model forward, GPU, GPTQ solve or new data was used, and no historical artifact was modified.

## Question

The held-out study found that execution-variant effects are nearly uncorrelated between WikiText-2 (WT2) and C4-val
(C4HO): 8/15 MoE sign flips, r ≈ −0.2 to −0.3. Two explanations compete:

- **A:** real effects whose ranking depends on the text distribution;
- **B:** effects too small to resolve with 128 evaluation windows.

## Outcomes

| setting | verdict (prespecified) | WT2 | C4HO |
|---|---|---|---|
| **MoE Family D** (15 FC − FR pairs) | **D-INCONCLUSIVE** | sampling fraction **25 %**; null p **0.0005**; split-half r **0.64** (null p < 0.002) | sampling fraction 80 %; null p 0.26; r 0.14 |
| **Dense 4-bit** (5 groupings) | **D-INCONCLUSIVE** | sampling 21 %; null p 0.002; r 0.75 (p 0.006) | V_obs < V_samp (fraction 2.6); r −0.43 |
| **Dense 3-bit** (5 groupings) | **C-MIXED** | **resolved**: V_corr 95 % [0.9, 10.4]×10⁻⁷; sampling 9 %; r 0.90 | fraction 4.1; r −0.62 |
| Dose, natural act-order, k = 1 / 4 / 16 / 64 | B / D / D / B | k = 4 and 16 reliable (r ≈ 0.6); k = 1 and 64 not | nothing resolved |
| Dose, fixed act-order (exploratory) | B / D / B / D | | |

**Headline.**
- **WT2 effects are real and reproducible.** About 75–90 % of the between-variant NLL variance is not evaluation
  sampling. Rankings reproduce across disjoint article halves, robust to every split scheme and to dropping any single
  article.
- **C4HO shows nothing beyond sampling.** With 128 documents, every setting's dispersion is fully explained by
  evaluation sampling.
- **No setting supports A** (distribution-dependent rankings): C4HO has no reliable ranking to compare against. What is
  supported is a **distribution-dependent magnitude**:
  - variants differ 6–70× more on WT2 (V_obs ratio), while per-window noise is only 1.5–2× higher;
  - the WT2 − C4HO difference is formally significant only for dense 3-bit (post hoc).
- **The cross-dataset sign flips are not contradictions.**
  - 0/15 MoE pairs have CIs excluding 0 in opposite directions.
  - The flips are fully compatible with C4HO sampling noise.
- **Why D rather than C for MoE and dense 4-bit.** The frozen rule also requires the pivot V_corr interval to exclude 0.
  On WT2 its lower bound is slightly negative (MoE −0.7×10⁻⁷, dense-4 −0.3×10⁻⁷), even though the null test and
  reliability pass. In synthetic test A that interval was conservative (60/60 coverage at 0). The verdict stays as
  frozen.

## 1. Data and structure audit

- **Inventory.** All 112 configurations are present on both datasets:
  - MoE: 30 + ORIGINAL pair + BF16;
  - dense 4/3-bit: 6 each;
  - dose: k = 0–64 × 8 replicates, under natural and fixed act-order.
- **Format.** Each window's stored NLL is a sum over 511 scored targets, 65,408 in all. Totals, mean NLL and PPL
  reproduce exactly, and WT2 arrays equal history.
- **Exact nulls.** G20261115 = ORIGINAL bitwise at 4 and 3 bits. Dose k = 0 = dense-4 ORIGINAL.
- **C4HO.** 128 distinct documents, one window each, so the **document is the resampling unit**.
- **WT2.** The 128 windows are sorted 512-token chunks of the test stream and fall in **49 of 62 articles** (1–10
  windows each). 31 windows are adjacent to the previous one.
  - **Primary unit:** article cluster.
  - **Sensitivity:** contiguous blocks of 4/8/16 (32/16/8 units) and a naive window bootstrap.
  - Raw window NLL is serially correlated (lag-1 0.37), but the paired FC − FR effect is not (−0.04).

## 2. Method

- **Estimand.**
  - Between-variant variance of dataset-average mean NLL: V_obs = θ̂ᵀPθ̂ / df.
  - Corrected: **V_corr = V_obs − tr(PΣ̂)/df**.
  - Σ̂ is the joint bootstrap covariance, with the same resampled windows for every variant.
- **MoE centering.** Within-seed block centering diag(P₅, P₅, P₅), df = 12.
- **Uncertainty.**
  - Nested bootstrap (2,000 outer × 400 inner; Σ re-estimated in every outer draw) with a pivot interval.
  - Common-effect null test: remove the variant effects, keep the window covariance.
  - Negative V_corr is reported raw; zero appears only as a separately labelled constrained value.
- **Split-half.**
  - 1,000 partitions: C4 64/64 documents; WT2 balanced article halves, plus block and chronological sensitivity.
  - Statistics: Pearson, Spearman, Kendall, pair-order, top/bottom and sign agreement.
  - Null: 500 effect-free datasets × 40 partitions.
- **Cross-dataset.**
  - Full-vs-full r, with split-half reliability stepped up to full length by Spearman–Brown before comparison.
  - Attenuated expectation √(λ₁λ₂) with λ = V_corr/V_obs, computed only when both λ are positive.
- **Synthetic validation** (6 tests, all passed):
  - **A:** no true effect gives an unbiased V_corr.
  - **B:** stable effects are recovered (ratio 0.95–1.00).
  - **C:** domain-dependent effects: within-domain r grows with n.
  - **D:** the naive bootstrap under-covers serially correlated windows (SE 0.32× the truth; block-16 0.78×).
  - **E:** within-seed centering is exact, while flattening inflates 40×.
  - **F:** the duplicate-reference contrast is exactly 0 and excluded from sign agreement.

## 3. Key numbers

| | MoE WT2 | MoE C4HO | dense-4 WT2 | dense-4 C4HO | dense-3 WT2 | dense-3 C4HO |
|---|---|---|---|---|---|---|
| V_obs (×10⁻⁸ nats²) | 54.9 | 8.72 | 15.9 | 0.88 | 72.7 | 1.03 |
| V_samp (×10⁻⁸) | 13.7 | 7.02 | 3.36 | 2.27 | 6.37 | 4.26 |
| V_corr (×10⁻⁸), 95 % | 41.2 [−7.1, 74.5] | 1.7 [−11.2, 10.5] | 12.5 [−3.5, 25.0] | −1.4 [−6.6, 1.3] | **66.3 [8.8, 104]** | −3.2 [−13.4, 1.3] |
| corrected SD (nats) | 6.4×10⁻⁴ (≈ 0.006 PPL) | 1.3×10⁻⁴ (≈ 0.0015 PPL) | 3.5×10⁻⁴ | n/a | 8.1×10⁻⁴ | n/a |
| split-half Pearson median | 0.64 | 0.14 | 0.75 | −0.43 | 0.90 | −0.62 |
| cross-dataset r (95 %) | −0.21 (−0.56, 0.27) | | −0.32 (−0.88, 0.76) | | −0.84 (−0.94, 0.64) | |

**WT2 block sensitivity.** V_corr changes by ≤ 7 % across article, b = 4/8/16 and naive. The block correlation changes
almost nothing.

**MoE window CIs excluding 0:**
- WT2: 3/15 (article), 2–4 across schemes, naive 4, matching history;
- C4HO: 0/15;
- opposite significant directions: 0.

**Execution dispersion vs window SE.** The across-execution SD is 1.55× the per-pair window SE on WT2 but 0.88× on
C4HO. Window-only CIs therefore understate execution uncertainty on WT2, while on C4HO they already cover it.

## 4. What this changes for the paper

- **Claim:**
  - GPTQ execution variants have real, reproducible differences in expected NLL on WT2 (corrected SD ≈ 0.006 PPL for
    MoE FC − FR).
  - Window-only CIs for a single fixed-weight comparison understate the uncertainty of an execution-averaged method
    ranking there.
- **Claim:** on C4-val with 128 documents, execution-variant differences are below evaluation resolution. The FR vs FC
  ranking is undetermined because of evaluation power.
- **Claim, with qualification:** the magnitude of execution sensitivity depends on the evaluation distribution. This is
  formally shown only for dense 3-bit.
- **Do not claim:**
  - that rankings reverse across distributions;
  - that a negative corrected variance means zero execution effect;
  - any disattenuated cross-dataset correlation (none was estimable).
- **Next step.**
  - A fresh WT2 subset (456 unused test windows, or WT2 validation) to test whether the WT2 rankings replicate on new
    text.
  - A 4–8× larger C4 set to tell "small" from "absent" C4 effects.

## 5. Deviations and notes

- **Split-half null fix (before the freeze).** The first null kept resampled copies of an article in one half, which
  biased it negative (≈ −0.36 on pure noise). It was replaced before the protocol freeze; the null is now centred at 0
  with uniform p. The code hash is unchanged since the freeze.
- **Post-hoc checks, labelled** (`posthoc_exploratory.json`): the WT2 − C4HO V_corr difference, leave-one-article-out
  (≥ 77 % of WT2 V_corr kept) and window concentration. They do not affect any verdict.
- **Partitions are not independent evidence.** Split-half partitions and correlated comparisons are not independent;
  significance comes only from the resampling nulls.

## Files

- **Code:**
  - `diagnostics/evaluation_uncertainty.py` (stages audit / protocol / synth / analyze)
  - `diagnostics/report_evaluation_uncertainty.py` (figures)
  - `diagnostics/evaluation_uncertainty_posthoc.py`
- **Outputs:** `runs_evaluation_uncertainty/`.
  - `summary.md` and all 15 required JSONs, plus `posthoc_exploratory.json`.
  - `figures/fig1`–`fig7` (PNG and PDF).
  - `logs/` (bootstrap draws, per-set parts, protocol hash).
