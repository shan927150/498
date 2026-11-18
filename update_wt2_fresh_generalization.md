# Final WikiText-2 fresh evaluation: article-level generalization

Protocol `runs_wt2_fresh_generalization/protocol.json` (sha256 `cd559666…`) was frozen on 2026-11-18 at 05:45, before
any model saw the fresh text. The full report is `runs_wt2_fresh_generalization/summary.md`.

**Setup.** 44 historical weight sets were reused and hash-verified: MoE Family D 30 plus the seed-0 ORIGINAL pair; dense
4-bit 6; dense 3-bit 6. There were no GPTQ solves. The evaluator is unchanged, and evaluator parity on the original 128
windows is bitwise.

## Outcomes

| model | article generalization | execution effect, fresh articles | same-article fresh windows (secondary) |
|---|---|---|---|
| **MoE Family D** | INCONCLUSIVE (descriptive transfer r 0.66 [0.17, 0.74]) | BELOW-EVALUATION-RESOLUTION | transfer SUPPORTED (r 0.77); effect INCONCLUSIVE (null p < 0.0005, split-half 0.70; CI lower bound −1×10⁻⁹) |
| **Dense 4-bit** | INCONCLUSIVE (r −0.28) | BELOW-EVALUATION-RESOLUTION | transfer SUPPORTED (r 0.85); effect **RESOLVED** |
| **Dense 3-bit** | INCONCLUSIVE (descriptive r 0.95 [0.25, 0.99]) | BELOW-EVALUATION-RESOLUTION | transfer SUPPORTED (r 0.92); effect **RESOLVED** |

**Headline.**
- **All unseen WT2 articles are now used, and there are few of them.** The test split has **60 articles, not 62**: the
  old regex counted two formula lines. Only **10** were never represented historically, giving 35 windows and 17,885
  targets, 27 % of the original budget.
- **On those 10 articles the variance can't be resolved, but the ranking is there.**
  - A power simulation written before any fresh result predicted 10 % power.
  - The MoE corrected variance is the same size as the original (4.2 vs 4.1×10⁻⁷) but not resolvable (null p 0.065).
  - The MoE ranking transfers at r = 0.66, exactly the sampling-attenuated expectation for identical true effects
    (0.64). Dense 3-bit transfers at r = 0.95.
  - The strict rule labels both INCONCLUSIVE because the fresh-article effect itself is below resolution.
- **On 381 unseen passages of the evaluated articles, execution effects are reproducible.**
  - Split-half r 0.70 / 0.87 / 0.89 (MoE / dense-4 / dense-3), all above the null.
  - Transfer r 0.77 / 0.85 / 0.92, each matching its attenuated expectation.
  - Corrected variance resolved for dense 4- and 3-bit.
  - **There is no evidence of article-specific effects beyond sampling.**
- **The FR vs FC ranking is undetermined in every cohort once execution uncertainty is included.**
  - Eval-only CIs favour FR on fresh articles (+6.7×10⁻⁴) and FC on fresh windows (−2.5×10⁻⁴).
  - The seed-level and combined intervals cover 0 everywhere.

## Historical significance audit

| Seed | Grouping | Δ (FC − FR) | Article CI | Favoured | familywise |
|---|---|---:|---|---|---|
| 1 | G20261114 | +9.2×10⁻⁴ | [+0.5, +19.0]×10⁻⁴ | FR | p_maxT 0.41 |
| 2 | G20261111 | +10.7×10⁻⁴ | [+0.6, +19.6]×10⁻⁴ | FR | 0.28 |
| 2 | G20261113 | −14.6×10⁻⁴ | [−26.2, −4.5]×10⁻⁴ | FC | 0.11 |

- **The directions are opposite**, and this does not come from the naive bootstrap. The naive bootstrap only adds
  s0_G20261114, which favours FC.
- **0/15 survive BH or max-|t|.**
- **Wording.** "Opposite significant conclusions" must be stated as *marginal* (`abstract_wording_recommendation.md`).

**Fresh cohorts** (article-cluster CIs):

| cohort | CIs excluding 0 | direction | familywise-significant |
|---|---|---|---|
| fresh articles | 4/15 | all FR, seed 2; s2_G20261111 FR in 9 of 10 articles | 2 |
| fresh windows | 4/15 | all FC | 3 |

- **The historical significant pairs do not reverse.** s2_G20261113 (FC) and s2_G20261111 (FR) replicate significantly
  on one fresh cohort each; no historical pair is significantly reversed.
- **One set of weights, opposite conclusions.** s2_G20261115 is significant FR on fresh articles (marginal) and FC on
  fresh windows (familywise p 0.002).

## Cohort audit

- **Unused grid.** 456 unused i·512 grid windows: A 30 (8 articles), B 381, C 45 (boundary-crossing, excluded).
- **Evaluated windows are article-aligned.** These are complete 512-token slices of the same token stream:
  - Cohort A: 35 windows / 10 articles;
  - Cohort B: 381 windows / 48 articles.
- **Overlap.** Zero token or article overlap with history.
- **Early-dev flag.** Early smoke-test PPL (`e3_allocation --ppl`, first 81,920 tokens) touched cohort-A articles 7, 13
  and 14. They are flagged, and a sensitivity analysis excludes them (MoE transfer r 0.52 [0.04, 0.67]).
- **Validation cohort omitted.** WT2 validation was used by the domain ablation, so that optional cohort was not run.

## Validation

- **Parity:** bitwise on the original windows for BF16, dense4 ×2, dense3 ×1 and MoE ×2.
- **Repeats:** bitwise.
- **Nulls:** G20261115 = ORIGINAL bitwise on all fresh cohorts at 4 and 3 bits.
- **MoE non-target parameters:** all 3,964 unchanged.
- **Restores:** pristine after every config.
- **Token accounting:** exact.
- **Synthetic check:** with 10 article clusters, the null-test size is 3.3 %.

## What this changes for the paper

- **Claim:** GPTQ execution-induced NLL differences are reproducible on unseen WikiText-2 text from the evaluated
  articles, and their rankings transfer. Dense effects are resolved; the MoE common-effect null is rejected.
- **Claim, descriptive:** the MoE within-seed and dense 3-bit rankings transfer to unseen articles at the strength
  sampling attenuation allows. Article-level variance resolution is power-limited, because only 10 unseen articles
  exist.
- **Claim:** fixed-weight FR/FC conclusions, including familywise-significant ones, depend on the evaluation sample.
  The execution-averaged FR/FC difference is indistinguishable from zero everywhere.
- **Do not claim:**
  - article- or domain-specific ranking reversals;
  - familywise "opposite significant conclusions" on the original windows.
- **Decision:** the evidence is sufficient to start paper writing. Further article-level power would need a new corpus
  (e.g. WikiText-103), which is optional.

## Files

- **Code:**
  - `diagnostics/wt2_fresh_generalization.py`
  - `diagnostics/wt2_fresh_analysis.py`
  - `diagnostics/report_wt2_fresh_generalization.py`
- **Outputs:** `runs_wt2_fresh_generalization/` (all required JSONs, `summary.md`, `figures/`, `results/`, `logs/`).
