## 5. Interpretation

### What the results establish
- **Expert weighting does not change any comparison.**
  - SC keeps 98–101 % of its R0 improvement under every weighting: frequency or squared router weight, C4 or WikiText reference.
  - The improvement is negative (better) on all 3 seeds under every metric.
  - FC keeps 96–100 % of its +2.5 % penalty. TR and TC stay small and seed-inconsistent under every weighting.
- **Why:** the weights are not very concentrated, and the changes are spread evenly across experts.
  - Effective experts per layer: 59–60 (R1_c4), 48–57 (R1_wiki), 34–57 (R2).
  - SC's per-expert improvement is nearly unrelated to expert weight (Spearman −0.15 to +0.06). Top- and bottom-quartile experts improve about equally (+0.64 to +0.83 %).
  - With a broad, even change, any convex reweighting returns about the same mean.
- **The grouped runs agree:** grouping-averaged per-expert errors give the same deltas as the ORIGINAL extraction.

### What remains uncertain
- Model-level NLL differences are at noise level.
  - ORIGINAL-extraction ΔNLL for SC vs FR is mixed in sign: +0.0009 / −0.0011 / −0.0003.
  - Grouping-averaged ΔNLL is −0.0002 to −0.0007 per seed, with grouping SDs of 0.0004–0.0006 per seed.
- The Spearman values in §4 are +0.62 on ORIGINAL NLL and −0.01 after grouping averaging, identical for every weighting.
  - The +0.62 is carried by FC's ORIGINAL-extraction NLL penalty, which earlier grouping studies showed is a grouping artifact.
  - 12 points from 4 strategies × 3 seeds cannot support a correlation claim.

### Does changing expert-importance weights explain the discrepancy?
**No.** Outcome A is ruled out.
- Equal-expert aggregation does not overstate SC's improvement relative to frequency- or router-weight-aware aggregation, nor FC's penalty.
- Whatever separates reconstruction from NLL lies elsewhere: in the data the error is measured on, the nonlinearity or aggregation (MVP B), downstream propagation, or detectability.
