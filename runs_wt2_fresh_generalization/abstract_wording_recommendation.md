# Abstract-wording recommendation (historical significance audit)

Generated from `historical_significance_audit.json`; this is a recommendation, not a measurement.

- Article-cluster bootstrap (49 articles): 3/15 marginal 95% CIs exclude 0: s1_G20261114 (favours FR), s2_G20261111 (favours FR), s2_G20261113 (favours FC).
- Directions: opposite (both FR- and FC-favouring significant comparisons exist).
- Multiplicity: BH-FDR q<0.05: 0/15; max-|t| familywise 5%: 0/15.

**Recommendation.** The statement "individual executions yield opposite statistically significant conclusions" holds only at the *marginal* (per-comparison) 95% level, not familywise. Suggested wording:

> Across 15 numerically equivalent calibration executions, individual FR-vs-FC comparisons on the same WikiText-2 windows reached marginally significant conclusions in opposite directions (2 favouring FR, 1 favouring FC at 95% article-cluster bootstrap; none survives a familywise correction), while the execution-averaged comparison is indistinguishable from zero.
