# Activation second-moment mismatch (post-hoc, descriptive)

Reference: tuning split rows [2000, 2128] (disjoint from pool and test); projection: gu input (gate/up). Mean over 240 experts and 3 seeds.

| condition | rel. Frobenius | rel. diagonal | eff-rank ratio cal/ref | top-16 alignment | recon |
|---|---|---|---|---|---|
| Full Random | 0.2731 | 0.0479 | 0.9683 | 0.6223 | 0.004615 |
| Full Coverage-LP | 0.3722 | 0.1329 | 1.0458 | 0.5092 | 0.004730 |
| Typical Random | 0.2885 | 0.0723 | 0.9463 | 0.6275 | 0.004606 |
| Typical Coverage-LP | 0.2755 | 0.0517 | 0.9797 | 0.6272 | 0.004613 |
| Random-half Random | 0.2698 | 0.0445 | 0.9870 | 0.6209 | 0.004615 |
| Random-half Coverage-LP | 0.3383 | 0.1032 | 1.0159 | 0.5406 | 0.004697 |
| Coverage-LP-Repr | 0.2927 | 0.0763 | 1.0548 | 0.6168 | 0.004653 |

Spearman(mean Frobenius mismatch, recon) over 21 runs: 0.736; within seed (7 conditions): 0.857, 0.750, 0.750; within expert across runs: median 0.776 (99 % of 240 experts positive). Descriptive only; correlation is not causation.
