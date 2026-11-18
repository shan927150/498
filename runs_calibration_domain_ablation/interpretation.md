# Interpretation — calibration-domain ablation (selection seed 0, 3 calibration groupings)

Scope:
- One selection seed and three calibration groupings per cell. This is exploratory, so nothing here is a significance claim.
- Effects are grouping means. ORIGINAL-extraction values are quoted separately and never averaged in.
- Reference SD for NLL (prespecified) = 5.26e-4 nats/token: the within-seed grouping SD of the paired C4 SC−FR ΔNLL from the SC E3 study.

## Answers

### 1. Does WikiText-train calibration improve WikiText-test NLL compared with C4?
**Yes, modestly, under the prespecified rule. One seed only.**

| comparison | ΔNLL, grouping mean | sign over 3 groupings | × reference SD | ORIGINAL extraction |
|---|---|---|---|---|
| WT-FR − C4-FR | −0.00093 | 3/3 negative | 1.76 | −0.00021 |
| WT-SC − C4-SC | −0.00064 | 3/3 negative | 1.22 | −0.0011 |

- Both comparisons are labelled CONSISTENT-NEGATIVE.
- The 4-layer quantization NLL penalty (grouping-mean NLL − BF16) is:
  - FR: 0.00392 (C4) → 0.00299 (WT), −24 %.
  - SC: 0.00368 (C4) → 0.00304 (WT), −17 %.
- Caveats:
  - The C4 and WT grouping caches are different activations, so a cross-source difference carries the grouping noise of both cells (cell SDs 2–4e-4).
  - The ORIGINAL-extraction FR difference alone (−0.0002) would not have been distinguishable.
- Price paid: WT-calibrated weights are **+15.4 %** worse on the C4 held-out expert reconstruction. In-domain calibration specializes the quantizer; it does not improve it in general.

### 2. Does SC improve over Random under WikiText-train calibration?
**Not at model level.**
- ΔNLL(WT-SC − WT-FR) = +0.00005 (+0.00009 / −0.00028 / +0.00035; 1 negative / 2 positive; 0.10 × reference SD): NOT-DISTINGUISHABLE.
- SC still lowers every local error, consistently across groupings:

| local metric (SC vs FR, WT calibration) | change |
|---|---|
| WikiText P_seq | −0.33 % |
| B2 pooled | −0.35 % |
| C4 test reconstruction | −0.58 % |

- SC also does what it optimizes: the WT-pool covariance objective is 0.125 vs 0.165 for FR.

### 3. Is the SC advantage different between C4 and WikiText calibration?
**No evidence that SC benefits more from in-domain calibration. If anything, the local gain shrinks.**
- NLL interaction = +0.00029 (−0.0004 / +0.0006 / +0.0006): NOT-DISTINGUISHABLE.
- Local interactions point the other way, toward a *smaller* SC gain under WT:

| metric | SC gain, C4 calibration | SC gain, WT calibration | groupings |
|---|---|---|---|
| B2 pooled | −0.69 % | −0.35 % | 3/3 |
| P_seq | −0.51 % | −0.33 % | 3/3 |

- Layer-equal B2 shows no interaction (+0.04 %, mixed signs).
- Outcome B ("SC helps only with WT calibration") is not supported.

### 4. Does in-domain calibration change P_seq or B2 substantially?
**Yes, by far the largest local effect in the study.**

| metric (WT vs C4) | change |
|---|---|
| WikiText P_seq | −21.9 % |
| B2 pooled | −19.2 % |
| B2 layer-equal | −20.6 % |

- Per layer (FR), B2 changes by −11 % (L1), −22 % (L8), −21 % (L16) and −19 % (L23).
- These are about 30–65× the size of SC's local effects, with grouping SDs ≤ 0.3 %.

### 5. Are the NLL effects distinguishable from calibration-grouping variability?
- **The domain effect is borderline yes.** It has the same sign in 3/3 groupings, at 1.8× (Random) and 1.2× (SC) the reference SD.
- **The SC effects and the interaction are no.** Both are below 0.6× the reference SD, with mixed signs.
- **The C4 SC − FR difference** in this 3-grouping subset is −0.00023 (2/1 signs). That matches the earlier 5-grouping Stage B result for seed 0 (−0.0002); the reruns reproduced those 6 stored configurations bitwise.

### 6. Is calibration-distribution mismatch sufficient to explain the original SC reconstruction/PPL discrepancy?
**No.**
- With matched-domain calibration, SC's local gains persist but get smaller, and NLL still shows nothing.
- What the study does show is scale. The large domain change cut local WikiText MoE-output error by ~19–22 % and the NLL penalty by ~17–24 %: roughly proportional.
- If that proportionality holds (observed once, for a large change; not established), SC's 0.35–0.7 % B2 gains are worth about 1–3e-5 nats. That is 20–50× below the grouping SD.
- So the PPL null for SC is consistent with an effect too small to detect, not with a domain mismatch hiding a real gain.
- Overall: **mixed Outcome A + C**.
  - Outcome A at model level: domain matching helps, SC adds nothing detectable.
  - Outcome C locally: SC's local improvements persist under both sources, while NLL does not move.

### 7. Smallest justified next experiment
Replicate the **domain effect only** (WT-FR vs C4-FR) on allocation seeds 1 and 2 with the same 3 groupings, recording B2 alongside NLL.
- It is the only effect here near detectability.
- It tests the B2 ↔ NLL proportionality on a perturbation large enough to measure.
- Cost:
  - C4-FR s1/s2 grouping NLLs already exist (SC E3 Stage B). They need only re-runs for B2 (bitwise-checked).
  - The WT side needs per seed: CPU selection, 3 grouping caches (~4 min) and 3 quantizations (~6 min each).
  - Total about 1.5 h.
- Further SC-vs-FR PPL runs are not justified: at the implied effect size they would need thousands of grouping evaluations.

## Limitations
- One selection seed; three groupings. Grouping replicates are not independent allocation seeds.
- C4 grouping caches batch the 374-row union of four seed-0 strategies; WT caches batch the 221-row WT-FR ∪ WT-SC union. Groupings are matched by rule and seed, not by activations.
- The WT pool covers 2,000 of 4,918 train windows from one small corpus, so its windows are more topically related to each other than C4's one-window-per-document pool.
- B2 / P_seq use the 32-window loss-alignment capture with frozen BF16 inputs and routing. They are local measures, not the propagated model-level error.
- One configuration (WT-SC G20261111) is flagged `install_ok=False`:
  - Cause: one prespecified check expert (L1/e0, quant effect only 3.2 %) exceeds the frozen BF16-noise output tolerance (0.00557 vs 0.00510).
  - Its weights are bitwise BF16(W_q) and all structural checks pass, so it is kept and flagged. Dropping it does not change any conclusion: WT-SC − WT-FR becomes +0.00003 over 2 groupings.
