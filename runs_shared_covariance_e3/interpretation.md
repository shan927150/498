## 7. Interpretation

Written after all Stage A and Stage B results were in. The decision rule (`moe_calib/shared_covariance_selection.py:decide`, threshold 0.5 %) and the protocol (sha `8ccc0ce6eaf496cb`) were frozen before any SC reconstruction and were not changed. The test split was never used for selection or tuning: SC optimizes a 64-d sketch objective over pool rows [0, 2000) only.

### Q1. Does SC beat Full Random?
**Yes, on isolated expert reconstruction.**
- Paired test improvement: +0.67 / +0.89 / +0.49 % (mean +0.68 %). Seed 2 is 0.006 pp below the 0.5 % materiality line on its own; the frozen rule uses the 3-seed mean together with sign consistency.
- Confirmation split: +0.54 / +1.07 / +0.59 %.
- Breadth: 229 / 234 / 227 of 240 experts improve, and all four layers improve (+0.52 to +0.90 %).
- The p90/p95 expert error also drops, so the gain is not driven by a few experts.
- Stage A label SUPPORTED; final label with grouping robustness **ROBUST-SUPPORTED**.

### Q2. Does SC beat the routing-aware baselines?
- **vs Typical Coverage-LP: yes.** Test +0.64 % mean (+0.81 / +0.53 / +0.58), every seed above 0.5 %, 4/4 layers. ROBUST-SUPPORTED.
- **vs Typical Random: no, under the frozen rule (NO-GAIN).** SC is better on every seed (+0.72 / +0.33 / +0.38 %, mean +0.47 %), every layer, and 15/15 grouping configurations. The mean, however, is below the pre-registered 0.5 % threshold, and the confirmation-split margin is only +0.28 %.
- Read this as: SC is consistently better than TR, but by less than the materiality bar. It does not mean SC is no better than TR.
- Part of SC's margin over FR is available more cheaply: TR alone gets +0.21 % test / +0.45 % confirmation over FR.

### Q3. Does the MVP result generalize?
**Yes, in direction and roughly in size, although the two settings are not directly comparable.**
- The MVP's SC-vs-SR effect (+0.55 % test, token-matched) reappears as +0.68 % for SC vs FR at full scale.
- Raw errors must not be compared across the two studies: the expert sets, metric aggregation and test subset all differ.

| aspect | MVP (`runs_expertwise_mvp`) | full E3 (this study) |
|---|---|---|
| layers / experts | L16 only / 8 experts | L1, L8, L16, L23 / all 240 routed experts |
| objective scope | 8 experts × 3 projections | 240 experts × 2 Hessian groups (gate+up, down), stacked |
| warm start | none | matched 32-sequence warm start shared with FR/TR/TC |
| token control | token-matched (subsampled to equal per-expert tokens) | none; natural exposure (median ≈ 4,320 tokens/expert for all strategies) |
| baseline | Shared Random (SR) | FR, TR, TC from the frozen E3 manifest (stored, reproduced bitwise) |
| metric | per-sequence relative error, mean over 8 experts, 128 test sequences | E3 relative error over 240 experts, full test split, plus confirmation split |
| downstream | none | partial-model WikiText-2 PPL, plus a 5-grouping Stage B |
| SC gain vs random | +0.55 % test / +0.62 % confirmation | +0.68 % test / +0.73 % confirmation (vs FR) |

### Q4. Is covariance matching associated with the reconstruction gains?
**Yes, descriptively, though the evidence is not causal.**
- SC lowers the selection surrogate J from ≈0.20 to 0.154.
- The gain is not specific to the 64-d sketch. The held-out full-dimensional covariance mismatch also falls, from 0.443 (FR) to 0.364. It falls for gate+up (0.196 → 0.155) and for down (0.691 → 0.574), and for 98–100 % of experts on each seed.
- Across the 9 within-seed deltas vs FR, test error tracks the full-dimensional mismatch (Spearman +0.82; down projection +0.90) and the sketch J (+0.78).
- It does not track token count (+0.07), Gini (+0.20) or routing atypicality d(s) (−0.53).
- At the per-expert level the association is weaker but positive: ρ(Δsketch, Δerr) = +0.27 to +0.51.
- SC does not achieve its gain by moving toward typical sequences: its mean d(s) is 0.306, between FR (0.324) and TR/TC (≈0.29).
- These are associations across 12 selected sets that differ in more than one respect, not a controlled intervention.

### Q5. Does the gain survive calibration-grouping noise?
**Yes, clearly.**
- In Stage B every (seed, grouping) configuration favors SC over each baseline: 15/15 vs FR, TR and TC.
- The pooled grouping SD of the reconstruction advantage is 0.004–0.007 pp, about 100× smaller than the effect.
- Hierarchical 95 % CI for SC vs FR: [+0.50 %, +0.89 %].
- The fresh-process control (SC s0 G20261111) reproduced weights, reconstruction, PPL and per-window NLL bitwise.
- The remaining uncertainty is between allocation seeds (0.50–0.89 %), not between groupings.

### Q6. Does the reconstruction gain transfer to perplexity?
**Not established. There is at most a small directional signal vs FR.**
- Stage A ΔPPL vs FR is mixed in sign: +0.0082 / −0.0103 / −0.0026.
- In Stage B the SC − FR seed means are all negative (−0.0018 / −0.0066 / −0.0040; overall −0.0041; hierarchical CI [−0.0072, −0.0011]; 11/15 configurations favor SC).
- But the pooled grouping SD (0.0049) is larger than |mean| (|M|/W = 0.85). By the pre-stated criterion, a PPL gain is therefore not claimed.
- For scale, −0.004 PPL would be about 13 % of the 4-layer quantization gap to BF16 (+0.032).
- Against TR and TC there is no PPL signal (overall −0.0001 and −0.0020; CIs straddle 0; 6/15 and 8/15).
- Conclusion: the reconstruction improvement is real but too small to separate from calibration-grouping noise in 4-layer partial-model PPL.

### Q7. Is the gain worth the extra profiling compute?
**Only when the number of calibration sequences is the binding constraint, not when compute is.**
- SC's extra cost over FR is ≈632 s: the sketch pass (617 s) plus selection (~15 s). That equals about 3,500 random sequences' worth of full-model forwards at 0.178 s/sequence, plus 8 GB of sketch storage.
- What do more random sequences buy?
  - In `runs_low_budget`, Full Random goes from 0.004816 at B = 64 to 0.004615 at B = 128: −4.2 % per doubling.
  - SC's 3-seed mean (0.004584) corresponds to an FR budget of about B ≈ 143 on a log-linear extrapolation of those two points. This extrapolation is an estimate, not a measurement.
  - So the SC gain is worth roughly 15 extra random calibration sequences, which cost seconds of Hessian replay.
- The sketch pass is amortized across allocation seeds and could be reused across budgets. Even so, under a compute budget, adding random sequences is far cheaper.
- SC is useful when B itself is fixed, for example by a cap on calibration data, on Hessian replay, or on per-expert token storage. It is not useful as a compute-efficient route to lower reconstruction error.

### Classification
**SUPPORTED (limited).**
- SC vs FR passes the frozen rule as ROBUST-SUPPORTED; SC vs TC is also ROBUST-SUPPORTED.
- The gain generalizes from the 8-expert, single-layer MVP to all 240 experts in 4 layers, and it survives calibration-grouping noise.
- Limits:
  - (i) Against Typical Random the gain is consistent but below the 0.5 % materiality bar (NO-GAIN under the rule).
  - (ii) No perplexity improvement is established.
  - (iii) The effect is small: about the value of ~15 extra random sequences, bought at roughly 3,500 sequence-forwards of profiling.
  - (iv) Only one model, 4 of 24 layers, B = 128, 4-bit GPTQ g128.
