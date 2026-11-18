# Small-scale c²-weighted Hessian pilot (EXPLORATORY extension of AGQ)

**Frozen screening verdict: `C2-NO-GAIN`.**
- Replacing AGQ's c-weighted Hessian with a c²-weighted Hessian does **not** help on the pilot.
- It is worse than q = 1 on every held-out expert metric, including the c²-weighted objective it targets
  (pooled +0.83 %, 11 of 12 experts worse).
- In subset replacement it raises the aggregated MoE output error slightly on both layers (L8 +0.10 %, L16 +0.07 %
  vs q = 1).
- Cross-expert terms are negligible. The likely mechanism is a halving of the effective calibration sample size.
- **Recommendation B:** retain the original AGQ (c¹).

## Status and limits

- c¹ AGQ is published (MoEQuant). c² is a **proposed modification**, and only a *diagonal* approximation of the MoE
  output-error objective.
- **Scope:**
  - One seed (FR s0, C4 Random B = 128 × 512, CANONICAL-B1).
  - Two layers, **chosen with knowledge of the AGQ results**: L8 was slightly hurt by AGQ, L16 improved most.
  - 12 experts, chosen only by calibration routed-token counts:
    - L8: low 21, 3 / median 37, 28 / high 44, 52.
    - L16: low 1, 6 / median 41, 47 / high 30, 9.
- **Not measured:** no PPL, and no claims of significance or of full-layer or model-level effects.
- **Freezing:** all rules were frozen in `runs_agq_c2_pilot/protocol.json` before any q = 2 result.

Artifacts:
- Run outputs: `runs_agq_c2_pilot/`.
- Code: `diagnostics/agq_c2_pilot.py`, `tests/test_agq_c2.py`, and helpers in `moe_calib/agq.py`.
- One additive argument in `moe_calib/profiler.py`: `affinity_power`, default 1.

## Correctness (`correctness_results.json`, `oracle_results.json`, `regression_results.json`)

- **Implementation:** the row weight is cᵠ, so the rows are scaled by c^(q/2). q = 2 gives Xᵀ diag(c²) X.
- **Real-data oracle:** on layer 8 with all 60 experts and q = 0/1/2, the worst relative error is 2.0e-7. The result is
  far from c⁴ and c¹, and the dead columns are unchanged.
- **Hard regression gates:**
  - q = 0 reproduces the historical GPTQ weights **bitwise** on all 36 projections.
  - q = 1 reproduces the historical AGQ weights **bitwise** on all 36 projections.
- **Tests:** 60/60 AGQ + c² tests pass, and the full repository suite passes.
- **Decomposition check:** D + C reconstructs ‖Δy‖² with a relative residual below 1e-6.
- **Consistency check:** condition A equals the earlier full GPTQ evaluation exactly.

## Answers

**1. Does c² change Hessian structure beyond c?** Yes, roughly twice as much. The changes below are trace-normalized,
with the shape residual after removing scale in parentheses.

| | gate/up | down |
|---|---|---|
| q0 → q1 | 0.145 (0.131) | 0.334 (0.267) |
| q0 → q2 | 0.309 (0.269) | 0.731 (0.481) |
| q1 → q2 | 0.162 (0.147) | 0.350 (0.247) |

- The diagonal correlation q0→q2 drops to 0.887 for down_proj.
- About 99 % of act-order positions change.
- About 59–64 % of integer codes change, versus q = 0 or q = 1.

**2. Effective sample size.** The mean N_eff/N is 1.00 for q = 0, **0.79** for q = 1 and **0.45** for q = 2.
- The minimum is 0.65 for q = 1 and **0.23** for q = 2.
- c² weighting roughly halves the effective calibration data relative to AGQ. Some experts are left with under a
  quarter of their nominal tokens.

**3. Does q = 2 improve the held-out c²-weighted expert objective?** **No.** These are the pooled held-out values over
the 12 experts:

| | E0 (unweighted) | E1 (c) | E2 (c²) |
|---|---|---|---|
| q1 vs q0 | +0.77 % | +0.30 % | −0.10 % |
| q2 vs q0 | +2.58 % | +1.61 % | +0.73 % |
| **q2 vs q1** | **+1.80 %** | **+1.32 %** | **+0.83 %** |

- **Mean of per-expert ratios, q2 vs q1:** E0 +1.87 % (0/12 improved), E1 +1.45 % (0/12), E2 +1.01 % (1/12).
- **By exposure, E2 q2 vs q1:** low +1.14 %, median +1.33 %, high +0.48 %. Every group is worse; the low and median
  exposure experts suffer most.
- **Projection level** (the Hessian's own objective, pooled, P2 = c²-weighted): q2 vs q1 is +1.08 % (gate),
  +1.02 % (up) and +0.84 % (down).

**4. Does it worsen ordinary reconstruction?** Yes: E0 is +1.8 % vs q = 1 and +2.6 % vs q = 0. The weight quantization
error rises slightly (0.1052 → 0.1054 → 0.1057 for q = 0/1/2).

**5–6. Aggregated output, subset replacement.** Six of 60 experts are replaced; the other 54 stay at the historical
q = 0 weights. These are **not** full c² layers.

| | A: all q = 0 | B: 6 experts q = 1 | C: 6 experts q = 2 | q1 vs q0 | q2 vs q0 | **q2 vs q1** | held-out sequences with C < B |
|---|---|---|---|---|---|---|---|
| L8 | 7.173198e-3 | 7.172916e-3 | 7.180264e-3 | −0.004 % | +0.098 % | **+0.102 %** | 25 % |
| L16 | 7.090592e-3 | 7.089892e-3 | 7.094984e-3 | −0.010 % | +0.062 % | **+0.072 %** | 23 % |

- The native BF16 block forward agrees: q2 vs q1 is +0.102 % (L8) and +0.073 % (L16).
- Neither layer improves. Both regress, but below the 0.5 % regression threshold, so the verdict is NO-GAIN rather
  than REGRESSION.
- The absolute effects are small because only 10 % of a layer's experts change.

**7. Does the diagonal c² error explain the change?** Yes. Expressed as a fraction of condition A's total, the
diagonal change is +0.096 % (L8) and +0.065 % (L16) for C, against total changes of +0.098 % and +0.062 %.

**8. Do cross-expert terms offset gains?** No. |cross / total| is below 1e-4 in every condition. The cross changes are
≤ 0.004 % and do not mask any gain; there is no gain to offset.

**9. Additional cost?** None measurable. Each figure is the sum over both layers of the Profiler replay plus 36 GPTQ
solves.

| | total time | peak GPU |
|---|---|---|
| q = 0 | 31.8 s | 7.76 GB |
| q = 1 | 32.9 s | 7.97 GB |
| q = 2 | 31.7 s | 8.17 GB |

- The differences are within run-to-run noise.
- q = 2 adds one `pow` per (expert, sequence) and no forward passes.
- The peak-GPU differences reflect the run order of the per-q replays within one process (+5 % vs q = 0, inside the
  frozen 10 % limit).

**10. Enough evidence for a full-scale c² experiment?** No.
- The frozen promising criterion requires a −1 % pooled E2 gain vs q = 1; the observed value is **+0.83 %**.
- It also requires an output improvement in at least one layer; neither layer improved.
- This is despite selecting L16, the layer where AGQ helped most.

## Interpretation

- **What c² gets right:** aligning the Hessian weight with the diagonal of the output objective is correct in
  expectation.
- **What goes wrong:**
  - With router affinities this skewed (CV of c about 0.5–0.7), c² concentrates the Hessian on few tokens, and N_eff
    halves.
  - GPTQ then fits those calibration tokens more closely but generalizes worse to held-out tokens, even under the c²
    metric itself.
  - The low- and median-exposure experts, with the fewest effective tokens, degrade most.
- **Where AGQ sits:** c¹ appears to sit near a bias–variance balance point between uniform weighting (low variance,
  misaligned objective) and c² (aligned objective, high variance).
- **Status of this explanation:** a hypothesis consistent with the N_eff diagnostics. No exponent sweep was run, by
  design.

## Recommendation

**B: retain the original AGQ (c¹).** c² shows no incremental benefit, so no full-layer c² experiment is warranted. This
also closes AGQ-method development. Combined with `update_agq_only_validation.md` (small, consistent, free c¹ gain in
layer output, with no PPL effect), the CS498 report can be finalized with AGQ as the retained variant.
