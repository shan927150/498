# Fixed GU-AGQ across four MoE layers and three C4 seeds

**Frozen verdict: `FIXED-GU-BLOCKED` → Recommendation B, retain Full AGQ.**

**Why BLOCKED:**
- One frozen correctness check failed: check 16, the native BF16 ranking of all four configurations in every cell.
- It failed in **1 of 24 cells**: old-test split, seed 0, L1.
- Two near-tied, non-primary pairs swap order there:
  - GPTQ vs GU-AGQ, fp32 gap −0.17 %.
  - Down-AGQ vs Full AGQ, fp32 gap −0.11 %.
- BF16 rounding alone adds +5.4 % to L1's native error, which is larger than either gap.
- The primary pair, GU-AGQ vs Full AGQ, ranks identically in native and fp32 in all 24 cells.
- The rule is not relaxed post hoc.

**Post hoc, for transparency only:** if check 16 were read as covering the primary pair only, the verdict would be
`FIXED-GU-SMALL-GAIN` → C. The quality conclusion is the same either way. Promising criteria 1 (gain ≥ 0.5 %) and 3 (no
layer regression > 0.5 %) **fail regardless of check 16**.

**Main result:**
- Fixed GU-AGQ beats Full AGQ on the frozen 4-layer mean-of-ratios by only **−0.096 %** on the confirmation split.
  - Per seed: −0.067 / −0.092 / −0.128 %, so 3/3 seeds improve.
  - Old test split: −0.129 %.
- That aggregate gain comes **entirely from L8**. GU-AGQ is *worse* than Full AGQ on L1, L16 and L23 in every seed:
  - L1: +1.20 %.
  - L16: +0.09 %.
  - L23: +0.35 %.
- The secondary pooled ratio (Σnum/Σden) **reverses the sign**: +0.25 to +0.37 % *against* GU-AGQ, because it is
  dominated by L23.
- Removing down-projection AGQ is therefore a net loss everywhere except L8.

**Scope and caveats:**
- **Setup:** Qwen1.5-MoE-A2.7B, routed experts of L1/8/16/23. FR s0/s1/s2, C4 Random B = 128 × 512, CANONICAL-B1.
  GPTQ 4-bit g128 percdamp 1.0 act-order.
- **No new quantization:** every configuration was assembled from the existing same-seed GPTQ and AGQ candidates, with
  0 new solves.
- **Retrospective:** GU-AGQ was motivated by L8/L16 seed-0, and both evaluation splits were inspected before. The
  previously inspected cells are s0-L8 and s0-L16; the other 10 seed-layer cells are new evaluations.
- **Replication:** three seeds is limited replication.
- **Metric:** local, non-sequential, frozen-routing distortion only. No PPL or task accuracy.

**Artifacts:**
- Run outputs: `runs_fixed_gu_agq/`.
- Code: `diagnostics/fixed_gu_agq_validation.py`, `diagnostics/report_fixed_gu_agq.py`, `tests/test_fixed_gu_agq.py`.
- The study reuses `diagnostics/projection_selective_agq.evaluate_layer` and `moe_calib/projection_select.py`.
- **Post-freeze code edits** (disclosed): after the protocol was frozen and before any GU/Down result, a numpy-bool JSON
  serialization fix was made in the regression stage. Report-only changes followed: native-disagreement detail, the
  post-hoc verdict line, the regression-timing source and a figure label. The frozen rule and thresholds are unchanged.

---

## 1. Audit and historical regression (`artifact_inventory.json`, `configuration_manifest.json`, `candidate_regression.json`)

**Audit:**
- 24 candidate files exist: 3 seeds × 4 layers × {GPTQ, AGQ}.
- Their sha256 values match the historical records.
- Each has 60 experts × 3 projections, uniform shapes, BF16, all finite, and no missing tensors.
- Each seed's GPTQ and AGQ use identical calibration rows and GPTQ configuration.
- **Configuration manifest:** traces every installed projection (tensor sha + source file). GU-AGQ takes 120/180
  projections from AGQ, and all sources are same-seed.

**Regression (old-test split, GPTQ and Full AGQ, all 12 cells, before any GU/Down evaluation):**
- Every per-window value equals the AGQ-only validation's, and the native BF16 values are exactly equal.
- The `LayerEvalAccumulator` agrees (per window equal, |Δ| ≤ 1e-17).
- The 4-layer AGQ-vs-GPTQ deltas reproduce the historical values:

  | seed | measured | reported |
  |---|---|---|
  | s0 | −1.505 % | −1.51 % |
  | s1 | −1.032 % | −1.03 % |
  | s2 | −1.159 % | −1.16 % |

- **Confirmation split:** GPTQ equals the legacy re-evaluation values in all 12 cells.

## 2. Primary result: per seed (confirmation split, E_4L = mean of 4 layer ratios, ×1e-3)

| Seed | GPTQ | Full AGQ | Fixed GU-AGQ | GU vs Full AGQ | window-bootstrap 95 % CI |
|---|---|---|---|---|---|
| s0 | 3.681321 | 3.625292 | 3.622856 | −0.067 % | [−0.107, −0.028] |
| s1 | 3.682688 | 3.640863 | 3.637497 | −0.092 % | [−0.128, −0.059] |
| s2 | 3.680302 | 3.634806 | 3.630162 | −0.128 % | [−0.159, −0.096] |
| Mean | 3.681437 | 3.633654 | 3.630172 | **−0.096 %** (median −0.092, spread 0.061 pt) | |

**Other comparisons:**
- GU vs GPTQ: −1.39 % (mean). Full AGQ vs GPTQ: −1.30 %.
- Old test split, GU vs Full AGQ: −0.087 / −0.162 / −0.138 % (mean −0.129 %).
- Native BF16 4L delta: −0.066 / −0.079 / −0.101 %.
- **Secondary pooled ratio, GU vs Full AGQ:**
  - Confirmation split: +0.368 / +0.302 / +0.248 %.
  - Old test split: +0.27 / +0.11 / +0.22 %.

## 3. Per layer (confirmation split, seed-averaged; per-seed values in brackets)

| Layer | GU vs Full AGQ | seeds GU better | GU vs GPTQ | Full AGQ vs GPTQ |
|---|---|---|---|---|
| L1 | **+1.203 %** [+1.93, +0.84, +0.84] | 0/3 | −2.01 % | −3.18 % |
| L8 | **−0.347 %** [−0.32, −0.34, −0.38] | 3/3 | −0.25 % | +0.10 % |
| L16 | +0.090 % [+0.09, +0.10, +0.08] | 0/3 | −2.55 % | −2.64 % |
| L23 | +0.351 % [+0.42, +0.35, +0.29] | 0/3 | −0.76 % | −1.11 % |

**Old test split:** the same signs in every layer. L1 +1.49 %, L8 −0.37 %, L16 +0.04 %, L23 +0.23 %.

**Window fractions (confirmation split):**
- On L8, GU-AGQ improves 85–89 % of windows.
- On L1, L16 and L23 it improves only 14–43 % of windows.

**Why the two aggregates disagree:**
- **Mean-of-ratios (frozen primary):** L8 and L16 ratios (≈ 7e-3) are about 24× those of L1 and L23 (≈ 3e-4). L8's
  −0.35 % therefore dominates.
- **Pooled ratio:** L23 dominates. Its Σ‖y‖² is 1.6e9, versus 0.9e6–1.2e7 for the other layers, and GU is +0.35 % worse
  there.
- **Conclusion:** the "aggregate gain" is a property of the aggregation convention, not of the layers.

## 4. Projection contributions (`projection_contributions.json`; exact 2-group Shapley, Down-AGQ evaluated as a diagnostic)

Confirmation split, seed-averaged, % of E_GPTQ; positive = improvement:

| Layer | φ_GU | φ_D | full AGQ gain | GU share | interaction I/E_A |
|---|---|---|---|---|---|
| L1 | 1.97 | **1.20** | 3.18 | 62 % (per seed: 28 / 66 / 80 %) | +0.07 (antagonistic) |
| L8 | 0.25 | **−0.35** | −0.10 | n/a (gain < 0.5 %) | −0.007 |
| L16 | 2.43 | 0.21 | 2.64 | **92 %** | +0.25 (antagonistic) |
| L23 | 0.76 | **0.35** | 1.11 | 69 % | −0.003 |

- **Gate/up AGQ** helps in every layer and every seed.
- **Down AGQ:**
  - **Hurts on L8 only**, consistently: −0.33 / −0.35 / −0.38 % per seed.
  - **Helps on L1, L16 and L23** in every seed.
  - On L1 it carries 38 % of the gain, unstable across seeds: φ_D is 1.86 / 0.87 / 0.88 %.
- **Generalization:** the "~93 % gate/up" pattern holds only for L16. L1 and L23 are mixed, at 62–69 %.
- **Old test split:** the same picture. GU share is 47 % on L1, 94 % on L16 and 76 % on L23, and φ_D is −0.37 % on L8.

## 5. Diagonal vs cross-expert terms (`error_decomposition.json`)

- **Identity:** D + C reproduces ‖Δy‖² with a residual ≤ 3e-16, and D + C agrees with the direct numerator to ≤ 3e-9.
- **L8, L16, L23:** cross terms are ≤ 0.07 % of the total.
- **L1:**
  - Cross terms are strongly negative: −23 to −30 % of the total in every configuration.
  - GU-AGQ has slightly more cancellation than Full AGQ. This narrows its L1 deficit (ΔD = +1.1e-5 to +1.7e-5,
    ΔC = −0.5e-5 to −1.4e-5) but never flips the ranking.
  - In 0/24 cells do cross terms reverse the GU-vs-Full ranking.
- **Conclusion:** the differences are driven by diagonal (per-expert c²-weighted) error.

## 6. Expert-level gain concentration (`expert_gain_concentration.json`, confirmation split)

G_e = D_e^GPTQ − D_e^method, the c²-weighted complete-expert error.

- **L16 is dominated by expert 40** in every seed, under both GU-AGQ and Full AGQ.
  - It is the top-1 expert in 3/3 seeds and holds 78–84 % of the positive gain.
  - Without it, L16's net diagonal gain drops from 2.2–3.0 % to 0.39–0.48 %.
- **L1 is dominated by expert 0:**
  - Full AGQ: top-1 in 3/3 seeds, 50–95 % of the positive gain.
  - GU-AGQ: e0 in 2/3 seeds; in s1 its GU-AGQ net gain is *negative* (−0.41 %).
- **L23 and L8 are diffuse:**
  - L23: the top-5 hold about 50 % of the gain. The stable top-1 is e8 for GU-AGQ and e13 for Full AGQ.
  - L8: GU-AGQ improves 51–55 of 60 experts with a top-5 share of 21–33 %, and the top experts are not stable across
    seeds (top-5 Jaccard 0–0.11).
- **Correlates:** gain correlates weakly-to-moderately with mean affinity (ρ ≈ 0.1–0.7) and not consistently with
  exposure or affinity CV.

## 7. L16 expert 40 across seeds (`expert40_stability.json`; post hoc)

| seed | calib tokens | affinity mean / CV (rank in L16) | GU-AGQ vs GPTQ | Full AGQ vs GPTQ | GU share of full | share of L16 net gain (GU / Full) |
|---|---|---|---|---|---|---|
| s0 | 4086 | 0.152 / 1.087 (1st) | −48.3 % | −49.3 % | 93 % | 84 % / 84 % |
| s1 | 4010 | 0.152 / 1.069 (1st) | −36.7 % | −38.2 % | 92 % | 79 % / 78 % |
| s2 | 4054 | 0.153 / 1.083 (1st) | −40.8 % | −41.9 % | 91 % | 84 % / 83 % |

Expert 40 is consistently the dominant L16 AGQ beneficiary, and its benefit is consistently about 92 % gate/up.

## 8. Cost (`cost_results.json`)

**Incremental (this study):**
- 0 new GPTQ solves.
- Artifact loading: 42 s.
- Regression: 682 s.
- Confirmation split: 2071 s (4 configs + accumulator + native, 12 cells).
- Old test split: 339 s.
- Peak GPU: 10.0 GB. Peak RSS: 11.5 GB.
- Temporary storage: 34 MB.

**Fresh deployment, 4 layers** (matched s0 runs; Hessian + GPTQ seconds):

| method | cost |
|---|---|
| GPTQ | 288 s |
| Full AGQ | 288 s |
| direct GU-AGQ | ≈ 288 s (**estimated, not measured**) |

- Direct GU-AGQ needs the same number of solves, and its replay is ≤ the Full AGQ replay. Cost ratio vs Full AGQ ≈ 1.0.
- `Profiler` applies `hessian_weighting` to both the gu and down Hessians, so a direct GU-only path would need a new
  opt-in option.
- That option was **not built**, because the frozen quality gate failed.
- The 2-candidate assembly used here is not a deployment cost. Inference was not measured.

## 9. Correctness (`correctness_results.json`: 19/20)

**Passing checks:**
- Each seed uses its own candidates, and all 240 experts are covered.
- Gate/up come from AGQ and down from GPTQ.
- No cross-seed weights and no BF16 fallback.
- Router logits are unchanged (torch.equal on every window and configuration).
- The shared expert and other non-target parameters are unchanged.
- Weights are restored after each evaluation.
- The uniform baselines reproduce, and reinstall is bitwise deterministic.
- Installed tensors equal their sources.
- The same windows and the same BF16 reference are used throughout.
- D + C reproduces the numerator.
- The historical files are unchanged (sha re-verified).
- Tests: 23 targeted tests pass, and the full suite passes 598/598.

**Failing check:**
- **16:** the native-vs-fp32 ranking of all four configurations disagrees in one cell (old-test, s0, L1), on two pairs
  separated by 0.11–0.17 %.

## 10. Answers

1. **Does it beat Full AGQ on the 4-layer mean?** Yes, by −0.096 % (frozen mean-of-ratios), about 5× below the 0.5 % bar.
   The pooled ratio favors Full AGQ (+0.25 to +0.37 %).
2. **Consistently across seeds?** Yes in sign (3/3 seeds on both splits), but the gain is tiny and comes from L8 alone.
3. **Which layers benefit from unweighted down Hessians?** Only **L8**: −0.35 %, 3/3 seeds, both splits.
4. **Do any layers regress substantially?** Yes. **L1 regresses +1.2 %** (0/3 seeds better; +0.84 to +1.93 %), beyond
   the 0.5 % layer limit. L23 regresses +0.35 % and L16 +0.09 %.
5. **Does GU dominance generalize to L1/L23?** Partly. Gate/up is the larger contributor (62 % / 69 %), but down AGQ
   also helps materially there. About 92 % GU dominance holds only for L16.
6. **Consistent across confirmation and old test splits?** Yes. Every layer's sign and the 4L mean-of-ratios sign agree
   on both splits.
7. **Diagonal or cross terms?** Diagonal. L1's large negative cross terms narrow GU-AGQ's deficit there but never flip
   the ranking.
8. **Are gains concentrated?** Yes on L16 (e40 holds about 80 %) and L1 (e0). No on L8 and L23, which are diffuse.
9. **Is L16 expert 40 consistently important?** Yes. It is top-1 in all seeds, holds 78–84 % of L16's gain, and that
   gain is about 92 % gate/up.
10. **Is the quality–cost trade-off better than Full AGQ?** No. The estimated cost is equal, and quality is better by
    0.1 % on one convention and worse on the other. GU-AGQ also loses 3 of 4 layers.
11. **Enough evidence for a direct single-pass variant?** No. The gate failed, so it was not built.

## Recommendation

**B: retain Full AGQ.**
- This is the frozen verdict (`FIXED-GU-BLOCKED`).
- The post-hoc reading (`SMALL-GAIN`, which maps to C) would not change the practical conclusion: GU-AGQ regresses
  three of four layers, including L1 by 1.2 %, and its only gain is L8-specific.

Mechanism findings worth reporting, without changing the method:
- Gate/up AGQ helps in every layer.
- Down-projection AGQ hurts consistently on L8 only and helps on L1, L16 and L23.
- AGQ's L16 benefit is concentrated in expert 40, consistently across seeds.

No further adaptive selection is proposed.
