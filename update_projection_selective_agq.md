# Projection-selective AGQ: small-scale feasibility MVP

**Frozen verdict: `PROJECTION-AGQ-NO-GAIN`. Recommendation: B, retain uniform AGQ or layer-wise uniform selection.**

The rule was frozen in `runs_projection_selective_agq/protocol.json` before any GU-AGQ / Down-AGQ result and is
implemented in `moe_calib.projection_select.classify`.

**Main result:**
- The validation-selected projection method uses GU-AGQ on L8 and Full AGQ on L16.
- On the primary confirmation split it beats layer-wise uniform by **−0.117 %** of the 2-layer mean (window bootstrap
  95 % CI [−0.156, −0.080] %). The old test split agrees at −0.130 %.
- It beats uniform Full AGQ by −0.163 %.
- The effect is consistent but **4× smaller than the frozen 0.5 % bar**, so criteria 2 and 6 fail.

**Main mechanism:**
- **AGQ's benefit lives almost entirely in gate/up.**
  - On L16, gate/up accounts for 93 % of the Full-AGQ gain (Shapley over the two projection groups).
  - On L8, AGQ on down *hurts* (+0.34 %) and cancels a real gate/up gain (−0.23 %). That explains L8's historical
    Full-AGQ regression.

**Scope:**
- Qwen1.5-MoE-A2.7B, routed experts of **L8 and L16** (chosen with knowledge of earlier AGQ results).
- FR s0 only: C4 Random B = 128 × 512, CANONICAL-B1, GPTQ 4-bit g128 percdamp 1.0 act-order.
- The four configurations were assembled **only** from the two existing validated candidate weight sets. There were no
  new solves, Hessians, data or windows.
- Both test splits were inspected in earlier studies, so this is a retrospective, exploratory pilot with one seed and
  no PPL.

**Artifacts:**
- Run outputs: `runs_projection_selective_agq/`.
- Code: `diagnostics/projection_selective_agq.py`, `diagnostics/report_projection_selective_agq.py`,
  `moe_calib/projection_select.py`, `tests/test_projection_selective_agq.py`.
- No production code was changed.

---

## 1. Audit and regression gates

**Audit** (`artifact_inventory.json`, `configuration_manifest.json`):
- Both candidate sets are present for L8 and L16, and their file sha256 values match the historical records.
- Each set has 60 experts × 3 projections, uniform shapes, BF16, all finite, and no missing tensors.
- Both sets use the same calibration rows, manifest and GPTQ configuration.
- **Configuration manifest:** `build_projection_config(cands, config, 60)` maps every installed projection to its
  source candidate and tensor hash. It raises on any missing tensor and never falls back to BF16.
- **Projections taken from AGQ:**

  | GPTQ | GU-AGQ | Down-AGQ | Full AGQ |
  |---|---|---|---|
  | 0 | 120 | 60 | 180 |

**Regression gate** (`candidate_regression.json`, old test split, A and D only, run before B/C were evaluated anywhere):

| | A (GPTQ) | historical | D (Full AGQ) | historical |
|---|---|---|---|---|
| L8 | 0.007173198 | 0.007173198 | 0.007177917 | 0.007177917 |
| L16 | 0.007090592 | 0.007090592 | 0.006871301 | 0.006871301 |

- Every per-window value matches exactly, the native BF16 values match exactly, and the totals agree to ≤ 2e-16
  relative.
- The validated `LayerEvalAccumulator` agrees on every window for all four mixed configurations on the tuning and
  confirmation splits (total |Δ| ≤ 6e-18). This confirms the complete gated MLP is computed with the actual mixed
  weights.

## 2. Validation and frozen selection (`validation_results.json`, `selection_manifest.json`)

Tuning rows 2000–2127, E_layer × 1e-3:

| Layer | GPTQ | GU-AGQ | Down-AGQ | Full AGQ | selected (margin vs best uniform) | layer-wise uniform |
|---|---|---|---|---|---|---|
| L8 | 7.115237 | **7.095939** | 7.137903 | 7.116387 | **GU-AGQ** (−0.271 %) | GPTQ |
| L16 | 7.145263 | 6.930766 | 7.121990 | **6.926337** | **Full AGQ** (0.000 %) | Full AGQ |

The manifest was frozen (sha recorded) before any confirmation or old-test evaluation. Its sha was re-verified at every
evaluation.

## 3. Test results (`confirmation_results.json`, `old_test_results.json`, `paired_comparisons.json`)

Confirmation split (primary), E_layer × 1e-3, lower is better:

| Layer | GPTQ | GU-AGQ | Down-AGQ | Full AGQ | Selected |
|---|---|---|---|---|---|
| L8 | 7.096580 | **7.080248** | 7.120846 | 7.102910 | 7.080248 (GU) |
| L16 | 7.029577 | 6.816676 | 7.004825 | **6.810734** | 6.810734 (Full) |
| Mean | 7.063079 | 6.948462 | 7.062835 | 6.956822 | **6.945491** |

**Selected method on the confirmation split (2-layer mean):**

| vs | 2-layer mean | 95 % CI | windows improved |
|---|---|---|---|
| uniform GPTQ | −1.665 % | — | — |
| uniform Full AGQ | −0.163 % | [−0.190, −0.139] % | L8: 85 % |
| validation-selected layer-wise uniform (GPTQ on L8, Full AGQ on L16) | **−0.117 %** | [−0.156, −0.080] % | L8: 64 % |

- The per-layer difference vs the best uniform is −0.230 % on L8 and 0 on L16.

**Old test split:**
- Selected vs layer-wise: −0.130 % [−0.184, −0.079] %.
- Selected vs Full AGQ: −0.163 %.
- The ranking is the same on all three splits.

**Native BF16 forward:** gives the identical ranking of the four configurations on every layer and split.

GU-AGQ applied uniformly to both layers would have scored 6.948462: −0.12 % vs Full AGQ. This is not the frozen
method and is shown for context only.

**Frozen criteria:**

| criterion | result | detail |
|---|---|---|
| 1 | ✓ | GU-AGQ is selected on L8, beats both uniforms on validation and keeps that on the confirmation split |
| 2 | ✗ | −0.117 % vs layer-wise; −0.5 % was required |
| 3 | ✓ | no layer regresses |
| 4 | ✓ | the old test split agrees |
| 5 | ✓ | all correctness checks pass |
| 6 | ✗ | −0.163 % vs uniform AGQ; −0.5 % was required |

- **Fallback categories:**
  - Not SELECTION-FAIL: the preference transfers.
  - Not LOCAL-GAIN: the best mixed-vs-uniform layer gain is −0.23 %, below the 0.5 % bar.
  - Not INTERACTION-ONLY: |I| / Full-AGQ gain on L16 is 0.086, below 0.25, and the share of experts where a mixed
    config wins by ≥ 1 % is 0 % on L8 and 1.7 % on L16.
- That leaves **NO-GAIN**.

## 4. Projection interactions (`projection_interactions.json`; confirmation split, % of E_A; gains positive = better)

| | G_GU given GPTQ down | G_D given GPTQ gate/up | G_GU given AGQ down | G_D given AGQ gate/up | I = E_D−E_B−E_C+E_A | Full-AGQ gain |
|---|---|---|---|---|---|---|
| L8 | +0.230 | **−0.342** | +0.253 | **−0.319** | −0.023 (slightly synergistic) | −0.089 |
| L16 | **+3.029** | +0.352 | **+2.761** | +0.085 | **+0.268 (antagonistic)** | +3.113 |

**L16:**
- AGQ on gate/up gives almost all of the benefit (Shapley share 93 %).
- Down-AGQ alone helps by 0.35 %. On top of AGQ gate/up it adds only 0.085 %.
- The interaction is **antagonistic**: the gains overlap, so the full effect is about 8.6 % smaller than the sum of the
  two separate gains.

**L8:**
- The two groups pull in opposite directions: gate/up AGQ helps by 0.23 % and down AGQ hurts by 0.34 %.
- The interaction is negligible (0.02 %), so the effects are close to additive.
- Full AGQ's historical L8 regression is the net of a gate/up gain and a larger down loss.

**Stability:** the same pattern holds on the tuning and old test splits. The interaction is −0.03 / −0.03 % on L8 and
+0.26 / +0.28 % on L16.

**Per expert** (`expert_statistics.json`, confirmation split, c²-weighted complete-expert error), the configuration with
the lowest error:

| | GU-AGQ | Full AGQ | GPTQ | Down-AGQ |
|---|---|---|---|---|
| L8 | 46 | 5 | 8 | 1 |
| L16 | 31 | 27 | 2 | 0 |

- The median per-expert change vs GPTQ on L8 is −0.22 % for GU-AGQ and +0.33 % for Down-AGQ.
- Almost no expert gets ≥ 1 % from mixing beyond the better uniform configuration.

## 5. L16 expert 40 (post hoc, `expert40_attribution.json`)

Expert 40 was identified from earlier results. This analysis is mechanism exploration only and played no part in
selection.

- **Profile:** calibration exposure of 4,086 tokens (natural-exposure decile 2), mean affinity 0.152, and affinity
  CV 1.07–1.09, the highest in L16.
- **Complete-expert c²-weighted error vs GPTQ (confirmation split):** GU-AGQ −48.3 %, Down-AGQ −6.1 %, Full AGQ −49.3 %.
  The tuning and old test splits agree to within 1 point.
- **Attribution:** gate/up accounts for **93 %** (Shapley), down for 7 %. The interaction is antagonistic and equal to
  10 % of the full gain.
- **Share of L16:** this one expert supplies about **84 %** of L16's summed Full-AGQ c²-error reduction.
- **Conclusion:** AGQ's L16 benefit is mostly a gate/up effect, concentrated in one high-variance-affinity expert.

## 6. Error decomposition (`error_decomposition.json`; D + C identity residual ≤ 2e-16, D + C vs direct ≤ 6e-11)

Confirmation split. D is shown as its change vs GPTQ; C is shown as its share of the total.

| | L8 D | L8 C | L16 D | L16 C |
|---|---|---|---|---|
| GU-AGQ | −0.252 % | +0.018 % | −3.043 % | +0.008 % |
| Down-AGQ | +0.335 % | +0.002 % | −0.355 % | −0.004 % |
| Full AGQ | +0.071 % | +0.014 % | −3.125 % | +0.006 % |

- **Every difference is diagonal.** Cross terms stay at most 0.02 % of the total in every configuration.
- **L8 GU-AGQ:** the cross term rises by 1.6e-6 relative to GPTQ. That erodes about 9 % of its diagonal gain (−0.252 %
  diagonal, −0.230 % total) but does not reverse it.

## 7. Cost (`cost_results.json`)

**Incremental MVP:**

| Item | Value |
|---|---|
| New quantization | 0 s |
| Weight loading | 21 s |
| Regression gate | 119 s |
| Validation | 200 s (4 configs + accumulator cross-check + native) |
| Selection | 36 s (mostly hashing the 4.7 GB tuning caches for the manifest) |
| Confirmation split | 352 s |
| Old test split | 93 s |
| Peak GPU | 10.0 GB |
| Peak RSS | 8.7 GB |
| Temporary storage | 8 MB |

**Fresh deployment, 2 layers (estimated):**

| Method | Estimate |
|---|---|
| uniform GPTQ | ≈ 144 s |
| uniform AGQ | ≈ 144 s |
| layer-wise | ≈ 319 s |
| projection-selective | ≈ 438 s (≈ 3.0× one run) |

- The validation pass in the projection-selective estimate is an upper bound because it includes the accumulator
  cross-check.
- Like layer-wise selection, it needs both candidate sets. Its only extra cost over layer-wise is scoring two more
  configurations on validation.
- Final storage is one tensor per projection, in an unchanged format, with no routing parameters. Inference was not
  measured.

## 8. Correctness (`correctness_results.json`: 20/20 pass)

- A, B, C and D each use the correct sources.
- Every layer has 60 experts with 3 projections each, and the candidates are unchanged.
- Router logits are torch.equal for every window and configuration.
- The shared expert, router, attention and norms are unchanged: the non-target layer digest was checked after every
  install.
- The historical A and D values are reproduced per window.
- Complete mixed expert outputs were cross-checked against the accumulator.
- The selection manifest was frozen before the test results, and its sha was verified at every evaluation.
- Reinstalling a configuration gives bitwise-identical outputs.
- The native ranking agrees with the evaluator.
- D + C reproduces the numerator.
- Pristine weights are restored.
- Tests: `tests/test_projection_selective_agq.py` 14/14, and the full suite 589/589.

## 9. Answers

1. **Does GU-AGQ beat GPTQ and Full AGQ on L8?** Yes: −0.23 % vs GPTQ and −0.32 % vs Full AGQ on the confirmation
   split. This is consistent across the tuning, confirmation and old test splits, with 64–72 % of windows improved
   vs GPTQ.
2. **Does Down-AGQ beat them on L8?** No. It is the worst configuration: +0.34 % vs GPTQ and +0.26 % vs Full AGQ.
3. **Best configuration for L16?** Full AGQ (6.8107e-3). GU-AGQ is close behind at +0.087 %.
4. **Does mixing add anything beyond Full AGQ on L16?** No. Both mixed configurations are worse.
5. **Does the tuning selection generalize?** Yes. The selections (GU on L8, Full on L16) rank the same on the
   confirmation and old test splits.
6. **Does it beat layer-wise uniform?** Yes, but only slightly: −0.117 % on the confirmation split and −0.130 % on the
   old test split. The 95 % CIs exclude 0, but the effect is far below the frozen 0.5 % bar.
7. **Additive, synergistic or antagonistic?**
   - L16: antagonistic. I = +0.27 % of E_A, about 9 % of the Full-AGQ gain.
   - L8: near-additive and slightly synergistic (−0.02 %). The main effect is that the two groups act in opposite
     directions.
8. **Expert 40:** mostly gate/up (93 %). Down contributes 7 %, and the interaction is antagonistic (10 %).
9. **Diagonal or cross terms?** Diagonal. Cross terms are ≤ 0.02 % of the total and only trim about 9 % of L8 GU-AGQ's
   gain.
10. **Is the gain worth two candidate sets?** Not as a method. It costs 2–3× the preparation of uniform AGQ for 0.16 %
    better 2-layer local error. Relative to layer-wise selection, which needs the same two candidates, the extra
    0.12 % comes almost for free, but it is still a sub-threshold local gain on one seed and two outcome-selected layers.

## Recommendation

**B: retain uniform AGQ, or layer-wise uniform selection.** Mixed projection configurations give no improvement that
meets the frozen 0.5 % criterion. The study does not proceed to multi-seed validation (A not met).

Two exploratory mechanism notes, worth carrying into the write-up but not used for any decision:
- **Gate/up carries the benefit:** AGQ's benefit is a gate/up effect.
- **Down can hurt:** affinity-weighting down was harmful on L8.

Both come from one seed and two outcome-selected layers.
