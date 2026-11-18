# Expert-wise GPTQ/AGQ hybrid: small-scale feasibility MVP

**Frozen verdict: `HYBRID-NO-GAIN`. Recommendation: B, prefer simple layer-wise objective selection.**

The rule was frozen in `runs_expertwise_agq_hybrid/protocol.json` before any selection or hybrid result and is implemented
in `moe_calib.hybrid.classify`.

Main result:
- The expert-wise hybrid beats uniform GPTQ (−1.53 %), as every AGQ-containing configuration does.
- It ties uniform AGQ (−0.03 %) and validation-selected layer-wise uniform (+0.015 %) on the primary confirmation split.
- The old test split agrees: −1.54 % / −0.04 % / −0.003 %.
- The only gain worth having comes from choosing **per layer** (GPTQ on L8, AGQ on L16), not per expert.
- Expert-level preferences on L8 are mostly noise; on L16 they are dominated by one expert.

**Scope and caveats:**
- Qwen1.5-MoE-A2.7B BF16. Routed experts of **L8 and L16 only** (120 experts × 2 existing candidates).
- FR s0 only: C4 Random B = 128 × 512, CANONICAL-B1, GPTQ 4-bit g128 percdamp 1.0 act-order, non-sequential.
- The layers were chosen **knowing** earlier AGQ outcomes, so they are not a confirmatory sample.
- Both test splits were inspected in earlier studies, so this is a retrospective, exploratory pilot.
- No PPL. One seed. Two-layer means are **not** comparable to the historical four-layer means.

Artifacts:
- Run outputs: `runs_expertwise_agq_hybrid/`.
- Code: `diagnostics/expertwise_agq_hybrid.py`, `diagnostics/report_expertwise_agq_hybrid.py`, `moe_calib/hybrid.py`,
  `tests/test_expertwise_hybrid.py`.
- No existing solver, Hessian, routing or historical artifact was modified.

---

## 1. Method (as frozen)

- **Candidates:** the two historical weight sets were reused with **no new quantization**:
  - q = 0, standard GPTQ: `~/moe_cache_ebss_official_validation/weights/FR_s0`.
  - q = 1, AGQ: `~/moe_cache_agq_only/weights/AGQ_FR_s0`.
- **Selection score:** computed on the tuning split, `~/moe_cache` rows 2000–2127. The FR s0 calibration rows are all < 2000.
  - J_e^(q) = Σ_{i: e∈top4(i)} c_{i,e}² ‖E_e^FP(x_i) − E_e^(q)(x_i)‖².
  - It uses the full expert MLP in fp32, frozen BF16 inputs and routing, and the fp32 softmax top-4 c.
- **Rule:** z_e = AGQ iff J_e^(1) < J_e^(0). Ties go to GPTQ. All three projections come from one candidate.
- **Baselines:**
  - A: uniform GPTQ.
  - B: uniform AGQ.
  - C: per layer, all-AGQ iff its *actual* tuning-split layer-output error is lower.
- **Primary metric:** E_layer = Σ‖y_BF16 − y_q‖² / Σ‖y_BF16‖². The shared expert is included and routing is frozen. This is
  the validated `LayerEvalAccumulator` definition.
  - Primary split: `~/moe_cache_confirmation` rows 0–255, the same windows as the legacy re-evaluation.
  - Secondary split: old test rows 2128–2255.

## 2. Artifact audit and regression (`artifact_inventory.json`, `candidate_regression.json`)

- **Candidate files:** both sets are present for L8/L16.
  - The file sha256 values match the historical records.
  - Each set has 60 experts × {gate, up, down}, in BF16, all finite, and the same calibration manifest.
  - Every expert's GPTQ and AGQ tensors differ.
  - **No reproduction was needed.**
- **Old-test regression before selection:**
  - The uniform GPTQ and AGQ layer-output errors reproduce the AGQ-only validation per window exactly. The totals agree
    to ≤ 2.2e-16 relative (float64 summation order).
  - L8: GPTQ 7.173198e-3, AGQ 7.177917e-3.
  - L16: GPTQ 7.090592e-3, AGQ 6.871301e-3.
  - The native BF16 values match exactly.
  - The validated `LayerEvalAccumulator` agrees to ≤ 1.7e-18 absolute.
- **Confirmation regression:** uniform GPTQ reproduces the legacy re-evaluation values (L8 7.096580e-3, L16 7.029577e-3).

## 3. Validation selection (`validation_expert_scores.json`, `selection_manifest.json`, `selection_stability.json`)

| | AGQ chosen | GPTQ chosen | half-split agreement | half-split Spearman of Δ | experts with \|Δ\| < 1 % |
|---|---|---|---|---|---|
| L8 | 30 | 30 | **0.48** | 0.11 | 95 % |
| L16 | 46 | 14 | 0.78 | 0.53 | 75 % |

**Layer-wise baseline (validation):**
- L8 → **GPTQ** (AGQ +0.016 %).
- L16 → **AGQ** (AGQ −3.06 %).
- The native BF16 forward picks the same.

**Exploratory, post hoc: L16's AGQ gain is concentrated in one expert.**
- Expert 40 improves by **49 %** on validation and 49 % on confirmation. It carries 5.3 % of L16's GPTQ c²-error, and its
  affinity CV is 1.09, the highest.
- Excluding it, AGQ lowers L16's summed c²-error by only 0.48 % instead of 3.05 %.

## 4. Primary result: aggregated MoE-layer output error (`layer_output_results.json`, `paired_comparisons.json`)

Confirmation split (primary), E_layer × 1e-3, lower is better:

| Layer | Uniform GPTQ | Uniform AGQ | Layer-wise selection | Expert-wise hybrid |
|---|---|---|---|---|
| L8 | 7.096580 | 7.102910 | 7.096580 (GPTQ) | **7.096236** |
| L16 | 7.029577 | **6.810734** | **6.810734** (AGQ) | 6.813160 |
| Mean | 7.063079 | 6.956822 | **6.953657** | 6.954698 |

Hybrid relative differences on the confirmation split:

| Hybrid vs | L8 | L16 | 2-layer mean | bootstrap 95 % CI (mean) | windows improved (L8 / L16) |
|---|---|---|---|---|---|
| Uniform GPTQ | −0.005 % | −3.08 % | **−1.53 %** | [−1.60, −1.47] % | 55 % / 100 % |
| Uniform AGQ | −0.094 % | +0.036 % | **−0.031 %** | [−0.063, +0.001] % | 59 % / 41 % |
| Layer-wise | −0.005 % | +0.036 % | **+0.015 %** | [−0.021, +0.051] % | 55 % / 41 % |

**Old test (secondary):**
- The hybrid's 2-layer mean is 7.022066e-3, against layer-wise 7.022250e-3 and AGQ 7.024609e-3.
- That is −1.54 % vs GPTQ, −0.036 % vs AGQ and −0.003 % vs layer-wise. This is the same conclusion.

**Native BF16 block forward:** gives the same ranking as the fp32 evaluator on both splits and on every layer. On the
confirmation split the hybrid is +0.015 % vs layer-wise.

## 5. Validation → test transfer (`test_expert_scores.json`)

| confirm | sign agreement | Spearman (val Δ, test Δ) | false selections | regret / test J0 | realized vs oracle gain over GPTQ |
|---|---|---|---|---|---|
| L8 | 0.67 | 0.49 | 20 / 60 | 0.076 % | 0.030 % of 0.106 % possible |
| L16 | 0.80 | 0.63 | 12 / 60 | 0.052 % | 3.09 % of 3.14 % possible |
| pooled | 0.73 | 0.67 | 32 / 120 | 0.054 % | — |

- **Transfer criterion:** the pooled, prespecified criterion (≥ 0.65 / ≥ 0.5) formally **passes** on the confirmation split.
  - It is carried by L16, where the large-margin experts transfer.
  - L8's Spearman (0.49) and its half-split agreement (0.48) show little real per-expert signal.
  - On the old test split the pooled Spearman is 0.47, so the criterion would fail there.
- **Margin terciles (confirmation, pooled):**
  - Agreement: small-margin experts 0.50, medium 0.75, large 0.95.
  - Small-margin experts (|Δ_val| < 0.22 %) carry **83 %** of the regret.
- **Even with perfect (oracle) per-expert choice**, the extra gain available beyond uniform AGQ/GPTQ is ≈ 0.1 % of the
  c²-error on L8 (over GPTQ) and ≈ 0.02 % on L16 (over AGQ). There is almost nothing to select.

## 6. Error decomposition (`error_decomposition.json`; D + C identity residual < 1e-6)

Confirmation split. D, C and the total are divided by Σ‖y‖².

| | L8 D | L8 C | L16 D | L16 C |
|---|---|---|---|---|
| GPTQ | 7.09691e-3 | −3.3e-7 | 7.03003e-3 | −4.5e-7 |
| AGQ | 7.10193e-3 | +9.8e-7 | 6.81031e-3 | +4.2e-7 |
| Layer-wise | = GPTQ | = GPTQ | = AGQ | = AGQ |
| Hybrid | **7.09479e-3** | +1.44e-6 | 6.81270e-3 | +4.6e-7 |

- **Cross terms:** they are ≤ 0.02 % of the total for every configuration, so the near-zero cross terms still hold with
  mixed candidate sets.
- **L8:** the hybrid lowers D by 0.03 % vs GPTQ (−2.1e-6). The cross term rises by +1.8e-6 and cancels about 85 % of
  that. The net is −0.005 %. This is negligible either way.
- **L16:** the hybrid's D is 0.035 % *above* uniform AGQ. The 14 GPTQ-selected experts are, on net, small-margin misses.

## 7. Cost (`cost_results.json`)

**Incremental MVP, with historical candidates reused:**
- Candidate generation: 0 s.
- Regression: 139 s.
- Validation scoring: 65 s, of which the lean pass is 31 s.
- Selection: 0.1 s.
- Test evaluation: 256 s on the confirmation split and 83 s on the old test split.
- Peak GPU: 9.9 GB. Peak RSS: 8.8 GB.

**Estimated fresh deployment for 2 layers**, from the historical Hessian + GPTQ seconds / 2:

| method | estimate | relative to one run |
|---|---|---|
| uniform GPTQ | 144 s | 1× |
| uniform AGQ | 144 s | 1× |
| layer-wise | ≈ 319 s | 2.2× |
| hybrid | ≈ 319 s | 2.2× |

- Layer-wise needs the same two candidates and the same validation pass as the hybrid.
- **Storage:** final storage and format are unchanged. Offline, both candidates must be held, which doubles storage.
- **Inference:** no inference change is expected. Nothing was measured.

## 8. Correctness (`correctness_results.json`: all 15 checks pass)

The following checks pass:
- The candidates match historical q = 0 / q = 1 bitwise, and both use the same calibration manifest.
- Router logits are torch.equal under every installed configuration.
- Only routed experts change: the shared expert, its gate and the router digests are unchanged.
- All three projections of each expert come from one candidate (tensor-equality check after install).
- The installed configuration hash equals the manifest hash. The manifest sha was verified before each evaluation, and
  it predates every test result.
- Pristine weights are restored after each layer.
- The evaluator equals the validated accumulator, D + C reproduces the total, and the historical uniform errors are
  reproduced.
- Tests: `tests/test_expertwise_hybrid.py` 13/13, and the full repository suite 575/575 passed.

## 9. Answers

1. **How many experts prefer GPTQ vs AGQ on validation?** 44 prefer GPTQ and 76 prefer AGQ: 30/30 on L8 and 14/46 on L16.
2. **Does preference vary within a layer?** Nominally yes. On L8 that variation is mostly noise: half-split agreement is
   0.48 and 95 % of margins are under 1 %. On L16 it is partly real but dominated by one expert, e40, at 49 %.
3. **Do validation preferences generalize?**
   - Large-margin ones do: 95 % agreement.
   - Small-margin ones do not: 50 % agreement, with 83 % of the regret.
   - The pooled criterion passes on the confirmation split and would fail on the old test split.
4. **Better than uniform GPTQ?** Yes, by −1.53 %. This is entirely the L16 AGQ effect.
5. **Better than uniform AGQ?** Not meaningfully: −0.03 %, CI [−0.063, +0.001] %. The tie comes from avoiding AGQ's
   small L8 penalty.
6. **Better than layer-wise selection?** No: +0.015 % on the confirmation split and −0.003 % on the old test split.
7. **Consistent across L8 and L16?** No.
   - On L8 the hybrid equals the best uniform (−0.005 %).
   - On L16 it is slightly worse than uniform AGQ (+0.036 %).
8. **Does it reduce the diagonal component?**
   - Versus GPTQ: yes, mostly on L16.
   - Versus the layer-wise choice: only by 0.03 % on L8, and it is +0.035 % higher on L16.
9. **Do cross terms offset diagonal gains?** On L8 they cancel about 85 % of a negligible D gain. Elsewhere they are
   immaterial (≤ 0.02 % of the total).
10. **Is the gain worth generating two candidates?** Not for per-expert selection. It costs about 2.2× one quantization
    run for ≤ 0.03 % over uniform AGQ and nothing over the layer-wise choice. The layer-wise choice costs the same and
    captures the whole gain. With L16 in scope, uniform AGQ alone (1×) is within 0.05 %.

## Recommendation

**B: prefer simple layer-wise objective selection.**
- It matches or exceeds expert-wise selection on both splits, and it is the best non-hybrid method on the 2-layer mean.
- The expert-wise hybrid is **not** taken to multi-seed validation (A not met).
- Given the cost, the practical default remains **uniform AGQ**. Layer-wise selection is an optional refinement worth
  ≈ 0.03–0.05 % here.
- This is one seed and two outcome-selected layers, so no generalizable claim is made.
