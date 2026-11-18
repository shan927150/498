# AGQ-only validation: standard GPTQ vs Affinity-Guided GPTQ on matched C4 Random B = 128

**Primary verdict (rule frozen in `runs_agq_only_validation/protocol.json` before any quality result): `AGQ-LAYER-QUALITY-GAIN`.**

Main result:
- AGQ lowers the held-out aggregated MoE-layer output error by **1.0–1.5 %**. This holds on all 3 matched C4 seeds and on
  3 of 4 layers (L8 is +0.1–0.2 %).
- The cost is essentially zero.

Secondary results:
- **Expert reconstruction:** the standard metric is consistently **worse** (+1.6–1.7 %, ≤ 3/240 experts improved).
- **AGQ's own surrogate:** the pre-registered c-weighted surrogate does **not** improve (+0.1–0.25 %).
- **Partial-model PPL:** inconclusive (ΔPPL +0.007 / −0.015 / −0.006).

An exploratory post-hoc probe explains the mismatch. The layer-output error is a **c²**-weighted expert error, and AGQ's
c-weighting moves GPTQ part of the way from uniform weighting toward it.

**Scope:**
- Qwen1.5-MoE-A2.7B, BF16, with routed experts of layers 1/8/16/23 quantized (240 experts, 720 tensors).
- GPTQ 4-bit, g128, percdamp 1.0, act-order, non-sequential.
- Calibration is the frozen FR s0/s1/s2 sets (128 × 512, CANONICAL-B1).
- This is AGQ only: no EBSS and no Hadamard rotation, and it is **not** a reproduction of the full MoEQuant framework.

Artifacts:
- **Run outputs:** `runs_agq_only_validation/`.
- **Code:** `moe_calib/agq.py`, `diagnostics/agq_only_validation.py`, `diagnostics/report_agq_only_validation.py`,
  `scripts/run_agq_only_validation.sh`, `tests/test_agq.py`.
- **Small default-preserving hooks:** `moe_calib/hessian.py`, `moe_calib/profiler.py`, `e3_allocation.py`,
  `moe_calib/partial_model_quant.py`, `diagnostics/official_ebss_validation.py`.

---

## 1. Source fidelity (`paper_fidelity.md`, `logs/source_review.json`)

| question | answer | source |
|---|---|---|
| affinity c | softmax(router logits, fp32) value of each **selected** top-4 expert | Alg. 2 l.7–10, 27; official `save_gate_res_qwen` |
| renormalized after top-k? | **No** for Qwen. `norm_topk_prob=False`, and the official code renormalizes only for Mixtral. | code + checkpoint config |
| c or c² in H | **c**: H = Σ c x xᵀ, implemented as (√c x)(√c x)ᵀ | Eq. 18/19, Alg. 2 l.13, `inp*sqrt(s)` |
| projections | all routed gate/up/down. Each gets its own input and the same c. | official `add_batch_score` hooks every `mlp.experts.*` Linear |
| Hessian normalization | not in the paper. The code uses GPTQ's 2/nsamples running mean, a per-expert global factor. | code |
| released AGQ code | yes (`--AGQ_GPTQ`), but tied to the QuaRot/sequential pipeline. Its rule is reproduced inside our unchanged solver. | repo `d77ba1b` |

Unresolved paper ambiguities, and how each was handled:
- **Pre/post top-k and renormalization:** the paper is silent. Resolved from the code and the model.
- **fp32 vs BF16 c (A1):** the fp32 value is used, as the code does. The difference is ≤ 2⁻⁹.
- **Normalization (A2):** none added. Our solver's relative damping makes GPTQ exactly invariant to a global scale of H.
- **Shared-expert sigmoid weighting:** out of scope, because the shared expert is not quantized.
- **Sequential vs non-sequential capture (A3):** non-sequential BF16 capture for both arms.

Published prior:
- In the paper's own Table 5, for this model, AGQ alone changes the 7-task average by **+0.02** points (49.00 → 49.02)
  and WikiText PPL by +0.01.
- The paper's text and Table 5 also disagree on the DeepSeek/Mixtral AGQ claims.

## 2. Routing affinity and activation alignment (`router_affinity_definition.json`, `capture_alignment_results.json`)

Checks on the installed `Qwen2MoeSparseMoeBlock` source:
- fp32 softmax over 60 experts, then top-4, with no renormalization.
- The weights are cast to BF16 and multiply the expert outputs.
- The shared expert is gated by `sigmoid(shared_expert_gate)`.
- Dispatch rows are **slot-major**, built by `one_hot(sel).permute(2,1,0)` → `where`. They are not in token order.

Real-model check on 2 FR s0 rows × 4 layers, at batch 1:
- Every expert's `gate_proj` input rows equal the router input at the recorded token positions, bitwise (120 checks per
  layer). The `down_proj` row count equals the routed pair count.
- Captured c equals the model's own routing weights exactly.
- Unrouted experts receive no rows. The model still calls them with an empty batch; the first unit-regression attempt
  surfaced this and it is handled, see §10.
- **The MoE block output is reconstructed bitwise** from the captured expert outputs × BF16(c) plus the shared expert.
- The coefficient sums are < 1 (mean 0.20–0.43 per layer), so the coefficients really are unnormalized.

## 3. Hessian oracle and GPTQ regression (`hessian_oracle_results.json`, `unit_regression.json`)

Hessian oracle:
- The production weighted Hessians on real layer-8 activations match an independent float64 oracle X^T diag(c) X. The
  worst relative Frobenius error is **1.8e-7**.
- They are far from the c² variant (≥ 0.78) and from the unweighted Hessian (≥ 0.86), and they are exactly symmetric.
- The 13 synthetic oracle cases pass: unit, constant, nonuniform, zero, single-token, repeated, linearly dependent,
  different dimensions, mixed routing, chunking/sequence boundaries, multiple slots, permutation, PSD.

Hard gate (passed):
- With the modified code, standard GPTQ (`hessian_weighting="none"`) reproduces the historical FR s0 result **bitwise**:
  weights sha `ecfdb234…`, PPL 9.269004, reconstruction 0.0046095, and every per-window NLL and per-expert score.
- The weighted path with **c ≡ 1** reproduces it bitwise as well.
- A constant power-of-two c gives bitwise-identical GPTQ weights, and c = 0.3 agrees within 1e-5. Damping is relative,
  so a global scale cancels.

## 4. Expert pilot (`smoke_results.json`; layer 8, two high-, two medium- and two low-exposure experts chosen by routed-token rank)

- **AGQ is a real intervention:**
  - The mean c per expert is 0.05–0.10, and N_eff/N is 0.68–0.91.
  - After removing global scale, the residual change in Hessian shape is 5–41 %.
  - About 50–64 % of integer codes and about 97 % of act-order positions change.
  - The quantized weights differ by about 12 % (relative L2), comparable to the quantization error itself.
- **Held-out projection errors:** these move by under about 2 % in either direction.
- **Cost:** Hessian replay took 10.1 s vs 8.1 s.

## 5. Primary matched comparison (FR s0; GPTQ = historical result, admissible via §3)

| Metric | GPTQ | AGQ | Relative difference |
|---|---|---|---|
| Standard expert reconstruction (E3, mean of 240) | 0.0046095 | 0.0046829 | **+1.59 %** |
| Affinity-weighted full-expert reconstruction (mean of 240) | 0.014750 | 0.014764 | +0.09 % |
| **MoE layer output error (fp32, frozen BF16 routing, mean of 4 layers)** | 0.0037166 | 0.0036607 | **−1.51 %** |
| MoE layer output error (native BF16 block forward) | 0.0037340 | 0.0036773 | −1.52 % |
| Partial-model WikiText-2 PPL (BF16 9.2371) | 9.26900 | 9.27556 | +0.07 % (+0.0066) |
| Quantization wall time, matched rerun (s) | 346.8 | 347.4 | +0.16 % |
| Peak GPU memory (GB) | 8.418 | 8.418 | +0.001 % |

## 6. All three seeds (paired; the same calibration rows in both arms)

| | FR s0 | FR s1 | FR s2 |
|---|---|---|---|
| MoE output error, AGQ vs GPTQ (mean of 4 layers) | −1.51 % | −1.03 % | −1.16 % |
| per layer L1 / L8 / L16 / L23 | −1.8 / +0.07 / −3.1 / −1.2 % | −2.1 / +0.23 / −2.3 / −0.9 % | −4.4 / +0.22 / −2.4 / −1.1 % |
| held-out sequences improved, L1 / L8 / L16 / L23 | 66 / 41 / 99 / 83 % | 65 / 30 / 99 / 77 % | 68 / 37 / 98 / 87 % |
| standard expert reconstruction | +1.59 % (3/240 improved) | +1.67 % (3/240) | +1.68 % (2/240) |
| weighted full-expert reconstruction | +0.09 % (91/240) | +0.25 % (80/240) | +0.20 % (77/240) |
| Eq. 18 projection-level weighted loss | +0.20 % | +0.29 % | +0.29 % |
| ΔPPL (AGQ − GPTQ), bootstrap 95 % CI on Δ mean NLL | +0.0066 [−0.0002, +0.0016] | −0.0150 [−0.0026, −0.0006] | −0.0055 [−0.0015, +0.0003] |

- **Gain is robust under the frozen definition:** the gain holds in every measured seed, 3/4 layers each time, and the
  native BF16 block agrees to within 0.02 points.
- **Seeds are not experts:** three seeds are three paired calibration draws. The 240 experts are not treated as
  independent replicates.
- **Standard reconstruction:** it is worse in every layer (+1.0–2.3 %), every exposure decile (+0.95–2.1 %) and the
  p90/worst tails.
- **PPL:** the three ΔPPL values straddle zero (mean −0.0046). Only s1 exceeds the frozen ±0.014 noise band, so the
  overall PPL classification is **inconclusive**. This matches every earlier finding that four-layer partial-model PPL
  barely responds to reconstruction differences.

## 7. Why the layer output improves while the surrogates do not (exploratory, post hoc: `logs/c2_probe_FR_s0.json`)

- **Error decomposition.** The aggregated error is ‖Σₖ cₖ Δₖ(x)‖². Its diagonal part is Σₖ cₖ² ‖Δₖ‖², a
  **c²-weighted** expert error. On L8/16/23 the cross-expert terms are about 0 (the diagonal is 100 % of the total). On
  L1 they are negative (the diagonal is 128 % of the total).
- **Per-layer comparison:**

  | layer | MoE output Δ | c²-weighted expert error Δ | c-weighted (token-pooled) Δ |
  |---|---|---|---|
  | L1 | −1.83 % | −2.20 % | −0.83 % |
  | L8 | +0.07 % | +0.09 % | +0.42 % |
  | L16 | −3.09 % | −3.10 % | −0.59 % |
  | L23 | −1.21 % | −1.18 % | −0.19 % |

- **Interpretation.**
  - Standard GPTQ weights tokens uniformly (c⁰). AGQ uses c¹, and the quantity that reaches the residual stream weights
    them by c².
  - AGQ therefore trades accuracy on low-affinity tokens, which barely affect the layer output, for accuracy on
    high-affinity ones.
  - The headline per-expert-mean c-weighted surrogate stays flat because it averages ratios over experts. Token-pooled,
    the c-weighted error does improve slightly on 3/4 layers.
  - L16, with the highest mean c (0.104), gains most. L8 is the one layer where the c² error does not improve.
- **Status of this explanation.** It is descriptive, from one seed, and was **not** part of the frozen decision. No c²
  variant was run (forbidden by protocol).

## 8. Mechanistic diagnostics (FR s0; `affinity_statistics.json`, `hessian_diagnostics.json`, `weight_diagnostics.json`)

- **Affinity distributions.**
  - Per-expert mean c is 0.029–0.169. The layer means are L1 0.052, L8 0.080, L16 0.104 and L23 0.077.
  - The within-expert CV is 0.52–0.71, and N_eff/N averages 0.74 (min 0.38).
  - Exposure and affinity are separate properties: Spearman(natural p, mean c) = +0.21, and
    Spearman(natural p, CV of c) = −0.41. Rarely used experts have somewhat more variable affinity, not lower affinity.
- **Hessians.**
  - The trace ratio is about 0.08–0.10, and that is pure scale. The raw relative Frobenius difference of about 0.9 is
    mostly that scale, so it is not reported as information.
  - After removing scale, the shape residual is 0.15–0.20 for gate/up and 0.27–0.31 for down. The trace-normalized
    difference is 0.19 for gate/up and 0.37 for down. The diagonal correlation is 0.99 for gate/up and 0.96 for down.
  - Act-order changes in **every** projection, about 98.6 % of positions, although the rank correlation of the diagonals
    stays 0.96. Near-ties in diag(H) reorder easily.
  - Spectral checks were run on the pilot experts only: the damped condition numbers are essentially unchanged.
- **Weights.**
  - About 84 % of dequantized weights, about 59 % of integer codes and about 96 % of group scales differ.
  - The relative L2 difference between the two quantized matrices (0.12–0.13) exceeds the quantization error itself
    (0.105).
  - The error relative to FP weights is nearly identical (0.1051 vs 0.1053).
  - AGQ selects a *different* rounding solution of equal size, redistributed toward high-affinity directions.
- **Variability vs effect (240 experts, not independent).**
  - Spearman(CV of c, Δ weighted error) = +0.17, and Spearman(N_eff/N, Δ weighted error) = −0.17. These are weak.
  - Experts with more variable affinity lose more on *unweighted* error (ρ = +0.51), which is consistent with
    reweighting.
- **Recomputation check.** Recomputing every Hessian and solve reproduced the installed weights of both arms bitwise
  (720 projections).

## 9. Systems cost (`cost_results.json`; FR s0 matched against a fresh GPTQ rerun in the same session; s1/s2 timings overlapped other GPU jobs and are excluded)

| | GPTQ | AGQ | Δ |
|---|---|---|---|
| total quantize + E3 scoring (s) | 346.8 | 347.4 | +0.6 s (+0.16 %) |
| Hessian replay (s) | 36.9 | 38.0 | +1.1 s (+2.9 %) |
| GPTQ solves + scoring (s) | 247.0 | 250.1 | +3.1 s (within run-to-run spread: the unit-affinity rerun took 264.5 s) |
| peak GPU / CPU (GB) | 8.418 / 42.40 | 8.418 / 42.42 | ≈ 0 |

- **Isolated replay** (diagnostics stage, with affinity tracing on): 69.4 s vs 61.2 s.
- **Affinity metadata:** 9.4 MB sparse for 4 layers × 128 × 512 × top-4, vs 63 MB dense.
- **No extra forward passes.** The affinity is computed inside the Hessian-capture replay already used by GPTQ. There
  is no new activation storage, no new calibration generation, and no extra CPU→GPU traffic beyond the c vectors.
- **No inference claims.** The bit-width is the same, so no inference-speed or memory claims follow.

## 10. Correctness record, tests and post-freeze changes

- **Tests:** `tests/test_agq.py` has 34 tests covering all 30 required items, and the full repository suite passes
  (`logs/pytest_all.log`). Every gate in `decision.json` is true.
- **Post-freeze changes, all disclosed:**
  1. **Empty-expert guard.** The first `unit_regression` attempt stopped at the alignment check: the model calls
     unrouted experts with a 0-row batch. A guard was added. It is a bug fix with no numerical effect, it came before
     any AGQ quality result, and the regression then passed bitwise.
  2. **Classifier alignment.** `classify()` used the mean of per-layer relative differences, while the protocol text
     defines the relative difference of the 4-layer mean. The code was aligned to the protocol text after the s0
     evaluation was seen. Both give the same verdict (−1.52 % vs −1.51 %).
  3. **c² probe.** The `c2_probe` stage (§7) was added post hoc as exploratory.
- **Code hashes:** `protocol.json`'s `code_sha256` records the pre-change hashes.

## 11. Verdict and recommendation

- **Primary: `AGQ-LAYER-QUALITY-GAIN`.** The gain is a 1.0–1.5 % reduction of the aggregated MoE output error, in
  3/3 seeds and 3/4 layers.
- **Secondary:**
  - standard reconstruction: regression (+1.6 %)
  - weighted surrogate: no gain
  - PPL: inconclusive
  - cost: about 0
- **Practical value.** It is a free, consistent, but **small** local improvement that does not reach model-level PPL in
  this four-layer setting. This matches the authors' own Qwen AGQ-only effect (+0.02 points).

**Recommendation: C — stop algorithm development and finalize the CS498 report.**
- Include AGQ as a validated, zero-cost local improvement.
- Include the methodological finding that the right evaluation target is the c²-weighted MoE output, not per-expert
  reconstruction, which AGQ actually worsens.
- Option A (larger scope) is not justified: the effect is small and invisible at the PPL level.
- Option B (EBSS × AGQ) is not justified: EBSS already failed (`update_ebss_*`).
- If anything were pursued later, the untested hypothesis from §7 would be the specific one: c²-weighted Hessians,
  which match the layer-output objective exactly. **Not executed.**
