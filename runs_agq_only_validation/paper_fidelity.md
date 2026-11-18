# AGQ source fidelity (written before any AGQ quantization result)

Sources:
- **Paper:** MoEQuant, arXiv 2505.03804v1 (HTML), Sec. 4.3 Eq. 15–19, Appendix A.1 Table 5.
- **Algorithm 2 figure:** `asserts/Algorithm_2_AGQ.png` in the official repo.
- **Official code:** `chenzx921020/MoEQuant` @ `d77ba1b`, `code/fake_quant/gptq_utils_moe.py`, `main.py --AGQ_GPTQ`,
  `quant_layers/quant_layer.py`.
- **Not used:** OpenReview author responses. They were unreachable (bot wall) in earlier studies. The rebuttal tables
  in the repo (`Tables_for_R*.md`) say nothing about AGQ.

Each statement below is labelled **PAPER**, **ALGO2**, **CODE** or **ASSUMPTION**.

## 1. Affinity definition

- **PAPER:** c = {c_1..c_b} are the "affinity scores provided by the gating network" for the b tokens routed to
  expert E. The routed output is y_i = c_i E(x_i) (Eq. 15). The gating network is a softmax linear layer (Sec. 2.1),
  and g_i(x) is "the weight assigned to the i-th expert" (Eq. 1).
- **ALGO2:** `g ← Softmax(W_g h)` (lines 25–28); `c ← gates[E]` for each `E ∈ topk_experts` (lines 7–10).
- **CODE (Qwen):** `save_gate_res_qwen`: `F.softmax(router_logits, dim=1, dtype=float)` then `topk(·, 4)`. The fp32
  values are stored and **not renormalized**. Mixtral is renormalized there because its model renormalizes; Qwen is
  not.
- **Paper ambiguity:** the paper never says whether c is taken before or after top-k, or whether it is renormalized.
- **Resolved for this checkpoint** from the official code and the actual model forward (`norm_topk_prob=False`): c is
  the top-4 entry of the full 60-way fp32 softmax. This is exactly the coefficient that multiplies the expert output
  in `Qwen2MoeSparseMoeBlock`, before its cast to BF16.
- **ASSUMPTION A1:** the Hessian uses the fp32 value, as the official code does, not its BF16 rounding (relative
  difference ≤ 2⁻⁹).

## 2. c versus c²

- **PAPER Eq. 18:** L = Σ_i c_i ‖W x_i − Ŵ x_i‖².
- **PAPER Eq. 19:** H = (X·√c)(X·√c)ᵀ = (X·c)Xᵀ.
- **ALGO2 line 13:** H ← H + c·aaᵀ.
- **CODE:** `s1 = torch.sqrt(s); inp = inp*s1`.
- **Conclusion:** all three sources agree on weight **c**, not c². Implemented as `(√c ⊙ X)ᵀ(√c ⊙ X)`.

## 3. Which projections

- **PAPER:** Eq. 17 argues that c "propagates through every layer of the expert network": up, down and (approximately)
  gate. It does not list which linear layers receive AGQ.
- **CODE:** `add_batch_score` hooks **every** Linear named `mlp.experts.*`, i.e. gate_proj, up_proj and down_proj. Each
  receives the same c of its (token, expert) pair, and down_proj gets its own input
  `act(gate x) ⊙ up x`.
- **CODE (shared expert):** the shared expert is additionally weighted by `sigmoid(shared_expert_gate)`.
- **Not applicable here:** the shared expert is not quantized in this study (historical scope), so that weighting
  does not arise.
- **Router:** in the official pipeline the router is quantized with plain GPTQ. Here it stays BF16, as in every prior
  study.

## 4. Hessian normalization and damping

- **PAPER:** none stated. Eq. 8 is H = XXᵀ and Eq. 19 is (X·c)Xᵀ.
- **CODE:** the standard GPTQ running mean, H ← H·n/(n+t) + (2/n)·Σ (√c x)(√c x)ᵀ, where `nsamples` counts forward
  calls (one per sequence), not tokens or Σc. Per expert this is one global positive factor.
- **Production solver here:** it uses the raw sum Σ x xᵀ, and damping percdamp·mean(diag H) is relative. GPTQ is
  therefore exactly invariant to a global positive scale of H. A constant c gives identical weights (bitwise for
  power-of-two c; tested).
- **Consequence:** no normalization is added (**ASSUMPTION A2**, equivalent to the official choice up to a per-expert
  scalar).
- **Damping:** damping relative to mean(diag H_AGQ) is the solver's unchanged rule. The paper does not discuss it.

## 5. Sequential versus non-sequential scope (**ASSUMPTION A3**)

- **Official pipeline:** sequential, so each layer's inputs come from the already-quantized previous layers. The
  routing scores are recomputed after the gate is quantized.
- **This study:** keeps the historical non-sequential protocol. Every layer's calibration inputs and affinities come
  from the BF16 model, CANONICAL-B1 at batch 1.
- **Why this is acceptable:** it isolates AGQ's Hessian weighting from error propagation. It is the same protocol for
  both arms.

## 6. Other differences from the paper's setting

- **Calibration:** the paper uses EBSS data with Hadamard (QuaRot) preprocessing, symmetric per-channel quantization,
  2048-token segments and all layers. Here it is C4 Random 128 × 512, no rotation, asymmetric g128, act-order,
  percdamp 1.0, and 4 layers only.
- **Status:** these are deliberate scope choices of an AGQ-only test. They are not claims about MoEQuant.

## 7. Published evidence for AGQ alone (Table 5, GPTQ baseline → AGQ only, 7-task average)

| model | baseline avg | AGQ-only avg | Δ |
|---|---|---|---|
| Qwen-MoE-14B (Qwen1.5-MoE-A2.7B) | 49.00 | 49.02 | **+0.02** |
| DeepSeek-MoE-16B | 39.01 | 39.50 | +0.49 |
| Mixtral-8x7B | 53.42 | 54.24 | +0.82 |

- **WikiText PPL (Qwen):** 7.43 → 7.44 for AGQ alone, so no PPL gain.
- **Text–table mismatches:** the text ("EBSS 1.3 %, AGQ about 2 %") matches Table 5 only if the labels are swapped. The
  Mixtral sentence ("AGQ not better than GPTQ") contradicts the table.
- **Prior for this study:** for the exact model studied here, the authors' own AGQ-only effect is +0.02 points, i.e.
  essentially null.

## 8. Released code status

- AGQ is implemented and selectable with `--AGQ_GPTQ` → `gptq_utils_moe.gptq_fwrd`.
- It is **not used directly** here because it is tied to the QuaRot / sequential pipeline. Its exact Hessian rule
  (`inp*sqrt(s)`) and affinity definition are reproduced inside our unchanged solver path.
- **Code oddities, not relevant to this reproduction:**
  - `add_batch_score` counts `nsamples` per call, not per token.
  - The DeepSeek branch reads `out[1]`/`out[0]` of its gate.
  - `bit_mask.py` has a `score1_cnt` typo for slot 4.

## 9. Unresolved

1. Pre/post top-k and renormalization are not stated in the paper. They are resolved by code and model behaviour.
2. Whether the BF16-rounded or the fp32 coefficient is intended is not stated (A1). The effect is numerically
   negligible.
3. The normalization of the weighted Hessian is not stated (A2). It is irrelevant under relative damping.
4. Whether the shared-expert sigmoid weighting is part of "AGQ" as published is not stated. It is out of scope here.
