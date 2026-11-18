# Dense-model control: GPTQ calibration-grouping reproducibility (Qwen2.5-1.5B)

**Outcome: `E-INTERMEDIATE`** under the prespecified rule in `runs_dense_grouping_scope/protocol.json` (sha256
`03886a9d…`). The rule was frozen before any grouping result existed.

**Headline.** Dense GPTQ weight sensitivity is clearly smaller than in the MoE Family D experiment. Dense PPL
variability across the same five groupings is of the same order.

| ratio (dense / MoE FR seed 0) | value | meaning |
|---|---|---|
| R_W | **0.142** | BF16 weight elements changed, mean of 10 grouping pairs: 7.7 % vs 54.0 % |
| R_PPL | **0.836** | SD(log PPL) over the 5 groupings: 3.99e-4 vs 4.77e-4 (F(4,4) 95 % 0.27–2.59) |
| R_relFrob | 0.311 | relative Frobenius weight difference: 0.028 vs 0.089 |

**Mandatory checks.** All checks A–F pass exactly:
- BF16 dense PPL is 12.523451, and repeat evaluation is bitwise identical.
- Every quantized config re-evaluates bitwise.
- A fresh-process re-quantization reproduces the Hessians and weights bitwise.
- The calibration multisets and token ids are identical across groupings, and reversed batch execution order reproduces
  every cache bitwise.
- Integrity holds for all 12 tensors in every config (D).
- Every config starts from pristine targets (E) and is restored to them (F).
- At the end, BF16 re-evaluates bitwise and the 3964 non-target parameters are unchanged.

The 60 targeted tests pass.

**Main finding.** The dense control isolates the grouping trigger. In the dense model a grouping changes the
calibration activations of **only one sequence**: the row executed alone in the grouping's final 1-row batch
(27 × 8 + 1). Even that single-sequence perturbation makes GPTQ:
- re-round 7.7 % of BF16 weight elements;
- flip 1.4 % of integer codes (about 2.2 M codes);
- produce a PPL spread comparable, in log terms, to the MoE grouping spread.

GPTQ's sensitivity to tiny calibration perturbations is therefore **not MoE-specific**. What looks MoE-amplified
is the trigger: in the MoE the same groupings perturb every calibration sequence.

## 1. Audit of the historical Family D path (`logs/audit.json`)

| step | historical code path |
|---|---|
| selection | FR s0 = `e3_allocation` random, B = 128, verified by `full_depth_pilot.load_selections` |
| grouping | `multigroup_robustness.grouping_batches`: the sorted 217-row union (FR s0 ∪ FC s0) → `default_rng(seed).permutation` → consecutive batches of 8 (27 × 8 + 1) |
| what a grouping changes | **only** the pristine BF16 *batched* forward that builds the per-sequence layer-input cache (`batch_sensitivity.forward_plan`) |
| Hessians | `Profiler.run`: replay layer L on cached inputs, batch 1, in the frozen chosen order; `BlockHessianAccumulator.add` (fp32 XᵀX, K = 4 contiguous blocks), summed |
| solver | `gptq.quantize_expert` → `gptq_quantize`: gate/up share H_gu; lazy-batch 128; act-order = argsort(diag H); per-128-column asymmetric 4-bit grid |
| BF16 weights | fp32 W_q → `p.copy_(W_q.to(bf16))` |
| evaluation | `eval_partial_ppl.layerwise_nll`, `runs_partial_ppl/windows.pt` (128 × 512), batch 8 |

**Note on `percdamp`.** `percdamp = 1.0` is passed verbatim, so the damping is **1.0 × mean(diag H)**, not 1 %. This
value was selected on the tuning split (`configs/frozen_b4.json`).

**Groupings.** All five groupings regenerate exactly from the frozen `runs_multigroup_robustness/grouping_manifests`.

**Historical seed-0 FR baseline** (the matched single-strategy, single-seed MoE comparison):

| statistic | value |
|---|---|
| PPL, G20261111–15 | 9.278182, 9.268159, 9.271035, 9.278102, 9.272717 |
| SD (ddof = 1) | **0.004422** |
| SD(log PPL) | 4.768e-4 |
| CV | 4.768e-4 |
| range | 0.010023 |

The 0.0052 figure quoted in the brief is the 3-seed *pooled* within-seed SD, so it is not used as a single-seed
estimate.

**Tokenization.** Qwen1.5-MoE and Qwen2.5 share all 151,646 base-vocabulary entries and identical `merges.txt`;
Qwen2.5 adds only special tokens. All calibration and evaluation ids are below 151,643, and re-tokenizing WikiText-2
with the dense tokenizer reproduces the historical windows bitwise. The **identical token-id tensors** are therefore
reused for both calibration and evaluation.

## 2. Setup

- **Model:** `Qwen/Qwen2.5-1.5B` (base) at revision `8faed761d45a`, checkpoint sha256 `a961db72…`.
  - 28 layers, hidden 1536, intermediate 8960, tied embeddings.
  - BF16, eager attention, the same layer-wise offload as the MoE.
- **Layer mapping:** round(L · 27 / 23) maps MoE layers 1/8/16/23 to dense layers **1/9/19/27**.
- **Quantized tensors:** `mlp.gate_proj` [8960, 1536], `up_proj` [8960, 1536] and `down_proj` [1536, 8960] in those
  four layers.
  - 12 tensors, 165,150,720 parameters, which is 10.7 % of 1.544 B. (MoE: 720 tensors, 2.08 B, 14.5 %.)
  - Everything else stays BF16.
- **GPTQ:** the unchanged `quantize_expert`, 4-bit, g128, percdamp 1.0, act-order, BF16 fake-quantized weights.
- **Adapter:** `moe_calib/dense_gptq.py`.
  - Hessians come from the existing `BlockHessianAccumulator`, with one pseudo-expert per layer fed by pre-hooks on
    `gate_proj` and `down_proj`.
  - Accumulation uses Profiler semantics: batch 1, frozen order, K = 4, summed.
  - The installation cast is identical to the MoE path.
  - The solver's read-only `info` capture yields the integer codes. It is bitwise identical to the default path, and the
    codes reproduce W_q exactly.
  - The MoE code path is unchanged.
- **Calibration:** FR s0, 128 × 512 = 65,536 tokens per projection, in its frozen order (`calibration_manifest.json`).
- **Configurations:**
  - The 5 groupings G20261111–15 of the same 217-row union (`grouping_manifest.json`).
  - **ORIGINAL** (cache_inputs.py consecutive batches) as a descriptive reference, outside every 5-grouping statistic.
  - The prespecified reference grouping is G20261111.
- **Evaluation:** the historical 128 × 512 WikiText-2 windows, batch 8, each evaluated twice. Calibration (C4) and
  evaluation (WikiText-2) come from disjoint sources.

## 3. The trigger: what a grouping actually changes in a dense model

| layer-input cache at layer | 1 | 9 | 19 | 27 |
|---|---|---|---|---|
| rows bitwise equal to G20261111 (217 rows) | 99.1 % | 99.1 % | 99.1 % | 99.1 % |
| rows that differ | exactly 2 (both singletons) | 2 | 2 | 2 |
| rel. L2 of the singleton row vs ORIGINAL | 2.6e-3 – 4.1e-3 | 3.1e-3 – 2.4e-2 | 3.3e-3 – 2.4e-2 | 1.1e-2 – 2.2e-2 |
| **MoE**: activation elements bitwise equal under grouping | 99.9 % (layer 1) | 13.0 % (layer 8) | 11.1 % (layer 16) | 10.5 % (layer 23) |

**Dense model.** Rows that share a batch of 8 get bitwise the same activations whatever their batch-mates are. Only
the leftover row, forwarded as a batch of 1, is computed differently, presumably through a different GEMM path for a
different batch size.

| grouping | singleton row | in FR's 128 rows? |
|---|---|---|
| G20261111 | 491 | yes (position 122) |
| G20261112 | 1024 | yes (position 108) |
| G20261113 | 1667 | yes (position 123) |
| G20261114 | 1064 | yes (position 20) |
| G20261115 | 694 | **no** |

G20261115's singleton is not a calibration row, so **its Hessians, weights, integer codes and PPL are bitwise identical
to ORIGINAL**. This is a built-in null control.

**MoE.** The same groupings perturb almost every activation from layer 8 onward. Expert dispatch, where per-expert token
counts and hence GEMM shapes depend on batch composition, is the leading explanation. It was **not tested here**.

## 4. Grouping PPL (`grouping_ppl_results.json`, `ppl_variability.json`, Fig. 1)

| config | PPL | − PPL(G20261111) | − BF16 |
|---|---|---|---|
| G20261111 | 12.618646 | 0 | +0.095196 |
| G20261112 | 12.611615 | −0.007031 | +0.088164 |
| G20261113 | 12.622945 | +0.004299 | +0.099494 |
| G20261114 | 12.616844 | −0.001803 | +0.093393 |
| G20261115 | 12.624145 | +0.005499 | +0.100695 |
| ORIGINAL (ref.) | 12.624145 | +0.005499 | +0.100695 (bitwise = G20261115) |

| statistic (5 groupings) | dense | MoE FR s0 |
|---|---|---|
| BF16 PPL | 12.523451 | 9.237090 |
| mean PPL | 12.618839 | 9.273639 |
| SD(PPL) | **0.005030** (95 % 0.0030–0.0145) | 0.004422 |
| SD(log PPL) | **3.99e-4** | 4.77e-4 |
| CV(PPL) | 3.99e-4 | 4.77e-4 |
| min / max | 12.611615 / 12.624145 | 9.268159 / 9.278182 |
| max pairwise difference | 0.012530 | 0.010023 |
| SD / quantization penalty (mean − BF16) | 0.053 | 0.121 |
| distinct installed weight sets | 5 | 5 |

- **Per-window differences:** the per-window NLL differs in **every** window for every pair of distinct weight sets. The
  mean per-window SD of paired mean-NLL differences is 2.8e-3.
- **Normalized spread:** after normalizing for baseline PPL (log or CV), the dense spread is 0.84× the MoE seed-0 FR
  spread. Relative to the quantization penalty it is about half.

## 5. Weights and integer codes (`weight_difference_analysis.json`, Fig. 3, Fig. 5)

Weight differences use `grouping_routing_amplification.weight_diff`, the identical Family D definition.

| | dense, 10 pairs | dense vs ORIGINAL | MoE FR s0, 10 pairs | MoE FR s0 vs ORIGINAL |
|---|---|---|---|---|
| BF16 elements changed | **7.66 %** (6.2–9.3 %) | 5.54 % | **53.99 %** | 54.06 % |
| rel. Frobenius | 0.0276 | 0.0209 | 0.0889 | 0.0889 |
| mean abs diff | 8.0e-5 | 5.7e-5 | 6.3e-4 | 6.3e-4 |
| ULP share 1 / 2 / 3–7 | 76.8 / 6.3 / 3.4 % | 62.4 / 4.9 / 2.5 % | 35.4 / 11.5 / 14.6 % | same |
| ULP share ≥ 8 | **12.5 %** | 9.4 % | **37.3 %** | 37.3 % |
| sign change | 1.1 % | 0.8 % | 1.3 % | 1.3 % |
| **integer codes changed** | **1.36 %** | 0.94 % | not measured | not measured |
| changed BF16 elements with a code change | 13.9 % | 10.6 % | | |
| code changes with identical BF16 value | 22.1 % | | | |
| elements in columns that changed quantization group | 0.19 % | 0.12 % | | |

**BF16 change and integer-code change are different quantities.**
- 86 % of changed BF16 elements keep the same integer code: the grid values were rescaled (mostly 1-ULP moves), not the
  rounding decision.
- 22 % of code flips leave the BF16 value unchanged.
- A large ULP change is therefore not by itself evidence of a code flip.

**Where it happens (10-pair mean, BF16 changed / codes changed):**

| layer | gate_proj | up_proj | down_proj |
|---|---|---|---|
| L1 | 2.4 % / 0.51 % | 2.4 % / 0.56 % | 1.5 % / 1.00 % |
| L9 | 8.0 % / 1.33 % | 7.8 % / 1.33 % | **32.3 % / 5.23 %** |
| L19 | 1.4 % / 0.28 % | 1.4 % / 0.27 % | **25.4 % / 3.85 %** |
| L27 | 1.1 % / 0.15 % | 1.1 % / 0.13 % | 7.0 % / 1.73 % |

`down_proj` dominates (16.6 % vs 3.2 % for gate and up), concentrated in the middle layers. In the MoE, `down_proj` was
the *least* changed projection (44 % vs 59 %).

## 6. Hessians, act-order and conditioning (`hessian_sensitivity.json`, Fig. 4)

The metric is `frozen_routing.hessian_metrics`. `rel_fro_trace_normalized` is the historical Family D metric.

| | gu | down | all |
|---|---|---|---|
| 10 pairs: trace-normalized rel. Frobenius, median | 5.9e-5 | 2.2e-4 | **7.0e-5** |
| 10 pairs: act-order positions changed, mean | 8.3 % | 33.3 % | 20.8 % |
| 10 pairs: columns changing quantization group, mean | 0.09 % | 0.40 % | 0.24 % |
| vs ORIGINAL: trace-normalized, median | 3.6e-5 | 5.3e-5 | 4.7e-5 |

**Matched MoE values** (FR, G20261111 vs ORIGINAL, 16 diagnostic expert Hessians):

| MoE quantity | trace-normalized median | act-order positions changed | columns changing group |
|---|---|---|---|
| activation-only (AO, routing frozen) | 0.0039 (gu 0.0030, down 0.0124) | 69 % | 2.4 % |
| natural (AA) | 0.0141 (gu 0.0092, down 0.0246) | 85 % | 7.3 % |

The dense model has no routing, so AO is the closer comparison. **The dense Hessian perturbation is about 55× smaller
than the MoE AO value and about 200× smaller than AA.**

**Per layer.** The weight changes track the down-Hessian change, as the next table shows.

| layer | down Hessian change (median) | down act-order positions changed | down_proj BF16 changed |
|---|---|---|---|
| L9 | 7.9e-4 | 42 % | 32 % |
| L19 | 4.7e-4 | 32 % | 25 % |
| L1 | 5.1e-5 | 25 % | 1.5 % |
| L27 | 5.7e-5 | 42 % | 7 % |

Across (pair, layer, projection), the weight change rises steeply with the Hessian change (Fig. 4).

**Conditioning.**
- The undamped gu Hessians are numerically rank-deficient (cond about 1e11–1e13; effective-rank fraction ≤ 0.013).
- After damping (1.0 × mean diag), cond is 115–8,937.
- The best-conditioned matrix (L9 down, 115) is also the most perturbed, so conditioning does not explain the layer
  pattern.

**Tokens per input dimension:** dense gu 42.7 and down 7.3; MoE expert median gu 2.1 and down 3.1 (median 4,312 tokens
per expert).

These are associations. **No GPTQ error-feedback cascade is claimed**; that needs the separate integer-code
experiment.

## 7. Dense vs MoE (`dense_moe_comparison.json`, Fig. 2)

| metric | dense control | MoE Family D (FR s0) |
|---|---|---|
| model | Qwen2.5-1.5B (dense, 28 layers) | Qwen1.5-MoE-A2.7B (24 layers, 60 experts, top-4) |
| quantized | 4 MLP layers, 12 matrices, 165 M params (10.7 %) | 4 routed-expert layers, 720 matrices, 2.08 B params (14.5 %) |
| matrix inputs gu / down | 1536 / 8960 | 2048 / 1408 |
| tokens per matrix | 65,536 | median 4,312 (p10 3,871, p90 4,913) |
| calibration rows perturbed by a grouping | 1 (0 for G20261115) | essentially all |
| Hessian change (trace-normalized) | 7.0e-5 | AO 0.0039 / AA 0.0141 |
| BF16 PPL / mean quantized PPL | 12.5235 / 12.6188 | 9.2371 / 9.2736 |
| within-seed SD(PPL) | 0.00503 | 0.00442 |
| SD(log PPL) = CV | 3.99e-4 | 4.77e-4 |
| SD / quantization penalty | 0.053 | 0.121 |
| BF16 elements changed (10 pairs) | 7.7 % | 54.0 % |
| rel. Frobenius (10 pairs) | 0.028 | 0.089 |
| ≥ 8-ULP share of changed elements | 12.5 % | 37.3 % |
| act-order positions changed vs ORIGINAL | 13 % | AO 69 % / AA 85 % |

The figures above are all within-seed. The MoE pooled 3-seed SDs are not used as single-seed estimates.

## 8. Answers

1. **Does calibration batch grouping cause numerical instability in dense GPTQ?** Yes, through a narrower trigger.
   - Batch composition alone does not change dense activations.
   - The leftover row's batch-size-1 execution does. When it is a calibration row (4 of 5 groupings), GPTQ output and PPL
     change; when it is not (G20261115), nothing changes, bitwise.
2. **How much do the quantized weights change?** 7.7 % of BF16 elements (rel. Frobenius 0.028), of which 12.5 % move
   ≥ 8 ULP. 1.4 % of integer codes change. The change is concentrated in mid-layer `down_proj`.
3. **How large is the PPL variability?** SD(PPL) is 0.0050, SD(log PPL) and CV are 4.0e-4, and the maximum pairwise
   difference is 0.0125. That is about 5 % of the quantization penalty.
4. **Is it comparable to MoE Family D?**
   - For PPL, yes after normalization: 0.84× in log or CV terms, with a 95 % CI that comfortably includes 1.
   - For weights, no: 7× fewer elements change, and rel. Frobenius is 0.31×.
   - Model size, matrix geometry, tokens per matrix and quantized fraction are not matched.
5. **Is the variability associated with Hessian perturbations, act-order changes or other solver properties?**
   - It is associated with the down-Hessian perturbation and with large act-order reordering (33 % of `down_proj`
     positions). Very few columns change quantization group (0.24 %).
   - Conditioning does not explain the layer pattern.
   - Causality is not established.
6. **What evidence supports general GPTQ sensitivity?** A single-sequence activation change produces about 2.2 M
   integer-code changes and a model-level PPL spread comparable to the MoE's. The perturbed row's relative L2 change is
   3e-3 to 2.4e-2; every other row is bitwise unchanged. There is no routing anywhere, and the controls are exact: a
   bitwise repeat, a bitwise fresh-process re-quantization and a bitwise G20261115 = ORIGINAL null.
7. **What remains unresolved about MoE-specific contributions?**
   - **The trigger.** In the MoE the grouping reaches essentially every calibration sequence, against one row here.
     Expert dispatch is the hypothesis; it was not tested.
   - **Perturbation size vs solver sensitivity.** The MoE input perturbation is 55–200× larger in Hessian terms, so this
     study cannot separate "larger perturbation" from "more sensitive per-expert solver problem" (about 2 tokens per
     input dimension).
   - **Similar PPL from fewer weight changes.** Why a 7× smaller weight change gives a similar PPL spread is open.
8. **Most useful next controlled experiment: a dose-matched perturbation study.**
   - Perturb **every** dense calibration sequence, for example batch size 1 vs 8 for all rows, or inject Hessian
     perturbations at the MoE AO/AA magnitudes into the unchanged solver.
   - Compare weight, integer-code and PPL response per unit of trace-normalized Hessian change across dense and MoE.
   - Then run the integer-code cascade experiment on the saved dense codes.

## Limitations

- **Small design.** One calibration seed and 5 groupings; in the dense model these are 4 distinct single-row
  perturbations plus one null. The dense "grouping SD" therefore samples a much smaller perturbation than the MoE's.
  R_PPL and R_W are screening ratios (F(4,4) CI factor about 3.1×), not tests. The optional Phase 10 replication over two
  more seeds was **not run**.
- **Narrow setting.**
  - One dense model, 1.5 B parameters (vs 14.3 B total for the MoE), with layers mapped by relative depth.
  - BF16 fake quantization, not packed INT4.
  - Tied embeddings in the dense model vs an untied MoE head.
  - The singleton-row mechanism depends on GEMM kernel selection by batch size in this stack: A100 MIG 2g.20gb,
    torch 2.11 cu128, transformers 4.51.3, eager attention.
- **Reference mismatch.** The MoE Hessian reference is 16 diagnostic expert Hessians, G20261111 vs ORIGINAL. The dense
  analysis uses all 8 layer Hessians.
- **Cholesky comparison.** The Cholesky-factor differences in `hessian_sensitivity.json` compare factors under each
  config's own act-order, so they mix value and ordering changes.
- **Fig. 4 floor.** Exact-zero Hessian differences are drawn at a 1e-12 floor.
- **Harmless warning.** The "sliding window not implemented for eager" warning is inert: `use_sliding_window=False`,
  and the window is 131,072 tokens, longer than the 512-token sequences.

### Protocol amendments

Neither amendment affects any reported result.

1. **Driver crash before quantization.** The GPU stage crashed in the activation-drift diagnostic (`torch.stack` over
   bools) after the BF16 baseline and caches were built, but before any quantization.
   - After a one-line fix the protocol was re-frozen with identical content except the driver's code hash.
   - `8bc12d8c…` became `03886a9d…`; see `logs/protocol_amendment.json` and `logs/protocol_v1_superseded.json`.
   - The caches and BF16 baseline were reused after sha verification.
2. **Weight-stage failure.** The driver's `--stage weights` failed because the reused Family D `weight_diff` hard-codes
   the MoE layer labels (KeyError: 9).
   - Neither the frozen driver nor the historical function was modified.
   - `diagnostics/dense_grouping_weights.py` calls the identical function with dense layers relabelled onto its slots
     and the labels mapped back.

## Cost (`cost_results.json`)

| item | value |
|---|---|
| model load | 6 s |
| cache build per grouping, with reversed-order control | 30–42 s (ORIGINAL 82 s) |
| Hessians per config | 11 s |
| GPTQ per config | 11–14 s |
| PPL eval | 7–8 s (×2 per config) |
| total per config | about 65 s |
| peak GPU | 3.7 GB |
| peak CPU RSS | 9.6 GB |
| storage | 19.2 GB in `~/moe_cache_dense_grouping_scope` (caches, BF16 weights, integer codes, scales/zeros, Hessians) |

## Files

- **Code:**
  - `moe_calib/dense_gptq.py` (adapter)
  - `diagnostics/dense_grouping_scope.py` (stages audit / protocol / gpu / repeat / hessian / weights / analyze)
  - `diagnostics/dense_grouping_weights.py` (weight stage, supersedes `--stage weights`)
  - `diagnostics/report_dense_grouping_scope.py` (analysis, decision, figures, summary)
  - `tests/test_dense_grouping_scope.py` (4 tests)
- **Outputs:** `runs_dense_grouping_scope/`
  - `protocol.json`, `artifact_inventory.json`, `calibration_manifest.json`, `grouping_manifest.json`
  - `correctness_results.json`, `bf16_baseline.json`, `grouping_ppl_results.json`, `ppl_variability.json`
  - `weight_difference_analysis.json`, `hessian_sensitivity.json`, `dense_moe_comparison.json`, `cost_results.json`
  - `summary.md`
  - `figures/fig1`–`fig5`
  - `logs/` (audit, raw Hessian, weight and code differences, repeat control, interpretation, pytest)
- **Preserved for mechanism studies:** `~/moe_cache_dense_grouping_scope/{weights,codes,hessians,cache}/<config>/`.
  All artifacts re-hash to the inventory.
