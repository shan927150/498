# Dense-model control: GPTQ calibration-grouping reproducibility (Qwen2.5-1.5B)

**Outcome: E-INTERMEDIATE** - R_PPL = 0.836 (SD(log PPL) dense / MoE FR s0), R_W = 0.142 (mean pairwise BF16 elements changed, dense / MoE FR s0); protocol sha256 03886a9d67711004.

**Interpretation.** Under the prespecified rule the outcome is E-INTERMEDIATE. Dense weight sensitivity is clearly smaller than in the MoE (R_W = 0.14), but dense PPL variability across groupings is of the same order (R_PPL = 0.84; F(4,4) 95 % 0.27–2.59). The dense run also isolates the trigger.

In the dense model the grouping changes the calibration activations of **exactly one sequence**: the row executed alone in the 27 x 8 + 1 grouping's final 1-row batch. Every other row of every layer-input cache is bitwise identical across groupings (99.1 % of rows vs G20261111 = all but the two singleton rows; reversed execution order reproduces every cache bitwise). For G20261115 that row (694) is not an FR calibration row, so its Hessians, weights and PPL are **bitwise identical to ORIGINAL**. In the MoE, by contrast, only about 11–13 % of activation elements stay bitwise equal from layer 8 onward under the same groupings (runs_multigroup_robustness/activation_diagnostics.json): there every calibration sequence is perturbed.

Even so, a one-sequence perturbation is enough for dense GPTQ. That one sequence out of 128 shifts the dense Hessians by a median trace-normalized 7e-5, about 55x below the MoE activation-only change (AO 0.0039) and about 200x below its natural change (AA 0.0141). It still re-rounds 7.7 % of BF16 weight elements and 1.4 % of integer codes (about 2.2 M codes), and gives a PPL SD of 0.0050 against the MoE's 0.0044.

## Setup

- Model `Qwen/Qwen2.5-1.5B` @ `8faed761d45a` (checkpoint sha256 `a961db72e75d52b1…`), depth 28, hidden 1536, intermediate 8960; BF16, eager attention, layer-wise offload.
- Quantized: mlp.gate/up/down_proj of layers [1, 9, 19, 27] (MoE 1/8/16/23 by relative depth); 12 tensors, 165,150,720 parameters = 10.7 % of the model.
- GPTQ: existing `moe_calib.gptq.quantize_expert`, 4-bit, g128, percdamp 1.0 (x mean diag H), act-order, BF16 effective weights.
- Calibration: FR s0 (128 x 512 C4 rows, identical token ids to Family D), K = 4 blocks; groupings G20261111-15 of the same 217-row union exactly as Family D (batched pristine BF16 forward -> layer-input cache; Hessians one sequence at a time).
- Evaluation: the historical 128 x 512 WikiText-2 windows (dense re-tokenization reproduces them bitwise), batch 8. BF16 PPL 12.523451.

## Mandatory checks

| check | result |
|---|---|
| A_bf16_checkpoint_equal | True |
| A_bf16_evaluated | True |
| B_bf16_repeat_bitwise | True |
| B_all_config_repeat_bitwise | True |
| B_fresh_process_requantization_bitwise | True |
| C_union_multiset_identical | True |
| C_tokens_identical | True |
| C_reversed_execution_order_bitwise | True |
| C_tokens_per_projection_65536 | True |
| C_same_layers | True |
| D_shapes | True |
| D_finite | True |
| D_all_targets_quantized | True |
| D_installed_equals_solver | True |
| D_non_targets_unchanged | True |
| D_integer_codes_exact | True |
| D_info_capture_bitwise_default | True |
| E_pristine_start | True |
| F_restoration | True |
| F_end_bf16_bitwise | True |
| F_end_pristine | True |
| F_end_untouched | True |
| all_configs_valid | True |
| targeted_tests | True |

Blocked: False.

## Grouping PPL

| config | PPL | − PPL(G20261111) | − BF16 | installed sha256 |
|---|---|---|---|---|
| G20261111 | 12.618646 | +0.000000 | +0.095196 | `3a4a82f05658` |
| G20261112 | 12.611615 | -0.007031 | +0.088164 | `3c0131b58129` |
| G20261113 | 12.622945 | +0.004299 | +0.099494 | `046a794ef0e1` |
| G20261114 | 12.616844 | -0.001803 | +0.093393 | `ac238612ef35` |
| G20261115 | 12.624145 | +0.005499 | +0.100695 | `380b9fcab1a7` |
| ORIGINAL | 12.624145 | +0.005499 | +0.100695 | `380b9fcab1a7` |

| statistic (5 groupings) | dense | MoE FR s0 |
|---|---|---|
| mean PPL | 12.618839 | 9.273639 |
| SD(PPL) | 0.005030 (95 % 0.003014–0.014455) | 0.004422 |
| SD(log PPL) | 3.987e-04 | 4.768e-04 |
| CV(PPL) | 3.986e-04 | 4.768e-04 |
| min / max | 12.611615 / 12.624145 | 9.268159 / 9.278182 |
| max pairwise difference | 0.012530 | 0.010023 |
| SD / (mean − BF16) | 0.0527 | 0.1210 |
| distinct installed weight sets | 5 | 5 |

R_PPL = 0.836 (F(4,4) 95 % 0.270–2.591); R_CV = 0.836. MoE pooled 3-seed within-seed SDs (FR 0.0052) are not used as single-seed estimates. Mean per-window paired NLL SD over the 10 pairs: 2.83e-03.

## Weights and integer codes

| | dense 10 pairs | dense vs ORIGINAL | MoE FR s0 10 pairs | MoE FR s0 vs ORIGINAL |
|---|---|---|---|---|
| BF16 elements changed | 0.0766 | 0.0554 | 0.5399 | 0.5406 |
| rel. Frobenius | 2.76e-02 | 2.09e-02 | 8.89e-02 | 8.89e-02 |
| mean abs diff | 8.02e-05 | 5.67e-05 | 6.34e-04 | 6.34e-04 |
| ULP share 1 | 0.768 | 0.624 | 0.354 | 0.354 |
| ULP share 2 | 0.063 | 0.049 | 0.115 | 0.115 |
| ULP share 3-7 | 0.034 | 0.025 | 0.146 | 0.146 |
| ULP share >=8 | 0.125 | 0.094 | 0.373 | 0.373 |
| ULP share sign_change | 0.011 | 0.008 | 0.013 | 0.013 |
| integer codes changed | 0.0136 | 0.0094 | n/a | n/a |
| changed BF16 elements with a code change | 0.139 | 0.106 | n/a | n/a |
| elements in columns whose group changed | 0.0019 | 0.0012 | n/a | n/a |

Per projection (10 pairs, BF16 changed): gate_proj 0.0323, up_proj 0.0317, down_proj 0.1657; per layer: L1 0.0210, L9 0.1604, L19 0.0941, L27 0.0308.

## Hessians and act-order

| | gu | down | all |
|---|---|---|---|
| 10 pairs: rel. Frob. (trace-norm.), median | 5.93e-05 | 2.20e-04 | 6.99e-05 |
| 10 pairs: act-order positions changed, mean | 0.0825 | 0.3326 | 0.2075 |
| 10 pairs: columns changing group, mean | 0.0009 | 0.0040 | 0.0024 |
| vs G20261111: rel. Frob. (trace-norm.), median | 7.81e-05 | 2.93e-04 | 7.87e-05 |
| vs G20261111: act-order positions changed, mean | 0.0994 | 0.3296 | 0.2145 |
| vs G20261111: columns changing group, mean | 0.0012 | 0.0039 | 0.0026 |
| vs ORIGINAL: rel. Frob. (trace-norm.), median | 3.60e-05 | 5.25e-05 | 4.66e-05 |
| vs ORIGINAL: act-order positions changed, mean | 0.0529 | 0.2127 | 0.1328 |
| vs ORIGINAL: columns changing group, mean | 0.0007 | 0.0024 | 0.0015 |

MoE FR s0 (G20261111 vs ORIGINAL, 16 diagnostic Hessians): trace-normalized median AA 0.0141 (gu 0.0092, down 0.0246); activation-only AO 0.0039 (gu 0.0030, down 0.0124); act-order positions changed AA 0.854, columns changing group AA 0.0731.

| layer.proj | cond(damped H), ref | cond(H), ref | eff. rank frac | diag CV |
|---|---|---|---|---|
| L1.gu | 1123.89 | 5.65e+13 | 0.001 | 14.60 |
| L1.down | 8937.49 | 5.65e+09 | 0.000 | 73.48 |
| L9.gu | 329.58 | 8.60e+13 | 0.013 | 4.36 |
| L9.down | 115.14 | 3.06e+03 | 0.182 | 0.59 |
| L19.gu | 418.09 | 4.16e+11 | 0.008 | 5.80 |
| L19.down | 203.40 | 1.40e+04 | 0.081 | 1.27 |
| L27.gu | 716.78 | 2.55e+13 | 0.003 | 11.91 |
| L27.down | 5947.20 | 2.15e+07 | 0.000 | 18.36 |

**Calibration activations (layer-input cache) vs G20261111:** L1: rows bitwise 12 0.991, 13 0.991, 14 0.991, 15 0.991; L9: rows bitwise 12 0.991, 13 0.991, 14 0.991, 15 0.991; L19: rows bitwise 12 0.991, 13 0.991, 14 0.991, 15 0.991; L27: rows bitwise 12 0.991, 13 0.991, 14 0.991, 15 0.991.

**vs ORIGINAL:** L1: rel L2 2.1e-04, 2.2e-04, 2.5e-04, 2.9e-04, 1.6e-04; L9: rel L2 5.4e-04, 1.6e-03, 2.2e-04, 2.2e-04, 3.7e-04; L19: rel L2 5.5e-04, 1.6e-03, 2.4e-04, 2.4e-04, 4.0e-04; L27: rel L2 1.1e-03, 1.2e-03, 7.8e-04, 9.5e-04, 1.5e-03.

## Dense vs MoE

| metric | dense control | MoE Family D (FR s0) |
|---|---|---|
| model | Qwen2.5-1.5B (dense, 28 layers) | Qwen1.5-MoE-A2.7B (24 layers, 60 routed experts, top-4) |
| quantized | 4 MLP layers [1, 9, 19, 27], 12 matrices | 4 routed-expert layers [1, 8, 16, 23], 720 matrices |
| quantized_params | 165150720 | 2076180480 |
| frac_of_model_params | 0.107 | 0.145 |
| matrix_dims_gu_down_in | [1536, 8960] | [2048, 1408] |
| calibration | FR s0 128 x 512, identical token ids | FR s0 128 x 512 |
| tokens_per_matrix | 65536 | median 4312 (p10 3871, p90 4913) |
| tokens_per_input_dim_gu | 42.67 | 2.105 |
| tokens_per_input_dim_down | 7.314 | 3.062 |
| groupings | 5 (seed 0) | 5 (seed 0) |
| bf16_ppl | 12.52 | 9.237 |
| mean_quantized_ppl | 12.62 | 9.274 |
| within_seed_sd_ppl | 0.00503 | 0.004422 |
| sd_log_ppl | 0.0003987 | 0.0004768 |
| cv_ppl | 0.0003986 | 0.0004768 |
| sd_over_quant_penalty | 0.05274 | 0.121 |
| max_pairwise_ppl_diff | 0.01253 | 0.01002 |
| bf16_elements_changed_10pairs | 0.07656 | 0.5399 |
| rel_frobenius_10pairs | 0.0276 | 0.08885 |
| ge8ulp_share_of_changed | 0.1251 | 0.3733 |
| bf16_elements_changed_vs_original | 0.05545 | 0.5406 |
| hessian_rel_fro_tn_G1111_vs_ORIGINAL_median | 6.994e-05 | AO 0.0039 / AA 0.0141 |
| hessian_rel_fro_tn_10pairs_median | 6.994e-05 | not measured |
| actorder_positions_changed_vs_original | 0.1328 | AO 0.694 / AA 0.854 |
| columns_group_changed_vs_original | 0.001506 | AO 0.0238 / AA 0.0731 |

R_W = 0.142; R_relFrob = 0.311; R_PPL = 0.836. screening ratios, not tests; 5 groupings, 1 seed; F(4,4) 95% factor on an SD ratio is ~3.1x; architecture, size, matrix geometry, calibration statistics and baseline PPL are not matched.

## Answers

1. **Does calibration batch grouping cause numerical instability in dense GPTQ?** Yes, but through a narrower trigger than in the MoE. Batch composition alone does not change dense activations here. The only change is the batch-size-1 execution of the grouping's leftover row; batched dense layers are row-invariant at a fixed batch size in this stack. When that row is a calibration row (4 of 5 groupings), GPTQ output and PPL change. When it is not (G20261115), nothing changes, bitwise.
2. **How much do the quantized weights change?** Across the 10 grouping pairs, 7.7 % of BF16 elements change (range 6.2–9.3 %), with rel. Frobenius 0.028; against ORIGINAL it is 5.5 %. 77 % of changed elements move 1 ULP and 12.5 % move 8 or more ULPs (MoE: 35 % / 37 %). Only 1.4 % of integer codes change. 86 % of BF16 changes keep the same code, so they are rescaled grid values, not code flips, and 22 % of code flips occur without a BF16 change. Large BF16 ULP change and code flips are clearly distinct quantities. Sensitivity sits mainly in down_proj (16.6 % vs 3.2 % for gate/up) and in the middle layers (L9 16 %, L19 9 %; L1 2 %, L27 3 %).
3. **How large is the PPL variability?** SD(PPL) = 0.0050 (95 % 0.0030–0.0145), SD(log PPL) = 3.99e-4, CV = 3.99e-4, max pairwise difference 0.0125 on a mean of 12.6188 (BF16 12.5235). This is about 5 % of the quantization penalty (MoE: 12 %). Per-window paired NLL differences are nonzero in every window for every pair that differs.
4. **Is it comparable to MoE Family D?** For model-level PPL, yes, after normalizing for baseline: SD(log PPL) and CV are 0.84x the MoE seed-0 FR values, with a CI that comfortably includes 1. For weights, no: 7x fewer elements change (R_W 0.14) and rel. Frobenius is 0.31x. Relative to the quantization penalty, the dense spread is about half the MoE's. Size, depth, matrix geometry (dense 1536/8960 vs expert 2048/1408 inputs), tokens per matrix (65,536 vs a median of about 4,300; 43 vs 2.1 tokens per gu input dimension) and quantized fraction (10.7 % vs 14.5 %) are all unmatched.
5. **What is it associated with?** The perturbation is tiny in Hessian terms (median trace-normalized 7e-5; gu 6e-5, down 2.2e-4) but reorders the act-order permutation substantially:
   - 21 % of positions change between groupings (gu 8 %, down 33 %), yet only 0.24 % of columns change quantization group.
   - Across (pair, layer, projection), weight change rises steeply with Hessian change (Fig. 4). down_proj, with fewer tokens per input dimension (7.3) and the largest act-order disruption, shows the largest changes.
   - The damped Hessians are moderately conditioned (cond 1e2–9e3), and the undamped gu Hessians are numerically rank-deficient (cond about 1e11–1e13).
   These are associations. No error-feedback cascade is claimed; that needs the separate integer-code experiment.
6. **What supports general GPTQ sensitivity?** A one-sequence activation change (that row's relative L2 change is 3e-3 to 2.4e-2 across layers; every other row is bitwise identical), with no routing anywhere, produces 2.2 M integer-code changes and a model-level PPL spread as large (in log terms) as the MoE grouping spread. This holds under exact controls: deterministic re-evaluation, a bitwise fresh-process re-quantization, a bitwise G20261115 = ORIGINAL identity, and pristine restoration. GPTQ's sensitivity to tiny calibration perturbations is therefore not MoE-specific.
7. **What remains unresolved about MoE-specific contributions?**
   - The **trigger** looks MoE-amplified. The same groupings perturb every MoE calibration sequence (about 87 % of activation elements change from layer 8) but only one dense sequence. Expert dispatch, where per-expert token counts and hence GEMM shapes depend on batch composition, is the leading hypothesis; it was not tested here.
   - The MoE weight-level response is 7x larger, but its input perturbation is also about 55–200x larger in Hessian terms. This study cannot separate 'larger perturbation' from 'more sensitive solver problem', such as sparse per-expert calibration with about 2 tokens per input dimension.
   - Why a 7x smaller weight change gives a similar PPL spread (per-element impact, down_proj concentration, model sensitivity) is also not resolved.
8. **Most useful next controlled experiment:** a dose-matched perturbation study.
   - In the dense model, perturb **every** calibration sequence's numerics, for example forwarding each sequence at batch size 1 vs 8, or varying batch size per grouping, so the activation perturbation reaches all 128 rows.
   - Alternatively, inject Hessian perturbations scaled to the MoE AO and AA magnitudes into the unchanged solver.
   - Compare weight, code and PPL response per unit of trace-normalized Hessian change across dense and MoE.
   - Run the integer-code cascade experiment on the saved dense codes.

## Limitations

- **One seed, 5 groupings.** In the dense model these are effectively 4 distinct one-sequence perturbations plus one null (G20261115 = ORIGINAL), so the dense 'grouping SD' is a sample of single-row perturbations, not of the MoE's all-row perturbations. R_PPL and R_W are screening ratios (F(4,4) CI factor about 3.1x), not tests. Phase 10 replication was not run.
- **Narrow setting.** One dense model at 1.5B parameters (vs 14.3B total MoE). The layer mapping is by relative depth. Quantization is BF16 fake quantization, not packed INT4. Results hold for this exact numerical stack (A100 MIG 2g.20gb, torch 2.11 cu128, transformers 4.51.3, eager attention); the singleton-row mechanism depends on GEMM kernel choice by batch size.
- **Reference differences.** MoE Hessian comparisons are G20261111 vs ORIGINAL on 16 diagnostic expert Hessians; dense uses all 8 Hessians. Cholesky-factor differences compare factors under each configuration's own act-order permutation, so they mix value and ordering changes. In Fig. 4, exactly-zero Hessian differences (G20261115 vs ORIGINAL) are plotted at a 1e-12 floor.
- **Protocol amendments, both before any affected result.** (1) The GPU stage crashed in the activation-drift diagnostic before any quantization; after a one-line fix the protocol was re-frozen with identical content (`logs/protocol_amendment.json`). (2) The driver's `--stage weights` failed because the reused Family D `weight_diff` hard-codes the MoE layer labels. It was replaced by `diagnostics/dense_grouping_weights.py`, which calls the same function with relabelled layers.
- The tied-embedding dense model and the MoE's untied lm_head are not matched. The 'sliding window not implemented for eager' warning is inert (use_sliding_window=False, window 131,072 > 512).

## Files

- Code: `moe_calib/dense_gptq.py`, `diagnostics/dense_grouping_scope.py`, `diagnostics/report_dense_grouping_scope.py`, `tests/test_dense_grouping_scope.py`.
- Outputs: `runs_dense_grouping_scope/` (required JSONs, `figures/fig1`-`fig5`, `logs/`).
- Weights, integer codes, Hessians and caches: `/home/kaixuan6/moe_cache_dense_grouping_scope/{weights,codes,hessians,cache}/<config>/`.

