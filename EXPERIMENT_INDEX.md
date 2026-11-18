# Experiment index

Chronological index of the project's experiment reports (`update_*.md`, one per experiment; each links its
`runs_*/` output directory and frozen protocol). Dates are the report dates. Historical protocols and results are never
rewritten; later reports qualify earlier ones explicitly.

## 1. Calibration-allocation methods and PPL alignment (Oct 18 – Nov 8)

| date | report | experiment |
|---|---|---|
| 2026-10-18 | [`update_10_18.md`](update_10_18.md) | Project Update — October 18, 2026 |
| 2026-10-30 | [`update_10.29.md`](update_10.29.md) | Project Update — October 29, 2026 (P0 → P2c) |
| 2026-11-06 | [`update_11.06.md`](update_11.06.md) | Project Update — November 6, 2026 (minimal fixed-budget allocation) |
| 2026-11-06 | [`update_matched_typicality_11.06.md`](update_matched_typicality_11.06.md) | Matched-warm-start validation of routing typicality — November 6, 2026 |
| 2026-11-07 | [`update_repr_constrained.md`](update_repr_constrained.md) | Representativeness-constrained Coverage-LP — November 7, 2026 |
| 2026-11-07 | [`update_aggregate_constraint.md`](update_aggregate_constraint.md) | Final experiment: aggregate-shift-constrained Coverage-LP on a locked confirmation split — November 7, 2026 |
| 2026-11-07 | [`update_activation_swap.md`](update_activation_swap.md) | Activation-aware calibration swap pilot — November 7, 2026 |
| 2026-11-08 | [`update_low_budget_sweep.md`](update_low_budget_sweep.md) | Low-budget calibration allocation sweep — November 8, 2026 |
| 2026-11-08 | [`update_extreme_low_budget.md`](update_extreme_low_budget.md) | Extreme low-budget exposure scarcity study (B = 4 / 8) — November 8, 2026 |
| 2026-11-08 | [`update_partial_ppl.md`](update_partial_ppl.md) | Partial-model perplexity validation — November 8, 2026 |
| 2026-11-08 | [`update_full_depth_pilot.md`](update_full_depth_pilot.md) | Full-depth routed-expert GPTQ pilot — Full Random s0 vs Full Coverage-LP s0 |

## 2. Numerical-execution sensitivity in MoE GPTQ (Nov 8 – Nov 9)

| date | report | experiment |
|---|---|---|
| 2026-11-08 | [`update_batch_sensitivity.md`](update_batch_sensitivity.md) | Calibration batch-composition sensitivity in MoE GPTQ (four layers, seed 0) |
| 2026-11-08 | [`update_multigroup_robustness.md`](update_multigroup_robustness.md) | Multi-grouping robustness of MoE GPTQ calibration (FR s0 vs FC s0, four layers) |
| 2026-11-09 | [`update_cross_seed_grouping.md`](update_cross_seed_grouping.md) | Cross-seed × calibration-grouping robustness (FR vs FC, four-layer GPTQ, allocation seeds 0/1/2) |
| 2026-11-09 | [`update_frozen_routing.md`](update_frozen_routing.md) | Frozen-routing 2×2 factorial ablation of MoE GPTQ calibration (FR s0 / FC s0, four layers, ORIGINAL vs G20261111) |
| 2026-11-09 | [`update_frozen_routing_robustness.md`](update_frozen_routing_robustness.md) | Frozen-routing robustness across five calibration batch groupings |
| 2026-11-09 | [`update_installation_audit.md`](update_installation_audit.md) | Installation audit: the FC AO G20261114 BF16 tolerance exceedance |
| 2026-11-09 | [`update_actorder_ablation.md`](update_actorder_ablation.md) | GPTQ act-order ablation after frozen-routing calibration |
| 2026-11-09 | [`update_hessian_order_null.md`](update_hessian_order_null.md) | FP32 Hessian accumulation-order null experiment |

## 3. Calibration-method follow-ups: EBSS, AGQ, expert-wise, shrinkage, SC (Nov 10 – Nov 17)

| date | report | experiment |
|---|---|---|
| 2026-11-10 | [`update_ebss_official_validation.md`](update_ebss_official_validation.md) | Official MoEQuant EBSS calibration-quality validation (four layers, B = 128 × 512) |
| 2026-11-10 | [`update_ebss_generation_pilot.md`](update_ebss_generation_pilot.md) | Model-matched EBSS generation pilot + Probability-Only ablation (16 × 512) |
| 2026-11-10 | [`update_ebss_score_audit.md`](update_ebss_score_audit.md) | EBSS score/sign correctness audit |
| 2026-11-10 | [`update_ebss_balance_fidelity.md`](update_ebss_balance_fidelity.md) | MoEQuant EBSS balance-strength fidelity experiment |
| 2026-11-11 | [`update_ebss_prefix_rescue.md`](update_ebss_prefix_rescue.md) | C4-prefix diversity rescue for modified EBSS (B=16, P=64 + G=448) |
| 2026-11-11 | [`update_agq_only_validation.md`](update_agq_only_validation.md) | AGQ-only validation: standard GPTQ vs Affinity-Guided GPTQ on matched C4 Random B = 128 |
| 2026-11-11 | [`update_agq_c2_pilot.md`](update_agq_c2_pilot.md) | Small-scale c²-weighted Hessian pilot (EXPLORATORY extension of AGQ) |
| 2026-11-11 | [`update_hessian_early_stop_mvp.md`](update_hessian_early_stop_mvp.md) | Per-expert Hessian early stopping: feasibility MVP |
| 2026-11-12 | [`update_legacy_metric_reeval.md`](update_legacy_metric_reeval.md) | Legacy metric re-evaluation: do failed MoE calibration methods still fail under actual MoE output error? |
| 2026-11-13 | [`update_expertwise_agq_hybrid.md`](update_expertwise_agq_hybrid.md) | Expert-wise GPTQ/AGQ hybrid: small-scale feasibility MVP |
| 2026-11-13 | [`update_projection_selective_agq.md`](update_projection_selective_agq.md) | Projection-selective AGQ: small-scale feasibility MVP |
| 2026-11-13 | [`update_fixed_gu_agq.md`](update_fixed_gu_agq.md) | Fixed GU-AGQ across four MoE layers and three C4 seeds |
| 2026-11-13 | [`update_hessian_pool_shrinkage_mvp.md`](update_hessian_pool_shrinkage_mvp.md) | Layer-pooled Hessian shrinkage for low-budget MoE GPTQ: feasibility MVP |
| 2026-11-16 | [`update_expertwise_calibration_mvp.md`](update_expertwise_calibration_mvp.md) | Expert-wise activation calibration for MoE GPTQ: MVP |
| 2026-11-16 | [`update_shared_covariance_e3.md`](update_shared_covariance_e3.md) | Shared covariance-aware calibration (SC): full E3 validation |
| 2026-11-16 | [`update_loss_alignment.md`](update_loss_alignment.md) | Loss-alignment diagnostic (MVP A + B): why SC's reconstruction gain does not show up in PPL |
| 2026-11-17 | [`update_calibration_domain_ablation.md`](update_calibration_domain_ablation.md) | Calibration-domain ablation: C4 vs WikiText-2-train calibration (SC vs Random) |

## 4. Reproducibility paper core: mechanism, dense control, uncertainty, external validity (Nov 13 – Nov 18)

| date | report | experiment |
|---|---|---|
| 2026-11-13 | [`update_fp64_canonical_reproducibility.md`](update_fp64_canonical_reproducibility.md) | FP64 canonical GPTQ and numerical-reproducibility synthesis |
| 2026-11-14 | [`update_inference_routing_amplification.md`](update_inference_routing_amplification.md) | Inference-time routing amplification MVP |
| 2026-11-14 | [`update_grouping_routing_amplification.md`](update_grouping_routing_amplification.md) | Grouping × routing amplification: frozen-routing intervention on calibration batch grouping (Family D) |
| 2026-11-14 | [`update_dense_grouping_scope.md`](update_dense_grouping_scope.md) | Dense-model control: GPTQ calibration-grouping reproducibility (Qwen2.5-1.5B) |
| 2026-11-15 | [`update_integer_code_mechanism.md`](update_integer_code_mechanism.md) | GPTQ integer-code mechanism: code-flip spatial structure and dense vs MoE code disagreement |
| 2026-11-15 | [`update_actorder_functional_stability.md`](update_actorder_functional_stability.md) | Functional impact of GPTQ code disagreement and act-order stabilization |
| 2026-11-15 | [`update_dense_dose_response.md`](update_dense_dose_response.md) | Dense GPTQ numerical-perturbation dose response and act-order stabilization |
| 2026-11-18 | [`update_heldout_bitwidth_validation.md`](update_heldout_bitwidth_validation.md) | Held-out evaluation validation and 3-bit GPTQ execution sensitivity |
| 2026-11-18 | [`update_evaluation_uncertainty.md`](update_evaluation_uncertainty.md) | Execution variability vs evaluation-sampling uncertainty |
| 2026-11-18 | [`update_wt2_fresh_generalization.md`](update_wt2_fresh_generalization.md) | Final WikiText-2 fresh evaluation: article-level generalization |

## Reproducing the final three experiments (CPU/GPU commands; weights and caches are not in git)

```bash
source ~/venvs/moe/bin/activate
# held-out C4 + 3-bit (GPU; needs ~/moe_cache_* weight/Hessian caches listed in artifact_inventory.json)
for s in select audit protocol dense dense3 dense3_repeat moe analyze; do python diagnostics/heldout_bitwidth_validation.py --stage $s; done
python diagnostics/heldout_bitwidth_functional.py
# evaluation-uncertainty decomposition (CPU only, reads runs_heldout_bitwidth_validation/results)
python diagnostics/evaluation_uncertainty.py --stage all && python diagnostics/report_evaluation_uncertainty.py
python diagnostics/evaluation_uncertainty_posthoc.py
# WikiText-2 fresh-article generalization (GPU for dense/moe)
for s in hist cohorts protocol dense moe synth analyze; do python diagnostics/wt2_fresh_generalization.py --stage $s; done
python diagnostics/report_wt2_fresh_generalization.py
```

Excluded from git (by `.gitignore` / size): model checkpoints, quantized weight dumps (`~/moe_cache_*`), Hessian and
activation caches, `*.pt` window tensors and `logs/` directories. Every run directory keeps the hashes needed to verify
those artifacts (`artifact_inventory.json`, `protocol.json`).
