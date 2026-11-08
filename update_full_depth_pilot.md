# Full-depth routed-expert GPTQ pilot — Full Random s0 vs Full Coverage-LP s0

**Decision (rule frozen in `runs_full_depth_pilot/protocol.json` before any quantized PPL): FULL-DEPTH-PENALTY-NOT-OBSERVED**

## Motivation

The partial-model PPL validation (`update_partial_ppl.md`) quantized the routed experts of only four layers (1/8/16/23). There, Full Coverage-LP was worse than Full Random on 3/3 seeds, by +0.010 PPL at seed 0. The question here: does that penalty persist, grow or vanish when the routed experts of **all 24** MoE layers are quantized?

## Scope (unchanged settings)

- Qwen1.5-MoE-A2.7B; routed experts (gate/up/down) of all 24 MoE layers: 1,440 experts, 4,320 projections.
- GPTQ 4-bit, g128, percdamp 1.0, act-order; B = 128; seed 0.
- The frozen partial-PPL selections are reused (no re-selection).
- Attention, routers, shared experts, embeddings, norms and lm_head stay BF16.
- **Full-depth routed-expert GPTQ, not fully 4-bit LLM quantization.**
- Weights are fake-quantized (dequantized GPTQ weights stored in BF16). No memory or speed claim follows from this.
- Calibration is **non-sequential**: Hessians come from the original BF16 layer inputs, never from quantized activations.
- Evaluation is a sequential 24-layer forward on the frozen `runs_partial_ppl` WikiText-2 windows: 128 × 512, 65,408 scored targets.

## Implementation

| file | role |
|---|---|
| `moe_calib/partial_model_quant.py` | Added `resolve_layers`, `target_tensor_digests`, `nonfinite_targets`, `layer_digest`, `save_layer` (atomic), `install_layer_file`. |
| `diagnostics/full_depth_pilot.py` | Stages: preflight, cache, regression, quant (one fresh process per strategy, resumable per layer with hash verification), report. |
| `diagnostics/report_full_depth_pilot.py` | Summary, paired bootstrap, decision, figures. |
| `diagnostics/full_depth_cache_probe.py` | Shows that the original batch composition reproduces `~/moe_cache` bitwise. |
| `scripts/run_full_depth_pilot.sh` | `STAGE=preflight\|tests\|cache\|regression\|fr\|fc\|report\|all` |
| `tests/test_full_depth_pilot.py` | 12 CPU tests (see below). |

The quantizer itself is E3's `quantize_and_score(write_back=True)`, unchanged. The only difference from partial PPL is that it runs one layer at a time; a test checks this equals all-at-once because calibration is non-sequential.

The 12 tests cover:
- layer enumeration;
- digests that cover every projection;
- non-finite detection;
- compact-cache remapping giving bitwise-identical GPTQ weights;
- layer-by-layer equal to all-at-once;
- no skipped experts and no hidden RTN;
- untouched parameters preserved and a fresh model pristine (no cross-run contamination);
- layer-artifact roundtrip and corruption detection;
- the full-depth layerwise forward matching HF;
- the paired bootstrap (joint, token-weighted, fixed seed);
- the decision-rule cases;
- the frozen seed-0 selections.

## Calibration cache (with a logged amendment)

Only the 345 needed rows are cached for all 24 layers, in `~/moe_cache_full_depth_pilot_v2` (17.4 GB, 6.2 min); `~/moe_cache` is untouched. The rows are the 217 rows in FR ∪ FC plus the 128 test rows 2128–2255.

- **Attempt 1 (v1, FAILED, kept as evidence):**
  - The needed rows were forwarded in new compact batches of 8.
  - BF16 MoE forward depends on batch composition, so the inputs were not bitwise equal to `~/moe_cache`. The mean relative difference was 4e-5 at L1, growing to 2.1 % at L23.
  - The prespecified four-layer regression failed (|ΔPPL| ≤ 0.1·|Δ4| was required):

    | | original cache | v1 cache |
    |---|---|---|
    | FR | 9.267851 | 9.272002 |
    | FC | 9.277908 | 9.268026 |
    | Δ4 | +0.010057 | −0.003975 |

  - Logs: `logs/regression_attempt1_FAILED_v1cache.json` and `logs/cache_manifest_v1.json`.
- **Amendment (v2), made before any full-depth quantization:**
  - Each needed row is forwarded inside its original batch (8 consecutive rows from row 0, as `cache_inputs.py` did).
  - Nothing else changed.
  - Log: `logs/amendment_cache_v2.json`.
- **v2 check:** 345/345 rows are bitwise equal to `~/moe_cache` at L1/8/16/23 (max abs diff 0).

## Regression (four layers, through the updated pipeline)

| | original cache | pilot v2 cache | partial-PPL reference |
|---|---|---|---|
| FR s0 PPL | 9.267851 | 9.267851 | 9.267851 |
| FC s0 PPL | 9.277908 | 9.277908 | 9.277908 |

- Installed weights are identical to the partial-PPL run (sha256).
- Δ4 = +0.010057 is reproduced exactly. **PASSED**, so the full-depth stage went ahead.

## Correctness (both strategies)

- 24 layers, 1,440 experts and 4,320 projections changed from BF16. None unchanged, none non-finite, no RTN fallbacks.
- Untouched parameters bitwise unchanged; each strategy ran in a fresh process with verified pristine hashes.
- Token check: every scored layer received exactly the expected number of routed tokens.
- Expert-output checks (L0/L11/L23 × experts 0/30) pass the BF16-justified tolerance (max 0.56 of tolerance).
- NLL finite. Repeated PPL evaluation gives |ΔPPL| = 0.

| | weights sha256 (prefix) |
|---|---|
| FR | 98fb44e0a716a16f |
| FC | f6358bbf581668bb |

## Results

| configuration | PPL | mean NLL | Δ vs BF16 | rel. degradation | mean expert recon |
|---|---|---|---|---|---|
| BF16 | 9.237090 | 2.223227 | — | — | — |
| Full Random s0, 4 layers | 9.267851 | 2.226552 | +0.0308 | +0.333 % | 0.004610 |
| Full Coverage-LP s0, 4 layers | 9.277908 | 2.227636 | +0.0408 | +0.442 % | 0.004727 |
| Full Random s0, 24 layers | 9.364730 | 2.236951 | +0.1276 | +1.382 % | 0.005061 |
| Full Coverage-LP s0, 24 layers | 9.370674 | 2.237585 | +0.1336 | +1.446 % | 0.005192 |

**Coverage penalty:**

| depth | Δ PPL (FC − FR) | ΔNLL | 95 % CI of ΔNLL |
|---|---|---|---|
| 4 layers | +0.010057 (+0.109 %) | +0.001085 | [+0.000202, +0.002004] |
| 24 layers | +0.005944 (+0.063 %) | +0.000634 | [−0.001578, +0.002778] |

The CIs come from a paired window bootstrap: 5,000 joint resamples, seed 20261109, token-weighted.

- Δ24 − Δ4 = −0.004113; Δ24/Δ4 = 0.59; τ = 1e-4 × PPL_BF16 = 0.000924.
- FC24 is worse on 69/128 windows. Per-window ΔNLL median is +0.00086 (5–95 %: [−0.0216, +0.0191]). The correlation with the 4-layer per-window differences is +0.23.
- Expert reconstruction: FC is worse than FR in **24/24 layers** (layer-mean +2.61 %). The reconstruction penalty is consistent, but at full depth it no longer shows up as a resolvable PPL difference.

Decision path: Δ24 = 0.0059 > τ, but the 95 % CI of ΔNLL includes 0, so the result is **NOT-OBSERVED**. The point estimate stays positive (FC worse) and smaller than at four layers. It is clearly not amplified.

Figures (`runs_full_depth_pilot/figures/`, PDF + PNG):
- `fig1_ppl_by_depth`
- `fig2_coverage_penalty_by_depth`
- `fig3_window_nll_differences`
- `fig4_runtime_memory`

## Calibration-numerics sensitivity (caveat)

The failed v1 attempt is an unplanned but informative measurement. Calibration inputs that differ only by BF16 batch-composition rounding moved four-layer PPL by up to 0.0099, and flipped the sign of Δ4. Reconstruction error moved by only ~3e-5 relative.

So GPTQ-induced PPL differences of order 0.01 between strategies are within the sensitivity of the pipeline to numerically equivalent calibration data. The window bootstrap does not capture this source of variation. This qualifies the four-layer partial-PPL "FC worse" result as well as the full-depth result. No additional runs were added to quantify it.

## Cost

| strategy | quantize (min) | Hessian (min) | GPTQ (min) | PPL eval (s) | peak GPU GB | peak CPU GB | weight artifacts GB |
|---|---|---|---|---|---|---|---|
| Full Random s0 | 31.8 | 3.0 | 20.8 | 23 | 8.4 | 37.8 | 24.9 |
| Full Coverage-LP s0 | 32.3 | 3.1 | 21.2 | 23 | 8.4 | 37.2 | 24.9 |

Hardware: A100 MIG 2g.20gb; the model stays in CPU RAM and one decoder layer at a time is moved to the GPU.

Disk:
- `~/moe_cache_full_depth_pilot_v2`: 63 GB (17.4 GB cache plus 2 × 24.9 GB per-layer weight artifacts).
- `~/moe_cache_full_depth_pilot` (v1, unused, kept as evidence): 17 GB. It can be deleted once the v1 evidence is no longer needed.

## Limitations

- One seed, two strategies. Seeds 1/2 and Typical Random were deliberately not run.
- Fake quantization only. Attention, routers and shared experts stay BF16.
- A single PPL metric on 65k WikiText-2 tokens; no downstream tasks.
- The window bootstrap captures evaluation-text variance only. It misses calibration-numerics variance, which is shown above to be of the same order as the effect.

## One next experiment

Measure the calibration-numerics noise floor directly:
- Repeat the four-layer FR s0 and FC s0 GPTQ with a few (e.g. 3) bitwise-different but numerically equivalent BF16 calibration caches (different batch compositions).
- Selections, settings and evaluation windows stay unchanged.
- This tells whether any between-strategy PPL difference at this size (≈ 0.01) is distinguishable from solver/numerics sensitivity, before any further allocation-strategy PPL comparison.
