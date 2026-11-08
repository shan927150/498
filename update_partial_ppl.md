# Partial-model perplexity validation — November 8, 2026

Follows [`update_extreme_low_budget.md`](update_extreme_low_budget.md). All earlier outputs are unchanged.
Outputs are in `runs_partial_ppl/`:
- `evaluation_protocol.json`
- `selected_set_manifest.json`
- `bf16_baseline.json`
- `reference_check.json`
- `ppl_{FR,TR,FC,TC}_s{0,1,2}.json` and `ppl_FR_s0_repeat.json`
- `summary.md` / `.json`
- `figures/`
- `logs/`

> **Scope: a PARTIALLY quantized model.**
> - Quantized: only the 240 routed experts (gate/up/down) of layers 1/8/16/23, with
>   GPTQ-4-bit, **fake-quantized** weights stored in BF16.
> - BF16, unchanged: attention, routers, shared experts, embeddings, norms, lm_head and the
>   other 20 decoder layers.
> - Not a fully 4-bit model. No memory or throughput claim follows.

**Decision (rule frozen in `evaluation_protocol.json` before any quantized evaluation):
PPL-RECON-ALIGNED, but only narrowly, and the agreement comes entirely from Full Coverage-LP.**
- **Full Coverage-LP is worse than Full Random on 3/3 seeds:** +0.003 to +0.013 PPL, i.e.
  +0.03 % to +0.14 %. That is 7–46 % more degradation than Full Random's.
- **It has the highest mean PPL of the four strategies.**
- **Correlation:** Spearman between reconstruction error and PPL over the 12 configurations is
  +0.52, just above the 0.5 threshold.
- **The other three strategies:** their reconstruction differences are ≤ 0.6 %, and
  reconstruction does **not** order their PPL (Spearman 0.00 over those 9 configurations).
- **Effect size:** every effect is a few thousandths of a PPL point on a 9.24 baseline.

## 1. Motivation

Every allocation result so far rests on held-out expert reconstruction error. This
experiment asks whether those differences show up in end-to-end autoregressive language
modeling. It installs the *same* frozen GPTQ weights into the real model and runs complete
sequential forward passes. No new allocation strategy is introduced.

## 2. Quantized model scope (inspection findings)

| question | finding |
|---|---|
| Projections quantized by E3 | `gate_proj`, `up_proj` (shared Hessian H_gu) and `down_proj` (H_down) of every routed expert (`moe_calib/gptq.quantize_expert`) |
| Experts / layers | all 60 routed experts × layers 1/8/16/23 = 240 (asserted) |
| What GPTQ returns | dequantized FP32 `scale·(q − zero)` (fake-quantized). Per-group asymmetric, act-order permutation undone. No packing |
| Transfer to the HF model | yes. E3's `quantize_and_score(write_back=True)` copies each W_q (cast to BF16) into `experts[e].{proj}.weight`, in native [out, in] orientation (no bias) |
| Calibration | non-sequential: Hessians from cached BF16 inputs of each layer (`~/moe_cache/L*.pt`). Each layer's Profiler runs before that layer is quantized |
| Untouched modules | attention, router `mlp.gate`, `shared_expert(_gate)`, embeddings, norms, lm_head, layers ≠ 1/8/16/23 |
| Full forward feasible? | yes. Model in CPU RAM (BF16 ≈ 28.6 GB; 226 GB host), one decoder layer at a time on the 20 GB MIG slice. Embed/norm/lm_head stay on the GPU |

## 3. GPTQ weight installation

`moe_calib/partial_model_quant.py` wraps E3's own `quantize_and_score(..., write_back=True)`.
The solver, settings and fallback rule are unchanged. It adds:
- **Pristine handling:** pristine snapshots of the 720 target tensors, restoration, and
  bitwise "pristine" checks before and after every configuration.
- **Capture:** an interception of E3's write-back that records the FP32 W_q of 8
  prespecified check experts (experts 0 and 30 of each layer).
- **Untouched-state digests:** sha1 of all 3,964 untouched parameters and buffers.
- **Run identity:** a sha256 over all installed expert weights.
- **Output check:** an expert-level comparison on identical inputs (`expert_output_check`).

Each configuration starts from the pristine BF16 weights. The installed weights are restored
afterwards, so no configuration ever sees another's weights.

## 4. Frozen calibration selections

`selected_set_manifest.json` re-validates the 12 selections from `runs_typicality_matched`
(B = 128; seeds 0/1/2):
- **Strategies:** Full Random, Typical Random, Full Coverage-LP and Typical Coverage-LP.
- **Membership:** 128 unique ids; the ordered 32-id warm start and its hash; pool and
  typical-half membership; no tune/test ids.
- **Settings:** frozen GPTQ settings; no RTN fallback.
- **Cross-check:** each selection's id hash equals the one used in the locked confirmation
  runs.

Nothing was regenerated.

## 5. Model loading and memory

- **Load:** `MOE_LOAD_LAYERS` is unset, so the full BF16 checkpoint is loaded into CPU RAM.
- **Forward:** all 128 windows go through layer L while that layer alone is on the GPU
  (batch 8), then layer L + 1, and so on. Embedding, final norm and lm_head stay on the GPU.
- **Precision:** BF16 weights and activations, eager attention, FP32 logits and loss.
- **No changes to fit memory:** no 8-bit loading, no additional quantization, no
  architecture or routing changes.
- **Peaks:**
  - GPU: 3.3 GB (BF16 evaluation), 8.4 GB (with GPTQ).
  - Process CPU: 40 GB (BF16), 68 GB (quantization process, including the pristine snapshot).

## 6. WikiText-2 evaluation protocol (frozen before any model evaluation)

- **Data:** `wikitext-2-raw-v1` test split, HF cache revision `b08601e0…`, datasets 3.6.0,
  text sha256 `696cca6b…`.
- **Tokenizer:** Qwen1.5-MoE-A2.7B snapshot `1a758c50…`, transformers 4.51.3. It adds no
  special tokens (no BOS).
- **Text and windows:**
  - `"\n\n".join(text)` is tokenized once: 299,078 tokens, i.e. 584 complete
    non-overlapping 512-token windows.
  - 128 windows were drawn with `random.Random(20261108)` and sorted. Offsets are in the
    protocol.
  - Window tensor sha256 `e646cf7c…`, checked on every load.
- **Scoring:**
  - `logits[:, :-1]` against `input_ids[:, 1:]`.
  - The first token of each window is context only, giving 511 targets per window and
    65,408 in total.
  - FP32 cross entropy with FP64 accumulation; PPL = exp(total NLL / total targets).
  - No padding, positions restart at 0, and per-batch PPLs are never averaged.
- **Same windows everywhere:** the BF16 baseline and all 12 configurations use identical
  windows.

## 7. Numerical correctness checks

| check | result |
|---|---|
| BF16 evaluated twice in one process | PPL 9.237090 both times (Δ = 0) |
| BF16 across separate processes (2 more) | Δ = 0 |
| BF16 after all 13 quantize → evaluate → restore cycles | Δ = 0; 0 untouched parameters changed |
| Independent reference: HF `AutoModelForCausalLM` with accelerate offload, `labels=`, batch 1, windows 0/1 | mean NLL 1.76340 vs 1.76428 and 2.38579 vs 2.38390 (rel. 5e-4, 8e-4). Our scorer on HF's logits matches HF's loss to 5e-8, so the gap is BF16 kernel / batch-composition noise, not scoring |
| E3 per-expert reconstruction, recomputed while installing (all 12) | identical to the frozen runs (max rel. diff **0.0**) |
| Installed weights (8 check experts × 12) | bitwise equal to the BF16 cast of E3's W_q; all differ from the originals |
| Installed BF16 module vs E3 FP32 output, identical routed inputs | rel. ≤ 4.8e-3 vs a quantization perturbation of ≈ 0.13 (median); ≤ 0.95 × the BF16-justified tolerance |
| Untouched parameters / buffers | 0 of 3,964 changed in every run |
| RTN fallbacks | 0 |
| Re-quantizing Full Random s0 from scratch | identical installed weights (sha256) and identical PPL (Δ = 0) |

**Amendment (before any quantized PPL existed).**
- **What happened:** the first smoke attempt (FR s0) failed the original installation rule
  ("installed-vs-E3 deviation ≤ 10 % of the quantization effect"). Layer 1 expert 0 deviated
  0.34 %, which was 10.5 % of its unusually small 3.2 % quantization effect.
- **Weights were not the problem:** they were bitwise exact. The deviation was BF16
  arithmetic, which the original-weight module shows equally (0.26 %), plus BF16 storage of
  W_q (0.11 %).
- **New rule:** deviation ≤ weight-rounding deviation + 1.5 × BF16-compute deviation of the
  same module with its original weights.
- **Records:** `logs/amendment_install_tolerance.json` and `logs/smoke_attempt1_FAILED.json`.
  No PPL had been computed at that point.

## 8. CPU tests

- **`tests/test_partial_ppl.py`:** 15 tests on a tiny synthetic Qwen2-MoE, plus the real
  selection manifest. They cover:
  - selection loading and hashes;
  - expert module mapping;
  - exact weight replacement with orientation preserved;
  - preservation of untouched modules;
  - act-order and group mapping;
  - agreement with E3 expert outputs;
  - RTN fallback reported rather than hidden;
  - sequential forward that propagates quantization and matches HF's own loss;
  - no cross-run contamination (restore, then reinstall, gives the same digest as a fresh
    model);
  - target shifting and token counting;
  - padding excluded;
  - token-weighted accumulation;
  - deterministic evaluation;
  - window fingerprint validation;
  - the frozen GPTQ config;
  - the amended tolerance.
- **Full suite:** **173 passed, 0 failed.**

## 9. GPU execution status

- **Evaluations:** 12 / 12 partially quantized evaluations, the BF16 baseline (×2, plus 2
  cross-process checks and 1 final check), the independent reference, and the repeatability
  rerun. All were sequential, one GPU process at a time.
- **Driver:** `scripts/run_partial_model_ppl.sh` (resumable).

## 10. BF16 baseline

**PPL 9.2371** over 65,408 scored targets (128 × 511).

## 11. PPL per method and seed

| strategy | s0 | s1 | s2 | mean ± SD | Δ vs BF16 | relative degradation |
|---|---|---|---|---|---|---|
| Full Random | 9.2679 | 9.2747 | 9.2656 | 9.2694 ± 0.0047 | +0.0323 | +0.350 % |
| Typical Random | 9.2680 | 9.2747 | 9.2623 | **9.2683** ± 0.0062 | +0.0312 | +0.338 % |
| Full Coverage-LP | 9.2779 | 9.2774 | 9.2787 | 9.2780 ± 0.0007 | +0.0409 | +0.443 % |
| Typical Coverage-LP | 9.2725 | 9.2775 | 9.2675 | 9.2725 ± 0.0050 | +0.0354 | +0.383 % |

- **Small total degradation.** Quantizing the experts of 4 of 24 layers costs only +0.34 % to
  +0.44 % PPL.
- **The seed matters more than the strategy for three of the four.** Seed 1 is the worst seed
  for FR, TR and TC alike. Seeds share their 32-sequence warm start across strategies, so
  this is plausibly a warm-start effect, but that is not established here.

## 12. Paired differences (same seed; > 0 = first method worse)

| pair | ΔPPL s0 / s1 / s2 | mean | share of the baseline's degradation | 95 % window bootstrap excludes 0 | Δrecon (rel.) |
|---|---|---|---|---|---|
| Full Coverage-LP − Full Random | +0.0101 / +0.0027 / +0.0131 | +0.0086 | +33 / +7 / +46 % | s0, s2 (not s1) | +2.55 / +2.19 / +2.71 % |
| Typical Coverage-LP − Typical Random | +0.0045 / +0.0028 / +0.0052 | +0.0042 | +15 / +7 / +21 % | none | +0.09 / +0.21 / +0.21 % |
| Typical Random − Full Random | +0.0002 / +0.0000 / −0.0033 | −0.0010 | +1 / 0 / −12 % | none | +0.05 / −0.57 / −0.12 % |
| Full Coverage-LP − Typical Random (context) | +0.0099 / +0.0027 / +0.0164 | +0.0097 | +32 / +7 / +65 % | s0, s2 | +2.50 / +2.77 / +2.83 % |

- **What the bootstrap covers.** It resamples the 128 evaluation windows for a *fixed*
  calibration set: benchmark-sampling uncertainty, not seed variance.
- **Numerical floor.** Repeatability is exactly 0, so the frozen τ sits at its floor,
  1e-4 × PPL = 0.00092.
- **No significance claims.** Three seeds do not support formal significance statements.

## 13. Reconstruction–PPL relationship (12 configurations; descriptive)

- **All 12 configurations:** Spearman / Pearson between reconstruction error and PPL is
  +0.52 / +0.66 against the old test split, and +0.62 / +0.67 against the locked
  confirmation split.
- **Rank of the means:**

  | metric | best → worst |
  |---|---|
  | PPL | TR < FR < TC < FC |
  | reconstruction (both splits) | TR < TC < FR < FC |

  Both agree on best (TR) and worst (FC). TC and FR swap places.
- **Without Full Coverage-LP** (FR / TR / TC, 9 configurations): Spearman is **0.00** (old
  test) and +0.18 (confirmation). The correlation comes from FC's ~2.5 % reconstruction gap.
- **Small reconstruction differences are not visible in PPL.** Differences of a few tenths of
  a percent (TR vs FR, TC vs TR) are swamped by seed-to-seed PPL variation (SD ≈ 0.005).
- **Typical Coverage-LP is the one partial exception.** TC is above TR on 3/3 seeds
  (+0.003 to +0.005). Its reconstruction gap is tiny, but no bootstrap interval excludes 0.

Figures: `fig1_ppl_by_strategy`, `fig2_ppl_degradation`, `fig3_recon_vs_ppl`,
`fig4_paired_differences`.

## 14. Compute overhead (mean per configuration)

| item | cost |
|---|---|
| Hessian replay (4 layers, 128 sequences) | ≈ 32 s |
| GPTQ, 240 experts | ≈ 205 s |
| Quantize + install + E3 rescoring + checks (wall) | ≈ 266 s |
| Sequential 24-layer PPL evaluation, 128 × 512 | 22–44 s |
| Model load (page-cached) | 8 s |
| CPU offloading | the whole model lives in CPU RAM; a layer-transfer cost is included in each evaluation |
| Peak GPU / process CPU | 8.4 GB / 68 GB |
| Packed artifact if produced | ≈ 1.1 GB for 240 experts (not produced; weights are fake-quantized) |
| Routing profile (TR, FC, TC only) | ≈ 357 s full-forward equivalent; Full Random needs none |

**Equal budget is not equal compute.** At equal B = 128, the two strategies with the best PPL
(TR and FR) are indistinguishable within seed-to-seed variation, and FR is the only one that needs no routing
profile.

## 15. Scientific interpretation

- **The headline result does reach the model.** Full Coverage-LP's ~2.5 % higher expert
  reconstruction error appears as worse partial-model PPL on all three seeds. That gives the
  main negative finding about exposure-based allocation model-level support, in this
  setting.
- **The scale is tiny.** Partial quantization barely moves PPL (+0.35 %). FC's penalty is
  +0.03–0.14 % PPL, and on one seed it is inside the window-bootstrap noise.
- **Small reconstruction differences don't predict PPL.** Differences of a few tenths of a
  percent (Typical vs Full Random, Typical Coverage vs Typical Random) do not predict PPL
  ordering. The local metric is reliable only for differences as large as FC's.
- **The decision rests on a near-threshold correlation.** "PPL-RECON-ALIGNED" relies on a
  Spearman of +0.52 against a 0.5 threshold, driven by one strategy. Read it as: *large
  reconstruction penalties transfer to the model; small ones are not resolvable at the model
  level with this evaluation.*

## 16. Limitations

- **Partial model.** Only 4 of 24 layers' routed experts are quantized. The full-model 4-bit
  effect, and how errors compound across all layers, is untested.
- **Fake quantization** (BF16 dequantized weights): no packed-kernel numerics.
- **Small evaluation and design.**
  - One benchmark: 65k WikiText-2 tokens, 512-token context.
  - Three allocation seeds.
  - The 12 configurations share warm starts within a seed, so they are not independent
    replicates.
- **Calibration is non-sequential.** Hessians come from BF16 inputs, as in E3, so it does not
  model error from earlier quantized layers. Inference is sequential.
- **Reconstruction errors used for comparison** come from the old test split (exploratory)
  and the locked confirmation split. PPL used neither.

## 17. Recommended next experiment

**Extend the same pipeline to all 24 MoE layers**, i.e. every routed expert quantized, for the
Full Random and Full Coverage-LP selections only (3 seeds, same frozen windows).
- **Why:** the decision currently rests on a +0.35 % total degradation. That is a reach-only
  test of the one contrast with a large reconstruction gap, and the full-depth version shows
  whether the gap compounds to a practically meaningful PPL difference.
- **What it needs:** routing caches and Hessians for the remaining 20 layers. No new
  allocation method, and the B = 128 selections stay frozen.
