# Multi-grouping robustness of MoE GPTQ calibration (FR s0 vs FC s0, four layers)

**Decision (rule frozen in `runs_multigroup_robustness/protocol.json`, sha256 `f97271a3a1e0d42c…`, before any quantization): GROUPING-DEPENDENT-RANKING**

## Question

The batch-sensitivity study showed that regrouping the same calibration sequences into different BF16 forward batches can reverse the Full Random (FR) vs Full Coverage-LP (FC) PPL ranking. It tested only three deterministic groupings.

This study asks whether that holds under five additional groupings, each drawn with a prespecified random seed. Selections and GPTQ stay fixed.

## Design (frozen before any result)

- **Fixed:**
  - Qwen1.5-MoE-A2.7B; routed experts of layers 1/8/16/23 (240 experts, 720 projection tensors).
  - GPTQ 4-bit, g128, percdamp 1.0, act-order. Fake-quantized BF16 weights; everything else BF16.
  - B = 128, allocation seed 0. The FR s0 (`ec355fb8d866e4ff`) and FC s0 (`0353229b6d3090db`) selections are re-verified against their source-file hashes: 128 unique IDs each, frozen ordered warm start, no tune or test leakage. Their union is 217 sequences.
  - Non-sequential calibration from pristine-BF16 layer inputs (E3 Profiler).
  - Reconstruction is always scored on `~/moe_cache` test rows 2128–2255.
  - Frozen WikiText-2 windows (128 × 512, 65,408 targets) and evaluation batch 8.
- **Varied:** only which calibration sequences share a forward batch.
- **New groupings:** seeds 20261111–20261115. Each takes the sorted 217-row union, applies `default_rng(seed).permutation`, and cuts consecutive batches of 8 (27 × 8 + 1). They contain only the 217 calibration rows; no evaluation or held-out rows enter. FR and FC use the **same** grouping, so the design is paired.
- **Policy difference:** COMPACT and SHUFFLED used 345 rows (the union plus 128 old-test rows). They are kept as historical references only and are never pooled with the five new groupings.

## Validity and determinism controls (all passed)

- **Inputs:**
  - Token IDs after reassembly by sequence ID are identical to the ORIGINAL rows for all five groupings (same sha256).
  - The decoder-layer kwargs (causal mask, positions, rotary) are batch-1 and broadcast, so they are the same for every batch.
  - There is no padding and no token dropping.
  - All parameters match the checkpoint digests. The model is in eval mode, BF16, with eager attention and no active dropout.
  - An FP64 CPU test shows every grouping gives the same per-sequence activations to 1e-12.
- **Execution order:** each grouping's batches, run in reversed order, reproduce its layer-1/8/16/23 cache bitwise (5/5).
- **ORIGINAL reproduction (fresh processes):** FR and FC from `~/moe_cache` reproduce the frozen installed-weight sha256, PPL (9.267851 / 9.277908) and reconstruction bitwise. Repeated quantization of an identical cache is therefore deterministic.
- **Repeated PPL evaluation:** |ΔPPL| = 0, with identical per-window NLL, in all 12 runs.
- **All 10 new configurations:**
  - 240 experts quantized and 720/720 projections changed.
  - 0 RTN fallbacks, 0 non-finite weights.
  - Untouched parameters unchanged.
  - All 8 expert-output checks are within the BF16 tolerance.
- **Environment:** torch 2.11 (CUDA 12.8), transformers 4.51.3, A100 MIG 2g.20gb. This is the same stack as the earlier study, since ORIGINAL reproduces bitwise.

## Results

### PPL (primary; BF16 PPL 9.237090)

| grouping | FR PPL | FC PPL | Δ_g = FC − FR | I_g = Δ_g − Δ_ORIGINAL | ranking |
|---|---|---|---|---|---|
| G20261111 | 9.278182 | 9.274113 | **−0.004069** | −0.014126 | FC better |
| G20261112 | 9.268159 | 9.275760 | **+0.007601** | −0.002455 | FC worse |
| G20261113 | 9.271035 | 9.265146 | **−0.005889** | −0.015946 | FC better |
| G20261114 | 9.278102 | 9.269707 | **−0.008394** | −0.018451 | FC better |
| G20261115 | 9.272717 | 9.273808 | **+0.001091** | −0.008966 | FC worse |

- **Δ_g summary:**
  - Mean −0.001932, median −0.004069, SD (ddof 1) 0.006364.
  - Minimum −0.008394, maximum +0.007601, range 0.015996.
- **Ranking:**
  - FC is worse in 2/5 groupings and better in 3/5.
  - The ranking reverses relative to ORIGINAL (+0.010057) in **3/5**. All three reversals exceed τ = 1e-4 × BF16 = 0.000924.
  - Every |Δ_g| is greater than τ, and every one exceeds the measured repeatability (0).
- **Strategy-specific ranges:**
  - FR: 0.010023 (1.00 × |Δ_ORIGINAL|).
  - FC: 0.010614 (1.06 ×).
- **Interaction:**
  - Max |I_g| = 0.018451 (1.83 × |Δ_ORIGINAL|).
  - SD(Δ_g) / |Δ_ORIGINAL| = 0.63. This only describes this batch-generation policy; it is not a universal signal-to-noise ratio.
- **No grouping reproduces the ORIGINAL gap:** every I_g is negative, and the largest new Δ_g (+0.0076) is still below +0.0101. ORIGINAL is one draw from a different policy, so this is an observation, not a test.

### Paired-window bootstrap (FC − FR token-weighted NLL; 5,000 replicates, seed 20261109)

| grouping | point | 95 % CI |
|---|---|---|
| G20261111 | −0.00044 | [−0.00129, +0.00047] |
| G20261112 | +0.00082 | [−0.00002, +0.00166] |
| G20261113 | −0.00064 | [−0.00152, +0.00024] |
| G20261114 | −0.00091 | [−0.00174, −0.00004] |
| G20261115 | +0.00012 | [−0.00083, +0.00110] |
| ORIGINAL (reference) | +0.00108 | [+0.00020, +0.00200] |

- Each interval reflects evaluation-text sampling for one fixed pair of quantized models. It says nothing about calibration-batching uncertainty.
- G20261114 alone would "significantly" favour FC, while ORIGINAL would "significantly" favour FR. Neither conclusion survives across groupings.

### Reconstruction (same 240 experts, same evaluation rows)

- **Mean reconstruction:**
  - FR: 0.0046091–0.0046093.
  - FC: 0.0047266–0.0047271.
  - Compared with ORIGINAL, the largest relative change is 8.8e-5 (FR) and 9.7e-5 (FC), i.e. ≤ 0.01 %.
- **Per-expert changes:**
  - Median |change| about 0.06 %, maximum 0.40 %.
  - About half of the experts get worse and half get better.
  - Per-layer means are stable to about 1e-4 relative.
- **FC − FR reconstruction gap:**
  - +2.54 % to +2.56 % in all five groupings, against +2.55 % for ORIGINAL.
  - FC reconstruction is worse in 5/5 groupings, and the gap varies by only 0.014 percentage points.
- **Reconstruction does not order PPL:**
  - Within-strategy correlation of reconstruction with PPL across the five groupings: r = 0.20 (FR), 0.28 (FC).
  - Correlation of the reconstruction gap with Δ_g: r = 0.46.
  - Each correlation has n = 5, so they are descriptive only.
- **PPL moves while reconstruction stays fixed (fig 4).** The earlier observation persists.

### Effective weights vs ORIGINAL

- FR: 54.0–54.2 % of weight elements differ, relative L2 0.089.
- FC: 55.1–55.2 % differ, relative L2 0.090–0.091.
- These are the same magnitudes as COMPACT and SHUFFLED.

### Activations and routing vs ORIGINAL (217 calibration rows; GPU top-4)

| layer input | rel. L2 | % elements bitwise equal | token rel. L2 p50 / p99 | % tokens top-4 set changed | routing L1 |
|---|---|---|---|---|---|
| 1 | 1.5–1.9e-4 | 99.9 | 0 / 2.1–2.4e-4 | 0.61–0.65 | 4.8–5.5e-4 |
| 8 | 5.7–5.8e-3 | 13.0 | 1.2e-2 / 8.0e-2 | 3.78–3.90 | 1.2–1.4e-3 |
| 16 | 1.47–1.48e-2 | 11.1 | 1.4e-2 / 7.4e-2 | 3.89–3.99 | 1.3–1.4e-3 |
| 23 | 2.13–2.15e-2 | 10.4 | 1.5e-2 / 7.4e-2 | 6.65–6.78 | 1.4–2.0e-3 |

- The five groupings are equally far from ORIGINAL and from each other. For example, G20261112–15 vs G20261111 at layer 23: rel. L2 about 0.021, and 6.7 % of top-4 sets change.
- The historical SHUFFLED grouping sits on the same curve (fig 5).
- The size of the activation and routing perturbation is nearly identical across groupings, yet Δ_g ranges from −0.0084 to +0.0076. The sign of the PPL outcome is not predictable from these aggregate diagnostics.
- Nothing here establishes a causal routing mechanism.

### Historical comparison (different policies; not pooled)

| grouping | policy | FR | FC | Δ |
|---|---|---|---|---|
| ORIGINAL | cache_inputs.py, 176 × 8 consecutive original rows | 9.267851 | 9.277908 | +0.010057 |
| COMPACT | 345 rows (union + 128 old-test), sorted, 43 × 8 + 1 | 9.272002 | 9.268026 | −0.003975 |
| SHUFFLED | 345 rows, default_rng(20261110), 43 × 8 + 1 | 9.278272 | 9.265841 | −0.012431 |
| G20261111–15 | 217 calibration rows only, 27 × 8 + 1 | see above | see above | −0.0084 … +0.0076 |

## Answers

1. **Is the ORIGINAL FR/FC ranking robust to the five new groupings?**
   - No. FC is better in 3/5 groupings, beyond the deterministic tolerance.
   - The mean Δ is −0.0019, which is close to zero relative to its SD of 0.0064.
2. **Is the grouping-induced variation comparable to the selection effect?**
   - Yes. Each strategy's PPL range (0.0100 and 0.0106) equals |Δ_ORIGINAL|.
   - The SD of Δ_g is 0.63 × |Δ_ORIGINAL|, and the largest interaction is 1.83 ×.
3. **Does reconstruction stay stable while PPL varies?**
   - Yes. Mean reconstruction changes by ≤ 0.01 %, and the FC reconstruction penalty (+2.55 %) is the same in every grouping.
   - Meanwhile PPL moves by about 0.01, and the FC−FR PPL difference changes sign.
4. **Reproducibility limitation:**
   - A model-level PPL difference of about 0.01 (0.1 %) between calibration selections cannot be identified from one calibration-extraction run. A window bootstrap that excludes zero does not change this.
   - Bitwise-deterministic code is not enough: a different but equally valid batch grouping gives a different 4-bit solution, and its PPL spread is as large as the effect being compared.
   - Any such comparison needs a grouping-level replicate distribution, or effects clearly larger than this spread.
5. **What remains unproven:**
   - Only one allocation seed and one selection pair were tested, so selection-sampling variance is not separated from grouping variance.
   - Five groupings give a rough spread estimate (the SD has wide uncertainty at n = 5). They do not give a noise distribution for other policies, GPUs, kernels, models or layer sets.
   - Whether FC is truly better, worse or equal in PPL remains undetermined. The data are consistent with a true difference near zero.
   - The weights → PPL step is observed, not causally isolated.
   - Everything here uses fake-quantized weights; nothing is claimed about packed INT4 inference.

## One recommended follow-up (not executed)

Before attributing any model-level PPL difference to calibration selection, compare strategies against a grouping-replicate baseline:
- Average PPL over a prespecified set of K groupings per configuration (e.g. these five) for FR and FC at allocation seeds 1 and 2 (selections already frozen).
- Test whether the selection effect, averaged over groupings and seeds, exceeds the grouping SD measured here.

## Compute

- **Cache build:** about 60–76 s per grouping (217 rows, 4 layers saved), plus about 37 s for each reversed-order control.
- **Diagnostics:** 216 s.
- **Quantization:** about 294 s per configuration on average, plus about 43 s per PPL evaluation (2 per configuration). 12 fresh-process runs in total (2 ORIGINAL controls + 10 new).
- **Peak memory:** 8.4 GB GPU, 45.5 GB host.
- **Disk:**
  - `~/moe_cache_multigroup_robustness`: caches 9.1 GB; effective weights 49.8 GB (12 × 4.15 GB).
  - Earlier caches and results are untouched.
- **Tests:** full suite 223 passed, including 18 new CPU tests in `tests/test_multigroup_robustness.py`.

## Files

| file | role |
|---|---|
| `moe_calib/multigroup_robustness.py` | Grouping construction and validation, manifests, statistics, frozen decision rule, bootstrap interpretation. |
| `diagnostics/multigroup_robustness.py` | Stages: preflight, manifest, cache, diagnostics, quant. |
| `diagnostics/report_multigroup_robustness.py` | Results, summary and figures 1–5. |
| `scripts/run_multigroup_robustness.sh` | Resumable runner (`STAGE=preflight|tests|manifest|cache|diagnostics|quant|report`), one fresh GPU process per configuration. |
| `tests/test_multigroup_robustness.py` | 18 CPU tests. |

Outputs are in `runs_multigroup_robustness/`: `protocol.json`, `grouping_manifests/`, `cache_manifest.json`, `activation_diagnostics.json`, `ppl_results.json`, `reconstruction_results.json`, `summary.json`, `summary.md`, `results/`, `logs/` and `figures/` (PDF + PNG).
