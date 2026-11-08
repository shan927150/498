# Calibration batch-composition sensitivity in MoE GPTQ (four layers, seed 0)

**Decision (rule frozen in `runs_batch_sensitivity/protocol.json` before any quantization): BATCH-SENSITIVITY-DOMINANT**

## Question

The full-depth pilot found that re-grouping the same calibration sequences into different BF16 forward batches changed four-layer GPTQ PPL, and reversed the Full Random (FR) vs Full Coverage-LP (FC) ranking.

The question here: is the FR-vs-FC PPL difference robust to how calibration activations are extracted, or is it the same size as that implementation sensitivity?

This is a numerical-reproducibility study. Nothing about selection, allocation or GPTQ changed.

## Design (frozen before any result)

- **Fixed:**
  - Qwen1.5-MoE-A2.7B; routed experts of layers 1/8/16/23 (240 experts, gate/up/down).
  - Frozen GPTQ: 4-bit, g128, percdamp 1.0, act-order. Fake-quantized BF16 weights; everything else BF16.
  - B = 128, seed 0. The FR s0 / FC s0 selections are re-verified against their source hashes (128 unique each; frozen warm start; no tune or test leakage). Their union is 217 sequences.
  - Non-sequential calibration with E3's Profiler, replaying one sequence at a time.
  - Reconstruction is scored for every variant on the same `~/moe_cache` test rows 2128–2255, using E3's own `eval_cache_dir`.
  - Frozen WikiText-2 windows (128 × 512, 65,408 targets) and frozen evaluation batching.
- **Varied:** only the cache's batch membership.

| variant | construction | batches |
|---|---|---|
| ORIGINAL | `cache_inputs.py`: batches of 8 consecutive original rows from row 0; each needed row in its original batch. This is `~/moe_cache`, and the pilot v2 cache for all 24 layers. | 176 × 8 |
| COMPACT | The failed v1 construction: the 345 needed rows in pilot order (sorted FR ∪ FC, then the 128 test rows), consecutive batches of 8. | 43 × 8 + 1 |
| SHUFFLED | The same 345 rows, permuted once with `numpy.default_rng(20261110)`, then batches of 8. | 43 × 8 + 1 |

- The final batch of 1 in COMPACT is inherited from v1; SHUFFLED uses the same pattern.
- Batch manifests listing every member, its order and the execution order are in `runs_batch_sensitivity/batch_manifests/`.
- After extraction, activations are always reassembled by original sequence id.

## Validity checks (all passed)

- **Model:** every parameter matches the checkpoint digests. Eval mode, no active dropout, BF16, eager attention.
- **Inputs:**
  - Token ids are identical across variants (same sha256 after reassembly).
  - The decoder-layer kwargs (causal mask, positions, rotary) are captured once at batch 1 and broadcast, so they are identical for every batch.
  - There is no padding: every row is a full 512-token chunk.
  - Qwen2-MoE routing has no capacity limit or token dropping.
  - A CPU test in FP64 shows every batch plan gives the same per-sequence activations to 1e-12, so there is no cross-sequence leakage. Only rounding depends on batch composition.
- **Negative control:** the ORIGINAL batches run in reversed execution order reproduce the cache bitwise at 24/24 layers. There is no hidden statefulness or run-to-run nondeterminism; only who shares a batch matters.
- **COMPACT provenance:**
  - The regenerated cache is bitwise equal to the v1 cache at all 24 layers.
  - The FR/FC GPTQ weights and PPLs equal the v1 regression run exactly (sha256 and PPL).
  - So v1 was a pure batch-composition change, not an implementation mismatch.
- **ORIGINAL regression:** in fresh processes, FR and FC reproduce the partial-PPL installed weights (sha256), PPL and reconstruction bitwise. The GPTQ solver is deterministic for an identical cache.
- **All six configurations:**
  - 240 experts each; 0 RTN fallbacks, 0 unchanged projections, 0 non-finite weights.
  - Untouched parameters unchanged.
  - Expert-output checks within the BF16-justified tolerance (≤ 0.90 of it).
  - Repeated PPL evaluation gives |ΔPPL| = 0.

## Results

### PPL (primary)

BF16 PPL is 9.237090.

| cache variant | FR PPL | FC PPL | FC − FR | ranking | interaction Δ(v) − Δ(ORIGINAL) |
|---|---|---|---|---|---|
| ORIGINAL | 9.267851 | 9.277908 | **+0.010057** | FC worse | 0 |
| COMPACT | 9.272002 | 9.268026 | **−0.003975** | FR worse | −0.014032 |
| SHUFFLED | 9.278272 | 9.265841 | **−0.012431** | FR worse | −0.022488 |

- **PPL range across the three groupings:**
  - FR: 0.010421 (1.04 × |Δ_ORIGINAL|).
  - FC: 0.012067 (1.20 ×).
- **Largest change from ORIGINAL:** 0.012067 (1.20 ×). The largest strategy-by-batching interaction is 0.022488 (2.24 ×).
- **Ranking:** FC is worse in 1/3 groupings; the ranking reverses in 2/3.
- **Every alternative-variant PPL change is in the same direction for each strategy:** FR rises in both alternatives (+0.0042, +0.0104) and FC falls in both (−0.0099, −0.0121). With only two deterministic alternatives this is an observation, not a trend.
- **Paired window bootstrap (95 % CI of the FC − FR NLL difference):**

  | variant | 95 % CI |
  |---|---|
  | ORIGINAL | [+0.00020, +0.00200] |
  | COMPACT | [−0.00132, +0.00046] |
  | SHUFFLED | [−0.00227, −0.00038] |

  Both ORIGINAL and SHUFFLED exclude zero, with opposite signs. This bootstrap only measures evaluation-text variability for a fixed pair of quantized models. It does not measure calibration-batch sensitivity, and it would have declared both opposite conclusions "significant".

These are three deterministic groupings, not random replicates. The ranges are observed sensitivity, not variance estimates.

### Reconstruction (same 240 experts, same evaluation rows)

| | FR | FC |
|---|---|---|
| ORIGINAL | 0.0046095 | 0.0047270 |
| COMPACT | 0.0046093 | 0.0047269 |
| SHUFFLED | 0.0046090 | 0.0047267 |

- Mean reconstruction error moves by ≤ 0.012 %.
- Per-expert changes: median about 0.07 %, maximum 0.31 %; about half of the experts get worse and half get better.
- The FC − FR reconstruction gap (+2.5 %) is unchanged in every variant.
- So PPL moves by up to 1.2 × the strategy effect while isolated expert reconstruction is essentially fixed (fig 6).

### Activation drift (217 calibration rows, aligned by sequence id and position)

| layer input | rel. L2 (COMPACT / SHUFFLED) | % elements bitwise equal | p99 token rel. L2 |
|---|---|---|---|
| 0 (embeddings) | 0 / 0 | 100 | 0 |
| 1 | 4.3e-5 / 1.8e-4 | 99.97 / 99.9 | 2e-4 |
| 2 | 1.7e-3 / 1.6e-3 | 70 | 8e-3 |
| 8 | 5.7e-3 | 13 | 8e-2 |
| 16 | 1.5e-2 | 11 | 7e-2 |
| 23 | 2.1e-2 | 10.5 | 7e-2 |

- Differences first appear at the input of layer 1, i.e. after one BF16 decoder layer: rounding from shape-dependent GEMMs on different token batches.
- They grow through depth.
- There are no NaN or Inf values.
- COMPACT and SHUFFLED drift almost identically: any regrouping is equally far from ORIGINAL.

### Routing (Profiler replay of each cache; the model's GPU top-4 op)

The share of calibration tokens whose selected top-4 **set** changes:

| layer | COMPACT | SHUFFLED |
|---|---|---|
| 1 | 0.60 % | 0.57 % |
| 2 | 3.2 % | 3.2 % |
| 8 | 4.0 % | 3.9 % |
| 16 | 3.9 % | 3.9 % |
| 23 | 6.7 % | 6.7 % |

- A further 0.5–7 % of tokens keep the same set but change order. Order matters for routing weights, not for the Hessians.
- Mean top-4 overlap stays at least 0.983, and the layer routing-distribution L1 is ≤ 1.7e-3.
- Even the 4e-5 relative perturbation at layer 1 flips 0.6 % of top-4 sets, because BF16 router logits sit at or near ties.
- Implementation note: a first diagnostic pass ran `topk` on the CPU, which breaks the frequent BF16 ties differently from the GPU kernel. Its routing counts did not match the Profiler's.
  - Superseded output is in `logs/superseded_cpu_topk/`.
  - The final pass uses the GPU op per sequence and matches the Profiler's routed-token counts exactly (64/64).
  - Quantization was never affected, because it always used the model's own routing.

### Hessians (existing E3 Profiler; layers 1/8/16/23 × experts 0, 30; gu and down)

- **Relative Frobenius difference vs ORIGINAL:**
  - Layer 1: 1.5–4.6e-3 (gu); 1.4e-5 to 2.5e-3 (down).
  - Layers 8/16/23: 0.6–5.3e-2.
- **Trace ratios** are within 0.5 % of 1.
- **Routed-token membership changes in every comparison:** from 1–6 tokens of about 4,500 at layer 1 to about 250 at layer 23.
- **Decomposition (gu; validated against the Profiler Hessian to 1.9e-7):**
  - The membership-change term exceeds the common-token activation term in 32/32 comparisons.
  - Layer 1: about 1e-4 (activation) vs 1.5–4.6e-3 (membership).
  - Later layers: about 3–6e-3 vs 0.5–2.9e-2.
- So these differences are **not** mainly floating-point drift of the same tokens. They come mostly from tokens entering or leaving an expert because of near-tie routing flips.
- **Act-order permutation** (argsort of diag H): differs in 64/64 cases. 0.1–21 % of columns change their 128-column quantization group.

### Effective GPTQ weights

| | % of weight elements different | rel. L2 vs ORIGINAL | identical projections |
|---|---|---|---|
| FR, COMPACT / SHUFFLED | 54.0 % / 54.0 % | 0.089 / 0.089 | 0/720 |
| FC, COMPACT / SHUFFLED | 55.1 % / 55.0 % | 0.090 / 0.090 | 0/720 |

- Per layer: 0.074 at layer 1, about 0.09–0.097 elsewhere.
- For scale, the quantization error itself (GPTQ weights vs BF16, FR) is 0.1055 at every layer, and it is the same for ORIGINAL and COMPACT to 4–5 significant figures (`logs/quantization_error_scale.json`).
- So a different calibration grouping gives a different 4-bit rounding solution: about half of all weights land on another grid point, with equal local quality.
- The chain is: activation perturbation → (mostly) routing-membership change in the expert Hessians → act-order permutation and group change → different GPTQ solution → different PPL. The intermediate steps are measured; the step from weights to PPL is observed, not causally isolated.
- Scales and zero-points are not exposed by the frozen solver; the act-order and group change above is the available proxy.

## Answers

1. **Are the same calibration sequences numerically equivalent across groupings?** No. They match exactly in exact arithmetic (FP64 test), but in BF16 every regrouping changes 30–90 % of activation elements from layer 2 on.
2. **Where do differences first appear?** At the input of layer 1, after the first decoder layer. Embeddings are identical.
3. **Does top-4 routing change?** Yes: 0.6 % of tokens at layer 1, 3–7 % at later layers, mostly through near-ties.
4. **Hessian differences: membership or activation?** Mostly routed-token membership change (32/32 gu comparisons), not activation perturbation with unchanged routing.
5. **Do GPTQ weights change substantially?** Yes: about 54 % of elements, relative L2 about 0.09, comparable to the quantization error itself.
6. **Is reconstruction stable?** Yes, to ≤ 0.012 % on average.
7. **Is the FR-vs-FC PPL ordering robust?** No. It reverses in 2/3 groupings.
8. **Is the strategy difference larger than batching sensitivity?** No. Single-strategy sensitivity is 1.0–1.2 × |Δ_ORIGINAL|, and the interaction is up to 2.2 ×.
9. **Implications:**
   - Model-level PPL differences of order 0.01 (≈ 0.1 %) between calibration-selection strategies are not identifiable from one calibration-extraction run, even when the window bootstrap excludes zero.
   - The earlier partial-PPL "FC worse" finding and the full-depth Δ24 should be treated as within implementation sensitivity.
   - Reconstruction-level conclusions under the frozen protocol are unaffected: the FC − FR reconstruction gap is stable across groupings.
   - These numbers apply to this GPU, kernel and software stack and to this model. No claim is made about others.

## Scope and limits

- Two strategies, one seed, three deterministic groupings (exactly one shuffle, chosen before any result). Four layers.
- Fake-quantized weights only; this is not packed INT4 inference.
- The routing and Hessian diagnostics use a prespecified subset (2 experts × 4 layers for the Hessians).

## Files

| file | role |
|---|---|
| `moe_calib/batch_sensitivity.py` | Batch plans, reassembly, drift / routing / Hessian / weight metrics, deltas, decision. |
| `moe_calib/partial_model_quant.py` | `quantize_install(..., eval_cache_dir=)` pass-through to E3's existing option. |
| `diagnostics/batch_sensitivity.py` | Stages: preflight, cache, forward, quant, regression, weights. |
| `diagnostics/report_batch_sensitivity.py` | Report and figures. |
| `scripts/run_batch_sensitivity.sh` | Resumable runner, one fresh process per configuration. |
| `tests/test_batch_sensitivity.py` | 20 CPU tests. |

Outputs are in `runs_batch_sensitivity/`: protocol, batch manifests, cache validation, activation drift, routing disagreement, Hessian sensitivity, weight sensitivity, results, summary, logs, and figures 1–6.

Caches and weights are in `~/moe_cache_batch_sensitivity` (56 GB: two 17.4 GB caches plus six 4-layer weight sets). The v1 evidence cache and logs are untouched.

Compute: per configuration, about 4.6 min to quantize and about 45 s per PPL evaluation (peak 8.4 GB GPU, 44 GB CPU). The cache stage took about 21 min; forward diagnostics took about 19 min.

## One next experiment

Turn this into a measured noise floor before any further strategy comparison:
- Rebuild the four-layer FR s0 and FC s0 GPTQ under a small, prespecified set of additional seeded groupings (e.g. 5 more shuffles).
- Report each strategy's PPL spread and the distribution of FC − FR.
- Selections, settings and windows stay unchanged.
- Only a strategy difference that exceeds this calibration-numerics spread (not just the window bootstrap) should count as a model-level effect.
