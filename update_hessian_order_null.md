# FP32 Hessian accumulation-order null experiment

**Setup:** Qwen1.5-MoE-A2.7B. Routed experts of layers 1/8/16/23 are GPTQ 4-bit (g128, percdamp 1.0, act-order), B = 128, allocation seed 0, strategies FR s0 / FC s0. The ORIGINAL activation cache, ORIGINAL routing R_O and ORIGINAL act-order P_O are used everywhere. Everything else stays BF16.
**Runner:** `scripts/run_hessian_order_null.sh`
**Artifacts:** `runs_hessian_order_null/`; weights and Hessians are in `~/moe_cache_hessian_order_null` (57 GB).

**Decision: ORDER-SENSITIVE-PPL** (rule frozen in `protocol.json`, sha256 `be5c440f…`, before any shuffled-order Hessian or GPTQ result).

## Question

Can changing *only* the FP32 order in which identical per-sequence Hessian contributions are added change the GPTQ solution and model-level PPL?

Held fixed:
- calibration sequences;
- activations;
- routing;
- act-order and group membership;
- GPTQ hyperparameters;
- evaluation.

## The actual accumulation mechanism (code inspection, unchanged)

**Answers to the Part A questions:**

| question | answer (`moe_calib/profiler.py`, `moe_calib/hessian.py`) |
|---|---|
| accumulation dtype | FP32 (`torch.zeros(K, d, d, float32)` on the GPU) |
| granularity | one contribution per (sequence, expert, projection): the E3 Profiler replays one layer per sequence, batch 1 |
| operation | `Xf = X.float(); outer = Xf.T @ Xf` (one FP32 cuBLAS GEMM), then `H[key][block] += outer` (FP32 elementwise, in place) |
| top-k slots | inside one sequence's X: the rows of all four slots, slot-major (`torch.where` on the expert mask); a token appears at most once per expert |
| "four blocks" | the accumulator's K = 4 *statistical* blocks: block = (i·4)//128 of the sequence's replay **position**. They are not routing slots. |
| normalization | none: weight = 1.0 for FR/FC, unnormalized sums; GPTQ damping is relative to mean diag |
| replay order matters? | yes: the FP32 `+=` sequence inside each block follows the replay order |
| canonicalization? | none. The final `H[key][0..3].sum(dim=0)` is a fixed-order reduction over blocks; nothing sorts the contributions. |
| passed to GPTQ | `H_sum` over the 4 blocks, via `quantize_from_acc` → `quantize_expert` (gate and up share H_gu) |
| replay order recorded | yes: the allocation's `chosen` list (unsorted; FR and FC share their first 32 warm-start sequences) |

**Intervention:**
- The replay order is permuted with `default_rng(seed).permutation(128)` for seeds 20261121–25. FR and FC use the same positional permutation.
- Every sequence **keeps the block of its original position**: an `OrderedForcedProfiler` passes explicit `blocks` to the unchanged `Profiler.run`.
  - So each (expert, block) receives exactly the same contributions, and the cross-block sum is unchanged.
  - Only the FP32 addition order *inside* each 32-sequence block changes.
  - A naive reshuffle with positional blocks would also have repartitioned the blocks.
- This is a genuine sequence-level FP32 addition-order intervention, not row reordering within one GEMM.

## Correctness

**Baseline regression** (original order through the new pathway, act-order override = P_O), FR and FC:
- Hessians are 480/480 bitwise equal to the frozen OO.
- The natural act-order equals P_O for all 480.
- Replay order, positional blocks, routing (every token) and token counts are identical.
- Installed weights and all 4 weight files are bitwise equal to OO.
- Reconstruction (mean and per expert) is identical.
- PPL is identical (FR 9.267851290916612, FC 9.277908190612935), and so is the per-window NLL.

**Hessian preflight invariance** (all 5 orders × 2 strategies):
- routing equals R_O for every token;
- the identical set of contributing sequences per (expert, block) for all 480 keys;
- identical token counts per expert and per block;
- bitwise-identical per-sequence contributions — both the sha256 of X and of `Xf.T @ Xf` — for the 16 diagnostic Hessians (2048 / 2040 sequence contributions);
- no non-finite values.

**Each of the 10 GPTQ configurations:**
- 240 experts / 720 tensors, 0 RTN fallbacks;
- applied permutation equals P_O 240/240;
- regenerated Hessians equal the preflight 480/480;
- integrity I1–I4 pass, and the frozen old tolerance check passes;
- non-target and router parameters are unchanged;
- no inference override; the repeated PPL evaluation is identical.

**Repeatability control** (prespecified: the first configuration whose Hessians differ, FR 20261121), re-run in a fresh process: weights, weight files, reconstruction (mean and per expert), PPL and per-window NLL are all **bitwise identical**.

**Tests:** `tests/test_hessian_accumulation_order.py` (30 tests); full suite 333 passed.

**Protocol note:** the first Hessian-preflight attempt ran out of GPU memory on the third replay of layer 1. The recording wrapper had created a reference cycle that kept each 5.9 GB accumulator alive. No shuffled-order result had been written. The fix was:
- an unbound call, plus `gc.collect()`;
- the original protocol kept as `logs/protocol_superseded_before_any_shuffled_result.json`;
- the protocol refrozen.

The baseline regression is unaffected because that wrapper is not installed in the quantization path.

## Results

### Hessians (480 per strategy and order, vs original order)

| | FR | FC |
|---|---|---|
| Hessians changed | 480/480 in every order | 480/480 in every order |
| elements differing bitwise | 71.3–71.9 % | 73.1–73.6 % |
| median rel. Frobenius difference | 8.4–8.7 × 10⁻⁸ | 9.1–9.3 × 10⁻⁸ |
| max rel. Frobenius difference | 2.1 × 10⁻⁷ | 2.5 × 10⁻⁷ |
| median ULP of changed elements (diagnostic subset) | 2 | 2 |
| max relative trace difference | ≤ 1 × 10⁻⁷ | ≤ 2 × 10⁻⁷ |

- The median relative Frobenius difference is gate/up ≈ 7.9 × 10⁻⁸ and down ≈ 9.6 × 10⁻⁸.
- Max absolute element differences are 0.25–2, on entries of order 10⁴.
- The max-ULP column in the JSON (~1.9 × 10⁹) is dominated by near-zero entries whose sign flips. It is not a meaningful precision measure.
- The order-only perturbation is about 4 × 10⁴ times smaller than the AO Hessian perturbation (trace-normalized differences of about 0.004).

**Diagnostic subset** (16 Hessians):
- The damped-Hessian perturbation is about 9 × 10⁻⁸.
- κ(damped H) ranges from 23 to 1350.
- The upper Cholesky factor of H⁻¹ changes by 3–6 × 10⁻⁷ (relative).

**Act-order before the override:** argsort(diag H) still equals P_O for 362–377 of 480 Hessians. The rest swap near-tied columns: at most 6 × 10⁻⁶ of columns change group. The override makes the order identical by construction.

### FP64 control (FP64 is diagnostic only)

The same FP32 contributions are accumulated per block in FP64 and summed in FP64, for orders 20261121 and 20261122:
- **Every Hessian is bitwise identical across orders**, both in FP64 and after casting to FP32 (0 of 480 change).
- FP32 accumulation changes 72–74 % of elements.

The FP64 sums of FP32 addends were exact here. The FP32 accumulation error relative to FP64 is about 7 × 10⁻⁸, the same size as the order effect.

The FP32 path inside the control reproduced the preflight Hessians bitwise.

### GPTQ weights vs OO (720 tensors)

| | FR | FC |
|---|---|---|
| elements differing | 0.0067–0.0072 % (140–150 k elements) | 0.0072–0.0076 % (150–158 k elements) |
| rel. L2 | 0.00080–0.00082 | 0.00083–0.00086 |
| identical projections | 235–267 / 720 | 216–243 / 720 |
| experts with any change | 229–235 / 240 | 230–233 / 240 |
| experts: Hessian changed but weights identical | 5–11 | 7–10 |

For comparison, under AOF (frozen routing and act-order, alternative activations) 17–18 % of elements differ, at rel. L2 0.041.

**Weight change by Hessian status:**
- Every changed expert had a changed Hessian. That is trivially true, because all Hessians changed.
- A handful of experts absorbed the perturbation completely.
- No monotone relation between Hessian drift and weight drift is visible across orders (Fig 6).

**Quantizer** (8 check experts × 3 projections, read-only capture):
- Zero-points are essentially unchanged: ≤ 7 × 10⁻⁶ of entries differ.
- About 24 % of FP32 group scales differ in some bits. The median relative change is 0; the maximum is 1.1–1.5 %.
- **Rounding decisions** (integer codes) differ in only 4–13 × 10⁻⁶ of elements: 274–914 codes per configuration.
- 88–92 % of the BF16 effective-weight differences in these experts have an *unchanged* integer code. The scale moved by a few FP32 bits, and the BF16 cast exposed it.
- The per-column weight-space error is unchanged (ratio 1.0000 ± 2 × 10⁻⁵).

**Reconstruction:**
- The mean changes by at most 1.3 × 10⁻⁶ (relative).
- Per expert, every expert moves, with a median of 4 × 10⁻⁶ and a maximum of 5 × 10⁻⁵.
- The FC-over-FR penalty is 2.549 % throughout.

### PPL (natural inference routing, frozen windows)

| order | FR PPL | FC PPL | FC − FR |
|---|---|---|---|
| original (OO, reference) | 9.267851 | 9.277908 | +0.010057 |
| 20261121 | 9.267791 | 9.275571 | +0.007779 |
| 20261122 | 9.267245 | 9.279571 | +0.012326 |
| 20261123 | 9.270495 | 9.277078 | +0.006583 |
| 20261124 | 9.271444 | 9.276058 | +0.004614 |
| 20261125 | 9.269601 | 9.278450 | +0.008849 |

| quantity | FR | FC | Δ |
|---|---|---|---|
| SD over 5 orders (ddof 1) | 0.00178 | 0.00166 | 0.00287 |
| mean | 9.26932 | 9.27735 | +0.00803 |
| mean / max \|PPL − OO\| | 0.00173 / 0.00359 | 0.00144 / 0.00234 | — |
| configurations with \|PPL − OO\| > τ = 0.000924 | 3/5 | 3/5 | 5/5 with \|Δ − Δ_OO\| > τ |
| Δ range | | | +0.0046 to +0.0123 |
| sign reversals vs OO | | | 0/5 |

- Per-window NLL (summed over each window's targets) differs in all 128 windows. The mean absolute difference is about 1.0 nat per window, out of about 1140. Mean-NLL SD is about 1.9 × 10⁻⁴.
- The original order is one fixed reference, not a sixth randomized observation.

**Context (different perturbation mechanisms; not pooled):**

| condition | FR SD | FC SD | Δ SD |
|---|---|---|---|
| natural alternative routing (AA) | 0.00442 | 0.00429 | 0.00636 |
| frozen routing (AO) | 0.00402 | 0.00175 | 0.00561 |
| frozen routing + act-order (AOF) | 0.00240 | 0.00240 | 0.00166 |
| **accumulation order only (this study)** | **0.00178** | **0.00166** | **0.00287** |

Relative to AOF, the per-strategy SDs are 0.74× (FR) and 0.69× (FC): of comparable magnitude. The Δ SD is 1.7× AOF's, from five deterministic draws.

## Decision (frozen precedence)

| step | result |
|---|---|
| correctness (baseline, invariance, 10 configs, repeatability) | all pass, no blockers |
| Hessian preflight | 480/480 Hessians changed in all 10 order × strategy sets |
| weight equality | weights differ from OO in 10/10 configurations |
| PPL materiality | max \|PPL − OO\| = 0.00359 > τ = 0.000924 (6/10 configurations exceed it) |

**ORDER-SENSITIVE-PPL:**
- Both strategies are affected.
- The paired Δ changes by more than τ in all five orders, ranging from −0.0054 to +0.0023 relative to Δ_OO.
- The FR/FC ranking never changes sign.

## Answers

1. **Order sensitivity:** yes. The FP32 `+=` inside each block is order-dependent, and nothing canonicalizes it.
2. **Contributions unchanged:** yes, verified bitwise for the diagnostic Hessians (X and Xᵀ X). Block membership and token counts match for all 480.
3. **Hessian difference magnitude:** relative Frobenius ≈ 9 × 10⁻⁸ (a few ULPs), affecting about 72 % of elements.
4. **FP64:** FP64 accumulation removes the order dependence completely in these tests: 0/480 Hessians change.
5. **Effective weights:** yes, but only about 0.007 % of elements change (rel. L2 ≈ 8 × 10⁻⁴), in about 96 % of experts.
6. **Act-order:** identical by construction (P_O, 240/240). Without the override, about 23 % of Hessians would have swapped near-tied columns.
7. **Scales and zero-points:** zero-points are essentially unchanged; about 24 % of scales change in low-order bits, with a median relative change of 0.
8. **Rounding decisions:** about 0.0004–0.0013 % of integer codes. Most BF16 weight differences come from scale bits surviving the BF16 cast, not from code flips.
9. **Reconstruction:** stable to about 10⁻⁶ (relative).
10. **Model-level PPL:** yes. The SD is about 0.0017 per strategy, and the largest deviation is 0.0036.
11. **Paired FC − FR difference:** changes by 0.001–0.005 in every order. It stays positive.
12. **Comparison with the AOF spread:** comparable per strategy (0.70–0.74× AOF's SD).
13. **Is accumulation-order rounding an important mechanism?** It shows that a bit-level Hessian perturbation, about 10⁴ times smaller than the calibration-data perturbations, already produces most of the per-strategy PPL spread seen under AOF. That is consistent with a large part of the residual AOF spread being GPTQ / BF16 discrete sensitivity rather than calibration content. It does **not** show that FP32 summation error caused the earlier batching sensitivity: those Hessians differed by much more than a few ULPs.
14. **Unexplained:**
    - The path from about 150 k changed BF16 elements (mostly scale-bit exposures) to PPL changes of 10⁻³.
    - Which layers or experts carry it.
    - Why Δ's spread is larger here than under AOF.
    - Whether the earlier AO/AOF spreads add to this floor or are bounded by it.
15. **Numerical floor:** a sub-ULP perturbation already moves PPL by up to 0.0036 in this pipeline. FR-versus-FC differences of order 0.01 are therefore only about 3–5 order-SDs above this one mechanism. This does not establish an irreducible floor: one perturbation mechanism, five draws per strategy, one hardware/software stack.

**Keep these levels separate:**
- FP32 accumulation perturbation (≈ 10⁻⁷);
- Hessian numerical error, comparable to FP32 accumulation error vs FP64;
- GPTQ discrete response: a few hundred code flips, plus scale-bit changes;
- BF16 effective weights: 0.007 % of elements;
- reconstruction: ≈ 10⁻⁶;
- PPL: ≈ 2 × 10⁻³.

A tiny changed-weight fraction is not by itself evidence of a negligible PPL effect, and vice versa.

## Limitations

- One allocation seed, five prespecified permutations, one MoE model, four quantized layers.
- FP32 Hessian accumulation (confirmed in code), fixed ORIGINAL BF16 activations, fixed routing, fixed act-order, and one GPTQ configuration.
- Fake-quantized BF16 weights; no packed INT4 measurements.
- One GPU/software stack: no hardware-noise estimate.
- Not a proof of an irreducible numerical floor.
- Not a causal decomposition of PPL variation.
- The FP64 control covers two orders, and FP64 sums of FP32 addends were exact here. That is a property of these magnitudes, not a general invariance.

## Next step (not executed)

**Average out the floor instead of measuring it again.** For FR and FC, quantize from the FP64-accumulated Hessian (cast to FP32 once):
- This is order-invariant by the control above.
- Then measure the FR − FC gap against the *existing* five order-shuffled FP32 configurations, as a reference distribution.

Concretely: 2 new GPTQ + PPL configurations. Check whether Δ_FP64 lies inside the order-induced range +0.0046 to +0.0123. Report it as one deterministic canonical estimate, not as a correction.

That gives a reproducible, order-free baseline for future strategy comparisons. It changes only the accumulation precision, which is outside this study's constraints and needs its own protocol.

## Compute

| item | value |
|---|---|
| baseline regression | 455 s (FC); 1081 s (FR, shared the CPU with the test suite) |
| Hessian preflight | about 5 min per strategy (6 orders × 4 layers) |
| FP64 control | 6 min per strategy |
| per configuration | 434–473 s quantization (Hessians 29–39 s) + 2 × 37–41 s PPL |
| completed configurations | 10 + 1 repeat |
| peak memory | 9.9 GB GPU / 41.7 GB host |
| disk | 57 GB in `~/moe_cache_hessian_order_null` |
