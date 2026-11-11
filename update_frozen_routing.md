# Frozen-routing 2×2 factorial ablation of MoE GPTQ calibration (FR s0 / FC s0, four layers, ORIGINAL vs G20261111)

**Decision (rule frozen in `runs_frozen_routing/protocol.json`, sha256 `2353231ba957503f…`, before any hybrid quantization): ROUTING-AND-ACTIVATION-INTERACT**

## One-paragraph answer

The intervention acts on the calibration Hessians only.

- **Hessians.** Changing routing membership while keeping the original activation values (OA) reproduces most of the natural change. The median Hessian difference from OO is 0.0123 for OA, against 0.0141 (FR) and 0.0158 (FC) for AA. Freezing the original routing under the alternative activations (AO) shrinks it to 0.0039 / 0.0049, about 30 % of the natural change.
- **Effective weights.** The GPTQ weights do *not* inherit that suppression. AO still changes 39–40 % of all 4-bit weight elements (relative L2 0.069–0.071, against 0.089–0.090 for AA). GPTQ's rounding is so sensitive that even the activation-only Hessian change re-rounds a large share of weights.
- **PPL.** Frozen routing kept PPL at the OO value in both strategies:
  - FR: AO − OO = −0.0020, against a natural change of +0.0103.
  - FC: AO − OO = −0.00003, against a natural change of −0.0038.
  - It also preserved the FR-vs-FC ranking: Δ_AO = +0.0120 vs Δ_OO = +0.0101.
- **Neither factor alone reproduces the natural PPL change.** FR's large natural jump appears only when both factors change: the interaction is +0.0112 (≈ the whole natural effect), while routing-only adds just +0.0011. For FC, routing-only overshoots (−0.0054 vs −0.0038).
- **Why the rule returned ROUTING-AND-ACTIVATION-INTERACT.** The local-suppression criterion fails on weights (w_AO / w_AA = 0.78 / 0.79 > 0.5), and FR's interaction is large.
- **Caveat.** These are single PPL draws whose size (≈0.001–0.01) is the same order as the grouping-to-grouping noise measured earlier (SD of FC − FR ≈ 0.0069). The PPL pattern is a pilot observation, not an established mechanism.

## Mechanism (from the code, not assumed)

In transformers 4.51.3 `Qwen2MoeSparseMoeBlock.forward`, the router computes softmax (float) followed by `topk(4)`. With `norm_topk_prob=False`, each expert receives `hidden_states[top_x]`, and its output is multiplied by the routing weight *after* the MLP. The E3 Profiler's pre-hooks on `gate_proj` and `down_proj` therefore accumulate **unweighted** sums over routed tokens:
- gate/up: x xᵀ;
- down: z zᵀ, with z = SiLU(gate x) ⊙ up x, computed by the expert in BF16 with its original weights.

The sums are kept in K = 4 blocks (by position in the chosen order) and summed; GPTQ uses relative damping, so the scale of H is irrelevant. Router weights are not part of any Hessian, so none were introduced. A test shows that Hessians are invariant to the gate weights under a fixed assignment.

## Implementation

- **`moe_calib/frozen_routing.py`**
  - `make_forced_forward`: a verbatim copy of the block forward in which `selected_experts` is replaced by a forced `[T, 4]` table, keyed by original sequence ID. If the provider returns the natural tensor, the forward is op-for-op the original.
  - `ForcedProfiler`: the unmodified Profiler, with the override attached only while it runs.
  - `forced_e3`: patches `e3_allocation` so that only the Hessian-building Profiler is forced. The held-out reconstruction Profiler, GPTQ, write-back and the PPL evaluator are untouched.
  - The block output under forced routing is discarded: calibration is non-sequential, and every layer reads its own precomputed pristine-BF16 cache.
  - Also: Hessian, act-order and weight metrics, factorial contrasts, and the frozen decision rule.
- **`diagnostics/frozen_routing.py`:** stages preflight, capture, regression, hybrid.
- **`diagnostics/report_frozen_routing.py`:** report and figures.
- **`scripts/run_frozen_routing.sh`:** stages `preflight | tests | capture | regression | hybrid_quant | report | all`.
- **`tests/test_frozen_routing.py`:** 19 CPU tests:
  - natural forced forward is bitwise equal to the original;
  - capture is the ordered top-k and matches the unmodified Profiler's counts;
  - natural forced Hessians are bitwise equal to the unmodified Profiler's;
  - `expect_natural` mismatches are detected;
  - manual gate/up and down Hessians under forced membership;
  - router weights are not in the Hessians;
  - alignment is by sequence ID, not row;
  - no cross-sequence leakage;
  - caches, router and inference are untouched;
  - reconstruction universe is unchanged;
  - weight-hash and reconstruction regression through E3;
  - GPTQ arguments are preserved and runs don't contaminate each other;
  - act-order, group, Hessian and weight metrics;
  - factorial contrasts and interaction;
  - a synthetic 2×2 where values change with fixed membership, membership changes with fixed values, and the effects interact;
  - decision precedence.

  Full suite: **261 passed**.

## Frozen setup and controls

- **Model and quantization:** Qwen1.5-MoE-A2.7B. Routed experts of layers 1/8/16/23 (240 experts, 720 projections) are GPTQ-quantized: 4-bit, g128, percdamp 1.0, act-order, fake-quantized BF16. B = 128, allocation seed 0.
- **Selections:** FR `ec355fb8d866e4ff` and FC `0353229b6d3090db`, re-verified (128 unique IDs each, frozen warm start, pool membership, no tune/test leakage). Their union is 217 sequences.
- **Activation sources:**
  - ORIGINAL: `~/moe_cache`, row = sequence ID.
  - ALTERNATIVE: G20261111, the first prespecified multigroup seed; row i = union[i]. Both cache hashes were verified.
  - Alignment is by (sequence ID, token position), with 512 tokens per sequence.
- **Routing capture (GPU, the Profiler's batch-1 replay path):** for every layer and both sources, the per-sequence routed-token counts equal those of the **unmodified** Profiler (class forward and its own gate-hook top-k): 217/217 sequences × 4 layers × 2 sources.
  - Stored as int8 ordered top-4 plus fp32 selected-expert probabilities in `routing_{original,alternative}.pt`; hashes are in the JSON files.
  - Tokens whose top-4 **set** differs between R_A and R_O, over the 217 sequences:

    | layer | tokens with a different top-4 set |
    |---|---|
    | 1 | 0.61 % |
    | 8 | 3.89 % |
    | 16 | 3.92 % |
    | 23 | 6.78 % |

    A further 0.5–7 % of tokens keep the same set but in a different order. This only reorders rows inside an expert's sum.
- **Natural-condition regression (all four passed, before any hybrid ran):** FR OO, FR AA, FC OO and FC AA each matched on:
  - routing: forced = natural for every token;
  - **480/480 Hessians bitwise identical** to the unmodified Profiler's, so act-order permutations are identical too;
  - token counts identical;
  - installed-weight sha256 equal to the frozen multigroup reference;
  - mean and per-expert reconstruction exactly equal.

  The PPL references (FR 9.267851 / 9.278182, FC 9.277908 / 9.274113) follow from bitwise weights and the deterministic evaluator, so they were not re-evaluated.
- **Hybrids (fresh processes):**
  - 240 experts and 720 projections changed; 0 RTN fallbacks; 0 non-finite values.
  - Untouched parameters and all 24 router weights unchanged; 8/8 expert-output checks pass.
  - Forced token counts equal the routing table's counts.
  - Evaluation verified free of overrides; repeat PPL |Δ| = 0.

## Results

### Hessians (prespecified subset: layers 1/8/16/23 × experts 0/30 × gate/up, down; trace-normalized relative Frobenius vs OO, median of 16)

| | AO (X_A, R_O) | OA (X_O, R_A) | AA (natural) | AA vs AO | AA vs OA |
|---|---|---|---|---|---|
| FR | 0.0039 | 0.0123 | 0.0141 | 0.0123 | 0.0039 |
| FC | 0.0049 | 0.0123 | 0.0158 | 0.0123 | 0.0049 |

- The routing-membership change is about 3× the activation-value change, for both gate/up and down.
- The two local effects are almost exactly separable:
  - changing activations has the same effect under either routing (AA vs OA = AO vs OO);
  - changing routing has the same effect under either activation source (AA vs AO = OA vs OO).
- Under AO the token counts are identical to OO (0/240 experts change). Under OA, 226 (FR) / 224 (FC) experts change count, by 0.15 % on average and at most 0.8–1.1 %.
- Trace ratios are 1.0000–1.0005.

### Act-order (all 480 Hessians per configuration)

No permutation is identical in any non-OO condition. Mean fraction of columns that move to a different 128-column quantization group:

| | AO | OA | AA |
|---|---|---|---|
| FR | 2.4 % | 6.8 % | 7.3 % |
| FC | 2.6 % | 7.0 % | 7.6 % |

For gate/up specifically: about 3 % (AO) vs about 10–11 % (OA, AA).

### Effective weights (all 720 tensors vs OO)

| | AO | OA | AA |
|---|---|---|---|
| FR: elements differing / relative L2 | 39.2 % / 0.069 | 52.9 % / 0.088 | 54.1 % / 0.089 |
| FC: elements differing / relative L2 | 40.3 % / 0.071 | 53.9 % / 0.089 | 55.1 % / 0.090 |

- No projection is identical in any non-OO condition.
- AA vs AO is ≈ OA vs OO, and AA vs OA is ≈ AO vs OO, mirroring the Hessians.
- Frozen routing removes only about 20 % of the weight change. GPTQ's discrete rounding amplifies a Hessian change about 3× smaller into a weight change only about 20 % smaller.

### Reconstruction (fixed held-out rows, natural routing)

- Every condition is within 0.011 % of OO in mean reconstruction. Per-expert changes are ≤ 0.4 %.
- The FC − FR reconstruction gap is +2.54 % to +2.56 % in all four conditions.
- Reconstruction does not track the PPL changes (fig 6).

### PPL (WikiText-2, 128 × 512 windows, BF16 9.237090)

| | OO | AO | OA | AA |
|---|---|---|---|---|
| FR | 9.267851 | 9.265861 | 9.268983 | 9.278182 |
| FC | 9.277908 | 9.277881 | 9.272497 | 9.274113 |
| Δ = FC − FR | **+0.010057** | **+0.012020** | +0.003514 | **−0.004069** |

**Factorial contrasts (PPL; τ = 0.000924):**

| | AO − OO (act \| R_O) | OA − OO (route \| X_O) | AA − OA (act \| R_A) | AA − AO (route \| X_A) | interaction | natural AA − OO |
|---|---|---|---|---|---|---|
| FR | −0.001990 | +0.001132 | +0.009199 | +0.012321 | **+0.011189** | +0.010331 |
| FC | −0.000028 | −0.005411 | +0.001616 | −0.003768 | +0.001644 | −0.003795 |
| Δ (FC − FR) | +0.001963 | −0.006543 | −0.007583 | −0.016089 | −0.009545 | −0.014126 |

- The mean-NLL contrasts have the same signs. For FR: AO −0.00021, OA +0.00012, interaction +0.00121, natural +0.00111.
- **Window bootstrap** (mean NLL vs OO; conditional on each fixed model pair): only FR AA excludes zero, [+0.00038, +0.00182]. FR AO, FR OA and all FC contrasts include zero.
- **Ranking:**
  - Freezing routing kept FC worse than FR, with a gap close to the original (Δ_AO +0.0120 vs Δ_OO +0.0101).
  - Routing-only moved the gap two-thirds of the way toward the alternative (Δ_OA +0.0035), but did not reverse it.
  - The reversal appears only with both factors changed (Δ_AA −0.0041).

### Decision path (frozen rule)

- `local_suppression` is **False** for both strategies: h_AO / h_AA = 0.28 / 0.31 passes, but w_AO / w_AA = 0.78 / 0.79 fails.
- `local_routing_dominant` is True for both.
- `ppl_suppression` is True for both: |AO − OO| ≤ 0.5 |AA − OO|.
- `ranking_preserved` is True.
- `interaction_large` is True for FR (|I| = 0.0112 ≥ 0.5 × max(|A_R0|, |R_X0|)) and False for FC.

Applying the precedence:
1. SUPPRESSES fails, because there is no local weight suppression and FR's interaction is large.
2. ACTIVATION-VALUES-DOMINATE fails.
3. LOCAL-ROUTING-EFFECT-WITHOUT-PPL-EXPLANATION fails, because PPL suppression *and* ranking preservation hold.
4. The result is therefore **ROUTING-AND-ACTIVATION-INTERACT**.

## Interpretation

1. **Routing membership is the larger driver of the Hessian change.** About 78–87 % of the natural trace-normalized Hessian change (0.0123 vs 0.0158 / 0.0141) and about 92 % of the act-order group changes are reproduced by swapping assignments alone. Holding membership fixed leaves only about 30 %. This upgrades the earlier observational decomposition to an intervention, at the Hessian level.
2. **GPTQ largely erases that distinction at the weight level.** Any Hessian perturbation of this size re-rounds 40–55 % of the 4-bit elements. So "frozen routing" does not give stable weights.
3. **The model-level picture is non-additive.**
   - FR: neither single factor moves PPL much (−0.0020, +0.0011), but together they give +0.0103.
   - FC: the routing change alone already gives the full natural change.
   - With one draw per cell and effects the size of grouping noise, these PPL contrasts could be chance re-roundings rather than mechanism. They should not be read as routing "explaining" or "not explaining" the ranking reversal.

## Limitations

- **What was intervened on:** this is an intervention on the calibration statistics, not the model's routing. Inference routing, router weights, layer caches and evaluation are unchanged.
- **Upstream drift is not separated:** X_O and X_A come from different natural BF16 forward trajectories, and those already differ partly because of routing in earlier layers. Upstream activation drift is not independently manipulated.
- **Scope:** one allocation seed, one alternative grouping, two strategies, four layers, one draw per cell. The PPL effects (≤ 0.012) are of the order of the grouping SD measured before. The local Hessian and weight comparisons are far better determined than the PPL contrasts.
- **Packed INT4:** fake-quantized weights only; no packed INT4 claim.
- **Not a full mediation result:** this does not prove routing membership is *the* mechanism of batching sensitivity.

## One recommended follow-up (not executed)

Replicate only the AO (frozen original routing) condition for FR and FC under the other four prespecified groupings, G20261112–15. That is 8 configurations, compared against their existing AA results.

If frozen routing really stabilizes PPL, the AO PPLs should cluster near OO (9.2679 / 9.2779) with a spread clearly below the AA spread (0.010). If they spread as widely as AA, the AO ≈ OO match observed here was chance.

## Compute

- **Routing capture:** 102 s.
- **Natural regressions:** 271–284 s quantization each, plus a 34 s reference Profiler pass.
- **Hybrids:** 282–300 s quantization each, plus 2 × 36–44 s PPL evaluation.
- **Peak memory:** 8.4 GB GPU, 42.7 GB host.
- **Disk (`~/moe_cache_frozen_routing`):** 16 GB hybrid weights, 1.6 GB Hessians (full matrices for the diagnostic subset; diagonals and permutations for all 480).
- **Earlier artifacts:** unchanged. The natural weights are read from `~/moe_cache_multigroup_robustness/weights`.

## Files

Outputs are in `runs_frozen_routing/`:
- `protocol.json`, `selected_sets.json`
- `routing_original.{json,pt}`, `routing_alternative.{json,pt}`, `routing_validation.json`
- `natural_regression.json`, `hessian_comparison.json`, `weight_comparison.json`
- `ppl_results.json`, `reconstruction_results.json`, `factorial_contrasts.json`
- `summary.json`, `summary.md`, `results/`, `logs/`
- `figures/` (PDF + PNG):
  - fig1 Hessian factorial
  - fig2 weight factorial
  - fig3 PPL factorial
  - fig4 factorial effects
  - fig5 ranking
  - fig6 reconstruction vs PPL
