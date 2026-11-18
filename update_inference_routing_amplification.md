# Inference-time routing amplification MVP

**Verdict: `ROUTING-AMPLIFICATION-SUPPORTED` (conclusion A).** We forced every MoE layer to use the BF16 reference
top-4 expert indices. Router logits and probabilities were still computed live from the quantized model's own hidden
states. Under this intervention, the spread of the paired difference across the five FP32 accumulation orders fell
from SD 0.002870 to SD 0.001032:

- **R_SD = 0.359**, against a frozen threshold of ≤ 0.5.
- Δ = PPL(FC) − PPL(FR).

All mandatory parity checks pass **bitwise**. Every natural evaluation reproduces the historical per-window NLL
bitwise. No weights were regenerated and no GPTQ solve was run.

The thresholds, the intervention, the primary metric and the parity requirements were frozen in
`runs_inference_routing_amplification/protocol.json` (sha256 `79849e02…`) before any new routing or PPL result
existed.

**Scope.**
- **Model:** Qwen1.5-MoE-A2.7B. Only the routed experts of layers 1/8/16/23 are fake-quantized (BF16 effective
  weights; GPTQ 4-bit, g128, percdamp 1.0, act-order).
- **Configurations:**
  - 10 primary: FR/FC seed 0 × FP32 accumulation orders 20261121–20261125.
  - 4 descriptive references: ORIGINAL FP32 FR/FC and FP64-canonical FR/FC. These are excluded from every
    five-order statistic.
- **Evaluation:** frozen WikiText-2 windows, 128 × 512, 65,408 targets, batch 8, using the historical sequential
  24-layer evaluator.
- **Hardware:** one A100 MIG 2g.20gb slice.

## 1. Audit (`artifact_inventory.json`, `weight_manifest.json`)

All 14 weight sets passed the audit:

- **Files:** each L{1,8,16,23}.pt file was re-hashed, and the file and layer sha256 values match the historical records.
- **Combined digest:** the combined installed-weights digest matches each historical result.
- **Tensors:** all 240 experts and 720 tensors are present per set. Every tensor is BF16, correctly shaped and finite.
- **No BF16 fall-back:** all 720 tensors differ from the BF16 checkpoint, so none silently fell back to BF16.
- **Settings and selections:** the GPTQ settings match the history, and the selection rows equal the seed-0
  FR/FC selections. For order runs only the replay order is permuted.
- **ORIGINAL weights:** taken from `CTRL_{FR,FC}_ORIGINAL`. These are byte-identical to the order-null `*_O0`
  regression weights.

## 2. Implementation (opt-in, diagnostic only)

**`moe_calib/inference_routing_override.py`** has two modes:

- **Recording:** a forward hook on each `mlp.gate` takes the router logits and recomputes the block's own
  `softmax(fp32) → topk(4)`. It stores the ordered indices (int16), the top-4 probabilities and the 5th probability.
  The block forward is untouched.
- **Forcing:** this reuses the existing verbatim class-forward copy `moe_calib.frozen_routing.make_forced_forward`,
  which substitutes only `selected_experts`.
  - The routing weights are the **current** softmax probabilities gathered at the forced indices.
  - No renormalization is applied, because `norm_topk_prob=False`.
  - Dispatch is the original expert loop with `index_add_`, plus the shared expert.
  - Hooks and overrides are removed on exit.

**Token mapping:** row r of evaluation batch bi maps to window 8·bi + r//512, position r%512.

**Evaluator:** the historical `EP.layerwise_nll` call sequence, extended to return per-token NLL. Its per-window sums
are bitwise those of `EP.nll_from_logits`.

**Unchanged:** the production GPTQ code and the historical evaluator's defaults were not modified.

## 3. Mandatory parity (`routing_override_validation.json`)

All tests below hold bitwise. "Layer hashes" means sha256 of the hidden states after each of the 24 layers.

| test | config | result |
|---|---|---|
| A: record-only vs no-wrapper | BF16 | per-window NLL bitwise; 24 per-layer hashes identical; PPL 9.237089824789335 = historical |
| A: record-only vs no-wrapper | FR_O20261121 (prespecified) | per-window bitwise; layer hashes identical; = historical order-null result |
| B: BF16 forced with its own indices | BF16 | per-window and per-token NLL, 24 layer hashes and final hidden states identical; dispatch counts identical |
| C: quantized model forced with its own natural indices | FR_O20261121 | same selections, router probabilities, layer and final hidden states, per-token NLL and PPL |

**Additional checks:**
- Repeated BF16 recording gives identical indices and probabilities.
- At the end of the process:
  - all 3,964 non-target parameters are unchanged;
  - the router weights are unchanged;
  - the target experts are back to pristine;
  - BF16 re-evaluates bitwise.
- In every frozen run, the dispatched token count per expert equals the BF16 reference exactly.
- Wherever the forced slot equals the model's own slot, the forced probability equals the model's own top-k value.

## 4. Natural routing divergence (`routing_divergence.json`, Fig. 1, Fig. 4)

### Against BF16

Each quantized configuration changes the top-4 **set** for about 8.8–9.0 % of (layer, token) slots, and 82 % of
tokens are affected in at least one layer. Pure slot-order changes (about 145k per configuration) are counted
separately and excluded.

- **Layers 0 and 1 are identical to BF16 in every configuration**, as expected: the layer-1 router acts before the
  layer-1 experts.
- Divergence first appears at layer 2 (about 14 %).
- It then declines to about 7 % around layer 15 and rises again to about 11–12 % in layers 19–23.

This divergence comes almost entirely from **quantization itself**. The five orders, ORIGINAL and FP64 overlap to
within about 0.2 percentage points.

### Order-induced divergence (order variant vs ORIGINAL FP32, same strategy)

- 3.65–3.71 % of (layer, token) slots change their top-4 set, from weight perturbations that alter only about
  0.007 % of weight elements.
- This is again zero in layers 0–1. It is 1.7–1.9 % at layer 2, reaches a maximum of about 6 % at layers 21–22, and
  roughly 48 % of scored positions have a local set flip.
- Almost all changed tokens swap exactly one expert (the 99th percentile is 2).

### Near-ties

Changed sets are strongly concentrated at small top-4/top-5 margins:
- the median BF16 margin is 7.1e-4 for changed sets versus 5.4e-3 for stable ones;
- 35 % of tokens in the lowest BF16-margin decile change their set, versus less than 0.1 % in the top decile.

**4.9 % of BF16 (layer, token) router decisions are exact ties (p4 = p5)**, because the router logits are BF16.
About 20 % of the changed sets sit on such exact ties, where any perturbation can flip the selection.

## 5. Frozen-routing PPL (`natural_ppl_results.json`, `frozen_ppl_results.json`, `paired_order_comparisons.json`, `variance_comparison.json`)

| order | FR natural | FR frozen | FC natural | FC frozen | Δ natural | Δ frozen | δ_g = Δnat − Δfro |
|---|---|---|---|---|---|---|---|
| 20261121 | 9.267791 | 9.268410 | 9.275571 | 9.278260 | +0.007779 | +0.009850 | −0.002070 |
| 20261122 | 9.267245 | 9.269271 | 9.279571 | 9.276344 | +0.012326 | +0.007073 | +0.005253 |
| 20261123 | 9.270495 | 9.270318 | 9.277078 | 9.278128 | +0.006583 | +0.007810 | −0.001227 |
| 20261124 | 9.271444 | 9.269121 | 9.276058 | 9.277549 | +0.004614 | +0.008428 | −0.003814 |
| 20261125 | 9.269601 | 9.269958 | 9.278450 | 9.277910 | +0.008849 | +0.007952 | +0.000897 |

| statistic | natural | frozen |
|---|---|---|
| SD(Δ) (ddof 1) | 0.002870 (= historical, recomputed) | 0.001032 |
| **R_SD** | | **0.359** |
| SD PPL FR | 0.001776 | 0.000746 |
| SD PPL FC | 0.001662 | 0.000772 |
| mean Δ | +0.008030 | +0.008222 |
| range Δ | 0.004614 – 0.012326 | 0.007073 – 0.009850 |
| FR better (Δ > 0) | 5/5 | 5/5 |

**Effect of freezing:**
- It roughly halves the order spread of FR and of FC separately, and cuts the paired spread by about 64 %.
- The mean paired difference barely moves (+0.00019).
- The FR-better ranking is unchanged in all five orders.
- Freezing raises mean PPL slightly (FR +0.00010, FC +0.00029) because it makes off-policy choices on about 9 % of
  slots.
- At the window level, frozen and natural are close to a coin flip: 44–60 % of windows improve per configuration.

The δ_g values are not consistent in sign, so freezing does not shift FC − FR in one direction. It removes
order-specific scatter.

Across configurations, the amount of BF16 disagreement does not track the natural-minus-frozen PPL change
(descriptive Spearman −0.30, n = 10). Disagreement rates are nearly identical across configurations, so this was not
expected to be informative.

## 6. ORIGINAL and FP64 canonical references (`canonical_reference_comparison.json`; descriptive only)

| reference | FR nat | FR frozen | FC nat | FC frozen | Δ nat | Δ frozen |
|---|---|---|---|---|---|---|
| ORIGINAL FP32 | 9.267851 | 9.270037 | 9.277908 | 9.278776 | +0.010057 | +0.008739 |
| FP64 canonical | 9.271421 | 9.268393 | 9.278551 | 9.277924 | +0.007129 | +0.009531 |

- Both natural values reproduce the historical results bitwise.
- Under frozen routing both references keep FR better, and both fall inside the frozen five-order range.
- The ORIGINAL − FP64 gap in Δ shrinks from +0.00293 (natural) to −0.00079 (frozen).
- The canonical conclusion (FR better by about 0.007–0.010 PPL) does not depend on downstream discrete routing.
- The specific canonical point value does move by about as much as the natural order spread.
- The canonical estimand is **not** redefined by these frozen results.

## 7. Token-level NLL (`token_nll_concentration.json`, `routing_nll_association.json`, Fig. 5, Fig. 6)

### Analysis A: each order variant vs same-strategy ORIGINAL, mean over the 10 configurations

| | mean abs ΔNLL | median | p99 | max | top 1 % share | top 10 % share |
|---|---|---|---|---|---|---|
| natural routing | 0.0319 | 0.0134 | 0.198 | 1.61 | 9.0 % | 42.0 % |
| frozen routing (both sides) | 0.0243 | 0.0092 | 0.126 | 0.48 | 6.7 % | 41.0 % |

- About 8.7k of 65.4k tokens carry half of the total absolute change.
- Changes are moderately concentrated, but not in a tiny subset: the top 1 % of tokens carries about 9 %.
- Freezing routing removes about 24 % of the total absolute token-level change and most of the extreme tail (the
  maximum falls from 1.6 to 0.5 nats).
- The rest is continuous propagation of the weight perturbation.
- These are absolute changes, not additive contributions to the signed PPL difference.

### Analysis B: local flips (events = set flips vs the ORIGINAL routing at the logit's own position)

- Positions with a local flip have about 1.23–1.26× larger mean |ΔNLL| than positions without one (0.035 vs 0.029).
- 79–88 % of the top-1 % |ΔNLL| tokens have a local flip, compared with 48 % (vs ORIGINAL) or 82 % (vs BF16) of all
  tokens.
- Measured against BF16, the association is roughly flat.

### Analysis C: causal-prefix flips

These are uninformative: 99.6 % of targets already have at least one order-induced flip earlier in the window, so
the comparison groups are tiny (n ≈ 150).

Analyses B and C are descriptive associations. The causal evidence is the intervention in §5.

## 8. Cost (`cost_results.json`)

| item | value |
|---|---|
| model load | 31 s |
| BF16 reference recording | 73 s |
| natural eval, median | about 35 s |
| frozen eval, median | about 54 s |
| install per configuration (NFS read and sha) | 56–114 s |
| peak GPU | 3.3 GB |
| peak CPU RSS | 50 GB |
| run storage | 1.67 GB (in `~/moe_cache_inference_routing`) |
| BF16 reference file | 45 MB |
| GPTQ solves | 0 |
| evaluation forwards | 37 |

Most primary frozen runs overlapped a 32-minute CPU pytest run. Natural and frozen wall times are therefore not a
clean overhead measurement, and the recorder overhead is not resolvable from them.

## 9. Answers to the research questions

1. **Do order-null variants route differently during natural inference?** Yes. About 3.7 % of (layer, token) top-4
   sets differ from the ORIGINAL FP32 model, and about 8.9 % differ from BF16 (mostly because of quantization
   itself).
2. **Where does disagreement first appear?** At layer 2, the first router downstream of a quantized expert block.
   Layers 0–1 are identical, as predicted.
3. **How does it change with depth?** It does not grow monotonically.
   - Against BF16: a peak at layer 2, a dip to about 7 % near layer 15, and a rise to 11–12 % at layers 19–23.
   - Order-induced: it rises from 1.7 % at layer 2 to about 6 % at layers 21–22.
4. **Are changes concentrated near small margins?** Yes, strongly. That includes 4.9 % exact BF16 ties.
5. **Does freezing reduce FR PPL variability?** Yes: SD 0.00178 → 0.00075.
6. **Does it reduce FC PPL variability?** Yes: SD 0.00166 → 0.00077.
7. **Does it reduce paired FC − FR variability?** Yes: SD 0.00287 → 0.00103, R_SD = 0.359.
8. **Is the method ranking stable under frozen routing?** Yes. FR is better in 5/5 orders under both regimes.
9. **Do ORIGINAL and FP64 behave similarly?** Their sign is preserved and their frozen Δ falls inside the frozen
   range. Their gap shrinks.
10. **Are large token-level changes associated with local flips?** Moderately: about 1.25× the mean |ΔNLL|, and the
    tail is enriched for local flips.
11. **Are they associated with earlier flips in the causal context?** This cannot be resolved here: almost every
    position has prefix flips.
12. **Are NLL changes concentrated in a few tokens?** Moderately. The top 10 % of tokens carry about 42 %; the top
    1 % carries about 9 %.
13. **Is there evidence for discrete routing amplification?** Yes. This is the intervention result, in this model,
    regime and stack.
14. **What remains after freezing routing?** SD(Δ) 0.00103 remains. This is continuous propagation of the order-induced
    weight perturbations through hidden states (and through router probabilities, which stay live), plus BF16
    arithmetic.
15. **Does this strengthen the Numerical Reproducibility contribution?** Yes. It identifies a concrete MoE-specific
    amplifier:
    - the perturbation is about 9e-8 relative in the Hessian;
    - this produces top-4 flips on about 3.7 % of layer-token slots, concentrated at near-ties and exact BF16 ties;
    - those flips account for most of the order-induced paired PPL spread.

**Conclusion A: routing amplification supported.** Freezing top-k indices materially reduces paired PPL variability.

## Limitations

- The five orders are deterministic variants of one seed's calibration. R_SD is a screening ratio, not a test. The SD
  difference is not an additive variance decomposition.
- The intervention is counterfactual and off-policy on about 9 % of slots. It freezes only the routing support: the
  router probabilities stay dynamic.
- The evaluation windows were inspected in earlier studies.
- The quantization is partial-model BF16 fake quantization, not packed INT4.
- The results are specific to this model, its hardware and its numerical stack.
- No claim is made about an irreducible noise floor.

## Files

- **Code:**
  - `moe_calib/inference_routing_override.py`
  - `diagnostics/inference_routing_amplification.py` (stages audit / protocol / gpu / analyze)
  - `diagnostics/report_inference_routing_amplification.py` (analysis and decision)
  - `diagnostics/report_inference_routing_figures.py`
  - `tests/test_inference_routing_override.py`
- **Outputs:** `runs_inference_routing_amplification/` (all required JSONs, `summary.md`, `figures/fig1`–`fig6`,
  `logs/`).
- **Large tensors:** `~/moe_cache_inference_routing/`, holding the BF16 reference routing, per-run routing records and
  token NLL.
