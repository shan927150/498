# Grouping × routing amplification: frozen-routing intervention on calibration batch grouping (Family D)

**Verdict: `WEAK-COLLAPSE (B)`.** We applied the Family A routing intervention unchanged
(`update_inference_routing_amplification.md`): every MoE layer is forced to use the BF16 reference top-4 expert
indices. This barely changes how much calibration batch grouping moves the paired FR-vs-FC difference:

- **R_SD = 0.915**, about an 8.5 % reduction. The pooled within-seed SD(Δ) goes from 0.006872 to 0.006286.
- Seed 0 alone shows **no** reduction (ratio 1.057).
- R1 ranking reversals go from **6/15 to 8/15**.
- Δ = PPL(FC) − PPL(FR); positive means FR is better.

Family A found that accumulation-order variability was mostly mediated by discrete routing flips (R_SD 0.359).
Grouping variability is **not**: it persists with routing held fixed, so it travels through the quantized expert
weights themselves.

All mandatory parity checks (A–F) pass **bitwise**. All 30 natural evaluations reproduce the historical per-window
NLL bitwise. No GPTQ solve was run.

The decision rule, primary metric, intervention and parity requirements were frozen in
`runs_grouping_routing_amplification/protocol.json` (sha256 `ca444093eb0bf1a7…`, created 2026-11-14 05:47) before
any frozen-routing result existed. These thresholds were:

| verdict | rule |
|---|---|
| STRONG-COLLAPSE (A) | R_SD ≤ 0.5 **and** frozen R1 ≤ ½ natural R1 |
| WEAK-COLLAPSE (B) | R_SD ≥ 0.9 **and** frozen R1 > ½ natural R1 |
| PARTIAL-COLLAPSE (C) | otherwise |
| BLOCKED | any parity check fails |

**Scope.**
- **Model and quantization:** the same as Family A. Qwen1.5-MoE-A2.7B, with only the routed experts of layers
  1/8/16/23 fake-quantized (GPTQ 4-bit, g128, percdamp 1.0, act-order, BF16 effective weights).
- **Configurations:** 30 = 3 allocation seeds {0, 1, 2} × 5 calibration groupings G20261111–15 × {FR, FC}.
  - These are the existing weight sets from `runs_multigroup_robustness` (seed 0) and `runs_cross_seed_grouping`
    (seeds 1/2). They were re-audited, not regenerated.
  - Groupings are nested within seeds, so the statistics are pooled within seed rather than over 15 flat samples.
- **Evaluation:** frozen WikiText-2 windows, 128 × 512, 65,408 targets, batch 8, using the historical sequential
  24-layer evaluator.
- **Reference routing:** reused from Family A (`~/moe_cache_inference_routing/bf16_reference_routing.pt`); its sha
  was re-verified, not regenerated.

## 1. Audit (`artifact_inventory.json`, `weight_manifest.json`, `logs/audit.log`)

All 30 weight sets passed the audit (30/30 valid, no duplicates, none missing). The checks were per-file and per-layer
sha256, 240 experts and 720 BF16 tensors per set, finite values, and no tensor equal to the BF16 checkpoint.

- The historical Δ table is reproduced from the stored result files for 15/15 pairs.
- `FR_s0_G20261111_CTRL_repeat` is excluded; it is a determinism control with identical weights.
- **References:**
  - The seed-0 ORIGINAL-extraction pair is the Family A `FR/FC_ORIGINAL` natural and frozen runs, reused.
  - The seed-1/2 ORIGINAL-extraction weights were never saved (`runs_partial_ppl` quantized in-process). They have
    only a historical natural PPL, so their frozen Δ cannot be evaluated.

## 2. Parity and correctness (`routing_override_validation.json`, `correctness_results.json`)

Everything below holds exactly; no tolerance was loosened. The parity configurations were prespecified:
`FR_s0_G20261111` and `FC_s2_G20261111`.

| test | scope | result |
|---|---|---|
| A: natural parity | all 30 | per-window NLL bitwise = historical |
| B: record-only vs plain evaluator | 2 parity configs | per-window bitwise; 24 layer hashes and final hidden state identical |
| C: self-forcing with own natural indices | 2 parity configs | NLL per window and per token, layer and final hashes, selections, router probabilities, dispatch, and PPL all identical; 16/16 forced calls per layer |
| D: BF16 self-forcing with the reference | BF16 | bitwise identical to plain and historical BF16 |
| E: frozen dispatch | all 30 | dispatch counts = BF16 reference; probabilities finite and model-dependent (~92 % of forced probabilities differ from BF16) |
| F: restoration | all 30 | installed digest unchanged during evaluation, routers unchanged, targets restored to pristine, no hooks left |

At the end of the process:
- all 3,964 non-target parameters are unchanged;
- the router weights are unchanged;
- no hooks remain;
- BF16 re-evaluates bitwise.

Per-token NLL sums reproduce the per-window NLL.

**Tests:** the targeted tests pass (`test_grouping_routing_amplification`, `test_inference_routing_override`,
`test_frozen_routing`, `test_partial_ppl`: 47 passed). The full suite was not re-run for this study.

**Family A artifacts:** all 14 Family A result files re-hash to their inventory sha256, the reference routing file is
unchanged, and no file in `runs_inference_routing_amplification/` was modified after this protocol was frozen. These
checks are recorded in `logs/pytest_results.json`.

## 3. Paired PPL (`natural_ppl_results.json`, `frozen_ppl_results.json`, `paired_grouping_comparisons.json`, Fig. 1)

| comparison | FR nat | FC nat | FR frozen | FC frozen | Δ nat | Δ frozen | δ_g = Δnat − Δfro |
|---|---|---|---|---|---|---|---|
| s0_G20261111 | 9.278182 | 9.274113 | 9.274681 | 9.275402 | −0.004069 | +0.000721 | −0.004790 |
| s0_G20261112 | 9.268159 | 9.275760 | 9.267316 | 9.276877 | +0.007601 | +0.009561 | −0.001960 |
| s0_G20261113 | 9.271035 | 9.265146 | 9.272412 | 9.266228 | −0.005889 | −0.006183 | +0.000295 |
| s0_G20261114 | 9.278102 | 9.269707 | 9.276955 | 9.269850 | −0.008394 | −0.007106 | −0.001289 |
| s0_G20261115 | 9.272717 | 9.273808 | 9.271482 | 9.272560 | +0.001091 | +0.001078 | +0.000013 |
| s1_G20261111 | 9.266099 | 9.265572 | 9.268815 | 9.267153 | −0.000527 | −0.001662 | +0.001135 |
| s1_G20261112 | 9.267534 | 9.263857 | 9.265103 | 9.266850 | −0.003677 | +0.001747 | −0.005424 |
| s1_G20261113 | 9.265318 | 9.268279 | 9.262063 | 9.270033 | +0.002961 | +0.007970 | −0.005009 |
| s1_G20261114 | 9.267026 | 9.275531 | 9.268506 | 9.277161 | +0.008505 | +0.008655 | −0.000150 |
| s1_G20261115 | 9.273307 | 9.269481 | 9.272534 | 9.266885 | −0.003826 | −0.005650 | +0.001824 |
| s2_G20261111 | 9.262634 | 9.272568 | 9.263155 | 9.270934 | +0.009934 | +0.007779 | +0.002155 |
| s2_G20261112 | 9.268870 | 9.269566 | 9.266980 | 9.266631 | +0.000696 | −0.000349 | +0.001045 |
| s2_G20261113 | 9.282398 | 9.268849 | 9.276153 | 9.267183 | −0.013549 | −0.008970 | −0.004578 |
| s2_G20261114 | 9.271457 | 9.266417 | 9.267137 | 9.265977 | −0.005041 | −0.001160 | −0.003880 |
| s2_G20261115 | 9.272849 | 9.268318 | 9.271092 | 9.269454 | −0.004531 | −0.001638 | −0.002893 |

The seed-0 ORIGINAL reference (Family A, not included in any SD) has Δ = +0.010057 natural and +0.008739 frozen.

**Window-level CIs:**
- Windows whose CI excludes zero: FR 2 natural → 4 frozen; FC 2 → 2.
- Mean CI width falls from 0.001904 to 0.001593 (mean-NLL units).
- These CIs condition on a single weight pair. They measure text-sampling uncertainty, not grouping variability.

## 4. Variability (`variance_comparison.json`, Fig. 2)

| statistic | natural | frozen | frozen / natural |
|---|---|---|---|
| **pooled within-seed SD(Δ)** (primary) | 0.006872 (= historical) | 0.006286 | **0.915** |
| seed 0 SD(Δ) | 0.006364 | 0.006728 | 1.057 |
| seed 1 SD(Δ) | 0.005176 | 0.006158 | 1.190 |
| seed 2 SD(Δ) | 0.008624 | 0.005945 | 0.689 |
| flat SD over 15 (descriptive) | 0.006522 | 0.005986 | 0.918 |
| pooled within-seed SD PPL FR | 0.005198 | 0.004214 | 0.811 |
| pooled within-seed SD PPL FC | 0.003812 | 0.003754 | 0.985 |
| mean Δ (mean of seed means) | −0.001248 | +0.000319 | |
| seed means | −0.00193, +0.00069, −0.00250 | −0.00039, +0.00221, −0.00087 | |
| range Δ | −0.01355 … +0.00993 | −0.00897 … +0.00956 | |
| FR better / FC better | 6 / 9 | 7 / 8 | |
| hierarchical bootstrap 95 % CI of mean Δ | [−0.00478, +0.00215] | [−0.00283, +0.00361] | |
| historical decision rule | GROUPING-VARIATION-DOMINATES | GROUPING-VARIATION-DOMINATES | |

**What the variability numbers show:**
- The per-seed ratios scatter around 1 (0.69–1.19), so freezing gives no consistent reduction.
- The only noticeable contraction is in FR's own PPL spread (0.81), and it does not carry over to the paired
  difference.
- δ_g has mean −0.00157 and SD 0.00269. Freezing therefore shifts individual Δ values by about 40 % of the grouping SD
  but in no consistent direction, and leaves the spread intact.

## 5. Ranking reversals (`ranking_reversal_analysis.json`, Fig. 3)

**R1** uses the historical decision-rule definition: a reversal is a Δ whose sign is opposite to the regime's overall
mean M.
- Natural: 6/15 (M = −0.00125). Frozen: 8/15 (M = +0.00032).
- Five of the six natural R1 reversals stay on the side opposite the natural M after freezing.
- Because both means are close to zero, R1 is fragile: the sign of M itself flips between regimes.

**R2** compares each Δ's sign with the same seed's ORIGINAL-extraction Δ sign, as in the FP64 mechanism table.
- 9/15 natural → 8/15 frozen.
- Seed 0 is fully frozen on both sides: 3/5 → 2/5.
- Seeds 1/2 compare frozen Δ against the natural ORIGINAL sign, a mixed regime.
- The seven comparisons that remain reversed are s0_G20261113, s0_G20261114, s1_G20261111, s1_G20261115,
  s2_G20261113, s2_G20261114 and s2_G20261115.

**3/15** comparisons change sign under freezing: s0_G20261111, s1_G20261112 and s2_G20261112.

## 6. Routing divergence (`routing_divergence.json`, Fig. 4)

### Against BF16

- Each quantized configuration changes the top-4 **set** for 8.77–9.01 % of (layer, token) slots, a mean of
  8.92 %. 81.7–82.8 % of tokens are affected in at least one layer.
- This is the same level as Family A (about 8.8–9.0 %), so it again comes from quantization itself.
- Layers 0–1 are identical to BF16 in all 30 configurations.

### Grouping-induced divergence

| layer | 2 | 4 | 7 | 9 | 12 | 15 | 18 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|
| D: s0 grouping vs ORIGINAL (%) | 5.92 | 6.00 | 4.39 | 7.17 | 6.19 | 5.23 | 7.95 | 9.35 | 9.40 | 8.83 |
| D: grouping vs grouping, within seed (%) | 6.00 | 6.02 | 4.43 | 7.16 | 6.12 | 5.30 | 8.05 | 9.39 | 9.40 | 8.88 |
| A: order vs ORIGINAL (%) | 1.76 | 3.59 | 3.13 | 3.68 | 3.72 | 3.44 | 4.61 | 5.84 | 6.04 | 5.88 |

- **Overall:** a grouping change flips about 6.1 % of (layer, token) sets, against Family A's 3.7 %. This holds both
  against ORIGINAL (6.09 % vs 3.67 %) and between variants (6.10 % vs 3.68 %).
- **Depth profile:** the profile is already high at layer 2 (about 6 % vs 1.8 %). It dips around layers 7–8 and rises
  to about 9.4 % at layers 21–22.
- **Order-only changes:** pure slot-order changes (6.87 % between groupings) are excluded from these counts.

**Near-ties.** As in Family A, the flips concentrate on low margins.
- Flip rate by BF16 margin decile, lowest to highest:
  - D: 27.5, 18.8, 8.3, 3.8, 1.7, 0.7, 0.2, 0.1, 0.0, 0.0 %
  - A: 20.0, 11.3, 3.4, 1.3, 0.6, 0.2, 0.1, 0.0, 0.0, 0.0 %
- The BF16 exact-tie rate is 4.88 %.
- 27.4 % of D's seed-0-vs-ORIGINAL flips sit on exact ORIGINAL ties (A: 35.1 %), with a flip rate on exact ties of
  34.0 % (A: 26.3 %).

**Association.** Spearman(|δ_g|, mean BF16 disagreement) = 0.38 and Spearman(|Δnat − Δfro|, BF16 disagreement) = 0.26
(n = 15, nested; descriptive only).

**Key point:** grouping perturbs routing **more** than accumulation order does (about 1.7×), yet removing that routing
divergence removes little of the PPL spread. The routing flips are real but are not the main carrier of the grouping
effect.

## 7. Weight differences (`weight_difference_analysis.json`, `logs/weight_differences_raw.json`, Fig. 5)

| comparison | BF16 elements changed | rel. Frobenius |
|---|---|---|
| D: seed-0 grouping vs ORIGINAL, FR | 54.0–54.2 % | ≈ 0.09 |
| D: seed-0 grouping vs ORIGINAL, FC | 55.1–55.2 % | ≈ 0.09 |
| D: within-seed grouping pairs (60) | mean 54.8 % (53.8–55.8 %) | mean 0.0898 |
| A: FR_O20261121 vs ORIGINAL | 0.0072 % | 8.16e-04 |
| A: FC_O20261121 vs ORIGINAL | 0.0076 % | 8.56e-04 |

- A grouping change rewrites over half of the quantized weight elements. That is about 7,000× more elements and about
  100× larger relative Frobenius than an accumulation-order change.
- **ULP distance of changed elements** (within-seed mean):

  | 1 ULP | 2 ULP | 3–7 ULP | ≥ 8 ULP | sign change |
  |---|---|---|---|---|
  | 34.7 % | 11.5 % | 14.8 % | 37.7 % | 1.3 % |

  Over a third of the changes are many-ULP moves: different quantization-grid choices, not rounding noise.
- Across the 60 within-seed pairs, the magnitude of the weight difference does not predict the PPL difference.
  - Spearman(rel. Frobenius, |ΔPPL|) = −0.18 natural and −0.22 frozen.
  - Rel. Frobenius varies little across pairs, so this test has little leverage.
  - Mean |ΔPPL| between two variants is 0.00515 natural → 0.00469 frozen.

## 8. Family A vs Family D

| metric | Family A (accumulation order) | Family D (grouping) |
|---|---|---|
| BF16 weight elements changed vs ORIGINAL | 0.0074 % | 54.6 % (seed 0) |
| rel. Frobenius vs ORIGINAL | 8.36e-04 | 0.0897 |
| natural paired SD | 0.002870 (5 orders, 1 seed) | 0.006872 pooled within-seed (seed 0: 0.006364) |
| frozen paired SD | 0.001032 | 0.006286 (seed 0: 0.006728) |
| **R_SD** | **0.359** | **0.915** (seed 0 only: 1.057) |
| sign reversals vs ORIGINAL sign, natural | 0/5 | 9/15 (seed 0: 3/5) |
| sign reversals vs ORIGINAL sign, frozen | 0/5 | 8/15 (seed 0: 2/5) |
| routing disagreement vs ORIGINAL | 3.67 % | 6.09 % (seed 0) |
| routing disagreement, variant vs variant | 3.68 % | 6.10 % |
| mean Δ natural → frozen | +0.00803 → +0.00822 | −0.00125 → +0.00032 |

**Overall picture:**
- **Accumulation order** is a tiny weight perturbation that is amplified by discrete routing flips at near-ties.
  Freezing routing removes most of its effect.
- **Grouping** is a large, direct weight perturbation. It also flips more routes, but its PPL effect survives
  freezing almost entirely.

## 9. Answers to the research questions

1. **Does frozen routing collapse grouping variability?** No. R_SD = 0.915, and the per-seed ratios are inconsistent
   (1.06, 1.19, 0.69). This is WEAK-COLLAPSE by the prespecified rule.
2. **Do ranking reversals disappear under frozen routing?** No. R1 goes 6 → 8/15, R2 goes 9 → 8/15, and 3/15 signs
   flip.
3. **Does the FR-vs-FC conclusion change?** No. Under grouping variation the ranking is undetermined in both regimes:
   - the mean Δ is about 0 (−0.00125 / +0.00032);
   - both bootstrap 95 % CIs contain zero;
   - FR-better / FC-better counts are 6/9 and 7/8;
   - the historical rule gives GROUPING-VARIATION-DOMINATES under both regimes.

   The seed-0 ORIGINAL pair (+0.0101 / +0.0087) is one draw from this spread, not a stable effect.
4. **Does grouping change routing?** Yes, more than accumulation order does: 6.1 % vs 3.7 % of (layer, token) sets.
   The flips concentrate at near-ties and exact BF16 ties, and are absent in layers 0–1.
5. **Where does the grouping effect come from?** Mainly from the quantized weights themselves. Grouping changes about
   55 % of effective weight elements, including about 38 % moves of 8 or more ULP. This propagates continuously
   through hidden states and through the router probabilities, which stay live, regardless of which experts are
   selected.
6. **Does Family A's routing-amplification mechanism generalize to grouping?** No. Routing amplification explains
   variability from *small* perturbations (order). It is not the mechanism behind grouping sensitivity.

**Conclusion B: weak collapse.** Calibration-grouping sensitivity of FR vs FC is a weight-level effect, not an
inference-time routing amplification effect. This study's result therefore does not change the earlier conclusion
that grouping variation dominates the FR/FC difference (`update_cross_seed_grouping.md`).

## Limitations

- **Small design.** There are 3 seeds × 5 groupings, nested. With n = 5 per seed the SD estimates are noisy. R_SD is a
  descriptive screening ratio, not a significance test, and the SD difference is not an additive variance
  decomposition.
- **Partial freeze.** The intervention freezes only the routing support. Gate probabilities stay live and differ from
  BF16 in about 92 % of forced entries, so gate-weight divergence is not removed. The intervention is also off-policy
  on about 9 % of slots.
- **Fragile R1.** R1 is defined against the regime mean, and that mean is close to zero in both regimes.
- **Missing seed-1/2 ORIGINAL weights.** These weights were never stored, so R2 for those seeds is a mixed-regime
  comparison.
- **Narrow setting.** The results cover 4 quantized layers, partial-model BF16 fake quantization (not packed INT4), one
  model, and the WikiText-2 partial-model PPL protocol, with windows inspected in earlier studies.
- **Small effects.** Effect sizes are about 1e-3 to 1e-2 PPL on a base of about 9.27, where window CIs exclude zero for
  only 2–4 of 15 pairs per strategy.

## Cost (`cost_results.json`)

| item | value |
|---|---|
| model load | 29–95 s (5 loads across restarts) |
| natural eval, median | 21.2 s |
| frozen eval, median | 21.3 s |
| install per configuration, median | 28 s |
| peak GPU | 3.32 GB |
| peak CPU RSS | 50.6 GB |
| run storage | 3.43 GB (in `~/moe_cache_grouping_routing`) |
| GPTQ solves | 0 |
| evaluation forwards | 69 |
| weight-difference stage (CPU) | about 95 min |

**Operational note:** the container RAM limit is a 72 GB cgroup cap (`/sys/fs/cgroup/memory.max`), even though `free`
reports 226 GB. The GPU stage was killed by the OOM killer twice:
1. once in the original run, after 1/30 configurations;
2. once when run concurrently with the CPU weight-difference stage (about 30 GB).

The stage is resumable: valid per-configuration results are skipped. After the weights stage finished, the GPU stage
completed in one pass. None of this affects the results; every configuration passed A/E/F after resuming.

## Files

- **Code:**
  - `diagnostics/grouping_routing_amplification.py` (stages audit / protocol / gpu / weights / analyze)
  - `diagnostics/report_grouping_routing_amplification.py` (analysis, decision and figures)
  - `tests/test_grouping_routing_amplification.py`
  - Reused from Family A: `moe_calib/inference_routing_override.py`, `diagnostics/inference_routing_amplification.py`
- **Outputs:** `runs_grouping_routing_amplification/`
  - `protocol.json`, `artifact_inventory.json`, `weight_manifest.json`
  - `routing_override_validation.json`, `natural_ppl_results.json`, `frozen_ppl_results.json`
  - `paired_grouping_comparisons.json`, `variance_comparison.json`, `ranking_reversal_analysis.json`
  - `routing_divergence.json`, `weight_difference_analysis.json`, `correctness_results.json`, `cost_results.json`
  - `decision.json`, `summary.json`, `summary.md`
  - `figures/fig1`–`fig5`
  - `logs/`, including `interpretation.json` and `pytest_results.json`
- **Large tensors:** `~/moe_cache_grouping_routing/runs/`, holding per-run routing records and token NLL.
