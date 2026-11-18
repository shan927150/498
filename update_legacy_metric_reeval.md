# Legacy metric re-evaluation: do failed MoE calibration methods still fail under actual MoE output error?

**Primary verdict (rule frozen in `runs_legacy_metric_reeval/protocol.json` before any output-error result):
FC vs FR = `LEGACY-NEGATIVE-CONFIRMED` (Case A).**

**Final recommendation: B — retain C4 Random and strengthen the negative-results report.**

The expert-reconstruction metric did not hide a benefit of any legacy calibration strategy:
- **Full Coverage-LP** is worse than Full Random on the actual aggregated MoE-layer output error in **3/3 paired seeds and
  on both evaluation splits**.
  - Primary (confirmation) split: +1.73 to +2.20 %, mean +1.97 %.
  - Secondary (old-test) split: +1.50 to +1.66 %.
- **Official EBSS** is +22 % worse.
- **Typical Coverage-LP** is indistinguishable from Typical Random (mean −0.10 %, seeds −0.48 / −0.02 / +0.19 %).

The old metric overstates the *size* of the differences, not their *direction*. Output-error penalties are about
0.55–0.74× the expert-reconstruction penalties.

**Scope:**
- Qwen1.5-MoE-A2.7B BF16, routed experts of layers 1/8/16/23.
- Standard GPTQ, q = 0: 4-bit, g128, percdamp 1.0, act-order, CANONICAL-B1 non-sequential calibration capture.
- Evaluation on frozen BF16 layer inputs with frozen routing. This is **local, non-sequential layer-output distortion,
  not full-model task performance.**
- This is a retrospective metric audit, not a new method.

Artifacts:
- **Run outputs:** `runs_legacy_metric_reeval/` — protocol, inventory, regression, correctness, metrics, decomposition,
  comparisons, ranking, cost, decision, summary, `figures/fig1–5`, `logs/`, `results/`.
- **Code:** `diagnostics/legacy_metric_reeval.py`, `diagnostics/report_legacy_metric_reeval.py` and
  `tests/test_legacy_metric_reeval.py` (9 CPU tests).
- No existing module was modified.

---

## 1. Artifact audit (`artifact_inventory.json`)

| condition | weights | status |
|---|---|---|
| FR / TR / FC s0–s2, official EBSS (B = 128) | `~/moe_cache_ebss_official_validation/weights/*` (CANONICAL-B1) | reused; all 4 layer files match their recorded sha256 |
| B = 16: C4R s0–s2, RS_PO, RS_EBSS (= ALG1_RATE), PX_PO, PX_EBSS | generation-pilot / balance-fidelity / prefix-rescue weights (CANONICAL-B1) | reused; sha256 match |
| **TC s0–s2** | none under CANONICAL-B1 (the `runs_partial_ppl` TC weights used the batch-8 ORIGINAL extraction) | **quantized here** from the frozen manifests (§2) |
| Coverage-LP-Repr | only E3 score JSONs, no weights; rows not in any B1 cache | **not evaluated** (optional; would need new capture + 3 quantizations) |

- **No selection was changed:** no new sequences, no LP solve, no routing profile, no EBSS generation.
- **All 20 evaluated weight sets are standard GPTQ:**
  - 0 RTN fallbacks
  - 240 experts quantized
  - routers and untouched parameters unchanged
  - every projection differs from BF16

## 2. TC under CANONICAL-B1 and the baseline regression (`baseline_regression.json`)

**Capture:**
- 343 C4 rows were re-captured at batch 1, BF16: the union of the TC s0/s1/s2 rows and the FR s0 rows.
- They were written to a new cache, `~/moe_cache_legacy_metric_reeval`.
- **The 242 rows shared with the historical B1 cache are bitwise identical at all four layers, and their routing counts
  are identical.**

**Gate (passed):**
- FR s0 was re-quantized from the **new** cache with the **current** code.
- It reproduces the historical weights **bitwise** (installed sha `ecfdb234…`), the per-expert E3 scores exactly, and
  PPL 9.269004 exactly.
- Only after this gate passed were TC s0/s1/s2 quantized, with the frozen selection order preserved:

  | | TC s0 | TC s1 | TC s2 |
  |---|---|---|---|
  | E3 (old-test) | 0.0046155 | 0.0046094 | 0.0046141 |
  | RTN fallbacks | 0 | 0 | 0 |

**Evaluator regressions (old-test, FR s0):**
- **Aggregated MoE-layer output:** 0.0037166 vs the AGQ-only validation's GPTQ arm 0.0037166 (max per-layer relative
  difference 5e-16). The native-BF16-block value is also reproduced (5e-16).
- **Equivalence with the validated `LayerEvalAccumulator`:** run alongside in every evaluation, the lean evaluator
  agrees within ≤ 1 ulp (≤ 5e-18 absolute).
- **Expert reconstruction:**
  - Recomputed on the **installed BF16** weights (the weights the layer metric uses), it is **uniformly +0.025 to
    +0.035 %** above the historical value. The historical value scored the solver's fp32 W_q before BF16 installation.
  - Per-expert Spearman = 1.000 for every condition, and every historical paired sign is preserved.
  - Tolerance (0.5 %) passed.
  - Historical values are unchanged; new deltas use the installed-weight values on both arms.
- **Correctness gates:** every gate in `correctness_results.json` passes.
  - Router logits are unchanged under every quantized install, asserted per sequence.
  - Pristine weights are restored after every condition.
  - The evaluation universe is identical.
  - The fast E3 path is bitwise equal to `quant_eval.per_seq_rel_errors` on the pre-specified check experts.
  - D + C = total holds to 4e-16.

## 3. Evaluation split (frozen before results)

| split | role | windows | status |
|---|---|---|---|
| `~/moe_cache_confirmation` (locked, document-disjoint) | **primary** | 256 × 512 | never used for selection. It was used once before (aggregate-constraint study) to score *expert reconstruction* of FR/TR/TC/FC built from the batch-8 cache, but never for MoE-output error. |
| `~/moe_cache` rows 2128–2255 | secondary / regression | 128 × 512 | inspected in many prior studies |

Both splits are therefore treated as **retrospective / exploratory**. All 11 family classifications agree between the two
splits.

## 4. Primary metric: aggregated MoE-layer output error (`moe_layer_output.json`)

Definition:
- Per layer, E_layer = Σ‖y − ŷ‖² / Σ‖y‖², with y = Σ c·E(x) + shared(x).
- Expert math is in fp32 with float64 sums, under frozen BF16 routing.
- The aggregate is the **mean of the four layer ratios**, the AGQ convention. The pooled ratio is stored for reference
  only.

Confirmation split, means over seeds (EBSS is one set):

| strategy | E_expert | E_layer | L1 | L8 | L16 | L23 |
|---|---|---|---|---|---|---|
| Full Random | 0.0045813 | 0.0036814 | 3.094e-4 | 7.108e-3 | 7.015e-3 | 2.928e-4 |
| Typical Random | 0.0045608 | 0.0036693 | 3.037e-4 | 7.091e-3 | 6.991e-3 | 2.919e-4 |
| Full Coverage-LP | 0.0047094 | 0.0037541 | 3.120e-4 | 7.229e-3 | 7.182e-3 | 2.934e-4 |
| Typical Coverage-LP | 0.0045631 | 0.0036656 | 3.045e-4 | 7.083e-3 | 6.985e-3 | 2.892e-4 |
| Official EBSS | 0.0059677 | 0.0045026 | 5.859e-4 | 8.374e-3 | 8.686e-3 | 3.646e-4 |

The native BF16 block forward with the installed weights agrees: every Δ is within 0.07 points of the fp32 Δ.

## 5. Paired comparisons and ranking agreement (`paired_comparisons.json`, `ranking_agreement.json`)

Δ = (method − reference) / reference, where negative means the method is better. The brackets are a *descriptive* paired
window bootstrap (95 %), resampling evaluation windows, not calibration draws.

**Confirmation split (primary):**

| comparison | seed | Δ expert recon | Δ MoE output | per layer L1 / L8 / L16 / L23 | agreement |
|---|---|---|---|---|---|
| FC − FR | s0 | +3.08 % | **+2.20 %** [1.73, 2.65] | −0.42 / +1.80 / +2.80 / +0.46 | yes (both worse) |
| | s1 | +2.39 % | **+1.73 %** [1.26, 2.17] | +4.98 / +1.57 / +1.83 / −0.19 | yes |
| | s2 | +2.92 % | **+1.99 %** [1.51, 2.46] | −1.95 / +1.74 / +2.50 / +0.42 | yes |
| TC − TR | s0 | −0.15 % | −0.48 % [−0.78, −0.17] | −3.32 / −0.22 / −0.59 / −0.98 | yes (both better) |
| | s1 | +0.08 % | −0.02 % [−0.35, 0.29] | +0.86 / −0.16 / +0.13 / −0.88 | no (both ≈ 0) |
| | s2 | +0.22 % | +0.19 % [−0.11, 0.50] | +3.21 / +0.07 / +0.23 / −1.00 | yes |
| TR − FR | s0 | −0.13 % | −0.16 % | +0.48 / +0.04 / −0.39 / −0.02 | yes |
| | s1 | −0.68 % | −0.30 % | −4.59 / −0.31 / −0.09 / −0.74 | yes |
| | s2 | −0.53 % | −0.53 % | −1.35 / −0.48 / −0.55 / −0.09 | yes |
| EBSS − FR | s0 / s1 / s2 | +30.5 / +29.9 / +30.4 % | +22.3 / +22.3 / +22.3 % | L1 ≈ +87 to +91 %, others +18 to +25 % | yes |

On the old-test split, the same pattern holds:
- FC − FR is +1.50 / +1.55 / +1.66 %.
- TC − TR is −0.38 / +0.03 / +0.08 %.
- TR − FR is +0.04 / −0.19 / −0.01 %.
- EBSS − FR is +21.6 to +21.8 %.

**Experts improved vs the same-seed FR (confirmation split):**

| | experts improved |
|---|---|
| FC | 0 / 240 in every seed |
| EBSS | 0 / 240 |
| TR | 70 / 94 / 95 % |
| TC | 80 / 90 / 80 % |

**Classification (frozen rule, τ = 0.5 %):**

| family | confirmation (primary) | old-test | mean Δ expert → mean Δ layer |
|---|---|---|---|
| FC vs FR (main) | **LEGACY-NEGATIVE-CONFIRMED** | LEGACY-NEGATIVE-CONFIRMED | +2.80 → +1.97 % (×0.71); old-test +2.48 → +1.57 % (×0.63) |
| TC vs TR (secondary) | NO-ROBUST-DIFFERENCE (legacy: tie) | NO-ROBUST-DIFFERENCE | +0.05 → −0.10 % |
| TR vs FR (additional) | NO-ROBUST-DIFFERENCE (tie) | NO-ROBUST-DIFFERENCE | −0.45 → −0.33 % |
| TC vs FR (extra) | NO-ROBUST-DIFFERENCE (tie) | NO-ROBUST-DIFFERENCE | −0.40 → −0.43 % |
| EBSS vs FR / vs TR (one set) | LEGACY-NEGATIVE-CONFIRMED | LEGACY-NEGATIVE-CONFIRMED | +30.3 → +22.3 % (×0.74) |

FC vs FR is Case A rather than Case D because its mean output penalty (+1.97 %) exceeds max(τ, 0.5 × the expert
penalty) = 1.40 %.

**Ranking-agreement categories, confirmation split (31 comparisons, B = 16 included):**
- 22 worsen both metrics.
- 8 improve both.
- 0 improve expert reconstruction while worsening output.
- **1 worsens expert reconstruction while improving output: TC s1 vs TR s1, +0.08 % → −0.02 %.** Both values are far
  below τ, so this is noise, not an AGQ-like reversal.
- On old-test, the two such cases (TC s0 vs TR s0 and TC s0 vs FR s0) have expert penalties of only +0.08 / +0.13 %.

**Rank correlation between E_expert and E_layer across configurations:**

| configurations | confirmation | old-test |
|---|---|---|
| 13 B = 128 configurations incl. EBSS | Spearman 0.98 (Kendall 0.92) | 0.75 |
| 12 C4 configurations | 0.98 | 0.69 |
| 7 B = 16 sets | 0.96 | 0.89 |

The lower old-test values come from reshuffling *within* the FR/TR/TC cluster, whose differences are below 0.5 %.

**Per-layer Spearman** (12 C4 configurations, confirmation): L1 0.15, L8 0.88, L16 0.82, L23 0.91. L1 is where the two
metrics decouple (§6).

## 6. Error decomposition (`error_decomposition.json`)

‖Δy‖² = D + C, with D = Σₖ cₖ²‖ΔEₖ‖² (diagonal, c²-weighted) and C the pairwise cross terms. Both are divided by the same
layer denominator. The identity holds to 4e-16.

- **L8 / L16 / L23:** the cross term is exactly about 0 (|C| / total < 1e-3) for every method. The output error *is* the
  c²-weighted expert error.
- **L1:** the cross term is substantially **negative** for every method: C / total = −0.23 to −0.32 for C4 sets, −0.11
  for EBSS, and −0.01 to −0.06 for the random-start B = 16 sets. Expert errors at L1 partially cancel in the weighted sum.
- **Driver of the method differences:**
  - Aggregated over layers, the method differences are almost entirely **diagonal**. On confirmation, FC − FR is
    ΔD = +2.09 / +1.79 / +1.96 % against ΔC = +0.12 / −0.06 / +0.03 %, as a percentage of FR E_layer. EBSS − FR is
    ΔD ≈ +22.2 % against ΔC ≈ +0.15 %.
  - **At L1 alone**, both ΔD and ΔC are large, seed-dependent and partly offsetting (`fig4`). For FC − FR at L1,
    ΔD = −5.16 / +8.07 / −3.29 % and ΔC = +4.74 / −3.09 / +1.34 % (s0 / s1 / s2, as a percentage of FR's L1
    E_layer).
  - That is why L1 is the only layer whose FC − FR sign flips between seeds (−0.42 / +4.98 / −1.95 %). It is also why
    L1's metric rank correlation is low.
  - No L1 sign is consistent across seeds, so the frozen rule flags **no layer-specific reversal**.

## 7. Partial-model PPL cross-check (descriptive, `ranking_agreement.json → method_level`)

**Protocol-compatible PPL:**
- FR / TR / FC / EBSS: the official-EBSS study's PPL measured on **these exact weights**.
- TC: produced on these exact weights by `quant_config`'s built-in validity check during the TC quantization. It is a
  byproduct, not a planned PPL experiment.
- Batch-8 `runs_partial_ppl` values are not used.

| strategy | E_expert rank | E_layer rank (conf. / old) | mean PPL (BF16 9.2371) | PPL rank |
|---|---|---|---|---|
| Typical Random | 1 | 2 / 2 | 9.26598 | 1 |
| Typical Coverage-LP | 2 | 1 / 1 | 9.26660 | 2 |
| Full Random | 3 | 3 / 3 | 9.27209 | 4 |
| Full Coverage-LP | 4 | 4 / 4 | 9.27080 | 3 |
| Official EBSS | 5 | 5 / 5 | 9.27324 | 5 |

- **Ends agree:** both local metrics and PPL put EBSS last and the typical-pool strategies first.
- **FR and FC swap under PPL.** The whole PPL spread (0.007) is inside the 0.014 noise band used in earlier studies.
- So, as before, four-layer partial PPL does not resolve differences of ≤ 2 % in local error. No PPL claim is made.

## 8. Exploratory B = 16 (separate, never ranked with B = 128)

| comparison | Δ expert | Δ MoE output (confirm) | output penalty / expert penalty |
|---|---|---|---|
| RS_EBSS − C4R | +13.5 % | +8.45 % | ×0.63 |
| RS_PO − C4R | +13.4 % | +8.31 % | ×0.62 |
| PX_EBSS − C4R | +6.1 % | +3.49 % | ×0.58 |
| PX_PO − C4R | +6.3 % | +3.56 % | ×0.57 |
| PX_EBSS − PX_PO (balance term) | −0.22 % | −0.07 % | no robust difference |

Each value is the mean over the three C4R seeds. These are single synthetic sets, so the comparisons are not
replications.
- Every synthetic set remains worse than C4 Random under the output metric, by the same ~0.6× shrinkage.
- The EBSS balance term itself has no detectable effect relative to its PO control.

## 9. Cost (`cost_results.json`)

**Evaluation:**
- One layer of 20 weight sets × 256 windows takes about 13 min (peak GPU 11.9 GB). 128 windows take about 15–19 min
  under GPU sharing (peak 9.3 GB).
- Total evaluation was about 2 h GPU across both splits.

**New compute:**

| step | time |
|---|---|
| TC B1 capture | 343 rows, 310 s |
| FR s0 regression | 1918 s |
| TC s0 | 1492 s |
| TC s1 | 413 s |
| TC s2 | 389 s |

- The FR s0 and TC s0 times are inflated by sharing the GPU with the evaluation.
- Each quantization includes `quant_config`'s two PPL passes.

**No new calibration data, LP, routing profile, EBSS generation, full-depth run or damping change.**

## 10. Answers to the report questions

1. **FC vs FR:** yes. Full Coverage-LP still underperforms Full Random on aggregated MoE output error:
   - +1.97 % mean, 3/3 seeds, on the primary split;
   - +1.57 % on old-test;
   - 0/240 experts improved.
2. **TC vs TR:** no. Typical Coverage-LP does not outperform Typical Random (−0.48 / −0.02 / +0.19 %, no robust
   difference). TC is nominally first by E_layer, by 0.1 %, which is inside seed spread.
3. **Across seeds:** yes. The FC penalty holds in all three seeds on both splits. TC/TR and TR/FR differences change sign
   or stay below τ.
4. **Layer-specific reversals:** L1 is the only layer where FC − FR changes sign (−0.4 / +5.0 / −2.0 %). It is
   seed-dependent, so no consistent layer reversal exists.
5. **Diagonal vs cross terms:** the differences are driven by the c²-weighted diagonal errors. Cross-expert terms are
   about 0 at L8/16/23. At L1 they are large and negative, but their method-to-method changes are seed-dependent and
   roughly offset the diagonal changes.
6. **Does the expert metric exaggerate?** It exaggerates the magnitude and preserves the direction. Output-error
   penalties are 0.55–0.74× the expert-reconstruction penalties for every failed method, and the confirmation rank
   correlation is 0.98.
7. **EBSS:** yes, the negative result is robust. Official EBSS is +22 % on MoE output (L1 ≈ +90 %). The B = 16 EBSS
   variants are +3.5 to +8.5 % vs C4 Random.
8. **Relation to PPL:** the two agree at the extremes (EBSS worst, typical-pool best). FR and FC swap, and the whole PPL
   spread is within noise. Small local-error differences do not translate into resolvable PPL differences.
9. **Interpretation:** the CS498 interpretation does not change materially. Coverage-oriented and synthetic calibration
   lose to natural C4 Random under the faithful layer-output objective as well. The penalty is about 30 % smaller than the
   expert metric suggested.
10. **New algorithm or C4 Random?** This strengthens the case for retaining C4 Random (typical or full). Unlike AGQ, no
    calibration strategy shows the "worse expert reconstruction, better output" signature, so nothing here motivates a new
    calibration algorithm.

## 11. Disclosures and limitations

**Post-freeze changes** (made before any comparative result was read; none changes a metric definition, rule or
threshold):
1. **Fast E3 path:** E3 recomputation was moved to a GPU-resident copy of the same fp16 held-out inputs, for speed. It is
   asserted bitwise equal to the production scorer on 4 pre-specified experts per layer per split.
2. **Accumulator gate tolerance:** the "lean evaluator vs `LayerEvalAccumulator`" gate was relaxed from exact equality to
   ≤ 1e-12 relative, because the observed differences are 1 ulp from summation order.
3. **Figure fixes:** a figure guard was added for not-yet-evaluated families, and Fig 1 was redesigned (zoom panel)
   after the first render was illegible.

**Limitations:**
- **No pristine split:** both splits had been seen before for expert reconstruction, so this is a retrospective audit.
- **Small replication:** there are three calibration seeds per C4 strategy, and EBSS and the B = 16 synthetic sets are
  single sets. No population-level significance is claimed. The bootstrap intervals reflect evaluation-window noise only.
- **Local metric only:** this measures local, non-sequential, frozen-routing layer output on four layers. It says
  nothing about full-model task quality.
- **Not evaluated:** Coverage-LP-Repr was not evaluated.
