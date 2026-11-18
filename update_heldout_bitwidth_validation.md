# Held-out evaluation validation and 3-bit GPTQ execution sensitivity

Protocol `runs_heldout_bitwidth_validation/protocol.json` (sha256 `f523b03478e2a5ac…`) was frozen on 2026-11-17 at 23:52.
That was after the held-out windows were frozen and the artifact audit passed, and before any new PPL result existed.
A read-only copy is in `logs/protocol_readonly_copy.json`. The full report is
`runs_heldout_bitwidth_validation/summary.md`; this note condenses it.

## Outcomes

| question | answer | key numbers |
|---|---|---|
| Held-out evaluation robustness | **A2: dataset-dependent** | the phenomenon reproduces on C4-val, but dispersion is 0.3–0.6× WikiText-2 |
| MoE paired FC − FR SD(Δ) | **About half** | 0.00687 (WT) → **0.00336** (HO); R_dataset 0.49, F band 0.27–0.89 |
| MoE FR vs FC ranking | **Still undetermined** | mean Δ +0.0004, CI spans 0, 8/7 split, R1 7/15; rule = GROUPING-VARIATION-DOMINATES on both |
| Dense 4-bit grouping SD | **Smaller** | 0.00503 → **0.00156** (0.31×, band 0.10–0.96); SD/Q 5.3 % → 1.8 % |
| Dose response (batch-1) | **Same shape, lower level** | k = 1 gives 76 % of the k = 64 spread; slope 0.06; level 0.58× WT |
| Reference sensitivity | **Persists; the best reference is not transferable** | π_ORIGINAL is worst on both; post-hoc p = 0.006; best reference changes |
| 3-bit vs 4-bit penalty | **5–6× larger** | Q3/Q4 = **6.44** (WT), **5.34** (HO) |
| 3-bit vs 4-bit execution SD | **Grows weakly (not established)** | S3/S4 = 2.23 (WT), 1.11 (HO) |
| Is SD/penalty stable across bit-widths? | **No (B2 on both datasets)** | R3/R4 = **0.346** [0.11, 0.75] (WT), **0.207** [0.04, 0.56] (HO) |
| Functional disagreement, 3 vs 4 bit | **Scales with quantization error** | R_func 3/4 = **0.91** |

**Headline.**
- **The execution-variance phenomenon is real on unseen text, but WikiText-2 overstates its size.** Dispersion, the
  undetermined MoE ranking, the flat dose response and the worst-case ORIGINAL reference all reproduce on a never-used
  C4 validation set. Every absolute spread is 2–3× smaller there.
- **Individual effects do not transfer.**
  - 8 of 15 MoE paired Δ signs flip between datasets (r = −0.23).
  - Dense grouping deviations (r = −0.32) and dose-replicate deviations (r ≈ −0.17 to −0.04) are uncorrelated.
  - "Which variant wins" is a property of the evaluation sample.
- **A penalty-normalized uncertainty threshold is not defensible.**
  - Going to 3 bits multiplies the PPL penalty by 5–6× but the execution SD by only 1.1–2.2×.
  - Dense SD/Q therefore varies 14× across bit-width × dataset, from 5.3 % (4-bit WT) to 0.37 % (3-bit HO).
  - At the function level the ratio is stable (R_func ×0.91), so the drop happens in the mapping from function to PPL.
    That is consistent with a locally quadratic loss: penalty ∝ error², spread ∝ error.

**Validation** (every mandatory check passed bitwise; `heldout_evaluator_validation.json`, `dense3_correctness.json`).
- **WikiText-2 parity.** Per-window NLL reproduces history for:
  - BF16 in both models;
  - all 6 dense 4-bit configs;
  - the dose subset;
  - all 25 reference-matrix models;
  - a 3-config MoE subset.

  The recomputed MoE pooled SD(Δ) is 0.006872, which matches history.
- **Weight integrity.** Hashes match the historical manifests for every config. Targets are restored bitwise after each
  config. All 3,964 MoE non-target parameters are unchanged.
- **3-bit solver checks.**
  - The **bits=4 re-solve reproduces all 6 historical weight sets, codes and permutations bitwise**.
  - 3-bit re-solves are bitwise identical in-process and in a fresh process.
  - Codes lie in [0, 7], and all 8 levels are used in every tensor.
  - **G20261115 = ORIGINAL holds bitwise at 3 bits**, so the null control survives.
- **Untouched history.** No historical artifact was modified. No historical Hessian or calibration cache was
  regenerated. No 4-bit GPTQ was re-run, except the bits=4 parity solve, which was used only for comparison.

## 1. Held-out set

- **Source.** C4 English **validation**, shard `c4-validation.00000-of-00008`, at the same revision as the calibration
  shard.
- **Never used before.** No historical code path loaded it. All calibration and tuning used `c4-train.00000-of-01024`
  or WikiText-2.
- **Selection.** Seed 20261117. One random 512-token window per document, using the calibration convention.
- **Format.** 128 windows × 512 tokens = 65,408 targets, matching the WikiText-2 evaluator.
- **Isolation audit** (all zero hits):
  - text hash and URL against 356 k train documents;
  - 13-gram overlap with 16 historical token sources (2.5 M n-grams);
  - internal near-duplicates.
- **BF16 PPL.** MoE 11.379 (WT 9.237); dense 16.585 (WT 12.523).

## 2. MoE Family D (30 weight sets + seed-0 ORIGINAL pair)

| | WikiText-2 | C4-val held-out |
|---|---|---|
| pooled within-seed SD(Δ) [χ² 95 %] | 0.00687 [0.0049, 0.0113] | **0.00336** [0.0024, 0.0056] |
| mean Δ (FC − FR) | −0.00125 | +0.00038 |
| hierarchical bootstrap 95 % CI | [−0.0048, +0.0022] | [−0.0015, +0.0022] |
| FR better / FC better | 6 / 9 | 8 / 7 |
| R1 reversals | 6 / 15 | 7 / 15 |
| decision rule | GROUPING-VARIATION-DOMINATES | GROUPING-VARIATION-DOMINATES |
| FR SD / Q | 0.153 | 0.180 |

- **ORIGINAL reshuffles.** The seed-0 ORIGINAL Δ moves from the extreme on WikiText (+0.0101, rank 6/6) to −0.0037
  (rank 2/6) on the held-out set.
- **Relative size holds.** Relative to the MoE penalty, the spread is *not* smaller on held-out text (SD/Q 0.18 vs 0.15).
  This is because the quantization penalty Q shrinks too (0.034 → 0.012).

## 3. Dense Qwen2.5-1.5B, 4-bit (grouping, dose, reference matrix)

| | WikiText-2 | held-out | ratio |
|---|---|---|---|
| grouping SD(PPL) | 0.00503 | 0.00156 | 0.31 |
| Q = mean − BF16 | 0.0954 | 0.0881 | 0.92 |
| SD / Q | 0.053 | 0.018 | 0.34 |
| dose s_k, k = 1 / 4 / 16 / 64 | 0.0033 / 0.0046 / 0.0048 / 0.0041 | 0.0018 / 0.0027 / 0.0028 / 0.0024 | ≈ 0.58 |
| reference-matrix range of means | 0.0096 | 0.0054 | 0.56 |
| worst / best reference | π_ORIGINAL / π_G20261112 | π_ORIGINAL / π_G20261111 | |

- **Same per-window perturbation, more averaging.** Per-window paired-difference SDs are similar on both datasets
  (2.8e-3 vs 2.4e-3 nats). The perturbation simply averages out more on C4 text.
- **Fixed act-order is still not a dependable PPL stabilizer.** The fixed/natural SD ratio on the held-out set is
  1.30 / 0.41 / 0.82 / 0.97.
- **A reference chosen on WikiText does not transfer.** π_G20261112 ranks only 4th of 5 on the held-out set.

## 4. Dense 3-bit GPTQ (historical Hessians, only the bit-width changed)

| | 4-bit WT | 3-bit WT | 4-bit HO | 3-bit HO |
|---|---|---|---|---|
| mean PPL | 12.6188 | 13.1375 | 16.6733 | 17.0560 |
| Q | 0.0954 | **0.6140** | 0.0881 | **0.4709** |
| S = SD(PPL) | 0.00503 | **0.01120** | 0.00156 | **0.00173** |
| R = S / Q | 0.0527 | **0.0182** | 0.0177 | **0.0037** |

| 3-bit / 4-bit | WikiText-2 | held-out |
|---|---|---|
| Q3/Q4 | 6.44 | 5.34 |
| S3/S4 (grouping-paired bootstrap) | 2.23 (0.69–4.7) | 1.11 (0.23–3.0) |
| R3/R4 | **0.346** (0.11–0.75) | **0.207** (0.04–0.56) |

**Functional check** (optional §14, saved BF16 MLP inputs on the WikiText-2 windows):

| | 4-bit | 3-bit | 3 / 4 |
|---|---|---|---|
| quantization error ‖f_q − f_BF16‖ / ‖f_BF16‖ | 0.0449 | 0.1040 | 2.32 |
| RMS R_func | 0.152 | 0.137 | 0.91 |

- **Inputs.** The 3-bit study used the same 6 Hessian sets (sha-verified) and the unchanged solver: asymmetric min/max
  g128 grid, percdamp 1.0, natural act-order.
- **Little shared output.** Only 14–16 % of the 3-bit BF16 elements equal their 4-bit counterparts.
- **Grouping deviations do not keep their direction across bit-widths.** Sign agreement is 1/5 on both datasets.

## 5. Replicate planning

The planning formula is n ≈ ((1.96 + 0.84) s / δ)², with two-sided α = 0.05 and power 0.8:

| design | WikiText-2 | held-out |
|---|---|---|
| MoE paired FC − FR, δ = 0.005 | 15 | 4 |
| MoE paired FC − FR, δ = 0.0025 | 60 | 15 |
| dense 4-bit, independent arms, δ = 0.005 | 16 | 2 |
| dense 3-bit, independent arms, δ = 0.005 | 79 | 2 |

- **No fixed replicate count fits.** The requirement varies up to ~40× across dataset and bit-width for the same
  models.
- **Recommendation.** Measure execution SD per model × bit-width × evaluation set, with **≥ 5–8 execution replicates per
  arm**. Report how the result depends on the evaluation set alongside any method ranking.

## 6. What this changes for the paper

- **Validated on unseen text:**
  - execution-induced PPL dispersion in both models;
  - the undetermined MoE FR/FC ranking;
  - the flat dose response (one perturbed row already gives most of the spread);
  - fixed act-order is not a variance fix;
  - π_ORIGINAL is a poor reference;
  - bitwise reproducibility, including the exact null at both 3 and 4 bits.
- **Specific to the WikiText-2 setup:**
  - the absolute magnitudes, which are 2–3× larger there;
  - the identity and sign of any individual grouping effect;
  - the "best" reference permutation.
- **Drop:**
  - any single SD/penalty threshold that is reused across bit-widths or datasets.
- **Not tested:**
  - other models;
  - packed INT kernels;
  - other numerical stacks (A100 MIG, torch 2.11 cu128, eager attention).

## 7. Deviations and notes

- **Code after the freeze.** The analysis and functional scripts were written after the freeze, but they implement the
  frozen definitions. After that, only two figure-only fixes were made, and no numbers changed.
- **Post-hoc additions, labelled.** The reference-matrix ANOVA is post hoc. The functional analysis was triggered by the
  3-bit result and uses WikiText-2 inputs.
- **Interpreting the intervals.** Window-bootstrap intervals are conditional on the weights. The F bands between
  datasets treat shared weights as independent, so they are descriptive only.
- **MoE R2 is seed-0 only on the held-out set**, because the ORIGINAL weights for seeds 1 and 2 were never stored.
- **Not run:** the strong-path and k = 32/96 dose families, and MoE frozen routing.
- **Cost.**
  - Dense: 29 min. 3-bit: 9 min (about 20 s per solve). MoE: 90 min (156 s per config).
  - Peak RSS 50.8 GB; peak GPU 3.3 GB.
  - The stages ran sequentially to stay under the 72 GB cgroup cap.

## Files

- **Code:**
  - `diagnostics/heldout_bitwidth_validation.py` (stages select / audit / protocol / dense / dense3 / dense3_repeat /
    moe / analyze)
  - `diagnostics/report_heldout_bitwidth_validation.py`
  - `diagnostics/heldout_bitwidth_functional.py`
- **Outputs:** `runs_heldout_bitwidth_validation/`.
  - `summary.md` and all required JSONs.
  - `figures/fig1`–`fig7` (PNG and PDF).
  - `results/` holds per-window NLL.
  - `logs/`
- **3-bit weights:** `~/moe_cache_heldout_bitwidth/dense3/<config>/L<l>.pt`.
