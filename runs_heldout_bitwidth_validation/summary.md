# Held-out evaluation validation and 3-bit GPTQ execution sensitivity

**Protocol:** `protocol.json` (sha256 `f523b03478e2a5ac…`). It was frozen at 2026-11-17 23:52, after the held-out windows
were frozen and the artifact audit passed, and before any new PPL result existed. A read-only copy is in
`logs/protocol_readonly_copy.json`. The driver code hash was re-checked at the start of every GPU stage and never
changed.

**Classifications** (prespecified rules in `protocol.json → decision_criteria`):

| question | label | key numbers |
|---|---|---|
| Held-out evaluation robustness | **A2 – dataset-dependent effect** | MoE paired SD(Δ) is 0.49× its WikiText value (F band 0.27–0.89). Dense 4-bit SD(PPL) is 0.31× (band 0.10–0.96). Both are measurable, but smaller. |
| Bit-width stability | **B2 – penalty grows faster than SD** (both datasets) | Q3/Q4 = 6.4 / 5.3, S3/S4 = 2.2 / 1.1, R3/R4 = **0.35 / 0.21** (paired-bootstrap 95 % 0.11–0.75 / 0.04–0.56) |

**Validation.** All mandatory checks passed with exact (bitwise) tolerance (`heldout_evaluator_validation.json`,
`dense3_correctness.json`, `logs/*end*`).

- **WikiText-2 parity.** Per-window NLL reproduces history bitwise for:
  - BF16 in both models;
  - all 6 dense 4-bit configs;
  - the dose subset (`b1_k001_r0`, `b1_k064_r7`, both policies);
  - all 25 reference-matrix models;
  - the MoE subset (`FR_s0_G20261111`, `FC_s2_G20261111`, `FR_ORIGINAL`).

  The recomputed historical MoE pooled SD(Δ) is 0.006872, which matches history.
- **Weight integrity, every configuration.**
  - Weight hashes match the historical manifests: file, layer and installed digest for MoE; installed digest and
    per-tensor sha for dense and dose.
  - Shapes, dtype and finiteness are correct.
  - Every configuration starts from pristine targets and is restored bitwise afterwards.
  - All 3,964 MoE non-target parameters are unchanged at the end. A light per-config digest also covers gates,
    quantized-layer non-expert parameters, embeddings and head.
- **Repeated evaluation is bitwise** for every dense 4-bit, dense 3-bit and MoE config and for the dose subset.
- **BF16 end-of-run re-evaluation** is bitwise on both datasets in every GPU process.
- **3-bit checks:**
  - codes lie in [0, 7], and all 8 levels are used in every tensor;
  - scale·(q − zero) reproduces the FP32 solver output bitwise;
  - the installed BF16 weights equal bf16(solver output);
  - re-solves of G20261111 are bitwise identical both in-process and in a fresh process;
  - the **bits=4 re-solve reproduces all 6 historical weight sets, codes and permutations bitwise**;
  - **G20261115 and ORIGINAL give bitwise-identical 3-bit weights, codes, scales/zeros and per-window NLL**, so the
    null control survives.

No historical artifact was modified. No historical Hessian or calibration cache was regenerated. No 4-bit result was
re-solved; the bits=4 solve served only as a parity check.

## 1. Held-out evaluation set (`evaluation_manifest.json`, `evaluation_overlap_audit.json`)

- **Source.** C4 English **validation** split, shard `en/c4-validation.00000-of-00008.json.gz`, at the same repo
  revision as the calibration shard (`1588ec45…`).
- **Never used before:**
  - All historical calibration and tuning data came from `c4-train.00000-of-01024` (data_utils, confirmation split)
    or from WikiText-2 train/validation (domain ablation).
  - No code path ever loaded C4 validation.
  - The shard was absent from the local HF cache until this study downloaded it.
- **Selection.**
  - Seed 20261117.
  - One random 512-token window per document, using the calibration convention, with the MoE tokenizer.
  - 426 documents were examined: 298 were too short, and 0 were excluded by the content rules.
  - No model was evaluated before the freeze.
- **Format.** 128 windows × 512 tokens = 65,408 targets, batch 8, ordered by shard row. Windows sha256 `e105f974…`.
- **Isolation checks** (all 0):
  - text-hash or URL match against the 356,317 train-shard documents;
  - 13-gram overlap with any of 16 historical token sources (all calibration/tuning `ids.pt` files, the WikiText-2
    windows and the EBSS tokens; 2.5 M n-grams);
  - near-duplicates within the held-out set.
- **Tokenizer.** The dense Qwen2.5 tokenizer re-tokenizes all 128 windows identically.

BF16 PPL on the held-out set is 11.379 for the MoE (WikiText 9.237) and 16.585 for the dense model (WikiText 12.523).

## 2. MoE Family D on held-out text (`moe_heldout_ppl_results.json`, `moe_heldout_pairing_analysis.json`, Fig. 1)

**Design:** 30 existing weight sets (3 seeds × 5 groupings × FR/FC) plus the seed-0 ORIGINAL pair. Inference routing
is natural. Δ = PPL(FC) − PPL(FR), and positive Δ means FR is better.

| | WikiText-2 (historical) | C4-val held-out |
|---|---|---|
| pooled within-seed SD(Δ) | **0.006872** [0.0049, 0.0113] | **0.003364** [0.0024, 0.0056] |
| seed SDs (s0 / s1 / s2) | 0.0064 / 0.0052 / 0.0086 | 0.0051 / 0.0007 / 0.0028 |
| pooled SD of paired mean-NLL Δ (nats/token) | 7.41e-4 | 2.95e-4 |
| mean Δ (mean of seed means) | −0.00125 | +0.00038 |
| seed means | −0.0019, +0.0007, −0.0025 | −0.0003, −0.0001, +0.0015 |
| FR better / FC better | 6 / 9 | 8 / 7 |
| R1 reversals (sign opposite to M) | 6 / 15 | **7 / 15** |
| R2, seed 0 (vs ORIGINAL Δ on the same dataset) | 3 / 5 | 2 / 5 |
| hierarchical bootstrap 95 % CI of mean Δ | [−0.0048, +0.0022] | [−0.0015, +0.0022] |
| historical decision rule | GROUPING-VARIATION-DOMINATES | **GROUPING-VARIATION-DOMINATES** |
| pairs whose window CI excludes 0 | 4 / 15 | 0 / 15 |
| FR pooled SD(PPL) / FC pooled SD(PPL) | 0.00520 / 0.00381 | 0.00209 / 0.00189 |
| FR quantization penalty Q (mean − BF16) | 0.0340 | 0.0116 |
| FR SD / Q (pooled, 3 seeds) | 0.153 | **0.180** (window-bootstrap 0.15–0.47) |

**Ratios between datasets:**
- R_dataset = SD_held-out / SD_WikiText = **0.490**; F(12,12) band 0.27–0.89.
- On the mean-NLL scale the ratio is **0.398** (band 0.22–0.72).
- The held-out paired spread is therefore about half of the WikiText spread, and the band excludes 1.

**Cross-dataset directions:**
- 8 of 15 Δ change sign between datasets.
- Pearson(Δ_WT, Δ_HO) = −0.23; within-seed centred r = −0.21.
- Individual grouping effects therefore do **not** carry over between evaluation texts. Which variant "wins" is itself
  a property of the evaluation sample.

**ORIGINAL reference:**
- On WikiText the seed-0 ORIGINAL Δ (+0.0101) sat at the extreme of the seed-0 groupings (z = +1.9, rank 6/6).
- On the held-out set it is −0.0037 (z = −0.7, rank 2/6).
- FR_ORIGINAL moves from the best seed-0 FR variant (rank 1) to rank 4. FC_ORIGINAL moves from worst (6) to best (1).
- The canonical reference is not systematically favourable or unfavourable: its position reshuffles with the text.

## 3. Dense 4-bit on held-out text (`dense4_heldout_ppl_results.json`, Fig. 2)

| | WikiText-2 | C4-val held-out | held-out / WikiText |
|---|---|---|---|
| BF16 PPL | 12.52345 | 16.58515 | |
| mean PPL (5 groupings) | 12.61884 | 16.67325 | |
| SD(PPL) | 0.00503 [0.0030, 0.0145] | **0.00156** [0.0009, 0.0045] | **0.31** (F(4,4) band 0.10–0.96) |
| SD(log PPL) | 3.99e-4 | 9.38e-5 | 0.24 |
| Q = mean − BF16 (ORIGINAL-based Q0) | 0.0954 (0.1007) | 0.0881 (0.0894) | 0.92 |
| SD / Q | **0.053** (window-bootstrap 0.033–0.086) | **0.018** (0.012–0.056) | 0.34 (band 0.11–1.04) |
| max pairwise difference | 0.0125 | 0.0035 | |
| mean per-window SD of paired NLL differences (nats/token) | 2.8e-3 | 2.4e-3 | |
| ORIGINAL rank among the 5 | 5 (worst, tie) | 4 | |

- The per-window functional perturbation is about as large on the new text: the per-window difference SDs are similar.
  It averages out much more at the aggregate level.
- Grouping deviations do not replicate across datasets: Pearson −0.32, sign agreement 2/5, and the rankings differ
  completely.
- G20261115 equals ORIGINAL bitwise on both datasets.

## 4. Dose response on held-out text (`dense4_dose_heldout_analysis.json`, Fig. 6)

Batch-1 family, 8 replicate subsets per dose, natural act-order. The replicate IDs are the historical ones.
Q0 = PPL(k = 0) − BF16 is 0.1007 on WikiText and 0.0894 on the held-out set.

| k | s_k WikiText | s_k held-out | s_k / Q0 held-out (χ² 95 %) | replicate-deviation r (WT vs HO) |
|---|---|---|---|---|
| 1 | 0.00327 | 0.00179 | 0.020 (0.013–0.041) | −0.04 |
| 4 | 0.00461 | 0.00274 | 0.031 (0.020–0.062) | −0.17 |
| 16 | 0.00480 | 0.00281 | 0.031 (0.021–0.064) | −0.12 |
| 64 | 0.00409 | 0.00236 | 0.026 (0.017–0.054) | −0.11 |

- **The historical pattern persists in shape.**
  - One perturbed row already gives 76 % of the k = 64 spread (WikiText: 80 %).
  - The log-log slope of s_k against k is 0.06 (WikiText 0.05).
  - So there is no proportional growth with k.
- **The level is about 0.58× WikiText.** The held-out R_PPL is 0.020–0.031 against 0.033–0.048, which matches the
  grouping result.
- **Replicate-level PPL deviations are uncorrelated across datasets** (r ≈ −0.17 to −0.04): the same weights rank
  differently on different text.
- **Fixed act-order:** held-out s_k is 0.0023 / 0.0011 / 0.0023 / 0.0023. The fixed/natural SD ratio is 1.30 / 0.41 /
  0.82 / 0.97, so fixing act-order is still not a dependable PPL-variance reducer.

## 5. Reference-selection sensitivity (optional, run; same file, `reference_matrix`)

| | WikiText-2 | held-out |
|---|---|---|
| between-reference range of mean PPL | 0.0096 | 0.0054 |
| mean within-reference SD | 0.0026 | 0.0019 |
| worst reference | **π_ORIGINAL (= π_G20261115)** | **π_ORIGINAL** |
| best reference | π_G20261112 | π_G20261111 |
| post-hoc two-way ANOVA, reference effect | F(4,16) = 9.5, p = 4e-4 | F = 5.4, p = 0.006 |

- **ORIGINAL is the worst reference on unseen text too**, and a reference-permutation effect is present on both
  datasets.
- Apart from ORIGINAL being worst, the ordering of the references does not replicate: Spearman of the reference means
  is 0.10, and the best reference changes.
- **Reference choice is a real source of quality sensitivity, but "the best reference" is not transferable.** A
  reference picked on WikiText (π_G20261112) is only 4th of 5 on the held-out set.
- No reference was selected on the held-out data.
- The ANOVA is post hoc and treats the interaction as error. It establishes a reference effect, not a formal bias of
  any particular choice.

## 6. Dense 3-bit vs 4-bit (`dense3_*`, `bitwidth_variability_comparison.json`, Figs. 3–5)

**Setup.** The same 6 Hessian sets (sha-verified) and the unchanged solver with bits = 3: 8 levels, the same asymmetric
min/max g128 grid, percdamp 1.0 and natural act-order. Only 14–16 % of the 3-bit BF16 elements equal their 4-bit
counterparts.

| | 4-bit WT | 3-bit WT | 4-bit HO | 3-bit HO |
|---|---|---|---|---|
| mean PPL | 12.6188 | 13.1375 | 16.6733 | 17.0560 |
| Q (mean − BF16) | 0.0954 | **0.6140** | 0.0881 | **0.4709** |
| S = SD(PPL) (χ² 95 %) | 0.00503 (0.0030–0.0145) | **0.01120** (0.0067–0.0322) | 0.00156 (0.0009–0.0045) | **0.00173** (0.0010–0.0050) |
| SD(log PPL) | 3.99e-4 | 8.53e-4 | 9.4e-5 | 1.02e-4 |
| R = S / Q | 0.0527 | **0.0182** | 0.0177 | **0.0037** |
| R_log | 0.0525 | 0.0178 | 0.0177 | 0.0036 |
| max pairwise difference | 0.0125 | 0.0252 | 0.0035 | 0.0045 |

| ratio, 3-bit / 4-bit | WikiText-2 | held-out |
|---|---|---|
| Q3 / Q4 | **6.44** | **5.34** |
| S3 / S4 | 2.23 (grouping-paired bootstrap 0.69–4.7) | 1.11 (0.23–3.0) |
| R3 / R4 | **0.346** (0.11–0.75; window bootstrap 0.20–0.59) | **0.207** (0.04–0.56; window 0.08–0.74) |
| classification | B2 | B2 |

- **Paired grouping table** (`bitwidth_variability_comparison.json → paired_table`). Every grouping is 0.50–0.54 PPL
  worse at 3 bits on WikiText and 0.38–0.39 worse on the held-out set.
- **Grouping deviations are not directionally consistent between bit-widths:**
  - WikiText: Pearson −0.68, sign agreement 1/5;
  - held-out: −0.24, sign agreement 1/5;
  - with n = 5 this is descriptive only.
- **Functional analysis** (optional, `dense3_functional_stability.json`). This uses the historical pooled full-MLP
  norm on the saved BF16 MLP inputs; the reused 4-bit E_MLP reproduces history exactly.

  | | 4-bit | 3-bit | 3 / 4 |
  |---|---|---|---|
  | quantization error ‖f_q − f_BF16‖ / ‖f_BF16‖ | 0.0449 | 0.1040 | 2.32 |
  | mean E_MLP of the 4 perturbations | 0.0068 | 0.0143 | 2.10 |
  | RMS R_func (disagreement / own quantization error) | 0.152 | 0.137 | **0.91** |
  | integer-code disagreement vs own ORIGINAL (within-bit) | 1.18 % | 1.00 % | (not comparable) |

  - At the function level, execution disagreement scales almost exactly with the quantization error (R_func ×0.91).
  - The drop in R therefore does **not** come from the solver output becoming less execution-sensitive. It happens in
    the mapping from function to PPL.
  - The PPL penalty grows roughly with the square of the functional error (2.3² ≈ 5.3, against Q ratios of 5.3–6.4).
  - The PPL spread grows roughly linearly or less (S ×1.1–2.2).
  - This is an interpretation consistent with a locally quadratic loss, not a tested law.

## 7. Joint dataset × bit-width table (`dataset_bitwidth_comparison.json`)

| model | bits | dataset | mean PPL | penalty Q | execution SD | SD / Q |
|---|---:|---|---:|---:|---:|---:|
| Dense | 4 | WikiText-2 (historical weights, re-verified) | 12.6188 | 0.0954 | 0.00503 | 0.053 |
| Dense | 4 | C4-val held-out | 16.6733 | 0.0881 | 0.00156 | 0.018 |
| Dense | 3 | WikiText-2 | 13.1375 | 0.6140 | 0.01120 | 0.018 |
| Dense | 3 | C4-val held-out | 17.0560 | 0.4709 | 0.00173 | 0.0037 |
| MoE FR | 4 | WikiText-2 (historical) | 9.2710 | 0.0340 | 0.00520 (seed 0: 0.00442) | 0.153 (seed 0: 0.121) |
| MoE FR | 4 | C4-val held-out | 11.3908 | 0.0116 | 0.00209 (seed 0: 0.00341) | 0.180 (seed 0: 0.263) |
| MoE FC | 4 | WikiText-2 (historical) | 9.2698 | 0.0327 | 0.00381 | 0.117 |
| MoE FC | 4 | C4-val held-out | 11.3912 | 0.0120 | 0.00189 | 0.158 |

**Sampling structures differ between the rows:**
- **Dense:** one calibration seed and 5 groupings, which are 4 single-row perturbations plus an exact null.
- **MoE:** a pooled within-seed SD over 3 seeds × 5 groupings (12 df). The seed-0-only values are given for a
  like-for-like comparison.

**MoE paired table:**

| dataset | mean Δ | pooled SD(Δ) [χ² 95 %] | FR better / FC better | R1 | rule |
|---|---|---|---|---|---|
| WikiText-2 | −0.00125 | 0.00687 [0.0049, 0.0113] | 6 / 9 | 6 | GROUPING-VARIATION-DOMINATES |
| held-out | +0.00038 | 0.00336 [0.0024, 0.0056] | 8 / 7 | 7 | GROUPING-VARIATION-DOMINATES |

## 8. Replicate planning (`replicate_power_analysis.json`, Fig. 7)

The planning formula is n ≈ ((1.96 + 0.84) s / δ)², with two-sided α = 0.05 and power 0.8. It is a normal
approximation, so small n are optimistic.

| design (s used) | dataset | δ = 0.0025 | δ = 0.005 | δ = 0.01 |
|---|---|---:|---:|---:|
| **paired FC − FR, MoE** (empirical pooled SD(Δ)) | WikiText-2, s = 0.00687 | 60 | 15 | 4 |
| **paired FC − FR, MoE** | held-out, s = 0.00336 | 15 | 4 | 1 |
| independent arms, dense 4-bit (√2 × 0.00503) | WikiText-2 | 64 | 16 | 4 |
| independent arms, dense 4-bit (√2 × 0.00156) | held-out | 7 | 2 | 1 |
| independent arms, dense 3-bit (√2 × 0.01120) | WikiText-2 | 316 | 79 | 20 |
| independent arms, dense 3-bit (√2 × 0.00173) | held-out | 8 | 2 | 1 |
| independent arms, MoE FR (√2 × pooled 0.00520 / 0.00209) | WT / HO | 68 / 11 | 17 / 3 | 5 / 1 |

- **Independence assumption.** The independent-arm rows assume independent executions per arm; the paired rows use
  measured paired SDs. The 3-bit study has one calibration strategy, so it gives no paired estimate.
- **Spread of the requirement.** The requirement differs by up to ~40× across evaluation sets and bit-widths for the
  same models.

## 9. Answers

1. **Does MoE Family D ranking instability persist on independent text?** Yes.
   - The FR/FC ranking is undetermined: the mean Δ is +0.0004, the hierarchical CI spans 0, the split is 8/7, R1 is
     7/15, and the historical rule gives GROUPING-VARIATION-DOMINATES.
   - The individual comparison signs do not carry over: 8/15 flip.
2. **Does the paired FC − FR execution SD remain substantial?** It remains measurable and many times larger than the
   mean difference: 0.0034 against |M| = 0.0004. But it is **about half** the WikiText value (R_dataset 0.49, band
   0.27–0.89; 0.40 in NLL units). Relative to the MoE quantization penalty it is not smaller: FR SD/Q is 0.18 against
   0.15.
3. **Does dense grouping sensitivity reproduce on unseen text?** Weakly.
   - The spread is present: SD 0.0016, max difference 0.0035, and every window differs.
   - It is **0.31×** the WikiText SD, and SD/Q falls from 5.3 % to 1.8 %.
   - The F(4,4) band (0.10–0.96) only just excludes equality, given n = 5.
4. **Does the low-dose PPL variability persist?** Yes in shape, smaller in level.
   - k = 1 already gives 76 % of the k = 64 spread, and s_k is flat in k (slope 0.06).
   - Levels are about 0.58× WikiText (R_PPL 0.020–0.031).
   - Replicate-level orderings do not replicate.
5. **Does reference-selection sensitivity persist?** Yes. π_ORIGINAL is again the worst reference and a reference
   effect exists (post hoc p = 0.006), but the range halves (0.0054) and the best reference changes. This is
   sensitivity, not a stable, selectable optimum.
6. **How much worse is dense 3-bit than 4-bit?** The penalty is 6.4× on WikiText (0.614 against 0.095) and 5.3× on the
   held-out set (0.471 against 0.088). That is +4.9 % and +2.8 % PPL over BF16.
7. **Does execution SD change at 3 bits?** It rises 2.2× on WikiText (0.0050 → 0.0112) and is about unchanged on the
   held-out set (1.1×). The CIs are wide; neither change is statistically established.
8. **Is SD / penalty stable across bit-widths?** **No.** It falls to 0.35× on WikiText and 0.21× on the held-out set.
   Both paired-bootstrap intervals lie below 1, so the label is B2. Functionally, R_func is unchanged (×0.91); the drop
   arises between function and PPL.
9. **Is that consistent across datasets?** The direction is consistent (B2 on both). The magnitude differs: R ranges
   from 5.3 % (4-bit WikiText) down to 0.37 % (3-bit held-out), a factor of 14.
10. **Can the paper use a common relative uncertainty threshold?** **No**, at least not one normalized by the PPL
    penalty.
    - In the dense model SD/Q varies 14× across bit-width × dataset.
    - In the MoE it is 0.12–0.26 depending on seed pooling and dataset.
    - A per-regime measured execution SD (or a functional-level ratio such as R_func, which was stable across bit-widths
      here) is more defensible.
11. **What replicate requirements does the data support?** Requirements are conditional on dataset, bit-width and model.
    - For the MoE paired FR/FC comparison at δ = 0.005: about 15 execution replicates on WikiText-2, about 4 on the
      held-out set.
    - At δ = 0.0025: 15–60.
    - Dense 3-bit on WikiText would need about 79 per arm at δ = 0.005 (under independence).
    - The defensible recommendation is to **measure execution SD in each evaluation regime with ≥ 5–8 execution
      replicates per arm**, rather than adopting a fixed count.
12. **What is independently validated, and what remains specific?**
    - **Validated on unseen text:**
      - execution-induced PPL dispersion exists in both models;
      - the MoE FR/FC ranking is not determined under grouping variation;
      - single-row perturbations already give most of the dense spread (flat dose response);
      - fixed act-order is not a dependable PPL-variance stabilizer;
      - π_ORIGINAL is a poor reference;
      - bitwise reproducibility under a controlled stack, including the exact G20261115 = ORIGINAL null at 3 and
        4 bits.
    - **Specific to the original setup:**
      - the **magnitudes**: WikiText SDs are 2–3× the held-out SDs;
      - the identity and sign of any individual grouping effect;
      - which reference permutation is "best";
      - any penalty-normalized threshold.
    - **Not tested here:** other models, packed INT kernels, and other numerical stacks (A100 MIG, torch 2.11 cu128,
      eager attention).

## Verdicts

- **Held-out evaluation robustness: A2 (dataset-dependent).**
  - The phenomenon (dispersion, ranking instability, flat dose response, ORIGINAL-worst reference) reproduces on C4
    validation.
  - Absolute dispersion is about 0.3–0.6× the WikiText value: MoE paired 0.49, dense grouping 0.31, dose 0.58.
  - The WikiText-2 windows somewhat overstate execution variance.
- **Bit-width sensitivity: the penalty scales strongly, the PPL spread weakly.** 3-bit raises Q 5–6×. Execution SD
  rises 1.1–2.2× (not established). Functional disagreement scales with the quantization error (R_func ×0.91).
- **Relative uncertainty scaling: not stable (B2 on both datasets).** SD/Q falls 3–5× from 4 to 3 bits and differs 3×
  between datasets at fixed bit-width. Do not use a single penalty-normalized threshold.
- **Method-ranking reproducibility: not reproducible.**
  - The FR vs FC ordering is undetermined on both datasets, and the decision rule gives GROUPING-VARIATION-DOMINATES on
    both.
  - Individual paired signs agree on only 7/15.
  - Dense grouping and dose-replicate orderings do not transfer between datasets.
- **Execution-replicate recommendations: regime-specific.**
  - Measure execution SD per model × bit-width × evaluation set.
  - Treat about 5–8 replicates per arm as a floor for 0.01-PPL comparisons.
  - Finer comparisons need 15–60+ paired replicates (or ~80–300 independent ones for 3-bit WikiText).
  - Report evaluation-set dependence alongside any ranking.

## Deviations and notes

- **Code written after the protocol freeze.** The analysis module (`diagnostics/report_heldout_bitwidth_validation.py`)
  and the optional functional script were written after the freeze but implement the frozen definitions. After the
  first figure pass, two figure-only fixes were made: Fig. 1 seed labels and Fig. 2 bars changed to points. No numbers
  changed.
- **Additions beyond the protocol, labelled.**
  - The reference-matrix ANOVA is post hoc.
  - The functional analysis (§14, optional) was triggered by the 3-bit result. It uses the saved BF16 MLP inputs on the
    **WikiText-2** windows: these are held out from calibration, but they are not the new C4 set.
- **What the uncertainty intervals mean.**
  - Window-bootstrap intervals measure evaluation-text uncertainty with the weights fixed. They do not replace execution
    replicates.
  - With 5 groupings, all dense SD intervals span about 3×.
  - The F bands for the dataset ratios treat the two SDs as independent, but the weights are shared, so the bands are
    descriptive.
- **MoE seed-1/2 ORIGINAL references.** The weights were never stored, so R2 on the held-out set is seed-0 only.
- **MoE non-target check.**
  - A full 14 B-parameter digest was taken at start, after the parity subset, and at end.
  - Per config, a light digest was taken instead.
  - Restore writes only targets, so the end match bounds every intermediate config.
- **Not run:**
  - the strong-path / k = 32 / k = 96 dose families;
  - MoE frozen routing (excluded by design).

## Cost (`cost_results.json`)

| item | value |
|---|---|
| dense GPU stage (BF16, 6 dense 4-bit, 33 dose × 2 policies, 25 reference-matrix models; both datasets where applicable) | 29 min |
| dense 3-bit stage (6 solves + 6 bits=4 parity re-solves + 4 repeat layers + 24 evaluations) | 9 min |
| solve time | about 20 s per 3-bit config |
| MoE stage (32 configs × 2 held-out evaluations + parity) | 90 min |
| MoE per config, median | 156 s (install 84 s, evaluation 27 s) |
| peak CPU RSS | 50.8 GB (MoE) |
| peak GPU | 3.3 GB |
| new GPTQ solves | 6 configs at 3-bit; the 4-bit parity re-solves were used only for comparison |
| new storage | about 3 GB (3-bit weights and codes, held-out windows) in `~/moe_cache_heldout_bitwidth/` |

## Files

- **Code:**
  - `diagnostics/heldout_bitwidth_validation.py` (stages select / audit / protocol / dense / dense3 / dense3_repeat /
    moe / analyze)
  - `diagnostics/report_heldout_bitwidth_validation.py`
  - `diagnostics/heldout_bitwidth_functional.py`
- **Outputs:** this directory.
  - All required JSONs, plus `dense3_functional_stability.json` and `heldout_windows.pt`.
  - `figures/fig1`–`fig7` (PNG and PDF).
  - `results/` holds raw per-window NLL for every evaluated configuration.
  - `logs/`
- **Preserved 3-bit weights:** `~/moe_cache_heldout_bitwidth/dense3/<config>/L<l>.pt` (weights, codes, scales/zeros,
  metadata).
