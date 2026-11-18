# GPTQ integer-code mechanism: code-flip spatial structure and dense vs MoE code disagreement

**Outcomes** (prespecified rules in `runs_integer_code_mechanism/protocol.json`, protocol hash `1e59c47e286b861d`, frozen
before any result existed; one pre-result amendment, see §7):

| question | classification | key number |
|---|---|---|
| Dense code flips: spatial structure (Experiment A) | **`MIXED-SPATIAL-STRUCTURE`** | 4 % of cells clustered, 60 % boundary-flagged |
| Dense vs MoE integer-code disagreement (Experiment C) | **`B-MUCH-LARGER-MOE-CODE-DISAGREEMENT`** | R_q = **17.86** (R_BF16 = 7.62) |

**Headline.**
- **No local clustering.** GPTQ code flips do not form local runs inside a quantization group. Given how many flips a
  (row, 128-column group) cell has, their positions inside the cell are exchangeable.
- **Structure across groups.** Flips concentrate in particular (row, group) cells, at group edges, in columns that
  changed group, and late in GPTQ's processing order (the order act-order uses to quantize columns).
- **MoE disagreement is real at code level.** MoE codes disagree about 18× more than dense codes. Grid refitting
  accounts for about two thirds of the MoE BF16 changes, but not for the code gap.
- **Act-order is the clearest amplifier.** Re-solving with the reference act-order permutation removes 85–92 % of
  code flips in both models.

## 1. Setup

| | dense control | MoE Family D |
|---|---|---|
| model | Qwen2.5-1.5B (rev `8faed761`), layers 1/9/19/27, 12 MLP tensors, all elements | Qwen1.5-MoE-A2.7B, layers 1/8/16/23, **20 / 240** routed experts (60 tensors) |
| source of codes | saved dense GPTQ artifacts from `runs_dense_grouping_scope` (**not re-solved**) | **re-solved** with the unchanged production solver plus a read-only info capture |
| primary comparisons | G20261111–14 vs ORIGINAL (4 single-row perturbations) | G20261111–15 vs ORIGINAL (5 groupings) |
| null / secondary | G20261115 vs ORIGINAL (bitwise identical, 0 flips); 6 grouping pairs | 10 grouping pairs |

**MoE expert manifest** (`moe_expert_manifest.json`). In each layer:
- experts 0 and 30, the check experts;
- the experts with the minimum, median and maximum token counts under historical CTRL_FR_ORIGINAL.

**Spatial statistics** (`moe_calib/code_spatial.py`):
- segment = 128, which equals both the group size and the lazy-batch block size;
- within-segment lag pairs at lags 1–64, all-flip windows of 2/3/4/8, edge counts over the first and last 8
  positions;
- run lengths and inter-flip distances;
- each statistic is computed in original order, the reference's execution order and the variant's execution order.

**Null.** The null is 1000 permutations per cell. Each permutation preserves the flip count of every (row, segment)
cell. An analytic version of the same null and a row-only null serve as cross-checks.

## 2. Validation

| check | result |
|---|---|
| Dense codes: valid permutation, group = position // 128, codes in [0, 15], `scale·(q − zero) → BF16` equals the saved weights bitwise | **72 / 72 tensors** |
| Dense null G20261115 vs ORIGINAL | 0 code flips, 0 BF16 changes |
| MoE replay vs historical Family D BF16 weights (6 configs × 20 experts × 3 projections) | **360 / 360 bitwise** |
| Fixed-act-order replay parity: the natural permutation supplied as an override reproduces the saved weights | **84 / 84** |
| `tests/test_code_spatial.py` (dequant parity, decomposition, null count preservation vs analytic, exec-order and run counts) | 4 / 4 pass |

## 3. Dense spatial structure (Experiment A): `MIXED-SPATIAL-STRUCTURE`

The label comes from **boundary concentration** (60 % of the 96 primary cells), not from adjacency clustering. Only
4 % of cells meet the clustered criterion: gate 0 %, up 0 %, down 12.5 %.

| measure (48 primary cells per order) | original order | ref exec order | var exec order |
|---|---|---|---|
| lag-1 pairs / (row, group) null, median (range) | 1.006 (0.91–1.23) | **1.001 (0.996–1.011)** | **1.001 (0.997–1.022)** |
| lag-64 pairs / (row, group) null, median | 1.05 | 1.000 | 1.000 |
| lag-1 pairs / row-only null, median | 1.01 | **8.27** (up to 36×, L1 down) | 8.27 |
| edge (first / last 8) flips / null | ~1 | 1.3–2.7× for down_proj and L1/L9 gate/up; ~1 for L19/L27 gate/up | same |
| flip rate, group-changed vs same-group columns | — | **15–230×** | — |
| lag-1 ratio by row flip-rate tertile (low / mid / high) | — | 1.009 / 0.998 / 1.001 | — |

**Interpretation.**
- Within a group, flips show no excess adjacency at any lag from 1 to 64.
- The real structure is at the **(row, group) level**: some cells flip heavily and others barely at all. This shows
  only in execution order, where a group's columns sit together.
- The reference and variant execution orders give nearly identical statistics, because only 13 % of dense column
  positions differ between them.
- Lazy-batch blocks coincide with quantization groups, so block and group boundaries cannot be separated.

**Post-hoc** (not prespecified; `diagnostics/integer_code_posthoc.py`, `logs/posthoc_execution_profile.json`,
Fig. 7). The flip rate rises steeply along the reference execution order:

| | first decile | later deciles |
|---|---|---|
| dense gate/up | 0.02 % | ~0.7 % |
| dense down | 0.27 % | ~4 % |
| MoE | ~4 % | ~30 % (falls again in the last decile) |

- **Whole-group shifts are rare.** Cells where all 128 codes flip account for only 0.1–1.7 % of flips.
- **Confound.** Execution position *is* the descending rank of diag(H). "Late" therefore also means "small
  curvature", so this profile is not evidence of error feedback.

**MoE spatial** (secondary, 200 permutations, 600 cells):
- lag-1 pairs / null is 1.038 in execution order;
- the row-only excess is 2.6×;
- edge enrichment is 1.5×;
- 0 / 600 cells meet the clustered criterion.

## 4. Code / grid decomposition and dense vs MoE (Experiment C): `B-MUCH-LARGER-MOE-CODE-DISAGREEMENT`

| metric | dense (4 vs ORIGINAL) | MoE (5 vs ORIGINAL, sampled) |
|---|---|---|
| BF16 element changed | 6.93 % | 52.8 % |
| **integer code changed** | **1.18 %** | **21.1 %** |
| zero-relative level (q − zero) changed | 0.62 % | 9.2 % |
| same code, given BF16 changed | 86.8 % | 65.6 % |
| BF16 change reproduced exactly by "a's codes on b's grid" | 86.8 % | 65.6 % |
| code changed on an identical grid | 6e-6 | 3e-4 |
| scale changed (elements) | 92.7 % | 93.1 % |
| median relative scale difference, among changed scales | ~5e-6 | 0.47 % |
| zero-point changed | 0.72 % | 18.3 % |
| group assignment changed | 0.15 % | 8.4 % |
| act-order position changed (columns) | 13 % | 87 % |

**R_q = 17.86** (gate 52.0, up 51.6, down 5.1); R_BF16 = 7.62.
- The historical dense figure of **1.36 %** included the duplicate G20261115 = ORIGINAL pairs. Matched definitions give
  1.18 % (4 vs ORIGINAL) or 1.49 % (6 grouping pairs).
- The code gap is *larger* than the BF16 gap, so it is not an artifact of the BF16 metric.

**MoE breakdown** (21.1 % overall):

| slice | values |
|---|---|
| projection | gate 25.1 %, up 25.2 %, down 13.0 % |
| layer | L1 11.6 %, L8 29.2 %, L16 21.0 %, L23 22.6 % |
| token tertile (low / mid / high) | 26.5 % / 21.8 % / 15.0 % |
| per expert × projection | min 2.3 %, median 20.6 %, max 42.8 % |

**down_proj sensitivity is model-specific.**
- **Dense:** down_proj is the most code-sensitive projection, 2.57 % vs 0.48 % for gate/up. It peaks at L9 (4.8 %) and
  L19 (3.4 %).
- **MoE:** down_proj is the least sensitive projection (13 % vs 25 %).

**Why this is not attributed to architecture.** The triggers are not matched between the two models:
- the MoE Hessian perturbation is 55–200× larger;
- every MoE calibration sequence is perturbed, against one sequence in dense;
- MoE experts see about 2 tokens per gate/up input dimension, against 43 for dense.

## 5. Fixed-act-order replay (optional controlled check)

Each variant's Hessian is solved with the **reference's** act-order permutation:

| | code flips natural → fixed | BF16 changed natural → fixed | group changes | scale changed |
|---|---|---|---|---|
| dense gate/up | 0.48 % → **0.04 %** | 2.7 % → 0.3 % | 0.08 % → 0 | 90 % → 90 % |
| dense down | 2.57 % → **0.47 %** | 15.5 % → 3.9 % | 0.30 % → 0 | 97 % → 91 % |
| MoE gate/up | 25 % → **3.9 %** | 58 % → 30 % | 10.7 % → 0 | 94 % → 94 % |
| MoE down | 13 % → **3.0 %** | 42 % → 24 % | 4.0 % → 0 | 91 % → 91 % |

- Fixing the permutation removes **85–92 % of code flips** and **all** group reassignments, while scales still refit
  almost everywhere.
- Dense residual flips are spatially exchangeable: lag-1 pairs / null = 1.000.
- **Caveat.** The fixed order still changes the variant's computation, because its grid and error-feedback trajectory
  still differ. The reduction is therefore not a clean attribution.

## 6. What is and is not supported

**Supported:**
- GPTQ code flips are exchangeable within a group, given the group's flip count.
- Flips are concentrated by (row, group) cell and in group-changed columns.
- MoE codes disagree about 18× more than dense codes on the sampled experts.
- Grid refitting explains about two thirds of the MoE BF16 changes.
- Act-order permutation instability is the dominant amplifier of code flips.

**Not supported:**
- That error feedback causes any particular flip.
- That the execution-order gradient comes from order rather than from diag(H) magnitude.
- That edge enrichment reflects solver dynamics rather than grid fitting at group starts.
- Any whole-MoE code rate.
- Any architecture attribution of the dense/MoE gap.
- That fixed-grid rounding sensitivity was measured: it was not measurable here, because the grid refits almost
  everywhere.

**Recommended next experiment: act-order dose-response.**
- In both models, measure code flips per unit trace-normalized Hessian perturbation, under natural vs fixed act-order.
- Run it at matched perturbation sizes.
- A direct error-feedback test, with a frozen permutation *and* a frozen scale/zero, would need a new opt-in solver
  option. Its design is documented but it was not run.

## 7. Deviations

- **Protocol amendment, before any result.** A one-line dict bug in the `dense_align` stage forced a re-freeze:
  v1 `e34a47a0` became `1e59c47e286b861d`. Records are in `logs/protocol_amendment.json` and
  `logs/protocol_v1_superseded.json`.
- **Post-hoc analysis.** The execution-order profile (Fig. 7) is labelled post-hoc and does not enter either
  classification.
- **Report module changed after results, output formatting only:**
  - NaN-aware medians;
  - extra adjacency and MoE spatial summary lines;
  - one figure key-index fix.
- **Not done:** the first-flip propagation analysis and the frozen-grid feedback test.
- **Dense GPTQ was not re-run** for the primary analysis. Production GPTQ is unchanged, and no earlier artifact was
  overwritten.

## 8. Files

- **Code:**
  - `moe_calib/code_spatial.py`
  - `diagnostics/integer_code_mechanism.py` (stages: audit, protocol, dense_align, dense_spatial, moe_replay,
    moe_spatial, fixed_actorder, analyze)
  - `diagnostics/report_integer_code_mechanism.py`
  - `diagnostics/integer_code_posthoc.py`
  - `tests/test_code_spatial.py`
- **Outputs:** everything is under `runs_integer_code_mechanism/`.
  - `summary.md`: full tables and answers to the 10 questions.
  - JSON deliverables: `protocol.json`, `dense_code_alignment_validation.json`, `dense_spatial_statistics.json`,
    `dense_permutation_nulls.json`, `moe_code_replay_validation.json`, `moe_code_difference_analysis.json`,
    `code_grid_decomposition.json`, `dense_moe_code_comparison.json`, `moe_expert_manifest.json`,
    `artifact_inventory.json`.
  - `figures/fig1`–`fig7`.
  - `logs/`: `fixed_actorder.json`, `moe_spatial.json`, `posthoc_execution_profile.json`, `interpretation.json`.
- **Cache:** sampled-expert MoE codes and Hessians are in
  `~/moe_cache_integer_code_mechanism/{moe_codes,moe_hessians}/<config>/L<l>.pt`.
