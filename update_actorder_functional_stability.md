# Functional impact of GPTQ code disagreement and act-order stabilization

Protocol `runs_actorder_functional_stability/protocol.json` (hash `d6202dc2b91f5cd3`) was frozen before any functional or
PPL result and has no amendments. The hashed code was identical at every stage (`logs/code_state.json`). Experiment 1
completed before any Experiment 2 evaluation.

## Outcomes

| question | classification | key numbers |
|---|---|---|
| Experiment 1: does code disagreement overstate functional disagreement? (F1 / F2 / F3) | **`F3-MIXED`** | R_MLP = **3.14** (MoE block vs dense MLP; F1 ≤ 3, F2 ≥ 8.93); projections conflict (functional/code ratio gate 0.36, up 0.18, down **1.98**) |
| Experiment 2: fixed reference act-order | **INTERMEDIATE** | code ×0.15, functional ×0.49, R_SD **0.508** (threshold 0.5), +0.0076 PPL (acceptable) |
| Experiment 2: bucketed act-order, 8 bits (reference-free) | **INTERMEDIATE, no effect** | code ×0.97, functional ×1.08, R_SD 1.09, −0.008 PPL |
| Experiment 2: disabled act-order | **`D-CODE-STABILITY-ONLY`** | code ×0.14, functional ×0.86, R_SD 0.37, **+0.117 PPL** (penalty ×2.2) |

**Headline.**
- **Integer-code counts are a poor proxy for functional and PPL instability.** They track functional error within one
  projection type of one model. They fail across projections, across models and across act-order policies.
- **The ~18× MoE/dense code gap is a 3–14× functional gap.** The exact figure depends on the denominator and on how
  layers are pooled.
- **The sharpest separator is functional difference relative to each model's own quantization error.** MoE grouping
  variants differ by about **0.8×** their quantization error, so they behave almost like independent quantizations.
  Dense variants differ by **0.06–0.37×**.
- **Fixing the act-order permutation is the only stabilizer that works without a large quality cost.** It halves the
  functional disagreement and the descriptive PPL SD. The reference-free bucketed variant does nothing.
- **The five-grouping SD(log PPL) cannot rank policies.** Bucketed variants with unchanged code and functional
  disagreement span R_SD 0.27–1.09.

**Validation** (all mandatory checks pass; `actorder_correctness.json`):
- Both activation-capture forwards reproduce the historical BF16 per-window NLL bitwise.
- All 3,840 MoE expert calls have gate inputs equal to `X[routed tokens]` bitwise.
- The natural re-solve from the saved Hessians reproduces 72/72 historical weight tensors and all six PPLs bitwise.
- Checks A–H pass:
  - **B:** the fixed-order path with each configuration's own permutation is bitwise equal to natural;
  - **C:** bucketed permutations are deterministic;
  - **D:** disabled act-order runs in original column order;
  - **F:** G20261115 equals ORIGINAL within every policy;
  - **G:** only the 12 target tensors change;
  - **H:** repeat evaluation is bitwise.
- At the end, BF16 re-evaluates bitwise and the targets are restored to pristine.

## 1. Experiment 1: functional sensitivity (no GPTQ solves)

**Setup.**
- **Reference and inputs.** BF16 reference activations are captured on the historical WikiText-2 windows (128 × 512).
  These are the actual projection inputs: the MLP/expert input for gate and up, and the real `down_proj` input.
- **Weight sets.** Saved weights only:
  - dense: 5 groupings + ORIGINAL;
  - MoE: Family D FR s0, **all 240 routed experts**.
- **MoE routing.** The MoE aggregate uses shared reference routing (the same selected experts and BF16 probabilities
  for both variants), so the shared expert cancels.
- **Arithmetic.** FP32 with TF32 off, accumulating in float64. Every error is relative to the BF16-original output.

**Primary comparisons**, pooled over layers by squared norms:

| | gate | up | down | MLP / MoE block | MoE routed-only | own quantization error |
|---|---|---|---|---|---|---|
| dense (4 single-row perturbations vs ORIGINAL) | 0.32 % | 0.71 % | 0.61 % | **0.68 %** | — | 4.49 % |
| MoE (5 groupings vs ORIGINAL) | 6.06 % | 6.73 % | 6.08 % | **2.14 %** | 9.87 % | 2.60 % block / 12.0 % routed |

- The dense G20261115 null is exactly 0 in every unit.
- Per-expert MoE MLP error has a median of 11.0 % (p10 6.5 %, p90 13.6 %).
- Low-token experts are only mildly more sensitive: 11.7 % vs 10.5 % for high-token experts (ρ with token count −0.22).

**Dense vs MoE ratios.** These are different quantities; the comparison is exploratory and not a causal decomposition.

| quantity | MoE / dense |
|---|---|
| integer codes (20 sampled experts) | 17.86 (gate 52, up 52, down 5.1) |
| BF16 elements (all experts) | 7.80 |
| functional gate / up / down | 19.0 / 9.4 / 10.0 |
| functional MLP vs MoE block / routed-only | **3.14** / 14.5 |
| SD(log PPL) | **1.20** |

**Post-hoc layer-matched sensitivity** (`logs/posthoc_layer_matched_ratios.json`; not in the classification).
- The pooled ratio is dominated by the layers with large output norms. Per depth-matched layer pair, R_block is
  7.0 / 1.6 / 2.2 / 2.6 and R_routed is 9.5 / 2.2 / 2.9 / 13.1.
- Grouping difference divided by the model's own quantization error, per layer:
  - **dense:** 0.06 / 0.37 / 0.24 / 0.14;
  - **MoE:** 0.42 / 0.87 / 0.79 / 0.83.
- For two errors of equal norm, a ratio of 0.82 implies an error correlation of about 0.66 between variants. Dense
  (ratio 0.15) gives about 0.99.

**Correlations** (descriptive; units share configurations):
- **Within one model:** code vs functional error has Spearman ρ = 0.74 for dense (n = 48) and 0.68 for MoE sampled
  experts (n = 300).
  - Within dense down_proj ρ = 0.99; within dense gate ρ = 0.55.
  - BF16 disagreement and the Hessian perturbation predict at least as well as codes.
- **Across models:** the two models lie on different lines, and down_proj on a separate one (Fig. 1).
- **Pair level:** functional error does **not** predict |Δ log PPL| (dense ρ = −0.13, n = 10; MoE ρ = −0.21, n = 15).

**Curvature and importance** (held-out quadratic forms; Fig. 5).
- **Code flips, not grid refits, carry the functional change.** Median energy shares from flipped elements:

  | | dense | MoE |
  |---|---|---|
  | gate / up | 99 % | 85 % |
  | down | 95 % | 89 % |

  Same-code grid refits contribute little, even though they account for most BF16 changes.
- **Low-impact concentration**, in the low-curvature half of held-out diag(H) columns:

  | | share of flips | share of energy |
  |---|---|---|
  | dense gate / up | 73 % | 73 % |
  | MoE gate / up | 56 % | 64 % |
  | **dense down** | **58 %** | **40 %** |
  | **MoE down** | **55 %** | **34 %** |

- Flips are therefore proportionate to impact for gate/up. **Only down_proj** shows a disproportionate share of flips
  in low-impact directions.
- The diagonal approximation overstates the full quadratic form by 6–21 %, so column contributions are not additive.

## 2. Experiment 2: dense act-order stabilization

**Setup.**
- **Inputs.** 24 configurations (6 policies × 6 configs) re-solved from the **saved historical Hessians**: the same
  128 calibration sequences, with sha256 verified.
- **Solver.** The production solver is unchanged. Policies use only its existing `actorder_override` and
  `actorder=False` switches (`moe_calib/actorder_policy.py`).
- **Groups.** Groups follow execution-order semantics; no static groups.
- **Bucketed key.**
  - frexp mantissa truncated to 8 significant bits;
  - descending order, ties broken by ascending column index;
  - fixed power-of-two boundaries;
  - frozen after inspecting diag(H) only.

**Results** (5 groupings each; code and E_MLP are means of the 4 primary pairs against the policy's own ORIGINAL;
BF16 PPL is 12.5235):

| policy | code | E_MLP | SD(PPL) | SD(log PPL) | R_SD | mean PPL | Δ vs natural | quant error |
|---|---|---|---|---|---|---|---|---|
| natural | 1.18 % | 0.68 % | 0.00503 | 3.99e-4 | 1.00 | 12.6188 | — | 4.49 % |
| **fixed** | **0.18 %** | **0.34 %** | 0.00256 | 2.03e-4 | **0.51** | 12.6265 | +0.0076 | 4.49 % |
| bucketed8 | 1.14 % | 0.74 % | 0.00548 | 4.35e-4 | 1.09 | 12.6110 | −0.0079 | 4.48 % |
| disabled | **0.16 %** | 0.59 % | 0.00189 | 1.48e-4 | 0.37 | **12.7361** | **+0.1173** | **7.12 %** |
| bucketed6 (exploratory) | 1.10 % | 0.76 % | 0.00137 | 1.09e-4 | 0.27 | 12.6215 | +0.0027 | 4.50 % |
| bucketed10 (exploratory) | 1.17 % | 0.74 % | 0.00241 | 1.91e-4 | 0.48 | 12.6148 | −0.0040 | 4.49 % |

**Grid and order changes** (vs the policy's ORIGINAL):

| | act-order positions changed | group reassignment | zero-point changes |
|---|---|---|---|
| natural | 13.3 % | 0.153 % | 0.72 % |
| fixed / disabled | 0 | 0 | ~0.07 % |
| bucketed8 | 10.8 % | 0.156 % | 0.71 % |

Scales still refit on 89–93 % of elements under every policy.

**Window-paired quality vs natural** (nats/token, bootstrap 95 % CI over 128 windows):

| policy | per-grouping mean difference | CI |
|---|---|---|
| fixed | 0 to +0.0013 | excludes 0 for 3 of 4 perturbations |
| disabled | +0.0088 to +0.0101 | excludes 0 for all groupings |
| bucketed8 | −0.0003 to −0.0013 | — |

**Why bucketing failed.**
- At 8 bits only 0.2–1.1 % of bucket keys change between groupings.
- But each changed key moves a column across a bucket boundary, and ties are then ordered by index.
- As a result, group-boundary crossings, the instability that matters, are not reduced.

**Statistical limitation.**
- The five groupings are four single-row perturbations plus an exact null, so every SD is descriptive.
- The F(4,4) 95 % band for an SD ratio is about [0.32, 3.1], and every observed R_SD lies essentially inside it.
- The bucketed variants show this directly: unchanged code and functional disagreement, yet R_SD 0.27–1.09.

## 3. Conclusions by dimension

- **Code-level stability.**
  - Fixed and disabled act-order remove act-order changes and group reassignment, leaving about 15 % of the natural
    flips.
  - Bucketed act-order does not reduce flips at any precision tested.
  - MoE code disagreement is 18× dense on the sampled experts.
- **Functional stability.**
  - Fixed act-order halves dense MLP disagreement; disabled barely helps (×0.86); bucketed does not help.
  - MoE variants sit at about 0.8× their own quantization error, dense at 0.06–0.37×.
- **PPL stability.** Descriptive R_SD is 0.51 for fixed, 0.37 for disabled and 0.27–1.09 for bucketed. None is
  distinguishable from five-grouping SD noise. Dense and MoE SD(log PPL) differ only 1.2×.
- **Quantization quality.** Fixed +0.0076 PPL (acceptable), bucketed8 −0.008, disabled +0.117 (penalty ×2.2,
  unacceptable).

**Practical vs diagnostic.**
- **Practical (conditional): fixed reference act-order.** It needs a canonical reference calibration run and 12 stored
  permutations, so it is not reference-free. Its PPL benefit is not statistically established.
- **Diagnostic only:**
  - code-flip counts as a stability metric;
  - bucketed act-order;
  - disabled act-order;
  - any R_SD ranking among policies.
- **Hypotheses:**
  - **A** (mostly low-impact flips) is supported only for down_proj.
  - **C** (act-order as a practical control) is partially supported for fixed order.
  - **D** (code stability without useful reproducibility) holds for disabled act-order.
  - **B** (saturation) is not tested here.

**Recommended next experiment: dense dose-response.**
- **Design:** vary the number of perturbed calibration rows (1, 4, 16, 64, 128) under natural vs fixed act-order.
- **Replicates:** at least 8 independent perturbations per dose.
- **Measure at each dose:**
  - E_MLP;
  - E divided by the quantization error;
  - SD(log PPL).
- **What would separate the hypotheses:**
  - **Saturation** predicts that E / quantization error approaches about 0.8 (the MoE value) while SD(log PPL)
    plateaus.
  - **Functional insensitivity** predicts that SD rises roughly in proportion to E.

## 4. Deviations

- **Protocol and code.** No protocol amendment. The hashed code did not change after the freeze.
- **Additions after results:**
  - an `artifact_inventory.json` builder in the report module;
  - the post-hoc layer-matched ratio script (`diagnostics/actorder_functional_posthoc.py`), labelled post-hoc and kept
    out of the classification.
- **MoE scope.**
  - Code, act-order and Hessian metrics cover only the 20 sampled experts.
  - The MoE aggregate is a shared-routing diagnostic, not deployment routing.
  - The block denominator includes the unquantized shared expert; the routed-only figure is reported alongside.
- **Experiment 2 scope.** Dense only, one calibration seed. The optional static-group analysis (§15) was not run.

## 5. Files

- **Code:**
  - `moe_calib/functional_diff.py`
  - `moe_calib/actorder_policy.py`
  - `diagnostics/actorder_functional_stability.py` (stages: audit, protocol, capture_dense, capture_moe, func_dense,
    func_moe, curvature, ao_solve, ao_func, analyze)
  - `diagnostics/report_actorder_functional_stability.py`
  - `diagnostics/actorder_functional_posthoc.py`
  - `tests/test_actorder_functional.py` (7 tests, all passing)
- **Outputs:** `runs_actorder_functional_stability/`
  - `summary.md`: answers to the 10 questions;
  - the required JSONs;
  - `figures/fig1`–`fig6`;
  - `logs/`.
- **Cache:** `~/moe_cache_actorder_functional_stability/`, sha256-inventoried in `artifact_inventory.json`:
  - `act/` (activations): 11 GB;
  - `ao/` (policy weights and codes): 17 GB;
  - `moe_bf16/` (MoE BF16 expert weights): 3.9 GB.
