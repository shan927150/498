# Matched-warm-start validation of routing typicality — November 6, 2026

Follows [`update_11.06.md`](update_11.06.md) §10 (the unmatched typicality
test). All earlier logs and outputs are unchanged. Outputs for this step are in
`runs_typicality_matched/` (`summary.md`, `summary.json`, `warm_starts.json`,
`figures/`, `logs/`).

**Decision (prespecified rule): MATCHED-TYPICALITY-SUPPORTED.**

## 1. Motivation

The unmatched experiment found that typical-half eligibility lowered
Coverage-LP's error by 1.47 % and removed its 1.32 % penalty against Random.
However, each pool × strategy drew its own warm start. Full and typical
Coverage-LP shared only 5.5–7.8 % of their final sequences. The gain could
therefore partly reflect *different initial calibration sets* rather than
the eligibility restriction on the subsequent picks. This experiment holds
the initial state fixed.

## 2. Design

| | Random | Coverage-LP |
|---|---|---|
| Full pool F (2,000) | F-R | F-C |
| Typical half T (1,000 lowest d(s)) | T-R | T-C |
| Random half H (1,000, pool seed 20261106) | H-R | H-C |

- **Fixed:** B = 128, seeds 0/1/2, frozen 4-bit GPTQ (group 128,
  `percdamp = 1.0`, act-order), layers 1/8/16/23, and the same test-split
  evaluation universe (240 experts).
- **Pools:** the pools are those of `runs_typicality/`, reloaded and verified
  by fingerprint. The metric and threshold are unchanged:
  d(s) = mean over layers of L1(p(s,L,·), p(original pool,L,·)); typical means
  d ≤ 0.321.
- **Warm start:** W_s = `random.Random(s).sample(sorted(T ∩ H), 32)`. T ∩ H
  has 479 sequences. It is built from pool membership only, never test data.
  The **same ordered** W_s is used by all six conditions of seed s. Hashes:
  s0 `71f4c1599ef18971`, s1 `1d2b3965567279e4`, s2 `ba99ad010c44de11`.
- **Remaining 96 sequences:**
  - Random draws them uniformly without replacement from its pool minus W_s
    (a warm-start-conditioned random baseline; the unconditioned random
    branch is unchanged).
  - Coverage-LP starts its routed-token counts from W_s and runs the
    existing two rounds of 48. The objective, capacities, LP, exploration,
    rounding and fallback are unchanged.
- **Estimand:** the effect of candidate eligibility for the 96 post-warm-start
  sequences, *conditional* on a common warm start drawn from T ∩ H. It is not
  the unconditional effect of selecting 128 sequences from each pool.
- **Status:** a stronger controlled follow-up, not an independent holdout
  confirmation. The hypothesis came from earlier results on the same test
  split.

## 3. Implementation (implemented + tested)

- **`e3_allocation.py --warm-start-file`**, used for `random` and the LP
  strategies:
  - It loads the seed's ordered ids and hash, and refuses a hash mismatch,
    duplicates, a missing seed, ids outside the eligible candidates, or a
    candidate pool whose fingerprint is not listed in the file.
  - LP strategies get the warm start via `select_lp_utility(...,
    warm_start=...)` and assert that its size equals the schedule's warm size.
  - Random is the warm start plus `rng.sample(pool − W_s, 96)`.
  - Run names carry `_warm-<hash8>`. Summaries record `warm_start_fixed`
    (file, hash, seed, size).
  - Without the flag, behaviour is unchanged (tested).
- **`diagnostics/typicality_warm_starts.py`** builds `warm_starts.json`
  (pools and fingerprints, eligible ids and hash, per-seed ordered ids and
  hashes). It stops if |T ∩ H| < 32.
- **`scripts/run_typicality_matched.sh`** checks artifacts, builds the warm
  starts and runs preflight. It runs seed 0 as a smoke test, then the
  six-way ordered warm-start check, then seeds 1 and 2 (two GPU runs at a
  time). It is resumable and skips only validated outputs.
- **`diagnostics/report_typicality_matched.py`** handles validation
  (`--validate`, `--check-warm SEED`), the summary and four figures.
  `report_typicality.set_metrics` gained the mean effective sources and the
  per-layer set L1.
- **CPU tests:** `tests/test_typicality_matched.py` has 8 tests covering:
  - deterministic intersection warm starts and an infeasible-design error;
  - an identical ordered warm start across all 6 conditions × 2 seeds;
  - exactly B unique sequences, pool restriction, no warm id repeated after
    the warm start, and no tune/test leakage;
  - LP counts initialised exactly from the warm start, with non-zero-based
    ids so the row ↔ id mapping is exercised;
  - a synthetic case where a shared warm start in different pools produces
    different picks;
  - reproducibility, backward compatibility, identical GPTQ settings and
    evaluation universe, and invalid warm-start files.

  **All 87 CPU tests pass** (52 existing, 20 allocation, 7 typicality, 8 new),
  run before the GPU experiment.

## 4. Execution

- **18 / 18 GPU runs complete and valid.** Hardware was an A100 20 GB MIG
  slice; peak memory was 7.8 GB per run, two runs at a time.
- The unmatched runs were **not** reused: their warm starts differ from W_s.
- The six-way ordered warm-start identity was verified for seeds 0, 1 and 2.
- Coverage-LP took 288/288 adaptive picks by LP, with zero fill or fallback.

```bash
python diagnostics/typicality_warm_starts.py         # CPU, builds warm_starts.json
bash scripts/run_typicality_matched.sh               # 18 runs, resumable
python diagnostics/report_typicality_matched.py      # summary + figures
```

## 5. Results

### Reconstruction (mean expert relative error, test split; lower is better)

| pool | Random | Coverage-LP | penalty P (C − R) |
|---|---|---|---|
| Full | 0.004615 | 0.004730 | +114.6e-6 (+2.48 %) |
| **Typical half** | **0.004606** | **0.004613** | **+7.7e-6 (+0.17 %)** |
| Random half | 0.004615 | 0.004697 | +82.3e-6 (+1.78 %) |

### Paired interactions (×1e-6; seed 0 / 1 / 2; mean ± sd)

| quantity | per seed | mean ± sd | seeds < 0 |
|---|---|---|---|
| **I = P_T − P_F** | −113.4 / −91.9 / −115.4 | −106.9 ± 13.0 | 3/3 |
| **J = P_T − P_H** | −84.6 / −85.2 / −54.2 | −74.6 ± 17.7 | 3/3 |
| **D = E(T,C) − E(F,C)** | −111.1 / −118.1 / −120.8 | −116.6 ± 5.0 | 3/3 |
| D / E(F,C) | −2.35 % / −2.50 % / −2.55 % | −2.47 % | 3/3 |

Other paired relative changes (mean over 3 seeds):

| comparison | change | seeds better |
|---|---|---|
| T-R vs F-R | −0.21 % | 2/3 |
| T-C vs T-R | +0.17 % | 0/3 |
| H-C vs F-C | −0.69 % | 3/3 |
| T-C vs H-C | −1.79 % | 3/3 |
| T-C vs F-R | −0.04 % | 1/3 |

Typical eligibility lowers Coverage-LP's error in **every layer and every
seed**: L1 −2.4 to −2.7 %, L8 −2.1 %, L16 −2.5 to −2.8 %, L23 −2.2 to −2.8 %.
Expert p90/p95 follow the same pattern.

### Routing, balance and source diversity (T-C − F-C, per seed)

| seed | Δ error | Δ d (post-warm picks) | Δ set JS | Δ effective sources (median) | Δ Gini of expert tokens |
|---|---|---|---|---|---|
| 0 | −2.35 % | −0.150 | −0.0014 | +8.1 | +0.007 |
| 1 | −2.50 % | −0.152 | −0.0011 | +8.3 | +0.003 |
| 2 | −2.55 % | −0.154 | −0.0013 | +8.6 | −0.001 |

- Full-pool Coverage-LP's post-warm picks are very atypical: d = 0.449,
  versus 0.336 for random picks. Typical eligibility brings them to 0.297.
- Token-count balance is essentially unchanged: Gini 0.043 vs 0.046, and
  median tokens per expert 4345 vs 4330.
- **Reconstruction improves 2.5 % while routed-token balance stays the same,
  so exposure balancing does not explain calibration quality here.**
- Effective source diversity rises at the same time (+8 effective sources per
  expert). Routing representativeness and source diversity therefore
  **remain confounded** in this design.
- Activations were not measured. No claim is made about activation
  distribution mismatch.

Selected-set overlap: T-C vs F-C is 0.258–0.289, i.e. the shared warm start
(0.25) plus very few common picks.

Figures (`runs_typicality_matched/figures/`, PNG + PDF):
1. `fig1_paired_full_vs_typical_coverage`: paired F-C → T-C per seed.
2. `fig2_coverage_penalty_by_pool`: penalty by pool, with seed points.
3. `fig3_atypicality_vs_error`: post-warm-start d(s) vs error, all 18 runs.
4. `fig4_per_layer_effect`: per-layer T-C − F-C.

## 6. Comparison with the unmatched experiment (not pooled)

| experiment | F-C | T-C | F-R | T-R | P_F | P_T | I | J | T-C vs F-C | warm start |
|---|---|---|---|---|---|---|---|---|---|---|
| unmatched (`runs_typicality/`) | 0.004677 | 0.004608 | 0.004616 | 0.004608 | +61.1e-6 | +0.4e-6 | −60.7e-6 | −65.6e-6 | −1.47 % | different per condition |
| matched (this) | 0.004730 | 0.004613 | 0.004615 | 0.004606 | +114.6e-6 | +7.7e-6 | −106.9e-6 | −74.6e-6 | −2.47 % | identical ordered W_s |

- **The matched effect is larger, not smaller:** D_matched / D_unmatched =
  1.68. The earlier gain was not an artefact of differing warm starts.
- **Why F-C gets worse when matched:** with a typical warm start (drawn from
  T ∩ H), full-pool Coverage-LP chases rare-expert tokens with even more
  atypical sequences. Its post-warm-start d is 0.449 here versus 0.399
  unmatched, and its penalty against random doubles (+2.48 %).
- The unmatched numbers stay as exploratory evidence. The two experiments
  share the test split and are not independent confirmations.

## 7. Interpretation

**Outcome A (prespecified: MATCHED-TYPICALITY-SUPPORTED).** D < 0, I < 0 and
J < 0 hold on all three seeds, and |D| = 2.47 % is well above the 0.5 %
practical effect size.

- With the initial calibration state held fixed, restricting Coverage-LP's
  subsequent candidates to routing-typical sequences reduces its error by
  ~2.5 % and removes almost all of its penalty against Random.
- A random half of the pool removes only part of it: H-C is 0.69 % better
  than F-C, and J is consistently negative. So most of the effect is
  specific to the typicality filter, not to pool size.
- **Outcome E is not observed.** Typical Coverage-LP is still slightly
  *worse* than typical Random (+0.17 %, 0/3 seeds). Even among typical
  candidates, count balancing gives no additional benefit at B = 128.

What it supports:
- Routing atypicality of the selected sequences is a meaningful contributor
  to coverage-selection failure under matched initial conditions.
- Exposure (routed-token) balancing does not explain calibration quality in
  this setting.

What it does not establish:
- That routing typicality is the unique causal mechanism, because source
  diversity moves with it.
- Activation-level distribution mismatch.
- Any benefit at the model level (no PPL).
- Results at other budgets, bit-widths or layers.

## 8. Limitations

- The estimand is conditional on W_s drawn from T ∩ H. That warm start is
  itself typical, which likely magnifies how far F-C drifts away from it.
- The test split is the same one that motivated the hypothesis; this is not
  a fresh holdout.
- Three seeds give three paired replicates, so no significance is claimed.
  Experts are not treated as replicates.
- d(s) may proxy text properties such as domain, language or repetition.
  These were not measured.
- Hessian effective rank was not measured.
- The routing profile of all 2,000 candidates is needed to define the pools
  (≈ 357 s full-forward equivalent). Only equal-budget comparisons are made.

## 9. Recommended next experiment (single)

**A soft, representativeness-constrained Coverage-LP on the full pool.** Run
the existing LP with one added linear constraint: the mean d(s) of the 96
post-warm-start picks must be at most the pool median, 0.321, the same
threshold as the typical half. Compare it with T-C, F-C and Random under the
same matched warm starts, at B = 128 with 3 seeds. The constraint level is
fixed in advance.

- If it matches T-C without discarding half the pool, it is a usable
  allocator.
- Because T-C still does not beat Random, a constrained allocator that also
  fails to beat Random would show that count-based allocation adds nothing
  once representativeness is controlled. The project should then report
  typicality as the finding and stop allocation work.
