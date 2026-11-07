# Project Update — November 6, 2026 (minimal fixed-budget allocation)

Follows the [10/29 update](update_10.29.md) (P0 → P2c). That history is
unchanged; in particular the **dynamic-state negative result (P2 CASE B,
P2b UNSTABLE) stands**.

Status legend used below: *implemented* (code exists) · *tested* (CPU unit
tests pass) · *evaluated* (real GPU runs on Qwen1.5-MoE-A2.7B finished) ·
*not evaluated*.

---

## 1. Previous status

| stage | result |
|---|---|
| P0 | `percdamp = 1.0` frozen (`configs/frozen_b4.json`) |
| P1 | target `future_rel_delta` PASS |
| P2 / P2b | dynamic state signals **not robust** (CASE B / UNSTABLE) |
| P2c | layer- and regime-aware exposure models predict held-out benefit better: M4 −30.3 % LOSO MSE vs global flexible exposure (M1), −22.5 % vs layer intercepts; Dlb (layer × budget) almost as good |
| Exp1 (within-layer ranking) | layer-aware utility orders experts *within* a layer exactly like coverage 1/n; its P2c ranking gain was entirely cross-layer |
| Exp4 (source diversity) | more distinct source sequences at matched routed-token count lowers expert error (e.g. spread/natural 0.970 at m = 32, 0.988 at m = 128; > 99 % of experts) |
| open | no learned utility had been tested through actual whole-sequence selection |

Question for this step: **can layer- and regime-aware marginal calibration
utility improve actual sequence selection over random and routed-token
coverage at a fixed calibration budget?**

## 2. What was implemented (implemented + tested)

**`moe_calib/alloc_models.py`** (new)
- `design()` — the P2c ladder design matrices, now the single implementation
  (`diagnostics/p2c_layer_exposure.py` calls it; its LOSO MSEs are reproduced
  to all printed digits: M1 7.0014e-05, Dlb 4.9168e-05, M4 4.8772e-05).
- `fit_export()` — fits M1 / Dlb / M4 on the full P2 table and exports a
  controller artifact: coefficients, column schema, train-only log-count
  standardization, characterized budgets and layers, per-(layer, budget)
  routed-count support, characterization seeds, frozen GPTQ / split / model,
  training-table SHA-256, E1 run dirs, and a content fingerprint.
- `predict_marginal_benefit(art, model, layer, count, m)` and
  `compute_expert_utility(strategy, n, layers, E, m, art)`.

**`diagnostics/export_allocation_models.py`** (new) → `runs/allocation_models_b4_d1.json`
(fingerprint `cbcd70db8f47d3dd`, 2,880 observations, char. seeds 0/1/2). It
checks every E1 run against the frozen config and the table's seeds / layers /
budgets against those runs. Test-split data are never read.

**`e3_allocation.py`** — four strategies on **one shared controller**
(`select_lp_utility`): tlp's warm start (`round_schedule`: 32 random, then two
rounds of 48), exploration rule, HiGHS LP, positive-marginal rounding, seeded
random fill. Only δ̂ differs:

| strategy | δ̂_e | LP value u_e |
|---|---|---|
| `coverage_lp` | 1 | 1 / n_e |
| `global_exposure` | f_M1(n_e, m) | max(δ̂, 0) / n_e |
| `layer_budget` | f_Dlb(n_e, L, m) | max(δ̂, 0) / n_e |
| `layer_aware` | f_M4(n_e, L, m) | max(δ̂, 0) / n_e |

LP capacity is n_e (the doubling the labels measure), weights uniform. The
existing greedy `coverage`, `random`, `tlp`, `oracle` are unchanged.
New flags: `--frozen-config` (validates model, bits, group size, damping,
layers, split, act-order; required by the new strategies), `--alloc-models`
(artifact, validated by fingerprint and frozen config), `--routing-cache`
(reuse `routing_profile_v2.npz`; cost is still charged as 2,000 full-model
forwards). n_e is read from the pool routing matrix over the selected set; every
run asserts it equals the Hessian token count of every scored expert (it did,
in all runs).

**Regimes.** `m` = selected sequences; `n_e` = routed tokens. Decisions are made
at m = 32 (characterized) and m = 80, which is between 64 and 128. The
prespecified policy interpolates **linearly in log m at fixed n_e**:
w = log(80/64)/log 2 = 0.32. There is no extrapolation outside 16 ≤ m ≤ 128.
An expert is predicted only if lo(L, b_lo) ≤ n_e ≤ hi(L, b_hi). Otherwise:
below the range → exploration target, above it → no utility, n_e = 0 →
exploration. All of these are logged per round.

**Tests** (`tests/test_allocation_minimal.py`, 20 tests, all pass; the 52
existing tests also pass):
- **Regression.** Exported predictions equal P2c's `fit()` (synthetic data
  and the real table). Standardization uses training rows only. Fits are
  deterministic. Layer-major indexing is correct. The interpolation formula
  holds. Out-of-range budgets, uncharacterized layers or budgets, invalid or
  zero counts, and below/above support never get a value.
- **Artifact guards.** An edited coefficient, a different damping, a missing
  layer, or a missing artifact or frozen config each fail.
- **Allocation.** Each strategy selects exactly B unique pool ids, never a
  tune/test id. The same seed gives the same set. All LP strategies share the
  warm start, which also equals greedy coverage's warm start at B = 128. The
  m schedule is [4, 8] on the tiny model, with an interpolated bracket.
  Every GPTQ call uses (4, 128, 1.0, act-order).
- **Routing cache.** Expert order equals the profiler's.
- **Synthetic control.** `coverage_lp` values are exactly 1/n. `layer_budget`
  values differ across layers. Two utilities with the same within-layer
  order but different cross-layer magnitudes pick different sequences.

## 3. Protocol

B = 128; seeds 0, 1, 2; 4-bit GPTQ, group size 128, `percdamp = 1.0`, act-order;
layers 1/8/16/23; 2,000-sequence C4 pool; scoring = mean per-expert relative
reconstruction error on the fixed 128-sequence **test** split (all 240 experts
in the evaluation universe; one universe file shared by every run). Strategies:
random, coverage (greedy), coverage_lp, global_exposure, layer_budget,
layer_aware. Random was **rerun** (not reused) so that K, outdir and code path
match. Its seed-0 error, 0.0046060696, agrees with the earlier headroom run
(0.0046060681) to within float summation order. No PPL, no budget sweep, no
3-bit. The decision rule was written into `diagnostics/report_minimal_allocation.py`
before any result was read.

```bash
python diagnostics/export_allocation_models.py          # CPU, seconds
SEEDS=0 bash scripts/run_minimal_allocation.sh           # smoke: seed 0, all strategies
SEEDS="1 2" PAR=2 bash scripts/run_minimal_allocation.sh # remaining seeds, 2 runs at a time
python diagnostics/report_minimal_allocation.py          # summary.md / summary.json
```

## 4. What was executed (evaluated)

All **18 / 18** strategy × seed runs finished and pass validation:
B = 128 unique pool ids, frozen settings, 240 experts quantized and scored,
a shared evaluation universe, and routing counts equal to Hessian tokens.
Hardware was one A100 MIG slice (20 GB); peak memory was 7.8 GB per run.
Seed 0 ran first as the smoke test. Seeds 1 and 2 ran two at a time, so
their Hessian/GPTQ wall-clock times are inflated by contention.

Full report: `runs_allocation_minimal/summary.md` / `summary.json`. Per-run JSONs
(`e3_<strategy>_B128_b4_d1_s<seed>.json`) contain the selected and warm-start
ids, per-expert errors, tokens and selected routed counts, the per-round
controller log, and costs.

**No adaptive controller decision needed a fallback.** In every round:
- 0 experts were zero-exposure, below support or above support;
- 0 exploration sequences, 0 random fill and 0 utility-fallback sequences
  were taken;
- 100 % of the 96 adaptive picks were LP-selected.

## 5. Results

### Reconstruction (mean expert relative error, test split; lower is better)

| strategy | mean (3 seeds) | sd | seed 0 / 1 / 2 | vs random | vs coverage | vs coverage_lp |
|---|---|---|---|---|---|---|
| **random** | **0.004616** | 9.9e-06 | 0.004606 / 0.004616 / 0.004626 | – | −1.43 % | −1.31 % |
| coverage (greedy) | 0.004683 | 5.4e-06 | 0.004677 / 0.004683 / 0.004688 | +1.45 % | – | +0.13 % |
| coverage_lp | 0.004677 | 9.2e-06 | 0.004669 / 0.004687 / 0.004676 | +1.32 % | −0.13 % | – |
| global_exposure (M1) | 0.004666 | 2.4e-05 | 0.004639 / 0.004675 / 0.004685 | +1.09 % | −0.36 % | −0.23 % |
| layer_budget (Dlb) | 0.004700 | 1.3e-05 | 0.004706 / 0.004686 / 0.004709 | +1.82 % | +0.36 % | +0.49 % |
| layer_aware (M4) | 0.004678 | 2.7e-05 | 0.004654 / 0.004674 / 0.004707 | +1.35 % | −0.10 % | +0.02 % |

Expert p90 and p95 show the same order. Random has the lowest p95 (0.00622);
the adaptive strategies range from 0.00629 to 0.00634.

### Paired seed differences

| comparison | seed 0 / 1 / 2 (relative) | mean | seeds better |
|---|---|---|---|
| **layer_aware − coverage_lp** | −0.32 % / −0.28 % / **+0.67 %** | +0.02 % | 2/3 |
| layer_budget − coverage_lp | +0.79 % / −0.03 % / +0.71 % | +0.49 % | 1/3 |
| global_exposure − coverage_lp | −0.64 % / −0.26 % / +0.21 % | −0.23 % | 2/3 |
| layer_aware − layer_budget | −1.10 % / −0.25 % / −0.04 % | −0.46 % | 3/3 |
| coverage_lp − coverage (greedy) | −0.19 % / +0.08 % / −0.27 % | −0.13 % | 2/3 |
| every adaptive strategy − random | +0.71 % to +2.16 % | +1.09 % to +1.82 % | **0/3 for each** |

- Against random, every adaptive strategy is worse in every seed and in
  every layer: +0.7 % to +2.0 % per layer.
- It is also worse for almost every expert. Random gives the lower error for
  93–100 % of experts, depending on strategy and seed.
- The layer-aware vs coverage_lp difference changes sign across seeds, and
  its per-layer differences are within ±0.1 %.

### Selection overlap and routing

- **The learned utilities do change the selection.**
  - No learned strategy picked the same set as coverage_lp in any seed.
    Overlap is 0.38–0.49 of the 128 sequences, including the shared
    32-sequence warm start.
  - Of the 96 adaptive picks, only 18–32 % are shared.
  - Per-layer L1 distance between the routed-count distributions and
    coverage_lp's: 0.03–0.05.
- **The utility magnitudes change across layers as P2c predicts.** At
  m = 32, layer_aware's mean δ̂ is 0.062 for L1 versus 0.051 for L23, so the
  LP objective is weighted toward L1. At m = 80 (interpolated) the order
  flips: L8/L16 0.043–0.044 versus L1 0.040. Global exposure is flat across
  layers (0.057 at m = 32, 0.042 at m = 80).
- **None of this translated into lower error.** All adaptive strategies,
  learned or not, lose to random by a similar margin.

### Why adaptive selection loses (descriptive, CPU-only diagnostics)

- **Counts barely change.** Adaptive selection moves an expert's routed
  tokens by a median of only ≈5 % relative to random (|log ratio| 0.046–0.051).
  - About 54 % of experts get *more* tokens than under random, yet 96–100 %
    of those experts end up *worse*.
  - Corr(log token ratio, log error ratio) ranges from −0.08 to +0.18.
  - The count change cannot explain the error change. Per P1 a full doubling
    is worth ≈3–4.5 %, so a 5 % count shift is worth ≲0.2 %, while the
    observed loss is 1.1–1.8 %.
- **The selected sequences are atypical.**
  - The 96 adaptive picks have per-sequence routing distributions much
    farther from the pool's: L1 0.40–0.42, versus 0.33 for random picks and
    a 0.34 pool mean.
  - Their routing entropy is lower: 3.93–3.95 versus 3.99 nats.
  - Layer_budget is the most atypical of the learned strategies and the
    worst. Global exposure is among the least atypical and the best adaptive
    strategy.
  - Routing JS divergence to the pool is 3–4.5× random's.
- **Mechanism (hypothesis, consistent with Exp4).**
  - Utilities of the form 1/n, or f/n, reward sequences that route many
    tokens to under-covered experts. Such sequences are unusual text.
  - They shift the *within-expert* activation distribution away from the
    natural test distribution. That costs more than the small count
    rebalancing gains.
  - Exposure-only utilities cannot see this, whatever their count curve.

### Compute

- Every adaptive strategy needs the pool routing profile. That costs 2,000
  candidates × 0.178 s = **357 s of full-model-forward equivalent**, i.e.
  ≈ 2,000 extra random sequences, on top of the LP selection (≈1.2 s).
- Hessian replay plus GPTQ is ≈ 4 min per run on an uncontended slice. It is
  the same for every strategy.
- Equal compute was **not run**. Random already wins at equal *sequence*
  budget, and the existing random headroom curve shows random at B = 512
  (0.00439) far below every adaptive strategy at B = 128.

## 6. Decision: **ALLOCATION-NO-GAIN**

The rule was prespecified, with 0.5 % as a practical effect size.

- **Layer-aware vs coverage_lp:** +0.02 % mean (−0.32 / −0.28 / +0.67 %),
  better on 2 of 3 seeds, with no consistent per-layer or tail effect. This
  is **Outcome B**: the learned utility changes which sequences are selected
  and how routing is distributed, but not reconstruction quality.
- **Layer-budget vs layer-aware:** layer_budget is worse (Outcome D does not
  occur). layer_aware beats it on all 3 seeds (−0.46 %), and layer_budget is
  the worst LP strategy.
- **Against random**, every adaptive strategy loses on every seed. This
  includes the existing greedy coverage baseline (+1.45 %). In this setting
  that is the dominant result.

## 7. What the results support

- At B = 128, frozen 4-bit GPTQ, expert-level reconstruction, routing-driven
  sequence selection from the C4 pool is **worse than random**. This holds
  for coverage of every kind (greedy, LP) and for every exposure-only learned
  utility (M1, Dlb, M4).
- P2c's improvements in offline prediction and pooled ranking **do not
  transfer** to whole-sequence selection: the learned utility changes the
  selected sets but not the outcome. Offline prediction gains are not
  allocation gains.
- Controller choice (greedy vs LP) matters little: coverage_lp is −0.13 %
  versus greedy, 2 of 3 seeds.
- Calibration quality depends on *which* sequences are chosen (their
  typicality), not only on routed-token counts. This agrees with the Exp4
  source-diversity result.

## 8. What the results do not support / limits

- No model-level (PPL) claim either way. PPL was not run, by design.
- Only B = 128, 4-bit, 4 layers, one candidate pool and one test split were
  tested. Low budgets, where P2c's ranking gain was largest (n0 = 16), were
  not tested.
- The atypicality mechanism is a descriptive correlation across six
  strategies × three seeds. It has not been tested causally.
- Three seeds give three paired replicates; no significance is claimed.
- The u_abs ablation (below) and diversity-aware utilities were **not
  implemented**.

### Cross-layer comparability (documented future ablation, not run)

- The utility here is relative: u_rel = max(δ̂_rel, 0)/n.
- A comparable-across-layers variant would be
  u_abs = max(δ̂_rel · ê_base, 0)/n. ê_base would be a *training-only*
  per-expert base-error estimate, for example err(n0) on the tuning split
  from E1 labels. The test split must never be used.
- It is not implemented. The P2 table has no base-error column, and no
  validated estimator exists.
- Given the NO-GAIN result against random, a re-weighting of the same
  exposure-only utility is unlikely to help. It would also select from the
  same atypical sequences.

### Source-diversity follow-up (analysis only)

Exp4 showed that more sources at matched token count help, but it did not
hold the number of whole selected sequences fixed. Today's result adds that,
at fixed B, utilities that chase rare-expert tokens pick atypical sequences
and lose. A feasible fixed-B, diversity- or representativeness-aware utility
would need one of the following:

| signal | available now (routing cache / CPU) | needs new GPU work |
|---|---|---|
| routing typicality (L1/JS of a sequence's routing vs pool), as a penalty or constraint | **yes**: `routing_profile_v2.npz` | – |
| routing-profile diversity among selected sequences (e.g. DPP / facility location on per-sequence routing vectors) | **yes** | – |
| per-expert activation novelty (incremental Hessian / covariance gain of a candidate) | no | per-candidate per-expert activations over the 2,000-sequence pool |
| low-rank activation coverage (projection of candidate activations onto the current Hessian's top eigenspace) | no | same, plus eigendecomposition per round |
| non-redundant routed activations (token-level dedup or k-center in activation space) | no | token activations for the pool (large; not cached) |

## 9. Recommended next experiment (single)

**A typicality-controlled allocation test.** Prespecify a filtered candidate
pool: the sequences whose routing L1 to the pool is at most the pool median.
This is computable now from the routing cache. Then run `random` and
`coverage_lp` from that filtered pool at B = 128, seeds 0/1/2, with
everything else unchanged.

- **If coverage_lp still loses to random from the typical pool:**
  count-rebalancing itself hurts under this metric. Stop exposure-based
  allocation and report the negative result.
- **If it matches or beats random:** the loss is due to atypical-sequence
  selection. A representativeness-constrained utility is the justified next
  method.

Do not run budget sweeps, PPL, 3-bit, or more complex layer curves before
this.

---

## 10. Follow-up (same day): routing-typicality test — **TYPICALITY-SUPPORTED**

**Hypothesis.** Coverage-style allocation over-selects sequences whose routing
departs from the natural pool distribution. That selection bias, not
count balancing, explains the loss to random.

**Design.**
- Pools:

  | pool | definition |
  |---|---|
  | Full | 2,000 sequences (runs reused from §5 after validation: same frozen config, routing cache SHA, evaluation universe) |
  | Typical half | the 1,000 sequences with the lowest d(s) |
  | Random half | 1,000 sequences drawn with pool seed 20261106, independent of the experiment seeds |

- Strategies: random and coverage_lp, with B = 128 and seeds 0/1/2. The only
  change is candidate eligibility (`--candidate-subset-file`).
- d(s) = mean over layers 1/8/16/23 of L1(p(s,L,·), p(original pool,L,·)),
  computed from the corrected routing cache only. The metric was fixed before
  any run. The original pool is the reference for every condition.
- Pool d(s) means: full 0.337, typical 0.288 (threshold d ≤ 0.321), random
  half 0.340. No malformed rows. The typical and random halves share 479
  sequences.

**Implementation (implemented + tested).**
- New code: `moe_calib/typicality.py`, `diagnostics/typicality_pools.py`,
  `e3_allocation.py --candidate-subset-file`, `scripts/run_typicality_experiment.sh`
  and `diagnostics/report_typicality.py`.
- The subset file is validated: only original pool ids, a fingerprint, and
  B ≤ |subset|. tlp/oracle are rejected because their legacy path indexes
  routing rows by id.
- Routing rows are sliced from the full-pool cache by original id.
- 7 new tests: scores, ties and zero rows, layer order, subset confinement,
  LP row ↔ id mapping, reproducibility, and a full-pool subset reproducing
  the no-flag run exactly. All 79 tests pass.

**Executed (evaluated):** 12/12 new runs, all validated, plus the 6 reused
full-pool runs. Outputs are in `runs_typicality/` (`summary.md`,
`summary.json`, `pools.json`, `typicality_scores.csv`, `pool_diagnostics.json`).

| pool | random | coverage_lp | coverage penalty P (×1e-6), seeds 0 / 1 / 2 |
|---|---|---|---|
| full | 0.004616 | 0.004677 | +62.6 / +70.9 / +49.8 (+1.32 %) |
| **typical half** | **0.004608** | **0.004608** | −2.1 / +2.9 / +0.3 (+0.01 %) |
| random half | 0.004622 | 0.004687 | +72.0 / +59.5 / +66.3 (+1.43 %) |

- **Interaction I = P_T − P_F:** −64.7 / −68.1 / −49.5 (×1e-6), negative on
  all seeds.
- **Pool-size control P_T − P_H:** −74.1 / −56.7 / −66.0, negative on all
  seeds.
- **Halving the pool at random does nothing:** P_H − P_F = +4.8 mean, with
  mixed signs.
- **Typical coverage_lp vs full coverage_lp:** −1.47 %, 3/3 seeds, every
  layer.
- **Typical random vs full random:** −0.17 %, 3/3 seeds, which is below the
  0.5 % practical effect size.
- **Typical coverage_lp vs typical random:** +0.01 %, so count balancing is
  neutral, not helpful.
- Expert p90/p95 follow the same pattern; typical coverage_lp has the lowest
  p90 (0.005993).

**Mechanism checks (all hold on all 3 seeds).**
1. Typical filtering lowers the selected sets' d(s): random 0.335 → 0.287,
   coverage_lp 0.386 → 0.296.
2. Coverage_lp's adaptive picks become typical: d 0.399 → 0.297.
3. Coverage_lp's set JS to the original pool falls from 0.00070 to 0.00031.
4. Reconstruction improves.
5. The coverage penalty shrinks, and by more than with the random half.

Other observations:
- Effective sources per expert rise (coverage_lp 113.8 → 120.5).
- Low-exposure experts: 0 everywhere.
- Count inequality (Gini) is *higher* for typical random (0.065) than full
  random (0.053), yet error is lower.

**What this supports.**
- At B = 128 the loss of routing-driven selection is explained by selecting
  routing-atypical sequences. A random pool reduction does not reproduce the
  effect; restricting to routing-typical candidates removes the penalty
  entirely.
- This is an intervention on candidate eligibility, with a size-matched
  control. It is causal evidence for the *eligibility* manipulation, at this
  budget and setting.

**What it does not support.**
- Count balancing does not become beneficial within typical candidates. It is
  merely harmless (P_T ≈ 0).
- Per-sequence typicality, not matching the pool's aggregate routing, is what
  matters. Typical-pool random sets have *higher* aggregate JS to the pool
  (0.00048) than full random (0.00024) because rare-expert-heavy sequences
  are excluded, but they still reconstruct better.
- d(s) may proxy other text properties (domain, language, repetitiveness).
  Those were not measured.
- Hessian effective rank was not measured (it needs GPU Hessian replays per
  condition).
- No PPL, other budgets, or 3-bit runs were done. Equal budget only: every
  filtered condition still needs the full-pool routing profile, about 357 s
  of full-forward equivalent.

**Recommended next experiment (single).** A typicality dose-response with
plain random selection:
- Arms: random from the lowest-d 25 % and 50 %, from the full pool, and from
  a size-matched random 25 % control.
- B = 128, seeds 0/1/2, with the quantiles prespecified.
- This tests whether typicality filtering is itself a useful selection rule.
  Today's gain is only −0.17 %, so check whether a stronger filter helps or
  starts hurting diversity, before building any typicality-constrained
  allocator.
