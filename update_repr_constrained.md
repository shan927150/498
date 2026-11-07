# Representativeness-constrained Coverage-LP — November 7, 2026

Follows [`update_matched_typicality_11.06.md`](update_matched_typicality_11.06.md).
All earlier results and logs are unchanged. Outputs of this step are in
`runs_repr_constrained/`: `summary.md`, `summary.json`, `feasibility.json`,
`tail_check.json`, `covariance/` and `logs/`.

**Scientific decision: REPR-ALLOCATION-NO-GAIN.**

The prespecified rule coded in `diagnostics/report_repr_constrained.py` prints
`PARTIAL`, because it only requires a ≥ 0.5 % gain over Full Coverage-LP on
3/3 seeds. The task's definition of PARTIAL also requires the new method to
*match* Typical Random. It does not: it is +1.03 % worse on 0/3 seeds. Under
the task's definitions the outcome is therefore NO-GAIN. The coded rule was
not changed after seeing the results; both labels are recorded.

## 1. Research motivation

The matched-warm-start experiment showed that hard Typical-half eligibility
removes most of Coverage-LP's penalty: −2.47 % vs Full Coverage-LP on 3/3
seeds. However, Typical Coverage-LP was still +0.17 % worse than Typical
Random.

Hard filtering discards half the pool. The question here is whether a *soft*
representativeness constraint can do better: keep the full pool, but bound
the atypicality of the selected set. The aim is to keep the exposure-balancing
benefit while limiting distribution mismatch, and so beat Random at B = 128.

## 2. Constrained allocation objective

Per adaptive round, there is the same LP as `coverage_lp` plus one row
(`e3_allocation.solve_lp_constrained`):

```
maximize   sum_e u_e y_e                     u_e = 1 / n_e   (n_e: routed tokens of the selected set)
s.t.       y_e <= sum_s C[s,e] z_s,   0 <= y_e <= n_e
           sum_s z_s = round_budget,  0 <= z_s <= 1
           sum_s d(s) z_s <= round_cap        (new)
```

The scientific constraint applies to the final 96 post-warm sequences:
mean d(s) ≤ τ. Equivalently, there is a shared budget A = 96 τ across both
rounds. The warm start is fixed and consumes nothing.

## 3. Typicality definition (unchanged)

```
p(s,L,e)    = C[s,L,e] / sum_j C[s,L,j]
p(pool,L,e) = sum_s C[s,L,e] / sum_s sum_j C[s,L,j]    (original 2,000-sequence pool, fixed)
d(s)        = mean_L sum_e |p(s,L,e) - p(pool,L,e)|    (equal layer weights)
```

- d(s) is computed by `moe_calib.typicality.sequence_scores`, the function
  that built the Typical-half pool.
- It is recomputed from `routing_profile_v2.npz`, whose sha256 matches the
  pool provenance.
- It agrees exactly with `runs_typicality/typicality_scores.csv` (max abs
  diff 0.0) and reproduces the Typical-half membership.
- Each GPU run re-verifies d(s) against the CSV (`--typicality-scores`).
- Only candidate-pool routing is used. No tune or test data are involved.

## 4. Threshold and feasibility

- **τ = median d(s) over the 2,000 candidates = 0.320968.** It is fixed by rule
  before any reconstruction (`--repr-tau median`) and was never tuned.
- **CPU feasibility check** (`diagnostics/repr_feasibility.py`, run before the
  GPU runs): for every warm start, the cheapest 96 completions are far below
  the budget, so the constraint is feasible.
- The check also dry-runs the exact selection: the controller is CPU-only and
  uses the same rng seed.
- It also replays unconstrained `coverage_lp`, which reproduces the stored
  matched Full Coverage-LP selections **exactly** for all three seeds.
- **Final discrete constraint:**

| seed | mean post-warm d | τ | tightness (spent/budget) | post-warm picks outside Typical half |
|---|---|---|---|---|
| 0 | 0.32023 | 0.32097 | 0.9977 | 37 / 96 |
| 1 | 0.32093 | 0.32097 | 0.9999 | 44 / 96 |
| 2 | 0.32015 | 0.32097 | 0.9974 | 41 / 96 |

The constraint binds, and 42 % of the post-warm picks lie outside the
Typical-half pool. The method is not equivalent to hard filtering.

## 5. Implementation

- **`moe_calib/repr_constraint.py`** (new) contains:
  - `pool_typicality` and `resolve_tau`;
  - **`ReprBudget`**, the cumulative accounting;
  - **`select_round_repr`**, the constrained round.
- **Round cap** (prespecified):
  - `min(A_rem · b_r / k_rem, A_rem − R_future)`. The first term is pacing,
    with slack carried forward. The second is a feasibility reserve.
  - `R_future` = the sum of sorted candidate d at positions
    `[b_r, b_r + k_future)`. This provably bounds the cheapest completion of
    the later rounds, whatever this round takes.
  - The last round's cap is exactly `A_rem`.
  - Pacing stops a myopic first round from spending the whole shared budget
    on atypical sequences.
- **Constraint-aware rounding.** It is the same descending-z, positive-marginal
  walk as `positive_marginal_selection`. A candidate is kept only if
  `spent + d_i + minsum(k_left − 1 remaining) ≤ cap`.
  - Skipped positive-marginal candidates count as **repairs**. The log records
    their objective loss relative to naive rounding of the same z.
  - Exploration cover and random fill are restricted the same way.
  - The invariant `spent + minsum(k_left) ≤ cap` always admits the cheapest
    candidate, so a feasible completion always exists. There is **no
    unconstrained fallback**.
  - An LP infeasibility or budget overrun raises `ReprInfeasible`, and τ is
    never relaxed.
- **`e3_allocation.py`**:
  - adds `--strategy coverage_lp_repr`, together with `--repr-tau` and
    `--typicality-scores`;
  - the strategy shares the `coverage_lp` utility (`AM.CONSTRAINED_LP_STRATEGIES`);
  - it refuses `--candidate-subset-file`, so it always uses the full pool;
  - `coverage_lp` and every other strategy are byte-for-byte unchanged in
    behaviour. The real-data replay above verifies this.
- **Logging** (per round): cap, paced cap, reserve, fractional LP d-sum,
  naive-rounding d-sum, whether naive rounding violates the cap, repairs,
  objective loss, cumulative spent, and timing. The summary records
  `repr_constraint`.
- **Also added:** `scripts/run_repr_constrained.sh` (resumable), a
  `diagnostics/report_repr_constrained.py` validator and report, and
  `diagnostics/covariance_mismatch.py`.

**Rounding mattered:**
- In 3 of 6 rounds, naive top-z rounding of the feasible fractional LP would
  have violated the round cap.
- Five repairs fixed this, at a total objective loss of **0.011 %**.
- All 288 adaptive picks came from the LP; there was no fill or fallback.

## 6. CPU tests

`tests/test_repr_constrained.py` has 13 tests. They cover:
- the typicality definition, the frozen reference and malformed rows;
- agreement with the stored artifact;
- pacing, reserve and cumulative accounting;
- infeasibility detection, without relaxation;
- **a fractional-feasible LP whose naive rounding is infeasible and is
  repaired**;
- deterministic, unique, exact-budget rounding;
- zero-count experts, and exploration and fill under the constraint;
- **unconstrained coverage picking atypical sequences while the constrained
  version is more representative *and* selects sequences above the median**;
- round-to-round budget logging;
- reproducibility and Coverage-LP backward compatibility;
- an end-to-end tiny-model run: B, warm start, uniqueness, split isolation,
  frozen GPTQ, the identical evaluation universe, and rejection of a bad
  typicality CSV.

**Full suite: 100 passed** (87 existing + 13 new; `logs/pytest_all.log`).

## 7. GPU experiment status

- **3 / 3 runs complete and validated:** `coverage_lp_repr` × seeds 0/1/2.
- **Hardware:** A100 80 GB MIG 2g.20gb slice; peak memory 7.8 GB.
- **Baselines reused without rerun:** the 18 matched runs in
  `runs_typicality_matched/`. Before reuse they were validated for warm-start
  ids/hash, frozen GPTQ (4-bit, g128, percdamp 1.0, act-order), model, split,
  eval universe and schema.
- `check-warm` confirmed, for each seed, that all 7 conditions share the exact
  ordered warm start, a single 240-expert evaluation universe and identical
  GPTQ settings.
- Every run quantized and evaluated all 240 routed experts. Nothing outside
  the candidate pool was selected.

```bash
python diagnostics/repr_feasibility.py           # CPU: d(s) check, tau, feasibility, dry run
bash scripts/run_repr_constrained.sh             # seed 0 -> check -> seeds 1, 2 -> report (resumable)
python diagnostics/report_repr_constrained.py    # summary.md / summary.json
python diagnostics/covariance_mismatch.py        # post-hoc, read-only (Part E)
```

## 8. Reconstruction comparison

Mean expert relative error on the test split; lower is better. This is
partial-model, non-sequential expert GPTQ.

| condition | mean | SD | s0 | s1 | s2 |
|---|---|---|---|---|---|
| Full Random | 0.004615 | 0.000009 | 0.004610 | 0.004626 | 0.004610 |
| Full Coverage-LP | 0.004730 | 0.000005 | 0.004727 | 0.004727 | 0.004735 |
| **Typical Random** | **0.004606** | 0.000006 | 0.004612 | 0.004600 | 0.004605 |
| Typical Coverage-LP | 0.004613 | 0.000003 | 0.004616 | 0.004609 | 0.004614 |
| Random-half Coverage-LP | 0.004697 | 0.000009 | 0.004698 | 0.004706 | 0.004687 |
| **Coverage-LP-Repr** | **0.004653** | 0.000017 | 0.004638 | 0.004672 | 0.004649 |

## 9. Paired-seed results (Coverage-LP-Repr − baseline)

| baseline | rel. per seed | mean | seeds RC better | per-layer (L1/L8/L16/L23) | expert p95 |
|---|---|---|---|---|---|
| **Typical Random (primary)** | +0.58 / +1.56 / +0.96 % | **+1.03 %** | **0/3** | +1.21 / +0.77 / +1.15 / +1.03 % | +1.09 % |
| Typical Coverage-LP | +0.49 / +1.35 / +0.75 % | +0.86 % | 0/3 | +0.77 / +0.57 / +1.06 / +1.01 % | +0.75 % |
| Full Random | +0.63 / +0.98 / +0.85 % | +0.82 % | 0/3 | +0.75 / +0.62 / +1.11 / +0.75 % | +0.76 % |
| Full Coverage-LP | −1.87 / −1.18 / −1.82 % | **−1.62 %** | **3/3** | −1.77 / −1.54 / −1.58 / −1.64 % | −1.72 % |
| Random-half Coverage-LP | −1.28 / −0.73 / −0.82 % | −0.94 % | 3/3 | −1.37 / −0.89 / −0.74 / −0.89 % | −1.25 % |

- The constraint removes about two thirds of Full Coverage-LP's penalty
  against Full Random: +115e-6 → +38e-6.
- It is worse than **every** random baseline and than hard Typical filtering,
  on every seed and in every layer.
- The expert median and tail (p90/p95) move with the mean, so this is not a
  tail-only effect.

## 10. Routing representativeness

| condition | d post-warm | set JS vs pool | set L1 (L1/L8/L16/L23) | post-warm outside T-half | post-warm > pool p90(d) |
|---|---|---|---|---|---|
| Full Random | 0.336 | 0.00023 | .022/.032/.032/.042 | 50 % | 8.3 % |
| Full Coverage-LP | 0.449 | 0.00154 | .055/.096/.104/.100 | 87 % | 51.0 % |
| Typical Random | 0.289 | 0.00045 | .032/.046/.048/.056 | 0 % | 0 % |
| Typical Coverage-LP | 0.297 | 0.00030 | .024/.033/.030/.052 | 0 % | 0 % |
| Random-half Coverage-LP | 0.413 | 0.00131 | .049/.084/.095/.096 | 80 % | — |
| Coverage-LP-Repr | 0.320 | 0.00065 | .029/.058/.064/.077 | 42 % | 7.3 % |

**Key observation:**
- RC's per-sequence atypicality is *lower* than Full Random's on every
  per-sequence measure: mean d, tail fraction above the pool p90, and the
  p90 of the picks' d (0.4047 vs 0.4053).
- Yet RC's aggregate selected-set routing shift is **2.8× Full Random's**
  (JS 0.00065 vs 0.00023), and its reconstruction is 0.82 % worse.
- Random atypical sequences deviate in random directions, which cancel. The
  coverage objective chooses atypical sequences that all deviate toward the
  same under-covered experts, so their deviations add up.
- A per-sequence mean-d bound cannot control this *coordinated* aggregate
  drift.

Across the six conditions, reconstruction error ranks more closely with set
JS than with mean d: FR, TC, TR < RC < HC < FC. This is a descriptive
pattern over six correlated conditions, not a fitted relation.

Selected-set overlap with RC, post-warm only: FC 9.4 %, TC 18.8 %, TR 5.6 %,
FR 4.5 %.

## 11. Source diversity and expert exposure

| condition | Gini of expert tokens | median tokens/expert | median effective sources |
|---|---|---|---|
| Full Random | 0.054 | 4307 | 118.5 |
| Full Coverage-LP | 0.043 | 4345 | 112.0 |
| Typical Random | 0.065 | 4310 | 120.9 |
| Typical Coverage-LP | 0.046 | 4330 | 120.4 |
| Coverage-LP-Repr | **0.032** | 4353 | 119.3 |

No condition has low-exposure experts.

Mechanism questions:

1. **Better token balance than Typical Random?** Yes. RC has the most
   balanced exposure of any condition: Gini 0.032 vs 0.065.
2. **Atypicality close to Typical Random?** Per sequence, only partly: mean d
   0.320 vs 0.289. In aggregate, no: set JS is 1.4× Typical Random's and 2.2×
   Typical Coverage-LP's.
3. **Uses sequences excluded by hard filtering?** Yes, 42 %. They do not help.
4. **Better reconstruction?** No, compared with any random baseline. Yes,
   compared with unconstrained coverage.
5. **Effective source diversity?** It largely recovers (119.3 vs FC 112.0) but
   stays below Typical Random (120.9).
6. **Layers?** RC is worse than Typical Random in all four layers and better
   than Full Coverage-LP in all four.

Better exposure balance again fails to translate into lower error.

## 12. Systems overhead

- **Selection cost:**
  - typicality scores: 0.02 s;
  - LP plus constraint-aware rounding: 1.2 s per run (vs 1.1 s for
    Coverage-LP);
  - an extra LP row and the feasibility checks: ~0.3–0.9 s per round.
- **Routing profile:** the candidate-pool profile still costs ~357 s of
  full-model-forward equivalent, about 2,000 extra random calibration
  sequences at equal compute.
  - Every condition except Full Random needs it. Typical Random also needs it
    to build its pool, even though its run JSON charges 0.
- **GPTQ is identical** across conditions: 280–285 s and 62–63 s Hessian
  replay for seeds 1/2.
  - Seed 0's 597 s / 103 s wall time reflects CPU contention from the
    concurrently running test suite.
- **Peak GPU memory:** 7.8 GB.
- Equal-B results do not establish equal-compute efficiency.

## 13. Covariance diagnostic status (Part E)

**Run.** This was done after the primary result was recorded; it is
read-only and post hoc (`diagnostics/covariance_mismatch.py` →
`runs_repr_constrained/covariance/`).

- **Inputs:** H = XᵀX / N over the gate/up-projection input, for all 240
  experts × 7 conditions × 3 seeds.
- **Reference split:** the **tuning split** (rows 2000–2127). It is disjoint
  from the pool and from the test split that scores reconstruction, and it
  played no part in this experiment's selection or τ.
- **Caveat:** that split *was* used earlier (E1 labels, P0 damping). It is an
  independent activation source, not a pristine confirmation split. Labelled
  descriptive.
- **Cost:** ~20 GPU-minutes.

| condition | rel. Frobenius | rel. diagonal | eff-rank ratio cal/ref | top-16 alignment | recon |
|---|---|---|---|---|---|
| Typical Random | 0.289 | 0.072 | 0.946 | 0.628 | 0.004606 |
| Typical Coverage-LP | 0.276 | 0.052 | 0.980 | 0.627 | 0.004613 |
| Full Random | 0.273 | 0.048 | 0.968 | 0.622 | 0.004615 |
| Random-half Random | 0.270 | 0.045 | 0.987 | 0.621 | 0.004615 |
| **Coverage-LP-Repr** | 0.293 | 0.076 | **1.055** | 0.617 | 0.004653 |
| Random-half Coverage-LP | 0.338 | 0.103 | 1.016 | 0.541 | 0.004697 |
| Full Coverage-LP | 0.372 | 0.133 | 1.046 | 0.509 | 0.004730 |

**Correlations.** Spearman correlation between mean Frobenius mismatch and
reconstruction error:
- over 21 runs: 0.74;
- within seed, across the 7 conditions: 0.86 / 0.75 / 0.75;
- within expert, across runs: median 0.78, and positive for 99 % of the 240
  experts.

**Reading (descriptive, not causal):**
- Routing-atypical coverage selections (FC, HC) also produce calibration
  second moments that are further from held-out activations and less aligned
  with them.
- The representativeness constraint largely closes the Frobenius and
  alignment gap. RC's Frobenius distance is close to Typical Random's.
- However, RC has the **highest effective-rank ratio** (1.055). Its
  calibration Hessians are more spread out, or more isotropic, than held-out
  activations.
- The three worst conditions are exactly the three with ratio > 1. All
  random and hard-filtered conditions are < 1.
- This is consistent with exposure balancing over-weighting rarely used
  directions. Seven correlated conditions cannot establish this, and routing
  divergence and covariance divergence are not assumed to be the same thing.

## 14. PPL diagnostic status (Part F)

**Not run (gate closed).** The gate requires a reproducible reconstruction
improvement over the strongest baseline. Coverage-LP-Repr is worse than
Typical Random on 3/3 seeds. No perplexity numbers were produced.

## 15. Scientific interpretation

- **Mitigation, not a gain.** A soft per-sequence representativeness
  constraint *mitigates* the coverage penalty (−1.62 % vs Full Coverage-LP,
  3/3 seeds, all layers). It does **not** make exposure-based allocation
  better than Random. It is worse than Typical Random, Typical Coverage-LP and
  Full Random on every seed.
- **Hard filtering beats the soft constraint.** It reaches a *lower* mean d
  and lower aggregate shift, while the soft constraint's extra access to the
  full pool is used to buy balance that does not help.
- **Coordinated drift (hypothesis, not established).** The constraint's
  failure suggests the harm is not driven only by individual atypical
  sequences. RC matches Full Random's per-sequence tail, yet is clearly worse.
  The data fit a *coordinated aggregate routing shift* better: the coverage
  objective balances expert counts by moving the set's routing distribution
  away from the pool distribution, which is not uniform.
- **Possible conflict between the two goals.** Exposure balancing and aggregate
  representativeness may therefore conflict directly at B = 128. Every
  condition that balanced more than Random (lower Gini) had a larger aggregate
  shift than Full Random, and none beat the best random baseline.
- **Status.** All of this is exploratory: the test split has been inspected by
  several earlier allocation and typicality hypotheses.

## 16. Limitations

- Three seeds = three paired replicates. Experts are not independent; no
  significance claims are made.
- Partial-model, non-sequential expert GPTQ on four layers; attention and
  shared experts are BF16. This says nothing about full-model accuracy.
- One prespecified τ (the median) and one pacing rule. Other τ values were
  not explored, by design (stopping rules).
- Matched warm starts are drawn from T ∩ H, so the estimand is conditional on
  that warm start.
- The test split has been reused across hypotheses. No locked confirmation
  was run.
- **Confirmation split:** none is available. The cache holds exactly
  pool 2000 + tune 128 + test 128 from a single shuffled pass over C4 shard
  00000.
  - A reproducible disjoint split can be created by replaying
    `get_splits(..., data_seed=0)` with `need = 2256 + 256`. The first 2,256
    windows must match `ids.pt` byte for byte; the next 256 are new
    documents, one window per document, so they are disjoint from the pool,
    the tune split and the inspected test split.
  - Caching them costs about 2–3 GPU-minutes. It was not done here (stopping
    rules), and nothing should be tuned on it.

## 17. Recommended next experiment

**One experiment: an aggregate-shift-constrained Coverage-LP, as a decisive
test of the coordinated-drift explanation.**

- **Why it is cheap and exact.** Each sequence routes exactly 2,048 tokens per
  layer (512 × top-4; verified for all 2,000 candidates). So the selected
  set's aggregate routing distribution is *linear* in z, and the bound
  `sum_e |p_set(L,e) − p_pool(L,e)| ≤ ε_L` is an exact LP constraint using
  auxiliary variables.
- **Prespecify ε_L** before any reconstruction from candidate-pool data only.
  For example, use the median per-layer set L1 of random 128-sequence draws
  from the pool.
- **Keep everything else fixed:** same warm starts, τ-free, one run per seed,
  evaluated once.
- **Run it on the new locked confirmation split above,** together with Full
  Random and Typical Random.
- **Reading the outcome:**
  - If it still cannot beat Random, the evidence would be strong that
    exposure balancing has no recoverable value at B = 128, and the active
    allocation line should stop.
  - If it does beat Random, it would be the first allocation result confirmed
    on untouched data.
