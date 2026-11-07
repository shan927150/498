# Final experiment: aggregate-shift-constrained Coverage-LP on a locked confirmation split — November 7, 2026

Follows [`update_repr_constrained.md`](update_repr_constrained.md). All earlier results and
logs are unchanged. Outputs: `runs_aggregate_constraint/` (`locked_protocol.json`,
`summary.md`, `summary.json`, `figures/`, `eval/`, `logs/`); confirmation activations in
`~/moe_cache_confirmation/`.

**Decision (rule frozen in the locked protocol): AGGREGATE-ALLOCATION-PARTIAL.**
Aggregate Coverage-LP satisfies every aggregate bound and beats Full Coverage-LP on 3/3 seeds
(−1.67 %), but on the untouched confirmation split it is **1.53 % worse than Typical Random
(0/3 seeds better)** and 1.08 % worse than Full Random (0/3). Successful mitigation, not a
better allocator. Exposure balancing still has no recoverable value at B = 128.

## 1. Previous findings

Matched-warm-start experiments (B = 128, 4-bit GPTQ, 3 seeds, 240 experts, old test split):
Typical Random 0.004606 < Typical Coverage-LP 0.004613 < Full Random 0.004615 <
Coverage-LP-Repr 0.004653 < Full Coverage-LP 0.004730. Every exposure-balancing allocator
lost to Random. The per-sequence representativeness constraint recovered most of Full
Coverage-LP's penalty but was still 1.03 % worse than Typical Random on 3/3 seeds.

## 2. Coordinated aggregate-shift hypothesis

Coverage-LP-Repr picked sequences that were *individually* no more atypical than Full Random's
(mean post-warm d 0.320 vs 0.336), yet its selected set's aggregate routing distribution moved
2.8× further from the pool (set JS 0.00065 vs 0.00023): the coverage objective chooses atypical
sequences that all lean toward the same under-covered experts, so their deviations add up. A
post-hoc covariance analysis was consistent with this (Spearman 0.74 between second-moment
mismatch and error over 21 runs) but descriptive. This experiment constrains the aggregate
shift itself and tests the consequence on untouched data.

## 3. Formulation

For the final set S (|S| = B = 128, **warm start included**) and every layer L:

```
D_L(S) = sum_e | p_S(L,e) - p_pool(L,e) |  <=  eps_L
p_pool(L,e) = sum_{s in pool} C[s,L,e] / sum_{s,j} C[s,L,j]      (original 2,000 candidates, fixed)
p_S(L,e)    = sum_{s in S} C[s,L,e] / (B * T),   T = 2048 = 512 tokens x top-4
```

T is **verified** on the routing cache (`aggregate_constraint.token_invariant`: every row and
layer routes exactly 2,048 tokens; any deviation raises). With |S| fixed, p_S is linear in z.
Exact linear form with auxiliary t[L,e] ≥ 0 (M = B·T, n_fix = counts already fixed):

```
t >=  (n_fix + C^T x)/M - p_pool,   t >= -(n_fix + C^T x)/M + p_pool,   sum_e t[L,e] <= eps_L
```

**Controller** (`moe_calib/aggregate_constraint.py`, strategy `coverage_lp_aggregate` in
`e3_allocation.py`; same `u_e = 1/n_e` objective, capacities, warm start and 2 × 48 schedule as
`coverage_lp`). Per round r with budget b_r and k_future slots after it:

1. **Horizon-aware LP**: coverage objective on the round's picks z only; future mass f
   (Σf = k_future, z + f ≤ 1); aggregate rows on n_fix + Cᵀ(z + f). The final-set
   constraint is never imposed per round or split between rounds.
2. **Rounding as in coverage_lp**: descending-z positive-marginal walk + seeded random fill.
3. **Integer certificate**: an exact MILP feasibility problem over binary f decides whether
   the rounded picks still admit an integer completion of k_future sequences satisfying all
   four bounds (last round: direct check).
4. **Repair** if not: the exact constrained round MILP (binary z, f) maximising the same
   coverage objective subject to an integer completion. HiGHS with a deterministic budget
   (relative gap 1e-4 or 200 nodes — a node limit, unlike a time limit, is reproducible); the
   incumbent is re-checked discretely and its gap logged. Infeasible → `AggInfeasible`; no
   certified integer solution → `AggUnresolved`; eps is never relaxed and nothing falls back
   to an unconstrained selection.

This is a **controller difference** from coverage_lp (horizon LP, integer certificate, MILP
repair), not only an added constraint. Zero-exposure exploration targets are refused
explicitly (never occurred: the 32-sequence warm starts expose all 240 experts).

## 4. Threshold construction (prespecified)

For each seed's fixed warm start W_s: 1,000 completions of 96 uniform draws without
replacement from pool \ W_s (`numpy.default_rng([20261107, s])`); eps_L = median of D_L over
all 3,000 pooled draws. No reconstruction result (old or new) was read.

| | L1 | L8 | L16 | L23 |
|---|---|---|---|---|
| eps_L | 0.021305 | 0.033924 | 0.028923 | 0.040310 |

Saved: `epsilon.json`, `epsilon_reference.npz` (every D_L and every draw).

## 5. Discrete feasibility (CPU, before any GPU run)

| seed | random completions meeting all 4 bounds | explicit integer certificate | MILP completion | AG final D (L1/L8/L16/L23) | repaired rounds | deterministic rerun |
|---|---|---|---|---|---|---|
| 0 | 17.5 % | yes | feasible (5.4 s) | .0204/.0329/.0289/.0403 | round 1 (32 swaps) | identical |
| 1 | 21.7 % | yes | feasible (5.3 s) | .0200/.0334/.0255/.0383 | round 1 (5 swaps) | identical |
| 2 | 15.6 % | yes | feasible (8.7 s) | .0213/.0335/.0287/.0393 | round 1 (28 swaps) | identical |

In every seed round 0's naive rounding was certified completable; round 1's naive rounding
violated the bounds in **all four layers** (e.g. s0: .0248/.0418/.0359/.0490), so the
fractional-feasible / discrete-infeasible case is real here. Repair cost 0.10–0.31 % of the
round's coverage objective; the MILP stopped at its node limit with a certified gap of
0.17–0.20 % to the LP bound. None of the four reused baseline selections satisfies all four
bounds in any seed; Typical Random's aggregate shift is the second largest
(mean ≈ .032/.046/.048/.056 in L1/L8/L16/L23) because the typical half is itself a shifted sub-population.

## 6. Confirmation split

`diagnostics/build_confirmation_split.py` replays `get_splits` verbatim (C4 shard
`en/c4-train.00000-of-01024`, data_seed 0, seqlen 512, same shuffle / skip / random window
rule) with headroom, recording document index, offset, text hash and URL per window.

- The first 2,256 replayed windows reproduce `~/moe_cache/ids.pt` (pool + tune + test)
  **token for token**.
- The first replay surfaced **one near-duplicate**: a templated SEO page
  (different document index and text) sharing ~200 consecutive tokens with pool row 1261.
  Different ids are therefore not enough. Rule (fixed then, before any evaluation): the
  confirmation split is the first 256 windows after position 2256 that share no document
  index, no document text (sha256) and no 32-token shingle with any earlier window
  (pool, tune, test or accepted confirmation). 1 window rejected; accepted windows come
  from replay positions 2256–2512.
- Result: 256 windows, document-index-, text- and 32-token-shingle-disjoint from all 2,256
  earlier windows and from each other; never used for any selection, threshold or tuning.
  Fingerprint `da4ff9fb5b29e90a4761091f39b8e6294ff298b2b8334a01b8e2c3b7ae60a05e`.
- Activation cache: layers 1/8/16/23 inputs, built exactly like `cache_inputs.py` (210 s).
  Pipeline check: recomputing the batch-aligned old test rows 2128–2135 the same way is
  **bit-exact** with `~/moe_cache` at all four layers.
- Scoring check (`--verify-old`): the confirmation scorer
  (`e3_allocation.quantize_and_score(..., eval_cache_dir=...)`, a new optional argument
  defaulting to the old behaviour) re-scores Full Random seed 0 on the OLD test split and
  reproduces the stored 0.004609548874536215 exactly (max per-expert difference 0).

## 7. Locked protocol

`runs_aggregate_constraint/locked_protocol.json` (read-only;
sha256 `1abbe55aa734c3c5102dfc88306e072f49fd2eb67aa214f6f799834776503223`), locked at commit
`7ec7aa8` on local branch `final-aggregate-constraint` with a clean code tree. It records model
and GPTQ settings, pool fingerprint `49f227928f299287`, routing sha256, ordered warm starts
(hashes 71f4c159 / 1d2b3965 / ba99ad01), all 15 selected id lists with hashes, eps_L and its
rule/seed/artifact hashes, the confirmation fingerprint and layer-cache hashes, primary and
secondary metrics, baseline definitions, the decision rule, test results and per-file code
hashes. `--evaluate` verifies the protocol hash and the confirmation cache hashes before
scoring, never writes the protocol, and skips only runs completed under the same protocol hash
and id hash.

Selected-id hashes (AG / FR / TR / TC / FC): s0 14f29f92 / ec355fb8 / cc3f2c76 / 98136bed /
0353229b; s1 8088f8d2 / e36d91e6 / adf3bf99 / 516df64f / 325e8efc; s2 04b737f5 / cf1f867a /
f04aaf80 / 8f2fd731 / f63a4a61.

## 8. CPU tests

`tests/test_aggregate_constraint.py` (15 tests): reference distribution, per-layer L1, the
2,048-token normalisation (and its failure on a malformed cache), auxiliary-variable exactness
at integer points, warm-start accounting, multi-round integer-completion certificates, a
deterministic fractional-feasible / naive-rounding-infeasible case repaired by the MILP,
non-zero-based id mapping, deterministic eps, reproducible selections, all per-layer bounds,
B / uniqueness, explicit refusal of zero-exposure targets and of infeasible bounds, protocol
tamper detection and no re-locking, confirmation-split reproduction and disjointness,
selection isolation (pool ids only), Coverage-LP backward compatibility.
**Full suite: 115 passed, 0 failed, 0 skipped** (`logs/pytest_all.log`).

## 9. GPU execution

15 / 15 confirmation runs complete and valid (5 strategies × 3 seeds; protocol hash and
selected-id hash checked per run; one identical 240-expert evaluation universe; all 240
experts quantized in every run). A100 80 GB MIG 2g.20gb slice, two evaluation processes, peak
7.8 GB. The evaluator printed only completion markers; no score was read before all 15 were
done. No run was repeated or dropped. Baseline selections are the reused matched ids; every
strategy was **re-scored** on the confirmation activations (no old score reused).

## 10. Reconstruction comparison (confirmation split, lower is better)

| strategy | mean | SD | s0 | s1 | s2 | expert median | p95 |
|---|---|---|---|---|---|---|---|
| **Typical Random** | **0.004559** | 0.000008 | 0.004566 | 0.004561 | 0.004550 | 0.004747 | 0.006168 |
| Typical Coverage-LP | 0.004562 | 0.000003 | 0.004559 | 0.004565 | 0.004561 | 0.004751 | 0.006175 |
| Full Random | 0.004580 | 0.000011 | 0.004573 | 0.004592 | 0.004575 | 0.004768 | 0.006188 |
| Aggregate Coverage-LP | 0.004629 | 0.000017 | 0.004647 | 0.004614 | 0.004626 | 0.004815 | 0.006267 |
| Full Coverage-LP | 0.004708 | 0.000006 | 0.004713 | 0.004702 | 0.004709 | 0.004888 | 0.006366 |

The ordering of the four reused baselines is the same as on the old test split
(TR < TC < FR < FC); Full Coverage-LP's penalty vs Full Random is +2.80 % (old split +2.49 %).

## 11. Paired-seed results (Aggregate Coverage-LP − baseline, relative)

| baseline | s0 / s1 / s2 | mean | seeds AG better | expert p95 |
|---|---|---|---|---|
| **Typical Random (primary)** | **+1.77 / +1.16 / +1.67 %** | **+1.53 %** | **0/3** | +1.61 % |
| Typical Coverage-LP | +1.93 / +1.07 / +1.44 % | +1.48 % | 0/3 | +1.49 % |
| Full Random | +1.63 / +0.48 / +1.12 % | +1.08 % | 0/3 | +1.27 % |
| Full Coverage-LP | −1.40 / −1.86 / −1.75 % | −1.67 % | 3/3 | −1.55 % |

Secondary (not prespecified as primary, reported for completeness): Typical Random vs Full
Random −0.14 / −0.68 / −0.53 % (mean −0.45 %, 3/3) — the earlier exploratory typicality
finding holds directionally on untouched data; Typical Coverage-LP vs Typical Random
−0.15 / +0.09 / +0.22 % (no difference).

## 12. Per-layer performance

| AG − baseline (mean rel.) | L1 | L8 | L16 | L23 |
|---|---|---|---|---|
| Typical Random | +1.56 % | +1.26 % | +1.60 % | +1.70 % |
| Full Random | +1.02 % | +0.87 % | +1.27 % | +1.11 % |
| Full Coverage-LP | −1.70 % | −1.69 % | −1.82 % | −1.49 % |

Worse than every random baseline in every layer; better than unconstrained coverage in every
layer. Not a tail effect: the expert median and p90/p95 move with the mean.

## 13. Routing shift comparison (selection side; mean over seeds)

| strategy | aggregate L1 (L1/L8/L16/L23) | mean | within all eps | set JS | post-warm d(s) | error |
|---|---|---|---|---|---|---|
| Typical Random | .032/.046/.048/.056 | .0455 | no | .00045 | **0.289** | **0.004559** |
| Typical Coverage-LP | .024/.033/.030/.052 | .0348 | no | .00030 | 0.297 | 0.004562 |
| Full Random | .022/.032/.032/.042 | .0322 | no | .00023 | 0.336 | 0.004580 |
| Aggregate Coverage-LP | .021/.033/.028/.039 | **.0302** | **yes** | **.00023** | 0.392 | 0.004629 |
| Full Coverage-LP | .055/.096/.104/.100 | .0887 | no | .00154 | 0.449 | 0.004708 |

**The coordinated-drift hypothesis is not supported as the explanation.** Aggregate
Coverage-LP has the *smallest* aggregate routing shift of all five strategies (lower L1 than
Full Random, equal set JS) and is still 1.08 % worse than Full Random. Typical Random has the
second-*largest* aggregate shift and the lowest error. To satisfy the aggregate bound while
balancing exposure, the controller combined individually *atypical* sequences whose deviations
cancel: its post-warm mean d rose to 0.392 (Full Random 0.336). Across the five strategies,
confirmation error is ordered exactly by post-warm per-sequence d, not by aggregate shift
(`figures/shift_vs_error`). Five correlated conditions; descriptive, not causal. It also does
not rescue per-sequence d as a complete explanation: on the old split Coverage-LP-Repr had a
lower post-warm d than Full Random (0.320 vs 0.336) yet higher error. Each representativeness
constraint was met, and the coverage objective found the atypicality along the dimension that
was left unconstrained.

## 14. Source diversity and exposure balance

| strategy | expert-token Gini | median tokens/expert | median effective sources |
|---|---|---|---|
| Typical Random | 0.0654 | 4310 | 120.9 |
| Typical Coverage-LP | 0.0457 | 4330 | 120.4 |
| Full Random | 0.0543 | 4307 | 118.5 |
| Aggregate Coverage-LP | **0.0333** | 4340 | 115.2 |
| Full Coverage-LP | 0.0427 | 4345 | 112.0 |

Aggregate Coverage-LP achieves much better token balance than Typical Random (Gini 0.033 vs
0.065), but lower effective source diversity (115.2 vs 120.9), and worse reconstruction. Post-warm
overlap with other strategies is small (FR 4.2 %, TR 2.4 %, TC 1.7 %, FC 11.8 %).

## 15. Covariance diagnostic

**Not run.** It is optional, would need a new ~15–20 GPU-minute profiling pass with a
re-targeted script, and could not change the decision. If run, it must be labelled post-hoc and
descriptive; the existing `diagnostics/covariance_mismatch.py` would need its run list pointed at
`runs_aggregate_constraint/` selections and the confirmation cache as reference.

## 16. Compute overhead

| item | cost |
|---|---|
| Routing profile of the 2,000 candidates | ~357 s full-model-forward equivalent (TR, TC, FC, AG; not FR) |
| eps construction | 0.2 s (3,000 draws) |
| AG selection per seed | 368 / 297 / 351 s: horizon LP ≈ 6 s, integer certificates ≈ 8 s, **repair MILP 278–349 s** |
| Feasibility study (MILP per warm start) | 5–9 s |
| Hessian replay / GPTQ per run | ≈ 52 s / ≈ 332 s, identical across strategies (two concurrent runs) |
| Peak GPU memory | 7.8 GB |

At equal B the new method costs the routing profile (~2,000 sequences of forwards) plus ~6 min
of CPU MILP per selection; at equal compute Full Random could have used far more calibration
data. No efficiency claim is made.

## 17. Scientific conclusion

- On a locked, document-disjoint confirmation split, enforcing aggregate routing
  representativeness (every per-layer L1 within the median random-completion bound) does
  **not** make exposure balancing beat Random: −1.53 % vs Typical Random is the wrong sign on
  every seed and every layer.
- It mitigates the unconstrained Coverage-LP penalty (−1.67 %, 3/3), as the per-sequence
  constraint did (−1.62 % on the old split); neither reaches Full Random.
- The aggregate-shift hypothesis is weakened by this experiment: the strategy with the least
  aggregate shift is not among the best, and the best strategy is not the least shifted.
  Exposure-balancing selection appears harmful through the atypicality of the individual
  sequences it needs, which a constraint on one routing summary displaces rather than removes.
- What holds up across both evaluation sources: Typical Random ≤ Full Random (−0.45 %, 3/3 on
  confirmation, secondary) and every exposure-balancing allocator ≥ Random.
- Three paired seeds, partial-model non-sequential four-layer expert GPTQ; no significance
  claims, no full-model claims.

## 18. Limitations

- Three seeds; 240 experts are correlated, not replicates.
- Expert reconstruction on four layers with attention/shared experts in BF16; no PPL.
- The repair MILP is solved to a deterministic node limit (certified gap ≤ 0.20 %), not to
  proven optimality; repair swapped 5–32 of the final round's 48 picks.
- eps is one prespecified rule (median of random completions); a looser or tighter bound was
  not tried, by design.
- The confirmation split is one shard and 256 windows; the reference p_pool and all selection
  use only the original candidate pool.
- Post-lock change: `diagnostics/report_aggregate_constraint.py` was edited after the scores
  were read, **only** to fix figure layout (title truncation, label collisions; the routing-proxy
  figure became two panels). No data, metric or decision logic changed; the locked code hash of
  that file therefore no longer matches.

## 19. Recommended final project direction

**End expert-exposure-based allocator development for this CS498 project.** Four controlled
attempts — learned utilities, hard typicality filtering, a per-sequence representativeness
constraint and now an aggregate representativeness constraint on untouched data — never beat
Random at B = 128. Focus the final report on the validated negative findings and their
mechanisms: (i) exposure-only allocation does not help once sequence-routing coupling is real;
(ii) coverage objectives buy balance with routing-atypical sequences; (iii) constraining one
representativeness summary (per-sequence mean or aggregate distribution) relocates the
atypicality rather than removing it; (iv) the only consistent positive is the modest
typical-half random effect, now seen on both evaluation sources.
