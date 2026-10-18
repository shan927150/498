# Project Update — October 18, 2026

Internal research and engineering log. Chronological: what changed, why,
what was run, what was learned, and what is still open. Tags used below:
**[finding]** measured result, **[fix]** code change, **[hypothesis]** not
established, **[open]** undecided.

Numbers are copied from `runs/` (characterization) and `runs_diag/`
(diagnostics). All diagnostics used the pool and the tuning split only.

---

## 1. Starting point

The first implementation (commit `16dac21`) ran E1 → E2 → E3 end to end, but
an audit against the proposal found several research-validity problems that
had to be fixed before spending GPU time on E3:

- **E2 and E3 were disconnected.** E2 analysed several signals, but E3
  hard-coded relSE and converted it into a token "need" via a CI-width rule.
- **Statistical precision was treated as sufficiency.** A converged Hessian
  estimate was assumed to mean "enough calibration", without checking that
  it predicts reconstruction benefit.
- **One holdout served two roles.** The same held-out sequences were used for
  E1/E2 tuning and for E3's final reconstruction evaluation.
- **Unfair evaluation.** In E3, experts with < 32 calibration tokens could be
  skipped by quantization, so a strategy that starved experts kept them in
  BF16 for free.
- **E1 run identity.** Resume logic could mix outputs from incompatible
  configurations.
- **Unstable blocks.** E3's statistical block assignment depended on
  selection order, so a sequence could change block between rounds.
- **The pooled predictor did not represent validated future benefit.**

These motivated a research-validity refactor before any expensive E3 runs.

## 2. Research-validity refactor

Committed as `360d547`. Later changes (C4 loader fix, characterization
script and analysis, diagnostics, the routing fix in §7) are uncommitted.

### Data splits
- **[fix]** Cached rows are split into three disjoint roles: pool (rows
  0–1999), tune (2000–2127), test (2128–2255). Every output records the
  split definition; E3 asserts it matches the E2 artifact.
- **[fix]** The test split is used only for final E3 reconstruction.

### E1
- **[fix]** Continuous target `future_rel_delta` (n0 → 2·n0, same tuning
  activations); t-test/BH significance kept only as a secondary diagnostic.
- **[fix]** No token-count quality threshold: experts are labelled whenever a
  Hessian exists (`--min-calib-tokens 1`) and ≥ 8 tuning sequences are paired.
- **[fix]** Explicit exclusion reasons in `excluded.csv`.
- **[fix]** Characterization seeds (`--char-seed`), deterministic draws per
  (seed, n0), saved in `draws.json`.
- **[fix]** Run fingerprint directory `e1_b{bits}_s{seed}_{hash}`; same
  config resumes, mismatching config is rejected.

### E2
- **[fix]** Signals count / drift / relSE / margin / entropy, each fitted as
  `b0 + b_count·log(n) + b_sig·z(signal)`.
- **[fix]** Multi-seed validation: leave-one-characterization-seed-out MSE
  (`--min-char-seeds`); single-seed results are flagged `preliminary`.
- **[fix]** Machine-readable artifact `e2_model_b{bits}.json` with selected
  signal, coefficients, orientation, fitted count support, seeds, criterion,
  split, K, block seed.

### E3
- **[fix]** Consumes the E2 predictor (`--signal auto`); relSE is no longer
  hard-coded. The old relSE CI-width rule survives only as the
  `--allocation relse_precision` ablation.
- **[fix]** Utility is expressed in predicted future reconstruction benefit:
  capacity n_e, per-token value max(δ̂_e, 0)/n_e.
- **[fix]** Exploration cover for zero-exposure and below-support experts,
  capped at the round budget.
- **[fix]** No extrapolation above the fitted count support.
- **[fix]** Seeded random fallback when no expert has positive utility, and
  seeded random fill for slots left after positive-utility capacities saturate.

### Evaluation fairness
- **[fix]** Every routed expert in the profiled layers is quantized under
  every strategy.
- **[fix]** Fixed evaluation universe derived only from test routing,
  asserted identical across strategies.
- **[fix]** Zero-calibration experts use GPTQ with H = I and act-order off,
  which is exactly per-group RTN.

### Reproducibility
- **[fix]** Deterministic statistical blocks: `block_of(seq, K, seed)` is a hash.
- **[fix]** A single `round_schedule()` shared by E3 and the support diagnostic.
- **[fix]** Support diagnostic evaluated at E3's actual decision checkpoints.

Synthetic CPU tests grew 11 → 19 → 22 → 23 over these steps (25 after the
routing tests in §7). A real Qwen/CUDA smoke run (cache → E1 → E2 → E3 tlp
and random → PPL) passed on ICRN.

## 3. Real 4-bit characterization

Configuration: Qwen1.5-MoE-A2.7B, 4-bit, group 128, `percdamp = 0.01`,
layers 1/8/16/23, characterization seeds 0/1/2, n0 = 16/32/64/128, 3
extension replicates per n0, pool 2000, tune 128, test 128.

- **[finding]** All 8,640 expert-replicate units (60 experts × 4 layers ×
  4 n0 × 3 reps = 2,880 per seed) were labelled; **zero exclusions**.
- Routed counts per expert at the base n0 (all seeds): n0=16 → 304–885
  (median 538); n0=32 → 749–1859 (1082); n0=64 → 1454–3707 (2160);
  n0=128 → 3164–6349 (4318).

**[finding]** Future benefit is non-monotone in n0:

| step | mean rel_delta | experts worsened |
|---|---|---|
| 16 → 32 | −0.086 | ~90% |
| 32 → 64 | +0.191 | 0% |
| 64 → 128 | +0.294 | 0% |
| 128 → 256 | +0.170 | 0% |

The pattern was consistent across all three seeds and all four layers.
Adding calibration data making held-out reconstruction worse was unexpected,
so E3 was **not** run.

## 4. Initial E2 result under 1% damping

E2 (2,880 observations, LOSO over 3 seeds) selected **relSE**:

| signal | LOSO MSE | vs count |
|---|---|---|
| count | 0.012728 | — |
| drift | 0.012518 | −1.65% |
| relSE | 0.012505 | −1.75% |
| margin | 0.012731 | +0.03% |
| entropy | 0.012730 | +0.02% |

**Not accepted as validation**, because:

- The improvement over count is small (1.75%; per held-out seed 0.2%, 2.6%,
  2.5%).
- The relSE coefficient changes sign across n0 regimes (within-n0 fits:
  ≈ −0.036 at n0=16, ≈ 0 at 32, ≈ +0.020 at 64, ≈ +0.004 at 128).
- The count slope also changes direction (+0.25 to +0.42 at n0=16/32,
  −0.18 to −0.22 at n0=64/128); the pooled slope +0.126 makes predicted
  benefit *increase* with count.
- Separate per-n0 models fit far better than the pooled model (count-only
  LOSO MSE 0.0030 / 0.0020 / 0.0017 / 0.00018 vs ≈ 0.012–0.014 for the pooled
  model on the same rows).
- The pooled gain comes from the n0=16 regime (−9%), which is dominated by
  negative future benefit; at n0=64 relSE is worse than count (+3.2%).

**The old E2 result is diagnostic only and must not be used for final E3.**

## 5. Support diagnostic

Fitted routed-count support of the old model: **[304, 6349]**. Decision
checkpoint coverage (3 seeds × 4 layers; zero and below-support were 0%
everywhere):

| final B | decision size | inside | above |
|---|---|---|---|
| 64 | 16, 40 | 100% | 0% |
| 128 | 32, 80 | 100% | 0% |
| 256 | 64 | 100% | 0% |
| 256 | 160 | 93.5% | 6.5% |
| 512 | 128 | 100% | 0% |
| 512 | 320 | 0% | 100% |

This suggested n0=256 might eventually be needed for B=512 **[open]**. It is
not the current bottleneck: E1/E2 must first be rerun under the final GPTQ
configuration. The support diagnostic also used the buggy routing profile
(§7) and must be regenerated.

## 6. Low-data GPTQ anomaly investigation

Before touching the predictor, the team checked whether the 16 → 32
degradation was (1) an E1/Hessian bug, (2) a GPTQ regularization artifact,
or (3) genuine low-data behaviour.

**Invariant checks** (L8, seed 0, n0=16 draw):
- **[finding]** H_base + H_extension matches a direct profile of the union to
  relative Frobenius ≈ 1e-7 (gate/up and down); token counts match exactly.
- **[finding]** E1's stored `rel_delta` and token counts reproduce exactly.
- **[finding]** Base, extensions, tune and test are disjoint for all seeds and
  n0; extensions add tokens to every expert; n0 and 2·n0 use the same
  tuning sequences.
- **[finding]** GPTQ's rounding is chaotic (a 1e-7 Hessian change or scaling
  H by 2× / 10× changes ~76–78% of quantized entries), but tuning error moves
  by < 0.07% relative — far below the effect being studied.

**Nested calibration curve** (one fixed pool ordering per seed, prefixes
C16 ⊂ C32 ⊂ … ⊂ C256; 3 seeds × 4 layers × 60 experts, `percdamp = 0.01`):

| sequences | tuning error | calibration-set error |
|---|---|---|
| 16 | 0.00949 | 0.00045 |
| 32 | 0.01038 | 0.00091 |
| 64 | 0.00834 | 0.00168 |
| 128 | 0.00581 | 0.00248 |
| 256 | 0.00482 | 0.00304 |

RTN tuning error ≈ 0.00679. **[finding]** The nested curve reproduces the
anomaly: 16 → 32 worsened 89% of experts (L1 68%, L8 99%, L16 97%, L23 91%);
every later step improved 100% of experts.

**Calibration vs held-out** (seed 0, L8; Hessian-exact relative error):
- q32 fits its own calibration set C32 far better than q16 (0.00096 vs 0.00496).
- On sequences 33–64, unseen by both, q32 is worse (0.01080 vs 0.00971); on
  the tuning split q32 is worse (0.01067 vs 0.00960).
- Same pattern in L1, L16 and L23.

**[finding]** This is evidence of genuine low-data overfitting /
under-regularization in GPTQ, not an E1 bug. In a seed-0 / L8 sensitivity check,
lowering damping to 0.001 made the 16 → 32 worsening much larger (mean
rel_delta −0.27 vs −0.11 at 0.01); raising it to 0.05 nearly removed it
(−0.002); act-order had a smaller effect.

**[hypothesis]** At 16–32 sequences an expert sees roughly 540–1,100 routed
tokens, below the gate/up (2048) and down (1408) input dimensions, i.e. a
low-rank, weakly damped Hessian. This is a plausible mechanism, not an
established one — both 16 and 32 are below those dimensions, yet 32 is worse.

## 7. Routing-profile bug

**[finding]** Found during the invariant checks: routing-profile counts
disagreed with Hessian token counts.

- Old: `selected experts = topk(probs, k+1)[:k]`.
- **[fix]** Now: `selected experts = topk(probs, k)` (the model's own op);
  `topk(k+1)` values are used only for the margin.
- Cause: router scores are bf16, so exact ties are common; on the GPU,
  `topk(k+1)` and `topk(k)` can break ties differently. Totals per token were
  right, but tied tokens were credited to the wrong expert.

Real-model attribution mismatch (32 pool sequences per layer, 16,384 tokens):

| layer | old rule vs model | fixed hook vs model |
|---|---|---|
| L1 | 10.5% | 0 |
| L8 | 4.6% | 0 |
| L16 | 3.8% | 0 |
| L23 | 6.6% | 0 |

Per-expert count errors went from up to 125 tokens to 0. Two CPU regression
tests were added (tied router rows; and a simulated k-dependent
tie-breaking test that fails on the old rule).

Impact:
- E1 labels and E1/E2 routed counts were **unaffected** (they come from
  expert inputs).
- The E3 routing matrix and the old support diagnostic were affected; the
  E2 margin/entropy signals were also averaged over tokens attributed by the
  old rule.
- **[open]** `runs/routing_profile.npz` is stale; `support_diagnostic.py`
  reuses it if layers and pool match, so it must be deleted/regenerated
  before the support diagnostic is rerun.

## 8. Damping study

Controlled study on the tuning split: `percdamp ∈ {0.01, 0.03, 0.05, 0.10}`,
act-order on, seeds 0/1, layers 1/8/16/23, nested prefixes 16–256, fixed RTN
reference (H = I, act-order off). Values averaged over prefixes; transitions
are err_n − err_2n (positive = more data helped).

| percdamp | mean tuning error | Δ vs RTN | % experts better than RTN | 16→32 |
|---|---|---|---|---|
| 0.01 | 0.00775 | +0.00097 | 44.5% | −0.00092 |
| 0.03 | 0.00673 | −0.00006 | 54.5% | −0.00019 |
| 0.05 | 0.00629 | −0.00050 | 62.3% | +0.00005 |
| 0.10 | 0.00576 | −0.00103 | 76.3% | +0.00025 |

Per budget for `percdamp = 0.10` (RTN ≈ 0.00679):

| prefix | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| GPTQ tuning error | 0.00683 | 0.00657 | 0.00582 | 0.00504 | 0.00454 |
| % experts better than RTN | 41.9% | 49.6% | 90.0% | 100% | 100% |

- **[finding]** 0.10 is the best tested value at every prefix, seed and layer.
- **[finding]** It largely removes the anomaly (16 → 32 improves for 93% of
  experts) and every later step improves 100% of experts.
- **[finding]** Seeds agree closely; layers rank the values identically, but
  L8 and L16 are still worse than RTN at 16–32 sequences even at 0.10
  (7–25% of experts better), while L1 and L23 beat RTN at every prefix.
- **[open]** 0.10 is the upper edge of the tested grid and error was still
  decreasing; the production default has **not** been changed. Optional
  follow-up: test 0.2 / 0.3 before freezing the GPTQ config.

## 9. Connection back to the proposal

Proposal claim: an expert's current calibration state may predict future
calibration benefit beyond routed-token count.

Accomplished so far:
- Future benefit is measured directly (E1) instead of being inferred from
  Hessian convergence.
- Signal validation (tuning split) is separated from final evaluation (test
  split).
- Multiple characterization seeds are used, with leave-one-seed-out selection.
- The adaptive allocator consumes a benefit model rather than raw
  statistical convergence.
- Natural MoE routing is preserved, and routing profiling now matches actual
  expert routing exactly.

Important refinement discovered during characterization: *before testing
whether calibration state predicts future benefit, the quantizer that
defines that benefit must itself behave sensibly in the calibration-budget
regime.* GPTQ with 1% damping violates this at low budgets (more data makes
held-out reconstruction worse, and GPTQ is worse than RTN up to 64
sequences). The damping study is therefore not a change of research
question; it removes a confounder before the central hypothesis is tested.

## 10. Current status

Completed:
- research-valid pipeline (splits, E1 target, E2 validation, E3 utility,
  fair evaluation)
- real smoke
- 3-seed / 4-layer characterization under the old 1% damping
- initial E2 analysis
- support diagnostic (old model, old routing profile)
- low-data anomaly diagnosis
- routing profiler fix
- controlled fixed-damping study

Not yet completed:
- final production damping freeze
- E1/E2 rerun under the final damping
- final signal-validation conclusion
- corrected support diagnostic
- n0=256 decision
- E3 Random / Coverage / Sufficiency comparison
- PPL, tail, overhead and equal-compute evaluation
- 3-bit robustness

Sufficiency-aware allocation has **not** been shown to work yet.

## 11. Immediate next steps

1. Decide whether to test `percdamp` = 0.2 / 0.3.
2. Freeze ONE fixed production damping.
3. Rerun E1 with characterization seeds 0, 1, 2.
4. Rerun E2.
5. Reassess whether any state signal predicts future benefit beyond count.
6. Redesign the predictor to be budget-aware only if the new
   characterization still requires it.
7. Delete/regenerate the stale `runs/routing_profile.npz`.
8. Rerun the support diagnostic.
9. Decide whether n0=256 is needed.
10. Only then run E3.
