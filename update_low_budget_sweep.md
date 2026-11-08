# Low-budget calibration allocation sweep — November 8, 2026

Follows [`update_activation_swap.md`](update_activation_swap.md). All earlier outputs are unchanged.
Outputs are in `runs_low_budget/`:
- `scarcity.md` / `.json`
- `summary.md` / `.json`
- `selected_sets.json`
- `warm_starts_B{16,32,64}.json`
- `figures/` (fig1–fig5, PDF and PNG)
- `logs/`

**Decision (prespecified rule, in `diagnostics/report_low_budget.py` before any GPU run):
LOW-BUDGET-COVERAGE-NO-GAIN.**
- Coverage-LP is worse than its matched Random baseline on **every seed at every budget**, in
  both families. The one exception is a tie, Typical B = 64 s0 at −0.02 %.
- The full-pool penalty does not shrink as the budget drops: mean G_full is +1.41 / +2.57 /
  +2.56 / +2.48 % at B = 16 / 32 / 64 / 128.
- The typical-pool penalty stays small and positive: G_typical is +0.34 / +0.66 / +0.27 /
  +0.17 %.
- Neither curve is monotone in B, so the PARTIAL criterion fails as well.

## 1. Motivation

At B = 128, routing-exposure allocators (Coverage-LP and its constrained variants) never beat
Random. A natural remaining hypothesis is that exposure balancing only pays off when the budget
is so small that some experts are starved of routed tokens. This sweep tests that hypothesis
directly. It keeps the controller, utility and GPTQ settings fixed and only scales the budget
down.

## 2. Frozen settings

- **Model and GPTQ:**
  - Qwen1.5-MoE-A2.7B.
  - 4-bit GPTQ: group 128, percdamp 1.0, act-order (`configs/frozen_b4.json`, verified per run).
  - Layers 1/8/16/23.
  - Non-sequential, partial-model expert GPTQ; attention and shared experts stay BF16.
- **Evaluation:** 240 routed experts, one evaluation universe (`eval_universe_b06e33af24.json`,
  copied byte-identical).
- **Splits:**
  - Pool: rows [0, 2000).
  - Evaluation: the **old test split, rows 2128–2255**. It was inspected in earlier
    experiments, so this sweep is **exploratory**.
  - The locked confirmation split was not used or opened.
- **Routing profile:** `routing_profile_v2.npz` (sha verified).
- **Typical pool:** `pool_typical_half.json` (fingerprint 1e5cbd9d8664f64c). It was not
  re-thresholded.
- **Budget-independent settings:** no GPTQ hyperparameter, threshold or utility was changed per
  budget. Zero-token experts would use the existing RTN fallback (the prespecified policy).
  It was never triggered: 0 fallbacks in all 48 runs.

## 3. Strategies, seeds and warm-start schedule

| code | strategy | candidate pool |
|---|---|---|
| FR | random | full pool |
| FC | coverage_lp | full pool |
| TR | random | typical half |
| TC | coverage_lp | typical half |

- **Seeds:** 0/1/2.
- **Warm start:** W(seed, B) = the **first B/4** ids of the matched W(seed, 128) from
  `runs_typicality_matched/warm_starts.json`.
  - The prefixes are nested across budgets and identical for all four strategies.
  - Each prefix gets its own hash.
- **Schedule:** the existing `round_schedule` with K = 4 and two equal rounds:

  | B | warm | round 1 | round 2 |
  |---|---|---|---|
  | 16 | 4 | 6 | 6 |
  | 32 | 8 | 12 | 12 |
  | 64 | 16 | 24 | 24 |
  | 128 | 32 | 48 | 48 |

- **Runs:**
  - 36 new GPU runs (B = 16/32/64).
  - B = 128 reuses the 12 matched runs. `--prepare` re-derived all 12 B = 128 selections and
    found them identical to the stored ones.

## 4. Scarcity (CPU; routing cache only)

Full Random, mean over seeds, 240 experts:

| B | uniform expectation | min | p10 | median | max | CV | Gini | zero / < 128 tokens | effective sources |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 546 | 361 | 448 | 539 | 934 | 0.155 | 0.083 | 0 / 0 | 14.9 |
| 32 | 1092 | 736 | 939 | 1078 | 1689 | 0.124 | 0.065 | 0 / 0 | 29.7 |
| 64 | 2185 | 1592 | 1912 | 2160 | 3175 | 0.111 | 0.059 | 0 / 0 | 59.3 |
| 128 | 4369 | 3171 | 3877 | 4323 | 6311 | 0.103 | 0.054 | 0 / 0 | 118.5 |

- **Starvation is weak.** Even at B = 16 (8,192 routed tokens per layer), no expert receives
  zero or fewer than 128 tokens. The least-routed expert still gets ~66 % of the uniform share.
- **Coverage-LP balances, but not where it matters.** It lowers Gini at every budget, but it
  does not lift the floor at B = 16: the minimum falls to 278 and 0.7 experts drop below 256.
- Figure: `fig3_exposure_scarcity`.

## 5. Implementation changes

No change to the controller, the utility, GPTQ or the evaluator. `coverage_lp` and `random`
run through the unchanged `e3_allocation.py` with `--warm-start-file` (and, for TR/TC,
`--candidate-subset-file`).

New files:

| file | purpose |
|---|---|
| `diagnostics/low_budget.py` | prefix warm-start files, exact pre-computation of every selection (same RNG path as `e3_allocation.main`), set statistics, scarcity report, per-run validator |
| `diagnostics/report_low_budget.py` | paired G(B), doubling returns R(B), mechanism, cost, the prespecified decision rule, figures 1/2/4/5 |
| `scripts/run_low_budget_sweep.sh` | staged driver (prepare / tests / smoke / rest / report; resumable; PAR = 2 controlled concurrent runs) |
| `tests/test_low_budget.py` | 14 CPU tests |

**Post-result edit:** a single formatting fix. The reconstruction-table separator row had one
column too many; it was corrected after the first report render. It changes neither a metric
nor the decision logic.

## 6. CPU tests

`tests/test_low_budget.py`: 14 tests, covering:
- the schedule for each budget;
- nested prefix hashes;
- acceptance of prefix warm starts, and rejection of a wrong length;
- zero-exposure exploration;
- a synthetic lower-tail improvement;
- "balance better while a separate metric gets worse";
- the exposure, Gini and effective-source statistics;
- paired gain and doubling return;
- the decision-rule cases;
- validity and leakage of the prepared selections, and typical-pool membership;
- reproducibility and B = 128 backward compatibility.

Full suite: **138 passed, 0 failed**.

## 7. GPU status

- **Runs:** 36 / 36 new runs completed, and each validated against its precomputed selection:
  - frozen settings;
  - 240 experts;
  - B unique ids;
  - warm hash;
  - pool membership.
- **Totals:** 48 / 48 valid runs, one evaluation universe.
- **Smoke stage** (B = 16, s0, 4 strategies) passed before the batch.
- **Concurrency:** at most 2 concurrent runs on the 20 GB MIG slice; peak 7.8 GB each.

## 8. Reconstruction (mean relative expert error, old test split, lower is better)

| B | FR | FC | TR | TC |
|---|---|---|---|---|
| 16 | **0.005386** ±7e-6 | 0.005462 ±27e-6 | **0.005376** ±6e-6 | 0.005394 ±9e-6 |
| 32 | 0.005093 ±12e-6 | 0.005224 ±38e-6 | **0.005065** ±14e-6 | 0.005098 ±14e-6 |
| 64 | 0.004816 ±9e-6 | 0.004940 ±27e-6 | **0.004804** ±7e-6 | 0.004817 ±8e-6 |
| 128 | 0.004615 ±9e-6 | 0.004730 ±5e-6 | **0.004606** ±6e-6 | 0.004613 ±3e-6 |

- Per-seed values, the expert median and p95, per-layer errors and RTN fallbacks are all in
  `runs_low_budget/summary.md`.
- The ordering is the same at every budget: **TR ≤ FR < TC < FC**. (At B = 16, FR and TR are
  within one SD of each other.)
- Figure: `fig1_error_vs_budget`.

## 9. Paired coverage effect G(B) = [E(Coverage) − E(matched Random)] / E(matched Random)

| B | G_full s0 / s1 / s2 | mean | G_typical s0 / s1 / s2 | mean |
|---|---|---|---|---|
| 16 | +0.94 / +1.36 / +1.95 % | +1.41 % | +0.39 / +0.55 / +0.09 % | +0.34 % |
| 32 | +2.47 / +1.66 / +3.60 % | +2.57 % | +0.80 / +0.38 / +0.80 % | +0.66 % |
| 64 | +2.14 / +3.32 / +2.22 % | +2.56 % | −0.02 / +0.41 / +0.41 % | +0.27 % |
| 128 | +2.55 / +2.19 / +2.71 % | +2.48 % | +0.09 / +0.21 / +0.21 % | +0.17 % |

Seeds where Coverage is better: 0 out of 12 (full) and 1 out of 12 (typical, a tie). Decision
rule:
- **PROMISING** needs some B ≤ 64 with 3/3 seeds < 0 and a mean ≤ −0.5 %. Not met.
- **PARTIAL** needs mean G strictly increasing in B. Not met: full is 1.41 → 2.57 → 2.56 →
  2.48; typical is 0.34 → 0.66 → 0.27 → 0.17.
- Result: **NO-GAIN.**

The smaller penalty at B = 16 (full) is not a coverage benefit. Coverage is still worse there on
3/3 seeds, and only 9.6 % of experts are better under Coverage.

Figure: `fig2_coverage_penalty`.

## 10. Diminishing returns R(B) = [E(B) − E(2B)] / E(B)

| strategy | 16→32 | 32→64 | 64→128 |
|---|---|---|---|
| FR | 5.44 % | 5.44 % | 4.17 % |
| FC | 4.35 % | 5.45 % | 4.24 % |
| TR | 5.78 % | 5.15 % | 4.14 % |
| TC | 5.48 % | 5.52 % | 4.23 % |

- **Returns diminish only slowly:** each doubling still buys ~4–5.5 %, and that dwarfs any
  allocation effect.
- **More data beats smarter selection.** Doubling from B to 2B with plain Random improves
  error by 4–5.8 %. That is about 2× the mean Full Coverage penalty and roughly 7–30× the
  typical-family difference (|G_typical| ≤ 0.8 %).
## 11. Mechanism

| B | Δ Gini, FC − FR | post-warm d(s), FR / FC / TR / TC | set JS ×1e-3, FR / FC | effective sources, FR / FC |
|---|---|---|---|---|
| 16 | −0.019 | 0.338 / 0.412 / 0.291 / 0.299 | 1.82 / 2.44 | 14.9 / 14.4 |
| 32 | −0.016 | 0.345 / 0.462 / 0.287 / 0.303 | 0.84 / 1.73 | 29.7 / 28.0 |
| 64 | −0.012 | 0.336 / 0.446 / 0.288 / 0.300 | 0.42 / 1.99 | 59.3 / 56.3 |
| 128 | −0.012 | 0.336 / 0.449 / 0.288 / 0.297 | 0.23 / 1.54 | 118.5 / 112.0 |

- **Atypical picks in the full pool.** Full Coverage-LP achieves balance by picking atypical
  sequences:
  - post-warm d(s) is ~+0.1 above Random at every B;
  - set routing JS to the pool is 1.3–7× higher;
  - effective sources per expert are 3–6 % lower.
- **Typical half.** Restricting to the typical half removes most of the atypicality (TC
  d(s) ≈ TR) and most of the penalty. It also removes any balance benefit.
- **No visible relation between balance and error.** In `fig4_balance_vs_recon`, the Gini
  reduction (−0.005 to −0.031) shows no visible relationship with G. All 24 paired points sit
  at or above zero.
- **Atypicality does not depend on budget.** `fig5_atypicality_vs_budget` shows Coverage
  atypicality is essentially flat in B, so low budgets do not change the controller's behaviour.

## 12. Per layer

Full Coverage is worse in every layer at every budget: mean relative difference +1.0 to +2.9 %.
L1 is the worst at small B (+1.9 % at B = 16).

Typical Coverage is +0.02 to +0.8 % per layer; its worst layer is also L1.

No layer shows a coverage benefit at any budget. p95 relative differences are all positive:
+1.3 to +2.5 % (full) and +0.2 to +1.0 % (typical).

## 13. Compute

| item | cost |
|---|---|
| Selection (LP) | 0.4–1.4 s per run |
| Hessian replay | 9–67 s; scales with B |
| GPTQ of 240 experts | ≈ 287 s; independent of B |
| Calibration forward-equivalent | 2.9 / 5.7 / 11.4 / 22.8 s for B = 16 / 32 / 64 / 128 |
| Routing profile of the 2,000-row pool | ≈ 357 s forward-equivalent |
| Peak GPU memory | 7.8 GB per run |

- **Who pays for the profile:** FC and TC need the routing profile, and TR also needs it, to
  build the typical half. Only FR avoids it.
- **Overhead dominates at low budget.** At B = 16 the profile is **~125× the calibration
  forwards** it is meant to make more efficient. The relative overhead grows exactly where the
  hypothesis predicted Coverage would help.

## 14. Interpretation

- **The scarcity hypothesis fails at the first step.** With top-4 routing over 60 experts and
  512-token sequences, even 16 random sequences give every studied expert several hundred
  routed tokens. There is no starved tail for an exposure allocator to rescue.
- **Coverage-LP has the same failure mode at every budget.** It buys a slightly lower Gini by
  adding atypical sequences, and reconstruction gets worse. The penalty is ~2.5 % in the full
  pool and ~0.2–0.7 % in the typical half.
- **This matches B = 128.** The B = 128 anchors and all earlier allocation experiments point
  the same way: per-expert token *count* is not the bottleneck; the routing/activation
  *representativeness* of the calibration set is.
- **Scope of the claim:** these are expert-level reconstruction results only; no full-model
  claim is made.

## 15. Limitations

- **Exploratory evaluation.** The old test split was inspected in earlier experiments, so this
  is not independent confirmation. The locked confirmation split was deliberately not used.
- **One policy.** This is one warm-start-conditioned, budget-scaled policy: prefix warm starts
  and two equal rounds. Other schedules were not tried, by design.
- **Narrow coverage.** Four layers, three seeds, one model, one bit width. Expert-level
  reconstruction only; no PPL.
- **Fixed 512-token sequences.** Shorter sequences or larger expert counts could create real
  scarcity. This sweep cannot speak to those regimes.

## 16. Recommended next step

**Stop allocation experiments and write the final project report.** It should consolidate this
sweep with the earlier allocation experiments and the activation-swap pilot as one coherent
negative result: routing-exposure allocation does not beat matched Random at any tested budget
from 16 to 128.
