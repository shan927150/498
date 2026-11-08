# Extreme low-budget exposure scarcity study (B = 4 / 8) — November 8, 2026

Follows [`update_low_budget_sweep.md`](update_low_budget_sweep.md). CPU only: **no GPTQ, no GPU, no
reconstruction error was read.** All earlier outputs are unchanged.

Outputs are in `runs_extreme_low_budget/`:
- `scarcity.json` / `.md`
- `matched_selection.json`
- `summary.md` / `.json`
- `figures/` (fig1–fig5, PDF and PNG)
- `logs/`

**Decision (prespecified gate): EXTREME-SCARCITY-NO-OPPORTUNITY.**
- **B = 4 is genuinely count-scarce.** In every one of 1,000 random sets, at least 10 % of
  experts get fewer than 128 tokens (median 42 %).
- **Coverage-LP makes that tail *worse*.** On every seed in both pools, the minimum, p5 and
  p10 fall, the count of experts below 128 rises, and even the Gini rises.
- **B = 8 is not scarce under the gate.** Full Random never has 10 % of experts below 128
  tokens, and no expert ever has zero.
- **Coverage gains at B = 8 stay below the bar.** Coverage's best lower-tail gain there, in
  the typical half, is +5–8 % at p10, under the prespecified 10 % threshold.

## 1. Research motivation

Earlier work (B = 16–128) showed Coverage-LP never beat Random, and that B = 16 is not
count-starved: its least-routed expert still gets about 361 tokens. If expert-count balancing
can help GPTQ anywhere, it should be where random calibration starves experts. This study asks
whether such a regime exists at B = 4 or 8, and whether the existing Coverage-LP actually
lifts the starved tail there. That is the precondition for any targeted GPU experiment.

## 2. Frozen setup

- **Model:** Qwen1.5-MoE-A2.7B, layers 1/8/16/23, 60 routed experts per layer, top-4,
  512-token sequences.
- **Routing data:** `routing_profile_v2.npz`, sha verified against `runs_typicality/pools.json`.
  Each sequence routes exactly 2,048 tokens per layer (asserted).
- **Pools:**
  - Full pool: rows [0, 2000).
  - Typical half: `pool_typical_half.json`, fingerprint 1e5cbd9d8664f64c, unchanged.
- **Typicality:** d(s) as before, measured against the original full-pool reference.
- **Uniform expectation per expert:** 136.5 tokens at B = 4 and 273.1 at B = 8.
- **Thresholds:** 32/64/128/256 tokens are **descriptive** scarcity indicators, not validated
  GPTQ sufficiency thresholds.

## 3. Random Monte Carlo exposure distributions (Experiment A)

- **Sets:** 1,000 sets per (pool, B), sampled without replacement.
- **RNG:** independent fixed numpy streams 2026111241/81 (full, B = 4 / 8) and 2026111242/82
  (typical, B = 4 / 8), saved in `scarcity.json`.
- **Scope:** these sets estimate an exposure distribution. They are **not** GPTQ replicates.

Probabilities are over the 1,000 sets, across all 240 experts (95 % Wilson CIs are in
`scarcity.md`):

| pool | B | P(any zero) | P(any < 32) | P(any < 64) | P(any < 128) | P(≥ 10 % of experts < 128) |
|---|---|---|---|---|---|---|
| full | 4 | 0.000 | 0.084 | 0.833 | 1.000 | 1.000 |
| full | 8 | 0.000 | 0.000 | 0.004 | 0.301 | 0.000 |
| typical | 4 | 0.000 | 0.019 | 0.728 | 1.000 | 1.000 |
| typical | 8 | 0.000 | 0.000 | 0.001 | 0.159 | 0.000 |

Per-set statistics, median [q05, q95] over the 1,000 sets:

| pool | B | min | p5 | p10 | median | Gini | experts < 128 | fraction < 128 | effective sources |
|---|---|---|---|---|---|---|---|---|---|
| full | 4 | 52 [28, 70] | 88 | 98 | 134 | 0.136 | 100 [88, 113] | 0.42 | 3.8 / 4 |
| full | 8 | 139 [93, 171] | 198 | 214 | 269 | 0.103 | 0 [0, 2] | 0.00 | 7.5 / 8 |
| typical | 4 | 57 [36, 74] | 91 | 100 | 134 | 0.126 | 98 [87, 109] | 0.41 | 3.9 / 4 |
| typical | 8 | 149 [108, 173] | 200 | 215 | 269 | 0.101 | 0 | 0.00 | 7.7 / 8 |

Full values (p1, p90, max, CV, every threshold, unique sources) are in `scarcity.md`.

- **Per layer.** Scarcity is concentrated in L8 and L23.
  - **B = 4, full pool:** median minima are 91 / 61 / 75 / 58 for L1/L8/L16/L23. P(any expert
    < 64) is 0.01 / 0.56 / 0.18 / 0.66. Experts < 128 per layer: 21 / 26 / 25 / 28 of 60.
  - **B = 8:** only L8 and L23 occasionally have an expert below 128 (P = 0.13 / 0.21).
- **Zero exposure never occurs.**
  - 90.6 % of single pool sequences touch all 240 experts, and every sequence touches at
    least 225.
  - So zero-exposure experts essentially cannot occur at B ≥ 4, and the B = 4 scarcity is a
    *thin tail of tens of tokens*, not missing experts.
- **The typical half is slightly less scarce** than the full pool: P(any < 64) is 0.73 vs 0.83
  at B = 4.
- Figures: `fig1_exposure_by_budget`, `fig2_scarcity_probability`.

## 4. Matched allocation design (Experiment B)

- **Strategies:** Full Random (FR), Full Coverage-LP (FC), Typical Random (TR), Typical
  Coverage-LP (TC). Seeds 0/1/2.
- **Warm start:** W(seed, B) = the first B/4 ids of the frozen matched warm start.
  - It is identical across all four strategies.
  - It lies in typical half ∩ random half (asserted).
- **Schedule:** B = 4 is warm 1, then rounds 1 + 2. B = 8 is warm 2, then rounds 3 + 3.
- **Controller:** the unchanged `e3_allocation.select_lp_utility` (u_e = 1/n_e, existing
  exploration, positive-marginal rounding, seeded random fill). Random uses the same
  `Random(seed)` path as `e3_allocation.main`.
- **Compatibility issue (caller-side only; no code changed):**
  - `round_schedule` sets warm = min(B, max(K, B//4)). With the E3 default K = 4 this gives
    warm = 4 at B = 4 (no allocation rounds at all) and (4; 2 + 2) at B = 8.
  - K has no other role in the controller. So the study passes **K = B//4** (1 and 2), which
    yields exactly the prescribed schedules.
  - For B ≥ 16 this is identical to K = 4. Tested, including exact reproduction of the B = 16
    FC/TC selections.
- **Validity:** all 24 selections are valid: B unique ids, exact warm prefix, pool membership,
  no tune/test ids.

## 5. Coverage lower-tail effects (paired, same seed and warm start; all 240 experts)

| B | family | Δmin, s0 / s1 / s2 | Δp10 relative | Δ experts < 128 | Δ experts < 64 | ΔGini (layer mean) |
|---|---|---|---|---|---|---|
| 4 | full | −25 / −10 / −7 | −22.1 / −18.4 / −6.9 % | +3 / +40 / +14 | +7 / +4 / +3 | +0.064 / +0.084 / +0.038 |
| 4 | typical | −2 / −17 / −18 | −10.3 / −6.5 / −13.7 % | +18 / +4 / +22 | +7 / +1 / +2 | +0.052 / +0.013 / +0.054 |
| 8 | full | −2 / −15 / +1 | +0.4 / −4.3 / +5.2 % | 0 / +2 / 0 | 0 / 0 / 0 | −0.006 / +0.023 / −0.018 |
| 8 | typical | −62 / +10 / +15 | +7.9 / +8.2 / +4.8 % | +1 / 0 / 0 | 0 / 0 / 0 | −0.019 / −0.035 / −0.023 |
| 16 (ref.) | full | +47 / −67 / −229 | +9.3 / +2.6 / +4.6 % | 0 / 0 / 0 | 0 | −0.030 / −0.017 / −0.009 |
| 16 (ref.) | typical | +52 / −121 / −10 | +5.5 / +9.8 / +7.2 % | 0 / 0 / 0 | 0 | −0.022 / −0.031 / −0.022 |

Per-seed p5 and below-256 values are in `summary.md`; the deltas are plotted in
`fig3_coverage_lower_tail`.

- **B = 4: Coverage-LP is uniformly harmful to the scarce tail.**
  - On 6/6 paired seeds it lowers p10, adds experts below 128 and 64 tokens, and lowers the
    minimum.
  - It also *raises* Gini on all 6.
- **Why (post-hoc check).** The LP credits each expert min(n_e, added_e) / n_e: E1's
  "double n_e" capacity.
  - With one warm sequence, this rewards picks whose routing resembles the warm sequence,
    roughly doubling its imbalance, rather than lifting the floor.
  - Cosine similarity between the post-warm picks' routing and the warm sequence is
    0.94 (FC) / 0.95 (TC) vs 0.90 (FR / TR) at B = 4, but equal (≈ 0.94) at B = 8
    (`logs/warm_imitation_check.json`).
  - This is a diagnostic, not a prespecified test.
- **B = 8: there is little scarcity to fix.** Typical Coverage lowers Gini on 3/3 seeds and
  raises p10 by 5–8 %. It also removes 13–28 experts below 256, but the threshold is 10 %.
  The minimum fell sharply on one seed (−62). Full Coverage is mixed.
- **B = 16 (reference):** p10 rises and Gini falls, but the minimum falls on 2/3 seeds in each
  family. This is the same "p10 up, minimum down" pattern seen before.

## 6. Zero-exposure handling

- No zero-exposure expert occurred in any trajectory, not even after a single warm sequence.
  So the existing exploration rule was never triggered (0 exploration sequences).
- Every post-warm slot was filled by the LP with positive marginal value. There was no random
  fill and no random-utility fallback.
- Synthetic tests confirm that the unchanged controller handles the cases that never arose
  here:
  - zero-exposure experts at B = 4;
  - more zero-exposure experts than slots: unreached targets are logged, zeros are reported,
    and no smoothing or substitution is applied.

## 7. Routing representativeness

| B | Δ post-warm d(s), full | Δ post-warm d(s), typical | set JS, FR → FC (×1e-3) | aggregate L1, L8, FR → FC |
|---|---|---|---|---|
| 4 | 0.000 / +0.018 / −0.046 | +0.018 / +0.011 / +0.014 | 5.5 → 14.3, 4.1 → 12.8, 5.6 → 8.3 | 0.148 → 0.296, 0.146 → 0.270, 0.154 → 0.250 |
| 8 | +0.053 / +0.064 / −0.034 | −0.012 / +0.006 / +0.003 | 3.7 → 3.8, 2.1 → 4.5, 5.2 → 3.1 | mixed |
| 16 (ref.) | +0.080 / +0.110 / +0.033 | +0.015 / −0.002 / +0.011 | — | — |

- **B = 4.** Coverage picks are not much more atypical one by one (Δd ≈ 0), yet the
  *selected-set* routing moves 1.5–3× further from the pool distribution (set JS, aggregate
  L1). That is the warm-imitation effect.
- **B = 8.** Full Coverage again picks more atypical sequences on 2/3 seeds (+0.05–0.06), as
  at B = 16. Typical Coverage stays representative.
- **Selection overlap.** Coverage and Random post-warm picks share no sequences (Jaccard 0 on
  every seed), so the comparison is between disjoint sets.
- Figure: `fig4_balance_vs_atypicality`.

## 8. Source diversity

- **Tokens come from every selected sequence.** Median unique sources per expert equal B for
  every strategy and budget.
- **Effective sources barely move.** They are 0.95–0.98 B under every strategy:
  - B = 4: FC 3.91 vs FR 3.80.
  - B = 8: FC 7.46–7.54 vs FR 7.58–7.61.
- **Diversity is not what is short.** At these budgets the binding constraint is the number of
  sequences, not how evenly each expert's tokens come from them.
- Figure: `fig5_source_diversity`.

## 9. CPU tests

`tests/test_extreme_low_budget.py`: 20 tests.
- **Schedule and compatibility:**
  - B = 4 / 8 schedules;
  - the documented K = 4 limitation;
  - K = B//4 compatibility for B = 16–128.
- **Metrics:**
  - token accounting;
  - lower-tail metrics on a known vector;
  - Gini;
  - unique and effective sources;
  - probabilities and Wilson intervals;
  - Monte Carlo reproducibility and independent streams.
- **Synthetic scenarios:**
  - random selection yields zero exposure;
  - Coverage eliminates zero exposure at B = 4;
  - a low-budget LP that cannot cover every expert;
  - lower Gini with a worse minimum;
  - a balanced but atypical selection.
- **Decision gate:** all four outcomes.
- **Real selections:**
  - sizes, prefixes, identical warm starts, pool membership, no duplicates or leakage;
  - reproducibility and atypicality consistency;
  - B = 16 backward compatibility with the low-budget sweep.

Full suite: see section 13.

## 10. Compute feasibility (estimated from the B = 16–128 runs)

| item | B = 4 | B = 8 |
|---|---|---|
| Routing profile, 2,000-row pool (once; FC, TC and TR) | 357 s | 357 s |
| Coverage-LP selection | ≤ 1.4 s | ≤ 1.4 s |
| Hessian replay (linear fit 1.8 s + 0.50 s × B, extrapolated) | ≈ 3.8 s | ≈ 5.8 s |
| GPTQ of 240 experts (independent of B) | ≈ 288 s | ≈ 288 s |
| Calibration forward-equivalent | 0.7 s | 1.4 s |
| Profile / calibration forwards | **≈ 500×** | **≈ 250×** |
| Minimal 4 × 3 comparison | ≈ 58 min GPU | ≈ 59 min GPU |
| Zero-calibration-token experts (RTN fallback) | 0 expected | 0 expected |
| Peak GPU memory | ≈ 7.8 GB | ≈ 7.8 GB |

- **The GPU comparison would be cheap**, but the routing profile it depends on costs hundreds
  of times the calibration it is meant to improve.
- **No efficiency claim follows from an equal-B comparison.** At these budgets, simply
  profiling the pool costs more forwards than calibrating with every pool sequence the
  profile reads.

## 11. Scientific decision: EXTREME-SCARCITY-NO-OPPORTUNITY

Prespecified gate, checked by `extreme_low_budget.decide`:

| B | c1: Full Random scarce in ≥ 2/3 seeds | c2: lower tail improved in ≥ 2/3 seeds, full / typical | c3: minimum not systematically worse, full / typical | outcome |
|---|---|---|---|---|
| 4 | **yes (3/3)** | no (0/3) / no (0/3) | no / no | fails |
| 8 | no (0/3) | no (0/3) / no (0/3) | no / yes | fails |

- **No PROMISING budget.** PROMISING needs c1, c2 and c3 at one budget.
- **Not PARTIAL either.** PARTIAL needs scarcity together with at least a Gini or partial
  lower-tail gain. At B = 4, the only scarce budget, Coverage improved neither Gini nor any
  lower-tail metric on any seed.
- **Outcome: NO-OPPORTUNITY.** At the scarce budget, Coverage does not materially improve
  the scarce experts.

## 12. Recommended next experiment

**Stop expert-count-based allocation research.**
- An extreme-scarcity regime does exist (B = 4). There, the existing Coverage-LP objective
  makes the starved tail *worse*: it imitates the warm start instead of lifting the floor.
- No targeted GPU experiment is justified.
- The project report should record this as the final negative result for exposure-based
  allocation.

## 13. Reproduction and status

```
PATH=$HOME/venvs/moe/bin:$PATH python diagnostics/extreme_low_budget.py      # ~35 s CPU
PATH=$HOME/venvs/moe/bin:$PATH python -m pytest tests -q
```

- **Overwrite guard:** the script refuses to overwrite an existing `summary.json`.
- **Post-result edits (layout only):** after the first render, figures 2, 3 and 5 had their
  legends moved and the fig3 title reworded. No data, metric or decision rule changed.
- **Tests:** full suite **158 passed, 0 failed**.
