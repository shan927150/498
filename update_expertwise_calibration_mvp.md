# Expert-wise activation calibration for MoE GPTQ: MVP

**Answer to the main question: no.** Each MoE expert was allowed to choose its own 128 calibration sequences instead of
all experts sharing one set. After matching per-expert routed-token counts, this does not improve held-out expert GPTQ
reconstruction.
- **Primary comparison is a null.** Under the same covariance-aware objective, expert-wise selection (EC) vs shared
  selection (SC) gives **−0.04 % ± 0.05** on the test split and **−0.03 %** on the confirmation split. Only 4/8 experts
  improve, and the sign changes across seeds.
- **The gain comes from covariance-aware selection, not from dropping the shared constraint.**
  - SC vs SR (shared covariance-aware vs shared random): **+0.55 %**, 8/8 experts.
  - EC vs ER (expert-wise covariance-aware vs expert-wise random): **+0.81 %**, 8/8 experts.
- **The unmatched expert-wise advantage is a token-quantity effect.** Without token matching, EC beats SC by +0.23 %
  (8/8 experts). That edge comes from EC choosing sequences that route more tokens to the expert, and it disappears once
  counts are matched.

This is a diagnostic experiment, not a replacement for the fixed global sequence budget E3 protocol. It measures
isolated-expert reconstruction only. No full-model PPL or inference-efficiency claim is made.

All settings, the expert choice and the decision rule were frozen in `runs_expertwise_mvp/protocol.json` before any
selection or reconstruction was measured. Code hashes were frozen in `freeze.json` after a passing seed-0 smoke run, and
nothing was tuned afterwards. The final seed-0 run reproduced the smoke run bit-for-bit.

## Setup

- **Model:** Qwen1.5-MoE-A2.7B, partial load of layer 16 only, BF16. Pretrained weights, routing, the GPTQ solver and
  the evaluation metric are unchanged. The expert and router weight digest is checked at the start and end of every
  stage.
- **Candidate pool:** the existing 2,000 C4 sequences (`~/moe_cache` rows 0–1999), length 512. The tune rows
  (2000–2127) are not used.
- **Experts:** 8, chosen by L16 pool routed-token count from the existing `runs_allocation_minimal/routing_profile_v2.npz`.
  - **Rule:** sort ascending, ties to the lower ID, then take ranks round(i·59/7).
  - **Selected (rank):** 1 (0), 26 (8), 52 (17), 23 (25), 29 (34), 2 (42), 7 (51), 30 (59).
  - **Pool tokens:** 51k–84k. Layer-16 routing is fairly balanced.
- **Budget and seeds:** B = 128 sequences, allocation seeds 0/1/2.
- **GPTQ:** `quantize_expert` unchanged; 4-bit, g128, act-order, percdamp 1.0, H = Σ x xᵀ in fp32.
- **Metric:** the E3 definition, `per_seq_rel_errors` over gate+up+down.
  - It is averaged over held-out sequences, then over the 8 experts.
  - The cap is 128 routed tokens per sequence per expert (Profiler default).
  - **Primary split:** test rows 2128–2255. **Secondary split:** confirmation, 256 sequences.
  - Every condition uses identical held-out activations.

### Conditions (2 × 2)

| | random | covariance-aware |
|---|---|---|
| **shared set** | SR: one random set for all experts | SC: one set minimizing mean_e D_e(S) |
| **expert-wise sets** | ER: an independent random set per expert | EC: per expert, S_e minimizing D_e(S_e) |

### Selection objective

- **Sketches:** fixed Gaussian projections R_p ∈ ℝ^{d_p×64} with N(0, 1/64) entries and seed 20261116, giving
  z = R_pᵀx for p ∈ {gu, down}.
- **Covariance of a set:** H_{e,p}(S) = Σ_{s∈S} Σ_t z zᵀ / Σ_{s∈S} n_{s,e}. The reference H^pool is the same quantity
  over all 2,000 pool sequences.
- **Mismatch:** D_e(S) = ½ Σ_p ‖H_{e,p}(S) − H_{e,p}^pool‖_F / (‖H_{e,p}^pool‖_F + ε).
- **Empty token set:** H(S) := 0, so that term is ≈ 1.
- **Data used:** only the pool, never tune or test data.

### Optimizer

1. **Seeded stochastic greedy.** At each step it picks the argmin over a uniform 10 % sample of the remaining candidates.
2. **Best-improvement swap passes.**
   - Slots are visited in seeded order.
   - Each out-of-set replacement is scored in closed form from cached per-sequence sketches, so no extra model forwards
     are needed.
   - A swap is accepted only if the *exact* objective decreases (relative tolerance 1e-9).
   - At most 5 passes.

The accepted objective trajectory was checked to be monotone in every run.

### Token matching

- **Quota:** k_e = min over SR/SC/EC/ER of the expert's routed-token count.
- **Sampling:** each condition draws exactly k_e (sequence, dispatch-row) pairs uniformly without replacement, with a
  seeded RNG per (seed, condition, expert).
- **Alignment:** the same row indices address both the gate/up input and the down input.

### Pipeline (`moe_calib/expertwise_calibration.py`)

- **Sketch pass:** one replay of the 2,000 pool sequences stores per-(sequence, expert, projection) 64-d sketches and
  counts on disk.
- **Diagnostic reference:** a full-dimensional pool Hessian is also stored. It is never used for selection.
- **Selection:** runs on the sketches only.
- **Final pass:** per seed, one replay of the union of all selected sequences builds the matched Hessians. It also
  builds unmatched Hessians for a secondary analysis that does not control token quantity.

## Correctness checks (all passed, every seed)

| Check | Result |
|---|---|
| Natural routing: gate_proj rows == router input at router-selected positions (bitwise, slot-major dispatch order) | 16,000 sketch-pass checks; 10,792 / 10,768 / 10,688 final-pass checks |
| gate/up–down alignment: down_proj rows == act(gate(x))·up(x) of the same rows (bitwise) | same counts |
| Layer outputs with / without hooks | bitwise identical |
| Per-sequence routed counts vs existing routing profile and Profiler | equal for all 2,000 × 8 |
| Exactly 8 predetermined experts; 128 distinct pool IDs per set; shared conditions share | yes |
| Identical matched token counts across SR/SC/EC/ER per expert | yes |
| Replay counts and per-sequence sketches reproduce the sketch pass | yes (max relative difference 0.0) |
| Pre-matching mismatch from the final replay == selection-time value | yes |
| Same-seed reruns give identical selections and matching plans | yes |
| Objective never increases on accepted swaps | yes |
| No tune/test rows in selection or references | yes |
| Frozen solver, evaluation and E3 code hashes unchanged | yes |
| Smoke seed 0 ≡ final seed 0 (selections, quotas, all errors) | bit-for-bit |

Unit tests: `tests/test_expertwise_calibration.py`, 17 passed. They cover:
- closed-form vs exact objective agreement;
- the empty-token rule;
- determinism and monotonicity;
- quota and alignment of token matching;
- tiny-model capture, misalignment detection, and an end-to-end GPTQ run.

The existing E3 tests (`test_p1p2.py`, `test_allocation_minimal.py`) still pass, 32/32. No existing module was modified.

## Results

### Mean held-out error (token-matched)

| condition | test, seeds 0 / 1 / 2 | test mean | confirm mean |
|---|---|---|---|
| SR | 0.005123 / 0.005136 / 0.005138 | 0.005133 | 0.005091 |
| SC | 0.005108 / 0.005101 / 0.005106 | **0.005105** | **0.005060** |
| EC | 0.005108 / 0.005103 / 0.005110 | 0.005107 | 0.005061 |
| ER | 0.005152 / 0.005153 / 0.005140 | 0.005148 | 0.005107 |

### Paired relative improvement

r_e = (err_Y − err_X)/err_Y, averaged over the 8 experts; positive means X is better.

The pre-registered rule has two levels:
- **Consistent:** positive in all 3 seeds AND ≥ 6/8 experts improved on the seed average.
- **Material:** also ≥ 0.5 %.

| comparison | isolates | test, seeds 0 / 1 / 2 | test mean | confirm mean | experts improved | verdict (test) |
|---|---|---|---|---|---|---|
| **EC vs SC** | dropping the shared constraint, with covariance-aware selection | +0.00 / −0.03 / −0.09 % | **−0.04 %** | −0.03 % | 4/8 | no reliable effect |
| EC vs ER | covariance-aware selection, expert-wise sets | +0.86 / +0.99 / +0.59 % | **+0.81 %** | +0.91 % | 8/8 (24/24 expert×seed) | consistent, material |
| SC vs SR | covariance-aware selection, shared set | +0.29 / +0.70 / +0.65 % | **+0.55 %** | +0.62 % | 8/8 | consistent, material |
| ER vs SR | dropping the shared constraint, random selection | −0.57 / −0.33 / −0.02 % | −0.31 % | −0.32 % | 1/8 | consistent degradation, below 0.5 % (mixed on confirm) |

**Secondary analysis without token matching (token quantity not controlled):** EC vs SC is +0.22 / +0.25 / +0.21 %,
mean **+0.23 %**. On test 8/8 experts improve and 22/24 expert×seed pairs win; confirm gives the same +0.23 %. Covariance-aware
selection prefers sequences rich in the expert's tokens. EC carries 4–14 % more tokens than SC, so matching discards
11.5 % of EC's tokens against 4.5 % of SC's. The unmatched advantage is therefore a token-quantity effect, and it
disappears after matching.

Per-expert results are in `runs_expertwise_mvp/summary.md` §3 and `figures/per_expert_improvement.png`. In EC vs SC,
experts 52 and 2 are slightly worse in every seed (−0.21 % and −0.22 % on average). The other six have 3-seed means
within ±0.1 %, with seed-level values flipping sign. No frequency stratum benefits.

### Covariance mismatch vs reconstruction

| condition | sketch D_e, pre / post matching | full-dim relative Frobenius, pre / post |
|---|---|---|
| SR | 0.178 / 0.179 | 0.426 / 0.429 |
| SC | 0.119 / 0.121 | 0.334 / 0.339 |
| EC | **0.093 / 0.0995** | **0.314 / 0.326** |
| ER | 0.177 / 0.178 | 0.421 / 0.421 |

- **EC lowers the surrogate mismatch more than SC** in **24/24** expert×seed cases (−18 % after matching), and it lowers
  full-dimensional mismatch on average.
- **That reduction does not predict reconstruction.** Spearman(Δmismatch, Δerror) for EC vs SC is ρ = −0.02 for the
  sketch (p = 0.91) and −0.25 for the full-dimensional mismatch (p = 0.23), n = 24.
- **The pooled correlation is driven by random vs covariance-aware.** Pooled over SC/EC/ER relative to SR, ρ ≈ +0.70
  (n = 72). Within the covariance-aware regime, further mismatch reduction buys nothing measurable
  (`figures/mismatch_vs_error.png`).

### Token counts, diversity, overlap

- **Matched quotas:** k_e ranges from 3.0k (expert 1) to 5.4k (expert 30) tokens per expert.
- **Distinct sequences:** every set has 128.
- **Effective source diversity** (inverse Simpson over the matched tokens):

  | condition | effective diversity |
  |---|---|
  | SR | 111.0 |
  | SC | 115.8 |
  | EC | 120.3 |
  | ER | 111.1 |

- **Union of sequences across the 8 experts:**

  | condition | union |
  |---|---|
  | SR / SC | 128 |
  | EC | 771–787 |
  | ER | 819–827 |

- **Overlap:** mean Jaccard between EC and SC is 0.06–0.07, and between pairs of EC experts 0.04–0.05. The expert-wise
  sets are nearly disjoint from each other and from the shared set.

### Cost

| item | value |
|---|---|
| Sketch pass (2,000 sequences, 8 experts, plus full-dim reference) | 66 s, vs 59 s for a plain replay (+11 %); peak GPU 2.0 GB, RSS 8.7 GB |
| Sketch storage | 533 MB for 8 experts (≈ 4 GB per layer for all 60); file incl. diagnostic reference 730 MB in `~/moe_cache_expertwise_mvp` |
| Selection (all four conditions, CPU) | 56–62 s per seed |
| Final capture replay | ~1,340 sequences, 33 s (0.024 s per sequence); peak GPU 3.4 GB |
| GPTQ + evaluation (64 quantizations, two splits) | 122–148 s; peak GPU 2.3 GB |
| Acquisition | SR/SC forward 128 sequences; EC/ER need ~6× more for only 8 experts, approaching the whole pool for 60 |

## Interpretation

The factors the protocol asked to separate:

| factor | finding |
|---|---|
| Relaxing the shared-sequence constraint | **null** once tokens are matched (EC ≈ SC). With random selection it is, if anything, slightly negative (ER vs SR), though that rests on only 3 independent shared draws. |
| Covariance-aware surrogate | the only consistent, material effect (+0.55 % shared, +0.81 % expert-wise), on both evaluation splits. |
| Larger union of unique sequences | EC reads ~6× more distinct sequences overall, but each expert still sees 128, and its slightly higher diversity does not turn into lower error. This is a cost, not a measured benefit. |
| Per-expert token budget vs global sequence budget | the unmatched EC advantage (+0.23 %) is per-expert token quantity. Matching removes it, and EC still needs a far larger global acquisition budget. |
| Surrogate vs actual GPTQ improvement | a lower covariance mismatch (24/24 cases) is not sufficient evidence of a better quantizer here. |

All effect sizes are below 1 % relative on an isolated-expert metric.

## Limitations

- One layer (L16, balanced routing), 8 of 60 experts, B = 128, 3 allocation seeds. Layers with rarer experts or lower
  budgets could behave differently.
- The shared conditions supply only 3 independent random draws (one per seed, shared by all experts), so SR-based
  comparisons carry more seed noise than their 8-expert averages suggest.
- The reference covariance is the candidate pool's own, not the evaluation distribution's. The optimizer is a heuristic
  on a 64-d sketch.
- The test and confirmation splits were inspected in earlier studies, but neither was used for any selection or setting
  here.
- Isolated-expert reconstruction only: no full-model PPL, no inference efficiency, no comparison against the existing
  E3 allocation strategies.

## Recommendation

Do not pursue expert-wise sequence selection as a replacement for shared-sequence E3 calibration. If anything is carried
forward, it is the *shared* covariance-aware objective. SC gains +0.55–0.62 % over random at unchanged acquisition cost.
It has not been compared against the existing fixed-budget E3 strategies and would need that test before any claim.

## Reproduction

```bash
source ~/venvs/moe/bin/activate
scripts/run_expertwise_mvp.sh all     # tests -> protocol -> sketch -> smoke (seed 0) -> freeze -> final seeds 0-2 -> report
# individual stages: scripts/run_expertwise_mvp.sh {tests|protocol|sketch|smoke|freeze|final|report}
```

Every stage resumes, skipping work whose outputs exist. Final runs refuse to start if the solver, evaluation or E3
code changed since `protocol.json`, or if the experiment code changed since `freeze.json`.

## Artifacts

- **Code:**
  - `moe_calib/expertwise_calibration.py`
  - `diagnostics/expertwise_calibration_mvp.py`
  - `diagnostics/report_expertwise_calibration_mvp.py`
  - `tests/test_expertwise_calibration.py`
  - `scripts/run_expertwise_mvp.sh`
- **Settings:** `runs_expertwise_mvp/protocol.json` (frozen settings, experts, decision rule) and `freeze.json` (code
  hashes after smoke).
- **Logs and selections:** `runs_expertwise_mvp/logs/sketch.json` (sketch-pass checks and cost) and
  `selections/seed{0,1,2}.json` (IDs, optimizer traces, pre-matching mismatch, overlap).
- **Results:** `runs_expertwise_mvp/results/{smoke,final}_seed*.json`, plus `*_perseq.npz` with per-sequence errors.
- **Report:** `runs_expertwise_mvp/summary.md`, `summary.json`, `interpretation.md`, and
  `figures/{per_expert_improvement,mismatch_vs_error}.png`.
- **Outside the repo:** `~/moe_cache_expertwise_mvp/sketch.pt` (per-sequence sketches and full-dim pool reference).
