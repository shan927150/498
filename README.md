# Calibration-Sufficiency-Aware Budget Allocation for MoE PTQ

CS 498 research project on calibration-data allocation for post-training
quantization (PTQ) of Mixture-of-Experts models.

**Original research question.** Can an MoE expert's current calibration
state predict the marginal GPTQ reconstruction benefit of additional
calibration data *beyond routed-token count*, and can that information improve
calibration allocation under a fixed sequence budget?

**Where it stands (P0–P2c).** Under strict held-out validation, the
*dynamic-state* hypothesis was **not supported**: drift, relSE, router margin
and entropy add no robust value beyond a flexible, layer-aware exposure
model (P2 CASE B, P2b UNSTABLE). What the data do support is a revised,
exposure-only hypothesis:

> The marginal value of routed-token exposure is strongly **layer- and
> calibration-regime-dependent**.

It holds on held-out seeds and held-out experts, and it ranks experts better
than plain coverage (P2c LA-A).

**Minimal allocation experiment (11/06): ALLOCATION-NO-GAIN.** The
question was whether that advantage survives sequence-routing coupling in
real whole-sequence selection.
- Setup: B = 128, seeds 0/1/2, frozen GPTQ, test split.
- Layer-aware utility matches matched-controller LP coverage: +0.02 % mean;
  per seed −0.32 / −0.28 / +0.67 %.
- **Every** routing-driven strategy is 1.1–1.8 % *worse than random* on every
  seed. This includes greedy coverage, LP coverage, M1, Dlb and M4.
- The adaptive picks are atypical sequences. Their per-expert token counts
  change by only ≈5 %. See [`update_11.06.md`](update_11.06.md).

> **Final:** [`update_aggregate_constraint.md`](update_aggregate_constraint.md)
> (aggregate-shift-constrained Coverage-LP on a locked, document-disjoint confirmation split:
> AGGREGATE-ALLOCATION-PARTIAL — beats Full Coverage-LP by 1.7 % but is 1.5 % worse than
> Typical Random on 3/3 seeds; exposure-based allocator development ends here;
> [`runs_aggregate_constraint/summary.md`](runs_aggregate_constraint/summary.md));
> **Previous:** [`update_repr_constrained.md`](update_repr_constrained.md)
> (representativeness-constrained Coverage-LP: REPR-ALLOCATION-NO-GAIN — mitigates
> the coverage penalty by 1.6 % but stays 1.0 % worse than Typical Random on 3/3 seeds);
> [`update_matched_typicality_11.06.md`](update_matched_typicality_11.06.md)
> (matched-warm-start typicality validation: MATCHED-TYPICALITY-SUPPORTED);
> [`update_11.06.md`](update_11.06.md) (minimal allocation + unmatched typicality test).
> **Detailed experimental history (P0 → P2c, with numbers):**
> [`update_10.29.md`](update_10.29.md). Earlier log:
> [`update_10_18.md`](update_10_18.md). Run order and stop rules:
> [`NEXT_STEPS.md`](NEXT_STEPS.md).

Setup: **Qwen/Qwen1.5-MoE-A2.7B** (60 routed experts/layer, top-4),
**4-bit weight-only GPTQ** (group size 128), **C4** calibration data,
**expert-level reconstruction** evaluation (plus WikiText-2 PPL).

## Research principle

*Calibration sufficiency is defined by FUTURE held-out GPTQ reconstruction
improvement from additional calibration data.* Candidate predictors of that
future benefit:

| signal | source |
|---|---|
| routed-token count | Hessian accumulator |
| normalized Hessian drift | prefix vs. full normalized block means |
| relative Hessian standard error (relSE) | between-block variability |
| router margin | k-th minus (k+1)-th router probability |
| router entropy | router distribution |

Statistical significance is **not** treated as sufficiency: a t-test only
says the benefit is nonzero, while allocation needs its *magnitude*. In P1,
100 % of expert/budget transitions were significant, but the size of the
remaining benefit varied systematically by budget and layer. A signal is
useful only if it predicts future benefit beyond a **flexible** exposure
baseline on held-out characterization seeds, held-out layers or held-out
experts.

The GPTQ algorithm is identical across all allocation strategies. A single
GPTQ configuration was chosen on tuning data and frozen *before*
characterization and final evaluation: **4-bit, group size 128, act-order on,
`percdamp = 1.0`**, recorded in [`configs/frozen_b4.json`](configs/frozen_b4.json)
(see [Damping study](#damping-study)).

## Current pipeline

```
Step 0  cache inputs                        cache_inputs.py
   ↓
E1      characterize future GPTQ benefit    e1_label_sufficiency.py
   ↓
E2      test signals beyond token count     e2_signals.py
   ↓
        support diagnostic                  support_diagnostic.py
   ↓
E3      adaptive fixed-budget allocation    e3_allocation.py
   ↓
        final GPTQ reconstruction / PPL     e3_allocation.py --ppl
```

Validation gates between E1/E2 and E3 (read-only, CPU, on E1/E2 outputs):

```
P1   e1_gate.py                           is future benefit a stable target?
P2   e2_gate.py                           state signals vs flexible exposure
P2b  diagnostics/p2b_regime_diagnostic.py budget- vs layer-conditioned signals
P2c  diagnostics/p2c_layer_exposure.py    layer/regime-aware exposure model + ranking
```

### Step 0 — Activation cache (`cache_inputs.py`)

Runs the full model once and stores the *input* hidden states of each
profiled layer. Defaults of the real setup:

| | |
|---|---|
| model | Qwen/Qwen1.5-MoE-A2.7B |
| profiled layers | 1, 8, 16, 23 |
| pool | 2000 C4 sequences |
| holdout | 256 (tune 128 + test 128) |
| sequence length | 512 |

The bf16 model (~29 GB) stays in CPU RAM and decoder layers are moved to the
GPU one at a time, so everything fits an ICRN A100 20 GB MIG slice
(72 GB RAM). E1–E3 replay single layers on the cache (characterization is
non-sequential, so layers are independent).

### E1 — Future-benefit characterization (`e1_label_sufficiency.py`)

For a starting calibration set of n0 sequences, E1 quantizes each expert
twice from the same FP weights — GPTQ with **n0** sequences vs. GPTQ with
**2·n0** sequences (the base plus a disjoint extension) — and scores both on
the **same tuning activations**. The primary label is continuous:

```
future_rel_delta = mean_s(err_s(n0) - err_s(2 n0)) / mean_s err_s(n0)
```

(positive = more data helped). Paired one-sided t-tests with BH-FDR are
secondary diagnostics only.

- `--char-seed` selects the characterization draws (real runs: 0, 1, 2).
- Output dir `runs/e1_b{bits}_s{seed}_{hash}/` is fingerprinted on the full
  config; same config resumes from `raw.jsonl`, a mismatching dir is rejected.
- Low-count experts are kept whenever technically valid
  (`--min-calib-tokens`, default 1; `--min-paired-seqs`, default 8).
- Every exclusion is written to `excluded.csv` with a reason
  (`zero_calib_tokens`, `below_min_calib_tokens`,
  `insufficient_paired_eval_seqs`, `nonfinite_error`, `nonpositive_base_error`).
- Only the tuning split is used; the final test split is never touched.

### E2 — Signal validation (`e2_signals.py`)

Fits, per signal, `future_rel_delta ~ b0 + b_count·log(n) + b_sig·z(signal)`
(count-only for `count`) and asks whether any signal predicts the label
better than routed-token count. With ≥2 characterization seeds the selection
criterion is **leave-one-characterization-seed-out MSE**
(`--min-char-seeds 3` for real runs); with one seed the result is marked
`preliminary` and E3 refuses it unless `--allow-preliminary`.

Outputs: `runs/e2_signals_b{bits}.csv` (observations) and
`runs/e2_model_b{bits}.json`, which stores the selected signal, predictor
coefficients, signal orientation, fitted count support, characterization
seeds, selection criterion and metrics, split, K and block seed.

> E2 artifacts from the original damping (`*_b4.*`, `percdamp = 0.01`) are
> diagnostic only. The frozen-damping artifacts are `*_b4_d1*`. Their
> `e2_model_b4_d1.json` names relSE via E2's own linear-count selection,
> which the P2 gate rejected; it **must not** be consumed by E3.

### Support diagnostic (`support_diagnostic.py`)

The predictor is only valid inside the routed-count range E2 was fitted on,
so coverage is checked in **per-expert routed-token space**, not in global
sequence-budget space. It uses the same `round_schedule()` as E3. With K=4
and 2 rounds (warm start `min(B, max(K, B//4))`):

| final B | predictor decisions at (sequences) |
|---|---|
| 64 | 16, 40 |
| 128 | 32, 80 |
| 256 | 64, 160 |
| 512 | 128, 320 |

At each checkpoint every expert falls into exactly one category: **zero**
tokens, **below-support** (nonzero), **inside-support**, or
**above-support**. `FINAL_STATE_ONLY` reports the final-B state, where the
predictor is *not* evaluated. Outputs:
`runs/support_diagnostic_b{bits}.{json,csv}`.

### E3 — Adaptive allocation (`e3_allocation.py`)

Strategies: `random`, `random_plus` (equal-compute baseline), `coverage`,
`tlp` (sufficiency-aware), `oracle` (E1 labels).

For an expert with current routed count n_e, the E2 model predicts
`delta_hat_e`, the expected reconstruction improvement from roughly doubling
its current calibration exposure. Each round solves an LP over candidate
sequences with their natural routing profiles:

```
capacity_e          = n_e
per_token_value_e   = max(delta_hat_e, 0) / n_e
maximize  sum_e w_e * per_token_value_e * y_e
```

- `--weighting uniform` (w_e = 1) is the default/main interpretation;
  `--weighting usage` (w_e = p_e) is an explicit ablation.
- Zero-exposure and below-support experts are reached by an exploration
  cover, capped at the round budget.
- Above-support experts receive no invented benefit estimate (no
  extrapolation outside the fitted support).
- If the predictor gives no positive marginal preference, remaining slots
  are filled by seeded random sampling.
- Every strategy uses the same fixed budget B; natural MoE routing is
  preserved; the GPTQ solver is not changed by the allocation method.

E3 has **not** produced final research results yet. Its utility is still the
original signal-based `delta_hat`. The next experiment replaces it with
layer/regime-aware exposure utilities, leaving the LP, routing and GPTQ
unchanged.

## Data isolation

```
C4 candidate pool   -> routing / calibration / Hessians / allocation
C4 tuning split     -> E1 labels, E2 model selection, GPTQ hyperparameter selection
C4 final test split -> final E3 reconstruction evaluation only
WikiText-2          -> model-level PPL
```

Test data is never used to select signals, select GPTQ damping, fit
predictors, or choose allocation hyperparameters.

## Fair expert evaluation

- All routed experts in the profiled layers are quantized under every strategy.
- Experts that receive zero calibration tokens use the same RTN-equivalent
  fallback (GPTQ with H = I, act-order off); no strategy keeps free BF16
  experts because it failed to calibrate them.
- The reconstruction evaluation universe is determined only from the fixed
  test routing and is identical (asserted) for all strategies.

## Routing profiler fix

The router hook used to infer selected experts with `topk(k+1)[:k]`, while
the model routes with `topk(k)`. Router scores are bf16 and contain exact
ties, which the two calls can break differently. Now selected experts come
from `topk(k)` and `topk(k+1)` values are used only for the router margin.
A real-model check after the fix showed **zero** disagreement between the
routing profile and actual expert execution. E1 labels and E1/E2 routed
counts were never affected (they come from expert inputs); the E3 routing
matrix, the old support diagnostic, and the per-expert attribution of the
E2 margin/entropy signals used the old rule.

## Current GPTQ calibration finding

The original GPTQ configuration uses `percdamp = 0.01`. Real
characterization revealed severe low-data overfitting. On strictly nested
calibration prefixes (3 seeds × 4 layers × 60 experts), mean tuning error was:

| sequences | 16 | 32 | 64 | 128 | 256 | RTN |
|---|---|---|---|---|---|---|
| tuning error | 0.00949 | 0.01038 | 0.00834 | 0.00581 | 0.00482 | 0.00679 |

Going from 16 to 32 sequences made held-out reconstruction worse for about
89–90% of experts. Diagnostics showed that Hessian accumulation and E1
labelling are correct, that nested calibration sets reproduce the effect,
and that error on the calibration activations improves while tuning error
worsens. The effect is therefore genuine low-data GPTQ overfitting /
under-regularization, not an E1 implementation bug. The exact mechanism is
not established.

## Damping study

Controlled fixed-damping study on the tuning split: `percdamp ∈ {0.01,
0.03, 0.05, 0.10}`, act-order on, 2 characterization seeds, 4 layers,
nested prefixes 16/32/64/128/256, RTN reference (values averaged over
prefixes):

| percdamp | mean tuning error | mean Δ vs RTN | % experts better than RTN |
|---|---|---|---|
| 0.01 | 0.00775 | +0.00097 | 44.5% |
| 0.03 | 0.00673 | −0.00006 | 54.5% |
| 0.05 | 0.00629 | −0.00050 | 62.3% |
| 0.10 | 0.00576 | −0.00103 | 76.3% |

0.10 was best but at the grid edge, so P0 (`scripts/p0_damping.sh`,
`diagnostics/damping_threshold.py`) extended the grid on the same design
(seeds 0/1, layers 1/8/16/23, prefixes 16–256, tuning split only):

| percdamp | 0.01 | 0.1 | 0.2 | 0.3 | 0.5 | **1.0** | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|---|
| mean tuning error | 0.00775 | 0.00576 | 0.00533 | 0.00514 | 0.00497 | **0.00487** | 0.00487 | 0.00490 |
| % experts better than RTN | 44.5 | 76.3 | 93.5 | 98.8 | 100 | 100 | 100 | 100 |
| 16 → 32 improved % | 10 | 93 | 100 | 100 | 100 | 100 | 100 | 100 |

- **P0 is complete; production damping is frozen at `percdamp = 1.0`**, the
  best overall value and interior to the grid (1.5 is within 0.05 % and not
  better; 2.0 is worse). All doubling transitions improve for 100 % of
  experts at 1.0.
- The per-prefix optimum drifts with budget (2.0 at 16–32 sequences, 0.5 at
  256); one fixed value is used by design.
- The value was selected on the tuning split before final characterization.
  It must **not** be retuned using E1/E2 results or the test split.
- `moe_calib/gptq.py` keeps its legacy default of 0.01 so that old runs stay
  interpretable; production runs pass the frozen value explicitly (E1 and the
  E1 gate validate against `configs/frozen_b4.json` via `--frozen-config`).

## Current experimental status

| component | status |
|---|---|
| Real Qwen/CUDA smoke | complete |
| Data split (pool / tune / test) and activation cache | complete |
| Routing profiler correctness fix | complete |
| Low-data GPTQ diagnosis | complete — genuine under-regularization, not a bug |
| Damping freeze (P0) | complete — `percdamp = 1.0` |
| P1 target validation (E1 + gate) | **PASS** |
| P2 dynamic-state validation (E2 + gate) | **CASE B** — not robust |
| P2b instability diagnosis | **UNSTABLE** — layer heterogeneity dominates |
| P2c layer-aware exposure validation | **LA-A** — supported |
| Minimal layer-aware allocation experiment (B = 128) | **evaluated — ALLOCATION-NO-GAIN** (all adaptive strategies lose to random) |
| Typicality-controlled allocation test | **evaluated — TYPICALITY-SUPPORTED** (coverage penalty +1.32 % → +0.01 % in the typical half; random-half control +1.43 %) |
| Matched-warm-start typicality validation | **evaluated — MATCHED-TYPICALITY-SUPPORTED** (T-C vs F-C −2.47 %, I and J < 0 on 3/3 seeds) |
| Representativeness-constrained Coverage-LP (mean d ≤ 0.321) | next — not yet run |
| Multi-budget allocation evaluation (64/256/512) | pending |
| PPL | pending |
| Equal-compute evaluation | pending |
| Overhead / peak memory | pending |
| 3-bit robustness | pending |

Key results (details and tables in [`update_10.29.md`](update_10.29.md);
reports in `runs/`, which is gitignored):

- **P1 — PASS** (`runs/e1_gate_b4_d1.md`). Median `future_rel_delta`:
  +0.057 / +0.057 / +0.045 / +0.031 for n0 = 16 / 32 / 64 / 128. 0 % of experts
  are negative at every budget, every seed and layer is positive, and returns
  diminish.
- **P2 — CASE B** (`runs/e2_gate_b4_d1.md`). The flexible exposure baseline
  alone beats linear count by +12.7 % (LOSO) / +11.0 % (LOLO). Against it,
  relSE/drift with budget-specific slopes reach ≈ +9 % pooled LOSO
  (permutation p = 0.005). But seed 0 and held-out layers 16/23 get worse,
  and the slopes change sign between low and high budgets. Margin/entropy
  collapse when a layer is held out. **Negative finding for the original
  hypothesis.**
- **P2b — UNSTABLE** (`runs/p2b_regime_b4_d1.md`).
  - Fixing the budget does not help: no signal is robust in any budget.
  - A per-layer exposure model alone gains +30.2 % LOSO; a per-budget one
    gains +0.1 %.
  - Against per-layer exposure, drift/relSE give −1.5 / −1.4 %.
  - Margin/entropy are ~99 % explained by expert identity, i.e. static
    expert descriptors rather than calibration state.
  - **This negative result stands.**
- **P2c — LA-A** (`runs/p2c_layer_exposure.md`). Exposure-only; no state
  signals.
  - Layer intercepts gain +10.1 % LOSO over global flexible exposure.
  - A layer-specific curve gains a further **+22.5 %** (every held-out seed;
    permutation p = 0.001), about two thirds of the total.
  - It transfers to held-out experts: +24.6 % over intercepts, every fold
    positive.
  - The most valuable layer changes with regime: L1 at n0 ≤ 32, L8 at n0 = 128.
  - Ranking experts at a fixed budget, it captures 0.962 of the oracle's
    utility, versus 0.934 for coverage 1/n. It wins in all 12 (seed, budget)
    cells: +7 pp at n0 = 16, +1–2 pp at n0 ≥ 32.
  - **Caveat:** routed count is ~proportional to budget, and layer × budget
    means (no continuous layer × count curve) perform just as well. The
    supported claim is *layer- and regime-dependent* marginal value, not a
    uniquely identified continuous curve. The ranking also ignores
    sequence-routing coupling.

- **Minimal allocation — ALLOCATION-NO-GAIN** (`runs_allocation_minimal/summary.md`,
  [`update_11.06.md`](update_11.06.md)). Setup: B = 128, seeds 0/1/2, 18/18
  runs, mean expert reconstruction error on the test split.

  | strategy | error | vs random |
  |---|---|---|
  | random | 0.004616 | – |
  | coverage (greedy) | 0.004683 | +1.45 % |
  | coverage_lp | 0.004677 | +1.32 % |
  | global_exposure (M1) | 0.004666 | +1.09 % |
  | layer_budget (Dlb) | 0.004700 | +1.82 % |
  | layer_aware (M4) | 0.004678 | +1.35 % |

  - Layer-aware vs coverage_lp: +0.02 % (seeds −0.32 / −0.28 / +0.67 %).
    The selected sets differ (18–32 % of adaptive picks shared), but quality
    does not.
  - Adaptive picks are atypical sequences (routing L1 to pool 0.40–0.42 vs
    0.33 for random). Counts move by only ≈5 %.
  - Selection overhead: 2,000 full-model forwards for the routing profile.

- **Typicality test — TYPICALITY-SUPPORTED** (`runs_typicality/summary.md`,
  [`update_11.06.md`](update_11.06.md) §10).
  - Typical half = the 1,000 sequences with the lowest routing L1 to the
    pool.
  - Coverage_lp's penalty versus random falls from +1.32 % (full pool) to
    +0.01 % (typical half). A random half does not do this (+1.43 %).
  - Interaction and pool-size control are both negative on 3/3 seeds.
  - Count balancing becomes harmless, not beneficial. Typical random beats
    full random by only 0.17 %.

- **Matched warm start — MATCHED-TYPICALITY-SUPPORTED**
  (`runs_typicality_matched/summary.md`,
  [`update_matched_typicality_11.06.md`](update_matched_typicality_11.06.md)).
  - All 6 conditions of a seed share the same ordered 32-sequence warm start
    drawn from typical ∩ random half.
  - Typical eligibility lowers Coverage-LP error by 2.47 %: −2.35 / −2.50 /
    −2.55 % per seed, in every layer.
  - Coverage penalty vs Random: +2.48 % (full), +0.17 % (typical),
    +1.78 % (random half). I and J are negative on 3/3 seeds.
  - Token balance is unchanged while error drops. Source diversity moves
    with typicality (confounded).
  - Typical Coverage-LP still does not beat typical Random.

**Next step — representativeness-constrained Coverage-LP** on the full pool
(mean d of adaptive picks ≤ 0.321) under the same matched warm starts. No
budget sweep, PPL or 3-bit before that.

## Running the code

Minimal allocation experiment (11/06):

```bash
python diagnostics/export_allocation_models.py           # P2c M1/Dlb/M4 -> runs/allocation_models_b4_d1.json
SEEDS=0 bash scripts/run_minimal_allocation.sh            # smoke (all 6 strategies, seed 0)
SEEDS="1 2" PAR=2 bash scripts/run_minimal_allocation.sh  # remaining seeds; resumable
python diagnostics/report_minimal_allocation.py           # runs_allocation_minimal/summary.{md,json}
```

Routing-typicality test (reuses the full-pool runs above):

```bash
python diagnostics/typicality_pools.py                    # pools from routing_profile_v2.npz (CPU)
bash scripts/run_typicality_experiment.sh                  # 12 filtered-pool runs; resumable, PAR=2
python diagnostics/report_typicality.py                    # runs_typicality/summary.{md,json}
```

Matched-warm-start typicality validation:

```bash
python diagnostics/typicality_warm_starts.py              # shared warm starts W_s from typical ∩ random half
bash scripts/run_typicality_matched.sh                     # 18 runs; seed-0 smoke + warm-start check; resumable
python diagnostics/report_typicality_matched.py            # summary.{md,json} + figures/
```


Environment (ICRN): a venv with a CUDA build of torch that matches the
driver, plus `requirements.txt`; `scripts/setup_icrn.sh` is the original
one-time setup. Use the A100 option (the model needs ~29 GB of CPU RAM).

```bash
# end-to-end smoke on a tiny setting (1 layer, 64-sequence pool; PRELIMINARY E2)
bash scripts/run_smoke.sh 2>&1 | tee logs/smoke.log

# P1: E1 seeds 0,1,2 under the frozen config (percdamp 1.0) -> E1 gate, then stop.
#   Resumable: rerun the same command. Without P1_ONLY=1 it continues to E2 + E2 gate (P2).
P1_ONLY=1 nohup bash scripts/p1p2_characterize.sh > logs/p1.log 2>&1 &

# legacy (1% damping) characterization: cache (if missing) -> E1 -> E2 -> support diagnostic
mkdir -p logs && nohup bash scripts/run_characterization_b4.sh > logs/characterization_b4.log 2>&1 &

# support diagnostic (delete the stale runs/routing_profile.npz first if it predates the routing fix)
python support_diagnostic.py --cache-dir ~/moe_cache --bits 4 --budgets 64 128 256 512 --seeds 0 1 2 --outdir runs

# read-only report on finished E1/E2 (LOSO metrics overall and per n0, per-fold coefficients)
python analyze_characterization.py --bits 4 --outdir runs

# validation gates on the frozen-damping E1/E2 outputs (CPU, seconds to a minute)
python e1_gate.py --bits 4 --percdamp 1.0 --outdir runs
python e2_gate.py --bits 4 --percdamp 1.0 --outdir runs
python diagnostics/p2b_regime_diagnostic.py --permutations 1000
python diagnostics/p2c_layer_exposure.py --permutations 1000
```

`scripts/run_all.sh` and `scripts/run_characterization_b4.sh` still use the
legacy GPTQ default (0.01); do not use them for final results. Use
`scripts/p1p2_characterize.sh` and `scripts/p3p4_e3_min.sh`, which read the
frozen damping from `configs/frozen_b4.json`.

Diagnostics (write only to `runs_diag/`, tuning split only):

```bash
python diagnostics/low_data_diag.py {invariants,nested,calibtune,damp,report} [--layer L] [--layers ...] [--seed S] [--seeds ...] [--percdamp ...]
python diagnostics/routing_fix_check.py [--layers ...] [--n-seqs 32]
python diagnostics/damping_selection.py {run,report} [--layers ...] [--seeds ...]
```

Tests (tiny random Qwen2-MoE on CPU):

```bash
MOE_DEVICE=cpu python -m pytest -q tests
```

## Repository layout

```
moe_calib/
  model_utils.py  CPU-resident model, layer_on_gpu(), run_layer()
  data_utils.py   C4 pool/holdout, pool/tune/test split, cache I/O, WikiText-2 PPL
  profiler.py     single-layer replay with router/expert hooks
  hessian.py      K-block Hessian accumulator; fixed hash blocks; relSE, drift
  gptq.py         standard GPTQ (lazy-batch), percdamp default 0.01 (legacy)
  frozen.py       load / validate configs/frozen_*.json (production GPTQ + E1 design)
  quant_eval.py   quantize from Hessians, zero-token RTN fallback, per-sequence error
  utility.py      E2 predictors, support-aware delta_hat, E3 utility, exploration cover
  runid.py        E1 run fingerprints
  stats.py        paired t-test, BH-FDR, Spearman, incremental R^2
cache_inputs.py, e1_label_sufficiency.py, e2_signals.py, e3_allocation.py,
support_diagnostic.py, analyze_characterization.py
configs/          frozen_b4.json (frozen production configuration)
diagnostics/      low-data, routing and damping diagnostics (not part of the pipeline)
scripts/          setup_icrn.sh, run_smoke.sh, run_characterization_b4.sh, run_all.sh
tests/            CPU tests
```

## Known limitations

- Only four profiled layers (1, 8, 16, 23). A layer-aware allocator over all
  24 layers needs every layer characterized, or a parameter-sharing scheme.
- Layer-aware exposure value is identified across calibration regimes: per-expert
  count and budget move together, so their effects cannot be separated.
- Non-sequential GPTQ: every layer is quantized from full-precision inputs.
- Only routed experts are quantized; attention and the shared expert stay BF16.
- Characterization is currently 4-bit only.
- Predictor validity is limited to the characterized routed-count support.
- The allocator approximates the benefit between n and 2n from the n0 → 2n0 labels.
- No allocation experiment has been run. E3 still uses the original
  signal-based utility.
- Equal-compute `random_plus` evaluation is pending.
- Final cost and peak-memory analysis is pending.
- `--unbiased` (Horvitz–Thompson weighting) is experimental and not part of
  the main comparison.
- The mechanism of the low-data GPTQ error peak is not established.
