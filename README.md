# Calibration-Sufficiency-Aware Budget Allocation for MoE PTQ

CS 498 research project on calibration-data allocation for post-training
quantization (PTQ) of Mixture-of-Experts models.

**Research question.** Can an MoE expert's current calibration state predict
the marginal GPTQ reconstruction benefit of additional calibration data
*beyond routed-token count*, and can that information improve calibration
allocation under a fixed sequence budget?

Setup: **Qwen/Qwen1.5-MoE-A2.7B** (60 routed experts/layer, top-4),
**4-bit weight-only GPTQ** (group size 128), **C4** calibration data,
**expert-level reconstruction** evaluation (plus WikiText-2 PPL). The
hypothesis has **not** been confirmed yet; see [Current status](#current-experimental-status).

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

Statistical stability of the Hessian alone is **not** treated as sufficiency;
a signal is useful only if it predicts future reconstruction benefit beyond
count on held-out characterization seeds.

The GPTQ algorithm is identical across all allocation strategies. A single
GPTQ configuration is to be chosen on tuning data and frozen *before*
characterization and final evaluation (see [Damping study](#damping-study)).

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

> The existing E2 artifact was produced under the original GPTQ damping
> (`percdamp = 0.01`) and is **diagnostic only**. It must not be used for
> final E3 until E1/E2 are rerun under the final frozen GPTQ configuration.

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

E3 has **not** produced final research results yet.

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

- 0.10 is the best tested value at every prefix, layer and seed.
- With 0.10 the 16 → 32 step becomes positive on average; later doubling
  steps stay positive.
- 0.10 is the current leading candidate, but it is the edge of the tested
  grid. The production default in `moe_calib/gptq.py` is **still 0.01**
  until the team decides whether to test larger fixed values (e.g. 0.2, 0.3).

## Current experimental status

| component | status |
|---|---|
| Real Qwen/CUDA smoke | Passed |
| Pool / tune / test split | Implemented |
| E1 future-benefit characterization | Implemented |
| 3-seed characterization | Completed at old 0.01 damping |
| E2 multi-seed signal validation | Completed at old 0.01 damping |
| Routing profile correctness | Fixed / validated |
| Low-data GPTQ diagnostic | Completed |
| Fixed damping study | Completed |
| Final production GPTQ damping | Pending freeze |
| E1/E2 rerun under final damping | Pending |
| Final E3 Random / Coverage / Sufficiency study | Not yet run |
| 3-bit robustness | Not yet run |
| PPL / overhead / peak-memory evaluation | Not yet run |

The E2 result under `percdamp = 0.01` (relSE selected, ~1.75% LOSO MSE
improvement over count) is diagnostic, **not** the final sufficiency-signal
conclusion.

## Running the code

Environment (ICRN): a venv with a CUDA build of torch that matches the
driver, plus `requirements.txt`; `scripts/setup_icrn.sh` is the original
one-time setup. Use the A100 option (the model needs ~29 GB of CPU RAM).

```bash
# end-to-end smoke on a tiny setting (1 layer, 64-sequence pool; PRELIMINARY E2)
bash scripts/run_smoke.sh 2>&1 | tee logs/smoke.log

# real 4-bit characterization: cache (if missing) -> E1 seeds 0,1,2 (n0 16 32 64 128)
#   -> E2 (--min-char-seeds 3) -> support diagnostic. Resumable: rerun the same command.
mkdir -p logs && nohup bash scripts/run_characterization_b4.sh > logs/characterization_b4.log 2>&1 &

# support diagnostic (delete the stale runs/routing_profile.npz first if it predates the routing fix)
python support_diagnostic.py --cache-dir ~/moe_cache --bits 4 --budgets 64 128 256 512 --seeds 0 1 2 --outdir runs

# read-only report on finished E1/E2 (LOSO metrics overall and per n0, per-fold coefficients)
python analyze_characterization.py --bits 4 --outdir runs
```

`scripts/run_all.sh` still runs 3-bit characterization and E3 under the
current GPTQ default; do not use it for final results before the GPTQ
configuration is frozen and E1/E2 are rerun.

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
  gptq.py         standard GPTQ (lazy-batch), percdamp default 0.01
  quant_eval.py   quantize from Hessians, zero-token RTN fallback, per-sequence error
  utility.py      E2 predictors, support-aware delta_hat, E3 utility, exploration cover
  runid.py        E1 run fingerprints
  stats.py        paired t-test, BH-FDR, Spearman, incremental R^2
cache_inputs.py, e1_label_sufficiency.py, e2_signals.py, e3_allocation.py,
support_diagnostic.py, analyze_characterization.py
diagnostics/      low-data, routing and damping diagnostics (not part of the pipeline)
scripts/          setup_icrn.sh, run_smoke.sh, run_characterization_b4.sh, run_all.sh
tests/            CPU tests
```

## Known limitations

- Only four profiled layers (1, 8, 16, 23).
- Non-sequential GPTQ: every layer is quantized from full-precision inputs.
- Only routed experts are quantized; attention and the shared expert stay BF16.
- Characterization is currently 4-bit only.
- Predictor validity is limited to the characterized routed-count support.
- The allocator approximates the benefit between n and 2n from the n0 → 2n0 labels.
- The final Random / Coverage / Sufficiency comparison has not been run.
- Equal-compute `random_plus` evaluation is pending.
- Final cost and peak-memory analysis is pending.
- `--unbiased` (Horvitz–Thompson weighting) is experimental and not part of
  the main comparison.
- The mechanism of the low-data GPTQ error peak is not established.
