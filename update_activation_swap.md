# Activation-aware calibration swap pilot — November 7, 2026

Follows [`update_aggregate_constraint.md`](update_aggregate_constraint.md). All earlier outputs are unchanged.
Outputs: `runs_activation_swap/` (`summary.md`, `summary.json`, `proposals.json` + `.sha256`,
`all_proposals.json`, `experts.json`, `prepare.json`, `stats.npz`, `eval/`, `figures/`, `logs/`).

**Decision (rule frozen in `proposals.json` before any GPTQ run): ACTIVATION-SIGNAL-INCONCLUSIVE.**
Across the nine swaps the predicted gain and the actual gain have a Spearman correlation of only
+0.12. The predicted-beneficial group (+1.75e-6) did no better than the random group (+2.23e-6)
and was essentially tied with the predicted-harmful group (+1.65e-6). Every swap changed the
8-expert error by at most 0.085 %, which is within the spread of the random swaps (SD 2.0e-6).
This pilot found no evidence that the diagonal activation-mismatch signal predicts which single
swap helps GPTQ.

## 1. Question

A post-hoc diagnostic associated expert activation second-moment mismatch with reconstruction
error (Spearman 0.74 across 21 runs). This pilot asks the prospective question instead: does a
cheap expert-specific statistic predict the *marginal* reconstruction effect of replacing one
calibration sequence? It is a predictive-validation pilot, not an allocator.

## 2. Setup (frozen)

- **Model and GPTQ:** Qwen1.5-MoE-A2.7B; frozen 4-bit GPTQ (group 128, percdamp 1.0,
  act-order), checked against `configs/frozen_b4.json`.
- **Scope:** layer 16 only, allocation seed 0, B = 128.
- **Experts:** the 8 most-routed experts of layer 16 over the *original pool* routing cache,
  ties to the lower id: **[30, 53, 9, 42, 24, 50, 16, 0]**. They were saved in `experts.json`
  before any quantization; no reconstruction error was used to choose them.
- **Baseline set S0:** the matched-warm-start Full Random seed-0 selection, reused and not
  regenerated (hash `ec355fb8d866e4ff`).
  - It has 128 unique ids, all from the pool, with no tune/test ids.
  - Its warm-start prefix and hash are verified.
- **Reference statistics:** the tuning split (rows 2000–2127; it was used for earlier
  decisions, so this pilot is exploratory).
- **Evaluation:** the old test split (rows 2128–2255; previously inspected, so exploratory),
  with the existing frozen evaluator.

## 3. Activation statistics

- **Which tokens:** `Qwen2MoeSparseMoeBlock` passes expert e only `hidden_states[top_x]`. The
  existing `Profiler` pre-hook on `gate_proj` therefore sees exactly the routed gate/up inputs.
- **Accumulator:** `moe_calib/activation_swap.DiagProfiler` subclasses `Profiler` and
  accumulates two quantities per (sequence, expert), in FP32 then float64. No token storage.
  - `n[s,e]`, the routed-token count;
  - `g[s,e,j] = Σ x_j²`, the per-coordinate sum of squares.
- **Set statistics:** `h(S) = Σ_S g / Σ_S n`, weighted by token count.
- **Mismatch** (prespecified):
  - `D_e = ||h − h_ref||² / (||h_ref||² + 1e-12)`, and `D = mean_e D_e` with equal weights.
  - `predicted_gain = D(S0) − D(S_swap)`.
  - Swaps are scored by additive updates of the summed statistics.
- **Validation:**
  - Per-sequence counts equal the routing cache and the gate-hook counts exactly (asserted).
  - On the tiny CPU model, the diagonal sums equal the diagonal of the existing Hessian
    accumulator.

## 4. Candidates, proposals and the nine swaps

- **Candidate profiling:**
  - Profiled S0 (128) plus 64 incoming candidates drawn without replacement from pool \ S0
    (`random.Random(20261108)`): 192 sequences in total.
  - The 16 outgoing sequences came from S0 via `random.Random(20261109)`.
- **Proposal pool:** all 16 × 64 = 1,024 swaps, 522 with predicted gain > 0 and 502 with < 0.
- **Baseline mismatch:** D(S0) = 1.658e-3.
- **Selection:**
  - The top 3 and bottom 3 by predicted gain (deterministic tie-break).
  - 3 random proposals from the remainder (`random.Random(20261110)`).
  - Unique outgoing and incoming ids were achieved for all nine.
- **Freezing:** the manifest `proposals.json` holds the ids, predicted gains, per-expert
  predictions, set hashes, the reference fingerprint (`0d125bc069f7424e`) and the decision
  rule. Its sha256 was recorded before evaluation, and `--evaluate` refuses to run if it changes.
- **Set construction:** each swapped set replaces the outgoing sequence *in place*. That keeps
  exactly 128 unique ids, differing from S0 by exactly one out and one in.

## 5. Results (10 GPTQ evaluations: S0 + 9 swaps; old test split; 8-expert mean error)

E0 = 0.004688. Recomputing S0 reproduces the stored full-run per-expert errors exactly (difference 0.0).

| swap | out → in | predicted gain | actual gain (×1e-6) | rel. change | experts improved | Δ tokens | Δ mean d(s) | Δ agg. L1 (L16) |
|---|---|---|---|---|---|---|---|---|
| beneficial0 | 988 → 1654 | +2.39e-4 | +3.58 | −0.076 % | 5/8 | −132 | +0.0022 | −0.0008 |
| beneficial1 | 674 → 1714 | +1.53e-4 | −0.13 | +0.003 % | 4/8 | −48 | +0.0007 | −0.0002 |
| beneficial2 | 887 → 32 | +1.34e-4 | +1.79 | −0.038 % | 5/8 | −66 | +0.0011 | −0.0017 |
| harmful0 | 1008 → 1981 | −1.40e-4 | +1.83 | −0.039 % | 6/8 | −82 | −0.0008 | +0.0016 |
| harmful1 | 872 → 432 | −1.28e-4 | +1.12 | −0.024 % | 5/8 | +48 | +0.0002 | +0.0015 |
| harmful2 | 797 → 1932 | −1.06e-4 | +2.01 | −0.043 % | 5/8 | +144 | −0.0013 | +0.0012 |
| random0 | 1463 → 920 | −1.5e-6 | +0.02 | −0.001 % | 4/8 | +260 | −0.0039 | +0.0019 |
| random1 | 517 → 1676 | +8.94e-5 | +2.71 | −0.058 % | 6/8 | −24 | +0.0004 | −0.0013 |
| random2 | 1284 → 235 | +2.33e-5 | +3.96 | −0.085 % | 7/8 | +27 | +0.0008 | −0.0008 |

| | value |
|---|---|
| Spearman(predicted, actual), n = 9 | **+0.12** |
| mean actual gain, beneficial / harmful / random (×1e-6) | +1.75 / +1.65 / +2.23 |
| swaps improving, beneficial / harmful / random | 2/3, 3/3, 3/3 |
| random-group SD (noise reference) | 2.0e-6 |

- **Frozen rule:**
  - PROMISING fails: ρ < 0.5, and the beneficial group does not beat the random group.
  - NOT-SUPPORTED fails: ρ > 0, and the beneficial mean is marginally above the harmful mean.
  - The outcome is therefore INCONCLUSIVE.
- **Eight of nine swaps lowered the error, including all predicted-harmful ones.** S0 appears
  to be slightly unlucky for these experts. Most single swaps help by a few 1e-6, whatever the
  prediction says.

**Per expert:**
- Per-expert rank correlations between predicted and actual gain range from −0.63 (expert 16)
  to +0.55 (expert 42).
- No expert dominates: the largest share of the mean absolute change is 22 % (expert 42).
- Neither the routed-token change, nor the change in sequence atypicality, nor the change in
  aggregate routing L1 visibly orders the outcomes either.

## 6. Cost

| item | cost |
|---|---|
| Candidate profiling (192 sequences, layer 16 replay on cached inputs) | 8.5 s |
| Reference profiling (128 tuning sequences) | 5.3 s |
| Stored statistics | 25.3 MB (`stats.npz`; no token-level activations) |
| Proposal scoring (1,024 swaps) | CPU, seconds |
| Per evaluation | Hessian replay ≈ 7.1 s + GPTQ of 8 experts ≈ 8.7 s |
| Peak GPU memory | 7.8 GB |

The replay relies on the existing layer-input cache. Building that cache costs full-model
forwards; a real allocator would also have to profile its candidates through the model up to
each layer.

## 7. Tests

`tests/test_activation_swap.py` has 9 tests, covering:
- routed-token extraction against the existing Hessian accumulator and the gate-hook counts;
- expert-id mapping and tie-breaking;
- token-weighted normalization, not averaged variances;
- incremental vs full-set statistics (agreement to 1e-12);
- a synthetic pair of swaps with known opposite effects;
- exactly B = 128, with one sequence out and one in;
- deterministic proposal picking that prefers unique ids;
- frozen GPTQ settings and reference/test split isolation;
- reproducibility of the frozen proposals.

Full suite: **124 passed, 0 failed**.

## 8. Interpretation and limitations

- **Big predicted changes, tiny actual effects.** Single-sequence swaps move the diagonal
  mismatch by up to ±14 % of D(S0). They move 8-expert GPTQ error by at most 0.085 %, and the
  direction of that change is unrelated to the prediction in this sample.
- **Underpowered.** Nine swaps with effects of the same size as the swap-to-swap noise cannot
  rule out a weak signal. They do rule out a strong, easily exploitable one.
- **Diagonal only.** The metric uses diagonal second moments only. GPTQ uses the full Hessian,
  including off-diagonal structure, and act-order. The earlier association was with
  full-matrix mismatch across *whole strategies*, not single swaps.
- **Not independent evidence.** One layer, eight experts, one seed. The reference (tuning) and
  evaluation (old test) splits were both used before, so nothing here is independent
  confirmation, and no causal claim is made.

## 9. Recommended next step

**Do not build an activation-aware allocator; write the final project report.** It should
present this pilot alongside the four allocation experiments as negative evidence. Per the
stopping rules, the inconclusive result does not justify expanding the experiment.
