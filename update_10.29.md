# Project Update — October 29, 2026 (P0 → P2c)

Research log for the stages after the [10/18 update](update_10_18.md):
GPTQ stabilization (P0), target validation (P1), dynamic-signal validation
(P2), failure analysis (P2b), and validation of the revised layer-aware
exposure hypothesis (P2c).

Numbers come from `runs/` and `runs_diag/`. Those folders are gitignored, so
each section names its report file. Everything here used the candidate pool
and the tuning split. **The final test split has not been touched.**

---

## 1. Current status

| stage | question | result |
|---|---|---|
| P0 | Is low-data GPTQ well-behaved? Freeze damping | `percdamp = 1.0` frozen (interior optimum) |
| P1 | Is future calibration benefit a stable target? | **PASS** — positive at every budget, 0 % negative |
| P2 | Do dynamic state signals predict benefit beyond exposure? | **CASE B** — some association, not robust held-out |
| P2b | Is P2's instability budget-, layer-dependent, or both? | **P2b-UNSTABLE** — dominated by layer heterogeneity |
| P2c | Is the marginal value of exposure layer-/regime-dependent? | **LA-A** — supported, generalizes to held-out seeds and experts |
| next | Does the advantage survive sequence-routing coupling? | minimal fixed-budget allocation experiment — **not yet run** |

**Short version.** The original hypothesis — that *dynamic*
Hessian/router state is a robust expert-level sufficiency signal — was
**not supported** under strict held-out validation. Instead, the marginal
value of routed-token exposure depends strongly on **layer** and **calibration
regime**. That structure holds up on held-out characterization seeds and
held-out experts. It also ranks experts better than simple coverage, but
only before sequence-routing constraints, which have not been tested yet.

### Research question and the sufficiency definition

Original question: *can an MoE expert's calibration state predict the
marginal future GPTQ reconstruction benefit of additional calibration data
beyond routed-token count, and can that improve allocation under a fixed
sequence budget?*

Operational definition (E1), on held-out tuning activations:

```
future_rel_delta = (err(n0) - err(2·n0)) / err(n0)      (> 0: more data helps)
```

Statistical significance is **not** sufficiency. A t-test only says the
benefit is nonzero. Allocation needs the **size** of the benefit that
remains. P1 makes this concrete (§3).

---

## 2. P0 — GPTQ stabilization

**Why.** At the original `percdamp = 0.01`, low-data GPTQ was pathological.
On strictly nested prefixes (16 ⊂ 32 ⊂ 64 ⊂ 128 ⊂ 256 sequences), going from
16 to 32 sequences made held-out reconstruction *worse* for ~90 % of experts
(mean tuning error 0.00949 → 0.01038; RTN 0.00679). Any benefit label built
on that would have measured a GPTQ artifact, not calibration sufficiency.

**Ruled out as a bug.** The diagnostics (`diagnostics/low_data_diag.py`) found:
- Hessian accumulation is correct (partial-sum equivalence, scale invariance).
- E1 labelling is correct.
- Nested calibration sets reproduce the effect.
- Error on the *calibration* activations improves while *tuning* error gets
  worse: the classic overfitting signature.

So this is genuine low-data overfitting / under-regularization of GPTQ.

**Extended damping grid** (`runs_diag/damping_threshold_report.md`; seeds
0/1, layers 1/8/16/23, prefixes 16–256, tuning split; error averaged over
prefixes):

| percdamp | 0.01 | 0.1 | 0.2 | 0.3 | 0.5 | **1.0** | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|---|
| mean tuning error | 0.00775 | 0.00576 | 0.00533 | 0.00514 | 0.00497 | **0.00487** | 0.00487 | 0.00490 |
| % experts better than RTN | 44.5 | 76.3 | 93.5 | 98.8 | 100 | 100 | 100 | 100 |
| 16 → 32 improved | 10 % | 93 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |

**Frozen production GPTQ:** 4-bit, group size 128, act-order on,
`percdamp = 1.0` (`configs/frozen_b4.json`). Reasons:
- 1.0 is an **interior** optimum: 1.5 does not improve on it and 2.0 is worse.
- Every doubling transition improves for 100 % of experts.
- Every expert beats RTN.

The per-prefix optimum drifts (2.0 at 16–32 sequences, 0.5 at 256), but one
fixed value is used by design. It was chosen on the tuning split before
characterization and will not be retuned on E1/E2 results or the test split.

**Threshold diagnostic.**
- At 0.01, the GPTQ/RTN error ratio peaks near *routed tokens / input
  dimension ≈ 1*: gate/up in [0.5, 0.75) at 1.39, down in [0.75, 1.0) at 2.56.
- At 1.0 the peak disappears: the ratio falls monotonically and stays below 1
  for gate/up.
- The Hessian's effective rank stays far below the raw token count even at
  256 sequences. The median erank/d_in is 0.25 for gate/up and 0.58 for down,
  with ~8.7k routed tokens.

*Interpretation (cautious):* this fits a low-rank / under-regularized
Hessian explanation. It does **not** prove a formal double-descent mechanism.

---

## 3. P1 — target validation

**Why.** Before asking what *predicts* future benefit, check that the target
is real and stable under the frozen GPTQ.

**Setup.** E1 rerun at `percdamp = 1.0`:
- characterization seeds 0/1/2; layers 1/8/16/23; 60 experts per layer
- n0 = 16/32/64/128; 3 extension replicates
- 0 excluded rows (`runs/e1_gate_b4_d1.md`)

| transition | median `future_rel_delta` | mean | % negative | median abs. Δ |
|---|---|---|---|---|
| 16 → 32 | +0.0569 | +0.0586 | 0.0 % | 3.09e-04 |
| 32 → 64 | +0.0567 | +0.0569 | 0.0 % | 2.93e-04 |
| 64 → 128 | +0.0447 | +0.0446 | 0.0 % | 2.22e-04 |
| 128 → 256 | +0.0310 | +0.0310 | 0.0 % | 1.49e-04 |

**Gate: PASS.** Every required check passed:
- positive median at every budget
- 0 % negative experts
- every seed's median positive
- every layer's median positive

Absolute improvement shows diminishing returns.

**What it means.**
- Future held-out GPTQ improvement is a stable, meaningful target.
- It shows directly that **significance ≠ sufficiency**: 100 % of
  expert/budget transitions are significant (paired t-test, BH-FDR
  q = 0.1). The *size* of the remaining benefit still varies systematically,
  by budget (median 0.057 → 0.031) and by layer (at n0 = 16, L1 0.072 vs.
  L23 0.050).
- A significance-based sufficiency rule would therefore say "still
  insufficient" for every expert and could not rank them.

---

## 4. P2 — dynamic state signals beyond exposure

**Why.** This is the core hypothesis test. Do the candidate state signals
predict `future_rel_delta` beyond routed-token exposure? The signals are
normalized Hessian drift, relSE, router margin and router entropy.

**A stricter baseline.** The old E2 compared signals with *linear* log count,
and a signal can "win" just by absorbing count non-linearity. P2
(`e2_gate.py`) compares against a **flexible exposure baseline**: log n,
(log n)², (log n)³, budget dummies and budget × log n. On its own, this
baseline beats linear count by **+12.65 %** with seeds held out (LOSO) and
**+10.96 %** with layers held out (LOLO). The old linear-count comparison is
therefore not a valid scientific conclusion.

**Result: CASE B** (`runs/e2_gate_b4_d1.md`). Gain = reduction in held-out
MSE versus flexible exposure.

| signal | model | LOSO pooled | LOSO folds (seed 0/1/2) | LOLO pooled | LOLO folds (L1/8/16/23) |
|---|---|---|---|---|---|
| relSE | + s | +2.8 % | −5.9 / +8.0 / +5.5 | +0.3 % | +4.1 / +8.2 / −7.7 / −9.9 |
| relSE | + s × budget | +9.2 % | −6.1 / +19.7 / +11.8 | +8.1 % | +14.4 / +46.9 / −13.1 / −17.4 |
| drift | + s × budget | +8.6 % | −5.4 / +16.9 / +12.9 | +6.6 % | +12.6 / +47.3 / −15.1 / −19.1 |
| entropy | + s × budget | +12.9 % | +12.7 / +11.8 / +14.4 | −42.8 % | −16.9 / −7.3 / −635 / +12.8 |

- relSE and drift have positive pooled gains, and permutation tests reject
  "no extra information" (p = 0.005).
- But seed 0 and held-out layers 16/23 get worse, sometimes substantially.
- The per-budget slopes of relSE/drift **change sign**: negative at
  n0 ≤ 32 (t ≈ −14), positive at n0 = 128 (t ≈ +14).
- Margin/entropy collapse when a layer is held out.

**Conclusion.** State signals carry *some* association with future benefit,
but the held-out evidence is not stable enough to support a global
sufficiency-aware predictor. P2 is **not** a successful sufficiency-signal
result. E3 was blocked.

---

## 5. P2b — why P2 was unstable

**Why.** Before giving up or adding complexity, find out *where* the
instability comes from: budget dependence, layer dependence, or both
(`diagnostics/p2b_regime_diagnostic.py`, `runs/p2b_regime_b4_d1.md`).

**Result: P2b-UNSTABLE.**

1. **Conditioning on budget does not fix it.** No signal is robust in any
   fixed budget (0/4 for every signal). At fixed n0, drift/relSE still lose
   on held-out layers 16/23 (−10 % to −27 %) and on seed 0 at n0 ≤ 32.
2. **Layer heterogeneity dominates.**
   - A per-layer exposure model alone improves LOSO by **+30.2 %** over P2's
     global flexible exposure baseline.
   - A per-budget exposure model improves it by only **+0.1 %**.
3. **Drift/relSE lose their value once layer-specific exposure is modelled:**
   −1.5 % (drift) and −1.4 % (relSE) versus the matched per-layer exposure
   model, with seed 0 negative.
4. **Margin/entropy are static expert descriptors, not calibration state.**
   ~98.8 % (margin) and ~98.9 % (entropy) of their variance is explained by
   expert identity: they barely change with seed or budget. Holding out a
   seed keeps the same experts in training and test, so it cannot separate
   "state predicts benefit" from "expert identity predicts benefit". Their
   within-layer slopes also change sign across layers.

**Conclusion.** The original dynamic sufficiency hypothesis is not robustly
supported. Most of the apparent predictive structure in P2 is layer
heterogeneity that the global exposure baseline did not model, not dynamic
calibration state. **This negative result stands.**

---

## 6. P2c — validating the layer-aware exposure hypothesis

**Revised hypothesis (motivated by P2b, not the original idea).** *The
marginal value of additional calibration exposure depends strongly on layer
and calibration regime.* The test is exposure-only: routed count, layer,
budget and `future_rel_delta`. Drift, relSE, margin and entropy are not used.

Script: `diagnostics/p2c_layer_exposure.py`; report:
`runs/p2c_layer_exposure.md`.

**Nested ladder** (lc = standardized log routed count):

| model | terms |
|---|---|
| M0 | 1 + lc (reference) |
| M1 | global flexible exposure (= P2 baseline) |
| M2 | M1 + layer intercept |
| M3 | M2 + layer × lc |
| M4 | M2 + layer × (lc, lc²) — selected; layer × lc³ adds nothing |
| Dlb | M2 + layer × budget means (diagnostic, no layer × count) |

**Result: LA-A — all six gate criteria pass.**

### Prediction (LOSO, primary)

| model | MSE | gain vs M1 [95 % CI] | gain vs M2 [95 % CI] | gain vs M2 per seed (0/1/2) |
|---|---|---|---|---|
| M1 | 7.001e-05 | 0 | – | – |
| M2 | 6.295e-05 | +10.1 % [+4.4, +15.3] | 0 | – |
| M3 | 4.926e-05 | +29.6 % | +21.8 % [+14.2, +29.3] | +25.0 / +19.6 / +21.2 % |
| M4 | 4.877e-05 | +30.3 % [+18.4, +40.5] | +22.5 % [+14.8, +30.2] | +26.4 / +20.1 / +21.7 % |
| Dlb | 4.917e-05 | +29.8 % | +21.9 % | +25.3 / +19.4 / +21.7 % |

Confidence intervals come from a cluster bootstrap over (layer, expert).

- About **33 %** of the M1 → M4 improvement comes from layer intercepts.
  About **67 %** comes from layer/regime structure beyond intercepts.
- M4 beats M2 in every held-out seed. The interaction is unlikely to be
  chance: residual-permutation and layer-label-shuffle tests both give
  p = 0.001 (1,000 permutations).

**Caveat — what is and is not identified.** Routed count is almost
proportional to budget (within a budget, counts vary only ±10–20 %). So the
"curve" is learned from how each layer's benefit changes *between*
calibration regimes. Dlb (layer × budget means, no continuous layer × count
term) performs almost as well as M4. Adding a within-budget layer-specific
count slope to Dlb gains only +0.1 %. The safe statement is:

> Calibration marginal value is strongly **layer- and regime-dependent**.

We have **not** uniquely identified a precise continuous layer-specific
count curve.

### Generalization beyond seeds

| validation | M2 vs M1 | M4 vs M1 | M4 vs M2 | folds |
|---|---|---|---|---|
| expert-grouped CV (5 folds on (layer, expert); all seeds of an expert held out) | +10.5 % | +32.5 % | +24.6 % | all folds positive; all layers positive |
| joint (held-out seed **and** held-out experts) | +9.6 % | +29.5 % | +22.0 % | all seeds positive |

So the model is not memorizing individual experts or one characterization
draw: the layer/regime structure transfers to unseen experts in the same
layers. (M2 alone gets slightly worse on L8 in expert CV; M4 is positive
everywhere.)

### Which layer is most valuable changes with regime (M4 fitted values)

| regime | n | highest benefit | lowest |
|---|---|---|---|
| n0 = 16 | ≈ 500 | L1 (0.074) | L23 (0.052) |
| n0 = 32 | ≈ 1000 | L1 (0.065) | L23 (0.053) |
| n0 = 64 | ≈ 2000 | L8 / L16 (0.048) | L23 (0.043) |
| n0 = 128 | ≈ 4000 | L8 (0.035) | L1 (0.029) |

L1 starts highest and drops fastest. It goes from first to last. The layer
order is identical in all three seeds at n0 = 16, 32 and 128; at n0 = 64,
where values are nearly tied, L8 and L16 swap. **A single fixed layer weight
is not enough.** M2, which has only layer intercepts, ranks experts no better
than coverage (below). The method is therefore *layer- and regime-aware
exposure allocation*, not plain layer-weighted coverage.

### Actionable expert ranking (fixed budget, held-out seed)

Models are fit on the other seeds. Each utility rule ranks experts, and the
score is how much realized benefit per token (`future_rel_delta / n`) the
top-q % captures, as a fraction of the oracle's top-q % (q = 10/25/50 %,
averaged).

| rule | utility | mean capture | Spearman(predicted, realized benefit) |
|---|---|---|---|
| coverage | 1 / n | 0.934 | – |
| global exposure | f_M1(n) / n | 0.934 | 0.29 |
| layer intercept | f_M2(n) / n | 0.936 | 0.40 |
| layer × budget | f_Dlb(n) / n | **0.963** | 0.64 |
| layer-aware | f_M4(n) / n | 0.962 | **0.65** |

- At a fixed budget, global exposure ranks experts exactly like coverage.
  Any advantage must come from layer/regime terms.
- Layer-aware beats coverage in **every** one of the 12 (seed, budget) cells.
  The largest gains are at low budget (+6.7 to +7.6 pp at n0 = 16); the
  gain is +0.6 to +2.2 pp at n0 ≥ 32.
- So the layer-aware / layer-budget models do more than improve regression
  fit: they give better expert priorities than simple under-coverage.

**Caveat.** This ranking assumes exposure can be given to experts directly.
It ignores that sequences route tokens to many experts at once. It is an
intermediate test, **not** a substitute for the allocation experiment.

---

## 7. What we learned

1. **Damping matters more than expected at low data.** Below ~1 routed token
   per input dimension, `percdamp = 0.01` GPTQ overfits, and benefit labels
   computed at that setting were partly artifacts.
2. **The target is sound.** Future benefit is positive, stable across seeds
   and layers, and decreasing in budget. Significance tests carry no ranking
   information here.
3. **Dynamic state ≠ sufficiency (so far).** Drift/relSE look predictive
   against weak baselines. Under a flexible, layer-aware exposure baseline
   with held-out seeds and layers, they add nothing robust.
4. **Router margin/entropy are expert descriptors.** Seed-held-out validation
   cannot tell them apart from expert identity; any future use needs
   expert-grouped validation.
5. **Baselines decide the conclusion.** The same signals went from "+4 % over
   count" (linear baseline) to "no robust value" (flexible + layer-aware
   baseline).
6. **The real structure is layer × regime.** Exposure value differs strongly
   across layers, and the order changes with calibration regime. This
   generalizes to held-out seeds and experts.

## 8. Revised research direction

| | original | revised (evidence-supported) |
|---|---|---|
| hypothesis | dynamic Hessian/router state predicts sufficiency beyond count | marginal value of routed-token exposure is layer- and regime-dependent |
| predictor input | count + one state signal | count + layer + calibration regime |
| status | not robust under strict validation (P2 CASE B, P2b UNSTABLE) | supported for prediction and expert ranking (P2c LA-A) |
| allocation utility | max(δ̂(n, signal), 0) / n | max(f_L(n, regime), 0) / n |

This progression from the original idea to a negative result to an
evidence-supported revision is itself a result. The negative P2/P2b findings
stay in the record.

## 9. Next experiment — minimal layer-aware allocation

**Question.** *Does the expert-level advantage of layer-aware utility survive
sequence-routing coupling?*

- Fixed budget **B = 128**, seeds 0/1/2.
- Strategies: Random, Coverage, Global exposure, **Layer × budget**,
  **Layer-aware**, Oracle.
- Keep the same standard frozen GPTQ, the same sequence budget, the same
  natural routing, and the same LP / sequence-selection machinery where
  possible. Only the utility changes.
- Evaluation: final expert reconstruction on the untouched test split.
- Layer × budget is a required comparator. It matched the layer-aware model
  in P2c, so the layer-aware method must beat it, not just coverage, before
  any claim about continuous curves.
- Expected size: at the matching regime (n0 = 128) the expert-ranking gain
  over coverage was +0.8 to +2.0 pp, much smaller than at n0 = 16. B = 128
  may therefore show only a small effect.

**Only if it succeeds:** budgets 64/256/512, all-layer PPL, equal-compute
baseline, overhead/peak memory, 3-bit robustness.

## 10. Open limitations

- **Four characterized layers** (1/8/16/23 of 24). A production allocator
  needs every layer characterized, or a parameter-sharing scheme (for
  example, interpolating by depth). This is untested.
- **Regime vs. exposure are confounded.** Per-expert count and budget move
  together, so the data cannot separate "more tokens for this expert" from
  "more calibration sequences overall".
- The allocator uses n0 → 2n0 labels to approximate the benefit of growing
  exposure from n to 2n. Predictions are limited to the observed count support.
- The ranking test ignores sequence-routing coupling. The allocation
  experiment is the real test.
- Non-sequential GPTQ: each layer is quantized from full-precision inputs.
  Only routed experts are quantized. 4-bit only so far.
- The E3 controller still implements the old signal-based utility. It has
  not yet been adapted to layer/regime utilities.
- The mechanism of the low-data GPTQ peak is not established; P0 shows
  consistency with under-regularization, not a proof.
