# FP64 canonical GPTQ and numerical-reproducibility synthesis

**Canonicalization verdict: `CANONICAL-VALIDATED`.** Accumulating the original FP32 per-sequence Hessian contributions
in FP64 and casting once to FP32 does three things:
- it gives bitwise-identical Hessians for every tested accumulation order;
- the unchanged GPTQ solver quantizes them deterministically;
- the canonical FR/FC pair evaluates to Δ_FP64 = PPL(FC) − PPL(FR) = **+0.007129**.

That value lies inside the FP32 order-null range (+0.004614 to +0.012326) and keeps the FR-better ranking of the
original pipeline.

**Execution-variability verdict: `EXECUTION-VARIABILITY-MATERIAL`.** This is decided from the existing studies only;
the FP64 result was not used.
- Matched calibration-execution variants (batch groupings) change the sign of FC − FR within every allocation seed.
- Repeated evaluation of identical weights is exactly deterministic.

Both verdict categories and all rules were frozen in `runs_fp64_canonical_reproducibility/protocol.json` before any
canonical Hessian or PPL was computed.

**Scope.**
- **Model:** Qwen1.5-MoE-A2.7B. Only the routed experts of layers 1/8/16/23 are GPTQ-quantized (4-bit, g128, percdamp
  1.0, act-order); everything else stays BF16. These are fake-quantized BF16 weights, not packed INT4.
- **Calibration:** FR s0 / FC s0, B = 128, using the ORIGINAL activation cache, ORIGINAL routing R_O and ORIGINAL
  act-order P_O.
- **Evaluation:** frozen WikiText-2 windows, 128 × 512, 65,408 scored targets, batch 8.

## 1. Audit and reuse decisions (`artifact_inventory.json`, `protocol.json`)

**Inputs verified against their recorded hashes:**
- FR/FC selection hashes and the selection sha256 used by the order-null study;
- the ORIGINAL cache files;
- the ORIGINAL routing file;
- the 720-permutation act-order file;
- the ORIGINAL FP32 (OO) weight files;
- all 6 + 5 order-null PPL results;
- the historical FP64 control JSONs.

**Leakage:** none. The maximum calibration row is 1978 (FR) / 1989 (FC), below the test rows starting at 2128.

**The FP64 control is reusable as evidence, but its matrices are not.** It stored comparison results only, not FP64
Hessians. The canonical Hessians were therefore **reconstructed** under these rules:
- the same `Fp64Accumulator` class;
- the same contributions;
- the original positional blocks;
- the same three orders (O0, 20261121, 20261122).

**Acceptance gates for the reconstruction:**
- the FP32 path must reproduce the stored preflight Hessians bitwise for all 480 keys and all 3 orders;
- every FP32-vs-FP64 relative-Frobenius value must reproduce the historical control's value *exactly*.

Both gates passed for FR and FC. No activation cache, selection or calibration data was regenerated.

## 2. Canonical Hessians and order invariance (`fp64_order_invariance.json`, `canonical_hessian_manifest.json`)

**Construction:**
1. Per-sequence contribution `Xf.T @ Xf`: an FP32 GEMM, unchanged.
2. Add it into the original positional block in FP64.
3. Sum the 4 blocks in FP64.
4. Cast once to FP32.
5. Serve the result to the unchanged E3/GPTQ path through `LoadedHessianAccumulator`.

**Pathway regression:** the same serving pathway fed with the ORIGINAL FP32 Hessians reproduces the frozen OO weights
and reconstruction bitwise.

| | FR | FC |
|---|---|---|
| FP64 sum bitwise equal across O0 / 20261121 / 20261122 | 480/480 | 480/480 |
| FP32 cast bitwise equal across the 3 orders | 480/480 | 480/480 |
| FP32 path == stored preflight Hessians (all 3 orders) | yes | yes |
| reproduces historical control FP64 values exactly | yes | yes |
| block token counts == original | yes | yes |
| finite, exactly symmetric | yes | yes |
| elements differing from ORIGINAL FP32 Hessian | 68.4 % | 70.5 % |
| rel. Frobenius vs ORIGINAL FP32, median / max | 7.1e-8 / 1.3e-7 | 7.5e-8 / 1.5e-7 |
| max \|trace rel. diff\| | 8.3e-8 | 9.9e-8 |
| damped-H perturbation / H⁻¹ factor change (16 diag. Hessians, median) | 6.1e-8 / 3.0e-7 | 6.8e-8 / 3.4e-7 |
| natural act-order == P_O (canonical H) | 384/480 | 390/480 |
| Hessians whose natural order would change a quantization group | 1 (max 0.1 % of columns) | 0 |

- **Tested orders only.** Order invariance is verified for three orders. It is not claimed universally.
- **Exactness of the sums.** The FP64 sums of these FP32 addends were exact at these magnitudes. That is a property of
  the data, not a general guarantee.
- **Act-order.** The ORIGINAL P_O was applied as an explicit override in 240/240 experts.
  - Without the override, 90–96 Hessians would swap near-tied columns, almost always within a group.
  - The canonical definition of act-order was not changed.

## 3. Canonical quantization and PPL (`canonical_quantization_results.json`, `canonical_ppl_results.json`)

**Every configuration (FR, FC, FR repeat) passed all of the following:**
- 240 experts, 720/720 projections changed from BF16, 0 RTN fallbacks, 0 non-finite values;
- routing equal to R_O for every token, and token counts equal to R_O;
- permutations equal to P_O, 240/240;
- 480/480 served Hessians equal to the canonical hashes;
- integrity checks I1–I4 and the frozen old tolerance check;
- routers and all untouched parameters unchanged;
- the repeated PPL evaluation identical.

**Determinism:** the FR repeat, run in a fresh process, equals the first FR run bitwise. This covers installed weights,
weight files, reconstruction (mean and per expert), PPL and per-window NLL.

| | FR | FC |
|---|---|---|
| PPL | 9.271421461854468 | 9.278550816989844 |
| mean NLL | — | Δ mean NLL = +0.000769 |
| Δ_FP64 = FC − FR | | **+0.007129355** |

The paired window bootstrap of the mean-NLL difference, *conditional on these two weight sets*, is +0.00077, 95 % CI
[−0.00013, +0.00168].

## 4. Comparison with the FP32 order-null distribution (`order_null_comparison.json`)

| Policy | FR PPL | FC PPL | FC − FR |
|---|---|---|---|
| Original FP32 | 9.267851 | 9.277908 | +0.010057 |
| FP32 order 20261121 | 9.267791 | 9.275571 | +0.007779 |
| FP32 order 20261122 | 9.267245 | 9.279571 | +0.012326 |
| FP32 order 20261123 | 9.270495 | 9.277078 | +0.006583 |
| FP32 order 20261124 | 9.271444 | 9.276058 | +0.004614 |
| FP32 order 20261125 | 9.269601 | 9.278450 | +0.008849 |
| **FP64 canonical** | **9.271421** | **9.278551** | **+0.007129** |

Exact stored values are in the JSON.

1. **Inside the order range?** Yes. Δ_FP64 is inside [+0.004614, +0.012326], FR inside [9.267245, 9.271444] (at the
   top) and FC inside [9.275571, 9.279571].
2. **Ranking preserved?** Yes; FR is better.
3. **Distance from ORIGINAL FP32:** FR +0.003570, FC +0.000643, Δ −0.002928.
4. **Distance from the order mean:** FR +0.00211 (1.19 order-SD), FC +0.00121 (0.72 SD), Δ −0.00090 (−0.31 SD).
5. **Variability reduced?** Only the accumulation-order variability, which canonicalization removes by construction.
   It does nothing to, and cannot measure, variability from any other execution choice.
6. **Reconstruction:** not materially different. The relative change vs ORIGINAL FP32 is −7.7e-7 (FR) and −1.4e-7
   (FC). The FC penalty is +2.5491 % in both.

The canonical point is **one deterministic reference configuration**. Its distance from the order mean is not an
estimation error. It is not an estimate of the grouping-marginal effect either.

## 5. Weight-level sensitivity (`weight_sensitivity.json`)

Canonical vs ORIGINAL FP32, which is the same perturbation size as one shuffled FP32 order:

| | FR | FC | FP32 order-null (5 orders) |
|---|---|---|---|
| BF16 elements differing | 145,392 (0.0070 %) | 140,009 (0.0067 %) | FR 0.0067–0.0072 %, FC 0.0072–0.0076 % |
| rel. L2 | 8.16e-4 | 8.08e-4 | 8.0e-4 to 8.6e-4 |
| identical projections / 720 | 267 | 240 | 216–267 |
| experts with any change / 240 | 234 | 226 | 229–235 |
| integer codes differing (24 check projections) | 593 (8.6e-6) | 237 (3.4e-6) | 274–914 |
| FP32 group scales with any bit change | 23.3 % (median rel. change 0, max 1.1 %) | 23.9 % (max 1.2 %) | about 24 % |
| zero-points differing | 1.8e-6 | 0 | ≤ 7e-6 |
| changed BF16 elements with the *same* integer code | 90.8 % | 91.3 % | 88–92 % |
| per-window NLL differing | 128/128 (mean \|Δ\| 1.13 nats) | 128/128 (1.12 nats) | 128/128 (about 1.0) |

**Three change types, kept separate:**
1. **Integer-code flips:** a few hundred per configuration in the check experts.
2. **FP32 scale-bit changes:** about a quarter of group scales, almost always in low-order bits.
3. **BF16 effective-weight changes:** about 0.007 %, of which about 91 % are scale-bit changes exposed by the BF16 cast,
   with no change in rounding decision.

## 6. Mechanism comparison (`mechanism_comparison.json`; existing studies, not pooled, not additive)

| Intervention | Activations varied? | Routing varied? | Act-order varied? | Hessian change vs OO | Elements ≠ OO (FR / FC) | rel. L2 (FR) | FR PPL SD | FC PPL SD | Δ SD | sign changes vs Δ_OO |
|---|---|---|---|---|---|---|---|---|---|---|
| A: FP32 accumulation order | no | no | no | rel. Frob. ≈ 9e-8 | 0.0067–0.0072 % / 0.0072–0.0076 % | 8.0–8.2e-4 | 0.00178 | 0.00166 | 0.00287 | 0/5 |
| B: AOF (frozen routing + act-order) | yes | no | no | trace-norm. 0.0038–0.0050 | 16.9 % / 17.9 % | 0.0405 | 0.00240 | 0.00240 | 0.00166 | 0/5 |
| C: AO (frozen routing) | yes | no | yes | trace-norm. 0.0038–0.0050 | 39.2 % / 40.2 % | 0.0694 | 0.00402 | 0.00175 | 0.00561 | 1/5 |
| D: natural grouping, seed 0 | yes | yes | yes | trace-norm. 0.014–0.016 | 54.1 % / 55.2 % | 0.0889 | 0.00442 | 0.00429 | 0.00636 | 3/5 |
| D3: natural grouping, 3 seeds × 5 | yes | yes | yes | — | — | — | — | — | pooled within-seed 0.00687 | 9/15 vs seed's ORIGINAL sign |

**Design of each row:**
- A–D each use 1 allocation seed and 5 deterministic variants, with FR and FC paired (same order or grouping).
- D3 is the only multi-seed analysis: 3 seeds with groupings nested within seeds.

**How to read the SDs:**
- **They are not additive.** A, B, C and D are different intervention families: some are nested, none is factorial. The
  SDs are not variance components and are not subtracted.
- **Hessian vs PPL scale.** A's Hessian perturbation is about 4 × 10⁴ times smaller than B/C's. Yet its per-strategy PPL
  SD is 0.69–0.74 × AOF's.
- **Act-order and group membership** were fixed *jointly* in B. Their separate effects are not isolated.

## 7. Cross-seed grouping reanalysis (`cross_seed_grouping_analysis.json`)

All values were recomputed from the 15 stored paired differences with the study's own nested functions, and match the
historical values exactly (mean, pooled SD and interval).

| seed | mean Δ | grouping SD | FC worse / better | ORIGINAL-extraction Δ (separate policy) |
|---|---|---|---|---|
| 0 | −0.001932 | 0.006364 | 2/3 | +0.010057 |
| 1 | +0.000687 | 0.005176 | 2/3 | +0.002706 |
| 2 | −0.002498 | 0.008624 | 2/3 | +0.013070 |

**PPL-difference estimand:**
- Overall paired mean −0.001248; median −0.003677.
- Pooled within-seed SD 0.006872; SD of the seed means 0.001699.
- 40 % of the 15 values have the opposite sign to the mean.
- Hierarchical interval (seeds, then groupings) [−0.004777, +0.002155].

**Paired mean-NLL estimand:** mean −0.000134, pooled SD 0.000741, interval [−0.000515, +0.000232]. The direction and
conclusion are the same; because PPL is nonlinear, the two estimands are reported separately.

**Caveats:**
- With 3 seeds there are only 10 distinct outer resamples, so the interval is descriptive and unstable.
- ORIGINAL is never pooled with the five regroupings.

## 8. Evaluation-window uncertainty vs execution variability (`evaluation_uncertainty_comparison.json`)

**The window bootstrap.** For one fixed weight pair (S, g), it resamples evaluation windows. It answers: *for these two
quantized models, how sure are we about the PPL gap on this text distribution?*

**What happens across groupings:**
- 4 of the 15 matched configurations have window intervals that exclude zero, **in opposite directions**:
  - FC favoured: s0 G20261114 [−0.00174, −0.00004]; s2 G20261113 [−0.00254, −0.00040].
  - FR favoured: s1 G20261114 [+0.00003, +0.00183]; s2 G20261111 [+0.00013, +0.00203].
- The ORIGINAL s0 pair also "favours FR" ([+0.00020, +0.00200]).

So each fixed-model comparison can look conclusive, while another equally valid calibration execution of the same
selections reverses it.

**The execution-level interval** resamples seeds and groupings: [−0.00478, +0.00216] in PPL. It is the relevant
quantity for a claim about a calibration *method*.

**Reading.** Neither interval is wrong. They condition on different things, and a window interval does not sample
quantization-execution variability.

## 9. Reconstruction vs PPL (`reconstruction_ppl_alignment.json`)

- **Expert reconstruction:** FC is worse in 15/15 grouping configurations, by +2.19 to +2.71 %.
  - Within a seed the penalty varies by ≤ 0.014 percentage points.
  - Canonical and ORIGINAL FP32 both give +2.549 %.
- **FC − FR PPL** over the same configurations: −0.0135 to +0.0099; FC worse in 6/15.
- **Actual MoE layer output:** never evaluated for the grouping weight sets, so it is reported as unavailable rather than
  imputed.
  - Under a *different* extraction protocol (CANONICAL-B1, `runs_legacy_metric_reeval`), FC's layer-output penalty is
    +2.20 / +1.73 / +1.99 % (seeds 0/1/2).
  - No matched PPL exists for those weights, so they are shown separately.

**Why the local ranking is stable while the PPL ranking is not.** These are hypotheses, not measured causes:
- Reconstruction averages squared error over many tokens per expert. PPL is sensitive to a small number of high-loss
  predictions.
- Perturbations propagate nonlinearly through 20 BF16 layers.
- BF16 effective weights and scale bits are sensitive.
- Local and model-level objectives differ.

The data establish only that the 2.5 % local penalty is far more stable than the about 0.1 % PPL effect. They do not
identify the experts or tokens responsible.

## 10. Estimands (`statistical_estimands.json`)

| Estimand | Definition | Value |
|---|---|---|
| Canonical | PPL(FC, S, G₀) − PPL(FR, S, G₀); G₀ = ORIGINAL extraction, FP64 block accumulation, P_O, this solver and stack | +0.007129 (one deterministic value, no interval) |
| Order-null | the same pathway over 5 FP32 accumulation orders | mean +0.008030, SD 0.002870 |
| Execution-marginal | E over (s ∈ 3 seeds, g ∈ 5 groupings) of the paired difference | −0.001248, interval [−0.004777, +0.002155]; mean-NLL version −0.000134 |
| Window-conditional | paired window bootstrap for fixed weights | per configuration; does not sample execution |

**The canonical and execution-marginal estimands answer different questions:**
- The canonical value is a reproducible reference for *one* execution policy.
- The execution-marginal value describes robustness over a *defined* set of policies.

**Why the canonical value is positive while the marginal mean is slightly negative.** These are different policies.
ORIGINAL extraction gives positive Δ at all three seeds. Neither value "corrects" the other.

## 11. Cost (`cost_results.json`)

- **FP64 accumulation timing:** layer 8, 15 experts × (gu, down), 128 sequences, medians of 3.
  - Replay without accumulation 4.47 s; FP32 accumulation +0.49 s; FP64 accumulation +1.15 s.
  - FP64 is therefore 2.3× the FP32 accumulation cost. That is a small share of the pipeline, which is dominated by
    GPTQ: about 367–369 s per configuration.
- **Memory:** FP64 block buffers double accumulator memory, from about 5.9 to 11.9 GB per layer at K = 4. Without
  chunking they would not fit next to the layer on this 20 GB slice. The MVP chunked 15 experts at a time; peak GPU was
  6.9 GB.
- **Experimental cost:**
  - Canonical Hessian reconstruction (3 orders × 4 layers × 4 chunks): 7.5 min per strategy.
  - Quantization: 420–430 s per configuration (×4, including the regression and the repeat).
  - PPL: 22–50 s per evaluation.
  - Peak host memory 47 GB.
  - Storage 11.9 GB of Hessians per strategy (FP32 canonical + original) and 4.2 GB per weight set, in
    `~/moe_cache_fp64_canonical`.
- **Deployment cost:** one FP64 accumulation and one cast. The 3-order replays, stored duplicate Hessians and repeat
  solves are experimental overhead.

## 12. Correctness (`correctness_results.json`): 24/24 pass

Selection hashes; no leakage; activation hashes; contributions unchanged (FP32 path == preflight, 480/480 × 3 orders);
block membership; FP64 protocol (exact reproduction of the historical control); order invariance; cast once and served
unchanged (480/480); finite and symmetric; P_O 240/240; GPTQ parameters; solver unchanged (sha256 + clean git diff);
0 RTN fallbacks; 240 experts; 720 tensors; BF16 installation integrity I1–I4; routers unchanged and routing equal to
R_O; non-target parameters unchanged; windows sha and 65,408 targets; repeated evaluation identical; repeated
quantization identical; historical FP32 controls (loaded-pathway regression, OO / order-null PPL equal to the stored
values); historical artifacts unchanged (sha256); tests pass.

**Disclosure about the test run.** The first full-suite run executed with CUDA hidden, a setting copied from the
order-null runner. One GPU-only test, `test_forward_b1_is_batch_independent_on_tiny_model`, then failed with "No CUDA
GPUs are available". That log is kept as `logs/pytest_full_cpu_only_first_attempt.log`. The suite was re-run with the
GPU visible: **625 passed**. No code under test changed between the two runs.

## Answers

1. **Same Hessian across tested orders in FP64?** Yes: 480/480 per strategy, bitwise, for 3 orders.
2. **Does the FP32 cast preserve it?** Yes, bitwise.
3. **Reproducible quantization?** Yes. The FR repeat in a fresh process is bitwise identical down to per-window NLL.
4. **Canonical PPL:** FR 9.271421, FC 9.278551.
5. **Ranking vs original FP32:** the same, FR better (+0.007129 vs +0.010057).
6. **Position in the order-null range:** inside it, 0.31 order-SD below the order mean.
7. **Effective-weight variation from accumulation order alone:** about 0.007 % of BF16 elements (rel. L2 about 8e-4).
   About 90 % of that is scale bits, not code flips. PPL moves by up to 0.0036.
8. **Act-order and groups:** fixing them jointly (AOF vs AO) cuts the elements changed from about 39 % to about 17 %,
   rel. L2 by about 41 %, and the paired Δ SD from 0.0056 to 0.0017. FR's SD falls, FC's rises. Their individual roles
   are not separated.
9. **Why reconstruction is stable while PPL varies:** the local penalty is large (2.5 %) and averaged. The model-level
   effect is about 0.1 % and sensitive to discrete weight changes. The mechanism is hypothesized, not measured.
10. **PPL variability under natural grouping:**
    - Per-strategy SD about 0.0044 at seed 0.
    - Paired Δ SD 0.0064 at seed 0, pooled within-seed 0.0069 over 3 seeds.
    - Range −0.0135 to +0.0099.
11. **Does grouping variability dominate the average effect?** Yes: |mean| = 0.18 × the within-seed SD, and 40 % of
    differences have the opposite sign.
12. **Why window intervals are insufficient:** they hold the weights fixed. Different valid executions give window
    intervals excluding zero in opposite directions.
13. **Canonical vs execution-marginal:** a reproducible single-policy reference vs robustness over a stated policy
    distribution. Neither estimates the other.
14. **Isolated vs unresolved:**
    - **Isolated:** FP32 accumulation order (A); activation perturbation with routing and act-order frozen (B); the
      joint act-order + group effect (B vs C).
    - **Unresolved:**
      - routing changes alone vs activation changes in natural grouping (only the earlier 2×2 factorial at one
        grouping);
      - act-order vs group membership separately;
      - which experts or tokens carry the PPL changes;
      - other GPUs or kernels;
      - full depth / packed INT4.
15. **Protocol:** see the recommendation below.

## Supported findings

- FP32 Hessian summation order alone changes GPTQ weights (about 0.007 % of elements) and PPL (up to 0.0036), with
  everything else fixed.
- FP64 block accumulation with a single cast removed the order dependence for all tested orders, and the canonical
  configuration is bitwise reproducible.
- The canonical FR/FC difference (+0.0071) agrees in sign with the ORIGINAL FP32 pipeline and lies inside its
  order-null range.
- Act-order and group membership, jointly, account for a large share of effective-weight sensitivity under activation
  perturbations.
- Equally valid batch groupings change FR-vs-FC PPL rankings within each seed. Expert reconstruction ranks FC worse in
  every configuration.
- Fixed-model window intervals do not capture execution-induced variability. The canonical and execution-marginal
  estimands answer different questions.

## Reproducibility recommendation (proposed practice, based on this project's evidence)

1. Publish a canonical extraction and accumulation policy: batch-1 or fixed grouping; FP64 block accumulation with one
   cast.
2. Record calibration sequence IDs, replay order and execution grouping.
3. Record GPTQ precision and the act-order policy, including any override.
4. Verify that re-quantizing from identical Hessians is bitwise deterministic, and that repeated evaluation is too.
5. Report method effects over multiple allocation seeds.
6. When effects are small (here below about 0.01 PPL), add paired execution-grouping replicates and nested intervals.
7. Label fixed-model window intervals as conditional on the weights; never present them as method-level uncertainty.
8. Report local reconstruction and aggregated MoE output separately from PPL.
9. Store weight, Hessian and activation hashes.
10. Do not claim a method is better on the basis of one small PPL difference.

## Limitations

- **Scope of the canonical result:** one allocation seed, ORIGINAL extraction, and 3 tested accumulation orders. FP64
  sums happened to be exact for these addends; this is not a general invariance proof.
- **What FP64 does not fix:** it removes accumulation-order variability only. BF16 activations, GEMM rounding, routing
  near-ties, act-order and the solver are untouched, and the GPTQ pipeline is not FP64.
- **Hardware:** one GPU and software stack (A100 MIG 2g.20gb, torch 2.11 / CUDA 12.8, TF32 off). No cross-hardware
  claim.
- **Statistics:** 5 deterministic permutations or groupings per mechanism and 3 allocation seeds give descriptive
  statistics, not significance tests. The between-seed variance is very poorly bounded.
- **Model scope:** partial-model fake quantization (4 layers, BF16 weights). Nothing is claimed about full-depth or
  packed INT4 inference.
- **What was not measured:**
  - no dense or shared-expert control, so MoE-specific routing amplification is not isolated (future work, needing its
    own matched protocol);
  - no causal attribution to experts or tokens;
  - no irreducible noise floor is inferred.

## Artifacts

- **Run outputs:** `runs_fp64_canonical_reproducibility/` (all required JSONs, `summary.md`, figures 1–6 in png and pdf,
  logs, results).
- **Code:**
  - `moe_calib/fp64_canonical.py`
  - `diagnostics/fp64_canonical_gptq.py`
  - `diagnostics/report_numerical_reproducibility.py`
  - `tests/test_fp64_canonical.py` (7 tests)
- **Stored Hessians and weights** (outside the repo): `~/moe_cache_fp64_canonical/`.
- **Unchanged:** the production solver, Profiler and accumulator, and all historical experiments and artifacts.
