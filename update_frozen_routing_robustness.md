# Frozen-routing robustness across five calibration batch groupings

**Setup:** Qwen1.5-MoE-A2.7B. Routed experts of layers 1/8/16/23 are GPTQ 4-bit (g128, percdamp 1.0, act-order), B = 128, allocation seed 0. Everything else stays BF16.
**Runner:** `scripts/run_frozen_routing_robustness.sh`
**Artifacts:** `runs_frozen_routing_robustness/`

## Question

The 2×2 factorial study (`update_frozen_routing.md`) found that at grouping G20261111:
- freezing the ORIGINAL expert routing while building the calibration Hessians (AO) kept PPL close to OO;
- AO also preserved the FR/FC ranking.

That was one grouping. This study asks whether the AO behaviour holds across all five prespecified groupings, G20261111–G20261115.

## Conditions

| condition | Hessian activations | Hessian routing | source |
|---|---|---|---|
| OO | X_O (ORIGINAL cache) | R_O | CTRL ORIGINAL (multigroup study), one reference per strategy |
| AA_g | X_g | R_g (natural) | multigroup study, reused (10 configs) |
| AO_g | X_g | R_O (frozen) | G20261111 reused from the factorial study; **8 new** (G20261112–15 × FR/FC) |

Frozen routing is applied only to the Hessian-building Profiler run. It goes through the validated `diagnostics/frozen_routing.run_config`, whose only change is that the grouping and output paths are now parameters with backward-compatible defaults. Everything else is the unmodified original code:
- inference and PPL evaluation;
- the reconstruction evaluator;
- GPTQ;
- router weights and the caches.

Alignment uses the original sequence id plus token position. Each grouping cache's row i is `union_calibration_ids[i]`, which I verified from every cache manifest.

## Correctness

- **Inputs:** the selections reproduce:
  - FR `ec355fb8d866e4ff` and FC `0353229b6d3090db`, 128 unique sequences each, 217 in the union;
  - the frozen warm start and pool membership;
  - no tune/test leakage.

  The five grouping manifests are unchanged, and the cache file sha256s match.
- **R_O was reused, not recomputed:** it is the factorial study's `routing_original.pt`, sha-checked per layer against the manifest. Each per-layer table is [512, 4], with distinct experts per token.
- **G20261111 AO regression (both strategies):**
  - forced Hessians recomputed bitwise: 480/480;
  - stored weights re-installed into the pristine model give the stored installed digest;
  - re-evaluated PPL equals the stored JSON float exactly (FR 9.265860908067534), per-window NLL included;
  - fingerprints of all 14 reused OO / AA / AO result JSONs and weight files match.
- **Natural AA_g Hessians:** recomputed for the diagnostics with the unmodified Profiler (no GPTQ). For G20261111 they equal the factorial study's AA Hessians bitwise, 480/480 for both strategies.
- **Token-count invariant:**
  - In every AO config, the routed-token count per expert equals the R_O count over the chosen sequences.
  - The AO count vector is identical across all five groupings for each strategy, while the AA counts differ across groupings.
  - All 240 experts receive tokens.
  - Down-projection counts equal gate/up counts.
- **Per-config checks** for all 8 new AO configs:
  - 240 experts and 720 tensors quantized, no RTN fallback, no NaN/Inf;
  - router and untouched parameters unchanged;
  - no forced routing at inference;
  - PPL evaluated twice per config, with identical results (|ΔPPL| = 0 and identical per-window NLL).
- **One installation check failed (blocker):**
  - In FC AO G20261114, the layer-1 expert-0 installed-output check gave `installed_vs_e3_rel` = 0.005595, above its tolerance of 0.005202 (1.075×).
  - The installed weights *are* bitwise the BF16 cast of E3's quantized weights. The check measures BF16 output drift relative to the FP32 solver output, and this expert has the smallest quantization effect (0.031) of the checked experts.
  - Across 972 such checks in all studies, this is the first exceedance. The highest earlier ratio was 0.95×, also at layer 1 expert 0.
  - The tolerance was not changed. Re-running would be bitwise deterministic, so it was not repeated.
  - Under the frozen rule, a failed correctness check makes the decision **RUN-BLOCKED**. All numbers below include this config and are descriptive only.

## Results (descriptive; includes the blocked FC AO G20261114)

| grouping | FR AA_g | FR AO_g | FC AA_g | FC AO_g | Δ_AA,g | Δ_AO,g |
|---|---|---|---|---|---|---|
| OO reference | 9.267851 | — | 9.277908 | — | Δ_OO +0.010057 | |
| G20261111 | 9.278182 | 9.265861 | 9.274113 | 9.277881 | −0.004069 | +0.012020 |
| G20261112 | 9.268159 | 9.270916 | 9.275760 | 9.274432 | +0.007601 | +0.003516 |
| G20261113 | 9.271035 | 9.276043 | 9.265146 | 9.273312 | −0.005889 | −0.002731 |
| G20261114 | 9.278102 | 9.273279 | 9.269707 | 9.274137 † | −0.008394 | +0.000858 † |
| G20261115 | 9.272717 | 9.268226 | 9.273808 | 9.274729 | +0.001091 | +0.006503 |

† failed one installation check (see above).

### Spread, distance and ranking

| quantity | AA | AO | AO/AA |
|---|---|---|---|
| FR SD (ddof 1) | 0.00442 | 0.00402 | **0.91** |
| FR range | 0.01002 | 0.01018 | 1.02 |
| FC SD | 0.00429 | 0.00175 | **0.41** |
| FC range | 0.01061 | 0.00457 | 0.43 |
| Δ (FC − FR) SD | 0.00636 | 0.00561 | **0.88** |
| Δ range | | | 0.92 |
| FR mean / median \|PPL − OO\| | 0.00579 / 0.00487 | 0.00381 / 0.00307 | AO closer in 3/5 |
| FC mean / median \|PPL − OO\| | 0.00620 / 0.00410 | 0.00301 / 0.00348 | AO closer in 4/5 |
| mean Δ | −0.00193 | +0.00403 | Δ_OO +0.01006 |
| mean \|Δ − Δ_OO\| | 0.01199 | 0.00681 | AO closer in 4/5 |
| FR/FC ranking preserved vs OO | 2/5 | 4/5 | |

**Paired intervention effects:**
- PPL_AO − PPL_AA:
  - FR: −0.0123, +0.0028, +0.0050, −0.0048, −0.0045
  - FC: +0.0038, −0.0013, +0.0082, +0.0044, +0.0009
- RankingInteraction, Δ_AO − Δ_AA: +0.0161, −0.0041, +0.0032, +0.0093, +0.0054

**Where the means sit relative to OO:**
- Under AA, both strategies' means drift toward each other: FR +0.0058 above OO, FC −0.0062 below OO.
- Under AO, the drifts are about half as large (FR +0.0030, FC −0.0030). So the mean Δ stays positive but still well below Δ_OO.
- A smaller distance to OO is a protocol-stability property, not better language modelling.

### Mechanism (prespecified diagnostic subset and all 720 tensors)

**Hessians** (median trace-normalized relative Frobenius difference vs OO over 16 Hessians):

| | AO, range over 5 groupings | AA, range over 5 groupings | AO/AA per grouping |
|---|---|---|---|
| FR | 0.0038–0.0039 | 0.0137–0.0147 | 0.27–0.28 |
| FC | 0.0049–0.0050 | 0.0143–0.0158 | 0.31–0.34 |

- AO is below AA in 5/5 groupings for both strategies.
- The worst diagnostic Hessian under AO is 0.016–0.021, against 0.042–0.053 under AA.

**Act-order:** columns changing GPTQ group vs OO are 2.4 % (FR) / 2.6 % (FC) under AO, against 7.3 % / 7.6 % under AA.

**Effective weights:**

| | elements differing vs OO, AO | elements differing vs OO, AA | rel L2 vs OO, AO | rel L2 vs OO, AA |
|---|---|---|---|---|
| FR | 39.0–39.3 % | 54.0–54.2 % | 0.069 | 0.089 |
| FC | 40.0–40.5 % | 55.1–55.2 % | 0.071 | 0.090 |

- AO_g and AA_g differ from each other in 53–54 % of elements.
- No projection is identical between any two conditions.

**Reconstruction:**
- Every AO and AA config is within ±0.0098 % of OO.
- The FC penalty over FR is +2.548–2.551 % under AO, +2.545–2.559 % under AA, and +2.549 % under OO.
- Reconstruction does not follow PPL.

## Decision (rule frozen in `protocol.json` before any new AO run; sha256 `21ba43d8…`)

**RUN-BLOCKED.** One AO configuration (FC G20261114) failed a prespecified installation-tolerance check. The failure is narrow: 1 of 8 expert checks, at 1.075× tolerance, with bitwise-correct installation. I did not retune the tolerance and did not reclassify the failure.

The frozen rule would return **FROZEN-ROUTING-PARTIALLY-STABILIZES** if that configuration were accepted. That result is **not the decision**. It satisfies:
- (b) FC SD ratio 0.41 ≤ 0.5;
- (e) ranking preserved in 4/5 under AO versus 2/5 under AA.

It fails:
- FR SD ratio 0.91;
- Δ SD ratio 0.88;
- FR closer-to-OO in only 3/5.

Note that criterion (b) itself depends on the blocked FC G20261114 value.

## Answers (descriptive, conditional on accepting the blocked configuration)

1. **Hessian sensitivity:** yes, consistently. AO is at 27–34 % of the AA Hessian change in every grouping, and the AO value is nearly constant across groupings (0.0038–0.0050).
2. **Weight variation:** only modestly. 39–40 % of elements still change, against 54–55 % under AA, and AO_g differs from AA_g in about 53 % of elements.
3. **FR PPL spread:** no (SD ratio 0.91, range ratio 1.02).
4. **FC PPL spread:** observed lower (SD ratio 0.41), but this rests partly on the blocked configuration.
5. **Paired Δ spread:** no (SD ratio 0.88). Δ_AO,g ranges from −0.0027 to +0.0120.
6. **Ranking preservation:** more often under AO (4/5) than AA (2/5). One AO grouping (G20261113) still reverses, and one (G20261114, blocked) is only +0.0009.
7. **Reconstruction:** essentially unchanged (≤ 0.01 %), with a stable FC penalty of about +2.55 %.
8. **G20261111:** an unusually favourable AO draw. Its FR AO (−0.0020 vs OO) and Δ_AO (+0.0120 ≈ Δ_OO) are the closest of the five. The other groupings put FR AO up to +0.0082 from OO.
9. **Is frozen routing sufficient?** No. Hessians are stabilized, but quantized weights are only partly stabilized, model-level PPL spread is not reduced for FR, and the strategy-difference spread is not reduced. Freezing routing is not sufficient to make single-grouping FR-vs-FC PPL comparisons reliable at this effect size (|Δ| ≈ 0.01, about the size of the grouping SD).

## Limitations

- Only allocation seed 0, five prespecified groupings and four quantized layers.
- Weights are fake-quantized BF16; nothing here measures packed INT4 inference.
- Frozen routing acts only on calibration-Hessian construction. The BF16 activation trajectories X_g are the natural ones.
- The ORIGINAL cache comes from a different batch-generation policy than the five randomized groupings.
- Five deterministic observations per cell cannot establish population-level variance reduction, and a smaller SD says nothing about mean language-model quality.
- The upstream sources of activation drift are not isolated, and this is not a full causal mediation analysis.

## Next step (not executed)

Find out why the GPTQ re-rounding that remains under frozen routing (about 39 % of elements) still moves PPL by the same amount as the natural runs. Re-quantize AO with the act-order permutation and the per-group scales/zeros frozen to OO's, so that only the within-group error-feedback updates see X_g. Run it for the same 10 AO cells (FR/FC × five groupings), with the same evaluation and decision structure. If PPL spread collapses, the remaining sensitivity sits in the scale/order choices rather than the Hessian off-diagonals.

## Compute

Each new AO config took 281–309 s to quantize plus 2 × 36–43 s for PPL. Peak memory was 8.4 GB GPU and 41.4 GB host.

| stage | time |
|---|---|
| capture (10 AA Hessian passes) | about 6 min |
| regression | 7 min |
| report | about 10 min |

Storage: 35 GB in `~/moe_cache_frozen_routing/robustness` (8 weight sets plus 18 Hessian files).

Tests: `tests/test_frozen_routing_robustness.py` (19 tests); full suite 280 passed.
