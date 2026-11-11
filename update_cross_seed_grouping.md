# Cross-seed × calibration-grouping robustness (FR vs FC, four-layer GPTQ, allocation seeds 0/1/2)

**Decision (rule frozen in `runs_cross_seed_grouping/protocol.json`, sha256 `404a0274a7309c3b…`, before any new quantization): GROUPING-VARIATION-DOMINATES** (not ambiguous)

## Question

The multi-grouping study (`update_multigroup_robustness.md`) found that at allocation seed 0, five equally valid calibration batch groupings move FC − FR PPL across zero (mean −0.0019, SD 0.0064). That study tested only one selection pair.

This study asks whether the grouping-averaged FR vs FC comparison is consistent across allocation seeds 0, 1 and 2. Equivalently: once grouping noise is averaged out, is there a selection effect, and does it depend on the seed?

## Design (frozen before any new result)

- **Fixed:** the same model, layers (1/8/16/23, 240 routed experts), GPTQ settings (4-bit, g128, percdamp 1.0, act-order, fake-quantized BF16), E3 non-sequential calibration, reconstruction rows (`~/moe_cache` test 2128–2255), WikiText-2 windows (128 × 512) and evaluation batch 8 as the multi-grouping study.
- **Allocation seed s ∈ {0, 1, 2}** is the outer unit. The B = 128 FR and FC selections are frozen and verified against their hashes:

  | seed | FR | FC | union size |
  |---|---|---|---|
  | 0 | `ec355fb8d866e4ff` | `0353229b6d3090db` | 217 |
  | 1 | `e36d91e6b151320b` | `325e8efc5aa125e7` | 220 |
  | 2 | `cf1f867af56f29d1` | `f63a4a6122b79aab` | 222 |

- **Grouping g ∈ {20261111 … 20261115}** is nested within each seed. The policy is exactly `multigroup_robustness.grouping_batches`: sort seed s's union U_s, apply `default_rng(g).permutation`, and cut consecutive batches of 8. The same RNG seed therefore permutes a *different* union for each allocation seed, so groupings are not crossed across seeds.
- **Paired:** FR and FC share the grouping within each (s, g). The analysis works only with Δ[s, g] = PPL(FC, s, g) − PPL(FR, s, g).
- **Seed 0 is reused** from `runs_multigroup_robustness`, after integrity checks. Seeds 1 and 2 are 20 new configurations, each run in a fresh process.
- **Statistics:**
  - Nested structure only: per-seed mean, SD and SE over 5 groupings; pooled within-seed SD W; SD of the seed means.
  - Hierarchical bootstrap: resample seeds, then groupings within each seed.
  - A flat 15-observation analysis is rejected in code, because it would be pseudo-replication.

## Validity and determinism controls (all passed)

- **Seed-0 reuse:** all 10 reused results match their frozen artifacts.
- **Determinism control:** FR s0 G20261111 was re-quantized in a fresh process from the reused cache. Weights, PPL, reconstruction and per-window NLL are all **bitwise identical**. Repeated PPL evaluation also gives identical results.
- **New caches (10):**
  - Reassembled token IDs match each union's ORIGINAL rows.
  - Each grouping's batches, executed in reversed order, reproduce the cache bitwise (10/10).
- **Hashes:** selection hashes, union sha256 and grouping manifests match the frozen protocol. Window tokens are unchanged.
- **All 20 new configurations:** 240 experts and 720/720 projections changed; 0 RTN fallbacks; 0 non-finite values; untouched parameters unchanged; expert-output checks pass.
- **Test suite:** 242 tests pass, including the new `tests/test_cross_seed_grouping.py`.
- **Interruption:** the run was interrupted once by a container restart during `FR_s2_G20261114`. That configuration had written no result, so it was re-run from scratch in a fresh process. All other results come from uninterrupted runs.

## Results

### Paired PPL differences (primary; BF16 PPL 9.237090)

| seed | grouping | FR PPL | FC PPL | Δ = FC − FR |
|---|---|---|---|---|
| 0 | 20261111 | 9.278182 | 9.274113 | −0.004069 |
| 0 | 20261112 | 9.268159 | 9.275760 | +0.007601 |
| 0 | 20261113 | 9.271035 | 9.265146 | −0.005889 |
| 0 | 20261114 | 9.278102 | 9.269707 | −0.008394 |
| 0 | 20261115 | 9.272717 | 9.273808 | +0.001091 |
| 1 | 20261111 | 9.266099 | 9.265572 | −0.000527 |
| 1 | 20261112 | 9.267534 | 9.263857 | −0.003677 |
| 1 | 20261113 | 9.265318 | 9.268279 | +0.002961 |
| 1 | 20261114 | 9.267026 | 9.275531 | +0.008505 |
| 1 | 20261115 | 9.273307 | 9.269481 | −0.003826 |
| 2 | 20261111 | 9.262634 | 9.272568 | +0.009934 |
| 2 | 20261112 | 9.268870 | 9.269566 | +0.000696 |
| 2 | 20261113 | 9.282398 | 9.268849 | −0.013549 |
| 2 | 20261114 | 9.271457 | 9.266417 | −0.005041 |
| 2 | 20261115 | 9.272849 | 9.268318 | −0.004531 |

| seed | mean Δ (m_s) | SD (ddof 1) | SE (SD/√5) | median | FC worse / better | ORIGINAL-extraction Δ (reference) |
|---|---|---|---|---|---|---|
| 0 | −0.001932 | 0.006364 | 0.002846 | −0.004069 | 2 / 3 | +0.010057 |
| 1 | +0.000687 | 0.005176 | 0.002315 | −0.000527 | 2 / 3 | +0.002706 |
| 2 | −0.002498 | 0.008624 | 0.003857 | −0.004531 | 2 / 3 | +0.013070 |

- **Overall:**
  - Mean of seed means M = **−0.001248**. Median of all 15 Δ values: −0.003677.
  - Range −0.013549 to +0.009934. FC is worse in 6/15.
- **No seed mean is distinguishable from zero.** Every |m_s| is below 1 SE, and no seed is confidently positive or negative (> 2 SE).
- **Scale of the effect against grouping noise:**
  - Pooled within-seed grouping SD: W = **0.006872**.
  - |M| / W = **0.18**.
  - 40 % of the 15 Δ values have the opposite sign to M.
- **Seed means agree more than grouping noise alone predicts:**
  - Observed SD of seed means: 0.001699.
  - Expected from grouping averaging alone (W/√5): 0.003073.
  - So the estimated between-seed variance component is 0 (descriptive; only 3 seeds).
- **Hierarchical bootstrap** (20,000 replicates, seed 20261120):
  - Point −0.001248, 95 % interval **[−0.004777, +0.002155]**. 24 % of replicates are positive.
  - There are only 10 distinct seed resamples, so the interval is unstable and descriptive, not population-level inference.
- **Decision path:**
  - Not ALLOCATION-SEED-DEPENDENT: there are no conflicting confident seeds, and SB = 0.0017 < 2 × EXP = 0.0061.
  - GROUPING-VARIATION-DOMINATES applies: |M| = 0.18 W < 0.5 W, and the opposite-sign fraction is 0.40 ≥ 1/3.

### Paired-window bootstrap (per (s, g); conditional on one fixed model pair)

- 4/15 window CIs exclude zero: 2 favour FC (s0 G20261114, s2 G20261113) and 2 favour FR (s1 G20261114, s2 G20261111).
- A window-level "significant" difference in either direction occurs for individual groupings within the same seed. A window CI on one extraction run says nothing about calibration-execution uncertainty.

### Reconstruction (same 240 experts, same evaluation rows)

- **FC reconstruction is worse in 15/15 configurations.**
- **Relative FC penalty:**
  - Seed 0: +2.55 %. Seed 1: +2.19 %. Seed 2: +2.71 %.
  - Within each seed the penalty varies by ≤ 0.014 percentage points across groupings. Overall range +2.19 % to +2.71 %.
- **Within-strategy reconstruction across groupings:** relative range ≤ 1.3e-4.
- **Reconstruction does not track PPL:**
  - Within-seed correlation of the reconstruction penalty with Δ: r = +0.46, −0.66, −0.21 for seeds 0, 1, 2. Each has n = 5, so they are descriptive only.
  - Across seeds, seed 2 has the largest reconstruction penalty and the most negative mean Δ.
- So the robust, seed-consistent reconstruction penalty of FC does **not** translate into a PPL penalty at this precision.

### Activations and routing (each grouping vs that seed's ORIGINAL extraction)

| layer input | rel. L2 (all seeds) | % tokens with top-4 set changed |
|---|---|---|
| 1 | 1.5e-4 – 1.6e-3 | 0.61 – 0.68 |
| 8 | 5.7 – 5.9e-3 | 3.8 – 3.9 |
| 16 | 1.47 – 1.49e-2 | 3.8 – 4.0 |
| 23 | 2.13 – 2.18e-2 | 6.6 – 6.8 |

- The perturbation size is essentially identical across seeds and groupings. Layer-1 rel. L2 is higher for seeds 1 and 2 (about 1e-3) than for seed 0 (about 1.7e-4), but routing changes are the same.
- As at seed 0, aggregate activation and routing diagnostics do not predict the sign or size of Δ. No causal mechanism is claimed.

### Observation: the ORIGINAL extraction favours FR in all three seeds

- The ORIGINAL-extraction Δ (from `cache_inputs.py`, with a different batching policy over different rows) is positive at every seed: +0.0101, +0.0027, +0.0131.
- It lies above the grouping-averaged mean at every seed, and above all five grouping replicates at seeds 0 and 2.
- Earlier FR-vs-FC PPL conclusions drawn from the ORIGINAL caches (e.g. "FC slightly worse in PPL") are therefore at least partly specific to that one extraction policy.
- With 3 seeds and one ORIGINAL draw per seed this is an observation, not a test. ORIGINAL is a different policy and is never pooled with G20261111–15.

## Answers

1. **Does grouping-averaged FC − FR agree across allocation seeds?**
   - Yes, in the sense that all three seed means are near zero (−0.0019, +0.0007, −0.0025) and agree with each other within grouping noise.
   - None is distinguishable from zero.
2. **Is there a selection effect larger than calibration-execution noise?**
   - No. |M| = 0.0012 is 0.18 × the within-seed grouping SD.
   - A single (seed, grouping) comparison spans −0.0135 to +0.0099.
3. **Is FC worse in PPL?**
   - Not at this precision. The hierarchical interval [−0.0048, +0.0022] contains zero, and 9/15 paired comparisons favour FC.
   - FC is consistently worse in **reconstruction** (+2.2 to +2.7 %, 15/15). That proxy does not carry over to PPL.
4. **Practical implication:**
   - A four-layer FR-vs-FC PPL difference of order 0.01 cannot be resolved by one extraction run per seed, nor by three seeds with one grouping each.
   - Resolving a true effect of about 0.001 would need far more replicates. At SD about 0.007 per paired comparison, roughly 200 or more paired (s, g) runs would be needed for SE ≈ 0.0005.

## What remains unproven

- Whether FC and FR truly differ in PPL. The data are consistent with a true difference of zero, or with a small difference of either sign (|Δ| ≲ 0.005).
- Generality beyond this grouping policy (217–222 calibration rows, batches of 8), this GPU and kernel stack, this model, and these four layers, including full depth.
- Whether the systematic positive ORIGINAL Δ reflects that extraction policy or chance (3 draws).
- Three allocation seeds bound the between-seed variance only very loosely. The "0" component estimate is a truncated, highly uncertain descriptive quantity.
- Everything uses fake-quantized weights; nothing is claimed about packed INT4 inference.

## Compute

- **New caches:** about 62–98 s per (seed, grouping), plus about 35 s per reversed-order control.
- **Diagnostics:** 417 s.
- **Quantization:** about 290 s per configuration (Hessian 29 s; GPTQ + reconstruction 225 s), plus about 40 s per PPL evaluation. 21 fresh-process runs (20 new + 1 control).
- **Peak memory:** 8.4 GB GPU, 43.4 GB host.
- **Disk:** new caches 18.5 GB, new effective weights 83.1 GB.

## Files

| file | role |
|---|---|
| `moe_calib/cross_seed_grouping.py` | Pure helpers: pairing, nested variance summary, hierarchical bootstrap, frozen decision rule. |
| `diagnostics/cross_seed_grouping.py` | Stages: preflight, manifest, cache, diagnostics, quant (plus `--control`), report. |
| `diagnostics/report_cross_seed_grouping.py` | Results, summary and figures 1–6. |
| `scripts/run_cross_seed_grouping.sh` | Resumable runner (`STAGE=preflight|tests|manifest|cache|diagnostics|quant|report`). |
| `tests/test_cross_seed_grouping.py` | CPU tests (pairing, nesting and pseudo-replication guard, decision rule, bootstrap). |

Outputs are in `runs_cross_seed_grouping/`: `protocol.json`, `selected_set_manifest.json`, `grouping_manifests/`, `cache_manifest.json`, `activation_diagnostics.json`, `ppl_results.json`, `reconstruction_results.json`, `paired_differences.json`, `variance_analysis.json`, `summary.json`, `summary.md`, `results/`, `logs/` and `figures/` (fig1 paired Δ, fig2 within-seed sensitivity, fig3 grouping-averaged effect, fig4 reconstruction vs PPL, fig5 sources of variation, fig6 activation/routing; PDF + PNG).
