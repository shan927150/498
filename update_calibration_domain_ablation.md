# Calibration-domain ablation: C4 vs WikiText-2-train calibration (SC vs Random)

**Answer:**
- **In-domain calibration helps a little.**
  - WikiText-2-train calibration lowers WikiText-test NLL vs C4 calibration by −0.00093 nats/token (Random) and −0.00064 (SC).
  - Both are negative in 3/3 calibration groupings, at 1.8× and 1.2× the prespecified grouping SD.
  - That is a ~17–24 % smaller 4-layer quantization NLL penalty.
  - It goes with a ~19–22 % drop in local WikiText MoE-output error (B2, P_seq) and a +15 % worse C4 reconstruction.
- **SC adds nothing detectable at model level under either source.**
  - ΔNLL(SC − FR) = −0.00023 (C4) and +0.00005 (WT), mixed signs, ≤ 0.44× the reference SD.
  - SC's local gains persist but are smaller with WT calibration: B2 −0.35 % vs −0.69 %.
  - The domain × selection interaction is not distinguishable (+0.00029 nats).
- **Calibration-domain mismatch does not explain SC's reconstruction-vs-PPL gap.**
  - The domain experiment suggests NLL moves roughly in proportion to local MoE-output error when that error changes by ~20 %.
  - At that rate, SC's < 1 % local gain is worth ~1e-5 nats, 20–50× below grouping noise.
- **Outcome: mixed A + C.** Exploratory: one selection seed, three groupings.

This is a controlled dataset ablation:
- No allocator, GPTQ solver, SC objective or evaluation was changed.
- The WT arm re-uses the existing SC selector and sketch pass with identical SETTINGS, the existing warm-start / FR / grouping conventions (unit-tested to reproduce the stored C4 selections bit for bit), the frozen partial-PPL installer and windows, and the loss-alignment MVP B capture.

## Setup
- **Model and quantization:** Qwen1.5-MoE-A2.7B; GPTQ 4-bit, g128, act-order, percdamp 1.0 on the routed-expert gate/up/down of layers 1/8/16/23. Non-sequential calibration on cached BF16 inputs; B = 128; selection seed 0.
- **Pools:**
  - **C4:** the existing 2,000-sequence pool.
  - **WikiText-2 train:** `"\n\n".join(train)`, one tokenizer call (the test-window convention), 4,918 complete 512-token windows. 2,000 sampled with seed 20261117.
  - **Leakage:** 0 test windows and 0 of 61,568 test 32-grams appear in the pool.
- **WT selections:**
  - Warm start from the WT typical∩random halves (same rule as C4).
  - WT-FR = warm + 96 random.
  - WT-SC = existing SC optimizer on WT-only sketches and references. Covariance objective 0.125 vs 0.165 for FR; 53 swaps; deterministic.
- **Extraction:**
  - ORIGINAL pool caches, reported separately.
  - 3 groupings (20261111–13, batches of 8): WT unions are new; C4 re-uses the SC E3 Stage B seed-0 caches, re-run for B2 and reproduced bitwise.
- **Metrics:**
  - Primary: mean NLL on the frozen 128 × 512 WikiText-2 test windows.
  - Secondary: P_seq and B2 on the 32-window BF16 capture; C4 held-out reconstruction.
- **Freezes:** protocol sha `9c817814f5456d9a` (frozen before any GPU result); selections frozen before any quantization.

## Results (grouping means over 3 groupings; NLL in nats/token, others relative)

| effect | ΔNLL (per grouping) | label | B2 pooled | P_seq | C4 recon |
|---|---|---|---|---|---|
| SC − FR, C4 calibration | −0.00023 (+0.00048 / −0.00090 / −0.00028) | not distinguishable | −0.69 % | −0.51 % | −0.66 % |
| SC − FR, WT calibration | +0.00005 (+0.00009 / −0.00028 / +0.00035) | not distinguishable | −0.35 % | −0.33 % | −0.58 % |
| WT − C4, Random | **−0.00093** (−0.00082 / −0.00073 / −0.00122) | consistent negative | **−19.2 %** | **−21.9 %** | +15.4 % |
| WT − C4, SC | **−0.00064** (−0.00122 / −0.00011 / −0.00059) | consistent negative | −19.0 % | −21.8 % | +15.5 % |
| interaction (SC−FR)_WT − (SC−FR)_C4 | +0.00029 (−0.00040 / +0.00062 / +0.00063) | not distinguishable | +0.34 % | +0.18 % | +0.08 % |

Labels are prespecified:
- **Consistent** = the same sign in 3/3 groupings and |mean| > 5.26e-4. That threshold is the reference SD: the within-seed grouping SD of C4 SC−FR from the SC E3 study.

Mean NLL per cell (grouping mean) and NLL − BF16 (BF16 = 2.223227):

| cell | mean NLL | NLL − BF16 |
|---|---|---|
| C4-FR | 2.227142 | +0.00392 |
| C4-SC | 2.226909 | +0.00368 |
| WT-FR | 2.226217 | +0.00299 |
| WT-SC | 2.226269 | +0.00304 |

ORIGINAL extraction (shown separately):

| cell | mean NLL |
|---|---|
| C4-FR | 2.226552 |
| C4-SC | 2.227439 |
| WT-FR | 2.226340 |
| WT-SC | 2.226315 |

Full per-configuration table, per-layer B2 and figures: `runs_calibration_domain_ablation/summary.md`. Answers to the 7 questions: `runs_calibration_domain_ablation/interpretation.md`.

## Checks
- **Baselines:**
  - The BF16 baseline reproduces the stored PPL (9.237090) exactly; B2 = 0 with the original weights.
  - All 7 C4 re-runs reproduce their stored results bitwise: installed-weight sha256, per-expert reconstruction and per-window NLL. C4-FR ORIG also reproduces the MVP B B2 sums.
- **WT pool and caches:**
  - The WT sketches pass identity (≤ 1.6e-6) and routing-count checks.
  - All 3 WT grouping caches pass the reversed-order bitwise control.
- **Every configuration:** 240 experts quantized, 0 RTN fallbacks, untouched parameters unchanged, no non-finite weights, 65,408 scored tokens, identical windows and capture.
- **One flag:**
  - WT-SC G20261111 exceeds the frozen BF16-noise output tolerance on one prespecified check expert, L1/e0 (0.00557 vs 0.00510; quant effect 3.2 %).
  - Its weights are bitwise BF16(W_q). It is kept and flagged; the tolerance was not changed.
  - Dropping it changes no conclusion.
- **Tests:** 8 new CPU tests (`tests/test_calibration_domain.py`). The loss-alignment and SC E3 suites still pass (41 total).

## Limitations
- One selection seed and three groupings. Grouping replicates are not allocation seeds.
- C4 and WT caches are different activations, so cross-source differences carry both cells' grouping noise.
- C4 grouping caches batch a 374-row 4-strategy union; WT caches batch a 221-row 2-strategy union. They are matched by rule and seed only.
- The WT pool comes from one small corpus: 2,000 of 4,918 windows.
- B2 / P_seq are fixed-input local metrics on 32 windows.
- The NLL ∝ local-error proportionality is observed once, not established.

## Next step
Replicate WT-FR vs C4-FR on allocation seeds 1–2 (3 groupings each, with B2), about 1.5 h. It is the only effect near detectability and a direct test of the B2 ↔ NLL proportionality. More SC-vs-FR PPL runs are not justified.

## Reproduction
```bash
scripts/run_calibration_domain_ablation.sh tests
scripts/run_calibration_domain_ablation.sh audit      # CPU: artifact_audit.md
scripts/run_calibration_domain_ablation.sh pool       # CPU: WikiText-train pool + leakage checks
scripts/run_calibration_domain_ablation.sh protocol   # CPU: freeze
scripts/run_calibration_domain_ablation.sh cache      # full model, ~10 min
scripts/run_calibration_domain_ablation.sh routing    # partial, ~5 min
scripts/run_calibration_domain_ablation.sh sketch     # partial, ~9 min
scripts/run_calibration_domain_ablation.sh select     # GPU, ~1 min (+ freeze)
scripts/run_calibration_domain_ablation.sh gmanifest
scripts/run_calibration_domain_ablation.sh gcache     # full model, ~4 min
scripts/run_calibration_domain_ablation.sh smoke      # BF16 check, C4-FR ORIG validation, WT-FR/SC ORIG (~20 min)
scripts/run_calibration_domain_ablation.sh grouped    # 12 configs, ~6 min each
scripts/run_calibration_domain_ablation.sh report
```
Run one full-model process at a time (72 GB host RAM cap). Environment: `~/venvs/moe`, torch 2.11.0+cu128, transformers 4.51.3, A100 MIG 2g.20gb, `OMP_NUM_THREADS=8`.

## Artifacts
- Code:
  - `moe_calib/calibration_domain.py`
  - `diagnostics/calibration_domain_ablation.py`
  - `diagnostics/report_calibration_domain_ablation.py`
  - `tests/test_calibration_domain.py`
  - `scripts/run_calibration_domain_ablation.sh`
- `runs_calibration_domain_ablation/`:
  - `protocol.json`, `dataset_manifest.json`, `selection_manifest.json`, `freeze.json`, `artifact_audit.md`
  - `per_configuration.csv`, `paired_grouping_results.csv`, `domain_interaction.csv`, `per_layer_b2.csv`, `results_summary.json`
  - `results/`, `grouping/`, `wt_sc/` (sketch validation)
  - `summary.md`, `interpretation.md`
  - `figures/` (5)
  - `logs/`
- `~/moe_cache_calibration_domain/`: WT pool ids and layer caches, routing profile, sketches, grouping caches (not in the repo).
