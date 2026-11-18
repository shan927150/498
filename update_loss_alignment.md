# Loss-alignment diagnostic (MVP A + B): why SC's reconstruction gain does not show up in PPL

**Answer:**
- **Expert weighting does not explain the mismatch.**
  - SC's −0.68 % R0 change is unchanged under routing-frequency weighting and under squared-router-weight weighting, on both C4 and WikiText references (98–101 % retained).
- **The gain survives the nonlinearity and the router aggregation.**
  - At fixed BF16 inputs and routing (seed 0, 32 WikiText windows), SC lowers the router-weighted routed-expert MoE output error B2 by **−0.68 %** (window bootstrap 95 % CI [−1.03, −0.32]).
  - It improves in all 4 layers.
- **SC therefore lands in Outcome C.** The local improvement reaches the MoE output; the PPL null comes from further downstream or from detectability.
  - A scale estimate (assumption: NLL proportional to local error energy) puts the expected NLL effect at about 2.5e-5 nats.
  - The per-configuration calibration-grouping SD of ΔNLL is about 5.5e-4.
- **FC behaves differently.** Its +2.55 % R0 penalty (C4 test tokens) shrinks to +0.49 % when the same projection objective is measured on the WikiText evaluation tokens. At the MoE output it is −0.14 % pooled and +0.46 % layer-equal, neither resolved.
  - So the FC "worse reconstruction, no PPL penalty" puzzle is largely about which tokens the error is measured on, not about expert weights or nonlinearity. This rests on one seed.

This is a diagnostic only:
- No allocator was introduced and nothing was tuned.
- Model weights, routing, the GPTQ solver and the frozen evaluation are unchanged.
- No existing artifact was modified. All new outputs are in `runs_loss_alignment/` and `~/moe_cache_loss_alignment/`.

## Inputs (what existed, what was regenerated)

| input | status |
|---|---|
| per-expert E3 test errors, FR/TR/FC/TC s0–2 | stored (`runs_typicality_matched`, via `runs_partial_ppl/selected_set_manifest.json`, sha256-checked) |
| per-expert errors + NLL, SC s0–2 | stored (`runs_shared_covariance_e3/results/A_SC_s*.json`) |
| ORIGINAL NLL, FR/TR/FC/TC | stored (`runs_partial_ppl/ppl_*.json`) |
| grouped NLL + per-expert errors | stored: FR/TR/TC/SC in Stage B (`runs_shared_covariance_e3/grouping_robustness`); FR/FC in `runs_multigroup_robustness` (s0) + `runs_cross_seed_grouping` (s1, s2). The two studies use different grouping caches, so grouped deltas are paired within a study only. |
| C4 routing frequency | stored (`runs_allocation_minimal/routing_profile_v2.npz`) |
| WikiText routing + router probabilities | stored (`~/moe_cache_inference_routing/bf16_reference_routing.pt`, same 128 PPL windows, sha-matched) |
| C4 router weights | **new**: layer replay of the 2,000-sequence pool at batch 1. Reproduces the stored per-sequence routing counts exactly; a first attempt at batch 8 differed by ±1 token on a few sequences because BF16 router ties depend on batch shape. |
| FR s0 / FC s0 effective GPTQ weights | stored (`~/moe_cache_batch_sensitivity/weights/{FR,FC}_ORIGINAL`). Installed sha256 equals the partial-PPL runs. |
| SC s0 effective GPTQ weights | **regenerated** with the unchanged E3 solver from the frozen selection. Per-expert errors and installed sha256 reproduce `A_SC_s0` bitwise. |

## Router semantics
Qwen1.5-MoE has `norm_topk_prob = false`. The applied weight is the FP32 softmax over all 60 experts, top-4, not renormalized (per-token sum 0.09–0.98), then cast to BF16.
- MVP A's R2 and all MVP B metrics use this BF16-cast applied weight.
- The shared expert is excluded from every routed-expert metric.

## Results

| metric | data / aggregation | FR | SC | FC | SC vs FR | FC vs FR |
|---|---|---|---|---|---|---|
| R0 projection reconstruction | C4 test, equal-expert, s0 | 0.004610 | 0.004579 | 0.004727 | −0.67 % | +2.55 % |
| R1 frequency-weighted | C4 pool / WikiText frequency, s0 | 0.004596 / 0.004564 | 0.004565 / 0.004532 | 0.004712 / 0.004679 | −0.67 / −0.68 % | +2.53 / +2.53 % |
| R2 router-weight²-weighted (heuristic) | C4 / WikiText, s0 | 0.004566 / 0.004640 | 0.004536 / 0.004608 | 0.004677 / 0.004753 | −0.66 / −0.68 % | +2.43 / +2.44 % |
| R0 / R1 / R2, 3-seed mean change | ORIGINAL, all seeds | | | | −0.67 to −0.69 % (all seeds negative) | +2.37 to +2.48 % |
| P_seq projection error on the B tokens | WikiText, E3 definition | 5.201e-3 | 5.175e-3 | 5.227e-3 | −0.51 % | +0.49 % |
| B0 expert MLP output error | equal-expert mean | 1.784e-2 | 1.775e-2 | 1.793e-2 | −0.50 % | +0.51 % |
| B1 uncancelled router-weighted error | Σ w²‖δ‖² / Σ‖m‖² | 1.4195e-2 | 1.4096e-2 | 1.4172e-2 | −0.70 % | −0.16 % |
| **B2 router-weighted MoE output error** | Σ‖Σ w δ‖² / Σ‖m‖², pooled | 1.4183e-2 | 1.4086e-2 | 1.4163e-2 | **−0.68 %** [−1.03, −0.32] | −0.14 % [−0.81, +0.47] |
| B2, layer-equal mean | | 1.1276e-2 | 1.1201e-2 | 1.1328e-2 | −0.67 % | +0.46 % |
| B3 cancellation ratio | Σ‖Σ w δ‖² / Σ w²‖δ‖² | 0.99912 | 0.99934 | 0.99935 | +0.02 % | +0.02 % |
| mean NLL, ORIGINAL (s0) | 128 windows | 2.226552 | 2.227439 | 2.227636 | +0.00089 | +0.00108 |
| ΔNLL, 5-grouping mean ± SD (s0) | paired within study | | | | −0.00020 ± 0.00052 | −0.00021 ± 0.00069 |
| ΔNLL, 5-grouping mean, all seeds | | | | | −0.00020 / −0.00071 / −0.00043 | −0.00021 / +0.00007 / −0.00027 |

R metrics use stored C4 test-split errors. B metrics are computed on 32 fixed WikiText windows (16,384 tokens per layer) with the BF16 model's own routing. Compare relative within-metric changes, never absolute values across rows.

### Correctness checks (all passed)
- **MVP A:**
  - R0 reproduces every stored macro reconstruction (1.9e-16).
  - R1_c4 reproduces E3's stored `recon_usage_weighted` (1.9e-16).
  - Every weight row sums to 1.
- **MVP B capture** (all 4 layers):
  - Router logits equal the module's output bitwise.
  - An exact re-execution of the HF sparse-MoE block (routed + shared) equals the module output bitwise.
  - Top-4 ids and FP32 weights equal the stored PPL-run routing bitwise.
  - Weights are ≤ 1, slots are descending, and there are no repeated ids.
  - The FP32 recompute of the experts deviates from the BF16 module by ≤ 0.4 % relative.
- **MVP B eval:**
  - Installed sha256 equals the expected value for BF16, FR, SC and FC.
  - All 83 non-target parameters are unchanged; there are no non-finite weights; the model is restored to pristine between runs.
  - The BF16 control gives exactly zero error.
  - Every expert has ≥ 323 routed eval tokens.
  - The FP32 routed reference deviates ≤ 0.5 % from the module's BF16 routed sum.
  - BF16-native compute reproduces the FP32 conclusions.
- 15 new CPU tests in `tests/test_loss_alignment.py`: weighted aggregation, routing-index alignment, HF top-k semantics, zero error, normalization, brute-force B metrics, constructive (ratio 2) and destructive (ratio 0) interference. All earlier test suites still pass.

## Answers

1. **Does SC's advantage persist under routing-frequency weighting?** Yes. It is −0.68 % (C4 pool) and −0.69 % (WikiText), negative on all 3 seeds, versus −0.68 % for R0.
2. **Under actual router-gate weighting?** Yes: −0.67 % (C4) and −0.69 % (WikiText). The improvement is broad: per-expert gains are uncorrelated with weight, and effective experts per layer are 34–60.
3. **Does better projection reconstruction give better nonlinear expert outputs?** Yes for SC, seed 0.
   - The equal-expert B0 change is −0.50 %, against −0.51 % for projection error on the same tokens. Per expert, Spearman(ΔP, ΔB0) = +0.61, and 178/240 experts improve.
   - Pooled B0 is flat (+0.01 %). One high-output-energy, low-router-weight expert (L23/26, 14 % of output energy) moves the other way.
4. **Does it give better router-weighted MoE outputs?** Yes for SC, seed 0.
   - B2 is −0.68 % [−1.03, −0.32] and improves in all 4 layers.
   - Expert errors are essentially uncorrelated per token (B3 ≈ 0.999), so the aggregation neither cancels nor amplifies the gain.
5. **Which metric aligns best with grouping-averaged NLL?** None can be shown to.
   - Grouping-averaged ΔNLL is within about 1 grouping SD of zero for every strategy and seed.
   - Every R metric gives Spearman ≈ 0 against it (12 dependent points).
   - Only the WikiText-token metrics (P_seq, B0, B2) agree with the observed equal PPL of FC and FR, because they remove most of FC's C4 penalty. This is suggestive, from one seed, not demonstrated alignment.
6. **Smallest justified next experiment:** two cheap steps, no new allocator.
   - **(a) Replicate MVP B on allocation seeds 1–2** for FR/SC/FC. This needs 6 weight regenerations (~9 min each, partial load) and ~2 min of evaluation. It tests whether SC's B2 gain and FC's shrinking penalty on WikiText tokens replicate.
   - **(b) Measure B2's calibration-grouping stability** on the already-saved grouped FR/FC weights of seeds 1–2 (`~/moe_cache_cross_seed_grouping/weights`, 20 configurations, no GPTQ).
   - If B2 differences are stable across groupings while NLL is not (as the reconstruction error was in Stage B), B2 is the right low-noise endpoint. Detecting a 2.5e-5-nat NLL effect directly would take thousands of grouped PPL evaluations, so more PPL runs are not a justified next step.

## Limitations
- MVP B covers one allocation seed and 32 windows; the bootstrap reflects token sampling only.
- B metrics freeze inputs and routing. They exclude error propagation through later layers and any routing changes, both of which the PPL evaluation includes. B2 is not model-level error.
- R2 is a heuristic: squared router weight ignores expert output scale and the nonlinearity.
- The NLL-scale argument assumes NLL ∝ local error energy. It was not measured.
- The FC token-dependence result compares a C4 test-split metric (128 sequences) with a 32-window WikiText metric. It is not yet replicated across seeds.
- Pooled B2 is dominated by layer 23 (84 % of routed output energy), so layer-equal B2 is reported alongside it.

## Reproduction
```bash
scripts/run_loss_alignment.sh tests        # 15 CPU tests
scripts/run_loss_alignment.sh a_router     # C4 router weights, GPU partial load, ~7 min
scripts/run_loss_alignment.sh a            # MVP A analysis (CPU, seconds)
scripts/run_loss_alignment.sh b_protocol   # freeze windows / sources (done before any measurement)
scripts/run_loss_alignment.sh b_scweights  # regenerate SC s0 weights, ~9 min, bitwise-checked
scripts/run_loss_alignment.sh b_capture    # BF16 full model (~32 GB RAM), ~70 s + load
scripts/run_loss_alignment.sh b_eval       # partial load, ~2 min
scripts/run_loss_alignment.sh b_report
```
Environment: `~/venvs/moe`, torch 2.11.0+cu128, transformers 4.51.3, A100 MIG 2g.20gb, TF32 off, `OMP_NUM_THREADS=8`. Peak GPU memory: 17.2 GB for eval, 3.1 GB for capture.

## Artifacts
- Code:
  - `moe_calib/loss_alignment.py`
  - `diagnostics/loss_alignment_mvp_a.py`
  - `diagnostics/loss_alignment_mvp_b.py`
  - `tests/test_loss_alignment.py`
  - `scripts/run_loss_alignment.sh`
- `runs_loss_alignment/mvp_a/`:
  - `metrics.csv`, `paired_comparisons.csv`, `results.json`, `reference_weights.npz`, `c4_router_stats.json`
  - `summary.md`, `interpretation.md`
  - `figures/`
- `runs_loss_alignment/mvp_b/`:
  - `protocol.json`, `capture_manifest.json`, `sc_weights.json`, `raw_sums.json`
  - `metrics.json`, `per_layer.csv`, `per_expert.csv`
  - `summary.md`, `interpretation.md`
  - `figures/`
- `~/moe_cache_loss_alignment/`: C4 router stats, BF16 captures (4 × ~200 MB), SC s0 weights.
