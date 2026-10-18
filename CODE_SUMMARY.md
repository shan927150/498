# Code Summary and Proposal Mapping

**Project:** *Calibration-Sufficiency-Aware Budget Allocation for MoE Post-Training Quantization* (CS 498)
**Sources reviewed:** `cs498_proposal_updated.pdf`, all `.py` / `.sh` files in the repo.

Legend for the **Maps to proposal?** column:

| Mark | Meaning |
|---|---|
| ✅ Direct | Implements something the proposal names explicitly |
| 🟡 Partial | Implements the proposal's idea, but narrower, broader, or in a different form |
| ➕ Extension | Not in the proposal; added on top of it (often from instructor feedback or cited papers) |
| ⚙️ Infra | Supporting plumbing the proposal assumes but does not describe |

---

## 1. Proposal components → code at a glance

| Proposal element (§) | Where in code | Status |
|---|---|---|
| Candidate pool **D**, select **C ⊆ D**, \|C\| = B (§3) | `data_utils.get_splits`, `cache_inputs.py`, `e3_allocation.py` (`--B`) | ✅ Direct |
| **Routing & statistics profiler** (§3) | `moe_calib/profiler.py`, `moe_calib/hessian.py` | ✅ Direct (stores full Hessians, so the "compact summaries" claim is only partly met) |
| Token-normalized drift H̄ₑ = Hₑ / nₑ (§3) | `BlockHessianAccumulator.normalized_block_means`, `.drift` | ✅ Direct |
| **Calibration sufficiency estimator** → remaining-need score uₑ (§3) | `e3.needs_from_stats` (relSE → need) | 🟡 Partial: uses relSE only. Drift is computed but not used for allocation |
| Offline characterization with held-out GPTQ reconstruction improvement (§3, §4) | `e1_label_sufficiency.py`, `e2_signals.py` | ✅ Direct |
| **Adaptive calibration controller** (initial subset → score → select batch → refresh) (§3) | `e3_allocation.py` Phase B (`tlp`) | ✅ Direct (selection is done with an LP, which the proposal does not specify) |
| Natural routing retained (§3) | Profiler hooks on actual router/expert inputs | ✅ Direct |
| Standard GPTQ, unchanged solver (§3) | `moe_calib/gptq.py` | ✅ Direct |
| Profiling and selection overhead in the evaluation (§3, §4) | `cost` dict in `e3_allocation.py` | ✅ Direct |
| Qwen1.5-MoE-A2.7B, 4-bit weight-only, single GPU (§4) | `model_utils.py`, `--bits 4` | ✅ Direct (3-bit is also run) |
| Calibration corpus C4 / WikiText-2 (§4) | C4 for calibration, WikiText-2 for PPL | ✅ Direct |
| Correlations, count-matched comparisons (§4) | `stats.spearman`, `binned_spearman`, `incremental_r2` | ✅ Direct |
| Expert error curves across budgets (§4) | E1 n0 ∈ {32, 64, 128} → 2·n0 only | 🟡 Partial: only doubling points, no curves up to 512 |
| Original weights restored before each quantization run (§4) | `quantize_expert` never mutates. E3 restores after PPL | ✅ Direct |
| Strategies: Random / Coverage-aware / Sufficiency-aware (§4) | `--strategy random / coverage / tlp` | ✅ Direct |
| Metrics: held-out recon error (mean + tail), PPL (§4) | `recon_mean/p90/p99`, `wikitext2_ppl` | ✅ Direct (PPL uses 40 windows by default as a proxy) |
| Budgets 64, 128, 256, 512 (§4) | `run_all.sh` runs only **B = 128** | 🟡 Partial: the code supports other budgets, but the script does not run them |
| Multiple seeds (§4) | seeds 0, 1, 2 in `run_all.sh` | ✅ Direct |
| Signal ablations (§4) | E2 compares 5 signals. E3 has no switch to choose the signal | 🟡 Partial |
| Secondary metrics: GPU time, **peak memory**, overhead (§4) | timings yes. Peak memory **not recorded** | 🟡 Partial |
| Separate calibration / **tuning** / evaluation data (§4) | pool vs. holdout only. No tuning split (e.g. for `--eps`) | 🟡 Partial |
| Figure 1 (routed tokens vs. drift, 256→512) | No plotting code | ❌ Missing from the repo |
| Global early stopping (optional, §3) | — | Not implemented (proposal marks it optional) |
| Additional model or corpus (stretch, §4) | `MOE_MODEL` env var only | Not exercised |

---

## 2. Per-file, per-function summary

### `moe_calib/model_utils.py`: model loading and layer-wise offload ⚙️ Infra

| Function / class | What it does | Maps to proposal? |
|---|---|---|
| `load_model` | Loads Qwen1.5-MoE-A2.7B in bf16 on **CPU**. Moves only embed, final norm, rotary and lm_head to the GPU. | ✅ Direct (the proposal's model, single-GPU). The CPU offload is infra |
| `num_layers`, `moe_block`, `num_experts`, `top_k`, `expert_dims` | Structural accessors (60 experts, top-4, d=2048 / 1408). | ⚙️ Infra |
| `expert_module`, `get_expert_weights`, `set_expert_weights` | Read and write one expert's gate/up/down weights. | ⚙️ Infra (used for PPL write-back and restore) |
| `_Caught`, `_Catcher` | Temporarily replaces layer 0 so it can capture the exact kwargs (mask, rotary) the model passes to decoder layers. | ⚙️ Infra |
| `layer_kwargs` | Runs a dummy forward with the catcher and caches the kwargs per seqlen. | ⚙️ Infra |
| `layer_on_gpu` | Context manager that moves one decoder layer to the GPU and back. | ⚙️ Infra (needed for the "single GPU" constraint on a 20 GB MIG slice) |
| `run_layer` | Runs one decoder layer on hidden states. | ⚙️ Infra |
| `embed` | Token IDs → embeddings on the GPU. | ⚙️ Infra |

### `moe_calib/data_utils.py`: data and cache ⚙️ Infra / ✅

| Function | What it does | Maps to proposal? |
|---|---|---|
| `get_splits` | Shuffles one C4 shard with `data_seed` and takes one window per document. The first P windows form the **candidate pool D** and the next H form the **held-out set** (disjoint). | ✅ Direct (pool D and separate evaluation data) |
| `load_meta` | Reads `meta.json` (sizes, layers, measured forward cost per sequence). | ⚙️ Infra |
| `load_layer_cache` | Memory-maps a cached layer-input tensor `L{l}.pt`. | ⚙️ Infra |
| `pool_and_holdout_ids` | Maps the pool and holdout to global row indices. | ⚙️ Infra |
| `wikitext2_ppl` | Layer-wise WikiText-2 perplexity (GPTQ protocol, non-overlapping 2048 windows) using the current (possibly quantized) weights. | ✅ Direct ("model-level perplexity") |

### `moe_calib/hessian.py`: `BlockHessianAccumulator` ✅ (part of the profiler)

| Method | What it does | Maps to proposal? |
|---|---|---|
| `__init__` | Allocates K Hessian blocks [K, d, d] per (layer, expert, proj ∈ {gu, down}), plus weighted and raw token counts. | ✅ "Hessian summaries", "routed-token counts". 🟡 Stores full d×d Hessians, which is not "compact" |
| `add` | Adds w·XᵀX for the activations routed to an expert into a given block and updates the counts. | ✅ "Statistics use activations actually routed to each expert" |
| `H_sum` | Sums a subset of blocks, e.g. H(n0) and H(2n0) with no new forward. | ✅ Supports characterization ➕ (partial-sum reuse, Canévet et al.) |
| `tokens` | Raw routed-token count. | ✅ nₑ (the exposure baseline) |
| `normalized_block_means` | H̄ₖ = Hₖ / nₖ per block. | ✅ **Direct**: the proposal's H̄ₑ = Hₑ / nₑ |
| `rel_std_err` | Between-block relative standard error of H̄ (a t-statistic / CI-width style signal). | ➕ Extension (the code calls it "the instructor's suggestion"). It is the signal E3 actually allocates on |
| `drift` | ‖H̄(first half) − H̄(all)‖ / ‖H̄(all)‖ within the given blocks. | ✅ Direct: the proposal's drift signal. 🟡 This is within-draw prefix drift, not the 256→512 cumulative drift shown in Fig. 1 |

### `moe_calib/profiler.py`: `Profiler` ✅ (proposal §3, component 1)

| Method | What it does | Maps to proposal? |
|---|---|---|
| `__init__` | Sets up the accumulator (optional), holdout capture (optional), per-sequence routing counts and router-uncertainty sums. | ✅ Routing and statistics profiler |
| `_gate_hook` | Router forward hook. Records per-sequence routed counts c₍s,e₎, plus the top-k margin and entropy per expert. | ✅ "records routing information for candidate sequences". ➕ margin/entropy are extra signals |
| `_expert_hook` | Pre-hook on gate_proj/down_proj that feeds routed activations to the Hessian accumulator and optionally stores held-out activations per sequence. | ✅ Direct |
| `attach` / `detach` | Register and remove hooks (`routing_only` skips the expert hooks). | ⚙️ Infra |
| `run` | Replays one layer on cached inputs, one sequence at a time, with block and weight assignment per sequence. | ✅ "incrementally tracks … for the selected data" |
| `free` | Releases the GPU Hessians. | ⚙️ Infra |
| `router_signals` | Mean top-k margin and mean entropy for an expert. | ➕ Extension (candidate signals not named in the proposal) |
| `usage_freq` | Expert's share of routed tokens. | ➕ Used as the LP weight pₑ |

### `moe_calib/gptq.py`: standard GPTQ ✅

| Function | What it does | Maps to proposal? |
|---|---|---|
| `_qparams` | Asymmetric per-group min/max scale and zero point. | ✅ Part of standard GPTQ |
| `gptq_quantize` | GPTQ Algorithm 1: 1% damping, act-order, Cholesky of H⁻¹, lazy-batch error feedback, group size 128. Returns dequantized weights. | ✅ **Direct**: "retain the standard GPTQ solver" |
| `quantize_expert` | Quantizes gate/up with H_gu and down with H_down. Never mutates the model. | ✅ Direct ("original weights restored before each independent run") |

### `moe_calib/quant_eval.py`: quantize and score ✅

| Function | What it does | Maps to proposal? |
|---|---|---|
| `fp_weights` | Fetches an expert's FP32 weights. | ⚙️ Infra |
| `quantize_from_H` | GPTQ from given Hessians. | ✅ Direct |
| `quantize_from_acc` | GPTQ from a subset of accumulator blocks. | ✅ Direct |
| `per_seq_rel_errors` | For each held-out sequence: Σ‖X(W−W_q)ᵀ‖² / Σ‖XWᵀ‖² (the relative GPTQ objective). | ✅ **Direct**: "held-out expert reconstruction error" on "fixed held-out activations" |

### `moe_calib/stats.py`: statistics ✅ / ➕

| Function | What it does | Maps to proposal? |
|---|---|---|
| `paired_t_one_sided` | One-sided paired t-test that mean(err_n0 − err_2n0) > 0. | ➕ Extension: turns "marginal improvement" into a significance label (Korattikara et al.) |
| `bh_fdr` | Benjamini–Hochberg q-values. | ➕ Extension (multiple-testing control) |
| `auroc` | Mann–Whitney AUROC of a signal vs. the significance label. | ✅ Supports "how well signals predict future reconstruction improvement" |
| `spearman` | Rank correlation. | ✅ Direct ("Correlations") |
| `binned_spearman` | Mean Spearman within routed-count quintiles. | ✅ **Direct**: "count-matched comparisons" |
| `_ols_r2` | OLS R² helper. | ⚙️ Infra |
| `incremental_r2` | R² gain from adding the signal on top of log(count). | ✅ **Direct**: "predictive information beyond routing exposure" |

### `cache_inputs.py`: step 0 ⚙️ Infra

| Function | What it does | Maps to proposal? |
|---|---|---|
| `main` | Draws the pool (2000) and holdout (256) at seqlen 512, runs the model layer by layer, saves the **input hidden states** of the profiled layers (default 1, 8, 16, 23), and records `forward_s_per_seq` for cost accounting. | ⚙️ Infra that lets the single-GPU setup work. 🟡 Profiles only 4 of 24 layers |

### `e1_label_sufficiency.py`: E1, ground-truth sufficiency labels ✅

| Function | What it does | Maps to proposal? |
|---|---|---|
| `main` | For n0 ∈ {32, 64, 128}: a fixed base draw plus R=3 disjoint extensions. For each expert with at least `min_tokens` tokens it runs GPTQ(H(n0)) and GPTQ(H(2n0)), computes per-held-out-sequence errors, runs a paired t-test, and applies BH-FDR within each (n0, rep). Writes `e1_labels_b{bits}.csv`. Resumable via JSONL. | ✅ **Direct**: "offline characterization … using held-out GPTQ reconstruction improvement", "define sufficiency through marginal improvement". 🟡 Budgets are smaller than in the proposal (32–256 vs. 64–512) |

### `e2_signals.py`: E2, do n0 signals predict E1? ✅

| Function | What it does | Maps to proposal? |
|---|---|---|
| `main` | Reprofiles each layer on the same base draw as E1. Computes 5 signals: count, drift, relSE, −margin, entropy. Scores each against E1's `rel_delta` and significance with AUROC, Spearman, binned Spearman and incremental R². Writes the report JSON and signal CSV. | ✅ **Direct**: §4 "Characterization" ("compare how well routing counts and statistical signals predict … signals use only data available at the starting budget"). It also serves as the "validate the signal" step from §3 |

### `e3_allocation.py`: E3, fixed-budget allocation ✅ (core method)

| Function | What it does | Maps to proposal? |
|---|---|---|
| `routing_profile` | Phase A: dense routing matrix C[s, expert] over the whole pool and usage pₑ. | ✅ "combines the scores uₑ with **candidate routing profiles**" |
| `expert_stats` | Per-expert token count and relSE over the currently chosen set. Blocks are assigned randomly (round-robin) so selection order does not inflate the variance. | ✅ "After each selected batch, the profiler updates expert statistics" |
| `needs_from_stats` | need = max(0, n·(z·relSE/ε)² − n). Experts never seen get a large constant (`RARE_NEED`). | ✅ **The remaining-need score uₑ**. 🟡 It is hard-wired to relSE. The proposal says the score is chosen from validated signals, and nothing here reads the E2 result |
| `solve_lp` | LP: maximize Σ pₑ·yₑ subject to yₑ ≤ coverage by selected sequences, yₑ ≤ needₑ, Σz = batch. | 🟡 Implements "prioritize unselected sequences that activate experts with greater remaining needs". The LP form and the usage weighting pₑ are an extension (Neyman-style) |
| `round_selection` | Top-z rounding, or randomized rounding with Horvitz–Thompson weights 1/π clipped at p99 (`--unbiased`). | 🟡 Top-z is direct. ➕ HT reweighting is an extension (it changes Hessian weights but not the GPTQ solver) |
| `select_coverage` | Greedy on Σₑ c₍s,e₎ / (nₑ + λ) after a random B/4 warm start. Uses counts only. | ✅ **Direct**: the "Coverage-aware" baseline, explicitly not MoEQuant/EAQuant, as the proposal says |
| `oracle_need` | Turns E1's mean `rel_delta` into a fixed need vector. | ➕ Extension (an upper-bound reference strategy) |
| `main` | Phase A routing profile → Phase B selection (`random`, `random_plus`, `coverage`, `tlp`, `oracle`), with `tlp`/`oracle` using a B/4 warm start plus `--rounds` LP rounds → Phase C per-layer Hessians over the chosen set, standard GPTQ, held-out mean/p90/p99 error, usage-weighted error, optional write-back and WikiText-2 PPL, and a cost breakdown including `equiv_extra_random_seqs`. Restores the original weights after PPL. | ✅ **Direct**: §3 controller loop and §4 fixed-budget comparison (random / coverage / sufficiency), tail errors, PPL, overhead. ➕ `random_plus` (equal-compute baseline) and `oracle` are extensions |

### Shell scripts

| Script | What it does | Maps to proposal? |
|---|---|---|
| `scripts/setup_icrn.sh` | Checks torch/CUDA, installs pinned requirements, reports the size of the GPU slice. | ⚙️ Infra |
| `scripts/run_smoke.sh` | Tiny end-to-end run (1 layer, pool 64, B=16). | ⚙️ Infra |
| `scripts/run_all.sh` | Cache → E1 + E2 for 3 and 4 bits → E3 for {random, coverage, tlp, oracle} × seeds {0, 1, 2} at **B=128**. | 🟡 Partial: matches the §4 protocol except that only one budget is run (proposal: 64/128/256/512) and there are no signal ablations in E3 |

---

## 3. Gaps and deviations to address or state in the report

1. **E2 → E3 gate isn't wired.** The proposal says the pipeline uses a *validated* signal. `tlp` always uses relSE, and drift, margin and entropy can't be selected in E3. Adding a `--signal` flag would cover both this and the proposal's "signal ablations".
2. **Budget sweep.** `run_all.sh` only runs B = 128. The proposal lists 64, 128, 256 and 512, and E1's n0 grid (32, 64, 128) stops below the proposal's 256→512 drift setting.
3. **Partial model coverage.** Only 4 of 24 layers are profiled, and only routed-expert weights are quantized (attention and shared expert stay bf16). So PPL measures a *partially* quantized model, which is not what "standard GPTQ quantizes the model" implies. The README already lists this as a simplification.
4. **Non-sequential GPTQ.** Each layer is quantized from full-precision inputs, which differs from standard sequential GPTQ. This is documented in the README.
5. **Peak memory isn't measured.** The proposal lists it as a secondary metric. Adding `torch.cuda.max_memory_allocated()` to `cost` would close this.
6. **No tuning split.** The proposal asks for separate calibration, tuning and evaluation data. `--eps`, `--rounds` and `λ` have no held-out tuning set.
7. **"Compact summaries".** The profiler stores full K×d×d Hessians per expert (about 1.5 GB per block per layer), not compact summaries. It works because only one layer is processed at a time.
8. **Figure 1 isn't reproducible from the repo.** There is no script for the routed-tokens vs. 256→512 drift plot.
9. **Extensions beyond the proposal** (worth calling out as contributions): relSE / t-statistic signal, router margin and entropy signals, paired t-test with BH-FDR labels, LP-based selection with usage weighting, Horvitz–Thompson unbiased reweighting, the `oracle` upper bound, and the `random_plus` equal-compute baseline with `equiv_extra_random_seqs`.

## 4. Bottom line

The repository maps closely onto the proposal's three components: the profiler (`profiler.py` + `hessian.py`), the sufficiency estimator (`needs_from_stats`, validated offline by E1 + E2), and the adaptive controller (`e3_allocation.py`, `tlp`). It keeps the standard GPTQ solver and implements all three comparison strategies the proposal names. The main gaps are in evaluation coverage: a single budget, no signal choice in E3, no peak-memory metric, no tuning split, and a partially quantized model. The core method itself is implemented.
