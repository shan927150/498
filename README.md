# MoE Calibration Sufficiency (CS 498 project)

Experiments E1–E3 for *Calibration-Sufficiency-Aware Budget Allocation for MoE
Post-Training Quantization*: **Qwen1.5-MoE-A2.7B, weight-only GPTQ (3/4-bit)**,
sized to run on an **ICRN A100 20 GB MIG slice (12 CPU / 72 GB RAM)**.

## Why the code is layer-wise

ICRN sessions are MIG *slices* (A100 20 GB or H200 16 GB VRAM), and the bf16
model alone is ~28.6 GB. So:

* the model stays in CPU RAM — **pick the A100 option**: the H200 option has
  only 32 GB RAM, too little to hold the model;
* `cache_inputs.py` runs the full model **once**, one decoder layer on the
  GPU at a time, and saves each profiled layer's *input* hidden states to disk;
* E1/E2/E3 only replay **single layers** on those cached inputs (layers are
  independent because characterization uses non-sequential GPTQ), so peak
  GPU use is ~1.2 GB (layer) + ~1.5 GB per Hessian block (60 experts).

## Layout

```
moe_calib/
  model_utils.py  load on CPU, layer_on_gpu(), run_layer() (kwargs captured
                  from the model itself, so masks/rotary are version-robust)
  data_utils.py   C4 pool/holdout (one fixed draw), cache I/O (mmap), layer-wise PPL
  hessian.py      K-block Hessian accumulator (partial sums -> H(n0), H(2n0), relSE, drift)
  profiler.py     single-layer replay with router/expert hooks
  gptq.py         standard GPTQ, lazy-batch form (identical output to column-wise)
  quant_eval.py   quantize from Hessians, per-held-out-sequence relative error
  stats.py        paired t-test, BH-FDR, AUROC, Spearman, incremental R^2
cache_inputs.py           step 0: build activation cache (once)
e1_label_sufficiency.py   E1: t-test sufficiency labels (resumable)
e2_signals.py             E2: do n0-signals predict E1 beyond token count?
e3_allocation.py          E3: fixed-budget allocation (random/coverage/tlp/oracle)
scripts/setup_icrn.sh     one-time environment setup
scripts/run_smoke.sh      ~30 min end-to-end check on a tiny setting
scripts/run_all.sh        full mini-scale pipeline
```

## Running on ICRN

```bash
git clone https://github.com/shan927150/498.git && cd 498
bash scripts/setup_icrn.sh                 # installs pinned transformers etc.
bash scripts/run_smoke.sh 2>&1 | tee logs/smoke.log
nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
tail -f logs/run_all.log
```

ICRN limits to respect: sessions end after **24 h**; processes may be killed
if the browser is disconnected for **> 1 h** (keep the tab open). E1 appends
each finished (layer, n0, rep) to `runs/e1_raw_b*.jsonl` and resumes from it.

Disk (100 GB quota): model ~29 GB (HF cache) + activation cache ~4.6 GB per
profiled layer (default 4 layers, ~18.5 GB).

## Outputs

| file | content |
|---|---|
| `runs/e1_draws.json` | the base/extension sequence ids (shared by E1/E2/E3, all bit-widths) |
| `runs/e1_labels_b{3,4}.csv` | per (layer, expert, n0, rep): t, p, BH q, delta, rel_delta, significant |
| `runs/e2_report_b{3,4}.json` | per n0 and signal: AUROC, Spearman, binned Spearman, incremental R^2 |
| `runs/e3_<strategy>_B<B>_b<bits>_s<seed>.json` | recon mean/p90/p99, usage-weighted recon, PPL, cost breakdown, chosen ids |

### Total cost (instructor's question)

Each E3 json has `cost.equiv_extra_random_seqs`: the number of additional
*random* calibration sequences the selection overhead (full-forward routing
profile of the pool + selection compute) would have paid for. The fair
baseline is

```bash
python e3_allocation.py --strategy random_plus --delta-seqs <that number> ...
```

If this number is large relative to B, the method only wins on equal compute
when routing information is cheap — itself an important finding to report.

## Design provenance

| Paper | Where | What was taken |
|---|---|---|
| Frantar et al., *GPTQ*, ICLR 2023 | `gptq.py`, metric in `quant_eval.py` | solver unchanged (1% damping, act-order, Cholesky error feedback, lazy batches); metric = GPTQ's objective on held-out inputs |
| Canévet, Jose & Fleuret, *Importance Sampling Tree for Large-scale Empirical Expectation*, ICML 2016 | `hessian.py`; `e3 --unbiased` | (1) an empirical expectation (the Hessian) may be acquired by prioritized non-uniform sampling and stay unbiased under importance weights → randomized rounding + Horvitz–Thompson weights; (2) partial sums allow re-estimation on subsets without new passes → K-block accumulator. Their MCTS tree is not needed at a 2k pool; it is the scaling path for 100k+ pools. |
| Korattikara, Chen & Welling, *Austerity in MCMC Land*, ICML 2014 | `stats.paired_t_one_sided`, `e3.needs_from_stats` | a t-test on data seen so far decides "enough data"; inverted into a token-need |
| Zhao & Zhang, ICML 2015; Neyman allocation | LP objective in `e3.solve_lp` | effort ∝ usage p_e × variability σ_e |
| Katharopoulos & Fleuret, *Not All Samples Are Created Equal*, ICML 2018 | `e2_signals.py` | estimate whether non-uniform selection will pay off before running it (E2 gates E3) |
| Williams & Aletras, ACL 2024 | E1 `--reps`, E3 seeds | calibration-set sensitivity must be measured |
| Benjamini & Hochberg, 1995 | `stats.bh_fdr` | FDR control over ~240 hypotheses per (n0, rep) |

## Simplifications to state in the report

1. Non-sequential GPTQ (every layer quantized from full-precision inputs).
2. Only routed-expert weights are quantized (attention, shared expert stay
   bf16) — PPL is comparable across strategies, not to full-model GPTQ papers.
3. The sequence is the sampling unit everywhere (blocks, t-test, selection).
4. `--unbiased` clips HT weights at the 99th percentile.
5. Candidate pool and holdout are fixed by `--data-seed`; seeds vary selection only.
6. E2/E3 use `rel_delta` (effect relative to the n0 error) so experts with
   different error scales are comparable.
