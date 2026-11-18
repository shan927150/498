# Code audit: Hessian early-stopping MVP

## Reused production pieces (unchanged)

- **`BlockHessianAccumulator.add`** (`moe_calib/hessian.py`) computes fp32 `X^T X` on the GPU and adds it in place into
  `H[key][block]`. It also updates the `n` / `raw_n` token counters.
- **`H_sum`** sums the K block tensors in index order. **`drift()` / `rel_std_err()`** are not used:
  - they operate on *statistical blocks*;
  - with K = 4 contiguous blocks, relSE would rest on ≤ 4 means;
  - the MVP's drift is defined between chronological prefixes, so it is implemented separately in
    `moe_calib/early_stop.py`.
- **`Profiler`** (`moe_calib/profiler.py`) replays one decoder layer per sequence at batch 1:
  - a forward hook on the router counts routed tokens;
  - pre-hooks on each expert's `gate_proj` (key `gu`) and `down_proj` (key `down`) call `acc.add`.
- **Block assignment, the key point:**
  - E3's *final* quantization calls `Profiler(...).run(cache, chosen)` with `blocks=None`. That gives contiguous blocks
    `(i*K)//n` in the frozen `chosen` order, which **are chronological prefixes**: block 0 = S32, blocks 0–1 = S64 for
    K = 4.
  - The hashed `block_of(seq_id)` blocks (`assign_blocks`) are used only by the E1/E2 characterization, never here.
- **GPTQ and scoring:** `quantize_expert` / `quantize_from_acc_or_fallback` → `gptq_quantize` (sha `58e40d1c…`,
  unchanged). Scoring is `per_seq_rel_errors`, the E3 held-out metric (rows 2128–2255, first 128 dispatch rows per
  sequence).

## New code

- **`moe_calib/early_stop.py`:**
  - prefixes;
  - trace and Frobenius drift on token-normalized Hessians;
  - stopping rules, with stop-budget assignment that uses only the statistics of the current checkpoint;
  - false-safe rate;
  - workload model;
  - frontier interpolation;
  - labelled time projection;
  - classifier.
- **`diagnostics/hessian_early_stop_mvp.py`:**
  - *Experiment 1:* a `Gate` wrapper around `acc.add` that skips the call entirely (fp32 conversion, GEMM, update) for
    masked experts at sequence positions ≥ 64. The forward pass is untouched, and output digests are compared across
    conditions. An optional CUDA-event split of conversion / GEMM / update is included.
  - *Experiment 2:* one production-equivalent replay per layer, split after 16 sequences to snapshot S16 (same blocks,
    same order). The final B = 128 Hessian is therefore the production Hessian, bitwise.
- **`diagnostics/report_hessian_early_stop_mvp.py`:** aggregation, frozen decision, figures.
- **`tests/test_early_stop.py`:** 10 tests.

## Fixes made during the run

The first bench attempt ran out of memory inside the MIG slice. The `Gate` ↔ accumulator reference cycle kept discarded
5.9 GB Hessian buffers alive until garbage collection. The cycle is now broken after every replay, followed by
`gc.collect()` and `empty_cache()`. Timings were not affected, because the repeat ran from scratch. No production file
was modified for this MVP.
