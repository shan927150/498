# AGQ implementation audit (code review before the B = 128 runs)

## Existing GPTQ Hessian path (audited, unchanged when `hessian_weighting="none"`)

`OV.quant_config` → `PQ.quantize_install` → `E3.quantize_and_score` → `Profiler(model, L, K).run(cache, chosen)`

1. **Hooks.** `Profiler` registers a forward hook on the router `mlp.gate`, used for the routing counts only. It also
   registers a forward **pre**-hook on each routed expert's `gate_proj` (key `gu`) and `down_proj` (key `down`).
   - `up_proj` shares `gate_proj`'s input, so it shares H_gu.
   - down_proj's hook sees the true input `act(gate x) ⊙ up x`.
2. **Accumulation.** `BlockHessianAccumulator.add` computes `outer = Xf.T @ Xf` in fp32 on the GPU, then
   `H[key][block] += outer` (times a sequence weight that is always 1.0 here).
   - Sample-count bookkeeping (`n`, `raw_n`) is **not** used by the solver.
   - There is **no 2/N factor**: H is the raw sum, and the K blocks are summed in fixed order (`H_sum`).
3. **Solver.** `quantize_from_acc_or_fallback` → `gptq_quantize(W, H, 4, 128, percdamp=1.0)`:
   - **Dead columns:** columns with diag(H) == 0 are handled first.
   - **Act-order:** `argsort(diag H, desc)`.
   - **Damping:** `H += 1.0 · mean(diag H) · I`.
   - **Inversion:** Cholesky inverse, then lazy-batch updates with blocksize 128.
   - **Fallback:** an expert with zero tokens falls back to RTN (H = I, no act-order). This never happens at B = 128
     (0 fallbacks historically).
   - **Scale invariance:** damping is relative to mean(diag H), and act-order depends only on the diagonal ordering.
     The solver is therefore exactly invariant to a global positive scale of H. This is verified by
     `test_constant_affinity_gptq_invariance`, which is bitwise for c = 0.25.

## New AGQ path (`hessian_weighting="router_affinity"`)

- **Router hook.** `Profiler._gate_hook` additionally keeps `vals, sel = topk(softmax(logits, fp32), 4)`. This is
  the model's own op on the same tensor.
  - `_record_rows` builds, per expert, `idx, top_x = where(one_hot(sel).permute(2,1,0)[e])` and `c = vals[top_x, idx]`.
  - This is the model's exact dispatch order, which is slot-major.
- **Expert hooks.** The expert pre-hooks call `_check_rows`, which raises if the number of activation rows differs from
  the number of routed (token, slot) pairs.
  - With `verify_alignment=True` it also requires `X == router_input[top_x]` bitwise.
  - They then call `acc.add(..., row_weight=c)`, which forms `(√c ⊙ X)ᵀ(√c ⊙ X)`. That is weight c, **not c²**.
  - The same c is used for gate/up (H_gu) and down (H_down).
- **Token counts.** These are unchanged, and no normalization is added (see `paper_fidelity.md` §4).
- **Solver.** It is byte-identical: the `moe_calib/gptq.py` sha256 is `58e40d1c…`, the same as in all prior studies.
  The damping, act-order, group size, bits and blocksize are untouched.
- **Regression control (`"unit_affinity"`).** It runs the identical weighted code path with c forced to 1.
  `X·√1` is exact in fp32, so the Hessians must be bitwise standard.

## Changed lines in existing files (all additive, default-preserving)

| file | change |
|---|---|
| `moe_calib/hessian.py` | `add(..., row_weight=None)`: optional √c row scaling with validation. The `None` branch is the original code. |
| `moe_calib/profiler.py` | ctor kwargs `hessian_weighting="none"`, `collect_affinity`, `verify_alignment`; `_record_rows`, `_check_rows`. The default gate hook still calls `topk(...).indices` exactly as before. |
| `e3_allocation.py` | `quantize_and_score(..., hessian_weighting="none")`, forwarded to the calibration Profiler only. With the default the call is `Profiler(model, L, K=K)` exactly as before. |
| `moe_calib/partial_model_quant.py` | `quantize_install(..., hessian_weighting="none")` pass-through. |
| `diagnostics/official_ebss_validation.py` | `quant_config(..., hessian_weighting="none", weights_root=None)` pass-through and weights directory. |

New files: `moe_calib/agq.py`, `diagnostics/agq_only_validation.py`, `diagnostics/report_agq_only_validation.py`,
`scripts/run_agq_only_validation.sh`, `tests/test_agq.py`.

## Protocol identity

- **Calibration.** Both arms use the same CANONICAL-B1 cache (`~/moe_cache_ebss_official_validation`, sha-verified)
  and the same 128 rows per seed (`calibration_manifest.json` equals the historical `cache_rows`). They also share the
  same K blocks, layers [1, 8, 16, 23], held-out rows 2128–2255 of `~/moe_cache`, and WikiText-2 windows (sha
  `e646cf7c…`).
- **GPTQ arm.** It uses the historical FR results, admissible because `unit_regression` reproduces FR_s0 bitwise with
  the modified code.
- **Cost comparison.** It uses the fresh `GPTQ_rerun_FR_s0` run made in the same session.

## Resource feasibility

- **Historical FR_s0 run:** 350 s quantize + 2 × 41 s PPL. Peak GPU 8.4 GB on the 20 GB MIG slice, peak CPU about
  43 GB.
- **AGQ additions:** about 36 bytes per token-layer of sparse metadata (4 × fp32 c + 4 × int32 token index +
  4 × int8 slot) and one `where` per
  (expert, sequence). No new forward passes and no new storage of activations.
