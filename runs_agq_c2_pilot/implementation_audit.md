# c² pilot: implementation audit

## Change

`moe_calib/profiler.py` gains one constructor argument, `affinity_power=1`. It is valid only with
`hessian_weighting="router_affinity"`, and only the values {1, 2} are accepted. q = 0 means standard GPTQ and is
requested with `hessian_weighting="none"`.

In the expert pre-hook the change is:

```python
c = self._cur_rows[e][2]                     # validated fp32 top-4 router softmax, dispatch order
if self.hessian_weighting == "unit_affinity": c = torch.ones_like(c)
elif self.affinity_power != 1: c = c.pow(self.affinity_power)    # NEW: row weight c^q
self.acc.add(key, X, block, w, row_weight=c)  # unchanged: forms (sqrt(w) X)^T (sqrt(w) X)
```

- **q = 2:** the row weight is c², so the rows are scaled by √(c²) = c. This gives H = Xᵀ diag(c²) X, not c⁴. It is
  checked against a float64 oracle; the distance to the c⁴ and c¹ Hessians is far above tolerance.
- **q = 1:** the existing AGQ code path, byte for byte. `pow` is not called.
- **Defaults:** all unchanged. Routing, top-4 selection, the absence of renormalization, the solver (`gptq.py` sha
  `58e40d1c…`), damping, act-order, group size and bits are untouched.
- **Normalization:** none added. Relative damping cancels a global scale. Realistic c² values (about 1e-3 to 3e-2) leave
  diag(H) strictly positive. Dead-column counts are identical for q = 0 and q = 2 on all 36 projections and in the
  real-data oracle.

New helpers in `moe_calib/agq.py`:
- `effective_sample_size_q`
- `weighted_hessian_reference_q`
- `slot_contributions` / `decompose`: per token, the 4 slot vectors c·ΔE. The cross term is computed pairwise and
  independently, then checked against the total.
- `classify_c2`

Runner: `diagnostics/agq_c2_pilot.py`. Tests: `tests/test_agq_c2.py` (26 tests).

## Gates

| gate | result |
|---|---|
| pytest AGQ + c² (60 tests) | pass |
| real-data oracle, layer 8, 4 calibration rows, all 60 experts × {gu, down}, q = 0/1/2 | worst relative error 2.0e-7; dead columns unchanged; symmetric |
| q = 0 reproduces historical FR_s0 GPTQ weights (36 projections) | bitwise |
| q = 1 reproduces historical AGQ_FR_s0 weights (36 projections) | bitwise |
| q = 2 finite, no new dead columns | yes |
| D + C reconstructs the total ‖Δy‖² | relative residual < 1e-6 in every condition |
| condition A equals the earlier full-layer GPTQ evaluation | L8 7.173198e-3 and L16 7.090592e-3, identical |
| pristine weights restored after evaluation | layer digest check |
| full repository test suite | `logs/pytest_all.log` |

## Same inputs in every arm

- The same 128 FR s0 rows from the CANONICAL-B1 cache, with the same K blocks.
- The same router coefficients, recomputed by the model's own op in the same Profiler replay; raw token counts are
  equal across q.
- The same pristine BF16 weights and solver call.
- The same held-out rows 2128–2255 under frozen BF16 routing.
