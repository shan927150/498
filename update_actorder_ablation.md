# GPTQ act-order ablation after frozen-routing calibration

**Setup:** Qwen1.5-MoE-A2.7B. Routed experts of layers 1/8/16/23 are GPTQ 4-bit (g128, percdamp 1.0), B = 128, allocation seed 0, strategies FR s0 / FC s0, groupings G20261111–15. Everything else stays BF16.
**Runner:** `scripts/run_actorder_ablation.sh`
**Artifacts:** `runs_actorder_ablation/`
**Correctness basis:** `update_installation_audit.md`.

## Question

Under frozen routing (AO), GPTQ still changes about 39 % of the effective weights relative to OO. Are the Hessian-dependent act-order permutation and the quantization-group membership it induces responsible for a large part of that residual weight and PPL sensitivity?

## Conditions

| | Hessian | act-order | source |
|---|---|---|---|
| OO | H_O | natural, from H_O (= P_O) | frozen reference |
| AO | H_AO,g (X_g, frozen R_O) | natural, from H_AO,g | frozen-routing robustness study (10 configs, reused) |
| **AOF** | H_AO,g (bitwise the AO Hessian) | **fixed to P_O** | new (10 configs) |

The solver groups permuted columns in blocks of 128, so fixing the permutation also fixes the column-to-group membership. **AOF therefore tests act-order and group membership jointly.**

Everything else is recomputed from H_AO,g:
- damping and the Cholesky factor;
- per-group scales and zero-points;
- error feedback.

## Implementation

**`moe_calib/gptq.py`:**
- `gptq_quantize(..., actorder_override=None, info=None)`. The override replaces `argsort(diag H)` at exactly that point. It is validated as a complete int64 permutation of the right length, and it requires act-order.
- `info` is a read-only capture of the applied permutation, the scale and zero fitted at each group start, and the per-column weight-space rounding error before feedback.
- `quantize_expert` gains `perms` / `infos`.

**`moe_calib/quant_eval.py`:** the call chain passes these through. A call that supplies neither runs the original code path.

**Regressions, all exact:**

| | result |
|---|---|
| default path vs the solver at git HEAD (synthetic, CPU) | bitwise |
| A — OO re-run through the modified solver, FR and FC | installed weights == frozen OO, reconstruction (mean and per expert) == frozen, Hessians 480/480, captured permutations == stored act-order 240/240 |
| B — natural permutation supplied as override | bitwise for all 240 experts (FR OO, FC OO, FR AO G20261111); AO default path == stored AO weights and reconstruction |
| C — synthetic | inverse permutation; override == explicitly permuted problem without act-order (bitwise); group grid membership; scales refitted, not copied; H never mutated |
| D — inside every AOF config | permutation == P_O 240/240; Hessians == AO 480/480; routed-token counts == R_O; AO default path on the same Hessian reproduces the stored AO weights for the 8 check experts; AOF weights ≠ OO |

**Integrity:** every AOF config passed the precision-aware integrity protocol (I1–I4, frozen before any AOF run). It also passed the frozen old tolerance check, at a maximum of 0.91×.

**Tests:** `tests/test_gptq_actorder_ablation.py` (23 tests); full suite 303 passed.

## Results

### Mechanism

- **Hessians:** AOF's are bitwise AO's in every config (480/480). The median trace-normalized difference from OO is 0.0038–0.0039 for FR and 0.0049–0.0050 for FC.
- **Act-order:**
  - AO moves 70 % of permutation positions relative to P_O, but only 2.6 % (FR) / 2.8 % (FC) of columns change quantization group. Most act-order changes are reorderings *within* a group.
  - AOF has 0 positions and 0 group changes by construction, verified.

**Effective weights vs OO** (720 tensors, mean over 5 groupings; every grouping lies within ±0.3 percentage points / ±0.0002 rel L2 of the mean):

| | elements ≠ OO, AO | elements ≠ OO, AOF | rel L2, AO | rel L2, AOF | WeightReduction (1 − D_AOF/D_AO) |
|---|---|---|---|---|---|
| FR | 39.2 % | 16.9 % | 0.0694 | 0.0405 | **0.416** (0.415–0.417) |
| FC | 40.2 % | 17.9 % | 0.0711 | 0.0419 | **0.410** (0.409–0.411) |

AOF and AO differ from each other in 38–40 % of elements.

**Quantizer parameters** (8 check experts × 3 projections, read-only capture):

| comparison | median scale rel. diff, FR | median scale rel. diff, FC | zero-points differing, FR | zero-points differing, FC |
|---|---|---|---|---|
| AO vs OO | 0.029–0.031 | 0.033–0.034 | 6.0–6.4 % | 6.6–7.1 % |
| AOF vs OO | 0.0024 | 0.0026 | 0.8 % | 0.9 % |

- Fixing the order nearly fixes the quantization grid.
- The total weight-space rounding error per column is unchanged (AOF/OO median ratio 1.000).
- The remaining AOF differences are individual rounding decisions moved by the alternative error feedback, on an almost unchanged grid.

**Reconstruction:**
- AOF is within ±0.0092 % of AO and ±0.0082 % of OO in every config.
- The FC penalty over FR is +2.55–2.56 % under AOF, +2.55 % under AO and +2.549 % under OO.

### PPL

| grouping | FR AO | FR AOF | FC AO | FC AOF | Δ_AO | Δ_AOF |
|---|---|---|---|---|---|---|
| OO | 9.267851 | | 9.277908 | | +0.010057 | |
| G20261111 | 9.265861 | 9.266665 | 9.277881 | 9.272543 | +0.012020 | +0.005878 |
| G20261112 | 9.270916 | 9.268927 | 9.274432 | 9.277243 | +0.003516 | +0.008316 |
| G20261113 | 9.276043 | 9.265309 | 9.273312 | 9.271201 | −0.002731 | +0.005892 |
| G20261114 | 9.273279 | 9.264406 | 9.274137 | 9.273505 | +0.000858 | +0.009099 |
| G20261115 | 9.268226 | 9.270099 | 9.274729 | 9.275548 | +0.006503 | +0.005449 |

| quantity | AO | AOF | AOF/AO |
|---|---|---|---|
| FR SD (ddof 1) | 0.00402 | 0.00240 | **0.60** |
| FR range | 0.0102 | 0.0057 | 0.56 |
| FC SD | 0.00175 | 0.00240 | **1.37** |
| FC range | 0.0046 | 0.0060 | 1.32 |
| FR mean \|PPL − OO\| | 0.0038 | 0.0021 | AOF closer in 4/5 |
| FC mean \|PPL − OO\| | 0.0030 | 0.0039 | AOF closer in 2/5 |
| FR mean PPL | 9.27087 | 9.26708 | −0.0038 |
| FC mean PPL | 9.27490 | 9.27401 | −0.0009 |
| Δ (FC − FR) SD | 0.00561 | 0.00166 | **0.30** |
| Δ range | 0.0148 | 0.0036 | 0.25 |
| mean Δ | +0.0040 | +0.0069 | (Δ_OO +0.0101) |
| FR/FC ranking preserved vs OO | 4/5 | **5/5** | (natural AA: 2/5) |

**How the spreads moved:**
- Under AOF, both strategies have nearly the same grouping SD (0.0024). FR's SD fell from 0.0040.
- FC's SD *rose* from an unusually small AO value (0.0017; the other grouping SDs were FR AO 0.0040, FR AA 0.0044 and FC AA 0.0043). That AO value includes FC AO G20261114, whose installation the audit confirmed.
- The paired FC − FR difference became much more stable (SD 0.0017). It stays positive in all five groupings, but at +0.0054 to +0.0091, below Δ_OO.
- The FR/FC differences are deterministic single draws per grouping, so these are descriptive counts, not tests.

## Decision (rule and code frozen in `protocol.json` before any AOF run; sha256 `6fd73236…`)

**ACTORDER-CONTRIBUTES-TO-WEIGHT-SENSITIVITY-ONLY**

| criterion | met? | values |
|---|---|---|
| W: WeightReduction ≥ 0.25, both strategies | **yes** | 0.416 / 0.410 |
| P: SD_AOF / SD_AO ≤ 0.70, both strategies | **no** | FR 0.60 yes, FC 1.37 no |
| D: SD(Δ_AOF) / SD(Δ_AO) ≤ 0.70 | **yes** | 0.30 |

## Answers

1. **Earlier failure:** BF16 arithmetic on outlier tokens, not a correctness problem (`update_installation_audit.md`).
2. **Default solver:** reproduces the original exactly (OO and AO, all 720 tensors; synthetic comparison with git HEAD).
3. **Residual under frozen routing:** about 39–40 % of elements, rel L2 ≈ 0.070 vs OO.
4. **Removed by fixing act-order:** about 41 % of the relative-L2 deviation; changed elements fall to 17–18 %.
5. **Quantization grid:** fixing act-order almost freezes it (scale difference 0.0024 vs 0.030; zero-points 0.8 % vs 6.4 %), even though only 2.6 % of columns changed group under AO.
6. **FR PPL spread:** reduced (SD ratio 0.60).
7. **FC PPL spread:** not reduced (1.37), and the AO baseline was atypically low.
8. **Paired FC − FR spread:** strongly reduced (0.30).
9. **Ranking:** preserved 5/5 under AOF versus 4/5 under AO and 2/5 under AA, with a smaller margin than OO.
10. **Reconstruction:** stable everywhere (≤ 0.01 %).
11. **Scales and zeros:** with the order fixed they barely move, yet 17 % of elements still differ. The sensitivity that remains is in error-feedback-driven rounding decisions, not in grid fitting.
12. **Most plausible remaining source:** per-element rounding flips driven by the alternative Hessian through GPTQ error feedback. These leave a per-strategy PPL SD of about 0.0024, comparable to the effect sizes being compared.

**What has not been causally established:**
- Whether about 0.0024 is an irreducible numerical floor of this GPTQ/BF16 pipeline.
- Whether specific layers, experts or tokens carry it.
- How act-order and group membership separate from each other; they were manipulated jointly.

**Keep these levels separate:**
- Hessian sensitivity: unchanged by AOF.
- Act-order and group sensitivity: removed by construction.
- Grid sensitivity: almost removed.
- Effective-weight sensitivity: reduced by about 41 %.
- Reconstruction: stable throughout.
- Model-level PPL: FR spread and the paired-difference spread reduced; FC spread not.

A lower changed-weight fraction is not by itself evidence of better PPL.

## Limitations

- Only allocation seed 0, five prespecified groupings and four quantized layers.
- A fixed BF16 environment and a fixed 4-bit g128 GPTQ configuration; the weights are fake-quantized BF16, not packed INT4.
- Act-order and group membership are manipulated jointly.
- Other discrete GPTQ operations (rounding under error feedback, min/max grid fitting) are not isolated.
- The alternative Hessians come from natural BF16 activation trajectories under frozen routing.
- Five deterministic observations per cell give no population-level variance estimates.
- This is not a causal decomposition of PPL variation.

## Next step (not executed)

**Measure the numerical floor directly.** Re-run OO calibration on the *same* 128 sequences, the same activations, the same routing and the act-order fixed to P_O. Accumulate the Hessians in five prespecified random sequence orders, so that H differs only by FP32 summation rounding. Do this for FR and FC: 10 configs. Then measure the changed-element fraction, the PPL SD and the Δ SD.

If that null perturbation already produces a PPL SD near 0.0024, the remaining AOF spread is GPTQ rounding chaos rather than calibration content. FR-versus-FC comparisons would then need effects well above that floor, or averaging over repeats.

## Compute

| stage | time / size |
|---|---|
| audit | 9 min |
| order capture (OO re-run with the natural-override regression) | 2 × 11 min |
| solver regression | 11 min |
| AOF (incl. read-only capture and the AO check recompute) | 443–471 s quantization + 2 × ~40 s PPL each |
| peak memory | 8.4 GB GPU / 41.7 GB host |
| disk | 39 GB in `~/moe_cache_actorder_ablation` (10 weight sets, quantizer captures) |
