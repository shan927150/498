# Layer-pooled Hessian shrinkage MVP: summary

**Frozen verdict: `POOL-SHRINKAGE-MISMATCH-ONLY` → C**, report a covariance-estimation finding and do not continue
algorithm development. Correctness: 21/21 checks pass.

The verdict label understates how negative the result is. Pooled shrinkage also loses to both controls:
- to the trace-matched identity control at the same α, and
- to the historical tuned damping, by a wide margin.

On practical grounds the conclusion matches B: keep standard GPTQ with tuned damping.

## Stage 1: covariance mismatch (tune-split reference, 12 experts, gate/up)

| Mean D_F | raw | pooled α=0.1 / 0.3 / 0.5 | identity α=0.1 / 0.3 / 0.5 | pure prior α=1 |
|---|---|---|---|---|
| B16 | 0.580 | 0.547 / 0.518 / **0.537** | 0.531 / 0.504 / 0.575 | 0.748 |
| B32 | 0.448 | 0.424 / **0.418** / 0.464 | 0.414 / 0.431 / 0.542 | 0.711 |

- **Screen passed.** The frozen screen picked α\* = 0.5 at B16 (−7.3 %, 8/12 experts) and α\* = 0.3 at B32 (−6.6 %,
  11/12).
- **Narrow pass.** Pooled beats identity *only* at those alphas. At α = 0.1 (both budgets) and α = 0.3 (B16), identity
  is closer to the reference.
- **Diagonal mismatch gets worse.** Pooling roughly doubles diagonal mismatch: 0.088 → 0.179 at B16, α = 0.3.
- **Top-16 eigenspace alignment** is unchanged.
- **Trace matching** makes almost no difference: the untraced D_F is within 0.001.

## Stage 2: complete-expert c²-weighted output error J (confirmation split)

| Method | B16 mean J | vs standard | B32 mean J | vs standard |
|---|---|---|---|---|
| Standard GPTQ (percdamp 1.0) | 0.017278 | — | 0.016469 | — |
| Identity shrinkage, α\* | 0.017061 | +1.26 % | 0.016375 | +0.57 % |
| **Pooled shrinkage, α\*** | 0.017169 | **+0.63 %** [0.42, 0.85] | 0.016386 | **+0.50 %** [0.32, 0.71] |
| Tuned damping (3.0 / 2.0) | 0.016448 | +4.81 % | 0.016019 | +2.73 % |
| Standard, B64 / B128 | 0.015710 / 0.015080 | | | |

- **Criteria:** C1 (≥ 1 %), C2 (beats identity) and C3 (competitive with tuned) fail at both budgets. C4–C6 pass: 9/12
  experts improve, the worst expert is +1.8 %, and both layers improve.
- **Gap recovery:**
  - B16 → B32: pooled 13 %, identity 27 %, tuned 103 %.
  - B32 → B64: pooled 11 %, identity 12 %, tuned 59 %.
- **Subset replacement** (B16, 6 of 60 experts, which carry about 10 % of the diagonal error): MoE output error changes
  by −0.077 % (L8) and −0.009 % (L16).
- **Cost:** pooled accumulation adds 0.11–0.13 s per layer, 0.22–0.27 % of replay + 60-expert solve. Cost is not the
  obstacle.

See `update_hessian_pool_shrinkage_mvp.md`.
