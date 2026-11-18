# Expert-wise GPTQ/AGQ hybrid MVP: summary

**Verdict (frozen rule, confirmation split): `HYBRID-NO-GAIN` → Recommendation B, prefer layer-wise objective selection.**

**Setup:**
- Qwen1.5-MoE-A2.7B, routed experts of L8 and L16 (120 experts).
- FR s0, C4 Random B = 128, CANONICAL-B1, GPTQ 4-bit g128 percdamp 1.0 act-order.
- Two existing candidates per expert: historical GPTQ q = 0 and AGQ q = 1, both bitwise verified.
- Experts were selected on tuning rows 2000–2127 by the c²-weighted full-expert output error, and the choice was frozen
  before any test evaluation.

## Aggregated MoE-layer output error (confirmation split, ×1e-3, lower is better)

| Layer | Uniform GPTQ | Uniform AGQ | Layer-wise selection | Expert-wise hybrid |
|---|---|---|---|---|
| L8 | 7.096580 | 7.102910 | 7.096580 (GPTQ) | 7.096236 |
| L16 | 7.029577 | 6.810734 | 6.810734 (AGQ) | 6.813160 |
| Mean (2 layers) | 7.063079 | 6.956822 | **6.953657** | 6.954698 |

**Hybrid vs the baselines on the 2-layer mean (confirmation split):**
- vs GPTQ: −1.53 %.
- vs AGQ: −0.031 %, 95 % CI [−0.063, +0.001].
- vs layer-wise: **+0.015 %**, CI [−0.021, +0.051].

**Old test split:** −1.54 %, −0.036 % and −0.003 %. Native BF16 gives the same ranking.

## Key facts

**Selection:**
- 76 experts chose AGQ and 44 chose GPTQ (L8 30/30, L16 46/14).
- On L8 the preferences are noise: half-split agreement is 0.48, and 95 % of the margins are under 1 %.
- On L16 they are dominated by expert 40 (49 % AGQ gain). Without it, L16's AGQ c²-gain falls from 3.05 % to 0.48 %
  (exploratory).

**Transfer (confirmation split):**
- Sign agreement is 0.73 and Spearman is 0.67. The criterion passes, carried by L16; on the old test split Spearman is
  0.47.
- Small-margin experts carry 83 % of the regret.

**Oracle headroom:** perfect per-expert selection would add ≈ 0.1 % (L8) or ≈ 0.02 % (L16) of the c²-error beyond the
best uniform method.

**Cross terms:** ≤ 0.02 % of the total in every configuration. On L8 they cancel about 85 % of a negligible diagonal gain.

**Cost:**
- Fresh deployment costs about 2.2× one quantization run, the same for layer-wise and hybrid.
- This MVP reused the historical candidates, so its incremental run was ≈ 10 min of evaluation.
- Final storage and inference are unchanged (inference not measured).

**Correctness:** all 15 checks pass (`correctness_results.json`).

See `update_expertwise_agq_hybrid.md` for the full report and the answers to the 10 questions.
