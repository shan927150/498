# Projection-selective AGQ MVP: summary

**Verdict (frozen rule, confirmation split): `PROJECTION-AGQ-NO-GAIN` → Recommendation B, retain uniform AGQ or
layer-wise uniform selection.**

**Setup:**
- Qwen1.5-MoE-A2.7B, L8 and L16 routed experts.
- FR s0, GPTQ 4-bit g128 percdamp 1.0 act-order.
- Four configurations, assembled only from the existing GPTQ and AGQ weights:
  - GPTQ.
  - GU-AGQ: AGQ gate/up, GPTQ down.
  - Down-AGQ: GPTQ gate/up, AGQ down.
  - Full AGQ.
- One configuration per layer was selected on tuning rows 2000–2127 and frozen before testing. The selection was GU-AGQ
  on L8 and Full AGQ on L16.

## Confirmation split (E_layer × 1e-3, lower is better)

| Layer | GPTQ | GU-AGQ | Down-AGQ | Full AGQ | Selected |
|---|---|---|---|---|---|
| L8 | 7.096580 | **7.080248** | 7.120846 | 7.102910 | 7.080248 |
| L16 | 7.029577 | 6.816676 | 7.004825 | **6.810734** | 6.810734 |
| Mean | 7.063079 | 6.948462 | 7.062835 | 6.956822 | **6.945491** |

**Selected method vs the baselines (confirmation split):**
- vs layer-wise uniform: **−0.117 %**, 95 % CI [−0.156, −0.080].
- vs Full AGQ: −0.163 %.
- vs GPTQ: −1.665 %.

**Old test split:** −0.130 % vs layer-wise and −0.163 % vs Full AGQ.

The 0.5 % bar is not met, so criteria 2 and 6 fail.

## Mechanism (exploratory)

- **L16:**
  - Gate/up carries 93 % of the Full-AGQ gain (Shapley).
  - The interaction is antagonistic: I = +0.27 % of E_A, about 9 % of the gain.
- **L8:**
  - AGQ on gate/up helps by 0.23 %, and AGQ on down hurts by 0.34 %.
  - The two effects are near-additive, so Full AGQ nets +0.09 %.
- **L16 expert 40 (post hoc):**
  - The Full-AGQ gain is −49 %. Gate/up explains 93 % of it and down 7 %.
  - This one expert supplies about 84 % of L16's summed c²-error reduction.
- **Where the differences are:** they are diagonal. Cross terms are ≤ 0.02 % of the total.

## Cost and checks

- **Cost:** no new solves for this MVP. Fresh deployment needs both candidate sets, ≈ 2–3× one quantization run.
- **Correctness:** 20/20 checks pass.
- **Tests:** 14 targeted tests pass, and the full suite passes 589/589.

See `update_projection_selective_agq.md`.
