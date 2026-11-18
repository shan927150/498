## Interpretation

Scope: allocation seed 0 only, and 32 WikiText windows (16,384 tokens per layer). The window bootstrap captures token-sampling uncertainty but not allocation-seed or calibration-grouping uncertainty.

### SC vs FR: the improvement survives to the router-weighted MoE output (Outcome C)
SC improves every link of the chain by roughly the same relative amount:

| link | SC vs FR |
|---|---|
| C4 projection error (R0) | −0.67 % |
| projection error on the WikiText tokens (P_seq) | −0.51 % |
| nonlinear expert output, equal-expert (B0) | −0.50 % |
| uncancelled router-weighted energy (B1) | −0.70 % |
| router-weighted MoE output (B2) | −0.68 % (window bootstrap 95 % CI [−1.03, −0.32]) |

- B2 improves in all four layers: −0.27 / −0.64 / −0.71 / −0.68 %.
- Per expert, the projection change predicts the output change (Spearman +0.61). 178 of 240 experts have lower output error.
- BF16-native compute gives the same answer (B2 −0.68 %).
- So, at fixed inputs and routing, better linear projection reconstruction does turn into better nonlinear expert outputs and a better routed MoE output. The nonlinearity and the router aggregation do not remove it.
- One caveat: **pooled B0** (all expert output energy summed before dividing) changes by only +0.01 %.
  - Output energy is very concentrated: expert L23/26 alone holds 14 % of it, and SC is 8 % worse on that expert.
  - That expert has a small router weight (energy-weighted mean w² = 0.009), so it matters little in B2.

### FC vs FR: most of the reconstruction penalty is specific to the C4 measurement tokens
- FC's +2.55 % R0 penalty survives every expert weighting (MVP A) but shrinks to +0.49 % when the same projection objective is measured on the WikiText tokens (P_seq). B0 is +0.51 %.
- At the routed MoE output, FC is −0.14 % pooled (CI [−0.81, +0.47]) and +0.46 % with layers weighted equally.
- Its per-layer B2 changes have mixed signs (L1 −2.05, L8 +1.08, L16 +0.63, L23 −0.24 %).
- Pooled B2 is about 84 % layer 23, because that layer carries 84 % of the routed output energy.
- The reweighted R1/R2_wiki metrics use C4-measured errors and stay at +2.4 %. So the shrinkage comes from the tokens the error is measured on, not from expert weights.
- With one seed and 32 windows, read this as a strong hint, not an established mechanism.

### Cancellation
- B3 = 0.9991–0.9994 for all three strategies, with no difference between them (+0.02 %).
- Errors of the four routed experts are essentially uncorrelated per token: they neither cancel nor reinforce. The router-weighted aggregation therefore adds nothing that a squared-router-weighted expert-output metric (B1) misses.

### Why B2 can improve while NLL does not show it (scale, not mechanism)
- The whole 4-layer quantization penalty is about 0.0037 nats/token (grouping-averaged FR vs BF16).
- Assume, as an untested linear-proportionality assumption, that NLL scales with local MoE-output error energy. Then SC's −0.68 % predicts about −2.5e-5 nats, and FC's C4 R0 penalty would predict about +9.5e-5.
- The per-configuration grouping SD of ΔNLL is about 5.5e-4.
- Under that assumption, the expected effects are 6–20× smaller than what 15 grouped evaluations can resolve, so the PPL null says little about whether the local improvement matters.
- This scale argument is an estimate, not a measurement of how errors propagate.
