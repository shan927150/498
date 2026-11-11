# EBSS score/sign audit: summary

**Decision: SCORE-SIGN-CORRECT** (secondary: SCORE-CORRECT-PAPER-AMBIGUOUS). Full report: `../update_ebss_score_audit.md`.

- **Representation.** The production code stores cumulative log-probability (≤ 0). It scores `-(cum + logp_v)/(|S|+1) + sigma_parent/tau` and keeps the w smallest over all parent × token pairs. The final output is the argmin of `-cum/|c| + sigma/tau`.
- **Fixed examples.** All match the oracle, including the cross-parent trap (A 0.2556 < B 0.6156; the wrong formula would prefer B) and the τ* = 2 tradeoff threshold.
- **Metamorphic tests.** 300 random pools, 0 failures. The wrong variants are detectable: maximise 230/230, wrong NLL sign 190, per-parent 183, reversed σ 122.
- **Real-model capture.** 4 × 32 tokens per method, full 151,646-token pool. 62/62 pruning decisions, all propagations, and all final picks match the oracle. Score agreement is ≤ 3.6e-15 given the same logP, and ≤ 6.6e-7 against an fp64 recomputation from the logits.
- **Historical pilot artifacts.** cum, counts, σ, final scores and picks are exact. All 640 logged top-8 candidates are consistent.
- **Paper ambiguities.** These affect the strength of the balance term (Alg. 1 √E factor; rate vs share), not its sign. See `paper_ambiguities.md`.
- **Pilot validity.** The pilot's conclusions stand. Nothing was regenerated, and there was no GPTQ or PPL evaluation.
