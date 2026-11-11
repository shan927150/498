# Paper ambiguities relevant to the score: documented, not resolved by results

For each item:
- (1) source location
- (2) literal reading
- (3) mathematically consistent reading
- (4) what the generator does
- (5) does it change score values?
- (6) does it change beam ranking?
- (7) is there an author clarification?

No author clarification is available for any item. The OpenReview thread is inaccessible, and the rebuttal tables in the official repository do not address any of these points.

### A1. Direction of top-k and of the final selection
1. Source: Eq. 14 text ("argtopk … the w values of x that maximize f"); Algorithm 1 line 20 (`arg max Score(c)`).
2. Literal: keep the w **largest** Eq. 13 scores; output the branch with the largest Score.
3. Consistent: Eq. 9/10 are argmin; Eq. 13's first term is an average NLL and its second term is an imbalance penalty. "Top"/"max" must therefore mean best = **smallest** cost.
4. Generator: minimises at both steps (`torch.sort` ascending; `torch.argmin`).
5. Score values: no.
6. Ranking: yes. The literal reading reverses every decision. On the 230 multi-parent synthetic pools it changes the kept set in 230/230 cases. It would select the least probable, most imbalanced extensions, which contradicts Eq. 9 and the paper's own rationale.

### A2. Balance on the prefix or on the candidate
1. Source: Eq. 13 writes σ(M, **S**); Sec. 4.2 "Deferred Expert Imbalance Calculation" excludes v; Algorithm 1 line 14 writes σ(M, **new_candidate**).
2. Literal (Alg. 1): σ of S‖v.
3. Consistent: the text is explicit, and v's routing is unknown without a forward pass of every v ∈ V. So σ(S).
4. Generator: σ of the parent prefix (verified: the same σ for all children of a parent, and the correct parent after reordering).
5. Score values: yes, in principle.
6. Ranking: yes, in principle. The candidate-level variant is infeasible at |V| = 151,646 and is not implemented.

### A3. Missing log in Eq. 12
1. Source: Eq. 12, `R_s + P(v|S)`.
2. Literal: adds a probability to a log-probability.
3. Consistent: log P(v|S), as in Eq. 11, Eq. 13 and Algorithm 1 line 14.
4. Generator: log.
5–6. The literal form would change both, but it is dimensionally inconsistent. Eq. 12 is not used by the search; only Eq. 13 is.

### A4. Length normaliser
1. Source: Eq. 10 −1/N (N undefined); Eq. 12 1/(n+1); Eq. 13 1/(l+1); Alg. 1 line 14 1/(j+1); Alg. 1 line 36 1/|candidate|.
2/3. All of these denote the candidate length. With the first token counted, Alg. 1's j+1 equals l+1.
4. Generator: l + 1, where l = |S| includes the first token. The final score uses |c| = 512.
5. Score values: the alternative "number of model-scored tokens" (l) scales the probability term by (l+1)/l.
6. Ranking within a step: no, because every candidate in a step shares l. It does change the probability-vs-balance tradeoff by a factor (l+1)/l: 2.0 at l = 1, 1.06 at l = 16, < 1.01 beyond l = 100. The audit records the generator's convention and does not change it.

### A5. First-token log-probability
1. Source: Eq. 11 / Alg. 1 line 26 sum from t = 1, i.e. P(S_1 | ∅). Alg. 1 line 4 draws S_1 at random.
2. Literal: include log P(S_1) from an empty context.
3. Qwen1.5-MoE has no BOS token (`bos_token_id = None`), so P(S_1 | ∅) is undefined without an assumed start symbol.
4. Generator: the first token contributes 0 to R but counts in |S|, and its routing counts in σ.
5. Score values: yes, any defined value c = log P(S_1) adds c/(l+1) to every candidate and c/|c| to every final score.
6. Ranking: **no**. All branches of a sentence share the same first token, so the constant is common to every candidate compared at a step and in the final selection. The audit proves this invariant (constant-shift test). A hypothetical BOS token would add one routed token to σ, which is a σ question, not a sign question.

### A6. σ scale: Eq. 4 vs Algorithm 1 line 30
1. Source: Eq. 4 sqrt(1/(E−1) Σ …); Alg. 1 line 30 sqrt(E/(E−1) Σ …), with no mean over layers.
2. Literal Alg.: σ multiplied by √E (√60 ≈ 7.75), i.e. an effective τ/√E.
3. Consistent: Eq. 4. The equation, Algorithm 1's "▷ Eq. 4" annotation, and the rebuttal's STD magnitudes all point to it.
4. Generator: Eq. 4.
5. Score values: yes.
6. Ranking: yes. It strengthens the balance term by 7.75×.
   - **Correction:** `runs_ebss_generation_pilot/source_review.md` stated this factor acts "like τ/E". The correct factor is τ/√E. The adopted convention is unaffected.

### A7. Usage-frequency normalisation
1. Source: Eq. 4 "frequency of expert usage", normalisation unstated.
2/3. Either a per-token rate (count/|S|) or a share (count/(|S|·k)).
4. Generator: per-token rate. This was chosen because it matches the magnitude of the rebuttal's STD table.
5–6. Yes: the share divides σ by k = 4, which is equivalent to τ × 4. This is a genuine ambiguity in the strength of the balance term, not a sign issue. It is not resolved by any result in this audit.

### A8. Final-beam selection and set extraction
1. Source: Alg. 1 lines 20 and 36. The paper text is silent.
4. Generator: argmin of line-36 Score over the w survivors. One sentence per start token.
6. Changes which survivor is output. Covered by A1.

### A9. Vocabulary restriction
- The generator computes the log-softmax over ids 0..151645 and excludes the 290 never-trained embedding-padding rows.
- The paper does not discuss this.
- Effect on log P: the log of (1 − excluded mass). This is measured on the real model in `production_trace_audit.json` and is negligible.
- It is common to all candidates of a parent but not across parents, so it is reported with its magnitude.
