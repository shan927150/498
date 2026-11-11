# MoEQuant Sec. 4.2 / Algorithm 1: equations as written, and the canonical specification

Sources:
- arXiv 2505.03804v1 HTML, re-read for this audit.
- `asserts/Algorithm_1_EBSS.png` in github.com/chenzx921020/MoEQuant @ d77ba1b, read directly.

The OpenReview discussion is not accessible (bot wall), so it is not used. The official repository has no EBSS generator code.

## Equations as printed

| eq. | as printed (transcribed) | remarks |
|---|---|---|
| (3) | σ = (1/L) Σ_l σ_l | L = number of MoE layers |
| (4) | σ_l = sqrt( 1/(E−1) Σ_e (u_l^e − û_l)² ) | sample SD; u = "frequency of expert usage" (normalisation unstated) |
| (9) | D* = argmin_D { PPL(M,D) · exp(σ(M,D)/τ) } | **minimisation**; τ "controls the impact of expert imbalance" |
| (10) | D* = argmin_D { −(1/N) Σ_{i=1}^{n} log P(D_i \| D_{1:i−1}) + σ(M,D)/τ } | the log of (9); N is undefined (n is the length) |
| (11) | R_S = Σ_{i=1}^{n} log P(S_i \| S_{1:i−1}) | **cumulative log-probability** (≤ 0). The i = 1 term is conditioned on an empty prefix. |
| (12) | PPL(M, S‖v) = exp( −(R_S + P(v\|S)) / (n+1) ) | **log missing** before P(v\|S); subscript printed as **s** |
| (13) | score(S‖v) = −(R_S + log P(v\|S)) / (l+1) + σ(M,S)/τ, v ∈ V | average NLL of the extension (≥ 0) + prefix imbalance / τ |
| (14) | Ŝ = argtopk_{S‖v}(w, score(S‖v)), "the set of w values of x that **maximize** f" | as written, this keeps the w **largest** scores |
| text | "Deferred expert imbalance calculation": v is not included in the balance assessment | σ uses the prefix S, so all children of one parent share it |

Algorithm 1 (image):
- Line 4: `first_token ← Random(V)`.
- Line 6: `for j ∈ [1, Seqlen−1]`.
- Line 10: `RS ← ComputeCumulativeProbability` (Eq. 11). Line 26: Σ_{t=1}^{|seq|} log P(seq[t] \| seq[1..t−1]).
- Line 13: `prob ← Softmax(logits)[v]`.
- Line 14: `score ← −(RS + log(prob))/(j+1) + σ(M, new_candidate)/τ`.
- Line 18: `Topk(expanded_candidates, w, key=score)`, taken globally over all candidates of all parents.
- Line 20: `best_sentence ← arg max Score(c)`.
- Line 30: σ ← sqrt( E/(E−1) Σ_e (expert_usage[e] − μ)² ).
- Line 36: `Score = −RS/|candidate| + σ/τ`.

## Canonical specification audited

Notation:
- L(S) = Σ_i log P(s_i \| s_<i) ≤ 0 (Eq. 11)
- N(S) = −L(S) ≥ 0
- p_v = log P(v\|S) ≤ 0

**Pruning score** (minimised):

    J(S‖v) = −(L(S) + p_v)/(|S|+1) + σ(S)/τ  =  (N(S) − p_v)/(|S|+1) + σ(S)/τ

- Keep the w smallest J over all (parent, token) pairs of one sentence.
- Ties are broken by (parent, token).

**Final selection:** argmin over the surviving branches c of −L(c)/|c| + σ(c)/τ. Probability-Only drops the σ/τ term everywhere.

**Why minimisation is the mathematically consistent reading:**
- Eq. 9 and 10 are argmin problems.
- The first term of Eq. 13 is an average negative log-likelihood, the log of the extension's PPL from Eq. 12. Lower means more probable.
- The second term grows with imbalance.

Maximising Eq. 13 as written would therefore select the *least* probable and *most* imbalanced extensions, the opposite of the stated objective (Eq. 9) and of the deferred-balance rationale ("pruning should depend on the LLM's probabilities").

The only reading in which every equation's sign agrees is: Eq. 13 is a cost, and "argtopk / arg max" means "best", i.e. the smallest cost. The other self-consistent repair would flip the sign of Eq. 13 and maximise. That ranks candidates identically, so the two are not separate interpretations.

A literal reading (maximise Eq. 13 exactly as printed) is internally inconsistent. It is documented in `paper_ambiguities.md` and not adopted. It was not tested against quality metrics.
