# EBSS score/sign correctness audit

**Verdict (frozen rule): `SCORE-SIGN-CORRECT`.** Secondary finding: `SCORE-CORRECT-PAPER-AMBIGUOUS`.

The production generator stores **cumulative log-probability** (≤ 0), not NLL. Its Eq. 13 score is
`-(cum_logp + logp_v)/(|S|+1) + σ(parent)/τ`, minimised over all parent × token pairs. The suspected sign trap,
`-(cum_nll + logp)/(l+1)`, is not present.

All captured production pruning decisions match an independent NumPy oracle:
- **62 real-model steps:** 4 sentences × 31 steps × 2 methods, each over the full 151,646-token vocabulary.
- **81 synthetic-backend steps.**

Also matched exactly:
- cumulative-score propagation, routing-count propagation and σ;
- final-beam selection for all 16 + 16 pilot sentences and all 4 + 4 audit sentences.

The B=16 pilot's generation results stand under the documented interpretation. No generator numerics were changed. Nothing was regenerated or re-quantized.

Artifacts: `runs_ebss_score_audit/`
- `audit_protocol.json`, frozen before any test
- `independent_oracle.py`
- `paper_equations_review.md`, `sign_convention_table.md`, `paper_ambiguities.md`
- the JSON results
- `decision.json`, `summary.md`, `logs/`

Code:
- `diagnostics/ebss_score_audit.py` (stages: synthetic / historical / production / report)
- `tests/test_ebss_score_audit.py` (14 tests)
- an opt-in, read-only `audit=` hook added to `moe_calib.ebss_generator.generate()`

---

## A. Mathematical specification

Let L(S) = Σ_i log P(s_i | s_<i) ≤ 0 (Eq. 11), N(S) = −L(S) ≥ 0, and p_v = log P(v | S) ≤ 0.

    J(S‖v) = −(L(S) + p_v)/(|S|+1) + σ(S)/τ          (logP form)
           =  (N(S) − p_v)/(|S|+1) + σ(S)/τ          (NLL form, identical)

- **σ (Eq. 3/4):** the mean over the 24 MoE layers of sqrt(1/(E−1) Σ_e (u_e − ū)²). Here u_e is the routed top-4 count of the **parent** prefix divided by |S| ("deferred" balance). The shared expert is excluded.
- **Pruning:** keep the w = 4 **smallest** J over all (parent, token) pairs of a sentence, with ties broken by (parent, token).
- **Final selection:** argmin over the survivors of −L(c)/|c| + σ(c)/τ.
- **Probability-Only:** the same procedure with the σ/τ terms removed.

The full derivation of why minimisation is the only sign-consistent reading of Eq. 9–14 / Alg. 1 is in `paper_equations_review.md`.

## B. Existing implementation (`moe_calib/ebss_generator.py`)

| line | code | meaning |
|---|---|---|
| 265 | `logits = out.logits[:, -1, :151646].float()` | last-position logits of each live branch, over V = ids 0..151645 |
| 342 | `cum = torch.zeros(N, float64)` | L(S) of the 1-token prefix. The first token is unconditioned and contributes 0. |
| 347–348 | `for j in range(1, seqlen): l = j` | l = \|S\|, counting the first token |
| 351 | `counts.scatter_add_(2, sel, 1)` | adds the routing of the most recently fed (last prefix) token |
| 353, 70–71 | `sigma = balance_sigma(counts, l)` → `(counts/l).std(unbiased=True).mean(-1)` | Eq. 3/4 on the prefix |
| 356 | `logp = log_softmax(logits.float(), -1)` | log P(v\|S) ≤ 0 (fp32) |
| 357, 77–79 | `s = -(cum[:,None] + logp)/(l+1)`; `s += (sigma/tau)[:,None]` if EBSS | Eq. 13 in logP form; σ is divided by τ and added |
| 362, 85–88 | `torch.sort(flat, stable=True).indices[:w]` on the flattened [B, V] per sentence | global top-w smallest; stable sort gives (parent, token) tie order |
| 396–402 | `newlp = logp[par, tok]`; `tokens = cat(tokens[par], tok)`; `cum = cum[par] + newlp`; `counts = counts[par]`; `backend.reorder(par)` | each child inherits **its own parent's** history, counts and KV cache, plus the selected token's logP |
| 412–417, 93–94 | last token's routing added; `fs = -cum/512 + sigma/tau`; `argmin` | final selection |

Probability-Only sets `use_balance=False`. This skips line 79 and the σ term on line 94; nothing else differs.

## C. Independent verification

The oracle uses NumPy float64 and `math.fsum`, re-sums L from per-token lists, recomputes σ from routing ids with explicit loops, and ranks with `np.lexsort`. It does not import production code; a test enforces this.

**Explicitly calculated examples** (`synthetic_counterexamples.json`; τ = 1.3):

| # | setup | expected | correct J (A / B) | production |
|---|---|---|---|---|
| 6 | same parent and history (\|S\| = 3), σ = 0.1; p = 0.8 vs 0.2 | A | lower for A | A |
| 7 | identical except σ = 0.1 vs 0.3 | A | A lower by 0.2/1.3 = 0.154 | A |
| **8** | **cross-parent trap**: L_A = −0.2, L_B = −2.0, p = 0.5, σ = 0.1, \|S\| = 4 | **A** | **0.25555 / 0.61555** | **A (0.25555 / 0.61555)** |
| 5a | \|S\| = 1 (first token only), p = 0.6 vs 0.3 | A | | A |
| 5b | p = 1e−3 vs 1e−30 (log p = −69.08) | A | finite, ordered | A |
| 5c | 4-token histories, B contains one p = 0.2 token | A | | A |
| 5e | exact tie: equal p, equal σ | (0,0),(0,1),(1,0) | | identical order |

The cross-parent trap, computed four ways:

| formula | J_A | J_B | prefers |
|---|---|---|---|
| correct logP: −(L + p)/5 + σ/τ | 0.255553 | 0.615553 | A |
| correct NLL: (N − p)/5 + σ/τ | 0.255553 | 0.615553 | A |
| deliberately wrong: −(N + p)/5 + σ/τ | 0.175553 | **−0.184447** | **B** |
| production `candidate_scores` | 0.255553 | 0.615553 | A |

The same trap also runs through the full production `generate()` with a scripted backend (V = 4, w = 2). EBSS and PO both keep both children of parent A and output A's continuation.

**Probability–balance tradeoff.**
- Setup: A has L = −1.0 and σ = 0.30; B has L = −1.5 and σ = 0.10; same p; \|S\| = 4.
- A's probability advantage is (1.5 − 1.0)/5 = 0.1. B's balance advantage is 0.2/τ.
- So B wins iff τ < τ\* = 2.
- Production and oracle agree at τ ∈ {0.5, 1.0, 1.3, 1.9, 2.0 (exact tie → A by parent index), 2.1, 3, 10, 1e9}.
- If τ multiplied σ instead of dividing it, B would win at τ = 3. That variant is detectable.

**Beam-width ranking.**
- Setup: 3 parents × 5 tokens, w = 2 and w = 4, with and without balance.
- Production kept sets and scores match the oracle exactly, in order.
- These variants each change the EBSS w = 4 kept set: maximisation, probability-only ranking, balance applied after pruning, rolled (wrong-parent) σ, rolled parent history, reversed σ.
- Per-parent top-k does not change the kept set in this hand-built case, but it does in 183/230 random pools.

**Deferred σ.**
- In a scripted test, two parents have equal L but σ = 0.577 vs 0.
- EBSS keeps both children of the balanced parent. PO keeps the other parent's children (tie order). Reversed σ does the same as PO.
- Production σ equals the oracle's Eq. 4 recomputation.

**Metamorphic tests** (300 random pools, seed 20261110): 0 failures across all 11 invariants.
- higher p_v is better; higher parent L is better; higher σ is worse;
- σ = 0 gives exactly PO; τ → ∞ gives PO; larger τ shrinks the balance spread;
- shifting all scores by a constant is invariant; the result does not depend on enumeration order;
- production = oracle; the logP and NLL forms are identical (≤ 1e−12);
- counterfactual ordering: disabled σ = PO, EBSS kept-σ ≤ PO, reversed ≥ PO.

The tests have discriminating power. On the 230 multi-parent pools, these wrong variants change the kept set in the following number of pools:

| wrong variant | pools changed (of 230) |
|---|---|
| maximise | 230 |
| wrong NLL sign | 190 |
| per-parent top-k | 183 |
| reversed σ | 122 |
| PO ranking | 72 |

## D. Production beam-trace comparison

**Historical 16 × 512 pilot.** These are the stored artifacts; no regeneration.
- cum_logp = fsum(token_logp) to within 6e−14.
- Routing counts equal an exact recount from the 512 × 24 × 4 routed ids.
- σ and final scores match the oracle to ≤ 1e−16.
- Final picks equal the oracle argmin for all 4 beams of all 32 sentences, and the picked beam is the output.
- All 640 logged top-8 candidates (steps 1/2/64/256/511) have scores equal to the oracle's Eq. 13, (score, parent, token) order, correct kept flags, and correct parent R/σ lookup.
- Teacher-forced B1 replay mean NLL vs generation −cum/511: |Δ| ≤ 0.0026 nats/token (EBSS) and ≤ 0.0017 (PO).
- Limitation: the logs kept only the top 8 candidates, so completeness over the full pool required a new capture (below).

**New opt-in capture on the real model.**
- Configuration: same snapshot, BF16, offload backend, the pilot's first 4 start tokens, 32 tokens, w = 4, τ = 1.3.
- At every step the hook recorded the full pre-pruning state: logits, logP, cum, σ, counts, routing ids, the full [16, 151646] production score matrix, and the production choice.

| check | EBSS | PO |
|---|---|---|
| pruning decisions = oracle (kept (parent, token) sets, all 4 sentences) | 31/31 steps | 31/31 steps |
| same against an oracle recomputing log-softmax in fp64 from the logits | 31/31 | 31/31 |
| max \|score − oracle\| (same per-token logP) | 3.6e−15 | 1.8e−15 |
| max \|score − fp64 oracle\| | 6.6e−7 | 6.6e−7 |
| max \|fp32 log-softmax − fp64\| | 3.9e−6 | 3.7e−6 |
| σ vs Eq. 4 recount | 5.6e−17 | 5.6e−17 |
| cum, token_logp, tokens, routing counts propagated parent → child | exact | exact |
| final pick = oracle; final score diff | yes; 1.1e−16 | yes; 0 |
| signs: logP ≤ 0, probability term ≥ 0, σ ≥ 0 | all | all |
| sentence-steps where σ changed the PO kept set | 4 | n/a |
| teacher-forced replay: mean \|Δlogp\|, min r | 0.021, 0.999 | 0.019, 0.999 |
| probability mass of the excluded padding rows | ≤ 1.7e−4 (mean 3e−6) | same |
| dispatch mismatches | 0 | 0 |

**Example of σ changing a real decision.** EBSS, sentence 2, step 3, \|S\| = 3. The table lists candidates ranked 1–5 by EBSS score (the JSON uses 0-based ranks 0–4).

| EBSS rank | parent | token | R_parent | logp_v | prob term | σ_parent | σ/τ | score (prod = oracle) | kept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 124376 | −3.8683 | −1.0430 | 1.22783 | 0.16475 | 0.12673 | 1.35456 | ✓ |
| 2 | 3 | 11071 | −5.2308 | −0.2504 | 1.37028 | 0.17179 | 0.13215 | 1.50243 | ✓ |
| 3 | 2 | 125006 | −4.5281 | −1.6867 | 1.55371 | 0.16299 | 0.12538 | 1.67909 | ✓ |
| 4 | 1 | 198 | −4.3558 | −1.8587 | 1.55361 | 0.16949 | 0.13038 | 1.68399 | ✓ |
| 5 | 1 | 124376 | −4.3558 | −1.8587 | 1.55361 | 0.16949 | 0.13038 | 1.68399 | ✗ |

By probability alone, parent 1's two tied tokens (1.55361) beat parent 2's (1.55371). Parent 2's lower σ moves its child ahead. This is the intended cross-parent effect, with the correct sign.

**Exact ties.**
- The cut sometimes falls on an exact tie (min gap 0). Pilot logs: 5/80 (EBSS) and 6/80 (PO) logged steps.
- Every such tie is two tokens of the **same parent** with identical BF16-derived log-probabilities, so σ cannot be involved.
- They are resolved by the documented token-id order, identically in production and oracle.

## E. Paper fidelity

See `paper_ambiguities.md`. Summary:

| # | ambiguity | effect on ranking | resolution |
|---|---|---|---|
| A1 | Eq. 14 "maximize" and Alg. 1 line 20 "arg max" vs argmin in Eq. 9/10 and a cost-shaped Eq. 13 | literal maximisation reverses every decision | minimisation; the only sign-consistent reading |
| A2 | σ(M, S) in Eq. 13 and the text vs σ(M, new_candidate) in Alg. 1 line 14 | yes, in principle | prefix S (stated explicitly in the text; per-candidate σ is infeasible) |
| A3 | Eq. 12 missing log | n/a | Eq. 12 is not used by the search |
| A4 | normaliser (N, n+1, l+1, j+1, \|c\|) | none within a step | changes the probability/σ weight by (l+1)/l: 2× at l = 1, < 1 % beyond l = 100 |
| A5 | log P of the first token (Eq. 11 sums from i = 1; Qwen has no BOS) | **none** | the first token is common to all branches of a sentence, so this is a constant shift |
| A6 | Alg. 1 line 30 E/(E−1) vs Eq. 4 1/(E−1) | yes: √60 ≈ 7.75× stronger σ | Eq. 4. **Correction:** the pilot's `source_review.md` described this as "τ/E"; it is τ/√E. |
| A7 | usage rate vs share (÷k) | yes: σ ÷ 4, equivalent to τ × 4 | rate (assumption) |
| A8 | final selection / set extraction undocumented | covered by A1 | argmin of Alg. 1 line 36 |
| A9 | vocabulary restricted to the trained rows | ≤ 1.7e−4 mass | documented |

A6 and A7 leave the **strength** of the balance term uncertain within roughly a factor of 4–8. They do not affect its **sign**. No variant was chosen or tested on quality.

The released code contains no generator, so exact reproduction of the authors' implementation is **not** claimed.

## F. Verdict

**`SCORE-SIGN-CORRECT`.** Every pre-registered check passed (`decision.json`):
- the fixed counterexamples, the tradeoff threshold, deferred σ end-to-end, and the cross-parent trap;
- metamorphic invariants and beam-width ranking;
- synthetic and real-model `generate()` traces, and the historical pilot artifacts.

The deliberately wrong variants were shown to be detectable.

To summarise:
- The score is correct under the documented interpretation, and the beam ordering is mathematically consistent.
- Probability-Only is a valid control: it differs only by the σ/τ terms (shown exactly: σ = 0 gives bit-identical scores and selections).
- No evidence of a sign bug was found.
- The paper ambiguities A1, A2, A4–A8 remain. This is not proof of fidelity to the authors' private implementation.

## G. Impact on previous experiments

The B=16 pilot (`update_ebss_generation_pilot.md`) remains valid as a test of the documented interpretation. Its pruning, cumulative-score and final-selection arithmetic is now verified on the stored artifacts and on a fresh full-pool capture.

The audit gives the pilot's measured failure modes a correct-implementation reading:
- Repetition, self-PPL ≈ 1.2, and a balance term that alters only 4.4 % of kept sets **are not caused by a sign error.**
- They follow from the objective itself, which minimises average NLL and so rewards loops. The σ/τ term is about 17× smaller in spread than the probability term and is shared by all children of a parent.

The open fidelity risk is the **strength** of the balance term (A6/A7). It is not its sign. Nothing needs rerunning.

Tests:
- `tests/test_ebss_score_audit.py` (14) and `tests/test_ebss_generator.py` (31): 45 passed.
- The audit hook is read-only: a test confirms identical outputs with and without it.
