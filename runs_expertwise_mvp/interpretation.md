## Interpretation

**Question: does expert-wise calibration improve reconstruction over shared-sequence calibration after controlling
for per-expert routed-token counts? — No.** Under the same covariance surrogate, letting each expert choose its own
128 sequences (EC) gives the same held-out error as one shared, covariance-selected set (SC):
−0.04 % ± 0.05 (test) and −0.03 % ± 0.10 (confirmation). Only 4/8 experts improve and the sign changes across seeds.
EC also costs about 6× the sequence acquisition (a union of 771–787 sequences for 8 experts vs. 128).

**Where the gains come from.** The 2 × 2 design separates the two factors:

| factor | comparison | matched test | matched confirm |
|---|---|---|---|
| covariance-aware selection (shared) | SC vs SR | **+0.55 %**, 8/8 experts, 3/3 seeds | +0.62 %, 8/8 |
| covariance-aware selection (expert-wise) | EC vs ER | **+0.81 %**, 8/8 experts, 24/24 expert×seed | +0.91 %, 8/8 |
| removing the shared constraint (surrogate) | EC vs SC | −0.04 % (null) | −0.03 % (null) |
| removing the shared constraint (random) | ER vs SR | −0.31 % (1/8 experts) | −0.32 % (mixed) |

The improvement comes entirely from **covariance-aware selection**, not from relaxing the shared-sequence
constraint. All effects are small: under 1 % relative, with absolute errors around 5.1e-3. Part of the EC-vs-ER gain
comes from ER sitting slightly *above* SR. The ER-vs-SR deficit should not be over-read: each seed has only one SR
draw shared by all 8 experts, so the shared conditions supply just 3 independent draws (a 1-in-8 chance that all 3
land on the same side of ER by luck).

**Token quantity versus selection quality.** Without token matching (secondary analysis), EC beats SC consistently,
by +0.23 % with 8/8 experts. EC's objective prefers sequences that route more tokens to the expert: it carries 4–14 %
more tokens than SC, and 11.5 % of its tokens are discarded by matching (SC: 4.5 %). Once counts are matched, that
advantage disappears. **The apparent expert-wise benefit is a per-expert token-quantity effect, not a selection-quality
effect.**

**Surrogate improvement is not quantizer improvement.** EC lowers the 64-d sketch mismatch relative to SC in **24/24**
expert×seed cases (0.121 → 0.0995 after matching, −18 %) and lowers the full-dimensional mismatch on average
(0.339 → 0.326). Neither predicts the EC-vs-SC error change: Spearman ρ = −0.02 (sketch, p = 0.91) and −0.25 (full-dim,
p = 0.23), n = 24. The pooled ρ ≈ +0.7 between mismatch and error in §4 comes from the large gap between random and
covariance-aware sets. Within the covariance-aware regime, further mismatch reduction buys nothing measurable.

**Separating the confounds the protocol asked about:**
* *Relaxing the shared constraint*: null after matching (EC vs SC).
* *Covariance-aware surrogate*: the only factor with a consistent, material effect, under both shared and expert-wise
  sets.
* *Larger union of unique sequences*: EC reads ~780 distinct sequences in total (ER ~820). Each expert still sees
  exactly 128 sequences, and its effective source diversity is only mildly higher (inverse Simpson 120 vs 116 for SC).
  The larger union is a cost, not a measured benefit.
* *Per-expert token budget versus global sequence budget*: k_e matches tokens per expert, but EC/ER need a much larger
  global acquisition budget (for all 60 experts the union would approach the whole pool). Even with that budget
  advantage, EC does not beat SC.

**Limitations.** One layer (L16), 8 of 60 experts, B = 128, 3 allocation seeds, and an isolated-expert reconstruction
metric only: no full-model PPL and no inference-efficiency claim. Layer 16's routing is fairly balanced (51k–84k pool
tokens per expert), and the result may differ in layers with rarer experts or at low budgets. The reference covariance
is the candidate pool's own, not the evaluation distribution's. The selection optimizer is a heuristic (stochastic
greedy plus swaps) on a 64-d sketch. The test rows and confirmation split were inspected in earlier studies, but
neither was used for any selection or setting here; all settings were frozen after the seed-0 smoke run, which the
final seed-0 run reproduced bit-for-bit.

**Recommendation.** Do not pursue expert-wise sequence selection as a replacement for shared-sequence E3. If anything
here is worth carrying forward, it is the *shared* covariance-aware objective (SC: +0.55–0.62 % over random at
unchanged acquisition cost). It should be tested against the existing fixed-budget E3 strategies before any claim is
made, since it has not been compared with them here.
