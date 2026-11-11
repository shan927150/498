# Model-matched EBSS generation pilot + Probability-Only ablation (16 × 512)

**Decision (frozen rule): `MODEL-MATCHED-EBSS-BALANCING-SUPPORTED`.**
EBSS lowers set-level expert-load imbalance relative to an otherwise identical
probability-only beam search (paper σ −4.9 %, mean load CV −4.9 %, 18/24 layers).
The effect is small, it is concentrated in short prefixes, and it does **not**
carry over to quantization quality. B=16 GPTQ reconstruction is +0.46 % *worse*
than Probability-Only and +13.8 % worse than C4 Random (0/240 experts improved).
PPL for every config is inside the C4 seed spread. Both generated sets are
strongly repetitive and much less balanced than random C4 at the same budget.

Scope: Qwen1.5-MoE-A2.7B base (snapshot `1a758c50…`), BF16, w = 4, τ = 1.3,
16 sequences × 512 tokens, one seed and one start-token draw. This is a pilot.
It says nothing about B = 128 effectiveness.

Artifacts: `runs_ebss_generation_pilot/` (protocol, generated sets, all JSONs,
figures 1–6, logs). Code: `moe_calib/ebss_generator.py`, `moe_calib/ebss_pilot.py`,
`diagnostics/ebss_generation_pilot.py`, `diagnostics/diagnostics_ebss_pilot.py`,
`diagnostics/report_ebss_generation_pilot.py`, `scripts/run_ebss_generation_pilot.sh`,
`tests/test_ebss_generator.py`.

---

## 1. Source fidelity

The sources are kept in four separate categories.

**Faithful paper reproduction.** These come from arXiv 2505.03804v1 and its Algorithm 1 figure in the official repo:
- Score (Eq. 13): `−(R_S + log P(v|S))/(l+1) + σ(M,S)/τ`, where R_S is the accumulated negative log-probability.
- Balance σ (Eq. 3/4): the mean over all L MoE layers of the per-layer
  `σ_l = sqrt(1/(E−1) Σ (u_e − ū)²)`.
- Beam search with width w, keeping the top-w (S||v) over all parents × vocabulary.
- "Deferred expert balance": the σ term is evaluated on the prefix S, because the
  routing of v is only known after the next forward.
- Paper defaults τ = 1.2, w = 4.

**Author-rebuttal clarifications** (`Tables_for_R*.md` in the official repo):
- Calibration size is 128 sequences × 512 tokens.
- τ sensitivity for Qwen is flat (1.2 → 49.60, 1.3 → 49.61), and w = 4 is used.
- Qwen EBSS generation takes 18 min on an A800. This is reported as-is and is not
  normalised to our hardware.
- DeepSeek expert-balance STD is 0.0427 for wikitext vs 0.0052 for EBSS.

The OpenReview discussion page could not be fetched (bot wall), so nothing from it is used.

**Implementation assumptions.** These are undocumented. Each was frozen in
`source_ambiguities.json` / `protocol.json` before any pilot generation.
- **Minimisation throughout.** Eq. 9–10 minimise, but the Eq. 14 text says "maximise"
  and Algorithm 1 line 20 says argmax. The only consistent reading is that the
  balance term penalises imbalance.
- u_e = routed top-k count / tokens. Only the routed top-4 counts. The shared expert is excluded. All 24 layers.
- The Eq. 4 scale is used. The E/(E−1) factor on Algorithm 1 line 30 is a constant
  rescaling and does not change any ranking.
- First token: uniform over ids 0..151642 with seed 20261110. The same ids are used for both methods.
- No EOS stop. Exactly 512 tokens including the first. The vocabulary is ids 0..151645.
- Ties are broken by a stable sort on (score, parent, token).
- Final sequence = argmin final score among the w beams.
- Sentences are decoded in lock-step batches. BF16 weights, fp64 scores.
- **τ = 1.3** (the rebuttal value) instead of the paper's 1.2. It was fixed before
  generation and never tuned.

**Remaining ambiguities:** first-token rule, stopping rule, final-beam selection,
the sign convention (Eq. 14 / Alg. 1), whether σ uses counts or rates, and which
checkpoint produced the released `EBSS_qwen.jsonl`. The official repo contains
**no EBSS generator code**.

## 2. Algorithm correctness

Unit tests (`tests/test_ebss_generator.py`, 31 tests) cover the following:
- σ matches a direct Eq. 4 computation.
- The incremental statistics equal a full recount.
- Scores and rankings match a brute-force recomputation.
- Probability-Only ignores the balance term and EBSS includes it.
- Determinism, token budgets and validation.
- The GPTQ source sha256 is unchanged (`58e40d1c…`).

The smoke gate on the real model (4 × 32 tokens, both methods) passed on attempt 2.
- Dispatch cross-check: 0 mismatches.
- Runs are deterministic and parents stay intact.
- Replay vs generation log-probabilities: mean |Δ| ≈ 0.02, r = 0.9994.
- Attempt 1 failed only because an exact-equality check compared GPU and CPU σ.
  The tolerance was set to 1e-12. The failed record is kept in `logs/smoke_attempt1_*`.

Both generated sets are valid: 16 distinct sequences × 512 real token ids, with 0 padding.

The diagnostics gate also passed. It checks two things:
- Capture regression: 28 shared C4 rows are bitwise identical at layers 1/8/16/23.
- Contamination: 0 generated windows share a 32-gram with held-out data.

Decision metrics use **CANONICAL-B1 replay routing** of the finished sequences.
This protocol amendment was made before generation. Generation-time routing
(batched, KV-cached) agrees with replay on 87–88 % of routed sets. The EBSS
balance advantage has the same sign under both routings.

## 3. Generation results (Q4–Q12)

All 24 MoE layers, routing from the B1 replay. The C4 Random column is the mean of 3 seeds at B=16.

| metric | Prob-Only | EBSS | EBSS − PO | C4 Random |
|---|---|---|---|---|
| paper σ | 0.01753 | 0.01667 | −0.00086 | 0.01038 |
| load CV | 0.2608 | 0.2480 | −0.0128 | 0.1544 |
| Gini | 0.1385 | 0.1329 | −0.0056 | 0.0847 |
| min exposure | 299.7 | 294.0 | −5.6 | 373.8 |
| p10 exposure | 390.8 | 402.2 | +11.4 | 450.5 |
| zero-exposure experts | 0 | 0 | 0 | 0 |
| routing L1 vs C4 | 0.210 | 0.201 | −0.009 | 0.091 |
| frac routing-atypical | 0.88 | 1.00 | +0.12 | 0.40 |
| model PPL of set | 1.206 | 1.198 | | 11.0 |
| distinct-1 / distinct-2 | 0.222 / 0.376 | 0.220 / 0.385 | | 0.51 / 0.87 |
| repetition rate | 0.778 | 0.780 | | 0.49 |
| max identical-token run | 508 | 508 | | |

**Q4: Is the intended balancing behaviour reproduced?** Directionally yes, weakly.
- EBSS is more balanced than Probability-Only on both label metrics and in 18/24 layers.
- The advantage is largest at short prefixes (CV 0.45 → 0.42 at 32 tokens) and mostly
  gone by 128–512 tokens. Per-sequence σ is identical from position 128 onward (Fig. 2).
- Both sets are **far less balanced than random C4**: CV 0.25 vs 0.15.
  The paper's claim that EBSS data is more balanced than natural text is **not** reproduced here.

**Q5: How different are the outputs?**
- 3/16 sequences are token-identical (indices 4, 5, 14).
- The others diverge after a shared prefix.
- Per-sequence σ: EBSS lower in 7, higher in 6. The mean difference is dominated by
  two sequences, −0.036 and +0.034.

**Q6: Does the balance term materially affect selection?** Rarely.
- The kept beam set differs from the pure-probability ranking in 362/8160 steps (4.4 %).
- The median spread of the balance term across candidates (0.0012) is ~17× smaller than
  that of the length-normalised probability term (0.020).
- σ changes slowly once the prefix is long, so the term's influence decays with length.

**Q7: Lower-tail exposure?** Mixed. p10 improves by +11 tokens, but the minimum
exposure gets worse by −6. Neither method leaves any expert at zero exposure.
Both stay well below C4 Random (min 374, p10 451).

**Q8: All layers or only some?** CV is lower in 18/24 layers. The improvement is not uniform (Fig. 1).

**Q9: Is the data repetitive?** Yes, for both methods.
- distinct-1 ≈ 0.22 vs 0.51 for C4.
- Repetition rate ≈ 0.78.
- Longest repeated n-gram: mean 181 (PO) and 148 (EBSS) tokens.
- At least one sequence in each set is a 508-token single-token run.
- Without a repetition penalty, which the spec forbids adding, beam search on the base model collapses into loops.

**Q10: Are repetitive outputs linked to low PPL?** Yes. Model PPL on its own outputs is ≈ 1.2,
against 11.0 for C4. The low self-PPL is a symptom of the repetition loops, not evidence of
high-quality calibration text. The released official artifact showed the same pattern
(self-PPL 1.26).

**Q11: Does EBSS mitigate or worsen diversity collapse?** Neither, materially.
- distinct-2/3 are marginally higher for EBSS (+0.009 / +0.005).
- The mean longest repeat is shorter (148 vs 181).
- distinct-1 is marginally lower.
- The 508-token run occurs under both methods.

**Q12: Does model matching change the extreme imbalance of the released artifact?**
Partly, but the numbers are only roughly comparable because the budgets differ.

| | layers 1/8/16/23 load CV |
|---|---|
| released EBSS_qwen.jsonl (B=128) | 0.331 |
| FR (B=128) | 0.100 |
| this pilot's EBSS (B=16) | 0.244 |
| this pilot's PO (B=16) | 0.264 |
| C4 Random (B=16) | 0.133–0.174 |

- The released file is 3.3× less balanced than random. The model-matched pilot is ~1.6× less balanced than random.
- Model-matched generation is still clearly *less* balanced than natural text.
- Imbalance does not converge to the random level at small B, so a B=16 vs B=128
  comparison cannot fully separate the effect of the checkpoint from the effect of budget.

## 4. Quantization (Q13)

Setup:
- Frozen GPTQ: same code and sha256 as all earlier runs.
- Layers 1/8/16/23. E3 held-out relative expert reconstruction, mean over 240 experts.
- WikiText-2 PPL with only those layers quantized.
- B=16 × 512 for every config. 0 RTN fallbacks in all configs.

| config | recon | PPL |
|---|---|---|
| C4R_s0 | 0.0053818 | 9.2675 |
| C4R_s1 | 0.0053950 | 9.2689 |
| C4R_s2 | 0.0053815 | 9.2754 |
| Probability-Only | 0.0061027 | 9.2683 |
| EBSS | 0.0061308 | 9.2715 |

(BF16 PPL 9.2371.)

- **EBSS vs PO: +0.46 % worse.** EBSS is worse in every layer (+0.17…+0.80 %), and only 29 % of experts improve.
- **EBSS vs C4 mean: +13.8 % worse.** 0/240 experts improve, in every layer and every exposure decile.
- **PO vs C4 mean: +13.3 % worse.**
- PPL: PO and EBSS both fall inside the C4 seed range (9.2675–9.2754). PPL cannot separate the methods.

Self-generated, repetitive beam-search text is worse GPTQ calibration data than random C4
at B=16. The small balancing gain does not translate into better reconstruction.
This is consistent with the official-artifact result (+29 % at B=128) and with this project's
earlier finding that routing balance is not a sufficient proxy for reconstruction quality.
It does **not** show that lower balance STD causes worse reconstruction. The sets differ
mainly in text distribution, not in balance alone.

## 5. Systems (Q14–Q16)

Hardware: one 20 GB MIG slice. Experts are offloaded: non-expert weights stay
resident on the GPU (4.76 GB), and each layer's experts (1.04 GB) are streamed from
pinned host memory for every forward. GPU-seconds = wall time.

| | Prob-Only | EBSS |
|---|---|---|
| wall (= GPU-s) | 1264 s | 1338 s |
| emitted tokens/s | 6.48 | 6.12 |
| forward calls / avg rows | 512 / 63.9 | 512 / 63.9 |
| candidates scored | 4.95e9 | 4.95e9 |
| expert weight transfer | 998 s (79.0 %) | 1065 s (79.6 %) |
| model forward compute | 252 s (19.9 %) | 253 s (18.9 %) |
| balance-specific (stats + routing hook) | 1.3 s | 1.1 s |
| beam pruning / selection | 1.7 s | 11.2 s (0.84 %) |
| peak GPU / KV cache / peak CPU | 11.6 / 6.44 / 40.2 GB | 11.6 / 6.44 / 40.2 GB |

The extra +74 s for EBSS over PO is ~10 s of selection plus variation in the H2D transfer
rate (12.8 vs 12.0 GB/s).

- **Q14:** ~22 min of GPU time for 16 × 512 on this hardware. A linear extrapolation to
  B=128 gives ~3 h. Because the run is transfer-bound, batching more rows per forward would
  amortise the transfers, so the true B=128 cost would likely be lower. The paper's
  18 min (A800, B=128) was measured on different hardware and is not normalised against this.
- **Q15:** Under 1 %. Balance-specific computation (routing capture, statistics, scoring)
  is ~0.1 %. Including the larger EBSS selection step it is ~0.9 %.
- **Q16:** The expert weight transfer from host to GPU, at ~80 % of wall time, is the
  best optimisation target. It is an artifact of the 20 GB slice, not of EBSS.
  Options are larger batches per transfer or keeping hot experts resident.

## 6. Decision and next step (Q17)

Frozen label: **`MODEL-MATCHED-EBSS-BALANCING-SUPPORTED`**.
- QUALITY-GAIN failed: EBSS is not lower than PO, nor lower than any C4 seed.
- The balancing class was assigned because both paper σ and CV are lower than PO.

The categories kept separate in this report:

| category | outcome |
|---|---|
| Algorithm correctness | Established (tests, smoke gate, diagnostics gate). |
| Expert balancing | Small, prefix-concentrated improvement over PO. Still worse than random C4. |
| Generation quality | Poor for both methods (repetition loops, self-PPL ≈ 1.2). |
| Quantization quality | No gain. Worse than PO and much worse than C4. |
| Systems efficiency | Feasible, but dominated by offload transfers that are not intrinsic to EBSS. |

**Q17: Does the pilot justify full B=128 generation?** No. A scale-up would cost ~3 h of GPU
time to test a mechanism whose selection effect is 4.4 % of steps. Its data quality already
loses 13–14 % to random C4 at the same budget, and the official B=128 artifact lost 29 %.
**Recommended next step: stop EBSS development and retain the negative result.**
The measured failure is repetition collapse in beam-search generation, which is shared by
both methods. Fixing it would require new sampling or penalty mechanisms, which the protocol excludes.

## 7. Limitations

- One seed and one first-token draw. B=16 only. GPTQ covers four layers.
- Reconstruction is measured on held-out C4 activations, a C4-matched objective.
  This favours C4 calibration by construction.
- τ = 1.3 instead of 1.2. The rebuttal reports these as equivalent.
- Generation-time routing differs from B1 replay for 12–13 % of routed sets. This comes from BF16 batching effects.
- Nothing here establishes B=128 behaviour or A800-level efficiency.
