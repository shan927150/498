# MoEQuant EBSS balance-strength fidelity experiment

**Primary (frozen rule): `BALANCE-NORMALIZATION-MATERIAL`.** **Secondary: `BALANCE-VARIANT-QUALITY-MIXED`.**

## The question

The paper's Eq. 4 and Algorithm 1 (line 30, annotated "▷ Eq. 4") disagree on the expert-balance STD constant:
- Eq. 4 uses 1/(E−1).
- Algorithm 1 uses E/(E−1).

They differ by √E = √60 = 7.746 for Qwen1.5-MoE-A2.7B. "Frequency of expert usage" is also never pinned down: it could be per-token rate c/n or assignment share c/(kn), a further factor of k = 4.

All earlier EBSS runs in this repo used **EQ4_RATE**. This study asks whether that reading choice matters for:
- beam pruning;
- the generated set;
- B=16 GPTQ.

It does **not** establish which reading the authors used.

## What was held fixed

The checkpoint, w = 4, τ = 1.3, the score sign, minimisation, cumulative logP, deferred parent σ, start tokens, token budgets and GPTQ were all unchanged. τ was not retuned, and no variant was chosen on reconstruction or PPL.

## Artifacts

All in `runs_ebss_balance_fidelity/`:
- frozen `protocol.json`, `normalization_definitions.json`, `source_ambiguities.json` and `source_equation_review.md`;
- result JSONs;
- `decision.json`, `summary.md`, `figures/fig1–7` and `logs/`.

Code:
- `moe_calib/ebss_generator.py` gains opt-in `balance_std_mode` / `usage_mode`. The default is bitwise unchanged; the pre-study copy is `logs/ebss_generator_pre_fidelity.py`.
- `diagnostics/ebss_balance_fidelity.py`
- `diagnostics/report_ebss_balance_fidelity.py`
- `runs_ebss_balance_fidelity/normalization_oracle.py`
- `scripts/run_ebss_balance_fidelity.sh`
- `tests/test_ebss_balance_fidelity.py` (46 tests)

## Variants

| ID | STD | usage | strength vs EQ4_RATE | equivalent τ (explanation only) |
|---|---|---|---|---|
| EQ4_RATE (historical) | Eq. 4 | c/n | 1 | 1.3 |
| EQ4_SHARE | Eq. 4 | c/(kn) | 0.25 | 5.2 |
| ALG1_SHARE | Alg. 1 | c/(kn) | 1.936 | 0.671 |
| ALG1_RATE | Alg. 1 | c/n | 7.746 | 0.168 |

---

## 1. Source fidelity

- **Eq. 4 vs Algorithm 1 is a genuine contradiction.** Algorithm 1 cites Eq. 4 but writes a constant √E larger, so at least one of the two contains a typo.
- **The sources only say "frequency".** That rules out raw counts (σ would grow with |S|) but does not choose between rate and share.
- **No author implementation exists.** The released code has no generator and no expert-frequency/STD code. OpenReview is not accessible.
- **Rebuttal Table R1** (DeepSeek, wikitext σ = 0.0427) implies a natural-text CV of:

  | variant | implied CV |
  |---|---|
  | EQ4_RATE | 0.46 |
  | EQ4_SHARE | 2.7 (implausible) |
  | ALG1_SHARE | 0.34 |
  | ALG1_RATE | 0.057 |

  It weakly disfavours EQ4_SHARE but cannot separate the other three. It is context only.
- **Correction to an earlier note.** The pilot `source_review.md` described Algorithm 1's factor as τ/E. It acts like τ/√E.

## 2. Correctness

- **Oracle.** An independent NumPy/`math.fsum` oracle covers all four definitions, the analytic ratios and the global ranking with stable tie-break. All 26 oracle tests pass.
- **Default path unchanged.** With options omitted, the generator is bitwise identical to the frozen pre-study generator on 16 synthetic cases: outputs plus every step's scores and decisions.
- **Historical pilot reproduced.** Pilot σ and final scores are reproduced to ≤ 5.6e-17 of the stored GPU values, and every pick matches.
- **Real-model checkpoint.** At |S| = 32, the 4 × 128 EQ4_RATE and PO beam states reproduce the score-audit 4 × 32 outputs token for token.
- **Invariances.** Probability-Only is identical under every variant's options. score − PO = σ/τ to 3.3e-16.
- **Production matches the oracle.** The oracle reproduces production pruning decisions on every captured trajectory:
  - 5 × 504 sentence-steps at 4 × 128;
  - 2 × 8,160 sentence-steps at 16 × 512.
- **How exactness is guaranteed.** Pools are the per-parent top-(w+1) candidates, which are provably sufficient for any per-parent σ.
- **Data integrity.** The B1 re-capture is compatible with the pilot cache. All generations are valid: shape, vocabulary, distinct sequences, no padding.
- **Test suite.** 46 new tests pass, and the full suite passes: 454 tests in 3 h 18 min (`runs_ebss_balance_fidelity/logs/pytest_all.log`).

## 3. Counterfactual rankings

This is local re-ranking of fixed candidate pools, **not** a trajectory prediction. D is the fraction of sentence-steps whose kept top-w set changes.

| variant | D vs EQ4_RATE (EQ4_RATE 4 × 128 pools) | D vs PO | median balance/probability spread | D vs EQ4_RATE on 16 × 512 pools (ALG1_SHARE / ALG1_RATE trajectories) |
|---|---|---|---|---|
| PO | 0.089 | 0 | 0 | 0.047 / 0.036 |
| EQ4_SHARE | 0.063 | 0.028 | 0.012 | 0.034 / 0.026 |
| ALG1_SHARE | 0.048 | 0.125 | 0.090 | 0.033 / 0.033 |
| ALG1_RATE | **0.282** | **0.315** | **0.359** | **0.219 / 0.183** |

- **Spread ratio.** The median probability spread is 0.035 at 4 × 128 and 0.012 at 16 × 512. At EQ4_RATE strength, the balance term is ~4.6 % of it.
- **Prefix length.** On the long pools, ALG1_RATE's change rate stays at 0.18–0.26 in every prefix-length bin, including 257–511. It does not fade the way the EQ4_RATE effect did in the pilot.
- **ALG1_SHARE.** It changes only ~3–5 % of decisions per step. Those small per-step changes compound over 511 steps (see §4).

## 4. Generation behaviour

Canonical B1 routing. Every variant is scored with the same EQ4_RATE metrics.

### B=16 × 512 (one generation draw per variant; no error bars)

| set | CV | min | p10 | distinct-2 | repetition | longest repeated n-gram | longest single-token run | self-PPL | routing L1 | atypical |
|---|---|---|---|---|---|---|---|---|---|---|
| PO | 0.261 | 300 | 391 | 0.376 | 0.778 | 181 | 508 | 1.206 | 0.210 | 0.88 |
| EQ4_RATE | 0.248 | 294 | 402 | 0.385 | 0.780 | 148 | 508 | 1.198 | 0.201 | 1.00 |
| **ALG1_SHARE** | **0.186** | 328 | 423 | 0.413 | 0.762 | 138 | 3 | 1.223 | 0.158 | 1.00 |
| **ALG1_RATE** | **0.166** | 330 | 437 | 0.421 | 0.748 | 152 | 3 | 1.222 | 0.142 | 0.94 |
| C4R s0/s1/s2 | 0.176/0.156/0.132 | 354–388 | 438–462 | 0.85–0.89 | 0.48–0.52 | 6–15 | 3 | 10.3–12.4 | 0.078–0.110 | 0.38–0.44 |

Relative to EQ4_RATE:
- **(Q6) Expert-usage CV improves.**
  - ALG1_SHARE −25 %, ALG1_RATE −33 %.
  - ALG1_RATE falls inside the C4 Random seed range.
  - The pilot's "EBSS less balanced than C4" finding is therefore specific to EQ4_RATE.
- **(Q7) Minimum exposure improves:** +11.5 % / +12.3 %, still below C4.
- **(Q8) p10 exposure improves:** +5.2 % / +8.5 %. ALG1_RATE reaches C4 s0.
- **(Q9) Repetition collapse is largely unchanged.**
  - distinct-2 rises 0.385 → 0.41–0.42, against 0.85–0.89 for C4.
  - Repetition rate 0.78 → 0.75–0.76.
  - Repeated n-grams of ~140–150 tokens remain.
  - Self-PPL stays at 1.22, against ~10 for C4.
  - The one visible change: the 508-token single-token run present in both PO and EQ4_RATE disappears (max run 3).
- **(Q10) Routing representativeness improves partially.**
  - Routing L1 −21 % / −29 %, but still 1.3–1.8× C4.
  - The atypical fraction stays at 0.94–1.0, against ~0.4 for C4.

### Short stage (4 × 128): direction is unreliable

- The short stage is a single draw. ALG1_SHARE's short-run CV was +7.8 % *above* EQ4_RATE, the opposite sign to B=16.
- The PO − EQ4_RATE short-run CV difference was 24.6 %.
- The short stage was used only for the frozen gate. Its direction is not evidence.
- At 128 tokens the 512-token atypicality reference is uninformative: every set, including C4, scores 1.0.

## 5. Quantization

B=16 GPTQ on layers 1/8/16/23. Held-out expert reconstruction; WikiText-2 PPL.

| config | recon | vs EQ4_RATE | vs PO | vs C4 mean | experts better than EQ4_RATE | PPL |
|---|---|---|---|---|---|---|
| C4R s0 / s1 / s2 | 0.0053818 / 0.0053950 / 0.0053815 | | | | | 9.2675 / 9.2689 / 9.2754 |
| PO | 0.0061027 | −0.46 % | — | +13.4 % | | 9.2683 |
| EQ4_RATE | 0.0061308 | — | +0.46 % | +13.8 % | | 9.2715 |
| **ALG1_SHARE** | 0.0060830 | **−0.78 %** | −0.32 % | **+12.9 %** | 75 % (4/4 layers) | 9.2769 |
| **ALG1_RATE** | 0.0061072 | **−0.38 %** | +0.07 % | **+13.4 %** | 64 % | 9.2744 |

- **(Q11) Reconstruction improves only marginally.**
  - ALG1_SHARE clears the frozen 0.5 % margin against EQ4_RATE, at −0.78 %. ALG1_RATE does not.
  - Each variant is one generation draw, and the EQ4_RATE − PO gap is itself 0.46 %. So −0.78 % is not clearly beyond generation-to-generation variation.
- **(Q12) No variant comes close to C4 Random.**
  - Both are 12.9–13.4 % worse than the C4 mean.
  - 0/240 experts are better than any C4 seed.
- **(Q13) Reconstruction and PPL rankings disagree.**
  - The reconstruction order is ALG1_SHARE < PO < ALG1_RATE < EQ4_RATE.
  - The PPL order is PO < EQ4_RATE < ALG1_RATE < ALG1_SHARE.
  - All PPLs sit within ~0.009 of each other, at the scale of the measured PPL noise (accumulation-order SD ≈ 0.002, C4 seed spread 0.008). ALG1_SHARE is 0.0015 above the worst C4 seed.
- **(Q14) Balance does not translate monotonically into quality.**
  - ALG1_RATE is the most balanced (CV 0.166, at C4 level) but reconstructs worse than ALG1_SHARE, and no better than PO.
  - Matching C4's balance leaves ~13 % of the C4 gap intact.
  - The gap therefore cannot be explained by expert imbalance. The remaining distinguishing factor in these data is text diversity/typicality.

Label rule (frozen): MIXED, because a variant is ≥ 0.5 % below EQ4_RATE but none is below the C4 seed mean.

## 6. Systems

Wall time is on one 20 GB MIG slice, so GPU-seconds = wall.

The new runs carry a read-only pool-capture hook, which the old runs did not. It is reported separately: 705 s for ALG1_SHARE and 1,039 s for ALG1_RATE.

| run | wall excl. capture | vs EQ4_RATE pilot (1,338 s) | tokens/s excl. capture |
|---|---|---|---|
| ALG1_SHARE | 1,370 s | +2.4 % | 6.0 |
| ALG1_RATE | 1,377 s | +2.9 % | 6.0 |

- **Memory.** Peak GPU 11.6 GB (pilot 11.6 GB); peak CPU 40.2 GB.
- **(Q15) Normalisation overhead.** About 32–39 GPU-s per 16 × 512 set over EQ4_RATE. Balance statistics, scoring and selection together take < 1 % of wall time, so the overhead is mostly run-to-run variation, not the normalisation.
- **(Q16) The bottleneck is still expert-weight host→device transfer.**
  - It is ~70 % of wall time excluding capture (~1,000 s per set).
  - Model compute is ~18 %.
- **(Q17) No quality gain is large enough to justify systems optimisation.**
  - The best gain is 4.8e-5 absolute reconstruction (−0.78 %) for ~32 extra GPU-s per set.
  - That gain is within draw noise and ~17× smaller than the gap to C4 Random, which costs no generation at all.

## 7. Scientific decision

- **Primary: `BALANCE-NORMALIZATION-MATERIAL`.**
  - At 4 × 128, ALG1_RATE passes G1 (D = 0.282) and G4 (distinct-2 +0.142, longest repeated n-gram −79 %).
  - EQ4_SHARE passes G1, G2, G3 and G5. ALG1_SHARE passes G4.
  - At B=16 both Algorithm-1 variants pass the same thresholds (CV −25 % / −33 %).
  - This label says the reading choice changes the generated set. It does not say the method is better.
- **Secondary: `BALANCE-VARIANT-QUALITY-MIXED`.**
  - ALG1_SHARE: −0.78 % vs EQ4_RATE, +12.9 % vs C4.
  - ALG1_RATE: −0.38 % vs EQ4_RATE, +13.4 % vs C4.
  - PPL is unchanged within noise.
- **Not run at B=16: EQ4_SHARE.** It qualified, but the protocol allows at most two new variants and gives the Algorithm-1 variants priority. Its counterfactual and short-stage behaviour places it between PO and EQ4_RATE (weaker balancing).

## 8. Previous conclusions

**Remain valid:**
- SCORE-SIGN-CORRECT. Normalisation does not touch sign or direction.
- EBSS-family sets reconstruct ~13 % worse than C4 Random with 0/240 experts improved. This now holds for PO and three balance strengths spanning 7.7× (1, 1.94, 7.75).
- EBSS text is highly repetitive and atypical: self-PPL ≈ 1.2, distinct-2 ≈ 0.4.
- PPL differences between all B=16 calibration sets are at noise scale.
- Expert-weight transfer dominates generation cost.
- OFFICIAL-EBSS-TRANSFER-LIMITED is unaffected (different data).

**Must be narrowed** (they hold for the EQ4_RATE reading only):
- "MODEL-MATCHED-EBSS-BALANCING-SUPPORTED" and its magnitudes: σ −4.9 %, CV 0.248, "less balanced than C4 Random", and "the balance effect fades after ~128 tokens".
- Under Algorithm-1 readings, EBSS balances to C4 level, and its per-step pruning effect persists through 512 tokens.
- "Weak balancing" was a property of the reading, not of EBSS as published.

## 9. Next experiment (recommendation; not executed)

Pursue **diversity-rescued generation**, framed as a diagnostic, not a MoEQuant reproduction.

- **Why balance is ruled out.** ALG1_RATE matches C4 expert balance yet still reconstructs ~13 % worse. Expert imbalance is therefore ruled out as the cause of the EBSS–C4 gap at this budget.
- **What remains.** The remaining large, measured difference is text diversity and typicality: distinct-2 0.42 vs 0.87, self-PPL 1.2 vs 10, atypical fraction 0.94 vs 0.4.
- **Why not reproduce at full scale.** A full B=128 / full-depth EBSS reproduction would be expensive and would inherit the same unresolved reading ambiguity. These data give no sign it would close a 13 % gap.

If diversity rescue also fails to approach C4 Random, stop EBSS development.
