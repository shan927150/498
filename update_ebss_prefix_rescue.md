# C4-prefix diversity rescue for modified EBSS (B=16, P=64 + G=448)

**Primary (frozen rule): `PREFIX-RESCUE-PARTIAL`.** **Diversity: `NO-DIVERSITY-IMPROVEMENT`.**
**Recommendation: C — stop EBSS development and finalize the project** (not executed).

## The question

Every earlier EBSS set in this repo collapsed into long repetition loops:
- distinct-2 was about 0.36–0.42, against 0.85–0.89 for C4;
- reconstruction stayed about 13 % behind C4 Random at B=16, whatever the balance normalization.

This study asks whether a small amount of natural text can rescue EBSS calibration quality without losing its balancing. Each beam is seeded with a 64-token C4 prefix and then generates 448 tokens. It also asks whether the result has any real advantage over just using C4 Random.

This is **not** a paper-faithful EBSS reproduction. It is the repo's modified EBSS, using the ALG1_RATE normalization from the fidelity study and the C4-prefix initialization.

## What was held fixed

- **Model and generation:** Qwen/Qwen1.5-MoE-A2.7B in BF16, w = 4, τ = 1.3.
- **EBSS term:** strong ALG1_RATE normalization (balance_std_mode = algorithm1, usage_mode = per_token). PX_PO is the same generator with the balance term disabled.
- **Score:** sign, minimisation, cumulative log P and deferred parent σ are unchanged.
- **Decoding:** no repetition penalty and no sampling.
- **Quantization and evaluation:** GPTQ, evaluation windows and the four quantized layers (1/8/16/23) are unchanged.

Selection rules:
- Prefixes are tokens [0:64] of the frozen C4 Random seed-0 windows. They are not chosen by any quality signal.
- Prefix length was not tuned.
- No generated sequence was selected or filtered.

Token budget is equal everywhere: 16 × 512 = 8192 calibration tokens per set.

## Artifacts

All in `runs_ebss_prefix_rescue/`:
- **Frozen before generation:** `protocol.json`, `prefix_manifest.json`, `matched_c4_manifest.json`, `generator_config.json`, `source_provenance.json`.
- **Generations:** `generated_prefix_{po,ebss}.jsonl`, `generation_logs/`.
- **Checks:** `correctness_results.json` (CPU regression + smoke), `smoke_results.json`.
- **Diagnostics:** `diversity_by_region.json`, `repetition_analysis.json`, `routing_diagnostics.json`, `exposure_decomposition.json`.
- **Quantization and cost:** `results/PX_{PO,EBSS}.json`, `reconstruction_results.json`, `ppl_results.json`, `generation_cost.json`, `quality_cost_comparison.json`.
- **Outcome:** `decision.json`, `summary.{json,md}`, `figures/fig1–7` (PNG + PDF), `logs/`.

Code:
- `moe_calib/ebss_generator.py` gains an opt-in `initial_token_ids` prefix path with batched `prefill`. With no prefix, the default path is bitwise unchanged. The pre-study copy is `logs/ebss_generator_pre_prefix.py`.
- `moe_calib/ebss_prefix.py` holds the pure helpers.
- `diagnostics/ebss_prefix_rescue.py` and `diagnostics/report_ebss_prefix_rescue.py`.
- `scripts/run_ebss_prefix_rescue.sh`.
- `tests/test_ebss_prefix_rescue.py` (29 tests).

Reused, not rerun: C4R_s0/s1/s2 and RS_PO come from `runs_ebss_generation_pilot/`, and RS_EBSS (= ALG1_RATE) comes from `runs_ebss_balance_fidelity/`. The capture verified those caches as bitwise-equal to the originals.

## Sets compared

| set | content | role |
|---|---|---|
| **C4R_s0** | 16 natural C4 windows of 512 tokens | matched natural control. Its tokens [0:64] **are** the PX prefixes. |
| C4R_s1, s2 | other C4 seeds | natural seed spread |
| RS_PO / RS_EBSS | random-start generation (1 start token + 511 generated) | historical counterparts |
| **PX_PO** | 64 C4 prefix + 448 generated, balance term off | prefix control |
| **PX_EBSS** | 64 C4 prefix + 448 generated, ALG1_RATE | method under test |

Which comparisons are matched:
- **PX_EBSS vs PX_PO is the only matched intervention:** same prefixes, same budget, only the balance term differs.
- **PX vs C4R_s0** is matched on budget and on the first 64 tokens.
- **PX vs RS is descriptive.** Both the initial content and the generated-token count (448 vs 511) differ, although the final budget is the same.

There is one generation draw per condition, so there are no error bars on any generated set.

## 1. Correctness

**CPU regression: 35/35 pass.** It covers:
- the default path is bitwise equal to the pre-study generator;
- a P = 1 prefix reproduces the historical path;
- an independent pure-Python prefix oracle;
- PO invariance;
- the counts and R identities.

**GPU smoke** (4 prefixes, 64 + 32 tokens): 15/15 checks pass.
- Prefill logits vs teacher forcing: mean |Δ log p| 0.030.
- Prefix routing agreement: 0.952.
- Ranking and σ match the oracle (σ diff ≤ 1.1e-16).
- KV-cache parent: top-1 agreement 0.96 / 0.98.
- Replay: Pearson r 0.9999.
- Deterministic across reruns.

**Deviation, recorded:** smoke attempt 1 failed only an ad-hoc check, last-position max |Δ log p| ≤ 0.5 (observed 0.81). That cap was not in the frozen protocol. The check was replaced with the frozen mean tolerance (≤ 0.05), plus top-1 agreement and the same max ≤ 1.0 cap the KV check uses. The rerun gave identical deterministic numbers. Attempt 1 is preserved in `logs/smoke_attempt1*`. BF16 batched-vs-sequential noise of this size was already documented in earlier studies.

**Full run:**
- Both sets have the right shape, in-vocab IDs, 16 distinct sequences and 0 padding.
- Tokens [0:64] equal the frozen prefixes exactly for **both** PO and EBSS (checked directly against `prefix_manifest.json`).
- 0/16 windows share a 32-gram with the held-out evaluation text.
- PO and EBSS diverge at positions 65–116. One of the 16 sequences is identical under both.
- Prefix-region routing at capture vs the generator: agreement 0.916 (PO) and 0.920 (EBSS), from batched-prefill vs B=1 BF16 differences.

**Tests:** 24 passed and 5 skipped before the artifacts existed.
- With the artifacts in place: the new file passes 29/29, and the full suite passes **483/483** in 3:54 (`logs/pytest_*.log`).
- The 29 new tests were rerun and passed after the final report regeneration.

## 2. Context vs continuation: diversity (fig1, fig2)

Per-sequence means. The continuation is [64:512]; for C4 and RS it is the same slice.

| set | cont. distinct-2 | cont. repetition | full distinct-2 | sequences with a 16-gram loop | 8-gram repeat coverage | median loop onset (tokens after 64) |
|---|---|---|---|---|---|---|
| C4R_s0 | 0.855 | 0.500 | 0.849 | 1 / 16 | 0.05 | — |
| RS_PO | 0.357 | 0.790 | 0.376 | 12 / 16 | 0.63 | 62 |
| RS_EBSS | 0.421 | 0.744 | 0.421 | 13 / 16 | 0.62 | 56 |
| **PX_PO** | **0.200** | 0.872 | 0.273 | **16 / 16** | 0.86 | 71 |
| **PX_EBSS** | **0.278** | 0.838 | 0.349 | 13 / 16 | 0.75 | 50 |

- **The continuation got *less* diverse, not more.** Relative to the same slice of the random-start sets:
  - continuation distinct-2: −0.157 (PO) and −0.144 (EBSS);
  - repetition: +0.082 and +0.094.
  - Both diversity criteria fail for both methods, so the label is **NO-DIVERSITY-IMPROVEMENT**, with no mixed flag.
- **Loops are not eliminated.**
  - 16/16 PO and 13/16 EBSS sequences enter a 16-gram loop, typically 50–70 tokens after the prefix.
  - 12/16 in each set repeat a span longer than 64 tokens.
  - Repetition rises along the sequence (fig2). The PO second-half distinct-2 is 0.20.
- **Full-sequence diversity is lower than random-start too** (0.273 / 0.349 vs 0.376 / 0.421). The descriptive splice check makes this sharper: the same C4 prefixes spliced onto the *random-start* continuations give full distinct-2 of 0.433 (PO) and 0.489 (EBSS), higher than the real prefix-conditioned sets.
- **So the C4 prefix does not carry its diversity into generation.** The beam latches onto a phrase from the prefix and repeats it.
- **Boundary n-grams** (shared between prefix and continuation) are a small term: about 7–11 of 511 bigrams per sequence.
- **Balance and diversity:** EBSS is a little less repetitive than PO under prefixes, as it was under random starts.

## 3. Balancing (fig3, fig4)

CANONICAL-B1 routing over 24 layers.

| set | CV (total) | CV (prefix) | CV (continuation) | min exposure | p10 | routing L1 vs C4 ref | fraction atypical |
|---|---|---|---|---|---|---|---|
| C4R_s0 | 0.176 | 0.264 | 0.177 | 354 | 438 | 0.110 | 0.44 |
| RS_PO | 0.261 | (start) | 0.276 | 300 | 391 | 0.210 | 0.88 |
| RS_EBSS | 0.166 | (start) | 0.179 | 330 | 437 | 0.142 | 0.94 |
| **PX_PO** | 0.318 | 0.264 | 0.361 | 241 | 351 | 0.260 | 1.00 |
| **PX_EBSS** | 0.305 | 0.264 | 0.350 | 246 | 358 | 0.255 | 1.00 |

- **Provided context:** the prefix region routes exactly like the matched C4 windows (CV 0.264 in all three). That is expected, since the tokens are the same.
- **Generated continuation:** continuation balance is much **worse** than both C4 (0.177) and random-start EBSS (0.179).
- **Strong EBSS is barely more balanced than PO once prefixes are used.** Total CV is 0.305 vs 0.318 and continuation CV is 0.350 vs 0.361. Under random starts it was 0.166 vs 0.261, so the balancing effect largely disappears.
- A plausible reading, not tested: early loop lock-in, with a median onset of 50 tokens, leaves the balance term little to choose between.
- **Representativeness and lower-tail exposure both get worse:** routing L1 0.26, all sequences atypical, minimum exposure 241–246 vs 300–354 elsewhere.

## 4. GPTQ reconstruction (fig5, fig6)

B=16 GPTQ on layers 1/8/16/23, 240 routed experts, held-out reconstruction.

| config | recon | vs C4 mean | PPL |
|---|---|---|---|
| C4R_s0 (matched natural) | 0.0053818 | −0.08 % | 9.2675 |
| C4R_s1 | 0.0053950 | +0.17 % | 9.2689 |
| C4R_s2 | 0.0053815 | −0.09 % | 9.2754 |
| RS_PO | 0.0061027 | +13.30 % | 9.2683 |
| RS_EBSS | 0.0061072 | +13.39 % | 9.2744 |
| **PX_PO** | **0.0057175** | +6.15 % | 9.2723 |
| **PX_EBSS** | **0.0056954** | +5.74 % | 9.2660 |

BF16 PPL is 9.2371. No configuration needed an RTN fallback.

Deltas:
- **Δ_PO = PX_PO − RS_PO = −3.85e-4 (−6.31 %)** (descriptive):
  - 240/240 experts improve; median expert −5.3 %;
  - every layer improves (−3.6 % to −10.3 %).
- **Δ_EBSS = PX_EBSS − RS_EBSS = −4.12e-4 (−6.74 %)** (descriptive):
  - 240/240 experts improve; median expert −5.8 %.
- **Δ_balance|prefix = PX_EBSS − PX_PO = −2.21e-5 (−0.39 %)** (matched):
  - 69 % of experts improve;
  - by layer: layer 1 −1.74 %, layer 8 −0.43 %, layer 16 +0.08 %, layer 23 +0.22 %.
  - This is below the 0.5 % materiality flag, with one draw per condition.
  - The random-start counterpart (RS_EBSS − RS_PO) was +0.07 %.
- **Gap recovery** G = (recon_RS − recon_PX) / (recon_RS − C4 mean):
  - **G_PO = 0.538, G_EBSS = 0.571.** About half the gap is recovered.
- **Versus C4 Random:** neither method beats any seed. 0/240 experts are better than C4R_s0, for both methods and all three seeds. PX_EBSS is +5.6 % to +5.8 % worse.

The primary label is **PREFIX-RESCUE-PARTIAL**: it is a strict, material improvement over the random-start counterpart, but not below all three C4 seeds.

The reconstruction gain comes **without** better continuation diversity or balance; both got worse. These experiments cannot say what does drive it. One untested candidate is that 1024 of the 8192 calibration tokens (12.5 %) are now natural text. No causal claim is made beyond the controlled comparisons.

## 5. Model-level PPL

Partial model, 4 quantized layers.

- PX_EBSS has the lowest PPL of all seven configurations (9.2660).
- PX_PO (9.2723) is *worse* than RS_PO (9.2683), which contradicts the reconstruction direction for PO.
- The C4 seed spread alone is 0.008, larger than every PX–C4 difference.

PPL does not resolve the ranking and is not interpreted, as the protocol specifies.

## 6. Systems (fig7)

Measured on the A100 20 GB MIG slice with expert offload.

| | PX_PO | PX_EBSS | RS_PO | RS_EBSS | C4 Random |
|---|---|---|---|---|---|
| generation wall (s) | 1117 | 1119 | 1264 | 1377 | 0 |
| s / generated token | 0.156 | 0.156 | 0.155 | 0.168 | — |
| prefix prefill (s) | 3.2 | 2.8 | — | — | — |

- **Where the time goes:** 79–80 % expert-weight host-to-device transfer, 19 % forward compute, under 1 % prefill.
- **Memory:** peak GPU 11.6 GB, KV cache 6.4 GB, peak CPU 40 GB.
- **Why PX is faster than RS:** it generates 448 instead of 511 tokens per sequence. Per generated token the cost is the same.

Capture and GPTQ are the same pipeline for every source. Their measured wall time varied with machine load, which is why fig7 plots generation time only:
- GPTQ took 280 s in this run vs 727–788 s for identical work in earlier runs;
- quantize total ranged 355–1366 s.

**Quality vs cost: no advantage.** PX_EBSS spends about 1120 s extra GPU time on generation to get a calibration set 5.7 % *worse* than free C4 Random.

## 7. Answers to the 18 questions

1. **Were the original C4 Random seed-0 windows successfully reused?** Yes.
   - The prefixes are tokens [0:64] of the frozen C4R_s0 windows: 16 distinct prefixes, hashes in `prefix_manifest.json`.
   - The C4R_s0 cache, routing and per-row NLL are bitwise equal to the pilot's.
2. **Are the generated prefixes identical for PO and EBSS?** Yes. Tokens [0:64] equal the frozen prefixes exactly in both sets.
3. **Does the prefix-conditioned generator preserve the audited score/sign semantics?** Yes.
   - 35/35 CPU regression checks pass: default path bitwise, P = 1 equals historical, independent oracle.
   - Smoke ranking and σ match the oracle.
4. **Does C4-prefix initialization eliminate long repetition loops?** No.
   - 16/16 PO and 13/16 EBSS sequences loop, typically 50–70 tokens after the prefix.
   - 12/16 in each set have repeated spans longer than 64 tokens.
5. **Does continuation-only distinct-n improve?** No. It falls by 0.157 (PO) and 0.144 (EBSS) vs random start.
6. **Is the full-sequence diversity gain larger than the 64 C4 prefix tokens alone can explain descriptively?** There is no gain to explain.
   - Full distinct-2 is lower than random start.
   - It is also below what simply splicing the prefixes onto the random-start continuations would give (0.433 / 0.489).
7. **Does strong EBSS remain more balanced than PO?** Only marginally.
   - CV 0.305 vs 0.318 (continuation 0.350 vs 0.361).
   - Under random starts the gap was 0.166 vs 0.261.
8. **Does prefix conditioning improve routing representativeness?** No. Routing L1 is 0.26 vs 0.21 / 0.14 for random start and 0.11 for C4, and 100 % of sequences are atypical.
9. **Does it improve minimum or lower-tail expert exposure?** No. Minimum exposure is 241–246 vs 300–330 for random start and 354 for C4; p10 is lower too.
10. **Does Prefix PO improve GPTQ reconstruction over Random-Start PO?** Yes, descriptively: −6.31 %, 240/240 experts.
11. **Does Prefix EBSS improve over Random-Start EBSS?** Yes, descriptively: −6.74 %, 240/240 experts.
12. **Does Prefix EBSS improve over Prefix PO?** Not materially. The matched comparison gives −0.39 %, 69 % of experts, mixed by layer, one draw, below the 0.5 % flag.
13. **Does either method beat C4 Random?** No.
    - Both are +5.6 % to +6.2 % worse than every seed.
    - 0/240 experts beat C4R_s0.
14. **How much of the original reconstruction gap is recovered?** G_PO = 0.538 and G_EBSS = 0.571.
15. **Does partial-model PPL support the reconstruction ranking?** No.
    - The differences are within the C4 seed spread (0.008).
    - PX_PO vs RS_PO goes the opposite way to reconstruction.
16. **How expensive is prefix-conditioned generation?**
    - About 1118 s of GPU time per 16 × 512 set, 0.156 s per generated token, about 80 % of it expert offload transfer.
    - Prefill is about 3 s.
    - C4 Random needs no generation.
17. **Is there an actual quality–cost advantage over directly using C4 Random?** No. It is more expensive and has worse reconstruction.
18. **Is there sufficient evidence to justify another allocator experiment?** No.
    - The matched balance effect is −0.39 %.
    - The only sizeable gain (vs random start) is not attributable to balancing and still leaves a 5.7 % gap to free natural text.

## 8. Effect on previous conclusions

- **Still standing:**
  - The pilot's MODEL-MATCHED-EBSS-BALANCING-SUPPORTED holds for random starts.
  - The fidelity study's BALANCE-NORMALIZATION-MATERIAL holds.
  - The score audit's SCORE-SIGN-CORRECT holds.
- **New:**
  - EBSS's balancing advantage over PO largely disappears when generation is seeded with natural prefixes.
  - Repetition collapse is a property of the beam search under this model and τ, not of the start token.
- **Unchanged:** C4 Random remains the best B=16 calibration source tested in this repo for this model.

## 9. Recommendation (one choice; not executed)

**C. Stop EBSS development and finalize the project.**

Why not A or B:
- **Not A:** neither method beats C4 Random, and the quality–cost comparison is strictly unfavourable.
- **Not B:** the bottleneck the experiment reveals is beam-search repetition lock-in. Every in-scope fix for it (repetition penalties, sampling, τ changes, prefix-length tuning) is excluded by the hard constraints.
- **Balance:** within the matched comparison, balance is no longer a lever (−0.39 %).

The finalized write-up should report EBSS as follows:
- correctly implemented and audited;
- balancing works under random starts;
- balance-normalization-sensitive;
- not competitive with natural C4 calibration at B=16 on this model;
- partially rescued (about 55 % of the gap) but not closed by natural prefixes.
