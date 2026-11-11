# Official MoEQuant EBSS calibration-quality validation (four layers, B = 128 × 512)

**Decision (rule frozen in `runs_ebss_official_validation/protocol.json` before any quality result): OFFICIAL-EBSS-TRANSFER-LIMITED**
(the generating checkpoint of the released file is UNDOCUMENTED). The quality class that would otherwise apply is
**OFFICIAL-EBSS-NO-CONSISTENT-GAIN**: neither the reconstruction criterion R nor the PPL criterion P holds.

## Question

MoEQuant (ICML 2025) proposes Expert-Balanced Self-Sampling (EBSS): calibration text the model generates itself, steered
toward balanced expert usage. The authors release `code/EBSS_data/EBSS_qwen.jsonl`.

Question: under our unchanged GPTQ pipeline, does that released file give lower held-out expert reconstruction error and lower
partial-model PPL than C4 Full Random (FR) and Typical Random (TR)? Full Coverage-LP (FC) is included as a negative control.

This tests the **released artifact used as calibration data**. It is not a reproduction of MoEQuant: there is no AGQ, no
rotation, and no regeneration.

## Source and provenance

- **File:** MoEQuant @ `d77ba1b`, `code/EBSS_data/EBSS_qwen.jsonl`.
  - 2,480,736 bytes, sha256 `691c7a50…253de9`.
  - The download equals the clone byte for byte, and the git blob sha1 matches.
  - The local copy is never edited.
- **Contents:** 600 records of raw `text`, with no token ids and no metadata.
  - Records 0–93 begin exactly with vocabulary token *i*, i.e. one self-generated continuation per start token.
  - Our Qwen1.5 tokenizer reproduces this pattern exactly, which confirms the id mapping.
- **Generating checkpoint: UNDOCUMENTED.**
  - For the base model: the paper cites `Qwen/Qwen1.5-MoE-A2.7B`, rebuttal Table R4 quotes record 0 as "Qwen-MoE-14B", and
    the file has no chat-template markers.
  - Against, or unclear: the repo's model directory is `qwen_moe_14b_chat`, the code's paths include a chat checkpoint, the
    authors describe "128 sequences of 512 length" while the file holds 600 records of up to about 2048 tokens, and the
    generation settings are not released.
  - This is why the rule returns TRANSFER-LIMITED.
- **The released EBSS loader is not runnable as published.** The text branch is gated on `--llmqat_calib`, which the argument
  parser never defines. It would also take the first 128 lines (only 55 have ≥ 512 tokens) and zero-pad them to 2048 with
  no attention mask.
- **Technical compatibility: passed.** The architecture matches the official Qwen code, all token ids are in vocabulary, and
  598/600 records decode–re-encode exactly. Only `max_position_embeddings` and `rope_theta` defaults differ in their config
  file; ours are the checkpoint's own.

## Data quality of the artifact (`official_dataset_audit.json`)

- **Lengths:** the median record has 2044 tokens, but 200/600 are under 128 tokens. 327 records have a full 512-token window.
- **Duplicates:** 47 records are exact duplicates (8 groups), and 40 first-512 windows are exact duplicates.
- **Repetition and diversity**, measured on the first-512 windows of long records:
  - Mean distinct-1 is 0.12, and 56 % of windows have distinct-1 below 0.1.
  - The mean repetition rate is 0.88. The median zlib ratio is 0.115.
  - Examples: record 5 is `&0.0000&0.0000…`, and record 127 is `×\n×\n…`.
- **Language:** 355/600 records contain CJK text, much of it Chinese multiple-choice exam items.

None of these properties were used to exclude or alter records.

## Extraction policy (frozen before any result)

- **Windows:** from each eligible record (≥ 512 tokens, file order), take the token window [0, 512).
  - No padding, no concatenation, and no window crosses a record boundary.
  - Exact-duplicate records are skipped (12 were skipped).
  - Windows with a 32-gram match to evaluation data would be excluded; there were none. One window has a 13-gram contact with
    held-out C4 and none with WikiText.
- **Result:** 266 records were scanned to obtain 128 windows (65,536 tokens) from 128 independent parent records.
- **One extraction for every configuration (CANONICAL-B1):** every calibration sequence, C4 or EBSS, is forwarded alone at
  batch size 1. The batch-composition effect found in `update_batch_sensitivity.md` therefore cannot differ between
  sources.
- **Fixed pipeline:** GPTQ 4-bit, g128, percdamp 1.0, act-order, routed experts of layers 1/8/16/23 (240 experts).
  - Reconstruction is scored on `~/moe_cache` test rows 2128–2255.
  - PPL is measured on the frozen 128 × 512 WikiText-2 windows.
  - BF16 PPL = 9.237090.

## Correctness gates (all passed)

- **All 10 configurations are valid:** 0 RTN fallbacks for C4 or EBSS, and repeated PPL evaluation is exact.
- **Quantization-pipeline regression:** FR s0 on the historical cache reproduces the partial-PPL weights, PPL and
  reconstruction exactly.
- **Extraction checks:** the extraction implementation regression passes. The batch-1 capture in a fresh process in reversed
  order is bitwise equal.
- **Repeat quantization:** FR s0 run again is bitwise equal (PPL 9.269004, recon 0.0046095).
- **Other checks:** BF16 PPL is exact, and the profiler expert-count cross-check has 0 mismatches.
- **Tests:** 30/30 `tests/test_official_ebss_validation.py` pass, and the full suite passes (`logs/pytest_all.log`).

## Results

### Reconstruction (primary)

| config | mean expert recon | EBSS relative penalty |
|---|---|---|
| FR s0/s1/s2 | 0.0046095 / 0.0046259 / 0.0046104 | +29.5 / +29.1 / +29.5 % |
| TR s0/s1/s2 | 0.0046120 / 0.0046000 / 0.0046051 | +29.5 / +29.8 / +29.7 % |
| FC s0/s1/s2 | 0.0047270 / 0.0047277 / 0.0047352 | +26.3 / +26.3 / +26.1 % |
| **EBSS** | **0.0059707** | — |

- **The EBSS penalty is large and uniform.**
  - The mean penalty is +29.4 % vs FR and +29.6 % vs TR.
  - **0 of 240 experts improve** against any baseline in any seed. The smallest per-expert penalty is +17 %, and the median is
    +29 %.
  - The penalty appears in every layer (L8 ≈ +24 %, L1 ≈ +34 %) and in every natural-exposure decile (+26–32 %). Even rarely
    routed experts, which balanced sampling is meant to help, get worse.
- **Comparison with FC:** EBSS is 12× further from FR than the FC negative control, whose penalty is about +2.5 %.
- **Caveat:** reconstruction is scored on held-out C4 activations, which matches the C4 baselines' distribution. This works
  against any non-C4 source. It is the same metric used in every prior study, and PPL on WikiText is the out-of-distribution
  check.

### PPL (secondary)

| | s0 | s1 | s2 |
|---|---|---|---|
| ΔPPL EBSS − FR | +0.0042 | +0.0007 | −0.0015 |
| ΔPPL EBSS − TR | +0.0021 | +0.0057 | +0.0140 |
| ΔPPL EBSS − FC | +0.0029 | +0.0058 | −0.0014 |

- **EBSS sits inside the C4 spread:** its PPL is 9.273237, ranked 8th of the 10 configurations, while the C4 range is
  9.2592–9.2747. EBSS is never below all three seeds of FR or TR, so P fails.
- **No PPL penalty:** the +29 % reconstruction penalty produces no clear PPL penalty either. Every |ΔPPL| is ≤ 0.014, the same
  scale as the calibration-grouping sensitivity measured earlier (up to 0.012 in `update_batch_sensitivity.md`; grouping SD
  0.0069 in `update_cross_seed_grouping.md`).
- **Consistent with earlier studies:** this repeats their pattern, where four-layer partial-model PPL barely responds to
  expert-reconstruction differences.

### Calibration diagnostics (means over layers 1/8/16/23)

| | min expert tokens | CV of expert load | routing L1 to C4 pool | atypical frac. | model PPL on calib text | distinct-1 |
|---|---|---|---|---|---|---|
| FR | ≈ 3400 | 0.10 | 0.03 | 0.37 | 11.2 | 0.51 |
| TR | ≈ 3310 | 0.12 | 0.05 | 0.02 | 12.3 | 0.53 |
| FC | ≈ 3630 | 0.08 | 0.09 | 0.65 | 9.2 | 0.47 |
| **EBSS** | **2167** | **0.33** | **0.25** | **0.96** | **1.26** | **0.13** |

- **Expert load: the released set is the *least* balanced** under our model. The coefficient of variation of expert load is
  3× FR's, and the least-exposed expert receives 36 % fewer tokens than under FR. This is the opposite of EBSS's stated goal.
  It is consistent with a set generated by a different checkpoint or with different settings, or with degenerate
  repetitive text concentrating routing.
- **Text: the windows are highly repetitive and very low in surprise.** The model's PPL on its own calibration text is 1.26,
  96 % of windows are atypical under the frozen typicality score, and 2 near-duplicate window pairs remain.
- **Interpretation:** this is the most likely mechanism for the reconstruction penalty. Hessians built from repetitive,
  far-from-C4 activations fit the held-out distribution poorly, despite equal token count.

### Efficiency

- **Measured calibration cost:** EBSS costs 338 s vs 345–361 s for C4. Tokenization, activation capture, Hessian
  accumulation and GPTQ cost about the same for every configuration.
- **Excluded:** the EBSS *generation* cost is unknown for this file and is not counted. The authors report about 18 min on an
  A800 for Qwen.

## Interpretation and limits

- **Main conclusion:** with this pipeline, the released `EBSS_qwen.jsonl` is a clearly worse calibration set than plain C4 for
  expert reconstruction (+29 %, in all 240 experts, every layer, every exposure decile). It is indistinguishable from C4 on
  WikiText partial-model PPL. No configuration-level evidence supports the claim that it improves calibration quality when
  transferred.
- **What this does not test:**
  - Model-matched EBSS generated by our exact checkpoint.
  - EBSS combined with AGQ and rotation, the setting in which MoEQuant reports its Qwen results.
  - A 2048-token or padded calibration protocol.
  - Full-depth quantization.
- **Repeatability:** this is one EBSS set, so there is no EBSS SD. Experts are not independent replicates.
- **Why the label is TRANSFER-LIMITED:** the provenance is undocumented, so the result says nothing about EBSS as a method.
  It only shows that the released artifact does not transfer to this setup.

## Files

- `runs_ebss_official_validation/`:
  - `summary.md` and `summary.json`
  - `decision.json`
  - `comparisons.json`, `reconstruction_results.json`, `ppl_results.json`
  - `routing_diagnostics.json`
  - `efficiency_results.json`
  - `figures/fig1–6`
  - provenance and audit: `source_provenance.json`, `official_dataset_audit.json`, `compatibility.json`
  - frozen protocol and manifests: `protocol.json`, `execution_policy.json`, `ebss_selection_manifest.json`, `baseline_manifest.json`
- Code:
  - `moe_calib/ebss_official_data.py`
  - `diagnostics/official_ebss_validation.py`
  - `diagnostics/report_official_ebss.py`
  - `scripts/run_official_ebss_validation.sh`
  - `tests/test_official_ebss_validation.py`
