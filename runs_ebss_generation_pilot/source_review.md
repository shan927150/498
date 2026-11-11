# Source review: MoEQuant EBSS (what each implementation decision rests on)

## Sources

- **paper**: MoEQuant, ICML 2025, PMLR 267:8245-8260 (https://proceedings.mlr.press/v267/chen25aa.html); text read from arXiv:2505.03804v1 HTML (https://arxiv.org/html/2505.03804)
- **algorithm_1_image**: github.com/chenzx921020/MoEQuant @ d77ba1b asserts/Algorithm_1_EBSS.png (rebuttal supplement, read directly)
- **rebuttal_R1**: Tables_for_R1_eTCn.md (Table R2: tau sweep; Table R3: AGQ c; Table R4: AGQ gate layer)
- **rebuttal_R2**: Tables_for_R2_NkUk.md (Table R3: time cost GPTQ vs MoEQuant, '128 sequences of 512 length', batching; Table R7)
- **rebuttal_R3**: Tables_for_R3_dN5E.md (Table R1 expert-balance STD by calibration set; Table R2 tau; Table R3 width w; Table R6 EBSS time; Algorithm 1 image)
- **rebuttal_R4**: Tables_for_R4_ZiDj.md (Table R3 time cost; Table R4 generated-sample snapshot)
- **openreview_note**: https://openreview.net/forum?id=0epuNvt5Dj&noteId=aKw8ICecFU -- NOT ACCESSIBLE (bot verification interstitial; API endpoints redirect to a challenge). Its content is not used and not inferred.
- **official_code**: the released repository contains NO EBSS generation code (only the EBSS_data/*.jsonl outputs and a calibration loader)

## Decisions

| item | implementation | source | status |
|---|---|---|---|
| objective | minimise -(1/N) sum log P + sigma/tau (Eq. 9-10) | paper Eq. 9-10 | documented |
| pruning score | score(S||v) = -(R_S + log P(v|S))/(l+1) + sigma(M,S)/tau | paper Eq. 13; Alg. 1 line 14 | documented |
| score direction | keep the w SMALLEST scores; final = argmin Score(c) | Eq. 9-10 argmin + minus sign of Eq. 13; Eq. 14 text 'maximize' and Alg. 1 line 20 'arg max' conflict | resolved (inconsistency) |
| length normalisation | divide by l+1 (= tokens of the candidate incl. the first token) | Eq. 13; Alg. 1 line 14 (j+1) | documented |
| balance uses prefix or candidate | prefix S ('deferred expert balance': the pruning metric excludes v) | paper Sec. 4.2 text + Eq. 13 sigma(M,S); Alg. 1 line 14 writes sigma(M,new_candidate) | resolved (inconsistency; v's routing is unknowable without a forward of v, the paper text is explicit) |
| sigma definition | sigma = mean over all MoE layers of sigma_l; sigma_l = sqrt(1/(E-1) sum (u - mean)^2) | paper Eq. 3/4 | documented (Alg. 1 line 30 writes E/(E-1): typo-level conflict, paper equations used) |
| usage frequency u | routed tokens of the branch per expert / tokens of the branch | rebuttal R3 Table R1 scale (0.0427 wikitext2 / 0.0052 EBSS on DeepSeek-MoE-16B) | ASSUMPTION |
| routed counts vs affinities | top-k membership counts | paper 'frequency of expert usage' | assumption (consistent) |
| tau meaning | divides sigma (expert-balance coefficient, called 'temperature'); never applied to logits | Eq. 13; Alg. 1 input list 'Expert balance temperature'; App. A.1 'controls the significance of expert balance' | documented |
| tau value | 1.3 (task protocol) | rebuttal R2/R3 Table R2 Qwen: 1.2 -> 49.60, 1.3 -> 49.61; paper sets 1.2 | protocol choice (paper default is 1.2) |
| w | 4 | paper Sec. 5.2 / App. Table 7; rebuttal R3 Table R3 | documented |
| first token | Random(V) | Alg. 1 line 4 | documented; distribution and seed undocumented -> uniform over non-special ids, seed 20261110 (ASSUMPTION) |
| number / length of sequences | 128 x 512 (full budget); this pilot 16 x 512 | rebuttal R2 Table R3, R3 Table R6 ('128 sequences of 512 length'); paper does not state it | rebuttal clarification |
| stop rule / EOS | none: Seqlen-1 expansion steps over all of V | Alg. 1 lines 6-18 | documented by omission |
| batching | sentences stitched along the batch dimension | rebuttal R2 Table R3 / R3 Table R6 | rebuttal clarification |
| generation hardware / time | A800, Qwen-MoE-14B 18 min (128 x 512) | rebuttal R3 Table R6 | reference only |
| checkpoint | Qwen/Qwen1.5-MoE-A2.7B base @ 1a758c50 | paper cites huggingface.co/Qwen/Qwen1.5-MoE-A2.7B; task protocol | documented for the paper; the released file's generator stays undocumented |

## Ambiguities and their resolution

- **score_direction**: Eq. 14 defines argtopk as maximising and Alg. 1 line 20 uses arg max, while Eq. 9-10 minimise and Eq. 13 carries a leading minus (negative log-likelihood + imbalance). Resolved: minimisation everywhere.
- **balance_prefix_vs_candidate**: Alg. 1 line 14 writes sigma(M, new_candidate); Eq. 13 writes sigma(M, S) and Sec. 4.2 describes deferred expert balance that excludes v. Resolved: prefix S.
- **sigma_scale**: Alg. 1 line 30 has sqrt(E/(E-1) sum ...), Eq. 4 has sqrt(1/(E-1) sum ...). Eq. 4 used (the factor E only rescales sigma, i.e. it would act like tau/E).
- **usage_frequency_normalisation**: per-token rate (counts / tokens) vs share (counts / (tokens * k)) is not stated; the share would shrink the balance term by k = 4 (equivalent to tau x 4). Per-token rate adopted (rebuttal STD scale).
- **eq12_missing_log**: Eq. 12 writes P(v|S) without log; Eq. 11/13 use log. Eq. 13 used.
- **first_token_distribution**: Random(V): distribution, special-token handling and seed undocumented. The released EBSS_qwen.jsonl instead starts record i with token id i (sequential), which contradicts Random(V) or indicates a different script.
- **sequence_count_and_length**: paper silent; rebuttal: 128 x 512; released file: 600 records up to ~2048 tokens.
- **eos_and_specials**: no stop rule; whether specials / EOS were masked is undocumented. Not masked here.
- **per_sentence_vs_global_balance**: sigma(M,S) is a property of one branch; no cross-sentence balance state is described. Per-branch adopted.
- **precision_and_batching_numerics**: authors' precision undocumented; BF16 used (our checkpoint dtype). Batched KV decoding is not bitwise equal to a batch-1 replay (measured in smoke).
- **openreview_response**: not accessible; nothing from it is used.

## Not documented anywhere accessible

- the authors' generation script, its first-token distribution, special-token handling, precision, batching numerics and stop rule;
- the generating checkpoint of the released EBSS_qwen.jsonl (see update_ebss_official_validation.md);
- any per-step balance diagnostics of generated Qwen data.

