# Author rebuttal material incorporated

Sources: the four `Tables_for_R*.md` files and `asserts/Algorithm_1_EBSS.png` of github.com/chenzx921020/MoEQuant
@ d77ba1b (local clone ~/ebss_official_src/moequant). The OpenReview note aKw8ICecFU could NOT be accessed (bot
verification wall; API redirects to a challenge) -- nothing from it is used or inferred.

| item | rebuttal value | file / table | used as |
|---|---|---|---|
| search width w (Qwen 7-task avg) | w=2 49.19, 3 49.45, **4 49.59**, 5 49.59, 6 49.61, 10 49.60, 20 49.63 | R3 Table R3 | w = 4 (paper default; diminishing returns) |
| tau (Qwen) | 1.0 49.47/49.49, 1.1 49.53, **1.2 49.60**, 1.3 49.61, 1.4 49.59, 1.5 49.55 | R1 Table R2, R3 Table R2 | tau = 1.3 (task protocol; paper default 1.2; difference 0.01 points) |
| tau name | "Temperature Parameter (tau)"; Alg. 1: "Expert balance temperature" | R1/R3 Table R2, Alg. 1 | divides sigma only; NOT a logit temperature |
| calibration budget | "MoEQuant generates 128 sequences of 512 length" | R2 Table R3, R3 Table R6, R4 Table R3 | full target 128 x 512 = 65,536 tokens; this pilot 16 x 512 |
| batching | "we stitch multiple sequences together in the batch dimension" | same | all 16 sentences x w branches decoded in one batch |
| EBSS generation time (A800) | Qwen-MoE-14B 18 min, DeepSeek-MoE-16B 12 min, Mixtral-8x7B 42 min | R3 Table R6 | reference only; not normalised to our MIG slice |
| total quantization time (A800) | Qwen GPTQ 37 min vs MoEQuant 54 min | R2 Table R3 | motivation for overhead analysis |
| expert balance STD | DeepSeek-MoE-16B: wikitext2 0.0427, humaneval 0.0877, gsm8k 0.0928, EBSS 0.0052 | R3 Table R1 (asserts/inter-expert_balance.md) | scale evidence for u = per-token usage rate |
| generated samples | Qwen sample begins '! ### GMAT Club Daily Prep ...' (mixed English/Chinese, exam items, code, web text, some repetition) | R4 Table R4 | context only; not evidence of a defect |
| Algorithm 1 | score = -(RS+log prob)/(j+1) + sigma(M,new_candidate)/tau; Topk(w, key=score); best = argmax Score | Alg. 1 image | see source_ambiguities.json for the two conflicts with the paper |

The rebuttal reports only downstream accuracy for w/tau; no tables of generated-data balance for Qwen exist.
