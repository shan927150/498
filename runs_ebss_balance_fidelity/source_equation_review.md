# Source review: how strong is MoEQuant's expert-balance term?

Sources:
- arXiv 2505.03804v1 HTML: Eq. 3, 4, 9–14 and Sec. 4.2. These were re-read during the score audit (`runs_ebss_score_audit/paper_equations_review.md`).
- `asserts/Algorithm_1_EBSS.png` in github.com/chenzx921020/MoEQuant @ d77ba1b, read directly.
- The rebuttal tables `Tables_for_R1..R4*.md`.
- The released `code/` tree, searched for expert-frequency / STD code; none was found.

OpenReview is not accessible.

## What each source says

| question | paper | Algorithm 1 | rebuttal | code |
|---|---|---|---|---|
| definition of u_e | Sec. 3: "the standard deviation in the **frequency of expert usage** across all layers"; u_l^e = "usage frequency" of expert e in layer l | line 29: `expert_usage ← M.get_expert_activations(sequence)` | R3 Table R1: "standard deviation on **frequency** of expert" | none |
| counts, rate or share | not stated | not stated ("activations") | not stated | none |
| STD constant | Eq. 4: sqrt( **1/(E−1)** Σ_e (u_l^e − û_l)² ) | line 30: sqrt( **E/(E−1)** Σ_{e=1}^{E} (expert_usage[e] − μ)² ), annotated "▷ Eq. 4" | — | none |
| mean over layers | Eq. 3: σ = Σ_l σ_l / L | line 30 shows a single σ, with no layer index | "across all layers" (Table R1 caption, Sec. 3) | none |
| prefix or candidate | Eq. 13 σ(M, S); Sec. 4.2 "Deferred Expert Imbalance Calculation" (v excluded) | line 14 σ(M, new_candidate) | — | none |
| τ | Eq. 9–13: σ/τ, "controlling the impact of expert imbalance" | line 14 / 36: σ/τ | Qwen τ: 1.2 → 49.60, 1.3 → 49.61 | none |
| which normalisation produced the published Qwen results | **not stated** | **not stated** | **not stated** | **no generator released** |

## Transcription check

- The task's transcription of Algorithm 1 line 30, sqrt(E/(E−1) Σ (u − ū)²), matches the image.
- Eq. 4 has 1/(E−1). The two differ by exactly √E, which is √60 = 7.746 for Qwen1.5-MoE-A2.7B.
- The model config confirms the setting: `num_experts` 60, `num_experts_per_tok` 4, `decoder_sparse_step` 1. So all 24 decoder layers are MoE layers, each with 60 routed experts plus one shared expert.

## Weight of the evidence

- Neither the equation nor the algorithm is privileged. The algorithm cites Eq. 4 but writes a different constant, which means at least one of the two contains a typo.
- **"Frequency"** implies a normalised usage, not raw counts. Raw counts would make σ grow with |S| and are excluded. Frequency per token (rate) and share of the k·n assignments are both "frequencies".
- **Rebuttal Table R1** (DeepSeek-MoE-16B, 64 routed experts, top-6) reports σ = 0.0427 for wikitext2 and 0.0052 for EBSS. Converting the wikitext2 value into the per-token-rate CV each variant implies:

  | variant | implied CV |
  |---|---|
  | EQ4_RATE | 0.46 |
  | EQ4_SHARE | 2.7 |
  | ALG1_SHARE | 0.34 |
  | ALG1_RATE | 0.057 |

  A CV of 2.7 for natural text over 64 experts would need extreme concentration. So EQ4_SHARE is the least consistent with the authors' reported magnitudes. The table cannot separate the other three without DeepSeek routing data, so it is used only as context, never to select a variant.
- **No source establishes the authors' actual implementation.** All four variants are paper-grounded readings. The study measures whether the choice matters; it does not claim which one the authors used.

## Variants (frozen in `normalization_definitions.json`)

| ID | STD | usage | strength vs EQ4_RATE | equivalent τ (explanation only) |
|---|---|---|---|---|
| EQ4_RATE | Eq. 4 | c/n | 1.000 | 1.300 |
| EQ4_SHARE | Eq. 4 | c/(kn) | 0.250 | 5.200 |
| ALG1_SHARE | Alg. 1 | c/(kn) | 1.936 | 0.671 |
| ALG1_RATE | Alg. 1 | c/n | 7.746 | 0.168 |

Everything else is unchanged in every variant:
- the Eq. 3 layer mean and the parent-prefix σ;
- τ = 1.3 applied as division;
- the minimisation and the global top-w;
- the cumulative logP;
- the final argmin.
