# Sign-convention table: production generator

Production file: `moe_calib/ebss_generator.py`. Line numbers are after the opt-in audit hook was added; that hook has no numerical effect. The production entry point is `generate()`, called by `EBSSGenerator` and `ProbabilityOnlyGenerator` and by `diagnostics/ebss_generation_pilot.py` (`stage_generate`, `stage_smoke`).

**Representation found: the code stores cumulative log-probability (≤ 0), not NLL.** The variable `cum` starts at 0 (line 342) and is updated by adding the next-token log-probability (line 398). The candidate score negates the sum of `cum` and the next-token log-probability (line 77). This is the logP form of Eq. 13, so the trap `-(cumulative_nll + logp)` does not apply.

Each sign was checked numerically. "Verified" refers to the captured production values, compared against the oracle: `production_trace_audit.json` and `logs/historical_pilot_audit.json`.

| Variable (code) | Mathematical quantity | Expected sign | Actual code | Verified |
|---|---|---|---|---|
| `cum` (l. 342, 398) | L(S) = Σ log P, cumulative **log-probability** of the prefix | ≤ 0 | `zeros` at start; `cum = cum[par] + newlp` | equals `fsum(token_logp)` of every branch (max abs 0 to 6e-14); propagation exact |
| `logp` (l. 356) | log P(v\|S) over V = ids 0..151645 | ≤ 0 | `torch.log_softmax(logits.float(), dim=-1)`; logits are `out.logits[:, -1, :151646]` (l. 265) | ≤ 0 everywhere; matches fp64 oracle log-softmax ≤ 1e-5 |
| `newlp` (l. 396) | log P of the **selected** token under its **own parent** | ≤ 0 | `logp[par, tok]`, with `par` the global parent row | `token_logp` propagation exact; teacher-forced replay agrees |
| `cum` after update | L(S‖v) | ≤ 0 | `cum[par] + newlp` (the parent's history, not the slot's) | exact against the oracle per row |
| probability component (l. 77) | (N(S) − p_v)/(\|S\|+1), average NLL | ≥ 0 | `-(cum_logp[:, None] + logp)/(prefix_len + 1)` | ≥ 0 at every captured candidate |
| `sigma` (l. 70–71, 353) | Eq. 3/4 on the parent prefix; u = count / \|S\| | ≥ 0 | `u.std(unbiased=True).mean(-1)` on `counts` of the l prefix tokens | equals oracle Eq. 4 recomputed from routing ids (≤ 3e-17) |
| balance penalty (l. 79) | σ(S)/τ | ≥ 0 | `+ (sigma / tau)[:, None]`, added with + and **divided** by τ | sign and τ role verified (tradeoff threshold τ* = 2) |
| final candidate score (l. 357) | J(S‖v), to be minimised | ≥ 0 | sum of the two terms | equals oracle to ≤ 1e-15 (synthetic) and ≤ 1e-9 (real model) |
| pruning (l. 86–88, 362) | w smallest J over all w·V (parent, token) pairs | — | `torch.sort(flat, stable=True)[:w]` on the flattened [B, V] per sentence | kept sets identical to the oracle's `np.lexsort` |
| `counts` (l. 351, 400) | routed top-4 counts of the prefix, per branch | ≥ 0 | scatter of the last fed token's routing, then `counts[par]` | exact recount from per-token routing ids; child = parent + new token |
| final score (l. 93–94, 416) | −L(c)/\|c\| + σ(c)/τ | ≥ 0 | `-cum_logp/length + sigma/tau` | equals the oracle |
| final pick (l. 417) | argmin over the w survivors | — | `torch.argmin` (first minimum on ties) | equals the oracle pick for all 16+16 pilot and all 4+4 audit sentences |

Additional checks:
- **Denominator.** `prefix_len = l = j` (line 348) is the number of prefix tokens, including the unconditioned first token. The divisor `l + 1` is the candidate's length.
- **Tie-break.** All candidates in a step share l, so the denominator cannot change the ranking within a step. It only scales the probability term relative to σ/τ by l/(l+1) compared with "mean NLL over scored tokens".
- **Probability-Only.** `use_balance=False` skips line 79. In the final score, line 94 returns `s` without σ. Nothing else differs: same candidates, same normalisation, same starts.
