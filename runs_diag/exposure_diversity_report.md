# Exp4 — token count vs source diversity (frozen GPTQ, tuning split)

Ratio = error(condition A) / error(condition B), paired per expert; '% A better' = share of experts where A has lower error.

## regime m = 32 sequences

| condition | tokens | sources | erank gate/up | erank down | mean err |
|---|---|---|---|---|---|
| natural | 1092 | 32.0 | 240 | 354 | 0.00507 |
| spread | 1092 | 244.4 | 300 | 432 | 0.00492 |
| conc | 1092 | 16.4 | 199 | 290 | 0.00535 |
| half_tok | 546 | 31.9 | 176 | 243 | 0.00531 |
| half_seq | 546 | 16.0 | 150 | 209 | 0.00538 |

| contrast | ratio A/B | % A better | n |
|---|---|---|---|
| spread / natural — diversity, same count (more sources) | 0.9696 | 99.2% | 480 |
| conc / natural — diversity, same count (fewer sources) | 1.0536 | 0.2% | 480 |
| half_tok / half_seq — sources at ~same count (m vs m/2) | 0.9866 | 96.0% | 480 |
| natural / half_tok — count at same sources (N vs N/2) | 0.9543 | 100.0% | 480 |

## regime m = 128 sequences

| condition | tokens | sources | erank gate/up | erank down | mean err |
|---|---|---|---|---|---|
| natural | 4369 | 127.9 | 424 | 638 | 0.00460 |
| spread | 4369 | 255.4 | 449 | 677 | 0.00454 |
| conc | 4369 | 87.7 | 403 | 610 | 0.00469 |
| half_tok | 2184 | 127.7 | 369 | 547 | 0.00474 |
| half_seq | 2185 | 63.9 | 341 | 506 | 0.00480 |

| contrast | ratio A/B | % A better | n |
|---|---|---|---|
| spread / natural — diversity, same count (more sources) | 0.9878 | 99.4% | 480 |
| conc / natural — diversity, same count (fewer sources) | 1.0192 | 0.4% | 480 |
| half_tok / half_seq — sources at ~same count (m vs m/2) | 0.9855 | 99.8% | 480 |
| natural / half_tok — count at same sources (N vs N/2) | 0.9710 | 100.0% | 480 |

Reading: spread < 1 and conc > 1 (with half_tok/half_seq < 1) means diversity of sources matters beyond the token count, so exposure should be measured in sources or effective rank, not raw routed tokens. Ratios ~1 mean the count is what matters.
