# Fixed GU-AGQ (AGQ gate/up + GPTQ down): summary

**Frozen verdict: `FIXED-GU-BLOCKED` → B, retain Full AGQ.**

The study is blocked by one correctness check, check 16:
- Native BF16 swaps two near-tied, non-primary pairs in 1 of 24 cells (old-test, seed 0, L1). The gaps are 0.11–0.17 %,
  while BF16 rounding adds +5.4 % to the error there.
- The primary GU vs Full AGQ ranking agrees in all 24 cells.

**Post-hoc reading:** without check 16 the verdict would be `FIXED-GU-SMALL-GAIN` → C. Promising criteria 1 and 3 fail
either way.

## Confirmation split, E_4L = mean of 4 layer ratios (×1e-3)

| Seed | GPTQ | Full AGQ | GU-AGQ | GU vs Full |
|---|---|---|---|---|
| s0 | 3.681321 | 3.625292 | 3.622856 | −0.067 % |
| s1 | 3.682688 | 3.640863 | 3.637497 | −0.092 % |
| s2 | 3.680302 | 3.634806 | 3.630162 | −0.128 % |
| Mean | 3.681437 | 3.633654 | 3.630172 | **−0.096 %** |

**Old test split:** −0.129 % (3/3 seeds).

**Pooled ratio (secondary):** +0.25 to +0.37 % *against* GU-AGQ, dominated by L23.

## Per layer (seed-averaged GU vs Full AGQ, confirmation split)

| Layer | GU vs Full AGQ | seeds GU better |
|---|---|---|
| L1 | **+1.20 %** | 0/3 |
| L8 | **−0.35 %** | 3/3 |
| L16 | +0.09 % | 0/3 |
| L23 | +0.35 % | 0/3 |

## Mechanism (exact Shapley; Down-AGQ evaluated as a diagnostic)

| Layer | gate/up share of the Full-AGQ gain | down AGQ |
|---|---|---|
| L1 | 62 % | helps |
| L8 | — | hurts (−0.35 %), every seed |
| L16 | 92 % | helps |
| L23 | 69 % | helps |

- **Cross terms:** cross terms never flip the GU-vs-Full ranking, even with L1's −23 to −30 % cross-term share.
- **L16 expert 40:** top-1 in every seed, holding 78–84 % of L16's gain, and that gain is about 92 % gate/up.

## Cost and checks

- **Cost:** 0 new solves. Direct GU-AGQ is estimated (not measured) at about 1.0× Full AGQ. It was not built, because
  the gate failed.
- **Correctness:** 19/20 checks pass.
- **Tests:** 23 targeted tests pass, and the full suite passes 598/598.

See `update_fixed_gu_agq.md`.
