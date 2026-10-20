# Next steps (plan of 10/19)

Order of work follows the 10/18 log: **freeze GPTQ → rerun E1 → rerun E2 →
yes/no on the core hypothesis**, and only then E3. Each step has a script and
an explicit pass/fail rule so nothing downstream runs on an unvalidated input.

| step | what | script | stop rule |
|---|---|---|---|
| P0 | extended damping grid (0.01 … 1.0) + per-projection threshold / effective-rank test | `scripts/p0_damping.sh` | freeze the best value only if it is **interior** to the grid |
| P1 | E1 under the frozen damping, 3 seeds × 4 layers × n0 16/32/64/128 | `scripts/p1p2_characterize.sh <d>` | `e1_gate.py` must PASS |
| P2 | E2 under the frozen damping, then the go/no-go gate | (same script) | `e2_gate.py` case A / A_budget → E3; B → discuss; C → negative finding |
| P3 | support diagnostic with the fixed routing cache → decide on n0 = 256 | `scripts/p3p4_e3_min.sh <d>` | only after case A |
| P4 | minimal E3: B = 128 × seeds 0/1/2 × random / coverage / tlp / oracle + equal-compute random | (same script) | no B sweep unless tlp or oracle beats random/coverage |
| P5 | budgets 64/256/512, all-layer PPL, candidate-pool multiplier k = 2/4/8, distribution constraint, 3-bit | — | only after P4 shows an improvement |

## What changed in the code

* **Damping is a first-class setting.** `--percdamp` on E1/E2/E3, support
  diagnostic and analysis; recorded in E1 fingerprints and in names
  (`e1_b4_d0.2_s0_<hash>`, `e2_model_b4_d0.2.json`, `e3_..._b4_d0.2_s0.json`).
  E3 refuses a damping that differs from its E2 model. Runs without the flag
  keep the legacy names and mean 1% damping.
* **`diagnostics/damping_threshold.py`** (P0): wider grid, errors per
  projection group (gate+up vs down), RTN per group, effective rank of every
  Hessian, resumable per damping value, report flags whether the optimum is
  interior and tabulates GPTQ/RTN error against tokens / input-dim.
* **`e1_gate.py`** (P1): median benefit > 0, ≤ 25 % negative experts, every
  seed median > 0 (required); per-layer medians and diminishing returns (warn).
* **`e2_gate.py`** (P2): compares signals against a **flexible exposure
  baseline** (cubic in log count + budget dummies + budget × log count), not
  just linear log count; leave-one-seed-out **and** leave-one-layer-out;
  permutation null within (seed, budget, layer, count-quintile) strata;
  per-budget slope signs; replicate-noise ceiling; prints case A / A_budget / B / C.
  On synthetic data a count-only proxy gains +68 % over linear count but 0 %
  over the flexible baseline and is correctly rejected (tests/test_p1p2.py).
* **E3**: logs routing-distribution shift of the selected set vs the pool
  (L1, JS per layer) and peak GPU memory; `random_plus` is capped at the pool
  size and records when equal compute would exceed it. Routing caches use a new
  file (`routing_profile_v2.npz`, model `topk(k)` rule); the stale pre-fix
  `routing_profile.npz` is never read.

## Instant check on existing data (CPU, seconds)

The 1 % characterization already on ICRN can be re-judged with the stricter
gate, no GPU needed:

```bash
python e2_gate.py --outdir runs --bits 4          # legacy (1 %) artifacts
```

(margin / entropy in that file still carry the old routing attribution bug.)

## Known gaps for P5

* **All-layer PPL vs disk.** Caching every layer's inputs for the 2 256-row
  pool is ~110 GB (> 100 GB ICRN quota). Final PPL should propagate only the
  selected B sequences + eval windows layer by layer (sequential GPTQ), not
  reuse the pool cache.
* `CODE_SUMMARY.md` describes the pre-refactor code (relSE-only allocation,
  no tuning split) and should be refreshed.
* Proposal Figure 1 (routed tokens vs drift, 256 → 512) has no plotting code.
