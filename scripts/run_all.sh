#!/usr/bin/env bash
# Full mini-scale pipeline. Launch with:
#   nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
# E1 is resumable per (config, characterization seed): if the session dies,
# re-run the same command.
# Signal selection must be validated across characterization seeds: E1 runs
# seeds 0 1 2, E2 refuses to fit with fewer (--min-char-seeds 3) and E3
# refuses a preliminary model (no --allow-preliminary here).
set -e
mkdir -p logs runs
C=~/moe_cache
CHAR_SEEDS="0 1 2"
[ -f $C/meta.json ] || python cache_inputs.py --cache-dir $C          # ~30-60 min, once
for BITS in 3 4; do
  for CS in $CHAR_SEEDS; do
    python e1_label_sufficiency.py --cache-dir $C --bits $BITS --char-seed $CS   # longest step
  done
  python e2_signals.py --cache-dir $C --bits $BITS --min-char-seeds 3
done
for BITS in 3 4; do
  for S in random coverage tlp oracle; do
    for SEED in 0 1 2; do
      python e3_allocation.py --strategy $S --B 128 --bits $BITS --seed $SEED --cache-dir $C --ppl
    done
  done
done
echo "ALL DONE"
