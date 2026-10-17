#!/usr/bin/env bash
# Full mini-scale pipeline. Launch with:
#   nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
# E1 is resumable: if the session dies, re-run the same command.
set -e
mkdir -p logs runs
C=~/moe_cache
[ -f $C/meta.json ] || python cache_inputs.py --cache-dir $C          # ~30-60 min, once
for BITS in 3 4; do
  python e1_label_sufficiency.py --cache-dir $C --bits $BITS          # longest step
  python e2_signals.py --cache-dir $C --bits $BITS
done
for BITS in 3 4; do
  for S in random coverage tlp oracle; do
    for SEED in 0 1 2; do
      python e3_allocation.py --strategy $S --B 128 --bits $BITS --seed $SEED --cache-dir $C --ppl
    done
  done
done
echo "ALL DONE"
