#!/usr/bin/env bash
# Real 4-bit characterization ONLY: cache (if missing) -> E1 x char seeds 0,1,2
# -> E2 (leave-one-char-seed-out) -> support diagnostic at E3 decision
# checkpoints. No E3, no 3-bit, no PPL. Resumable: re-run the same command.
#   nohup bash scripts/run_characterization_b4.sh > logs/characterization_b4.log 2>&1 &
set -euo pipefail
mkdir -p logs runs
C=~/moe_cache
O=runs
N0="16 32 64 128"
[ -f $C/meta.json ] || python cache_inputs.py --cache-dir $C
for CS in 0 1 2; do
  python e1_label_sufficiency.py --cache-dir $C --bits 4 --n0 $N0 --char-seed $CS --outdir $O
done
python e2_signals.py --cache-dir $C --bits 4 --min-char-seeds 3 --outdir $O
python support_diagnostic.py --cache-dir $C --bits 4 --budgets 64 128 256 512 --seeds 0 1 2 --outdir $O
echo "CHARACTERIZATION DONE"
