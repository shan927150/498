#!/usr/bin/env bash
# ~20-40 min end-to-end check on a tiny setting. Run this BEFORE the real runs.
set -e
C=~/moe_cache_smoke; O=runs_smoke; mkdir -p logs
python cache_inputs.py --layers 8 --pool-size 64 --holdout-size 32 --cache-dir $C
python e1_label_sufficiency.py --cache-dir $C --n0 8 --reps 1 --min-tokens 64 --outdir $O
python e2_signals.py --cache-dir $C --outdir $O
python e3_allocation.py --strategy tlp --B 16 --K 4 --cache-dir $C --outdir $O
python e3_allocation.py --strategy random --B 16 --K 4 --cache-dir $C --outdir $O --ppl --ppl-windows 4
echo "SMOKE TEST PASSED"
