#!/usr/bin/env bash
# Exp6b (GPU, 3 layers only): E1 on layers 4/12/20 (seed 0, 2 reps), then depth analysis (CPU).
set -euo pipefail; source scripts/parallel/env.sh
export MOE_LOAD_LAYERS=4,12,20
python e1_label_sufficiency.py --cache-dir $NEWCACHE --bits 4 --n0 16 32 64 128 --reps 2 \
    --K 1 --char-seed 0 --percdamp $DAMP --outdir runs_newlayers
unset MOE_LOAD_LAYERS
python diagnostics/depth_generalization.py --old-outdir runs --new-outdir runs_newlayers \
    --percdamp $DAMP
