#!/usr/bin/env bash
# Own labels for the 4 characterized layers (seed 0, frozen damping) -> runs_own/.
# Replaces the P1 labels from another account; feeds Exp1 and the Exp6 analysis.
set -euo pipefail; source scripts/parallel/env.sh
export MOE_LOAD_LAYERS=1,8,16,23
python e1_label_sufficiency.py --cache-dir $CACHE --layers 1 8 16 23 --bits 4 \
    --n0 16 32 64 128 --reps 2 --K 1 --char-seed 0 --percdamp $DAMP --outdir runs_own
