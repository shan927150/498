#!/usr/bin/env bash
# Exp2 (GPU, full model ~30 GB RAM): random B = 0/32/64/128/256/512 x 3 seeds, recon + PPL.
set -euo pipefail; source scripts/parallel/env.sh
python diagnostics/headroom.py run --cache-dir $CACHE --outdir runs_headroom --percdamp $DAMP \
    --layers 1 8 16 23
python diagnostics/headroom.py report --outdir runs_headroom
