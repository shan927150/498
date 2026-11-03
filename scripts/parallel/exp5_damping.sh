#!/usr/bin/env bash
# Exp5 (GPU, 4 layers only): extend the P0 grid to 3.0/5.0, then the adaptive-damping analysis.
set -euo pipefail; source scripts/parallel/env.sh
export MOE_LOAD_LAYERS=1,8,16,23
python diagnostics/damping_threshold.py run --cache-dir $CACHE --seeds 0 1 \
    --layers 1 8 16 23 --damps 3.0 5.0
python diagnostics/damping_threshold.py report
python diagnostics/adaptive_damping.py --frozen $DAMP
