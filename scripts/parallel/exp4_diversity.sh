#!/usr/bin/env bash
# Exp4 (GPU, 4 layers only ~11 GB RAM): token count vs source diversity.
set -euo pipefail; source scripts/parallel/env.sh
export MOE_LOAD_LAYERS=1,8,16,23
python diagnostics/exposure_diversity.py run --cache-dir $CACHE --percdamp $DAMP
python diagnostics/exposure_diversity.py report
