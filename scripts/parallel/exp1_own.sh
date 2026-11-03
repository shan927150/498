#!/usr/bin/env bash
# Exp1 on our own labels (single seed -> expert-grouped 5-fold validation). CPU, seconds.
set -euo pipefail; source scripts/parallel/env.sh
python diagnostics/within_layer_ranking.py --outdir runs_own --percdamp $DAMP
