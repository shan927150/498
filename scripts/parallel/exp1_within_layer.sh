#!/usr/bin/env bash
# Exp1 (CPU, seconds): is the P2c ranking gain actionable within a layer?
set -euo pipefail; source scripts/parallel/env.sh
python diagnostics/within_layer_ranking.py --outdir runs --percdamp $DAMP
