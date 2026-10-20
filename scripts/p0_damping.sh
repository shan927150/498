#!/usr/bin/env bash
# P0: extended fixed-damping study + per-projection threshold / effective-rank test.
# Tuning split only. Resumable. Output: runs_diag/damping_threshold_report.md
#   nohup bash scripts/p0_damping.sh > logs/p0_damping.log 2>&1 &
set -euo pipefail
mkdir -p logs runs_diag
python diagnostics/damping_threshold.py run --cache-dir ~/moe_cache \
    --seeds 0 1 --layers 1 8 16 23 --damps 0.01 0.1 0.2 0.3 0.5 1.0
python diagnostics/damping_threshold.py report
echo "P0 DONE -> read runs_diag/damping_threshold_report.md, then freeze ONE percdamp"
