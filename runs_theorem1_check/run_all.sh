#!/bin/bash
# Theorem-1 check: Part B grid then Part C sweep (sequential: each loads the bf16 model in CPU RAM; 72 GB cap)
set -eo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/venvs/moe/bin/python
$PY diagnostics/theorem1_check.py grid  2>&1 | grep --line-buffered -v "Loading checkpoint" | tee -a runs_theorem1_check/logs/grid.log
$PY diagnostics/theorem1_check.py sweep 2>&1 | grep --line-buffered -v "Loading checkpoint" | tee -a runs_theorem1_check/logs/sweep.log
echo "ALL DONE"
