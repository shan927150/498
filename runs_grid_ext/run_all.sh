#!/bin/bash
# Damping-grid extension: E1 + E2 (resumable), then the report.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=~/venvs/moe/bin/python
$PY diagnostics/damping_grid_extension.py run 2>&1 | tee -a runs_grid_ext/logs/run.log
