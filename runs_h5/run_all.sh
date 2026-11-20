#!/bin/bash
# H5 rules: protocol (frozen) -> fit (CPU) -> eval (GPU, resumable) -> kappa (GPU) -> report.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=~/venvs/moe/bin/python
[ -f runs_h5/fit.json ] || $PY diagnostics/h5_rules.py fit 2>&1 | tee -a runs_h5/logs/fit.log
$PY diagnostics/h5_rules.py eval 2>&1 | tee -a runs_h5/logs/eval.log
$PY diagnostics/h5_rules.py kappa 2>&1 | tee -a runs_h5/logs/kappa.log
