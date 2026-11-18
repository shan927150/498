#!/usr/bin/env bash
# Loss-alignment diagnostic MVP A + B. Run ONE full-model process at a time (72 GB host RAM cap).
# Usage: scripts/run_loss_alignment.sh [tests|a_router|a|b_protocol|b_scweights|b_capture|b_eval|b_report|all]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/venvs/moe/bin/python
export OMP_NUM_THREADS=8
mkdir -p runs_loss_alignment/logs
step=${1:-all}
run() { echo "== $*"; "$PY" "$@" 2>&1 | grep --line-buffered -v "Loading checkpoint" | tee -a "runs_loss_alignment/logs/$(basename "$1" .py)_${3}.out"; }

case $step in
  tests)       "$PY" -m pytest -q tests/test_loss_alignment.py ;;
  a_router)    run diagnostics/loss_alignment_mvp_a.py --stage c4router ;;   # GPU, partial load, ~7 min (batch 1)
  a)           run diagnostics/loss_alignment_mvp_a.py --stage analyze ;;    # CPU, needs b-independent inputs only
  b_protocol)  run diagnostics/loss_alignment_mvp_b.py --stage protocol ;;
  b_scweights) run diagnostics/loss_alignment_mvp_b.py --stage scweights ;;  # GPU, partial load, ~9 min
  b_capture)   run diagnostics/loss_alignment_mvp_b.py --stage capture ;;    # GPU, FULL model, ~2 min + load
  b_eval)      run diagnostics/loss_alignment_mvp_b.py --stage eval ;;       # GPU, partial load, ~2 min
  b_report)    run diagnostics/loss_alignment_mvp_b.py --stage report ;;     # CPU (reads mvp_a outputs)
  all)
    "$0" tests; "$0" a_router; "$0" a
    "$0" b_protocol; "$0" b_scweights; "$0" b_capture; "$0" b_eval; "$0" b_report ;;
  *) echo "unknown step $step"; exit 1 ;;
esac
