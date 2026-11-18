#!/usr/bin/env bash
# Expert-wise activation calibration MVP (L16, 8 experts, B=128, SR/SC/EC/ER). Resumable: every stage skips work
# whose output exists; `all` stops at the first failing stage (set -e).
# Usage: scripts/run_expertwise_mvp.sh <stage>
#   stages: tests protocol sketch smoke freeze final report all
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/venvs/moe/bin/activate
PY="python diagnostics/expertwise_calibration_mvp.py"
LOG=runs_expertwise_mvp/logs
mkdir -p "$LOG"
run() { echo "== $*"; $PY "$@" 2>&1 | tee -a "$LOG/runner.log"; }
stage="${1:-all}"
case "$stage" in
  tests)    python -m pytest tests/test_expertwise_calibration.py -q 2>&1 | tee -a "$LOG/tests.log" ;;
  smoke)    run --stage select --seed 0; run --stage run --seed 0 --tag smoke ;;
  final)    for s in 0 1 2; do run --stage select --seed "$s"; run --stage run --seed "$s" --tag final; done ;;
  all)
    "$0" tests
    run --stage protocol
    run --stage sketch
    "$0" smoke
    run --stage freeze
    "$0" final
    run --stage report ;;
  *) run --stage "$stage" ;;
esac
