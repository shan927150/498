#!/usr/bin/env bash
# Full-E3 validation of Shared Covariance-aware calibration (SC) vs FR / TR / TC. Resumable; every stage skips
# finished work. Full-model stages (evalA, gcache, gquant) peak at ~68 GB host RAM: run them ONE AT A TIME.
# Usage: scripts/run_shared_covariance_e3.sh <stage>
#   stages: tests preflight protocol sketch mvpcheck select freeze stageA diag stageB report all
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/venvs/moe/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}   # as every earlier full-model runner
PY="python diagnostics/shared_covariance_e3.py"
LOG=runs_shared_covariance_e3/logs
mkdir -p "$LOG"
run() { echo "== $*"; $PY "$@" 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$LOG/runner.log"; return "${PIPESTATUS[0]}"; }
stage="${1:-all}"
case "$stage" in
  tests)   python -m pytest tests/test_shared_covariance_e3.py tests/test_expertwise_calibration.py -q 2>&1 | tee -a "$LOG/tests.log" ;;
  preflight) run --stage preflight --hash-caches ;;
  select)  for s in 0 1 2; do run --stage select --seed "$s"; done ;;
  stageA)  run --stage evalA --configs FR:0 TR:1 TC:2 SC:0 SC:1 SC:2 ;;   # validation re-runs first
  stageB)
    run --stage gmanifest
    for s in 0 1 2; do run --stage gcache --seed "$s"; done
    for s in 0 1 2; do for g in 20261111 20261112 20261113 20261114 20261115; do
      run --stage gquant --seed "$s" --grouping "$g"; done; done
    run --stage gquant --seed 0 --grouping 20261111 --strategy SC --control ;;   # fresh-process determinism control
  all)
    "$0" tests; "$0" preflight
    for st in protocol sketch mvpcheck; do run --stage "$st"; done
    "$0" select; run --stage freeze; "$0" stageA; run --stage diag; "$0" stageB; run --stage report ;;
  *) run --stage "$stage" ;;
esac
