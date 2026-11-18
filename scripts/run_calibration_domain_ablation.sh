#!/usr/bin/env bash
# C4 vs WikiText-2-train calibration-domain ablation (2 x 2: source x {FR, SC}). Resumable; every stage skips finished
# work. Full-model stages (cache, gcache, quant) need up to ~65 GB host RAM: run ONE process at a time (72 GB cap).
# Usage: scripts/run_calibration_domain_ablation.sh <stage>
#   stages: tests audit pool protocol cache routing sketch select gmanifest gcache smoke grouped report all
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/venvs/moe/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
PY="python diagnostics/calibration_domain_ablation.py"
LOG=runs_calibration_domain_ablation/logs
mkdir -p "$LOG"
run() { echo "== $*"; $PY "$@" 2>&1 | grep --line-buffered -v -i "warn\|Loading checkpoint\|Token indices" | tee -a "$LOG/runner.log"; return "${PIPESTATUS[0]}"; }
G="G20261111 G20261112 G20261113"
stage="${1:-all}"
case "$stage" in
  tests)  python -m pytest -q tests/test_calibration_domain.py | tee -a "$LOG/tests.log" ;;
  smoke)  run --stage quant --bf16 --configs C4:FR:ORIG WT:FR:ORIG WT:SC:ORIG ;;      # Stage 3 (+ validation re-run)
  grouped)
    cfgs=(); for g in $G; do for s in WT C4; do for c in FR SC; do cfgs+=("$s:$c:$g"); done; done; done
    run --stage quant --configs "${cfgs[@]}" ;;                                          # Stage 4 (12 configurations)
  all)
    "$0" tests
    for st in audit pool protocol cache routing sketch select gmanifest gcache; do run --stage "$st"; done
    "$0" smoke; "$0" grouped; run --stage report ;;
  *) run --stage "$stage" ;;
esac
