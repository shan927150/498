#!/usr/bin/env bash
# AGQ-only validation (standard GPTQ vs AGQ-GPTQ, matched C4 Random B=128, CANONICAL-B1). Resumable: every stage
# skips work whose output exists; `all` stops at the first failed correctness gate (set -e).
# Usage: scripts/run_agq_only_validation.sh <stage>
#   stages: source_review inspect tests capture_smoke hessian_oracle unit_regression expert_pilot full_capture
#           agq_quantize evaluate diagnostics report all
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/venvs/moe/bin/activate
PY="python diagnostics/agq_only_validation.py"
LOG=runs_agq_only_validation/logs
mkdir -p "$LOG"
run() { echo "== $*"; $PY "$@" 2>&1 | tee -a "$LOG/runner.log"; }
stage="${1:-all}"
case "$stage" in
  agq_quantize|evaluate)
    for s in 0 1 2; do
      [[ $s != 0 && ! -f runs_agq_only_validation/results/AGQ_FR_s0.json ]] && break
      run --stage "$stage" --config "FR_s$s"
    done ;;
  all)
    for st in source_review inspect tests capture_smoke hessian_oracle unit_regression expert_pilot full_capture; do
      run --stage "$st"
    done
    for s in 0 1 2; do run --stage agq_quantize --config "FR_s$s"; run --stage evaluate --config "FR_s$s"; done
    run --stage diagnostics
    run --stage report ;;
  *) run --stage "$stage" ;;
esac
