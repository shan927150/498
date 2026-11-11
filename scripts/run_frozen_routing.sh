#!/usr/bin/env bash
# Frozen-routing 2x2 factorial ablation (activation source x routing source of the calibration Hessians) for
# FR s0 / FC s0, four layers, ORIGINAL vs G20261111. Natural OO / AA must reproduce the frozen multigroup
# results bitwise before any hybrid (AO / OA) configuration runs. Resumable: completed work is skipped.
#
#   bash scripts/run_frozen_routing.sh                     # all stages in order
#   STAGE=preflight|tests|capture|regression|hybrid_quant|report bash scripts/run_frozen_routing.sh
# One GPU process at a time; every quantized configuration runs in a FRESH process from the checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
unset MOE_LOAD_LAYERS
PY=${PY:-python}
OUT=runs_frozen_routing
STAGE=${STAGE:-all}
mkdir -p "$OUT/logs" "$OUT/figures"
FRR="$PY diagnostics/frozen_routing.py"
run() { local name=$1; shift; $FRR "$@" 2>&1 | grep -v '^\[E3\]' | tee -a "$OUT/logs/$name.log"; return "${PIPESTATUS[0]}"; }

stage_preflight() { run preflight --stage preflight; }
stage_tests() {
    CUDA_VISIBLE_DEVICES= $PY -m pytest tests -q -p no:cacheprovider > "$OUT/logs/pytest_all.log" 2>&1 \
        || { tail -15 "$OUT/logs/pytest_all.log"; exit 1; }
    tail -1 "$OUT/logs/pytest_all.log"
}
stage_capture() { run capture --stage capture; }
stage_regression() {
    for c in FR FC; do for k in OO AA; do run "regression_${c}_${k}" --stage regression --strategy $c --cond $k; done; done
}
stage_hybrid_quant() {
    for c in FR FC; do for k in AO OA; do run "hybrid_${c}_${k}" --stage hybrid --strategy $c --cond $k; done; done
}
stage_report() { run report --stage report; }

case $STAGE in
    all) stage_preflight; stage_tests; stage_capture; stage_regression; stage_hybrid_quant; stage_report ;;
    preflight|tests|capture|regression|hybrid_quant|report) "stage_$STAGE" ;;
    *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
