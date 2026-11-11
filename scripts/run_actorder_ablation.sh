#!/usr/bin/env bash
# GPTQ act-order ablation after frozen-routing calibration (FR s0 / FC s0, layers 1/8/16/23, five groupings).
# Stages run in order and never bypass the installation audit or the solver regression:
#   preflight -> audit -> tests -> capture_order -> solver_regression -> aof_quant -> report
#   bash scripts/run_actorder_ablation.sh                  # all
#   STAGE=<stage> bash scripts/run_actorder_ablation.sh
# Resumable; one GPU process at a time; every quantization runs in a FRESH process from the pristine checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
unset MOE_LOAD_LAYERS
PY=${PY:-python}
OUT=runs_actorder_ablation
STAGE=${STAGE:-all}
mkdir -p "$OUT/logs" "$OUT/figures" "$OUT/results"
DRV="$PY diagnostics/actorder_ablation.py"
run() { local name=$1; shift; $DRV "$@" 2>&1 | grep -v '^\[E3\]' | tee -a "$OUT/logs/$name.log"; return "${PIPESTATUS[0]}"; }

stage_preflight() { run preflight --stage preflight; }
stage_audit() { run audit --stage audit; }
stage_tests() {
    CUDA_VISIBLE_DEVICES= $PY -m pytest tests -q -p no:cacheprovider > "$OUT/logs/pytest_all.log" 2>&1 \
        || { tail -15 "$OUT/logs/pytest_all.log"; exit 1; }
    tail -1 "$OUT/logs/pytest_all.log"
}
stage_capture_order() {
    for c in FR FC; do run "capture_order_$c" --stage capture_order --strategy $c; done
    run merge_order --stage merge_order
}
stage_solver_regression() { run solver_regression --stage solver_regression; }
stage_aof_quant() {
    for g in G20261111 G20261112 G20261113 G20261114 G20261115; do for c in FR FC; do
        run "aof_${c}_${g}" --stage aof --strategy $c --grouping $g
    done; done
}
stage_report() { run report --stage report; }

case $STAGE in
    all) stage_preflight; stage_audit; stage_tests; stage_capture_order; stage_solver_regression; stage_aof_quant; stage_report ;;
    preflight|audit|tests|capture_order|solver_regression|aof_quant|report) "stage_$STAGE" ;;
    *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
