#!/usr/bin/env bash
# Frozen-routing robustness across the five prespecified calibration groupings (FR s0 / FC s0, layers 1/8/16/23):
# AO_g = GPTQ(X_g, R_O) for G20261112..15 (8 new configs) + reused G20261111 AO, natural AA_g and OO references.
# Resumable: completed work is skipped. One GPU process at a time; every AO configuration runs in a FRESH process.
#
#   bash scripts/run_frozen_routing_robustness.sh
#   STAGE=preflight|tests|capture|regression|ao_quant|report bash scripts/run_frozen_routing_robustness.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
unset MOE_LOAD_LAYERS
PY=${PY:-python}
OUT=runs_frozen_routing_robustness
STAGE=${STAGE:-all}
mkdir -p "$OUT/logs" "$OUT/figures" "$OUT/results"
DRV="$PY diagnostics/frozen_routing_robustness.py"
run() { local name=$1; shift; $DRV "$@" 2>&1 | grep -v '^\[E3\]' | tee -a "$OUT/logs/$name.log"; return "${PIPESTATUS[0]}"; }

stage_preflight() { run preflight --stage preflight; }
stage_tests() {
    CUDA_VISIBLE_DEVICES= $PY -m pytest tests -q -p no:cacheprovider > "$OUT/logs/pytest_all.log" 2>&1 \
        || { tail -15 "$OUT/logs/pytest_all.log"; exit 1; }
    tail -1 "$OUT/logs/pytest_all.log"
}
stage_capture() { run capture --stage capture; }
stage_regression() { run regression --stage regression; }
stage_ao_quant() {
    for g in G20261112 G20261113 G20261114 G20261115; do for c in FR FC; do
        run "ao_${c}_${g}" --stage ao --strategy $c --grouping $g
    done; done
}
stage_report() { run report --stage report; }

case $STAGE in
    all) stage_preflight; stage_tests; stage_capture; stage_regression; stage_ao_quant; stage_report ;;
    preflight|tests|capture|regression|ao_quant|report) "stage_$STAGE" ;;
    *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
