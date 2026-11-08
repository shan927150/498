#!/usr/bin/env bash
# Controlled calibration batch-composition sensitivity (four layers 1/8/16/23, FR s0 / FC s0,
# variants ORIGINAL / COMPACT / SHUFFLED). Resumable: every stage skips completed work.
#
#   bash scripts/run_batch_sensitivity.sh                    # all stages in order
#   STAGE=preflight|tests|cache|forward|original|regression|alternatives|weights|report bash scripts/run_batch_sensitivity.sh
# One GPU process at a time; every quantized configuration runs in a FRESH process from the checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
unset MOE_LOAD_LAYERS
PY=${PY:-python}
OUT=runs_batch_sensitivity
STAGE=${STAGE:-all}
mkdir -p "$OUT/logs" "$OUT/figures"
DB="$PY diagnostics/batch_sensitivity.py"
run() { local name=$1; shift; $DB "$@" 2>&1 | grep -v '^\[E3\]' | tee -a "$OUT/logs/$name.log"; return "${PIPESTATUS[0]}"; }
done_json() { [ -f "$1" ] && $PY -c "import json,sys;d=json.load(open('$1'));sys.exit(0 if d.get('$2') else 1)"; }

stage_preflight() { run preflight --stage preflight; }
stage_tests() {
    CUDA_VISIBLE_DEVICES= $PY -m pytest tests -q > "$OUT/logs/pytest_all.log" 2>&1 || { tail -5 "$OUT/logs/pytest_all.log"; exit 1; }
    tail -1 "$OUT/logs/pytest_all.log"
}
stage_cache() {
    if done_json "$OUT/cache_validation.json" valid; then echo "[skip] caches valid"; return 0; fi
    run cache --stage cache
    done_json "$OUT/cache_validation.json" valid || { echo "cache semantics invalid: stop"; exit 1; }
}
stage_forward() { [ -f "$OUT/hessian_sensitivity.json" ] && { echo "[skip] forward diagnostics done"; return 0; }; run forward --stage forward; }
stage_original() { for c in FR FC; do run "quant_${c}_ORIGINAL" --stage quant --strategy $c --variant ORIGINAL; done; }
stage_regression() { run regression --stage regression; }
stage_alternatives() {
    done_json "$OUT/regression_checks.json" passed || { echo "ORIGINAL regression not passed: stop"; exit 1; }
    for v in COMPACT SHUFFLED; do for c in FR FC; do run "quant_${c}_${v}" --stage quant --strategy $c --variant $v; done; done
}
stage_weights() { run weights --stage weights; }
stage_report()  { run report --stage report; }

case $STAGE in
    all) stage_preflight; stage_tests; stage_cache; stage_forward; stage_original; stage_regression
         stage_alternatives; stage_weights; stage_report ;;
    preflight|tests|cache|forward|original|regression|alternatives|weights|report) "stage_$STAGE" ;;
    *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
