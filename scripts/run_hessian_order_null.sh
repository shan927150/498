#!/usr/bin/env bash
# FP32 Hessian accumulation-order null experiment (FR s0 / FC s0, layers 1/8/16/23, ORIGINAL activations / routing /
# act-order; only the FP32 addition order of the per-sequence Hessian contributions changes).
#   preflight -> tests -> baseline_regression -> order_manifests -> hessian_preflight -> fp64_control -> quant -> report
#   bash scripts/run_hessian_order_null.sh                 # all (honors the Hessian early-stop condition)
#   STAGE=<stage> bash scripts/run_hessian_order_null.sh
# Resumable; one GPU process at a time; every quantization runs in a FRESH process from the pristine checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
unset MOE_LOAD_LAYERS
PY=${PY:-python}
OUT=runs_hessian_order_null
STAGE=${STAGE:-all}
SEEDS="20261121 20261122 20261123 20261124 20261125"
mkdir -p "$OUT/logs" "$OUT/figures" "$OUT/results"
DRV="$PY diagnostics/hessian_order_null.py"
run() { local name=$1; shift; $DRV "$@" 2>&1 | grep -v '^\[E3\]' | tee -a "$OUT/logs/$name.log"; return "${PIPESTATUS[0]}"; }

stage_preflight() { run preflight --stage preflight; }
stage_tests() {
    CUDA_VISIBLE_DEVICES= $PY -m pytest tests -q -p no:cacheprovider > "$OUT/logs/pytest_all.log" 2>&1 \
        || { tail -15 "$OUT/logs/pytest_all.log"; exit 1; }
    tail -1 "$OUT/logs/pytest_all.log"
}
stage_baseline_regression() { for c in FR FC; do run "baseline_$c" --stage baseline_regression --strategy $c; done; }
stage_order_manifests() { run order_manifests --stage order_manifests; }
stage_hessian_preflight() { for c in FR FC; do run "hessian_$c" --stage hessian_preflight --strategy $c; done; }
stage_fp64_control() { for c in FR FC; do run "fp64_$c" --stage fp64_control --strategy $c; done; }
early_stop() {   # prints true / false from the aggregated Hessian preflight (fails if checks did not pass)
    $PY -c "import json,sys; h=json.load(open('$OUT/hessian_differences.json')); \
sys.exit('Hessian preflight checks failed: RUN-BLOCKED') if not h['checks_passed'] else print(str(h['early_stop']).lower())"
}
stage_quant() {
    if [ "$(early_stop)" = "true" ]; then
        echo "all shuffled-order Hessians bitwise identical: NULL-ORDER-INVARIANT, GPTQ skipped" | tee -a "$OUT/logs/quant.log"
        return 0
    fi
    for s in $SEEDS; do for c in FR FC; do run "quant_${c}_O$s" --stage quant --strategy $c --order $s; done; done
    rc=$($PY -c "import json; r=json.load(open('$OUT/hessian_differences.json'))['repeat_config']; print(' '.join(map(str, r)) if r else '')")
    if [ -n "$rc" ]; then set -- $rc; run "quant_$1_O$2_repeat" --stage quant --strategy "$1" --order "$2" --repeat; fi
}
stage_report() { run report --stage report; }

case $STAGE in
    all) stage_preflight; stage_tests; stage_baseline_regression; stage_order_manifests; stage_hessian_preflight
         stage_fp64_control; stage_quant; stage_report ;;
    preflight|tests|baseline_regression|order_manifests|hessian_preflight|fp64_control|quant|report) "stage_$STAGE" ;;
    *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
