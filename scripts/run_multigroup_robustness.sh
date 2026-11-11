#!/usr/bin/env bash
# Multi-grouping robustness of the FR s0 vs FC s0 four-layer GPTQ PPL comparison (5 prespecified
# calibration-only BF16 batch groupings). Resumable: every stage verifies its inputs and skips done work.
#
#   bash scripts/run_multigroup_robustness.sh                         # all stages in order
#   STAGE=preflight|tests|manifest|cache|diagnostics|quant|report bash scripts/run_multigroup_robustness.sh
# One GPU process at a time; every quantized configuration runs in a FRESH process from the checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
unset MOE_LOAD_LAYERS
PY=${PY:-python}
OUT=runs_multigroup_robustness
STAGE=${STAGE:-all}
GROUPINGS="G20261111 G20261112 G20261113 G20261114 G20261115"
mkdir -p "$OUT/logs" "$OUT/figures"
MR="$PY diagnostics/multigroup_robustness.py"
run() { local name=$1; shift; $MR "$@" 2>&1 | grep -v '^\[E3\]' | tee -a "$OUT/logs/$name.log"; return "${PIPESTATUS[0]}"; }
done_json() { [ -f "$1" ] && $PY -c "import json,sys;d=json.load(open('$1'));sys.exit(0 if d.get('$2') else 1)"; }

stage_preflight() { run preflight --stage preflight; }
stage_tests() {
    CUDA_VISIBLE_DEVICES= $PY -m pytest tests -q -p no:cacheprovider > "$OUT/logs/pytest_all.log" 2>&1 \
        || { tail -15 "$OUT/logs/pytest_all.log"; exit 1; }
    tail -1 "$OUT/logs/pytest_all.log"
}
stage_manifest() { run manifest --stage manifest; }
stage_cache() {
    if done_json "$OUT/cache_manifest.json" valid; then echo "[skip] caches valid"; return 0; fi
    run cache --stage cache
    done_json "$OUT/cache_manifest.json" valid || { echo "cache semantics invalid: stop"; exit 1; }
}
stage_diagnostics() {
    [ -f "$OUT/activation_diagnostics.json" ] && { echo "[skip] diagnostics done"; return 0; }
    run diagnostics --stage diagnostics
}
stage_quant() {
    # determinism controls first: ORIGINAL cache must reproduce the frozen FR / FC weights, PPL, reconstruction
    for c in FR FC; do run "quant_CTRL_${c}_ORIGINAL" --stage quant --strategy $c --grouping ORIGINAL; done
    for g in $GROUPINGS; do for c in FR FC; do run "quant_${c}_${g}" --stage quant --strategy $c --grouping $g; done; done
}
stage_report() { run report --stage report; }

case $STAGE in
    all) stage_preflight; stage_tests; stage_manifest; stage_cache; stage_diagnostics; stage_quant; stage_report ;;
    preflight|tests|manifest|cache|diagnostics|quant|report) "stage_$STAGE" ;;
    *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
