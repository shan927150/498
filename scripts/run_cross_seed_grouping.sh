#!/usr/bin/env bash
# Cross-seed (allocation seeds 0/1/2) x calibration-grouping (5 prespecified seeds) robustness of the FR vs FC
# four-layer GPTQ PPL comparison. Seed 0 = the validated runs_multigroup_robustness results (reused); seeds 1/2 new.
# Resumable: every stage verifies its inputs and skips completed work.
#
#   bash scripts/run_cross_seed_grouping.sh                       # all stages in order
#   STAGE=preflight|tests|manifest|cache|diagnostics|quant|report bash scripts/run_cross_seed_grouping.sh
# One GPU process at a time; every quantized configuration runs in a FRESH process from the checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
unset MOE_LOAD_LAYERS
PY=${PY:-python}
OUT=runs_cross_seed_grouping
STAGE=${STAGE:-all}
GSEEDS="20261111 20261112 20261113 20261114 20261115"
mkdir -p "$OUT/logs" "$OUT/figures"
CSG="$PY diagnostics/cross_seed_grouping.py"
run() { local name=$1; shift; $CSG "$@" 2>&1 | grep -v '^\[E3\]' | tee -a "$OUT/logs/$name.log"; return "${PIPESTATUS[0]}"; }
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
    done_json "$OUT/cache_manifest.json" valid || { echo "cache controls invalid: stop"; exit 1; }
}
stage_diagnostics() { run diagnostics --stage diagnostics; }
stage_quant() {
    # determinism control first: re-quantize seed-0 FR G20261111 from the reused cache (must match bitwise)
    run quant_CTRL_FR_s0 --stage quant --seed 0 --strategy FR --grouping 20261111 --control
    for s in 1 2; do for g in $GSEEDS; do for c in FR FC; do
        run "quant_${c}_s${s}_G${g}" --stage quant --seed $s --strategy $c --grouping $g
    done; done; done
}
stage_report() { run report --stage report; }

case $STAGE in
    all) stage_preflight; stage_tests; stage_manifest; stage_cache; stage_diagnostics; stage_quant; stage_report ;;
    preflight|tests|manifest|cache|diagnostics|quant|report) "stage_$STAGE" ;;
    *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
