#!/usr/bin/env bash
# Full-depth routed-expert GPTQ pilot (all 24 MoE layers; Full Random s0 vs Full Coverage-LP s0).
# NOT fully 4-bit LLM quantization: attention / routers / shared experts / embeddings / lm_head stay BF16.
#
#   bash scripts/run_full_depth_pilot.sh                     # all stages, resumable
#   STAGE=preflight|tests|cache|regression|fr|fc|report bash scripts/run_full_depth_pilot.sh
# One GPU process at a time; each strategy runs in a FRESH process from the pretrained checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
unset MOE_LOAD_LAYERS
PY=${PY:-python}
OUT=runs_full_depth_pilot
STAGE=${STAGE:-all}
mkdir -p "$OUT/logs" "$OUT/figures"
FD="$PY diagnostics/full_depth_pilot.py"
run() { local name=$1; shift; $FD "$@" 2>&1 | grep -v '^\[E3\]' | tee -a "$OUT/logs/$name.log"; }

stage_preflight()  { run preflight --stage preflight; }
stage_tests() {
    CUDA_VISIBLE_DEVICES= $PY -m pytest tests -q > "$OUT/logs/pytest_all.log" 2>&1 || { tail -5 "$OUT/logs/pytest_all.log"; exit 1; }
    tail -1 "$OUT/logs/pytest_all.log"
}
stage_cache()      { run cache --stage cache; }
stage_regression() {
    if [ -f "$OUT/regression_checks.json" ] && $PY -c "import json,sys;sys.exit(0 if json.load(open('$OUT/regression_checks.json'))['passed'] else 1)"; then
        echo "[skip] regression passed"; return 0; fi
    run regression --stage regression
}
stage_fr()     { [ -f "$OUT/full_random_s0.json" ] && { echo "[skip] FR done"; return 0; }; run quant_FR --stage quant --strategy FR; }
stage_fc()     { [ -f "$OUT/full_coverage_s0.json" ] && { echo "[skip] FC done"; return 0; }; run quant_FC --stage quant --strategy FC; }
stage_report() { run report --stage report; }

case $STAGE in
    all) stage_preflight; stage_tests; stage_cache; stage_regression; stage_fr; stage_fc; stage_report ;;
    preflight|tests|cache|regression|fr|fc|report) "stage_$STAGE" ;;
    *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
