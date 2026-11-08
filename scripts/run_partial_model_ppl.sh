#!/usr/bin/env bash
# Partial-model WikiText-2 PPL for the frozen B = 128 matched selections
# (routed experts of layers 1/8/16/23 GPTQ fake-quantized; everything else BF16).
#
#   bash scripts/run_partial_model_ppl.sh                  # all stages, resumable
#   STAGE=protocol|manifest|tests|bf16|reference|smoke|quant|report bash scripts/run_partial_model_ppl.sh
# One GPU process at a time (no parallel jobs). Needs ~45 GB CPU RAM, ~8 GB GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
unset MOE_LOAD_LAYERS                       # full model: the PPL forward needs all 24 layers
PY=${PY:-python}
CACHE=${CACHE:-$HOME/moe_cache}
OUT=runs_partial_ppl
STAGE=${STAGE:-all}
mkdir -p "$OUT/logs" "$OUT/figures"
EP="$PY diagnostics/eval_partial_ppl.py --cache-dir $CACHE"

stage_protocol()  { $EP --stage protocol  2>&1 | tee "$OUT/logs/protocol.log"; }
stage_manifest()  { $EP --stage manifest  2>&1 | tee "$OUT/logs/manifest.log"; }
stage_tests() {
    CUDA_VISIBLE_DEVICES= $PY -m pytest tests -q > "$OUT/logs/pytest_all.log" 2>&1 || { tail -5 "$OUT/logs/pytest_all.log"; exit 1; }
    tail -1 "$OUT/logs/pytest_all.log"
}
stage_bf16() {
    [ -f "$OUT/bf16_baseline.json" ] && { echo "[skip] bf16 baseline exists"; return 0; }
    $EP --stage bf16 2>&1 | grep -v '^\[E3\]' | tee "$OUT/logs/bf16.log"
}
stage_reference() {
    [ -f "$OUT/reference_check.json" ] && { echo "[skip] reference exists"; return 0; }
    $EP --stage reference 2>&1 | tee "$OUT/logs/reference.log"
}
stage_smoke() { $EP --stage quant --only FR:0 2>&1 | grep -v '^\[E3\]' | tee "$OUT/logs/smoke.log"; }
stage_quant() { $EP --stage quant 2>&1 | grep -v '^\[E3\]' | tee "$OUT/logs/quant.log"; }
stage_report() { $PY diagnostics/report_partial_ppl.py 2>&1 | tee "$OUT/logs/report.log"; }

case $STAGE in
    all) stage_protocol; stage_manifest; stage_tests; stage_bf16; stage_reference; stage_smoke; stage_quant; stage_report ;;
    protocol|manifest|tests|bf16|reference|smoke|quant|report) "stage_$STAGE" ;;
    *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
