#!/usr/bin/env bash
# Minimal fixed-budget allocation experiment (11/06): B = 128, seeds 0/1/2,
# frozen 4-bit GPTQ (configs/frozen_b4.json), layers 1/8/16/23, reconstruction
# on the fixed test split only (no PPL, no B sweep, no 3-bit).
#
#   strategies: random | coverage (greedy, unchanged) | coverage_lp |
#               global_exposure (M1) | layer_budget (Dlb) | layer_aware (M4)
#
# Resumable: a run is skipped only if its JSON validates
# (diagnostics/report_minimal_allocation.py --validate). Seed 0 is the smoke
# test; seeds 1/2 run only if every seed-0 output validates.
#
#   bash scripts/run_minimal_allocation.sh            # everything
#   SEEDS="0" bash scripts/run_minimal_allocation.sh  # smoke only
#   PAR=2 bash scripts/run_minimal_allocation.sh      # 2 runs at a time (GPU >= 20 GB)
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOE_LOAD_LAYERS=1,8,16,23                   # partial load: no PPL in this experiment
CACHE=${CACHE:-$HOME/moe_cache}
OUT=${OUT:-runs_allocation_minimal}
FROZEN=configs/frozen_b4.json
MODELS=${MODELS:-runs/allocation_models_b4_d1.json}
SEEDS=${SEEDS:-"0 1 2"}
PAR=${PAR:-1}
STRATS="random coverage coverage_lp global_exposure layer_budget layer_aware"
mkdir -p "$OUT/logs"

[ -f "$MODELS" ] || python diagnostics/export_allocation_models.py --frozen $FROZEN \
    --signals-csv runs/e2_signals_b4_d1.csv --out "$MODELS"
python diagnostics/report_minimal_allocation.py --preflight --outdir "$OUT" \
    --cache-dir "$CACHE" --frozen $FROZEN --alloc-models "$MODELS"

run_one() {   # strategy seed
    local s=$1 seed=$2 f="$OUT/e3_$1_B128_b4_d1_s$2.json"
    if [ -f "$f" ] && python diagnostics/report_minimal_allocation.py --validate "$f" \
            --frozen $FROZEN >/dev/null; then
        echo "[skip] $f (validated)"; return 0
    fi
    echo "[run] $s seed $seed"
    python e3_allocation.py --strategy "$s" --seed "$seed" --B 128 --bits 4 --group-size 128 \
        --percdamp 1.0 --layers 1 8 16 23 --cache-dir "$CACHE" --outdir "$OUT" \
        --frozen-config $FROZEN --alloc-models "$MODELS" --routing-cache \
        > "$OUT/logs/${s}_s${seed}.log" 2>&1
    python diagnostics/report_minimal_allocation.py --validate "$f" --frozen $FROZEN
}

# the pool routing matrix is shared; build it once before any parallel runs
for seed in $SEEDS; do
    if [ "$seed" != 0 ]; then
        for s in $STRATS; do
            python diagnostics/report_minimal_allocation.py --validate \
                "$OUT/e3_${s}_B128_b4_d1_s0.json" --frozen $FROZEN >/dev/null \
                || { echo "smoke (seed 0) incomplete or invalid: $s"; exit 1; }
        done
    fi
    run_one random "$seed"                          # first run also builds the routing cache
    n=0
    for s in $STRATS; do
        [ "$s" = random ] && continue
        run_one "$s" "$seed" &
        n=$((n + 1))
        if [ "$n" -ge "$PAR" ]; then wait -n; n=$((n - 1)); fi
    done
    wait
done
python diagnostics/report_minimal_allocation.py --outdir "$OUT" --frozen $FROZEN
