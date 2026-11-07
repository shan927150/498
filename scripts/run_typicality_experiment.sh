#!/usr/bin/env bash
# Routing-typicality experiment (11/06 follow-up): random vs coverage_lp at
# B = 128, seeds 0/1/2, from three candidate pools:
#   full          existing runs in runs_allocation_minimal/ (reused after validation)
#   typical_half  1000 lowest d(s) (routing L1 to the original pool)
#   random_half   1000 uniform at random (pool seed 20261106)
# Frozen 4-bit GPTQ, layers 1/8/16/23, test-split reconstruction only.
# Resumable (validated outputs are skipped); seed 0 first as the smoke test.
#
#   bash scripts/run_typicality_experiment.sh            # everything
#   SEEDS=0 bash scripts/run_typicality_experiment.sh    # smoke only
#   PAR=2 ...                                            # 2 GPU runs at a time (20 GB slice: ok)
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOE_LOAD_LAYERS=1,8,16,23
CACHE=${CACHE:-$HOME/moe_cache}
OUT=${OUT:-runs_typicality}
FULL=${FULL:-runs_allocation_minimal}
FROZEN=configs/frozen_b4.json
SEEDS=${SEEDS:-"0 1 2"}
PAR=${PAR:-2}
mkdir -p "$OUT/logs"

[ -f "$OUT/pool_typical_half.json" ] || python diagnostics/typicality_pools.py \
    --routing "$FULL/routing_profile_v2.npz" --frozen $FROZEN --out "$OUT"
python diagnostics/report_minimal_allocation.py --preflight --outdir "$OUT" --cache-dir "$CACHE" \
    --frozen $FROZEN
python diagnostics/report_typicality.py --check-full --full-dir "$FULL" --outdir "$OUT"

run_one() {   # strategy pool seed
    local s=$1 pool=$2 seed=$3 f="$OUT/e3_$1_B128_b4_d1_s$3_pool-$2.json"
    if [ -f "$f" ] && python diagnostics/report_typicality.py --validate "$f" --outdir "$OUT" >/dev/null; then
        echo "[skip] $f (validated)"; return 0
    fi
    echo "[run] $s $pool seed $seed"
    python e3_allocation.py --strategy "$s" --seed "$seed" --B 128 --bits 4 --group-size 128 \
        --percdamp 1.0 --layers 1 8 16 23 --cache-dir "$CACHE" --outdir "$OUT" \
        --frozen-config $FROZEN --routing-cache --candidate-subset-file "$OUT/pool_$pool.json" \
        > "$OUT/logs/${s}_${pool}_s${seed}.log" 2>&1
    python diagnostics/report_typicality.py --validate "$f" --outdir "$OUT"
}

for seed in $SEEDS; do
    if [ "$seed" != 0 ]; then
        for pool in typical_half random_half; do for s in random coverage_lp; do
            python diagnostics/report_typicality.py --validate \
                "$OUT/e3_${s}_B128_b4_d1_s0_pool-${pool}.json" --outdir "$OUT" >/dev/null \
                || { echo "smoke (seed 0) incomplete or invalid: $s $pool"; exit 1; }
        done; done
    fi
    n=0
    for pool in typical_half random_half; do
        for s in random coverage_lp; do
            run_one "$s" "$pool" "$seed" &
            n=$((n + 1))
            if [ "$n" -ge "$PAR" ]; then wait -n; n=$((n - 1)); fi
        done
    done
    wait
done
python diagnostics/report_typicality.py --full-dir "$FULL" --outdir "$OUT"
