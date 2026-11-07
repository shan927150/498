#!/usr/bin/env bash
# Matched-warm-start typicality experiment: random vs coverage_lp from the
# full / typical_half / random_half pools (runs_typicality/), B = 128, seeds
# 0/1/2; EVERY run of seed s starts from the same ordered 32-sequence warm
# start W_s drawn from typical_half & random_half. 18 GPU runs, no reuse of the
# unmatched runs. Frozen 4-bit GPTQ, layers 1/8/16/23, test-split recon only.
#
#   bash scripts/run_typicality_matched.sh             # smoke (seed 0) -> check -> seeds 1, 2
#   SEEDS=0 bash scripts/run_typicality_matched.sh     # smoke only
# Resumable: validated outputs are skipped. PAR (default 2) GPU runs at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOE_LOAD_LAYERS=1,8,16,23
CACHE=${CACHE:-$HOME/moe_cache}
POOLDIR=${POOLDIR:-runs_typicality}
OUT=${OUT:-runs_typicality_matched}
FROZEN=configs/frozen_b4.json
SEEDS=${SEEDS:-"0 1 2"}
PAR=${PAR:-2}
mkdir -p "$OUT/logs"
REPORT="python diagnostics/report_typicality_matched.py --outdir $OUT --prev-dir $POOLDIR"

for f in "$POOLDIR/pools.json" "$POOLDIR/pool_typical_half.json" "$POOLDIR/pool_random_half.json" \
         "$POOLDIR/routing_profile_v2.npz"; do
    [ -f "$f" ] || { echo "RUN-BLOCKED: missing $f (run diagnostics/typicality_pools.py)"; exit 1; }
done
[ -f "$OUT/warm_starts.json" ] || python diagnostics/typicality_warm_starts.py \
    --pool-dir "$POOLDIR" --frozen $FROZEN --out "$OUT"
python diagnostics/report_minimal_allocation.py --preflight --outdir "$OUT" --cache-dir "$CACHE" --frozen $FROZEN

run_one() {   # strategy pool seed
    local s=$1 pool=$2 seed=$3 sub=() tag=""
    if [ "$pool" != full ]; then sub=(--candidate-subset-file "$POOLDIR/pool_$pool.json"); tag="_pool-$pool"; fi
    local h; h=$(python -c "import json;print(json.load(open('$OUT/warm_starts.json'))['seeds']['$seed']['hash'][:8])")
    local f="$OUT/e3_${s}_B128_b4_d1_s${seed}${tag}_warm-${h}.json"
    if [ -f "$f" ] && $REPORT --validate "$f" >/dev/null 2>&1; then echo "[skip] $f (validated)"; return 0; fi
    echo "[run] $s $pool seed $seed"
    python e3_allocation.py --strategy "$s" --seed "$seed" --B 128 --bits 4 --group-size 128 \
        --percdamp 1.0 --layers 1 8 16 23 --cache-dir "$CACHE" --outdir "$OUT" \
        --frozen-config $FROZEN --routing-cache --warm-start-file "$OUT/warm_starts.json" "${sub[@]}" \
        > "$OUT/logs/${s}_${pool}_s${seed}.log" 2>&1
    $REPORT --validate "$f"
}

for seed in $SEEDS; do
    if [ "$seed" != 0 ]; then
        $REPORT --check-warm 0 || { echo "smoke (seed 0) incomplete or warm starts differ"; exit 1; }
    fi
    n=0
    for pool in full typical_half random_half; do
        for s in random coverage_lp; do
            run_one "$s" "$pool" "$seed" &
            n=$((n + 1))
            if [ "$n" -ge "$PAR" ]; then wait -n; n=$((n - 1)); fi
        done
    done
    wait
    $REPORT --check-warm "$seed"
done
$REPORT
