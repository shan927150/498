#!/usr/bin/env bash
# Representativeness-constrained Coverage-LP (coverage_lp_repr), B = 128, seeds 0/1/2.
# Full 2,000-sequence candidate pool, the matched warm starts of
# runs_typicality_matched/warm_starts.json (reused, never resampled), tau =
# median d(s) (prespecified), frozen 4-bit GPTQ, layers 1/8/16/23, test-split
# reconstruction over the fixed evaluation universe. Baselines are REUSED from
# runs_typicality_matched/ after validation (no reruns).
#
#   bash scripts/run_repr_constrained.sh            # CPU checks -> seed 0 -> check -> seeds 1, 2 -> report
#   SEEDS=0 bash scripts/run_repr_constrained.sh    # smoke only
# Resumable: validated outputs are skipped. PAR (default 2) GPU runs at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOE_LOAD_LAYERS=1,8,16,23
PY=${PY:-python}
CACHE=${CACHE:-$HOME/moe_cache}
POOLDIR=${POOLDIR:-runs_typicality}
MATCHED=${MATCHED:-runs_typicality_matched}
OUT=${OUT:-runs_repr_constrained}
FROZEN=configs/frozen_b4.json
SEEDS=${SEEDS:-"0 1 2"}
PAR=${PAR:-2}
mkdir -p "$OUT/logs"
REPORT="$PY diagnostics/report_repr_constrained.py --outdir $OUT --matched-dir $MATCHED --pool-dir $POOLDIR"

for f in "$POOLDIR/pools.json" "$POOLDIR/typicality_scores.csv" "$MATCHED/warm_starts.json" \
         "$MATCHED/routing_profile_v2.npz" "$MATCHED/eval_universe_b06e33af24.json"; do
    [ -f "$f" ] || { echo "RUN-BLOCKED: missing $f"; exit 1; }
done
# same routing cache and evaluation universe as the matched baselines (byte-identical copies;
# e3_allocation.check_eval_universe then asserts the identical universe for every new run)
for f in routing_profile_v2.npz eval_universe_b06e33af24.json warm_starts.json; do
    if [ -f "$OUT/$f" ]; then cmp -s "$MATCHED/$f" "$OUT/$f" || { echo "$OUT/$f differs from $MATCHED/$f"; exit 1; }
    else cp "$MATCHED/$f" "$OUT/$f"; fi
done
$PY diagnostics/repr_feasibility.py --pool-dir "$POOLDIR" --matched-dir "$MATCHED" --out "$OUT" \
    > "$OUT/logs/feasibility.log" 2>&1 || { tail -5 "$OUT/logs/feasibility.log"; echo "constraint infeasible: not relaxing tau"; exit 1; }
tail -4 "$OUT/logs/feasibility.log"
$PY diagnostics/report_minimal_allocation.py --preflight --outdir "$OUT" --cache-dir "$CACHE" --frozen $FROZEN

run_one() {   # seed
    local seed=$1 h
    h=$($PY -c "import json;print(json.load(open('$OUT/warm_starts.json'))['seeds']['$seed']['hash'][:8])")
    local f="$OUT/e3_coverage_lp_repr_B128_b4_d1_s${seed}_warm-${h}.json"
    if [ -f "$f" ] && $REPORT --validate "$f" >/dev/null 2>&1; then echo "[skip] $f (validated)"; return 0; fi
    echo "[run] coverage_lp_repr seed $seed"
    $PY e3_allocation.py --strategy coverage_lp_repr --seed "$seed" --B 128 --bits 4 --group-size 128 \
        --percdamp 1.0 --layers 1 8 16 23 --cache-dir "$CACHE" --outdir "$OUT" \
        --frozen-config $FROZEN --routing-cache --warm-start-file "$OUT/warm_starts.json" \
        --repr-tau median --typicality-scores "$POOLDIR/typicality_scores.csv" \
        > "$OUT/logs/coverage_lp_repr_s${seed}.log" 2>&1
    $REPORT --validate "$f"
}

# seed 0 first (smoke); later seeds only once seed 0 is complete and valid
case " $SEEDS " in *" 0 "*) run_one 0 ;; esac
$REPORT --check-warm 0 || { echo "smoke (seed 0) incomplete or invalid"; exit 1; }
n=0
for seed in $SEEDS; do
    [ "$seed" = 0 ] && continue
    run_one "$seed" &
    n=$((n + 1)); if [ "$n" -ge "$PAR" ]; then wait -n; n=$((n - 1)); fi
done
wait
for seed in $SEEDS; do $REPORT --check-warm "$seed"; done
$REPORT
