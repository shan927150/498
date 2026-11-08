#!/usr/bin/env bash
# Low-budget calibration allocation sweep (exploratory; old test split).
# Budgets 16/32/64 x {full_random, coverage_lp, typical_random, typical_coverage_lp} x seeds 0/1/2
# = 36 new E3 runs; B = 128 anchors are the validated matched-warm-start runs (selections verified
# identical by --prepare). Warm starts W(seed,B) = first B/4 of the matched W(seed,128).
#
#   bash scripts/run_low_budget_sweep.sh            # all stages (resumable)
#   STAGE=prepare|tests|smoke|rest|report bash scripts/run_low_budget_sweep.sh
# PAR (default 2) controlled concurrent GPU runs on the 20 GB slice.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOE_LOAD_LAYERS=1,8,16,23
PY=${PY:-python}
CACHE=${CACHE:-$HOME/moe_cache}
OUT=${OUT:-runs_low_budget}
MATCHED=runs_typicality_matched
POOLDIR=runs_typicality
FROZEN=configs/frozen_b4.json
PAR=${PAR:-2}
STAGE=${STAGE:-all}
mkdir -p "$OUT/logs" "$OUT/figures"

stage_prepare() {
    for f in "$MATCHED/routing_profile_v2.npz" "$MATCHED/warm_starts.json" "$MATCHED/eval_universe_b06e33af24.json" \
             "$POOLDIR/pool_typical_half.json" "$POOLDIR/pools.json" "$FROZEN"; do
        [ -f "$f" ] || { echo "RUN-BLOCKED: missing $f"; exit 1; }
    done
    for f in routing_profile_v2.npz eval_universe_b06e33af24.json; do     # identical routing rows / universe
        if [ -f "$OUT/$f" ]; then cmp -s "$MATCHED/$f" "$OUT/$f" || { echo "$OUT/$f differs"; exit 1; }
        else cp "$MATCHED/$f" "$OUT/$f"; fi
    done
    $PY diagnostics/low_budget.py --outdir "$OUT" --prepare 2>&1 | grep -v '^\[E3\]' | tee "$OUT/logs/prepare.log"
    $PY diagnostics/report_minimal_allocation.py --preflight --outdir "$OUT" --cache-dir "$CACHE" --frozen $FROZEN | tail -1
}

stage_tests() {
    CUDA_VISIBLE_DEVICES= $PY -m pytest tests -q > "$OUT/logs/pytest_all.log" 2>&1 || { tail -5 "$OUT/logs/pytest_all.log"; exit 1; }
    tail -1 "$OUT/logs/pytest_all.log"
}

run_one() {   # code B seed
    local code=$1 B=$2 seed=$3 st sub=() tag=""
    case $code in FR|TR) st=random ;; FC|TC) st=coverage_lp ;; esac
    case $code in TR|TC) sub=(--candidate-subset-file "$POOLDIR/pool_typical_half.json"); tag="_pool-typical_half" ;; esac
    local h; h=$($PY -c "import json;print(json.load(open('$OUT/warm_starts_B$B.json'))['seeds']['$seed']['hash'][:8])")
    local f="$OUT/e3_${st}_B${B}_b4_d1_s${seed}${tag}_warm-${h}.json"
    if [ -f "$f" ] && $PY diagnostics/low_budget.py --outdir "$OUT" --validate "$f" >/dev/null 2>&1; then
        echo "[skip] $code B$B s$seed (validated)"; return 0; fi
    echo "[run] $code B$B s$seed"
    $PY e3_allocation.py --strategy "$st" --seed "$seed" --B "$B" --bits 4 --group-size 128 --percdamp 1.0 \
        --layers 1 8 16 23 --cache-dir "$CACHE" --outdir "$OUT" --frozen-config $FROZEN --routing-cache \
        --warm-start-file "$OUT/warm_starts_B$B.json" "${sub[@]}" > "$OUT/logs/${code}_B${B}_s${seed}.log" 2>&1
    $PY diagnostics/low_budget.py --outdir "$OUT" --validate "$f"
}

run_batch() {  # list of "code B seed"
    local n=0
    for job in "$@"; do
        run_one $job &
        n=$((n + 1)); if [ "$n" -ge "$PAR" ]; then wait -n; n=$((n - 1)); fi
    done
    wait
}

stage_smoke() {
    run_batch "FR 16 0" "FC 16 0" "TR 16 0" "TC 16 0"
    $PY - "$OUT" <<'EOF'
import glob, json, sys
fs = sorted(glob.glob(f"{sys.argv[1]}/e3_*_B16_*_s0_*warm-*.json"))
u = {json.dumps(json.load(open(f))["eval_universe"]) for f in fs}
n = {json.load(open(f))["summary"]["n_experts_test_observed"] for f in fs}
assert len(fs) == 4 and len(u) == 1 and n == {240}, (len(fs), len(u), n)
print("smoke ok: 4 runs, one evaluation universe, 240 experts each")
EOF
}

stage_rest() {
    jobs=()
    for B in 16 32 64; do for s in 0 1 2; do for c in FR FC TR TC; do
        [ "$B" = 16 ] && [ "$s" = 0 ] && continue
        jobs+=("$c $B $s")
    done; done; done
    run_batch "${jobs[@]}"
}

stage_report() { $PY diagnostics/report_low_budget.py --outdir "$OUT"; }

case $STAGE in
    all) stage_prepare; stage_tests; stage_smoke; stage_rest; stage_report ;;
    prepare|tests|smoke|rest|report) "stage_$STAGE" ;;
    *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
