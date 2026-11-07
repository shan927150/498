#!/usr/bin/env bash
# Final aggregate-shift-constrained allocation experiment (locked confirmation).
# Explicit, auditable stages; each can be run alone:
#
#   bash scripts/run_aggregate_constraint.sh split      # replay + verify the confirmation split, build its cache
#   bash scripts/run_aggregate_constraint.sh prepare    # eps_L, feasibility, selections (CPU; no recon read)
#   bash scripts/run_aggregate_constraint.sh verify     # scorer reproduces a stored OLD-split score exactly
#   bash scripts/run_aggregate_constraint.sh tests      # full CPU test suite -> tests_passed.json
#   bash scripts/run_aggregate_constraint.sh lock       # freeze locked_protocol.json (never re-locked)
#   bash scripts/run_aggregate_constraint.sh evaluate   # 15 confirmation runs (resumable, blind)
#   bash scripts/run_aggregate_constraint.sh report     # summary + figures (RUN-BLOCKED if < 15 runs)
#   bash scripts/run_aggregate_constraint.sh all        # every stage in order
# Frozen 4-bit GPTQ, layers 1/8/16/23, B = 128, matched warm starts, seeds 0/1/2.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
OUT=${OUT:-runs_aggregate_constraint}
CONF=${CONF:-$HOME/moe_cache_confirmation}
DRV="$PY diagnostics/aggregate_constraint.py --outdir $OUT --conf-dir $CONF"
mkdir -p "$OUT/logs"
stage=${1:-all}

run_split()    { $PY diagnostics/build_confirmation_split.py --out "$CONF" --replay
                 [ -f "$CONF/meta.json" ] || $PY diagnostics/build_confirmation_split.py --out "$CONF" --cache; }
run_prepare()  { [ -f "$OUT/locked_protocol.json" ] && { echo "locked: prepare refused"; exit 1; }
                 $DRV --prepare 2>&1 | tee "$OUT/logs/prepare.log" | grep -v '^\[E3\]'; }
run_verify()   { MOE_LOAD_LAYERS=1,8,16,23 $DRV --verify-old > "$OUT/logs/verify_old.log" 2>&1; tail -1 "$OUT/logs/verify_old.log"; }
run_tests()    { CUDA_VISIBLE_DEVICES= $PY -m pytest tests -q > "$OUT/logs/pytest_all.log" 2>&1 || true
                 $PY - "$OUT" <<'EOF'
import json, re, sys
t = open(f"{sys.argv[1]}/logs/pytest_all.log").read().splitlines()[-1]
g = lambda k: int(m.group(1)) if (m := re.search(rf"(\d+) {k}", t)) else 0
r = dict(summary=t, passed=g("passed"), failed=g("failed") + g("error"), skipped=g("skipped"))
json.dump(r, open(f"{sys.argv[1]}/tests_passed.json", "w")); print(r)
sys.exit(1 if r["failed"] else 0)
EOF
}
run_lock()     { $DRV --lock; }
run_evaluate() { [ -f "$OUT/locked_protocol.json" ] || { echo "RUN-BLOCKED: protocol not locked"; exit 1; }
                 MOE_LOAD_LAYERS=1,8,16,23 $DRV --evaluate --seeds 0,1 > "$OUT/logs/evaluate_a.log" 2>&1 &
                 MOE_LOAD_LAYERS=1,8,16,23 $DRV --evaluate --seeds 2 > "$OUT/logs/evaluate_b.log" 2>&1 &
                 wait; grep -h '^\[' "$OUT"/logs/evaluate_*.log | grep -v '^\[E3\]\|^\[model\]' || true; }
run_report()   { $PY diagnostics/report_aggregate_constraint.py --outdir "$OUT"; }

case $stage in
    split|prepare|verify|tests|lock|evaluate|report) "run_$stage" ;;
    all) run_split; run_prepare; run_verify; run_tests; run_lock; run_evaluate; run_report ;;
    *) echo "unknown stage $stage"; exit 2 ;;
esac
