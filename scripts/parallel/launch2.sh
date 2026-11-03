#!/usr/bin/env bash
# Self-contained plan (no labels from other accounts). Keeps GPU < 20 GB:
#   A: Exp2 + Exp4 + own E1 on layers 1/8/16/23           (~15 GB GPU)
#   B: own E1 done -> Exp1 (CPU) + E1 on new layers 4/12/20
#   C: Exp2 and Exp4 done -> Exp5 (lowest priority; may not finish in 8 h)
# Waits for a cache build that is already running instead of starting another.
# Resumable: run it again after any failure.
#   nohup bash scripts/parallel/launch2.sh > logs/launch2.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/parallel/env.sh
echo "[launch2] start $(date)"
while pgrep -f "cache_inputs.py" >/dev/null; do
  echo "[launch2] cache build still running... $(date +%H:%M)"; sleep 120
done
if [ ! -f "$CACHE/meta.json" ]; then
  echo "[launch2] building cache layers 1 4 8 12 16 20 23 (logs/cache.log)"
  python cache_inputs.py --layers 1 4 8 12 16 20 23 --cache-dir "$CACHE" >> logs/cache.log 2>&1 \
    || { echo "[launch2] cache build FAILED, see logs/cache.log"; exit 1; }
fi
python -c "import json,sys;L=json.load(open('$CACHE/meta.json'))['layers'];sys.exit(0 if {4,12,20}<=set(L) else 1)" \
  && export NEWCACHE=$CACHE
echo "[launch2] cache ready $(date); new-layer cache = $NEWCACHE"
bash scripts/parallel/exp2_headroom.sh  > logs/exp2.log 2>&1 & P2=$!
bash scripts/parallel/exp4_diversity.sh > logs/exp4.log 2>&1 & P4=$!
bash scripts/parallel/exp6_own_old.sh   > logs/exp6_old.log 2>&1 & P6O=$!
wait $P6O; echo "[launch2] own E1 (old layers) exit $? $(date)"
bash scripts/parallel/exp1_own.sh > logs/exp1.log 2>&1 \
  && echo "[launch2] exp1 done" || echo "[launch2] exp1 FAILED (logs/exp1.log)"
[ -f "$NEWCACHE/meta.json" ] || bash scripts/parallel/exp6a_cache.sh > logs/exp6a.log 2>&1
bash scripts/parallel/exp6_own_new.sh > logs/exp6_new.log 2>&1 & P6N=$!
wait $P4; echo "[launch2] exp4 exit $? $(date)"
wait $P2; echo "[launch2] exp2 exit $? $(date)"
bash scripts/parallel/exp5_damping.sh > logs/exp5.log 2>&1 & P5=$!
wait $P6N; echo "[launch2] exp6 (new layers + depth analysis) exit $? $(date)"
wait $P5;  echo "[launch2] exp5 exit $? $(date)"
echo "[launch2] all done $(date)"
