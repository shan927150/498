#!/usr/bin/env bash
# Runs Exp1/2/4/5/6 in parallel on ONE ICRN session, staged so that GPU stays
# < 20 GB and RAM < 72 GB:
#   stage 1: Exp4 + Exp5 + Exp6a (cache)            ~14 GB GPU, ~55 GB RAM
#   stage 2: Exp6a done -> Exp2 starts              Exp2 + Exp4 + Exp5
#   stage 3: Exp5 done  -> Exp6b starts             Exp2 + Exp4 + Exp6b
# Every experiment is resumable: if something dies, just run this file again.
#   nohup bash scripts/parallel/launch.sh > logs/launch.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
mkdir -p logs
echo "[launch] start $(date)"
source scripts/parallel/env.sh
if [ ! -f "$CACHE/meta.json" ]; then
  # This account has no layer cache yet: build ONE cache with the 4 characterized
  # layers + the 3 new ones (one forward pass, ~45-60 min, ~32 GB disk), and let
  # Exp6 use it too (Exp6a then has nothing to do).
  echo "[launch] no cache at $CACHE -> building layers 1 4 8 12 16 20 23 (see logs/cache.log)"
  python cache_inputs.py --layers 1 4 8 12 16 20 23 --cache-dir "$CACHE" > logs/cache.log 2>&1 \
    || { echo "[launch] cache build FAILED, see logs/cache.log"; exit 1; }
  export NEWCACHE=$CACHE
  echo "[launch] cache built $(date)"
fi
ls -d runs/e1_b4_d1_s* >/dev/null 2>&1 || echo "[launch] WARNING: no frozen-damping E1 labels \
in runs/ (runs/e1_b4_d1_s*): Exp1 and the final Exp6 analysis need them (copy them from \
the account that ran P1). Exp2/4/5 and the new-layer E1 are not affected."
bash scripts/parallel/exp1_within_layer.sh > logs/exp1.log 2>&1 \
  && echo "[launch] exp1 done" || echo "[launch] exp1 FAILED (see logs/exp1.log)"
bash scripts/parallel/exp4_diversity.sh > logs/exp4.log 2>&1 & P4=$!
bash scripts/parallel/exp5_damping.sh   > logs/exp5.log 2>&1 & P5=$!
bash scripts/parallel/exp6a_cache.sh    > logs/exp6a.log 2>&1 & P6A=$!
wait $P6A; echo "[launch] exp6a exit $? $(date)"
bash scripts/parallel/exp2_headroom.sh  > logs/exp2.log 2>&1 & P2=$!
wait $P5;  echo "[launch] exp5 exit $? $(date)"
bash scripts/parallel/exp6b_newlayers.sh > logs/exp6b.log 2>&1 & P6B=$!
wait $P2;  echo "[launch] exp2 exit $? $(date)"
wait $P4;  echo "[launch] exp4 exit $? $(date)"
wait $P6B; echo "[launch] exp6b exit $? $(date)"
echo "[launch] all done $(date)"
