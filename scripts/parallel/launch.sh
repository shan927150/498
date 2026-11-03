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
