#!/usr/bin/env bash
# P1 + P2 under ONE frozen GPTQ damping:  bash scripts/p1p2_characterize.sh 0.2
# E1 (3 char seeds x 4 layers x n0 16/32/64/128) -> E1 gate -> E2 -> E2 gate.
# Stops after the E1 gate if the future-benefit target is not sane.
# Resumable (E1 run dirs are fingerprinted, damping included).
#   nohup bash scripts/p1p2_characterize.sh 0.2 > logs/p1p2.log 2>&1 &
set -euo pipefail
D=${1:?usage: p1p2_characterize.sh <frozen percdamp>}
C=~/moe_cache; O=runs; B=${BITS:-4}
mkdir -p logs $O
for CS in 0 1 2; do
  python e1_label_sufficiency.py --cache-dir $C --bits $B --n0 16 32 64 128 \
      --char-seed $CS --percdamp $D --outdir $O
done
if ! python e1_gate.py --bits $B --percdamp $D --outdir $O; then
  echo "E1 GATE FAILED: future benefit is not sane at percdamp=$D; do NOT run E2."; exit 1
fi
python e2_signals.py --cache-dir $C --bits $B --min-char-seeds 3 --percdamp $D --outdir $O
python analyze_characterization.py --bits $B --percdamp $D --outdir $O
set +e
python e2_gate.py --bits $B --percdamp $D --outdir $O --permutations 200
code=$?
set -e
case $code in
  0) echo "E2 GATE: case A -> run scripts/p3p4_e3_min.sh $D";;
  1) echo "E2 GATE: case B (weak/unstable) -> stop and discuss before E3";;
  2) echo "E2 GATE: case C (no signal beats exposure) -> negative finding; E3 optional";;
esac
