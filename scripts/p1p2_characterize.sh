#!/usr/bin/env bash
# P1 + P2 under the frozen GPTQ configuration (configs/frozen_b4.json, percdamp 1.0):
#   bash scripts/p1p2_characterize.sh            # damping read from the frozen config
#   bash scripts/p1p2_characterize.sh 1.0        # optional; must equal the config
# E1 (3 char seeds x 4 layers x n0 16/32/64/128) -> E1 gate -> E2 -> E2 gate.
# Stops after the E1 gate if the future-benefit target is not sane, and always
# stops there with P1_ONLY=1.
# Resumable (E1 run dirs are fingerprinted, damping included).
#   P1_ONLY=1 nohup bash scripts/p1p2_characterize.sh > logs/p1.log 2>&1 &
set -euo pipefail
CFG=${FROZEN_CONFIG:-configs/frozen_b4.json}
D=$(python -c "import json;print(json.load(open('$CFG'))['gptq']['percdamp'])")
if [ -n "${1:-}" ] && ! python -c "import sys;sys.exit(float('$1') != float('$D'))"; then
  echo "percdamp $1 != frozen $D in $CFG; refusing to run."; exit 2
fi
C=~/moe_cache; O=runs; B=${BITS:-4}
mkdir -p logs $O
for CS in 0 1 2; do
  python e1_label_sufficiency.py --cache-dir $C --bits $B --n0 16 32 64 128 \
      --char-seed $CS --percdamp $D --frozen-config $CFG --outdir $O
done
if ! python e1_gate.py --bits $B --percdamp $D --frozen-config $CFG --outdir $O; then
  echo "P1 FAIL: future benefit is not sane at percdamp=$D; do NOT run E2."; exit 1
fi
if [ "${P1_ONLY:-0}" = 1 ]; then
  echo "P1 PASS — ready for P2, but P2 has not been started."; exit 0
fi
python e2_signals.py --cache-dir $C --bits $B --min-char-seeds 3 --percdamp $D --outdir $O
python analyze_characterization.py --bits $B --percdamp $D --outdir $O
set +e
python e2_gate.py --bits $B --percdamp $D --outdir $O --permutations 200
code=$?
set -e
case $code in
  0) echo "E2 GATE: case A -> run scripts/p3p4_e3_min.sh";;
  1) echo "E2 GATE: case B (weak/unstable) -> stop and discuss before E3";;
  2) echo "E2 GATE: case C (no signal beats exposure) -> negative finding; E3 optional";;
esac
