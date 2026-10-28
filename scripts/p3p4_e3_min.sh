#!/usr/bin/env bash
# P3 + P4, ONLY after E2 gate case A / A_budget:  bash scripts/p3p4_e3_min.sh
# (damping read from configs/frozen_b4.json; an optional argument must equal it)
# Corrected support diagnostic (fixed routing cache) -> minimal E3 at B=128 x 3 seeds:
# random / coverage / tlp (E2 predictor) / oracle, then the equal-compute baseline.
set -euo pipefail
CFG=${FROZEN_CONFIG:-configs/frozen_b4.json}
D=$(python -c "import json;print(json.load(open('$CFG'))['gptq']['percdamp'])")
if [ -n "${1:-}" ] && ! python -c "import sys;sys.exit(float('$1') != float('$D'))"; then
  echo "percdamp $1 != frozen $D in $CFG; refusing to run."; exit 2
fi
DT=$(python -c "print(f'd{float($D):g}')")    # name tag, as moe_calib.runid.damp_tag
C=~/moe_cache; O=runs; B=${BITS:-4}; BUDGET=${BUDGET:-128}
mkdir -p logs $O
python support_diagnostic.py --cache-dir $C --bits $B --percdamp $D \
    --budgets 64 128 256 512 --seeds 0 1 2 --outdir $O
for SEED in 0 1 2; do
  for S in random coverage tlp oracle; do
    python e3_allocation.py --strategy $S --B $BUDGET --bits $B --percdamp $D \
        --seed $SEED --cache-dir $C --outdir $O
  done
  # equal-compute baseline: random with the extra sequences tlp's overhead would buy
  F=$(ls $O/e3_tlp_*_B${BUDGET}_b${B}_${DT}_s${SEED}.json | head -1)
  X=$(python -c "import json;print(json.load(open('$F'))['summary']['cost']['equiv_extra_random_seqs'])")
  python e3_allocation.py --strategy random_plus --delta-seqs $X --B $BUDGET --bits $B \
      --percdamp $D --seed $SEED --cache-dir $C --outdir $O
done
echo "P3/P4 DONE"
