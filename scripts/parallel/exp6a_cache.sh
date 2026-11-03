#!/usr/bin/env bash
# Exp6a (GPU, full model ~36 GB RAM, ~40 min): cache inputs of the NEW layers 4/12/20.
set -euo pipefail; source scripts/parallel/env.sh
if [ -f $NEWCACHE/meta.json ]; then echo "cache exists: $NEWCACHE"; exit 0; fi
python cache_inputs.py --layers 4 12 20 --cache-dir $NEWCACHE
