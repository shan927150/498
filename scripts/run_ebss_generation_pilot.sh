#!/usr/bin/env bash
# Model-matched MoEQuant EBSS reproduction + Probability-Only ablation (pilot 16 x 512). Resumable: every stage
# skips finished work and refuses to run past a failed gate.
# Usage: STAGE=<preflight|source_review|tests|smoke|feasibility|generate|diagnostics|quantize|report|all> \
#        scripts/run_ebss_generation_pilot.sh
# Full B=128 generation is never launched by this script.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venvs/moe/bin/python}
STAGE=${STAGE:-all}
OUT=runs_ebss_generation_pilot
mkdir -p $OUT/logs
D="$PY diagnostics/ebss_generation_pilot.py"

run() {  # run <logname> <args...>
  local name=$1; shift
  echo "== $name"
  $D "$@" 2>&1 | grep --line-buffered -v "Loading checkpoint" | tee -a "$OUT/logs/${name}.log"
  return "${PIPESTATUS[0]}"
}

s_preflight() { run preflight --stage preflight; }
s_source()    { run source_review --stage source_review; }
s_tests()     { $PY -m pytest -q tests/test_ebss_generator.py 2>&1 | tee $OUT/logs/pytest_ebss_generator.log
                $PY -m pytest -q tests 2>&1 | tee $OUT/logs/pytest_all.log; }
s_smoke()     { run smoke --stage smoke; }
s_feas()      { run feasibility --stage feasibility; }
s_generate()  { run generate_po --stage generate --method po; run generate_ebss --stage generate --method ebss; }
s_diag()      { run diagnostics --stage diagnostics; }
s_quantize()  { $PY -c "import json,sys; sys.exit(0 if json.load(open('$OUT/logs/diagnostics_gate.json'))['passed'] else 1)" ||
                  { echo "diagnostics gate failed: no GPTQ"; exit 1; }
                for c in C4R_s0 PO EBSS C4R_s1 C4R_s2; do run quantize_$c --stage quantize --config $c; done; }
s_report()    { run report --stage report; }

case $STAGE in
  preflight) s_preflight ;; source_review) s_source ;; tests) s_tests ;; smoke) s_smoke ;;
  feasibility) s_feas ;; generate) s_generate ;; diagnostics) s_diag ;; quantize) s_quantize ;; report) s_report ;;
  all) s_preflight; s_source; s_tests; s_smoke; s_feas; s_generate; s_diag; s_quantize; s_report ;;
  *) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
