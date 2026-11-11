#!/usr/bin/env bash
# C4-prefix diversity rescue (modified EBSS; B=16, P=64 + G=448). Resumable; every stage refuses to run past a failed gate.
# Usage: STAGE=<freeze|regression|smoke|generate|capture|diagnostics|quantize|report|tests|all> scripts/run_ebss_prefix_rescue.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venvs/moe/bin/python}
STAGE=${STAGE:-all}
OUT=runs_ebss_prefix_rescue
mkdir -p $OUT/logs $OUT/generation_logs
D="$PY diagnostics/ebss_prefix_rescue.py"
run() { local name=$1; shift; echo "== $name"
        $D "$@" 2>&1 | grep --line-buffered -v "Loading checkpoint" | tee -a "$OUT/logs/${name}.log"; return "${PIPESTATUS[0]}"; }
s_freeze() { run freeze --stage freeze; }
s_reg()    { run regression --stage regression; }
s_smoke()  { run smoke --stage smoke; }
s_gen()    { for c in PX_PO PX_EBSS; do run generate_$c --stage generate --cond $c; done; }
s_cap()    { run capture --stage capture; }
s_diag()   { run diagnostics --stage diagnostics; }
s_quant()  { for c in PX_PO PX_EBSS; do run quantize_$c --stage quantize --cond $c; done; }
s_report() { run report --stage report; }
s_tests()  { $PY -m pytest -q tests/test_ebss_prefix_rescue.py 2>&1 | tee $OUT/logs/pytest_prefix_rescue.log
             $PY -m pytest -q tests 2>&1 | tee $OUT/logs/pytest_all.log; }
case $STAGE in
  freeze) s_freeze ;; regression) s_reg ;; smoke) s_smoke ;; generate) s_gen ;; capture) s_cap ;;
  diagnostics) s_diag ;; quantize) s_quant ;; report) s_report ;; tests) s_tests ;;
  all) s_freeze; s_reg; s_smoke; s_gen; s_cap; s_diag; s_quant; s_report ;;
  *) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
