#!/usr/bin/env bash
# Official MoEQuant EBSS calibration-quality validation (resumable; every stage skips finished work).
# Usage: STAGE=<preflight|download|inspect_data|compatibility|freeze_selection|tests|baseline_regression|
#               capture|quantize|reconstruction|ppl|report|all> scripts/run_official_ebss_validation.sh
# `all` runs the stages in order and stops at the first failed correctness gate.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venvs/moe/bin/python}
STAGE=${STAGE:-all}
OUT=runs_ebss_official_validation
mkdir -p $OUT/logs
D="$PY diagnostics/official_ebss_validation.py"
QUANT_ORDER="FR_s0 EBSS TR_s0 FC_s0 FR_s1 TR_s1 FC_s1 FR_s2 TR_s2 FC_s2"

run() {  # run <stage> [args...] with a log file
  local name=$1; shift
  echo "== $name $*"
  $D --stage "$name" "$@" 2>&1 | tee -a "$OUT/logs/${name}${LOGSUF:-}.log"
}

gate() {  # stop unless a JSON boolean is true
  $PY -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if d.get('$2') else 1)" ||
    { echo "GATE FAILED: $1 [$2]"; exit 1; }
}

s_preflight()  { run preflight; }
s_download()   { run download; }
s_inspect()    { run inspect_data; }
s_compat()     { run compatibility;
                 $PY -c "import json,sys; d=json.load(open('$OUT/compatibility.json')); sys.exit(0 if d['gate']['proceed_to_gpu'] else 1)" ||
                   { echo "compatibility gate failed: stop before GPU"; exit 1; }; }
s_freeze()     { run freeze_selection;
                 $PY -c "import json,sys; d=json.load(open('$OUT/ebss_selection_manifest.json')); sys.exit(0 if d['status']=='OK' else 1)" ||
                   { echo "DATA-BLOCKED"; exit 1; }; }
s_tests()      { $PY -m pytest -q tests/test_official_ebss_validation.py 2>&1 | tee $OUT/logs/pytest_ebss.log;
                 $PY -m pytest -q tests 2>&1 | tee $OUT/logs/pytest_all.log; }
s_regression() { run baseline_regression; gate $OUT/logs/baseline_regression.json passed; }
s_capture()    { run capture; run capture_repeat; gate $OUT/logs/capture_repeat.json passed; }
s_quantize()   { for c in $QUANT_ORDER; do LOGSUF=_$c run quantize --config $c; done
                 LOGSUF=_repeat run quantize --config FR_s0 --repeat; }
s_recon()      { run reconstruction; }
s_ppl()        { run ppl; }
s_report()     { run report; }

case $STAGE in
  preflight) s_preflight ;; download) s_download ;; inspect_data) s_inspect ;; compatibility) s_compat ;;
  freeze_selection) s_freeze ;; tests) s_tests ;; baseline_regression) s_regression ;; capture) s_capture ;;
  quantize) s_quantize ;; reconstruction) s_recon ;; ppl) s_ppl ;; report) s_report ;;
  all) s_preflight; s_download; s_inspect; s_compat; s_freeze; s_tests; s_regression; s_capture; s_quantize
       s_recon; s_ppl; s_report ;;
  *) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
