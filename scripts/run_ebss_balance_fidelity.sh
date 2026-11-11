#!/usr/bin/env bash
# MoEQuant EBSS balance-strength fidelity experiment (Eq. 4 vs Algorithm 1 x rate vs share). Resumable; every stage
# refuses to run past a failed gate. B=16 generation/quantisation only for variants selected by the frozen gate.
# Usage: STAGE=<source_review|tests|regression|counterfactual|short_generate|short_diagnostics|gate|full_generate|
#               full_diagnostics|quantize|report|all> scripts/run_ebss_balance_fidelity.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venvs/moe/bin/python}
STAGE=${STAGE:-all}
OUT=runs_ebss_balance_fidelity
mkdir -p $OUT/logs
D="$PY diagnostics/ebss_balance_fidelity.py"

run() {  # run <logname> <args...>
  local name=$1; shift
  echo "== $name"
  $D "$@" 2>&1 | grep --line-buffered -v "Loading checkpoint" | tee -a "$OUT/logs/${name}.log"
  return "${PIPESTATUS[0]}"
}
selected() { $PY -c "import json; print(' '.join(json.load(open('$OUT/scientific_gate.json'))['selected_for_B16']))"; }

s_source()  { for f in protocol.json normalization_definitions.json source_ambiguities.json source_equation_review.md; do
                test -f $OUT/$f || { echo "missing frozen $f"; exit 1; }; done; echo "source review frozen"; }
s_tests()   { $PY -m pytest -q tests/test_ebss_balance_fidelity.py 2>&1 | tee $OUT/logs/pytest_balance_fidelity.log
              $PY -m pytest -q tests 2>&1 | tee $OUT/logs/pytest_all.log; }
s_reg()     { run regression --stage regression; }
s_short()   { run short_generate --stage short_generate; }
s_cf()      { run counterfactual --stage counterfactual; }
s_sdiag()   { run short_diagnostics --stage short_diagnostics; }
s_gate()    { run gate --stage gate; }
s_full()    { for v in $(selected); do run full_generate_$v --stage full_generate --variant $v; done; }
s_fdiag()   { [ -n "$(selected)" ] && run full_diagnostics --stage full_diagnostics || echo "no B=16 variant selected"; }
s_quant()   { for v in $(selected); do run quantize_$v --stage quantize --variant $v; done; }
s_report()  { run counterfactual --stage counterfactual; run report --stage report; }

case $STAGE in
  source_review) s_source ;; tests) s_tests ;; regression) s_reg ;; short_generate) s_short ;;
  counterfactual) s_cf ;; short_diagnostics) s_sdiag ;; gate) s_gate ;; full_generate) s_full ;;
  full_diagnostics) s_fdiag ;; quantize) s_quant ;; report) s_report ;;
  all) s_source; s_reg; s_short; s_cf; s_sdiag; s_gate; s_full; s_fdiag; s_quant; s_report ;;
  *) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
