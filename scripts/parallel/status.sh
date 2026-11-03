#!/usr/bin/env bash
# Progress at a glance:  bash scripts/parallel/status.sh
cd "$(dirname "$0")/../.."
echo "== launcher"; tail -n 4 logs/launch2.log 2>/dev/null || tail -n 4 logs/launch.log 2>/dev/null
for f in cache exp1 exp2 exp4 exp5 exp6_old exp6_new; do
  echo "== $f"; tail -n 2 logs/$f.log 2>/dev/null | cut -c1-160
done
echo "== GPU"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null \
  || nvidia-smi | grep -i mib
M=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null)
[ -n "$M" ] && echo "== RAM used by this session: $((M/1024/1024/1024)) GB of $((${MEM_LIMIT:-0}/1024/1024/1024)) GB"
echo "== python processes"; pgrep -af "python .*(diagnostics|e1_label|cache_inputs)" | cut -c1-120
