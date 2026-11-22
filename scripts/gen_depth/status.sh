#!/bin/bash
# Queue / GPU status for gen_depth. Usage: bash scripts/status.sh
R=${BREWING_WORKSPACE:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}/experiments/gen_depth
date -u '+%F %T UTC'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for g in 0 1; do
  echo "== gpu$g queue: $(grep -c . $R/queue/gpu$g.txt 2>/dev/null) pending"
  tail -n 2 $R/queue/gpu$g.history 2>/dev/null
  cur=$(tail -n 1 $R/queue/gpu$g.history 2>/dev/null | awk '/START/{print $4}')
  [ -n "$cur" ] && tail -n 1 $R/logs/$cur.log
done
ls $R/logs/*.failed 2>/dev/null
