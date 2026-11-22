#!/bin/bash
# Wait for the trained probe of one run, then run the final taxonomy + early exit (CPU only).
# Usage: nohup bash scripts/taxonomy_when_ready.sh qwen3-4b-base base_fewshot > /dev/null 2>&1 &
# Trigger: results/{model}__{fmt}/probe/{summary.json,oof_full.npy,oof_null2.npy} all exist (summary.json is written
# last by train_probe.py), then 60 s of unchanged sizes. Runs classify (full, raw) -> classify/, classify (full, sub)
# -> classify_csdsub/, early_exit (full) -> early_exit/. Log logs/taxonomy_{model}__{fmt}.log, marker .done/.failed.
# Stop: touch logs/taxonomy_stop. Gives up after 14 h.
ROOT=${BREWING_WORKSPACE:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}/experiments/depth_intervention
PY=${PYTHON:-python}
M=$1; F=$2; P=$ROOT/results/${M}__${F}/probe; NAME=taxonomy_${M}__${F}
cd $ROOT
exec >> logs/$NAME.log 2>&1
echo "$(date '+%F %T') pid $$ waiting for $P/summary.json"
ready=0
for i in $(seq 1 1680); do
  [ -f logs/taxonomy_stop ] && { echo "$(date '+%F %T') stop file"; exit 0; }
  if [ -f $P/summary.json ] && [ -f $P/oof_full.npy ] && [ -f $P/oof_null2.npy ]; then
    s1=$(du -b $P/*.npy | md5sum); sleep 60; s2=$(du -b $P/*.npy | md5sum)
    [ "$s1" == "$s2" ] && { ready=1; break; }
  fi
  sleep 30
done
[ $ready -eq 1 ] || { echo "$(date '+%F %T') gave up"; touch logs/$NAME.failed; exit 1; }
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
$PY scripts/classify.py --model $M --fmt $F --probe full --csd raw --out classify && \
$PY scripts/classify.py --model $M --fmt $F --probe full --csd sub --out classify_csdsub && \
$PY scripts/early_exit.py --model $M --fmt $F --probe full --classify_dir classify
rc=$?
echo "$(date '+%F %T') END rc=$rc"
if [ $rc -eq 0 ]; then touch logs/$NAME.done; else touch logs/$NAME.failed; fi
