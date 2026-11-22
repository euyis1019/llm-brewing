"""Append main-evaluation jobs to the GPU queues (balanced round-robin), each followed by grading.

Usage: python -I scripts/queue_main.py --model qwen3-4b --pids data/main_v1.txt --seeds 2,3,4,5 \
          --arms 'M_Lp_c35=loop:win=28-33,it=1,cond=commit>=35;M_T_c35=temp:mult=1.5,cond=commit>=35' [--gpus 0,1] [--front]
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import argparse
import fcntl
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))
P = f'P={_sys.executable};'


def push(gpu, lines, front):
    p = os.path.join(GD, 'queue', f'gpu{gpu}.txt')
    with open(p, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        cur = [l for l in f.read().split('\n') if l.strip()]
        new = lines + cur if front else cur + lines
        f.seek(0)
        f.truncate()
        f.write('\n'.join(new) + '\n')
    return [l.split('\t')[0] for l in new]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--pids', required=True)
    ap.add_argument('--seeds', default='2,3,4,5')
    ap.add_argument('--arms', required=True)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--pool', default='data/pool.jsonl')
    ap.add_argument('--front', action='store_true')
    ap.add_argument('--extra', default='', help='extra run_gen args')
    a = ap.parse_args()
    gpus = a.gpus.split(',')
    per = {g: [] for g in gpus}
    for i, arm in enumerate([x for x in a.arms.split(';') if x]):
        tag, pol = arm.split('=', 1)
        line = (f"{tag}\t{P} $P -I scripts/run_gen.py --model {a.model} --pool {a.pool} --pids {a.pids} --seeds {a.seeds} "
                f"--policy '{pol}' --tag {tag} {a.extra} && $P -I scripts/grade.py --model {a.model} --tag {tag} --pool {a.pool}")
        per[gpus[i % len(gpus)]].append(line)
    for g, lines in per.items():
        if lines:
            print(f'gpu{g}:', push(g, lines, a.front))


if __name__ == '__main__':
    main()
