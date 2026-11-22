"""Grade gen*.jsonl of a run: writes results/<model>/<tag>/graded.jsonl and prints accuracy per source.

Usage: python -I scripts/grade.py --model qwen3-4b --tag base [--pool data/pool.jsonl]
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(GD, 'pylib'))

POOL = {}


def _grade(r):
    import grader
    it = POOL[r['pid']]
    try:
        ok = bool(grader.grade(r['text'], it['answer'], it['answer_type']))
        ext = grader.extract_answer(r['text'], it['answer_type'])
    except Exception as e:  # grader must not raise; record and count as wrong
        ok, ext = False, f'ERR {e!r}'[:200]
    return dict(jid=r['jid'], pid=r['pid'], seed=r['seed'], source=it['source'], correct=ok, extracted=ext,
                n=r['n'], stop=r['stop'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--pool', default=os.path.join(GD, 'data/pool.jsonl'))
    ap.add_argument('--procs', type=int, default=24)
    a = ap.parse_args()
    for l in open(a.pool):
        it = json.loads(l)
        POOL[it['pid']] = it
    rd = os.path.join(GD, 'results', a.model, a.tag)
    recs = []
    for fn in sorted(os.listdir(rd)):
        if fn.startswith('gen') and fn.endswith('.jsonl'):
            for l in open(os.path.join(rd, fn)):
                r = json.loads(l)
                recs.append({k: r[k] for k in ('jid', 'pid', 'seed', 'text', 'n', 'stop')})
    with Pool(a.procs) as p:
        out = p.map(_grade, recs, chunksize=8)
    with open(os.path.join(rd, 'graded.jsonl'), 'w') as f:
        for g in out:
            f.write(json.dumps(g) + '\n')
    acc = defaultdict(list)
    for g in out:
        acc[g['source']].append(g['correct'])
        acc['ALL'].append(g['correct'])
    for k, v in sorted(acc.items()):
        print(f'{k:10s} n={len(v):5d} acc={sum(v) / len(v):.4f}')
    trunc = sum(g['stop'] == 'len' for g in out)
    print(f'truncated at max_new: {trunc}/{len(out)}; mean tokens {sum(g["n"] for g in out) / len(out):.0f}')


if __name__ == '__main__':
    main()
