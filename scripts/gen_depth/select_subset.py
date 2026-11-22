"""Freeze an evaluation subset from graded baseline runs, stratified by per-problem solve rate.

Strata: frontier (0 < rate < 1), solved (rate == 1, measures harm), unsolved (rate == 0). Within each stratum,
problems are drawn per source in proportion (seeded), with caps so that easy sources do not dominate.
Writes data/<name>.txt (one pid per line) and data/<name>.json (composition and baseline rates).

Usage: python -I scripts/select_subset.py --model qwen3-4b --tags base,base_rest --name subset_v1 \
           --n_frontier 240 --n_solved 100 --n_unsolved 60 [--cap gsm8k=40]
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import argparse
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--tags', default='base')
    ap.add_argument('--name', default='subset_v1')
    ap.add_argument('--n_frontier', type=int, default=240)
    ap.add_argument('--n_solved', type=int, default=100)
    ap.add_argument('--n_unsolved', type=int, default=60)
    ap.add_argument('--cap', default='gsm8k=40')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--use_seeds', default='', help='only these generation seeds define the strata (keep others for evaluation)')
    a = ap.parse_args()
    res = defaultdict(list)
    src = {}
    for tag in a.tags.split(','):
        for l in open(os.path.join(GD, 'results', a.model, tag, 'graded.jsonl')):
            r = json.loads(l)
            if a.use_seeds and r['seed'] not in {int(x) for x in a.use_seeds.split(',')}:
                continue
            res[r['pid']].append(r['correct'])
            src[r['pid']] = r['source']
    rate = {p: sum(v) / len(v) for p, v in res.items()}
    caps = {k: int(v) for k, v in (c.split('=') for c in a.cap.split(',') if c)}
    rng = random.Random(a.seed)
    strata = dict(frontier=[p for p in rate if 0 < rate[p] < 1], solved=[p for p in rate if rate[p] == 1],
                  unsolved=[p for p in rate if rate[p] == 0])
    want = dict(frontier=a.n_frontier, solved=a.n_solved, unsolved=a.n_unsolved)
    chosen, comp = [], {}
    for st, pids in strata.items():
        by = defaultdict(list)
        for p in sorted(pids):
            by[src[p]].append(p)
        for v in by.values():
            rng.shuffle(v)
        n = min(want[st], len(pids))
        # round-robin over sources so each source contributes, respecting caps
        taken = defaultdict(int)
        out = []
        while len(out) < n:
            progressed = False
            for s in sorted(by):
                if len(out) >= n:
                    break
                cap = caps.get(s)
                if by[s] and (cap is None or taken[s] < cap):
                    out.append(by[s].pop())
                    taken[s] += 1
                    progressed = True
            if not progressed:
                break
        chosen += out
        comp[st] = dict(taken=dict(taken), available={s: len(v) + taken[s] for s, v in by.items()})
    with open(os.path.join(GD, 'data', a.name + '.txt'), 'w') as f:
        f.write('\n'.join(chosen) + '\n')
    json.dump(dict(args=vars(a), composition=comp, rates={p: rate[p] for p in chosen}, source={p: src[p] for p in chosen}),
              open(os.path.join(GD, 'data', a.name + '.json'), 'w'), indent=1)
    print(json.dumps(comp, indent=1))
    print('total', len(chosen))


if __name__ == '__main__':
    main()
