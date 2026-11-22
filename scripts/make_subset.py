"""Draw a stratified subset of the round-1 question set for fast runs.

Questions are sampled per (source, fold) cell in proportion to the full set,
so every source and all five folds (including the fold-4 lockbox) keep their
share. Selected questions are renumbered qid = 0..n-1, as the study scripts
require rows = 4*qid + v; the original id is kept in `orig_qid`.

Usage (from the repository root):
    python scripts/make_subset.py --n 500 --name core500
    BREW_BENCH=core500 python experiments/depth_intervention/scripts/extract.py ...

Runs with BREW_BENCH=<name> read data/bench/questions_<name>.jsonl and write to
results/<model>__<fmt>__<name>, so they never touch full-set results.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments',
                     'depth_intervention', 'data', 'bench')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--name', default='core500')
    ap.add_argument('--seed', type=int, default=20261122)
    a = ap.parse_args()

    qs = [json.loads(l) for l in open(os.path.join(BENCH, 'questions.jsonl'))]
    cells = defaultdict(list)
    for q in qs:
        cells[(q['source'], q['fold'])].append(q)

    # Largest-remainder allocation so the cell counts sum to exactly n.
    keys = sorted(cells)
    exact = np.array([len(cells[k]) * a.n / len(qs) for k in keys])
    take = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - take))[: a.n - take.sum()]:
        take[i] += 1

    rng = np.random.default_rng(a.seed)
    picked = []
    for k, t in zip(keys, take):
        idx = rng.choice(len(cells[k]), size=t, replace=False)
        picked += [cells[k][i] for i in idx]
    picked.sort(key=lambda q: q['qid'])

    out = os.path.join(BENCH, f'questions_{a.name}.jsonl')
    with open(out, 'w') as f:
        for new_qid, q in enumerate(picked):
            row = dict(q, orig_qid=q['qid'], qid=new_qid)
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    by_src = defaultdict(int)
    by_fold = defaultdict(int)
    for q in picked:
        by_src[q['source']] += 1
        by_fold[q['fold']] += 1
    print(f'{len(picked)} questions -> {out}')
    print('by source:', dict(sorted(by_src.items())))
    print('by fold:', dict(sorted(by_fold.items())))


if __name__ == '__main__':
    main()
