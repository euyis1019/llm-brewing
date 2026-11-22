"""Summarize a token-choice run: per intervention and position category, how often the intervention changes the
token, and the continuation success of its token minus that of the original token (cluster bootstrap by problem).

Usage: python -I scripts/analyze_tokchoice.py --model qwen3-4b --tag base --out tokchoice_v1
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import argparse
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))


def cluster_boot(vals, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for v, g in zip(vals, groups):
        by[g].append(v)
    keys = list(by)
    sums = np.array([np.sum(by[k]) for k in keys])
    cnts = np.array([len(by[k]) for k in keys])
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(keys), len(keys))
        bs.append(sums[i].sum() / cnts[i].sum())
    return float(np.mean(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--tag', default='base')
    ap.add_argument('--out', default='tokchoice_v1')
    a = ap.parse_args()
    od = os.path.join(GD, 'results', a.model, a.tag, a.out)
    pos = [json.loads(l) for l in open(os.path.join(od, 'positions.jsonl'))]
    V = defaultdict(lambda: defaultdict(list))
    for l in open(os.path.join(od, 'conts.jsonl')):
        c = json.loads(l)
        V[c['pos']][c['tok']].append(c['ok'])
    names = sorted({k[:-4] for k in pos[0] if k.endswith('_tok') and k not in ('base_tok',)})
    cats = sorted({p['cat'] for p in pos})
    S = {}
    print(f"positions {len(pos)}; base reproduces orig: {np.mean([p['base_tok'] == p['orig'] for p in pos]):.4f}")
    hdr = f"{'interv':6s} {'cat':11s} {'n':>5s} {'chg':>6s} {'n_eval':>6s} {'dV(cand-orig)':>14s} {'95% CI':>17s} {'V_orig':>6s}"
    print(hdr)
    for nm in names:
        for cat in cats + ['ALL']:
            ps = [(i, p) for i, p in enumerate(pos) if cat == 'ALL' or p['cat'] == cat]
            chg = [p[nm + '_tok'] != p['orig'] for _, p in ps]
            d, g, vo = [], [], []
            for i, p in ps:
                t = p[nm + '_tok']
                if t == p['orig'] or i not in V or t not in V[i] or p['orig'] not in V[i]:
                    continue
                d.append(np.mean(V[i][t]) - np.mean(V[i][p['orig']]))
                vo.append(np.mean(V[i][p['orig']]))
                g.append(p['pid'])
            row = dict(n=len(ps), chg=float(np.mean(chg)) if chg else 0.0, n_eval=len(d))
            if d:
                m, lo, hi = cluster_boot(d, g)
                row.update(dV=m, ci=[lo, hi], V_orig=float(np.mean(vo)))
                print(f"{nm:6s} {cat:11s} {len(ps):5d} {row['chg']:6.3f} {len(d):6d} {m:+14.3f} [{lo:+.3f},{hi:+.3f}] {np.mean(vo):6.3f}")
            S[f'{nm}|{cat}'] = row
    json.dump(S, open(os.path.join(od, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
