"""Does the brewing indicator predict which problems static truncation damages?

Per problem (screen set): damage = mean base accuracy (seeds 0-3) - mean accuracy under a static-exit run (seeds 0,1).
Predictors from the base trajectories (all seeds): share and count of late-committing tokens (engine commit >= 34),
the same restricted to digit tokens, mean final entropy, response length. Reports Spearman correlations and the damage
in predictor tertiles.

Usage: python -I scripts/pruning_predict.py --model qwen3-4b --runs X_s34,X_s33 --pids data/screen_m500.txt
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


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after removing (rank-linear) dependence on z."""
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.vstack([rz, np.ones_like(rz)]).T
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--runs', default='X_s34,X_s33')
    ap.add_argument('--pids', required=True)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    keep = set(l.strip() for l in open(a.pids) if l.strip())
    rd = lambda t: os.path.join(GD, 'results', a.model, t)  # noqa: E731
    from transformers import AutoTokenizer
    import sys
    sys.path.insert(0, HERE)
    import engine as E
    tok = AutoTokenizer.from_pretrained(E.MODELS[a.model][0])
    digit_ids = set()
    base_acc = defaultdict(list)
    for l in open(os.path.join(rd('base'), 'graded.jsonl')):
        g = json.loads(l)
        if g['pid'] in keep and g['seed'] in (0, 1):
            base_acc[g['pid']].append(g['correct'])
    feats = defaultdict(lambda: defaultdict(list))
    for fn in os.listdir(rd('base')):
        if fn.startswith('gen') and fn.endswith('.jsonl'):
            for l in open(os.path.join(rd('base'), fn)):
                r = json.loads(l)
                if r['pid'] not in keep or r['seed'] not in (0, 1, 2, 3):
                    continue
                t = r['tokens']
                for x in set(t):
                    if x not in digit_ids and tok.decode([x]).strip().isdigit():
                        digit_ids.add(x)
                com = np.array(r['commit'])
                ent = np.array(r['ent_m']) / 1000
                dig = np.array([x in digit_ids for x in t])
                late = com >= 34
                f = feats[r['pid']]
                f['late_share'].append(late.mean())
                f['late_count'].append(late.sum())
                f['late_digit_count'].append((late & dig).sum())
                f['late_digit_share'].append((late & dig).sum() / max(1, dig.sum()))
                f['ent'].append(ent.mean())
                f['length'].append(len(t))
    dump = {}
    for run in a.runs.split(','):
        acc = defaultdict(list)
        for l in open(os.path.join(rd(run), 'graded.jsonl')):
            g = json.loads(l)
            acc[g['pid']].append(g['correct'])
        pids = [p for p in acc if p in base_acc and p in feats]
        dmg = np.array([np.mean(base_acc[p]) - np.mean(acc[p]) for p in pids])
        bacc = np.array([np.mean(base_acc[p]) for p in pids])
        solved = bacc == 1.0
        print(f'{run}: problems {len(pids)}, mean damage {dmg.mean() * 100:.1f}pp (seeds 0,1 paired); '
              f'base-solved problems {solved.sum()}, broken share among them {dmg[solved].mean():.3f}')
        dump[run] = dict(n=len(pids), mean_damage=float(dmg.mean()), n_solved=int(solved.sum()),
                         broken_share_solved=float(dmg[solved].mean()), predictors={})
        for k in ('late_share', 'late_count', 'late_digit_count', 'late_digit_share', 'ent', 'length'):
            x = np.array([np.mean(feats[p][k]) for p in pids])
            q = np.quantile(x, [1 / 3, 2 / 3])
            terc = [dmg[x <= q[0]].mean(), dmg[(x > q[0]) & (x <= q[1])].mean(), dmg[x > q[1]].mean()]
            rs = spearman(x[solved], dmg[solved]) if solved.sum() > 5 else float('nan')
            rp = partial_spearman(x, dmg, bacc)
            dump[run]['predictors'][k] = dict(spearman=spearman(x, dmg), spearman_solved=rs, partial_on_base=rp,
                                              tertiles=[float(t) for t in terc])
            print(f'   {k:17s} spearman {spearman(x, dmg):+.3f} | base-solved only {rs:+.3f} | partial on base acc {rp:+.3f} | tertiles '
                  f'{terc[0] * 100:5.1f} / {terc[1] * 100:5.1f} / {terc[2] * 100:5.1f} pp')
    if a.out:
        json.dump(dump, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
