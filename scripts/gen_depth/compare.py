"""Paired comparison of intervention runs against a baseline run (same model, same (pid, seed) jobs).

For each run tag: accuracy, paired difference vs baseline with a problem-level bootstrap CI, fixed/broken counts,
harm on problems the baseline always solves, divergence (share of jobs whose token sequence differs), intervention
rate, and compute per token in layer-equivalents (blocks + K/V propagation of skipped blocks + full-vocab readouts),
relative to full depth.

Usage: python -I scripts/compare.py --model qwen3-4b --base base --runs loopA,loopB [--pids FILE] [--seeds 0,1]
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))


def load(model, tag):
    """tag may be 'a+b' to merge several runs of the same policy (e.g. baselines over different pools)."""
    g, tok, cfg = {}, {}, None
    for t in tag.split('+'):
        rd = os.path.join(GD, 'results', model, t)
        cfg = cfg or json.load(open(os.path.join(rd, 'config.json')))
        for l in open(os.path.join(rd, 'graded.jsonl')):
            r = json.loads(l)
            g[r['jid']] = r
        for fn in os.listdir(rd):
            if fn.startswith('gen') and fn.endswith('.jsonl'):
                for l in open(os.path.join(rd, fn)):
                    r = json.loads(l)
                    tok[r['jid']] = r
    return cfg, g, tok


def token_cost(r, cfg):
    """Per-token cost in block-equivalents for one job record."""
    L, rc, kv = cfg['L'], cfg['c_head'] / cfg['c_layer'], cfg['c_kv'] / cfg['c_layer']
    ex = np.array(r.get('exit', [L - 1] * r['n']), float)
    nr = np.array(r.get('nread', [1] * r['n']), float)
    lp = np.array(r.get('loop', [0] * r['n']), float)
    c = (ex + 1) + (L - 1 - ex) * kv + nr * rc
    pol0 = cfg.get('policy', '')
    if 'fullkv=1' in pol0:  # diagnostic: exited rows still run every block
        c = L + nr * rc
    if pol0.startswith('skip'):
        s0, s1 = map(int, dict(p.split('=', 1) for p in pol0.partition(':')[2].split(',') if p)['range'].split('-'))
        sk = np.array(r.get('skip', [0] * r['n']), float)
        c = c - sk * (s1 - s0 + 1) * (1 - kv)
    pol = cfg.get('policy', '')
    if pol.startswith('loop'):
        kvs = dict(p.split('=', 1) for p in pol.partition(':')[2].split(',') if p)
        a, b = map(int, kvs['win'].split('-'))
        it = int(kvs.get('it', 1))
        mode = kvs.get('mode', 'naive')
        passes = it if mode == 'naive' else it  # damped: K-1 = it extra window passes
        c = c + lp * (passes * (b - a + 1) + (L - 1 - b) + rc)
    elif pol.startswith('contrast'):
        c = c + lp * rc
    return c, L + rc


def mcnemar(b, c):
    """Exact two-sided McNemar p-value for b fixed vs c broken."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def boot(diffs_by_pid, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(diffs_by_pid)
    arr = np.array([np.mean(diffs_by_pid[k]) for k in keys])
    bs = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
    return float(arr.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--base', default='base', help="baseline tag(s), '+'-joined to merge")
    ap.add_argument('--runs', required=True)
    ap.add_argument('--pids', default='')
    ap.add_argument('--seeds', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--by_source', action='store_true')
    ap.add_argument('--by_stratum', default='', help='subset json with per-pid selection rates (e.g. data/main_v1.json)')
    a = ap.parse_args()
    bcfg, bg, btok = load(a.model, a.base)
    keep_p = set(l.strip() for l in open(a.pids)) if a.pids else None
    keep_s = set(int(s) for s in a.seeds.split(',')) if a.seeds else None
    # baseline per-problem solve rate (all seeds available in the baseline run)
    rate = defaultdict(list)
    for r in bg.values():
        rate[r['pid']].append(r['correct'])
    rate = {k: np.mean(v) for k, v in rate.items()}
    rows = []
    for tag in a.runs.split(','):
        cfg, g, tok = load(a.model, tag)
        jids = [j for j in g if j in bg and (keep_p is None or g[j]['pid'] in keep_p)
                and (keep_s is None or g[j]['seed'] in keep_s)]
        if not jids:
            print(tag, 'no overlapping jobs')
            continue
        d_by = defaultdict(list)
        fixed = broken = div = 0
        harm_n = harm_b = 0
        costs, base_cost, nt, irate, chg, fdiv = [], None, 0, [], [], []
        n_int = n_tok = n_chg = 0
        for j in jids:
            x, y = bg[j]['correct'], g[j]['correct']
            d_by[g[j]['pid']].append(float(y) - float(x))
            fixed += (not x) and y
            broken += x and (not y)
            div += btok[j]['tokens'] != tok[j]['tokens']
            if rate[g[j]['pid']] == 1.0:
                harm_n += 1
                harm_b += not y
            c, base_cost = token_cost(tok[j], cfg)
            costs.append(c)
            r = tok[j]
            if 'chg' in r:
                chg.append(sum(r['chg']) / max(1, sum(r.get('loop', [0]))))
                n_chg += sum(r['chg'])
            pol_ = cfg.get('policy', '')
            flags = (r.get('skip', []) if pol_.startswith('skip') else r.get('loop', []) if pol_.startswith(('loop', 'contrast', 'temp'))
                     else [int(x < cfg['L'] - 1) for x in r.get('exit', [])])
            n_int += sum(flags)
            n_tok += r['n']
            bt = btok[j]['tokens']
            fdiv.append(next((k for k, (u, w) in enumerate(zip(bt, r['tokens'])) if u != w), min(len(bt), len(r['tokens']))))
            irate.append(np.mean(r.get('skip', [0])) if cfg.get('policy', '').startswith('skip') else
                         np.mean(r.get('loop', [0])) if cfg.get('policy', '').startswith(('loop', 'contrast', 'temp'))
                         else np.mean(np.array(r.get('exit', [cfg['L'] - 1])) < cfg['L'] - 1))
        cc = np.concatenate(costs)
        bcc = np.concatenate([token_cost(btok[j], bcfg)[0] for j in jids])
        m, lo, hi = boot(d_by)
        acc_b = np.mean([bg[j]['correct'] for j in jids])
        acc_r = np.mean([g[j]['correct'] for j in jids])
        tr_b = np.mean([bg[j]['stop'] == 'len' for j in jids])
        tr_r = np.mean([g[j]['stop'] == 'len' for j in jids])
        fin_b = [bg[j]['correct'] for j in jids if bg[j]['stop'] != 'len']
        fin_r = [g[j]['correct'] for j in jids if g[j]['stop'] != 'len']
        row = dict(tag=tag, policy=cfg.get('policy'), jobs=len(jids), problems=len(d_by), acc_base=acc_b, acc=acc_r,
                   diff=m, ci=[lo, hi], fixed=int(fixed), broken=int(broken), diverged=div / len(jids),
                   harm_on_always_solved=harm_b / max(1, harm_n), n_always_solved_jobs=harm_n,
                   intervention_rate=n_int / max(1, n_tok), chg_per_intervention=(n_chg / max(1, n_int)) if chg else None,
                   mcnemar_p=mcnemar(int(fixed), int(broken)),
                   trunc_base=float(tr_b), trunc=float(tr_r), acc_finished_base=float(np.mean(fin_b)),
                   acc_finished=float(np.mean(fin_r)),
                   first_div_median=float(np.median(fdiv)), cost_rel=float(cc.mean() / base_cost),
                   cost_rel_total=float(cc.sum() / (bcc.sum())), tokens_rel=float(len(cc) / len(bcc)))
        if a.by_source or a.by_stratum:
            groups = defaultdict(dict)
            strat = json.load(open(a.by_stratum))['rates'] if a.by_stratum else {}
            for pid, v in d_by.items():
                if a.by_source:
                    groups['src:' + g[next(j for j in jids if g[j]['pid'] == pid)]['source']][pid] = v
                if a.by_stratum and pid in strat:
                    r_ = strat[pid]
                    groups['stratum:' + ('frontier' if 0 < r_ < 1 else 'solved' if r_ == 1 else 'unsolved')][pid] = v
            row['groups'] = {}
            for k in sorted(groups):
                m_, lo_, hi_ = boot(groups[k])
                row['groups'][k] = dict(n_problems=len(groups[k]), diff=m_, ci=[lo_, hi_])
                print(f"    {k:22s} problems {len(groups[k]):4d} d={m_ * 100:+.2f}pp [{lo_ * 100:+.2f},{hi_ * 100:+.2f}]")
        rows.append(row)
        print(f"{tag:28s} n={len(jids):4d} acc {acc_b:.4f}->{acc_r:.4f} d={m * 100:+.2f}pp [{lo * 100:+.2f},{hi * 100:+.2f}] "
              f"fix/brk {fixed}/{broken} harm(R) {row['harm_on_always_solved']:.3f} div {row['diverged']:.2f} "
              f"rate {row['intervention_rate']:.3f} chg {row['chg_per_intervention'] or 0:.3f} fdiv {row['first_div_median']:.0f} McN p={row['mcnemar_p']:.3f} "
              f"cost/tok {row['cost_rel']:.3f} total {row['cost_rel_total']:.3f} len {row['tokens_rel']:.3f} "
              f"trunc {tr_b:.3f}->{tr_r:.3f} acc|finished {np.mean(fin_b):.3f}->{np.mean(fin_r):.3f}")
    if a.out:
        json.dump(rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
