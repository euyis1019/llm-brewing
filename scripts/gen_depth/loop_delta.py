"""Does a loop push the next-token distribution along the post-training direction?

At sampled positions of baseline trajectories (teacher-forced prefixes), on the instruct model's natural top-K ids:
  d_loop = log p_loop - log p_nat      (one loop on that position only; several window specs)
  d_post = log p_nat  - log p_base     (instruct vs its base model on the same prefix: the post-training shift)
(log-probs renormalized over the K ids, then centered). Reports per position category and window: mean cosine
(d_loop, d_post), the share of positions with positive cosine, the change of the natural top-1 log-prob under the loop
and its correlation with the post-training change, and a shuffled-position null for the cosine.

Usage: python -I scripts/loop_delta.py --model qwen3-4b --base_model qwen3-4b-base --tag base --pids data/screen_m500.txt \
          --specs 'w18:loop:win=18-25,it=1;w28:loop:win=28-33,it=1' --per_traj 6 --out loopdelta_v1
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
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import engine as E  # noqa: E402
import policies as P  # noqa: E402
from run_gen import build_prompt  # noqa: E402

FORK = {'So', 'so', 'Let', 'Now', 'But', 'Then', 'Therefore', 'Wait', 'wait', 'However', 'Thus', 'Hence', 'Alternatively',
        'Since', 'Because', 'First', 'Next', 'Finally', 'We', 'This', 'The', 'Note', 'Check', 'Hmm', 'Actually'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--base_model', default='qwen3-4b-base')
    ap.add_argument('--tag', default='base')
    ap.add_argument('--pool', default=os.path.join(GD, 'data/pool.jsonl'))
    ap.add_argument('--pids', required=True)
    ap.add_argument('--specs', default='w18:loop:win=18-25,it=1;w24:loop:win=24-31,it=1;w28:loop:win=28-33,it=1;'
                                       'damp:loop:win=18-25,it=1,mode=damped')
    ap.add_argument('--per_traj', type=int, default=6)
    ap.add_argument('--max_traj', type=int, default=300)
    ap.add_argument('--K', type=int, default=32)
    ap.add_argument('--out', default='loopdelta_v1')
    a = ap.parse_args()
    t0 = time.time()
    rd = os.path.join(GD, 'results', a.model, a.tag)
    od = os.path.join(rd, a.out)
    os.makedirs(od, exist_ok=True)
    pool = {json.loads(l)['pid']: json.loads(l) for l in open(a.pool)}
    keep = set(l.strip() for l in open(a.pids) if l.strip())
    recs = []
    for fn in sorted(os.listdir(rd)):
        if fn.startswith('gen') and fn.endswith('.jsonl'):
            for l in open(os.path.join(rd, fn)):
                r = json.loads(l)
                if r['pid'] in keep and r['seed'] in (0, 1):
                    recs.append(r)
    rng = random.Random(0)
    rng.shuffle(recs)
    recs = recs[:a.max_traj]
    eng = E.Engine(a.model, R=24, Tmax=4096)
    tokstr = {}

    def ts(t):
        if t not in tokstr:
            tokstr[t] = eng.tok.decode([t])
        return tokstr[t]
    pos = []
    for r in recs:
        n = r['n']
        lo, hi = 1, max(2, int(n * 0.9))
        com = np.array(r['commit'])
        ent = np.array(r['ent_m']) / 1000
        cats = {'hient': [], 'late': [], 'fork': [], 'digit': [], 'eqdigit': [], 'rand': list(range(lo, hi))}
        for t in range(lo, hi):
            s = ts(r['tokens'][t]).strip()
            if ent[t] >= 1.0:
                cats['hient'].append(t)
            if com[t] >= 35 and ent[t] < 1.0:
                cats['late'].append(t)
            if s in FORK:
                cats['fork'].append(t)
            if s.isdigit():
                cats['digit'].append(t)
                prev = ts(r['tokens'][t - 1]).strip()
                if prev.endswith('='):
                    cats['eqdigit'].append(t)
        for c, v in cats.items():
            if v:
                for t in rng.sample(v, min(len(v), max(1, a.per_traj // 6))):
                    pos.append(dict(jid=r['jid'], pid=r['pid'], seed=r['seed'], t=t, cat=c, tok=r['tokens'][t],
                                    ent=float(ent[t]), commit=int(com[t])))
    prompts = {r['jid']: build_prompt(pool[r['pid']], eng.tok, eng.fam) for r in recs}
    by = {r['jid']: r for r in recs}
    print(f'{len(recs)} trajectories, {len(pos)} positions', flush=True)

    def jobs_for(eval_ids=None):
        js = []
        for i, p in enumerate(pos):
            j = dict(jid=i, key=E.job_key(p['pid'], p['seed']), step0=p['t'], max_new=1,
                     prompt=prompts[p['jid']] + by[p['jid']]['tokens'][:p['t']])
            if eval_ids is not None:
                j['eval_ids'] = eval_ids[i]
            js.append(j)
        return js
    res = {}
    nat_ids = None
    for spec in a.specs.split(';'):
        name, pol = spec.split(':', 1)
        out = eng.generate(jobs_for(nat_ids), P.make_policy(pol, eng), max_new=1, record_first=a.K)
        if nat_ids is None:
            nat_ids = [out[i]['first']['ids'] for i in range(len(pos))]
            res['nat'] = [out[i]['first']['lp_nat'] for i in range(len(pos))]
            res['ent_nat'] = [out[i]['first']['ent_nat'] for i in range(len(pos))]
        res[name] = [out[i]['first']['lp_used'] for i in range(len(pos))]
        print(f'{name} done ({time.time() - t0:.0f}s)', flush=True)
    del eng
    torch.cuda.empty_cache()
    beng = E.Engine(a.base_model, R=24, Tmax=4096)
    out = beng.generate(jobs_for(nat_ids), P.make_policy('base:feat=0', beng), max_new=1, record_first=a.K)
    res['base'] = [out[i]['first']['lp_nat'] for i in range(len(pos))]
    print(f'base model done ({time.time() - t0:.0f}s)', flush=True)
    json.dump(dict(pos=pos, ids=nat_ids, **res), open(os.path.join(od, 'deltas.json'), 'w'))

    def renorm(x):
        x = np.array(x, np.float64)
        return x - np.logaddexp.reduce(x)
    nat = [renorm(x) for x in res['nat']]
    base = [renorm(x) for x in res['base']]
    dpost = [n - b for n, b in zip(nat, base)]
    names = [s.split(':', 1)[0] for s in a.specs.split(';')]
    cats = sorted({p['cat'] for p in pos})
    summary = {}
    rs = np.random.default_rng(0)
    for nm in names:
        dl = [renorm(x) - n for x, n in zip(res[nm], nat)]
        cos = []
        for u, v in zip(dl, dpost):
            u, v = u - u.mean(), v - v.mean()
            den = np.linalg.norm(u) * np.linalg.norm(v)
            cos.append(float(u @ v / den) if den > 1e-9 else np.nan)
        cos = np.array(cos)
        perm = rs.permutation(len(dl))
        null = []
        for i, j in enumerate(perm):
            u, v = dl[i] - dl[i].mean(), dpost[j] - dpost[j].mean()
            den = np.linalg.norm(u) * np.linalg.norm(v)
            null.append(float(u @ v / den) if den > 1e-9 else np.nan)
        null = np.array(null)
        top1_dl = np.array([d[0] for d in dl])      # natural top-1 is ids[0]
        top1_dp = np.array([d[0] for d in dpost])
        mag = np.array([np.abs(d).max() for d in dl])
        for c in cats + ['ALL']:
            m = np.array([c == 'ALL' or p['cat'] == c for p in pos])
            ok = m & ~np.isnan(cos)
            row = dict(n=int(m.sum()), cos=float(np.nanmean(cos[ok])), cos_pos=float(np.mean(cos[ok] > 0)),
                       null_cos=float(np.nanmean(null[m & ~np.isnan(null)])), d_top1=float(top1_dl[m].mean()),
                       corr_top1=float(np.corrcoef(top1_dl[m], top1_dp[m])[0, 1]) if m.sum() > 3 else None,
                       max_abs_delta=float(np.median(mag[m])))
            summary[f'{nm}|{c}'] = row
            print(f"{nm:5s} {c:8s} n={row['n']:4d} cos={row['cos']:+.3f} (null {row['null_cos']:+.3f}) "
                  f"cos>0 {row['cos_pos']:.2f} dlogp(top1) {row['d_top1']:+.3f} corr(top1 loop, top1 post) "
                  f"{row['corr_top1'] if row['corr_top1'] is not None else float('nan'):+.3f} |d|med {row['max_abs_delta']:.3f}")
    json.dump(summary, open(os.path.join(od, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
