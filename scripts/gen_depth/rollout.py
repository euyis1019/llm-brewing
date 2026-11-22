"""Token-level loop effect by coupled rollouts.

From baseline trajectories of frontier problems, pick positions t by category (late-commit, high-entropy, random,
...). For each position and each of M noise keys, continue generation from the exact prefix (prompt + tokens[:t])
twice with shared noise: (a) normally, (b) with a loop on the first generated token only (token t). Everything after
token t is computed normally (but sees the looped K/V of token t in (b)). The paired success difference estimates the
loop effect at that token; the base success rate estimates the prefix value V(t).

Usage: python -I scripts/rollout.py --model qwen3-4b --tag base --pids data/screen_m500.txt --pool data/pool.jsonl \
          --loop 'loop:win=18-25,it=1,cond=step<1' --M 4 --per_cat 2 --out rollout_v1
Output: results/<model>/<tag>/<out>/rollouts.jsonl (one line per (position, m) with both outcomes).
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
sys.path.insert(0, os.path.join(GD, 'pylib'))

import torch  # noqa: E402

import engine as E  # noqa: E402
import policies as P  # noqa: E402
from run_gen import build_prompt  # noqa: E402


def pick_positions(r, per_cat, rng, lo_frac=0.02, hi_frac=0.8):
    n = r['n']
    lo, hi = max(1, int(n * lo_frac)), max(2, int(n * hi_frac))
    idx = np.arange(lo, hi)
    com = np.array(r['commit'])[lo:hi]
    ent = np.array(r['ent_m'])[lo:hi] / 1000
    cats = {
        'late': idx[(com >= 35) & (ent < 1.0)],
        'hient': idx[(ent >= 1.0) & (com < 35)],
        'late_hient': idx[(com >= 35) & (ent >= 1.0)],
        'mid': idx[(ent >= 0.3) & (ent < 1.0) & (com < 35)],
        'rand': idx,
    }
    out = []
    for c, v in cats.items():
        v = list(v)
        rng.shuffle(v)
        out += [(c, int(t)) for t in v[:per_cat]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--tag', default='base')
    ap.add_argument('--pool', default=os.path.join(GD, 'data/pool.jsonl'))
    ap.add_argument('--pids', required=True)
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--loop', default='loop:win=18-25,it=1,cond=step<1')
    ap.add_argument('--M', type=int, default=4)
    ap.add_argument('--per_cat', type=int, default=2)
    ap.add_argument('--frontier_only', type=int, default=1)
    ap.add_argument('--max_total', type=int, default=3072)
    ap.add_argument('--R', type=int, default=48)
    ap.add_argument('--out', default='rollout_v1')
    ap.add_argument('--shard', default='0/1')
    a = ap.parse_args()
    t0 = time.time()
    import grader
    rd = os.path.join(GD, 'results', a.model, a.tag)
    pool = {json.loads(l)['pid']: json.loads(l) for l in open(a.pool)}
    keep = set(l.strip() for l in open(a.pids) if l.strip())
    seeds = set(int(s) for s in a.seeds.split(','))
    graded = {json.loads(l)['jid']: json.loads(l) for l in open(os.path.join(rd, 'graded.jsonl'))}
    rate = {}
    for g in graded.values():
        rate.setdefault(g['pid'], []).append(g['correct'])
    recs = []
    for fn in sorted(os.listdir(rd)):
        if fn.startswith('gen') and fn.endswith('.jsonl'):
            for l in open(os.path.join(rd, fn)):
                r = json.loads(l)
                if r['pid'] in keep and r['seed'] in seeds and r['stop'] == 'eos':
                    if a.frontier_only and not (0 < np.mean(rate[r['pid']]) < 1):
                        continue
                    recs.append(r)
    recs.sort(key=lambda r: r['jid'])
    si, sn = map(int, a.shard.split('/'))
    recs = recs[si::sn]
    by_jid = {r['jid']: r for r in recs}
    rng = random.Random(1234 + si)
    eng = E.Engine(a.model, R=a.R, Tmax=4096)
    pol_base = P.make_policy('base', eng)
    pol_loop = P.make_policy(a.loop, eng)
    jobs, meta = [], {}
    for r in recs:
        prompt = build_prompt(pool[r['pid']], eng.tok, eng.fam)
        for cat, t in pick_positions(r, a.per_cat, rng):
            for m in range(a.M):
                jid = f"{r['jid']}|{t}|{m}"
                jobs.append(dict(jid=jid, key=E.job_key(r['pid'], r['seed'], t, m), prompt=prompt + r['tokens'][:t],
                                 max_new=min(2048, a.max_total - t)))
                meta[jid] = dict(src=r['jid'], pid=r['pid'], t=t, m=m, cat=cat, base_ok=bool(graded[r['jid']]['correct']),
                                 commit=r['commit'][t], ent=r['ent_m'][t] / 1000, n=r['n'], orig_tok=r['tokens'][t])
    print(f'{len(recs)} trajectories, {len(jobs)} rollout jobs per condition', flush=True)
    od = os.path.join(rd, a.out)
    os.makedirs(od, exist_ok=True)
    res = {}
    for cond, pol in (('base', pol_base), ('loop', pol_loop)):
        def on_done(jid, b, cond=cond):
            mm = meta[jid]
            src = mm['src']
            prefix = by_jid[src]['tokens'][:mm['t']]
            text = eng.tok.decode(prefix + [int(x) for x in b['tokens']], skip_special_tokens=True)
            it = pool[mm['pid']]
            res.setdefault(jid, {})[cond] = dict(ok=bool(grader.grade(text, it['answer'], it['answer_type'])),
                                                 tok0=int(b['tokens'][0]), n=len(b['tokens']),
                                                 chg0=int(b['chg'][0]) if 'chg' in b else 0)
        eng.generate(jobs, pol, max_new=2048, log_every=500, log=lambda s: print(cond, s, flush=True), on_done=on_done)
        print(f'{cond} done ({time.time() - t0:.0f}s)', flush=True)
    with open(os.path.join(od, f'rollouts{"" if sn == 1 else f".shard{si}"}.jsonl'), 'w') as f:
        for jid, v in res.items():
            f.write(json.dumps(dict(jid=jid, **meta[jid], **{f'{c}_{k}': x for c in v for k, x in v[c].items()})) + '\n')
    print(f'DONE {len(res)} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
