"""Token-choice test: at position t of a baseline trajectory, is the token a loop would emit better than the original?

For frontier baseline trajectories and positions t chosen by category, compute with the run's own noise (step0=t):
  orig   the generated token (the base policy reproduces it up to bf16 noise)
  loop   token emitted when position t is computed with the loop policy (same noise)
  alt    token from an ordinary resample of the base distribution (different noise): the null comparator
For each position, continue M times from prefix+orig, and from prefix+loop / prefix+alt when they differ from orig
(base policy; continuation noise keys shared across candidates), and grade. A loop that improves the token choice gives
V(loop) > V(orig) on positions where it changed the token; an uninformative perturbation gives about V(alt) - V(orig).

Usage: python -I scripts/tokchoice.py --model qwen3-4b --tag base --pids data/screen_m500.txt \
          --specs 'w18:loop:win=18-25,it=1;w24:loop:win=24-31,it=1' --M 4 --per_cat 1 --out tokchoice_v1
Output: results/<model>/<tag>/<out>/positions.jsonl (step 1) and conts.jsonl (one line per continuation).
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

import engine as E  # noqa: E402
import policies as P  # noqa: E402
from run_gen import build_prompt  # noqa: E402

CATS = ('late', 'hient', 'late_hient', 'mid', 'rand', 'digit_late')


def categories(r, lo, hi, tokstr):
    idx = np.arange(lo, hi)
    com = np.array(r['commit'])[lo:hi]
    ent = np.array(r['ent_m'])[lo:hi] / 1000
    dig = np.array([tokstr(t).strip().isdigit() for t in r['tokens'][lo:hi]])
    return {
        'late': idx[(com >= 35) & (ent < 1.0)],
        'hient': idx[(ent >= 1.0) & (com < 35)],
        'late_hient': idx[(com >= 35) & (ent >= 1.0)],
        'mid': idx[(ent >= 0.3) & (ent < 1.0) & (com < 35)],
        'rand': idx,
        'digit_late': idx[dig & (com >= 34)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--tag', default='base')
    ap.add_argument('--pool', default=os.path.join(GD, 'data/pool.jsonl'))
    ap.add_argument('--pids', required=True)
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--specs', default='w18:loop:win=18-25,it=1',
                    help="';'-separated name:policy pairs; each policy computes the token at t (same noise)")
    ap.add_argument('--M', type=int, default=4)
    ap.add_argument('--per_cat', type=int, default=1)
    ap.add_argument('--max_total', type=int, default=3072)
    ap.add_argument('--R', type=int, default=48)
    ap.add_argument('--out', default='tokchoice_v1')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    import grader
    rd = os.path.join(GD, 'results', a.model, a.tag)
    od = os.path.join(rd, a.out)
    os.makedirs(od, exist_ok=True)
    pool = {}
    for l in open(a.pool):
        it = json.loads(l)
        pool[it['pid']] = it
    keep = set(l.strip() for l in open(a.pids) if l.strip())
    seeds = set(int(s) for s in a.seeds.split(','))
    graded = {}
    for l in open(os.path.join(rd, 'graded.jsonl')):
        g = json.loads(l)
        graded[g['jid']] = g
    rate = {}
    for g in graded.values():
        rate.setdefault(g['pid'], []).append(g['correct'])
    recs = []
    for fn in sorted(os.listdir(rd)):
        if fn.startswith('gen') and fn.endswith('.jsonl'):
            for l in open(os.path.join(rd, fn)):
                r = json.loads(l)
                if r['pid'] in keep and r['seed'] in seeds and r['stop'] == 'eos' and 0 < np.mean(rate[r['pid']]) < 1:
                    recs.append(r)
    recs.sort(key=lambda r: r['jid'])
    eng = E.Engine(a.model, R=a.R, Tmax=4096)
    tokstr = lambda t: eng.tok.decode([t])  # noqa: E731
    pol_base = P.make_policy('base', eng)
    specs = [(x.split(':', 1)[0], P.make_policy(x.split(':', 1)[1], eng)) for x in a.specs.split(';') if x]
    rng = random.Random(a.seed)
    prompts = {r['jid']: build_prompt(pool[r['pid']], eng.tok, eng.fam) for r in recs}
    pos = []
    for r in recs:
        lo, hi = max(1, int(r['n'] * 0.02)), max(2, int(r['n'] * 0.8))
        for c, v in categories(r, lo, hi, tokstr).items():
            v = list(v)
            rng.shuffle(v)
            for t in v[:a.per_cat]:
                pos.append(dict(src=r['jid'], pid=r['pid'], seed=r['seed'], t=int(t), cat=c, orig=r['tokens'][t],
                                commit=r['commit'][t], ent=r['ent_m'][t] / 1000, traj_ok=bool(graded[r['jid']]['correct'])))
    print(f'{len(recs)} trajectories, {len(pos)} positions', flush=True)
    by_src = {r['jid']: r for r in recs}

    # step 1: one token under base (same noise), loop (same noise), base with alternative noise
    def one_token(pol, keyfn, name):
        jobs = []
        for i, p in enumerate(pos):
            r = by_src[p['src']]
            jobs.append(dict(jid=i, key=keyfn(p), step0=p['t'], max_new=1,
                             prompt=prompts[p['src']] + r['tokens'][:p['t']]))
        out = eng.generate(jobs, pol, max_new=1)
        for i, p in enumerate(pos):
            p[name] = int(out[i]['tokens'][0])
        print(f'step1 {name} done ({time.time() - t0:.0f}s)', flush=True)
    one_token(pol_base, lambda p: E.job_key(p['pid'], p['seed']), 'base_tok')
    for name, pol in specs:
        one_token(pol, lambda p: E.job_key(p['pid'], p['seed']), f'{name}_tok')
    one_token(pol_base, lambda p: E.job_key(p['pid'], p['seed'], 'alt'), 'alt_tok')
    with open(os.path.join(od, 'positions.jsonl'), 'w') as f:
        for p in pos:
            f.write(json.dumps(p) + '\n')
    for c in CATS:
        ps = [p for p in pos if p['cat'] == c]
        if ps:
            print(f"{c:11s} n={len(ps):4d} base==orig {np.mean([p['base_tok'] == p['orig'] for p in ps]):.3f} " +
                  ' '.join(f"{nm}!=orig {np.mean([p[nm + '_tok'] != p['orig'] for p in ps]):.3f}"
                           for nm in [n for n, _ in specs] + ['alt']), flush=True)

    # step 2: continuations for positions where loop or alt changed the token
    jobs, meta = [], {}
    for i, p in enumerate(pos):
        toks = {p['orig']} | {p[nm + '_tok'] for nm in [n for n, _ in specs] + ['alt']}
        if len(toks) == 1:
            continue
        r = by_src[p['src']]
        for tk in sorted(toks):
            name = 'orig' if tk == p['orig'] else str(tk)
            for m in range(a.M):
                jid = f'{i}|{name}|{m}'
                jobs.append(dict(jid=jid, key=E.job_key(p['pid'], p['seed'], p['t'], m),
                                 prompt=prompts[p['src']] + r['tokens'][:p['t']] + [tk],
                                 max_new=min(2048, a.max_total - p['t'] - 1)))
                meta[jid] = (i, name, m, tk)
    print(f'step2: {len(jobs)} continuation jobs', flush=True)
    fh = open(os.path.join(od, 'conts.jsonl'), 'w')

    def on_done(jid, b):
        i, name, m, tk = meta[jid]
        p = pos[i]
        r = by_src[p['src']]
        text = eng.tok.decode(r['tokens'][:p['t']] + [tk] + [int(x) for x in b['tokens']], skip_special_tokens=True)
        it = pool[p['pid']]
        ok = bool(grader.grade(text, it['answer'], it['answer_type']))
        fh.write(json.dumps(dict(pos=i, cand=name, tok=tk, m=m, ok=ok, n=len(b['tokens']))) + '\n')
        fh.flush()
    eng.generate(jobs, pol_base, max_new=2048, log_every=500, log=lambda s: print(s, flush=True), on_done=on_done)
    fh.close()
    print(f'DONE in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
