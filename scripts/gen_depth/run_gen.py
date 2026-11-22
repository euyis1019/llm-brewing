"""Generate with a controller policy over a problem selection; stream one record per (pid, seed) to jsonl.

Usage:
  python -I scripts/run_gen.py --model qwen3-4b --select source=math500 --seeds 0,1 --policy base --tag base
        [--pool data/pool.jsonl] [--pids FILE] [--shard 0/2] [--R 48] [--max_new 3072]
Output: results/<model>/<tag>/gen[.shardI].jsonl, one line per job:
  pid, seed, jid, n, stop ('eos'|'len'), text, and per-token arrays tokens, exit, loop, nread, commit, flips, chg,
  ent_m / p1_m (entropy and top-1 prob of the final-layer distribution, x1000, int).
Resumes by skipping jids already in the output file.
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
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(GD, 'pylib'))

import torch  # noqa: E402

import engine as E  # noqa: E402
import policies as P  # noqa: E402

SUFFIX = {'math': '\nPlease reason step by step, and put your final answer within \\boxed{}.',
          'mcq': '\nPlease reason step by step, and put the letter of your final answer (A, B, C, or D) within \\boxed{}.'}


def build_prompt(item, tok, fam):
    """Chat prompt (Qwen3: non-thinking). Kept self-contained so every run of a model uses identical prompts."""
    kw = dict(add_generation_prompt=True, tokenize=False)
    if fam == 'qwen3':
        kw['enable_thinking'] = False
    text = tok.apply_chat_template([{'role': 'user', 'content': item['problem'] + SUFFIX[item['answer_type']]}], **kw)
    return tok(text, add_special_tokens=False)['input_ids']


def select(pool, sel, pids_file):
    if pids_file:
        keep = [l.strip() for l in open(pids_file) if l.strip()]
        by = {it['pid']: it for it in pool}
        return [by[p] for p in keep]
    if sel in ('', 'all'):
        return pool
    k, v = sel.split('=', 1)
    vals = set(v.split('+'))
    return [it for it in pool if str(it.get(k)) in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--pool', default=os.path.join(GD, 'data/pool.jsonl'))
    ap.add_argument('--select', default='all')
    ap.add_argument('--pids', default='')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--policy', default='base')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--R', type=int, default=48)
    ap.add_argument('--Tmax', type=int, default=4096)
    ap.add_argument('--max_new', type=int, default=3072)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--salt', type=int, default=0, help='fresh sampling noise for the same (pid, seed) labels (A/A runs)')
    a = ap.parse_args()
    t0 = time.time()
    pool = [json.loads(l) for l in open(a.pool)]
    items = select(pool, a.select, a.pids)
    if a.limit:
        items = items[:a.limit]
    si, sn = map(int, a.shard.split('/'))
    seeds = [int(s) for s in a.seeds.split(',')]
    od = os.path.join(GD, 'results', a.model, a.tag)
    os.makedirs(od, exist_ok=True)
    of = os.path.join(od, 'gen.jsonl' if sn == 1 else f'gen.shard{si}.jsonl')
    done = set()
    if os.path.exists(of):
        for l in open(of):
            try:
                done.add(json.loads(l)['jid'])
            except json.JSONDecodeError:
                pass
    eng = E.Engine(a.model, R=a.R, Tmax=a.Tmax)
    pol = P.make_policy(a.policy, eng)
    jobs, meta = [], {}
    for n, it in enumerate(items):
        for s in seeds:
            jid = f"{it['pid']}|{s}"
            if (n * len(seeds) + seeds.index(s)) % sn != si or jid in done:
                continue
            p = build_prompt(it, eng.tok, eng.fam)
            if len(p) + a.max_new + 1 > a.Tmax:
                print('skip (prompt too long)', jid, len(p), flush=True)
                continue
            key = E.job_key(it['pid'], s) if a.salt == 0 else E.job_key(it['pid'], s, 'salt', a.salt)
            jobs.append(dict(jid=jid, key=key, prompt=p))
            meta[jid] = (it['pid'], s, len(p))
    # long prompts first so slots stay busy at the tail
    jobs.sort(key=lambda j: -len(j['prompt']))
    with open(os.path.join(od, 'config.json'), 'w') as f:
        json.dump(dict(vars(a), policy_name=pol.name, feat_layers=list(pol.feat_layers), L=eng.L,
                       c_layer=eng.c_layer, c_head=eng.c_head, c_kv=eng.c_kv, sampling=E.SAMPLING[eng.fam]), f, indent=1)
    print(f'{a.model} policy={a.policy} jobs={len(jobs)} (done before {len(done)}) -> {of}', flush=True)
    fh = open(of, 'a')

    def on_done(jid, b):
        pid, s, plen = meta[jid]
        toks = [int(t) for t in b['tokens']]
        stop = 'eos' if toks[-1] in E.EOS else 'len'
        rec = dict(pid=pid, seed=s, jid=jid, plen=plen, n=len(toks), stop=stop,
                   text=eng.tok.decode(toks, skip_special_tokens=True), tokens=toks,
                   ent_m=[int(round(x * 1000)) for x in b['ent']], p1_m=[int(round(x * 1000)) for x in b['p1']])
        for k in ('exit', 'loop', 'nread', 'commit', 'flips', 'chg', 'skip'):
            if k in b:
                rec[k] = [int(x) for x in b[k]]
        fh.write(json.dumps(rec) + '\n')
        fh.flush()

    with torch.no_grad():
        eng.generate(jobs, pol, max_new=a.max_new, log_every=500, log=lambda m: print(m, E.time.strftime('%H:%M:%S'), flush=True),
                     on_done=on_done)
    fh.close()
    print(f'DONE {len(jobs)} jobs in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
