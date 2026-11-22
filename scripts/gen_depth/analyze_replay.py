"""Inner-loop analysis of teacher-forced replays: where depth can be cut, which tokens commit late.

Reads results/<model>/<tag>/replay/part*.npz (+ gen*.jsonl, + graded.jsonl if present) and writes
results/<model>/<tag>/replay/summary.json and prints tables:
  1. oracle exit depth: first listed layer from which the layer's sampled token (same noise as the run) equals the
     final one at every later listed layer; distribution overall and per token category
  2. online exit rules evaluated offline (static, conf, stable, kl-jump): mean executed blocks, readout overhead,
     token agreement overall / digits / first 16 tokens / inside the final \\boxed{}
  3. commitment vs outcome (if graded): late-commit rate and entropy in correct vs incorrect trajectories of the
     same problem
Usage: python -I scripts/analyze_replay.py --model qwen3-4b --tag base
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))
sys.path.insert(0, HERE)

MARKERS = {'wait', 'but', 'so', 'therefore', 'thus', 'however', 'let', 'first', 'next', 'now', 'hence', 'since',
           'because', 'alternatively', 'check', 'verify', 'then', 'note', 'recall', 'suppose', 'assume', 'step',
           'finally', 'hmm', 'actually'}


def categorize(tok_strs):
    """Coarse token category per token string."""
    cats = []
    for s in tok_strs:
        t = s.strip()
        if t == '':
            cats.append('space')
        elif re.fullmatch(r'[0-9]+', t):
            cats.append('digit')
        elif re.fullmatch(r'[+\-*/=^<>]+|\\(times|cdot|div|pm|le|ge|neq)', t):
            cats.append('op')
        elif t.lower() in MARKERS:
            cats.append('marker')
        elif t.startswith('\\'):
            cats.append('latex')
        elif re.fullmatch(r'[A-Za-z]+', t):
            cats.append('word')
        else:
            cats.append('punct/other')
    return np.array(cats)


def suffix_all(eq):
    """eq (T, nl) bool -> (T,) index of the first column from which eq holds through the last column."""
    rev = np.logical_and.accumulate(eq[:, ::-1], axis=1)[:, ::-1]
    first = np.argmax(rev, axis=1)
    first[~rev[:, -1]] = eq.shape[1] - 1
    return first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--tag', default='base')
    ap.add_argument('--c_ratio', type=float, default=0.0, help='readout cost / block cost (default from config)')
    a = ap.parse_args()
    rd = os.path.join(GD, 'results', a.model, a.tag)
    cfg = json.load(open(os.path.join(rd, 'config.json')))
    L = cfg['L']
    c_ratio = a.c_ratio or cfg['c_head'] / cfg['c_layer']
    parts = sorted(f for f in os.listdir(os.path.join(rd, 'replay')) if f.startswith('part') and f.endswith('.npz') and '_h' not in f)
    D = defaultdict(list)
    offs, jids = [0], []
    for f in parts:
        z = np.load(os.path.join(rd, 'replay', f))
        layers = z['layers'].tolist()
        for k in ('top1', 'samp', 'p1', 'ent', 'lpf', 'kl', 'klp', 'cand'):
            D[k].append(z[k])
        o = z['offsets']
        offs += (o[1:] + offs[-1]).tolist()
        jids += z['jids'].tolist()
    D = {k: np.concatenate(v) for k, v in D.items()}
    offs = np.array(offs)
    T, nl = D['samp'].shape
    lay = np.array(layers)
    recs = {}
    for fn in os.listdir(rd):
        if fn.startswith('gen') and fn.endswith('.jsonl'):
            for l in open(os.path.join(rd, fn)):
                r = json.loads(l)
                recs[r['jid']] = r
    graded = {}
    gp = os.path.join(rd, 'graded.jsonl')
    if os.path.exists(gp):
        for l in open(gp):
            g = json.loads(l)
            graded[g['jid']] = g['correct']
    # per-token context
    from transformers import AutoTokenizer
    import engine as E
    tok = AutoTokenizer.from_pretrained(E.MODELS[a.model][0])
    toks = np.concatenate([np.array(recs[j]['tokens'][:offs[i + 1] - offs[i]]) for i, j in enumerate(jids)])
    vocab_str = {}
    for t in np.unique(toks):
        vocab_str[int(t)] = tok.decode([int(t)])
    cat = categorize([vocab_str[int(t)] for t in toks])
    pos = np.concatenate([np.arange(offs[i + 1] - offs[i]) for i in range(len(jids))])
    job_of = np.concatenate([np.full(offs[i + 1] - offs[i], i) for i in range(len(jids))])
    # inside the last \boxed{...}: mark tokens after the last occurrence of 'boxed'
    inbox = np.zeros(T, bool)
    for i, j in enumerate(jids):
        s, e = offs[i], offs[i + 1]
        strs = [vocab_str[int(t)] for t in toks[s:e]]
        idx = [k for k, x in enumerate(strs) if 'boxed' in x]
        if idx:
            inbox[s + idx[-1] + 1: min(e, s + idx[-1] + 12)] = True
    final = D['samp'][:, -1]
    e_samp = suffix_all(D['samp'] == final[:, None])
    e_top = suffix_all(D['top1'] == D['top1'][:, -1:])
    e_cand = suffix_all(D['cand'])
    blocks = lambda j: lay[j] + 1  # noqa: E731
    S = dict(model=a.model, tag=a.tag, tokens=int(T), jobs=len(jids), layers=layers, L=L, c_ratio=c_ratio)
    S['oracle'] = {
        'mean_exit_blocks_samp': float(blocks(e_samp).mean()), 'mean_exit_blocks_top1': float(blocks(e_top).mean()),
        'mean_commit_cand': float(blocks(e_cand).mean()),
        'frac_exit_by_layer': {int(lay[j]): float((e_samp <= j).mean()) for j in range(nl)},
        'late_frac(e_samp>=L-3)': float((lay[e_samp] >= L - 3).mean()),
    }
    print(f"tokens {T} jobs {len(jids)} L {L}; oracle mean exit blocks (sampled-token stable) "
          f"{S['oracle']['mean_exit_blocks_samp']:.2f} / top1 {S['oracle']['mean_exit_blocks_top1']:.2f} / "
          f"cand-commit {S['oracle']['mean_commit_cand']:.2f}")
    print('cumulative share of tokens whose output is fixed by layer:',
          ' '.join(f'{lay[j]}:{(e_samp <= j).mean():.2f}' for j in range(0, nl, max(1, nl // 12))))
    # categories
    S['categories'] = {}
    print(f"{'cat':12s} {'share':>6s} {'exitblk':>7s} {'late':>6s} {'ent':>6s} {'p1':>5s}")
    ent_f = D['ent'][:, -1].astype(np.float32)
    p1_f = D['p1'][:, -1].astype(np.float32)
    for c in ['digit', 'op', 'marker', 'latex', 'word', 'space', 'punct/other', 'first16', 'inbox']:
        m = (pos < 16) if c == 'first16' else inbox if c == 'inbox' else cat == c
        if m.sum() == 0:
            continue
        row = dict(share=float(m.mean()), exit_blocks=float(blocks(e_samp[m]).mean()),
                   late=float((lay[e_samp[m]] >= L - 3).mean()), ent=float(ent_f[m].mean()), p1=float(p1_f[m].mean()))
        S['categories'][c] = row
        print(f"{c:12s} {row['share']:6.3f} {row['exit_blocks']:7.2f} {row['late']:6.3f} {row['ent']:6.3f} {row['p1']:5.2f}")
    # online rules offline
    rules = []
    for j in range(nl - 1):
        rules.append((f'static@{lay[j]}', np.full(T, j)))
    ck = [j for j in range(nl - 1) if lay[j] >= L // 2]
    p1 = D['p1'].astype(np.float32)
    klp = D['klp'].astype(np.float32)
    for tau in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        ex = np.full(T, nl - 1)
        for j in reversed(ck):
            ex = np.where(p1[:, j] >= tau, j, ex)
        rules.append((f'conf>={tau}', ex))
    for m_ in (2, 3, 4):
        for tau in (0.0, 0.5, 0.8):
            ex = np.full(T, nl - 1)
            for idx in reversed(range(m_ - 1, len(ck))):
                js = ck[idx - m_ + 1:idx + 1]
                same = np.all(D['top1'][:, js] == D['top1'][:, [ck[idx]]], axis=1) & (p1[:, ck[idx]] >= tau)
                ex = np.where(same, ck[idx], ex)
            rules.append((f'stable m={m_} p>={tau}', ex))
    for kap in (0.5, 1.0, 2.0):
        for tau in (0.5, 0.8):
            seen = np.zeros(T, bool)
            ex = np.full(T, nl - 1)
            done = np.zeros(T, bool)
            for j in ck:
                seen |= klp[:, j] >= kap
                hit = seen & (klp[:, j] < 0.1) & (p1[:, j] >= tau) & ~done
                ex = np.where(hit, j, ex)
                done |= hit
            rules.append((f'kljump k>={kap} p>={tau}', ex))
    S['rules'] = {}
    print(f"{'rule':28s} {'blocks':>6s} {'cost':>6s} {'agree':>6s} {'digit':>6s} {'first16':>7s} {'inbox':>6s}")
    for name, ex in rules:
        emit = D['samp'][np.arange(T), ex]
        ok = emit == final
        nread = 1 + (0 if name.startswith('static') else np.array([sum(1 for c in ck if c <= x) for x in range(nl)])[ex])
        cost = blocks(ex) + (L - 1 - lay[ex]) * (cfg['c_kv'] / cfg['c_layer']) + (nread - 1) * c_ratio
        row = dict(blocks=float(blocks(ex).mean()), cost=float(np.mean(cost)), agree=float(ok.mean()),
                   agree_digit=float(ok[cat == 'digit'].mean()), agree_first16=float(ok[pos < 16].mean()),
                   agree_inbox=float(ok[inbox].mean()) if inbox.any() else None)
        S['rules'][name] = row
        if not name.startswith('static') or lay[int(ex[0])] % 4 == 3 or lay[int(ex[0])] >= L - 4:
            print(f"{name:28s} {row['blocks']:6.2f} {row['cost']:6.2f} {row['agree']:6.4f} {row['agree_digit']:6.4f} "
                  f"{row['agree_first16']:7.4f} {row['agree_inbox'] if row['agree_inbox'] is not None else float('nan'):6.4f}")
    # commitment vs outcome
    if graded:
        late_tok = lay[e_samp] >= L - 3
        per = {}
        for i, j in enumerate(jids):
            if j not in graded:
                continue
            s, e = offs[i], offs[i + 1]
            per[j] = dict(pid=recs[j]['pid'], ok=bool(graded[j]), late=float(late_tok[s:e].mean()),
                          ent=float(ent_f[s:e].mean()), hi_ent=float((ent_f[s:e] > 1.0).mean()),
                          exitb=float(blocks(e_samp[s:e]).mean()), n=int(e - s))
        byp = defaultdict(list)
        for v in per.values():
            byp[v['pid']].append(v)
        diffs = defaultdict(list)
        for pid, vs in byp.items():
            oks = [v for v in vs if v['ok']]
            bad = [v for v in vs if not v['ok']]
            if oks and bad:
                for k in ('late', 'ent', 'hi_ent', 'exitb', 'n'):
                    diffs[k].append(np.mean([v[k] for v in bad]) - np.mean([v[k] for v in oks]))
        S['outcome'] = {k: dict(mean_wrong_minus_right=float(np.mean(v)), n_problems=len(v),
                                frac_positive=float(np.mean(np.array(v) > 0))) for k, v in diffs.items()}
        print('within-problem (wrong - right):', json.dumps(S['outcome'], indent=None))
    json.dump(S, open(os.path.join(rd, 'replay', 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
