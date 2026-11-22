"""Sequence-level early stop from per-token signals: detect degenerate (runaway / repeating) generations early.

Online detectors over a sliding window of the last W generated tokens (causal), fired at the first step where the
statistic crosses a threshold (after a minimum of M tokens):
  rep     share of the window's tokens that complete a 16-gram already seen earlier in the response (text baseline)
  lowent  mean final-layer entropy of the window (copy loops are near-deterministic)
  commit  mean candidate-restricted commitment layer of the window (depth signal recorded by the engine)
  both    rep and commit/lowent combined (AND)
Thresholds are chosen on fit problems (pid hash parity) for a target false-stop rate on correct runs and evaluated on
the other half: share of tokens saved, share of truncated runs caught, and accuracy lost (correct runs stopped before
their final answer).

Usage: python -I scripts/degen_stop.py --model qwen3-4b --tags base,base_rest
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import argparse
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))


def rep_flags(tok, n=16):
    """flags[i] = 1 if the n-gram ending at i appeared earlier in the sequence."""
    seen = set()
    out = np.zeros(len(tok), np.float32)
    for i in range(n - 1, len(tok)):
        g = tuple(tok[i - n + 1:i + 1])
        if g in seen:
            out[i] = 1
        seen.add(g)
    return out


def first_fire(stat, thr, M, above=True):
    idx = np.arange(len(stat))
    hit = (stat >= thr) if above else (stat <= thr)
    hit &= idx >= M
    w = np.flatnonzero(hit)
    return int(w[0]) if len(w) else -1


def window_mean(x, W):
    c = np.cumsum(np.concatenate([[0], x]))
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(0, i - W)
    return (c[i] - c[lo]) / (i - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--tags', default='base,base_rest')
    ap.add_argument('--W', type=int, default=256)
    ap.add_argument('--M', type=int, default=512)
    a = ap.parse_args()
    runs = []
    for tag in a.tags.split(','):
        rd = os.path.join(GD, 'results', a.model, tag)
        G = {json.loads(l)['jid']: json.loads(l) for l in open(os.path.join(rd, 'graded.jsonl'))}
        for fn in os.listdir(rd):
            if fn.startswith('gen') and fn.endswith('.jsonl'):
                for l in open(os.path.join(rd, fn)):
                    r = json.loads(l)
                    g = G[r['jid']]
                    tok = r['tokens']
                    # position where the final answer is written: last occurrence of 'boxed' in text -> approx by tokens
                    ans_pos = len(tok)
                    txt_end = r['text'].rfind('\\boxed')
                    if txt_end >= 0:
                        ans_pos = int(len(tok) * (txt_end / max(1, len(r['text']))))
                    runs.append(dict(pid=r['pid'], ok=g['correct'], trunc=r['stop'] == 'len', n=len(tok), ans=ans_pos,
                                     rep=window_mean(rep_flags(tok), a.W),
                                     ent=window_mean(np.array(r['ent_m'], np.float32) / 1000, a.W),
                                     com=window_mean(np.array(r['commit'], np.float32), a.W)))
    fit = np.array([int(hashlib.md5(r['pid'].encode()).hexdigest(), 16) % 2 == 0 for r in runs])
    N = len(runs)
    tot = sum(r['n'] for r in runs)
    print(f'runs {N}, tokens {tot}, truncated {np.mean([r["trunc"] for r in runs]):.3f}, '
          f'acc {np.mean([r["ok"] for r in runs]):.4f}, token share of truncated runs '
          f'{sum(r["n"] for r in runs if r["trunc"]) / tot:.3f}')
    dets = {
        'rep': (lambda r: r['rep'], True, np.linspace(0.2, 0.95, 16)),
        'lowent': (lambda r: r['ent'], False, np.linspace(0.005, 0.08, 16)),
        'commit_low': (lambda r: r['com'], False, np.linspace(20, 28, 17)),
        'commit_high': (lambda r: r['com'], True, np.linspace(28, 33, 11)),
    }
    for name, (f, above, grid) in dets.items():
        best = None
        for thr in grid:
            fires = [first_fire(f(r), thr, a.M, above) for r in runs]
            # false stop: a correct run stopped before its answer
            fs = [(fi >= 0 and fi < r['ans']) for fi, r in zip(fires, runs)]
            fs_rate_fit = np.mean([x for x, r, ff in zip(fs, runs, fit) if r['ok'] and ff])
            if fs_rate_fit <= 0.005 and (best is None or thr != best[0]):
                cand = (thr, fires)
                # prefer the most aggressive threshold that keeps false stops <= 0.5% on fit
                if best is None or (above and thr < best[0]) or (not above and thr > best[0]):
                    best = cand
        if best is None:
            print(f'{name}: no threshold with <=0.5% false stops on fit half')
            continue
        thr, fires = best
        ev = ~fit
        saved = sum((r['n'] - fi) for fi, r, e in zip(fires, runs, ev) if e and fi >= 0)
        tot_ev = sum(r['n'] for r, e in zip(runs, ev) if e)
        caught = np.mean([fi >= 0 for fi, r, e in zip(fires, runs, ev) if e and r['trunc']])
        lost = sum(1 for fi, r, e in zip(fires, runs, ev) if e and r['ok'] and 0 <= fi < r['ans'])
        n_ok = sum(1 for r, e in zip(runs, ev) if e and r['ok'])
        n_ev = int(ev.sum())
        print(f'{name:12s} thr {thr:.3f}: eval half tokens saved {saved / tot_ev:.3f}, truncated runs caught {caught:.3f}, '
              f'correct runs stopped early {lost}/{n_ok} -> accuracy change {-lost / n_ev * 100:+.2f}pp')


if __name__ == '__main__':
    main()
