"""Data-driven critical-token vocabularies from baseline runs (no gold labels).

A token id is 'critical' when the tokens it produces (top1crit) or the tokens that follow it (prevcrit) are often
late-committing (engine 'commit' >= --late), i.e. the final top-1 only wins in the last block(s). Ids need at least
--min_count occurrences. Only problems in the train half (pid hash parity) are used, so evaluations on the other
half are out of sample.

Writes results/<model>/<tag>/crit_{top1,prev}_<name>.npy and a readable table crit_<name>.txt.
Usage: python -I scripts/build_crit_vocab.py --model qwen3-4b --tag base --late 35 --min_count 30 --top 300
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
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--tag', default='base')
    ap.add_argument('--late', type=int, default=35)
    ap.add_argument('--min_count', type=int, default=30)
    ap.add_argument('--top', type=int, default=300)
    ap.add_argument('--name', default='v1')
    ap.add_argument('--all', action='store_true', help='use all problems (no train/test split)')
    a = ap.parse_args()
    rd = os.path.join(GD, 'results', a.model, a.tag)
    cnt_t, late_t = Counter(), Counter()
    cnt_p, late_p = Counter(), Counter()
    ent_t = defaultdict(float)
    n_tok = n_late = 0
    for fn in sorted(os.listdir(rd)):
        if not (fn.startswith('gen') and fn.endswith('.jsonl')):
            continue
        for l in open(os.path.join(rd, fn)):
            r = json.loads(l)
            if not a.all and int(hashlib.md5(r['pid'].encode()).hexdigest(), 16) % 2 == 1:
                continue
            toks, com, ent = r['tokens'], r['commit'], r['ent_m']
            for i, t in enumerate(toks):
                lt = com[i] >= a.late
                cnt_t[t] += 1
                late_t[t] += lt
                ent_t[t] += ent[i] / 1000
                n_tok += 1
                n_late += lt
                if i > 0:
                    cnt_p[toks[i - 1]] += 1
                    late_p[toks[i - 1]] += lt
    base = n_late / n_tok
    from transformers import AutoTokenizer
    import engine as E
    tok = AutoTokenizer.from_pretrained(E.MODELS[a.model][0])

    def pick(cnt, late):
        cand = [(late[t] / c, c, t) for t, c in cnt.items() if c >= a.min_count]
        cand.sort(reverse=True)
        return cand[:a.top]
    top1 = pick(cnt_t, late_t)
    prev = pick(cnt_p, late_p)
    np.save(os.path.join(rd, f'crit_top1_{a.name}.npy'), np.array([t for _, _, t in top1], dtype=np.int64))
    np.save(os.path.join(rd, f'crit_prev_{a.name}.npy'), np.array([t for _, _, t in prev], dtype=np.int64))
    cov_t = sum(cnt_t[t] for _, _, t in top1) / n_tok
    cov_p = sum(cnt_p[t] for _, _, t in prev) / n_tok
    with open(os.path.join(rd, f'crit_{a.name}.txt'), 'w') as f:
        f.write(f'tokens {n_tok} late rate {base:.4f}; top1 vocab covers {cov_t:.3f} of tokens, prev vocab {cov_p:.3f}\n')
        f.write('\n# produced token (top1crit): late_rate count mean_ent repr\n')
        for r_, c, t in top1:
            f.write(f'{r_:.3f} {c:6d} {ent_t[t] / cnt_t[t]:.3f} {tok.decode([t])!r}\n')
        f.write('\n# previous token (prevcrit): late_rate_of_next count repr\n')
        for r_, c, t in prev:
            f.write(f'{r_:.3f} {c:6d} {tok.decode([t])!r}\n')
    print(f'tokens {n_tok} late rate {base:.4f}; top1 vocab ({len(top1)}) covers {cov_t:.3f}; prev vocab covers {cov_p:.3f}')


if __name__ == '__main__':
    main()
