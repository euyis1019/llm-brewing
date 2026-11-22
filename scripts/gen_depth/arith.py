"""Does extra depth fix computed tokens? Direct-answer synthetic arithmetic under greedy decoding.

Problems (fixed seed): sum of three 7-digit numbers, and 4-digit x 3-digit products. The model answers with the number
only (chat template, Qwen3 non-thinking), so every generated token is a computed digit. Greedy decoding makes base and
intervention runs deterministic per problem (paired, no resampling noise). Arms are policy specs applied to every
generated token (the answer); accuracy = exact match of the digits.

Usage: python -I scripts/arith.py --model qwen3-4b --n 300 --arms 'base;w18:loop:win=18-25,it=1;w24:loop:win=24-29,it=1'
Writes results/<model>/arith/<name>.json and prints accuracy, fixed/broken vs base, McNemar p.
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import argparse
import json

import numpy as np
import os
import random
import re
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))
sys.path.insert(0, HERE)

import engine as E  # noqa: E402
import policies as P  # noqa: E402


def problems(n, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if i % 2 == 0:
            a, b, c = (rng.randint(10 ** 6, 10 ** 7 - 1) for _ in range(3))
            out.append(dict(id=f'add3x7:{i}', q=f'What is {a} + {b} + {c}?', ans=str(a + b + c)))
        else:
            a, b = rng.randint(1000, 9999), rng.randint(100, 999)
            out.append(dict(id=f'mul4x3:{i}', q=f'What is {a} * {b}?', ans=str(a * b)))
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--arms', required=True)
    ap.add_argument('--name', default='arith_v1')
    a = ap.parse_args()
    eng = E.Engine(a.model, R=48, Tmax=512)
    eng.temperature = 0.0
    probs = problems(a.n)
    kw = dict(add_generation_prompt=True, tokenize=False)
    if eng.fam == 'qwen3':
        kw['enable_thinking'] = False
    jobs = []
    for p in probs:
        text = eng.tok.apply_chat_template([{'role': 'user', 'content': p['q'] + ' Answer with only the final number, no other text.'}], **kw)
        jobs.append(dict(jid=p['id'], key=E.job_key(p['id']), prompt=eng.tok(text, add_special_tokens=False)['input_ids']))
    res, extra = {}, {}
    for arm in [x for x in a.arms.split(';') if x]:
        name, _, spec = arm.partition(':')
        spec = spec or 'base'
        out = eng.generate(jobs, P.make_policy(spec + ('' if 'feat=' in spec else (',' if ':' in spec else ':') + 'feat=0'), eng),
                           max_new=24)
        ok, texts, wf, dig = {}, {}, {}, {}
        for p in probs:
            txt = eng.tok.decode(out[p['id']]['tokens'], skip_special_tokens=True)
            texts[p['id']] = txt
            digits = re.sub(r'[^0-9]', '', txt.split('\n')[0])
            ok[p['id']] = digits == p['ans']
            # well-formed: the whole output is one number (optional commas/spaces/period), nothing else
            wf[p['id']] = bool(re.fullmatch(r'\s*[0-9][0-9,]*\.?\s*', txt))
            d = re.sub(r'[^0-9]', '', txt) if wf[p['id']] else ''
            dig[p['id']] = sum(x == y for x, y in zip(d, p['ans'])) / len(p['ans']) if d else 0.0
        res[name] = ok
        extra[name] = dict(texts=texts, wellformed=wf, digit_acc=dig)
        nwf = sum(wf.values())
        print(f'   {name:10s} well-formed {nwf / len(wf):.3f}; acc among well-formed '
              f'{sum(ok[k] for k in wf if wf[k]) / max(1, nwf):.4f}; mean digit accuracy {np.mean(list(dig.values())):.4f}', flush=True)
        acc = sum(ok.values()) / len(ok)
        line = f'{name:10s} acc {acc:.4f}'
        if 'base' in res and name != 'base':
            b = sum(1 for k in ok if ok[k] and not res['base'][k])
            c = sum(1 for k in ok if not ok[k] and res['base'][k])
            line += f'  vs base {(acc - sum(res["base"].values()) / len(ok)) * 100:+.2f}pp fixed/broken {b}/{c} McNemar p={mcnemar(b, c):.4f}'
            for kind in ('add3x7', 'mul4x3'):
                ks = [k for k in ok if k.startswith(kind)]
                line += f' | {kind} {sum(ok[k] for k in ks) / len(ks):.3f} (base {sum(res["base"][k] for k in ks) / len(ks):.3f})'
        print(line, flush=True)
    od = os.path.join(GD, 'results', a.model, 'arith')
    os.makedirs(od, exist_ok=True)
    json.dump(res, open(os.path.join(od, f'{a.name}.json'), 'w'))
    json.dump(extra, open(os.path.join(od, f'{a.name}_texts.json'), 'w'))


if __name__ == '__main__':
    main()
