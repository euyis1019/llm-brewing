"""Check scripts/prompts.py against the real tokenizers and write prompt token-length statistics.

Usage: python -I check_prompts.py --pool GD/data/pool.jsonl --out GD/data/prompt_stats.json
           [--prompts GD/scripts/prompts.py] [--qwen3 /nvme0/if/models/Qwen3-4B]
           [--qwen25 /nvme0/if/models/huggingface/Qwen/Qwen2.5-7B-Instruct]

Checks per family, over every pool item:
  - prompt_ids(item) == tok(prompt_text(item), add_special_tokens=False)['input_ids']
  - qwen3: the rendered prompt ends with '<think>\\n\\n</think>\\n\\n' (non-thinking mode)
Prints one rendered prompt per family and token-length stats per source. No GPU is used.
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import argparse
import collections
import importlib.util
import json
import os
import statistics
import sys
import time

GD = str(_experiment_root('gen_depth'))
sys.path.insert(0, os.path.join(GD, 'pylib'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

EXAMPLE_PID = 'gsm8k:1'
EMPTY_THINK = '<think>\n\n</think>\n\n'


def load_module(path):
    spec = importlib.util.spec_from_file_location('prompts', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarize(xs):
    return {'n': len(xs), 'min': min(xs), 'mean': round(statistics.mean(xs), 1), 'median': pct(xs, 0.5),
            'p90': pct(xs, 0.9), 'p99': pct(xs, 0.99), 'max': max(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prompts', default=os.path.join(GD, 'scripts', 'prompts.py'))
    ap.add_argument('--qwen3', default=str(_model_root() / 'Qwen3-4B'))
    ap.add_argument('--qwen25', default=str(_model_root() / 'huggingface/Qwen/Qwen2.5-7B-Instruct'))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import transformers

    P = load_module(args.prompts)
    pool = [json.loads(line) for line in open(args.pool)]
    by_pid = {it['pid']: it for it in pool}
    sources = list(dict.fromkeys(it['source'] for it in pool))
    report = {'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'script': 'scripts/data/check_prompts.py',
              'transformers': transformers.__version__, 'n_items': len(pool),
              'suffix': P.SUFFIX, 'families': {}}
    ok = True
    for family, path in (('qwen3', args.qwen3), ('qwen2.5', args.qwen25)):
        tok = AutoTokenizer.from_pretrained(path)
        raw_type = type(tok.apply_chat_template([{'role': 'user', 'content': 'x'}], add_generation_prompt=True,
                                                tokenize=True)).__name__
        lens = collections.defaultdict(list)
        mismatch, bad_end = [], []
        t0 = time.time()
        for it in pool:
            ids = P.prompt_ids(it, tok, family)
            text = P.prompt_text(it, tok, family)
            assert isinstance(ids, list) and all(isinstance(t, int) for t in ids)
            if ids != tok(text, add_special_tokens=False)['input_ids']:
                mismatch.append(it['pid'])
            if family == 'qwen3' and not text.endswith(EMPTY_THINK):
                bad_end.append(it['pid'])
            lens[it['source']].append(len(ids))
            lens['ALL'].append(len(ids))
        ex = P.prompt_text(by_pid[EXAMPLE_PID], tok, family)
        ex_ids = P.prompt_ids(by_pid[EXAMPLE_PID], tok, family)
        print(f'===== {family} ({path}); apply_chat_template(tokenize=True) returns {raw_type}; '
              f'{time.time() - t0:.1f}s')
        print(f'----- rendered prompt for {EXAMPLE_PID} ({len(ex_ids)} tokens):')
        print(ex + '<<END>>')
        if family == 'qwen3':
            print(f'ends with {EMPTY_THINK!r}: {ex.endswith(EMPTY_THINK)}; items not ending so: {len(bad_end)}')
        print(f'prompt_ids == tok(prompt_text, add_special_tokens=False): mismatches {len(mismatch)}/{len(pool)}')
        print(f'{"source":10s} {"n":>5s} {"min":>5s} {"mean":>7s} {"median":>6s} {"p90":>5s} {"p99":>5s} {"max":>5s}')
        stats = {}
        for s in sources + ['ALL']:
            st = summarize(lens[s])
            stats[s] = st
            print(f'{s:10s} {st["n"]:5d} {st["min"]:5d} {st["mean"]:7.1f} {st["median"]:6d} {st["p90"]:5d} '
                  f'{st["p99"]:5d} {st["max"]:5d}')
        ok &= not mismatch and not bad_end
        report['families'][family] = {
            'tokenizer': path, 'chat_template_kwargs': P._chat_kwargs(family),
            'apply_chat_template_tokenize_returns': raw_type,
            'example_pid': EXAMPLE_PID, 'example_prompt_text': ex, 'example_prompt_ids_len': len(ex_ids),
            'ids_equal_tokenized_text_mismatches': mismatch,
            'qwen3_prompts_not_ending_with_empty_think': bad_end if family == 'qwen3' else None,
            'token_length_by_source': stats}
    with open(args.out + '.tmp', 'w') as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    os.replace(args.out + '.tmp', args.out)
    print('OK' if ok else 'NOT OK')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
