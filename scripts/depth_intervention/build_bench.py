"""Build the 4-choice single-token benchmark (canonical option order + few-shot demo pool).

Writes data/bench/questions.jsonl, data/bench/demos.json, data/bench/build_meta.json.
Versions are not materialized here: version v shows option (i+v)%4 at letter i (see common.py).
Raw inputs are downloaded/untrusted: run with `python -I`.
Usage: python -I scripts/build_bench.py
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import hashlib, json, os, random, re, sys, collections

ROOT = str(_experiment_root('depth_intervention'))
RAW = os.path.join(ROOT, 'data/raw/mcev_data/data')
NORM = str(_normalized_benchmark_root())
OUT = os.path.join(ROOT, 'data/bench')
SEED = 20261016
N_DEMO = 3


def clean(s):
    s = str(s).replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r'[ \t]+\n', '\n', s)
    return s.strip()


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def mc_rows(path, source, subject_key=None, group_key=None):
    out = []
    for i, r in enumerate(read_jsonl(path)):
        opts = [clean(r[k]) for k in 'ABCD']
        out.append(dict(source=source, question=clean(r['Question']), options=opts, answer='ABCD'.index(r['Answer'].strip()),
                        subject=str(r.get(subject_key, '')) if subject_key else '',
                        group_raw=str(r[group_key]) if group_key else None, raw_row=i))
    return out


def valid(q):
    o = q['options']
    return len(o) == 4 and all(o) and len(set(o)) == 4 and q['question'] and 0 <= q['answer'] < 4


def main():
    rng = random.Random(SEED)
    qs, demos, meta = [], {}, {'seed': SEED, 'sources': {}}

    def pick_demos(pool, source):
        # short, valid, distinct canonical answer letters (so no cyclic shift makes them all the same letter)
        pool = [p for p in pool if valid(p)]
        lens = sorted(len(p['question']) + sum(map(len, p['options'])) for p in pool)
        cap = lens[len(lens) // 2]
        cand = [p for p in pool if len(p['question']) + sum(map(len, p['options'])) <= cap]
        rng.shuffle(cand)
        chosen, used = [], set()
        for p in cand:
            if p['answer'] not in used:
                chosen.append(p); used.add(p['answer'])
            if len(chosen) == N_DEMO:
                break
        assert len(chosen) == N_DEMO, source
        return [dict(question=p['question'], options=p['options'], answer=p['answer'], subject=p['subject']) for p in chosen]

    # --- MC-Evaluation: GSM-MC (all test), MATH-MC (1000), PythonIO (1000) ---
    for source, sub, n, skey, gkey in [('gsm_mc', 'gsm8k-mc', None, None, None),
                                        ('math_mc', 'math-mc', 1000, 'Type', None),
                                        ('pythonio', 'pythonio-mc', 1000, None, 'Task_id')]:
        test = [q for q in mc_rows(os.path.join(RAW, sub, 'test.jsonl'), source, skey, gkey) if valid(q)]
        train = mc_rows(os.path.join(RAW, sub, 'train.jsonl'), source, skey, gkey)
        n_valid = len(test)
        if n is not None and len(test) > n:
            test = rng.sample(test, n)
        demos[source] = pick_demos(train, source)
        qs += test
        meta['sources'][source] = dict(n_test_valid=n_valid, n_used=len(test), demo_from='train.jsonl')

    # --- MMLU-Pro: 1000 rows, correct + 3 seeded distractors ---
    rows = read_jsonl(os.path.join(NORM, 'mmlu_pro/mmlu_pro.jsonl'))
    def to4(r):
        ch = [clean(c) for c in r['choices']]
        ai = 'ABCDEFGHIJ'.index(r['target'].strip())
        correct = ch[ai]
        distract = [c for j, c in enumerate(ch) if j != ai and c and c != correct and c.upper() != 'N/A']
        distract = list(dict.fromkeys(distract))
        if len(distract) < 3 or not correct:
            return None
        r2 = random.Random(f"{SEED}:{r['sample_id']}")
        opts = r2.sample(distract, 3) + [correct]
        r2.shuffle(opts)
        return dict(source='mmlu_pro', question=clean(r['question']), options=opts, answer=opts.index(correct),
                    subject=r['metadata'].get('category', ''), group_raw=None, raw_row=r['sample_id'])
    conv = [q for q in (to4(r) for r in rows) if q and valid(q)]
    idx = list(range(len(conv))); rng.shuffle(idx)
    test = [conv[i] for i in idx[:1000]]
    demos['mmlu_pro'] = pick_demos([conv[i] for i in idx[1000:]], 'mmlu_pro')
    qs += test
    meta['sources']['mmlu_pro'] = dict(n_rows=len(rows), n_valid_4opt=len(conv), n_used=len(test), demo_from='other rows')

    # --- C-Eval: all val rows except 3 held out as demos ---
    rows = read_jsonl(os.path.join(NORM, 'ceval/ceval.jsonl'))
    conv = []
    for r in rows:
        opts = [clean(c) for c in r['choices']]
        if len(opts) != 4:
            continue
        conv.append(dict(source='ceval', question=clean(r['question']), options=opts, answer='ABCD'.index(r['target'].strip()),
                         subject=r['metadata'].get('subject', ''), group_raw=None, raw_row=r['sample_id']))
    conv = [q for q in conv if valid(q)]
    d = pick_demos(conv, 'ceval')
    dkeys = {(x['question'], tuple(x['options'])) for x in d}
    test = [q for q in conv if (q['question'], tuple(q['options'])) not in dkeys]
    demos['ceval'] = d
    qs += test
    meta['sources']['ceval'] = dict(n_rows=len(rows), n_valid=len(conv), n_used=len(test), demo_from='3 held-out val rows')

    # --- ids, groups, folds (5 folds by group, stratified by source) ---
    for i, q in enumerate(qs):
        q['qid'] = i
        q['group'] = f"{q['source']}:{q['group_raw']}" if q['group_raw'] is not None else f"{q['source']}:{i}"
    fold_of = {}
    for source in sorted({q['source'] for q in qs}):
        groups = sorted({q['group'] for q in qs if q['source'] == source})
        r2 = random.Random(f'{SEED}:fold:{source}'); r2.shuffle(groups)
        for j, g in enumerate(groups):
            fold_of[g] = j % 5
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'questions.jsonl'), 'w') as f:
        for q in qs:
            rec = dict(qid=q['qid'], source=q['source'], group=q['group'], fold=fold_of[q['group']], subject=q['subject'],
                       question=q['question'], options=q['options'], answer=q['answer'], raw_row=q['raw_row'])
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    json.dump(demos, open(os.path.join(OUT, 'demos.json'), 'w'), ensure_ascii=False, indent=1)
    meta['n_questions'] = len(qs)
    meta['by_source'] = dict(collections.Counter(q['source'] for q in qs))
    meta['answer_dist'] = dict(collections.Counter(q['answer'] for q in qs))
    meta['fold_dist'] = dict(collections.Counter(fold_of[q['group']] for q in qs))
    meta['script_md5'] = hashlib.md5(open(__file__, 'rb').read()).hexdigest()
    json.dump(meta, open(os.path.join(OUT, 'build_meta.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
