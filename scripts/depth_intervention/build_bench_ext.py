"""O-enrichment extension of the benchmark (computational sources only): GSM-MC train, the MATH-MC test rows and
PythonIO test rows not used in the core set. Same prompt format/demos as the core; separate qid space and folds.
Writes data/bench/questions_ext.jsonl + build_meta_ext.json. Select it with env BREW_BENCH=ext (see common.py).
Usage: python -I scripts/build_bench_ext.py"""
import collections, hashlib, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_bench import RAW, OUT, SEED, mc_rows, valid

def main():
    core = [json.loads(l) for l in open(os.path.join(OUT, 'questions.jsonl'))]
    demos = json.load(open(os.path.join(OUT, 'demos.json')))
    used = {(q['source'], q['question'], tuple(q['options'])) for q in core}
    used |= {(s, d['question'], tuple(d['options'])) for s, ds in demos.items() for d in ds}
    qs, meta = [], {'seed': SEED, 'sources': {}}
    for source, sub, split, skey, gkey in [('gsm_mc', 'gsm8k-mc', 'train', None, None),
                                           ('math_mc', 'math-mc', 'test', 'Type', None),
                                           ('pythonio', 'pythonio-mc', 'test', None, 'Task_id')]:
        rows = [q for q in mc_rows(os.path.join(RAW, sub, f'{split}.jsonl'), source, skey, gkey) if valid(q)]
        new = [q for q in rows if (source, q['question'], tuple(q['options'])) not in used]
        # PythonIO: drop Task_ids that appear in the core set so folds/groups never straddle core and ext
        if source == 'pythonio':
            core_tasks = {q['group'].split(':', 1)[1] for q in core if q['source'] == 'pythonio'}
            new = [q for q in new if q['group_raw'] not in core_tasks]
        meta['sources'][source] = dict(split=split, n_valid=len(rows), n_used=len(new))
        qs += new
    for i, q in enumerate(qs):
        q['qid'] = i
        q['group'] = f"{q['source']}:{q['group_raw']}" if q['group_raw'] is not None else f"{q['source']}:ext{i}"
    fold_of = {}
    for source in sorted({q['source'] for q in qs}):
        groups = sorted({q['group'] for q in qs if q['source'] == source})
        r2 = random.Random(f'{SEED}:extfold:{source}'); r2.shuffle(groups)
        for j, g in enumerate(groups):
            fold_of[g] = j % 5
    with open(os.path.join(OUT, 'questions_ext.jsonl'), 'w') as f:
        for q in qs:
            f.write(json.dumps(dict(qid=q['qid'], source=q['source'], group=q['group'], fold=fold_of[q['group']],
                                    subject=q['subject'], question=q['question'], options=q['options'],
                                    answer=q['answer'], raw_row=q['raw_row']), ensure_ascii=False) + '\n')
    meta['n_questions'] = len(qs)
    meta['by_source'] = dict(collections.Counter(q['source'] for q in qs))
    meta['fold_dist'] = dict(collections.Counter(fold_of[q['group']] for q in qs))
    meta['script_md5'] = hashlib.md5(open(__file__, 'rb').read()).hexdigest()
    json.dump(meta, open(os.path.join(OUT, 'build_meta_ext.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))

if __name__ == '__main__':
    main()
