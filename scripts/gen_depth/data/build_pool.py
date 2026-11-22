"""Build the evaluation pool (data/pool.jsonl) and its statistics (data/pool_stats.json) from the raw downloads.

Usage (raw files are untrusted data: isolated interpreter, paths as arguments):
  python -I build_pool.py --raw-root GD/data/raw --manifest GD/data/raw_manifest.json \
      --out GD/data/pool.jsonl --stats GD/data/pool_stats.json

Every raw file is checked against the sha256 recorded in the manifest before it is read.
One JSON object per problem:
  pid "<source>:<idx>"  (idx = 0-based row index in the primary raw file; olympiad pids have gaps at dropped rows)
  source, problem (text shown to the model), answer (str), answer_type ("math" | "mcq"), level (int | null),
  orig_id (id in the source), subject (math500 subject / olympiad subfield / gpqa domain / null), meta (dict).
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import argparse
import collections
import difflib
import hashlib
import json
import os
import random
import re
import sys
import time

import pandas as pd

GD = str(_experiment_root('gen_depth'))
sys.path.insert(0, os.path.join(GD, 'pylib'))

SOURCE_ORDER = ['gsm8k', 'math500', 'aime24', 'aime25', 'aime26', 'amc', 'olympiad', 'minerva', 'gpqa']
EXPECTED_CANDIDATES = {'gsm8k': 1319, 'math500': 500, 'aime24': 30, 'aime25': 30, 'aime26': 30, 'amc': 83,
                       'olympiad': 675, 'minerva': 272, 'gpqa': 198}
# raw manifest entries used by each pool source (first = primary, others = cross-check copies)
RAW_USED = {'gsm8k': ['gsm8k'], 'math500': ['math500'], 'aime24': ['aime24', 'aime24_maxwelljia'],
            'aime25': ['aime25', 'aime25_opencompass'], 'aime26': ['aime26'], 'amc': ['amc'],
            'olympiad': ['olympiad', 'olympiad_knoveleng'], 'minerva': ['minerva'],
            'gpqa': ['gpqa_jingzeshi', 'gpqa_talzoomanzoo', 'gpqa_official']}

# Manual answer fixes (decided by reading the items; every application is logged in pool_stats.json).
OLYMPIAD_ANSWER_FIX = {
    2045: ('(0,4]', 'gold "t(0,4]" has a stray variable name before the interval (t in (0,4])'),
    1782: ('n^{2}-n-1', 'gold "m_{\\max }=n^{2}-n-1" (does not parse); kept the value of m_max'),
}
MINERVA_ANSWER_FIX = {
    5: ('\\arcsin(10/13)', 'numpy syntax "np.arcsin(10/13)"'),
    38: ('-1/3', 'python float syntax "-1./3"'),
    49: ('-3/2', 'python float syntax "-3./2"'),
    72: ('x_{0} \\cos (\\omega t)+\\dot{x}_{0} \\sin (\\omega t) / \\omega', 'stray "$ $" inside the answer'),
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Raw:
    def __init__(self, root, manifest):
        self.root, self.man, self.checked = root, manifest, {}

    def path(self, name, fn):
        e = self.man[name]
        assert e['status'] == 'ok', (name, e.get('error'))
        f = e['files'][fn]
        p = os.path.join(self.root, f['local_path'])
        digest = sha256_of(p)
        assert digest == f['sha256'], f'sha256 mismatch for {p}'
        self.checked[f'{name}/{fn}'] = digest
        return p


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def item(source, idx, problem, answer, answer_type='math', level=None, orig_id=None, subject=None, meta=None):
    assert isinstance(answer, str) and answer.strip() == answer and answer, (source, idx, answer)
    assert isinstance(problem, str) and problem.strip(), (source, idx)
    return {'pid': f'{source}:{idx}', 'source': source, 'problem': problem, 'answer': answer,
            'answer_type': answer_type, 'level': level, 'orig_id': str(orig_id), 'subject': subject,
            'meta': meta or {}}


def words(s):
    return set(re.findall(r'[A-Za-z]+|\d+', s))


def jaccard(a, b):
    wa, wb = words(a), words(b)
    return len(wa & wb) / max(1, len(wa | wb))


# --------------------------------------------------------------------------------------------- sources
def build_gsm8k(raw, st):
    df = pd.read_parquet(raw.path('gsm8k', 'main/test-00000-of-00001.parquet'))
    out, n_comma = [], 0
    for idx, r in enumerate(df.to_dict('records')):
        assert r['answer'].count('####') == 1, idx
        tail = r['answer'].split('####')[-1].strip()
        n_comma += ',' in tail
        ans = tail.replace(',', '')
        assert re.fullmatch(r'-?\d+(?:\.\d+)?', ans), (idx, tail)
        out.append(item('gsm8k', idx, r['question'].strip(), ans, orig_id=f'test-{idx}'))
    st['notes']['gsm8k'] = {'answers_with_thousands_commas_stripped': n_comma}
    return out


def build_math500(raw, st):
    rows = jsonl(raw.path('math500', 'test.jsonl'))
    out = [item('math500', idx, r['problem'].strip(), r['answer'].strip(), level=int(r['level']),
                orig_id=r['unique_id'], subject=r['subject']) for idx, r in enumerate(rows)]
    st['notes']['math500'] = {'levels': dict(sorted(collections.Counter(x['level'] for x in out).items())),
                              'subjects': dict(collections.Counter(x['subject'] for x in out))}
    return out


def build_aime24(raw, st):
    df = pd.read_parquet(raw.path('aime24', 'data/train-00000-of-00001.parquet'))
    mj = pd.read_parquet(raw.path('aime24_maxwelljia', 'aime_2024_problems.parquet'))
    mjb = {r['ID']: r for r in mj.to_dict('records')}
    out, mism, zero_pad, sims = [], [], [], {}
    for idx, r in enumerate(df.to_dict('records')):
        m = re.search(r'/(\d{4})_AIME_(I{1,2})_Problems/Problem_(\d+)$', r['url'])
        key = f'{m.group(1)}-{m.group(2)}-{int(m.group(3))}'
        ans = str(int(r['answer']))
        if ans != r['answer']:
            zero_pad.append({'orig_id': key, 'raw': r['answer'], 'answer': ans})
        o = mjb[key]
        sims[key] = round(jaccard(r['problem'], o['Problem']), 3)
        if int(o['Answer']) != int(ans):
            mism.append({'orig_id': key, 'HuggingFaceH4': r['answer'], 'Maxwell-Jia': int(o['Answer'])})
        out.append(item('aime24', idx, r['problem'].strip(), ans, orig_id=key,
                        meta={'url': r['url'], 'hf_id': int(r['id'])}))
    assert len(mjb) == 30 and {x['orig_id'] for x in out} == set(mjb)
    st['cross_checks']['aime24'] = {
        'copies': ['HuggingFaceH4/aime_2024 (used)', 'Maxwell-Jia/AIME_2024'], 'aligned_by': 'AoPS url -> ID',
        'n_compared': len(out), 'answer_value_mismatches': mism,
        'leading_zero_answers_normalized': zero_pad,
        'problem_word_jaccard_min': min(sims.values()),
        'note': 'problem texts differ only in formatting / [asy] figures (checked by eye for the lowest-similarity '
                'pairs)'}
    return out


def build_aime25(raw, st):
    df = pd.read_parquet(raw.path('aime25', 'data/train-00000-of-00001.parquet'))
    oc = (jsonl(raw.path('aime25_opencompass', 'aime2025-I.jsonl')) +
          jsonl(raw.path('aime25_opencompass', 'aime2025-II.jsonl')))
    assert len(df) == 30 and len(oc) == 30
    out, str_mism, val_mism, sims, argmax_ok = [], [], [], [], True
    for idx, r in enumerate(df.to_dict('records')):
        n = int(r['problem_idx'])
        assert n == idx + 1
        key = f'2025-{"I" if n <= 15 else "II"}-{n if n <= 15 else n - 15}'
        ans = str(int(r['answer']))
        o = oc[idx]
        o_ans = str(o['answer']).strip()
        s = [jaccard(r['problem'], x['question']) for x in oc]
        sims.append(round(s[idx], 3))
        argmax_ok &= max(range(30), key=lambda k: s[k]) == idx
        if o_ans != ans:
            o_val = re.sub(r'\^\s*\{?\\circ\}?', '', o_ans)
            entry = {'orig_id': key, 'MathArena': ans, 'opencompass': o_ans}
            (str_mism if o_val.isdigit() and int(o_val) == int(ans) else val_mism).append(entry)
        out.append(item('aime25', idx, r['problem'].strip(), ans, orig_id=key,
                        meta={'problem_idx': n, 'problem_type': list(r['problem_type'])}))
    st['cross_checks']['aime25'] = {
        'copies': ['MathArena/aime_2025 (used)', 'opencompass/AIME2025 (I + II jsonl)'],
        'aligned_by': 'position (problem_idx 1-15 = I, 16-30 = II); confirmed: each MathArena problem is most '
                      'similar to its aligned opencompass problem' if argmax_ok else 'ALIGNMENT CHECK FAILED',
        'n_compared': 30, 'answer_value_mismatches': val_mism, 'answer_format_only_mismatches': str_mism,
        'problem_word_jaccard_min': min(sims),
        'note': 'MathArena statements include TikZ figure code for figure problems; opencompass are text-only'}
    return out


def build_aime26(raw, st):
    df = pd.read_parquet(raw.path('aime26', 'data/train-00000-of-00001.parquet'))
    out = []
    for idx, r in enumerate(df.to_dict('records')):
        n = int(r['problem_idx'])
        assert n == idx + 1
        out.append(item('aime26', idx, r['problem'].strip(), str(int(r['answer'])), orig_id=f'problem_idx={n}',
                        meta={'problem_idx': n}))
    st['cross_checks']['aime26'] = {'copies': ['MathArena/aime_2026 (only copy)'], 'note': 'no second copy; '
                                    'answers not cross-checked'}
    return out


def build_amc(raw, st):
    df = pd.read_parquet(raw.path('amc', 'data/train-00000-of-00001.parquet'))
    out = []
    for idx, r in enumerate(df.to_dict('records')):
        a = float(r['answer'])
        assert a == int(a), (idx, a)
        oid = re.search(r'index\.php/(.+)$', r['url']).group(1)
        out.append(item('amc', idx, r['problem'].strip(), str(int(a)), orig_id=oid,
                        meta={'url': r['url'], 'hf_id': int(r['id'])}))
    assert len({x['orig_id'] for x in out}) == len(out)
    st['notes']['amc'] = {'years': dict(collections.Counter(x['orig_id'][:4] for x in out)),
                          'note': 'hf "id" column is not unique (49 distinct); orig_id = AoPS page name'}
    return out


def top_level_split(s):
    parts, depth, cur = [], 0, ''
    for ch in s:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def build_olympiad(raw, st):
    ob = pd.read_parquet(raw.path('olympiad', 'OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet'))
    kn = pd.read_parquet(raw.path('olympiad_knoveleng', 'data/train-00000-of-00001.parquet'))
    obr, knr = ob.to_dict('records'), kn.to_dict('records')
    knb = {r['id']: r for r in knr}
    assert all(r['modality'] == 'Text-only' and r['question_type'] == 'Open-ended' and r['language'] == 'English'
               and r['subject'] == 'Math' and r['difficulty'] == 'Competition' for r in obr)
    assert all(len(r['final_answer']) == 1 for r in obr + knr)
    out, drops, fixes = [], [], []
    q_diff, a_diff = [], []
    for idx, r in enumerate(obr):
        k = knb[r['id']]
        if k['question'] != r['question']:
            q_diff.append(int(r['id']))
        if list(k['final_answer']) != list(r['final_answer']):
            a_diff.append({'id': int(r['id']), 'official': r['final_answer'][0], 'knoveleng': k['final_answer'][0]})
        raw_ans = r['final_answer'][0]
        s = raw_ans.strip().rstrip('.').strip()
        parts = top_level_split(s.replace('$', ''))
        base = {'pid': f'olympiad:{idx}', 'orig_id': str(r['id']), 'answer': raw_ans,
                'is_multiple_answer': bool(r['is_multiple_answer'])}
        if len(parts) > 1:
            reason = 'multiple_answers' if r['is_multiple_answer'] else 'multi_part_answer'
            drops.append(dict(base, reason=reason))
            continue
        ans = s
        if ans.startswith('$') and ans.endswith('$') and ans.count('$') == 2:
            ans = ans[1:-1].strip()
        assert '$' not in ans, (r['id'], raw_ans)
        ans = re.sub(r'^(?:\\q?quad\s*)+|(?:\s*\\q?quad)+$', '', ans).strip()
        if ans != raw_ans.strip().strip('$').strip():
            fixes.append({'pid': f'olympiad:{idx}', 'orig_id': str(r['id']), 'raw': raw_ans, 'answer': ans,
                          'why': 'generic cleanup: trailing period / \\quad outside the math'})
        if int(r['id']) in OLYMPIAD_ANSWER_FIX:
            new, why = OLYMPIAD_ANSWER_FIX[int(r['id'])]
            fixes.append({'pid': f'olympiad:{idx}', 'orig_id': str(r['id']), 'raw': raw_ans, 'answer': new,
                          'why': why})
            ans = new
        meta = {'ob_answer_type': r['answer_type'], 'is_multiple_answer': bool(r['is_multiple_answer'])}
        if isinstance(r['unit'], str):
            meta['unit'] = r['unit']
        if isinstance(r['error'], str):
            meta['error_tolerance'] = r['error']
        if ans != raw_ans:
            meta['raw_final_answer'] = raw_ans
        out.append(item('olympiad', idx, r['question'].strip(), ans, orig_id=r['id'], subject=r['subfield'],
                        meta=meta))
    only_kn = [r for r in knr if r['id'] not in {x['id'] for x in obr}]
    for r in only_kn:
        drops.append({'pid': None, 'orig_id': str(r['id']), 'answer': r['final_answer'][0],
                      'is_multiple_answer': bool(r['is_multiple_answer']), 'reason': 'free_text_answer',
                      'note': 'present only in the knoveleng copy (absent from Hothan/OlympiadBench); answer is a '
                              'function family with free-text conditions'})
    assert len(obr) + len(only_kn) == EXPECTED_CANDIDATES['olympiad']
    st['cross_checks']['olympiad'] = {
        'copies': ['Hothan/OlympiadBench OE_TO_maths_en_COMP (used, 674 rows)',
                   'knoveleng/OlympiadBench (675 rows = official 674 + id 1965)'],
        'ids_only_in_knoveleng': [int(r['id']) for r in only_kn],
        'question_text_differs': q_diff,
        'question_text_differs_note': 'knoveleng keeps the relay placeholder "Let $T=T N Y W R$" (unsolvable); '
                                      'the official copy substitutes the value of T',
        'final_answer_differs': a_diff}
    st['answer_fixes']['olympiad'] = fixes
    st['dropped']['olympiad'] = drops
    st['candidates']['olympiad'] = len(obr) + len(only_kn)
    return out


def build_minerva(raw, st):
    rows = jsonl(raw.path('minerva', 'test.jsonl'))
    out, fixes = [], []
    for idx, r in enumerate(rows):
        ans = r['answer'].strip()
        meta = {}
        if idx in MINERVA_ANSWER_FIX:
            new, why = MINERVA_ANSWER_FIX[idx]
            fixes.append({'pid': f'minerva:{idx}', 'raw': r['answer'], 'answer': new, 'why': why})
            meta['raw_answer'] = r['answer']
            ans = new
        out.append(item('minerva', idx, r['question'].strip(), ans, orig_id=f'test-{idx}', meta=meta))
    st['answer_fixes']['minerva'] = fixes
    return out


def build_gpqa(raw, st):
    gj = pd.read_parquet(raw.path('gpqa_jingzeshi', 'data/test-00000-of-00001.parquet'))
    gt = pd.read_parquet(raw.path('gpqa_talzoomanzoo', 'data/train-00000-of-00001.parquet'))
    rows, trows = gj.to_dict('records'), gt.to_dict('records')
    cols = ['Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']
    out, dup_incorrect = [], []
    for idx, r in enumerate(rows):
        opts_raw = [str(r[c]).strip() for c in cols]
        assert all(isinstance(r[c], str) and r[c].strip() for c in cols), idx
        assert opts_raw[0] not in opts_raw[1:], idx
        if len(set(opts_raw[1:])) < 3:
            dup_incorrect.append({'pid': f'gpqa:{idx}', 'record_id': r['Record ID'], 'options': opts_raw})
        perm = [0, 1, 2, 3]
        random.Random(1234 + idx).shuffle(perm)
        opts = [opts_raw[k] for k in perm]
        letter = 'ABCD'[perm.index(0)]
        problem = r['Question'].strip() + '\n\n' + '\n'.join(f'{L}) {o}' for L, o in zip('ABCD', opts))
        out.append(item('gpqa', idx, problem, letter, answer_type='mcq', orig_id=r['Record ID'],
                        subject=r['High-level domain'],
                        meta={'subdomain': r['Subdomain'], 'perm': perm, 'options': opts}))
    same_q = sum(a['Question'].strip() == b['Question'].strip() for a, b in zip(rows, trows))
    same_c = sum(str(a['Correct Answer']).strip() == str(b['Correct Choice']).strip() for a, b in zip(rows, trows))
    canary = collections.Counter(r['Canary String'] for r in rows)
    st['cross_checks']['gpqa'] = {
        'official': 'Idavidrein/gpqa gpqa_diamond.csv is gated (401) -> not downloaded',
        'copies': ['JingzeShi/gpqa_diamond (used; full original GPQA csv columns)',
                   'talzoomanzoo/gpqa_diamond_raw (Question / Correct Choice only; Choices column is null)'],
        'n_rows': len(rows), 'unique_record_ids': len({r['Record ID'] for r in rows}),
        'all_rows_have_gpqa_canary': len(canary) == 1 and next(iter(canary)).startswith('gpqa:'),
        'all_rows_1_correct_3_incorrect_nonempty': True,
        'correct_equals_an_incorrect': 0,
        'rows_with_duplicate_incorrect_options': dup_incorrect,
        'second_mirror_same_question_same_order': same_q, 'second_mirror_same_correct_answer': same_c,
        'domains': dict(collections.Counter(r['High-level domain'] for r in rows)),
        'diamond_criteria': {
            'expert_validator_1_correct': int(sum(r['Validator Answered Correctly_EV_1'] == 1 for r in rows)),
            'expert_validator_2_correct': int(sum(r['Validator Answered Correctly_EV_2'] == 1 for r in rows)),
            'ev2_wrong_post_hoc_agree': int(sum(r['Validator Answered Correctly_EV_2'] == 0 and
                                                r['Post hoc agreement_EV_2'] == 'Agree' for r in rows)),
            'ev2_wrong_post_hoc_disagree': int(sum(r['Validator Answered Correctly_EV_2'] == 0 and
                                                   r['Post hoc agreement_EV_2'] == 'Disagree' for r in rows)),
            'majority_non_experts_incorrect': int(sum(r['Majority Non-Expert Vals Incorrect'] == 1 for r in rows)),
        },
        'shuffle': 'perm = [0,1,2,3] (0 = correct); random.Random(1234 + idx).shuffle(perm); option at display '
                   'position j is original option perm[j]',
        'answer_letter_counts': dict(sorted(collections.Counter(x['answer'] for x in out).items())),
    }
    return out


BUILDERS = {'gsm8k': build_gsm8k, 'math500': build_math500, 'aime24': build_aime24, 'aime25': build_aime25,
            'aime26': build_aime26, 'amc': build_amc, 'olympiad': build_olympiad, 'minerva': build_minerva,
            'gpqa': build_gpqa}


def gold_parse_report(pool):
    """Gold answers that the grader's math-verify path cannot parse into a sympy object (they can only be graded
    by normalized string comparison). Uses the same parse as scripts/grader.py."""
    sys.path.insert(0, os.path.join(GD, 'scripts'))
    import grader
    bad = []
    for x in pool:
        if x['answer_type'] != 'math':
            continue
        g = grader._strip_text_units(grader._pre(x['answer']))
        p = grader._call_with_timeout(lambda: grader._mv_parse(g, False), grader.TIMEOUT_S, ())
        if not p or isinstance(p[0], str):
            bad.append({'pid': x['pid'], 'answer': x['answer']})
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stats', required=True)
    args = ap.parse_args()
    with open(args.manifest) as f:
        man = json.load(f)
    raw = Raw(args.raw_root, man)
    st = {'notes': {}, 'cross_checks': {}, 'answer_fixes': {}, 'dropped': {}, 'candidates': {}}
    pool = []
    for src in SOURCE_ORDER:
        items = BUILDERS[src](raw, st)
        st['candidates'].setdefault(src, len(items))
        assert st['candidates'][src] == EXPECTED_CANDIDATES[src], (src, st['candidates'][src])
        pool += items
    pids = [x['pid'] for x in pool]
    assert len(set(pids)) == len(pids)
    counts = collections.Counter(x['source'] for x in pool)

    with open(args.out + '.tmp', 'w') as f:
        for x in pool:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    os.replace(args.out + '.tmp', args.out)

    dropped = {}
    for src in SOURCE_ORDER:
        d = st['dropped'].get(src, [])
        dropped[src] = {'count': len(d), 'by_reason': dict(collections.Counter(x['reason'] for x in d)), 'items': d}
    sources = {}
    for src in SOURCE_ORDER:
        sources[src] = {}
        for name in RAW_USED[src]:
            e = man[name]
            sources[src][name] = {'repo_id': e['repo_id'], 'revision': e.get('revision'), 'status': e['status'],
                                  'files': {fn: v['sha256'] for fn, v in e.get('files', {}).items()}}
            if e['status'] != 'ok':
                sources[src][name]['error'] = e.get('error', '')[:200]
    stats = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'script': 'scripts/data/build_pool.py',
        'total': len(pool),
        'counts': {s: counts[s] for s in SOURCE_ORDER},
        'candidates_before_drops': {s: st['candidates'][s] for s in SOURCE_ORDER},
        'expected_candidates': EXPECTED_CANDIDATES,
        'answer_type_counts': dict(collections.Counter(x['answer_type'] for x in pool)),
        'dropped': dropped,
        'drop_reasons': {
            'multiple_answers': 'is_multiple_answer=True and the final answer lists more than one value / tuple / '
                                'function (top-level comma-separated); not a single answer, and last-\\boxed '
                                'extraction cannot grade lists',
            'multi_part_answer': 'is_multiple_answer=False but the final answer has several parts '
                                 '("$f(n)=n$, $g(n)=1$")',
            'free_text_answer': 'answer is a family of objects described with free-text conditions'},
        'answer_fixes': st['answer_fixes'],
        'sources': sources,
        'raw_files_sha256_verified': sorted(raw.checked),
        'cross_checks': st['cross_checks'],
        'notes': st['notes'],
        'gold_not_parsed_by_math_verify': gold_parse_report(pool),
    }
    with open(args.stats + '.tmp', 'w') as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    os.replace(args.stats + '.tmp', args.stats)
    print('pool', len(pool), dict(stats['counts']))
    print('dropped', {s: (v['count'], v['by_reason']) for s, v in dropped.items() if v['count']})
    print('gold not parsed by math-verify:', stats['gold_not_parsed_by_math_verify'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
