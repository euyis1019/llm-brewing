"""Collect round-1 headline numbers from every run into results/round1_summary.json (for RESULTS/report)."""
import csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RUNS = ['qwen3-4b-base__base_fewshot', 'qwen3-4b__chat0', 'qwen3-4b__base_fewshot', 'llama3.1-8b__base_fewshot',
        'qwen2.5-7b__base_fewshot', 'qwen2.5-7b-instruct__chat0',
        'qwen3-4b-base__base_fewshot__ext', 'qwen3-4b__chat0__ext']


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    out = {}
    for run in RUNS:
        d = os.path.join(C.ROOT, 'results', run)
        meta = jl(f'{d}/extract/meta.json')
        if meta is None:
            continue
        r = dict(acc=meta['acc4'], compliance=meta['compliance'], N=meta['N'], L=meta['L'])
        rows = json.load(open(f'{d}/extract/rows.json'))
        gold = np.array([x['gold'] for x in rows])
        csd = np.load(f'{d}/extract/csd_raw4.npy', mmap_mode='r')
        ok = np.asarray(csd).argmax(-1) == gold[:, None]
        r['csd_acc'] = ok.mean(0).round(4).tolist()
        r['csd_acc4c'] = ok.reshape(-1, 4, ok.shape[1]).all(1).mean(0).round(4).tolist()
        ps = jl(f'{d}/probe/summary.json')
        if ps:
            r['probe_acc'] = {k: np.round(v, 4).tolist() for k, v in ps['acc'].items() if k in ('lens', 'bias', 'full')}
            r['probe_acc4c_full'] = np.round(ps['acc4c']['full'], 4).tolist()
            no = ps['acc_null_own']
            r['null_own_range'] = [float(np.min(list(no.values()) if isinstance(no, dict) else no)),
                                   float(np.max(list(no.values()) if isinstance(no, dict) else no))]
        cs = jl(f'{d}/classify/summary.json')
        if cs:
            r['classify_config'] = cs['config']
            r['v1'] = {K: v.get('counts', {}).get('all') for K, v in cs['v1'].items()}
            r['v0'] = cs['v0']['counts']['all']
        q = f'{d}/classify/questions.csv'
        if os.path.exists(q):
            Q = list(csv.DictReader(open(q)))
            oc = np.array([x['outcome_v1_K3'] for x in Q])
            fp = np.array([int(x['FPCL_c_K3']) for x in Q]); fj = np.array([int(x['FJC_c_K3']) for x in Q])
            R = oc == 'R'
            r['R_FPCL_quart'] = np.percentile(fp[R & (fp >= 0)], [25, 50, 75]).tolist() if R.any() else None
            r['R_FJC_quart'] = np.percentile(fj[R], [25, 50, 75]).tolist() if R.any() else None
            r['by_source_K3'] = {s: {c: int(((oc == c) & (np.array([x['source'] for x in Q]) == s)).sum()) for c in 'ROMUX'}
                                 for s in C.SOURCES}
        ee = jl(f'{d}/early_exit/summary.json')
        if ee:
            r['early_exit_verdict'] = ee.get('A1') or ee.get('verdict')
        for ld in ('loop_allq', 'loop'):
            le = jl(f'{d}/{ld}/eval/summary.json')
            if le:
                r[f'{ld}_A2'] = {k: dict(passing=v['passing'], verdict=v['verdict']) for k, v in le['A2'].items()}
        out[run] = r
    C.save_json(out, os.path.join(C.ROOT, 'results', 'round1_summary.json'))
    print('runs:', list(out))


if __name__ == '__main__':
    main()
