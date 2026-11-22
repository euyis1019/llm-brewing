"""Row/question-level net accuracy of each loop config vs the same-batch none run (and vs extraction)."""
import json, os, sys, glob, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C
run, ld = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'loop_allq')
d = os.path.join(C.ROOT, 'results', run)
rows = json.load(open(f'{d}/extract/rows.json')); gold = np.array([r['gold'] for r in rows])
q = {int(r['qid']): r['outcome_v1_K3'] for r in csv.DictReader(open(f'{d}/classify/questions.csv'))}
def load(cfg):
    p = f'{d}/{ld}/{cfg}'
    if not os.path.exists(f'{p}/meta.json'): return None
    return np.load(f'{p}/rows.npy'), np.load(f'{p}/final4.npy'), np.load(f'{p}/final_top1.npy')
for mode in ['last', 'all']:
    base = load(f'{mode}_none')
    if base is None: continue
    r0, f0, t0 = base; g = gold[r0]; ok0 = f0.argmax(-1) == g
    qid = r0 // 4; cls = np.array([q[int(x)] for x in qid])
    print(f'{run} {ld} mode={mode} n_rows={len(r0)} none acc={ok0.mean():.4f}')
    for p in sorted(glob.glob(f'{d}/{ld}/{mode}_*/meta.json')):
        cfg = os.path.basename(os.path.dirname(p))
        r, f, t = load(cfg); assert (r == r0).all()
        ok = f.argmax(-1) == g
        letters = np.isin(t, json.load(open(f'{d}/extract/meta.json'))['letter_ids'])
        byc = {c: f'{(ok[cls == c].mean() - ok0[cls == c].mean()) * 100:+.1f}' for c in 'ROMU' if (cls == c).any()}
        print(f'  {cfg:16s} acc {ok.mean():.4f} net {100 * (ok.mean() - ok0.mean()):+.2f}pp  fix {int((ok & ~ok0).sum())} break {int((~ok & ok0).sum())}  letter-compliance {letters.mean():.3f}  by-class pp {byc}')
