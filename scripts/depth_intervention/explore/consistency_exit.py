"""Quick test: label-free (L0/L1) question-level early exit by cross-version content consistency.
For each question and layer l in [w0, L-1], readout r in {csd, lens}: content of each version's argmax; consistency =
size of the modal content. Policies:
  final      : real output per version (reference)
  vote_final : majority content of the 4 final outputs (permutation ensembling, no depth)
  cons_exit  : layer with max consistency (ties -> latest), output that layer's modal content for all 4 versions
  cons_gate  : keep real outputs unless final consistency < 4 and some layer >= w0 has consistency 4 -> that layer's content
Scored at question level (all 4 versions correct) and row level, and on O / decoy-O / R / U / M questions."""
import csv, json, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C
run = sys.argv[1]
d = os.path.join(C.ROOT, 'results', run)
rows = json.load(open(f'{d}/extract/rows.json'))
gold = np.array([r['gold'] for r in rows]); v = np.array([r['v'] for r in rows])
ans = ((gold + v) % 4).reshape(-1, 4)[:, 0]
out = np.load(f'{d}/extract/out_logits4.npy')
q = {int(r['qid']): r for r in csv.DictReader(open(f'{d}/classify/questions.csv'))}
oc = np.array([q[i]['outcome_v1_K3'] for i in range(len(q))])
L = out.shape[0] and np.load(f'{d}/extract/lens4.npy', mmap_mode='r').shape[1]
w0 = round(0.45 * L)
def contents(lg):  # (N, L, 4) -> (Q, 4, L)
    c = (lg.argmax(-1) + v[:, None]) % 4
    return c.reshape(-1, 4, c.shape[-1])
def modal(c):  # c (Q, 4, L) -> modal content (Q, L), count (Q, L)
    cnt = np.stack([(c == k).sum(1) for k in range(4)], -1)  # (Q, L, 4)
    return cnt.argmax(-1), cnt.max(-1)
fin_c = ((out.argmax(-1) + v) % 4).reshape(-1, 4)
fm, fcnt = modal(fin_c[..., None]); fm, fcnt = fm[:, 0], fcnt[:, 0]
res = {}
def score(name, qc):  # qc (Q, 4) contents per version
    corr = qc == ans[:, None]
    r = dict(q4=float(corr.all(1).mean()), row=float(corr.mean()))
    for cl in 'ROMU':
        m = oc == cl
        if m.any(): r[f'{cl}_q4'] = f'{int(corr[m].all(1).sum())}/{int(m.sum())}'
    res[name] = r
score('final', fin_c)
score('vote_final', np.repeat(fm[:, None], 4, 1))
for rd in ['csd_raw4', 'lens4']:
    lg = np.load(f'{d}/extract/{rd}.npy')
    c = contents(lg)
    mc, cnt = modal(c)
    sub = cnt[:, w0:]
    lay = w0 + (sub.shape[1] - 1 - np.argmax(sub[:, ::-1], 1))  # latest layer with max consistency
    score(f'cons_exit[{rd}]', np.repeat(mc[np.arange(len(mc)), lay][:, None], 4, 1))
    full4 = cnt[:, w0:] == 4
    has = full4.any(1)
    last4 = w0 + (full4.shape[1] - 1 - np.argmax(full4[:, ::-1], 1))
    gate = (fcnt < 4) & has
    qc = fin_c.copy(); qc[gate] = mc[np.arange(len(mc)), last4][gate][:, None]
    score(f'cons_gate[{rd}]', qc)
    res[f'cons_gate[{rd}]']['n_gated'] = int(gate.sum())
print(run, f'w0={w0}')
for k, r in res.items(): print(f'  {k:18s}', json.dumps(r))
