"""Adversarial review of steer.py (CPU): independent re-derivation of the L1 selection and the dev / lockbox numbers from
the saved run arrays (RUN/steer/*.npy), written without steer.py's evaluate(). Classes come from RUN/classify/questions.csv
(last_layer=output). Gold is used only to score.
Usage: python scripts/explore/review_steer_cpu.py RUN"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import csv, json, os, sys
import numpy as np

ROOT = str(_experiment_root('depth_intervention'))
run = sys.argv[1]
rd = os.path.join(ROOT, 'results', run)
od = os.path.join(rd, 'steer')
meta = json.load(open(os.path.join(od, 'meta.json')))
names = meta['configs']
ci_ = {n: i for i, n in enumerate(names)}
layers, alphas, heads = meta['layers'], meta['alphas'], meta['heads']
rows = json.load(open(os.path.join(rd, 'extract', 'rows.json')))
rid = np.load(os.path.join(od, 'rows.npy'))
assert (rid == np.arange(len(rows))).all()
N = len(rid); Q = N // 4
gold = np.array([r['gold'] for r in rows]); v = np.arange(N) % 4
foldq = np.array([r['fold'] for r in rows])[0::4]
qc = list(csv.DictReader(open(os.path.join(rd, 'classify', 'questions.csv'))))
cls = np.array([r['outcome_v1_K3'] for r in qc]); ua = np.array([r['u_avail_K3'] == '1' for r in qc])
keepq = cls != 'X'

f4 = np.load(os.path.join(od, 'final4.npy'))                     # (N, nl, nc, 4)
top1 = np.load(os.path.join(od, 'top1.npy'))
s = np.sort(f4, -1)
let = np.where((s[..., -1] - s[..., -2] >= 1e-4) & np.isin(top1, meta['letter_ids']), f4.argmax(-1), -1)
head4 = np.load(os.path.join(od, 'head4.npy')).astype(np.float64)
p = np.exp(head4 - head4.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
hconf, hk = p.max(-1), head4.argmax(-1)
lensl = np.load(os.path.join(od, 'lens_l.npy'))
notem = lensl.argmax(-1)[:, :, None] != hk
tz = np.load(os.path.join(rd, 'consensus_probe', 'targets.npz'))
vote_c = np.repeat(tz['vote'].astype(int), 4)
cont = lambda x: np.where(x >= 0, (x + v) % 4, -1)
i0 = ci_['none']
TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def gate(li, hi, g, tau):
    if g == 'none':
        return np.ones(N, bool)
    if g == 'cons_conf':
        return hconf[:, li, hi] >= tau
    if g == 'not_emitted':
        return notem[:, li, hi]
    return (hconf[:, li, hi] >= tau) & notem[:, li, hi]


cands = [None]
for t in ('maxcons', 'vote'):
    for li in range(len(layers)):
        for a in alphas:
            for g in ('none', 'cons_conf', 'not_emitted', 'both'):
                for tau in (TAUS if g in ('cons_conf', 'both') else (None,)):
                    cands.append((t, li, a, g, tau))
devr = np.repeat(keepq & (foldq != 4), 4)
agree0 = {li: (cont(let[:, li, i0]) == vote_c).astype(np.float64) for li in range(len(layers))}
S = np.zeros((len(cands), N))
for k, cd in enumerate(cands[1:], 1):
    t, li, a, g, tau = cd
    c = ci_[f'{t}:cstar_perp@{a:g}']
    S[k] = gate(li, heads.index(t), g, tau) * ((cont(let[:, li, c]) == vote_c) - agree0[li])
sc = S[:, devr].mean(1)
best = int(np.argmax(sc))
fmt = lambda cd: 'abstain' if cd is None else f'{cd[0]} L{layers[cd[1]]} a={cd[2]:g} gate={cd[3]}' + (f'@{cd[4]:g}' if cd[4] is not None else '')
print(run, 'my L1 selection on folds 0-3:', fmt(cands[best]), f'{100 * sc[best]:+.3f}pp; declared:',
      json.load(open(os.path.join(od, 'lockbox_declared.json')))['final'])
fr = np.repeat(foldq, 4)
cf = {f: int(np.argmax(S[:, devr & (fr != f)].mean(1))) for f in range(4)}
print('  my cross-fit picks', {f: fmt(cands[cf[f]]) for f in range(4)})

rng = np.random.default_rng(7)  # independent bootstrap (different seed from steer.py's)


def net_ci(after, before, qm, B=4000):
    okA, okB = (after == gold).reshape(Q, 4).sum(1), (before == gold).reshape(Q, 4).sum(1)
    d = (okA - okB)[qm] / 4.0
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [round(100 * d.mean(), 3), round(100 * np.percentile(bs, 2.5), 3), round(100 * np.percentile(bs, 97.5), 3)]


def pol(t, li, dname, a, g, tau, gate_head=None):
    c = ci_[f'rand_perp@{a:g}'] if dname == 'rand_perp' else ci_[f'{t}:{dname}@{a:g}']
    gm = gate(li, heads.index(gate_head or t), g, tau)
    return np.where(gm, let[:, li, c], let[:, li, i0]), let[:, li, i0], gm


def crossfit_after():
    after, before = let[:, 0, i0].copy(), let[:, 0, i0].copy()
    for f in range(4):
        cd = cands[cf[f]]
        if cd is None:
            continue
        a_, b_, _ = pol(cd[0], cd[1], 'cstar_perp', cd[2], cd[3], cd[4])
        m = fr == f
        after[m], before[m] = a_[m], b_[m]
    return after, before


dev_q, lock_q = keepq & (foldq != 4), keepq & (foldq == 4)
out = dict(selection=fmt(cands[best]), crossfit={f: fmt(cands[cf[f]]) for f in range(4)})
A_, B_ = crossfit_after()
out['dev_crossfit_net_pp'] = net_ci(A_, B_, dev_q)
t, li, a, g, tau = cands[best]
res = {}
for nm, dn in (('method', 'cstar_perp'), ('rand', 'rand_perp'), ('runner', 'runner_perp'), ('par', 'cstar_par')):
    aft, bef, gm = pol(t, li, dn, a, g, tau, gate_head=t)
    res[nm] = aft
    out[nm] = dict(dev=net_ci(aft, bef, dev_q), lock=net_ci(aft, bef, lock_q), steered_lock=round(float(gm[np.repeat(lock_q, 4)].mean()), 4))
bef = let[:, li, i0]
for nm in ('rand', 'runner', 'par'):
    out[f'method_minus_{nm}_lock'] = net_ci(res['method'], res[nm], lock_q)
# R harm / U->R on the lockbox
okA, okB = (res['method'] == gold).reshape(Q, 4).sum(1), (bef == gold).reshape(Q, 4).sum(1)
isR4 = (cls == 'R') & lock_q & (okB == 4)
out['R_harm_lock'] = [int((isR4 & (okA < 4)).sum()), int(isR4.sum())]
for k, m in (('R', cls == 'R'), ('O', cls == 'O'), ('M', cls == 'M'), ('U', cls == 'U'), ('U_avail', (cls == 'U') & ua),
             ('U_none', (cls == 'U') & ~ua)):
    out[f'lock_by_class_{k}'] = net_ci(res['method'], bef, lock_q & m) if (lock_q & m).any() else None
# ungated method, all layers/alphas (descriptive, lockbox) for the selected head
out['lock_ungated'] = {f'L{layers[l_]} a={x:g}': net_ci(let[:, l_, ci_[f'{t}:cstar_perp@{x:g}']], let[:, l_, i0], lock_q)
                       for l_ in range(len(layers)) for x in alphas}
print(json.dumps(out, indent=1))
os.makedirs(os.path.join(rd, 'steer_review'), exist_ok=True)
json.dump(out, open(os.path.join(rd, 'steer_review', 'cpu_recompute.json'), 'w'), indent=1)
