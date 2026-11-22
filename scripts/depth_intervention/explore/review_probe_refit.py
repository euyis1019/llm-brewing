"""Review of train_probe.py: independent float64 CPU refits (scipy L-BFGS, whitening on training rows only).
Checks that weights_full[f, l] = argmin mean CE + lam*||W||^2 on rows fold != f (labels gold / null), that lam_path's
inner CE (fit on fold not in {f, g}, g = (f+1)%5, CE on fold g) matches, and the bias rung.
Usage: python scripts/explore/review_probe_refit.py RUN [quick]"""
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import common as C

run = sys.argv[1]
d = os.path.join(C.ROOT, 'results', run); pd = os.path.join(d, 'probe')
meta = json.load(open(f'{d}/extract/meta.json')); N, L, D = meta['N'], meta['L'], meta['d']
hid = np.memmap(f'{d}/extract/hidden_bf16.u16', dtype=np.uint16, mode='r', shape=(N, L, D))
rows = json.load(open(f'{d}/extract/rows.json'))
fold = np.array([r['fold'] for r in rows]); gold = np.array([r['gold'] for r in rows])
lens4 = np.load(f'{d}/extract/lens4.npy'); eps = float(np.load(f'{d}/extract/lens_params.npz')['eps'])
nulls = np.load(f'{pd}/null_labels.npy').astype(int)
wf = np.load(f'{pd}/weights_full.npz'); wb = np.load(f'{pd}/weights_bias.npz'); lp = np.load(f'{pd}/lam_path.npz')
LAMS = lp['lams']


def ce_of(lg, y):
    m = lg.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(lg - m).sum(1))
    return lse - lg[np.arange(len(y)), y], lse


class Fitter:
    def __init__(self, X, lens, y):
        self.mu = X.mean(0); Xc = X - self.mu
        ev, self.Q = np.linalg.eigh(Xc.T @ Xc / len(X))
        self.ev = np.maximum(ev, 0); self.Zq = Xc @ self.Q; self.lens, self.y = lens, y

    def fit(self, lam):
        n, Dd = self.Zq.shape
        sc = 1 / np.sqrt(self.ev + lam); Z = self.Zq * sc; Y = np.eye(4)[self.y]

        def f(p):
            V, c = p[:4 * Dd].reshape(4, Dd), p[4 * Dd:]
            lg = self.lens + Z @ V.T + c
            ce, lse = ce_of(lg, self.y)
            G = (np.exp(lg - lse[:, None]) - Y) / n
            return ce.mean() + lam * ((V * sc) ** 2).sum(), np.concatenate([(G.T @ Z + 2 * lam * V * sc ** 2).ravel(), G.sum(0)])
        r = minimize(f, np.zeros(4 * Dd + 4), jac=True, method='L-BFGS-B',
                     options=dict(maxiter=20000, maxfun=40000, maxcor=30, gtol=1e-11, ftol=1e-15))
        V, c = r.x[:4 * Dd].reshape(4, Dd), r.x[4 * Dd:]
        W = (V * sc) @ self.Q.T
        return W, c - W @ self.mu, r


def obj(X, lens, y, W, b, lam):
    return ce_of(lens + X @ W.T + b, y)[0].mean() + lam * (W ** 2).sum()


def hn_of(l):
    h = C.bf16_bits_to_f32(hid[:, l]).astype(np.float64)
    return h / np.sqrt((h ** 2).mean(-1, keepdims=True) + eps)


t0 = time.time()
cases = [(20, 0, 0), (35, 3, 0), (23, 1, 0), (35, 2, 1), (24, 4, 2)]  # (layer, outer fold, label set: 0 gold, s>0 null s-1)
quick = len(sys.argv) > 2
if quick:
    cases = [(int(round(20 * L / 36)), 0, 0), (L - 1, 3, 0)]
for l, f, s in cases:
    X = hn_of(l); y = gold if s == 0 else nulls[s - 1]
    oof = np.load(f'{pd}/oof_full.npy' if s == 0 else f'{pd}/oof_null{s - 1}.npy', mmap_mode='r')[:, l]
    lam = float(wf['lam'][f, l]) if s == 0 else float(lp['lam_null'][s - 1, f, l])
    tr, te = fold != f, fold == f
    W, b, r = Fitter(X[tr], lens4[tr, l], y[tr]).fit(lam)
    z = lens4[te, l] + X[te] @ W.T + b
    msg = (f'[{run}] L{l} f{f} set{s} lam={lam:.1e}: scipy nit={r.nit} |g|max={np.abs(r.jac).max():.1e}  '
           f'max|dlogit| oof vs ref={np.abs(z - oof[te]).max():.4f} argmax agree={(z.argmax(1) == np.asarray(oof[te]).argmax(1)).mean():.5f} '
           f'acc ref/oof={(z.argmax(1) == y[te]).mean():.4f}/{(np.asarray(oof[te]).argmax(1) == y[te]).mean():.4f}')
    if s == 0:  # their stored W, b vs reference on the training objective; and a wrong-fold hypothesis
        gap = obj(X[tr], lens4[tr, l], y[tr], wf['W'][f, l].astype(np.float64), wf['b'][f, l].astype(np.float64), lam) - obj(X[tr], lens4[tr, l], y[tr], W, b, lam)
        msg += f' objgap(theirs-ref)={gap:.2e} relW={np.linalg.norm(wf["W"][f, l] - W) / np.linalg.norm(W):.2e}'
    print(msg, C.stamp(t0), flush=True)

# inner lam path for one (f, l), label sets gold and null0: fit on fold not in {f, g}, CE on fold g
for l, f in [] if quick else [(20, 0), (30, 2)]:
    X = hn_of(l); g = (f + 1) % 5
    tr, va = (fold != f) & (fold != g), fold == g
    for s in (0, 1):
        y = gold if s == 0 else nulls[0]
        F = Fitter(X[tr], lens4[tr, l], y[tr]); ref = []
        for j, lam in enumerate(LAMS):
            if j < 2: ref.append(np.nan); continue  # lam 1 and 0.32: slow and irrelevant (never selected)
            W, b, r = F.fit(lam)
            ref.append(ce_of(lens4[va, l] + X[va] @ W.T + b, y[va])[0].mean())
        theirs = lp['val_ce'][s, f, l]
        print(f'[{run}] inner L{l} f{f} set{s}: ref CE={np.round(ref, 5).tolist()}\n      theirs CE={np.round(theirs, 5).tolist()}\n'
              f'      max|diff|={np.nanmax(np.abs(np.array(ref) - theirs)):.1e} argmin ref={np.nanargmin(ref)} theirs={int(np.argmin(theirs))} '
              f'stored lam={(wf["lam"][f, l] if s == 0 else lp["lam_null"][0, f, l]):.1e}', C.stamp(t0), flush=True)

# bias rung: 4 params on fold != f
for l, f in [(24, 0), (L - 1, 4), (10, 2)]:
    tr, te = fold != f, fold == f
    def fb(c):
        lg = lens4[tr, l].astype(np.float64) + c; ce, lse = ce_of(lg, gold[tr])
        return ce.mean(), ((np.exp(lg - lse[:, None]) - np.eye(4)[gold[tr]]).mean(0))
    c = minimize(fb, np.zeros(4), jac=True, method='L-BFGS-B', options=dict(gtol=1e-12, ftol=1e-16)).x
    ob = np.load(f'{pd}/oof_bias.npy', mmap_mode='r')[te, l]
    zb = lens4[te, l] + c
    dz = (zb - zb.mean(1, keepdims=True)) - (ob - ob.mean(1, keepdims=True))
    print(f'[{run}] bias L{l} f{f}: ref b-mean={np.round(c - c.mean(), 4)} theirs={np.round(wb["b"][f, l] - wb["b"][f, l].mean(), 4)} '
          f'max|dlogit (centred)|={np.abs(dz).max():.1e}', C.stamp(t0), flush=True)
