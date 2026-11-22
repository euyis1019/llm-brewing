"""Self-supervised exit probes: from the final-normed residual at layer l, predict whether exiting there would emit
the same token as full depth (sampled with the run's noise). Labels come from the model's own replay (no gold).

Per probe layer: logistic regression (L2) on train problems; evaluated on held-out problems (split by pid hash):
AUROC, and the offline exit rule "exit at the first probe layer whose probability >= tau" (agreement with full depth,
mean executed blocks) for a grid of tau. Saves results/<model>/<tag>/exit_probe.npz (W (nl, d), b (nl,), layers).

Usage: python -I scripts/train_exit_probe.py --model qwen3-4b --tag base
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import argparse
import hashlib
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))


def auroc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def fit_logreg(X, y, lam=1e-3, steps=300, dev='cuda'):
    X = torch.tensor(X, dtype=torch.float32, device=dev)
    y = torch.tensor(y, dtype=torch.float32, device=dev)
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Xs = (X - mu) / sd
    w = torch.zeros(X.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=steps, history_size=20, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b, y) + lam * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    # fold standardization into the weights
    W = (w / sd).detach()
    B = (b - (mu / sd * w).sum()).detach()
    return W.cpu().numpy(), float(B.cpu().numpy()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--tag', default='base')
    ap.add_argument('--lam', type=float, default=1e-3)
    a = ap.parse_args()
    rd = os.path.join(GD, 'results', a.model, a.tag, 'replay')
    cfg = json.load(open(os.path.join(GD, 'results', a.model, a.tag, 'config.json')))
    L = cfg['L']
    parts = sorted(f for f in os.listdir(rd) if f.startswith('part') and f.endswith('.npz') and '_h' not in f)
    samp, jids, offs, hs, idxs = [], [], [0], [], []
    for f in parts:
        z = np.load(os.path.join(rd, f))
        layers = z['layers'].tolist()
        base = offs[-1]
        samp.append(z['samp'])
        o = z['offsets']
        offs += (o[1:] + base).tolist()
        jids += z['jids'].tolist()
        zh = np.load(os.path.join(rd, f.replace('.npz', '_h.npz')))
        hl = zh['layers'].tolist()
        hs.append(zh['h'])
        idxs.append(zh['idx'] + base)
    samp = np.concatenate(samp)
    H = np.concatenate(hs)          # (n, nhl, d) fp16
    idx = np.concatenate(idxs)
    offs = np.array(offs)
    job = np.searchsorted(offs, idx, side='right') - 1
    pid = np.array([jids[j].split('|')[0] for j in job])
    test = np.array([int(hashlib.md5(p.encode()).hexdigest(), 16) % 2 == 1 for p in pid])
    final = samp[idx, -1]
    col = {l: layers.index(l) for l in hl}
    # "safe" label: the layer's sampled token equals the final one at this and every later listed layer
    eq = samp[idx] == final[:, None]
    rev = np.logical_and.accumulate(eq[:, ::-1], axis=1)[:, ::-1]
    W, B, res = [], [], {}
    P = np.zeros((len(idx), len(hl)), np.float32)
    for j, l in enumerate(hl):
        y = rev[:, col[l]].astype(np.float32)
        X = H[:, j].astype(np.float32)
        w, b = fit_logreg(X[~test], y[~test], lam=a.lam)
        s = X @ w + b
        P[:, j] = 1 / (1 + np.exp(-s))
        res[l] = dict(pos_rate=float(y.mean()), auroc_test=auroc(y[test], s[test]))
        W.append(w)
        B.append(b)
        print(f'layer {l}: safe rate {y.mean():.3f}  test AUROC {res[l]["auroc_test"]:.3f}', flush=True)
    np.savez(os.path.join(GD, 'results', a.model, a.tag, 'exit_probe.npz'), W=np.stack(W), b=np.array(B),
             layers=np.array(hl))
    # offline exit rule on held-out problems
    rows = []
    for tau in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99):
        ex = np.full(len(idx), len(layers) - 1)
        exl = np.full(len(idx), L - 1)
        for j in reversed(range(len(hl))):
            hit = P[:, j] >= tau
            ex = np.where(hit, col[hl[j]], ex)
            exl = np.where(hit, hl[j], exl)
        agree = samp[idx, ex] == final
        rows.append(dict(tau=tau, blocks=float((exl[test] + 1).mean()), agree=float(agree[test].mean()),
                         exit_rate=float((exl[test] < L - 1).mean())))
        print(f'tau {tau}: mean blocks {rows[-1]["blocks"]:.2f}  agree {rows[-1]["agree"]:.4f}  exit rate {rows[-1]["exit_rate"]:.3f}')
    json.dump(dict(layers=res, rule=rows, n=int(len(idx)), n_test=int(test.sum())),
              open(os.path.join(rd, 'exit_probe_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
