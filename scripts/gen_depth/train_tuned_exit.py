"""Exit probes for the tuned lens: from norm(h_l), predict whether the tuned-lens readout at l already emits the
full-depth token (same noise). Self-supervised (labels from the model's own outputs), trained on train problems.

Offline rule on held-out problems: exit at the first probe layer whose probability >= tau, emit the tuned-lens token.
Reports mean executed blocks, token agreement (overall / digits), exit rate, for a tau grid. Saves
results/<model>/<tag>/tuned_exit.npz with probe W (nl, d), b (nl,), layers, and tuned-lens U (nl, d, r), V, c.

Usage: python -I scripts/train_tuned_exit.py --model qwen3-4b --tag base
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
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import engine as E  # noqa: E402
from train_exit_probe import auroc, fit_logreg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--tag', default='base')
    ap.add_argument('--lam', type=float, default=1e-3)
    a = ap.parse_args()
    dev = 'cuda'
    rd = os.path.join(GD, 'results', a.model, a.tag)
    z = np.load(os.path.join(rd, 'replay', 'part0.npz'))
    zh = np.load(os.path.join(rd, 'replay', 'part0_h.npz'))
    offs, jids = z['offsets'], z['jids'].tolist()
    idx, hl, H = zh['idx'], zh['layers'].tolist(), zh['h']
    job = np.searchsorted(offs, idx, side='right') - 1
    step = idx - offs[job]
    pid = np.array([jids[j].split('|')[0] for j in job])
    seed = np.array([int(jids[j].split('|')[1]) for j in job])
    test = np.array([int(hashlib.md5(p.encode()).hexdigest(), 16) % 2 == 1 for p in pid])
    ys = z['samp'][idx, -1].astype(np.int64)
    st = torch.load(os.path.join(rd, 'tuned_lens.pt'))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path, fam = E.MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    W = m.lm_head.weight.detach().to(dev)
    L = m.config.num_hidden_layers
    del m
    sm = E.SAMPLING[fam]
    digit = np.array([tok.decode([int(t)]).strip().isdigit() for t in ys])
    keys = torch.tensor([E.job_key(p, s) for p, s in zip(pid, seed)], dtype=torch.long, device=dev)
    steps = torch.tensor(step, dtype=torch.long, device=dev)

    def sample(lg, sel):
        lg = lg / sm['temperature']
        tv, ti = lg.topk(sm['top_k'], dim=-1)
        p = torch.softmax(tv, -1)
        keep = (p.cumsum(-1) - p) < sm['top_p']
        u = E.hash_uniform(keys[sel], steps[sel], ti)
        g = (-torch.log(-torch.log(u))).float()
        return ti.gather(1, torch.where(keep, tv + g, torch.full_like(tv, float('-inf'))).argmax(-1)[:, None]).squeeze(1)

    TS = np.zeros((len(idx), len(hl)), np.int64)
    with torch.no_grad():
        for j, l in enumerate(hl):
            U, V, c = (st[l][k].to(dev) for k in ('U', 'V', 'c'))
            X = torch.tensor(H[:, j], device=dev)
            for s in range(0, len(idx), 4096):
                b = torch.arange(s, min(len(idx), s + 4096), device=dev)
                xb = X[b].float()
                TS[s:s + len(b), j] = sample(((xb + (xb @ V) @ U.T + c).to(W.dtype) @ W.T).float(), b).cpu().numpy()
    ok = TS == ys[:, None]
    Wp, Bp, res = [], [], {}
    P = np.zeros((len(idx), len(hl)), np.float32)
    for j, l in enumerate(hl):
        y = ok[:, j].astype(np.float32)
        X = H[:, j].astype(np.float32)
        w, b = fit_logreg(X[~test], y[~test], lam=a.lam)
        s = X @ w + b
        P[:, j] = 1 / (1 + np.exp(-s))
        res[l] = dict(agree_rate=float(y[test].mean()), auroc_test=auroc(y[test], s[test]))
        Wp.append(w)
        Bp.append(b)
        print(f'layer {l}: tuned agree {y[test].mean():.3f}  probe AUROC {res[l]["auroc_test"]:.3f}', flush=True)
    rows = []
    for tau in (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99):
        exl = np.full(len(idx), L - 1)
        agree = np.ones(len(idx), bool)
        done = np.zeros(len(idx), bool)
        for j, l in enumerate(hl):
            hit = (P[:, j] >= tau) & ~done
            exl = np.where(hit, l, exl)
            agree = np.where(hit, ok[:, j], agree)
            done |= hit
        t = test
        rows.append(dict(tau=tau, blocks=float((exl[t] + 1).mean()), agree=float(agree[t].mean()),
                         agree_digit=float(agree[t & digit].mean()), exit_rate=float((exl[t] < L - 1).mean())))
        r = rows[-1]
        print(f"tau {tau}: blocks {r['blocks']:.2f} agree {r['agree']:.4f} digits {r['agree_digit']:.4f} exit {r['exit_rate']:.3f}")
    np.savez(os.path.join(rd, 'tuned_exit.npz'), W=np.stack(Wp), b=np.array(Bp), layers=np.array(hl),
             U=np.stack([st[l]['U'].numpy() for l in hl]), V=np.stack([st[l]['V'].numpy() for l in hl]),
             c=np.stack([st[l]['c'].numpy() for l in hl]))
    json.dump(dict(layers=res, rule=rows), open(os.path.join(rd, 'replay', 'tuned_exit_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
