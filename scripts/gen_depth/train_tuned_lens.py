"""Low-rank tuned lens on replay residual dumps: is the full-depth token decodable earlier than the logit lens shows?

For each dumped layer l (final-normed residual z_l), learn z -> z + (z V) U^T + c (rank r, init identity) followed by
the frozen unembedding, trained with cross-entropy to the model's own final top-1 token (self-supervised, no gold),
on train problems (pid-hash split). On held-out problems, per layer: top-1 agreement with the final top-1 and
sampled-token agreement (same noise as the run) for the logit lens vs the tuned lens, overall and for digit tokens,
and the oracle exit depth (first dumped layer from which the readout's sampled token equals the final one through
the last dumped layer and the final layer).

Usage: python -I scripts/train_tuned_lens.py --model qwen3-4b --tag base [--rank 128 --epochs 3]
Saves results/<model>/<tag>/tuned_lens.pt and replay/tuned_lens_summary.json.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--tag', default='base')
    ap.add_argument('--rank', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1e-3)
    a = ap.parse_args()
    dev = 'cuda'
    rd = os.path.join(GD, 'results', a.model, a.tag)
    z = np.load(os.path.join(rd, 'replay', 'part0.npz'))
    zh = np.load(os.path.join(rd, 'replay', 'part0_h.npz'))
    layers = z['layers'].tolist()
    offs, jids = z['offsets'], z['jids'].tolist()
    idx = zh['idx']
    hl = zh['layers'].tolist()
    H = zh['h']                                  # (n, nhl, d) fp16, final-normed residuals
    job = np.searchsorted(offs, idx, side='right') - 1
    step = idx - offs[job]
    pid = np.array([jids[j].split('|')[0] for j in job])
    seed = np.array([int(jids[j].split('|')[1]) for j in job])
    test = np.array([int(hashlib.md5(p.encode()).hexdigest(), 16) % 2 == 1 for p in pid])
    y = z['top1'][idx, -1].astype(np.int64)      # final top-1
    ys = z['samp'][idx, -1].astype(np.int64)     # final sampled token
    lens_samp = z['samp'][idx]                    # (n, nl) logit-lens sampled tokens at replay layers
    from transformers import AutoTokenizer
    path = E.MODELS[a.model][0]
    tok = AutoTokenizer.from_pretrained(path)
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    W = m.lm_head.weight.detach().to(dev)        # (V, d) bf16
    del m
    sm = E.SAMPLING[E.MODELS[a.model][1]]
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
        j = torch.where(keep, tv + g, torch.full_like(tv, float('-inf'))).argmax(-1)
        return ti.gather(1, j[:, None]).squeeze(1)

    d = H.shape[2]
    tr = np.where(~test)[0]
    te = np.where(test)[0]
    yt = torch.tensor(y, device=dev)
    res, state = {}, {}
    tuned_samp = np.zeros((len(idx), len(hl)), np.int64)
    logit_samp = np.zeros((len(idx), len(hl)), np.int64)
    for j, l in enumerate(hl):
        X = torch.tensor(H[:, j], device=dev)    # fp16
        U = torch.zeros(d, a.rank, device=dev, requires_grad=True)
        Vm = (torch.randn(d, a.rank, device=dev) * 0.01).requires_grad_(True)
        c = torch.zeros(d, device=dev, requires_grad=True)
        opt = torch.optim.Adam([U, Vm, c], lr=a.lr)
        rng = np.random.default_rng(j)
        for ep in range(a.epochs):
            perm = rng.permutation(tr)
            for s in range(0, len(perm), a.bs):
                b = torch.tensor(perm[s:s + a.bs], device=dev)
                xb = X[b].float()
                zb = xb + (xb @ Vm) @ U.T + c
                loss = torch.nn.functional.cross_entropy((zb.to(W.dtype) @ W.T).float(), yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            for s in range(0, len(idx), 4096):
                b = torch.arange(s, min(len(idx), s + 4096), device=dev)
                xb = X[b].float()
                lt = ((xb + (xb @ Vm) @ U.T + c).to(W.dtype) @ W.T).float()
                ll = (xb.to(W.dtype) @ W.T).float()
                tuned_samp[s:s + len(b), j] = sample(lt, b).cpu().numpy()
                logit_samp[s:s + len(b), j] = sample(ll, b).cpu().numpy()
        state[l] = dict(U=U.detach().cpu(), V=Vm.detach().cpu(), c=c.detach().cpu())
        ag_t = tuned_samp[te, j] == ys[te]
        ag_l = logit_samp[te, j] == ys[te]
        res[l] = dict(samp_agree_logit=float(ag_l.mean()), samp_agree_tuned=float(ag_t.mean()),
                      digit_logit=float(ag_l[digit[te]].mean()), digit_tuned=float(ag_t[digit[te]].mean()),
                      final_loss=float(loss.item()))
        print(f'layer {l}: sampled-token agreement logit {ag_l.mean():.3f} tuned {ag_t.mean():.3f} | digits '
              f'{ag_l[digit[te]].mean():.3f} -> {ag_t[digit[te]].mean():.3f}', flush=True)
        del X
    # oracle exit depth on held-out positions (stable agreement from layer l through the last dumped layer)
    for name, S in (('logit', logit_samp), ('tuned', tuned_samp)):
        eq = S[te] == ys[te][:, None]
        rev = np.logical_and.accumulate(eq[:, ::-1], axis=1)[:, ::-1]
        first = np.where(rev.any(1), rev.argmax(1), len(hl))
        blocks = np.array([hl[f] + 1 if f < len(hl) else E.Engine.__init__.__defaults__ and 36 for f in first], float)
        L = 36 if a.model.startswith('qwen3') else 28
        blocks = np.where(first < len(hl), np.array(hl + [L - 1])[first] + 1, L)
        res[f'oracle_exit_blocks_{name}'] = float(blocks.mean())
        res[f'oracle_exit_blocks_{name}_digit'] = float(blocks[digit[te]].mean())
        print(f'{name} lens: oracle exit blocks {blocks.mean():.2f} (digits {blocks[digit[te]].mean():.2f})', flush=True)
    torch.save(state, os.path.join(rd, 'tuned_lens.pt'))
    json.dump(res, open(os.path.join(rd, 'replay', 'tuned_lens_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
