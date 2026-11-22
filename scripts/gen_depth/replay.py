"""Teacher-forced replay of generated trajectories: per response token, per layer logit-lens statistics.

For each job in a gen.jsonl (prompt rebuilt from the pool), runs the full sequence once and, for every response
position i (the state that produced response token i) and every layer l in --layers, records:
  top1   lens argmax                        p1    lens top-1 prob (T=1)          ent   lens entropy (nats)
  samp   token the engine's sampler would pick from this layer's lens with the run's noise (key, step=i)
  lpf    lens log-prob of the final layer's top-1 token
  kl     KL(p_final || p_l)  (how far layer l still is from the final distribution)
  cand   whether the final top-1 wins among the final top-16 candidates under layer l's lens (engine 'commit' basis)
  klp    KL(p_l || p_prev) between this and the previous listed layer (0 for the first): the "emission jump" signal
  cos    cosine(h_l, h_{l-1}) of the residual stream (state saturation, free)   upd  |h_l - h_{l-1}| / |h_l|
With --h_every k, also saves the final-normed residual at --h_layers for every k-th response position (fp16) to
part{shard}_h.npz (inputs for self-supervised exit probes).
Final-layer quantities are included as layer L-1.
Output: results/<model>/<tag>/replay/part{shard}.npz with flat arrays over (job, position) and offsets + jids.

Usage: python -I scripts/replay.py --model qwen3-4b --tag base [--layers 12-35] [--max_jobs N] [--shard 0/2]
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(GD, 'pylib'))

import torch  # noqa: E402

import engine as E  # noqa: E402
import policies as P  # noqa: E402
from run_gen import build_prompt  # noqa: E402


@torch.no_grad()
def lens_stats(eng, H, Hprev, final_logits, keys, steps, chunk=512, n_cand=16):
    """H (nl, T, d) residuals at the chosen layers for T response states. Returns dict of (T, nl) arrays."""
    nl, T, _ = H.shape
    out = {k: np.zeros((T, nl), dt) for k, dt in
           [('top1', np.int32), ('samp', np.int32), ('p1', np.float16), ('ent', np.float16), ('lpf', np.float16),
            ('kl', np.float16), ('klp', np.float16), ('cand', np.bool_), ('cos', np.float16), ('upd', np.float16)]}
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        fl = final_logits[s:e]
        flp = torch.log_softmax(fl, -1)
        ftop = fl.argmax(-1)
        cand = fl.topk(n_cand, -1).indices
        Wc = eng.W[cand]
        prev = None
        for j in range(nl):
            hj, hp = H[j, s:e].float(), Hprev[j, s:e].float()
            out['cos'][s:e, j] = torch.nn.functional.cosine_similarity(hj, hp, dim=-1).cpu().numpy()
            out['upd'][s:e, j] = ((hj - hp).norm(dim=-1) / hj.norm(dim=-1)).cpu().numpy()
            z = eng.norm(H[j, s:e])
            lg = eng.lm_head(z).float()
            lp = torch.log_softmax(lg, -1)
            p = lp.exp()
            p1, t1 = p.max(-1)
            out['top1'][s:e, j] = t1.cpu().numpy()
            out['p1'][s:e, j] = p1.cpu().numpy()
            out['ent'][s:e, j] = (-(p * lp).sum(-1)).cpu().numpy()
            out['lpf'][s:e, j] = lp.gather(1, ftop[:, None])[:, 0].cpu().numpy()
            out['kl'][s:e, j] = (flp.exp() * (flp - lp)).sum(-1).clamp_min(0).cpu().numpy()
            out['samp'][s:e, j] = eng.sample(lg, keys[s:e], steps[s:e]).cpu().numpy()
            out['cand'][s:e, j] = (torch.einsum('nd,ncd->nc', z, Wc).argmax(-1) == 0).cpu().numpy()
            if prev is not None:
                out['klp'][s:e, j] = (p * (lp - prev)).sum(-1).clamp_min(0).cpu().numpy()
            prev = lp
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument('--tag', default='base')
    ap.add_argument('--pool', default=os.path.join(GD, 'data/pool.jsonl'))
    ap.add_argument('--layers', default='')
    ap.add_argument('--max_jobs', type=int, default=0)
    ap.add_argument('--seeds', default='')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--h_every', type=int, default=0)
    ap.add_argument('--h_layers', default='16-34/2')
    a = ap.parse_args()
    t0 = time.time()
    rd = os.path.join(GD, 'results', a.model, a.tag)
    recs = []
    for fn in sorted(os.listdir(rd)):
        if fn.startswith('gen') and fn.endswith('.jsonl'):
            recs += [json.loads(l) for l in open(os.path.join(rd, fn))]
    if a.seeds:
        keep = {int(s) for s in a.seeds.split(',')}
        recs = [r for r in recs if r['seed'] in keep]
    recs.sort(key=lambda r: r['jid'])
    si, sn = map(int, a.shard.split('/'))
    recs = recs[si::sn]
    if a.max_jobs:
        recs = recs[:a.max_jobs]
    pool = {}
    for l in open(a.pool):
        it = json.loads(l)
        pool[it['pid']] = it
    eng = E.Engine(a.model, R=1, Tmax=16)
    layers = P.parse_layers(a.layers) if a.layers else list(range(eng.L // 3, eng.L))
    if eng.L - 1 not in layers:
        layers.append(eng.L - 1)
    acc = {k: [] for k in ('top1', 'samp', 'p1', 'ent', 'lpf', 'kl', 'klp', 'cand', 'cos', 'upd')}
    hl = P.parse_layers(a.h_layers) if a.h_every else []
    hbuf, hpos = [], []
    offs, jids, n_bad = [0], [], 0
    for n, r in enumerate(recs):
        prompt = build_prompt(pool[r['pid']], eng.tok, eng.fam)
        ids = prompt + r['tokens']
        Hall = eng.forward_all(ids)                                   # (L, T, d)
        st = len(prompt) - 1                                          # state producing response token 0
        Tr = len(r['tokens'])
        H = Hall[layers, st:st + Tr]
        Hprev = Hall[[l - 1 for l in layers], st:st + Tr]
        final = eng.readout(Hall[-1, st:st + Tr])
        if hl:
            sel = torch.arange(0, Tr, a.h_every, device=eng.dev)
            hbuf.append(eng.norm(Hall[hl][:, st + sel]).transpose(0, 1).half().cpu().numpy())
            hpos.append(offs[-1] + sel.cpu().numpy())
        keys = torch.full((Tr,), E.job_key(r['pid'], r['seed']), dtype=torch.long, device=eng.dev)
        steps = torch.arange(Tr, device=eng.dev)
        o = lens_stats(eng, H, Hprev, final, keys, steps)
        n_bad += int((o['samp'][:, -1] != np.array(r['tokens'])).sum())
        for k in acc:
            acc[k].append(o[k])
        offs.append(offs[-1] + Tr)
        jids.append(r['jid'])
        del Hall, H, Hprev
        if n % 100 == 0:
            print(f'{n}/{len(recs)} tokens {offs[-1]} final-sample mismatch {n_bad} ({time.time() - t0:.0f}s)', flush=True)
    od = os.path.join(rd, 'replay')
    os.makedirs(od, exist_ok=True)
    np.savez(os.path.join(od, f'part{si}.npz'), layers=np.array(layers), offsets=np.array(offs), jids=np.array(jids),
             **{k: np.concatenate(v) for k, v in acc.items()})
    if hl:
        np.savez(os.path.join(od, f'part{si}_h.npz'), h=np.concatenate(hbuf), idx=np.concatenate(hpos), layers=np.array(hl))
    print(f'DONE {len(recs)} jobs {offs[-1]} tokens; replayed final sample != generated token: {n_bad} '
          f'({n_bad / max(1, offs[-1]):.4%}) in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
