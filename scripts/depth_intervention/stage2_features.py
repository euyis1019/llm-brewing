"""Stage-2 preview, part 1: per row x layer label-free features (L0/L1) and probe features (L2). CPU only.

Inputs : RUN/extract/* (required); RUN/probe/oof_full.npy (optional: gold-trained probe -> L2 features).
Output : RUN/stage2/features.npz and features.json (config, supervision tags, tie counts, seconds, md5).
Row arrays (N, L) f32, row = 4*qid + v:
  {lens,csd,probe}_{maxp,margin,ent,kl,run,agree}: 4-way max softmax / top1-top2 logit / entropy / KL(p_l || p_{l-1})
      (0 at l=0) / length of the top-1 run ending at l / top-1 == real-output top-1 (agreement with the model's own
      final output, L1 for lens/csd)
  csd_mass: CSD probability mass on the 4 letters (full vocabulary)
  avail_{lens,probe}: 'available-but-not-emitted' candidate (a) = readout max softmax - CSD prob of the readout's class
      (probe version = L2 as specified; lens version = its L0 analogue)
  h_norm (RMS of h_l), h_cos (cos(h_l, h_{l-1})), h_upd (|h_l - h_{l-1}| / |h_{l-1}|), h_u4 (energy fraction of hn in
      span(w*U4), the subspace through which hn moves the lens letter logits), h_u4d (same for the 3-d letter-difference
      subspace), h_updu4 (fraction of the update h_l - h_{l-1} lying in span(w*U4)); update-based ones are NaN at l=0
Question arrays (Q, L); version v maps letter i to content (i + v) % 4:
  xv_{R}_n    size of the modal-content set (1..4) of readout R's top-1 over the 4 versions
  xv_{R}_fin  layer-l modal content == modal content of the real output
  xv_{R}_pmax max of the version-averaged content distribution
  xv_{R}_gap  n_l - n_out: candidate (b), cross-version consistency at l relative to the final output
  xv_{R}_lost n_l - (number of real outputs on the layer-l modal content); > 0 = a consensus the output does not keep
  Modal ties are broken by the version-averaged content probability (never by gold).
Untagged: modal_{R} (Q, L) layer-l modal content; out_votes (Q, 4) final votes per content; out_n, out_modal (Q,);
  row vectors qid, v, fold, src (index into common.SOURCES). No gold is stored.
Usage: python scripts/stage2_features.py --model qwen3-4b-base --fmt base_fewshot
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from scipy.special import logsumexp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

IDX = (np.arange(4)[None, :] - np.arange(4)[:, None]) % 4  # IDX[v, c] = letter showing content c in version v
TIE = 1e-4


def readout(z, out_top1):
    """z (N, L, 4) letter logits -> ((N, L) feature dict, top1 (N, L), probs (N, L, 4), tie mask)."""
    z = z.astype(np.float64)
    lp = z - logsumexp(z, -1, keepdims=True)
    p = np.exp(lp)
    s = np.sort(z, -1)
    top1 = z.argmax(-1)
    run = np.ones(top1.shape)
    for l in range(1, top1.shape[1]):
        run[:, l] = np.where(top1[:, l] == top1[:, l - 1], run[:, l - 1] + 1, 1)
    kl = np.zeros(top1.shape)
    kl[:, 1:] = (p[:, 1:] * (lp[:, 1:] - lp[:, :-1])).sum(-1)
    f = dict(maxp=p.max(-1), margin=s[..., -1] - s[..., -2], ent=-(p * lp).sum(-1), kl=kl, run=run,
             agree=top1 == out_top1[:, None])
    return {k: v.astype(np.float32) for k, v in f.items()}, top1, p, f['margin'] < TIE


def votes_of(top1, p):
    """top1 (N, L), p (N, L, 4) -> content votes (Q, L, 4), modal content (Q, L), version-mean content probs (Q, L, 4)."""
    Q, L = top1.shape[0] // 4, top1.shape[1]
    c = (top1.reshape(Q, 4, L) + np.arange(4)[None, :, None]) % 4
    votes = (c[..., None] == np.arange(4)).sum(1)
    P = p.reshape(Q, 4, L, 4)
    pbar = np.stack([P[:, v][..., IDX[v]] for v in range(4)], 1).mean(1)
    return votes, (votes + 0.5 * pbar).argmax(-1), pbar


def cross_version(top1, p, out_votes, out_modal):
    votes, modal, pbar = votes_of(top1, p)
    n = votes.max(-1)
    kept = np.take_along_axis(np.broadcast_to(out_votes[:, None, :], votes.shape), modal[..., None], -1)[..., 0]
    f = dict(n=n, fin=modal == out_modal[:, None], pmax=pbar.max(-1), gap=n - out_votes.max(-1)[:, None], lost=n - kept)
    return {k: v.astype(np.float32) for k, v in f.items()}, modal.astype(np.int8)


def hidden_geometry(model, fmt, lp, chunk=256):
    hid, meta = C.open_hidden(model, fmt)
    N, L, d = hid.shape
    E = torch.from_numpy(lp['U4'] * lp['norm_w'][None]).double()  # effective letter rows: lens = hn @ E.T
    Q4 = torch.linalg.qr(E.T)[0].float()                          # (d, 4) basis of span(w*U4)
    Qd = torch.linalg.qr((E[1:] - E[:1]).T)[0].float()            # (d, 3) letter-difference subspace
    names = ('h_norm', 'h_cos', 'h_upd', 'h_u4', 'h_u4d', 'h_updu4')
    out = {k: np.full((N, L), np.nan, np.float32) for k in names}
    for i in range(0, N, chunk):
        H = torch.from_numpy(np.array(hid[i:i + chunk]).view(np.int16)).view(torch.bfloat16).float()
        n2 = H.pow(2).sum(-1); nr = n2.sqrt()
        D = H[:, 1:] - H[:, :-1]; dn2 = D.pow(2).sum(-1)
        sl = slice(i, i + chunk)
        out['h_norm'][sl] = nr / d ** 0.5
        out['h_cos'][sl, 1:] = (H[:, 1:] * H[:, :-1]).sum(-1) / (nr[:, 1:] * nr[:, :-1])
        out['h_upd'][sl, 1:] = dn2.sqrt() / nr[:, :-1]
        out['h_u4'][sl] = (H @ Q4).pow(2).sum(-1) / n2
        out['h_u4d'][sl] = (H @ Qd).pow(2).sum(-1) / n2
        out['h_updu4'][sl, 1:] = (D @ Q4).pow(2).sum(-1) / dn2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
    ap.add_argument('--no_probe', action='store_true'); ap.add_argument('--threads', type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(a.threads)
    run = C.run_dir(a.model, a.fmt)
    ex = os.path.join(run, 'extract')
    od = os.path.join(run, 'stage2'); os.makedirs(od, exist_ok=True)
    rows = json.load(open(os.path.join(ex, 'rows.json')))
    N = len(rows); Q = N // 4
    assert all(r['row'] == i and r['qid'] == i // 4 and r['v'] == i % 4 for i, r in enumerate(rows))
    lp = dict(np.load(os.path.join(ex, 'lens_params.npz')))
    out4 = np.load(os.path.join(ex, 'out_logits4.npy'))
    F, tags, ties = {}, {}, {}

    o, out_top1, out_p, out_tie = readout(out4[:, None], out4.argmax(-1))
    out_votes, out_modal, _ = votes_of(out_top1, out_p)
    out_votes, out_modal = out_votes[:, 0], out_modal[:, 0]
    out_top1 = out_top1[:, 0]
    ties['out'] = int(out_tie.sum())

    readouts = {'lens': np.load(os.path.join(ex, 'lens4.npy')), 'csd': np.load(os.path.join(ex, 'csd_raw4.npy'))}
    probe_path = os.path.join(run, 'probe', 'oof_full.npy')
    if os.path.exists(probe_path) and not a.no_probe:
        readouts['probe'] = np.load(probe_path)
    P = {}
    for name, z in readouts.items():
        lvl = 'L2' if name == 'probe' else 'L0'
        f, top1, p, tie = readout(z, out_top1)
        ties[name] = int(tie.sum())
        P[name] = p
        for k, v in f.items():
            F[f'{name}_{k}'] = v
            tags[f'{name}_{k}'] = ('L1' if lvl == 'L0' and k == 'agree' else lvl)
        xv, F[f'modal_{name}'] = cross_version(top1, p, out_votes, out_modal)
        for k, v in xv.items():
            F[f'xv_{name}_{k}'] = v
            tags[f'xv_{name}_{k}'] = 'L2' if name == 'probe' else 'L1'
        print(f'{name}: done', C.stamp(t0), flush=True)
    F['csd_mass'] = np.load(os.path.join(ex, 'csd_mass.npy')).astype(np.float32); tags['csd_mass'] = 'L0'
    for name in [r for r in readouts if r != 'csd']:
        cls = P[name].argmax(-1)[..., None]
        F[f'avail_{name}'] = (P[name].max(-1) - np.take_along_axis(P['csd'], cls, -1)[..., 0]).astype(np.float32)
        tags[f'avail_{name}'] = 'L2' if name == 'probe' else 'L0'
    for k, v in hidden_geometry(a.model, a.fmt, lp).items():
        F[k] = v; tags[k] = 'L0'
    print('hidden geometry: done', C.stamp(t0), flush=True)

    F.update(out_n=out_votes.max(-1).astype(np.int8), out_modal=out_modal.astype(np.int8), out_votes=out_votes.astype(np.int8),
             qid=np.array([r['qid'] for r in rows], np.int32), v=np.array([r['v'] for r in rows], np.int8),
             fold=np.array([r['fold'] for r in rows], np.int8),
             src=np.array([C.SOURCES.index(r['source']) for r in rows], np.int8))
    np.savez(os.path.join(od, 'features.npz'), **F)
    L = readouts['lens'].shape[1]
    meta = dict(model=a.model, fmt=a.fmt, N=N, Q=Q, L=L, has_probe='probe' in readouts, tags=tags, ties_lt_1e4=ties,
                csd_final_agree=float(F['csd_agree'][:, -1].mean()), lens_final_agree=float(F['lens_agree'][:, -1].mean()),
                out_n_hist=np.bincount(F['out_n'], minlength=5).tolist(), seconds=time.time() - t0,
                script_md5=C.md5_file(__file__), common_md5=C.md5_file(C.__file__), args=vars(a))
    C.save_json(meta, os.path.join(od, 'features.json'))
    print(json.dumps({k: meta[k] for k in ('N', 'Q', 'L', 'has_probe', 'ties_lt_1e4', 'csd_final_agree',
                                           'lens_final_agree', 'out_n_hist', 'seconds')}), flush=True)
    print('DONE', C.stamp(t0), flush=True)


if __name__ == '__main__':
    main()
