"""Layer-wise lens-anchored linear probe with 5-fold cross-fitting (contract: INTERFACES.md 'Probe'; PLAN.md 2.2, 2.5).

Form, per layer l:  logits4 = lens(h) + hn @ W.T + b,  hn = h / sqrt(mean(h^2) + eps),  lens(h) = (hn * norm_w) @ U4.T
(the lens term is extract/lens4.npy: the same fp32 readout of the bf16 state). Rungs: lens (no params), bias (lens + b),
full (lens + W, b). Objective: mean CE + lam * ||W||^2 (b unpenalized); full-batch torch.optim.LBFGS (strong Wolfe), fp32.
Optimisation (exact reparametrisation, same minimiser): features are centred (the bias absorbs the mean) and W is
optimised as W = (V * s) @ Q.T, Q the eigenbasis of the feature covariance (eigenvalues ev), s = 1/sqrt(ev + 2 lam/S_BAR).
Centring and Q use features only (no labels). The K independent fits of one call (lam grid x label sets x outer folds)
form one block-separable objective (sum of the K losses); convergence is checked per block as max |gradient| w.r.t.
(W, centred b). Fits stop at max|grad| <= 1e-5 or when the strong-Wolfe search stalls at the fp32 floor of the objective.
Cross-fitting: outer fold f trains on fold != f and predicts fold == f. lam per (f, l): fit on folds not in
{f, g = (f+1) % 5} for every lam in LAMS, pick the lowest mean CE on fold g, refit on fold != f (cold start).
Nulls (seeds 0..2): answers permuted among questions within source (question level), row label (perm_answer - v) % 4;
same pipeline, lam re-selected on the null labels.
Outputs in results/{model}__{fmt}/probe/: oof_{lens,bias,full}.npy, oof_null{0,1,2}.npy (N, L, 4) f32,
weights_full.npz (W (5,L,4,d), b (5,L,4), lam (5,L)), weights_bias.npz (b (5,L,4)), null_labels.npy (3, N) int8,
lam_path.npz (inner-validation CE/acc (4 label sets, 5, L, G); lam_null (3, 5, L)), summary.json.
Usage: CUDA_VISIBLE_DEVICES=0 python scripts/train_probe.py --model qwen3-4b-base --fmt base_fewshot
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

LAMS = [10 ** (-k / 2) for k in range(11)]  # 1 .. 1e-5, strongest first (val-CE ties -> stronger)
NULL_SEEDS = (0, 1, 2)
S_BAR = 0.2  # typical softmax curvature; sets the preconditioner scale only
TIE = 1e-4
TOL_BAD = 1e-4  # max |grad| above this = non-converged (1e-5 .. 1e-4 is the fp32 noise floor of the objective)


def null_answers(answer, source, seed):
    """Permute canonical answers among questions within each source (question level)."""
    rng = np.random.default_rng(seed)
    out = answer.copy()
    for s in C.SOURCES:
        idx = np.flatnonzero(source == s)
        out[idx] = answer[rng.permutation(idx)]
    return out


def load_layer(hid, l, lens4, eps, dev):
    """Centred hn (N, d) fp32 on GPU, its mean, covariance eigenbasis, and the lens logits of layer l."""
    x = torch.from_numpy(np.ascontiguousarray(hid[:, l]).view(np.int16)).to(dev).view(torch.bfloat16).float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    mu = x.mean(0)
    x -= mu
    ev, Qb = torch.linalg.eigh(x.T @ x / len(x))
    return x, mu, ev.clamp(min=0), Qb, torch.from_numpy(lens4[:, l]).to(dev)


def block_ce(X, lens, W, b, Y, M):
    """Per-block mean CE over rows with M = 1 (K,), and logits (K, N, 4). W (K, 4, d) or None, b (K, 4) centred bias."""
    Z = lens + b[:, None]
    if W is not None:
        K = W.shape[0]
        Z = Z + (X @ W.reshape(4 * K, -1).T).view(-1, K, 4).transpose(0, 1)
    ce = torch.logsumexp(Z, -1) - Z.gather(-1, Y[..., None])[..., 0]
    return (ce * M).sum(1) / M.sum(1), Z


def kkt(X, lens, W, b, Y, M, lam):
    """Max |gradient| per block w.r.t. (W, centred b), the coordinates being optimised. (In the uncentred coordinates
    the W-gradient gains grad_b * mean(hn); the massive dims, mean ~30, inflate it at the fp32 floor of grad_b.)"""
    b = b.detach().requires_grad_()
    W = None if W is None else W.detach().requires_grad_()
    ce = block_ce(X, lens, W, b, Y, M)[0]
    (ce if W is None else ce + lam * W.pow(2).sum((1, 2))).sum().backward()
    g = b.grad.abs().amax(1)
    return g if W is None else torch.maximum(g, W.grad.abs().amax((1, 2)))


def fit(X, lens, Y, M, lam, pre, init, a):
    """Joint L-BFGS over K = len(Y) blocks; lam None -> bias rung (b only). pre = (Q, ev); init = (W, b) or None.
    Returns W (K, 4, d) or None, centred b (K, 4), info (iterations, per-block max |grad|)."""
    K, dev = Y.shape[0], Y.device
    b = (init[1].clone() if init else torch.zeros(K, 4, device=dev)).requires_grad_()
    params, V = [b], None
    if lam is not None:
        Qb, ev = pre
        s = torch.rsqrt(ev[None] + 2 * lam[:, None] / S_BAR)[:, None]  # (K, 1, d)
        V = ((init[0] @ Qb) / s if init else torch.zeros(K, 4, X.shape[1], device=dev)).requires_grad_()
        params.append(V)
    getW = (lambda: (V * s) @ Qb.T) if V is not None else (lambda: None)
    opt = torch.optim.LBFGS(params, lr=1, max_iter=a.chunk, max_eval=20 * a.chunk, tolerance_grad=1e-12,
                            tolerance_change=1e-12, history_size=a.history, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        W = getW()
        ce = block_ce(X, lens, W, b, Y, M)[0]
        loss = (ce if W is None else ce + lam * W.pow(2).sum((1, 2))).double().sum()  # fp64 sum over blocks
        loss.backward()
        return loss

    done = 0
    while True:
        opt.step(closure)
        n = opt.state[b]['n_iter']
        with torch.no_grad():
            W = getW()
        g = kkt(X, lens, W, b, Y, M, lam)
        if a.verbose:
            print(f'    K={K} it={n} gmax max {g.max():.1e} median {g.median():.1e}', flush=True)
        if g.max() <= a.tol or n >= a.max_iter or n - done < a.chunk:
            break
        done = n
    return W, b.detach(), dict(n_iter=int(n), gmax=g.cpu().numpy())


def metrics(oof, lab, qsrc=None):
    """Per-layer row accuracy, content-level 4/4 rate, tie count of oof (N, L, 4) against labels lab (N,)."""
    top2 = np.sort(oof, -1)[..., -2:]
    ok = oof.argmax(-1) == lab[:, None]  # (N, L)
    r = dict(acc=ok.mean(0), acc4c=ok.reshape(-1, 4, ok.shape[1]).all(1).mean(0),
             ties=(top2[..., 1] - top2[..., 0] < TIE).sum(0))
    if qsrc is not None:
        r['acc_by_source'] = {s: ok[qsrc.repeat(4) == s].mean(0) for s in C.SOURCES}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
    ap.add_argument('--out', default='probe'); ap.add_argument('--layers', default='', help='comma list (test only)')
    ap.add_argument('--max_iter', type=int, default=1000); ap.add_argument('--chunk', type=int, default=25)
    ap.add_argument('--tol', type=float, default=1e-5); ap.add_argument('--history', type=int, default=30)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = 'cuda'
    hid, meta = C.open_hidden(a.model, a.fmt)
    N, L, d = meta['N'], meta['L'], meta['d']
    ed = os.path.join(C.run_dir(a.model, a.fmt), 'extract')
    od = os.path.join(C.run_dir(a.model, a.fmt), a.out)
    os.makedirs(od, exist_ok=True)
    rows = json.load(open(os.path.join(ed, 'rows.json')))
    assert all(r['row'] == i and r['qid'] == i // 4 and r['v'] == i % 4 for i, r in enumerate(rows)) and N % 4 == 0
    fold = np.array([r['fold'] for r in rows]); gold = np.array([r['gold'] for r in rows]); v = np.arange(N) % 4
    qsrc = np.array([r['source'] for r in rows])[::4]
    answer = (gold + v)[::4] % 4
    assert ((answer.repeat(4) - v) % 4 == gold).all()
    nulls = np.stack([(null_answers(answer, qsrc, s).repeat(4) - v) % 4 for s in NULL_SEEDS])
    labels = np.concatenate([gold[None], nulls])  # (S, N): real, null0..2
    S, G = len(labels), len(LAMS)
    lens4 = np.load(os.path.join(ed, 'lens4.npy'))
    eps = float(np.load(os.path.join(ed, 'lens_params.npz'))['eps'])
    layers = [int(x) for x in a.layers.split(',')] if a.layers else list(range(L))
    print(f'{a.model}/{a.fmt}: N={N} L={L} d={d} layers={len(layers)} K_inner={S * 5 * G}', C.stamp(t0), flush=True)

    def retry(fn, *args, tries=240, wait=30):
        """GPUs are shared: on OOM free our cache, wait and retry."""
        for t in range(tries):
            try:
                return fn(*args)
            except torch.OutOfMemoryError:
                pass
            torch.cuda.empty_cache()
            print(f'  OOM (shared GPU), retry {t + 1}/{tries} in {wait}s', flush=True)
            time.sleep(wait)
        raise RuntimeError('persistent OOM')

    def setup():
        fo = torch.from_numpy(fold).to(dev)
        tr_in = torch.stack([(fo != f) & (fo != (f + 1) % 5) for f in range(5)]).float()  # inner train
        va_in = torch.stack([fo == (f + 1) % 5 for f in range(5)]).float()  # inner validation (never fold f)
        tr_out = torch.stack([fo != f for f in range(5)]).float()
        Yt = torch.from_numpy(labels).to(dev).long()
        return dict(Yt=Yt, tr_out=tr_out, lam_i=torch.tensor(LAMS, device=dev)[None, None].expand(S, 5, G).reshape(-1),
                    Yi=Yt[:, None, None].expand(S, 5, G, N).reshape(-1, N),  # inner blocks (s, f, j)
                    Mi=tr_in[None, :, None].expand(S, 5, G, N).reshape(-1, N),
                    Vi=va_in[None, :, None].expand(S, 5, G, N).reshape(-1, N),
                    Yf=Yt[:, None].expand(S, 5, N).reshape(-1, N),  # final blocks (s, f)
                    Mf=tr_out[None].expand(S, 5, N).reshape(-1, N))
    B = retry(setup)

    def layer_fits(l):
        """All fits of layer l (bias rung; inner lam path; final refit for real + nulls). Returns numpy results."""
        X, mu, ev, Qb, lens = load_layer(hid, l, lens4, eps, dev)
        _, bb, ib = fit(X, lens, B['Yt'][:1].expand(5, N), B['tr_out'], None, None, None, a)
        Wi, bi, ii = fit(X, lens, B['Yi'], B['Mi'], B['lam_i'], (Qb, ev), None, a)
        with torch.no_grad():
            ce, Zi = block_ce(X, lens, Wi, bi, B['Yi'], B['Vi'])
            acc = (((Zi.argmax(-1) == B['Yi']).float() * B['Vi']).sum(1) / B['Vi'].sum(1)).view(S, 5, G)
            del Zi
            ce = ce.view(S, 5, G)
        jstar = ce.argmin(-1)  # (S, 5)
        sel = torch.arange(S * 5, device=dev) * G + jstar.flatten()
        del Wi, bi
        Wf, bf, i_f = fit(X, lens, B['Yf'], B['Mf'], B['lam_i'][sel], (Qb, ev), None, a)  # cold start
        with torch.no_grad():
            Zf = block_ce(X, lens, Wf, bf, B['Yf'], B['Mf'])[1]
            return dict(Zf=Zf.cpu().numpy(), W=Wf[:5].cpu().numpy(), b=(bf - Wf @ mu)[:5].cpu().numpy(),
                        bb=bb.cpu().numpy(), js=jstar.cpu().numpy(), ce=ce.cpu().numpy(), acc=acc.cpu().numpy(),
                        ib=ib, ii=ii, i_f=i_f)

    oof = {k: np.zeros((N, L, 4), np.float32) for k in ['bias', 'full'] + [f'null{i}' for i in range(len(NULL_SEEDS))]}
    Wfull = np.zeros((5, L, 4, d), np.float32); bfull = np.zeros((5, L, 4), np.float32)
    lamfull = np.zeros((5, L), np.float32); bbias = np.zeros((5, L, 4), np.float32)
    lamnull = np.zeros((len(NULL_SEEDS), 5, L), np.float32)
    val_ce = np.full((S, 5, L, G), np.nan, np.float32); val_acc = np.full((S, 5, L, G), np.nan, np.float32)
    conv = []
    for l in layers:
        tl = time.time()
        r = retry(layer_fits, l)
        for f in range(5):
            te = fold == f
            oof['bias'][te, l] = lens4[te, l] + r['bb'][f]
            oof['full'][te, l] = r['Zf'][f, te]
            for s in range(len(NULL_SEEDS)):
                oof[f'null{s}'][te, l] = r['Zf'][(s + 1) * 5 + f, te]
        Wfull[:, l], bfull[:, l], bbias[:, l] = r['W'], r['b'], r['bb']
        lamfull[:, l] = np.array(LAMS)[r['js'][0]]; lamnull[:, :, l] = np.array(LAMS)[r['js'][1:]]
        val_ce[:, :, l], val_acc[:, :, l] = r['ce'], r['acc']
        ib, ii, i_f = r['ib'], r['ii'], r['i_f']
        conv.append(dict(layer=l, sec=time.time() - tl, iter_bias=ib['n_iter'], iter_inner=ii['n_iter'],
                         iter_final=i_f['n_iter'], gmax_bias=ib['gmax'].tolist(), gmax_inner=ii['gmax'].tolist(),
                         gmax_final=i_f['gmax'].tolist()))
        ok = lambda k, lab=gold: float((oof[k][:, l].argmax(-1) == lab).mean())
        print(f'L{l:02d} {time.time() - tl:5.1f}s it(b/in/fin)={ib["n_iter"]}/{ii["n_iter"]}/{i_f["n_iter"]} '
              f'gmax(in/fin)={ii["gmax"].max():.1e}/{i_f["gmax"].max():.1e} lam={lamfull[:, l].tolist()} '
              f'acc lens/bias/full={float((lens4[:, l].argmax(-1) == gold).mean()):.4f}/{ok("bias"):.4f}/{ok("full"):.4f} '
              f'null(own)={[round(ok(f"null{s}", nulls[s]), 4) for s in range(len(NULL_SEEDS))]}', C.stamp(t0), flush=True)

    # ---- outputs
    oof['lens'] = lens4
    for k in ('lens', 'bias', 'full') + tuple(f'null{s}' for s in range(len(NULL_SEEDS))):
        np.save(os.path.join(od, f'oof_{k}.npy'), oof[k])
    np.savez(os.path.join(od, 'weights_full.npz'), W=Wfull, b=bfull, lam=lamfull)
    np.savez(os.path.join(od, 'weights_bias.npz'), b=bbias)
    np.save(os.path.join(od, 'null_labels.npy'), nulls.astype(np.int8))
    np.savez(os.path.join(od, 'lam_path.npz'), lams=np.array(LAMS), val_ce=val_ce, val_acc=val_acc, lam_null=lamnull)

    # ---- check: stored fold weights reproduce the OOF logits from raw states (contract formula)
    rng = np.random.default_rng(0)
    rr = np.sort(rng.choice(N, 256, replace=False))
    lp = np.load(os.path.join(ed, 'lens_params.npz'))
    rep = 0.0
    for l in layers[:: max(1, len(layers) // 4)]:
        h = C.bf16_bits_to_f32(hid[rr, l]).astype(np.float64)
        hn = h / np.sqrt((h ** 2).mean(-1, keepdims=True) + eps)
        z = (hn * lp['norm_w']) @ lp['U4'].T.astype(np.float64)
        z += np.einsum('nd,ncd->nc', hn, Wfull[fold[rr], l]) + bfull[fold[rr], l]
        rep = max(rep, float(np.abs(z - oof['full'][rr, l]).max()))

    # ---- summary
    sel = np.array(layers)
    m = {k: metrics(oof[k][:, sel], gold, qsrc if k in ('lens', 'bias', 'full') else None) for k in oof}
    own = {f'null{s}': metrics(oof[f'null{s}'][:, sel], nulls[s]) for s in range(len(NULL_SEEDS))}
    tl_ = lambda x: np.asarray(x).round(5).tolist()
    gfin, ginn = np.array([c['gmax_final'] for c in conv]), np.array([c['gmax_inner'] for c in conv])
    nonconv = [dict(layer=c['layer'], stage=st, label_set=0 if st == 'bias' else k // 5, fold=k % 5, gmax=float(g))
               for c in conv for st in ('bias', 'final') for k, g in enumerate(c[f'gmax_{st}']) if g > TOL_BAD]
    acc = {k: m[k]['acc'] for k in ('lens', 'bias', 'full')}
    summary = dict(
        model=a.model, fmt=a.fmt, N=N, Q=N // 4, L=L, d=d, layers=layers,
        config=dict(lams=LAMS, lam_select='min mean CE on inner fold g=(f+1)%5, fits on folds not in {f,g}',
                    null_seeds=list(NULL_SEEDS), null_lam='re-selected on the null labels (same pipeline)',
                    null_perm='canonical answer permuted among questions within source; row label (perm-v)%4',
                    lbfgs=dict(line_search='strong_wolfe', history=a.history, chunk=a.chunk, max_iter=a.max_iter,
                               tol_gmax=a.tol, S_BAR=S_BAR, precond='centred features, covariance eigenbasis',
                               gmax='max |grad| w.r.t. (W, centred b)', final_init='cold'),
                    tie=TIE, layout='blocks (label set s: 0 real, 1..3 null; outer fold f; lam j)'),
        acc={k: tl_(m[k]['acc']) for k in m}, acc4c={k: tl_(m[k]['acc4c']) for k in m},
        ties={k: m[k]['ties'].tolist() for k in m},
        acc_null_own={k: tl_(own[k]['acc']) for k in own}, acc4c_null_own={k: tl_(own[k]['acc4c']) for k in own},
        acc_by_source={k: {s: tl_(x) for s, x in m[k]['acc_by_source'].items()} for k in ('lens', 'bias', 'full')},
        lam_full=lamfull[:, sel].tolist(), lam_null=lamnull[:, :, sel].tolist(),
        lam_boundary=dict(full_max=int((lamfull[:, sel] == LAMS[0]).sum()), full_min=int((lamfull[:, sel] == LAMS[-1]).sum()),
                          null_max=int((lamnull[:, :, sel] == LAMS[0]).sum()), null_min=int((lamnull[:, :, sel] == LAMS[-1]).sum())),
        convergence=dict(tol_stop=a.tol, tol_nonconverged=TOL_BAD, n_final_fits=gfin.size + 5 * len(conv),
                         n_inner_fits=ginn.size, final_above_tol_stop=int((gfin > a.tol).sum()),
                         nonconverged_final=nonconv, inner_above_tol_stop=int((ginn > a.tol).sum()),
                         nonconverged_inner=int((ginn > TOL_BAD).sum()), gmax_final_max=float(gfin.max()),
                         gmax_inner_max=float(ginn.max()),
                         per_layer=[{k: c[k] for k in ('layer', 'sec', 'iter_bias', 'iter_inner', 'iter_final')} |
                                    dict(gmax_inner=max(c['gmax_inner']), gmax_final=max(c['gmax_final'])) for c in conv]),
        sanity=dict(top_lens_acc=float((lens4[:, -1].argmax(-1) == gold).mean()), meta_acc4=meta['acc4'],
                    layers_full_ge_bias=int((acc['full'] >= acc['bias']).sum()),
                    layers_bias_ge_lens=int((acc['bias'] >= acc['lens']).sum()),
                    weights_reproduce_oof_maxabs=rep),
        seconds=time.time() - t0, script_md5=C.md5_file(__file__), common_md5=C.md5_file(C.__file__), args=vars(a))
    C.save_json(summary, os.path.join(od, 'summary.json'))
    print(json.dumps(dict(sanity=summary['sanity'], lam_boundary=summary['lam_boundary'],
                          convergence={k: v for k, v in summary['convergence'].items() if k != 'per_layer'})), flush=True)
    print('DONE', C.stamp(t0), flush=True)


if __name__ == '__main__':
    main()
