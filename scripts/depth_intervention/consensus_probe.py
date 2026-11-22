"""Consensus probe: a single-pass, label-free (L1) exit head trained on the model's own 4-version consensus.

Training (GPU; gold is never read). L1 targets per question, from the model's own outputs:
  vote     modal content of the 4 versions' real final outputs (ties -> largest summed 4-way softmax over versions)
  maxcons  modal content of the CSD readout at the layer l >= w0 = round(0.45 L) where its 4-version consistency (size
           of the modal content) is maximal; ties -> latest layer, content ties -> summed CSD softmax at that layer
  Row target letter = (c* - v) % 4. Readouts are tie-aware (top-2 margin < 1e-4 = no vote).
Head = train_probe.py's lens-anchored probe (logits = lens(h) + hn @ W.T + b per layer; mean CE + lam ||W||^2; L-BFGS;
its load_layer / fit / block_ce are imported), 5-fold cross-fitted by 'fold' (fold-f logits from a fit on fold != f).
lam per (f, layer) = min CE on the L1 targets of inner fold g = (f+1) % 4 (never f, never the lockbox fold 4) after fits
on folds not in {f, g}. A row's logits depend on that row's own hidden state only (single pass); the 4 versions enter
only through the training targets, and a question's 4 rows share one fold (never trained on when evaluated).
Outputs RUN/consensus_probe/: oof_cons_{t}.npy (N, L, 4), weights_cons_{t}.npz (W (5,L,4,d), b (5,L,4), lam (5,L)),
targets.npz, train_summary.json.

Evaluation (--eval, CPU; gold used only to score). Levels: L0 lens / CSD / real output; L1 consensus head and 4-pass
consensus policies; L2-L3 gold probe (probe/oof_full.npy). Lockbox = fold 4: rule parameters for fold f in 0..3 are
chosen on folds {0..3} minus f, for fold 4 on folds 0..3; fold-4 numbers are computed only with --open_lockbox.
  curves    per-layer row accuracy and content-level 4/4 (vs gold), agreement with the L1 targets, 4-version
            self-consistency of each head; bootstrap CIs of consensus - probe / - lens at the report layers
  exit      early_exit.py rules (H[:, L-1] = real output): fix, stab(k, tau), conf; E-cons parameters chosen by agreement
            with the head's own L1 target, E-head (gold probe, reference) by gold accuracy; 'diag' cross conditions (L1 head
            with gold-selected parameters, gold probe with L1-selected ones; not label-free); fixed-layer sweep (descriptive);
            NOEXIT = head's own L-1
            readout; 4-pass L1 references vote_final, cons_exit (= maxcons target), cons_gate (RESULTS 3). Metrics as
            early_exit.py: O->R, matched decoy, decoy-O pairs, net single-version accuracy, R harm, bootstrap CIs
  official  v1 K=3 classes recomputed with classify_v1 under --last_layer (default 'output' = contract since 2026-10-16
            evening: the probe readout at L-1 is the real output), cross-checked against RUN/classify/questions.csv
  taxonomy  label-free v1 = classify_v1 with the consensus head as probe and the L1 target as 'answer' (K=3; layer
            L-1 convention = --last_layer) vs the
            official v1 K=3: crosstab, kappa (R,M merged: no label can split them), O recall / precision, share of
            O~ whose target is the gold content; plus v1 with the consensus head and the gold answer
Outputs RUN/consensus_probe/eval_f03.json (lockbox closed) or eval_final.json (--open_lockbox).
Usage: CUDA_VISIBLE_DEVICES=0 python scripts/consensus_probe.py --model qwen3-4b-base --fmt base_fewshot [--target vote]
       python scripts/consensus_probe.py --model qwen3-4b-base --fmt base_fewshot --eval [--open_lockbox]
"""
import argparse, csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from classify import top1, softmax, to_q, contents, classify_v1, load_inputs, boot_weights, wci, ratio, ci
from early_exit import KGRID, TGRID, stab_runs, first_fire, strat_rate, decoy_fn

TARGETS, LOCK, B_BOOT = ('vote', 'maxcons'), 4, 2000
REPORT = (16, 20, 21, 22, 23, 24, 28)  # plus L-1


# ----------------------------------------------------------------------------- L1 targets (no gold)
def content_probs(z):
    """Row logits (N, 4) -> 4-way softmax summed over the 4 versions in content space (Q, 4)."""
    p = to_q(softmax(z.astype(np.float64)))
    return sum(p[:, v][:, (np.arange(4) - v) % 4] for v in range(4))


def modal(cont, probs):
    """cont (Q, 4) contents (-1 = tie), probs (Q, 4) summed content probs (< 5) -> modal content, its count."""
    cnt = (cont[..., None] == np.arange(4)).sum(1)
    return (cnt + probs / 5).argmax(-1), cnt.max(-1)


def l1_targets(out4, csd, w0):
    q = np.arange(len(out4) // 4)
    vote, fcons = modal(contents(to_q(top1(out4))), content_probs(out4))
    cl = contents(to_q(top1(csd)))  # (Q, 4, L)
    cons = (cl[..., None] == np.arange(4)).sum(1).max(-1)  # (Q, L) 4-version consistency
    lay = cons.shape[1] - 1 - cons[:, w0:][:, ::-1].argmax(1)  # latest layer >= w0 with maximal consistency
    mc, mcount = modal(cl[q, :, lay], content_probs(to_q(csd)[q, :, lay].reshape(-1, 4)))
    return dict(vote=vote, maxcons=mc, maxcons_layer=lay, maxcons_cons=mcount, final_cons=fcons, csd_cons=cons)


def run_paths(model, fmt, out):
    rd = C.run_dir(model, fmt)
    return rd, os.path.join(rd, 'extract'), os.path.join(rd, out)


# ----------------------------------------------------------------------------- training (GPU)
def train(a):
    import torch
    import train_probe as TP
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = 'cuda'
    hid, meta = C.open_hidden(a.model, a.fmt)
    N, L, d = meta['N'], meta['L'], meta['d']
    rd, ed, od = run_paths(a.model, a.fmt, a.out)
    os.makedirs(od, exist_ok=True)
    rows = json.load(open(os.path.join(ed, 'rows.json')))
    assert all(r['row'] == i and r['qid'] == i // 4 and r['v'] == i % 4 for i, r in enumerate(rows))
    fold, v = np.array([r['fold'] for r in rows]), np.arange(N) % 4  # gold is deliberately not read
    assert (to_q(fold) == to_q(fold)[:, :1]).all(), 'a question must lie in one fold'
    w0 = int(round(0.45 * L))
    T = l1_targets(np.load(os.path.join(ed, 'out_logits4.npy')), np.load(os.path.join(ed, 'csd_raw4.npy')), w0)
    tg = a.target.split(',')
    labels = np.stack([(np.repeat(T[t], 4) - v) % 4 for t in tg])  # (S, N) row target letters
    S, G = len(tg), len(TP.LAMS)
    lens4 = np.load(os.path.join(ed, 'lens4.npy'))
    eps = float(np.load(os.path.join(ed, 'lens_params.npz'))['eps'])
    fa = argparse.Namespace(chunk=25, history=30, max_iter=1000, tol=1e-5, verbose=False)
    gf = [(f + 1) % 4 for f in range(5)]
    fo, Yt = torch.from_numpy(fold).to(dev), torch.from_numpy(labels).to(dev).long()
    tr_in = torch.stack([(fo != f) & (fo != gf[f]) for f in range(5)]).float()
    va_in = torch.stack([fo == gf[f] for f in range(5)]).float()
    tr_out = torch.stack([fo != f for f in range(5)]).float()
    lam_i = torch.tensor(TP.LAMS, device=dev)[None, None].expand(S, 5, G).reshape(-1)  # blocks (s, f, j)
    Yi, Mi, Vi = (Yt[:, None, None].expand(S, 5, G, N).reshape(-1, N), tr_in[None, :, None].expand(S, 5, G, N).reshape(-1, N),
                  va_in[None, :, None].expand(S, 5, G, N).reshape(-1, N))
    Yf, Mf = Yt[:, None].expand(S, 5, N).reshape(-1, N), tr_out[None].expand(S, 5, N).reshape(-1, N)  # blocks (s, f)
    print(f'{a.model}__{a.fmt} N={N} L={L} d={d} targets={tg} inner folds {gf}', C.stamp(t0), flush=True)

    def layer_fits(l):
        X, mu, ev, Qb, lens = TP.load_layer(hid, l, lens4, eps, dev)
        Wi, bi, ii = TP.fit(X, lens, Yi, Mi, lam_i, (Qb, ev), None, fa)
        with torch.no_grad():
            ce = TP.block_ce(X, lens, Wi, bi, Yi, Vi)[0].view(S, 5, G)
        js = ce.argmin(-1)
        del Wi, bi
        Wf, bf, i_f = TP.fit(X, lens, Yf, Mf, lam_i[torch.arange(S * 5, device=dev) * G + js.flatten()], (Qb, ev), None, fa)
        with torch.no_grad():
            Zf = TP.block_ce(X, lens, Wf, bf, Yf, Mf)[1]
            return dict(Z=Zf.cpu().numpy(), W=Wf.cpu().numpy(), b=(bf - Wf @ mu).cpu().numpy(), js=js.cpu().numpy(),
                        ce=ce.cpu().numpy(), it=[ii['n_iter'], i_f['n_iter']], gmax=[float(ii['gmax'].max()), float(i_f['gmax'].max())])

    def retry(fn, *args, tries=240, wait=30):  # shared GPU: on OOM free our cache, wait, retry
        for t in range(tries):
            try:
                return fn(*args)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f'  OOM, retry {t + 1}/{tries} in {wait}s', flush=True)
                time.sleep(wait)
        raise RuntimeError('persistent OOM')

    oof = np.zeros((S, N, L, 4), np.float32)
    W, b = np.zeros((S, 5, L, 4, d), np.float32), np.zeros((S, 5, L, 4), np.float32)
    lam, val_ce, conv = np.zeros((S, 5, L), np.float32), np.zeros((S, 5, L, G), np.float32), []
    for l in range(L):
        tl = time.time()
        r = retry(layer_fits, l)
        for s in range(S):
            for f in range(5):
                oof[s, fold == f, l] = r['Z'][s * 5 + f, fold == f]
            W[s, :, l], b[s, :, l] = r['W'][s * 5:(s + 1) * 5], r['b'][s * 5:(s + 1) * 5]
        lam[:, :, l], val_ce[:, :, l] = np.array(TP.LAMS)[r['js']], r['ce']
        conv.append(dict(layer=l, sec=round(time.time() - tl, 1), iters=r['it'], gmax=r['gmax']))
        agr = [round(float((oof[s, :, l].argmax(-1) == labels[s]).mean()), 4) for s in range(S)]
        print(f'L{l:02d} {time.time() - tl:5.1f}s it={r["it"]} gmax={r["gmax"][0]:.1e}/{r["gmax"][1]:.1e} '
              f'lam={lam[:, :, l].tolist()} L1-agreement {dict(zip(tg, agr))}', C.stamp(t0), flush=True)
    for s, t in enumerate(tg):
        np.save(os.path.join(od, f'oof_cons_{t}.npy'), oof[s])
        np.savez(os.path.join(od, f'weights_cons_{t}.npz'), W=W[s], b=b[s], lam=lam[s], val_ce=val_ce[s])
    np.savez(os.path.join(od, 'targets.npz'), **{k: x.astype(np.int16) for k, x in T.items()}, w0=w0)
    # single-pass check: stored fold weights reproduce the OOF logits from each row's own raw state (contract formula)
    rr = np.sort(np.random.default_rng(0).choice(N, 256, replace=False))
    lp = np.load(os.path.join(ed, 'lens_params.npz'))
    rep = 0.0
    for l in range(0, L, max(1, L // 6)):
        h = C.bf16_bits_to_f32(hid[rr, l]).astype(np.float64)
        hn = h / np.sqrt((h ** 2).mean(-1, keepdims=True) + eps)
        for s in range(S):
            z = (hn * lp['norm_w']) @ lp['U4'].T.astype(np.float64) + np.einsum('nd,ncd->nc', hn, W[s, fold[rr], l]) + b[s, fold[rr], l]
            rep = max(rep, float(np.abs(z - oof[s, rr, l]).max()))
    gm = np.array([c['gmax'] for c in conv])
    S_ = dict(model=a.model, fmt=a.fmt, N=N, L=L, d=d, w0=w0, targets=tg, supervision='L1 (targets from the model\'s own '
              'outputs; gold never read)', lams=TP.LAMS, inner_fold=gf, lockbox_fold=LOCK,
              target_stats=dict(vote_eq_maxcons=round(float((T['vote'] == T['maxcons']).mean()), 4),
                                maxcons_layer_lt_last=int((T['maxcons_layer'] < L - 1).sum()),
                                final_cons_hist=np.bincount(T['final_cons'], minlength=5).tolist(),
                                maxcons_cons_hist=np.bincount(T['maxcons_cons'], minlength=5).tolist()),
              l1_agreement={t: np.round((oof[s].argmax(-1) == labels[s][:, None]).mean(0), 4).tolist() for s, t in enumerate(tg)},
              lam={t: lam[s].tolist() for s, t in enumerate(tg)},
              lam_boundary=dict(max=int((lam == TP.LAMS[0]).sum()), min=int((lam == TP.LAMS[-1]).sum())),
              convergence=dict(gmax_inner_max=float(gm[:, 0].max()), gmax_final_max=float(gm[:, 1].max()),
                               final_gmax_above_1e4=int((gm[:, 1] > 1e-4).sum()), per_layer=conv),
              single_pass_reproduce_maxabs=rep, seconds=round(time.time() - t0, 1), script_md5=C.md5_file(__file__),
              common_md5=C.md5_file(C.__file__), train_probe_md5=C.md5_file(TP.__file__), args=vars(a))
    C.save_json(S_, os.path.join(od, 'train_summary.json'))
    print(json.dumps({k: S_[k] for k in ('target_stats', 'lam_boundary', 'single_pass_reproduce_maxabs')}),
          {k: v for k, v in S_['convergence'].items() if k != 'per_layer'}, 'DONE', C.stamp(t0), flush=True)


# ----------------------------------------------------------------------------- evaluation (CPU)
def split_boot(qm):
    """By-question bootstrap weights (B, Q) resampling only the questions in mask qm (others weight 0)."""
    Wq = np.zeros((B_BOOT, len(qm)))
    Wq[:, np.flatnonzero(qm)] = boot_weights(int(qm.sum()), B_BOOT)
    return Wq


def lockbox_crossfit(score, fold, pop):
    """score (n_cand, N) per-row criterion. Chosen candidate per fold f: best mean on pop rows of folds {0..3} - f."""
    return np.array([int(score[:, pop & (fold != f) & (fold != LOCK)].mean(1).argmax()) for f in range(5)])


def kappa(x, y):
    labs = np.union1d(x, y)
    po = (x == y).mean()
    pe = sum((x == k).mean() * (y == k).mean() for k in labs)
    return round(float((po - pe) / (1 - pe)), 4)


def evaluate(a):
    t0 = time.time()
    D = load_inputs(a.model, a.fmt, 'full', 'raw')
    N, L, gold, fold = D['N'], D['L'], D['gold'], D['fold']
    Q, w0, ar, v = N // 4, int(round(0.45 * L)), np.arange(N), np.arange(N) % 4
    rd, ed, od = run_paths(a.model, a.fmt, a.out)
    tz = np.load(os.path.join(od, 'targets.npz'))
    assert int(tz['w0']) == w0
    tq = {t: tz[t].astype(int) for t in TARGETS}  # L1 target contents (Q,)
    tl = {t: (np.repeat(tq[t], 4) - v) % 4 for t in TARGETS}  # L1 target letters (N,)
    heads = dict(lens=np.load(os.path.join(ed, 'lens4.npy')), probe=D['probe'],
                 **{f'cons_{t}': np.load(os.path.join(od, f'oof_cons_{t}.npy')) for t in TARGETS})
    level = dict(lens='L0', probe='L2/L3 (gold-trained)', cons_vote='L1', cons_maxcons='L1')
    out4 = D['out4']
    base_let = top1(out4)
    base_ok = base_let == gold
    answer, qf, excl = gold[0::4], fold[0::4], ~to_q(D['compliant']).all(1)
    csd_c, final_c = contents(to_q(top1(D['csd']))), contents(to_q(base_let))
    # official v1 K=3 recomputed under the contract's L-1 convention (--last_layer; 'output' official since 2026-10-16
    # evening) and cross-checked against RUN/classify/questions.csv (decoy-O pairs do not depend on the convention)
    p_let = top1(D['probe'])
    if a.last_layer == 'output':
        p_let[:, L - 1] = base_let
    off = classify_v1(contents(to_q(p_let)), csd_c, final_c, answer, 3, excl)
    cls, nfc, keep = off['outcome'], off['n_final'], ~excl
    qcsv, cconf = os.path.join(rd, 'classify', 'questions.csv'), json.load(open(os.path.join(rd, 'classify', 'summary.json')))['config']
    cls_csv = np.array([r['outcome_v1_K3'] for r in csv.DictReader(open(qcsv))])
    diff = {f'{x}->{y}': int(((cls_csv == x) & (cls == y)).sum()) for x in 'ROMUX' for y in 'ROMUX' if x != y and ((cls_csv == x) & (cls == y)).any()}
    if cconf.get('last_layer', 'probe') == a.last_layer:
        assert not diff, diff
    az = np.load(os.path.join(rd, 'classify', 'arrays.npz'))
    dO, nfc_c = az['decoyO_K3'] & keep[:, None], (az['final_c'][:, :, None] == np.arange(4)).sum(1)
    splits = {'f0-3': keep & (qf != LOCK)}
    if a.open_lockbox:
        splits['f4'] = keep & (qf == LOCK)
    Wb = {s: split_boot(m) for s, m in splits.items()}
    pop = np.repeat(keep, 4)
    rep = [l for l in REPORT if l < L - 1] + [L - 1]
    print(f'{a.model}__{a.fmt} Q={Q} L={L} w0={w0} classes {dict(zip(*np.unique(cls, return_counts=True)))} '
          f'splits {[(s, int(m.sum())) for s, m in splits.items()]}', C.stamp(t0), flush=True)
    R = dict(config=dict(model=a.model, fmt=a.fmt, N=N, Q=Q, L=L, w0=w0, lockbox_fold=LOCK, lockbox_open=a.open_lockbox,
                         splits={s: int(m.sum()) for s, m in splits.items()}, report_layers=rep, n_boot=B_BOOT,
                         levels=level, kgrid=KGRID, tgrid=TGRID, official_taxonomy=dict(
                             last_layer=a.last_layer, classes={k: int((cls == k).sum()) for k in 'ROMUX'},
                             csv_md5=C.md5_file(qcsv), csv_mtime=time.ctime(os.path.getmtime(qcsv)),
                             csv_last_layer=cconf.get('last_layer', 'probe'), changed_vs_csv=diff)),
             targets={}, curves={}, exit={}, taxonomy={})

    # ---- targets (4-pass L1 policies) scored against gold
    for t in TARGETS:
        R['targets'][t] = {s: dict(row_acc=ci(Wb[s], to_q(tl[t] == gold).sum(1) * m, 4 * m),
                                   q_correct=ci(Wb[s], (tq[t] == answer) & m, m),
                                   by_class={k: round(float((tq[t] == answer)[m & (cls == k)].mean()), 4) for k in 'ROMU'})
                           for s, m in splits.items()}

    # ---- curves
    lt = {h: top1(H) for h, H in heads.items()}
    ok = {h: x == gold[:, None] for h, x in lt.items()}
    for h in heads:
        hq = contents(to_q(lt[h]))  # (Q, 4, L)
        c4 = (hq == answer[:, None, None]).all(1)
        self4 = (hq == hq[:, :1]).all(1) & (hq[:, 0] >= 0)
        R['curves'][h] = {s: dict(acc=np.round(ok[h][np.repeat(m, 4)].mean(0), 4).tolist(),
                                  acc4c=np.round(c4[m].mean(0), 4).tolist(), self4=np.round(self4[m].mean(0), 4).tolist(),
                                  **{f'agree_{t}': np.round((lt[h] == tl[t][:, None])[np.repeat(m, 4)].mean(0), 4).tolist()
                                     for t in TARGETS}) for s, m in splits.items()}
    for t in TARGETS:
        h = f'cons_{t}'
        R['curves'][f'{h}_minus'] = {s: {ref: {int(l): ci(Wb[s], (to_q(ok[h][:, l]).sum(1) - to_q(ok[ref][:, l]).sum(1)) * m, 4 * m)
                                               for l in rep} for ref in ('probe', 'lens')} for s, m in splits.items()}

    # ---- early exit
    def metrics(let, e, qm, Wq):
        after = let == gold
        na, nb = to_q(after).sum(1), to_q(base_ok).sum(1)
        isO, isR, isUM = (cls == 'O') & qm, (cls == 'R') & qm, np.isin(cls, ['U', 'M']) & qm
        fixed = (na == 4) & qm
        oR, dec = ratio(isO & fixed, isO), decoy_fn(isO, isUM, nfc, fixed)
        on = (contents(to_q(let))[:, :, None] == np.arange(4)).all(1)  # (Q, 4): all 4 versions on content c
        dm = dO & qm[:, None]
        st = lambda x: np.stack([(x & (nfc_c == n)).sum(1) for n in range(4)], 1).astype(float)
        sym = strat_rate(np.stack([isO & (nfc == n) for n in range(4)], 1).astype(float), st(dm), st(dm & on))
        pr = np.repeat(qm, 4)
        return dict(O_to_R=[int((isO & fixed).sum()), int(isO.sum())], O_to_R_rate=wci(oR, Wq), decoy_4of4=wci(dec, Wq),
                    excess=wci(lambda w: oR(w) - dec(w), Wq), decoyO_pairs_to_c=[int((dm & on).sum()), int(dm.sum())],
                    decoyO_4of4=wci(sym, Wq), excess_vs_decoyO=wci(lambda w: oR(w) - sym(w), Wq),
                    net_acc=ci(Wq, (na - nb) * qm, 4 * qm), acc_after=round(float(after[pr].mean()), 5),
                    acc_before=round(float(base_ok[pr].mean()), 5), R_harm=[int((isR & ~fixed).sum()), int(isR.sum())],
                    R_harm_rate=ci(Wq, isR & ~fixed, isR), mean_exit=round(float(e[pr].mean()), 2),
                    frac_early=round(float((e < L - 1)[pr].mean()), 4))

    policies = {}  # name -> (letters (N,), exit layers (N,), chosen parameters per fold, level)
    grid = [(k, t) for k in KGRID for t in TGRID]
    sweep = {}
    for name, h, crit_name in [('E-cons[vote]', 'cons_vote', 'vote'), ('E-cons[maxcons]', 'cons_maxcons', 'maxcons'),
                               ('E-head[gold probe]', 'probe', 'gold'), ('diag E-cons[maxcons] gold-selected', 'cons_maxcons', 'gold'),
                               ('diag E-head L1-selected', 'probe', 'maxcons')]:
        diag = name.startswith('diag')  # cross conditions: separate the head from the selection criterion (fix, stab only)
        H = heads[h].copy()
        H[:, L - 1] = out4
        let, conf = top1(H), softmax(H).max(-1)
        sH = np.sort(H.astype(np.float64), -1)
        lconf = -np.log1p(np.exp(sH[..., :-1] - sH[..., -1:]).sum(-1))
        crit = let == (gold if crit_name == 'gold' else tl[crit_name])[:, None]  # selection criterion per (row, layer)
        lev = level[h] + (' head, rule params by gold accuracy' if crit_name == 'gold' else ' head, rule params by L1 agreement')
        layers = np.arange(w0, L)
        if not diag:  # descriptive fixed-layer sweep (nothing selected): net accuracy, O->R, R harm per exit layer
            okl, base4 = to_q(let == gold[:, None]).sum(1), to_q(base_ok).sum(1)
            sweep[h] = {s_: dict(net=np.round([((okl[:, l] - base4) * m).sum() / (4 * m.sum()) for l in layers], 5).tolist(),
                                 O_to_R=[int(((okl[:, l] == 4) & (cls == 'O') & m).sum()) for l in layers],
                                 R_harm=[int(((okl[:, l] < 4) & (cls == 'R') & m).sum()) for l in layers])
                        for s_, m in splits.items()}
        ch = lockbox_crossfit(crit[:, w0:].T, fold, pop)
        e = layers[ch[fold]]
        policies[f'{name} fix'] = (let[ar, e], e, layers[ch].tolist(), lev)
        run = stab_runs(let, w0)
        E = np.stack([first_fire((run >= k) & (conf >= t), w0) for k, t in grid])
        ch = lockbox_crossfit(crit[ar[None], E], fold, pop)
        e = E[ch[fold], ar]
        policies[f'{name} stab'] = (let[ar, e], e, [list(grid[c]) for c in ch], lev)
        if diag:
            continue
        e = w0 + lconf[:, w0:].argmax(1)
        policies[f'{name} conf'] = (let[ar, e], e, None, lev)
        policies[f'{name} NOEXIT(own L-1 readout)'] = (top1(heads[h][:, L - 1]), np.full(N, L - 1), None, level[h])
    R['sweep'] = dict(layers=list(range(w0, L)), note='exit at layer l for every row (l = L-1: real output); '
                      'descriptive, nothing selected', **sweep)
    # 4-pass L1 references (need all 4 versions)
    full = np.full(N, L - 1)
    policies['4pass vote_final'] = (tl['vote'], full, None, 'L1, 4 passes, no depth')
    policies['4pass cons_exit (=maxcons target)'] = (tl['maxcons'], np.repeat(tz['maxcons_layer'], 4).astype(int), None, 'L1, 4 passes')
    f4 = tz['csd_cons'][:, w0:] == 4
    last4 = L - 1 - f4[:, ::-1].argmax(1)
    gate = (tz['final_cons'] < 4) & f4.any(1)
    gc = csd_c[np.arange(Q), 0, last4]
    policies['4pass cons_gate'] = (np.where(np.repeat(gate, 4), (np.repeat(gc, 4) - v) % 4, base_let),
                                   np.where(np.repeat(gate, 4), np.repeat(last4, 4), L - 1), None, 'L1, 4 passes')
    for p, (let, e, ch, lev) in policies.items():
        R['exit'][p] = dict(level=lev, chosen_per_fold=ch, **{s: metrics(let, e, m, Wb[s]) for s, m in splits.items()})
        for s in splits:
            m = R['exit'][p][s]
            print(f'{p:42s} {s:4s} O->R {m["O_to_R"]} {m["O_to_R_rate"]} dec {m["decoy_4of4"][0]:.3f} decO '
                  f'{m["decoyO_pairs_to_c"]} exc_vs_decO {m["excess_vs_decoyO"]} net {m["net_acc"]} Rharm {m["R_harm"]} '
                  f'exit {m["mean_exit"]} early {m["frac_early"]} | {ch}', flush=True)

    # ---- label-free taxonomy (L1) vs official v1 K=3
    merge = lambda o: np.where(np.isin(o, ['R', 'M']), 'RM', o)
    for t in TARGETS:
        cl_ = lt[f'cons_{t}'].copy()
        if a.last_layer == 'output':
            cl_[:, L - 1] = base_let
        cons_c = contents(to_q(cl_))
        lf = classify_v1(cons_c, csd_c, final_c, tq[t], 3, excl)['outcome']  # fully label-free
        g1 = classify_v1(cons_c, csd_c, final_c, answer, 3, excl)['outcome']  # consensus head, gold answer
        R['taxonomy'][t] = {}
        for s, m in splits.items():
            o, x, y = cls[m], lf[m], g1[m]
            isO, isOt = (cls == 'O') & m, (lf == 'O') & m
            R['taxonomy'][t][s] = dict(
                counts_label_free={k: int((x == k).sum()) for k in 'ROMU'},
                crosstab_official_x_labelfree={k: {k2: int(((o == k) & (x == k2)).sum()) for k2 in 'ROMU'} for k in 'ROMU'},
                kappa_RM_merged=kappa(merge(o), merge(x)), agree_RM_merged=round(float((merge(o) == merge(x)).mean()), 4),
                O_recall=ci(Wb[s], isO & isOt, isO), O_precision=ci(Wb[s], isO & isOt, isOt),
                Otilde_target_is_gold=ci(Wb[s], isOt & (tq[t] == answer), isOt),
                conshead_gold_v1=dict(agree=round(float((o == y).mean()), 4), kappa=kappa(o, y),
                                      counts={k: int((y == k).sum()) for k in 'ROMU'},
                                      O_recall=round(float(((o == 'O') & (y == 'O')).sum() / max(1, (o == 'O').sum())), 4)))
            r = R['taxonomy'][t][s]
            print(f'taxonomy[{t}] {s}: label-free {r["counts_label_free"]} kappa(RM merged) {r["kappa_RM_merged"]} '
                  f'O recall {r["O_recall"]} precision {r["O_precision"]} O~ target=gold {r["Otilde_target_is_gold"]} | '
                  f'cons-head+gold v1 agree {r["conshead_gold_v1"]["agree"]} kappa {r["conshead_gold_v1"]["kappa"]}', flush=True)
    R['config'].update(seconds=round(time.time() - t0, 1), script_md5=C.md5_file(__file__), common_md5=C.md5_file(C.__file__))
    C.save_json(R, os.path.join(od, 'eval_final.json' if a.open_lockbox else 'eval_f03.json'))
    print('wrote', od, C.stamp(t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
    ap.add_argument('--target', default=','.join(TARGETS)); ap.add_argument('--out', default='consensus_probe')
    ap.add_argument('--eval', action='store_true'); ap.add_argument('--open_lockbox', action='store_true')
    ap.add_argument('--last_layer', default='output', choices=['output', 'probe'], help='official v1 L-1 convention')
    a = ap.parse_args()
    assert all(t in TARGETS for t in a.target.split(','))
    evaluate(a) if a.eval else train(a)


if __name__ == '__main__':
    main()
