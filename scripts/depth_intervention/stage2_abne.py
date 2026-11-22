"""Stage 2: the 'available-but-not-emitted' (ABNE) quantity. CPU parts scan/features/geometry/predict; GPU part causal.

Background (RESULTS.md §2): the gold-trained probe reads the answer at L18-L21 while lens/CSD emit at L24 (Qwen3-4B).
Supervision levels: L0 no training (lens/CSD readouts, hidden geometry); L1 self-supervised (consensus-trained probe,
RUN/consensus_probe/oof_cons_{maxcons,vote}.npy, used if present); L2 outputs of the gold-trained probe (RUN/probe) as features; L3 trained
on gold. Gold labels only evaluate. Lockbox = fold 4: every selection (layer, feature, model, residualization) is fit on
folds 0-3 (cross-fitted there by `fold`) and the selected rule is scored once on fold 4. Bootstrap 95% CIs by question.

scan (one pass over extract/hidden_bf16.u16 -> RUN/abne/scan.npz). hn = h / sqrt(mean(h^2) + eps); E = U4 * norm_w (lens
  = hn @ E.T, so span(E) is span(U4) in hn coordinates); Q4 = orthonormal basis of span(E) whose first 3 columns D span
  the letter-difference subspace (the part that moves the 4-way softmax); 10 random orthonormal 3-d bases (seed 0).
  Per row x layer: r4 = hn @ Q4, rr = random coordinates, hn2 = |hn|^2. Per layer: M (4, d) = mean over questions of the
  question-centred hn of the version whose gold letter is k (each question has one row per gold letter), SSw = total
  question-centred energy. Transfer: hn_l @ Wc[f, j].T for source layers j in J = {e-6, e-4, e-2}, Wc = class-centred
  probe W of the row's fold (no lens, no bias).
features (RUN/abne/features.npz, (N, L) f32). z_p = probe/oof_full.npy, z_l = extract/lens4.npy, k1/k2 = probe top-1/2.
  L2: abne = (z_p[k1] - z_p[k2]) - (z_l[k1] - max_{j!=k1} z_l[j]); dis = [probe top-1 != lens top-1];
      kl = KL(softmax z_p || softmax z_l); corr = |c~|^2 / (|c~|^2 + |z_l~|^2), c = z_p - z_l = hn @ W.T + b, ~ = centred
      over the 4 classes; corrW = the same without b.
  L1: the same with the consensus heads of scripts/consensus_probe.py: suffix _c = 'maxcons' target (4-version CSD
      consensus at the most consistent layer; main L1), _cv = 'vote' target (4-version final-output consensus).
  L0: u4 = |r4|^2 / hn2, u4d = |r4[:, :3]|^2 / hn2, u4d_g = log u4d[l] - log u4d[l-1], cmass = extract/csd_mass.npy,
      cmass_g = cmass[l] - cmass[l-1]; confidences {p,l,c}maxp/{p,l,c}ent (probe/lens/CSD), fmaxp/fent (real output).
  e = emission layer (label-free): largest one-layer rise of the population CSD top-1 agreement with the real output.
geometry (RUN/abne/geometry.json): July 'amplification vs relocation' check (discussion 2026-10-14 V4) on this run:
  readout / random 3-d subspace probes (raw and question-centred), probe-direction overlap with span(E), where the trained
  probe reads from, letter energy in span(E) vs its complement, direction persistence across e, transfer of the
  pre-emission probe directions to later layers, per-row regression of the post-emission readout coordinates.
predict (RUN/abne/predict.json): (a) emission timing: Spearman of features at l with FJC_c - l (question) and with the
  row CSD commit layer - l (row), plus 'the probe's class at l is eventually output'; (b) AUROC tasks O vs U,
  U_avail vs U_none, loop rescue (loop_allq: rows wrong under last_none fixed by w16_23_k1; direction among changed
  rows; U questions -> R/O by loop_eval classes (old last-layer contract) and loop_U_to_R_q = U under 'last layer = real
  output' with the none run not 4/4 -> w16_23_k1 output 4/4 correct). Baselines: final max softmax + entropy (base_f); strict = base_f + layer-l max softmax and
  entropy of lens, CSD (+ probe for L1/L2). gain = AUROC(base + x) - AUROC(base); resid = AUROC of x residualized on base.
  'New quantity' = lockbox gain >= 0.05.
causal (GPU, exploratory): add a direction to the critical-position residual at block l and run the remaining blocks with
  the row's own prefix cache (as loop.py mode last); measure CSD/lens emission of the probe's class and the final output.
Usage: python scripts/stage2_abne.py --model qwen3-4b-base --fmt base_fewshot [--parts scan,features,geometry,predict]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
import argparse, csv, glob, json, sys, time, warnings
import numpy as np
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import classify as CL
import stage2_analysis as SA

B_BOOT, SEED, NRAND, LOCK = 2000, 0, 10, 4
warnings.filterwarnings('ignore', category=RuntimeWarning)


def rnd(x, k=4):
    return SA.rnd(np.atleast_1d(x), k) if np.ndim(x) else round(float(x), k)


def jsonable(o):
    """Recursively convert numpy scalars / arrays and non-string keys for json."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def ctr(x):
    return x - x.mean(-1, keepdims=True)


def qctr(x):
    """Question-centre a row array (N, ...) over the 4 versions."""
    q = x.reshape((-1, 4) + x.shape[1:])
    return (q - q.mean(1, keepdims=True)).reshape(x.shape)


def load_run(model, fmt):
    rd = C.run_dir(model, fmt)
    ex = os.path.join(rd, 'extract')
    rows = json.load(open(os.path.join(ex, 'rows.json')))
    lp = dict(np.load(os.path.join(ex, 'lens_params.npz')))
    E = (lp['U4'] * lp['norm_w'][None]).astype(np.float64)
    D = np.linalg.qr((E[1:] - E[:1]).T)[0]                                   # (d, 3) letter-difference span
    m = E.mean(0) - D @ (D.T @ E.mean(0))
    Q4 = np.c_[D, m / np.linalg.norm(m)]                                      # (d, 4) span(E), D first
    rng = np.random.default_rng(SEED)
    QR = np.stack([np.linalg.qr(rng.standard_normal((E.shape[1], 3)))[0] for _ in range(NRAND)])  # (NRAND, d, 3)
    Wz = np.load(os.path.join(rd, 'probe', 'weights_full.npz'))
    return dict(model=model, fmt=fmt, rd=rd, ex=ex, od=os.path.join(rd, 'abne'), rows=rows, N=len(rows), lp=lp, E=E, Q4=Q4, QR=QR,
                eps=float(lp['eps']), W=Wz['W'], b=Wz['b'], gold=np.array([r['gold'] for r in rows]),
                fold=np.array([r['fold'] for r in rows]), src=np.array([r['source'] for r in rows]))


def emission_layer(csd4, out4):
    """Label-free emission layer: largest one-layer rise of the population CSD top-1 agreement with the real output
    (the lens gives the same layer on Qwen; on Llama its letter priors make it noisy)."""
    agree = (csd4.argmax(-1) == out4.argmax(-1)[:, None]).mean(0)
    return int(np.argmax(np.diff(agree)) + 1), agree


# ----------------------------------------------------------------------------- scan
def part_scan(R, model, fmt, e):
    hid, meta = C.open_hidden(model, fmt)
    N, L, d = hid.shape
    Q = N // 4
    answer, fold = R['gold'][0::4], R['fold']
    J = [e - 6, e - 4, e - 2]
    Wc = R['W'][:, J] - R['W'][:, J].mean(2, keepdims=True)                   # (5, |J|, 4, d)
    Q4, QR = R['Q4'].astype(np.float32), R['QR'].transpose(1, 0, 2).reshape(d, -1).astype(np.float32)
    r4 = np.zeros((N, L, 4), np.float32); rr = np.zeros((N, L, NRAND, 3), np.float32)
    hn2 = np.zeros((N, L), np.float32); M = np.zeros((L, 4, d), np.float32); SSw = np.zeros(L)
    T = np.zeros((len(J), N, L, 4), np.float32)
    t0 = time.time()
    for l in range(L):
        h = C.bf16_bits_to_f32(hid[:, l])
        hn = h / np.sqrt((h.astype(np.float64) ** 2).mean(-1, keepdims=True) + R['eps']).astype(np.float32)
        r4[:, l], rr[:, l] = hn @ Q4, (hn @ QR).reshape(N, NRAND, 3)
        hn2[:, l] = (hn.astype(np.float64) ** 2).sum(-1)
        hc = hn.reshape(Q, 4, d) - hn.reshape(Q, 4, d).mean(1, keepdims=True)
        SSw[l] = (hc.astype(np.float64) ** 2).sum()
        M[l] = np.stack([hc[np.arange(Q), (answer - k) % 4].mean(0) for k in range(4)])
        for f in range(5):
            i = fold == f
            T[:, i, l] = np.einsum('nd,jkd->jnk', hn[i], Wc[f])
        if l % 6 == 0:
            print(f'  scan L{l}', C.stamp(t0), flush=True)
    np.savez(os.path.join(R['od'], 'scan.npz'), r4=r4, rr=rr, hn2=hn2, M=M, SSw=SSw, T=T, J=np.array(J), e=e)


# ----------------------------------------------------------------------------- features
def readout_stats(z):
    lp = log_softmax(z.astype(np.float64), -1)
    p = np.exp(lp)
    return p.max(-1), -(p * lp).sum(-1), lp


def abne_quantities(zp, zl, b=None):
    """L2/L1 quantities of a lens-anchored probe zp vs the lens zl, both (N, L, 4); b (N, L, 4) the probe bias."""
    zp, zl = zp.astype(np.float64), zl.astype(np.float64)
    o = np.argsort(zp, -1)
    k1, k2 = o[..., -1:], o[..., -2:-1]
    tk = lambda z, k: np.take_along_axis(z, k, -1)[..., 0]
    zl_other = np.where(np.arange(4) == k1, -np.inf, zl).max(-1)
    _, _, lpp = readout_stats(zp)
    _, _, lpl = readout_stats(zl)
    c = zp - zl
    ce = lambda c: (ctr(c) ** 2).sum(-1) / ((ctr(c) ** 2).sum(-1) + (ctr(zl) ** 2).sum(-1))
    F = dict(abne=(tk(zp, k1) - tk(zp, k2)) - (tk(zl, k1) - zl_other), dis=(k1[..., 0] != zl.argmax(-1)),
             kl=(np.exp(lpp) * (lpp - lpl)).sum(-1), corr=ce(c))
    if b is not None:
        F['corrW'] = ce(c - b)
    return {k: v.astype(np.float32) for k, v in F.items()}, k1[..., 0]


def consensus_probe(rd, N, L, target):
    """L1 head logits (N, L, 4): RUN/consensus_probe/oof_cons_{target}.npy (scripts/consensus_probe.py), if present."""
    p = os.path.join(rd, 'consensus_probe', f'oof_cons_{target}.npy')
    if os.path.exists(p) and np.load(p, mmap_mode='r').shape == (N, L, 4):
        return np.load(p).astype(np.float32), p
    return None, p


def part_features(R, e):
    ex, N = R['ex'], R['N']
    zl, zp = np.load(os.path.join(ex, 'lens4.npy')), np.load(os.path.join(R['rd'], 'probe', 'oof_full.npy'))
    L = zl.shape[1]
    F, k1 = abne_quantities(zp, zl, R['b'][R['fold']])
    S = np.load(os.path.join(R['od'], 'scan.npz'))
    F['u4'] = (S['r4'] ** 2).sum(-1) / S['hn2']
    F['u4d'] = (S['r4'][..., :3] ** 2).sum(-1) / S['hn2']
    F['u4d_g'] = np.c_[np.zeros((N, 1)), np.diff(np.log(F['u4d']), axis=1)]
    F['cmass'] = np.load(os.path.join(ex, 'csd_mass.npy')).astype(np.float32)
    F['cmass_g'] = np.c_[np.zeros((N, 1)), np.diff(F['cmass'], axis=1)]
    out4 = np.load(os.path.join(ex, 'out_logits4.npy'))
    for nm, z in (('p', zp), ('l', zl), ('c', np.load(os.path.join(ex, 'csd_raw4.npy'))), ('f', out4[:, None])):
        mx, en, _ = readout_stats(z)
        F[f'{nm}maxp'], F[f'{nm}ent'] = mx, en
    F['fmaxp'], F['fent'] = F['fmaxp'][:, 0], F['fent'][:, 0]
    cons = {}
    for target, sfx in (('maxcons', '_c'), ('vote', '_cv')):   # L1 heads: CSD-consensus target (main), final-vote target
        zc, cons[target] = consensus_probe(R['rd'], N, L, target)
        if zc is not None:
            Fc, _ = abne_quantities(zc, zl, np.load(cons[target].replace('oof_cons_', 'weights_cons_').replace('.npy', '.npz'))['b'][R['fold']])
            F.update({f'{k}{sfx}': v for k, v in Fc.items()})
            F[f'pmaxp{sfx}'], F[f'pent{sfx}'] = readout_stats(zc)[:2]
        else:
            cons[target] = f'absent ({cons[target]})'
    F = {k: np.asarray(v, np.float32) for k, v in F.items()}
    F['k1'] = k1.astype(np.int8)
    F['csd_top1'] = np.load(os.path.join(ex, 'csd_raw4.npy')).argmax(-1).astype(np.int8)
    F['lens_top1'] = zl.argmax(-1).astype(np.int8)
    F['out_top1'] = out4.argmax(-1).astype(np.int8)
    np.savez(os.path.join(R['od'], 'features.npz'), **F)
    return dict(consensus_probe=cons, e=e)


# ----------------------------------------------------------------------------- geometry
def lr_acc(X, y, fold):
    """Cross-fitted (by fold) multinomial LR (standardized, C=1) row accuracy."""
    ok = np.zeros(len(y), bool)
    for f in np.unique(fold):
        tr = fold != f
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr], y[tr])
        ok[~tr] = m.predict(X[~tr]) == y[~tr]
    return float(ok.mean())


def cv_r2(Y, X, fold):
    """Cross-fitted OLS R^2 of Y (n, k) on [1, X] (pooled over held-out folds)."""
    P = np.zeros_like(Y)
    for f in np.unique(fold):
        tr = fold == f
        A = np.c_[np.ones(len(X)), X]
        P[tr] = A[tr] @ np.linalg.lstsq(A[~tr], Y[~tr], rcond=None)[0]
    return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def maxcos(A, B):
    """Largest cosine between span(A) and span(B), A (d, a) and B (d, b) with orthonormal columns."""
    return float(np.linalg.svd(A.T @ B, compute_uv=False)[0])


def part_geometry(R, e, layers=None):
    S = dict(np.load(os.path.join(R['od'], 'scan.npz')))
    F = np.load(os.path.join(R['od'], 'features.npz'))
    zl = np.load(os.path.join(R['ex'], 'lens4.npy')).astype(np.float64)
    zp = np.load(os.path.join(R['rd'], 'probe', 'oof_full.npy')).astype(np.float64)
    r4, rr, M, J, T = S['r4'].astype(np.float64), S['rr'], S['M'].astype(np.float64), [int(j) for j in S['J']], S['T']
    N, L, d = r4.shape[0], r4.shape[1], M.shape[2]
    gold, fold, Q = R['gold'], R['fold'], r4.shape[0] // 4
    E, Q4, QR, W, b = R['E'], R['Q4'], R['QR'], R['W'].astype(np.float64), R['b']
    D = Q4[:, :3]
    Ec = ctr(E.T).T
    layers = layers or list(range(max(0, e - 12), L))
    out = dict(e=e, J=J, d=d, random_energy_frac_3d=3 / d, random_energy_frac_4d=4 / d, layers=layers, per_layer={})
    rand_cos = np.mean([maxcos(QR[i], QR[(i + 1) % NRAND]) for i in range(NRAND)])
    out['random_maxcos_3d_vs_3d'] = rnd(rand_cos)
    sW = ctr(zp - zl - b[fold])                                                # (N, L, 4) = hn @ Wc[f, l].T
    prev = None
    for l in layers:
        g = {}
        # (1) subspace probes: readout 3-d / 4-d vs random 3-d; raw and question-centred (transductive diagnostic)
        x3, x4 = r4[:, l, :3], r4[:, l]
        ra = [lr_acc(rr[:, l, k], gold, fold) for k in range(NRAND)]
        rc = [lr_acc(qctr(rr[:, l, k]), gold, fold) for k in range(NRAND)]
        g['acc'] = dict(full_probe=rnd((zp[:, l].argmax(-1) == gold).mean()), lens=rnd((zl[:, l].argmax(-1) == gold).mean()),
                        readout3=rnd(lr_acc(x3, gold, fold)), readout4=rnd(lr_acc(x4, gold, fold)),
                        random3=rnd([np.mean(ra), np.min(ra), np.max(ra)]),
                        readout3_qc=rnd(lr_acc(qctr(x3), gold, fold)), random3_qc=rnd([np.mean(rc), np.min(rc), np.max(rc)]))
        # (2) probe direction vs span(E): class-centred W (correction) and E + W (full effective direction)
        ov = {k: [] for k in ('W_frac_spanE', 'W_frac_D', 'W_maxcos_D', 'P_frac_spanE', 'P_maxcos_D', 'W_maxcos_rand')}
        for f in range(5):
            Wc = ctr(W[f, l].T).T
            for nm, A in (('W', Wc), ('P', Wc + Ec)):
                U = np.linalg.svd(A.T, full_matrices=False)[0][:, :3]
                ov[f'{nm}_frac_spanE'].append(((A @ Q4) ** 2).sum() / (A ** 2).sum())
                ov[f'{nm}_maxcos_D'].append(maxcos(U, D))
                if nm == 'W':
                    ov['W_frac_D'].append(((A @ D) ** 2).sum() / (A ** 2).sum())
                    ov['W_maxcos_rand'].append(np.mean([maxcos(U, QR[k]) for k in range(NRAND)]))
        g['probe_dir'] = {k: rnd(np.mean(v)) for k, v in ov.items()}
        # (3) where the trained probe reads from: its correction split into span(E) and the complement
        Wq = np.einsum('fkd,dj->fkj', ctr(W[:, l].transpose(0, 2, 1)).transpose(0, 2, 1), Q4)  # (5, 4, 4) Wc @ Q4
        sW_E = np.einsum('nj,nkj->nk', r4[:, l], Wq[fold])                    # part of hn @ Wc.T from span(E)
        sW_perp = sW[:, l] - sW_E
        tot = (sW[:, l] - sW[:, l].mean(0)).var(0).sum()
        g['corr_var_share_spanE'] = rnd((sW_E - sW_E.mean(0)).var(0).sum() / tot)
        bl = b[fold, l]
        g['acc_probe_parts'] = dict(perp_only=rnd(((sW_perp + bl).argmax(-1) == gold).mean()),
                                    spanE_only=rnd(((zl[:, l] + sW_E + bl).argmax(-1) == gold).mean()))
        g['R2_probe_logits_by_readout3'] = rnd(cv_r2(ctr(zp[:, l]), x3, fold))
        # (4) letter signal (question-centred, by gold letter): energy in D / span(E) / complement / random 3-d
        Ml = M[l]
        etot, eD, e4 = (Ml ** 2).sum(), ((Ml @ D) ** 2).sum(), ((Ml @ Q4) ** 2).sum()
        eR = np.mean([((Ml @ QR[k]) ** 2).sum() for k in range(NRAND)])
        c3 = qctr(x3)
        g['letter'] = dict(e_tot=rnd(etot, 6), e_D=rnd(eD, 6), e_spanE=rnd(e4, 6), e_perp=rnd(etot - e4, 6),
                           e_rand3=rnd(eR, 6), frac_D=rnd(eD / etot, 5), frac_spanE=rnd(e4 / etot, 5),
                           letter_var_frac_full=rnd(Q * etot / S['SSw'][l]),
                           letter_var_frac_D=rnd(Q * eD / (c3 ** 2).sum()),
                           letter_var_frac_rand3=rnd(np.mean([Q * ((Ml @ QR[k]) ** 2).sum() / (qctr(rr[:, l, k]) ** 2).sum()
                                                              for k in range(NRAND)])),
                           between_q_share_D=rnd(1 - (c3 ** 2).sum() / ((x3 - x3.mean(0)) ** 2).sum()),
                           mean_energy_D_rowwise=rnd((x3 ** 2).sum(-1).mean(), 5))
        Mp = Ml - (Ml @ Q4) @ Q4.T
        cosv = lambda a, c: float((a * c).sum() / np.sqrt((a ** 2).sum() * (c ** 2).sum()))
        g['persist'] = dict(cos_MD_prev=rnd(cosv(Ml @ D, M[l - 1] @ D)) if l else None,
                            cos_MD_final=rnd(cosv(Ml @ D, M[-1] @ D)),
                            cos_Mperp_prev=rnd(cosv(Mp, M[l - 1] - (M[l - 1] @ Q4) @ Q4.T)) if l else None,
                            cos_M_prev=rnd(cosv(Ml, M[l - 1])) if l else None)
        # (5) transfer: pre-emission probe correction directions (layer j) read at layer l (no lens, no bias)
        g['transfer'] = {int(j): dict(acc=rnd((T[i][:, l].argmax(-1) == gold).mean()),
                                      acc_qc=rnd((qctr(T[i][:, l]).argmax(-1) == gold).mean()))
                         for i, j in enumerate(J) if l >= j}
        # (6) per-row: is the post-step readout content an amplified copy of the pre-step readout content, or written
        #     from the complement? Cross-fitted R^2 of question-centred D coordinates at l on those at l-1 and on the
        #     probe correction read from the complement at l-1.
        if prev is not None:
            Y, Xa, Xw = c3, prev['c3'], qctr(prev['sW_perp'])
            g['row_regress'] = dict(R2_from_D_prev=rnd(cv_r2(Y, Xa, fold)), R2_from_perp_probe_prev=rnd(cv_r2(Y, Xw, fold)),
                                    R2_from_both=rnd(cv_r2(Y, np.c_[Xa, Xw], fold)),
                                    gain_norm=rnd(np.sqrt((Y ** 2).sum() / (Xa ** 2).sum())),
                                    R2_from_gold_onehot=rnd(cv_r2(Y, np.eye(4)[gold], fold)))
        prev = dict(c3=c3, sW_perp=sW_perp)
        out['per_layer'][l] = g
        print(f'  L{l}: acc {g["acc"]} dir {g["probe_dir"]} letter fracD {g["letter"]["frac_D"]} eD {g["letter"]["e_D"]} '
              f'eperp {g["letter"]["e_perp"]} {g.get("row_regress", "")}', flush=True)
    pl = out['per_layer']
    out['amplification_at_e'] = {
        f'{l - 1}->{l}': dict(e_D=rnd(pl[l]['letter']['e_D'] / pl[l - 1]['letter']['e_D']),
                              e_perp=rnd(pl[l]['letter']['e_perp'] / pl[l - 1]['letter']['e_perp']),
                              e_rand3=rnd(pl[l]['letter']['e_rand3'] / pl[l - 1]['letter']['e_rand3']),
                              e_tot=rnd(pl[l]['letter']['e_tot'] / pl[l - 1]['letter']['e_tot']),
                              rowwise_D_energy=rnd(pl[l]['letter']['mean_energy_D_rowwise'] /
                                                   pl[l - 1]['letter']['mean_energy_D_rowwise']))
        for l in range(e - 2, min(e + 3, L)) if l - 1 in pl}
    return out


# ----------------------------------------------------------------------------- prediction helpers
def lr_fit(X, y, tr, balanced=True):
    return make_pipeline(StandardScaler(), LogisticRegression(C=SA.CREG, class_weight='balanced' if balanced else None,
                                                              max_iter=5000)).fit(X[tr], y[tr])


def scores(x, y, tr, te, base=None, kind='single'):
    """Scores on rows te of a rule fit on rows tr. kind: 'single' (sign of x chosen on tr), 'resid' (x residualized on base
    by OLS on tr, then sign), 'lr' (logistic regression on [base, x] or x), 'base' (logistic regression on base)."""
    x = np.nan_to_num(np.asarray(x, np.float64)).reshape(len(y), -1)
    if kind == 'resid':
        x = SA.residualize(x, base, tr)
        if x.shape[1] == 0:
            return None
    if kind in ('single', 'resid'):
        x = x[:, 0]
        return x[te] * (1.0 if SA.auc(y[tr], x[tr]) >= 0.5 else -1.0)
    X = base if kind == 'base' else (x if base is None else np.c_[base, x])
    return lr_fit(X, y, tr).decision_function(X[te])


def dev_oof(x, y, fold, base, kind):
    """Cross-fitted (by fold, within the given rows) scores."""
    s = np.zeros(len(y))
    for f in np.unique(fold):
        sc = scores(x, y, fold != f, fold == f, base, kind)
        if sc is None:
            return None
        s[fold == f] = sc
    return s


def auc_block(x, y, fold, grp, base, dev, lock, boot=True):
    """dev (cross-fitted within folds 0-3) and lock (fit on dev, scored on fold 4): AUROC of x alone, gain over base,
    residualized AUROC. With boot: [pt, lo, hi] by-question bootstrap, else point estimates."""
    out = {}
    for part, m in (('dev', dev), ('lock', lock)):
        if part == 'dev':
            get = lambda kind: dev_oof(x[m], y[m], fold[m], base[m], kind)
        else:
            tr, te = np.flatnonzero(dev), np.flatnonzero(lock)
            idx = np.r_[tr, te]
            sel = np.r_[np.ones(len(tr), bool), np.zeros(len(te), bool)]
            get = lambda kind: scores(x[idx], y[idx], sel, ~sel, base[idx], kind)
        yy = y[m]
        if min(yy.sum(), (1 - yy).sum()) < 5:
            out[part] = dict(skipped='fewer than 5 in a class', n_pos=int(yy.sum()), n=int(m.sum()))
            continue
        W = SA.boot(grp[m], B_BOOT) if boot else None
        e_ = (lambda s, ref=None: SA.est(yy, s, W, ref)) if boot else \
            (lambda s, ref=None: None if s is None else round(SA.auc(yy, s) - (SA.auc(yy, ref) if ref is not None else 0), 4))
        sb = get('base')
        out[part] = dict(n_pos=int(yy.sum()), n=int(m.sum()), auc=e_(get('single')), gain=e_(get('lr'), sb),
                         resid=e_(get('resid')), base=e_(sb))
    return out


def spearman_block(x, t, base, grp, m):
    """Spearman rho(x, t) and partial rho (ranks residualized on the ranks of base by OLS) on rows m; bootstrap by grp."""
    if m.sum() < 30:
        return dict(skipped='fewer than 30', n=int(m.sum()))
    rk = lambda a: np.argsort(np.argsort(a, kind='mergesort'), kind='mergesort').astype(np.float64)
    xr, tr_ = rk(x[m]), rk(t[m])
    A = np.c_[np.ones(m.sum()), np.stack([rk(c) for c in base[m].T], 1)]
    rx = xr - A @ np.linalg.lstsq(A, xr, rcond=None)[0]
    rt = tr_ - A @ np.linalg.lstsq(A, tr_, rcond=None)[0]
    W = SA.boot(grp[m], B_BOOT)

    def wcorr(a, c, w):
        w = w / w.sum(1, keepdims=True)
        ma, mc = w @ a, w @ c
        return (w @ (a * c) - ma * mc) / np.sqrt((w @ (a * a) - ma ** 2) * (w @ (c * c) - mc ** 2))
    ci = lambda a, c: rnd([wcorr(a, c, np.ones((1, len(a))))[0], *np.nanpercentile(wcorr(a, c, W), [2.5, 97.5])])
    return dict(n=int(m.sum()), rho=ci(xr, tr_), partial_rho=ci(rx, rt))


def dev_gain(x, y, fold, base, base_auc):
    """Cross-fitted (within the given rows) AUROC gain of LR(base + x) over LR(base): point estimate (selection)."""
    s = dev_oof(x, y, fold, base, 'lr')
    return SA.auc(y, s) - base_auc


# ----------------------------------------------------------------------------- predict
def load_labels(R):
    """Official v1 K=3 classes under the 'last layer = real output' contract (INTERFACES 2026-10-16 evening; identical to
    classify.py --last_layer output), computed here with classify.classify_v1 so the result does not depend on when
    RUN/classify/questions.csv is regenerated; cross-checked against that file."""
    if 'labels' in R:
        return R['labels']
    D = CL.load_inputs(R['model'], R['fmt'], 'full', 'raw')
    L = D['L']
    p_let, f_let = CL.top1(D['probe']), CL.top1(D['out4'])
    p_let[:, L - 1] = f_let
    excl = ~CL.to_q(D['compliant']).all(1)
    answer = D['gold'][0::4]
    r = CL.classify_v1(CL.contents(CL.to_q(p_let)), CL.contents(CL.to_q(CL.top1(D['csd']))), CL.contents(CL.to_q(f_let)),
                       answer, 3, excl)
    qs = list(csv.DictReader(open(os.path.join(R['rd'], 'classify', 'questions.csv'))))
    o_csv = np.array([q['outcome_v1_K3'] for q in qs])
    R['labels'] = dict(o=r['outcome'], ua=r['u_avail'], fjc=r['fjc'], fpcl=r['fpcl'], fold=D['fold'][0::4], nfc=r['n_final'],
                       check=dict(csv_md5=C.md5_file(os.path.join(R['rd'], 'classify', 'questions.csv')),
                                  agree_with_csv=float((o_csv == r['outcome']).mean()),
                                  changed={f'{a}->{b}': int(((o_csv == a) & (r['outcome'] == b)).sum())
                                           for a in CL.V1 for b in CL.V1 if a != b and ((o_csv == a) & (r['outcome'] == b)).any()},
                                  counts={k: int((r['outcome'] == k).sum()) for k in CL.V1},
                                  U_avail=int(((r['outcome'] == 'U') & r['u_avail']).sum())))
    return R['labels']


def loop_labels(R, L):
    """Per-row fixes / changes under the w16_23_k1 analogue of loop_allq (mode last) vs the same-batch none run."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from loop import scale_name
    cfg = scale_name('w16_23_k1', L)
    d = os.path.join(R['rd'], 'loop_allq')
    if not all(os.path.exists(os.path.join(d, f'last_{c}', 'meta.json')) for c in ('none', cfg)):
        return None
    rows = np.load(os.path.join(d, 'last_none', 'rows.npy'))
    assert (rows == np.arange(R['N'])).all() and (np.load(os.path.join(d, f'last_{cfg}', 'rows.npy')) == rows).all()
    pre, post = [CL.top1(np.load(os.path.join(d, f'last_{c}', 'final4.npy'))) for c in ('none', cfg)]
    pc = os.path.join(d, 'eval', 'post_classes.csv')
    qn = qw = None
    if os.path.exists(pc):
        P = list(csv.DictReader(open(pc)))
        qn, qw = np.array([r['last_none'] for r in P]), np.array([r[f'last_{cfg}'] for r in P])
    return dict(cfg=cfg, pre=pre, post=post, q_none=qn, q_loop=qw)


def part_predict(R, e, lo=None, tasks=None):
    F = dict(np.load(os.path.join(R['od'], 'features.npz')))
    lab = load_labels(R)
    N, L = F['abne'].shape
    Q = N // 4
    gold, qid = R['gold'], np.arange(N) // 4
    qfold = lab['fold']
    lo = lo if lo is not None else max(0, e - 8)
    has_c = 'abne_c' in F
    LV = {'L2': ['abne', 'dis', 'kl', 'corr', 'corrW'], 'L0': ['u4', 'u4d', 'u4d_g', 'cmass', 'cmass_g']}
    if has_c:
        LV['L1'] = ['abne_c', 'dis_c', 'kl_c', 'corr_c'] + (['abne_cv'] if 'abne_cv' in F else [])
    CONF = {'L0': ['lmaxp', 'lent', 'cmaxp', 'cent'],
            'L1': ['lmaxp', 'lent', 'cmaxp', 'cent', 'pmaxp_c', 'pent_c'] + (['pmaxp_cv', 'pent_cv'] if 'abne_cv' in F else []),
            'L2': ['lmaxp', 'lent', 'cmaxp', 'cent', 'pmaxp', 'pent']}
    qm = lambda a: np.nanmean(a.reshape((Q, 4) + a.shape[1:]), 1)
    res = dict(config=dict(e=e, labels=lab['check'], layers_from=lo, levels=list(LV), consensus_probe=has_c, n_boot=B_BOOT, lockbox_fold=LOCK,
                           creg=SA.CREG, baseline_f='final max softmax + entropy', strict='+ layer-l max softmax/entropy '
                           'of lens, CSD (+ probe of the same level)'), tasks={})
    excl = lab['o'] == 'X'

    def run_auc_task(name, m, y, unit, layers, prespec):
        """m: rows/questions in the task; y: labels (same length as m); unit 'q' or 'row'."""
        if unit == 'q':
            G = lambda k: qm(F[k]) if F[k].ndim == 2 else qm(F[k][:, None])[:, 0]
            fold_, grp = qfold, np.arange(Q)
        else:
            G = lambda k: F[k]
            fold_, grp = R['fold'], qid
        dev, lock = m & (fold_ != LOCK), m & (fold_ == LOCK)
        yv = np.zeros(len(m), int)
        yv[m] = y
        bf = np.stack([G('fmaxp'), G('fent')], 1)
        r = dict(n_pos=int(y.sum()), n=int(m.sum()), n_pos_lock=int(yv[lock].sum()), n_lock=int(lock.sum()),
                 base_f=auc_block(bf[:, :1], yv, fold_, grp, bf, dev, lock), selected={}, prespecified={})
        cache = {}
        for lvl, names in LV.items():
            cand = {}
            for nm in names:
                A = G(nm)
                for l in layers:
                    x = A[:, l]
                    bs = np.c_[bf, np.stack([G(k)[:, l] for k in CONF[lvl]], 1)]
                    cand[f'{nm}@{l}'] = (x, bs)
            # selection on dev only: highest cross-fitted gain over base_f
            dm = np.flatnonzero(dev)
            if min(yv[dm].sum(), len(dm) - yv[dm].sum()) < 5:
                continue
            b_auc = SA.auc(yv[dm], dev_oof(bf[dm, :1], yv[dm], fold_[dm], bf[dm], 'base'))
            ok = {k: round(dev_gain(x[dm], yv[dm], fold_[dm], bf[dm], b_auc), 4) for k, (x, _) in cand.items()}
            best = max(ok, key=ok.get)
            x, bs = cand[best]
            r['selected'][lvl] = dict(feature=best, dev_gain_rank_top5=sorted(ok.items(), key=lambda kv: -kv[1])[:5],
                                      vs_base_f=auc_block(x[:, None], yv, fold_, grp, bf, dev, lock),
                                      vs_strict=auc_block(x[:, None], yv, fold_, grp, bs, dev, lock))
            # all features of the level at the selected layer, jointly (LR), vs base_f
            l_sel = int(best.split('@')[1])
            Xs = np.stack([G(nm)[:, l_sel] for nm in names], 1)
            r['selected'][lvl]['set_at_layer'] = auc_block(Xs, yv, fold_, grp, bf, dev, lock)
            cache[lvl] = cand
            print(f'    {name} {lvl}: selected {best} dev-gain {ok[best]} lock {r["selected"][lvl]["vs_base_f"].get("lock")}',
                  flush=True)
        for k in prespec:
            lvl = 'L0' if k.startswith(('u4', 'cmass')) else 'L1' if '_c' in k.split('@')[0] else 'L2'
            if lvl not in cache or k not in cache[lvl]:
                continue
            x, bs = cache[lvl][k]
            r['prespecified'][k] = dict(vs_base_f=auc_block(x[:, None], yv, fold_, grp, bf, dev, lock),
                                        vs_strict=auc_block(x[:, None], yv, fold_, grp, bs, dev, lock))
        return r

    win = [l for l in (e - 6, e - 4, e - 2, e - 1) if l >= 0]
    pre = [f'abne@{l}' for l in win] + [f'abne_c@{l}' for l in win] + [f'abne_cv@{l}' for l in win] + \
        [f'u4d_g@{e}', f'u4d@{e - 1}', f'cmass@{e - 1}']
    lay_all = list(range(lo, L))
    o, ua = lab['o'], lab['ua']
    T = res['tasks']
    want = lambda k: tasks is None or k in tasks
    t0 = time.time()
    # (b) question tasks
    if want('O_vs_U'):
        m = np.isin(o, ['O', 'U'])
        T['O_vs_U'] = run_auc_task('O_vs_U', m, (o == 'O')[m], 'q', lay_all, pre)
    if want('Uavail_vs_Unone'):
        m = o == 'U'
        T['Uavail_vs_Unone'] = run_auc_task('Uavail_vs_Unone', m, ua[m], 'q', lay_all, pre)
    print('  question tasks', C.stamp(t0), flush=True)
    LP = loop_labels(R, L) if any(want(k) for k in ('loopfix_wrong_rows', 'loop_direction_changed_rows',
                                                    'loop_U_to_correct_q', 'loop_U_to_R_q')) else None
    if LP is not None:
        from loop import parse_config
        b_ = parse_config(LP['cfg'], L)['b']
        lay_loop = list(range(lo, b_ + 1))
        pre_loop = [k for k in pre if int(k.split('@')[1]) <= b_]
        keep_r = ~np.repeat(excl, 4)
        wrong = (LP['pre'] != gold) & keep_r
        chg = (LP['post'] != LP['pre']) & keep_r & (LP['post'] >= 0)
        if want('loopfix_wrong_rows'):
            T['loopfix_wrong_rows'] = run_auc_task('loopfix_wrong_rows', wrong, (LP['post'] == gold)[wrong], 'row', lay_loop,
                                                   pre_loop)
        if want('loop_direction_changed_rows'):
            T['loop_direction_changed_rows'] = run_auc_task('loop_direction_changed_rows', chg, (LP['post'] == gold)[chg],
                                                            'row', lay_loop, pre_loop)
        if want('loop_U_to_correct_q') and LP['q_none'] is not None:   # loop_eval classes (old last-layer contract)
            m = LP['q_none'] == 'U'
            T['loop_U_to_correct_q'] = run_auc_task('loop_U_to_correct_q', m, np.isin(LP['q_loop'], ['R', 'O'])[m], 'q',
                                                    lay_loop, pre_loop)
        if want('loop_U_to_R_q'):   # contract-independent: U (last layer = output) and none run not 4/4 -> loop output 4/4
            n_pre, n_post = CL.to_q(LP['pre'] == gold).sum(1), CL.to_q(LP['post'] == gold).sum(1)
            m = (o == 'U') & (n_pre < 4)
            T['loop_U_to_R_q'] = run_auc_task('loop_U_to_R_q', m, (n_post == 4)[m], 'q', lay_loop, pre_loop)
        res['config']['loop'] = dict(cfg=LP['cfg'], b=b_, n_wrong=int(wrong.sum()), n_fixed=int((wrong & (LP['post'] == gold)).sum()),
                                     n_changed=int(chg.sum()))
        print('  loop tasks', C.stamp(t0), flush=True)
    else:
        res['config']['loop'] = 'loop_allq none / w16_23_k1 analogue not complete: skipped'
    # (a) emission timing. Question level: FJC_c - l among R/O questions not yet emitted at l. Row level: CSD commit layer
    #     (first layer from which the CSD top-1 equals the real output to the end; label-free) - l among rows not yet
    #     committed; and 'the probe class at l is the real output' among rows whose lens top-1 at l differs from it.
    agree = F['csd_top1'] == F['out_top1'][:, None]
    commit = L - np.cumprod(agree[:, ::-1], 1).sum(1)
    TM = {}
    for nm in (['abne', 'u4d', 'u4d_g', 'cmass'] + (['abne_c'] if has_c else [])) if want('timing') else []:
        lvl = 'L1' if nm.endswith('_c') else 'L2' if nm == 'abne' else 'L0'
        for unit in ('q_fjc', 'row_commit'):
            tab = {}
            for l in range(max(lo, e - 7), e + 1):
                if unit == 'q_fjc':   # R/O questions whose FJC_c is later than l
                    mm, x, t = np.isin(o, ['R', 'O']) & (lab['fjc'] > l), qm(F[nm])[:, l], lab['fjc'] - l
                    bb = np.stack([qm(F['fmaxp'][:, None])[:, 0], qm(F['fent'][:, None])[:, 0]] +
                                  [qm(F[k])[:, l] for k in CONF[lvl]], 1)
                    fo, gr = qfold, np.arange(Q)
                else:                 # rows whose CSD commit layer is later than l
                    mm, x, t = (commit > l) & ~np.repeat(excl, 4), F[nm][:, l], commit - l
                    bb = np.stack([F['fmaxp'], F['fent']] + [F[k][:, l] for k in CONF[lvl]], 1)
                    fo, gr = R['fold'], qid
                tab[l] = (spearman_block(x, t, bb, gr, mm & (fo != LOCK)), (x, t, bb, gr, mm & (fo == LOCK)))
            dev_tab = {l: v[0] for l, v in tab.items()}
            okl = {l: abs(v['partial_rho'][0]) for l, v in dev_tab.items() if 'partial_rho' in v}
            if not okl:
                continue
            l_sel = max(okl, key=okl.get)   # layer chosen on dev (largest |partial rho|), then scored once on fold 4
            TM[f'{nm}|{unit}'] = dict(level=lvl, dev=dev_tab, selected_layer=l_sel, lock=spearman_block(*tab[l_sel][1]))
            print(f'    timing {nm}|{unit}: L{l_sel} dev {dev_tab[l_sel]} lock {TM[f"{nm}|{unit}"]["lock"]}', flush=True)
    if TM:
        T['timing'] = TM
    # probe class at l eventually output (rows whose lens top-1 at l differs from the probe top-1)
    EM = {}
    for l in range(max(lo, e - 7), e) if want('probe_class_emitted') else []:
        mr = (F['lens_top1'][:, l] != F['k1'][:, l]) & ~np.repeat(excl, 4)
        y = (F['out_top1'] == F['k1'][:, l])[mr]
        EM[l] = run_auc_task(f'probe_class_emitted@{l}', mr, y, 'row', [l], [f'abne@{l}'])
    if EM:
        T['probe_class_emitted'] = EM
    # profiles (descriptive): version-mean feature by class per layer
    cls = {c: o == c for c in 'ROMU'} | {'U_avail': (o == 'U') & ua, 'U_none': (o == 'U') & ~ua}
    res['profiles'] = {k: {c: rnd(qm(F[k])[mm].mean(0)) for c, mm in cls.items() if mm.any()}
                       for k in ['abne', 'dis', 'corr', 'corrW', 'u4d', 'cmass', 'pmaxp'] + (['abne_c'] if has_c else [])}
    return res


# ----------------------------------------------------------------------------- causal (GPU, exploratory)
CONDS = ('perp', 'par', 'rand', 'perp_other')


def part_causal(R, model, fmt, e, layers, n_q, alphas, max_tokens, gb):
    """Rows: n_q random U_avail and n_q random R questions (x 4 versions). At block l the critical-position state h gets
    h + a * sqrt(d) * rms(h) * u (relative size a in RMS-normalized units), u a unit vector: 'perp' / 'par' = component of
    the probe's class direction for its top class k (class-centred E + W of the row's fold, hn space) orthogonal to / inside
    span(E); 'rand' = random; 'perp_other' = 'perp' for another random class k2. Blocks l+1..L-1 then run with the row's
    own prefix keys/values (loop.py mode last); per step: lens and CSD letter logits; final letters + full-vocab top-1."""
    import torch
    from loop import csd4, batches
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
    lab = load_labels(R)
    rng = np.random.default_rng(SEED)
    qa = np.flatnonzero((lab['o'] == 'U') & lab['ua'])
    qr = np.flatnonzero(lab['o'] == 'R')
    qids = np.sort(np.r_[rng.choice(qa, min(n_q, len(qa)), replace=False), rng.choice(qr, min(n_q, len(qr)), replace=False)])
    lm = C.LM(model, device='cpu')                     # fp32 vocab copy stays on the CPU; only bf16 weights go to the GPU
    lm.model.to('cuda'); lm.device = 'cuda'; lm.norm_w = lm.norm_w.cuda()
    qs = C.load_questions()
    rows = C.build_rows(model, fmt, lm.tok, [qs[q] for q in qids])
    rid = np.array([r['row'] for r in rows])
    target = C.target_text(fmt, lm.tok)
    lm.set_letters(lm.find_letter_ids(rows[0]['text'])); lm.U4 = lm.U4.cuda(); lm.WU = None
    WH = lm.model.lm_head.weight                       # bf16 (V, d) for the full-vocab top-1

    def vocab_top1(h):
        hn = lm.fnorm32(h)
        best, arg = torch.full((len(h),), -torch.inf, device='cuda'), torch.zeros(len(h), dtype=torch.long, device='cuda')
        for i in range(0, WH.shape[0], 16384):
            m, a = (hn @ WH[i:i + 16384].float().T).max(-1)
            best, arg = torch.where(m > best, m, best), torch.where(m > best, a + i, arg)
        return arg
    lm.vocab_stats = lambda h, chunk=None: (vocab_top1(h), torch.full((len(h),), float('nan'), device='cuda'))  # csd_setup
    idl = [lm.encode(r['text']) for r in rows]
    plen = np.array([len(x) for x in idl])
    assert np.array_equal(np.load(os.path.join(R['ex'], 'prompt_len.npy'))[rid], plen), 'prompts differ from extraction'
    L, d, dev = lm.L, lm.d, 'cuda'
    E = (lm.U4 * lm.norm_w[None]).double()
    Q4 = torch.from_numpy(R['Q4']).to(dev)
    Wl = torch.from_numpy(R['W'][:, layers]).to(dev).double()           # (5, nl, 4, d)
    bl = torch.from_numpy(R['b'][:, layers]).to(dev).double()
    fold = torch.from_numpy(R['fold'][rid]).to(dev)
    S = L - 1 - min(layers)
    nc = 1 + len(CONDS) * len(alphas)
    out = dict(lens=np.full((len(rid), len(layers), nc, S, 4), np.nan, np.float32), csd=np.full((len(rid), len(layers), nc, S, 4), np.nan, np.float32),
               final4=np.zeros((len(rid), len(layers), nc, 4), np.float32), top1=np.zeros((len(rid), len(layers), nc), np.int64),
               k=np.zeros((len(rid), len(layers)), np.int8), k2=np.zeros((len(rid), len(layers)), np.int8),
               relnorm_par=np.zeros((len(rid), len(layers)), np.float32))
    gen = torch.Generator(device=dev).manual_seed(SEED)
    t0 = time.time()

    with torch.no_grad():
        for bi, b in enumerate(batches(plen, max_tokens, 16)):
            ids, am, pos = lm.batch_tokens([idl[x] for x in b])
            B, T = ids.shape
            H, cache = lm.forward_capture(ids, am, pos, use_cache=True)
            kv = C.PrefixKV([cache.layers[i].keys[:, :, :T - 1] for i in range(L)],
                            [cache.layers[i].values[:, :, :T - 1] for i in range(L)])
            mask, lastpos = am.bool()[:, None, None, :], pos[:, -1:]
            bt = torch.from_numpy(b).to(dev)
            for li, l in enumerate(layers):
                h = H[:, l].double()
                rms = torch.sqrt(h.pow(2).mean(-1, keepdim=True) + R['eps'])
                hn = h / rms
                Wf, bf_ = Wl[fold[bt], li], bl[fold[bt], li]                     # (B, 4, d), (B, 4)
                z = lm.letters32(H[:, l]).double() + torch.einsum('bd,bkd->bk', hn, Wf) + bf_
                k = z.argmax(-1)
                k2 = (k + torch.randint(1, 4, (B,), generator=gen, device=dev)) % 4
                P = (E[None] + Wf) - (E[None] + Wf).mean(1, keepdim=True)        # class-centred directions
                dk, dk2 = P[torch.arange(B), k], P[torch.arange(B), k2]
                proj = lambda v: (v @ Q4) @ Q4.T
                unit = lambda v: v / v.norm(dim=-1, keepdim=True)
                U = dict(perp=unit(dk - proj(dk)), par=unit(proj(dk)), rand=unit(torch.randn(B, d, generator=gen, device=dev,
                                                                                            dtype=torch.float64)),
                         perp_other=unit(dk2 - proj(dk2)))
                out['k'][b, li], out['k2'][b, li] = k.cpu().numpy(), k2.cpu().numpy()
                out['relnorm_par'][b, li] = (proj(dk).norm(dim=-1) / dk.norm(dim=-1)).cpu().numpy()
                conds = [(0, None, 0.0)] + [(1 + ci * len(alphas) + ai, U[c], a) for ci, c in enumerate(CONDS)
                                            for ai, a in enumerate(alphas)]
                X, lay, meta_ = [], [], []
                for c, u, a in conds:
                    x = (h if u is None else h + a * d ** 0.5 * rms * u).to(torch.bfloat16)
                    tr = lm.run_single(x, list(range(l + 1, L)), kv, lastpos, mask=mask, record=True)[1]   # (B, L-1-l, d)
                    X.append(tr.reshape(-1, d)); lay.append(torch.arange(l + 1, L, device=dev).repeat(B)); meta_.append(c)
                    out['top1'][b, li, c] = vocab_top1(tr[:, -1]).cpu().numpy()
                cs = csd4(lm, torch.cat(X), torch.cat(lay), target, 4096)
                Sl = L - 1 - l
                for j, c in enumerate(meta_):
                    cj = cs[j * B * Sl:(j + 1) * B * Sl].reshape(B, Sl, 4).cpu().numpy()
                    le = lm.letters32(X[j]).reshape(B, Sl, 4).cpu().numpy()
                    out['csd'][b, li, c, S - Sl:], out['lens'][b, li, c, S - Sl:] = cj, le
                    out['final4'][b, li, c] = le[:, -1]
            del H, cache, kv
            if bi % 10 == 0:
                print(f'  causal batch {bi} rows {b[-1]} peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}GB', C.stamp(t0), flush=True)
    cd = os.path.join(R['od'], 'causal')
    os.makedirs(cd, exist_ok=True)
    np.savez(os.path.join(cd, 'causal.npz'), rows=rid, qids=qids, layers=np.array(layers), alphas=np.array(alphas),
             conds=np.array(['none'] + [f'{c}@{a}' for c in CONDS for a in alphas]), letter_ids=np.array(lm.letter_ids), **out)
    return dict(n_rows=len(rid), n_q=len(qids), peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30, seconds=time.time() - t0)


def causal_report(R, e):
    """Effect of each condition vs the same-batch unperturbed run (bootstrap by question). Emission of class c = first layer
    from which the CSD (or lens) top-1 equals c through the last layer (L = never)."""
    Z = dict(np.load(os.path.join(R['od'], 'causal', 'causal.npz')))
    lab = load_labels(R)
    rid, layers, conds = Z['rows'], list(Z['layers']), list(Z['conds'])
    gold = R['gold'][rid]
    q = rid // 4
    uq, inv = np.unique(q, return_inverse=True)
    grp = {'U_avail': (lab['o'][q] == 'U'), 'R': lab['o'][q] == 'R'}
    Wb = CL.boot_weights(len(uq))[:, inv]
    S = Z['csd'].shape[3]
    L = S + 1 + min(layers)
    compliant = np.isin(Z['top1'], Z['letter_ids'])
    fin = np.where(compliant, Z['final4'].argmax(-1), -1)                    # (n, nl, nc)

    def commit(top, c):
        """top (n, nc, S) top-1 per step (NaN-padded steps = -9), c (n,) class -> absolute layer of commitment (L = never)."""
        hit = top == c[:, None, None]
        run = np.cumprod(hit[..., ::-1], -1).sum(-1)                           # length of the final run on c
        return L - run
    res = {}
    for li, l in enumerate(layers):
        Sl = L - 1 - l
        ctop = Z['csd'][:, li, :, S - Sl:].argmax(-1)
        ltop = Z['lens'][:, li, :, S - Sl:].argmax(-1)
        k, k2 = Z['k'][:, li].astype(int), Z['k2'][:, li].astype(int)
        em = {'csd_k': commit(ctop, k), 'lens_k': commit(ltop, k), 'csd_k2': commit(ctop, k2)}
        for g, m in grp.items():
            w = Wb * m
            stat = lambda x: [round(float(v), 4) for v in (x[m].mean(), *np.percentile((w @ x) / w.sum(1), [2.5, 97.5]))]
            rr = dict(n_rows=int(m.sum()), probe_class_is_gold=round(float((k == gold)[m].mean()), 4),
                      relnorm_par=round(float(Z['relnorm_par'][m, li].mean()), 4), conds={})
            base = dict(fk=(fin[:, li, 0] == k).astype(float), fg=(fin[:, li, 0] == gold).astype(float),
                        fk2=(fin[:, li, 0] == k2).astype(float), ek=em['csd_k'][:, 0].astype(float),
                        lk=em['lens_k'][:, 0].astype(float), ek2=em['csd_k2'][:, 0].astype(float))
            for ci, c in enumerate(conds):
                cur = dict(fk=(fin[:, li, ci] == k).astype(float), fg=(fin[:, li, ci] == gold).astype(float),
                           fk2=(fin[:, li, ci] == k2).astype(float), ek=em['csd_k'][:, ci].astype(float),
                           lk=em['lens_k'][:, ci].astype(float), ek2=em['csd_k2'][:, ci].astype(float))
                rr['conds'][c] = dict(final_is_k=stat(cur['fk']), d_final_is_k=stat(cur['fk'] - base['fk']),
                                      d_final_is_gold=stat(cur['fg'] - base['fg']),
                                      d_final_is_k2=stat(cur['fk2'] - base['fk2']),
                                      csd_commit_k_by_e=stat((cur['ek'] <= e).astype(float)),
                                      d_csd_commit_layer_k=stat(cur['ek'] - base['ek']),
                                      d_lens_commit_layer_k=stat(cur['lk'] - base['lk']),
                                      d_csd_commit_layer_k2=stat(cur['ek2'] - base['ek2']),
                                      noncompliant=stat((~compliant[:, li, ci]).astype(float)))
            res[f'L{l}|{g}'] = rr
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
    ap.add_argument('--parts', default='scan,features,geometry,predict', help='also: causal, causal_report')
    ap.add_argument('--layers', default='20,21,22'); ap.add_argument('--n_q', type=int, default=100)
    ap.add_argument('--alphas', default='0.05,0.1,0.2'); ap.add_argument('--max_tokens', type=int, default=6000)
    ap.add_argument('--gb', type=float, default=9.5, help='GPU memory cap of the causal part (allocator)')
    ap.add_argument('--tasks', default='', help='predict: comma list of tasks (default all) -> predict_{tasks}.json')
    a = ap.parse_args()
    t0 = time.time()
    R = load_run(a.model, a.fmt)
    os.makedirs(R['od'], exist_ok=True)
    e, agree = emission_layer(np.load(os.path.join(R['ex'], 'csd_raw4.npy')), np.load(os.path.join(R['ex'], 'out_logits4.npy')))
    print(f'{a.model}__{a.fmt}{"" if C.BENCH == "core" else "__" + C.BENCH}: N={R["N"]} emission layer e={e} '
          f'(CSD-output agreement {agree[e - 1]:.3f} -> {agree[e]:.3f})', flush=True)
    meta = dict(model=a.model, fmt=a.fmt, bench=C.BENCH, e=e, csd_output_agree=rnd(agree), script_md5=C.md5_file(__file__),
                common_md5=C.md5_file(C.__file__), classify_md5=C.md5_file(CL.__file__), stage2_analysis_md5=C.md5_file(SA.__file__))
    parts = a.parts.split(',')
    if 'scan' in parts and not os.path.exists(os.path.join(R['od'], 'scan.npz')):
        part_scan(R, a.model, a.fmt, e); print('scan done', C.stamp(t0), flush=True)
    if 'features' in parts:
        meta['features'] = part_features(R, e); print('features done', meta['features'], C.stamp(t0), flush=True)
    if 'geometry' in parts:
        G = part_geometry(R, e)
        C.save_json(jsonable(dict(meta=meta, seconds=round(time.time() - t0, 1), **G)), os.path.join(R['od'], 'geometry.json'))
        print('geometry done', json.dumps(G['amplification_at_e']), C.stamp(t0), flush=True)
    if 'predict' in parts:
        tk = a.tasks.split(',') if a.tasks else None
        P = part_predict(R, e, tasks=tk)
        C.save_json(jsonable(dict(meta=meta, seconds=round(time.time() - t0, 1), **P)),
                    os.path.join(R['od'], 'predict.json' if tk is None else f'predict_{"+".join(tk)}.json'))
        print('predict done', C.stamp(t0), flush=True)
    if 'causal' in parts:
        meta['causal'] = part_causal(R, a.model, a.fmt, e, [int(x) for x in a.layers.split(',')], a.n_q,
                                     [float(x) for x in a.alphas.split(',')], a.max_tokens, a.gb)
        print('causal done', meta['causal'], C.stamp(t0), flush=True)
    if 'causal' in parts or 'causal_report' in parts:
        CR = causal_report(R, e)
        C.save_json(jsonable(dict(meta=meta, args=vars(a), **CR)), os.path.join(R['od'], 'causal', 'report.json'))
        for k, v in CR.items():
            print(k, 'n', v['n_rows'], 'k==gold', v['probe_class_is_gold'], ' '.join(
                f"{c}: dk {x['d_final_is_k'][0]:+.3f} dg {x['d_final_is_gold'][0]:+.3f} dcommit {x['d_csd_commit_layer_k'][0]:+.2f}"
                for c, x in v['conds'].items() if c != 'none'), flush=True)
    print('DONE', C.stamp(t0), flush=True)


if __name__ == '__main__':
    main()
