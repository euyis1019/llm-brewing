"""Stage 2: who benefits from the depth loop, a gated loop, and perturbation stability as a resolution indicator. CPU.

Inputs: RUN/{loop_dir}/last_*/ (loop.py --mode last on all rows; reference = the same-batch `none` run, i.e. the same
numerics as every loop config), official v1 K=3 classes (RUN/classify/questions.csv if made with --last_layer output,
else recomputed here with classify.classify_v1 under that rule), RUN/extract/{rows.json, meta.json, csd_base4.npy,
lens_params.npz}, loop_eval's post_classes.csv (only if its classes match). Rows of X questions are left out.
Per row and config c: letter = 4-way argmax of final4 (ties -1, never correct); change = letter != none letter; fix =
none wrong -> c right; break = right -> wrong; gain = fix - break. 'strict': a row whose full-vocabulary top-1 is not a
letter counts as wrong. Gold only scores fix/break/correctness; it never selects anything applied to the scored rows.
Supervision: L0 = no training (none-pass readouts, letter prior csd_base4, hidden geometry); L1 = trained on a target
from the model's own outputs: the 4-version permutation consensus, leave-one-out (unique plurality content of the other 3
versions' none outputs, else undefined; the self-inclusive plurality scores every change of a majority row as a loss,
see proxy_* in the config table); L3 = trained on gold. Feature/sign/threshold choices made on training folds by gold net gain are marked 'gold-tuned'.
Lockbox: fold 4. Default = folds 0-3 only, cross-fitted by fold; --lockbox adds fold 4 (and all folds for the purely
descriptive parts) with every choice (configs, features, signs, thresholds, models) made on folds 0-3. Selection:
best_unsup / best_L0L1 pick family x config by dev GOLD net gain; best_unsup_goldfree / always_goldfree by the dev
consensus net gain (no gold anywhere).
(a) Perturbation stability (L0, but one extra mode-last loop per window config): stab = fraction of window configs
    (w*, d*) leaving the row's letter unchanged, stab_q = fraction leaving all 4 versions unchanged, soft = mean
    post-loop probability of the none letter, kl = mean KL(none || loop). By class; Spearman vs FJC_c (R/O) and vs the
    label-free CSD commit layer; AUROC for correctness (row: none letter correct; question: 4/4 correct, R vs rest,
    M vs U = resolved-wrong vs unresolved) next to confidence (log 4-way max softmax of the final letters) and the L1
    permutation consensus (question level); AUROC within confidence quintiles; single-config 'unchanged' indicators incl.
    the late placebo; L3 = logistic regression on confidence + stability, trained on correctness.
(b) Gated loop per config (gate fires -> the loop's letter). L0 features of the none pass at the window start a, end b
    and the final layer: CSD/lens log max softmax, margin, entropy, KL to the previous layer, top-1 run, agreement with
    the final letter; commit layer and flips from w0; prior prob of the final letter, KL(final || prior), baseline-
    subtracted CSD at a; hidden RMS norm, cos and relative update to the previous block, letter-subspace energy of the
    state and of the update. Gates (score s, loop where s > t): conf (-final max softmax), L0 (best single feature x
    sign), L1 (multinomial logistic regression of the consensus gain: +1 the loop moves the row onto the consensus
    content, -1 off it; s = P(+1) - P(-1)), L3 (same model, gold gain), L3h (L3 + PCA(64) of the RMS-normalized state at
    a and L-1). Thresholds: gold-tuned on the training folds (on inner out-of-fold scores for trained gates); '@0':
    s > 0 (expected gain > 0); 'u': tuned on the consensus gain (no gold anywhere). Multi-pass L0 reference: vote =
    plurality letter over none + all window configs. Net accuracy vs never/always, coverage, fixes/breaks, paired
    difference to the conf gate, fix-vs-break AUROC, Pareto (top-q fraction of each fold by score). Multi-pass
    references: xv4 (L1, 4 passes: loop iff the 4 versions' none outputs disagree), permutation vote without loop.
(2) ABNE: U_avail (FPCL_c exists, no FJC_c) vs U_none (+ R/O/M = who benefits): row fix/break/net vs none and minus the
    late placebo, question rescue (none < 4/4 -> 4/4), U_none reweighted to U_avail's none n_correct, U_avail by FPCL_c
    vs the window, trajectory-based post-loop class R/O (loop_eval). Bootstrap 95% CIs by question (2000).
Outputs RUN/{out}/summary{_lockbox}.json, rows{_lockbox}.npz (OOF gate scores, post letters, stability).
Usage: python scripts/stage2_loopgate.py --model qwen3-4b-base --fmt base_fewshot [--loop_dir loop_allq] [--lockbox]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
import argparse, csv, glob, json, sys, time, warnings
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import classify as CL
from loop import scale_name

DEV, LOCK, B_BOOT, SEED, CREG, NPCA = (0, 1, 2, 3), 4, 2000, 0, 0.1, 64
GATE36, PRIMARY36, DAMPED36 = ('w16_23_k1', 'w18_25_k1', 'w20_27_k1', 'w18_25_k2', 'd18_25_k2'), 'w18_25_k2', 'd18_25_k2'
COVER = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
FAMILIES = {  # name -> (supervision of the score, how the threshold / feature is chosen)
    'conf': ('L0', 'gold-tuned threshold'), 'conf_u': ('L0', 'consensus-tuned threshold (no gold)'),
    'L0': ('L0', 'gold-tuned feature, sign, threshold'), 'L0u': ('L0', 'consensus-tuned feature, sign, threshold (no gold)'),
    'L1': ('L1', 'gold-tuned threshold'), 'L1@0': ('L1', 'expected consensus gain > 0 (no gold)'),
    'L1u': ('L1', 'consensus-tuned threshold (no gold)'), 'L3': ('L3', 'gold-tuned threshold'),
    'L3@0': ('L3', 'expected gain > 0'), 'L3h': ('L3', 'gold-tuned threshold; + hidden PCA features'),
    'xv4': ('L1, 4 passes', "no tuning: loop iff the 4 versions' none outputs are not all on one content")}
UNSUP = ('conf_u', 'L0u', 'L1@0', 'L1u')
warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------------- statistics
def auc(y, s, W=None):
    """AUROC of score s for binary y (ties count 1/2); W (B, n) bootstrap multiplicities -> (B,) values."""
    o = np.argsort(s, kind='mergesort')
    s, y = s[o], y[o].astype(bool)
    g = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    w = np.ones((1, len(s)), np.float32) if W is None else W[:, o]
    pos, neg = np.add.reduceat(w * y, g, 1), np.add.reduceat(w * ~y, g, 1)
    a = (pos * (np.cumsum(neg, 1) - 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))
    return float(a[0]) if W is None else a


def r5(x):
    return [None if x_ is None or not np.isfinite(x_) else round(float(x_), 5) for x_ in x]


def boot_q(mq):
    """(B, Q) by-question bootstrap multiplicities over the questions in mask mq (0 elsewhere)."""
    W, i = np.zeros((B_BOOT, len(mq))), np.flatnonzero(mq)
    W[:, i] = CL.boot_weights(len(i))
    return W


def auc_rows(y, scores, m, W, ref=None):
    """{name: [AUROC, lo, hi]} of each score for y over rows m (by-question bootstrap), plus paired differences to ref."""
    i = np.flatnonzero(m)
    Wr, yy, out = W[:, i // 4].astype(np.float32), y[i], {}
    for k, s in scores.items():
        b = auc(yy, s[i], Wr)
        out[k] = r5([auc(yy, s[i]), *np.nanpercentile(b, [2.5, 97.5])])
        if ref is not None and k != ref:
            d = b - auc(yy, scores[ref][i], Wr)
            out[f'{k}-{ref}'] = r5([out[k][0] - auc(yy, scores[ref][i]), *np.nanpercentile(d, [2.5, 97.5])])
    return out


def strat_auc(y, s, strata, m, W):
    """Mean within-stratum AUROC (strata: label-free confidence quintiles) [pt, lo, hi]."""
    pt, bs = [], []
    for k in np.unique(strata[m]):
        j = np.flatnonzero(m & (strata == k))
        if 0 < y[j].sum() < len(j):
            pt.append(auc(y[j], s[j])); bs.append(auc(y[j], s[j], W[:, j // 4].astype(np.float32)))
    return r5([np.mean(pt), *np.nanpercentile(np.nanmean(bs, 0), [2.5, 97.5])])


def spear(x, y, qs, rowlevel, B=500):
    """Spearman rho [pt, lo, hi] over the rows (rowlevel) or entries of questions qs; bootstrap over questions."""
    idx = (lambda q: (4 * q[:, None] + np.arange(4)).ravel()) if rowlevel else (lambda q: q)
    rng = np.random.default_rng(SEED)
    bs = [spearmanr(x[idx(q)], y[idx(q)])[0] for q in (qs[rng.integers(0, len(qs), len(qs))] for _ in range(B))]
    return r5([spearmanr(x[idx(qs)], y[idx(qs)])[0], *np.nanpercentile(bs, [2.5, 97.5])])


def pstats(g, gain, m, W, ref=None):
    """Net accuracy change (fraction of rows m) of using the loop letter where g; coverage; paired diff to gate ref."""
    gm = g & m
    num, den = CL.to_q(gm * gain).sum(1), CL.to_q(m).sum(1)
    r = dict(net=CL.ci(W, num, den), coverage=CL.ci(W, CL.to_q(gm).sum(1), den), n_loop=int(gm.sum()),
             fixes=int((gm & (gain > 0)).sum()), breaks=int((gm & (gain < 0)).sum()))
    if ref is not None:
        nr = CL.to_q((ref & m) * gain).sum(1)
        r['minus_ref'] = CL.wci(lambda w: CL.ratio(num, den)(w) - CL.ratio(nr, den)(w), W)
    return r


def rate(W, num, den, sel):
    return CL.ci(W, num * sel, den * sel)


def rate_diff(W, n1, d1, s1, n2, d2, s2):
    """Paired bootstrap of ratio(n1*s1, d1*s1) - ratio(n2*s2, d2*s2)."""
    return CL.wci(lambda w: CL.ratio(n1 * s1, d1 * s1)(w) - CL.ratio(n2 * s2, d2 * s2)(w), W)


# ----------------------------------------------------------------------------- inputs
def official_classes(model, fmt, rd):
    """v1 K=3 classes under the official rule (INTERFACES: last layer = real output): RUN/classify/questions.csv if it was
    made with --last_layer output, else recomputed here with classify.classify_v1 on the same inputs and rule."""
    if json.load(open(os.path.join(rd, 'classify', 'summary.json')))['config'].get('last_layer') == 'output':
        qs = list(csv.DictReader(open(os.path.join(rd, 'classify', 'questions.csv'))))
        col = lambda k, f=str: np.array([f(r[k]) for r in qs])
        return dict(cls=col('outcome_v1_K3'), ua=col('u_avail_K3') == '1', fpcl=col('FPCL_c_K3', int),
                    fjc=col('FJC_c_K3', int), class_source='classify/questions.csv (--last_layer output)')
    I = CL.load_inputs(model, fmt, 'full', 'raw')
    p, f = CL.top1(I['probe']), CL.top1(I['out4'])
    p[:, I['L'] - 1] = f
    cont = lambda x: CL.contents(CL.to_q(x))
    r = CL.classify_v1(cont(p), cont(CL.top1(I['csd'])), cont(f), I['gold'][0::4], 3, ~CL.to_q(I['compliant']).all(1))
    return dict(cls=r['outcome'], ua=r['u_avail'], fpcl=r['fpcl'], fjc=r['fjc'],
                class_source='recomputed here: classify.classify_v1, K=3, probe full, CSD raw, last layer = real output')


def load(model, fmt, loop_dir):
    rd = C.run_dir(model, fmt)
    ex = os.path.join(rd, 'extract')
    meta, rows = json.load(open(os.path.join(ex, 'meta.json'))), json.load(open(os.path.join(ex, 'rows.json')))
    N, L = meta['N'], meta['L']
    D = dict(rd=rd, ex=ex, meta=meta, N=N, L=L, Q=N // 4, d=meta['d'], gold=np.array([r['gold'] for r in rows]),
             fold=np.array([r['fold'] for r in rows]), **official_classes(model, fmt, rd))
    loops = {}
    for d in sorted(glob.glob(os.path.join(rd, loop_dir, 'last_*'))):
        if not os.path.exists(os.path.join(d, 'meta.json')):
            continue
        m = json.load(open(os.path.join(d, 'meta.json')))
        assert np.array_equal(np.load(os.path.join(d, 'rows.npy')), np.arange(N)), f'{d}: need all rows in order'
        f4 = np.load(os.path.join(d, 'final4.npy'))
        loops[m['config']['name']] = dict(cfg=m['config'], dir=d, S=m['S'], f4=f4, let=CL.top1(f4),
                                         comp=np.isin(np.load(os.path.join(d, 'final_top1.npy')), meta['letter_ids']))
    assert 'none' in loops, 'need the same-batch none run'
    return D, loops


def outcomes(D, loops):
    """ok / gain / strict gain / consensus gain of every config vs the none run. Consensus of a row (L1 target) = unique
    plurality content of the OTHER 3 versions' none outputs (leave-one-out; -2 = undefined). The self-inclusive 4-version
    plurality ('all') counts every change of a majority row as a loss, so it is kept only as a calibration reference."""
    gold, n0, N = D['gold'], loops['none'], D['N']
    ok0, ok0s = n0['let'] == gold, (n0['let'] == gold) & n0['comp']
    own = CL.contents(CL.to_q(n0['let']))[..., None] == np.arange(4)  # (Q, 4 versions, 4 contents)
    votes = own.sum(1)
    mx, v = votes.max(1), np.arange(N) % 4

    def letter_of(cnt):  # unique plurality content per row -> letter showing it (-2: undefined)
        c = np.where((cnt == cnt.max(-1, keepdims=True)).sum(-1) == 1, cnt.argmax(-1), -1).reshape(-1)
        return np.where(c >= 0, (c - v) % 4, -2)
    consL, consL_all = letter_of(votes[:, None, :] - own), letter_of(np.repeat(votes, 4, 0).reshape(-1, 4, 4))
    for x in loops.values():
        x['ok'] = x['let'] == gold
        x['gain'] = x['ok'].astype(np.int8) - ok0
        x['gain_s'] = (x['ok'] & x['comp']).astype(np.int8) - ok0s
        x['gain_c'] = (x['let'] == consL).astype(np.int8) - (n0['let'] == consL)
        x['gain_c_all'] = (x['let'] == consL_all).astype(np.int8) - (n0['let'] == consL_all)
    D.update(ok0=ok0, consL=consL, consL_all=consL_all, out_n=mx)


def softmax(z):
    z = np.asarray(z, np.float64)
    p = np.exp(z - z.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def readouts(z, fin, w0):
    """Per-layer L0 features of letter logits z (N, L, 4) (fp64; log max softmax does not saturate); fin: final letter."""
    z = np.asarray(z, np.float64)
    lp = z - logsumexp(z, -1, keepdims=True)
    p, s, t = np.exp(lp), np.sort(z, -1), CL.top1(z)
    run = np.ones(t.shape)
    for l in range(1, t.shape[1]):
        run[:, l] = np.where((t[:, l] == t[:, l - 1]) & (t[:, l] >= 0), run[:, l - 1] + 1, 1)
    kl = np.zeros(t.shape)
    kl[:, 1:] = (p[:, 1:] * (lp[:, 1:] - lp[:, :-1])).sum(-1)
    agree = t == fin[:, None]
    return dict(lmaxp=lp.max(-1), margin=s[..., -1] - s[..., -2], ent=-(p * lp).sum(-1), kl=kl, run=run, top1=t,
                agree=agree, commit=t.shape[1] - np.cumprod(agree[:, ::-1], 1).sum(1),
                flips=(t[:, w0 + 1:] != t[:, w0:-1]).sum(1))


def geometry(hid, layers, lp, cache):
    """Hidden geometry of the none pass at each layer l (needs l-1): RMS norm, cos / relative update to l-1, energy
    fraction of the state and of the update in span(norm_w * U4) (the subspace that moves the lens letter logits)."""
    E = lp['U4'].astype(np.float64) * lp['norm_w'].astype(np.float64)[None]
    Q4 = np.linalg.qr(E.T)[0].astype(np.float32)
    get = lambda l: cache[l] if l in cache else cache.setdefault(l, C.bf16_bits_to_f32(hid[:, l]))
    out = {}
    for l in layers:
        h, hp = get(l), get(l - 1)
        dl = h - hp
        n2, p2, d2 = (h * h).sum(1), (hp * hp).sum(1), (dl * dl).sum(1)
        out[l] = dict(h_norm=np.sqrt(n2 / h.shape[1]), h_cos=(h * hp).sum(1) / np.sqrt(n2 * p2), h_upd=np.sqrt(d2 / p2),
                      h_u4=((h @ Q4) ** 2).sum(1) / n2, h_updu4=((dl @ Q4) ** 2).sum(1) / d2)
    return out


def pca(h, fit, eps, k=NPCA):
    """PCA(k) scores of the z-scored RMS-normalized states h (N, d), fit on rows `fit` (label-free)."""
    h = h / np.sqrt((h * h).mean(1, keepdims=True) + eps)
    mu, sd = h[fit].mean(0), h[fit].std(0) + 1e-6
    h = (h - mu) / sd
    return h @ randomized_svd(h[fit], k, random_state=SEED)[2].T


def gate_features(R, G, f4, base4, fin, a, b, L):
    """(names, X (N, F)) L0 features of the none pass for the window [a, b]."""
    F = {}
    for r, A in R.items():
        for tag, l in (('a', a), ('b', b)):
            for k in ('lmaxp', 'margin', 'ent', 'kl', 'run', 'agree'):
                F[f'{r}_{k}@{tag}'] = A[k][:, l]
        F[f'{r}_commit'], F[f'{r}_flips'] = A['commit'], A['flips']
    for k in ('lmaxp', 'margin', 'ent'):
        F[f'fin_{k}'] = R['csd'][k][:, L - 1]
    pb, pf = softmax(base4), softmax(f4)
    fi = np.where(fin >= 0, fin, pf.argmax(1))
    zs = softmax(R['csd']['z_a'] - base4)
    F.update(prior_p_fin=pb[fi], prior_is_fin=fi == pb.argmax(), kl_fin_prior=(pf * (np.log(pf) - np.log(pb))).sum(1),
             prior_p_csd_a=pb[R['csd']['z_a'].argmax(1)], csdsub_lmaxp_a=np.log(zs.max(1)), csdsub_agree_a=zs.argmax(1) == fin)
    for tag, l in (('a', a), ('b', b), ('fin', L - 1)):
        for k, v in G[l].items():
            F[f'{k}@{tag}'] = v
    return list(F), np.nan_to_num(np.stack([np.asarray(v, np.float64) for v in F.values()], 1))


# ----------------------------------------------------------------------------- gates
def fit(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=CREG, max_iter=3000)).fit(X, y)


def egain(m, X):
    """Expected gain P(+1) - P(-1) of a fitted {-1, 0, 1} classifier."""
    p, k = m.predict_proba(X), list(m.classes_)
    return sum(sg * p[:, k.index(sg)] for sg in (1, -1) if sg in k) + np.zeros(len(X))


def best_cut(s, g):
    """(t, net): threshold maximizing net = sum(g[s > t]); (+inf, 0) = never fire unless some cut is positive; ties in
    s are never split."""
    o = np.argsort(-s, kind='stable')
    ss, cg = s[o], np.cumsum(g[o], dtype=np.float64)
    ok = np.r_[ss[1:] < ss[:-1], True]
    i = np.flatnonzero(ok)[cg[ok].argmax()]
    if cg[i] <= 0:
        return np.inf, 0.0
    return (0.5 * (ss[i] + ss[i + 1]) if i + 1 < len(ss) else -np.inf), float(cg[i])


def single_gate(X, cand, fold, keep, gsel, lockbox):
    """Gate on one feature: (column, sign, threshold) maximizing the net gain gsel on the training folds' rows.
    Returns score (N,), gate (N,) (dev rows cross-fitted; fold-4 rows from the choice on all dev folds), choices."""
    s, g, ch = np.zeros(len(fold)), np.zeros(len(fold), bool), {}
    for f in list(DEV) + ([LOCK] if lockbox else []):
        m = keep & np.isin(fold, [h for h in DEV if h != f])
        (t, _), j, sg = max(((best_cut(sg * X[m, j], gsel[m]), j, sg) for j, sg in cand), key=lambda z: z[0][1])
        te = fold == f
        s[te] = sg * X[te, j]
        g[te] = s[te] > t
        ch[f] = (int(j), int(sg), float(t))
    return s, g, ch


def trained_gate(X, y, fit_ok, fold, keep, gains, lockbox):
    """Multinomial logistic regression of y in {-1, 0, 1}, score = P(1) - P(-1). Dev fold f: model fit on the other dev
    folds; thresholds per entry of gains (name -> per-row gain) tuned on that model's inner out-of-fold scores;
    '@0' = score > 0. Fold 4 (lockbox): model on all dev folds, thresholds from the dev out-of-fold scores."""
    N = len(y)
    s, G, thr = np.zeros(N), {k: np.zeros(N, bool) for k in list(gains) + ['@0']}, {}
    fit_on = lambda T: fit(X[fit_ok & np.isin(fold, T)], y[fit_ok & np.isin(fold, T)])
    for f in DEV:
        T, inner = [h for h in DEV if h != f], np.zeros(N)
        for h in T:
            inner[fold == h] = egain(fit_on([x for x in T if x != h]), X[fold == h])
        te, mT = fold == f, keep & np.isin(fold, T)
        s[te] = egain(fit_on(T), X[te])
        for k, gk in gains.items():
            thr.setdefault(k, {})[f] = t = best_cut(inner[mT], gk[mT])[0]
            G[k][te] = s[te] > t
        G['@0'][te] = s[te] > 0
    if lockbox:
        te, mD = fold == LOCK, keep & np.isin(fold, DEV)
        s[te] = egain(fit_on(list(DEV)), X[te])
        for k, gk in gains.items():
            thr[k][LOCK] = t = best_cut(s[mD], gk[mD])[0]
            G[k][te] = s[te] > t
        G['@0'][te] = s[te] > 0
    return s, G, {k: {f: round(float(t), 5) for f, t in v.items()} for k, v in thr.items()}


def pareto(s, gain, m, fold, W):
    """Net gain when looping the top-q fraction of each fold's rows by score s (random tie-breaking)."""
    r = np.zeros(len(s))
    jit = np.random.default_rng(SEED).random(len(s))
    for f in np.unique(fold[m]):
        i = np.flatnonzero(m & (fold == f))
        o = np.lexsort((jit[i], -s[i]))
        r[i[o]] = (np.arange(len(i)) + 1) / len(i)
    return {q: pstats(r <= q, gain, m, W)['net'] for q in COVER}


# ----------------------------------------------------------------------------- analyses
def stability(D, loops, WC, late, sets, Wq, R, keep):
    """(a) perturbation stability as a label-free resolution indicator."""
    N, Q, ok0, let0, cls = D['N'], D['Q'], D['ok0'], loops['none']['let'], D['cls']
    P0 = softmax(loops['none']['f4'])
    li = np.where(let0 >= 0, let0, P0.argmax(1))
    unch = np.stack([loops[c]['let'] == let0 for c in WC], 1)
    Pc = [softmax(loops[c]['f4']) for c in WC]
    st = dict(stab=unch.mean(1), soft=np.mean([p[np.arange(N), li] for p in Pc], 0),
              kl=np.mean([(P0 * (np.log(P0 + 1e-300) - np.log(p + 1e-300))).sum(1) for p in Pc], 0))
    st['stab_q'] = CL.to_q(unch).all(1).mean(1)
    conf, margin = R['csd']['lmaxp'][:, -1], R['csd']['margin'][:, -1]
    rows = dict(conf=conf, margin=margin, negent=-R['csd']['ent'][:, -1], stab=st['stab'], soft=st['soft'],
                negkl=-st['kl'])
    rows['rank(conf)+rank(stab)'] = np.argsort(np.argsort(conf)) + np.argsort(np.argsort(st['stab'] + 1e-9 * conf))
    for c in WC + ([late] if late else []):
        rows[f'unch_{c}'] = (loops[c]['let'] == let0).astype(float)
    qsc = dict(stab_q=st['stab_q'], stab_qmean=CL.to_q(st['stab']).mean(1), conf_q=CL.to_q(conf).mean(1),
               consensus_n_L1=D['out_n'].astype(float))
    n0 = CL.to_q(ok0).sum(1)
    out = dict(window_configs=WC, late_placebo=late, note='conf = log 4-way max softmax of the none final letters (L0); '
               'stab*, soft, kl need one mode-last loop per window config (L0, no training); consensus_n_L1 = number of '
               'versions on the plurality content (4 passes; for 4/4-correct and R it is near-definitional)')
    for k, m in sets.items():
        mq, W = CL.to_q(m).any(1), Wq[k]
        i_q = np.flatnonzero(mq)
        cl = {c: (cls == c) & mq for c in 'ROMU'} | {'U_avail': (cls == 'U') & D['ua'] & mq,
                                                      'U_none': (cls == 'U') & ~D['ua'] & mq}
        o = dict(by_class={c: dict(n=int(v.sum()), stab_row=CL.ci(W, CL.to_q(st['stab']).sum(1) * v, 4 * v),
                                   stab_q=CL.ci(W, st['stab_q'] * v, v), conf_row=CL.ci(W, CL.to_q(conf).sum(1) * v, 4 * v))
                           for c, v in cl.items()})
        rl = auc_rows(ok0, rows, m, W, ref='conf')
        edges = np.quantile(conf[m], [.2, .4, .6, .8])
        strata = np.searchsorted(edges, conf)
        # L3 reference: logistic regression on confidence + stability trained on correctness (cross-fitted by fold)
        X = np.stack([rows[k_] for k_ in ('conf', 'margin', 'negent', 'stab', 'soft', 'negkl')], 1)
        s3 = np.zeros(N)  # dev rows: model on the other dev folds; fold 4: model on all dev folds
        for f in np.unique(D['fold'][m]):
            tr = keep & np.isin(D['fold'], [h for h in DEV if h != f])
            mod = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr], ok0[tr])
            s3[D['fold'] == f] = mod.decision_function(X[D['fold'] == f])
        rl.update(auc_rows(ok0, dict(L3_conf_stab=s3, conf=conf), m, W, ref='conf'))
        o['row_correct_auc'] = rl
        o['row_correct_auc_within_conf_quintiles'] = {k_: strat_auc(ok0, rows[k_], strata, m, W)
                                                      for k_ in ('stab', 'soft', 'negkl', 'margin', 'conf')}
        o['row_acc_by_conf_quintile'] = [round(float(ok0[m & (strata == j)].mean()), 4) for j in range(5)]
        o['stab_by_conf_quintile'] = [round(float(st['stab'][m & (strata == j)].mean()), 4) for j in range(5)]
        qtasks = {'q_4of4_correct': (mq, n0 == 4), 'R_vs_rest': (mq & np.isin(cls, list('ROMU')), cls == 'R'),
                  'M_vs_U': (mq & np.isin(cls, ['M', 'U']), cls == 'M')}
        o['question_auc'] = {}
        for t, (mm, y) in qtasks.items():
            sc = {k_: v for k_, v in qsc.items() if not (t == 'M_vs_U' and k_.startswith('consensus'))}
            i = np.flatnonzero(mm)
            Wr = W[:, i].astype(np.float32)
            o['question_auc'][t] = {k_: r5([auc(y[i], v[i]), *np.nanpercentile(auc(y[i], v[i], Wr), [2.5, 97.5])])
                                    for k_, v in sc.items()}
            for k_ in ('stab_q', 'stab_qmean'):
                d = auc(y[i], sc[k_][i], Wr) - auc(y[i], sc['conf_q'][i], Wr)
                o['question_auc'][t][f'{k_}-conf_q'] = r5([auc(y[i], sc[k_][i]) - auc(y[i], sc['conf_q'][i]),
                                                          *np.nanpercentile(d, [2.5, 97.5])])
        qf = i_q[D['fjc'][i_q] >= 0]
        o['fjc_onset'] = dict(n=len(qf), spearman_stab_q_vs_FJC_c=spear(st['stab_q'], D['fjc'], qf, False),
                              spearman_conf_q_vs_FJC_c=spear(qsc['conf_q'], D['fjc'], qf, False),
                              stab_q_by_FJC_c={int(l): [int((D['fjc'][qf] == l).sum()),
                                                        round(float(st['stab_q'][qf][D['fjc'][qf] == l].mean()), 4)]
                                               for l in np.unique(D['fjc'][qf])})
        o['commit_onset'] = dict(spearman_stab_vs_csd_commit=spear(st['stab'], R['csd']['commit'].astype(float), i_q, True),
                                 spearman_conf_vs_csd_commit=spear(conf, R['csd']['commit'].astype(float), i_q, True))
        out[k] = o
    return out, st


def abne(D, loops, cfgs, late, sets, Wq, post_csv):
    """(2) who benefits; U_avail vs U_none rescue rates."""
    cls, ua, ok0 = D['cls'], D['ua'], D['ok0']
    pc = list(csv.DictReader(open(post_csv))) if os.path.exists(post_csv) else None
    if pc is not None and not (np.array([p['orig'] for p in pc]) == cls).all():
        pc = None  # loop_eval ran under another taxonomy rule (e.g. last layer = probe)
    out = {}
    for k, m in sets.items():
        W, mq = Wq[k], CL.to_q(m).any(1)
        n0 = CL.to_q(ok0).sum(1)
        grp = {c: (cls == c) & mq for c in 'ROMU'} | {'U_avail': (cls == 'U') & ua & mq, 'U_none': (cls == 'U') & ~ua & mq}
        wrong, right = CL.to_q(~ok0 & m).sum(1), CL.to_q(ok0 & m).sum(1)
        o = dict(n={g: int(v.sum()) for g, v in grp.items()},
                 none_acc={g: round(float(n0[v].mean() / 4), 4) for g, v in grp.items()})
        Un_w = {n: (grp['U_avail'] & (n0 == n)).astype(float) @ wrong for n in range(4)}  # U_avail strata (wrong rows)
        for c in cfgs + ([late] if late else []):
            x, a_, b_ = loops[c], loops[c]['cfg']['a'], loops[c]['cfg']['b']
            fix, brk = CL.to_q((x['gain'] > 0) & m).sum(1), CL.to_q((x['gain'] < 0) & m).sum(1)
            gq, nc = CL.to_q(x['gain'] * m).sum(1), CL.to_q(x['ok']).sum(1)
            r = {g: dict(fix_rate=rate(W, fix, wrong, v), break_rate=rate(W, brk, right, v), net=rate(W, gq, 4 * mq, v),
                         rescue_4of4=CL.ci(W, (nc == 4) & v & (n0 < 4), v & (n0 < 4))) for g, v in grp.items()}
            Ua, Un = grp['U_avail'], grp['U_none']
            r['Uavail_minus_Unone'] = dict(fix_rate=rate_diff(W, fix, wrong, Ua, fix, wrong, Un),
                                           net=rate_diff(W, gq, 4 * mq, Ua, gq, 4 * mq, Un),
                                           rescue_4of4=rate_diff(W, nc == 4, 1, Ua & (n0 < 4), nc == 4, 1, Un & (n0 < 4)))

            def strat(w, fx=fix):  # U_none fix rate per wrong row reweighted to U_avail's none n_correct strata
                num = sum(Un_w[n] * (w @ (fx * Un * (n0 == n))) / np.maximum(w @ (wrong * Un * (n0 == n)), 1e-12)
                          for n in range(4))
                return num / sum(Un_w.values())
            r['Unone_fix_rate_reweighted_to_Uavail'] = CL.wci(strat, W)
            r['Uavail_minus_Unone_reweighted'] = CL.wci(lambda w: CL.ratio(fix * Ua, wrong * Ua)(w) - strat(w), W)
            if late and c != late:
                lf = CL.to_q((loops[late]['gain'] > 0) & m).sum(1)
                r['fix_minus_placebo'] = {g: CL.wci(lambda w, v=v: CL.ratio(fix * v, wrong * v)(w) -
                                                    CL.ratio(lf * v, wrong * v)(w), W) for g, v in grp.items()}
                r['DiD_Uavail_minus_Unone'] = CL.wci(
                    lambda w: (CL.ratio(fix * Ua, wrong * Ua)(w) - CL.ratio(lf * Ua, wrong * Ua)(w)) -
                              (CL.ratio(fix * Un, wrong * Un)(w) - CL.ratio(lf * Un, wrong * Un)(w)), W)
            fp = D['fpcl']
            r['Uavail_by_FPCL_c'] = {nm: dict(n=int(v.sum()), fix_rate=rate(W, fix, wrong, v)) for nm, v in
                                     ((f'<{a_}', Ua & (fp < a_)), (f'{a_}-{b_}', Ua & (fp >= a_) & (fp <= b_)),
                                      (f'>{b_}', Ua & (fp > b_)))}
            if pc is not None and f'last_{c}' in pc[0]:
                post = np.array([p[f'last_{c}'] for p in pc])
                pn = np.array([p['last_none'] for p in pc])
                good = np.isin(post, ['R', 'O']).astype(float) - np.isin(pn, ['R', 'O'])
                r['v1_to_RO_minus_none'] = {g: CL.ci(W, good * v, v) for g, v in grp.items() if g.startswith('U')}
            o[c] = r
        out[k] = o
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
    ap.add_argument('--loop_dir', default='loop_allq'); ap.add_argument('--out', default='loopgate')
    ap.add_argument('--lockbox', action='store_true', help='also report fold 4 (every choice made on folds 0-3)')
    a = ap.parse_args()
    t0 = time.time()
    D, loops = load(a.model, a.fmt, a.loop_dir)
    N, L, Q, fold, gold = D['N'], D['L'], D['Q'], D['fold'], D['gold']
    outcomes(D, loops)
    WC = [c for c in loops if c[0] in 'wd']
    PL = sorted((c for c in loops if c[0] == 'p'), key=lambda c: -loops[c]['cfg']['a'])
    late = PL[0] if PL else None
    gcfg = [scale_name(c, L) for c in GATE36 if scale_name(c, L) in loops]
    k1 = [c for c in gcfg if loops[c]['cfg']['k'] == 1 and c[0] == 'w']
    keep = np.repeat(D['cls'] != 'X', 4)
    sets = {'dev': np.isin(fold, DEV) & keep}
    if a.lockbox:
        sets.update(lockbox=(fold == LOCK) & keep, all=keep)
    Wq = {k: boot_q(CL.to_q(m).any(1)) for k, m in sets.items()}
    od = os.path.join(D['rd'], a.out)
    os.makedirs(od, exist_ok=True)
    sfx = '_lockbox' if a.lockbox else ''
    print(f'{a.model}__{a.fmt} {a.loop_dir}: configs {list(loops)} window {WC} late placebo {late} gate {gcfg}; '
          f'rows ' + ' '.join(f'{k}={m.sum()}' for k, m in sets.items()), C.stamp(t0), flush=True)
    S = dict(config=dict(model=a.model, fmt=a.fmt, loop_dir=a.loop_dir, lockbox=a.lockbox, dev_folds=DEV,
                         lockbox_fold=LOCK, n_boot=B_BOOT, seed=SEED, C=CREG, npca=NPCA, window_configs=WC,
                         late_placebo=late, gate_configs=gcfg, families=FAMILIES,
                         classes={c: int((D['cls'] == c).sum()) for c in 'ROMUX'},
                         consensus_defined=dict(loo=round(float((D['consL'] >= 0).mean()), 4),
                                                all=round(float((D['consL_all'] >= 0).mean()), 4)),
                         class_source=D['class_source']))

    # ---- per config: change / fix / break / net (letter-restricted and strict), compliance, consensus-proxy calibration
    S['configs'] = {k: {} for k in sets}
    for k, m in sets.items():
        for c, x in loops.items():
            if c == 'none':
                continue
            ch = m & (x['let'] != loops['none']['let'])
            corr = lambda g2: round(float(np.corrcoef(x['gain'][ch], g2[ch])[0, 1]), 4) if ch.sum() > 2 else None
            S['configs'][k][c] = dict(
                change=CL.ci(Wq[k], CL.to_q(ch).sum(1), CL.to_q(m).sum(1)), fix=int(((x['gain'] > 0) & m).sum()),
                brk=int(((x['gain'] < 0) & m).sum()), net=pstats(np.ones(N, bool), x['gain'], m, Wq[k])['net'],
                net_strict=pstats(np.ones(N, bool), x['gain_s'], m, Wq[k])['net'],
                compliance=round(float(x['comp'][m].mean()), 4),
                proxy_sum_gain=dict(gold=int(x['gain'][m].sum()), cons_loo=int(x['gain_c'][m].sum()),
                                    cons_all=int(x['gain_c_all'][m].sum())),
                proxy_corr_with_gold_gain_changed_rows=dict(cons_loo=corr(x['gain_c']), cons_all=corr(x['gain_c_all'])))
    for c, r in S['configs']['dev'].items():
        print(f'  {c:12s} dev change {r["change"][0]:.4f} fix {r["fix"]} break {r["brk"]} net {100 * r["net"][0]:+.2f}pp '
              f'strict {100 * r["net_strict"][0]:+.2f}pp compliance {r["compliance"]}', flush=True)

    # ---- none-pass readouts (L0)
    nd = loops['none']['dir']
    w0, let0 = int(round(0.45 * L)), loops['none']['let']
    zc, zl = np.load(os.path.join(nd, 'traj_csd4.npy')), np.load(os.path.join(nd, 'traj_lens4.npy'))
    R = {'csd': readouts(zc, let0, w0), 'lens': readouts(zl, let0, w0)}
    base4 = np.load(os.path.join(D['ex'], 'csd_base4.npy')).astype(np.float64)
    lp = dict(np.load(os.path.join(D['ex'], 'lens_params.npz')))
    hid = np.memmap(os.path.join(nd, 'traj_hidden_bf16.u16'), np.uint16, 'r', shape=(N, L, D['d']))

    # ---- (a) perturbation stability
    S['stability'], st = stability(D, loops, WC, late, sets, Wq, R, keep)
    o = S['stability']['dev']
    print('(a) dev row AUROC correct:', {k: o['row_correct_auc'][k] for k in ('conf', 'stab', 'soft', 'negkl',
                                                                               'stab-conf', 'soft-conf', 'L3_conf_stab')},
          '| within conf quintiles', o['row_correct_auc_within_conf_quintiles'], C.stamp(t0), flush=True)
    print('    question AUROC', json.dumps(o['question_auc']), '| FJC', o['fjc_onset']['spearman_stab_q_vs_FJC_c'],
          o['fjc_onset']['spearman_conf_q_vs_FJC_c'], '| by class stab_q',
          {c: v['stab_q'][0] for c, v in o['by_class'].items()}, flush=True)

    # ---- (b) gated loop
    cache, G, gates, scores = {}, {}, {}, {}
    G = geometry(hid, sorted({loops[c]['cfg'][e] for c in gcfg for e in 'ab'} | {L - 1}), lp, cache)
    eps = float(lp['eps'])
    devr = np.isin(fold, DEV) & keep
    Hp = {l: pca(cache[l], devr, eps) for l in sorted({loops[c]['cfg']['a'] for c in gcfg} | {L - 1})}
    print('features: geometry + PCA done', C.stamp(t0), flush=True)
    vote_let = np.stack([let0] + [loops[c]['let'] for c in WC], 1)
    cnt = (vote_let[..., None] == np.arange(4)).sum(1) + 0.5 * (let0[:, None] == np.arange(4))
    vote_gain = (cnt.argmax(1) == gold).astype(np.int8) - D['ok0']
    perm_gain = (np.where(D['consL_all'] >= 0, D['consL_all'], let0) == gold).astype(np.int8) - D['ok0']
    xv_dis = np.repeat(D['out_n'] < 4, 4)
    S['gate'] = {}
    for c in gcfg:
        x, a_, b_ = loops[c], loops[c]['cfg']['a'], loops[c]['cfg']['b']
        R['csd']['z_a'] = zc[:, a_].astype(np.float64)
        names, X = gate_features(R, G, loops['none']['f4'], base4, let0, a_, b_, L)
        gain, gc = x['gain'].astype(float), x['gain_c'].astype(float)
        ic = names.index('fin_lmaxp')
        cand = [(j, sg) for j in range(len(names)) for sg in (1, -1)]
        sc, gt, info = {}, {}, {}
        for fam, cd, gs in (('conf', [(ic, -1)], gain), ('conf_u', [(ic, -1)], gc), ('L0', cand, gain), ('L0u', cand, gc)):
            sc[fam], gt[fam], ch = single_gate(X, cd, fold, keep, gs, a.lockbox)
            info[fam] = {f: (names[j], sg, round(t, 5)) for f, (j, sg, t) in ch.items()}
        s1, g1, info['L1'] = trained_gate(X, x['gain_c'], keep & (D['consL'] >= 0), fold, keep,
                                          {'gold': gain, 'cons': gc}, a.lockbox)
        sc.update({'L1': s1, 'L1@0': s1, 'L1u': s1}); gt.update({'L1': g1['gold'], 'L1@0': g1['@0'], 'L1u': g1['cons']})
        s3, g3, info['L3'] = trained_gate(X, x['gain'], keep, fold, keep, {'gold': gain}, a.lockbox)
        sc.update({'L3': s3, 'L3@0': s3}); gt.update({'L3': g3['gold'], 'L3@0': g3['@0']})
        Xh = np.c_[X, Hp[a_], Hp[L - 1]]
        s3h, g3h, info['L3h'] = trained_gate(Xh, x['gain'], keep, fold, keep, {'gold': gain}, a.lockbox)
        sc['L3h'], gt['L3h'] = s3h, g3h['gold']
        sc['xv4'], gt['xv4'] = xv_dis.astype(float), xv_dis
        gates[c], scores[c] = gt, sc
        res = dict(window=[a_, b_], k=x['cfg']['k'], features=names, choices=info)
        for k, m in sets.items():
            if k == 'all':
                continue
            W = Wq[k]
            r = dict(always=pstats(np.ones(N, bool), gain, m, W), oracle_uses_gold=pstats(gain > 0, gain, m, W),
                     always_strict=pstats(np.ones(N, bool), x['gain_s'].astype(float), m, W))
            for fam in FAMILIES:
                r[fam] = pstats(gt[fam], gain, m, W, ref=gt['conf'])
                r[fam]['minus_always'] = CL.wci(lambda w, g_=gt[fam]: CL.ratio(CL.to_q((g_ & m) * gain).sum(1),
                                                                                 CL.to_q(m).sum(1))(w) -
                                                CL.ratio(CL.to_q(m * gain).sum(1), CL.to_q(m).sum(1))(w), W)
                r[fam]['net_strict'] = pstats(gt[fam], x['gain_s'].astype(float), m, W)['net']
            ch_ = m & (gain != 0)
            r['fix_vs_break_auc'] = auc_rows(gain > 0, {f: sc[f] for f in ('conf', 'L0', 'L0u', 'L1', 'L3', 'L3h')},
                                             ch_, W, ref='conf')
            if k == 'dev':
                r['fix_vs_break_auc_single_features'] = dict(sorted(
                    {n: round(auc(gain[ch_] > 0, X[ch_, j]), 4) for j, n in enumerate(names)}.items(),
                    key=lambda kv: -abs(kv[1] - 0.5))[:12])
                r['pareto'] = {f: pareto(sc[f], gain, m, fold, W) for f in ('conf', 'L0', 'L1', 'L3', 'L3h')}
                r['pareto']['oracle'] = pareto(gain, gain, m, fold, W)
            res[k] = r
        S['gate'][c] = res
        r = res['dev']
        print(f'(b) {c} dev: always {100 * r["always"]["net"][0]:+.2f} oracle {100 * r["oracle_uses_gold"]["net"][0]:+.2f} | '
              + ' '.join(f'{f} {100 * r[f]["net"][0]:+.2f}[{100 * r[f]["net"][1]:+.2f},{100 * r[f]["net"][2]:+.2f}]'
                         f'(cov {r[f]["coverage"][0]:.3f})' for f in FAMILIES), C.stamp(t0), flush=True)
        print(f'    vs conf: ' + ' '.join(f'{f} {100 * r[f]["minus_ref"][0]:+.2f}[{100 * r[f]["minus_ref"][1]:+.2f},'
                                         f'{100 * r[f]["minus_ref"][2]:+.2f}]' for f in FAMILIES if f != 'conf') +
              ' | fix-vs-break AUROC ' + ' '.join(f'{f} {v[0]:.3f}' for f, v in r['fix_vs_break_auc'].items()
                                                if '-' not in f), flush=True)
    S['vote_multipass'] = {k: pstats(np.ones(N, bool), vote_gain.astype(float), m, Wq[k]) for k, m in sets.items()}
    S['permutation_vote_no_loop'] = {k: pstats(np.ones(N, bool), perm_gain.astype(float), m, Wq[k]) for k, m in sets.items()}

    # ---- pre-declared selection on folds 0-3 (dev) -> lockbox
    dnet = lambda c, f: S['gate'][c]['dev'][f]['net'][0]
    best_k1 = max(k1, key=lambda c: S['gate'][c]['dev']['always']['net'][0]) if k1 else None
    sel = dict(best_k1_by_dev_always=best_k1, always=max(gcfg, key=lambda c: S['gate'][c]['dev']['always']['net'][0]),
               per_family={f: max(gcfg, key=lambda c: dnet(c, f)) for f in FAMILIES})
    sel['best_unsup'] = max(((f, c) for f in UNSUP for c in gcfg), key=lambda z: dnet(z[1], z[0]))
    sel['best_L0L1'] = max(((f, c) for f in ('L0', 'L0u', 'L1', 'L1@0', 'L1u') for c in gcfg), key=lambda z: dnet(z[1], z[0]))
    # gold-free selection (review fix): best_unsup above picks family x config by dev GOLD net gain; here by the dev
    # consensus (L1) net gain instead, so the selected policy uses no gold anywhere
    cnet = lambda c, f: float((gates[c][f] * loops[c]['gain_c'])[sets['dev']].sum() / sets['dev'].sum())
    sel['best_unsup_goldfree'] = max(((f, c) for f in UNSUP for c in gcfg), key=lambda z: cnet(z[1], z[0]))
    sel['always_goldfree'] = max(gcfg, key=lambda c: float(loops[c]['gain_c'][sets['dev']].mean()))
    S['selection'] = sel
    rep = {}
    for k, m in sets.items():
        if k == 'all':
            continue
        W = Wq[k]
        g_conf = gates[sel['per_family']['conf']]['conf']
        rr = dict(always=pstats(np.ones(N, bool), loops[sel['always']]['gain'].astype(float), m, W),
                  vote_multipass=S['vote_multipass'][k], permutation_vote_no_loop=S['permutation_vote_no_loop'][k])
        rr[f'always_goldfree[{sel["always_goldfree"]}]'] = pstats(np.ones(N, bool),
                                                                   loops[sel['always_goldfree']]['gain'].astype(float), m, W)
        for f, c in sel['per_family'].items():
            rr[f'{f}[{c}]'] = pstats(gates[c][f], loops[c]['gain'].astype(float), m, W)
        for nm in ('best_unsup', 'best_L0L1', 'best_unsup_goldfree'):
            f, c = sel[nm]
            gg, gn = gates[c][f], loops[c]['gain'].astype(float)
            rr[f'{nm}={f}[{c}]'] = pstats(gg, gn, m, W)
            # paired difference to the selected conf gate (each uses its own config's gain)
            num1 = CL.to_q((gg & m) * gn).sum(1)
            num0 = CL.to_q((g_conf & m) * loops[sel['per_family']['conf']]['gain']).sum(1)
            rr[f'{nm}_minus_conf'] = CL.wci(lambda w: CL.ratio(num1, CL.to_q(m).sum(1))(w) -
                                            CL.ratio(num0, CL.to_q(m).sum(1))(w), W)
        rep[k] = rr
    S['selected_report'] = rep
    for k, rr in rep.items():
        print(f'selected ({k}): ' + ' '.join(f'{n} {100 * v["net"][0]:+.2f}[{100 * v["net"][1]:+.2f},{100 * v["net"][2]:+.2f}]'
                                            if 'net' in v else f'{n} {100 * v[0]:+.2f}[{100 * v[1]:+.2f},{100 * v[2]:+.2f}]'
                                            for n, v in rr.items()), flush=True)

    # ---- (2) ABNE / who benefits
    S['abne'] = abne(D, loops, gcfg, late, sets, Wq, os.path.join(D['rd'], a.loop_dir, 'eval', 'post_classes.csv'))
    for c in gcfg:
        r = S['abne']['dev'][c]
        print(f'(2) {c} dev fix rate U_avail {r["U_avail"]["fix_rate"]} U_none {r["U_none"]["fix_rate"]} diff '
              f'{r["Uavail_minus_Unone"]["fix_rate"]} reweighted diff {r["Uavail_minus_Unone_reweighted"]} DiD '
              f'{r.get("DiD_Uavail_minus_Unone")} | net by class ' +
              ' '.join(f'{g} {100 * r[g]["net"][0]:+.2f}' for g in ('R', 'O', 'M', 'U_avail', 'U_none')), flush=True)

    S['config'].update(seconds=round(time.time() - t0, 1), script_md5=C.md5_file(__file__),
                       common_md5=C.md5_file(C.__file__), classify_md5=C.md5_file(CL.__file__))
    C.save_json(S, os.path.join(od, f'summary{sfx}.json'))
    np.savez_compressed(os.path.join(od, f'rows{sfx}.npz'), post_letters=np.stack([loops[c]['let'] for c in loops], 1)
                        .astype(np.int8), configs=np.array(list(loops)), stab=st['stab'], soft=st['soft'], kl=st['kl'],
                        stab_q=st['stab_q'], **{f'score_{c}_{f}': scores[c][f].astype(np.float32)
                                                for c in gcfg for f in ('conf', 'L0', 'L0u', 'L1', 'L3', 'L3h')},
                        **{f'gate_{c}_{f}': gates[c][f] for c in gcfg for f in FAMILIES})
    print('wrote', od, C.stamp(t0), flush=True)


if __name__ == '__main__':
    main()
