"""Stage-2 preview, part 2: what do label-free per-layer indicators predict? CPU only (needs stage2_features.py output).

Labels (--labels auto): RUN/classify/questions.csv (scripts/classify.py) if present; else classify.py's own v1 rule
  (classify_v1, K=3) run here on probe/oof_full.npy (= classify.py --probe full --last_layer output); else a stand-in with
  the logit lens in place of the probe. The last two are written to RUN/stage2/{probe,lens}_questions.csv.
Question tasks (X excluded): O_vs_U, U_vs_M, Uavail_vs_Unone, R_vs_rest. Question features = version means of the row
  features plus the cross-version xv_* features: (i) at fixed relative depths ('name@l'), only features computable at
  layer l (no reference to the final output); (ii) post-hoc summaries over the window [w0, L-1] ('name:wmean', ':wmax',
  '_flips:w' = top-1 changes, ':commit' = first layer from which agreement with the final output holds to the end,
  ':run4' = longest run of 4/4 cross-version consensus), which may reference the final output (*_agree/_fin/_gap/_lost).
Supervision level of a set = highest tag in it (L0 < L1 < L2); L3 = logistic regression on PCA(128) of the z-scored
  RMS-normalized row hidden state, row scores averaged per question (supervised router). Sets are supervised
  combinations (an upper bound for that feature level); single features use only a sign chosen on the training folds,
  i.e. they are the unsupervised indicators proper. U_vs_M drops every xv_* feature (the U/M split is defined by the
  cross-version consistency of the final output).
Model: standardized logistic regression (C=0.1, balanced classes) cross-fitted over the `fold` field; pooled out-of-fold
  AUROC with a by-question bootstrap 95% CI. Baseline = final 4-way max softmax + entropy (version means).
  gain = AUROC(baseline + set) - AUROC(baseline) (paired bootstrap); resid = AUROC after residualizing the features on
  the baseline (OLS on the training folds; columns with no residual variance dropped). 'New quantity': gain >= 0.05.
Label-free lost consensus (the O rule without gold): readouts A and B (lens & CSD = L1; probe & CSD = L2) agree 4/4
  across versions on one content for K=3 consecutive layers and the final output does not keep 4/4 on it.
Row tasks (bootstrap by question): per-layer safe exit, L1 target = the layer-l CSD (or lens) top-1 equals the final
  top-1 and stays equal to the end, from layer-l L0 (all; 'L0nocsd' = lens + hidden only) and L2 features vs layer-l
  max softmax, plus an exit policy at a
  target agreement with the final output (threshold chosen on the training folds; baselines: max-softmax threshold
  and the stab rule = top-1 unchanged for k layers and max softmax >= tau); loop: for each
  RUN/{loop,loop_allq}/{mode}_{cfg} with meta.json, 'final output changes' (all rows) and 'loop fixes the row' (rows
  wrong before; gold only defines this evaluation target) from features at the window start a and end b, and the
  net accuracy change of gating the loop on a label-free score (threshold chosen on the training folds).
Output: RUN/stage2/analysis{,_probe,_standin}.json and tables{...}.tsv (labels from classify / probe / lens).
Usage: python scripts/stage2_analysis.py --model qwen3-4b-base --fmt base_fewshot [--labels lens] [--skip_rows|--rows_only]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '12')
import argparse, csv, glob, json, sys, time, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import classify as CL

DEPTHS, CREG, B_Q, B_ROW, SEED, KRUN = (0.3, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0), 0.1, 2000, 500, 0, 3
RANK = {'L0': 0, 'L1': 1, 'L2': 2}
FINAL_REF = ('_agree', '_fin', '_gap', '_lost')
CAND = ('avail_', 'xv_lens_gap', 'xv_csd_gap', 'xv_probe_gap', 'xv_lens_lost', 'xv_csd_lost', 'xv_probe_lost')
PROFILE = ('csd_maxp', 'lens_maxp', 'csd_agree', 'lens_agree', 'avail_lens', 'avail_probe', 'probe_maxp', 'xv_csd_n',
           'xv_lens_n', 'xv_probe_n', 'xv_csd_lost', 'xv_lens_lost', 'xv_probe_lost', 'h_u4', 'h_upd', 'h_cos')
warnings.filterwarnings('ignore', category=RuntimeWarning)


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


def boot(groups, B):
    """(B, n) by-cluster bootstrap multiplicities for items with cluster ids `groups`."""
    u, inv = np.unique(groups, return_inverse=True)
    return CL.boot_weights(len(u), B, SEED)[:, inv].astype(np.float32)


def rnd(x, k=4):
    return [None if x_ is None or not np.isfinite(x_) else round(float(x_), k) for x_ in x]


def est(y, s, W, ref=None):
    """[point, lo, hi]: AUROC(s), or AUROC(s) - AUROC(ref) with a paired bootstrap; None if s is None."""
    if s is None:
        return [None] * 3
    pt = auc(y, s) - (auc(y, ref) if ref is not None else 0.0)
    b = auc(y, s, W) - (auc(y, ref, W) if ref is not None else 0.0)
    return rnd([pt, *np.nanpercentile(b, [2.5, 97.5])])


def residualize(X, resid, tr):
    """Regress X on [1, resid] (OLS on rows tr); keep residual columns that retain variance."""
    A = np.c_[np.ones(len(X)), resid]
    R = X - A @ np.linalg.lstsq(A[tr], X[tr], rcond=None)[0]
    keep = R[tr].std(0) > 1e-6 * (X[tr].std(0) + 1e-12)
    return R[:, keep]


def fold_models(X, y, fold, resid=None, balanced=True, prob=False):
    """One standardized logistic regression per outer fold (trained on the other folds): (n_folds, n) scores of every
    fold model on all items, or None if residualizing on `resid` leaves no variance."""
    X = np.nan_to_num(np.asarray(X, np.float64).reshape(len(y), -1))
    folds = np.unique(fold)
    out = np.zeros((len(folds), len(y)))
    for i, f in enumerate(folds):
        tr = fold != f
        if len(np.unique(y[tr])) < 2:
            continue
        Xf = X if resid is None else residualize(X, resid, tr)
        if Xf.shape[1] == 0:
            return None
        m = make_pipeline(StandardScaler(), LogisticRegression(C=CREG, class_weight='balanced' if balanced else None,
                                                               max_iter=5000)).fit(Xf[tr], y[tr])
        out[i] = m.predict_proba(Xf)[:, 1] if prob else m.decision_function(Xf)
    return out


def oof(P, fold):
    """Pool the held-out fold of each fold model: (n_folds, n) -> (n,)."""
    return None if P is None else P[np.searchsorted(np.unique(fold), fold), np.arange(P.shape[1])]


def crossfit(X, y, fold, **kw):
    return oof(fold_models(X, y, fold, **kw), fold)


def single_oof(x, y, fold, resid=None):
    """Out-of-fold score of one feature: its sign is chosen on the training folds, nothing else is fitted."""
    s = np.zeros(len(y))
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        xf = np.nan_to_num(x) if resid is None else residualize(np.nan_to_num(x)[:, None], resid, tr)
        if resid is not None:
            if xf.shape[1] == 0:
                return None
            xf = xf[:, 0]
        s[te] = xf[te] * (1.0 if auc(y[tr], xf[tr]) >= 0.5 else -1.0)
    return s


def evaluate(X, y, fold, W, base, s_base):
    s = crossfit(X, y, fold)
    return dict(n_feat=int(np.asarray(X).reshape(len(y), -1).shape[1]), auc=est(y, s, W),
                gain=est(y, crossfit(np.c_[base, X], y, fold), W, s_base),
                resid=est(y, crossfit(X, y, fold, resid=base), W))


# ----------------------------------------------------------------------------- inputs
def labels(run, rows, mode):
    """v1 outcome (Q,), u_avail (Q,), info. mode: 'classify' = RUN/classify/questions.csv; 'probe' = classify.py's own
    classify_v1 (K=3) run here on probe/oof_full.npy (same rule as `classify.py --probe full`); 'lens' = stand-in with
    the logit lens in place of the probe; 'auto' = the first of these that exists."""
    path, pp = os.path.join(run, 'classify', 'questions.csv'), os.path.join(run, 'probe', 'oof_full.npy')
    if mode == 'auto':
        mode = 'classify' if os.path.exists(path) else 'probe' if os.path.exists(pp) else 'lens'
    Q = len(rows) // 4
    if mode == 'classify':
        qs = list(csv.DictReader(open(path)))
        assert [int(r['qid']) for r in qs] == list(range(Q))
        sp = os.path.join(run, 'classify', 'summary.json')
        ll = json.load(open(sp))['config'].get('last_layer', 'probe') if os.path.exists(sp) else None
        if ll != 'output':
            print(f'WARNING: classify/ was made with last_layer={ll}, not the official "output" (INTERFACES 2026-10-16)')
        return (np.array([r['outcome_v1_K3'] for r in qs]), np.array([r['u_avail_K3'] == '1' for r in qs]),
                dict(mode=mode, standin=False, source=path, md5=C.md5_file(path), last_layer=ll))
    ex = os.path.join(run, 'extract')
    meta = json.load(open(os.path.join(ex, 'meta.json')))
    gold = np.array([r['gold'] for r in rows])
    excl = ~CL.to_q(np.isin(np.load(os.path.join(ex, 'out_top1.npy')), meta['letter_ids'])).all(1)
    pl, cl, fl = [CL.top1(np.load(f).astype(np.float32)) for f in
                  (pp if mode == 'probe' else os.path.join(ex, 'lens4.npy'), os.path.join(ex, 'csd_raw4.npy'),
                   os.path.join(ex, 'out_logits4.npy'))]
    pl[:, -1] = fl  # last layer = real output (INTERFACES 2026-10-16 evening = classify.py --last_layer output)
    pc, cc, fc = [CL.contents(CL.to_q(x)) for x in (pl, cl, fl)]
    r = CL.classify_v1(pc, cc, fc, gold[0::4], KRUN, excl)
    out = os.path.join(run, 'stage2', f'{mode}_questions.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['qid', 'source', 'fold', 'answer', 'n_final_correct', 'outcome_v1_K3', 'FPCL_c_K3', 'FJC_c_K3',
                    'u_avail_K3'])
        for q in range(Q):
            w.writerow([q, rows[4 * q]['source'], rows[4 * q]['fold'], gold[4 * q], r['n_final'][q], r['outcome'][q],
                        r['fpcl'][q], r['fjc'][q], int(r['u_avail'][q])])
    info = dict(mode=mode, standin=mode == 'lens', source=out, rule=f'classify.classify_v1, K=3, {mode} as probe, last layer = real output')
    if os.path.exists(path):
        ref = np.array([q['outcome_v1_K3'] for q in csv.DictReader(open(path))])
        info['agree_with_classify_csv'] = float((ref == r['outcome']).mean())
    return r['outcome'], r['u_avail'], info


def maxrun(b):
    cur = best = np.zeros(b.shape[0])
    for j in range(b.shape[1]):
        cur = (cur + 1) * b[:, j]
        best = np.maximum(best, cur)
    return best


def question_features(F, tags, L, w0, Q):
    """-> X {name: (Q,)}, T {name: level}, depth layers, version-mean (Q, L) arrays for profiles."""
    depth = sorted({int(f * (L - 1) + 0.5) for f in DEPTHS})
    X, T, AQ = {}, {}, {}

    def add(n, a, lvl):
        X[n], T[n] = np.asarray(a, np.float64), lvl
    for k, lvl in tags.items():
        a = F[k].astype(np.float64)
        row = a.shape[0] == 4 * Q
        qm = (lambda x: np.nanmean(x.reshape((Q, 4) + x.shape[1:]), 1)) if row else (lambda x: x)
        AQ[k] = aq = qm(a)
        if not k.endswith(FINAL_REF):
            for l in depth:
                add(f'{k}@{l}', aq[:, l], lvl)
        w = a[:, w0:]
        if k.endswith(('_agree', '_fin')):
            add(f'{k}:commit', qm(L - np.cumprod(a[:, ::-1] > 0.5, 1).sum(1)), lvl)
        if k.endswith('_run'):
            add(f'{k[:-4]}_flips:w', qm((w[:, 1:] == 1).sum(1)), lvl)
            continue
        add(f'{k}:wmean', qm(np.nanmean(w, 1)), lvl)
        if k.endswith(('_kl', '_gap', '_lost', 'h_upd', 'h_updu4')) or k.startswith('avail_'):
            add(f'{k}:wmax', qm(np.nanmax(w, 1)), lvl)
        if k.startswith('xv_') and k.endswith('_n'):
            add(f'{k}:run4', maxrun(w == 4), lvl)
    return X, T, depth, AQ


def pca_rows(model, fmt, layers, eps, k=128):
    """Label-free PCA(k) of the z-scored RMS-normalized row hidden state at each layer (inputs of the L3 router)."""
    hid, _ = C.open_hidden(model, fmt)
    out = {}
    for l in layers:
        h = C.bf16_bits_to_f32(hid[:, l, :])
        h = h / np.sqrt((h ** 2).mean(-1, keepdims=True) + eps)
        h = (h - h.mean(0)) / (h.std(0) + 1e-6)
        U, S, _ = randomized_svd(h, k, random_state=SEED)
        out[l] = (U * S).astype(np.float32)
    return out


# ----------------------------------------------------------------------------- analyses
def lost_consensus(F, A, B, answer, out_votes, cls, K=KRUN):
    """Label-free O analogue: readouts A and B agree 4/4 across versions on one content for K consecutive layers and the
    final output does not keep 4/4 on it. Gold is used only to score the flags."""
    both = (F[f'xv_{A}_n'] == 4) & (F[f'xv_{B}_n'] == 4) & (F[f'modal_{A}'] == F[f'modal_{B}'])
    lost = np.stack([(CL.first_run(both & (F[f'modal_{A}'] == c), K) >= 0) & (out_votes[:, c] < 4)
                     for c in range(4)], 1)
    flag, right = lost.any(1), lost[np.arange(len(answer)), answer]
    wrong = (lost & (np.arange(4)[None] != answer[:, None])).any(1)
    return dict(n_flag=int(flag.sum()), flag_by_class={c: int((flag & m).sum()) for c, m in cls.items()},
                n_class={c: int(m.sum()) for c, m in cls.items()},
                lost_content_is_answer=int(right.sum()), lost_content_is_wrong=int(wrong.sum()),
                precision_O=round(float((flag & cls['O']).sum() / max(flag.sum(), 1)), 4),
                recall_O=round(float((flag & cls['O']).sum() / max(cls['O'].sum(), 1)), 4),
                U_with_wrong_lost=int((wrong & cls['U']).sum()))


def question_tasks(X, T, depth, P3, base, o, ua, qfold, levels):
    tasks = {'O_vs_U': (np.isin(o, ['O', 'U']), o == 'O'), 'U_vs_M': (np.isin(o, ['U', 'M']), o == 'U'),
             'Uavail_vs_Unone': (o == 'U', ua), 'R_vs_rest': (np.isin(o, list('ROMU')), o == 'R')}
    res = {}
    for task, (m, pos) in tasks.items():
        y, fold, b = pos[m].astype(int), qfold[m], base[m]
        if min(y.sum(), (1 - y).sum()) < 10:
            res[task] = dict(n_pos=int(y.sum()), n_neg=int((1 - y).sum()), skipped='fewer than 10 in a class')
            continue
        W = boot(np.flatnonzero(m), B_Q)
        s_base = crossfit(b, y, fold)
        allowed = [n for n in X if not (task == 'U_vs_M' and n.startswith('xv_'))]
        r = dict(n_pos=int(y.sum()), n_neg=int((1 - y).sum()), base=est(y, s_base, W), sets={}, singles={})
        for lvl in levels:
            for kind in ['sum'] + depth:
                names = [n for n in allowed if RANK[T[n]] <= RANK[lvl] and
                         (':' in n if kind == 'sum' else n.endswith(f'@{kind}'))]
                r['sets'][f'{lvl}|{kind}'] = evaluate(np.stack([X[n][m] for n in names], 1), y, fold, W, b, s_base)
        rows = np.repeat(m, 4)
        for l, Z in P3.items():   # L3: row-level router, scores averaged over the 4 versions
            s3 = crossfit(Z[rows], np.repeat(y, 4), np.repeat(fold, 4)).reshape(-1, 4).mean(1)
            r['sets'][f'L3|{l}'] = dict(n_feat=Z.shape[1], auc=est(y, s3, W),
                                        gain=est(y, crossfit(np.c_[b, s3], y, fold), W, s_base),
                                        resid=est(y, single_oof(s3, y, fold, resid=b), W))
        big = np.stack([X[n][m] for n in allowed if RANK[T[n]] <= RANK[levels[-1]] and ':' in n], 1)
        yp = y.copy()
        for f in np.unique(fold):
            yp[fold == f] = np.random.default_rng(SEED + int(f)).permutation(y[fold == f])
        r['null_auc_permuted_labels'] = round(auc(yp, crossfit(big, yp, fold)), 4)
        single = {n: auc(y, single_oof(X[n][m], y, fold)) for n in allowed}
        pick = set(n for n in allowed if n.startswith(CAND))
        for lvl in levels:
            pick |= set(sorted((n for n in allowed if T[n] == lvl), key=lambda n: -single[n])[:8])
        for n in sorted(pick, key=lambda n: -single[n]):
            x = X[n][m]
            r['singles'][n] = dict(level=T[n], final_ref=n.split('@')[0].split(':')[0].endswith(FINAL_REF),
                                   auc=est(y, single_oof(x, y, fold), W),
                                   gain=est(y, crossfit(np.c_[b, x], y, fold), W, s_base),
                                   resid=est(y, single_oof(x, y, fold, resid=b), W))
        r['single_auc_all'] = {n: round(v, 4) for n, v in sorted(single.items(), key=lambda kv: -kv[1])}
        res[task] = r
        best = {lvl: max((v['auc'][0], k) for k, v in r['sets'].items() if k.startswith(lvl))
                for lvl in list(levels) + (['L3'] if P3 else [])}
        print(f'  {task}: n={r["n_pos"]}/{r["n_neg"]} base={r["base"]} best sets {best} '
              f'null={r["null_auc_permuted_labels"]}', flush=True)
    return res


def exit_layers(P, tau, w0, L):
    """First layer in [w0, L-2] whose score >= tau, else L-1."""
    hit = P[:, w0:L - 1] >= tau
    return np.where(hit.any(1), w0 + hit.argmax(1), L - 1)


def policy(cands, agree, fold, w0, L, target):
    """cands: score arrays (n_folds or 1, n, L). Per fold, the (candidate, threshold) with the lowest mean exit layer
    whose agreement with the final output on the training folds is >= target. Returns test-fold exit layers (n,)."""
    e = np.full(len(fold), L - 1)
    for i, f in enumerate(np.unique(fold)):
        tr, te = fold != f, fold == f
        atr, best = agree[tr], (np.inf, None, None)
        for P in cands:
            Pi = P[i if P.shape[0] > 1 else 0]
            Ptr = Pi[tr]
            for t in np.unique(np.quantile(Ptr[:, w0:L - 1], np.linspace(0, 1, 201))):
                et = exit_layers(Ptr, t, w0, L)
                if atr[np.arange(len(et)), et].mean() >= target and et.mean() < best[0]:
                    best = (et.mean(), Pi, t)
        if best[1] is not None:
            e[te] = exit_layers(best[1][te], best[2], w0, L)
    return e


def mean_ci(v, W, scale=1.0):
    """[mean, lo, hi] of v (n,) under bootstrap multiplicities W (B, n)."""
    return rnd([v.mean() * scale, *np.nanpercentile((W @ v) / W.sum(1) * scale, [2.5, 97.5])])


def gate(score, delta, fold, W):
    """Loop only rows whose label-free score >= tau; tau per fold maximizes the training-fold accuracy gain (never and
    always included). delta = +1 fixed / -1 harmed / 0 (gold scores the policy only). Net accuracy change in pp."""
    use = np.zeros(len(delta), bool)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        cand = np.r_[np.inf, np.unique(np.quantile(score[tr], np.linspace(0, 1, 101)))]
        use[te] = score[te] >= cand[int(np.argmax([(delta[tr] * (score[tr] >= t)).mean() for t in cand]))]
    return dict(net_acc_pp=mean_ci(delta * use, W, 100), looped=round(float(use.mean()), 4))


def exit_tasks(F, tags, r_, qid, fold, gold, L, w0):
    N = len(qid)
    f0 = [k for k, t in tags.items() if t == 'L0' and F[k].shape[0] == N]
    f2 = f0 + [k for k, t in tags.items() if t == 'L2' and F[k].shape[0] == N and k != 'probe_agree']
    feats = {'L0': f0, 'L0nocsd': [k for k in f0 if not k.startswith(('csd_', 'avail_'))]} | \
        ({'L2': f2} if len(f2) > len(f0) else {})   # L0nocsd: lens + hidden only (no extra compute for a truncation exit)
    W, fd, out = boot(qid[r_], B_ROW), fold[r_], {}
    for head in ('csd', 'lens'):
        agree = F[f'{head}_agree'][r_] > 0.5
        commit = L - np.cumprod(agree[:, ::-1], 1).sum(1)
        top1, g = F[f'{head}_top1'][r_], gold[r_]
        maxp = F[f'{head}_maxp'][r_][None]
        Pf = {'base': maxp} | {k: np.zeros((5, len(r_), L)) for k in feats}
        stab = [maxp * (F[f'{head}_run'][r_][None] >= k) for k in (1, 2, 3, 4, 6)]  # top-1 stable k layers & conf
        per = []
        for l in range(w0, L - 1):
            y = (l >= commit).astype(int)
            ok = min(y.sum(), len(y) - y.sum()) >= 50     # AUROCs only where both classes have >= 50 rows
            na = [None] * 3
            d = dict(layer=l, n_safe=int(y.sum()), safe_rate=round(float(y.mean()), 4),
                     base=est(y, maxp[0, :, l], W) if ok else na,
                     single_raw_auc={k: round(auc(y, np.nan_to_num(F[k][r_, l])), 4) for k in feats[list(feats)[-1]]}
                     if ok else {})
            for lvl, fs in feats.items():
                Pf[lvl][:, :, l] = fold_models(np.stack([F[k][r_, l] for k in fs], 1), y, fd, balanced=False, prob=True)
                d[lvl] = est(y, oof(Pf[lvl][:, :, l], fd), W) if ok else na
                d[f'{lvl}_gain'] = est(y, oof(Pf[lvl][:, :, l], fd), W, maxp[0, :, l]) if ok else na
            per.append(d)
        pol = dict(oracle_mean_exit=round(float(np.maximum(commit, w0).mean()), 3),
                   final_acc=round(float((top1[:, -1] == g).mean()), 4), note='acc uses gold for reporting only')
        for target in (0.99, 0.98, 0.95):
            for k, P in list(Pf.items()) + [('stab', None)]:
                e = policy(stab if k == 'stab' else [P], agree, fd, w0, L, target)
                pol[f'{k}@{target}'] = dict(mean_exit=round(float(e.mean()), 3),
                                            agree=round(float(agree[np.arange(len(e)), e].mean()), 4),
                                            acc=round(float((top1[np.arange(len(e)), e] == g).mean()), 4),
                                            exit_before_last=round(float((e < L - 1).mean()), 4))
        out[head] = dict(per_layer=per, policy=pol, n_rows=len(r_), features=feats)
        print(f'  exit[{head}]: policy@0.99 oracle {pol["oracle_mean_exit"]} | ' +
              ' | '.join(f'{k} {pol[f"{k}@0.99"]["mean_exit"]} ({pol[f"{k}@0.99"]["agree"]})'
                         for k in list(Pf) + ['stab']), flush=True)
    return out


def loop_tasks(run, F, tags, qid, fold, gold, out4, keep_q, L):
    N, res = len(qid), {}
    f0 = [k for k, t in tags.items() if t == 'L0' and F[k].shape[0] == N]
    f1 = [k for k, t in tags.items() if t == 'L1' and not k.endswith(FINAL_REF)]  # online L1: xv_*_n, xv_*_pmax
    for mp in sorted(glob.glob(os.path.join(run, 'loop', '*', 'meta.json')) +
                     glob.glob(os.path.join(run, 'loop_allq', '*', 'meta.json'))):
        meta = json.load(open(mp))
        cfg, d = meta['config'], os.path.dirname(mp)
        name = f'{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}'
        if cfg['name'] == 'none':
            continue
        rows = np.load(os.path.join(d, 'rows.npy'))
        post = CL.top1(np.load(os.path.join(d, 'final4.npy')))
        sel = keep_q[qid[rows]]
        rows, post = rows[sel], post[sel]
        # pre-loop reference: the same-batch identity run of this mode (same batch and first pass; the extraction pass
        # differs from it by bf16 re-batching noise on ~1% of rows; loop_eval.py vs_none), else the extraction pass
        nd, ref4, ref = os.path.join(os.path.dirname(d), f"{meta['mode']}_none"), out4, 'extraction'
        if os.path.exists(os.path.join(nd, 'meta.json')) and np.isin(rows, np.load(os.path.join(nd, 'rows.npy'))).all():
            ref4, ref = out4.copy(), f"{meta['mode']}_none"
            ref4[np.load(os.path.join(nd, 'rows.npy'))] = np.load(os.path.join(nd, 'final4.npy'))
        pre, pre_x = CL.top1(ref4[rows]), CL.top1(out4[rows])
        targets = {'change': (np.ones(len(rows), bool), post != pre),
                   'fix': (pre != gold[rows], post == gold[rows])}
        r = dict(mode=meta['mode'], reference=ref, ref_differs_from_extraction=round(float((pre != pre_x).mean()), 4),
                 n_rows=len(rows), change_rate=round(float((post != pre).mean()), 4),
                 fix_rate_among_wrong=round(float((post == gold[rows])[pre != gold[rows]].mean()), 4),
                 harm_rate_among_right=round(float((post != gold[rows])[pre == gold[rows]].mean()), 4))
        delta = (post == gold[rows]).astype(float) - (pre == gold[rows])
        Wr, fr = boot(qid[rows], B_ROW), fold[rows]
        r['net_acc_pp'] = dict(always=mean_ci(delta, Wr, 100), oracle=mean_ci(np.maximum(delta, 0), Wr, 100),
                               always_vs_extraction=mean_ci((post == gold[rows]).astype(float) - (pre_x == gold[rows]), Wr, 100),
                               gate_low_final_conf=gate(-F['csd_maxp'][rows, -1], delta, fr, Wr),
                               gate_low_conf_at_b=gate(-F['csd_maxp'][rows, cfg['b']], delta, fr, Wr))
        for tname, (sub, yy) in targets.items():
            rr, y = rows[sub], yy[sub].astype(int)
            if min(y.sum(), len(y) - y.sum()) < 20:
                r[tname] = dict(n=len(y), n_pos=int(y.sum()), skipped='fewer than 20 in a class')
                continue
            W, fd = boot(qid[rr], B_ROW), fold[rr]
            base = np.stack([F['csd_maxp'][rr, -1], F['csd_ent'][rr, -1]], 1)
            s_base = crossfit(base, y, fd)
            t = dict(n=len(y), n_pos=int(y.sum()), base_final_conf=est(y, s_base, W))
            for at in (cfg['a'], cfg['b']):
                X0 = np.stack([F[k][rr, at] for k in f0], 1)
                X1 = np.c_[X0, np.stack([F[k][qid[rr], at] for k in f1], 1)]
                t[f'csd_maxp@{at}'] = est(y, single_oof(F['csd_maxp'][rr, at], y, fd), W)
                for lvl, Xs in (('L0', X0), ('L1', X1)):
                    t[f'{lvl}@{at}'] = evaluate(Xs, y, fd, W, base, s_base)
            r[tname] = t
        res[name] = r
        nap = r['net_acc_pp']
        print(f'  loop {name}: n={r["n_rows"]} change={r["change_rate"]} fix={r["fix_rate_among_wrong"]} '
              f'harm={r["harm_rate_among_right"]} net pp always {nap["always"]} gate(final conf) '
              f'{nap["gate_low_final_conf"]} gate(conf@b) {nap["gate_low_conf_at_b"]}', flush=True)
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
    ap.add_argument('--labels', default='auto', choices=['auto', 'classify', 'probe', 'lens'],
                    help='auto: classify/questions.csv, else in-process probe classification, else lens stand-in')
    ap.add_argument('--skip_rows', action='store_true', help='skip the row-level exit/loop analyses')
    ap.add_argument('--rows_only', action='store_true', help='recompute only exit/loop into the existing JSON')
    a = ap.parse_args()
    t0 = time.time()
    run = C.run_dir(a.model, a.fmt)
    od = os.path.join(run, 'stage2')
    fmeta = json.load(open(os.path.join(od, 'features.json')))
    F = dict(np.load(os.path.join(od, 'features.npz')))
    tags, L, Q, N = fmeta['tags'], fmeta['L'], fmeta['Q'], fmeta['N']
    ex = os.path.join(run, 'extract')
    rows = json.load(open(os.path.join(ex, 'rows.json')))
    if os.path.exists(os.path.join(run, 'probe', 'oof_full.npy')) and not fmeta['has_probe']:
        print('WARNING: probe/oof_full.npy exists but features.npz has no probe features: rerun stage2_features.py')
    o, ua, lab = labels(run, rows, a.labels)
    sfx = {'classify': '', 'probe': '_probe', 'lens': '_standin'}[lab['mode']]
    qid, fold, gold = F['qid'], F['fold'].astype(int), np.array([r['gold'] for r in rows])
    qfold, answer = fold[0::4], gold[0::4]
    w0 = int(round(0.45 * L))
    has_probe = fmeta['has_probe']
    levels = ['L0', 'L1'] + (['L2'] if has_probe else [])
    for k, f in (('lens', 'extract/lens4.npy'), ('csd', 'extract/csd_raw4.npy')) + \
            ((('probe', 'probe/oof_full.npy'),) if has_probe else ()):
        F[f'{k}_top1'] = np.load(os.path.join(run, f)).argmax(-1)
    counts = {k: int((o == k).sum()) for k in 'ROMUX'}
    counts['U_avail'] = int(((o == 'U') & ua).sum())
    print(f'{a.model}__{a.fmt} labels={lab["mode"]}{" (STAND-IN, lens as probe)" if lab["standin"] else ""} {counts} '
          f'levels={levels} {lab.get("agree_with_classify_csv", "")}', C.stamp(t0), flush=True)

    out4 = np.load(os.path.join(ex, 'out_logits4.npy'))
    if a.rows_only:
        S = json.load(open(os.path.join(od, f'analysis{sfx}.json')))
        return finish(a, S, F, tags, o, qid, fold, gold, out4, run, od, sfx, L, w0, t0)
    X, T, depth, AQ = question_features(F, tags, L, w0, Q)
    base = np.stack([AQ['csd_maxp'][:, -1], np.nanmean(F['csd_ent'][:, -1].reshape(Q, 4), 1)], 1)
    eps = float(np.load(os.path.join(ex, 'lens_params.npz'))['eps'])
    P3 = pca_rows(a.model, a.fmt, [l for l in depth if l in (w0, int(0.65 * (L - 1) + 0.5), L - 1)], eps)
    print(f'question features: {len(X)}; depths {depth}; w0 {w0}; L3 layers {list(P3)}', C.stamp(t0), flush=True)
    S = dict(config=dict(model=a.model, fmt=a.fmt, labels=lab, counts=counts, levels=levels, depths=depth, w0=w0,
                         C=CREG, class_weight='balanced (question tasks)', n_boot_q=B_Q, n_boot_rows=B_ROW, seed=SEED,
                         baseline='final max softmax + entropy (version means)',
                         features_md5=C.md5_file(os.path.join(od, 'features.npz'))))
    cls = {c: (o == c) for c in 'ROMU'} | {'U_avail': (o == 'U') & ua, 'U_none': (o == 'U') & ~ua}
    S['lost_consensus'] = {f'{A}&csd': lost_consensus(F, A, 'csd', answer, F['out_votes'], cls)
                           for A in ['lens'] + (['probe'] if has_probe else [])}
    print('lost consensus:', json.dumps(S['lost_consensus']), flush=True)
    S['questions'] = question_tasks(X, T, depth, P3, base, o, ua, qfold, levels)
    print('question tasks done', C.stamp(t0), flush=True)
    S['profiles'] = {k: {c: rnd(AQ[k][m].mean(0)) for c, m in cls.items() if m.any()} for k in PROFILE if k in AQ}
    finish(a, S, F, tags, o, qid, fold, gold, out4, run, od, sfx, L, w0, t0)


def finish(a, S, F, tags, o, qid, fold, gold, out4, run, od, sfx, L, w0, t0):
    """Row-level analyses (exit, loop), then write the JSON (and the question table unless --rows_only)."""
    if not a.skip_rows:
        keep_q = o != 'X'
        S['exit'] = exit_tasks(F, tags, np.flatnonzero(np.repeat(keep_q, 4)), qid, fold, gold, L, w0)
        print('exit tasks done', C.stamp(t0), flush=True)
        S['loop'] = loop_tasks(run, F, tags, qid, fold, gold, out4, keep_q, L) or 'no RUN/loop*/*/meta.json yet'
    S['config'].update(seconds=round(time.time() - t0, 1), script_md5=C.md5_file(__file__),
                       common_md5=C.md5_file(C.__file__), classify_md5=C.md5_file(CL.__file__),
                       rows_only=bool(a.rows_only))
    C.save_json(S, os.path.join(od, f'analysis{sfx}.json'))
    if a.rows_only:
        print('wrote (rows only)', od, C.stamp(t0), flush=True)
        return
    with open(os.path.join(od, f'tables{sfx}.tsv'), 'w') as f:
        f.write('task\tlevel\tset\tn_feat\tauc\tlo\thi\tgain\tlo\thi\tresid\tlo\thi\n')
        for task, r in S['questions'].items():
            if 'skipped' in r:
                continue
            f.write(f'{task}\tbase\tfinal_conf\t2\t' + '\t'.join(map(str, r['base'])) + '\n')
            for k, v in list(r['sets'].items()) + [(f'{v["level"]}|{n}', v) for n, v in r['singles'].items()]:
                lvl, name = k.split('|', 1)
                f.write(f'{task}\t{lvl}\t{name}\t{v.get("n_feat", 1)}\t' +
                        '\t'.join(map(str, v['auc'] + v['gain'] + v['resid'])) + '\n')
    print('wrote', od, C.stamp(t0), flush=True)


if __name__ == '__main__':
    main()
