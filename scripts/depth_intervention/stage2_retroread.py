"""Stage 2: single-pass label-free O detector and gated retro-read (RESULTS §3: content-level O is depth-specific, but
un-gated early exit loses because the late layers also fix wrong mid-layer answers). CPU only.

Question: from one forward pass and without gold, when should a mid-layer readout replace the final output?
Candidate per row (readout R in {csd, lens}, both L0; layer L-1 of both is set to the real output):
  stable  m = latest layer in [w0, L-2] whose R top-1 has been the same non-tie letter on m-2..m (run counted from w0)
  fixed   m = 24 (L = 36) / 21 (L = 32), sensitivity
Disagreement row: candidate letter a != final letter b (real output; ties never switch). Gold, for evaluation only:
  rescue (a right, b wrong), late-fix (a wrong, b right), both wrong.
Features (one pass, no cross-version information, all L0):
  readouts  for X in {csd, lens} at m: max p, entropy, log p(a), log p(b), top-1 == a, run length at m; dwell on a / b
            and top-1 flips in [w0, L-1]; run of b ending at L-1; KL(p_l || p_l-1) sum / max over the last third and at
            L-1; max p and log p(a) at L-2, L-3
  final     max p, entropy, top1 - top2 margin, logit gap z(b) - z(a), log p(a), letter mass; CSD letter mass at m
  prior     csd_base4 (letter logits on the bare target prompt): base4[b] - base4[a], b / a is its top letter; population
            constants from unlabelled dev-fold rows: late drift (mean centred real-output log-probs minus mean centred R
            log-probs at m) at b minus at a, output-letter frequency at b minus at a; one-hot letter identity of a and b
            (gold letters are balanced over the 4 versions, so identity can only carry the model's letter bias)
  geometry  RMS norm at L-1 and its ratio to m, cos(h_L-1, h_L-2), cos(h_m, h_L-1), |h_L-1 - h_m| / |h_m|, relative
            update |h_l - h_l-1| / |h_l-1| at the last 3 layers and summed over the last third
Detectors (score high = trust a), by supervision level:
  L0  hand-made scores, signs fixed a priori, nothing fitted (combo: z-scores on training-fold disagreement rows)
  L1  logistic regression (lr; gradient boosting gb with --gb) on all features; target = which of a / b equals the leave-one-out
      4-version consensus (majority content of the OTHER 3 versions' real outputs, tie-break by summed 4-way probability;
      rows where neither does are dropped): computed from the model's own outputs, no gold
  L3  the same models trained on gold (rescue = 1, late-fix = 0); reference only
Policy: replace b by a when score >= tau; tau maximizes the summed gain over training-fold disagreement rows on 101
  coverage quantiles or 'never' (ties -> fewer switches); gain = [a = consensus] - [b = consensus] (L0/L1) or the gold
  [a right] - [b right] (L3); trained detectors pick tau on inner out-of-fold scores within the training folds.
Protocol: dev = folds 0-3, cross-fitted (train on 3 dev folds, evaluate the 4th, pooled); lockbox = fold 4 (train and pick
  tau on folds 0-3, evaluate once; only with --lockbox). The best label-free configuration (readout x rule x L0/L1
  detector) is selected by dev L1 net agreement, never by gold; the best L3 one by dev gold (reference).
Signals (descriptive, dev, gold for evaluation only): raw AUROC of every single feature (no sign fitted; < 0.5 = reverse
  direction) and of LR detectors trained on one feature group only / on all but that group (L1 and L3 targets).
Metrics: AUROC rescue vs late-fix (+ paired difference to the confidence baseline = -final max p), AUROC vs the L1 target,
  net single-version accuracy change (non-X rows), coverage, precision, O->R (O question 4/4 after), R harm (R question no
  longer 4/4); by-question bootstrap 95% CIs (2000). Pareto = coverage vs gold net gain over thresholds (descriptive, dev
  pooled scores). References: all (switch every disagreement row, = un-gated retro-read), oracle (rescue rows only).
Depth controls: runner_up = the same switches but to the real output's second-choice letter instead of the mid readout
  (does depth add anything beyond 'the final is unreliable here'?); global debias_freq = real output re-ranked by
  log p - log(dev output-letter frequency) on every row (L0, no depth: how much is plain letter-bias removal).
Outputs RUN/retroread/: summary.json, pareto.tsv, geom.npz (hidden-geometry cache).
Usage: [BREW_BENCH=ext] python scripts/stage2_retroread.py --model qwen3-4b-base --fmt base_fewshot [--lockbox]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
import argparse, csv, json, sys, time
import numpy as np
import torch
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from classify import top1, to_q, boot_weights, wci, ratio
from early_exit import stab_runs
from stage2_analysis import boot, est

FIX, KSTAB, DEV, LOCK, NQ, CREG, B_BOOT = {36: 24, 32: 21}, 3, (0, 1, 2, 3), 4, 101, 0.1, 2000
IDX = (np.arange(4)[None, :] - np.arange(4)[:, None]) % 4  # IDX[v, c] = letter showing content c in version v
L0 = ('conf_baseline', 'final_gap', 'conf_diff', 'dwell', 'prior', 'drift', 'freq', 'readouts_agree', 'combo')


def group_of(n):
    return ('csd' if n.startswith('csd_') else 'lens' if n.startswith('lens_') else 'geometry' if n.startswith('g_') else
            'prior' if n.startswith(('prior_', 'let_')) or n in ('drift_gap', 'freq_gap') else 'final')


# ----------------------------------------------------------------------------- readouts, candidates, consensus
def readout(z, out4, w0):
    """z (N, L, 4) letter logits, layer L-1 replaced by the real output -> log-probs, max p, entropy, top-1, KL, runs."""
    z = z.astype(np.float64)
    z[:, -1] = out4
    lp = z - logsumexp(z, -1, keepdims=True)
    p = np.exp(lp)
    kl = np.zeros(z.shape[:2])
    kl[:, 1:] = (p[:, 1:] * (lp[:, 1:] - lp[:, :-1])).sum(-1)
    t = top1(z)
    return dict(lp=lp, maxp=p.max(-1), ent=-(p * lp).sum(-1), t=t, kl=kl, run=stab_runs(t, w0))


def candidates(Rd, L):
    """{(readout, rule): (layer m (N,), letter a (N,))}; -1 = no candidate (no stable run, or a tie)."""
    out = {}
    for R, d in Rd.items():
        ar = np.arange(len(d['t']))
        ok = d['run'][:, :L - 1] >= KSTAB  # runs are counted from w0 (0 before it)
        m = np.where(ok.any(1), L - 2 - ok[:, ::-1].argmax(1), -1)
        out[(R, 'stable')] = (m, np.where(m >= 0, d['t'][ar, np.maximum(m, 0)], -1))
        out[(R, 'fixed')] = (np.full(len(ar), FIX[L]), d['t'][:, FIX[L]])
    return out


def consensus_letter(out4, b):
    """Leave-one-out 4-version consensus per row as a letter of the row's version (L1, no gold): majority content of the
    other 3 versions' real outputs, tie-break by their summed 4-way content probabilities."""
    N = len(b)
    v = np.arange(N) % 4
    P = to_q(np.exp(out4 - logsumexp(out4, -1, keepdims=True)))
    P = np.stack([P[:, k][:, IDX[k]] for k in range(4)], 1)  # (Q, version, content)
    S = (to_q(np.where(b >= 0, (b + v) % 4, -1))[..., None] == np.arange(4)) + 0.01 * P
    return ((S.sum(1, keepdims=True) - S).reshape(N, 4).argmax(1) - v) % 4


def geometry(model, fmt, ms, cache, chunk=512):
    """Hidden-state geometry (L0): per layer RMS, cos to the previous layer, relative update; per candidate-layer array in
    ms {key: (N,)}: cos(h_m, h_L-1), |h_L-1 - h_m| / |h_m|, |h_L-1| / |h_m|. Cached (recomputed if the m arrays change)."""
    if os.path.exists(cache):
        g = dict(np.load(cache))
        if all(f'm_{k}' in g and np.array_equal(g[f'm_{k}'], m) for k, m in ms.items()):
            return g
    hid, _ = C.open_hidden(model, fmt)
    N, L, d = hid.shape
    g = dict(rms=np.zeros((N, L), np.float32), cos=np.zeros((N, L), np.float32), upd=np.zeros((N, L), np.float32))
    g.update({f'{s}_{k}': np.zeros(N, np.float32) for k in ms for s in ('cosm', 'dist', 'rmsr')})
    for i in range(0, N, chunk):
        H = torch.from_numpy(np.array(hid[i:i + chunk]).view(np.int16)).view(torch.bfloat16).float()
        sl, nr, ar = slice(i, i + len(H)), H.norm(dim=-1), torch.arange(len(H))
        g['rms'][sl] = (nr / d ** 0.5).numpy()
        g['cos'][sl, 1:] = ((H[:, 1:] * H[:, :-1]).sum(-1) / (nr[:, 1:] * nr[:, :-1])).numpy()
        g['upd'][sl, 1:] = ((H[:, 1:] - H[:, :-1]).norm(dim=-1) / nr[:, :-1]).numpy()
        for k, m in ms.items():
            j = torch.from_numpy(np.maximum(m[sl], 0)).long()
            Hm, nm = H[ar, j], nr[ar, j]
            g[f'cosm_{k}'][sl] = ((Hm * H[:, -1]).sum(-1) / (nm * nr[:, -1])).numpy()
            g[f'dist_{k}'][sl] = ((H[:, -1] - Hm).norm(dim=-1) / nm).numpy()
            g[f'rmsr_{k}'][sl] = (nr[:, -1] / nm).numpy()
    g.update({f'm_{k}': m for k, m in ms.items()})
    np.savez(cache, **g)
    return g


# ----------------------------------------------------------------------------- features and L0 scores
def features(Rd, m, a, b, out4, base4, drift, freq, csd_mass, out_mass, g, gk, L, w0):
    """Single-pass features of one candidate configuration (all L0): name -> (N,) float64."""
    ar, mm, aa, bb, t3 = np.arange(len(b)), np.maximum(m, 0), np.maximum(a, 0), np.maximum(b, 0), int(round(2 * L / 3))
    F = {}
    for X, d in Rd.items():
        lp, t = d['lp'], d['t']
        F.update({f'{X}_m_maxp': d['maxp'][ar, mm], f'{X}_m_ent': d['ent'][ar, mm], f'{X}_m_lpa': lp[ar, mm, aa],
                  f'{X}_m_lpb': lp[ar, mm, bb], f'{X}_m_isa': t[ar, mm] == a, f'{X}_m_run': d['run'][ar, mm],
                  f'{X}_dwell_a': (t[:, w0:] == a[:, None]).sum(1), f'{X}_dwell_b': (t[:, w0:] == b[:, None]).sum(1),
                  f'{X}_flips': (t[:, w0 + 1:] != t[:, w0:-1]).sum(1), f'{X}_run_b': d['run'][:, -1],
                  f'{X}_kl_sum': d['kl'][:, t3:].sum(1), f'{X}_kl_max': d['kl'][:, t3:].max(1),
                  f'{X}_kl_last': d['kl'][:, -1]})
        for j in (2, 3):
            F[f'{X}_L-{j}_maxp'], F[f'{X}_L-{j}_lpa'] = d['maxp'][:, L - j], lp[ar, L - j, aa]
    lpo, so = out4 - logsumexp(out4, -1, keepdims=True), np.sort(out4, -1)
    F.update(fin_maxp=np.exp(lpo.max(-1)), fin_ent=-(np.exp(lpo) * lpo).sum(-1), fin_margin=so[:, -1] - so[:, -2],
             fin_gap=out4[ar, bb] - out4[ar, aa], fin_lpa=lpo[ar, aa], out_mass=out_mass, csd_mass_m=csd_mass[ar, mm],
             prior_gap=base4[bb] - base4[aa], prior_top_b=bb == base4.argmax(), prior_top_a=aa == base4.argmax(),
             drift_gap=drift[bb] - drift[aa], freq_gap=freq[bb] - freq[aa],
             g_rms_last=g['rms'][:, -1], g_rms_ratio=g[f'rmsr_{gk}'], g_cos_last=g['cos'][:, -1], g_cos_m=g[f'cosm_{gk}'],
             g_dist_m=g[f'dist_{gk}'], g_upd_late=g['upd'][:, t3:].sum(1), **{f'g_upd_L-{j}': g['upd'][:, L - j] for j in (1, 2, 3)},
             **{f'let_{n}{i}': x == i for i in range(4) for n, x in (('a', a), ('b', b))})
    return {k: np.nan_to_num(np.asarray(v, np.float64)) for k, v in F.items()}


def l0_scores(F, R, ref):
    """Hand-made L0 detectors (signs fixed a priori; high = trust the mid readout). ref: rows for the combo's z-scores."""
    O = 'lens' if R == 'csd' else 'csd'
    S = {'conf_baseline': -F['fin_maxp'], 'final_gap': -F['fin_gap'], 'conf_diff': F[f'{R}_m_maxp'] - F['fin_maxp'],
         'dwell': F[f'{R}_dwell_a'] - F[f'{R}_dwell_b'], 'prior': F['prior_gap'], 'drift': F['drift_gap'],
         'freq': F['freq_gap'], 'readouts_agree': F[f'{O}_m_isa'] + 1e-3 * F[f'{O}_m_lpa']}
    S['combo'] = sum((S[k] - S[k][ref].mean()) / (S[k][ref].std() + 1e-12) for k in ('final_gap', 'dwell', 'prior'))
    return S


# ----------------------------------------------------------------------------- detectors, thresholds, evaluation
def fit_score(kind, X, y, tr, te):
    """Fit on rows tr (labels y in {0, 1}) and return P(1) on rows te."""
    if len(np.unique(y[tr])) < 2:
        return np.zeros(int(te.sum()))
    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=CREG, class_weight='balanced', max_iter=5000)) \
        if kind == 'lr' else HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                            min_samples_leaf=40, l2_regularization=1.0,
                                                            class_weight='balanced', random_state=0)
    return mdl.fit(X[tr], y[tr].astype(int)).predict_proba(X[te])[:, 1]


def best_tau(s, gain):
    """Threshold maximizing sum(gain[s >= tau]) over the coverage quantiles of s; inf (never switch) if no gain > 0."""
    if len(s) == 0:
        return np.inf
    taus = np.unique(np.quantile(s, np.linspace(0, 1, NQ)))[::-1]  # increasing coverage: ties -> fewer switches
    obj = np.array([gain[s >= t].sum() for t in taus])
    return float(taus[obj.argmax()]) if obj.max() > 0 else np.inf


def split_scores(det, s0, X, y, gain, dis, fold, trf, te):
    """Scores of the disagreement rows in te and tau picked on folds trf (inner out-of-fold scores if trained)."""
    trn, ten = dis & np.isin(fold, trf), dis & te
    if det[0] == 'L0':
        return s0[ten], best_tau(s0[trn], gain[trn])
    lab = ~np.isnan(y)
    s_in = np.zeros(int(trn.sum()))
    for g in trf:
        s_in[fold[trn] == g] = fit_score(det[1], X, y, trn & lab & (fold != g), trn & (fold == g))
    return fit_score(det[1], X, y, trn & lab, ten), best_tau(s_in, gain[trn])


def crossfit(X, y, dis, fold, folds=DEV, kind='lr'):
    """Out-of-fold scores of the disagreement rows of `folds`, each fold scored by a model trained on the others."""
    s, lab = np.full(len(y), np.nan), ~np.isnan(y)
    for f in folds:
        te = dis & (fold == f)
        s[te] = fit_score(kind, X, y, dis & lab & np.isin(fold, [g for g in folds if g != f]), te)
    return s


def signals(X, names, y1, yg, dis, fold, rows, qid):
    """Descriptive (dev): raw single-feature AUROC rescue vs late-fix (no sign fitted) and LR detectors on one feature
    group only / all but that group (L1 and L3 targets), cross-fitted over the dev folds; by-question bootstrap CIs."""
    r = rows & ~np.isnan(yg)
    W, y, out = boot(qid[r], B_BOOT), yg[r].astype(int), dict(groups={}, single={})
    for G in sorted(set(map(group_of, names))):
        cols = np.array([group_of(n) == G for n in names])
        for key, yy, c in (('L1|only', y1, cols), ('L3|only', yg, cols), ('L3|drop', yg, ~cols)):
            out['groups'][f'{key}|{G}'] = est(y, crossfit(X[:, c], yy, dis, fold)[r], W)
    single = {n: est(y, X[r, j], W) for j, n in enumerate(names)}
    out['single'] = dict(sorted(single.items(), key=lambda kv: -abs(kv[1][0] - 0.5)))
    return out


def aurocs(s, yg, y1, base, rows, qid):
    """AUROC rescue vs late-fix (CI by question), paired difference to the confidence baseline, AUROC vs the L1 target."""
    r, r1 = rows & ~np.isnan(yg), rows & ~np.isnan(y1)
    W, W1, y = boot(qid[r], B_BOOT), boot(qid[r1], B_BOOT), yg[r].astype(int)
    return dict(n_rescue=int(y.sum()), n_late=int((1 - y).sum()), auc=est(y, s[r], W), vs_base=est(y, s[r], W, base[r]),
                auc_L1=est(y1[r1].astype(int), s[r1], W1))


def policy_eval(sw, a, b, b2, gold, gain1, qsel, cls, Wb):
    """Gold evaluation of switch decisions sw (N,) on the questions qsel (non-X), plus the L1 net agreement and the
    runner-up control (same switches to the real output's second choice b2)."""
    ok0, ok1 = to_q(b == gold)[qsel], to_q(np.where(sw, a, b) == gold)[qsel]
    s, c, n4 = to_q(sw)[qsel], cls[qsel], np.full(int(qsel.sum()), 4.0)
    dq, q4, isO, isR = (ok1.astype(int) - ok0).sum(1), ok1.all(1), c == 'O', c == 'R'
    r = dict(net=wci(ratio(dq, n4), Wb), L1_net=wci(ratio(to_q(sw * gain1)[qsel].sum(1), n4), Wb),
             coverage=round(float(s.mean()), 5), n_switch=int(s.sum()), n_rescue=int((s & ok1 & ~ok0).sum()),
             n_late=int((s & ok0 & ~ok1).sum()), n_both=int((s & ~ok0 & ~ok1).sum()),
             O_to_R=[int((q4 & isO).sum()), int(isO.sum())], O_to_R_rate=wci(ratio(q4 & isO, isO), Wb),
             R_harm=[int((~q4 & isR).sum()), int(isR.sum())], R_harm_rate=wci(ratio(~q4 & isR, isR), Wb),
             q_new_4of4=int((q4 & ~ok0.all(1)).sum()), q_lost_4of4=int((~q4 & ok0.all(1)).sum()),
             net_rows_by_class={k: round(float(dq[c == k].sum() / max(4 * (c == k).sum(), 1)), 5) for k in 'ROMU'})
    r['precision'] = round(r['n_rescue'] / max(r['n_rescue'] + r['n_late'], 1), 4)
    rs = np.repeat(qsel, 4) & sw
    r['runner_up'] = dict(net=round(float(((b2 == gold).astype(int) - (b == gold))[rs].sum() / (4 * qsel.sum())), 5),
                          frac_a_is_runner_up=round(float((a[rs] == b2[rs]).mean()), 4) if rs.any() else None)
    return r


def pareto(s, rows, a, b, gold, n_rows, ngrid=51):
    """Coverage (fraction of the split's rows switched) vs gold net gain, switching the top-k candidate rows by score."""
    o = np.flatnonzero(rows)[np.argsort(-s[rows], kind='mergesort')]
    cg = np.r_[0, np.cumsum((a[o] == gold[o]).astype(int) - (b[o] == gold[o]))]
    cr = np.r_[0, np.cumsum((a[o] == gold[o]) & (b[o] != gold[o]))]
    k = np.unique(np.round(np.linspace(0, len(o), ngrid)).astype(int))  # late-fixes switched = cr - cg
    return [(int(x), x / n_rows, cg[x] / n_rows, cr[x] / max(2 * cr[x] - cg[x], 1)) for x in k], \
        (int(cg.argmax()), float(cg.max() / n_rows))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
    ap.add_argument('--lockbox', action='store_true', help='also train on folds 0-3 and evaluate fold 4 (run once)')
    ap.add_argument('--out', default='retroread'); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--gb', action='store_true', help='also gradient-boosting detectors (slow on a loaded CPU)')
    A = ap.parse_args()
    trained = [(lvl, k) for k in (('lr', 'gb') if A.gb else ('lr',)) for lvl in ('L1', 'L3')]
    t0 = time.time()
    torch.set_num_threads(A.threads)
    rd = C.run_dir(A.model, A.fmt)
    ex, od = os.path.join(rd, 'extract'), os.path.join(rd, A.out)
    os.makedirs(od, exist_ok=True)
    meta = json.load(open(os.path.join(ex, 'meta.json')))
    N, L = meta['N'], meta['L']
    w0 = int(round(0.45 * L))
    rows = json.load(open(os.path.join(ex, 'rows.json')))
    assert all(r['row'] == i and r['qid'] == i // 4 and r['v'] == i % 4 for i, r in enumerate(rows))
    gold, fold, qid = np.array([r['gold'] for r in rows]), np.array([r['fold'] for r in rows]), np.arange(N) // 4
    qs = list(csv.DictReader(open(os.path.join(rd, 'classify', 'questions.csv'))))
    cls = np.array([r['outcome_v1_K3'] for r in qs])
    keep_q = cls != 'X'
    keep = np.repeat(keep_q, 4)
    out4 = np.load(os.path.join(ex, 'out_logits4.npy')).astype(np.float64)
    base4 = np.load(os.path.join(ex, 'csd_base4.npy')).astype(np.float64)
    b, b2 = top1(out4), np.argsort(out4, -1)[:, -2]
    nfc = np.array([int(r['n_final_correct']) for r in qs])
    assert (to_q(b == gold).sum(1)[keep_q] == nfc[keep_q]).all(), 'final correctness != classify n_final_correct'
    Rd = {k: readout(np.load(os.path.join(ex, f)), out4, w0) for k, f in (('csd', 'csd_raw4.npy'), ('lens', 'lens4.npy'))}
    cand = candidates(Rd, L)
    cons = consensus_letter(out4, b)
    csd_mass, out_mass = np.load(os.path.join(ex, 'csd_mass.npy')), np.load(os.path.join(ex, 'out_mass.npy'))
    G = geometry(A.model, A.fmt, {'csd_stable': cand[('csd', 'stable')][0], 'lens_stable': cand[('lens', 'stable')][0],
                                  'fixed': cand[('csd', 'fixed')][0]}, os.path.join(od, 'geom.npz'))
    qfold = fold[0::4]
    qsplit = {'dev': keep_q & np.isin(qfold, DEV), 'lock': keep_q & (qfold == LOCK)}
    splits = {'dev': [(f, [g for g in DEV if g != f]) for f in DEV]}
    if A.lockbox:
        splits['lock'] = [(LOCK, list(DEV))]
    Wq = {s: boot_weights(int(qsplit[s].sum()), B_BOOT) for s in splits}
    rsplit = {s: np.repeat(qsplit[s], 4) for s in qsplit}
    print(f'{A.model}__{A.fmt}{"" if C.BENCH == "core" else "__" + C.BENCH} N={N} L={L} w0={w0} fixed={FIX[L]} '
          f'classes={ {k: int((cls == k).sum()) for k in "ROMUX"} } splits={list(splits)} final ties={(b < 0).sum()}',
          C.stamp(t0), flush=True)
    S = dict(config=dict(model=A.model, fmt=A.fmt, bench=C.BENCH, N=N, L=L, w0=w0, fixed_layer=FIX[L], kstab=KSTAB,
                         dev_folds=DEV, lockbox_fold=LOCK, lockbox_evaluated=A.lockbox, n_boot=B_BOOT, C=CREG, nq=NQ,
                         base4=base4.round(3).tolist(), levels=dict(
                             L0='no training: single-pass readouts, csd_base4, hidden geometry, population constants of '
                                'unlabelled dev rows; thresholds by L1', L1='trained on the leave-one-out 4-version '
                                'consensus of the real outputs (no gold)', L3='trained / thresholded on gold (reference)'),
                         consensus='leave-one-out majority content of the other 3 versions\' real outputs', gb=A.gb,
                         classify_md5=C.md5_file(os.path.join(rd, 'classify', 'questions.csv')),
                         classify_last_layer=json.load(open(os.path.join(rd, 'classify', 'summary.json')))['config'].get(
                             'last_layer', 'probe')),
             configs={})
    freq0 = np.bincount(b[rsplit['dev'] & (b >= 0)], minlength=4) / (rsplit['dev'] & (b >= 0)).sum()
    bd = top1(out4 - logsumexp(out4, -1, keepdims=True) - np.log(freq0))
    swd = keep & (b >= 0) & (bd >= 0) & (bd != b)
    S['global_references'] = dict(debias_freq=dict(freq=freq0.round(4).tolist(), **{sp: policy_eval(
        swd, bd, b, b2, gold, (bd == cons).astype(int) - (b == cons), qsplit[sp], cls, Wq[sp]) for sp in splits}))
    print('debias_freq (no depth): ' + ' | '.join(f'{sp}: net {S["global_references"]["debias_freq"][sp]["net"]} cov '
                                                  f'{S["global_references"]["debias_freq"][sp]["coverage"]}' for sp in splits))
    ptsv, sel = [], []
    for (R, rule), (m, am) in cand.items():
        cname, gk = f'{R}/{rule}', (f'{R}_stable' if rule == 'stable' else 'fixed')
        dis = keep & (am >= 0) & (b >= 0) & (am != b)
        pop = rsplit['dev'] & (am >= 0)
        lpR = Rd[R]['lp'][np.arange(N), np.maximum(m, 0)]
        lpo = out4 - logsumexp(out4, -1, keepdims=True)
        drift = ((lpo - lpo.mean(-1, keepdims=True))[pop].mean(0) - (lpR - lpR.mean(-1, keepdims=True))[pop].mean(0))
        freq = np.bincount(b[rsplit['dev'] & (b >= 0)], minlength=4) / (rsplit['dev'] & (b >= 0)).sum()
        F = features(Rd, m, am, b, out4, base4, drift, freq, csd_mass, out_mass, G, gk, L, w0)
        names = sorted(F)
        X = np.stack([F[k] for k in names], 1)
        g1 = (am == cons).astype(int) - (b == cons)
        y1 = np.where(am == cons, 1.0, np.where(b == cons, 0.0, np.nan))
        gg = (am == gold).astype(int) - (b == gold)
        yg = np.where(gg == 1, 1.0, np.where(gg == -1, 0.0, np.nan))
        base = -F['fin_maxp']
        cs = dict(n_features=len(names), features=names, drift=drift.round(4).tolist(), freq=freq.round(4).tolist(),
                  ties_candidate=int((keep & (am < 0)).sum()), detectors={}, references={}, by_split={})
        for sp in splits:
            r = dis & rsplit[sp]
            dec = r & ~np.isnan(yg)
            l1d = r & ~np.isnan(y1)
            cs['by_split'][sp] = dict(
                rows=int(rsplit[sp].sum()), n_dis=int(r.sum()), n_rescue=int((r & (yg == 1)).sum()),
                n_late=int((r & (yg == 0)).sum()), n_both=int((r & np.isnan(yg)).sum()),
                m_hist=np.bincount(m[r], minlength=L)[w0:].tolist(),
                L1_target=dict(n=int(l1d.sum()), pos=int((l1d & (y1 == 1)).sum()), coverage_of_gold_decided=round(
                    float((dec & l1d).sum() / max(dec.sum(), 1)), 4), agree_with_gold=round(float(
                    (y1[dec & l1d] == yg[dec & l1d]).mean()), 4) if (dec & l1d).any() else None),
                prior=dict(rescue_share_b_prior_top=[int((dec & (yg == 1) & (F['prior_top_b'] > 0)).sum()),
                                                     int((dec & (F['prior_top_b'] > 0)).sum())],
                           rescue_share_b_not_prior_top=[int((dec & (yg == 1) & (F['prior_top_b'] == 0)).sum()),
                                                         int((dec & (F['prior_top_b'] == 0)).sum())],
                           rescue_share_a_prior_top=[int((dec & (yg == 1) & (F['prior_top_a'] > 0)).sum()),
                                                     int((dec & (F['prior_top_a'] > 0)).sum())],
                           transitions_a_to_b_rescue_late_both=[[[int((r & (am == i) & (b == j) & (yg == 1)).sum()),
                                                                  int((r & (am == i) & (b == j) & (yg == 0)).sum()),
                                                                  int((r & (am == i) & (b == j) & np.isnan(yg)).sum())]
                                                                 for j in range(4)] for i in range(4)],
                           letters_final=np.bincount(b[rsplit[sp] & (b >= 0)], minlength=4).tolist(),
                           letters_mid=np.bincount(am[rsplit[sp] & (am >= 0)], minlength=4).tolist()))
            for ref, sw in (('all', r), ('oracle', r & (yg == 1))):
                cs['references'].setdefault(ref, {})[sp] = policy_eval(sw, am, b, b2, gold, g1, qsplit[sp], cls, Wq[sp])
        cs['signals'] = signals(X, names, y1, yg, dis, fold, dis & rsplit['dev'], qid)
        sg = cs['signals']
        print(f'  {cname}: top single features {[(k, v[0]) for k, v in list(sg["single"].items())[:8]]}\n'
              f'  {cname}: groups {[(k, v[0]) for k, v in sg["groups"].items()]}', flush=True)
        for det in [('L0', k) for k in L0] + trained:
            dn = f'{det[0]}:{det[1]}'
            res = dict(level=det[0])
            for sp, folds in splits.items():
                s, sw, taus = np.full(N, np.nan), np.zeros(N, bool), []
                for f, trf in folds:
                    te = fold == f
                    s0 = l0_scores(F, R, dis & np.isin(fold, trf))[det[1]] if det[0] == 'L0' else None
                    st, tau = split_scores(det, s0, X, y1 if det[0] != 'L3' else yg, g1 if det[0] != 'L3' else gg,
                                           dis, fold, trf, te)
                    s[dis & te], taus = st, taus + [tau]
                    sw[dis & te] = st >= tau
                r = dis & rsplit[sp]
                res[sp] = dict(taus=[None if not np.isfinite(t) else round(t, 5) for t in taus],
                               **aurocs(s, yg, y1, base, r, qid),
                               policy=policy_eval(sw, am, b, b2, gold, g1, qsplit[sp], cls, Wq[sp]))
                if sp == 'dev':
                    curve, (kbest, nbest) = pareto(s, r, am, b, gold, int(rsplit[sp].sum()))
                    ptsv += [(R, rule, det[0], det[1]) + c for c in curve]
                    res['dev']['pareto_max_net_descriptive'] = dict(n_switch=kbest, net=round(nbest, 5))
            cs['detectors'][dn] = res
            p = res['dev']['policy']
            sel.append((cname, dn, det[0], p['L1_net'][0], p['net'][0]))
            print(f'  {cname:12s} {dn:18s} AUC {res["dev"]["auc"]} d_base {res["dev"]["vs_base"]} AUC_L1 '
                  f'{res["dev"]["auc_L1"][0]} | net {p["net"]} L1net {p["L1_net"][0]:+.4f} cov {p["coverage"]} prec '
                  f'{p["precision"]} O->R {p["O_to_R"]} Rharm {p["R_harm"]} | pareto {res["dev"]["pareto_max_net_descriptive"]}'
                  + (f' || LOCK AUC {res["lock"]["auc"]} net {res["lock"]["policy"]["net"]} cov '
                     f'{res["lock"]["policy"]["coverage"]} O->R {res["lock"]["policy"]["O_to_R"]} Rharm '
                     f'{res["lock"]["policy"]["R_harm"]}' if 'lock' in res else ''), flush=True)
        bs = cs['by_split']['dev']
        print(f'  {cname}: dev dis={bs["n_dis"]} rescue={bs["n_rescue"]} late={bs["n_late"]} both={bs["n_both"]} '
              f'L1 target {bs["L1_target"]} | all {cs["references"]["all"]["dev"]["net"]} oracle '
              f'{cs["references"]["oracle"]["dev"]["net"]}', C.stamp(t0), flush=True)
        S['configs'][cname] = cs
    # selection on dev: label-free by L1 net agreement (never gold); the L3 reference by dev gold net
    lf = max((x for x in sel if x[2] in ('L0', 'L1')), key=lambda x: x[3])
    l3 = max((x for x in sel if x[2] == 'L3'), key=lambda x: x[4])
    S['selection'] = {}
    for key, (cn, dn, lvl, _, _) in (('label_free_by_dev_L1', lf), ('L3_by_dev_gold', l3)):
        d = S['configs'][cn]['detectors'][dn]
        S['selection'][key] = dict(config=cn, detector=dn, level=lvl, **{sp: dict(policy=d[sp]['policy'], auc=d[sp]['auc'],
                                                                                   vs_base=d[sp]['vs_base']) for sp in splits})
        print(f'SELECTED {key}: {cn} {dn} ' + ' | '.join(
            f'{sp}: net {d[sp]["policy"]["net"]} cov {d[sp]["policy"]["coverage"]} O->R {d[sp]["policy"]["O_to_R"]} '
            f'Rharm {d[sp]["policy"]["R_harm"]} AUC {d[sp]["auc"]}' for sp in splits), flush=True)
    with open(os.path.join(od, 'pareto.tsv'), 'w') as f:
        f.write('readout\trule\tlevel\tdetector\tn_switch\tcoverage\tnet_gain\tprecision\n')
        for row in ptsv:
            f.write('\t'.join(str(round(x, 6)) if isinstance(x, float) else str(x) for x in row) + '\n')
    S['config'].update(seconds=round(time.time() - t0, 1), script_md5=C.md5_file(__file__), common_md5=C.md5_file(C.__file__))
    C.save_json(S, os.path.join(od, 'summary.json'))
    print('wrote', od, C.stamp(t0), flush=True)


if __name__ == '__main__':
    main()
