"""Taxonomy (INTERFACES.md 'Taxonomy'): v1 content-level outcomes, v0 first-paper outcomes, controls. CPU only.

Readouts (fp32, row = 4*qid + v): probe logits RUN/probe/oof_{probe}.npy (lens falls back to extract/lens4.npy),
CSD extract/csd_raw4.npy (--csd sub: minus csd_base4), real output extract/out_logits4.npy (+ out_top1.npy for
compliance). A readout points to a letter only if top1 - top2 >= 1e-4 logits; a tie is uncertain (never correct).
v1 (per question; P4/C4[l] = all 4 versions' probe/CSD readouts point to the correct content at layer l):
  FPCL_c / FJC_c = first l with P4 / P4&C4 true on l..min(l+K-1, L-1) (end truncation, INTERFACES 2026-10-16).
  FJC_c exists -> R (final 4/4 correct) or O;
  else M (all 4 final outputs on one wrong content) or U. X = any non-compliant row (excluded).
  Sub-tags: u_avail (FPCL_c exists), n_final_correct.
v0 (per row, /nvme0/if/llm-brewing/source_code/brewing/diagnostics/{metrics,outcome}.py): NB if the probe is never
  correct; R/O if probe and CSD are correct at a common layer; else M if the CSD tail confidence >= 0.5 else U.
Controls: null probes (oof_null*: same v1 rule vs the real gold) and matched decoy O (wrong contents c').
Importable core (used by early_exit.py / loop_eval.py): top1, to_q, contents, first_run, content_level, final_info,
  outcome_v1, classify_v1, decoy_o, outcome_v0, tail_confidence, load_inputs, boot_weights, ci, wci.
Outputs RUN/{out}/: questions.csv, rows_v0.csv, arrays.npz, summary.json.
Usage: python scripts/classify.py --model qwen3-4b-base --fmt base_fewshot [--probe full] [--csd raw] [--out classify]
"""
import argparse, csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

TIE, KS, B_BOOT, SEED = 1e-4, (1, 2, 3), 2000, 0
V1, V0 = 'ROMUX', ('R', 'O', 'M', 'U', 'NB', 'X')


# ----------------------------------------------------------------------------- core rule (importable)
def top1(logits, tie=TIE):
    """Argmax over the last axis; -1 where the top-2 margin is below `tie` (uncertain)."""
    s = np.sort(logits, -1)
    return np.where(s[..., -1] - s[..., -2] >= tie, logits.argmax(-1), -1)


def softmax(x):
    p = np.exp(x - x.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def to_q(x):
    """(N, ...) with row = 4*qid + v -> (Q, 4, ...)."""
    return x.reshape((-1, 4) + x.shape[1:])


def contents(letters):
    """Letters (Q, 4, ...) per version v (-1 = tie) -> canonical content (letter + v) % 4 (-1 kept)."""
    v = np.arange(4).reshape((1, 4) + (1,) * (letters.ndim - 2))
    return np.where(letters >= 0, (letters + v) % 4, -1)


def first_run(x, K):
    """x (..., S) bool -> first s with x[..., s:min(s+K, S)] all true, else -1 (end truncation: a run reaching the
    last step counts even if shorter than K)."""
    xp = np.concatenate([x, np.ones(x.shape[:-1] + (K - 1,), bool)], -1)
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,), np.int32), np.cumsum(xp, -1, dtype=np.int32)], -1)
    win = (c[..., K:] - c[..., :-K]) == K
    return np.where(win.any(-1), win.argmax(-1), -1)


def content_level(probe_ok, csd_ok, K):
    """probe_ok, csd_ok (Q, 4, S) bool: version-level 'readout points to the target content' along a layer (or loop
    step) axis. Returns P4, C4 (Q, S) and FPCL_c, FJC_c (Q,) for K consecutive steps (-1 = none)."""
    P4, C4 = probe_ok.all(1), csd_ok.all(1)
    return P4, C4, first_run(P4, K), first_run(P4 & C4, K)


def final_info(final_c, answer):
    """final_c (Q, 4) content of the real output per version (-1 = tie/none). Returns n_final_correct (Q,) and
    final_wrong_consistent (Q,): all 4 outputs on one wrong content."""
    n = (final_c == answer[:, None]).sum(1)
    wc = (final_c == final_c[:, :1]).all(1) & (final_c[:, 0] >= 0) & (final_c[:, 0] != answer)
    return n, wc


def outcome_v1(fjc, n_final_correct, wrong_consistent, excluded=None):
    """'R'/'O' if FJC_c exists (final 4/4 correct or not), else 'M' (final wrong-consistent) or 'U'; 'X' if excluded."""
    o = np.where(fjc >= 0, np.where(n_final_correct == 4, 'R', 'O'), np.where(wrong_consistent, 'M', 'U'))
    return o if excluded is None else np.where(excluded, 'X', o)


def classify_v1(probe_c, csd_c, final_c, answer, K, excluded=None):
    """Contents (Q, 4, S) of probe / CSD readouts and (Q, 4) of the real output -> dict of v1 fields."""
    t = answer[:, None, None]
    P4, C4, fpcl, fjc = content_level(probe_c == t, csd_c == t, K)
    n, wc = final_info(final_c, answer)
    return dict(P4=P4, C4=C4, fpcl=fpcl, fjc=fjc, n_final=n, wrong_consistent=wc, u_avail=fpcl >= 0,
                outcome=outcome_v1(fjc, n, wc, excluded))


def decoy_o(probe_c, csd_c, final_c, answer, K):
    """Matched decoy: for each content c the FJC_c rule with 'correct' replaced by 'points to c'. Returns decoy-FJC
    (Q, 4) and decoy-O (Q, 4) = decoy-FJC and final not 4/4 on c; both False at c = answer."""
    dF = np.zeros(final_c.shape, bool)
    for c in range(4):
        dF[:, c] = (content_level(probe_c == c, csd_c == c, K)[3] >= 0) & (answer != c)
    return dF, dF & ~(final_c[:, :, None] == np.arange(4)).all(1)


def tail_confidence(logits, L):
    """First-paper CSD tail confidence: max over classes of the mean 4-way softmax over layers >= int(0.75 L)."""
    return softmax(logits[:, int(L * 0.75):]).mean(1).max(-1)


def outcome_v0(probe_ok, csd_ok, final_ok, tail_conf, excluded=None, thr=0.5):
    """First-paper per-row rule; probe_ok, csd_ok (n, L) bool. Returns FPCL, FJC (-1 = none), outcome."""
    fpcl, fjc = first_run(probe_ok, 1), first_run(probe_ok & csd_ok, 1)
    o = np.where(fpcl < 0, 'NB', np.where(fjc >= 0, np.where(final_ok, 'R', 'O'),
                                          np.where(tail_conf >= thr, 'M', 'U')))
    return fpcl, fjc, (o if excluded is None else np.where(excluded, 'X', o))


# ----------------------------------------------------------------------------- data + statistics
def probe_path(rd, name):
    p = os.path.join(rd, 'probe', f'oof_{name}.npy')
    return p if os.path.exists(p) or name != 'lens' else os.path.join(rd, 'extract', 'lens4.npy')


def load_inputs(model, fmt, probe='full', csd='raw'):
    """Row-level readouts of one run (fp32) plus per-row metadata."""
    rd = C.run_dir(model, fmt)
    ex = os.path.join(rd, 'extract')
    meta = json.load(open(os.path.join(ex, 'meta.json')))
    rows = json.load(open(os.path.join(ex, 'rows.json')))
    assert all(r['row'] == i and r['qid'] == i // 4 and r['v'] == i % 4 for i, r in enumerate(rows))
    cs = np.load(os.path.join(ex, 'csd_raw4.npy'))
    if csd == 'sub':
        cs = cs - np.load(os.path.join(ex, 'csd_base4.npy'))
    pp = probe_path(rd, probe)
    return dict(rd=rd, meta=meta, N=meta['N'], L=meta['L'], probe_file=pp, probe=np.load(pp).astype(np.float32),
                csd=cs.astype(np.float32), out4=np.load(os.path.join(ex, 'out_logits4.npy')),
                compliant=np.isin(np.load(os.path.join(ex, 'out_top1.npy')), meta['letter_ids']),
                gold=np.array([r['gold'] for r in rows]), fold=np.array([r['fold'] for r in rows]),
                source=np.array([r['source'] for r in rows]))


def boot_weights(Q, B=B_BOOT, seed=SEED):
    """(B, Q) multiplicities of a by-question bootstrap."""
    idx = np.random.default_rng(seed).integers(0, Q, (B, Q)) + Q * np.arange(B)[:, None]
    return np.bincount(idx.ravel(), minlength=B * Q).reshape(B, Q).astype(np.float64)


def wci(fn, Wb):
    """Estimate (unit weights) and bootstrap 95% CI of a statistic fn(w), w (B, Q) per-question weights -> (B,)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        pt, b = fn(np.ones((1, Wb.shape[1])))[0], fn(Wb)
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return [round(float(x), 5) for x in (pt, lo, hi)]


def ratio(num, den=None):
    num = np.asarray(num, np.float64)
    den = np.ones_like(num) if den is None else np.asarray(den, np.float64)
    return lambda w: (w @ num) / (w @ den)


def ci(Wb, num, den=None):
    """[estimate, lo, hi] of sum(num)/sum(den) over questions (den=None: mean)."""
    return wci(ratio(num, den), Wb)


def layer_hist(x, L):
    x = x[x >= 0]
    return dict(n=int(len(x)), mean=round(float(x.mean()), 2) if len(x) else None,
                median=float(np.median(x)) if len(x) else None, hist=np.bincount(x, minlength=L).tolist())


def rnd(x, k=4):
    return [round(float(v), k) for v in x]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
    ap.add_argument('--probe', default='full', choices=['full', 'bias', 'lens'])
    ap.add_argument('--csd', default='raw', choices=['raw', 'sub']); ap.add_argument('--out', default='classify')
    ap.add_argument('--last_layer', default='probe', choices=['probe', 'output'],
                    help="'output': at layer L-1 the probe readout is replaced by the real output (CSD at L-1 already is),"
                         " so every final-4/4-correct question has FJC_c and is R (fixes 'U_final4')")
    a = ap.parse_args()
    t0 = time.time()
    D = load_inputs(a.model, a.fmt, a.probe, a.csd)
    N, L, gold = D['N'], D['L'], D['gold']
    Q = N // 4
    od = os.path.join(D['rd'], a.out)
    os.makedirs(od, exist_ok=True)
    answer, src, fold = gold[0::4], D['source'][0::4], D['fold'][0::4]
    excl = ~to_q(D['compliant']).all(1)
    keep = ~excl
    p_let, c_let, f_let = top1(D['probe']), top1(D['csd']), top1(D['out4'])
    p_let_raw = p_let.copy()  # v0 and per-layer probe accuracy always use the probe itself (first-paper compatible)
    if a.last_layer == 'output':
        p_let[:, L - 1] = f_let
    probe_c, csd_c, final_c = contents(to_q(p_let)), contents(to_q(c_let)), contents(to_q(f_let))
    Wb = boot_weights(Q)
    print(f'{a.model}__{a.fmt} probe={a.probe} ({D["probe_file"]}) csd={a.csd} Q={Q} L={L} X={excl.sum()}',
          C.stamp(t0), flush=True)

    def tally(o, labels, mask=None):
        m = np.ones(len(o), bool) if mask is None else mask
        d = {'all': {k: int(((o == k) & m).sum()) for k in labels}}
        d.update({s: {k: int(((o == k) & m & (src == s)).sum()) for k in labels} for s in C.SOURCES})
        return d

    # ---- v1 for K = 1, 2, 3
    S = dict(config=dict(model=a.model, fmt=a.fmt, probe=a.probe, probe_file=D['probe_file'], csd=a.csd, tie=TIE, last_layer=a.last_layer,
                         Ks=KS, n_boot=B_BOOT, seed=SEED, N=N, Q=Q, L=L),
             exclusions=dict(noncompliant_rows=int((~D['compliant']).sum()), X_questions=tally(excl, [True])),
             v1={})
    t = answer[:, None, None]
    rev = ((probe_c == t).all(1) & (csd_c == t).all(1))[:, ::-1]
    suffix = np.where(rev.all(1), L, rev.argmin(1))  # length of the P4&C4 run ending at the last layer
    res, decoys = {}, {}
    for K in KS:
        r = res[K] = classify_v1(probe_c, csd_c, final_c, answer, K, excl)
        o, nf = r['outcome'], r['n_final']
        dF, dO = decoys[K] = decoy_o(probe_c, csd_c, final_c, answer, K)
        isO = (o == 'O').astype(float)
        dOq = dO.sum(1) / 3.0 * keep  # X questions out of every numerator below (denominators are keep)
        s = dict(counts=tally(o, V1),
                 frac_nonX={k: ci(Wb, (o == k), keep) for k in 'ROMU'},
                 frac_nonX_by_source={sr: {k: ci(Wb, (o == k) & (src == sr), keep & (src == sr)) for k in 'ROMU'}
                                      for sr in C.SOURCES} if K == 3 else None,
                 U_avail=int(((o == 'U') & r['u_avail']).sum()),
                 U_none=int(((o == 'U') & ~r['u_avail']).sum()),
                 U_avail_by_source={sr: int(((o == 'U') & r['u_avail'] & (src == sr)).sum()) for sr in C.SOURCES},
                 U_avail_share=ci(Wb, (o == 'U') & r['u_avail'], o == 'U'),
                 U_final4=int(((o == 'U') & (nf == 4)).sum()),
                 R_end_truncated=int(((o == 'R') & (r['fjc'] > L - K)).sum()),  # FJC_c only via a final run < K
                 U_avail_end_truncated=int(((o == 'U') & (r['fpcl'] > L - K)).sum()),
                 n_final_correct_by_class={k: np.bincount(nf[o == k], minlength=5).tolist() for k in 'ROMU'},
                 FPCL_c={k: layer_hist(r['fpcl'][o == k], L) for k in 'ROMU'},
                 FJC_c={k: layer_hist(r['fjc'][o == k], L) for k in 'RO'},
                 FPCL_c_exists=ci(Wb, (r['fpcl'] >= 0) & keep, keep), FJC_c_exists=ci(Wb, (r['fjc'] >= 0) & keep, keep),
                 decoy=dict(O_rate=ci(Wb, isO, keep), decoyO_rate=ci(Wb, dOq, keep),
                            O_minus_decoyO=wci(lambda w: ratio(isO, keep)(w) - ratio(dOq, keep)(w), Wb),
                            FJC_rate=ci(Wb, (r['fjc'] >= 0) & keep, keep),
                            decoyFJC_rate=ci(Wb, dF.sum(1) / 3.0 * keep, keep),
                            decoyO_by_source={sr: ci(Wb, dOq * (src == sr), keep & (src == sr)) for sr in C.SOURCES},
                            O_by_source={sr: ci(Wb, isO * (src == sr), keep & (src == sr)) for sr in C.SOURCES},
                            n_questions_with_any_decoyO=int((dO.any(1) & keep).sum())))
        if K == 3:
            s['per_layer'] = {k: dict(n=int((o == k).sum()), P4=rnd(r['P4'][o == k].mean(0)),
                                      C4=rnd(r['C4'][o == k].mean(0)),
                                      P4C4=rnd((r['P4'] & r['C4'])[o == k].mean(0))) for k in 'ROMU' if (o == k).any()}
            s['per_layer']['all_nonX'] = dict(n=int(keep.sum()), P4=rnd(r['P4'][keep].mean(0)),
                                              C4=rnd(r['C4'][keep].mean(0)),
                                              P4C4=rnd((r['P4'] & r['C4'])[keep].mean(0)))
        S['v1'][f'K{K}'] = s
        c = s['counts']['all']
        print(f'v1 K={K}: ' + ' '.join(f'{k}={c[k]}' for k in V1) + f"  U_avail={s['U_avail']} U_final4={s['U_final4']}"
              f" R_trunc={s['R_end_truncated']}"
              f"  O={s['decoy']['O_rate'][0]:.4f} decoyO={s['decoy']['decoyO_rate'][0]:.4f}"
              f" diff={s['decoy']['O_minus_decoyO']}", flush=True)

    # ---- row-level accuracy per layer (non-X rows) and v0
    keep_r = np.repeat(keep, 4)
    S['row_acc_per_layer'] = dict(probe=rnd((p_let_raw == gold[:, None])[keep_r].mean(0)),
                                  csd=rnd((c_let == gold[:, None])[keep_r].mean(0)),
                                  final=round(float((f_let == gold)[keep_r].mean()), 5))
    tconf = tail_confidence(D['csd'], L)
    fpcl0, fjc0, o0 = outcome_v0(p_let_raw == gold[:, None], c_let == gold[:, None], f_let == gold, tconf,
                                 ~D['compliant'])
    brew = np.isin(o0, ['R', 'O', 'M', 'U'])
    src_r = D['source']
    S['v0'] = dict(counts={'all': {k: int((o0 == k).sum()) for k in V0},
                           **{sr: {k: int(((o0 == k) & (src_r == sr)).sum()) for k in V0} for sr in C.SOURCES}},
                   frac_brewing={k: ci(Wb, to_q(o0 == k).sum(1), to_q(brew).sum(1)) for k in 'ROMU'},
                   nb_rate=ci(Wb, to_q(o0 == 'NB').sum(1), to_q(D['compliant']).sum(1)),
                   mean_fpcl_normalized=round(float(fpcl0[(fpcl0 >= 0) & D['compliant']].mean() / L), 4),
                   mean_fjc_normalized=round(float(fjc0[(fjc0 >= 0) & D['compliant']].mean() / L), 4),
                   mean_delta_brew=round(float((fjc0 - fpcl0)[(fjc0 >= 0) & D['compliant']].mean()), 3),
                   tail_start=int(L * 0.75), threshold=0.5,
                   crosstab_v1K3_rows={k: {k0: int(((np.repeat(res[3]['outcome'], 4) == k) & (o0 == k0)).sum())
                                           for k0 in V0} for k in V1})
    print('v0 rows: ' + ' '.join(f'{k}={v}' for k, v in S['v0']['counts']['all'].items()), flush=True)

    # ---- null-probe control: same v1 rule with oof_null*, vs the real gold and vs the null's own permuted answers
    nulls, own, pd_ = {}, {}, os.path.join(D['rd'], 'probe')
    nlab = np.load(os.path.join(pd_, 'null_labels.npy')) if os.path.exists(os.path.join(pd_, 'null_labels.npy')) else None
    for sd in range(3):
        p = os.path.join(pd_, f'oof_null{sd}.npy')
        if not os.path.exists(p):
            continue
        nc = contents(to_q(top1(np.load(p).astype(np.float32))))
        nulls[sd] = {K: classify_v1(nc, csd_c, final_c, answer, K, excl) for K in KS}
        if nlab is not None:
            pa = contents(to_q(nlab[sd].astype(np.int64)))
            assert (pa == pa[:, :1]).all(), 'null labels are not one content per question'
            own[sd] = (pa[:, 0], {K: classify_v1(nc, csd_c, final_c, pa[:, 0], K, excl) for K in KS})

    def nstats(rr, K, ne=None):
        o, m = rr[K]['outcome'], keep if ne is None else keep & ne
        return dict(FPCL_c_exists=ci(Wb, (rr[K]['fpcl'] >= 0) & m, m), FJC_c_exists=ci(Wb, (rr[K]['fjc'] >= 0) & m, m),
                    O_rate=ci(Wb, (o == 'O') & m, m), frac={k: round(float(((o == k) & m).sum() / m.sum()), 5) for k in 'ROMU'})

    S['null'] = dict(seeds=sorted(nulls), own_labels=nlab is not None, note=(
        'vs_gold: the v1 rule with the null probe in place of the probe, scored against the real answer (the null keeps '
        'the frozen lens term, so this includes lens-readable signal); vs_own: scored against the null probe\'s own '
        'permuted answer, restricted to questions where it differs from the real answer = chance-level false-positive '
        'rate of FPCL_c / FJC_c / O for a content the probe was trained on but the state does not encode'))
    for K in KS if nulls else ():
        per = {}
        for sd, rr in nulls.items():
            per[sd] = dict(vs_gold=nstats(rr, K), agree_with_real=round(float((rr[K]['outcome'] == res[K]['outcome'])[keep].mean()), 4))
            if K == 3:
                per[sd]['P4_per_layer_vs_gold'] = rnd(rr[K]['P4'][keep].mean(0))
            if sd in own:
                pa, ro = own[sd]
                per[sd].update(vs_own=nstats(ro, K, pa != answer), perm_equals_answer=round(float((pa == answer).mean()), 4))
        mean = {f'{v}_{k}': round(float(np.mean([per[sd][v][k][0] for sd in per])), 5)
                for v in ('vs_gold', 'vs_own') if all(v in per[sd] for sd in per) for k in ('FPCL_c_exists', 'FJC_c_exists', 'O_rate')}
        real = dict(FPCL_c_exists=S['v1'][f'K{K}']['FPCL_c_exists'][0], FJC_c_exists=S['v1'][f'K{K}']['FJC_c_exists'][0],
                    O_rate=S['v1'][f'K{K}']['decoy']['O_rate'][0])
        S['null'][f'K{K}'] = dict(per_seed=per, mean_over_seeds=mean, real_probe=real)
        print(f'null K={K}: mean {mean}  real {real}', flush=True)

    # ---- ties
    alt = classify_v1(contents(to_q(top1(D['probe'], 0.0))), contents(to_q(top1(D['csd'], 0.0))),
                      contents(to_q(top1(D['out4'], 0.0))), answer, 3, excl)['outcome']
    S['ties'] = dict(threshold=TIE, probe_row_layer=int((p_let_raw < 0).sum()), csd_row_layer=int((c_let < 0).sum()),
                     final_rows=int((f_let < 0).sum()), of_row_layer=N * L, of_rows=N,
                     v1K3_changed_if_argmax_breaks_ties=int((alt != res[3]['outcome']).sum()))
    print('ties:', S['ties'], flush=True)

    # ---- per-question / per-row outputs
    with open(os.path.join(od, 'questions.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        cols = ['qid', 'source', 'fold', 'answer', 'n_noncompliant'] + [f'final_c{v}' for v in range(4)] + \
               ['n_final_correct', 'final_wrong_consistent', 'pc4_suffix_len']
        for K in KS:
            cols += [f'outcome_v1_K{K}', f'FPCL_c_K{K}', f'FJC_c_K{K}', f'u_avail_K{K}', f'n_decoyO_K{K}']
        cols += [f'outcome_null{sd}_K3' for sd in sorted(nulls)] + [f'outcome_null{sd}_own_K3' for sd in sorted(own)]
        w.writerow(cols)
        ncomp = (~to_q(D['compliant'])).sum(1)
        for q in range(Q):
            line = [q, src[q], fold[q], answer[q], ncomp[q]] + final_c[q].tolist() + \
                   [res[3]['n_final'][q], int(res[3]['wrong_consistent'][q]), suffix[q]]
            for K in KS:
                r = res[K]
                line += [r['outcome'][q], r['fpcl'][q], r['fjc'][q], int(r['u_avail'][q]), int(decoys[K][1][q].sum())]
            line += [nulls[sd][3]['outcome'][q] for sd in sorted(nulls)] + [own[sd][1][3]['outcome'][q] for sd in sorted(own)]
            w.writerow(line)
    with open(os.path.join(od, 'rows_v0.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row', 'qid', 'v', 'source', 'fold', 'gold', 'compliant', 'final_letter', 'final_correct',
                    'FPCL', 'FJC', 'tail_conf', 'outcome_v0'])
        for i in range(N):
            w.writerow([i, i // 4, i % 4, src_r[i], D['fold'][i], gold[i], int(D['compliant'][i]), f_let[i],
                        int(f_let[i] == gold[i]), fpcl0[i], fjc0[i], round(float(tconf[i]), 5), o0[i]])
    np.savez_compressed(os.path.join(od, 'arrays.npz'), P4=res[3]['P4'], C4=res[3]['C4'], answer=answer, suffix=suffix,
                        final_c=final_c, excluded=excl, **{f'outcome_K{K}': res[K]['outcome'] for K in KS},
                        **{f'fpcl_K{K}': res[K]['fpcl'] for K in KS}, **{f'fjc_K{K}': res[K]['fjc'] for K in KS},
                        **{f'decoyO_K{K}': decoys[K][1] for K in KS})  # (Q, 4) decoy-O per wrong content (early_exit)
    S['config'].update(seconds=round(time.time() - t0, 1), script_md5=C.md5_file(__file__),
                       common_md5=C.md5_file(C.__file__))
    C.save_json(S, os.path.join(od, 'summary.json'))
    print('wrote', od, C.stamp(t0), flush=True)


if __name__ == '__main__':
    main()
