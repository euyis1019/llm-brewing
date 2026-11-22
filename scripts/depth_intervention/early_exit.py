"""Early exit (INTERFACES.md 'Early exit', PLAN 3.1, acceptance A1). CPU only.

Heads (N, L, 4) fp32: E-head = probe OOF logits RUN/probe/oof_{probe}.npy, E-csd = raw CSD logits. Layer L-1 of both
heads is the real output, so an exit at L-1 means "no early exit". Window start w0 = round(0.45 L).
Rules pick an exit layer per row without that row's gold; parameters are cross-fitted by fold (chosen on the
population rows of the other 4 folds by single-version accuracy):
  fix        one layer in [w0, L-1]
  stab(k,t)  first l >= w0 + k - 1 whose (non-tie) top-1 is the same on l-k+1..l with max 4-way softmax >= t at l;
             else L-1. Grid k in (2,3,4,6), t in (0.3,...,0.9)
  conf       layer of maximal 4-way softmax confidence in [w0, L-1] (no parameters; ranked as fp64 log max-softmax,
             since fp32 softmax saturates at 1.0 and argmax would then pick the first saturated layer)
  ORACLE     upper bound, uses gold: a row is fixed if any layer in [w0, L-1] reads it correctly
  NOEXIT_head_at_L-1  reference: the head's own L-1 readout, no exit. For E-head this isolates the label-trained probe's
             recalibration of the real output (its gain is not an early-exit effect)
Population: rows of non-X questions; classes = v1 K=3 outcomes from RUN/{classify_dir}/questions.csv.
Metrics: O->R (O question becomes 4/4 correct), per-version fix/break rates inside O, matched decoy (U and M questions
reweighted to O's n_final_correct distribution) 4/4 rate, excess = O->R - decoy, net single-version accuracy change,
R harm (R no longer 4/4), mean exit layer; bootstrap 95% CIs by question (2000). PLAN 2.4 decoy (decoyO_4of4,
excess_vs_decoyO, A1 verdict_vs_decoyO): decoy-O pairs (q, wrong c') from classify arrays.npz -> exit puts all 4
versions on c', stratified by #final versions on c' and reweighted to O's n_final_correct. Also a descriptive sweep over
fixed parameters (not selected). A1: some E-head rule with excess CI > 0 and accuracy drop <= 1 pp.
Outputs RUN/{out}/summary.json, exits.npz.
Usage: python scripts/early_exit.py --model qwen3-4b-base --fmt base_fewshot [--probe full] [--classify_dir classify]
"""
import argparse, csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from classify import top1, softmax, to_q, contents, load_inputs, boot_weights, wci, ratio, ci

KGRID, TGRID = (2, 3, 4, 6), (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def stab_runs(a, w0):
    """Length of the run of identical non-tie top-1 ending at each layer, counted from w0 (0 before w0 / at ties)."""
    run = np.zeros(a.shape, np.int16)
    run[:, w0] = a[:, w0] >= 0
    for l in range(w0 + 1, a.shape[1]):
        run[:, l] = np.where(a[:, l] < 0, 0, np.where(a[:, l] == a[:, l - 1], run[:, l - 1] + 1, 1))
    return run


def first_fire(fire, w0):
    """First layer >= w0 where `fire` is true, else L-1."""
    f = fire.copy()
    f[:, :w0] = False
    f[:, -1] = True
    return f.argmax(1)


def crossfit(cand_ok, fold, pop):
    """cand_ok (n_cand, N) bool. Per fold f, the candidate with the best accuracy on population rows of other folds."""
    return np.array([int(cand_ok[:, pop & (fold != f)].mean(1).argmax()) for f in range(5)])


def strat_rate(A, Dn, Nm):
    """Rate Nm/Dn per n_final stratum (columns), reweighted to O's strata counts A, as a statistic of weights w."""
    def f(w):
        pO, den, num = w @ A, w @ Dn, w @ Nm
        pO = np.where(den > 0, pO, 0.0)
        return (pO * num / np.maximum(den, 1e-12)).sum(1) / pO.sum(1)
    return f


def decoy_fn(isO, isUM, nfc, fixed):
    """U/M questions' 4/4-after rate reweighted to O's n_final_correct distribution, as a statistic of weights w."""
    A = np.stack([isO & (nfc == n) for n in range(4)], 1).astype(float)
    Dn = np.stack([isUM & (nfc == n) for n in range(4)], 1).astype(float)
    return strat_rate(A, Dn, Dn * fixed[:, None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
    ap.add_argument('--probe', default='full', choices=['full', 'bias', 'lens'])
    ap.add_argument('--classify_dir', default='classify'); ap.add_argument('--out', default='early_exit')
    a = ap.parse_args()
    t0 = time.time()
    D = load_inputs(a.model, a.fmt, a.probe, 'raw')
    N, L, gold, fold = D['N'], D['L'], D['gold'], D['fold']
    Q, w0 = N // 4, int(round(0.45 * L))
    cdir = os.path.join(D['rd'], a.classify_dir)
    qrows = list(csv.DictReader(open(os.path.join(cdir, 'questions.csv'))))
    cls = np.array([r['outcome_v1_K3'] for r in qrows])
    nfc = np.array([int(r['n_final_correct']) for r in qrows])
    src = np.array([r['source'] for r in qrows])
    cconf = json.load(open(os.path.join(cdir, 'summary.json')))['config']
    keep = cls != 'X'
    pop = np.repeat(keep, 4)
    az = np.load(os.path.join(cdir, 'arrays.npz'))  # decoy-O pairs (Q, 4); None if classify predates decoyO_K3
    dO = az['decoyO_K3'] & keep[:, None] if 'decoyO_K3' in az else None
    nfc_c = (az['final_c'][:, :, None] == np.arange(4)).sum(1)  # (Q, 4) final versions on each content
    isO, isR, isUM = cls == 'O', cls == 'R', np.isin(cls, ['U', 'M'])
    Wb = boot_weights(Q)
    od = os.path.join(D['rd'], a.out)
    os.makedirs(od, exist_ok=True)
    print(f'{a.model}__{a.fmt} probe={a.probe} classify={a.classify_dir} (probe {cconf["probe"]}, csd {cconf["csd"]})'
          f' w0={w0} O={isO.sum()} R={isR.sum()} U+M={isUM.sum()} X={(~keep).sum()}', C.stamp(t0), flush=True)
    ar = np.arange(N)

    def evaluate(after, e, full=True, let=None):
        na, nb = to_q(after).sum(1), to_q(base_ok).sum(1)
        fixed = (na == 4) & keep
        oR, dec = ratio(isO & fixed, isO), decoy_fn(isO, isUM, nfc, fixed)
        m = dict(O_to_R_count=int((isO & fixed).sum()), O_to_R=wci(oR, Wb), decoy_4of4=wci(dec, Wb),
                 excess=wci(lambda w: oR(w) - dec(w), Wb), net_acc_change=ci(Wb, (na - nb) * keep, 4 * keep),
                 R_harm_count=int((isR & ~fixed).sum()), R_harm=ci(Wb, isR & ~fixed, isR),
                 mean_exit_layer=round(float(e[pop].mean()), 2), frac_rows_exit_early=round(float((e < L - 1)[pop].mean()), 4))
        if full:
            wO, rO = to_q(~base_ok).sum(1) * isO, to_q(base_ok).sum(1) * isO
            m.update(O_partial_fix_count=int((isO & (na > nb) & (na < 4)).sum()), O_worse_count=int((isO & (na < nb)).sum()),
                     O_fix_rate_wrong_versions=ci(Wb, to_q(after & ~base_ok).sum(1) * isO, wO),
                     O_break_rate_right_versions=ci(Wb, to_q(~after & base_ok).sum(1) * isO, rO),
                     acc_before=round(float(base_ok[pop].mean()), 5), acc_after=round(float(after[pop].mean()), 5),
                     rows_wrong_to_right=int((after & ~base_ok & pop).sum()),
                     rows_right_to_wrong=int((~after & base_ok & pop).sum()),
                     mean_exit_layer_O_rows=round(float(e[np.repeat(isO, 4)].mean()), 2) if isO.any() else None,
                     O_to_R_by_source={s: [int((isO & fixed & (src == s)).sum()), int((isO & (src == s)).sum())]
                                       for s in C.SOURCES},
                     strata_nfc_O_Ofixed_UM_UMfixed={n: [int((isO & (nfc == n)).sum()), int((isO & (nfc == n) & fixed).sum()),
                                                         int((isUM & (nfc == n)).sum()),
                                                         int((isUM & (nfc == n) & fixed).sum())] for n in range(4)})
            if dO is not None:  # PLAN 2.4 decoy: decoy-O pair (q, c') 'fixed' = exit puts all 4 versions on c'
                on = (contents(to_q(lt[ar, e] if let is None else let))[:, :, None] == np.arange(4)).all(1)
                st = lambda x: np.stack([(x & (nfc_c == n)).sum(1) for n in range(4)], 1).astype(float)
                sym = strat_rate(np.stack([isO & (nfc == n) for n in range(4)], 1).astype(float), st(dO), st(dO & on))
                m.update(decoyO_pairs_to_c=[int((dO & on).sum()), int(dO.sum())], decoyO_4of4=wci(sym, Wb),
                         excess_vs_decoyO=wci(lambda w: oR(w) - sym(w), Wb))
        return m

    out4 = D['out4']
    S = dict(config=dict(model=a.model, fmt=a.fmt, probe=a.probe, probe_file=D['probe_file'],
                         classify_dir=a.classify_dir, classify_config=dict(probe=cconf['probe'], csd=cconf['csd']),
                         w0=w0, L=L, kgrid=KGRID, tgrid=TGRID, n_boot=len(Wb), population_rows=int(pop.sum()),
                         classes={k: int((cls == k).sum()) for k in 'ROMUX'},
                         note='exit at L-1 = real output (no early exit); cross-fitted parameters use the other '
                              'folds\' OOF readouts (their probes saw this fold\'s labels in training)'),
             heads={}, sweep={})
    exits = {}
    for head in ('E-head', 'E-csd'):
        H = (D['probe'] if head == 'E-head' else D['csd']).copy()
        H[:, L - 1] = out4
        lt, conf = top1(H), softmax(H).max(-1)
        sH = np.sort(H.astype(np.float64), -1)
        lconf = -np.log1p(np.exp(sH[..., :-1] - sH[..., -1:]).sum(-1))  # log max-softmax, no fp32 saturation
        ok = lt == gold[:, None]
        base_ok = ok[:, L - 1]
        assert (to_q(base_ok).sum(1)[keep] == nfc[keep]).all(), 'base correctness != classify n_final_correct'
        res, sw = {}, dict(fix={}, stab={})
        # fix
        layers = np.arange(w0, L)
        ch = crossfit(ok[:, w0:].T, fold, pop)
        e = layers[ch[fold]]
        res['fix'] = dict(chosen_layer_per_fold=layers[ch].tolist(), **evaluate(ok[ar, e], e))
        exits[f'{head}|fix'] = e
        for l in layers:
            sw['fix'][int(l)] = evaluate(ok[:, l], np.full(N, l), full=False)
        # stab(k, tau)
        run = stab_runs(lt, w0)
        grid = [(k, t) for k in KGRID for t in TGRID]
        E = np.stack([first_fire((run >= k) & (conf >= t), w0) for k, t in grid])
        cand_ok = ok[ar[None], E]
        ch = crossfit(cand_ok, fold, pop)
        e = E[ch[fold], ar]
        res['stab'] = dict(chosen_k_tau_per_fold=[list(grid[c]) for c in ch], **evaluate(ok[ar, e], e))
        exits[f'{head}|stab'] = e
        for gi, (k, t) in enumerate(grid):
            sw['stab'][f'k{k}_t{t}'] = evaluate(cand_ok[gi], E[gi], full=False)
        # conf
        e = w0 + lconf[:, w0:].argmax(1)
        res['conf'] = evaluate(ok[ar, e], e)
        exits[f'{head}|conf'] = e
        # oracle (uses gold)
        anyok = ok[:, w0:].any(1)
        e = np.where(anyok, w0 + ok[:, w0:].argmax(1), L - 1)
        res['ORACLE_upper_bound_uses_gold'] = evaluate(anyok, e)
        lh = top1((D['probe'] if head == 'E-head' else D['csd'])[:, L - 1])  # the head's own L-1 readout
        res['NOEXIT_head_at_L-1'] = evaluate(lh == gold, np.full(N, L - 1), let=lh)
        S['heads'][head], S['sweep'][head] = res, sw
        for r, m in res.items():
            print(f'{head:6s} {r:28s} O->R {m["O_to_R_count"]:3d}/{isO.sum()} {m["O_to_R"]}  decoy {m["decoy_4of4"]}  '
                  f'excess {m["excess"]}  decoyO {m.get("decoyO_pairs_to_c")} {m.get("decoyO_4of4")} excess_vs_decoyO '
                  f'{m.get("excess_vs_decoyO")}  net {m["net_acc_change"]}  Rharm {m["R_harm_count"]} {m["R_harm"][0]}  '
                  f'exit {m["mean_exit_layer"]} early {m["frac_rows_exit_early"]}', flush=True)
        best = max(sw['stab'].items(), key=lambda kv: kv[1]['excess'][1])
        print(f'{head:6s} sweep: best stab by excess-CI-lo {best[0]} {best[1]["excess"]} net {best[1]["net_acc_change"][0]}',
              flush=True)

    # acceptance A1 (PLAN 7): some E-head rule with excess CI excluding 0 and accuracy drop <= 1 pp
    A1 = {}
    for r in ('fix', 'stab', 'conf'):
        m = S['heads']['E-head'][r]
        sig, harm_ok = m['excess'][1] > 0, m['net_acc_change'][0] >= -0.01
        A1[r] = dict(excess=m['excess'], net_acc_change=m['net_acc_change'], excess_significant=sig,
                     acc_drop_le_1pp=harm_ok, pass_=bool(sig and harm_ok))
    verdict = 'PASS' if any(v['pass_'] for v in A1.values()) else \
        'PARTIAL (excess significant only with >1pp accuracy drop)' if any(v['excess_significant'] for v in A1.values()) else 'FAIL'
    S['A1'] = dict(rules=A1, verdict=verdict, probe=a.probe, note='any of 3 rules (no multiplicity correction); compare '
                   'with head_at_L1_no_exit: the E-head is label-trained, so part of any gain is recalibration, not exit',
                   head_at_L1_no_exit={k: S['heads']['E-head']['NOEXIT_head_at_L-1'][k]
                                       for k in ('O_to_R_count', 'O_to_R', 'net_acc_change', 'R_harm_count')})
    if dO is not None:  # same rule against the PLAN 2.4 decoy (decoy-O pairs reaching 4/4 on their wrong content)
        sig2 = {r: S['heads']['E-head'][r]['excess_vs_decoyO'][1] > 0 for r in A1}
        S['A1'].update(excess_vs_decoyO={r: S['heads']['E-head'][r]['excess_vs_decoyO'] for r in A1},
                       verdict_vs_decoyO='PASS' if any(sig2[r] and A1[r]['acc_drop_le_1pp'] for r in A1) else
                       'PARTIAL (excess significant only with >1pp accuracy drop)' if any(sig2.values()) else 'FAIL')
    print('A1 verdict:', S['A1']['verdict'], '| vs decoy-O:', S['A1'].get('verdict_vs_decoyO'), '(probe=%s)' % a.probe,
          {r: (v['excess'], v['net_acc_change'][0], v['pass_']) for r, v in A1.items()}, flush=True)
    np.savez_compressed(os.path.join(od, 'exits.npz'), **{k: v.astype(np.int8) for k, v in exits.items()})
    S['config'].update(seconds=round(time.time() - t0, 1), script_md5=C.md5_file(__file__),
                       common_md5=C.md5_file(C.__file__))
    C.save_json(S, os.path.join(od, 'summary.json'))
    print('wrote', od, C.stamp(t0), flush=True)


if __name__ == '__main__':
    main()
