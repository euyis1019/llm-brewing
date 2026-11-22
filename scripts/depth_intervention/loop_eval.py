"""Post-loop v1 classes and the A2 report (PLAN §3.2, §7; INTERFACES "Depth loop"). CPU.

For every complete RUN/{loop_dir}/{mode}_{cfg}/: probe logits per step = origin-layer probe of the question's fold,
lens(h) + hn @ W[f, j].T + b[f, j] (RUN/probe/weights_full.npz) on traj_hidden (logit-lens stand-in with --probe lens or
when the weights are missing); CSD = traj_csd4; final real output = final4 / final_top1. Ties, contents, the v1 rule
(FJC_c over consecutive steps) and the bootstrap are imported from classify.py.
Original class: RUN/classify/questions.csv when made with the same probe, else recomputed with classify.classify_v1 on
the extraction readouts. Per mode x config: transition matrix orig -> post; U change rate (-> R/O/M) and its paired
difference to each placebo (bootstrap 95% CI by question); direction among U (to correct: R or O, i.e. a correct-content
FJC; to wrong: M or a wrong-content FJC absent from the original trajectory); R harm; row-level final flip rate per
original class; ties. Three references: `vs_orig` (official class of the extraction pass), `vs_none` (the same-batch
identity run of the mode: same numerics, so only the loop changes anything; the extraction differs from any re-batched
pass by bf16 noise, ~1.5% final-argmax flips on near-ties) and `vs_none_letters` (same, compliance ignored: classes from
letter-restricted outputs, so a format-breaking config such as p4_11 is not hidden in X).
A2 (PLAN §7) per reference and mode: some window config (w*/d*) whose U change minus every placebo has CI > 0, with
to-correct > to-wrong.  Outputs RUN/{loop_dir}/eval/summary.json, post_classes.csv.
Usage: python scripts/loop_eval.py --model qwen3-4b-base --fmt base_fewshot [--loop_dir loop] [--probe full] [--K 3]
"""
import argparse, csv, glob, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import classify as CL


def probe_apply(hid, lens, seq, folds, Wz, eps, chunk=256):
    """Probe logits (n, S, 4) = lens + fold-f correction of each step's origin block; hid (n, S, d) bf16 bits."""
    if Wz is None:
        return lens
    n, S, _ = lens.shape
    W, b = torch.from_numpy(Wz['W'][:, seq].astype(np.float32)), Wz['b'][:, seq]  # (5, S, 4, d), (5, S, 4)
    out = lens.copy()
    for i in range(0, n, chunk):
        h = torch.from_numpy(C.bf16_bits_to_f32(hid[i:i + chunk]))
        hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
        f = folds[i:i + chunk]
        for ff in np.unique(f):
            s = np.where(f == ff)[0]
            out[i + s] += torch.einsum('msd,skd->msk', hn[s], W[ff]).numpy() + b[ff]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
    ap.add_argument('--loop_dir', default='loop'); ap.add_argument('--probe', default='full', choices=['full', 'lens'])
    ap.add_argument('--K', type=int, default=3); ap.add_argument('--eval_out', default='eval')
    ap.add_argument('--last_layer', default='probe', choices=['probe', 'output'],
                    help="'output': probe readout at the last step replaced by the real output (as classify.py)")
    a = ap.parse_args()
    t0 = time.time()
    rd = C.run_dir(a.model, a.fmt)
    ld, od = os.path.join(rd, a.loop_dir), os.path.join(rd, a.loop_dir, a.eval_out)
    os.makedirs(od, exist_ok=True)
    wp = os.path.join(rd, 'probe', 'weights_full.npz')
    probe = 'full' if a.probe == 'full' and os.path.exists(wp) else 'lens'
    Wz = dict(np.load(wp)) if probe == 'full' else None
    eps = float(np.load(os.path.join(rd, 'extract', 'lens_params.npz'))['eps'])
    dirs = sorted(d for d in glob.glob(os.path.join(ld, '*_*')) if os.path.exists(os.path.join(d, 'meta.json')))
    rows = np.load(os.path.join(dirs[0], 'rows.npy'))
    assert all(np.array_equal(np.load(os.path.join(d, 'rows.npy')), rows) for d in dirs), 'configs differ in rows'
    assert (rows.reshape(-1, 4) == 4 * (rows[::4] // 4)[:, None] + np.arange(4)).all(), 'need 4 versions per question'
    qid, Q = rows[::4] // 4, len(rows) // 4

    # original trajectory (extraction readouts of the target rows, same probe kind)
    D = CL.load_inputs(a.model, a.fmt, probe)
    ans, folds, src = D['gold'][rows[::4]], D['fold'][rows], D['source'][rows[::4]]
    cont = lambda x: CL.contents(CL.to_q(CL.top1(x)))
    def pcont(pr, f4):
        let = CL.top1(pr)
        if a.last_layer == 'output':
            let[:, -1] = CL.top1(f4)
        return CL.contents(CL.to_q(let))
    pc0, cc0, fc0 = pcont(D['probe'][rows], D['out4'][rows]), cont(D['csd'][rows]), cont(D['out4'][rows])
    r0 = CL.classify_v1(pc0, cc0, fc0, ans, a.K, ~CL.to_q(D['compliant'][rows]).all(1))
    dF0 = CL.decoy_o(pc0, cc0, fc0, ans, a.K)[0]
    orig, orig_src, csv_agree = r0['outcome'], f'recomputed from extraction (probe {probe})', None
    qcsv, qsum = os.path.join(rd, 'classify', 'questions.csv'), os.path.join(rd, 'classify', 'summary.json')
    if os.path.exists(qcsv) and os.path.exists(qsum) and json.load(open(qsum))['config']['probe'] == probe and \
            json.load(open(qsum))['config'].get('last_layer', 'probe') == a.last_layer:
        off = {int(r['qid']): r[f'outcome_v1_K{a.K}'] for r in csv.DictReader(open(qcsv))}
        o_csv = np.array([off[q] for q in qid])
        csv_agree = float((o_csv == orig).mean())
        orig, orig_src = o_csv, qcsv
    isU, isR = orig == 'U', orig == 'R'
    f0row = CL.top1(D['out4'][rows])
    probe_check = None  # fold weights applied here vs train_probe's OOF logits, on extraction states of 256 rows
    if Wz is not None:
        k = rows[:256]
        lens0 = np.load(os.path.join(rd, 'extract', 'lens4.npy'), mmap_mode='r')[k]
        pk = probe_apply(C.open_hidden(a.model, a.fmt)[0][k], lens0, np.arange(D['L']), folds[:256], Wz, eps)
        probe_check = dict(maxdiff=float(np.abs(pk - D['probe'][k]).max()),
                           argmax_agree=float((pk.argmax(-1) == D['probe'][k].argmax(-1)).mean()))
    print(f'{a.model}__{a.fmt} {a.loop_dir}: Q={Q} probe={probe} orig={orig_src} (csv agree {csv_agree}) '
          f'counts', {k: int((orig == k).sum()) for k in CL.V1}, 'probe_check', probe_check, C.stamp(t0), flush=True)

    # post-loop classes
    post = {}
    for d in dirs:
        name, m = os.path.basename(d), json.load(open(os.path.join(d, 'meta.json')))
        lens = np.load(os.path.join(d, 'traj_lens4.npy'))
        hid = np.memmap(os.path.join(d, 'traj_hidden_bf16.u16'), np.uint16, 'r', shape=lens.shape[:2] + (D['meta']['d'],))
        pl = probe_apply(hid, lens, np.load(os.path.join(d, 'seq.npy')), folds, Wz, eps)
        cs, f4 = np.load(os.path.join(d, 'traj_csd4.npy')), np.load(os.path.join(d, 'final4.npy'))
        excl = ~CL.to_q(np.isin(np.load(os.path.join(d, 'final_top1.npy')), D['meta']['letter_ids'])).all(1)
        pc, cc, fc = pcont(pl, f4), cont(cs), cont(f4)
        r = CL.classify_v1(pc, cc, fc, ans, a.K, excl)
        post[name] = dict(mode=m['mode'], cfg=m['config']['name'], o=r['outcome'], nf=r['n_final'], nX=int(excl.sum()),
                          oL=CL.outcome_v1(r['fjc'], r['n_final'], r['wrong_consistent']),
                          dF=CL.decoy_o(pc, cc, fc, ans, a.K)[0], frow=CL.top1(f4),
                          ties=dict(final_rows=int((CL.top1(f4) < 0).sum()), probe_steps=int((CL.top1(pl) < 0).sum()),
                                    csd_steps=int((CL.top1(cs) < 0).sum())))
        print('  read', name, C.stamp(t0), flush=True)

    # statistics against three references: vs_orig = official class of the extraction pass; vs_none = class of the
    # same-batch `none` run of the mode; vs_none_letters = same but compliance ignored (letter-restricted outputs, no X)
    Wb = CL.boot_weights(Q)

    def stats(ref, dFr, frr, p, plac, key):
        """Post classes p[key] vs reference classes ref (dFr: reference wrong-content FJC, frr: final letters)."""
        o, isU, isR = p[key], ref == 'U', ref == 'R'
        chg = lambda x: np.isin(x, ['R', 'O', 'M']) & isU
        ch, toC = chg(o), np.isin(o, ['R', 'O']) & isU
        newW = (p['dF'] & ~dFr).any(1) & isU
        toW = (o == 'M') & isU | newW
        flip = CL.to_q(p['frow'] != frr).sum(1)
        return dict(
            transition={k: {k2: int(((ref == k) & (o == k2)).sum()) for k2 in CL.V1} for k in CL.V1 if (ref == k).any()},
            U=dict(n=int(isU.sum()), n_change=int(ch.sum()), n_to_correct=int(toC.sum()), n_to_wrong=int(toW.sum()),
                   n_wrong_fjc_formed=int(newW.sum()), post={k: int(((o == k) & isU).sum()) for k in CL.V1},
                   to_X=CL.ci(Wb, (o == 'X') & isU, isU),
                   change=CL.ci(Wb, ch, isU), to_correct=CL.ci(Wb, toC, isU), to_wrong=CL.ci(Wb, toW, isU),
                   correct_minus_wrong=CL.wci(lambda w: CL.ratio(toC, isU)(w) - CL.ratio(toW, isU)(w), Wb),
                   minus_placebo={c: CL.wci(lambda w, x=chg(pp[key]): CL.ratio(ch, isU)(w) - CL.ratio(x, isU)(w), Wb)
                                  for c, pp in plac.items()}),
            R_harm=dict(n=int(isR.sum()), not_R=CL.ci(Wb, (o != 'R') & isR, isR),
                        not_4of4=CL.ci(Wb, (p['nf'] < 4) & isR, isR)),
            class_changed=CL.ci(Wb, o != ref),
            flip_rows={k: CL.ci(Wb, flip * (ref == k), 4 * (ref == k)) for k in 'ROMU' if (ref == k).any()})
    res = {}
    for name, p in post.items():
        plac = {pp['cfg']: pp for pn, pp in post.items()
                if pp['mode'] == p['mode'] and pp['cfg'].startswith('p') and pn != name}
        nn = post.get(f"{p['mode']}_none")
        res[name] = dict(mode=p['mode'], cfg=p['cfg'], ties=p['ties'], noncompliant_questions=p['nX'],
                         vs_orig=stats(orig, dF0, f0row, p, plac, 'o'),
                         vs_none=None if nn is None else stats(nn['o'], nn['dF'], nn['frow'], p, plac, 'o'),
                         vs_none_letters=None if nn is None else stats(nn['oL'], nn['dF'], nn['frow'], p, plac, 'oL'))
    REFS = ('vs_orig', 'vs_none', 'vs_none_letters')
    A2 = {}
    for ref in REFS:
        for mode in sorted({r['mode'] for r in res.values()}):
            rr = {r['cfg']: r[ref] for r in res.values() if r['mode'] == mode and r[ref] is not None}
            plac, win = [c for c in rr if c.startswith('p')], [c for c in rr if c[0] in 'wd']
            ok = lambda c: all(rr[c]['U']['minus_placebo'][p][1] > 0 for p in plac) and \
                rr[c]['U']['n_to_correct'] > rr[c]['U']['n_to_wrong']
            passing = [c for c in win if plac and ok(c)]
            A2[f'{ref}|{mode}'] = dict(windows=win, placebos=plac, passing=passing,
                                       n_U=rr[win[0]]['U']['n'] if win else 0,
                                       verdict=bool(passing) if plac and win and rr[win[0]]['U']['n'] else None)
    S = dict(config=dict(model=a.model, fmt=a.fmt, loop_dir=a.loop_dir, probe=probe, K=a.K, Q=Q,
                         orig_source=orig_src, orig_csv_agree_recomputed=csv_agree, probe_check=probe_check,
                         end_truncation_in_classify=bool(CL.first_run(np.array([[False, False, True]]), 3)[0] == 2),
                         seconds=round(time.time() - t0, 1), script_md5=C.md5_file(__file__),
                         classify_md5=C.md5_file(CL.__file__), common_md5=C.md5_file(C.__file__)),
             orig_counts={k: int((orig == k).sum()) for k in CL.V1},
             orig_counts_by_source={s: {k: int(((orig == k) & (src == s)).sum()) for k in CL.V1} for s in C.SOURCES},
             A2=A2, A2_rule='U change minus every placebo: CI lower bound > 0, and n_to_correct > n_to_wrong; '
                            'uncorrected over windows. vs_orig: U = official class (extraction pass, differs from any '
                            're-batched pass by bf16 noise); vs_none: U = class of the same-batch none run; '
                            'vs_none_letters: same with compliance ignored (no X; a format-breaking config counts)',
             configs=res)
    C.save_json(S, os.path.join(od, 'summary.json'))
    with open(os.path.join(od, 'post_classes.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['qid', 'source', 'orig', 'orig_recomputed'] + list(post))
        for i in range(Q):
            w.writerow([qid[i], src[i], orig[i], r0['outcome'][i]] + [p['o'][i] for p in post.values()])
    f3 = lambda x: f'{x[0]:+.3f}[{x[1]:+.3f},{x[2]:+.3f}]'
    for ref in REFS:
        print(f'--- reference {ref}: {"config":16s} {"U change":>24s} {"C/W":>7s} {"minus placebos":>52s} '
              f'{"R harm":>6s} {"changed":>7s} {"U->X":>5s}')
        for name, r in res.items():
            if r[ref] is None:
                continue
            u = r[ref]['U']
            print(f'{name:18s} n_U={u["n"]:<4d} {f3(u["change"]):>24s} {u["n_to_correct"]:>3d}/{u["n_to_wrong"]:<3d} '
                  f'{" ".join(f"{c}:{f3(x)}" for c, x in u["minus_placebo"].items()):>52s} '
                  f'{r[ref]["R_harm"]["not_R"][0]:6.3f} {r[ref]["class_changed"][0]:7.3f} {u["to_X"][0]:5.2f}')
    print('A2', json.dumps(A2), flush=True)
    print('DONE', C.stamp(t0), flush=True)


if __name__ == '__main__':
    main()
