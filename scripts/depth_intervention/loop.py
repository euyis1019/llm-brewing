"""Depth loop at the critical token (PLAN §3.2; INTERFACES "Depth loop").

Layer sequence 0..b, (a..b) x k, b+1..L-1 (same weights, no training); damped configs (d*) apply
h + beta*(block(h) - h) on the repeated steps only. Modes:
  all   every position follows the looped sequence (LM.forward_sequence, one pass per config)
  last  only the critical position loops: one normal pass per batch (use_cache=True) shared by all configs; the stream
        restarts from the block-b output and attends to the first-pass prefix keys/values (LM.run_single)
Per step (fp32): logit lens and CSD at the step's origin layer; final letter logits and full-vocab top-1.
Steps 0..b are the normal pass (mode last by construction; mode all checked, `prefix_maxdiff` in meta), so their CSD is
computed once per batch and shared by all configs. Mode last also re-runs every batch from block L//2 without
repetition (`selfcheck` in meta: must reproduce the normal pass).
Outputs RUN/{out}/{mode}_{cfg}/: rows.npy seq.npy step_kind.npy final4.npy final_top1.npy traj_csd4.npy traj_lens4.npy
traj_hidden_bf16.u16 (n, S, d) meta.json; done.npy makes an interrupted run resumable; configs with meta.json are skipped.
Config names are literal block indices; the default list is the L=36 list scaled by L/36.
Targets RUN/loop/targets.json: all v1 (K=3) U questions + 300 random each of R/M/O (seed 0) from
RUN/classify/questions.csv. --debug_n n: n random questions -> RUN/loop_debug/targets_debug.json, outputs there
(--out names another directory under RUN).
Usage: CUDA_VISIBLE_DEVICES=0 python scripts/loop.py --model qwen3-4b-base --fmt base_fewshot --mode last
"""
import argparse, csv, json, os, re, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CONFIGS36 = ('none', 'w16_23_k1', 'w16_23_k2', 'w18_25_k1', 'w18_25_k2', 'w20_27_k1', 'w20_27_k2', 'd18_25_k2',
             'p4_11_k1', 'p28_33_k1')
BETA = 0.5
PAT = re.compile(r'([wdp])(\d+)_(\d+)_k(\d+)')


def scale_name(name, L):
    """L=36 config name -> the same config for an L-block model (block i -> round(i*L/36))."""
    m = PAT.fullmatch(name)
    if m is None or L == 36:
        return name
    s = lambda i: int(int(i) * L / 36 + 0.5)
    return f'{m[1]}{s(m[2])}_{s(m[3])}_k{m[4]}'


def parse_config(name, L):
    """seq: origin block of each step; kind: 0 first pass, r repeat pass r, -1 post."""
    if name == 'none':
        return dict(name=name, seq=list(range(L)), kind=[0] * L, a=None, b=L - 1, k=0, beta=None)
    m = PAT.fullmatch(name)
    assert m, f'bad config name {name}'
    a, b, k = int(m[2]), int(m[3]), int(m[4])
    assert 0 <= a <= b < L and k >= 1, name
    seq = list(range(b + 1)) + list(range(a, b + 1)) * k + list(range(b + 1, L))
    kind = [0] * (b + 1) + [r for r in range(1, k + 1) for _ in range(a, b + 1)] + [-1] * (L - 1 - b)
    return dict(name=name, seq=seq, kind=kind, a=a, b=b, k=k, beta=BETA if m[1] == 'd' else None)


def make_targets(run, debug_n, path, allow_standin=False):
    if debug_n:
        qids = np.random.default_rng(0).choice(len(C.load_questions()), debug_n, replace=False)
        tg = dict(qids=sorted(int(q) for q in qids), rule=f'{debug_n} random questions, numpy default_rng(0)')
    else:
        src = os.path.join(run, 'classify', 'questions.csv')
        probe = json.load(open(os.path.join(run, 'classify', 'summary.json')))['config']['probe']
        assert probe == 'full' or allow_standin, f'classify was run with probe={probe}; use --allow_standin_targets'
        cls = {int(r['qid']): r['outcome_v1_K3'] for r in csv.DictReader(open(src))}
        rng = np.random.default_rng(0)
        pick = {}
        for c in 'URMO':
            qs = sorted(q for q, o in cls.items() if o == c)
            pick[c] = qs if c == 'U' or len(qs) <= 300 else sorted(int(q) for q in rng.choice(qs, 300, replace=False))
        tg = dict(qids=sorted(q for c in pick for q in pick[c]), cls=pick,
                  available={c: sum(o == c for o in cls.values()) for c in 'URMOX'},
                  rule='all U (outcome_v1_K3) + 300 random each of R/M/O (all if fewer), numpy default_rng(0)',
                  source=src, source_md5=C.md5_file(src), classify_probe=probe)
    C.save_json(tg, path)
    return tg


def batches(plen, max_tokens, max_bs):
    order, out, i = np.argsort(plen, kind='stable'), [], 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * plen[order[j]] <= max_tokens and j - i < max_bs:
            j += 1
        j = max(j, i + 1)
        out.append(order[i:j]); i = j
    return out


def csd4(lm, X, lay, target, chunk):
    """CSD letter logits (fp32) of states X (n, d) at origin blocks lay (n,); chunks are sorted by block."""
    out = torch.empty((X.shape[0], 4), dtype=torch.float32, device=X.device)
    order = torch.argsort(lay, stable=True)
    for i in range(0, len(order), chunk):
        idx = order[i:i + chunk]
        out[idx] = lm.letters32(lm.csd_states(X[idx], lay[idx], target))
    return out


class Out:
    """One config's output files (npy memmaps + raw bf16-bit memmap); done.npy (created last) marks resumable."""

    def __init__(self, d, cfg, rows, dm):
        self.d, self.cfg = d, cfg
        n, S = len(rows), len(cfg['seq'])
        os.makedirs(d, exist_ok=True)
        new = not os.path.exists(os.path.join(d, 'done.npy'))
        if new:
            np.save(os.path.join(d, 'rows.npy'), rows); np.save(os.path.join(d, 'seq.npy'), np.array(cfg['seq']))
            np.save(os.path.join(d, 'step_kind.npy'), np.array(cfg['kind']))
        assert np.array_equal(np.load(os.path.join(d, 'rows.npy')), rows), f'{d}: target rows changed'

        def arr(name, shape, dt):
            p = os.path.join(d, name)
            x = np.lib.format.open_memmap(p, 'w+', dt, shape) if new else np.lib.format.open_memmap(p, 'r+')
            assert x.shape == shape, (p, x.shape, shape)
            return x
        self.final4, self.top1 = arr('final4.npy', (n, 4), np.float32), arr('final_top1.npy', (n,), np.int32)
        self.csd, self.lens = arr('traj_csd4.npy', (n, S, 4), np.float32), arr('traj_lens4.npy', (n, S, 4), np.float32)
        self.hid = np.memmap(os.path.join(d, 'traj_hidden_bf16.u16'), np.uint16, 'w+' if new else 'r+', shape=(n, S, dm))
        self.done = arr('done.npy', (n,), bool)

    def write(self, idx, traj, csd, lens, top1):
        self.hid[idx] = traj.view(torch.int16).cpu().numpy().view(np.uint16)
        self.csd[idx], self.lens[idx] = csd.cpu().numpy(), lens.cpu().numpy()
        self.final4[idx], self.top1[idx] = self.lens[idx][:, -1], top1.cpu().numpy()
        for x in (self.hid, self.csd, self.lens, self.final4, self.top1):
            x.flush()
        self.done[idx] = True; self.done.flush()


def sanity(ex, rid, o):
    """Config none vs extraction (same prompts, different batch composition): argmax agreement and max |diff|."""
    out4 = np.load(os.path.join(ex, 'out_logits4.npy'))[rid]
    f4 = np.asarray(o.final4)
    srt = np.sort(out4, -1); margin = srt[:, -1] - srt[:, -2]
    dis = f4.argmax(-1) != out4.argmax(-1)
    r = dict(final_argmax_agree=float(1 - dis.mean()), final_maxdiff=float(np.abs(f4 - out4).max()),
             final_disagree_n=int(dis.sum()), final_disagree_extract_margin=np.round(margin[dis], 4).tolist()[:20],
             extract_ties=int((margin < 1e-4).sum()),
             top1_agree=float((np.asarray(o.top1) == np.load(os.path.join(ex, 'out_top1.npy'))[rid]).mean()))
    for nm, fn, x in (('csd', 'csd_raw4.npy', o.csd), ('lens', 'lens4.npy', o.lens)):
        ref, x = np.load(os.path.join(ex, fn))[rid], np.asarray(x)
        ag = (x.argmax(-1) == ref.argmax(-1)).mean(0)
        r.update({f'{nm}_argmax_agree_mean': float(ag.mean()), f'{nm}_argmax_agree_min_layer': float(ag.min()),
                  f'{nm}_maxdiff': float(np.abs(x - ref).max())})
    r['pass'] = r['final_argmax_agree'] >= 0.995
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
    ap.add_argument('--mode', required=True, choices=('all', 'last'))
    ap.add_argument('--configs', default=''); ap.add_argument('--debug_n', type=int, default=0)
    ap.add_argument('--max_tokens', type=int, default=8000); ap.add_argument('--max_bs', type=int, default=64)
    ap.add_argument('--csd_chunk', type=int, default=4096)
    ap.add_argument('--out', default='', help='output dir under RUN (default loop, or loop_debug with --debug_n)')
    ap.add_argument('--allow_standin_targets', action='store_true', help='build targets from a non-full-probe taxonomy')
    a = ap.parse_args()
    t0 = time.time()
    run = C.run_dir(a.model, a.fmt)
    od = os.path.join(run, a.out or ('loop_debug' if a.debug_n else 'loop')); os.makedirs(od, exist_ok=True)
    tpath = os.path.join(od, 'targets_debug.json' if a.debug_n else 'targets.json')
    tg = json.load(open(tpath)) if os.path.exists(tpath) else make_targets(run, a.debug_n, tpath, a.allow_standin_targets)
    assert not a.debug_n or len(tg['qids']) == a.debug_n, f'{tpath} holds {len(tg["qids"])} questions'
    lm = C.LM(a.model)
    L, dm, dev = lm.L, lm.d, lm.device
    qs = C.load_questions()
    rows = C.build_rows(a.model, a.fmt, lm.tok, [qs[q] for q in tg['qids']])
    rid = np.array([r['row'] for r in rows])
    target = C.target_text(a.fmt, lm.tok)
    ids_crit = lm.find_letter_ids(rows[0]['text'])
    assert ids_crit == lm.find_letter_ids(target), 'letter ids differ between prompt and CSD target'
    lm.set_letters(ids_crit)
    idl = [lm.encode(r['text']) for r in rows]
    plen = np.array([len(x) for x in idl])
    ex = os.path.join(run, 'extract')
    assert np.array_equal(np.load(os.path.join(ex, 'prompt_len.npy'))[rid], plen), 'prompts differ from extraction'
    cfgs = [parse_config(n, L) for n in (a.configs.split(',') if a.configs else [scale_name(c, L) for c in CONFIGS36])]
    outs = []
    for c in cfgs:
        d = os.path.join(od, f'{a.mode}_{c["name"]}')
        if os.path.exists(os.path.join(d, 'meta.json')):
            print('skip (complete):', d, flush=True)
        else:
            outs.append(Out(d, c, rid, dm))
    bl = batches(plen, a.max_tokens, a.max_bs)
    print(f'{a.mode}: {len(tg["qids"])} questions, {len(rid)} rows, {plen.sum()} tokens, {len(bl)} batches, '
          f'configs {[o.cfg["name"] for o in outs]}', C.stamp(t0), flush=True)
    tm = dict(fwd=0., loop=0., csd=0., write=0.)
    prefix_maxdiff, chk = 0., dict(from_block=L // 2, maxdiff=0., agree=0, n=0)
    n_rows = n_tok = 0
    torch.cuda.reset_peak_memory_stats()
    t_run = time.time()

    def tick(key, t):
        torch.cuda.synchronize(); tm[key] += time.time() - t
        return time.time()
    with torch.no_grad():
        for bi, b in enumerate(bl):
            live = [o for o in outs if not o.done[b].all()]
            if not live:
                continue
            t = time.time()
            ids, am, pos = lm.batch_tokens([idl[x] for x in b])
            B, T = ids.shape
            if a.mode == 'last':
                H, cache = lm.forward_capture(ids, am, pos, use_cache=True)
                kv = C.PrefixKV([cache.layers[i].keys[:, :, :T - 1] for i in range(L)],
                                [cache.layers[i].values[:, :, :T - 1] for i in range(L)])
                mask, lastpos = am.bool()[:, None, None, :], pos[:, -1:]
                j0 = chk['from_block']
                l1 = lm.letters32(lm.run_single(H[:, j0], list(range(j0 + 1, L)), kv, lastpos, mask=mask))
                l2 = lm.letters32(H[:, -1])
                chk['maxdiff'] = max(chk['maxdiff'], (l1 - l2).abs().max().item())
                chk['agree'] += int((l1.argmax(-1) == l2.argmax(-1)).sum()); chk['n'] += B
            else:
                H = lm.forward_sequence(ids, am, pos, list(range(L)))
            t = tick('fwd', t)
            new = []
            for o in live:
                c, b_ = o.cfg, o.cfg['b']
                if b_ == L - 1 and c['k'] == 0:
                    new.append(H[:, :0])
                elif a.mode == 'last':
                    damp = range(c['k'] * (b_ - c['a'] + 1)) if c['beta'] else None
                    new.append(lm.run_single(H[:, b_], c['seq'][b_ + 1:], kv, lastpos, mask=mask, beta=c['beta'],
                                             damp_steps=damp, record=True)[1])
                else:
                    damp = [s for s, kk in enumerate(c['kind']) if kk > 0] if c['beta'] else None
                    tr = lm.forward_sequence(ids, am, pos, c['seq'], beta=c['beta'], damp_steps=damp)
                    prefix_maxdiff = max(prefix_maxdiff, (tr[:, :b_ + 1] - H[:, :b_ + 1]).abs().max().item())
                    new.append(tr[:, b_ + 1:])
            t = tick('loop', t)
            X = [H.reshape(-1, dm)] + [nw.reshape(-1, dm) for nw in new]
            lay = [torch.arange(L, device=dev).repeat(B)] + \
                  [torch.tensor(o.cfg['seq'][o.cfg['b'] + 1:], device=dev, dtype=torch.long).repeat(B) for o in live]
            c4 = torch.split(csd4(lm, torch.cat(X), torch.cat(lay), target, a.csd_chunk), [len(x) for x in X])
            cH, lH = c4[0].reshape(B, L, 4), lm.letters32(H)
            res = []
            for o, nw, cn in zip(live, new, c4[1:]):
                b_ = o.cfg['b']
                traj = torch.cat([H[:, :b_ + 1], nw], 1)
                res.append((o, traj, torch.cat([cH[:, :b_ + 1], cn.reshape(B, -1, 4)], 1),
                            torch.cat([lH[:, :b_ + 1], lm.letters32(nw)], 1), lm.vocab_stats(traj[:, -1])[0]))
            t = tick('csd', t)
            for o, traj, cs, le, top1 in res:
                o.write(b, traj, cs, le, top1)
            del res, X, c4, new, H
            if a.mode == 'last':
                del cache, kv
            t = tick('write', t)
            n_rows += B; n_tok += int(plen[b].sum())
            if bi % 10 == 0 or bi == len(bl) - 1:
                print(f'batch {bi + 1}/{len(bl)} B={B} T={T} rows {n_rows} {n_rows / (time.time() - t_run):.2f} rows/s '
                      f'peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB',
                      {k: round(v, 1) for k, v in tm.items()}, C.stamp(t0), flush=True)
    sec = time.time() - t_run
    for o in outs:
        assert o.done.all(), o.d
        meta = dict(model=a.model, fmt=a.fmt, mode=a.mode, config=o.cfg, n=len(rid), n_questions=len(tg['qids']),
                    S=len(o.cfg['seq']), targets=tpath, letter_ids=ids_crit, target=target,
                    seconds=time.time() - t0, run_seconds=sec, rows_this_call=n_rows, tokens_this_call=n_tok,
                    timing=tm, configs_this_call=[x.cfg['name'] for x in outs],
                    peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                    script_md5=C.md5_file(__file__), common_md5=C.md5_file(C.__file__), args=vars(a))
        if a.mode == 'all':
            meta['prefix_maxdiff'] = prefix_maxdiff
        else:
            meta['selfcheck'] = dict(chk, agree=chk['agree'] / max(chk['n'], 1))
        if o.cfg['name'] == 'none':
            meta['sanity'] = sanity(ex, rid, o)
            print('SANITY', a.mode, json.dumps(meta['sanity']), flush=True)
        C.save_json(meta, os.path.join(o.d, 'meta.json'))
    print('timing', {k: round(v, 1) for k, v in tm.items()}, 'prefix_maxdiff' if a.mode == 'all' else 'selfcheck',
          prefix_maxdiff if a.mode == 'all' else chk, flush=True)
    print('DONE', C.stamp(t0), flush=True)


if __name__ == '__main__':
    main()
