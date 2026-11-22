"""L1 complement steering: a label-free, hot-pluggable intervention built on the ABNE quantity (RESULTS 2-7, stage2_abne).

Idea. Before the emission layer (L24 on Qwen3-4B) the answer is linearly available in the orthogonal complement of the
A-D readout subspace span(E), E = U4 * final-norm weight (lens = hn @ E.T, hn = h / rms(h)), but not yet emitted into it.
The L1 consensus head (scripts/consensus_probe.py; logits = hn @ (E + W).T + b per layer, trained on the model's own
4-version permutation consensus, 5-fold cross-fitted, gold never read) reads that content from a single pass.

Intervention (per row, block l). z = fold-specific head logits on the critical-position state h (weights of fold f were
fit without fold f, so a row is never steered by a head that saw its question); c* = argmax z. Raw direction
w = W[c*] - mean_k W[k] (hn space). The head sees h only through hn = h / rms(h); its residual-space gradient is the
hn-space direction with the radial component along h removed (the radial part only rescales h). So d = component of w
orthogonal to span(E) and to h (Gram-Schmidt on [E rows, h]), normalised; push h <- h + alpha * ||h|| * d (relative size
alpha, as stage2_abne causal) and run blocks l+1..L-1 at the critical position with the row's own prefix keys/values
(loop.py mode last via LM.run_single; the K configs of one row are packed along the batch axis and share the row's prefix
cache, RepPrefixKV). Controls at equal norm: rand_perp (seeded N(0, I) per row, same orthogonalisation), runner_perp (the
head's runner-up class c2), cstar_par (the head's full class contrast (E + W)[c*] - mean projected INTO span(E),
orthogonal to h). Reference (not label-free, never selected): gold:cstar_perp = the same with the gold-trained probe
(probe/weights_full.npz, also cross-fitted by fold).
Gates (label-free, applied per row; rows are independent at the critical position, so gating after the run equals gating at
inference): none; cons_conf (head max 4-way softmax >= tau); not_emitted (lens top-1 at l != head top-1, i.e. available
but not emitted); both.

GPU part (default) -> RUN/{out}/: final4.npy (n, nl, nc, 4) f32 final letter logits (fp32 readout); top1.npy (n, nl, nc)
full-vocab top-1 (compliance); lens_top.npy (n, nl, nc, L - l0) int8 lens top-1 of the stream at layers l0..L-1 (-1 = not
run; layer l = the pushed state); lensH_top.npy (n, L) first-pass lens top-1; head4.npy (n, nl, nh, 4) head logits on the
in-batch state; lens_l.npy (n, nl, 4); fwd4.npy (n, 4) same-batch normal-forward output letters; diag.npy (n, nl, nh, 4):
raw c* contrast's norm share outside span(E), |cos(raw c* contrast, h)|, runner share outside span(E), full contrast share
inside span(E); rows.npy, configs.json, done.npy (resumable), meta.json. Checks (meta['checks']): 'zero' (alpha = 0
through the push code path, last chunk) and padded replays of 'none' must equal the same-batch 'none' bit-for-bit
(final state bits, letters, lens per step); packed 'none' vs the unpacked loop.py path (C.PrefixKV) and vs the normal
forward (bf16 noise expected, as loop.py selfcheck).

Eval part (--eval, CPU; gold only scores). Classes: official v1 K=3 with the 'last layer = real output' contract, recomputed
with classify.py functions (RUN/classify/summary.json has no last_layer='output'), cross-checked with consensus_probe's
eval. Letters: steered letter = tie-aware argmax of final4 if the full-vocab top-1 is a letter, else -1 (wrong).
Reference = the same-batch alpha = 0 ('none') stream of the same layer. Population = all rows of non-X questions.
Selection (folds 0-3 only, L1 criterion only): candidates (head in maxcons/vote, layer, alpha, gate, tau) with direction
cstar_perp, plus 'abstain' (no steering); criterion = mean change in agreement of the row's output content with the
4-version consensus content (targets.npz 'vote' = modal content of the 4 versions' real outputs; label-free). Dev numbers
are cross-fitted (fold f in 0..3 uses the candidate chosen on the other three dev folds); the final candidate (chosen on
all of folds 0-3) is declared in lockbox_declared.json before fold 4 is scored (--open_lockbox). Controls are scored at
the selected (head, layer, alpha, gate, tau). Metrics: net single-version accuracy (bootstrap 95% CI by question, 2000),
by v1 class, fixes/breaks, R harm (R questions 4/4 under 'none' no longer 4/4), U->R (U questions not 4/4 under 'none'
that become 4/4), compliance, L1 criteria, steered share, lens emission layer shift; full grid descriptively.
Usage: CUDA_VISIBLE_DEVICES=1 python scripts/steer.py --model qwen3-4b-base --fmt base_fewshot [--layers 20,21,22]
         [--alphas 0.05,0.1,0.2,0.3] [--n_q 0]
       python scripts/steer.py --model qwen3-4b-base --fmt base_fewshot --eval [--open_lockbox]
"""
import argparse, csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

L1_HEADS, DIRS = ('maxcons', 'vote'), ('cstar_perp', 'runner_perp', 'cstar_par')
GATES, TAUS = ('none', 'cons_conf', 'not_emitted', 'both'), (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
LOCK, B_BOOT, RSEED = 4, 2000, 12345


def config_list(l1_heads, alphas, gold_ref):
    cfgs = [dict(name='none', head=None, dir='none', alpha=0.0)]
    for t in l1_heads:
        for dn in DIRS:
            cfgs += [dict(name=f'{t}:{dn}@{x:g}', head=t, dir=dn, alpha=x) for x in alphas]
    cfgs += [dict(name=f'rand_perp@{x:g}', head=None, dir='rand_perp', alpha=x) for x in alphas]
    if gold_ref:
        cfgs += [dict(name=f'gold:cstar_perp@{x:g}', head='gold', dir='cstar_perp', alpha=x) for x in alphas]
    cfgs.append(dict(name='zero', head=l1_heads[0], dir='cstar_perp', alpha=0.0))  # alpha = 0 through the push path
    return cfgs


def select_qids(Q, n_q):
    if not n_q or n_q >= Q:
        return np.arange(Q)
    return np.sort(np.random.default_rng(0).choice(Q, n_q, replace=False))


class RepPrefixKV:
    """Duck-typed cache like common.PrefixKV for packed streams: row b's prefix keys/values (B, H, P, D) are shared by
    `rep` consecutive streams (stream index b * rep + c); returns cat(prefix, own new key/value) per stream."""

    def __init__(self, ks, vs, rep):
        self.ks, self.vs, self.rep = ks, vs, rep

    def _cat(self, p, x):
        import torch
        B, H, P, D = p.shape
        out = torch.empty((B, self.rep, H, P + 1, D), dtype=x.dtype, device=x.device)
        out[:, :, :, :P] = p[:, None]
        out[:, :, :, P:] = x.view(B, self.rep, H, 1, D)
        return out.view(B * self.rep, H, P + 1, D)

    def update(self, k, v, layer_idx, *args, **kwargs):
        return self._cat(self.ks[layer_idx], k), self._cat(self.vs[layer_idx], v)


# ----------------------------------------------------------------------------- GPU run
def run(a):
    import torch
    from loop import batches
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
    rd = C.run_dir(a.model, a.fmt)
    ex, od = os.path.join(rd, 'extract'), os.path.join(rd, a.out)
    os.makedirs(od, exist_ok=True)
    layers = [int(x) for x in a.layers.split(',')]
    alphas = [float(x) for x in a.alphas.split(',')]
    heads = list(L1_HEADS) + (['gold'] if a.gold_ref else [])
    cfgs = config_list(L1_HEADS, alphas, a.gold_ref)
    nl, nc, nh, l0 = len(layers), len(cfgs), len(heads), min(layers)
    qs = C.load_questions()
    qids = select_qids(len(qs), a.n_q)
    run_cfg = dict(model=a.model, fmt=a.fmt, layers=layers, alphas=alphas, heads=heads, configs=[c['name'] for c in cfgs],
                   n_q=int(len(qids)), qids_md5=C.hashlib.md5(qids.astype(np.int64).tobytes()).hexdigest(), chunk=a.chunk,
                   max_tokens=a.max_tokens, max_bs=a.max_bs, rand_seed=RSEED)
    cpath = os.path.join(od, 'configs.json')
    if os.path.exists(cpath):
        old = json.load(open(cpath))
        assert old == run_cfg, f'{cpath} differs from this call (use another --out): {old} vs {run_cfg}'
    else:
        C.save_json(run_cfg, cpath)
    # model: fp32 vocab copy stays on the CPU (memory cap), bf16 weights on the GPU (as stage2_abne causal)
    lm = C.LM(a.model, device='cpu')
    lm.model.to('cuda'); lm.device = 'cuda'; lm.norm_w = lm.norm_w.cuda()
    dev = 'cuda'
    rows = C.build_rows(a.model, a.fmt, lm.tok, [qs[q] for q in qids])
    rid = np.array([r['row'] for r in rows])
    target = C.target_text(a.fmt, lm.tok)
    ids_crit = lm.find_letter_ids(rows[0]['text'])
    assert ids_crit == lm.find_letter_ids(target)
    lm.set_letters(ids_crit); lm.U4 = lm.U4.cuda(); lm.WU = None
    WH = lm.model.lm_head.weight  # bf16 (V, d), tied with the embeddings
    L, d = lm.L, lm.d
    eps = float(lm.norm_eps)
    lp = np.load(os.path.join(ex, 'lens_params.npz'))
    assert np.allclose(lp['U4'], lm.U4.cpu().numpy()) and np.allclose(lp['norm_w'], lm.norm_w.cpu().numpy())
    assert abs(float(lp['eps']) - eps) < 1e-12 and list(lp['letter_ids']) == ids_crit
    E = (lm.U4 * lm.norm_w[None]).double()                                    # (4, d): lens = hn @ E.T
    Q4 = torch.linalg.qr(E.T)[0]                                              # (d, 4) orthonormal basis of span(E)
    HW = {}
    for t in heads:
        p = os.path.join(rd, 'probe', 'weights_full.npz') if t == 'gold' else \
            os.path.join(rd, 'consensus_probe', f'weights_cons_{t}.npz')
        z = np.load(p)
        HW[t] = (torch.from_numpy(z['W'][:, layers]).to(dev).double(), torch.from_numpy(z['b'][:, layers]).to(dev).double())
    fold_row = np.array([r['fold'] for r in rows])
    idl = [lm.encode(r['text']) for r in rows]
    plen = np.array([len(x) for x in idl])
    assert np.array_equal(np.load(os.path.join(ex, 'prompt_len.npy'))[rid], plen), 'prompts differ from extraction'
    n, S = len(rid), L - l0
    new = not os.path.exists(os.path.join(od, 'done.npy'))
    if new:
        np.save(os.path.join(od, 'rows.npy'), rid)
    assert np.array_equal(np.load(os.path.join(od, 'rows.npy')), rid)

    def arr(name, shape, dt, fill=0):
        p = os.path.join(od, name)
        if new:
            x = np.lib.format.open_memmap(p, 'w+', dt, shape)
            x[:] = fill
            return x
        x = np.lib.format.open_memmap(p, 'r+')
        assert x.shape == shape and x.dtype == dt, (p, x.shape, shape)
        return x
    O = dict(final4=arr('final4.npy', (n, nl, nc, 4), np.float32), top1=arr('top1.npy', (n, nl, nc), np.int32),
             lens_top=arr('lens_top.npy', (n, nl, nc, S), np.int8, -1), lensH_top=arr('lensH_top.npy', (n, L), np.int8),
             head4=arr('head4.npy', (n, nl, nh, 4), np.float32), lens_l=arr('lens_l.npy', (n, nl, 4), np.float32),
             fwd4=arr('fwd4.npy', (n, 4), np.float32), diag=arr('diag.npy', (n, nl, nh, 4), np.float32))
    done = arr('done.npy', (n,), bool, False)
    R_ = a.chunk
    slots = list(range(nc)) + [0] * ((-nc) % R_)   # padded slots replay 'none' (extra bit-exactness checks)
    nch = len(slots) // R_
    i_none, i_zero = 0, nc - 1
    assert cfgs[i_zero]['name'] == 'zero' and i_zero // R_ == nch - 1 and nch > 1, 'zero must sit in another chunk than none'
    bl = batches(plen, a.max_tokens, a.max_bs)
    print(f'{a.model}__{a.fmt}: {len(qids)} questions, {n} rows, {len(bl)} batches, layers {layers}, alphas {alphas}, '
          f'{nc} configs in {nch} chunks of {R_} ({len(slots) - nc} padded), heads {heads}, done {int(done.sum())}',
          C.stamp(t0), flush=True)
    chk = dict(zero_vs_none_final_maxabs=0.0, zero_vs_none_state_bits_differ=0, zero_vs_none_lens_differ=0,
               zero_vs_none_top1_differ=0, pad_vs_none_state_bits_differ=0, n_pad_compared=0,
               none_vs_loop_path_maxabs=0.0, none_vs_loop_path_argmax_agree=0, none_vs_fwd_maxabs=0.0,
               none_vs_fwd_argmax_agree=0, n_row_layers=0)
    tm = dict(fwd=0., prep=0., stream=0., vocab=0., write=0.)

    def tick(k, t):
        torch.cuda.synchronize(); tm[k] += time.time() - t
        return time.time()

    def vocab_top1(h):
        hn = lm.fnorm32(h)
        best = torch.full((len(h),), -torch.inf, device=dev)
        arg = torch.zeros(len(h), dtype=torch.long, device=dev)
        for i in range(0, WH.shape[0], 8192):
            m, ix = (hn @ WH[i:i + 8192].float().T).max(-1)
            upd = m > best
            best, arg = torch.where(upd, m, best), torch.where(upd, ix + i, arg)
        return arg

    def process(b):
        """One batch of rows b. Returns (B, T, check deltas); writes the outputs and marks the rows done."""
        ck = {k: 0 for k in chk}
        t = time.time()
        ids, am, pos = lm.batch_tokens([idl[x] for x in b])
        B, T = ids.shape
        H, cache = lm.forward_capture(ids, am, pos, use_cache=True)
        ks = [cache.layers[i].keys[:, :, :T - 1] for i in range(L)]
        vs = [cache.layers[i].values[:, :, :T - 1] for i in range(L)]
        kv1, kvr = C.PrefixKV(ks, vs), RepPrefixKV(ks, vs, R_)
        mask1, pos1 = am.bool()[:, None, None, :], pos[:, -1:]
        maskr, posr = mask1.repeat_interleave(R_, 0), pos1.repeat_interleave(R_, 0)
        fw = lm.letters32(H[:, -1])
        t = tick('fwd', t)
        fb = torch.from_numpy(fold_row[b]).to(dev)
        ar = torch.arange(B, device=dev)
        rnd = torch.from_numpy(np.stack([np.random.default_rng([RSEED, int(r)]).standard_normal(d) for r in rid[b]])).to(dev)
        fin = torch.empty((B, nl, nc, d), dtype=torch.bfloat16, device=dev)
        ltop = torch.full((B, nl, nc, S), -1, dtype=torch.int8, device=dev)
        head4 = torch.zeros((B, nl, nh, 4), dtype=torch.float64, device=dev)
        lensl = torch.zeros((B, nl, 4), dtype=torch.float64, device=dev)
        diag = torch.zeros((B, nl, nh, 4), dtype=torch.float64, device=dev)
        for li, l in enumerate(layers):
            hb = H[:, l]
            hd = hb.double()
            nrm = hd.norm(dim=-1, keepdim=True)
            hn = hd / torch.sqrt(hd.pow(2).mean(-1, keepdim=True) + eps)
            lensl[:, li] = hn @ E.T
            g = (hd @ Q4) @ Q4.T                                              # part of h inside span(E)
            uh, ug = (hd - g) / (hd - g).norm(dim=-1, keepdim=True), g / g.norm(dim=-1, keepdim=True)

            def perp(w):                                                      # orthogonal to span(E) and to h
                w = w - (w @ Q4) @ Q4.T
                w = w - (w * uh).sum(-1, keepdim=True) * uh
                return w / w.norm(dim=-1, keepdim=True)

            def par(w):                                                       # inside span(E), orthogonal to h
                p = (w @ Q4) @ Q4.T
                p = p - (p * ug).sum(-1, keepdim=True) * ug
                return p / p.norm(dim=-1, keepdim=True)
            Dd = {(None, 'rand_perp'): perp(rnd)}
            for hi, tname in enumerate(heads):
                Wf, bf = HW[tname][0][fb, li], HW[tname][1][fb, li]            # (B, 4, d), (B, 4) fold-specific
                z = lensl[:, li] + torch.einsum('bd,bkd->bk', hn, Wf) + bf
                o = torch.argsort(z, -1, descending=True)
                cs, c2 = o[:, 0], o[:, 1]
                Wc = Wf - Wf.mean(1, keepdim=True)
                Pc = (E[None] + Wf) - (E[None] + Wf).mean(1, keepdim=True)
                w1, w2, w3 = Wc[ar, cs], Wc[ar, c2], Pc[ar, cs]
                Dd[(tname, 'cstar_perp')], Dd[(tname, 'runner_perp')], Dd[(tname, 'cstar_par')] = perp(w1), perp(w2), par(w3)
                head4[:, li, hi] = z
                sh = lambda w: (w - (w @ Q4) @ Q4.T).norm(dim=-1) / w.norm(dim=-1)
                diag[:, li, hi] = torch.stack([sh(w1), (w1 * hd).sum(-1).abs() / (w1.norm(dim=-1) * nrm[:, 0]), sh(w2),
                                               ((w3 @ Q4) @ Q4.T).norm(dim=-1) / w3.norm(dim=-1)], -1)
            X = torch.stack([hb if c['dir'] == 'none' else
                             (hd + c['alpha'] * nrm * Dd[(c['head'] if c['dir'] != 'rand_perp' else None, c['dir'])]
                              ).to(torch.bfloat16) for c in cfgs], 1)       # (B, nc, d)
            Xp = X[:, slots]
            t = tick('prep', t)
            blocks = list(range(l + 1, L))
            for ch in range(nch):
                x = Xp[:, ch * R_:(ch + 1) * R_].reshape(B * R_, d)
                f, tr = lm.run_single(x, blocks, kvr, posr, mask=maskr, record=True)
                f, tr = f.view(B, R_, d), tr.view(B, R_, L - 1 - l, d)
                lt = lm.letters32(torch.cat([x.view(B, R_, 1, d), tr], 2)).argmax(-1).to(torch.int8)  # layers l..L-1
                for j in range(R_):
                    p_ = ch * R_ + j
                    if p_ < nc:
                        fin[:, li, p_], ltop[:, li, p_, l - l0:] = f[:, j], lt[:, j]
                    else:                                                     # padded replay of 'none'
                        ck['pad_vs_none_state_bits_differ'] += int((f[:, j].view(torch.int16) != fin[:, li, i_none].view(torch.int16)).any(-1).sum())
                        ck['n_pad_compared'] += B
                del f, tr, lt
            # unpacked loop.py path (C.PrefixKV, B streams) for the unmodified state: bf16-noise reference
            f1 = lm.run_single(hb, blocks, kv1, pos1, mask=mask1)
            l_one, l_none = lm.letters32(f1), lm.letters32(fin[:, li, i_none])
            ck['none_vs_loop_path_maxabs'] = max(ck['none_vs_loop_path_maxabs'], float((l_one - l_none).abs().max()))
            ck['none_vs_loop_path_argmax_agree'] += int((l_one.argmax(-1) == l_none.argmax(-1)).sum())
            ck['none_vs_fwd_maxabs'] = max(ck['none_vs_fwd_maxabs'], float((fw - l_none).abs().max()))
            ck['none_vs_fwd_argmax_agree'] += int((fw.argmax(-1) == l_none.argmax(-1)).sum())
            ck['zero_vs_none_state_bits_differ'] += int((fin[:, li, i_zero].view(torch.int16) != fin[:, li, i_none].view(torch.int16)).any(-1).sum())
            ck['zero_vs_none_lens_differ'] += int((ltop[:, li, i_zero] != ltop[:, li, i_none]).any(-1).sum())
            ck['n_row_layers'] += B
            del X, Xp, Dd
            t = tick('stream', t)
        f4 = lm.letters32(fin)                                                # (B, nl, nc, 4) fp32
        top = vocab_top1(fin.reshape(-1, d)).view(B, nl, nc)
        ck['zero_vs_none_final_maxabs'] = float((f4[:, :, i_zero] - f4[:, :, i_none]).abs().max())
        ck['zero_vs_none_top1_differ'] += int((top[:, :, i_zero] != top[:, :, i_none]).sum())
        t = tick('vocab', t)
        O['final4'][b], O['top1'][b], O['lens_top'][b] = f4.cpu().numpy(), top.cpu().numpy().astype(np.int32), ltop.cpu().numpy()
        O['lensH_top'][b] = lm.letters32(H).argmax(-1).to(torch.int8).cpu().numpy()
        O['head4'][b], O['lens_l'][b] = head4.float().cpu().numpy(), lensl.float().cpu().numpy()
        O['fwd4'][b], O['diag'][b] = fw.cpu().numpy(), diag.float().cpu().numpy()
        for x in O.values():
            x.flush()
        done[b] = True; done.flush()
        t = tick('write', t)
        return B, T, ck

    def safe(b, depth=0):
        """process(b); on CUDA OOM (shared GPU / our cap) free the cache and split the batch (each part keeps its own
        same-batch 'none' reference, so splitting changes nothing in the comparisons)."""
        for attempt in range(20):
            try:
                return process(b)
            except torch.OutOfMemoryError:
                pass
            torch.cuda.empty_cache()
            print(f'  OOM on batch of {len(b)} rows (depth {depth}, attempt {attempt})', flush=True)
            if len(b) > 1:
                h = len(b) // 2
                r1, r2 = safe(b[:h], depth + 1), safe(b[h:], depth + 1)
                return r1[0] + r2[0], max(r1[1], r2[1]), {k: (max(r1[2][k], r2[2][k]) if k.endswith('maxabs') else r1[2][k] + r2[2][k])
                                                          for k in chk}
            time.sleep(30)
        raise RuntimeError('persistent OOM')

    torch.cuda.reset_peak_memory_stats()
    t_run, n_done = time.time(), 0
    with torch.no_grad():
        for bi, b in enumerate(bl):
            if done[b].all():
                continue
            B, T, ck = safe(b)
            for k, x in ck.items():
                chk[k] = max(chk[k], x) if k.endswith('maxabs') else chk[k] + x
            n_done += B
            if bi % 20 == 0 or bi == len(bl) - 1:
                el = time.time() - t_run
                print(f'batch {bi + 1}/{len(bl)} B={B} T={T} rows {n_done} {n_done / el:.2f} rows/s peak '
                      f'{torch.cuda.max_memory_allocated() / 2 ** 30:.2f}GB', {k: round(x, 1) for k, x in tm.items()},
                      {k: chk[k] for k in ('zero_vs_none_state_bits_differ', 'pad_vs_none_state_bits_differ',
                                           'none_vs_loop_path_argmax_agree', 'n_row_layers')}, C.stamp(t0), flush=True)
    assert done.all()
    out4 = np.load(os.path.join(ex, 'out_logits4.npy'))[rid]
    fwd = np.asarray(O['fwd4'])
    nr = max(chk['n_row_layers'], 1)
    checks = dict(chk, none_vs_loop_path_argmax_agree=chk['none_vs_loop_path_argmax_agree'] / nr,
                  none_vs_fwd_argmax_agree=chk['none_vs_fwd_argmax_agree'] / nr,
                  fwd_vs_extract_argmax_agree=float((fwd.argmax(-1) == out4.argmax(-1)).mean()),
                  none_vs_extract_argmax_agree=[float((np.asarray(O['final4'])[:, li, i_none].argmax(-1) == out4.argmax(-1)).mean())
                                                for li in range(nl)],
                  note='this call only (resumed runs: counts cover the batches computed in this call)')
    checks['alpha0_bit_exact'] = (checks['zero_vs_none_state_bits_differ'] == 0 and checks['zero_vs_none_lens_differ'] == 0
                                  and checks['zero_vs_none_top1_differ'] == 0 and checks['zero_vs_none_final_maxabs'] == 0.0
                                  and checks['pad_vs_none_state_bits_differ'] == 0)
    meta = dict(run_cfg, configs_full=cfgs, n=n, L=L, d=d, l0=l0, letter_ids=ids_crit, target=target, checks=checks,
                seconds=round(time.time() - t0, 1), run_seconds=round(time.time() - t_run, 1), rows_this_call=n_done,
                timing=tm, peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30, script_md5=C.md5_file(__file__),
                common_md5=C.md5_file(C.__file__), args=vars(a))
    C.save_json(meta, os.path.join(od, 'meta.json'))
    print('CHECKS', json.dumps(checks), flush=True)
    print('DONE', C.stamp(t0), flush=True)


# ----------------------------------------------------------------------------- evaluation (CPU)
def evaluate(a):
    import classify as CL
    t0 = time.time()
    rd = C.run_dir(a.model, a.fmt)
    od = os.path.join(rd, a.out)
    meta = json.load(open(os.path.join(od, 'meta.json')))
    cfgs, layers, heads = meta['configs_full'], meta['layers'], meta['heads']
    cix = {c['name']: i for i, c in enumerate(cfgs)}
    alphas, L, l0 = meta['alphas'], meta['L'], meta['l0']
    gates = GATES if a.gates == 'all' else tuple(a.gates.split(','))
    rid = np.load(os.path.join(od, 'rows.npy'))
    n = len(rid)
    assert (rid.reshape(-1, 4) == rid[0::4, None] + np.arange(4)).all() and (rid[0::4] % 4 == 0).all()
    qid = rid[0::4] // 4
    nq = len(qid)
    # official v1 K=3 classes, last layer = real output (contract since 2026-10-16 evening), recomputed in-process
    D = CL.load_inputs(a.model, a.fmt, 'full', 'raw')
    cconf = json.load(open(os.path.join(rd, 'classify', 'summary.json')))['config']
    p_let, f_let = CL.top1(D['probe']), CL.top1(D['out4'])
    p_let[:, D['L'] - 1] = f_let
    excl_all = ~CL.to_q(D['compliant']).all(1)
    answer_all = D['gold'][0::4]
    off = CL.classify_v1(CL.contents(CL.to_q(p_let)), CL.contents(CL.to_q(CL.top1(D['csd']))), CL.contents(CL.to_q(f_let)),
                         answer_all, 3, excl_all)
    cls_all = off['outcome']
    ref_counts = None
    cp = os.path.join(rd, 'consensus_probe', 'eval_final.json')
    if os.path.exists(cp):
        ref_counts = json.load(open(cp))['config']['official_taxonomy']['classes']
        assert {k: int((cls_all == k).sum()) for k in 'ROMUX'} == ref_counts, 'class recomputation differs'
    cls, ua, excl = cls_all[qid], off['u_avail'][qid], excl_all[qid]
    gold = D['gold'][rid]
    v = rid % 4
    fold_q = D['fold'][rid][0::4]
    fold_r = np.repeat(fold_q, 4)
    keep = ~excl
    tz = np.load(os.path.join(rd, 'consensus_probe', 'targets.npz'))
    tq4 = {t: np.repeat(tz[t].astype(int)[qid], 4) for t in ('vote', 'maxcons')}
    # letters (n, nl, nc): tie-aware argmax of final4 if the full-vocab top-1 is a letter, else -1 (wrong)
    f4 = np.load(os.path.join(od, 'final4.npy'))
    comp = np.isin(np.load(os.path.join(od, 'top1.npy')), meta['letter_ids'])
    let = np.where(comp, CL.top1(f4), -1)
    lens_top = np.load(os.path.join(od, 'lens_top.npy'))
    lensH = np.load(os.path.join(od, 'lensH_top.npy'))
    head4 = np.load(os.path.join(od, 'head4.npy')).astype(np.float64)
    lens_l = np.load(os.path.join(od, 'lens_l.npy'))
    diag = np.load(os.path.join(od, 'diag.npy'))
    hp, hk = CL.softmax(head4).max(-1), head4.argmax(-1)                     # (n, nl, nh)
    notem = lens_l.argmax(-1)[:, :, None] != hk
    hidx = {t: i for i, t in enumerate(heads)}
    i0 = cix['none']
    content = lambda x: np.where(x >= 0, (x + v) % 4, -1)
    agree = lambda x, t: content(x) == tq4[t]
    to_q = CL.to_q

    def gate_mask(li, t, gate, tau):
        hi = hidx[t]
        if gate == 'none':
            return np.ones(n, bool)
        if gate == 'cons_conf':
            return hp[:, li, hi] >= tau
        if gate == 'not_emitted':
            return notem[:, li, hi]
        return (hp[:, li, hi] >= tau) & notem[:, li, hi]

    def emission(li, c, x):
        """Lens emission layer of letter x (n,): first layer from which the lens top-1 equals x through L-1 (L = never);
        first-pass lens below the push layer, the stream from the (pushed) state at the push layer on."""
        l = layers[li]
        hit = np.concatenate([lensH[:, :l], lens_top[:, li, c, l - l0:]], 1) == x[:, None]
        return L - np.cumprod(hit[:, ::-1], 1).sum(1)

    def policy(t, li, dname, alpha, gate, tau):
        """Gated config (head t defines the gate): after letters, steered mask, compliance, emission pair, reference."""
        c = cix[f'rand_perp@{alpha:g}' if dname == 'rand_perp' else f'{t}:{dname}@{alpha:g}']
        g = gate_mask(li, t, gate, tau)
        ref = let[:, li, i0]
        eb = emission(li, i0, ref)
        return dict(after=np.where(g, let[:, li, c], ref), g=g, comp=np.where(g, comp[:, li, c], comp[:, li, i0]),
                    emis=(np.where(g, emission(li, c, let[:, li, c]), eb), eb), ref=ref)

    splits = {'f0-3': keep & (fold_q != LOCK)}
    if a.open_lockbox:
        splits['f4'] = keep & (fold_q == LOCK)

    def boot(qm):
        W = np.zeros((B_BOOT, nq))
        W[:, np.flatnonzero(qm)] = CL.boot_weights(int(qm.sum()), B_BOOT)
        return W
    Wb = {s: boot(m) for s, m in list(splits.items()) + ([('all', keep)] if a.open_lockbox else [])}

    def metrics(P, qm, W, full=True):
        after, before = P['after'], P['ref']
        ok_a, ok_b = after == gold, before == gold
        na, nb = to_q(ok_a).sum(1), to_q(ok_b).sum(1)
        rm = np.repeat(qm, 4)
        est = (lambda num, den: CL.ci(W, num, den)) if full else \
            (lambda num, den: round(float(np.sum(num) / max(np.sum(den), 1e-12)), 5))
        out = dict(n_q=int(qm.sum()), acc_before=round(float(ok_b[rm].mean()), 5), acc_after=round(float(ok_a[rm].mean()), 5),
                   net=CL.ci(W, (na - nb) * qm, 4 * qm), fixes=int((ok_a & ~ok_b & rm).sum()),
                   breaks=int((~ok_a & ok_b & rm).sum()), changed=int(((after != before) & rm).sum()))
        bc = {}
        for k, m in (('R', cls == 'R'), ('O', cls == 'O'), ('M', cls == 'M'), ('U', cls == 'U'),
                     ('U_avail', (cls == 'U') & ua), ('U_none', (cls == 'U') & ~ua)):
            mm = m & qm
            mr = np.repeat(mm, 4)
            bc[k] = dict(n_q=int(mm.sum()), net=est((na - nb) * mm, 4 * mm) if mm.any() else None,
                         fixes=int((ok_a & ~ok_b & mr).sum()), breaks=int((~ok_a & ok_b & mr).sum()))
        out['by_class'] = bc
        isR4, isU = (cls == 'R') & qm & (nb == 4), (cls == 'U') & qm & (nb < 4)
        out['R_harm'] = [int((isR4 & (na < 4)).sum()), int(isR4.sum())]
        out['R_harm_rate'] = est(isR4 & (na < 4), isR4)
        out['U_to_R'] = [int((isU & (na == 4)).sum()), int(isU.sum())]
        out['U_to_R_rate'] = est(isU & (na == 4), isU)
        mU = (cls == 'U') & qm
        out['U_4of4_net'] = int(((na == 4) & mU).sum() - ((nb == 4) & mU).sum())
        out['all4_correct_net'] = est(((na == 4).astype(float) - (nb == 4)) * qm, qm)
        self4 = lambda x: (lambda c: (c == c[:, :1]).all(1) & (c[:, 0] >= 0))(to_q(content(x)))
        out['L1'] = {f'd_agree_{t}': est((to_q(agree(after, t)).sum(1) - to_q(agree(before, t)).sum(1)) * qm, 4 * qm)
                     for t in ('vote', 'maxcons')}
        out['L1']['d_self4'] = est((self4(after).astype(float) - self4(before)) * qm, qm)
        out['steered_share'] = round(float(P['g'][rm].mean()), 4)
        out['compliance_after'] = round(float(P['comp'][rm].mean()), 5)
        out['compliance_before'] = round(float(comp[:, 0, i0][rm].mean()), 5)
        ea, eb = P['emis']
        ok = rm & (ea < L) & (eb < L)
        out['lens_emission_shift'] = round(float((ea - eb)[ok].mean()), 3) if ok.any() else None
        ok = ok & P['g'] & (after == before)
        out['lens_emission_shift_steered_unchanged_answer'] = round(float((ea - eb)[ok].mean()), 3) if ok.any() else None
        return out

    # ---- selection on folds 0-3 by the L1 criterion: candidates (head, layer, alpha, gate, tau), direction cstar_perp
    def cand_list(hs):
        out = [('abstain', None, None, None, None)]
        for t in hs:
            for li in range(len(layers)):
                for x in alphas:
                    for gname in gates:
                        out += [(t, li, x, gname, tau) for tau in (TAUS if gname in ('cons_conf', 'both') else (None,))]
        return out
    fmt_c = lambda cd: 'abstain' if cd[0] == 'abstain' else \
        f'{cd[0]} L{layers[cd[1]]} a={cd[2]:g} gate={cd[3]}' + (f'@{cd[4]:g}' if cd[4] is not None else '')
    dev_rows = np.repeat(keep & (fold_q != LOCK), 4)
    crit = a.criterion

    def select(cands):
        S = np.zeros((len(cands), n), np.float32)
        for i, cd in enumerate(cands):
            if cd[0] != 'abstain':
                t, li, x, gname, tau = cd
                c = cix[f'{t}:cstar_perp@{x:g}']
                S[i] = gate_mask(li, t, gname, tau) * (agree(let[:, li, c], crit).astype(np.float32) - agree(let[:, li, i0], crit))
        pick = lambda m: int(S[:, m].mean(1).argmax())  # ties -> first candidate ('abstain' first)
        cf = {f: pick(dev_rows & (fold_r != f)) for f in range(4)}
        return S, cf, pick(dev_rows)

    def as_policy(cd):
        if cd[0] == 'abstain':
            ref = let[:, 0, i0]
            return dict(after=ref, g=np.zeros(n, bool), comp=comp[:, 0, i0], emis=(np.zeros(n), np.zeros(n)), ref=ref)
        t, li, x, gname, tau = cd
        return policy(t, li, 'cstar_perp', x, gname, tau)

    def crossfit_policy(cands, cf):
        P = {k: (x.copy() if not isinstance(x, tuple) else tuple(y.copy() for y in x)) for k, x in as_policy(cands[0]).items()}
        P['emis'] = (np.zeros(n), np.zeros(n))
        for f in range(4):
            Q_ = as_policy(cands[cf[f]])
            m = fold_r == f
            for k in ('after', 'g', 'comp', 'ref'):
                P[k][m] = Q_[k][m]
            P['emis'][0][m], P['emis'][1][m] = Q_['emis'][0][m], Q_['emis'][1][m]
        return P

    R = dict(config=dict(model=a.model, fmt=a.fmt, n_rows=n, n_q=nq, subset=meta['n_q'] < 5659, layers=layers, alphas=alphas,
                         heads=heads, criterion=f'agree_{crit} (row output content vs 4-version consensus content of the '
                         'real outputs, targets.npz; label-free)', gates=gates, taus=TAUS, lockbox_fold=LOCK,
                         lockbox_open=a.open_lockbox, classes={k: int((cls == k).sum()) for k in 'ROMUX'},
                         classes_ref_consensus_probe=ref_counts,
                         classify_summary_last_layer=cconf.get('last_layer', 'absent -> recomputed with output'),
                         n_boot=B_BOOT, run_checks=meta['checks'], run_seconds=meta['run_seconds'],
                         none_layer_agreement={f'L{layers[i]}_vs_L{layers[j]}': round(float((let[:, i, i0] == let[:, j, i0]).mean()), 5)
                                               for i in range(len(layers)) for j in range(i + 1, len(layers))}))
    sel_sets = {'L1': list(L1_HEADS)} | ({'gold_ref': ['gold']} if 'gold' in heads else {})
    R['selection'] = {}
    picks = {}
    for nm, hs in sel_sets.items():
        cands = cand_list(hs)
        S, cf, final = select(cands)
        sc = S[:, dev_rows].mean(1)
        order = np.argsort(-sc, kind='stable')
        picks[nm] = (cands, cf, final)
        P = crossfit_policy(cands, cf)
        R['selection'][nm] = dict(level='L1' if nm == 'L1' else 'L3 reference (gold-trained probe; same L1 selection rule)',
                                  n_candidates=len(cands), crossfit={f: fmt_c(cands[cf[f]]) for f in range(4)},
                                  final=fmt_c(cands[final]), final_dev_L1=round(float(sc[final]), 6),
                                  top15_dev_L1=[(fmt_c(cands[i]), round(float(sc[i]), 6)) for i in order[:15]],
                                  dev_crossfit=metrics(P, splits['f0-3'], Wb['f0-3']))
        print(f'[{nm}] top dev L1 ({crit}): ' + '; '.join(f'{fmt_c(cands[i])} {sc[i] * 100:+.3f}pp' for i in order[:6]), flush=True)
        print(f'[{nm}] cross-fit picks {[fmt_c(cands[cf[f]]) for f in range(4)]} -> final {fmt_c(cands[final])}; '
              f'dev cross-fitted net {R["selection"][nm]["dev_crossfit"]["net"]}', flush=True)
    # ---- final L1 selection, scored on dev (in-sample for the selection, gold-clean) and lockbox, with controls
    cands, cf, final = picks['L1']
    cd = cands[final]
    sel = {}
    if cd[0] != 'abstain':
        t, li, x, gname, tau = cd
        ctrl = [('method L1 cstar_perp', t, 'cstar_perp', gname, tau), ('ctrl rand_perp', t, 'rand_perp', gname, tau),
                ('ctrl runner_perp', t, 'runner_perp', gname, tau), ('ctrl cstar_par (readout subspace)', t, 'cstar_par', gname, tau),
                ('method, gate none (descriptive)', t, 'cstar_perp', 'none', None)]
        if 'gold' in heads:
            ctrl.append(('ref gold:cstar_perp, same layer/alpha/gate (L3)', 'gold', 'cstar_perp', gname, tau))
        Pm = None
        for nm, tt, dn, gg, ta in ctrl:
            P = policy(tt, li, dn, x, gg, ta)
            sel[nm] = dict(cfg=f'L{layers[li]} {tt}:{dn}@{x:g} gate={gg}' + (f'@{ta:g}' if ta is not None else ''),
                           **{s: metrics(P, m, Wb[s]) for s, m in splits.items()})
            if Pm is None:
                Pm = P
            else:
                for s, m in splits.items():   # paired: method - control, net single-version accuracy
                    oka, okc = to_q(Pm['after'] == gold).sum(1), to_q(P['after'] == gold).sum(1)
                    sel[nm][s]['method_minus_this'] = CL.ci(Wb[s], (oka - okc) * m, 4 * m)
    if 'gold_ref' in picks:
        gc, gcf, gfin = picks['gold_ref']
        if gc[gfin][0] != 'abstain':
            P = as_policy(gc[gfin])
            sel['ref gold:cstar_perp, own L1 selection (L3)'] = dict(cfg=fmt_c(gc[gfin]), **{s: metrics(P, m, Wb[s]) for s, m in splits.items()})
    R['selected'] = sel
    # ---- full grid (descriptive; point estimates except net)
    grid = {}
    gsplits = dict(splits) | ({'all': keep} if a.open_lockbox else {})
    for li, l in enumerate(layers):
        for c in cfgs:
            if c['dir'] == 'none' or c['name'] == 'zero':
                continue
            for gname, tau in (('none', None), ('cons_conf', 0.5), ('cons_conf', 0.7), ('cons_conf', 0.9), ('not_emitted', None),
                               ('both', 0.7)):
                t = c['head'] if c['head'] is not None else L1_HEADS[0]
                P = policy(t, li, c['dir'], c['alpha'], gname, tau)
                key = f'L{l}|{c["name"]}|{gname}' + (f'@{tau:g}' if tau is not None else '')
                grid[key] = {}
                for s, m in gsplits.items():
                    mt = metrics(P, m, Wb[s], full=False)
                    grid[key][s] = dict(net=mt['net'], d_agree_vote=mt['L1']['d_agree_vote'], d_agree_maxcons=mt['L1']['d_agree_maxcons'],
                                        d_self4=mt['L1']['d_self4'], R_harm=mt['R_harm'], U_to_R=mt['U_to_R'], fixes=mt['fixes'],
                                        breaks=mt['breaks'], **{f'net_{k}': mt['by_class'][k]['net'] for k in ('R', 'O', 'M', 'U')},
                                        compliance=mt['compliance_after'], steered=mt['steered_share'], emis_shift=mt['lens_emission_shift'])
    R['grid'] = grid
    R['diag'] = {f'L{l}|{t}': dict(cstar_share_outside_spanE=round(float(diag[:, li, hi, 0].mean()), 4),
                                   cstar_abs_cos_h=round(float(diag[:, li, hi, 1].mean()), 4),
                                   runner_share_outside_spanE=round(float(diag[:, li, hi, 2].mean()), 4),
                                   full_contrast_share_inside_spanE=round(float(diag[:, li, hi, 3].mean()), 4),
                                   head_acc_gold=round(float((hk[:, li, hi] == gold)[np.repeat(keep, 4)].mean()), 4),
                                   head_agree_vote=round(float((content(hk[:, li, hi]) == tq4['vote'])[np.repeat(keep, 4)].mean()), 4),
                                   none_acc=round(float((let[:, li, i0] == gold)[np.repeat(keep, 4)].mean()), 4),
                                   none_agree_vote=round(float(agree(let[:, li, i0], 'vote')[np.repeat(keep, 4)].mean()), 4),
                                   gate_pass={f'cons_conf@{tau:g}': round(float((hp[:, li, hi] >= tau).mean()), 4) for tau in TAUS} |
                                   {'not_emitted': round(float(notem[:, li, hi].mean()), 4)})
                 for li, l in enumerate(layers) for t, hi in hidx.items()}
    R['config'].update(seconds=round(time.time() - t0, 1), script_md5=C.md5_file(__file__))
    name = 'eval_final.json' if a.open_lockbox else 'eval_dev.json'
    lp = os.path.join(od, 'lockbox_declared.json')
    if a.open_lockbox:
        dec = json.load(open(lp))
        assert dec['final'] == fmt_c(cd), f'declared {dec["final"]} != recomputed {fmt_c(cd)}'
    C.save_json(R, os.path.join(od, name))
    if not a.open_lockbox:
        dec = dict(declared_at=time.ctime(), source=f'{name} (folds 0-3 only; fold 4 not scored before this file)',
                   eval_dev_md5=C.md5_file(os.path.join(od, name)), criterion=R['config']['criterion'], final=fmt_c(cd),
                   crossfit=R['selection']['L1']['crossfit'], dev_crossfit_net=R['selection']['L1']['dev_crossfit']['net'],
                   dev_selected_net=sel.get('method L1 cstar_perp', {}).get('f0-3', {}).get('net'),
                   gold_ref_final=R['selection'].get('gold_ref', {}).get('final'))
        if os.path.exists(lp):
            assert json.load(open(lp))['final'] == dec['final'], 'declared selection changed'
        else:
            C.save_json(dec, lp)
    # ---- console summary
    for nm, x in sel.items():
        for s in splits:
            m = x[s]
            print(f'  {nm:50s} {s:4s} net {m["net"]} R_harm {m["R_harm"]} U->R {m["U_to_R"]} fix/brk {m["fixes"]}/{m["breaks"]} '
                  f'dvote {m["L1"]["d_agree_vote"][0]:+.4f} steered {m["steered_share"]} comp {m["compliance_after"]} '
                  f'emis {m["lens_emission_shift"]} | ' +
                  ' '.join(f'{k}:{v["net"][0]:+.4f}' for k, v in m['by_class'].items() if v['net']) +
                  (f' | method-this {m["method_minus_this"]}' if 'method_minus_this' in m else ''), flush=True)
    print('wrote', os.path.join(od, name), C.stamp(t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
    ap.add_argument('--layers', default='20,21,22'); ap.add_argument('--alphas', default='0.05,0.1,0.2,0.3')
    ap.add_argument('--gate', dest='gates', default='all', help="gates entering the selection grid (comma list of "
                    "none,cons_conf,not_emitted,both or 'all'); applied post hoc, rows are independent")
    ap.add_argument('--n_q', type=int, default=0, help='0 = all questions, else a fixed random subset (numpy default_rng(0))')
    ap.add_argument('--no_gold_ref', dest='gold_ref', action='store_false', help='skip the gold-probe reference configs')
    ap.add_argument('--chunk', type=int, default=17, help='configs packed per run_single call')
    ap.add_argument('--max_tokens', type=int, default=5000); ap.add_argument('--max_bs', type=int, default=48)
    ap.add_argument('--gb', type=float, default=11.5, help='GPU memory cap (allocator)')
    ap.add_argument('--out', default='steer')
    ap.add_argument('--eval', action='store_true'); ap.add_argument('--open_lockbox', action='store_true')
    ap.add_argument('--criterion', default='vote', choices=['vote', 'maxcons'])
    a = ap.parse_args()
    evaluate(a) if a.eval else run(a)


if __name__ == '__main__':
    main()
