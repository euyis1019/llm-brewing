"""Adversarial review of steer.py (GPU): independent re-computation of the selected L1 complement push through a different
code path. A forward hook on decoder block l edits the critical (last) position of a normal full forward (HF model, no
PrefixKV / run_single / RepPrefixKV); the head, c*, direction and random control are recomputed here from the fold weights
(projection onto span(E) by a normal-equation projector, not QR). Each row appears 3x in one batch: copy 0 = no edit
(the reference is the model's real output in that batch), copy 1 = method push, copy 2 = random push (own seed). Readout:
fp32 final norm of the pre-norm last block output (captured by hook), letters + full-vocab top-1, as steer.py.
Gold only scores. Outputs RUN/steer_review/hook_{tag}.npz + .json (small).
Usage: CUDA_VISIBLE_DEVICES=0 python scripts/explore/review_steer_hook.py --model qwen3-4b --fmt chat0 --layer 22 \
         --head maxcons --alpha 0.1 --tau 0.3 [--folds 4]"""
import argparse, csv, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
ap.add_argument('--layer', type=int, required=True); ap.add_argument('--head', required=True)
ap.add_argument('--alpha', type=float, required=True); ap.add_argument('--tau', type=float, required=True)
ap.add_argument('--folds', default='4'); ap.add_argument('--max_tokens', type=int, default=3000)
ap.add_argument('--max_bs', type=int, default=24); ap.add_argument('--gb', type=float, default=11.5)
ap.add_argument('--tag', default='')
a = ap.parse_args()
t0 = time.time()
torch.backends.cuda.matmul.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(min(1.0, a.gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
from transformers import AutoModelForCausalLM, AutoTokenizer
path = C.MODELS[a.model]
tok = AutoTokenizer.from_pretrained(path)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
rd = C.run_dir(a.model, a.fmt)
qs = C.load_questions()
folds = [int(x) for x in a.folds.split(',')]
qids = [q['qid'] for q in qs if q['fold'] in folds]
rows = C.build_rows(a.model, a.fmt, tok, [qs[q] for q in qids])
rid = np.array([r['row'] for r in rows])
ids = [tok(r['text'], add_special_tokens=True)['input_ids'] for r in rows]
plen = np.array([len(x) for x in ids])
assert np.array_equal(np.load(os.path.join(rd, 'extract', 'prompt_len.npy'))[rid], plen)
lp = np.load(os.path.join(rd, 'extract', 'lens_params.npz'))
letter_ids = [int(x) for x in lp['letter_ids']]
eps = float(lp['eps'])
dev = 'cuda'
norm_w = model.model.norm.weight.detach().double()
assert abs(model.model.norm.variance_epsilon - eps) < 1e-12
WH = model.lm_head.weight.detach()                                                  # bf16 (V, d)
U4 = WH[letter_ids].double()
assert np.allclose(U4.cpu().numpy(), lp['U4'], atol=1e-6)
E = U4 * norm_w[None]                                                       # (4, d)
PE = E.T @ torch.linalg.inv(E @ E.T) @ E                                    # (d, d) projector onto span(E)
z = np.load(os.path.join(rd, 'consensus_probe', f'weights_cons_{a.head}.npz'))
Wl = torch.from_numpy(z['W'][:, a.layer]).to(dev).double()                  # (5, 4, d)
bl = torch.from_numpy(z['b'][:, a.layer]).to(dev).double()                  # (5, 4)
fold_r = np.array([r['fold'] for r in rows])
L, d = model.config.num_hidden_layers, model.config.hidden_size
state = {}


def hook_l(mod, args, out):
    o = out[0] if isinstance(out, tuple) else out
    B = state['B']
    h = o[:, -1].double()                                                   # (3B, d)
    state['same'] = int((o[:B, -1].view(torch.int16) != o[B:2 * B, -1].view(torch.int16)).any(-1).sum()
                        + (o[:B, -1].view(torch.int16) != o[2 * B:, -1].view(torch.int16)).any(-1).sum())
    h0 = h[:B]
    hn = h0 / torch.sqrt(h0.pow(2).mean(-1, keepdim=True) + eps)
    Wf, bf = Wl[state['fold']], bl[state['fold']]
    zz = hn @ E.T + torch.einsum('bd,bkd->bk', hn, Wf) + bf
    cs = zz.argmax(-1)
    state['z'] = zz
    hperp = h0 - h0 @ PE

    def orth(w):
        w = w - w @ PE
        w = w - ((w * hperp).sum(-1, keepdim=True) / (hperp * hperp).sum(-1, keepdim=True)) * hperp
        return w / w.norm(dim=-1, keepdim=True)
    dm = orth(Wf[torch.arange(B, device=dev), cs] - Wf.mean(1))
    dr = orth(state['rnd'])
    nh = h0.norm(dim=-1, keepdim=True)
    o2 = o.clone()
    o2[B:2 * B, -1] = (h0 + a.alpha * nh * dm).to(torch.bfloat16)
    o2[2 * B:, -1] = (h0 + a.alpha * nh * dr).to(torch.bfloat16)
    state['ortho'] = max(state.get('ortho', 0.0), float(((o2[B:2 * B, -1].double() - h0) @ E.T).abs().max() /
                                                        (E.norm(dim=-1).max() * (a.alpha * nh).min())))
    return (o2,) + tuple(out[1:]) if isinstance(out, tuple) else o2


def hook_last(mod, args, out):
    o = out[0] if isinstance(out, tuple) else out
    state['last'] = o[:, -1].detach()


hs = [model.model.layers[a.layer].register_forward_hook(hook_l), model.model.layers[L - 1].register_forward_hook(hook_last)]
order = np.argsort(plen, kind='stable')
batches, i = [], 0
while i < len(order):
    j = i
    while j < len(order) and (j - i + 1) * plen[order[j]] <= a.max_tokens and j - i < a.max_bs:
        j += 1
    j = max(j, i + 1); batches.append(order[i:j]); i = j
n = len(rid)
let = np.full((n, 3), -9, np.int64); f4 = np.zeros((n, 3, 4), np.float32); top = np.zeros((n, 3), np.int64)
zh = np.zeros((n, 4), np.float32)
same_bad, ortho = 0, 0.0
with torch.no_grad():
    for bi, b in enumerate(batches):
        B = len(b)
        T = int(plen[b].max())
        x = torch.full((B, T), tok.pad_token_id, dtype=torch.long); am = torch.zeros((B, T), dtype=torch.long)
        for k, r in enumerate(b):
            x[k, T - plen[r]:] = torch.tensor(ids[r]); am[k, T - plen[r]:] = 1
        x, am = x.repeat(3, 1).to(dev), am.repeat(3, 1).to(dev)
        pos = (am.cumsum(-1) - 1).clamp(min=0)
        state.update(B=B, fold=torch.from_numpy(fold_r[b]).to(dev),
                     rnd=torch.from_numpy(np.stack([np.random.default_rng([777, int(rid[r])]).standard_normal(d) for r in b])).to(dev))
        model.model(input_ids=x, attention_mask=am, position_ids=pos, use_cache=False)
        same_bad += state['same']
        hl = state['last'].double()
        hn = hl / torch.sqrt(hl.pow(2).mean(-1, keepdim=True) + eps) * norm_w
        lg4 = (hn @ U4.T).float()
        best = torch.full((3 * B,), -torch.inf, device=dev, dtype=torch.float64); arg = torch.zeros(3 * B, dtype=torch.long, device=dev)
        for c0 in range(0, WH.shape[0], 16384):
            m, ix = (hn @ WH[c0:c0 + 16384].double().T).max(-1)
            upd = m > best
            best, arg = torch.where(upd, m, best), torch.where(upd, ix + c0, arg)
        lg4 = lg4.view(3, B, 4).permute(1, 0, 2).cpu().numpy(); arg = arg.view(3, B).T.cpu().numpy()
        f4[b], top[b], zh[b] = lg4, arg, state['z'].float().cpu().numpy()
        if bi % 50 == 0:
            print(f'batch {bi + 1}/{len(batches)} B={B} T={T} peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}GB '
                  f'same-h mismatches {same_bad} ortho {state["ortho"]:.2e}', C.stamp(t0), flush=True)
for h_ in hs:
    h_.remove()
s = np.sort(f4, -1)
let = np.where((s[..., -1] - s[..., -2] >= 1e-4) & np.isin(top, letter_ids), f4.argmax(-1), -1)
pz = np.exp(zh - zh.max(-1, keepdims=True)); pz /= pz.sum(-1, keepdims=True)
conf = pz.max(-1)
# ---- score (gold only here)
gold = np.array([r['gold'] for r in rows])
qc = {int(r['qid']): r for r in csv.DictReader(open(os.path.join(rd, 'classify', 'questions.csv')))}
cls = np.array([qc[q]['outcome_v1_K3'] for q in rid[0::4] // 4]); ua = np.array([qc[q]['u_avail_K3'] == '1' for q in rid[0::4] // 4])
keep = cls != 'X'
Q = n // 4
rng = np.random.default_rng(11)
bidx = rng.integers(0, int(keep.sum()), (4000, int(keep.sum())))


def net(after, before, qm=None):
    qm = keep if qm is None else keep & qm
    dq = ((after == gold).reshape(Q, 4).sum(1) - (before == gold).reshape(Q, 4).sum(1)) / 4.0
    if qm is not keep:
        dd = dq[qm]
        bs = dd[rng.integers(0, len(dd), (4000, len(dd)))].mean(1)
        return [round(100 * dd.mean(), 3), round(100 * np.percentile(bs, 2.5), 3), round(100 * np.percentile(bs, 97.5), 3)]
    dd = dq[keep]
    bs = dd[bidx].mean(1)
    return [round(100 * dd.mean(), 3), round(100 * np.percentile(bs, 2.5), 3), round(100 * np.percentile(bs, 97.5), 3)]


g = conf >= a.tau
none, meth, rnd = let[:, 0], let[:, 1], let[:, 2]
mg, rg = np.where(g, meth, none), np.where(g, rnd, none)
R = dict(args=vars(a), n_rows=n, n_q=Q, n_q_kept=int(keep.sum()), folds=folds, steered=round(float(g[np.repeat(keep, 4)].mean()), 4),
         same_h_mismatch_rows=same_bad, max_readout_leak_rel=state['ortho'],
         acc_none=round(float((none == gold)[np.repeat(keep, 4)].mean()), 5),
         net_method_gated=net(mg, none), net_rand_gated=net(rg, none), method_minus_rand=net(mg, rg),
         net_method_ungated=net(meth, none), net_rand_ungated=net(rnd, none),
         fixes_breaks=[int(((mg == gold) & (none != gold) & np.repeat(keep, 4)).sum()), int(((mg != gold) & (none == gold) & np.repeat(keep, 4)).sum())],
         by_class={k: net(mg, none, m) for k, m in (('R', cls == 'R'), ('U', cls == 'U'), ('U_avail', (cls == 'U') & ua),
                                                    ('U_none', (cls == 'U') & ~ua), ('M', cls == 'M'))},
         compliance=[round(float(np.isin(top[:, k], letter_ids).mean()), 5) for k in range(3)],
         seconds=round(time.time() - t0, 1))
# ---- cross-check against steer.py's saved arrays (same rows)
sd = os.path.join(rd, 'steer')
if os.path.exists(os.path.join(sd, 'meta.json')):
    sm = json.load(open(os.path.join(sd, 'meta.json')))
    li, cix = sm['layers'].index(a.layer), {c: i for i, c in enumerate(sm['configs'])}
    sf4, st1 = np.load(os.path.join(sd, 'final4.npy'), mmap_mode='r'), np.load(os.path.join(sd, 'top1.npy'), mmap_mode='r')
    def slet(c):
        x, t_ = np.asarray(sf4[rid, li, cix[c]]), np.asarray(st1[rid, li, cix[c]])
        ss = np.sort(x, -1)
        return np.where((ss[:, -1] - ss[:, -2] >= 1e-4) & np.isin(t_, letter_ids), x.argmax(-1), -1)
    s_none, s_meth = slet('none'), slet(f'{a.head}:cstar_perp@{a.alpha:g}')
    sh = np.load(os.path.join(sd, 'head4.npy'), mmap_mode='r')[rid, li, sm['heads'].index(a.head)]
    sp = np.exp(sh - sh.max(-1, keepdims=True)); sp /= sp.sum(-1, keepdims=True)
    sg = sp.max(-1) >= a.tau
    R['vs_steer_py'] = dict(none_letter_agree=round(float((s_none == none).mean()), 5), method_letter_agree=round(float((s_meth == meth).mean()), 5),
                            cstar_agree=round(float((sh.argmax(-1) == zh.argmax(-1)).mean()), 5), gate_agree=round(float((sg == g).mean()), 5),
                            steer_py_net_same_rows=net(np.where(sg, s_meth, s_none), s_none),
                            hook_method_vs_steer_none=net(mg, s_none))
    out4 = np.load(os.path.join(rd, 'extract', 'out_logits4.npy'))[rid]
    R['vs_steer_py']['hook_none_vs_extract_argmax_agree'] = round(float((f4[:, 0].argmax(-1) == out4.argmax(-1)).mean()), 5)
print(json.dumps(R, indent=1))
od = os.path.join(rd, 'steer_review'); os.makedirs(od, exist_ok=True)
tag = a.tag or f'{a.head}_L{a.layer}_a{a.alpha:g}_tau{a.tau:g}_f{"".join(map(str, folds))}'
np.savez(os.path.join(od, f'hook_{tag}.npz'), rid=rid, let=let, conf=conf, cstar=zh.argmax(-1))
json.dump(R, open(os.path.join(od, f'hook_{tag}.json'), 'w'), indent=1)
print('DONE', C.stamp(t0), flush=True)
