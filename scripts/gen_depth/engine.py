"""Per-token variable-depth decoding engine over frozen HF weights (Qwen3 / Qwen2.5).

Own forward (HF modules for the weights, own attention/cache), so inside one batch every token can stop early,
loop a layer window, or run normally.

- KV cache: per layer K/V of shape (R slots, kvh, Tmax, hd), bf16. Slot r holds positions [0, pos[r]).
- Decode step: all R slots run every block (uniform batch), as CUDA-graph segments split at the layers where the
  controller needs the residual. Two row masks drive a segment: `upd` (rows whose residual advances) and `wm`
  (rows whose K/V at the current position are written).
  * Early exit: an exited row stops advancing (upd=False) but keeps writing K/V, so its K/V in the skipped
    layers are projections of the exit state (CALM state propagation). The discarded compute is not counted in the
    budget (see the cost model c_layer / c_head / c_kv).
  * Loop: flagged rows re-run the window [a, b] from the window output (naive) or from its input with damped
    sub-steps (TFLT anchored), then the layers after the window; K/V at the current position are overwritten by the
    last pass (kv='last') or kept from the normal pass inside the window (kv='first').
- Sampling: temperature, top-k, top-p, then Gumbel-max with noise hashed from (job key, step, token id). Runs that
  differ only in the intervention share randomness, so trajectories diverge only where the intervention changed the
  sampled token.
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import hashlib
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    'qwen3-4b': (str(_model_root() / 'Qwen3-4B'), 'qwen3'),
    'qwen3-4b-base': (str(_model_root() / 'Qwen3-4B-Base'), 'qwen3'),
    'qwen2.5-7b-instruct': (str(_model_root() / 'huggingface/Qwen/Qwen2.5-7B-Instruct'), 'qwen2.5'),
    'qwen2.5-7b': (str(_model_root() / 'huggingface/Qwen/Qwen2.5-7B'), 'qwen2.5'),
}
# Qwen3: non-thinking recommendation. Qwen2.5-Instruct: generation_config (its repetition penalty 1.05 is not applied).
SAMPLING = {'qwen3': dict(temperature=0.7, top_p=0.8, top_k=20), 'qwen2.5': dict(temperature=0.7, top_p=0.8, top_k=20)}
EOS = (151645, 151643)
_LO31 = (1 << 31) - 1
_LO53 = (1 << 53) - 1


def job_key(*parts):
    return int(hashlib.md5('|'.join(map(str, parts)).encode()).hexdigest()[:15], 16)


def _mix(x):
    """splitmix64 finalizer on int64 tensors (masked arithmetic shifts emulate logical shifts; products wrap)."""
    x = x ^ ((x >> 33) & _LO31)
    x = x * -49064778989728563          # 0xff51afd7ed558ccd
    x = x ^ ((x >> 33) & _LO31)
    x = x * -4265267296055464877        # 0xc4ceb9fe1a85ec53
    return x ^ ((x >> 33) & _LO31)


def hash_uniform(key, step, tok, salt=0):
    """Uniform(0,1) noise, deterministic in (key, step, tok, salt). key/step (n,), tok (n, k) int64."""
    st = (salt * 3935559000370003845) & ((1 << 64) - 1)
    st = st - (1 << 64) if st >= (1 << 63) else st
    h = _mix(key[:, None] * 6364136223846793005 + step[:, None] * 1442695040888963407
             + tok * 2862933555777941757 + st)
    return ((h & _LO53).double() + 0.5) / float(1 << 53)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Policy:
    """Per-token controller. Default: plain full-depth decoding.

    exit_layers: layers after which a full-vocab readout is taken and exit_decide() may stop rows.
    feat_layers: layers whose residual is kept for cheap candidate-restricted features (commitment layer).
    win (a, b): loop window (inclusive), iters extra passes, mode 'naive' | 'damped' (beta anchor), kv 'last' | 'first'.
    loop_decide(feats, ctx) flags rows to loop after the normal pass (exited rows are never looped).
    """
    name = 'base'
    exit_layers = ()
    feat_layers = ()
    win = None
    iters = 1
    mode = 'naive'
    beta = 0.5
    kv = 'last'
    n_cand = 16
    contrast = None    # (layer, alpha, plausibility): DoLa-style readout correction on rows flagged by loop_decide
    exit_probe = None  # {layer: (w (d,) float tensor, b float)}: exit when sigmoid(w . norm(h) + b) >= exit_tau,
    exit_tau = 0.9     # decided before any readout (one readout only for rows that exit)
    exit_tuned = None  # {layer: (U, V, c)}: exited rows read out through the low-rank tuned lens z + (z V) U^T + c
    exit_fullkv = False  # diagnostic: exited rows keep computing (full-depth K/V), only the emitted token is early
    temp_mult = None   # zero-depth control: flagged rows (loop_decide) sample from final logits / temp_mult
    loop_accept = None  # 'sharper': keep the looped distribution only if its top-1 prob exceeds the natural one
    exit_oracle = None  # diagnostic "perfect exit classifier": (layers, tuned dict or None). After the full pass, each
                        # token gets the earliest listed layer whose readout already samples the full-depth token (and
                        # every later listed layer does too); the emitted token is the full-depth one, but that token's
                        # K/V above its exit layer are overwritten with projections of the exit state (CALM propagation)
    feat_tuned = None  # {layer: (U, V, c)}: commitment features read through the tuned lens instead of the logit lens
    skip = None        # (s0, s1): rows flagged at block s0-1 skip blocks s0..s1 (K/V propagated), then run the tail
    skip_probe = None  # (w, b) on norm(h_{s0-1}); None = skip every token
    skip_tau = 0.9

    def exit_decide(self, l, logits, ctx):
        return None

    def loop_decide(self, feats, ctx):
        return None


class Engine:
    def __init__(self, key, R=48, Tmax=4096, device='cuda', dtype=torch.bfloat16, graphs=True, bucket=256):
        path, fam = MODELS[key]
        self.key, self.fam, self.dev = key, fam, device
        self.tok = AutoTokenizer.from_pretrained(path)
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        m.requires_grad_(False)
        self.m = m
        cfg = m.config
        self.layers = m.model.layers
        self.L = len(self.layers)
        self.d, self.nh, self.kvh = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, 'head_dim', None) or self.d // self.nh
        self.g = self.nh // self.kvh
        self.embed, self.norm, self.lm_head, self.rotary = m.model.embed_tokens, m.model.norm, m.lm_head, m.model.rotary_emb
        self.W = self.lm_head.weight  # (V, d)
        self.V = self.W.shape[0]
        self.qk_norm = hasattr(self.layers[0].self_attn, 'q_norm')
        self.R, self.Tmax, self.graphs_on, self.bucket = R, Tmax, graphs, bucket
        self.Kc = torch.zeros(self.L, R, self.kvh, Tmax, self.hd, dtype=dtype, device=device)
        self.Vc = torch.zeros_like(self.Kc)
        s = SAMPLING[fam]
        self.temperature, self.top_p, self.top_k = s['temperature'], s['top_p'], s['top_k']
        at = self.layers[0].self_attn
        # cost model in multiply-accumulates per token: a block, a full-vocab readout, K/V projection of one block
        self.c_layer = sum(p.numel() for p in self.layers[0].parameters())
        self.c_head = self.V * self.d
        self.c_kv = at.k_proj.weight.numel() + at.v_proj.weight.numel()
        self.ar = torch.arange(Tmax, device=device)
        self.arR = torch.arange(R, device=device)
        # static buffers for CUDA-graph segments
        self.s_x = torch.zeros(R, self.d, dtype=dtype, device=device)
        self.s_out = torch.zeros_like(self.s_x)
        self.s_pos = torch.zeros(R, dtype=torch.long, device=device)
        self.s_upd = torch.zeros(R, dtype=torch.bool, device=device)
        self.s_wm = torch.zeros(R, dtype=torch.bool, device=device)
        self.graphs, self.pool = {}, None

    # ------------------------------------------------------------------ blocks
    def _qkv(self, l, x):
        lay = self.layers[l]
        at = lay.self_attn
        h = lay.input_layernorm(x)
        sh = x.shape[:-1]
        q = at.q_proj(h).view(*sh, self.nh, self.hd)
        k = at.k_proj(h).view(*sh, self.kvh, self.hd)
        v = at.v_proj(h).view(*sh, self.kvh, self.hd)
        if self.qk_norm:
            q, k = at.q_norm(q), at.k_norm(k)
        return q, k, v

    @staticmethod
    def _rope(q, k, cos, sin):
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

    def _block(self, l, x, pos, upd, wm, cos, sin, Tb):
        """One block for one new position in every slot (graph-safe: no host syncs, fixed shapes)."""
        lay = self.layers[l]
        at = lay.self_attn
        R = x.shape[0]
        q, k, v = self._qkv(l, x)
        q, k = self._rope(q[:, :, None], k[:, :, None], cos, sin)
        k = k[:, :, 0]
        Kl, Vl = self.Kc[l], self.Vc[l]
        w = wm[:, None, None]
        Kl[self.arR, :, pos] = torch.where(w, k, Kl[self.arR, :, pos])
        Vl[self.arR, :, pos] = torch.where(w, v, Vl[self.arR, :, pos])
        Kt, Vt = Kl[:, :, :Tb], Vl[:, :, :Tb]
        qg = q.view(R, self.kvh, self.g, self.hd)
        sc = torch.matmul(qg, Kt.transpose(-1, -2)).float() * at.scaling  # (R, kvh, g, Tb)
        mask = self.ar[:Tb][None, :] <= pos[:, None]
        sc = sc.masked_fill(~mask[:, None, None, :], float('-inf'))
        o = torch.matmul(torch.softmax(sc, -1).to(Vt.dtype), Vt)
        xn = x + at.o_proj(o.reshape(R, self.nh * self.hd))
        xn = xn + lay.mlp(lay.post_attention_layernorm(xn))
        return torch.where(upd[:, None], xn, x)

    def _seg_body(self, l0, l1, Tb):
        x = self.s_x
        cos, sin = self.rotary(x[:, None], self.s_pos[:, None])
        for l in range(l0, l1):
            x = self._block(l, x, self.s_pos, self.s_upd, self.s_wm, cos, sin, Tb)
        self.s_out.copy_(x)

    def run_seg(self, l0, l1, x, pos, upd, wm, Tcur):
        """Blocks l0..l1-1 for all slots. Writes K/V of rows in wm; advances rows in upd. Returns new residual."""
        Tb = min(self.Tmax, -(-Tcur // self.bucket) * self.bucket)
        self.s_x.copy_(x)
        self.s_pos.copy_(pos)
        self.s_upd.copy_(upd)
        self.s_wm.copy_(wm)
        if not self.graphs_on:
            self._seg_body(l0, l1, Tb)
            return self.s_out.clone()
        key = (l0, l1, Tb)
        g = self.graphs.get(key)
        if g is None:
            # warm-up executes the segment once (idempotent: same inputs, same K/V writes), then capture
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                self._seg_body(l0, l1, Tb)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._seg_body(l0, l1, Tb)
            self.pool = g.pool()
            self.graphs[key] = g
        g.replay()
        return self.s_out.clone()

    def readout(self, x):
        return self.lm_head(self.norm(x)).float()

    def _prefill(self, slots, prompts, max_tokens=12288):
        """Prefill new slots in sub-batches of at most max_tokens padded tokens."""
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        grp = []
        for i in order:
            if grp and (len(grp) + 1) * len(prompts[i]) > max_tokens:
                self._prefill_batch([slots[k] for k in grp], [prompts[k] for k in grp])
                grp = []
            grp.append(i)
        if grp:
            self._prefill_batch([slots[k] for k in grp], [prompts[k] for k in grp])

    def _prefill_batch(self, slots, prompts):
        n, T = len(prompts), max(len(p) for p in prompts)
        ids = torch.zeros(n, T, dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, :len(p)] = torch.tensor(p)
        ids = ids.to(self.dev)
        sl = torch.tensor(slots, device=self.dev)
        x = self.embed(ids)
        cos, sin = self.rotary(x, torch.arange(T, device=self.dev)[None].expand(n, T))
        for l in range(self.L):
            lay = self.layers[l]
            at = lay.self_attn
            q, k, v = self._qkv(l, x)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q, k = self._rope(q, k, cos, sin)
            self.Kc[l][sl, :, :T] = k
            self.Vc[l][sl, :, :T] = v
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=at.scaling, enable_gqa=True)
            x = x + at.o_proj(o.transpose(1, 2).reshape(n, T, -1))
            x = x + lay.mlp(lay.post_attention_layernorm(x))

    def sample(self, logits, keys, step):
        if self.temperature <= 0:
            return logits.argmax(-1)
        lg = logits / self.temperature
        tv, ti = lg.topk(self.top_k, dim=-1)
        p = torch.softmax(tv, -1)
        keep = (p.cumsum(-1) - p) < self.top_p
        u = hash_uniform(keys, step, ti)
        gum = (-torch.log(-torch.log(u))).float()
        j = torch.where(keep, tv + gum, torch.full_like(tv, float('-inf'))).argmax(-1)
        return ti.gather(1, j[:, None]).squeeze(1)

    # ------------------------------------------------------------------ features
    def features(self, final, saved, pol):
        """Final-layer features + candidate-restricted commitment layer over pol.feat_layers."""
        lp = torch.log_softmax(final, -1)
        p = lp.exp()
        ent = -(p * lp).sum(-1)
        t2 = p.topk(2, dim=-1).values
        f = dict(ent=ent, p1=t2[:, 0], margin=t2[:, 0] - t2[:, 1], top1=final.argmax(-1))
        if pol.feat_layers:
            cand = final.topk(pol.n_cand, dim=-1).indices      # (R, C); cand[:, 0] = final top-1
            Wc = self.W[cand]                                  # (R, C, d)
            commit = torch.full_like(f['top1'], self.L - 1)
            ok_all = torch.ones_like(f['top1'], dtype=torch.bool)
            flips = torch.zeros_like(f['top1'])
            prev = None
            for l in sorted(pol.feat_layers, reverse=True):
                z = self.norm(saved[l])
                if pol.feat_tuned is not None and l in pol.feat_tuned:
                    U, V, c = pol.feat_tuned[l]
                    zf = z.float()
                    z = (zf + (zf @ V) @ U.T + c).to(z.dtype)
                win = torch.einsum('nd,ncd->nc', z, Wc).argmax(-1) == 0
                ok_all = ok_all & win
                commit = torch.where(ok_all, torch.full_like(commit, l), commit)
                if prev is not None:
                    flips += (win != prev).long()
                prev = win
            f['commit'] = commit   # earliest feat layer from which the final top-1 wins among candidates to the end
            f['flips'] = flips
        return f

    # ------------------------------------------------------------------ one decode step
    def _step(self, active, pend, pos, gstep, keys, pol):
        R, L = self.R, self.L
        x = self.embed(pend)
        Tcur = int(pos.max()) + 1
        need = set(pol.feat_layers)
        if pol.exit_oracle:
            need |= set(pol.exit_oracle[0])
        if pol.win:
            need |= {pol.win[0] - 1, pol.win[1]}
        if pol.contrast:
            need.add(pol.contrast[0])
        if pol.skip:
            need |= {pol.skip[0] - 1, pol.skip[1]}
        bounds = sorted({l for l in need | set(pol.exit_layers) if 0 <= l < L - 1}) + [L - 1]
        ones = torch.ones(R, dtype=torch.bool, device=self.dev)
        exited = torch.zeros(R, dtype=torch.bool, device=self.dev)
        ex_layer = torch.full((R,), L - 1, dtype=torch.long, device=self.dev)
        ex_logits = None
        n_read = torch.zeros(R, dtype=torch.long, device=self.dev)
        saved = {}
        ctx = dict(active=active, pend=pend, gstep=gstep, keys=keys, pos=pos, state={})
        skipped = torch.zeros(R, dtype=torch.bool, device=self.dev)
        l0 = 0
        for lb in bounds:
            ex_frz = torch.zeros_like(exited) if pol.exit_fullkv else exited
            frozen = ex_frz | skipped if (pol.skip and l0 >= pol.skip[0] and lb <= pol.skip[1]) else ex_frz
            x = self.run_seg(l0, lb + 1, x, pos, ~frozen, ones, Tcur)
            l0 = lb + 1
            if lb in need:
                saved[lb] = x
            if pol.skip and lb == pol.skip[0] - 1:
                if pol.skip_probe is None:
                    skipped = active.clone()
                else:
                    w, b0 = pol.skip_probe
                    skipped = (torch.sigmoid(self.norm(x).float() @ w + b0) >= pol.skip_tau) & active
            if lb in pol.exit_layers and lb < L - 1 and pol.exit_probe is not None:
                w, b0 = pol.exit_probe[lb]
                sc = self.norm(x).float() @ w + b0
                dec = (torch.sigmoid(sc) >= pol.exit_tau) & active & ~exited
                if bool(dec.any()):
                    if pol.exit_tuned is not None and lb in pol.exit_tuned:
                        U, V, c = pol.exit_tuned[lb]
                        zz = self.norm(x).float()
                        lg = self.lm_head((zz + (zz @ V) @ U.T + c).to(x.dtype)).float()
                    else:
                        lg = self.readout(x)
                    n_read += dec.long()
                    ex_logits = lg if ex_logits is None else torch.where(dec[:, None], lg, ex_logits)
                    ex_layer = torch.where(dec, torch.full_like(ex_layer, lb), ex_layer)
                    exited = exited | dec
            elif lb in pol.exit_layers and lb < L - 1:
                lg = self.readout(x)
                n_read += (active & ~exited).long()
                dec = pol.exit_decide(lb, lg, ctx)
                if dec is not None:
                    dec = dec & active & ~exited
                    ex_logits = lg if ex_logits is None else torch.where(dec[:, None], lg, ex_logits)
                    ex_layer = torch.where(dec, torch.full_like(ex_layer, lb), ex_layer)
                    exited = exited | dec
        final = self.readout(x)
        n_read += (active & ~exited).long()
        logits = final if ex_logits is None else torch.where(exited[:, None], ex_logits, final)
        feats = self.features(final, saved, pol)
        looped = torch.zeros(R, dtype=torch.bool, device=self.dev)
        if pol.win:
            flag = pol.loop_decide(feats, ctx)
            if flag is not None:
                flag = flag & active & ~exited
                if bool(flag.any()):
                    xl = self._loop(flag, saved, pos, Tcur, pol)
                    lgl = self.readout(xl)
                    if pol.loop_accept == 'sharper':
                        flag = flag & (torch.softmax(lgl, -1).max(-1).values > feats['p1'])
                    logits = torch.where(flag[:, None], lgl, logits)
                    looped = flag
        if pol.contrast:
            flag = pol.loop_decide(feats, ctx)
            if flag is not None:
                flag = flag & active & ~exited
                if bool(flag.any()):
                    cl, alpha, plaus = pol.contrast
                    lpf = torch.log_softmax(final, -1)
                    lpl = torch.log_softmax(self.readout(saved[cl]), -1)
                    sc = lpf + alpha * (lpf - lpl)
                    sc = sc.masked_fill(lpf < lpf.max(-1, keepdim=True).values + float(torch.log(torch.tensor(plaus))),
                                        float('-inf'))
                    logits = torch.where(flag[:, None], sc, logits)
                    looped = flag
        if pol.exit_oracle:
            olayers, otuned = pol.exit_oracle
            tok_full = self.sample(final, keys, gstep)
            agree = []
            for l in olayers:
                z = self.norm(saved[l])
                if otuned is not None and l in otuned:
                    U, V, c = otuned[l]
                    zf = z.float()
                    lg = self.lm_head((zf + (zf @ V) @ U.T + c).to(z.dtype)).float()
                else:
                    lg = self.lm_head(z).float()
                agree.append(self.sample(lg, keys, gstep) == tok_full)
            ok = torch.ones_like(active)
            oex = torch.full_like(ex_layer, L - 1)
            for i in reversed(range(len(olayers))):
                ok = ok & agree[i]
                oex = torch.where(ok, torch.full_like(oex, olayers[i]), oex)
            oex = torch.where(active, oex, torch.full_like(oex, L - 1))
            zeros = torch.zeros_like(active)
            for e in sorted(set(oex.tolist())):
                if e >= L - 1:
                    continue
                m = oex == e
                self.run_seg(e + 1, L, saved[e], pos, zeros, m, Tcur)   # frozen residual -> propagated K/V for rows m
            ex_layer = oex
        if pol.temp_mult:
            flag = pol.loop_decide(feats, ctx)
            if flag is not None:
                flag = flag & active & ~exited
                logits = torch.where(flag[:, None], final / pol.temp_mult, logits)
                looped = flag
        self._last = (final, logits)
        tok = self.sample(logits, keys, gstep)
        info = dict(exit=ex_layer, loop=looped, ent=feats['ent'], p1=feats['p1'], nread=n_read)
        if pol.skip:
            info['skip'] = skipped
        if pol.feat_layers:
            info['commit'], info['flips'] = feats['commit'], feats['flips']
        if pol.win or pol.contrast or pol.temp_mult:  # did the intervention change the sampled token vs the normal pass?
            info['chg'] = looped & (tok != self.sample(final, keys, gstep))
        return tok, info

    def _loop(self, flag, saved, pos, Tcur, pol):
        a, b = pol.win
        none = torch.zeros_like(flag)
        wwin = flag if pol.kv == 'last' else none
        wpost = none if pol.kv == 'none' else flag
        if pol.mode == 'naive':
            x = saved[b]
            for _ in range(pol.iters):
                x = self.run_seg(a, b + 1, x, pos, flag, wwin, Tcur)
        else:  # damped: K sub-steps of size 1/K from the window input, then beta-anchor to the natural window output
            x0, g0 = saved[a - 1], saved[b]
            K = pol.iters + 1
            xk = x0 + (g0 - x0) / K
            for _ in range(K - 1):
                y = self.run_seg(a, b + 1, xk, pos, flag, wwin, Tcur)
                xk = xk + (y - xk) / K
            x = pol.beta * g0 + (1 - pol.beta) * xk
        return self.run_seg(b + 1, self.L, x, pos, flag, wpost, Tcur)

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, jobs, pol, max_new=3072, log_every=0, log=print, on_done=None, record_first=0):
        """jobs: list of dict(jid, key, prompt list[int][, max_new][, step0: noise step of the first new token]).
        Continuous batching over R slots.
        Returns {jid: dict(tokens, exit, loop, ent, p1, nread[, commit, flips][, chg])} (per generated token);
        with on_done(jid, record) each finished job is handed over instead of kept.
        record_first=K: at each job's first generated token, record log-probs of the natural final distribution and
        of the distribution actually sampled (after any intervention) on K ids (job['eval_ids'] or natural top-K)."""
        R, dev = self.R, self.dev
        queue = list(jobs)
        slot_job = [None] * R
        bufs = [None] * R
        pos = torch.zeros(R, dtype=torch.long, device=dev)
        pend = torch.zeros(R, dtype=torch.long, device=dev)
        gstep = torch.zeros(R, dtype=torch.long, device=dev)
        keys = torch.zeros(R, dtype=torch.long, device=dev)
        out, nstep, ntok, t0 = {}, 0, 0, time.time()
        while queue or any(s is not None for s in slot_job):
            free = [r for r in range(R) if slot_job[r] is None]
            if free and queue:
                new = []
                while free and queue:
                    r, j = free.pop(0), queue.pop(0)
                    assert len(j['prompt']) + j.get('max_new', max_new) + 1 <= self.Tmax, (j['jid'], len(j['prompt']))
                    slot_job[r] = j
                    new.append(r)
                self._prefill(new, [slot_job[r]['prompt'][:-1] for r in new])
                for r in new:
                    p = slot_job[r]['prompt']
                    pos[r], pend[r], gstep[r], keys[r] = len(p) - 1, p[-1], slot_job[r].get('step0', 0), slot_job[r]['key']
                    bufs[r] = None
            act_l = [s is not None for s in slot_job]
            active = torch.tensor(act_l, device=dev)
            pos = torch.where(active, pos, torch.zeros_like(pos))
            tok, info = self._step(active, pend, pos, gstep, keys, pol)
            ik = list(info)
            pack = torch.stack([tok.double()] + [info[k].double() for k in ik]).cpu().tolist()
            tl = [int(t) for t in pack[0]]
            for r in range(R):
                if not act_l[r]:
                    continue
                if bufs[r] is None:
                    bufs[r] = {k: [] for k in ['tokens'] + ik}
                    if record_first:
                        fin, used = self._last
                        ev = slot_job[r].get('eval_ids')
                        ids = torch.tensor(ev, device=dev) if ev is not None else fin[r].topk(record_first).indices
                        lpn, lpu = torch.log_softmax(fin[r], -1), torch.log_softmax(used[r], -1)
                        bufs[r]['first'] = dict(ids=ids.tolist(), lp_nat=lpn[ids].tolist(), lp_used=lpu[ids].tolist(),
                                                ent_nat=float(-(lpn.exp() * lpn).sum()))
                b = bufs[r]
                b['tokens'].append(tl[r])
                for i, k in enumerate(ik):
                    b[k].append(pack[i + 1][r])
                ntok += 1
                if tl[r] in EOS or len(b['tokens']) >= slot_job[r].get('max_new', max_new):
                    if on_done is not None:
                        on_done(slot_job[r]['jid'], b)
                        out[slot_job[r]['jid']] = None
                    else:
                        out[slot_job[r]['jid']] = b
                    slot_job[r] = None
            pos = pos + active.long()
            gstep = gstep + active.long()
            pend = torch.where(active, tok, pend)
            nstep += 1
            if log_every and nstep % log_every == 0:
                el = time.time() - t0
                log(f'step {nstep} done {len(out)}/{len(jobs)} tok {ntok} {ntok / el:.0f} tok/s active {sum(act_l)}')
        return out

    # ------------------------------------------------------------------ teacher-forced reference forward
    @torch.no_grad()
    def forward_all(self, ids):
        """Full-sequence forward (single sequence) returning the residual after every block: (L, T, d)."""
        ids = torch.tensor(ids, device=self.dev)[None]
        T = ids.shape[1]
        x = self.embed(ids)
        cos, sin = self.rotary(x, torch.arange(T, device=self.dev)[None])
        hs = []
        for l in range(self.L):
            lay = self.layers[l]
            at = lay.self_attn
            q, k, v = self._qkv(l, x)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q, k = self._rope(q, k, cos, sin)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=at.scaling, enable_gqa=True)
            x = x + at.o_proj(o.transpose(1, 2).reshape(1, T, -1))
            x = x + lay.mlp(lay.post_attention_layernorm(x))
            hs.append(x[0])
        return torch.stack(hs)
