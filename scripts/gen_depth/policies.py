"""Controller policies for engine.Engine, built from compact spec strings.

Spec '<kind>[:k=v,...]':
  base                                   full depth
  exit:rule=static,layer=30              every token exits after block 30
  exit:rule=conf,thr=0.9,layers=20-34/2  exit at the first listed layer whose lens top-1 prob >= thr
  exit:rule=probe,probe=<exit_probe.npz>,thr=0.9[,layers=...]
                                         exit at the first probe layer whose self-supervised probe says 'same token
                                         as full depth' with probability >= thr (no readout needed to decide)
  skip:range=26-31[,probe=<exit_probe.npz>,thr=0.9]
                                         rows (all, or those whose probe at block 25 says committed) skip blocks
                                         26..31 (K/V propagated from the frozen state) and run the remaining tail
  exit:rule=tprobe,probe=<tuned_exit.npz>,thr=0.99
                                         same, with probes for the tuned lens; exited rows read out through it
  exit:rule=stable,m=2,thr=0.5,layers=20-34/2
                                         exit when the lens top-1 is unchanged over the last m listed layers
                                         and its prob >= thr
  loop:win=18-25,it=1,mode=naive,kv=last,cond=ent>=1.0;commit>=30
  temp:mult=1.5,cond=...                 zero-depth control: gated tokens sample at temperature x mult
  contrast:layer=24,alpha=0.5,plaus=0.1,cond=...   DoLa-style: log p_final + alpha (log p_final - log p_layer),
                                                   restricted to tokens with p_final >= plaus * max p_final
      cond: ';'-joined AND of '<feat><op><value>', feat in ent, p1, margin, commit, flips, step, rand,
            prevcrit, top1crit (rand: hashed uniform per token; *crit: 1 if previous token / final top-1 is in
            the critical vocabulary given by crit=<path to .npy of token ids>). Empty cond = every token.
tfeat=<tuned_exit.npz> reads the commitment features through the tuned lens.
All policies record the candidate-restricted commitment features over feat layers (default: every 2nd block from
L*0.44 to L-2), unless feat=0.
"""
import re

import numpy as np
import torch

import engine as E

OPS = {'>=': torch.ge, '<=': torch.le, '>': torch.gt, '<': torch.lt, '==': torch.eq}


def parse_layers(s):
    """'20-34/2' -> [20, 22, ..., 34]; '18,24,30' -> [18, 24, 30]."""
    out = []
    for part in s.split('+'):
        m = re.fullmatch(r'(\d+)-(\d+)(?:/(\d+))?', part)
        if m:
            a, b, st = int(m[1]), int(m[2]), int(m[3] or 1)
            out += list(range(a, b + 1, st))
        else:
            out += [int(x) for x in part.split(',') if x]
    return out


def default_feat_layers(L):
    return list(range(int(round(L * 0.44)), L - 1, 2))


class Spec(E.Policy):
    def __init__(self, spec, eng):
        self.spec = spec
        kind, _, rest = spec.partition(':')
        kv = dict(p.split('=', 1) for p in rest.split(',') if p) if rest else {}
        self.kind, self.kvs = kind, kv
        self.name = spec
        self.feat_layers = default_feat_layers(eng.L) if kv.get('feat', '1') != '0' else ()
        self.crit = None
        if 'tfeat' in kv:  # commitment features through the tuned lens (npz with layers, U, V, c)
            z = np.load(kv['tfeat'])
            T = lambda a_: torch.tensor(a_, dtype=torch.float32, device=eng.dev)  # noqa: E731
            self.feat_tuned = {int(l): (T(z['U'][i]), T(z['V'][i]), T(z['c'][i])) for i, l in enumerate(z['layers'].tolist())}
        if 'crit' in kv:
            ids = np.load(kv['crit'])
            self.crit = torch.zeros(eng.V, dtype=torch.bool, device=eng.dev)
            self.crit[torch.as_tensor(ids, device=eng.dev, dtype=torch.long)] = True
        if kind == 'base':
            pass
        elif kind == 'exit':
            self.rule = kv['rule']
            self.exit_fullkv = kv.get('fullkv', '0') == '1'
            if self.rule == 'static':
                self.exit_layers = (int(kv['layer']),)
            elif self.rule == 'tprobe':
                z = np.load(kv['probe'])
                keep = set(parse_layers(kv['layers'])) if 'layers' in kv else set(z['layers'].tolist())
                ls = [int(l) for l in z['layers'].tolist()]
                T = lambda a_: torch.tensor(a_, dtype=torch.float32, device=eng.dev)  # noqa: E731
                self.exit_probe = {l: (T(z['W'][i]), float(z['b'][i])) for i, l in enumerate(ls) if l in keep}
                self.exit_tuned = {l: (T(z['U'][i]), T(z['V'][i]), T(z['c'][i])) for i, l in enumerate(ls) if l in keep}
                self.exit_layers = tuple(sorted(self.exit_probe))
                self.exit_tau = float(kv.get('thr', 0.9))
            elif self.rule == 'probe':
                z = np.load(kv['probe'])
                keep = set(parse_layers(kv['layers'])) if 'layers' in kv else set(z['layers'].tolist())
                self.exit_probe = {int(l): (torch.tensor(z['W'][i], dtype=torch.float32, device=eng.dev), float(z['b'][i]))
                                   for i, l in enumerate(z['layers'].tolist()) if int(l) in keep}
                self.exit_layers = tuple(sorted(self.exit_probe))
                self.exit_tau = float(kv.get('thr', 0.9))
            else:
                self.exit_layers = tuple(parse_layers(kv['layers']))
                self.thr = float(kv.get('thr', 0.0))
                self.m = int(kv.get('m', 2))
        elif kind == 'skip':
            s0, s1 = map(int, kv['range'].split('-'))
            self.skip = (s0, s1)
            if 'probe' in kv:
                z = np.load(kv['probe'])
                ls = z['layers'].tolist()
                i = ls.index(s0 - 1)
                self.skip_probe = (torch.tensor(z['W'][i], dtype=torch.float32, device=eng.dev), float(z['b'][i]))
                self.skip_tau = float(kv.get('thr', 0.9))
        elif kind == 'oracle':
            layers = parse_layers(kv.get('layers', '16-34/2'))
            tuned = None
            if kv.get('read', 'logit') == 'tuned':
                z = np.load(kv['probe'])
                T = lambda a_: torch.tensor(a_, dtype=torch.float32, device=eng.dev)  # noqa: E731
                tuned = {int(l): (T(z['U'][i]), T(z['V'][i]), T(z['c'][i])) for i, l in enumerate(z['layers'].tolist())}
                layers = [l for l in layers if l in tuned]
            self.exit_oracle = (layers, tuned)
        elif kind in ('loop', 'contrast', 'temp'):
            if kind == 'temp':
                self.temp_mult = float(kv.get('mult', 1.5))
            elif kind == 'loop':
                a, b = map(int, kv['win'].split('-'))
                self.win = (a, b)
                self.iters = int(kv.get('it', 1))
                self.mode = kv.get('mode', 'naive')
                self.beta = float(kv.get('beta', 0.5))
                self.kv = kv.get('kv', 'last')         # 'last' | 'first' | 'none' (prediction only, natural K/V)
                self.loop_accept = kv.get('accept')
            elif kind == 'contrast':
                self.contrast = (int(kv['layer']), float(kv.get('alpha', 0.5)), float(kv.get('plaus', 0.1)))
            self.conds = []
            for c in [c for c in kv.get('cond', '').split(';') if c]:
                m = re.fullmatch(r'([a-z0-9]+)(>=|<=|==|>|<)([-0-9.e]+)', c)
                assert m, c
                self.conds.append((m[1], OPS[m[2]], float(m[3])))
            if any(f in ('commit', 'flips') for f, _, _ in self.conds):
                assert self.feat_layers, 'commit/flips conditions need feat layers'
        else:
            raise ValueError(spec)

    # ---------------------------------------------------------------- exit
    def exit_decide(self, l, logits, ctx):
        if self.rule == 'static':
            return torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        p = torch.softmax(logits, -1)
        p1, top = p.max(-1)
        if self.rule == 'conf':
            return p1 >= self.thr
        if self.rule == 'stable':
            hist = ctx['state'].setdefault('tops', [])
            hist.append(top)
            if len(hist) < self.m:
                return torch.zeros_like(p1, dtype=torch.bool)
            same = torch.ones_like(p1, dtype=torch.bool)
            for h in hist[-self.m:-1]:
                same &= h == top
            return same & (p1 >= self.thr)
        raise ValueError(self.rule)

    # ---------------------------------------------------------------- loop
    def _feat(self, name, feats, ctx):
        if name in feats:
            return feats[name].float()
        if name == 'step':
            return ctx['gstep'].float()
        if name == 'rand':
            return E.hash_uniform(ctx['keys'], ctx['gstep'], torch.zeros_like(ctx['keys'])[:, None], salt=7)[:, 0].float()
        if name == 'prevcrit':
            return self.crit[ctx['pend']].float()
        if name == 'top1crit':
            return self.crit[feats['top1']].float()
        raise KeyError(name)

    def loop_decide(self, feats, ctx):
        flag = torch.ones_like(feats['ent'], dtype=torch.bool)
        for name, op, val in self.conds:
            flag &= op(self._feat(name, feats, ctx), val)
        return flag


def make_policy(spec, eng):
    return Spec(spec, eng)
