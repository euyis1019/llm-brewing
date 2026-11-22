"""Correctness and speed checks for engine.py against the HF forward.

Usage: python scripts/test_engine.py --model qwen3-4b [--speed]
"""
import argparse
import sys
import os
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E

QS = [
    'Find the number of positive divisors of 3600 that are perfect squares.',
    'Let $f(x)=2x^2-3x+1$. Compute $f(f(2))$.',
    'A bag has 5 red and 7 blue marbles. Two are drawn without replacement. What is the probability both are red?',
    'Solve for $x$: $\\log_2(x+3) + \\log_2(x-1) = 5$.',
]
SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'


class Always(E.Policy):
    def __init__(self, win, iters=0, mode='naive'):
        self.win, self.iters, self.mode = win, iters, mode

    def loop_decide(self, feats, ctx):
        return torch.ones_like(feats['ent'], dtype=torch.bool)


class Static(E.Policy):
    def __init__(self, layer):
        self.exit_layers = (layer,)

    def exit_decide(self, l, logits, ctx):
        return torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)


def prompt_ids(eng, q):
    kw = dict(add_generation_prompt=True, tokenize=False)
    if eng.fam == 'qwen3':
        kw['enable_thinking'] = False
    text = eng.tok.apply_chat_template([{'role': 'user', 'content': q + SUFFIX}], **kw)
    return eng.tok(text, add_special_tokens=False)['input_ids']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-4b')
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--R", type=int, default=48)
    a = ap.parse_args()
    eng = E.Engine(a.model, R=8, Tmax=2048)
    print(f'L={eng.L} d={eng.d} nh={eng.nh} kvh={eng.kvh} hd={eng.hd} V={eng.V} qk_norm={eng.qk_norm} '
          f'c_layer={eng.c_layer/1e6:.1f}M c_head={eng.c_head/1e6:.1f}M c_kv={eng.c_kv/1e6:.1f}M')
    prompts = [prompt_ids(eng, q) for q in QS]

    # 1. full-sequence forward vs HF
    ids = prompts[0]
    with torch.no_grad():
        hs = eng.forward_all(ids)
        mine = eng.readout(hs[-1])
        hf = eng.m(torch.tensor([ids], device=eng.dev)).logits[0].float()
    d = (mine - hf).abs().max().item()
    agree = (mine.argmax(-1) == hf.argmax(-1)).float().mean().item()
    print(f'[1] forward_all vs HF: max|dlogit|={d:.3f} argmax agree={agree:.4f} (T={len(ids)})')

    # 2. greedy decode vs HF teacher-forced argmax
    eng.temperature = 0.0
    jobs = [dict(jid=i, key=E.job_key('t', i), prompt=p) for i, p in enumerate(prompts)]
    out = eng.generate(jobs, E.Policy(), max_new=160)
    tot = ag = 0
    maxd = 0.0
    for i, p in enumerate(prompts):
        toks = out[i]['tokens']
        full = p + toks
        with torch.no_grad():
            hl = eng.m(torch.tensor([full], device=eng.dev)).logits[0].float()
        pred = hl[len(p) - 1:len(full) - 1].argmax(-1).tolist()
        ag += sum(int(x == y) for x, y in zip(pred, toks))
        tot += len(toks)
        first_div = next((k for k, (x, y) in enumerate(zip(pred, toks)) if x != y), None)
        print(f'   job {i}: {len(toks)} tokens, first disagreement at {first_div}')
    print(f'[2] greedy engine vs HF argmax: {ag}/{tot} = {ag / tot:.4f}')
    eng.graphs_on = False
    oe = eng.generate(jobs, E.Policy(), max_new=160)
    eng.graphs_on = True
    print(f"[2b] graphs vs eager identical jobs: {sum(int(oe[i]['tokens'] == out[i]['tokens']) for i in range(4))}/4")
    print('   sample:', repr(eng.tok.decode(out[0]['tokens'][:80])))

    # 3. loop identity (iters=0 recomputes post-window layers on the gathered subset path)
    out0 = eng.generate(jobs, E.Policy(), max_new=96)
    outl = eng.generate(jobs, Always((18, 25) if eng.L == 36 else (14, 19), iters=0), max_new=96)
    same = sum(sum(int(x == y) for x, y in zip(out0[i]['tokens'], outl[i]['tokens'])) for i in range(4))
    n = sum(min(len(out0[i]['tokens']), len(outl[i]['tokens'])) for i in range(4))
    print(f'[3] loop iters=0 vs base (greedy): {same}/{n} positions equal')

    # 4. static exit: first-step logits equal readout of layer e at the last prompt position
    e = eng.L - 6
    with torch.no_grad():
        hs = eng.forward_all(prompts[1])
        ref = eng.readout(hs[e][-1:])
    outx = eng.generate([jobs[1]], Static(e), max_new=8)
    print(f'[4] static exit at {e}: exit layers {outx[1]["exit"][:8]}, first token engine={outx[1]["tokens"][0]} '
          f'ref argmax={ref.argmax(-1).item()}')

    # 5. sampling determinism and batch-composition robustness
    eng.temperature = 0.7
    oa = eng.generate(jobs, E.Policy(), max_new=128)
    ob = eng.generate(jobs, E.Policy(), max_new=128)
    oc = eng.generate(list(reversed(jobs)), E.Policy(), max_new=128)
    dab = sum(int(oa[i]['tokens'] != ob[i]['tokens']) for i in range(4))
    dac = [next((k for k, (x, y) in enumerate(zip(oa[i]['tokens'], oc[i]['tokens'])) if x != y), None) for i in range(4)]
    print(f'[5] sampled: identical reruns differ in {dab}/4 jobs; reversed slot order first divergence {dac}')

    if a.speed:
        eng2 = None
        del eng
        torch.cuda.empty_cache()
        eng2 = E.Engine(a.model, R=a.R, Tmax=4096)
        prompts2 = [prompt_ids(eng2, q) for q in QS]
        jobs2 = [dict(jid=i, key=E.job_key("s", i), prompt=prompts2[i % 4]) for i in range(a.R)]
        for name, pol in [('base', E.Policy()), ('loop-all w', Always((18, 25) if eng2.L == 36 else (14, 19), 1))]:
            torch.cuda.synchronize()
            t0 = time.time()
            o = eng2.generate(jobs2, pol, max_new=384)
            torch.cuda.synchronize()
            nt = sum(len(v['tokens']) for v in o.values())
            print(f'[speed] {name}: {nt} tokens in {time.time() - t0:.1f}s = {nt / (time.time() - t0):.0f} tok/s '
                  f'mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')


if __name__ == '__main__':
    main()
