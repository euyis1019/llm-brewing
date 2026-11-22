"""Quick B-line format exploration: compliance of the first assistant token under several chat formats."""
import json, os, sys, random, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C

SYS = 'You are an exam assistant. For each multiple-choice question, reply with only the letter (A, B, C, or D) of the correct choice. Do not explain.'
def wrap(tok, user, system=None, prefill=''):
    msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefill

VARIANTS = {
    'chat0': dict(system=None, prefill=''),
    'sys': dict(system=SYS, prefill=''),
    'pre_answer': dict(system=None, prefill='Answer:'),
    'sys_pre_answer': dict(system=SYS, prefill='Answer:'),
    'pre_theansweris': dict(system=None, prefill='The answer is'),
}
model = sys.argv[1]
lm = C.LM(model)
tok = lm.tok
qs = C.load_questions()
t0 = time.time()
res = {}
for name, cfg in VARIANTS.items():
    # letter ids for both no-space and space variants after this prefix
    q0 = qs[0]
    pre = wrap(tok, C.q_block(q0['question'], C.shown(q0['options'], 0), False) + '\n' + C.CHAT_INSTR, **cfg)
    nosp = [lm.encode(pre + L)[-1] for L in C.LETTERS]
    sp = [lm.encode(pre + ' ' + L)[-1] for L in C.LETTERS]
    allowed = set(nosp) | set(sp)
    lm.set_letters(nosp)
    out = {}
    for s in C.SOURCES:
        qsub = random.Random(1).sample([q for q in qs if q['source'] == s], 50)
        rows = [(q, v) for q in qsub for v in range(4)]
        idl = [lm.encode(wrap(tok, C.q_block(q['question'], C.shown(q['options'], v), False) + '\n' + C.CHAT_INSTR, **cfg)) for q, v in rows]
        n_ok = n_sp = n_corr = 0; bad = {}
        for i in range(0, len(idl), 32):
            ids, am, pos = lm.batch_tokens(idl[i:i + 32])
            H, _ = lm.forward_capture(ids, am, pos)
            lg = lm.fnorm32(H[:, -1]) @ lm.WU.T
            top = lg.argmax(-1)
            l4n = lg[:, nosp]; l4s = lg[:, sp]
            l4 = torch.logaddexp(l4n, l4s)
            for r in range(len(top)):
                q, v = rows[i + r]
                t = int(top[r])
                n_ok += t in allowed; n_sp += t in set(sp)
                n_corr += int(l4[r].argmax()) == C.gold_letter(q['answer'], v)
                if t not in allowed:
                    k = tok.convert_ids_to_tokens(t); bad[k] = bad.get(k, 0) + 1
        out[s] = dict(compliance=n_ok / 200, space_share=n_sp / max(n_ok, 1), acc4=n_corr / 200,
                      bad=sorted(bad.items(), key=lambda x: -x[1])[:4])
    res[name] = dict(nosp=nosp, sp=sp, per_source=out)
    print(name, json.dumps(out, ensure_ascii=False), f'{time.time()-t0:.0f}s', flush=True)
C.save_json(res, os.path.join(C.run_dir(model, 'chat0'), 'explore_chat_formats.json'))
