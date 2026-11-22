"""Smoke tests for one model/format.
1) core checks: letter token ids; left-padded batch == unpadded; fp32 readout == model argmax; CSD prefix-KV ==
   naive hook Patchscope; forward_sequence(identity) == normal forward; last-position re-run == normal forward.
2) format compliance per source (base: k in {0,1,3} -> picks smallest k with compliance >= 0.95; chat: chat0).
Usage: python scripts/smoke.py --model qwen3-4b-base --fmt base_fewshot
"""
import argparse, json, os, sys, time, random
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
    ap.add_argument('--nq', type=int, default=50); ap.add_argument('--ks', default='0,1,3')
    ap.add_argument('--max_tokens', type=int, default=24000)
    a = ap.parse_args()
    t0 = time.time()
    lm = C.LM(a.model)
    tok = lm.tok
    out = {'model': a.model, 'fmt': a.fmt, 'checks': {}}
    target = C.target_text(a.fmt, tok)
    qs = C.load_questions()
    demos = C.load_demos()
    # letter ids at the critical position
    probe_prefix = C.prompt_text(qs[0], 0, a.fmt, tok, demos[qs[0]['source']], 0)
    ids_crit = lm.find_letter_ids(probe_prefix)
    ids_tgt = lm.find_letter_ids(target)
    assert ids_crit == ids_tgt, (ids_crit, ids_tgt)
    lm.set_letters(ids_crit)
    out['letter_ids'] = ids_crit
    out['letter_tokens'] = [tok.convert_ids_to_tokens(i) for i in ids_crit]
    sp = [tok.encode(' ' + L, add_special_tokens=False) for L in C.LETTERS]
    out['space_letter_ids'] = sp
    out['target_text'] = target
    out['target_ids'] = lm.encode(target)
    out['prompt_tail_tokens'] = tok.convert_ids_to_tokens(lm.encode(probe_prefix)[-6:])
    print('letters', out['letter_ids'], out['letter_tokens'], 'tail', out['prompt_tail_tokens'], C.stamp(t0), flush=True)

    # --- core checks on 8 prompts with mixed lengths
    rng = random.Random(0)
    sub = rng.sample(qs, 8)
    texts = [C.prompt_text(q, rng.randrange(4), a.fmt, tok, demos[q['source']], 3 if a.fmt == 'base_fewshot' else 0) for q in sub]
    idl = [lm.encode(t) for t in texts]
    ids, am, pos = lm.batch_tokens(idl)
    H, cache = lm.forward_capture(ids, am, pos, use_cache=True)
    single = torch.stack([lm.forward_capture(*lm.batch_tokens([x]))[0][0] for x in idl])
    d_pad = (lm.letters32(H[:, -1]) - lm.letters32(single[:, -1])).abs().max().item()
    out['checks']['leftpad_vs_single_maxdiff_letterlogit'] = d_pad
    with torch.no_grad():
        full = lm.model(input_ids=ids, attention_mask=am, position_ids=pos, logits_to_keep=1).logits[:, -1].float()
    agree = (full.argmax(-1) == (lm.fnorm32(H[:, -1]) @ lm.WU.T).argmax(-1)).float().mean().item()
    out['checks']['fp32_vs_model_vocab_argmax_agree'] = agree
    # CSD vs naive patchscope
    tids = torch.tensor([lm.encode(target)], device=lm.device)
    diffs = {}
    for l in sorted({x for x in [0, 5, 12, 18, 24, 30, lm.L - 2, lm.L - 1] if x < lm.L}):
        mine = lm.letters32(lm.csd_states(H[:, l], torch.full((8,), l), target))
        def rep(mod, args, o, l=l):
            o2 = o[0] if isinstance(o, tuple) else o
            o2 = o2.clone(); o2[:, -1] = H[:, l]
            return (o2,) + tuple(o[1:]) if isinstance(o, tuple) else o2
        hk = lm.layers[l].register_forward_hook(rep)
        with torch.no_grad():
            st, _ = lm.forward_capture(tids.expand(8, -1), torch.ones_like(tids.expand(8, -1)),
                                       torch.arange(tids.shape[1], device=lm.device)[None].expand(8, -1))
        hk.remove()
        naive = lm.letters32(st[:, -1])
        diffs[l] = [(mine - naive).abs().max().item(), (mine.argmax(-1) == naive.argmax(-1)).float().mean().item()]
    out['checks']['csd_vs_naive_patchscope'] = diffs
    # identity sequence and last-position re-run
    seqH = lm.forward_sequence(ids, am, pos, list(range(lm.L)))
    out['checks']['forward_sequence_identity_maxdiff_letterlogit'] = (lm.letters32(seqH[:, -1]) - lm.letters32(H[:, -1])).abs().max().item()
    T = ids.shape[1]
    ks = [cache.layers[i].keys[:, :, :T - 1] for i in range(lm.L)]
    vs = [cache.layers[i].values[:, :, :T - 1] for i in range(lm.L)]
    kv = C.PrefixKV(ks, vs)
    mask = am.bool()[:, None, None, :]
    lastpos = pos[:, -1:]
    b = 20
    hb = lm.run_single(H[:, b], list(range(b + 1, lm.L)), kv, lastpos, mask=mask)
    out['checks']['lastpos_rerun_maxdiff_letterlogit'] = (lm.letters32(hb) - lm.letters32(H[:, -1])).abs().max().item()
    print(json.dumps(out['checks'], indent=1), C.stamp(t0), flush=True)

    # --- compliance per source
    comp = {}
    by_src = {s: [q for q in qs if q['source'] == s] for s in C.SOURCES}
    klist = [int(x) for x in a.ks.split(',')] if a.fmt == 'base_fewshot' else [0]
    for s in C.SOURCES:
        qsub = random.Random(1).sample(by_src[s], min(a.nq, len(by_src[s])))
        for k in klist:
            rows = [(q, v) for q in qsub for v in range(4)]
            texts = [C.prompt_text(q, v, a.fmt, tok, demos[s], k) for q, v in rows]
            idl = [lm.encode(t) for t in texts]
            order = sorted(range(len(idl)), key=lambda i: len(idl[i]))
            n_ok = n_correct = n_correct_given_ok = 0
            first_tokens = {}
            i = 0
            while i < len(order):
                T = len(idl[order[i]]); j = i
                while j < len(order) and (j - i + 1) * len(idl[order[j]]) <= a.max_tokens and j - i < 64:
                    j += 1
                j = max(j, i + 1)
                bidx = order[i:j]
                ids, am, pos = lm.batch_tokens([idl[x] for x in bidx])
                Hb, _ = lm.forward_capture(ids, am, pos)
                top, mass = lm.vocab_stats(Hb[:, -1])
                l4 = lm.letters32(Hb[:, -1]).argmax(-1)
                for r, x in enumerate(bidx):
                    q, v = rows[x]
                    ok = int(top[r]) in lm.letter_ids
                    corr = int(l4[r]) == C.gold_letter(q['answer'], v)
                    n_ok += ok; n_correct += corr; n_correct_given_ok += (ok and corr)
                    if not ok:
                        t = tok.convert_ids_to_tokens(int(top[r]))
                        first_tokens[t] = first_tokens.get(t, 0) + 1
                i = j
            n = len(rows)
            comp[f'{s}|k={k}'] = dict(n=n, compliance=n_ok / n, acc4=n_correct / n,
                                      mean_len=sum(map(len, idl)) / n, max_len=max(map(len, idl)),
                                      noncompliant_top=sorted(first_tokens.items(), key=lambda x: -x[1])[:5])
            print(s, k, json.dumps(comp[f'{s}|k={k}'], ensure_ascii=False), C.stamp(t0), flush=True)
    out['compliance'] = comp
    if a.fmt == 'base_fewshot':
        kmap = {}
        for s in C.SOURCES:
            good = [k for k in klist if comp[f'{s}|k={k}']['compliance'] >= 0.95]
            kmap[s] = min(good) if good else max(klist, key=lambda k: comp[f'{s}|k={k}']['compliance'])
        out['chosen_k'] = kmap
        # An existing fewshot_k file is the study protocol (round 1 fixed k=1 for every model); smoke only
        # reports its own choice then. A new model gets the automatic choice written as its starting point.
        if os.path.exists(C.fewshot_k_path(a.model)):
            out['fewshot_k_kept'] = C.fewshot_k_path(a.model)
        else:
            C.save_json({'k': kmap, 'rule': 'smallest k with compliance>=0.95 on 50q x 4 versions', 'compliance': comp},
                        C.fewshot_k_path(a.model))
    out['seconds'] = time.time() - t0
    C.save_json(out, os.path.join(C.run_dir(a.model, a.fmt), 'smoke.json'))
    print('DONE', C.stamp(t0))


if __name__ == '__main__':
    main()
