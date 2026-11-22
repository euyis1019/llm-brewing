"""Extract per-block critical-position states and all fixed readouts for one model/format (fused with CSD).

Outputs in results/{model}__{fmt}/extract/:
  hidden_bf16.u16  (N, L, d) uint16 memmap of bf16 bits: output of block l (pre final norm), last prompt position
  lens4.npy        (N, L, 4) f32  logit lens at the 4 letter ids (final norm + unembedding in fp32)
  csd_raw4.npy     (N, L, 4) f32  raw Patchscope: block-l state injected at the target prompt's last position, blocks
                                   l+1..L-1 run there; letter logits. csd_raw4[:, L-1] == lens4[:, L-1] == real output
  csd_top1.npy     (N, L) i32     full-vocab argmax of that readout;  csd_mass.npy (N, L) f16 prob mass on the letters
  out_logits4.npy  (N, 4) f32     real output letter logits; out_top1.npy (N,) i32; out_mass.npy (N,) f32
  csd_base4.npy    (4,) f32       letter logits of the bare target prompt (baseline-subtracted readout = raw - base)
  rows.json, prompt_len.npy, meta.json
Usage: python scripts/extract.py --model qwen3-4b-base --fmt base_fewshot
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--fmt', required=True)
    ap.add_argument('--max_tokens', type=int, default=32000); ap.add_argument('--max_bs', type=int, default=64)
    ap.add_argument('--csd_chunk', type=int, default=160); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='extract')
    a = ap.parse_args()
    t0 = time.time()
    lm = C.LM(a.model)
    tok = lm.tok
    qs = C.load_questions()
    if a.limit:
        qs = qs[:a.limit]
    rows = C.build_rows(a.model, a.fmt, tok, qs)
    N, L, d = len(rows), lm.L, lm.d
    assert all(r['row'] == i for i, r in enumerate(rows))
    target = C.target_text(a.fmt, tok)
    ids_crit = lm.find_letter_ids(rows[0]['text'])
    assert ids_crit == lm.find_letter_ids(target), 'letter ids differ between prompt and CSD target'
    lm.set_letters(ids_crit)
    print(f'N={N} L={L} d={d} letters={ids_crit} target={target!r}', C.stamp(t0), flush=True)
    idl = [lm.encode(r['text']) for r in rows]
    plen = np.array([len(x) for x in idl], dtype=np.int32)
    print(f'tokenized: mean len {plen.mean():.1f} max {plen.max()} total {plen.sum()}', C.stamp(t0), flush=True)

    od = os.path.join(C.run_dir(a.model, a.fmt), a.out)
    os.makedirs(od, exist_ok=True)
    hid = np.memmap(os.path.join(od, 'hidden_bf16.u16'), dtype=np.uint16, mode='w+', shape=(N, L, d))
    lens4 = np.zeros((N, L, 4), np.float32); csd4 = np.zeros((N, L, 4), np.float32)
    ctop = np.zeros((N, L), np.int32); cmass = np.zeros((N, L), np.float16)
    otop = np.zeros(N, np.int32); omass = np.zeros(N, np.float32)
    done = np.zeros(N, bool)
    st = lm.csd_setup(target)
    np.save(os.path.join(od, 'csd_base4.npy'), st['base4'].cpu().numpy())

    buf_rows, buf_H = [], []

    def flush_csd():
        if not buf_rows:
            return
        rr = np.concatenate(buf_rows)
        Hb = torch.cat(buf_H)  # (n, L, d)
        n = Hb.shape[0]
        flat = Hb.reshape(n * L, d)
        lay = torch.arange(L, device=Hb.device).repeat(n)
        fin = lm.csd_states(flat, lay, target)
        csd4[rr] = lm.letters32(fin).reshape(n, L, 4).cpu().numpy()
        top, mass = lm.vocab_stats(fin)
        ctop[rr] = top.reshape(n, L).cpu().numpy(); cmass[rr] = mass.reshape(n, L).cpu().numpy()
        buf_rows.clear(); buf_H.clear()

    order = np.argsort(plen, kind='stable')
    i = 0; nb = 0; nbuf = 0
    while i < N:
        j = i
        while j < N and (j - i + 1) * plen[order[j]] <= a.max_tokens and j - i < a.max_bs:
            j += 1
        j = max(j, i + 1)
        b = order[i:j]
        ids, am, pos = lm.batch_tokens([idl[x] for x in b])
        H, _ = lm.forward_capture(ids, am, pos)  # (B, L, d) bf16
        hid[b] = H.view(torch.int16).cpu().numpy().view(np.uint16)
        lens4[b] = lm.letters32(H).cpu().numpy()
        top, mass = lm.vocab_stats(H[:, -1])
        otop[b] = top.cpu().numpy(); omass[b] = mass.cpu().numpy()
        buf_rows.append(b); buf_H.append(H); nbuf += len(b)
        if nbuf >= a.csd_chunk:
            flush_csd(); nbuf = 0
        done[b] = True
        i = j; nb += 1
        if nb % 50 == 0:
            el = time.time() - t0
            print(f'batch {nb} rows {done.sum()}/{N} T={ids.shape[1]} elapsed {el:.0f}s', flush=True)
    flush_csd()
    hid.flush()
    assert done.all()
    np.save(os.path.join(od, 'lens4.npy'), lens4); np.save(os.path.join(od, 'csd_raw4.npy'), csd4)
    np.save(os.path.join(od, 'csd_top1.npy'), ctop); np.save(os.path.join(od, 'csd_mass.npy'), cmass)
    np.save(os.path.join(od, 'out_logits4.npy'), lens4[:, -1].copy())
    np.save(os.path.join(od, 'out_top1.npy'), otop); np.save(os.path.join(od, 'out_mass.npy'), omass)
    np.save(os.path.join(od, 'prompt_len.npy'), plen)
    C.save_json([{k: r[k] for k in ('row', 'qid', 'v', 'source', 'fold', 'gold')} for r in rows], os.path.join(od, 'rows.json'))
    gold = np.array([r['gold'] for r in rows])
    comp = np.isin(otop, ids_crit)
    acc = (lens4[:, -1].argmax(-1) == gold)
    meta = dict(model=a.model, fmt=a.fmt, N=N, L=L, d=d, letter_ids=ids_crit, target=target, target_ids=st['ids'],
                target_base_top1=st['base_top1'], target_base_lettermass=st['base_mass'],
                compliance=float(comp.mean()), acc4=float(acc.mean()), acc4_compliant=float(acc[comp].mean()),
                csd_top_equals_out=float(np.abs(csd4[:, -1] - lens4[:, -1]).max()),
                prompt_len_mean=float(plen.mean()), prompt_len_max=int(plen.max()), seconds=time.time() - t0,
                script_md5=C.md5_file(__file__), common_md5=C.md5_file(C.__file__), args=vars(a))
    C.save_json(meta, os.path.join(od, 'meta.json'))
    print(json.dumps(meta, indent=1), flush=True)
    print('DONE', C.stamp(t0), flush=True)


if __name__ == '__main__':
    main()
