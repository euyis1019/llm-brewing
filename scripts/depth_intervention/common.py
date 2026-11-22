"""Shared core for the depth-intervention experiments (see ../PLAN.md).

Conventions
- Question q has canonical options q['options'][0..3] and correct index q['answer'].
- Version v (0..3) shows canonical option (i+v)%4 at letter i. Gold letter in version v = (answer-v)%4.
  Content shown at letter i in version v = (i+v)%4.
- Prompt rows are ordered row = 4*qid + v (N = 4*Q rows).
- Hidden states: output of decoder block l (l = 0..L-1, PRE final norm) at the critical (last prompt) position,
  stored as raw bf16 bits in a uint16 memmap of shape (N, L, d).
- Readouts at the four letter tokens are fp32 (final norm and unembedding computed in fp32 from the bf16 state).
- CSD / Patchscope: inject a block-l output state at the last position of a fixed target prompt and run blocks
  l+1..L-1 there (prefix keys/values of the target prompt are fixed). Raw logits are the main readout; the
  baseline-subtracted readout is raw minus the target prompt's own logits.
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import hashlib, json, os, time
import numpy as np
import torch

ROOT = str(_experiment_root('depth_intervention'))
PY = _sys.executable
MODELS = {
    'qwen3-4b-base': str(_model_root() / 'Qwen3-4B-Base'),
    'qwen3-4b': str(_model_root() / 'Qwen3-4B'),
    'llama3.1-8b': str(_model_root() / 'Meta-Llama-3.1-8B-nous'),
    'qwen2.5-7b-instruct': str(_model_root() / 'huggingface/Qwen/Qwen2.5-7B-Instruct'),
    'qwen2.5-7b': str(_model_root() / 'huggingface/Qwen/Qwen2.5-7B'),
}
FMTS = ('base_fewshot', 'chat0')
LETTERS = 'ABCD'
SOURCES = ('gsm_mc', 'math_mc', 'pythonio', 'mmlu_pro', 'ceval')
HEADER = {
    'gsm_mc': 'The following are multiple choice questions (with answers) about grade school math.\n\n',
    'math_mc': 'The following are multiple choice questions (with answers) about high school math.\n\n',
    'pythonio': 'The following are multiple choice questions (with answers) about Python program reasoning.\n\n',
    'mmlu_pro': 'The following are multiple choice questions (with answers).\n\n',
    'ceval': 'The following are multiple choice questions (with answers).\n\n',
}
CHAT_INSTR = 'Answer with only the letter (A, B, C, or D) of the correct choice.'
CHAT_SYS = ('You are an exam assistant. For each multiple-choice question, reply with only the letter '
            '(A, B, C, or D) of the correct choice. Do not explain.')
BASE_TARGET = 'Answer:\n'
# Benchmark selector: 'core' (data/bench/questions.jsonl, results/{model}__{fmt}) or e.g. 'ext'
# (data/bench/questions_ext.jsonl, results/{model}__{fmt}__ext). Set with env BREW_BENCH.
BENCH = os.environ.get('BREW_BENCH', 'core')


# ----------------------------------------------------------------------------- data
def load_questions():
    name = 'questions.jsonl' if BENCH == 'core' else f'questions_{BENCH}.jsonl'
    with open(os.path.join(ROOT, 'data/bench', name)) as f:
        return [json.loads(l) for l in f]


def load_demos():
    return json.load(open(os.path.join(ROOT, 'data/bench/demos.json')))


def shown(options, v):
    return [options[(i + v) % 4] for i in range(4)]


def gold_letter(answer, v):
    return (answer - v) % 4


def content_of(letter, v):
    """Canonical option index shown at `letter` in version v (works on numpy arrays)."""
    return (letter + v) % 4


def q_block(question, opts, with_answer_tag=True):
    s = f"Question:\n{question}\nChoices:\n" + ''.join(f"{L}. {o}\n" for L, o in zip(LETTERS, opts))
    return s + ('Answer:\n' if with_answer_tag else '')


def fewshot_k_path(model_key):
    return os.path.join(ROOT, f'data/bench/fewshot_k__{model_key}.json')


def prompt_text(q, v, fmt, tok=None, demos=None, k=0):
    if fmt == 'base_fewshot':
        s = HEADER[q['source']]
        for d in (demos or [])[:k]:
            s += q_block(d['question'], shown(d['options'], v)) + LETTERS[gold_letter(d['answer'], v)] + '\n\n'
        return s + q_block(q['question'], shown(q['options'], v))
    if fmt == 'chat0':
        user = q_block(q['question'], shown(q['options'], v), with_answer_tag=False) + '\n' + CHAT_INSTR
        return chat_wrap(tok, user)
    raise ValueError(fmt)


def chat_wrap(tok, user):
    """chat0 = system instruction + user turn, non-thinking mode (bare user turn: 2% compliance on GSM/MATH)."""
    msgs = [{'role': 'system', 'content': CHAT_SYS}, {'role': 'user', 'content': user}]
    kw = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(msgs, **kw)


def target_text(fmt, tok):
    """CSD target prompt: 'Answer:\\n' for base, the empty chat shell (instruction only) for chat."""
    if fmt == 'base_fewshot':
        return BASE_TARGET
    return chat_wrap(tok, CHAT_INSTR)


def build_rows(model_key, fmt, tok, qs=None, k_override=None):
    """All prompt rows (row = 4*qid + v)."""
    qs = qs if qs is not None else load_questions()
    demos = load_demos()
    kmap = {}
    if fmt == 'base_fewshot':
        if k_override is not None:
            kmap = {s: k_override for s in SOURCES}
        else:
            kmap = json.load(open(fewshot_k_path(model_key)))['k']
    rows = []
    for q in qs:
        for v in range(4):
            rows.append(dict(row=4 * q['qid'] + v, qid=q['qid'], v=v, source=q['source'], fold=q['fold'],
                             gold=gold_letter(q['answer'], v),
                             text=prompt_text(q, v, fmt, tok, demos.get(q['source']), kmap.get(q['source'], 0))))
    return rows


def run_dir(model_key, fmt):
    d = os.path.join(ROOT, 'results', f'{model_key}__{fmt}' + ('' if BENCH == 'core' else f'__{BENCH}'))
    os.makedirs(d, exist_ok=True)
    return d


def md5_file(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def save_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def bf16_bits_to_f32(a):
    """numpy uint16 array holding bf16 bits -> float32 array."""
    return (np.asarray(a).astype(np.uint32) << 16).view(np.float32)


def open_hidden(model_key, fmt, mode='r'):
    d = os.path.join(run_dir(model_key, fmt), 'extract')
    meta = json.load(open(os.path.join(d, 'meta.json')))
    return np.memmap(os.path.join(d, 'hidden_bf16.u16'), dtype=np.uint16, mode=mode,
                     shape=(meta['N'], meta['L'], meta['d'])), meta


# ----------------------------------------------------------------------------- model
class PrefixKV:
    """Duck-typed cache: returns cat(prefix_kv[layer], new_kv) without mutating anything."""

    def __init__(self, ks, vs):
        self.ks, self.vs = ks, vs

    def update(self, k, v, layer_idx, *args, **kwargs):
        pk, pv = self.ks[layer_idx], self.vs[layer_idx]
        if pk is None:
            return k, v
        if pk.shape[0] != k.shape[0]:
            pk = pk.expand(k.shape[0], -1, -1, -1)
            pv = pv.expand(k.shape[0], -1, -1, -1)
        return torch.cat([pk, k], 2), torch.cat([pv, v], 2)


class LM:
    def __init__(self, model_key, device='cuda', attn='sdpa'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.key = model_key
        path = MODELS[model_key]
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=attn)
        self.model.to(device).eval()
        self.device = device
        self.cfg = self.model.config
        self.layers = self.model.model.layers
        self.L = len(self.layers)
        self.d = self.cfg.hidden_size
        self.norm_w = self.model.model.norm.weight.detach().float()
        self.norm_eps = self.model.model.norm.variance_epsilon
        self.WU = self.model.lm_head.weight.detach().float()  # (V, d) fp32 copy
        self.letter_ids = None
        self._csd = {}

    # --- tokens
    def encode(self, text):
        return self.tok(text, add_special_tokens=True)['input_ids']

    def find_letter_ids(self, prefix_text):
        """Token id of each letter right after `prefix_text` (must be a single new token)."""
        base = self.encode(prefix_text)
        ids = []
        for L in LETTERS:
            full = self.encode(prefix_text + L)
            assert full[:len(base)] == base and len(full) == len(base) + 1, (L, full[-3:], base[-3:])
            ids.append(full[-1])
        return ids

    def set_letters(self, ids):
        self.letter_ids = list(ids)
        self.U4 = self.WU[self.letter_ids]  # (4, d)

    # --- readouts (fp32)
    def fnorm32(self, h):
        h = h.float()
        return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.norm_w

    def letters32(self, h):
        return self.fnorm32(h) @ self.U4.T

    def vocab_stats(self, h, chunk=1024):
        """For states h (n, d): (top1 id over full vocab, prob mass on the 4 letters)."""
        top, mass = [], []
        for i in range(0, h.shape[0], chunk):
            lg = self.fnorm32(h[i:i + chunk]) @ self.WU.T
            top.append(lg.argmax(-1))
            mass.append(torch.exp(torch.logsumexp(lg[:, self.letter_ids], -1) - torch.logsumexp(lg, -1)))
        return torch.cat(top), torch.cat(mass)

    # --- batching
    def batch_tokens(self, id_lists):
        """Left-pad. Returns input_ids, attention_mask, position_ids (positions count real tokens only)."""
        T = max(len(x) for x in id_lists)
        pad = self.tok.pad_token_id
        ids = torch.full((len(id_lists), T), pad, dtype=torch.long)
        am = torch.zeros((len(id_lists), T), dtype=torch.long)
        for i, x in enumerate(id_lists):
            ids[i, T - len(x):] = torch.tensor(x)
            am[i, T - len(x):] = 1
        pos = (am.cumsum(-1) - 1).clamp(min=0)
        return ids.to(self.device), am.to(self.device), pos.to(self.device)

    def rope(self, x, pos):
        return self.model.model.rotary_emb(x, pos)

    # --- forward with per-block capture at the last position
    @torch.no_grad()
    def forward_capture(self, ids, am, pos, use_cache=False):
        """Normal forward. Returns (states (B, L, d) bf16 block outputs at last position, cache or None)."""
        caps = [None] * self.L
        hooks = []
        for i, layer in enumerate(self.layers):
            def f(mod, args, out, i=i):
                o = out[0] if isinstance(out, tuple) else out
                caps[i] = o[:, -1].detach()
            hooks.append(layer.register_forward_hook(f))
        try:
            out = self.model.model(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=use_cache)
        finally:
            for h in hooks:
                h.remove()
        return torch.stack(caps, 1), (out.past_key_values if use_cache else None)

    @torch.no_grad()
    def forward_sequence(self, ids, am, pos, seq, beta=None, damp_steps=None):
        """All-position forward through an arbitrary block sequence `seq` (list of block indices).
        Steps in `damp_steps` are applied as h + beta*(block(h)-h). Returns last-position states (B, S, d)."""
        from transformers.masking_utils import create_causal_mask
        h = self.model.model.embed_tokens(ids)
        mask = create_causal_mask(config=self.cfg, inputs_embeds=h, attention_mask=am, past_key_values=None,
                                  position_ids=pos)
        pe = self.rope(h, pos)
        traj = []
        damp_steps = set(damp_steps or [])
        for s, j in enumerate(seq):
            o = self.layers[j](h, attention_mask=mask, position_ids=pos, position_embeddings=pe)
            o = o[0] if isinstance(o, tuple) else o
            h = h + beta * (o - h) if s in damp_steps else o
            traj.append(h[:, -1])
        return torch.stack(traj, 1)

    @torch.no_grad()
    def run_single(self, x, blocks, kv, pos, mask=None, beta=None, damp_steps=None, record=False):
        """Run a single-token stream x (B, d) through `blocks`, attending to fixed prefix keys/values `kv`
        (PrefixKV) plus itself, at position `pos` ((B,1) or (1,1)). mask: (B,1,1,P+1) bool or None."""
        h = x[:, None, :]
        pe = self.rope(h, pos)
        traj = []
        damp_steps = set(damp_steps or [])
        for s, j in enumerate(blocks):
            o = self.layers[j](h, attention_mask=mask, position_ids=pos, past_key_values=kv, position_embeddings=pe)
            o = o[0] if isinstance(o, tuple) else o
            h = h + beta * (o - h) if s in damp_steps else o
            if record:
                traj.append(h[:, 0])
        return (h[:, 0], torch.stack(traj, 1)) if record else h[:, 0]

    # --- CSD / Patchscope
    @torch.no_grad()
    def csd_setup(self, target):
        """Prepare the fixed target prompt. Returns dict with prefix kv, position, baseline letter logits."""
        if target in self._csd:
            return self._csd[target]
        ids = self.encode(target)
        x = torch.tensor([ids], device=self.device)
        if len(ids) > 1:
            out = self.model.model(input_ids=x[:, :-1], use_cache=True)
            cache = out.past_key_values
            ks = [cache.layers[i].keys for i in range(self.L)]
            vs = [cache.layers[i].values for i in range(self.L)]
        else:
            ks = vs = [None] * self.L
        states, _ = self.forward_capture(x, torch.ones_like(x), torch.arange(len(ids), device=self.device)[None])
        base4 = self.letters32(states[:, -1])[0]
        top, mass = self.vocab_stats(states[:, -1])
        st = dict(ids=ids, kv=PrefixKV(ks, vs), pos=torch.tensor([[len(ids) - 1]], device=self.device),
                  base4=base4, base_top1=int(top[0]), base_mass=float(mass[0]))
        self._csd[target] = st
        return st

    @torch.no_grad()
    def csd_states(self, H, layers, target):
        """H: (n, d) bf16 states, layers: (n,) long tensor/list of the block index each state is the output of.
        Returns final-block outputs (n, d) after running blocks layer+1..L-1 on the target prompt (staircase batch)."""
        st = self.csd_setup(target)
        layers = torch.as_tensor(layers, device=H.device)
        out = torch.empty_like(H)
        top = layers == self.L - 1
        out[top] = H[top]
        active_idx = torch.empty(0, dtype=torch.long, device=H.device)
        x = torch.empty((0, H.shape[1]), dtype=H.dtype, device=H.device)
        for j in range(1, self.L):
            new = (layers == j - 1).nonzero().flatten()
            if len(new):
                active_idx = torch.cat([active_idx, new])
                x = torch.cat([x, H[new]])
            if len(active_idx) == 0:
                continue
            x = self.run_single(x, [j], st['kv'], st['pos'])
        if len(active_idx):
            out[active_idx] = x
        return out


def stamp(t0):
    return f'{time.time() - t0:.1f}s'
