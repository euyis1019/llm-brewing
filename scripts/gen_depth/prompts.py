"""Prompt construction for the gen_depth evaluation pool (data/pool.jsonl).

    user_text(item) -> str
    prompt_text(item, tok, family) -> str          rendered chat prompt (generation prompt included)
    prompt_ids(item, tok, family) -> list[int]     token ids of the same prompt

family: 'qwen3' (chat template with enable_thinking=False: the prompt ends with '<think>\\n\\n</think>\\n\\n') or
'qwen2.5' (Qwen2.5-Instruct template; it inserts its default system prompt). Importing has no side effects.
"""

__all__ = ['SUFFIX', 'user_text', 'prompt_text', 'prompt_ids']

SUFFIX = {
    'math': '\nPlease reason step by step, and put your final answer within \\boxed{}.',
    'mcq': '\nPlease reason step by step, and put the letter of your final answer (A, B, C, or D) within \\boxed{}.',
}
FAMILIES = ('qwen3', 'qwen2.5')


def user_text(item):
    """The user message: problem text (GPQA items already contain the A-D options) + answer-format instruction."""
    return item['problem'] + SUFFIX[item['answer_type']]


def _chat_kwargs(family):
    if family == 'qwen3':
        return {'enable_thinking': False}
    if family == 'qwen2.5':
        return {}
    raise ValueError(f'unknown family {family!r} (expected one of {FAMILIES})')


def _messages(item):
    return [{'role': 'user', 'content': user_text(item)}]


def prompt_text(item, tok, family):
    out = tok.apply_chat_template(_messages(item), add_generation_prompt=True, tokenize=False,
                                  **_chat_kwargs(family))
    if not isinstance(out, str):                 # some versions wrap single conversations in a list
        out = out[0]
    return out


def _as_int_list(out):
    if hasattr(out, 'keys') and 'input_ids' in out:   # dict / BatchEncoding (transformers 5 default)
        out = out['input_ids']
    if hasattr(out, 'tolist'):                       # tensor / ndarray
        out = out.tolist()
    out = list(out)
    if out and not isinstance(out[0], int):          # batched [[...]]
        assert len(out) == 1, 'expected a single conversation'
        out = list(out[0])
    return [int(t) for t in out]


def prompt_ids(item, tok, family):
    out = tok.apply_chat_template(_messages(item), add_generation_prompt=True, tokenize=True,
                                  **_chat_kwargs(family))
    return _as_int_list(out)
