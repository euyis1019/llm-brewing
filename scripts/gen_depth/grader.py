"""Answer extraction and grading for the gen_depth evaluation pool (data/pool.jsonl).

    extract_answer(text, answer_type, fallback=True) -> str | None
    grade(text, answer, answer_type, fallback=True) -> bool        (never raises, except KeyboardInterrupt)
    math_equal(pred, gold) -> bool                                  (never raises, except KeyboardInterrupt)

answer_type is 'math' (free-form answer) or 'mcq' (gold is a letter A-D).

Extraction
  math: content of the LAST complete, non-empty \\boxed{...} / \\fbox{...} (nested braces; nested boxes are
        unwrapped; an unterminated box, e.g. at the end of a truncated generation, is ignored). Boxes written as
        one list ('\\boxed{3}, \\boxed{5}, \\boxed{7}', 'x = \\boxed{2} or x = \\boxed{-1}': only an explicit
        separator , ; and or & \\quad between them) are joined into '3, 5, 7' (duplicates dropped), so a hedge
        '\\boxed{3} or \\boxed{4}' is graded as the list it is. A final box missing its closing brace is used
        when the text ends with a closing math delimiter ('\\boxed{13\\n$$'). If there is no box and
        fallback=True: the short math-like statement after the last 'answer is' / 'answer:' / 'Final Answer' cue
        (text before an unterminated box only; a sentence is rejected). There is no last-number fallback: in a
        generation without a box or answer cue (in practice a truncated one) the last number is not an answer.
  mcq:  the letter of the last box whose content is a choice letter: 'B', '(B)', '\\text{B}', 'B) <text>',
        'B. <text>', 'B - <text>', 'Option B'; a lowercase letter only alone ('b', '(b)'); not 'B = ...' /
        'a = 2' (a variable). A last lettered box that hedges ('A or C') gives no answer. A final box missing its
        closing brace at the very end of the text ('\\boxed{A)\\n$$') counts. Else (fallback=True) the last
        explicit cue: 'Answer: X', 'answer is (X)', 'correct option is X', 'option X is correct', 'X is the
        correct answer', 'choose / pick / select X', 'conclude X.' (not followed by 'or Y'); else a letter that
        is the very last token of the text ('... it is (C).', not '... while option B'). A letter merely
        mentioned in the text ('compound A', 'C(=O)', 'Option B gives ...') is not an answer.
Grading (math), after rewriting plain scientific notation 4.5e33 -> 4.5\\times 10^{33} (math-verify's LaTeX parser
  would read the e as Euler's number), a few unicode symbols (-, x, sqrt, pi, 1/2, ...) to LaTeX and dropping
  markdown bold around the answer:
  1. normalized strings equal (\\text{} unwrapped, \\dfrac->\\frac, \\left/\\right, spacing, degrees, %, $,
     x_{1}->x_1) -> True
  1b. gold is a choice letter like \\text{(C)} (MATH): the leading letter of the prediction decides
     ('C', '(C)', '\\textbf{(C)} <text>').
  2. gold is a plain number (int, decimal, a/b, \\frac{a}{b}, 1.2e5, 1.2\\times10^{5}, thousands separators; trailing
     \\text{} / \\mathrm{} units (also m^{-2}, km/s, \\cdot), %, degrees, ordinal suffixes ignored): the
     prediction's numeric value (plain-number parse incl. number words 'one hundred' and repeating decimals
     0.\\overline{3}, else math-verify parse, taking the right-hand side of an equation and the element of a
     one-element set) decides alone. A prediction with digits on both sides of a \\text{} or a space
     ('5 \\text{ or } 7', '1 \\text{ h } 30 \\text{ min}') is not a plain number (no digit concatenation).
     Tolerance by gold form: integer -> exact; fraction -> relative 5e-6 (accepts a 6-significant-digit
     decimal); decimal / scientific notation -> relative 1e-4 OR agreement to the precision the gold is written
     in (|pred - gold| < half a unit in the gold's last digit: 1.57 and 1.63 match 1.6, 3.275e12 matches 3.3e12,
     0.1921 matches 0.192; Minerva golds are rounded values, e.g. '3.3e12' for 3.275e12).
     A prediction written with % also matches value/100.
  3. otherwise math-verify parse('\\boxed{gold}') / parse('\\boxed{pred}') + verify(allow_set_relation_comp=True,
     so x \\in [1,2] matches 1 \\le x \\le 2). parse() runs with extraction_mode='first_match': a box whose
     LaTeX does not parse stays an unparsed string (graded by string comparison) instead of being replaced by a
     parsed fragment of it (the default made '\\frac{1}{3} E_{1}+\\frac{2}{3} E_{2}' equal to 2/3 and accepted
     a prediction 2/3); a subscripted E (E_0, E_{1}), which latex2sympy cannot parse, is renamed to \\Xi on both
     sides. Trailing \\text{}/\\mbox{} units are dropped first, and the gold
     is parsed without math-verify's unit stripping (it turns '\\frac{37}{4} m' into 37/4); the prediction is
     also tried with unit stripping when the gold has no free symbols. A True verdict between two
     symbol-free numeric values is re-checked with the relative tolerances above (verify() rounds floats to 6
     decimals, which makes all tiny numbers equal: 2.88e-19 == 1.3e-27). If verify says False it is retried
     (a) with canonical symbols when the two parses disagree on a symbol's assumptions or contain i (latex2sympy
     builds 'i' in '2i\\sqrt{3}' and in '2\\sqrt{3} i' as different symbols; i is the imaginary unit), and
     (b) for answers with arbitrary constants (C, c, C_1 .. C_4 on both sides), under every renaming of the
     prediction's constants onto the gold's (C_1 \\cos 2t + C_2 \\sin 2t == C_1 \\sin 2t + C_2 \\cos 2t).
  4. exact form + approximation: if the last box is a decimal (37.7) that is the rounding of the value of the
     box right before it (12\\pi, within 400 characters), that earlier exact box is graded too.
Timeouts: each math-verify / sympy step runs under a TIMEOUT_S-second limit: SIGALRM (setitimer) in the main
  thread (a pending caller alarm is re-armed afterwards), else a daemon worker thread joined with a timeout
  (a timed-out worker is abandoned). A timeout counts as "not equal".
Importing this module has no side effects. math_verify is imported on first use (GD/pylib is put on sys.path if it
  is not importable); its "timeout disabled" warning flags are then set, since timeouts are handled here.
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import collections
import functools
import itertools
import math
import os
import random
import re
import signal
import sys
import threading
import time

__all__ = ['extract_answer', 'grade', 'math_equal', 'TIMEOUT_S']

TIMEOUT_S = 3                 # seconds per guarded math-verify / sympy step
_MAX_BOX_CONTENT = 4000       # chars; a longer box content is treated as unterminated (degenerate output)
_PYLIB = os.path.join(str(_experiment_root('gen_depth')), 'pylib')

# ----------------------------------------------------------------------------------------------- boxes
_BOX_START = re.compile(r'\\(?:boxed|fbox)\s*\{')


def _match_brace(text, i, limit):
    """text[i-1] is an opening brace; return the index of its matching closing brace or -1.
    Escaped characters (\\{ \\} \\\\) are skipped; the search stops after `limit` characters."""
    depth = 1
    n = min(len(text), i + limit)
    j = i
    while j < n:
        c = text[j]
        if c == '\\':
            j += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


def _box_spans(text):
    """(start, end, content) of all complete \\boxed{...} / \\fbox{...}, ordered by start position
    (end = index after the closing brace; a box nested in another one comes after it)."""
    out = []
    for m in _BOX_START.finditer(text):
        j = _match_brace(text, m.end(), _MAX_BOX_CONTENT)
        if j >= 0:
            out.append((m.start(), j + 1, text[m.end():j]))
    return out


def _boxes(text):
    """Contents of all complete \\boxed{...} / \\fbox{...}, ordered by start position."""
    return [c for _, _, c in _box_spans(text)]


def _unbox(s):
    """Remove nested \\boxed{...} wrappers inside a box content."""
    for _ in range(20):
        m = _BOX_START.search(s)
        if not m:
            break
        j = _match_brace(s, m.end(), len(s))
        if j < 0:
            break
        s = s[:m.start()] + s[m.end():j] + s[j + 1:]
    return s.strip()


def _cut_unterminated(text, spans=None):
    """The text before an unterminated box that follows the last complete box (a generation cut off mid-box);
    fallback cues are only looked for there."""
    spans = _box_spans(text) if spans is None else spans
    m = _BOX_START.search(text, max((e for _, e, _ in spans), default=0))
    return text[:m.start()] if m else text


def _open_box_tail(text, spans):
    """(content, closed) of an unterminated box at the very end of the text (at most 200 characters after
    '\\boxed{'), with trailing math delimiters / whitespace removed; closed: the text ends with a closing math
    delimiter ($$, $, \\], \\)), i.e. the model finished the formula and only forgot the closing brace
    ('\\boxed{A)\\n$$'). (None, False) if there is none."""
    m = _BOX_START.search(text, max((e for _, e, _ in spans), default=0))
    if not m or len(text) - m.end() > 200:
        return None, False
    rest = text[m.end():]
    closed = re.search(r'(?:\$|\\\]|\\\))\s*$', rest) is not None
    c = re.sub(r'(?:\s|\$|\\\]|\\\))+$', '', rest).strip()
    return (c or None), closed


def _balanced(s):
    depth = 0
    i = 0
    while i < len(s):
        if s[i] == '\\':
            i += 2
            continue
        depth += {'{': 1, '}': -1}.get(s[i], 0)
        if depth < 0:
            return False
        i += 1
    return depth == 0


# ------------------------------------------------------------------------------------- math extraction
_SEP = (r'(?:,|;|&|\\quad|\\qquad|\b(?i:and|or)\b|\\(?:text|textrm|mathrm|mbox)\s*\{\s*(?:,|;|(?i:and|or))?\s*\})')
_FILL = r'(?:\s|\$|\\[,;:! ]|~|\\\\|\\[()\[\]])*'
_VAR = r'(?:[A-Za-z](?:_\{?\w+\}?)?\s*=\s*)?'
_JOIN_GAP = re.compile(_FILL + _SEP + r'(?:' + _FILL + _SEP + r')*' + _FILL + _VAR + _FILL)
_ANS_CUE = re.compile(r'(?i:final\s+answer|answer)\s*(?:\*\*)?\s*(?:(?i:is)\b\s*:?|:)\s*'
                      r'|(?i:final\s+answer)\s*(?:\*\*)?\s*(?=\n)')
_MATH_DELIMS = (re.compile(r'^\$\$(.+?)\$\$', re.S), re.compile(r'^\$(.+?)\$'),
                re.compile(r'^\\\((.+?)\\\)'), re.compile(r'^\\\[(.+?)\\\]', re.S))


def _plausible_answer(s):
    """A short math-like statement, not a sentence ('18 dollars', 'x = 5', '\\frac{1}{2}'; not 'not 5, let me
    recompute' or a whole paragraph). Guards against math-verify's lenient extraction of any number in text."""
    if not s or len(s) > 80 or _BOX_START.search(s):
        return False
    bare = _TEXT_WRAP.sub(' ', s)
    words = re.findall(r'(?<![\\A-Za-z])[A-Za-z]{2,}', bare)
    return len(words) <= 2 and not (words and re.match(r'\s*[A-Za-z]{2,}', bare))   # not a sentence


def _after_answer_cue(text):
    last = None
    for m in _ANS_CUE.finditer(text):
        last = m
    if last is None:
        return None
    rest = re.sub(r'^[\s*#:]+', '', text[last.end():])
    for rx in _MATH_DELIMS:
        m = rx.match(rest)
        if m:
            s = m.group(1).strip()
            return s if _plausible_answer(s) else None
    rest = rest.split('\n', 1)[0].strip().strip('*').strip()
    if rest[:1] == '$':
        rest = rest.strip('$').strip()
    else:
        rest = re.split(r'(?<!\d)\.(?!\d)|\.(?=\s|$)', rest)[0]
        rest = rest.strip().strip('*').strip()
    return rest if _plausible_answer(rest) else None


def _approx_partner(text, spans):
    """The box right before a final decimal approximation of its value ('\\boxed{12\\pi} ... approximately
    \\boxed{37.7}'): returned when the last box is a decimal that is the rounding of that box's value."""
    if len(spans) < 2:
        return None
    ls, _, lc = spans[-1]
    ps, pe, pc = spans[-2]
    if pe > ls or ls - pe > 400:
        return None
    fl = _to_float(_pre(_unbox(lc)))
    if fl is None or fl.kind != 'approx' or fl.pct:
        return None
    prev = _unbox(pc)
    p = _pre(prev)
    fp = _to_float(p)
    if fp is not None:
        if fp.kind == 'approx' or fp.pct:
            return None
        vp = fp.value
    else:
        vp = _call_with_timeout(lambda: _mv_value(p), TIMEOUT_S, None)
    if vp is None:
        return None
    if abs(vp - fl.value) <= fl.hulp * (1 + 1e-9) + 1e-12 * abs(fl.value):
        return prev
    return None


def _math_final(text, fallback):
    """(answer, spans): the extracted math answer (see the module docstring) and the non-empty box spans it
    came from (None for a joined list or the cue fallback); grade() uses the spans for rule 4."""
    all_spans = _box_spans(text)
    tail, closed = _open_box_tail(text, all_spans)
    if tail is not None and closed and _balanced(tail) and len(tail) <= 120:
        return _unbox(tail), None                 # final box missing its closing brace: '\\boxed{13\\n$$'
    spans = [s for s in all_spans if _unbox(s[2])]
    if spans:
        k = len(spans) - 1
        start = spans[k][0]
        parts = [_unbox(spans[k][2])]
        while k > 0:
            ps, pe, pc = spans[k - 1]
            if pe > start:                                # an enclosing box
                break
            gap = text[pe:start]
            if len(gap) > 48 or not _JOIN_GAP.fullmatch(gap):
                break
            parts.insert(0, _unbox(pc))
            start = ps
            k -= 1
        if len(parts) > 1:
            seen, uniq = set(), []
            for p in parts:
                key = _norm_str(_pre(p))
                if key not in seen:
                    seen.add(key)
                    uniq.append(p)
            vals = [_to_float(_pre(p)) for p in uniq]
            if all(v is not None and abs(v.value - vals[0].value) <= 1e-12 * max(1.0, abs(v.value))
                   for v in vals):
                return uniq[-1], None                     # one value written twice: '\\frac{1}{2} or 0.5'
            return ', '.join(uniq), None
        return parts[0], spans
    if not fallback:
        return None, None
    return _after_answer_cue(_cut_unterminated(text)), None


# -------------------------------------------------------------------------------------- mcq extraction
_TEXT_WRAP = re.compile(r'\\(?:text|textbf|textit|textrm|textsf|texttt|mathrm|mathbf|mathit|mbox)\s*\{([^{}]*)\}')
_LETTER_OP = re.compile(r'[=<>+\-*/^_|]|\\(?:cdot|times|approx|leq?|geq?|neq?|in|pm|equiv|sim|to)(?![A-Za-z])')
_LETTER_HEDGE = re.compile(r'\s*(?:,|/|&|\b(?i:and|or)\b)\s*\(?\s*[A-Ha-h]\s*\)?(?![A-Za-z0-9])')
_MCQ_PREFIX = re.compile(r'^(?i:(?:the\s+)?(?:final\s+|correct\s+|best\s+)?(?:answer|option|choice)(?:\s+is)?)'
                         r'\s*[:\uff1a]?\s*')


_HEDGE = 'hedge'


def _lead_letter(c, letters):
    """Leading choice letter of a box content (see the module docstring); _HEDGE for 'A or C'; else None."""
    m = re.match(r'\(\s*([A-Za-z])\s*\)', c)
    if m:
        L, rest = m.group(1), c[m.end():]
        if L.upper() not in letters:
            return None
        return _HEDGE if _LETTER_HEDGE.match(rest) else L.upper()
    m = re.match(r'([A-Za-z])', c)
    if not m or m.group(1).upper() not in letters:
        return None
    L, rest = m.group(1), c[1:]
    r = rest.strip()
    if L.islower():
        return L.upper() if r in ('', '.', ')', ').') else None
    if not r:
        return L
    if _LETTER_HEDGE.match(rest):
        return _HEDGE
    if r[0] in ').:,]' or re.match(r'(?:-|\u2013|\u2014)\s', r):   # 'B) ..', 'B. ..', 'B - <text>'
        return L
    if not rest[:1].isspace() or _LETTER_OP.match(r):        # 'Br', 'B_1', 'B^2', 'B = 3'
        return None
    return L


def _box_letter(content, letters='ABCD'):
    c = _TEXT_WRAP.sub(r'\1', _unbox(content))
    c = _TEXT_WRAP.sub(r'\1', c)
    for x in ('\\!', '\\,', '\\;', '\\:', '\\ ', '~', '\\quad', '\\qquad'):
        c = c.replace(x, ' ')
    c = c.replace('$', '').strip().strip('*').strip()
    c = _MCQ_PREFIX.sub('', c).strip().strip('*').strip()
    return _lead_letter(c, letters)


def _letter_from_box(content, letters='ABCD'):
    """The choice letter of a box content, or None (also for a hedge 'A or C')."""
    L = _box_letter(content, letters)
    return None if L == _HEDGE else L


_CHOICE_GOLD = re.compile(r'\(([A-H])\)')                       # MATH golds like \text{(C)}
_NOHEDGE = r'(?!\s*(?:,|/|&|\b(?i:or|and)\b)\s*\(?[A-D]\b)'           # not 'A or B'
_L = r'\(?\s*([A-D])\s*\)?(?![A-Za-z0-9])' + _NOHEDGE
_MCQ_CUES = tuple(re.compile(p) for p in (
    r'(?i:answer)\s*(?:\*\*)?\s*[:\uff1a]\s*(?:\*\*)?\s*\$?\s*(?:\\text(?:bf)?\{\s*)?(?i:option\s+|choice\s+)?' + _L,
    r'(?i:answer\s+is)\s*:?\s*(?:\*\*)?\s*\$?\s*(?:\\text(?:bf)?\{\s*)?(?i:option\s+|choice\s+)?' + _L,
    r'(?i:(?:correct|right|best|final)\s+(?:option|choice)(?:\s+is|\s+would\s+be)?)\s*:?\s*(?:\*\*)?\s*' + _L,
    r'(?i:option|choice)\s+\(?([A-D])\)?\s*(?:\*\*)?\s*(?i:is\s+(?:the\s+)?(?:correct|right|best|answer))',
    r'(?<![A-Za-z0-9])\(?([A-D])\)?\s*(?:\*\*)?\s*(?i:is\s+(?:the\s+)?(?:correct|right|best)\s+(?:answer|option|choice))',
    r'(?i:\b(?:choose|pick|select|go\s+with)\s+(?:option\s+|choice\s+)?)' + _L
    + r'(?!\s+(?i:is|are|was|would|has|cannot|can|does|will|must|gives|leads)\b)',
    r'(?i:\bconclude\s+(?:that\s+)?(?:the\s+(?:correct\s+)?answer\s+is\s+)?(?:option\s+|choice\s+)?)'
    r'\(?\s*([A-D])\s*\)?(?=\s*(?:[.!]|$))',
))
_NOT_ANSWER_WORDS = frozenset('option options choice choices and or vs versus than compound compounds product products '
                              'structure molecule reagent reagents step part statement point between from to with of in '
                              'for on at by like if whether only both'.split())
_END_LETTER = re.compile(r'(?<![A-Za-z0-9_\\])\(?\s*(?:\*\*)?\s*\$?\s*(?:\\text(?:bf)?\{\s*)?\(?([A-D])\)?\s*\}?\s*\$?'
                         r'\s*(?:\*\*)?\s*\)?[\s.!*$]*$')


def _mcq_extract(text, fallback):
    spans = _box_spans(text)
    tail, _ = _open_box_tail(text, spans)
    contents = [c for _, _, c in spans] + ([tail] if tail is not None and len(tail) <= 120 else [])
    for content in reversed(contents):           # a final box missing its closing brace counts: '\\boxed{A)\\n$$'
        letter = _box_letter(content)
        if letter == _HEDGE:
            return None                           # the last lettered box is a hedge: no answer
        if letter:
            return letter
    if not fallback:
        return None
    t = _cut_unterminated(text, spans)
    best = None
    for rx in _MCQ_CUES:
        for m in rx.finditer(t):
            if best is None or m.start(1) > best.start(1):
                best = m
    if best is not None:
        return best.group(1)
    tail = t[-80:]
    m = _END_LETTER.search(tail)
    if not m:
        return None
    before = re.findall(r'[A-Za-z]+', tail[:m.start()].rsplit('\n', 1)[-1])
    if before and before[-1].lower() in _NOT_ANSWER_WORDS:
        return None                                  # '... while option B' (cut off), 'compared to B'
    return m.group(1)


def extract_answer(text, answer_type, fallback=True):
    """Extract the final answer string (math) or letter (mcq) from a generation; None if nothing found."""
    try:
        if not isinstance(text, str) or not text:
            return None
        if answer_type == 'mcq':
            return _mcq_extract(text, fallback)
        return _math_final(text, fallback)[0]
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return None


# --------------------------------------------------------------------------------------- normalization
_SCI = re.compile(r'(?<![\w.\\])([+-]?\d+(?:\.\d+)?)[eE]([+-]?\d+)(?![\w.])')
_DEGREE = re.compile(r'\^\s*\{\s*\\circ\s*\}|\^\s*\\circ|\\circ|\u00b0')
_SPACING = ('\\!', '\\,', '\\;', '\\:', '\\ ', '~')
_APPROX = re.compile(r'\d\.\d|\.\d|\d\.(?!\d)|\\times\s*10\^')
_UNICODE = (('\u2212', '-'), ('\u00d7', '\\times '), ('\u22c5', '\\cdot '), ('\u00b7', '\\cdot '),
            ('\u221a', '\\sqrt'), ('\u03c0', '\\pi '), ('\u221e', '\\infty '), ('\u00bd', '\\frac{1}{2}'),
            ('\u00bc', '\\frac{1}{4}'), ('\u00be', '\\frac{3}{4}'), ('\u00b2', '^{2}'), ('\u00b3', '^{3}'),
            ('\u2264', '\\le '), ('\u2265', '\\ge '), ('\u2260', '\\ne '), ('\u00b1', '\\pm '),
            ('\u2248', '\\approx '))


def _pre(s):
    """Light normalization applied before every comparison."""
    s = s.strip()
    for a, b in _UNICODE:
        if a in s:
            s = s.replace(a, b)
    for _ in range(2):
        m = re.fullmatch(r'\*\*(.+)\*\*', s, re.S)
        if m:
            s = m.group(1).strip()
    if len(s) >= 2 and s[0] == '$' and s[-1] == '$':
        s = s.strip('$').strip()
    s = s.rstrip('.').strip()
    s = re.sub(r'(\d)\\,(?=\d{3}(?!\d))', r'\1', s)          # 1\,000 -> 1000
    return _SCI.sub(lambda m: '%s\\times 10^{%d}' % (m.group(1), int(m.group(2))), s)


def _norm_str(s):
    t = s.replace('$', '')
    t = _TEXT_WRAP.sub(r'\1', t)
    t = t.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    t = t.replace('\\left', '').replace('\\right', '')
    for x in _SPACING:
        t = t.replace(x, '')
    t = _DEGREE.sub('', t).replace('\\%', '').replace('%', '')
    t = re.sub(r'\s+', '', t)
    t = re.sub(r'([_^])\{(\w)\}', r'\1\2', t)                 # x_{1} -> x_1, x^{2} -> x^2
    return t.rstrip('.')


_NUM = r'[+-]?(?:\d+\.?\d*|\.\d+)'
_ORDINAL = re.compile(r'(?<=\d)\s*(?:\^\s*\{\s*(?:\\(?:text|mathrm)\s*\{\s*)?(?:st|nd|rd|th)\s*\}?\s*\}'
                      r'|\\(?:text|mathrm)\s*\{\s*(?:st|nd|rd|th)\s*\}|(?:st|nd|rd|th)\b)')
_UNIT_BLOCK = (r'\\(?:text|textrm|textup|mathrm|mbox|operatorname|rm)\s*\{(?:[^{}]|\{[^{}]*\})*\}'
               r'(?:\s*\^\s*(?:\{[^{}]*\}|-?\d))?')
_UNIT_TAIL = re.compile(r'(?:\s|~|\\[,;:! ]|\\cdot|\\times|/|\\quad)*' + _UNIT_BLOCK + r'\s*$')
_UNIT_HEAD = re.compile(r'^\s*' + _UNIT_BLOCK)
_HAS_TEXT = re.compile(r'\\(?:text|textrm|textup|mathrm|mbox|operatorname)\b')
_WORD_NUM = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen '
                     'sixteen seventeen eighteen nineteen'.split(), range(20)))
_WORD_NUM.update(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
_WORD_SCALE = {'thousand': 10 ** 3, 'million': 10 ** 6, 'billion': 10 ** 9}

_Num = collections.namedtuple('_Num', 'value kind pct hulp')   # hulp: half a unit in the last written digit


def _words_to_int(t):
    """'one hundred twenty-three' -> 123; None unless every word is a number word."""
    ws = [w for w in re.split(r'[\s-]+', t.lower().strip()) if w]
    if not ws or any(w not in _WORD_NUM and w not in _WORD_SCALE and w not in ('hundred', 'and') for w in ws):
        return None
    if all(w == 'and' for w in ws):
        return None
    total = cur = 0
    for w in ws:
        if w in _WORD_NUM:
            cur += _WORD_NUM[w]
        elif w == 'hundred':
            cur = max(cur, 1) * 100
        elif w in _WORD_SCALE:
            total += max(cur, 1) * _WORD_SCALE[w]
            cur = 0
    return total + cur


def _decimals(num_str):
    return len(num_str.split('.', 1)[1]) if '.' in num_str else 0


def _to_float(s):
    """_Num(value, kind, pct, hulp) if s is a plain number, else None. kind: 'int' | 'frac' | 'approx' (written
    with a decimal point or in scientific notation); pct: written with %; hulp: half a unit in the last written
    digit for 'approx' numbers (0.05 for 1.6, 0.05e12 for 3.3e12), else 0. Trailing \\text{} / \\mathrm{} units
    and degrees are ignored; digits are never concatenated across a \\text{} or a space."""
    w = _TEXT_WRAP.sub(r'\1', s).strip().rstrip('.').strip()
    if re.fullmatch(r'[A-Za-z][A-Za-z\s-]*', w):
        n = _words_to_int(w)
        return _Num(float(n), 'int', False, 0.0) if n is not None else None
    t = _ORDINAL.sub('', s)                                                # 12^{\text{th}} -> 12
    for _ in range(8):                                                     # trailing units
        u = _UNIT_TAIL.sub('', t)
        if u == t or not u.strip():
            break
        t = u
    t = _UNIT_HEAD.sub(' ', t)
    if _HAS_TEXT.search(t):
        return None
    t = t.replace('\\$', '').replace('$', '').replace('{,}', ',')
    for x in _SPACING:
        t = t.replace(x, ' ')
    pct = '%' in t
    t = _DEGREE.sub('', t).replace('\\%', '').replace('%', '')
    t = t.replace('\\left', '').replace('\\right', '').replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    t = t.strip().rstrip('.').strip()
    if re.search(r'\d\s+[\d.]', t):
        if not re.fullmatch(r'[+-]?\d{1,3}(?:\s\d{3})+(?:\.\d+)?', t):         # '1 000' only
            return None
    t = re.sub(r'\s+', '', t)
    if not t:
        return None
    if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', t):
        t = t.replace(',', '')
    hulp = 0.0
    try:
        m = re.fullmatch(r'(%s)' % _NUM, t)
        if m:
            v = float(m.group(1))
            if '.' in t:
                kind, hulp = 'approx', 0.5 * 10.0 ** -_decimals(t)
            else:
                kind = 'int'
        else:
            m = re.fullmatch(r'([+-]?)\\frac\{(%s)\}\{(%s)\}' % (_NUM, _NUM), t) or \
                re.fullmatch(r'([+-]?)\(?(%s)\)?/\(?(%s)\)?' % (_NUM, _NUM), t)
            if m:
                v, kind = float(m.group(2)) / float(m.group(3)), ('approx' if '.' in t else 'frac')
                if m.group(1) == '-':
                    v = -v
            else:
                m = re.fullmatch(r'(%s)(?:[eE]|(?:\\times|\\cdot|\*|x)10\^)\{?([+-]?\d+)\}?' % _NUM, t)
                if m:
                    e = int(m.group(2))
                    v, kind = float(m.group(1)) * 10.0 ** e, 'approx'
                    hulp = 0.5 * 10.0 ** (e - _decimals(m.group(1)))
                else:
                    m = re.fullmatch(r'([+-]?)10\^\{?([+-]?\d+)\}?', t)
                    if m:
                        v = 10.0 ** int(m.group(2))
                        kind = 'int' if int(m.group(2)) >= 0 else 'frac'
                        if m.group(1) == '-':
                            v = -v
                    else:
                        m = re.fullmatch(r'([+-]?)(\d*)\.(\d*)\\overline\{(\d+)\}', t)   # 0.1\overline{36}
                        if not m:
                            return None
                        a, b, r = m.group(2) or '0', m.group(3), m.group(4)
                        v = int(a) + (int(b or '0') + int(r) / (10 ** len(r) - 1)) / 10 ** len(b)
                        kind = 'frac'
                        if m.group(1) == '-':
                            v = -v
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if kind == 'int' and v != int(v):
        kind = 'frac'
    return _Num(v, kind, pct, hulp) if math.isfinite(v) else None


_REL_TOL = {'int': 1e-12, 'frac': 5e-6, 'approx': 1e-4}


def _close(p, g, kind, hulp=0.0):
    if kind == 'int':
        return abs(p - g) <= 1e-12 * max(1.0, abs(g))
    tol = _REL_TOL[kind] * abs(g)
    if kind == 'approx':
        tol = max(tol, hulp * (1 - 1e-9))              # agrees to the precision the gold is written in
    if g == 0:
        return abs(p) <= max(1e-12, tol)
    return abs(p - g) <= tol


def _num_match(pv, ppct, g):
    """Prediction value pv (written with % if ppct) vs gold _Num; x% may also mean x/100."""
    golds = [(g.value, g.kind, g.hulp)]
    if g.pct:
        golds.append((g.value / 100.0, 'frac' if g.kind == 'int' else g.kind, g.hulp / 100.0))
    preds = [pv, pv / 100.0] if ppct else [pv]
    return any(_close(c, v, k, h) for c in preds for v, k, h in golds)


# ------------------------------------------------------------------------------------------- timeouts
class _Timeout(BaseException):
    pass


def _call_with_timeout(fn, seconds, default):
    """fn() with a time limit; `default` on timeout or on any exception."""
    if threading.current_thread() is threading.main_thread():
        state = {'armed': True}

        def handler(signum, frame):
            if state['armed']:
                raise _Timeout()

        try:
            old_handler = signal.signal(signal.SIGALRM, handler)
        except (ValueError, OSError, AttributeError):
            old_handler = None
        if old_handler is not None:
            t0 = time.monotonic()
            old_delay, old_interval = signal.setitimer(signal.ITIMER_REAL, seconds, 0.2)
            try:
                try:
                    return fn()
                finally:
                    state['armed'] = False
                    signal.setitimer(signal.ITIMER_REAL, 0)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                return default
            finally:
                signal.signal(signal.SIGALRM, old_handler)
                if old_delay > 0:                     # re-arm the caller's pending alarm
                    signal.setitimer(signal.ITIMER_REAL, max(1e-3, old_delay - (time.monotonic() - t0)),
                                     old_interval)
    box = []

    def run():
        try:
            box.append(fn())
        except BaseException:
            pass

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    return box[0] if box else default


# ---------------------------------------------------------------------------------------- math-verify
_MV = None


def _mv():
    global _MV
    if _MV is None:
        try:
            import math_verify  # noqa: F401
        except ImportError:
            if _PYLIB not in sys.path:
                sys.path.insert(0, _PYLIB)
        import math_verify.grader as mg
        import math_verify.parser as mp
        mg.TIMEOUT_WARNING_SHOWN = mp.TIMEOUT_WARNING_SHOWN = True   # timeouts are handled by this module
        _MV = (mp.parse, mg.verify)
    return _MV


_CFG = {}


def _cfg(units):
    """math-verify extraction config; units=True strips trailing unit words / letters (default math-verify
    behaviour), which corrupts answers ending in a one-letter symbol ('\frac{37}{4} m' -> 37/4)."""
    if units not in _CFG:
        _mv()
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, LatexNormalizationConfig
        norm = LatexNormalizationConfig(basic_latex=True, units=units, malformed_operators=True, nits=True,
                                        boxed='all', equations=False)
        _CFG[units] = (LatexExtractionConfig(normalization_config=norm), ExprExtractionConfig())
    return _CFG[units]


_E_SUB = re.compile(r'(?<![A-Za-z\\])E(?=\s*_)')


@functools.lru_cache(maxsize=8192)
def _mv_parse(s, units):
    """math-verify parse of '\\boxed{s}'. extraction_mode='first_match': if the whole box does not parse, the result
    is the unparsed string; the default 'any_match' then parses sub-matches of the text instead, a garbage value
    ('\\frac{1}{3} E_{1}+\\frac{2}{3} E_{2}' -> 2/3, so a prediction 2/3 would be graded correct). A subscripted E
    (E_0, E_{1}: latex2sympy reserves E for Euler's number and fails) is renamed to \\Xi on both sides."""
    parse, _ = _mv()
    s = _E_SUB.sub(r'\\Xi', s)
    return tuple(parse('\\boxed{' + s + '}', extraction_config=_cfg(units), parsing_timeout=None,
                       extraction_mode='first_match'))


_TRAIL_UNIT = re.compile(r'\s*(?:\\(?:text|textrm|mbox)\s*\{[^{}]*\}|\\mathrm\s*\{\s*~?[^{}]{2,}\})'
                         r'(?:\s*\^\s*(?:\{\s*-?\d+\s*\}|-?\d))?\s*$')


def _strip_text_units(s):
    """Drop trailing \\text{...} / \\mbox{...} / \\mathrm{..} unit blocks (unless nothing would remain)."""
    for _ in range(4):
        t = _TRAIL_UNIT.sub('', s)
        if t == s or not t.strip():
            break
        s = t
    return s.strip()


def _canon(e):
    """Symbols rebuilt without assumptions (latex2sympy gives the same name different assumptions depending on
    context, and sympy then treats them as different symbols); a free 'i' becomes the imaginary unit."""
    import sympy
    if not isinstance(e, sympy.Basic):
        return e
    try:
        rep = {s: (sympy.I if s.name == 'i' else sympy.Symbol(s.name))
               for s in e.free_symbols if isinstance(s, sympy.Symbol)}
        return e.xreplace(rep) if rep else e
    except _Timeout:
        raise
    except Exception:
        return e


def _sym_value(expr, take_rhs):
    """Float value of a parsed math-verify object, or None if it is not a single real number. take_rhs: also
    accept the right-hand side of an equation and the element of a one-element set (predictions)."""
    import sympy
    try:
        if isinstance(expr, str):
            return None
        if take_rhs and isinstance(expr, sympy.FiniteSet) and len(expr) == 1:
            expr = next(iter(expr))
        if isinstance(expr, sympy.core.relational.Relational):
            if not take_rhs:
                return None
            expr = expr.rhs
        expr = _canon(expr)
        if not isinstance(expr, sympy.Expr) or expr.free_symbols:
            return None
        z = complex(sympy.N(expr, 30))
    except _Timeout:
        raise
    except Exception:
        return None
    if not (math.isfinite(z.real) and math.isfinite(z.imag)) or abs(z.imag) > 1e-12 * max(1.0, abs(z.real)):
        return None
    return z.real


def _symbol_free(expr):
    import sympy
    try:
        return isinstance(expr, sympy.Basic) and not expr.free_symbols
    except _Timeout:
        raise
    except Exception:
        return False


def _mv_value(p):
    """Numeric value of a prediction (units stripped, right-hand side of an equation), or None."""
    pp = _mv_parse(p, True)
    return _sym_value(pp[0], take_rhs=True) if pp else None


def _free(e):
    import sympy
    try:
        return {s for s in e.free_symbols if isinstance(s, sympy.Symbol)} if isinstance(e, sympy.Basic) else set()
    except _Timeout:
        raise
    except Exception:
        return set()


_CONST_NAME = re.compile(r'^[cC](?:_?\{?\d\}?)?$')


def _num_close(a, b, syms, rng):
    """a == b at two random points (all free symbols real in [0.3, 1.7]); None if not numerically evaluable."""
    import sympy
    for _ in range(2):
        sub = {s: sympy.Float(rng.uniform(0.3, 1.7)) for s in syms}
        try:
            x, y = complex(a.evalf(subs=sub)), complex(b.evalf(subs=sub))
        except _Timeout:
            raise
        except Exception:
            return None
        if not abs(x - y) <= 1e-8 * max(1.0, abs(x), abs(y)):
            return False
    return True


def _const_perm_equal(gp, pp, verify):
    """verify() under the renamings of the prediction's arbitrary constants (C, c, C_1 ..) onto the gold's."""
    import sympy
    g0, p0 = gp[0], pp[0]
    if not isinstance(g0, sympy.Basic) or not isinstance(p0, sympy.Basic):
        return False
    gc = sorted((s for s in _free(g0) if _CONST_NAME.match(s.name)), key=lambda s: s.name)
    pc = sorted((s for s in _free(p0) if _CONST_NAME.match(s.name)), key=lambda s: s.name)
    if not gc or len(gc) != len(pc) or len(gc) > 4:
        return False
    ge = g0.rhs if isinstance(g0, sympy.core.relational.Relational) else g0
    pe = p0.rhs if isinstance(p0, sympy.core.relational.Relational) else p0
    syms = _free(ge) | _free(pe) | set(gc)
    rng = random.Random(0)
    for perm in itertools.permutations(gc):
        mapping = dict(zip(pc, perm))
        if all(a == b for a, b in mapping.items()):
            continue                                       # identity: already verified
        if isinstance(ge, sympy.Expr) and isinstance(pe, sympy.Expr):
            ok = _num_close(ge, pe.xreplace(mapping), syms, rng)
            if ok is False:
                continue
        pp2 = [x.xreplace(mapping) if isinstance(x, sympy.Basic) else x for x in pp]
        if verify(list(gp), pp2, allow_set_relation_comp=True, timeout_seconds=None):
            return True
    return False


def _mv_equal_impl(g, p):
    """math-verify equivalence. Gold: trailing \\text units dropped, parsed without unit stripping. Prediction: the
    same, plus (only for a symbol-free gold) a parse with math-verify's unit stripping ('3.5 cm' -> 3.5)."""
    _, verify = _mv()
    gp = _mv_parse(_strip_text_units(g), False)
    if not gp:
        return False
    p2 = _strip_text_units(p)
    cands = [_mv_parse(p2, False)]
    if _symbol_free(gp[0]):
        cands.append(_mv_parse(p2, True))
    vg = _sym_value(gp[0], False)
    kind = 'approx' if _APPROX.search(g) else 'frac'
    gfree = {s.name: s for s in _free(gp[0])}
    gpc = None
    for pp in cands:
        if not pp:
            continue
        if verify(list(gp), list(pp), allow_set_relation_comp=True, timeout_seconds=None):
            vp = _sym_value(pp[0], False)
            if vg is None or vp is None or _close(vp, vg, kind):
                return True
            continue
        pfree = {s.name: s for s in _free(pp[0])}
        clash = 'i' in gfree or 'i' in pfree or any(n in gfree and gfree[n] != s for n, s in pfree.items())
        ppc = [_canon(x) for x in pp]
        if clash:
            gpc = gpc or [_canon(x) for x in gp]
            if verify(gpc, ppc, allow_set_relation_comp=True, timeout_seconds=None):
                vp, vgc = _sym_value(ppc[0], False), _sym_value(gpc[0], False)
                if vgc is None or vp is None or _close(vp, vgc, kind):
                    return True
                continue
        if _const_perm_equal(gpc or [_canon(x) for x in gp], ppc, verify):
            return True
    return False


# ---------------------------------------------------------------------------------------------- grading
def math_equal(pred, gold, timeout=TIMEOUT_S):
    """Equivalence of an extracted answer and a gold answer (strings). Never raises."""
    try:
        p, g = _pre(str(pred)), _pre(str(gold))
        if not p or not g:
            return False
        if _norm_str(p) == _norm_str(g):
            return True
        mc = _CHOICE_GOLD.fullmatch(_norm_str(g))
        if mc:                                   # gold is a choice letter: compare the leading letter
            return _letter_from_box(p, 'ABCDEFGH') == mc.group(1)
        fg = _to_float(g)
        if fg is not None:                       # plain-number gold: the numeric value decides
            fp = _to_float(p)
            if fp is not None:
                return _num_match(fp.value, fp.pct, fg)
            pv = _call_with_timeout(lambda: _mv_value(p), timeout, None)
            return pv is not None and _num_match(pv, '%' in p, fg)
        return bool(_call_with_timeout(lambda: _mv_equal_impl(g, p), timeout, False))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return False


def grade(text, answer, answer_type, fallback=True):
    """True iff the generation `text` answers `answer` correctly. Never raises (except on Ctrl-C)."""
    try:
        if answer is None or not isinstance(text, str) or not text:
            return False
        gold = str(answer).strip()
        if not gold:
            return False
        if answer_type == 'mcq':
            pred = _mcq_extract(text, fallback)
            return pred is not None and pred.strip().upper() == gold.strip('() ').upper()
        pred, spans = _math_final(text, fallback)
        if pred is None:
            return False
        if math_equal(pred, gold):
            return True
        alt = _approx_partner(text, spans) if spans else None
        return alt is not None and math_equal(alt, gold)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return False
