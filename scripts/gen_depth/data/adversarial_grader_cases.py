"""Adversarial checks of the evaluation layer: realistic model endings against scripts/grader.py, and a sanity
sample of data/pool.jsonl.

Usage (raw files are untrusted data: isolated interpreter, paths as arguments):
  python -I adversarial_grader_cases.py --pool GD/data/pool.jsonl \
      [--raw-root GD/data/raw --manifest GD/data/raw_manifest.json] [--grader GD/scripts/grader.py]

Part 1  CASES: how Qwen3-4B (non-thinking) / Qwen2.5-7B-Instruct end chain-of-thought answers (Qwen3 ends with
        '### Final Answer:\\n\\n$$\\n\\boxed{..}\\n$$', Qwen2.5 with 'the final answer is $\\boxed{..}$'), each with
        the gold answer and the verdict a careful human grader gives. Cases marked [real] are taken from Qwen3-4B
        generations in results/qwen3-4b (base, base_rest; tail of the text, shortened).
Part 2  POOL: 40 items sampled per source with random.Random(0) (olympiad 8, minerva 8, math500 5, gsm8k 5, amc 4,
        gpqa 4, aime24/25/26 2 each); per-source format checks, gold self-consistency through the grader, and
        (with --raw-root) the stored answer re-derived from the raw file (sha256-checked against the manifest).
        Plus whole-pool checks: GPQA letter distribution (chi-square vs uniform) and option/answer consistency,
        AIME/AMC integer formats.
Part 3  near-miss control over the whole pool: every plain-number gold against wrong neighbours (n+-1, 10n, -n;
        decimals 3 precision steps away) and against a value that agrees to the gold's written precision, and every
        math gold against a hedge '\\boxed{gold} or \\boxed{other}'.
Exit status 0 iff every case passes and every pool / near-miss check passes.
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import argparse
import collections
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import random
import re
import sys

GD = str(_experiment_root('gen_depth'))
sys.path.insert(0, os.path.join(GD, 'pylib'))

M, Q = 'math', 'mcq'


def q3(ans, pre='Adding the contributions gives the result.'):
    """Qwen3-4B non-thinking ending."""
    return f'{pre}\n\n---\n\n### \u2705 Final Answer:\n\n$$\n\\boxed{{{ans}}}\n$$'


def q25(ans, pre='Combining the steps above,'):
    """Qwen2.5-7B-Instruct ending."""
    return f'{pre} the final answer is $\\boxed{{{ans}}}$.'


CASES = [
    # ---------------------------------------------------------------- fractions, decimals, equations, units
    (q3(r'\frac{3}{4}'), '0.75', M, True, 'fraction vs decimal gold'),
    (q3('0.75'), r'\frac{3}{4}', M, True, 'decimal vs fraction gold'),
    (q25(r'\dfrac{3}{4}'), '3/4', M, True, 'dfrac, Qwen2.5 ending'),
    (q3('0.7'), r'\frac{3}{4}', M, False, 'rounded decimal of an exact fraction'),
    (q3('x = 5'), '5', M, True, 'equation x=5 vs value'),
    (q3(r'x = 2 \text{ or } x = 5'), '5', M, False, 'hedged two values vs one gold'),
    (q3(r'2 + 3 = 5'), '5', M, True, 'computation chain in the box: right-hand side'),
    (q3(r'5 \text{ cm}'), '5', M, True, '\\text unit'),
    (q3(r'5\ \mathrm{cm}^2'), '5', M, True, '\\mathrm unit with exponent'),
    (q3(r'\$12'), '12', M, True, 'dollar sign'),
    (q3(r'\$12.00'), '12', M, True, 'dollar amount with cents'),
    (q25(r'18 \text{ dollars}'), '18', M, True, 'Qwen2.5 dollars'),
    (q3('18.5'), '18', M, False, 'integer gold is exact'),
    (q3(r'12\%'), '12', M, True, 'percent vs number'),
    (q3(r'12\%'), '0.12', M, True, 'percent vs fraction'),
    (q3(r'70,000'), '70000', M, True, 'thousands comma'),
    (q3(r'70\,000'), '70000', M, True, 'thousands thin space'),
    (q3('073'), '73', M, True, 'AIME zero padding'),
    (q3('-4'), '-4', M, True, 'negative'),
    (q3('4'), '-4', M, False, 'sign error'),
    (q3(r'\text{one hundred}'), '100', M, True, 'number in words'),
    (q3(r'5 \text{ or } 7'), '57', M, False, 'digits are not concatenated across \\text'),
    (q3(r'1 \text{ hour } 30 \text{ minutes}'), '130', M, False, 'h/min is not 130'),
    (q3(r'15,000 \, \text{kcal} \, \text{m}^{-2} \, \text{yr}^{-1}'), '15000', M, True,
     '[real minerva:58] unit with negative exponents'),
    (q3(r'300 \, \text{km/s}'), '300', M, True, '[real minerva:11] km/s'),
    (q3(r'v = 300 \text{ km/s}'), '300', M, True, 'equation with unit'),
    (q3(r'\{5\}'), '5', M, True, 'one-element set vs value'),
    (q3('\u22122'), '-2', M, True, 'unicode minus'),
    (q3('\u00bd'), '1/2', M, True, 'unicode one half'),
    (q3(r'0.\overline{3}'), r'\frac{1}{3}', M, True, 'repeating decimal'),
    # ---------------------------------------------------------------------- roots, pi, intervals, tuples
    (q3(r'\sqrt{2}/2'), r'\frac{\sqrt{2}}{2}', M, True, 'sqrt(2)/2 forms'),
    (q3(r'\frac{1}{\sqrt{2}}'), r'\frac{\sqrt{2}}{2}', M, True, 'rationalized vs not'),
    (q3('0.707'), r'\frac{\sqrt{2}}{2}', M, False, '3-digit decimal of an exact surd'),
    (q3(r'\frac{\sqrt{3}}{2} \approx 0.866'), r'\frac{\sqrt{3}}{2}', M, True, 'exact form with approximation'),
    (q3(r'2\sqrt{2}'), '2.83', M, True, 'exact value vs rounded decimal gold'),
    (q3('37.7'), r'12\pi', M, False, 'decimal only, exact gold'),
    (r'The speed is $\boxed{12\pi}$ inches per second.' '\n\nOr approximately:\n\n$$\n\\boxed{37.7} '
     r'\text{ inches per second}' '\n$$', r'12\pi', M, True, '[real math500:313] exact box then its approximation'),
    (r'The speed is $\boxed{12\pi}$ inches per second, i.e. about $\boxed{40}$.', r'12\pi', M, False,
     'second box is not a rounding of the first'),
    (q3('(2,3)'), '(2,3)', M, True, 'tuple'),
    (q3('(3, 2)'), '(2,3)', M, False, 'tuple order'),
    (q3('[1,2)'), '[1,2)', M, True, 'half-open interval'),
    (q3('[1,2]'), '[1,2)', M, False, 'interval endpoint'),
    (q3(r'x \in [-2, 7]'), '[-2,7]', M, True, 'interval with x \\in'),
    (q3(r'-2 \le x \le 7'), r'x \in [-2,7]', M, True, 'inequality vs interval'),
    (q3(r'(-\infty, 0) \cup \{1\}'), r'(-\infty, 0) \cup\{1\}', M, True, 'union with a point'),
    # --------------------------------------------------------------------------------- multiple boxes
    ('So the answer is $\\boxed{3}$.\n\nWait, let me re-check the case n = 2 ... that gives one more.\n\n'
     + q3('4'), '4', M, True, 'last box wins'),
    ('So the answer is $\\boxed{3}$.\n\nWait, let me re-check the case n = 2 ... that gives one more.\n\n'
     + q3('4'), '3', M, False, 'earlier box ignored'),
    ('---\n\n### \u2705 Final Answer:\n\n$$\n\\boxed{3}, \\boxed{5}, \\boxed{7}\n$$', '3, 5, 7', M, True,
     '[real math500:36] list written as several boxes'),
    ('The solutions are $x = \\boxed{2}$ or $x = \\boxed{-1}$.', '2, -1', M, True, 'x = box or x = box'),
    ('It is either $\\boxed{3}$ or $\\boxed{4}$.', '4', M, False, 'hedge between two boxes'),
    ('$$\n\\boxed{x = 5}\n$$\n\n---\n\n### \u2705 Final Answer:\n\n$$\n\\boxed{5}\n$$', '5', M, True,
     '[real math500:22] restated final box'),
    (r'$$\boxed{216 + 224 = \boxed{440}}$$', '440', M, True, '[real math500:285] nested boxes'),
    ('I will put the final answer in \\boxed{}.\n\n... so 6 + 3 + 4 = 13.\n\nThe answer is 13.', '13', M, True,
     'empty instruction box ignored, answer cue used'),
    ('So the probability is $\\boxed{\\frac{1}{2}}$ or $\\boxed{0.5}$.', '1/2', M, True,
     'one value written twice in two boxes'),
    (q3('**18**'), '18', M, True, 'markdown bold inside the box'),
    # ------------------------------------------------------------------------- no box / truncation
    ('Therefore, the final answer is $\\frac{1}{2}$.', '1/2', M, True, "no box: 'final answer is $..$'"),
    ('**Final Answer:** 42', '42', M, True, 'no box: markdown final answer'),
    ('### Final Answer\n\n$$\n18\n$$', '18', M, True, 'no box: Final Answer heading + display math'),
    ('So she makes 16 - 3 - 4 = 9 eggs, and 9 * 2 = 18 dollars. Let me double check the number of eggs: 16',
     '16', M, False, 'truncated: last number is not an answer'),
    ('... the remainder is r = 1, and the next term is (x+1)^2', '2', M, False, 'truncated: exponent is not an answer'),
    ('So the probability is $\\frac{3}{4}$.\n\n### Final Answer\n\n$$\n\\boxed{\\frac{3}{4', '4', M, False,
     'truncated mid-box: denominator is not an answer'),
    ('So the probability is 3/4.\n\n### Final Answer\n\n$$\n\\boxed{\\frac{3}{4', '3/4', M, False,
     'truncated mid-box: no final answer'),
    ('First guess $\\boxed{7}$. Let me double check the count ... it is $\\boxed{8', '7', M, True,
     'truncated mid-box after a complete box: the complete box counts'),
    ('### \u2705 Final Answer:\n\n$$\n\\boxed{13\n$$', '13', M, True, 'final box missing its closing brace'),
    ('So **13** is the minimum.\n\n---\n\n### \u2705 Final Answer', '13', M, False,
     '[real L_e10 math500:485] truncated right before the answer'),
    ('answer: - **Group A1** (start with truth): Same as truth-tellers \u2192 contributes 1 yes to Q1', '1', M,
     False, '[real amc:21] free text after an answer cue'),
    ('The answer is not 5, let me recompute the sum.', '5', M, False, "'answer is not 5'"),
    ('To find the answer: Step 2 gives the count of the remaining', '2', M, False, "'answer: Step 2 ...' (truncated)"),
    ('So the answer is 13. Let me double-check: 6 + 3 + 4 = 13, and 5 + 3 + 5 = 13, and', '13', M, True,
     'truncated after an explicit answer statement'),
    # ------------------------------------------------------------------- scientific notation, Minerva
    (q3(r'3.2 \times 10^{-4}'), '3.2e-4', M, True, 'sci vs e-notation'),
    (q3(r'3.2 \times 10^{-4}'), '0.00032', M, True, 'sci vs decimal'),
    (q3(r'3.2 \cdot 10^{-4}\ \text{s}^{-1}'), '3.2e-4', M, True, 'cdot sci with unit'),
    (q3(r'3.2 \times 10^{-5}'), '3.2e-4', M, False, 'wrong exponent'),
    (q3('1.57'), '1.6', M, True, '[real minerva:0] more precise than the 2-s.f. gold'),
    (q3('1.66'), '1.6', M, False, 'does not round to the gold'),
    (q3(r'3.275 \times 10^{12}'), '3.3e12', M, True, '[real minerva:18] 3.275e12 rounds to 3.3e12'),
    (q3('0.1921'), '0.192', M, True, '[real minerva:54]'),
    (q3(r'4.57 \times 10^{-24} \, \text{m}'), '4.6e-24', M, True, '[real minerva:267]'),
    (q3(r'1.4899 \times 10^4'), '1.4903e4', M, False, '[real minerva:149] wrong at the requested 4 decimals'),
    (q3('24.306'), '24.310', M, False, '[real minerva:170] arithmetic slip at the requested 3 decimals'),
    (q3(r'6 \times 10^{15} \text{ years}'), '6e13', M, False, '[real minerva:43] wrong magnitude'),
    (q3(r'1.3 \times 10^{-27}'), '2.88e-19', M, False, 'tiny numbers compared relatively'),
    ('R = (L / 4 pi sigma T^4)^{1/2}\n\nFinal answer: The final answer is 8.7e8. I hope it is correct.', '8.7e8', M, True,
     'Minerva preamble style without a box (70 Minerva prompts show this format)'),
    (q3(r'30^\circ'), '30', M, True, 'degree sign ^\\circ'),
    (q3('30\u00b0'), '30', M, True, 'degree sign \u00b0'),
    (q3(r'30 \text{ degrees}'), r'30^\circ', M, True, 'degrees in words vs ^\\circ gold'),
    (q3(r'-2 + 2i\sqrt{3}'), r'-2+2 \sqrt{3} i', M, True, '[real minerva:93] i before the root'),
    (q3(r'-2 - 2i\sqrt{3}'), r'-2+2 \sqrt{3} i', M, False, 'conjugate'),
    (q3(r'z_p(t) = \frac{2 A}{3} (1 - i) e^{i t}'), r'\frac{4 A}{3+3 i} e^{i t}', M, True,
     '[real minerva:85] i is the imaginary unit'),
    (q3(r'-\frac{1}{5} \sin(3t) + C_1 \cos(2t) + C_2 \sin(2t)'),
     r'-\sin (3 t) / 5+ C_{1} \sin (2 t)+C_{2} \cos (2 t)', M, True, '[real minerva:101] constants renamed'),
    (q3(r'-\frac{1}{4} \sin(3t) + C_1 \cos(2t) + C_2 \sin(2t)'),
     r'-\sin (3 t) / 5+ C_{1} \sin (2 t)+C_{2} \cos (2 t)', M, False, 'wrong particular solution'),
    (q3(r'\frac{1}{3} e^t + C e^{-2t}'), r'\frac{e^{t}} {3}+c e^{-2 t}', M, True, 'constant c vs C'),
    (q3(r'\frac{2}{3}'), r'\frac{1}{3} E_{1}+\frac{2}{3} E_{2}', M, False,
     'gold whose LaTeX parse fails must not become a sub-match (2/3) [minerva:268]'),
    (q3(r'\frac{E_1 + 2E_2}{3}'), r'\frac{1}{3} E_{1}+\frac{2}{3} E_{2}', M, True, 'subscripted E parsed'),
    (q3('2'), r'\hbar \omega(v+1 / 2)-\frac{E_{0}^{2} e^{2}}{2 m \omega^{2}}', M, False,
     'gold parse garbage 2 [minerva:261]'),
    (q3(r'\hbar\omega\left(v+\frac{1}{2}\right) - \frac{E_0^2 e^2}{2m\omega^2}'),
     r'\hbar \omega(v+1 / 2)-\frac{E_{0}^{2} e^{2}}{2 m \omega^{2}}', M, True, '[minerva:261] equivalent form'),
    (q3(r'P(x = 2x^3 - 2ax^2 + (a^2 - 81)x - c'), '81', M, False,
     '[real math500:285, L_c35] unparseable box is not its sub-match 81'),
    # ------------------------------------------------------------------------ text / special golds
    (q3(r'\text{East}'), r'\text{east}', M, True, 'text answer, case'),
    (q3(r'\text{west}'), r'\text{east}', M, False, 'wrong text answer'),
    (q3('52'), '52_8', M, True, 'base subscript dropped'),
    (q3('52_{8}'), '52_8', M, True, 'base subscript braces'),
    (q3(r'\text{C}'), r'\text{(C)}', M, True, 'MATH choice letter'),
    (q3(r'\text{(C) Plane}'), r'\text{(C)}', M, True, 'MATH choice letter with option text'),
    (q3('y = 2x + 3'), 'y = 2x + 3', M, True, 'line equation'),
    (q3('f(x) = 2x'), 'f(x)=2 x', M, True, 'function equation'),
    # ------------------------------------------------------------------------------------- mcq (GPQA)
    ('**Final Answer**\n\\boxed{B}', 'B', Q, True, '**Final Answer** box'),
    (q3(r'\text{(B)}'), 'B', Q, True, '\\text{(B)}'),
    (q3(r'\textbf{D}'), 'D', Q, True, '\\textbf{D}'),
    (q3(r'D) \frac{9}{2} \hbar \omega, \ 10'), 'D', Q, True, '[real gpqa:124] letter + option text'),
    (q3(r'B) \, 3,060,000 \, \text{km}'), 'B', Q, True, '[real] letter + value'),
    (q3(r'\text{A) 0}'), 'A', Q, True, '[real gpqa:66]'),
    (q3(r'\text{Option C}'), 'C', Q, True, 'Option C'),
    (q3('c'), 'C', Q, True, 'lowercase letter alone'),
    (q3('(b)'), 'B', Q, True, 'lowercase letter in parentheses'),
    (q3('a = 2'), 'A', Q, False, 'variable a, not option A'),
    (q3(r'd = 5 \text{ cm}'), 'D', Q, False, 'variable d, not option D'),
    (q3(r'A \text{ or } C'), 'A', Q, False, 'hedge between two options'),
    (q3('B'), 'C', Q, False, 'wrong letter'),
    (q3(r'D - 10^{-4} \text{ eV}'), 'D', Q, True, "'D - <option text>'"),
    (q3(r'\textbf{(B)}\ 2.5'), 'B', Q, True, 'bold (B) + value'),
    (q3(r'\text{Answer: (B)}'), 'B', Q, True, 'Answer: (B) inside the box'),
    ('First guess \\boxed{A}.\n\nOn reflection both remaining options fit:\n\n' + q3(r'B \text{ or } C'), 'A', Q, False,
     'final hedge box does not fall back to an earlier box'),
    ('Rounded to **one decimal place**:\n\n$$\n\\boxed{-0.7}\n$$\n\n---\n\n### \u2705 Final Answer:\n\n'
     '$$\n\\boxed{A)\n$$', 'A', Q, True, '[real gpqa:2] final box missing its closing brace'),
    ('The answer is B.', 'B', Q, True, "no box: 'The answer is B.'"),
    ('**Answer: C**', 'C', Q, True, 'no box: bold answer'),
    ('The correct option is **D**.', 'D', Q, True, "no box: 'correct option is'"),
    ('Option B is correct, since only it conserves charge.', 'B', Q, True, "no box: 'Option B is correct'"),
    ('Comparing the four energies, I would choose (A).', 'A', Q, True, "no box: 'choose (A)'"),
    ('Therefore, the answer is (B).', 'B', Q, True, 'no box: Qwen2.5 style'),
    ('Answer: B. In numbers that is \\boxed{10^{-4}\\text{ eV}}', 'B', Q, True, 'non-letter box, answer cue'),
    ("Let's evaluate option C: the molecular formula C\u2088H\u2089NO requires one degree of unsaturation, "
     'while option B', 'B', Q, False, 'truncated mid-reasoning, last letter mentioned'),
    ("So the correct **B sequence is**: **2, 1, 3, 4**\n\nBut this is **not** among the options.\n\nWait \u2014 the "
     '**options for B are**:\n\n- A) B = 3, 1, 2, 4\n- B) B = 3, 1, 2, 4\n- C) B = 4, 2, 1,', 'B', Q, False,
     '[real gpqa:25] truncated'),
    ('Actually, the correct product is **1-cyano-2-cyclopentene**.\n\n```\n       CN\n        |\n'
     'CH\u2082CH\u2082C(=O)CH\u2082CH\u2082CN\n```\n\nBut this is not correct.\n\nActually', 'C', Q, False,
     '[real gpqa:121] looping, C(=O) is not an answer'),
    ('A = 65  \nC = 67  \nG = 71  \nT = 84\n\nSo the sum is 622, and 622 mod 100 = 22 \u2192 not matching.',
     'C', Q, False, '[real gpqa:79] truncated, letters are variables'),
    ('So compound A must be the diene and product B the adduct. Considering the regiochemistry of',
     'A', Q, False, 'truncated, compound labels'),
    ('The answer is \\boxed{B}.\n\nHmm, but C also fits the NMR data. Let me reconsider', 'B', Q, True,
     'boxed letter, later doubt without a new box'),
]


def run_cases(grader):
    fails = []
    for text, gold, typ, want, note in CASES:
        got = grader.grade(text, gold, typ)
        if got is not want:
            fails.append((note, gold, typ, want, got, grader.extract_answer(text, typ), text[-120:]))
    n = len(CASES)
    real = sum('[real' in c[4] for c in CASES)
    print(f'adversarial cases: {n - len(fails)}/{n} pass ({real} from real generations; '
          f'{sum(c[2] == Q for c in CASES)} mcq, {sum(c[3] is False for c in CASES)} expected False)')
    for f in fails:
        print('  FAIL', f)
    return fails


# ------------------------------------------------------------------------------------------- pool sample
QUOTA = {'olympiad': 8, 'minerva': 8, 'math500': 5, 'gsm8k': 5, 'amc': 4, 'gpqa': 4, 'aime24': 2, 'aime25': 2,
         'aime26': 2}
INT_RE = re.compile(r'-?(?:0|[1-9]\d*)')


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Raw:
    """Raw rows by pool index, read only from sha256-verified files."""

    def __init__(self, root, manifest_path):
        import pandas as pd
        self.pd = pd
        self.root, self.man, self.cache = root, json.load(open(manifest_path)), {}

    def rows(self, name, fn):
        key = (name, fn)
        if key not in self.cache:
            f = self.man[name]['files'][fn]
            p = os.path.join(self.root, f['local_path'])
            assert sha256_of(p) == f['sha256'], f'sha256 mismatch {p}'
            if p.endswith('.jsonl'):
                self.cache[key] = [json.loads(line) for line in open(p) if line.strip()]
            else:
                self.cache[key] = self.pd.read_parquet(p).to_dict('records')
        return self.cache[key]

    def answer(self, it):
        """The answer re-derived from the raw row (independently of build_pool.py), or None if not applicable."""
        src, idx = it['source'], int(it['pid'].split(':')[1])
        if src == 'gsm8k':
            return self.rows('gsm8k', 'main/test-00000-of-00001.parquet')[idx]['answer'].split('####')[-1] \
                .strip().replace(',', '')
        if src == 'math500':
            return self.rows('math500', 'test.jsonl')[idx]['answer'].strip()
        if src in ('aime24', 'aime25', 'aime26'):
            return str(int(self.rows(src, 'data/train-00000-of-00001.parquet')[idx]['answer']))
        if src == 'amc':
            a = float(self.rows('amc', 'data/train-00000-of-00001.parquet')[idx]['answer'])
            return str(int(a)) if a == int(a) else str(a)
        if src == 'olympiad':
            r = self.rows('olympiad', 'OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet')[idx]
            return str(r['final_answer'][0]).strip()
        if src == 'minerva':
            return self.rows('minerva', 'test.jsonl')[idx]['answer'].strip()
        if src == 'gpqa':
            r = self.rows('gpqa_jingzeshi', 'data/test-00000-of-00001.parquet')[idx]
            return str(r['Correct Answer']).strip()
        return None


def documented_fix(it, stats):
    for f in stats.get('answer_fixes', {}).get(it['source'], []):
        if f['pid'] == it['pid']:
            return f
    return None


def check_item(it, grader, raw, stats):
    """List of problems with one pool item (empty = sane)."""
    bad = []
    a, src = it['answer'], it['source']
    if not isinstance(a, str) or not a or a != a.strip():
        return ['answer empty or not stripped']
    if '$' in a.replace('\\$', ''):
        bad.append('math delimiter $ in the answer')
    if src in ('gsm8k', 'aime24', 'aime25', 'aime26', 'amc'):
        if not INT_RE.fullmatch(a):
            bad.append('not a plain integer')
        elif src.startswith('aime') and not 0 <= int(a) <= 999:
            bad.append('AIME answer outside 0..999')
    if src == 'gpqa':
        if a not in 'ABCD' or len(a) != 1 or it['answer_type'] != 'mcq':
            bad.append('gpqa answer is not a letter A-D')
        else:
            pos = [it['problem'].find('\n%s) %s' % (L, o)) for L, o in zip('ABCD', it['meta']['options'])]
            if min(pos) < 0 or pos != sorted(pos) or not it['problem'].endswith(it['meta']['options'][3]):
                bad.append('options A-D (meta) not found in order at the end of the problem')
    elif it['answer_type'] != 'math':
        bad.append('answer_type should be math')
    if src == 'olympiad':
        depth, top_comma = 0, False
        for ch in a:
            depth += ch in '([{'
            depth -= ch in ')]}'
            top_comma |= ch == ',' and depth == 0
        if top_comma:
            bad.append('olympiad answer lists several values')
    text = 'Some reasoning ... so the answer is \\boxed{' + a + '}.'
    if not grader.grade(text, a, it['answer_type']):
        bad.append('gold does not grade against itself')
    if raw is not None:
        ra = raw.answer(it)
        if src == 'gpqa':
            opts = it['meta']['options']
            if opts['ABCD'.index(a)] != ra:
                bad.append(f'gold letter {a} is not the raw correct answer {ra[:40]!r}')
        elif src == 'olympiad':
            fix = documented_fix(it, stats)
            stripped = ra.strip('$').strip()
            if fix is None and stripped != a and not grader.math_equal(stripped, a):
                bad.append(f'differs from raw final_answer {ra!r}')
        elif src == 'minerva':
            fix = documented_fix(it, stats)
            if fix is None and ra != a:
                bad.append(f'differs from raw answer {ra!r}')
        elif ra != a:
            bad.append(f'differs from raw answer {ra!r}')
    return bad


def run_pool(path, grader, raw, stats):
    pool = [json.loads(line) for line in open(path)]
    by = collections.defaultdict(list)
    for it in pool:
        by[it['source']].append(it)
    rng = random.Random(0)
    sample = []
    for src in sorted(QUOTA):
        sample += rng.sample(by[src], QUOTA[src])
    problems = []
    print(f'\npool sample: {len(sample)} items (seed 0){"" if raw else "; no raw cross-check (--raw-root not given)"}')
    for it in sample:
        bad = check_item(it, grader, raw, stats)
        problems += [(it['pid'], b) for b in bad]
        print(f"  {'ok ' if not bad else 'BAD'} {it['pid']:13s} {it['answer_type']:4s} {it['answer']!r:45.45}"
              f" {'; '.join(bad)}")
    # whole-pool checks
    g = [it['answer'] for it in by['gpqa']]
    cnt = collections.Counter(g)
    exp = len(g) / 4
    chi2 = sum((cnt[L] - exp) ** 2 / exp for L in 'ABCD')
    print(f'gpqa letters {dict(sorted(cnt.items()))}: chi2 = {chi2:.2f} (df 3; 7.81 = p 0.05)')
    if chi2 > 7.81:
        problems.append(('gpqa', f'letter distribution not uniform (chi2 {chi2:.2f})'))
    longest = max(collections.Counter(zip(g, g[1:])).values())
    for it in by['gpqa']:
        for b in check_item(it, grader, None, stats):
            problems.append((it['pid'], b))
    for src in ('aime24', 'aime25', 'aime26', 'amc', 'gsm8k'):
        nonint = [it['pid'] for it in by[src] if not INT_RE.fullmatch(it['answer'])]
        if nonint:
            problems.append((src, f'non-integer answers {nonint[:5]}'))
    dup = [k for k, v in collections.Counter(it['pid'] for it in pool).items() if v > 1]
    if dup:
        problems.append(('pool', f'duplicate pids {dup[:5]}'))
    print(f'whole-pool checks: gpqa option/letter consistency on all {len(by["gpqa"])} items, integer formats '
          f'(aime, amc, gsm8k), unique pids; most frequent consecutive gpqa letter pair x{longest}')
    print(f'pool problems: {len(problems)}')
    for p in problems:
        print('  PROBLEM', p)
    return problems


def plain_number(g):
    """(value, kind, half unit in the last digit) of a plain-number gold string (independent of the grader)."""
    g = g.strip()
    m = re.fullmatch(r'-?\d+', g)
    if m:
        return float(g), 'int', 0.0
    m = re.fullmatch(r'-?(\d*)\.(\d+)', g)
    if m:
        return float(g), 'approx', 0.5 * 10.0 ** -len(m.group(2))
    m = re.fullmatch(r'(-?\d+(?:\.(\d+))?)e(-?\d+)', g)
    if m:
        return float(g), 'approx', 0.5 * 10.0 ** (int(m.group(3)) - len(m.group(2) or ''))
    m = re.fullmatch(r'(-?)\\frac\{(\d+)\}\{(\d+)\}|(-?)(\d+)/(\d+)', g)
    if m:
        sign, a, b = (m.group(1), m.group(2), m.group(3)) if m.group(2) else (m.group(4), m.group(5), m.group(6))
        return (-1 if sign else 1) * int(a) / int(b), 'frac', 0.0
    return None


def fmt(v):
    """Short decimal string of a float (12 significant digits; e-notation for tiny / huge values)."""
    return '%.12g' % v


def run_near_miss(path, grader):
    """Every plain-number gold vs wrong neighbours (must be False) and vs a value that agrees to the gold's
    written precision (must be True); every gold vs a hedge '\\boxed{gold} or \\boxed{other}' (False)."""
    pool = [json.loads(line) for line in open(path)]
    fails, n_neg, n_pos = [], 0, 0
    for it in pool:
        if it['answer_type'] != 'math':
            continue
        g = it['answer']
        n = plain_number(g)
        negs, poss = [], []
        if n is not None:
            v, kind, hulp = n
            if kind == 'int':
                negs += [str(int(v) + 1), str(int(v) - 1)] + ([str(int(v) * 10), str(-int(v))] if v else [])
            elif kind == 'approx':
                step = max(hulp, 1e-4 * abs(v))
                negs += [fmt(v + 3 * step), fmt(v - 3 * step)]
                if hulp > 1e-4 * abs(v):
                    poss.append(fmt(v + 0.4 * hulp))
            else:
                negs += [fmt(v * 1.001), fmt(v + 1)]
        mc = re.fullmatch(r'\\text\{\(([A-H])\)\}', g)
        other = ('\\text{(%s)}' % ('A' if mc.group(1) != 'A' else 'B')) if mc else '7919'
        texts = [(q3(p), p) for p in negs]
        texts += [(f'It is either $\\boxed{{{g}}}$ or $\\boxed{{{other}}}$.', 'hedge gold first'),
                  (f'It is either $\\boxed{{{other}}}$ or $\\boxed{{{g}}}$.', 'hedge gold last')]
        for text, p in texts:
            n_neg += 1
            if grader.grade(text, g, M):
                fails.append((it['pid'], g, 'wrong value graded True', p))
        for p in poss:
            n_pos += 1
            if not grader.grade(q3(p), g, M):
                fails.append((it['pid'], g, 'value at the gold precision graded False', p))
    print(f'\nnear-miss control: {n_neg} wrong predictions (neighbours of numeric golds, hedges) + {n_pos} '
          f'predictions within the gold precision; failures: {len(fails)}')
    for f in fails[:40]:
        print('  FAIL', f)
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--raw-root', default=None)
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--stats', default=None, help='pool_stats.json (documented answer fixes); default next to pool')
    ap.add_argument('--grader', default=os.path.join(GD, 'scripts', 'grader.py'),
                    help='grader source file to test (default scripts/grader.py; e.g. data/grader_v1_builder.py.orig)')
    a = ap.parse_args()
    loader = importlib.machinery.SourceFileLoader('grader', a.grader)     # any suffix (e.g. a .orig backup)
    spec = importlib.util.spec_from_file_location('grader', a.grader, loader=loader)
    grader = importlib.util.module_from_spec(spec)
    loader.exec_module(grader)
    stats = json.load(open(a.stats or os.path.join(os.path.dirname(a.pool), 'pool_stats.json')))
    raw = Raw(a.raw_root, a.manifest) if a.raw_root else None
    fails = run_cases(grader)
    problems = run_pool(a.pool, grader, raw, stats)
    near = run_near_miss(a.pool, grader)
    ok = not fails and not problems and not near
    print('OK' if ok else 'NOT OK')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
