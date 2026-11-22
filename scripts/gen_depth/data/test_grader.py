"""Tests for scripts/grader.py: hand-written cases + self-consistency over the pool.

Usage: python -I test_grader.py GD/data/pool.jsonl
Exit status 0 iff all hand cases pass and the self-consistency rate is >= 99.5%.

Pool checks
  self-consistency (required): grade('... so the answer is \\boxed{<answer>}.', answer, type) is True.
  math-verify parse (diagnostic): golds that math-verify cannot parse into a sympy object can only be graded
      by normalized string comparison.
  perturbed self-consistency (diagnostic): an equivalent rewrite of each math gold (1234 -> 1,234, 18 -> 18.0,
      4.5 -> 4.50, 4.5e33 -> 4.5 \\times 10^{33}, \\frac -> \\dfrac, ', ' spacing) must grade True; and
      1 \\cdot \\left(gold\\right) for single-expression golds, which forces the math-verify / sympy path.
  negative control (diagnostic): each gold is graded against 4 random other golds of the same source (seed 0);
      a True for two different golds is a potential false positive of the grader.
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import json
import os
import random
import re
import sys
import threading
import time

GD = str(_experiment_root('gen_depth'))
sys.path.insert(0, os.path.join(GD, 'pylib'))
sys.path.insert(0, os.path.join(GD, 'scripts'))

import grader  # noqa: E402

M, Q = 'math', 'mcq'
# (generation text, gold answer, answer_type, expected grade, note)
CASES = [
    # boxes
    (r'Thus the result is \boxed{\boxed{12}}.', '12', M, True, 'nested boxes'),
    (r'We get \boxed{\frac{\sqrt{3}}{2}}.', r'\frac{\sqrt3}{2}', M, True, 'nested braces + \\sqrt3'),
    (r'First \boxed{3}. Correction: the answer is \boxed{4}.', '4', M, True, 'last box wins'),
    (r'First \boxed{3}. Correction: the answer is \boxed{4}.', '3', M, False, 'earlier box ignored'),
    (r'Answer: \fbox{42}', '42', M, True, '\\fbox'),
    (r'So \boxed{7}. Let me double check ... \boxed{8', '7', M, True, 'unterminated last box ignored'),
    (r'\boxed{\left\{ 1, 2 \right\}}', r'\{1,2\}', M, True, 'escaped braces inside box'),
    # fractions / decimals
    (r'\boxed{\frac{3}{4}}', '0.75', M, True, '\\frac vs decimal'),
    (r'\boxed{0.75}', r'\frac{3}{4}', M, True, 'decimal vs \\frac'),
    (r'\boxed{\dfrac{5}{2}}', r'\frac52', M, True, '\\dfrac vs \\frac52'),
    (r'\boxed{\tfrac{1}{3}}', '1/3', M, True, '\\tfrac vs a/b'),
    (r'\boxed{0.3}', r'\frac{1}{3}', M, False, 'rounded decimal is not 1/3'),
    # roots / pi
    (r'\boxed{2\sqrt{3}}', r'\sqrt{12}', M, True, 'sqrt forms'),
    (r'\boxed{\sqrt3}', r'\sqrt{3}', M, True, 'sqrt without braces'),
    (r'\boxed{\sqrt{2}}', r'\sqrt{3}', M, False, 'different roots'),
    (r'\boxed{\frac{\pi}{2}}', r'\pi/2', M, True, 'pi fraction forms'),
    (r'\boxed{2\pi}', r'2 \pi', M, True, 'pi spacing'),
    (r'\boxed{3.14}', r'\pi', M, False, 'pi approximation'),
    # degrees, units, text, percent
    (r'\boxed{30^\circ}', '30', M, True, 'degrees in pred'),
    (r'\boxed{45}', r'45^{\circ}', M, True, 'degrees in gold'),
    (r'\boxed{18 \text{ dollars}}', '18', M, True, '\\text{} unit'),
    (r'\boxed{5\text{ cm}^2}', '5', M, True, '\\text{} unit with exponent'),
    (r'\boxed{\$18.90}', '18.90', M, True, 'currency sign'),
    (r'\boxed{\text{Evelyn}}', r'\text{Evelyn}', M, True, 'text answer'),
    (r'\boxed{25\%}', '25', M, True, 'percent'),
    # thousands separators
    (r'\boxed{1,234}', '1234', M, True, 'comma thousands'),
    (r'\boxed{12{,}000}', '12000', M, True, '{,} thousands'),
    (r'\boxed{1\,000\,000}', '1000000', M, True, '\\, thousands'),
    (r'\boxed{100001}', '100000', M, False, 'integers compared exactly'),
    # intervals / tuples / sets
    (r'\boxed{[-2, 5)}', '[-2,5)', M, True, 'interval'),
    (r'\boxed{(-2, 5)}', '[-2,5)', M, False, 'interval endpoint type'),
    (r'\boxed{(-\infty, 0) \cup \{1\}}', r'(-\infty, 0) \cup\{1\}', M, True, 'interval union'),
    (r'\boxed{x \in [1, 2)}', '[1,2)', M, True, 'interval with x \\in'),
    (r'\boxed{(3, \frac{\pi}{2})}', r'\left( 3, \frac{\pi}{2} \right)', M, True, 'tuple with \\left( \\right)'),
    (r'\boxed{(2, 1)}', '(1, 2)', M, False, 'tuple order matters'),
    (r'\boxed{2, -1}', '-1,2', M, True, 'unordered list of solutions'),
    # scientific notation (Minerva)
    (r'\boxed{4.5 \times 10^{33}}', '4.5e33', M, True, 'sci notation vs e-notation gold'),
    (r'\boxed{4.5e33}', r'4.5 \times 10^{33}', M, True, 'e-notation pred'),
    (r'\boxed{4.5 \times 10^{32}}', '4.5e33', M, False, 'wrong exponent'),
    (r'\boxed{4.0319}', '4.03188', M, True, 'decimal gold: relative tolerance 1e-4'),
    (r'\boxed{4.1}', '4.03188', M, False, 'decimal gold: outside tolerance'),
    (r'\boxed{6 \times 10^{-3}}', '0.006', M, True, 'sci vs decimal'),
    (r'\boxed{1.3 \times 10^{-27}}', '2.88e-19', M, False, 'tiny numbers compared relatively, not rounded'),
    (r'\boxed{2.9 \times 10^{-19}}', '2.88e-19', M, False, 'tiny numbers: 0.7% off'),
    (r'\boxed{2.88 \times 10^{-19}\ \mathrm{J}}', '2.88e-19', M, True, 'tiny number with unit'),
    (r'\boxed{0.333333}', r'\frac{1}{3}', M, True, '6-significant-digit decimal of a fraction'),
    (r'\boxed{0.3333}', r'\frac{1}{3}', M, False, '4-digit decimal of a fraction'),
    (r'\boxed{0.707107}', r'\frac{\sqrt{2}}{2}', M, True, '6-digit decimal of a surd'),
    (r'\boxed{0.0000001}', '0', M, False, 'tiny number is not zero'),
    (r'\boxed{3.3 \times 10^{-10}}', '0', M, False, 'tiny number is not integer zero'),
    (r'\boxed{\sqrt{4}}', '2', M, True, 'numeric gold, symbolic prediction'),
    (r'\boxed{0.25}', '25\\%', M, True, 'percent gold vs fraction'),
    # expressions / equations / signs
    (r'\boxed{x = 5}', '5', M, True, 'equation pred vs value gold'),
    (r'\boxed{p - q}', 'p-q', M, True, 'symbolic'),
    (r'\boxed{q - p}', 'p-q', M, False, 'symbolic sign'),
    (r'\boxed{-7}', '-7', M, True, 'negative'),
    (r'\boxed{7}', '-7', M, False, 'missing sign'),
    (r'\boxed{\frac{1}{2n+2}}', r'\frac{1}{2 n+2}', M, True, 'expression in n'),
    (r'\boxed{y = 2x + 1}', 'y=2x+1', M, True, 'line equation'),
    # fallbacks (no box)
    (r'Therefore, the final answer is $\frac{1}{2}$.', '1/2', M, True, "fallback 'answer is'"),
    (r'The answer is: **42**', '42', M, True, "fallback 'answer is' with markdown"),
    (r'So she makes 1,250 dollars in total', '1250', M, False,
     'no last-number fallback: a bare number in an unboxed (in practice truncated) text is not an answer'),
    (r'So she makes 1,250 dollars in total. The answer is 1,250.', '1250', M, True, "fallback 'answer is'"),
    ('', '1', M, False, 'empty text'),
    # mcq
    (r'... so the answer is \boxed{B}.', 'B', Q, True, 'boxed letter'),
    (r'\boxed{(B)}', 'B', Q, True, '(B) vs B'),
    (r'\boxed{\text{C}}', 'C', Q, True, '\\text{C}'),
    (r'\boxed{\textbf{(D)}}', 'D', Q, True, '\\textbf{(D)}'),
    (r'\boxed{D) 10^{-4} eV}', 'D', Q, True, 'letter + option text'),
    (r'\boxed{b}', 'B', Q, True, 'lowercase boxed letter'),
    (r'\boxed{A} ... wait, recheck ... \boxed{C}', 'C', Q, True, 'last boxed letter'),
    (r'\boxed{A}', 'B', Q, False, 'wrong letter'),
    (r'Answer: B. In numbers that is \boxed{10^{-4}\text{ eV}}', 'B', Q, True, "non-letter box -> 'Answer: X'"),
    (r'Answer: C', 'C', Q, True, "'Answer: X'"),
    (r'The correct answer is (A).', 'A', Q, True, "'answer is (X)'"),
    (r'**Answer:** D', 'D', Q, True, 'markdown answer'),
    (r'After comparing, I would pick B', 'B', Q, True, 'final standalone letter'),
    (r'I conclude D. A careful check confirms it', 'D', Q, True, "article 'A' is not a letter answer"),
    (r'The answer is a bit subtle; it is (C)', 'C', Q, True, "'answer is a' is not letter A"),
]

# Realistic predictions against gold formats that occur in data/pool.jsonl.
POOL_FORMAT = [
    ('C', r'\text{(C)}', True), ('(C)', r'\text{(C)}', True), ('D', r'\text{(C)}', False),
    ('East', r'\text{east}', True), ('odd', r'\text{even}', False),
    ('32,348', r'\$32,\!348', True), ('32348', r'\$32,\!348', True),
    ('864', r'864 \mbox{ inches}^2', True), ('865', r'864 \mbox{ inches}^2', False),
    (r'\frac{270}{7}^\circ', r'\frac{270}7\text{ degrees}', True),
    (r'1+\sqrt{19}, 1-\sqrt{19}', r'1 \pm \sqrt{19}', True), (r'1 + \sqrt{19}', r'1 \pm \sqrt{19}', False),
    (r'-5x + 7y - 11z - 4 = 0', r'5x - 7y + 11z + 4 = 0', True),
    (r'5x - 7y + 11z - 4 = 0', r'5x - 7y + 11z + 4 = 0', False),
    (r'2x+3', r'y = 2x + 3', True), ('2x', 'f(x)=2 x', True), ('1', 'k=1', True), ('2', 'k=1', False),
    ('0.625', '62.5%', True), ('62.5', '62.5%', True), ('10080', r'10,\!080', True),
    ('11111111100', r'11,\! 111,\! 111,\! 100', True),
    (r'\{-2, 1+\sqrt{5}, 1-\sqrt{5}\}', r'\{1\pm\sqrt{5},-2\}', True),
    (r'[-2, 7]', r'x \in [-2,7]', True), (r'-2 \le x \le 7', r'x \in [-2,7]', True),
    (r'\sin^{-1}(10/13)', r'\arcsin(10/13)', True), (r'E_1, E_2', r'E_{1},E_{2}', True),
    (r'\frac{a M^{1/3}}{G M^{2/3} + b}', r'\frac{a M^{1 / 3}}{G M^{2 / 3}+b}', True),
    (r'1.6 \mathrm{~cm}', '1.6', True), (r'2.7778 \times 10^{-6}', '2.7778e-6', True),
    (r'2.78 \times 10^{-6}', '2.7778e-6', False), ('4.2', '4.2', True), ('023', '23', True),
    (r'\frac{37}{4}', r'\frac{37}{4} m', False), (r'\frac{37 m}{4}', r'\frac{37}{4} m', True),
    ('6', '6m', False), ('6 m', '6m', True), (r'\frac{c}{\sqrt{2}}', r'\frac{1}{\sqrt{2}}c', True),
    (r'\frac{\sqrt{2}}{2} \text{ m}', r'\frac{\sqrt{2}}{2}', True), (r'0.707107 m', r'\frac{\sqrt{2}}{2}', True),
    ('45 minutes', '45', True), (r'7.6 Z', '7.6 Z', True), ('7.6', '7.6 Z', False),
    (r'12^{\text{th}} \text{ grade}', '12', True), ('21st', '21', True),
    (r'\textbf{(B)}\ 3(1 + r/10)^x = 7', r'\text{(B)}', True), (r'\text{(E) Hyperbola}', r'\text{(E)}', True),
    ('(D) Hyperbola', r'\text{(E)}', False), ('E', r'\text{(E)}', True),
]


def run_cases():
    fails = []
    for text, gold, typ, want, note in CASES:
        got = grader.grade(text, gold, typ)
        if got != want:
            fails.append((note, text, gold, typ, want, got, grader.extract_answer(text, typ)))
    for pred, gold, want in POOL_FORMAT:
        got = grader.grade('\\boxed{' + pred + '}', gold, M)
        if got != want:
            fails.append(('pool format', pred, gold, M, want, got, None))
    # fallback switch and robustness
    extra = [
        ('fallback=False disables answer-is', grader.grade('The answer is 5.', '5', M, fallback=False), False),
        ('fallback=False keeps boxes', grader.grade(r'\boxed{5}', '5', M, fallback=False), True),
        ('None text never raises', grader.grade(None, '5', M), False),
        ('None answer never raises', grader.grade(r'\boxed{5}', None, M), False),
        ('garbage never raises', grader.grade('\\boxed{' + '{' * 50 + '}', '5', M), False),
        ('huge box content is ignored', grader.extract_answer('\\boxed{' + 'x' * 5000 + '}', M) is None, True),
        ('extract math', grader.extract_answer(r'so \boxed{\frac{1}{2}} done', M), r'\frac{1}{2}'),
        ('extract mcq', grader.extract_answer(r'Final: \boxed{(c)}', Q), 'C'),
    ]
    box = []
    th = threading.Thread(target=lambda: box.append(grader.grade(r'\boxed{\frac{6}{8}}', '0.75', M)))
    th.start()
    th.join()
    extra.append(('grading from a non-main thread', box[0] if box else None, True))
    t0 = time.time()
    slow = grader.grade(r'\boxed{m l \ddot{\theta}(t)-m g \sin \theta(t)=f(t) \cos \theta(t)}',
                        r'\frac{d^{2} v_{o}}{d t^{2}}+\frac{v_{o}}{L C}=\frac{v_{i}}{L C}', M)
    dt = time.time() - t0
    extra.append((f'slow sympy comparison is cut by the timeout ({dt:.1f}s)', (slow, dt < grader.TIMEOUT_S + 1.5),
                  (False, True)))
    for note, got, want in extra:
        if got != want:
            fails.append((note, '', '', '', want, got, None))
    n = len(CASES) + len(POOL_FORMAT) + len(extra)
    print(f'hand cases: {n - len(fails)}/{n} pass')
    for f in fails:
        print('  FAIL', f)
    return n, fails


def run_pool(path):
    pool = [json.loads(line) for line in open(path)]
    t0 = time.time()
    fails = []
    for it in pool:
        text = 'Some reasoning ... so the answer is \\boxed{' + it['answer'] + '}.'
        if not grader.grade(text, it['answer'], it['answer_type']):
            fails.append(it)
    rate = 1 - len(fails) / len(pool)
    print(f'self-consistency: {len(pool) - len(fails)}/{len(pool)} = {100 * rate:.2f}%  ({time.time() - t0:.1f}s)')
    for it in fails:
        print('  FAIL', it['pid'], repr(it['answer']), 'extracted:',
              repr(grader.extract_answer('\\boxed{' + it['answer'] + '}', it['answer_type'])))

    t0 = time.time()
    mv_fail = []
    for it in pool:
        if it['answer_type'] == 'math':
            g = grader._pre(it['answer'])
            gp = grader._call_with_timeout(lambda: grader._mv_parse(grader._strip_text_units(g), False),
                                           grader.TIMEOUT_S, ())
            if not gp or isinstance(gp[0], str):
                mv_fail.append(it)
    n_math = sum(it['answer_type'] == 'math' for it in pool)
    print(f'math-verify parses gold to a sympy object: {n_math - len(mv_fail)}/{n_math} '
          f'({time.time() - t0:.1f}s); golds graded by (normalized) string comparison only:')
    for it in mv_fail:
        print('  ', it['pid'], repr(it['answer']))

    t0 = time.time()
    pert_fail, n_pert = [], 0
    for it in pool:
        if it['answer_type'] != 'math':
            continue
        alt = perturb(it['answer'])
        n_pert += 1
        if not grader.grade('so the answer is \\boxed{' + alt + '}', it['answer'], 'math'):
            pert_fail.append((it['pid'], it['answer'], alt))
    print(f'perturbed self-consistency (equivalent rewrite of the gold): {n_pert - len(pert_fail)}/{n_pert} '
          f'({time.time() - t0:.1f}s)')
    for x in pert_fail:
        print('  ', x)

    t0 = time.time()
    mv_fail, n_mv = [], 0
    for it in pool:
        alt = perturb_mv(it['answer']) if it['answer_type'] == 'math' else None
        if alt is None:
            continue
        n_mv += 1
        if not grader.grade('so the answer is \\boxed{' + alt + '}', it['answer'], 'math'):
            mv_fail.append((it['pid'], it['answer'], alt))
    print(f'perturbed self-consistency through math-verify (gold -> 1 \\cdot \\left(gold\\right)): '
          f'{n_mv - len(mv_fail)}/{n_mv} ({time.time() - t0:.1f}s)')
    for x in mv_fail:
        print('  ', x)

    t0 = time.time()
    rng = random.Random(0)
    by_src = {}
    for it in pool:
        by_src.setdefault(it['source'], []).append(it)
    fp, n_neg = [], 0
    for it in pool:
        others = by_src[it['source']]
        for other in rng.sample(others, min(4, len(others))):
            if grader._norm_str(grader._pre(other['answer'])) == grader._norm_str(grader._pre(it['answer'])):
                continue
            n_neg += 1
            if grader.grade('\\boxed{' + other['answer'] + '}', it['answer'], it['answer_type']):
                fp.append((it['pid'], it['answer'], other['pid'], other['answer']))
    print(f'negative control: {len(fp)}/{n_neg} (gold, other gold of the same source) pairs graded equal '
          f'({time.time() - t0:.1f}s)')
    for x in fp:
        print('  ', x)
    return rate, fails


def perturb(g):
    """An equivalent rewrite of a gold answer (formatting only)."""
    if re.fullmatch(r'-?\d+', g):
        n = int(g)
        return f'{n:,}' if abs(n) >= 1000 else g + '.0'
    if re.fullmatch(r'-?\d+\.\d+', g):
        return g + '0'
    m = re.fullmatch(r'(-?\d+(?:\.\d+)?)e(-?\d+)', g)
    if m:
        return m.group(1) + ' \\times 10^{' + m.group(2) + '}'
    return g.replace('\\frac', '\\dfrac').replace(',', ', ') + ' '


def perturb_mv(g):
    """1 * (gold) for single-expression golds (no sets, tuples, intervals, equations, text): forces the
    math-verify / sympy path instead of the string shortcut. None if not applicable."""
    if any(x in g for x in ('=', ',', '\\in', '\\cup', '\\pm', '\\{', '[', ']', '\\text', '\\mbox', '%')):
        return None
    if g.lstrip('-').startswith('('):
        return None
    if re.fullmatch(r'\\?[A-Za-z]+', g) or any(x in g for x in ('<', '>', '\\le', '\\ge')):
        return None                     # bare symbols (math-verify quirk) and inequalities
    return '1 \\cdot \\left(' + g + '\\right)'


def main():
    n, case_fails = run_cases()
    rate, pool_fails = run_pool(sys.argv[1])
    ok = not case_fails and rate >= 0.995
    print('OK' if ok else 'NOT OK')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
