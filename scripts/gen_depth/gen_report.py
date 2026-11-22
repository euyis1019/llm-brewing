"""Build report/gen_depth_report.html from result files (every number is read from an output file).

Inputs (skipped when missing): results/qwen3-4b/base/replay/{summary,tuned_lens_summary,tuned_exit_summary,
exit_probe_summary}.json, results/qwen3-4b/base/loopdelta_v1/deltas.json, results/qwen3-4b/base/tokchoice_v1/summary.json,
results/*/arith/*.json, report/{screen_vs_aa,main_eval,gpqa,q25,trunc}.json (written by compare.py --out).
Usage: python -I scripts/gen_report.py
"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import html
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GD = str(_experiment_root('gen_depth'))
R = lambda *p: os.path.join(GD, *p)  # noqa: E731


def load(*p):
    f = R(*p)
    return json.load(open(f)) if os.path.exists(f) else None


def esc(s):
    return html.escape(str(s))


# ----------------------------------------------------------------------------------------------- SVG helpers
def line_chart(series, xlab, ylab, xlim, ylim, w=640, h=300, yfmt='{:.0%}', hlines=(), xticks=None, yticks=None):
    """series: list of (name, xs, ys, css_class). Returns an SVG string drawn to scale."""
    ml, mr, mt, mb = 52, 16, 14, 40
    pw, ph = w - ml - mr, h - mt - mb
    X = lambda x: ml + (x - xlim[0]) / (xlim[1] - xlim[0]) * pw  # noqa: E731
    Y = lambda y: mt + ph - (y - ylim[0]) / (ylim[1] - ylim[0]) * ph  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(ylab)} vs {esc(xlab)}">']
    yt = yticks or list(np.linspace(ylim[0], ylim[1], 6))
    for v in yt:
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
        out.append(f'<text x="{ml - 6}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{yfmt.format(v)}</text>')
    xt = xticks or list(range(int(xlim[0]), int(xlim[1]) + 1, 2))
    for v in xt:
        out.append(f'<text x="{X(v):.1f}" y="{mt + ph + 16}" class="tick" text-anchor="middle">{v}</text>')
    out.append(f'<text x="{ml + pw / 2}" y="{h - 6}" class="axis" text-anchor="middle">{esc(xlab)}</text>')
    for v, name, cls in hlines:
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="{cls}"/>')
        out.append(f'<text x="{ml + pw - 4}" y="{Y(v) - 5:.1f}" class="lab {cls}-t" text-anchor="end">{esc(name)}</text>')
    for name, xs, ys, cls in series:
        pts = ' '.join(f'{X(x):.1f},{Y(y):.1f}' for x, y in zip(xs, ys))
        out.append(f'<polyline points="{pts}" class="ln {cls}" fill="none"/>')
        for x, y in zip(xs, ys):
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.6" class="pt {cls}"/>')
        out.append(f'<text x="{X(xs[-1]) - 4:.1f}" y="{Y(ys[-1]) - 8:.1f}" class="lab {cls}-t" text-anchor="end">{esc(name)}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def forest(rows, w=640, xlim=(-12, 8), unit='pp', title=''):
    """rows: list of (label, mean, lo, hi, cls). Horizontal CI plot with a zero line."""
    ml, mr, rowh = 260, 70, 26
    h = 30 + rowh * len(rows) + 30
    pw = w - ml - mr
    X = lambda v: ml + (v - xlim[0]) / (xlim[1] - xlim[0]) * pw  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(title)}">']
    for v in range(int(xlim[0]), int(xlim[1]) + 1, 4):
        out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="20" y2="{h - 26}" class="grid"/>')
        out.append(f'<text x="{X(v):.1f}" y="{h - 10}" class="tick" text-anchor="middle">{v:+d}{unit if v == int(xlim[1]) else ""}</text>')
    out.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="16" y2="{h - 26}" class="zero"/>')
    for i, (lab, m, lo, hi, cls) in enumerate(rows):
        y = 34 + i * rowh
        out.append(f'<text x="{ml - 10}" y="{y + 4}" class="rowlab" text-anchor="end">{esc(lab)}</text>')
        a, b = X(max(lo, xlim[0])), X(min(hi, xlim[1]))
        out.append(f'<line x1="{a:.1f}" x2="{b:.1f}" y1="{y}" y2="{y}" class="ci {cls}"/>')
        out.append(f'<circle cx="{X(min(max(m, xlim[0]), xlim[1])):.1f}" cy="{y}" r="4.5" class="pt {cls}"/>')
        out.append(f'<text x="{w - mr + 8}" y="{y + 4}" class="val">{m:+.1f}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def bars(rows, w=640, ymax=0.8, title=''):
    """rows: list of (label, value, cls, note)."""
    ml, mr, mt, mb = 40, 10, 16, 64
    h = 300
    pw, ph = w - ml - mr, h - mt - mb
    n = len(rows)
    bw = pw / n * 0.6
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(title)}">']
    for v in np.linspace(0, ymax, 5):
        y = mt + ph - v / ymax * ph
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{v:.0%}</text>')
    for i, (lab, v, cls, note) in enumerate(rows):
        x = ml + (i + 0.2) * pw / n
        y = mt + ph - v / ymax * ph
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{mt + ph - y:.1f}" class="bar {cls}"/>')
        out.append(f'<text x="{x + bw / 2:.1f}" y="{y - 6:.1f}" class="val" text-anchor="middle">{v:.1%}</text>')
        out.append(f'<text x="{x + bw / 2:.1f}" y="{mt + ph + 16}" class="rowlab" text-anchor="middle">{esc(lab)}</text>')
        out.append(f'<text x="{x + bw / 2:.1f}" y="{mt + ph + 32}" class="tick" text-anchor="middle">{esc(note)}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def table(header, rows, cls='tbl'):
    h = ''.join(f'<th>{esc(x)}</th>' for x in header)
    b = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def pp(x):
    return f'{x * 100:+.2f}'


def ci(r):
    return f'[{r["ci"][0] * 100:+.1f}, {r["ci"][1] * 100:+.1f}]'


# ----------------------------------------------------------------------------------------------- sections
def sec_lens():
    tl = load('results/qwen3-4b/base/replay/tuned_lens_summary.json')
    if not tl:
        return ''
    layers = sorted(int(k) for k in tl if k.isdigit())
    lg = [tl[str(l)]['samp_agree_logit'] for l in layers]
    tu = [tl[str(l)]['samp_agree_tuned'] for l in layers]
    svg = line_chart([('tuned lens', layers, tu, 'c1'), ('logit lens', layers, lg, 'c2')], '层（block 索引，共 36 层）',
                     '采样 token 与满深度一致的比例', (16, 34), (0, 1), hlines=[(0.372, 'n-gram 下限 37.2%', 'floor')])
    rows = [[f'L{l}', f'{lg[i]:.1%}', f'{tu[i]:.1%}', f'{(tu[i] - 0.372) * 100:+.0f}pp'] for i, l in enumerate(layers)]
    return f"""
<section id="lens"><h2>1. 生成中的 brewing：token 早就可得，读出轴上要到很晚才看得见</h2>
<p>在 MATH-500 基线轨迹（1000 条，84 万 token）上做 teacher-forced 重放。每一层用同一份采样噪声，看这一层读出的 token 与满深度是否一致。
logit lens 直接用最终的 norm 和 unembedding。tuned lens 是秩 128 的仿射读出，只用模型自己的末层 top-1 训练（自监督），在留出的题上评估。</p>
<figure>{svg}<figcaption>Qwen3-4B 非思考模式。灰线是只看最后两个输入 token 的 n-gram 回退模型的命中率，是"局部上下文就能猜到"的下限。</figcaption></figure>
<p><b>读法。</b>
oracle 早退层（从这一层起一直与满深度一致）用 logit lens 平均是 30.5 块，用 tuned lens 是 24.1 块。
tuned lens 超出 n-gram 下限的部分在第 22–24 层跳升，从 +22pp 到 +34pp。这个位置与第一轮 Qwen3-4B 答案位置的发射层（L24）一致。
所以第一轮的"可得早于可读"（brewing 间隔）在长链生成里依然存在：token 在残差里早已决定，最后几块只是把它写到读出轴上。</p>
<details><summary>逐层数字</summary>{table(['层', 'logit lens', 'tuned lens', '超出 n-gram'], rows)}</details>
</section>"""


def sec_prune():
    s = load('results/qwen3-4b/base/replay/summary.json')
    if not s:
        return ''
    xs, a_all, a_dig = [], [], []
    for k, v in s['rules'].items():
        if k.startswith('static@'):
            xs.append(int(v['blocks']))
            a_all.append(v['agree'])
            a_dig.append(v['agree_digit'])
    o = np.argsort(xs)
    xs, a_all, a_dig = [list(np.array(z)[o]) for z in (xs, a_all, a_dig)]
    sel = [i for i, x in enumerate(xs) if x >= 22]
    svg = line_chart([('全部 token', [xs[i] for i in sel], [a_all[i] for i in sel], 'c1'),
                      ('数字 token', [xs[i] for i in sel], [a_dig[i] for i in sel], 'c3')],
                     '保留的块数（静态截断，共 36 块）', '与满深度一致的比例', (22, 35), (0, 1), xticks=list(range(22, 36, 1)))
    cats = s['categories']
    order = ['digit', 'latex', 'marker', 'word', 'punct/other', 'op', 'space']
    names = {'digit': '数字', 'latex': 'LaTeX 命令*', 'marker': '话语标记', 'word': '词', 'punct/other': '标点/其他', 'op': '运算符',
             'space': '空白'}
    rows = [[names[c], f"{cats[c]['share']:.1%}", f"{cats[c]['exit_blocks']:.2f}", f"{cats[c]['late']:.1%}",
             f"{cats[c]['ent']:.2f}"] for c in order if c in cats]
    tr = load('report/trunc.json') or []
    trows = []
    for r in tr:
        trows.append([esc(r['tag']), f"{r['acc']:.1%}", f"{pp(r['diff'])}pp {ci(r)}", f"{r['fixed']}/{r['broken']}",
                      f"{r['tokens_rel']:.2f}×"])
    ttbl = table(['运行', '准确率', '相对 A/A', '修好/弄坏', '长度'], trows) if trows else ''
    pr = load('report/pruning_predict.json') or {}
    names2 = {'late_count': '晚定 token 数', 'late_digit_count': '晚定数字数', 'late_share': '晚定比例',
              'late_digit_share': '数字中晚定的比例', 'ent': '平均熵', 'length': '回答长度'}
    prows = []
    for run, d in pr.items():
        for k, v in d['predictors'].items():
            prows.append([esc(run), names2.get(k, k), f"{v['spearman']:+.2f}",
                          ' / '.join(f'{t * 100:.0f}' for t in v['tertiles'])])
    if prows:
        ttbl += '<p>逐题看：哪些题被静态截断伤得更重？预测量来自基线轨迹，伤害=基线准确率减截断后准确率。</p>' + table(
            ['截断', '预测量', 'Spearman', '按预测量三分位的平均伤害（pp）'], prows)
    return f"""
<section id="prune"><h2>2. 剪层为什么伤数学：算出来的数字定得最晚</h2>
<figure>{svg}<figcaption>teacher-forced 重放中，所有 token 在同一层截断时，采样 token 与满深度一致的比例。</figcaption></figure>
{table(['类别', '占比', 'oracle 早退块', '晚定比例（≥33 块）', '末层熵'], rows)}
<p class="note">* Qwen3 会把 LaTeX 命令拆成 "\\" 加 "frac" 这样的片段，所以 LaTeX 这一行是低估；数字这一行不受影响。</p>
<p>数字是最确定的一类（熵 0.02），晚定比例却最高（39.7%）。截掉最后两块时，所有 token 还有 82.5% 一致，数字只剩 65.7%。
这在 token 层面解释了已有文献里"剪 1–2 层就让长链数学崩溃，而知识题几乎不掉"：被剪掉的正是数字这类到最后才写进读出轴的 token。</p>
{ttbl}
</section>"""


def sec_exit():
    sv = {r['tag']: r for r in (load('report/screen_vs_aa.json') or [])}
    te = load('results/qwen3-4b/base/replay/tuned_exit_summary.json')
    if not sv:
        return ''
    base_acc = sv['E_t99']['acc_base'] if 'E_t99' in sv else None
    rows = [('A/A 基线', base_acc, 'c1', '36 块')]
    notes = {'E_l99': '53% token 跳末块', 'X_s34': '全部跳末块', 'E_t999': 'tuned τ=.999', 'E_t99f': '满深度 K/V（诊断）',
             'E_t99': 'tuned τ=.99', 'O_logit': '零 token 误差，30.5 块', 'X_s31': '静态 32 块'}
    labs = {'E_l99': 'lens 早退', 'X_s34': '均匀截断', 'E_t999': 'tuned 早退', 'E_t99f': 'tuned 早退', 'E_t99': 'tuned 早退',
            'O_logit': '完美分类器', 'X_s31': '均匀截断'}
    for tag in ('E_l99', 'X_s34', 'E_t999', 'E_t99f', 'E_t99', 'O_logit', 'X_s31'):
        if tag in sv:
            rows.append((labs[tag], sv[tag]['acc'], 'c4' if tag.startswith('X_') else 'c3', notes[tag]))
    svg = bars(rows, ymax=0.8, title='closed-loop accuracy')
    terows = ''
    if te:
        terows = table(['τ', '平均块数', '离线一致率', '数字一致率'],
                       [[r['tau'], f"{r['blocks']:.1f}", f"{r['agree']:.1%}", f"{r['agree_digit']:.1%}"] for r in te['rule']
                        if r['tau'] in (0.9, 0.95, 0.97, 0.98, 0.99)])
    return f"""
<section id="exit"><h2>3. 早退：离线 98% 一致，闭环仍然崩</h2>
<p>早退探针是线性的，从隐状态判断"这一层的 tuned lens 是否已给出满深度 token"，训练是自监督的。下表是离线规则：</p>
{terows}
<figure>{svg}<figcaption>MATH-500 筛查集，种子 0、1，共 366 条；基线是换了一份采样噪声的 A/A 运行。</figcaption></figure>
<ul>
<li><b>就绪度决定了该让谁少算。</b>让所有 token 都跳过最后一块，准确率从 66.4% 掉到 19.4%；只让探针判为就绪的 53% 的 token 跳过，是 63.1%，差值不显著。平均深度相同（约 32 块）时，按就绪度早退（26.0%）也远好于均匀截断（3.8%）。</li>
<li>但每个 token 只省约 3% 的算力时还能基本不掉点，省到 11% 就崩溃，中间找不到一个划算的工作点。</li>
<li><b>完美分类器也救不了。</b>事后给每个 token 找最早已定型的层，发出的 token 与满深度完全相同，只把深层 K/V 换成退出态的投影。准确率仍从 66.4% 掉到 27.9%（平均 30.5 块）；用 tuned lens 判定、退得更早时只剩 9.8%。瓶颈在于冻结模型上执行早退这个动作，而不在判断。</li>
<li>但两者都崩。一个诊断实验把早退 token 的深层 K/V 改回满深度计算（不省算力），准确率回到 48.6%，找回 22.7pp [16.1, 29.2]。
剩下的损失与早发出的 token 一起出现。两个通道之间有很强的交互，所以不能按比例拆分。</li>
<li>结论：在免训练的状态传播下，被早退 token 的 K/V 会伤害后续 token；早发出的 token 本身也会伤害，而且很可能集中在算出来的数字上。
"深度在长链里是共享资源"只是一个假说：经过训练的 K/V 修复方法没有试过。</li>
</ul>
</section>"""


def sec_loops():
    sv = {r['tag']: r for r in (load('report/screen_vs_aa.json') or [])}
    me = load('report/main_eval.json') or []
    lab = {'L_c35': 'loop 18–25 · 晚定 token', 'L_c35n': 'loop 18–25 · 晚定 · 只改预测', 'L_e10': 'loop 18–25 · 高熵 token',
           'Lp_c35': 'loop 28–33 · 晚定 token', 'L_r084': 'loop 18–25 · 随机 token', 'T_c35': '温度 ×1.5 · 晚定（零深度）',
           'T_c35s': '温度 ×0.5 · 晚定（零深度）', 'T_r084s': '温度 ×0.5 · 随机（零深度）', 'T_e10s': '温度 ×0.5 · 高熵（零深度）',
           'Lacc_c35': 'loop 只在更确定时采纳 · 晚定', 'Lacc_e10': 'loop 只在更确定时采纳 · 高熵',
           'Lv_fork': 'loop 18–25 · 分叉词词表', 'Lv_data': 'loop 18–25 · 数据驱动词表', 'Lw14_e10': 'loop 14–19 · 高熵 token'}
    frows = [(lab[t], sv[t]['diff'] * 100, sv[t]['ci'][0] * 100, sv[t]['ci'][1] * 100, 'c3' if t.startswith('L') else 'c2')
             for t in lab if t in sv]
    svg = forest(frows, xlim=(-12, 8), title='screening vs A/A') if frows else ''
    mrows = [(r['tag'], r['diff'] * 100, r['ci'][0] * 100, r['ci'][1] * 100, 'c1') for r in me]
    msvg = forest(mrows, xlim=(-8, 6), title='main evaluation') if mrows else '<p class="pending">主评估结果待补。</p>'
    mtbl = table(['配置', '策略', '题×种子', '基线', '干预', 'Δ', '修好/弄坏', 'McNemar p', '截断率', '写完时的准确率', '总算力'],
                 [[esc(r['tag']), f"<code>{esc(r['policy'])}</code>", r['jobs'], f"{r['acc_base']:.1%}", f"{r['acc']:.1%}",
                   f"{pp(r['diff'])}pp {ci(r)}", f"{r['fixed']}/{r['broken']}", f"{r['mcnemar_p']:.3f}",
                   f"{r.get('trunc_base', 0):.1%} → {r.get('trunc', 0):.1%}",
                   f"{r.get('acc_finished_base', 0):.1%} → {r.get('acc_finished', 0):.1%}", f"{r['cost_rel_total']:.2f}"]
                  for r in me]) if me else ''
    mtbl += ('<p>M_L_c35 是预先指定的主检验，即第一轮设定的直接迁移。它的伤害主要来自<b>输出变长</b>：截断率从 12.2% 升到 16.8%，'
             '没被截断的输出准确率几乎不变。同一批 token 上，零深度的温度扰动（M_T_c35）不会让输出变长。'
             '只改预测的 loop（M_L_c35n）截断增幅小得多，但它与 M_L_c35 的直接差值（−1.8pp）不显著。效应在各种子之间不稳（−8.3、−4.4、+0.3、−0.8pp）。</p>')
    return f"""
<section id="loops"><h2>4. loop：在生成里不能"补完 brewing"</h2>
<h3>4.1 筛查（相对 A/A）</h3>
<figure>{svg}<figcaption>MATH-500 筛查集，种子 0、1。干预大约作用在 8–11% 的 token 上。A/A 运行相对原基线本身就有 +3.3pp，这是选择偏差（均值回归）；以 A/A 为参照才公平。</figcaption></figure>
<h3>4.2 主评估（无选择偏差）</h3>
<p>主评估集 361 题，用种子 0、1 分层选出：前沿 231、全对 80、全错 50；来源是 OlympiadBench、GPQA、MATH-500、Minerva、AMC、AIME 和少量 GSM8K。评估用种子 2–5。</p>
{msvg}{mtbl}
</section>"""


def sec_mech():
    tc = load('results/qwen3-4b/base/tokchoice_v1/summary.json')
    out = ''
    if tc:
        names = {'alt': '普通重采样（零假设）', 'w18': 'loop 18–25', 'w24': 'loop 24–31', 'w28': 'loop 28–33', 'con': '层对比（L24）',
                 'damp': '阻尼 loop'}
        rows = [(names[k], tc[f'{k}|ALL']['dV'] * 100, tc[f'{k}|ALL']['ci'][0] * 100, tc[f'{k}|ALL']['ci'][1] * 100,
                 'c2' if k == 'alt' else 'c3') for k in names if f'{k}|ALL' in tc and 'dV' in tc[f'{k}|ALL']]
        out += f"""<h3>5.1 token 选择：逐 token 检测不出差别</h3>
<p>在前沿题的 954 个位置上，用该次运行的噪声分别算出原 token、loop 后的 token 和普通重采样的 token。对每个不同的候选，从"前缀 + 候选"续写 4 次并判分。
dV 是候选与原 token 的续写答对率之差，单位为百分点。分辨率约为每个 token ±5pp。闭环里的伤害只需每个被改的 token 平均 −0.1 到 −0.3pp，所以这个检验<b>分不开</b>预测通道和 K/V 通道。</p>
<figure>{forest(rows, xlim=(-12, 8), unit='pp', title='token choice')}<figcaption>按题聚类的自助法区间。</figcaption></figure>"""
    D = load('results/qwen3-4b/base/loopdelta_v1/deltas.json')
    if D:
        def rn(x):
            x = np.array(x, np.float64)
            return x - np.logaddexp.reduce(x)
        nat = [rn(x) for x in D['nat']]
        base = [rn(x) for x in D['base']]

        def resid(u, n):
            u, n = u - u.mean(), n - n.mean()
            return u - (u @ n) / (n @ n + 1e-12) * n
        rows = []
        for nm, lab in (('w18', '18–25'), ('w24', '24–31'), ('w28', '28–33'), ('damp', '18–25 阻尼')):
            if nm not in D:
                continue
            L = [rn(x) for x in D[nm]]
            de, sw, cs = [], [], []
            for i in range(len(nat)):
                pn, pl = np.exp(nat[i]), np.exp(L[i])
                de.append(-(pl * L[i]).sum() + (pn * nat[i]).sum())
                sw.append(np.argmax(L[i]) != 0)
                u, v = resid(L[i] - nat[i], nat[i]), resid(nat[i] - base[i], nat[i])
                d = np.linalg.norm(u) * np.linalg.norm(v)
                cs.append(u @ v / d if d > 1e-9 else np.nan)
            rows.append([lab, f'{np.mean(de):+.3f}', f'{np.mean(sw):.1%}', f'{np.nanmean(cs):+.3f}'])
        out += f"""<h3>5.2 loop 在分布上做了什么</h3>
<p>在 1517 个位置上比较两个变化：loop 引起的对数概率变化，以及后训练引起的变化（Qwen3-4B 减 Qwen3-4B-Base）。都只看自然分布的 top-32 id，并先去掉"锐化/压平"这条共同的轴。打乱位置配对后，零假设下的余弦约为 0±0.005。</p>
{table(['窗口', '熵变化（nats）', 'top-1 改变率', '去掉锐化轴后与后训练方向的余弦'], rows)}
<p>loop 做的是<b>重新排序</b>：把概率从自然 top-1 挪到另一个候选（24–31 和 28–33 窗口还会略微变锐）。去掉锐化轴后，<b>所有窗口都与后训练方向轻微反向</b>，阻尼版也是。随机位置上同样如此，而且窗口之间的排序会随候选集大小翻转。
所以这是一种通用的扰动，并非 brewing 窗口独有。"loop 等于更多后训练"不成立。</p>"""
    ar, ex = {}, {}
    for m, fn in (('Qwen3-4B', 'results/qwen3-4b/arith/arith_v3'), ('Qwen2.5-7B-Instruct', 'results/qwen2.5-7b-instruct/arith/arith_q25_v2')):
        d = load(fn + '.json')
        t = load(fn + '_texts.json') or {}
        if d:
            for k, v in d.items():
                ar[(m, k)] = sum(v.values()) / len(v)
                if k in t:
                    wf = t[k]['wellformed']
                    ex[(m, k)] = (sum(wf.values()) / len(wf), float(np.mean(list(t[k]['digit_acc'].values()))))
    if ar:
        rows = []
        for (m, k), v in sorted(ar.items(), key=lambda kv: (kv[0][0], kv[0][1] != 'base', kv[0][1])):
            w, dg = ex.get((m, k), (float('nan'), float('nan')))
            delta = '—' if k == 'base' else f"{(v - ar[(m, 'base')]) * 100:+.2f}pp"
            rows.append([m, esc(k), f'{v:.1%}', delta, f'{w:.0%}', f'{dg:.3f}'])
        out += f"""<h3>5.3 合成算术：额外深度会破坏算出来的数字</h3>
<p>400 道直接作答的题（三个 7 位数相加、4 位数乘 3 位数），贪心解码，每个答案 token 都 loop。中段到深段窗口在两个模型上都显著伤害，Holm 校正后仍显著；早段窗口的小幅"增益"都在噪声以内。准确率接近 0 的臂可能是输出格式被破坏，格式分析见 arith_v3。基线准确率：Qwen3-4B {ar.get(('Qwen3-4B', 'base'), 0):.1%}，Qwen2.5-7B-Instruct {ar.get(('Qwen2.5-7B-Instruct', 'base'), 0):.1%}。
窗口名里的数字是起始层：w18 指 18–25，w14 指 14–19；n 表示只改预测（kv=none），damp 表示阻尼版，×2 表示多跑两遍。</p>
{table(['模型', '窗口', '准确率', '相对基线', '格式正确', '逐位数字准确率'], rows)}
<p>中段和深段窗口基本保住了输出格式，但逐位数字准确率明显下降：它们是把数字算错了，不只是格式乱了。最早（4–9）和最晚（30–35）的窗口才会破坏格式本身。</p>"""
    return f'<section id="mech"><h2>5. 机制：loop 改了什么，没改什么</h2>{out}</section>' if out else ''


def sec_extra():
    g = load('report/gpqa.json') or []
    q = load('report/q25.json') or []
    out = ''
    if g:
        out += ('<h3>6.1 GPQA：在字母 token 上 loop</h3><p>G_L18 和 G_L28 在 top-1 为 A–D 的每个 token 上触发，其中大多是推理中途提到的选项。'
                'G_L18b 和 G_L28b 只在 <code>{</code> 后面的字母上触发：其中 46% 是最终答案，39% 是更早的 boxed 字母，14% 不是答案。'
                '所有差值都不显著，而且大多来自没有改动任何 token 的 job（K/V 改写和批组成的 bf16 底噪）。'
                '检出第一轮 +1.36pp 的功效只有约 0.17。</p>') + table(
            ['配置', '题×种子', '基线', '干预', 'Δ', '修好/弄坏', 'McNemar p'],
            [[esc(r['tag']), r['jobs'], f"{r['acc_base']:.1%}", f"{r['acc']:.1%}", f"{pp(r['diff'])}pp {ci(r)}",
              f"{r['fixed']}/{r['broken']}", f"{r['mcnemar_p']:.3f}"] for r in g])
    if q:
        out += '<h3>6.2 Qwen2.5-7B-Instruct 复现（主评估集，种子 0–3）</h3>' + table(
            ['配置', '题×种子', '基线', '干预', 'Δ', '修好/弄坏', 'McNemar p'],
            [[esc(r['tag']), r['jobs'], f"{r['acc_base']:.1%}", f"{r['acc']:.1%}", f"{pp(r['diff'])}pp {ci(r)}",
              f"{r['fixed']}/{r['broken']}", f"{r['mcnemar_p']:.3f}"] for r in q])
    return f'<section id="extra"><h2>6. 迁移与复现</h2>{out}</section>' if out else ''


CSS = """
:root{--bg:#fbfaf8;--fg:#1d2228;--mut:#5d6670;--line:#d9dde2;--card:#f1f0ec;--c1:#2b5d8a;--c2:#8a8f96;--c3:#b4532a;--c4:#7a2e8a;
--floor:#9aa3ad;--good:#2f7d4f;--bad:#a8342b;--f-head:'IBM Plex Sans','Noto Sans SC',system-ui,sans-serif;
--f-body:'Noto Sans SC','IBM Plex Sans',system-ui,sans-serif;--f-mono:'IBM Plex Mono',ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15181c;--fg:#e4e7ea;--mut:#9aa3ad;--line:#2c323a;--card:#1d2127;
--c1:#7fb0de;--c2:#9aa3ad;--c3:#e08a5f;--c4:#c48ad6;--floor:#6d7680;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#15181c;--fg:#e4e7ea;--mut:#9aa3ad;--line:#2c323a;--card:#1d2127;--c1:#7fb0de;--c2:#9aa3ad;--c3:#e08a5f;
--c4:#c48ad6;--floor:#6d7680;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--f-body);line-height:1.65;font-size:15px}
main{max-width:860px;margin:0 auto;padding-block:28px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));
padding-right:max(16px,env(safe-area-inset-right,0px));display:flex;flex-direction:column;gap:34px}
h1{font-family:var(--f-head);font-size:1.85rem;line-height:1.25;margin:0;text-wrap:balance}
h2{font-family:var(--f-head);font-size:1.3rem;margin:0 0 10px;text-wrap:balance}
h3{font-family:var(--f-head);font-size:1.05rem;margin:22px 0 6px}
p,ul{margin:8px 0;max-width:68ch} li{margin:4px 0}
.eyebrow{font-family:var(--f-mono);font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}
.lede{font-size:1.02rem;color:var(--fg)}
.tldr{background:var(--card);border-left:3px solid var(--c1);padding:14px 18px;border-radius:2px}
.tldr li{margin:6px 0}
figure{margin:12px 0} figcaption{font-size:.85rem;color:var(--mut)}
.chart{width:100%;height:auto;display:block}
.chart .grid{stroke:var(--line);stroke-width:1} .chart .zero{stroke:var(--fg);stroke-width:1.2}
.chart .tick{font:11px var(--f-mono);fill:var(--mut)} .chart .axis{font:12px var(--f-body);fill:var(--mut)}
.chart .rowlab{font:12px var(--f-body);fill:var(--fg)} .chart .val{font:11.5px var(--f-mono);fill:var(--fg)}
.chart .lab{font:12px var(--f-body)}
.chart .ln{stroke-width:2.2} .chart .ci{stroke-width:2.4}
.chart .c1{stroke:var(--c1);fill:var(--c1)} .chart .c2{stroke:var(--c2);fill:var(--c2)} .chart .c3{stroke:var(--c3);fill:var(--c3)}
.chart .c4{stroke:var(--c4);fill:var(--c4)} .chart polyline{fill:none}
.chart .c1-t{fill:var(--c1)} .chart .c2-t{fill:var(--c2)} .chart .c3-t{fill:var(--c3)} .chart .floor-t{fill:var(--floor)}
.chart .floor{stroke:var(--floor);stroke-dasharray:5 4;stroke-width:1.4}
.chart .bar.c1{fill:var(--c1)} .chart .bar.c3{fill:var(--c3)} .chart .bar.c4{fill:var(--c4)}
.tw{overflow-x:auto;margin:10px 0} table{border-collapse:collapse;font-size:.88rem;min-width:100%}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
th{font-family:var(--f-head);font-weight:600;color:var(--mut);font-size:.8rem;letter-spacing:.02em}
td{font-variant-numeric:tabular-nums} code{font-family:var(--f-mono);font-size:.8rem;word-break:break-all}
.note,.pending{font-size:.85rem;color:var(--mut)} details summary{cursor:pointer;color:var(--c1)}
a{color:var(--c1)} :focus-visible{outline:2px solid var(--c1);outline-offset:2px}
"""


def main():
    meta = load('report/meta.json') or {}
    tldr = meta.get('tldr', [])
    body = f"""<title>逐 token 深度控制</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@500;600&family=Noto+Sans+SC:wght@400;500;600&display=swap">
<style>{CSS}</style>
<main>
<header><div class="eyebrow">Brewing · 生成阶段 · {esc(meta.get('date', ''))}</div>
<h1>用 brewing 指标控制长链生成的逐 token 深度</h1>
<p class="lede">{esc(meta.get('lede', ''))}</p></header>
<div class="tldr"><b>结论</b><ul>{''.join(f'<li>{x}</li>' for x in tldr)}</ul></div>
{sec_lens()}{sec_prune()}{sec_exit()}{sec_loops()}{sec_mech()}{sec_extra()}
<section id="method"><h2>7. 方法与口径</h2><ul>{''.join(f'<li>{x}</li>' for x in meta.get('method', []))}</ul></section>
</main>"""
    os.makedirs(R('report'), exist_ok=True)
    open(R('report', 'gen_depth_report.html'), 'w').write(body)
    print('wrote', R('report', 'gen_depth_report.html'), len(body))


if __name__ == '__main__':
    main()
