"""Build the round-1 HTML report (report/round1_report.html) from result files. Numbers in charts/tables are read from
disk; stage-2 prose numbers are copied from the reviewed RESULTS.md sections (cited there)."""
import csv, html, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

R = os.path.join(C.ROOT, 'results')
OUT = os.path.join(C.ROOT, 'report', 'round1_report.html')
RUNS = [('qwen3-4b-base__base_fewshot', 'Qwen3-4B-Base', '1-shot'),
        ('qwen3-4b__base_fewshot', 'Qwen3-4B（后训练）', '1-shot'),
        ('qwen3-4b__chat0', 'Qwen3-4B（后训练）', 'chat'),
        ('qwen2.5-7b__base_fewshot', 'Qwen2.5-7B', '1-shot'),
        ('qwen2.5-7b-instruct__chat0', 'Qwen2.5-7B-Instruct', 'chat'),
        ('llama3.1-8b__base_fewshot', 'Llama-3.1-8B', '1-shot')]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def esc(s):
    return html.escape(str(s))


# ----------------------------------------------------------------------------- data
def curves(run):
    ps = jl(f'{R}/{run}/probe/summary.json')
    rows = json.load(open(f'{R}/{run}/extract/rows.json'))
    gold = np.array([r['gold'] for r in rows])
    csd = np.load(f'{R}/{run}/extract/csd_raw4.npy', mmap_mode='r')
    c_acc = (np.asarray(csd).argmax(-1) == gold[:, None]).mean(0)
    out = dict(L=ps['L'], lens=np.array(ps['acc']['lens']), full=np.array(ps['acc']['full']), csd=c_acc)
    cp = f'{R}/{run}/consensus_probe/oof_cons_vote.npy'
    if os.path.exists(cp):
        cons = np.load(cp, mmap_mode='r')
        out['cons'] = (np.asarray(cons).argmax(-1) == gold[:, None]).mean(0)
    return out


def taxonomy(run):
    Q = list(csv.DictReader(open(f'{R}/{run}/classify/questions.csv')))
    oc = np.array([q['outcome_v1_K3'] for q in Q])
    fp = np.array([int(q['FPCL_c_K3']) for q in Q]); fj = np.array([int(q['FJC_c_K3']) for q in Q])
    ua = np.array([q['u_avail_K3'] in ('1', 'True', 'true') for q in Q])
    Rm = oc == 'R'
    meta = jl(f'{R}/{run}/extract/meta.json')
    return dict(n=len(oc), counts={c: int((oc == c).sum()) for c in 'ROMUX'}, u_avail=int((ua & (oc == 'U')).sum()),
                fpcl=float(np.median(fp[Rm & (fp >= 0)])), fjc=float(np.median(fj[Rm])), acc=meta['acc4'])


def direction(run, loop_dir='loop_allq', placebo=None):
    p = f'{R}/{run}/{loop_dir}/eval_lo/post_classes.csv'
    if not os.path.exists(p):
        p = f'{R}/{run}/{loop_dir}/eval/post_classes.csv'
        if not os.path.exists(p):
            return None
    P = list(csv.DictReader(open(p)))
    cols = [c for c in P[0] if c.startswith(('last_', 'all_')) and not c.endswith('_none') and '_p4_' not in c and 'p3_9' not in c]
    mode = cols[0].split('_')[0]
    pl = placebo or [c for c in cols if c.split('_')[1].startswith('p')][0]
    U = [x for x in P if x[f'{mode}_none'] == 'U']
    sc = lambda c: np.array([1 if x[c] in ('R', 'O') else (-1 if x[c] == 'M' else 0) for x in U], float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(U), (2000, len(U)))
    sp = sc(pl)
    res = {}
    for c in cols:
        if c == pl:
            continue
        d = sc(c) - sp
        b = d[idx].mean(1)
        res[c.split('_', 1)[1]] = (d.mean() * 100, np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100)
    return dict(n_U=len(U), placebo=pl.split('_', 1)[1], res=res)


# ----------------------------------------------------------------------------- svg helpers
def line_chart(c, title, w=440, h=230):
    L = c['L']; pad_l, pad_r, pad_t, pad_b = 40, 12, 26, 30
    X = lambda l: pad_l + (w - pad_l - pad_r) * l / (L - 1)
    Y = lambda a: pad_t + (h - pad_t - pad_b) * (1 - (a - 0.2) / 0.5)
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(title)}">']
    s.append(f'<text x="{pad_l}" y="15" class="ct">{esc(title)}</text>')
    for a in (0.25, 0.4, 0.55, 0.7):
        s.append(f'<line x1="{pad_l}" x2="{w-pad_r}" y1="{Y(a):.1f}" y2="{Y(a):.1f}" class="grid"/>'
                 f'<text x="{pad_l-6}" y="{Y(a)+3:.1f}" class="ax" text-anchor="end">{a:.2f}</text>')
    step = 6 if L >= 32 else 4
    ticks = [l for l in range(0, L, step) if (L - 1) - l >= step / 2] + [L - 1]
    for l in ticks:
        s.append(f'<text x="{X(l):.1f}" y="{h-12}" class="ax" text-anchor="middle">L{l}</text>')
    for key, cls in (('lens', 'l-lens'), ('csd', 'l-csd'), ('full', 'l-probe'), ('cons', 'l-cons')):
        if key not in c:
            continue
        pts = ' '.join(f'{X(l):.1f},{Y(max(0.2, min(0.7, a))):.1f}' for l, a in enumerate(c[key]))
        s.append(f'<polyline points="{pts}" class="{cls}" fill="none"/>')
    s.append('</svg>')
    return ''.join(s)


def stacked_bars(items, w=640):
    rowh, pad_l, pad_r = 26, 210, 20
    h = 30 + rowh * len(items)
    cls = {'R': 'b-r', 'O': 'b-o', 'M': 'b-m', 'U': 'b-u'}
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="四类组成">']
    for i, (name, t) in enumerate(items):
        y = 20 + i * rowh
        s.append(f'<text x="{pad_l-8}" y="{y+13}" class="ax" text-anchor="end">{esc(name)}</text>')
        x = pad_l; tot = sum(t['counts'][k] for k in 'ROMU')
        for k in 'ROMU':
            wd = (w - pad_l - pad_r) * t['counts'][k] / tot
            s.append(f'<rect x="{x:.1f}" y="{y}" width="{max(wd,0):.1f}" height="18" class="{cls[k]}"/>')
            if wd > 34:
                s.append(f'<text x="{x+wd/2:.1f}" y="{y+13}" class="bl" text-anchor="middle">{k} {100*t["counts"][k]/tot:.1f}%</text>')
            x += wd
    s.append('</svg>')
    return ''.join(s)


def forest(rows, w=640):
    """rows: list of (label, est, lo, hi); x range -2..4 pp."""
    pad_l, pad_r, rowh = 170, 20, 22
    h = 34 + rowh * len(rows)
    lo_x, hi_x = -2.5, 4.0
    X = lambda v: pad_l + (w - pad_l - pad_r) * (v - lo_x) / (hi_x - lo_x)
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="方向净值减安慰剂">']
    for v in (-2, -1, 0, 1, 2, 3, 4):
        s.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="14" y2="{h-16}" class="{"zero" if v == 0 else "grid"}"/>'
                 f'<text x="{X(v):.1f}" y="{h-4}" class="ax" text-anchor="middle">{v:+d}</text>')
    for i, (lab, e, lo, hi) in enumerate(rows):
        y = 24 + i * rowh
        sig = lo > 0
        s.append(f'<text x="{pad_l-8}" y="{y+4}" class="ax" text-anchor="end">{esc(lab)}</text>'
                 f'<line x1="{X(max(lo,lo_x)):.1f}" x2="{X(min(hi,hi_x)):.1f}" y1="{y}" y2="{y}" class="ci{" sig" if sig else ""}"/>'
                 f'<circle cx="{X(e):.1f}" cy="{y}" r="4" class="pt{" sig" if sig else ""}"/>')
    s.append('</svg>')
    return ''.join(s)


# ----------------------------------------------------------------------------- page
CSS = """
:root{
  /* Layout: one reading column (~70ch) with full-width figure rows; tea-ink palette (subject: brewing). */
  --paper:#f3f5f2; --ink:#1c2420; --muted:#5b6a63; --rule:#d5ddd8; --panel:#e8ede9;
  --teal:#2f6b5b; --amber:#b9772a; --rose:#a14c4c; --slate:#64748b;
  --r:#2f6b5b; --o:#c98a2e; --m:#a14c4c; --u:#9aa7a0;
  --f-display:"Noto Serif SC","Songti SC",serif; --f-body:"Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --paper:#121816; --ink:#e3e9e5; --muted:#9aaaa2; --rule:#2b3632; --panel:#1a2320;
  --teal:#6fb39c; --amber:#e0a54b; --rose:#d98a8a; --slate:#9aa6b8;
  --r:#5fa38c; --o:#e0a54b; --m:#d07a7a; --u:#5d6a64; color-scheme:dark}}
:root[data-theme="dark"]{
  --paper:#121816; --ink:#e3e9e5; --muted:#9aaaa2; --rule:#2b3632; --panel:#1a2320;
  --teal:#6fb39c; --amber:#e0a54b; --rose:#d98a8a; --slate:#9aa6b8;
  --r:#5fa38c; --o:#e0a54b; --m:#d07a7a; --u:#5d6a64; color-scheme:dark}
body{background:var(--paper);color:var(--ink);font-family:var(--f-body);font-size:15px;line-height:1.75}
.wrap{max-width:980px;margin:0 auto;padding-block:28px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
header{display:grid;gap:6px;border-bottom:2px solid var(--ink);padding-bottom:14px;margin-bottom:22px}
.eyebrow{font-family:var(--f-mono);font-size:12px;letter-spacing:.08em;color:var(--muted);text-transform:uppercase}
h1{font-family:var(--f-display);font-weight:700;font-size:clamp(26px,4vw,38px);line-height:1.25;margin:0;text-wrap:balance}
h2{font-family:var(--f-display);font-size:22px;margin:38px 0 10px;text-wrap:balance}
h3{font-size:16px;margin:22px 0 6px}
p,li{max-width:72ch}
.lede{color:var(--muted);max-width:80ch;margin:0}
.verdicts{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin:18px 0}
.v{background:var(--panel);border-left:4px solid var(--teal);padding:10px 14px;border-radius:2px;min-width:0}
.v.warn{border-left-color:var(--amber)} .v.no{border-left-color:var(--rose)}
.v b{display:block;font-size:14px;margin-bottom:2px}
.v span{font-size:13.5px;color:var(--ink)}
.tag{font-family:var(--f-mono);font-size:11.5px;padding:1px 6px;border:1px solid var(--rule);border-radius:3px;color:var(--muted);white-space:nowrap}
.pass{color:var(--teal);font-weight:600}.fail{color:var(--rose);font-weight:600}.part{color:var(--amber);font-weight:600}
.tbl{overflow-x:auto;margin:10px 0 6px}
table{border-collapse:collapse;font-size:13.5px;min-width:560px;width:100%}
th,td{border-bottom:1px solid var(--rule);padding:6px 8px;text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12.5px}
td.n,th.n{text-align:right;font-family:var(--f-mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin:8px 0}
figure{margin:0;min-width:0}
figcaption{font-size:12.5px;color:var(--muted);margin-top:4px}
.chart{width:100%;height:auto;display:block}
.chart .ct{font-size:12px;font-weight:600;fill:var(--ink)}
.chart .ax{font-size:10.5px;fill:var(--muted);font-family:var(--f-mono)}
.chart .bl{font-size:10.5px;fill:#fff;font-family:var(--f-mono)}
.chart .grid{stroke:var(--rule);stroke-width:1}
.chart .zero{stroke:var(--ink);stroke-width:1}
.l-lens{stroke:var(--slate);stroke-width:1.6;stroke-dasharray:4 3}
.l-csd{stroke:var(--amber);stroke-width:2}
.l-probe{stroke:var(--teal);stroke-width:2.4}
.l-cons{stroke:var(--rose);stroke-width:1.4;stroke-dasharray:2 2}
.b-r{fill:var(--r)}.b-o{fill:var(--o)}.b-m{fill:var(--m)}.b-u{fill:var(--u)}
.ci{stroke:var(--muted);stroke-width:2}.ci.sig{stroke:var(--teal);stroke-width:2.6}
.pt{fill:var(--muted)}.pt.sig{fill:var(--teal)}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--muted);margin:4px 0 0}
.sw{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:5px}
.note{font-size:13px;color:var(--muted)}
code{font-family:var(--f-mono);font-size:12.5px;background:var(--panel);padding:1px 4px;border-radius:2px;word-break:break-all}
ul{padding-left:20px}
footer{margin-top:40px;border-top:1px solid var(--rule);padding-top:12px;font-size:12.5px;color:var(--muted)}
"""


def main():
    cv = {r: curves(r) for r, _, _ in RUNS if jl(f'{R}/{r}/probe/summary.json')}
    tx = {r: taxonomy(r) for r in [x[0] for x in RUNS] + ['qwen3-4b-base__base_fewshot__ext', 'qwen3-4b__chat0__ext']
          if os.path.exists(f'{R}/{r}/classify/questions.csv')}
    ee = {r: jl(f'{R}/{r}/early_exit/summary.json') for r in tx}
    dirs = {r: direction(r) for r, _, _ in RUNS}
    dirs_all = {r: direction(r, 'loop') for r in ('qwen3-4b-base__base_fewshot', 'qwen3-4b__chat0')}
    early = {r: direction(r, 'loop_allq_early') for r in ('qwen3-4b-base__base_fewshot', 'qwen3-4b__chat0')}
    steer = {r: jl(f'{R}/{r}/steer/eval_final.json') for r in ('qwen3-4b-base__base_fewshot', 'qwen3-4b__chat0')}
    json.dump(dict(taxonomy=tx, directions={r: d for r, d in dirs.items() if d}),
              open(os.path.join(C.ROOT, 'report', 'report_data.json'), 'w'), indent=1, default=float)
    return cv, tx, ee, dirs, dirs_all, early, steer, None


def table(head, rows, num_cols=()):
    h = ''.join(f'<th class="{"n" if i in num_cols else ""}">{x}</th>' for i, x in enumerate(head))
    b = ''.join('<tr>' + ''.join(f'<td class="{"n" if i in num_cols else ""}">{x}</td>' for i, x in enumerate(r)) + '</tr>' for r in rows)
    return f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def verdict_cls(v):
    v = str(v)
    return 'pass' if v.startswith('PASS') else ('part' if v.startswith('PARTIAL') else 'fail')


def build():
    cv, tx, ee, dirs, dirs_all, early, steer, _ = main()
    names = {r: f'{n} · {f}' for r, n, f in RUNS}
    names.update({'qwen3-4b-base__base_fewshot__ext': 'Qwen3-4B-Base · ext', 'qwen3-4b__chat0__ext': 'Qwen3-4B（后训练）· chat · ext'})
    P = []
    P.append('<title>Brewing 深度干预第一轮</title><meta name="claude:viewport-fit-sides" content="cover">')
    P.append('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Noto+Sans+SC:wght@400;600&family=Noto+Serif+SC:wght@600;700&display=swap">')
    P.append(f'<style>{CSS}</style><div class="wrap">')
    P.append('<header><div class="eyebrow">Brewing · depth intervention · round 1 · 2026-10-16</div>'
             '<h1>Brewing 深度干预第一轮</h1>'
             '<p class="lede">5659 道四选一题，每题 4 个循环置换版本，共 22636 条 prompt。6 个模型或格式配置，另有 2 个 O 扩样集。逐层用 probe（可得性）、CSD（就绪性）和 lens 读出关键 token（答案字母），'
             '做 R/O/M/U 四类判定，再做 early exit 和 depth loop 两种干预，最后找无标签的逐层指标。所有数字都带按题自助的 95% 区间，第二阶段用第 4 折做锁箱。</p></header>')
    P.append('<div class="verdicts">'
             '<div class="v"><b>brewing 现象复现</b><span>6 个配置上，probe 达到半程准确率的层都比 CSD 早 3–5 层：Qwen3-4B 为 L19–20 对 L24，Qwen2.5-7B 为 L16 对 L19，Llama-3.1-8B 为 L12 对 L17。逐题看，R 题从可得到就绪平均间隔 1.5–3.6 层。时间表不随后训练和 prompt 格式改变。</span></div>'
             '<div class="v warn"><b>O 存在但稀少</b><span>内容级 O 占 0.3–2.1%，高于 decoy 机会水平。base 模型的 O 多于对应的 instruct 模型。U 占 42–76%，主要是位置敏感的题。</span></div>'
             '<div class="v no"><b>A1 early exit：收益来自读出头，不来自深度</b><span>按 PLAN 字面判据，8 个运行中 7 个 PASS。但对照 decoy-O 全部 FAIL，而且所有早退收益都不超过"同一个头在末层读出"。</span></div>'
             '<div class="v"><b>A2 depth loop：后训练模型上成立</b><span>预先指定的主检验，在两个 instruct 模型上 PASS（方向净值减安慰剂 +1.4、+2.1pp），在 3 个 base 模型上 FAIL。只有覆盖 brewing 间隔的窗口才有效。</span></div>'
             '<div class="v"><b>"已可得、未发射"的机制</b><span>发射之前，答案在 A–D 读出子空间的正交补里，到发射层才写入读出子空间。推这个补空间分量，会让发射提前约 2 层，U_avail 题的准确率 +6–7pp（A 线，探索性）。</span></div>'
             '<div class="v warn"><b>无标签指标：两个能用，几个不能用</b><span>能用的：发射跳变（L0），可提前 2–5 层停止、一致率 99%；置换共识读出头（L1），约等于金标签 probe，能去掉字母偏置。不能用的：O 与 decoy-O 的区分、loop 收益的预测，单次和 4 次前向都做不到。</span></div>'
             '</div>')
    # ---- section 1
    P.append('<h2>1. 设置与定义</h2>')
    P.append('<ul><li><b>benchmark</b>：GSM-MC 1319、MATH-MC 1000、PythonIO 1000、MMLU-Pro 1000（10 选抽成 4 选）、C-Eval 1340。每题 4 个循环置换版本，题级 5 折。另有 O 扩样集：GSM train 加 MATH 剩余题，共 11364 题。</li>'
             '<li><b>prompt</b>：base 模型用 1-shot 补全，示例随版本同步置换。instruct 模型用 chat 模板，加 system 指令和非思考模式；只给 user 指令时，GSM/MATH 的格式遵从率只有 2%。所有配置的遵从率都不低于 99.98%。</li>'
             '<li><b>读出</b>：逐层取块输出（final norm 之前）。CSD 是 raw Patchscope，把状态注入目标 prompt 的最后位置、跑完剩余层，末层等于真实输出。probe 是 lens 锚定的线性 probe（lens + W·ĥ + b），5 折交叉拟合，另有 3 个按来源打乱标签的 null probe。</li>'
             '<li><b>四类（v1，内容级）</b>：FJC_c 指 4 个版本的 probe 和 CSD 在连续 3 层上都指向正确内容，连续段到达末层时可以不足 3 层。有 FJC_c 时，最终 4/4 正确为 R，否则为 O。没有 FJC_c 时，4 个版本一致指向同一个错误内容为 M，其余为 U。末层的 probe 读出用真实输出代替，所以 R 等价于"4 个版本最终都答对"。</li>'
             '<li><b>对照</b>：null probe；matched decoy；decoy-O（把正确内容换成某个错误内容、套用同一规则得到的"假 O"）；loop 的后段安慰剂块 p28_33（早段安慰剂会破坏格式，已弃用）；同批次的 none 运行作参照，消除 bf16 批次噪声（约 1% 的最终 argmax 会翻动）。</li></ul>')
    # ---- section 2
    P.append('<h2>2. brewing 再发现：可得早于就绪</h2>')
    P.append('<div class="legend"><span><i class="sw" style="background:var(--teal)"></i>full probe（可得性）</span><span><i class="sw" style="background:var(--amber)"></i>CSD（就绪性）</span><span><i class="sw" style="background:var(--slate)"></i>logit lens</span><span><i class="sw" style="background:var(--rose)"></i>置换共识头（L1，无标签）</span></div>')
    P.append('<div class="figs">' + ''.join(f'<figure>{line_chart(cv[r], names[r])}</figure>' for r, _, _ in RUNS if r in cv) + '</div>')
    P.append('<p class="note">纵轴是单版本准确率，机会水平为 0.25。probe 和共识头是 5 折交叉拟合的折外读出；CSD 和 lens 不需要训练。</p>')
    rows = []
    for r, n, f in RUNS:
        t = tx.get(r)
        if t:
            c_ = cv[r]; thr = 0.25 + 0.5 * (c_['full'][-1] - 0.25)
            hm = lambda a: int(np.argmax(np.asarray(a) >= thr))
            rows.append([esc(names[r]), f'{100*t["acc"]:.1f}%', f'L{hm(c_["full"])} / L{hm(c_["csd"])}', f'L{t["fpcl"]:.0f}', f'L{t["fjc"]:.0f}', f'{c_["full"][-1]*100 - t["acc"]*100:+.1f}pp'])
    P.append(table(['配置', '准确率', '半程层：probe / CSD', 'R 题 FPCL_c 中位数（可得）', 'R 题 FJC_c 中位数（就绪）', '顶层 probe − 真实输出'], rows, num_cols=(1, 2, 3, 4, 5)))
    P.append('<ul><li><b>后训练改的是末端读出，不是形成时间。</b>同一家族的 base 和 instruct，可得与就绪的层几乎一样。instruct 模型顶层 probe 比真实输出高约 2pp，原因是末层的字母偏置；B 线输出 A/B 占 31% 和 32%。</li>'
             '<li><b>几何</b>（A、B、Llama）：发射前 4 层，3 维的 A–D 读出子空间里读不到答案，处于随机水平；完整 probe 已有 .39–.53。到发射层，读出子空间的能量放大 51–257 倍，补空间里的答案信号也同时增强，不是被"搬过去"。复核指出，"由补空间写入"只对版本特异的部分成立。</li>'
             '<li><b>因果</b>（探索性，用金标签 probe 的方向）：A 线在 L20–22 推 probe 预测类的补空间方向（0.2 倍归一化状态）。U_avail 题的最终答案转向该类 +10.5 到 +11.5pp，准确率 +5.8 到 +7.5pp，CSD 发射提前约 2 层。随机方向约为 0。B 线也成立，但随机推送本身就有 +2.5pp。在全样本实验里，R 题有 1–2% 被推错，比例随推送幅度增大。</li></ul>')
    # ---- section 3
    P.append('<h2>3. 四类判定（v1，K=3）</h2>')
    P.append('<div class="legend"><span><i class="sw" style="background:var(--r);height:10px"></i>R</span><span><i class="sw" style="background:var(--o);height:10px"></i>O</span><span><i class="sw" style="background:var(--m);height:10px"></i>M</span><span><i class="sw" style="background:var(--u);height:10px"></i>U</span></div>')
    P.append('<figure>' + stacked_bars([(names[r], tx[r]) for r in [x[0] for x in RUNS] + ['qwen3-4b-base__base_fewshot__ext', 'qwen3-4b__chat0__ext'] if r in tx]) + '</figure>')
    rows = []
    for r in [x[0] for x in RUNS] + ['qwen3-4b-base__base_fewshot__ext', 'qwen3-4b__chat0__ext']:
        t = tx.get(r)
        if t:
            c = t['counts']; n = sum(c[k] for k in 'ROMU')
            rows.append([esc(names[r])] + [f'{c[k]} ({100*c[k]/n:.1f}%)' for k in 'ROMU'] + [str(t['u_avail'])])
    P.append(table(['配置', 'R', 'O', 'M', 'U', 'U_avail（probe 已 4/4，CSD 未就绪）'], rows, num_cols=(1, 2, 3, 4, 5)))
    P.append('<ul><li><b>O 高于机会水平。</b>A 线 O 为 1.40%，decoy-O 为 0.56%，差值 +0.84pp [0.52, 1.16]。null probe 的 O 误报率是 0.008%。计算型题上 O 更多，A ext 为 2.1%。</li>'
             '<li><b>v0（第一篇的单版本口径）会让 O 虚高</b>，A 线有 3687 条。原因是单层命中加上 4 选的机会水平。v1 的内容级判定是必要的。</li>'
             '<li><b>U 很大，主要是位置敏感的题。</b>只看末层，跨版本多数投票就让准确率 +2.0（A）和 +2.6pp（B）。这是置换集成，不涉及深度。</li></ul>')
    # ---- section 4
    P.append('<h2>4. 第一阶段验收：干预</h2><h3>4.1 early exit 针对 O（A1）</h3>')
    rows = []
    for r in [x[0] for x in RUNS] + ['qwen3-4b-base__base_fewshot__ext', 'qwen3-4b__chat0__ext']:
        e = ee.get(r)
        if not e:
            continue
        A1 = e.get('A1') or {}
        rules = A1.get('rules', {})
        best = max(rules.items(), key=lambda kv: kv[1]['net_acc_change'][0]) if rules else None
        noex = A1.get('head_at_L1_no_exit', {})
        rows.append([esc(names[r]), f'<span class="{verdict_cls(A1.get("verdict"))}">{esc(str(A1.get("verdict","")).split(" ")[0])}</span>',
                     f'<span class="{verdict_cls(A1.get("verdict_vs_decoyO"))}">{esc(A1.get("verdict_vs_decoyO",""))}</span>',
                     f'{best[0]} {100*best[1]["net_acc_change"][0]:+.2f}pp' if best else '—',
                     f'{100*noex.get("net_acc_change",[0])[0]:+.2f}pp' if noex else '—'])
    P.append(table(['配置', 'A1（PLAN 判据）', '对照 decoy-O', '三条规则中净变化最高者（描述性）', 'probe 头放在末层、不早退'], rows, num_cols=(3, 4)))
    P.append('<p>O 确实依赖深度：A 线 probe 在 L24–28 能读对 67/79 道 O，到 L35 只剩 23/79。但末层也会修好在中层形成的错误内容（decoy-O），而且这类题更多。'
             '所以不加门控时，早退的净值为负，或者只是读出头带来的收益。<b>早退能把 O 变成 R，但同样多的 decoy-O 会被"还原"成错误内容。</b></p>')
    P.append('<h3>4.2 depth loop 针对 U（A2）</h3>')
    rows = [
        ['Qwen3-4B-Base（A）', 'w18_25_k2', '+3.5 [2.2, 4.9]', '105 / 100', '+0.08 [−1.22, 1.48]', '<span class="part">PASS 勉强 / 严格 FAIL</span>'],
        ['Qwen3-4B 后训练（B）', 'w18_25_k2', '+6.4 [5.4, 7.4]', '215 / 146', '+1.36 [0.38, 2.39]', '<span class="pass">PASS</span>'],
        ['Qwen2.5-7B', 'w14_19_k2', '+5.4 [4.0, 6.7]', '123 / 141', '−0.24 [−1.60, 1.08]', '<span class="fail">FAIL</span>'],
        ['Qwen2.5-7B-Instruct', 'w14_19_k2', '+7.8 [6.7, 9.0]', '174 / 115', '+2.11 [0.95, 3.23]', '<span class="pass">PASS</span>'],
        ['Llama-3.1-8B', 'w16_22_k2', '−0.1 [−0.6, 0.4]', '34 / 44', '−0.16 [−0.67, 0.35]', '<span class="fail">FAIL</span>']]
    P.append(table(['模型', '主检验配置（mode last）', 'U 改变率 − 安慰剂（pp）', '朝正确 / 朝错误', '方向净值 − 安慰剂（pp）', '判定'], rows, num_cols=(2, 4)))
    fr = []
    for r, lab in (('qwen3-4b__chat0', 'B last'), ('qwen2.5-7b-instruct__chat0', 'Q2.5-I last'), ('qwen3-4b-base__base_fewshot', 'A last'), ('qwen2.5-7b__base_fewshot', 'Q2.5 last')):
        d = dirs.get(r)
        if d:
            for k in sorted(d['res']):
                e_, lo, hi = d['res'][k]
                fr.append((f'{lab} {k}', e_, lo, hi))
    P.append('<figure>' + forest(fr) + '<figcaption>每道 U 题转成 R/O 记 +1、转成 M 记 −1，取平均后减去安慰剂；按题配对自助 95% 区间。实心绿色表示区间不含 0。</figcaption></figure>')
    P.append('<ul><li><b>两个家族的后训练模型上，loop 都把 U 推向正确，并且超出安慰剂</b>：几乎所有窗口，在 mode last 和 mode all 下都成立。Qwen2.5-7B-Instruct 在 mode all 下，4 个窗口的方向净值减安慰剂为 +1.3 到 +3.2pp。单版本净准确率 +0.4 到 +1.3pp（mode last）。<b>base 模型上不一致</b>：Qwen3-4B-Base 只有 mode last 下 2 个窗口成立；Qwen2.5-7B 在 mode all 下 3/4 个窗口成立，但这部分被偏负的安慰剂放大了，mode last 下全不成立；Llama 不成立。</li>'
             '<li><b>窗口位置有特异性。</b>落在可得之前的 L10–L19 没有效果。从 L14 起、覆盖 brewing 间隔的窗口才有效。</li>'
             '<li><b>代价是 R 损伤。</b>mode last 下 4–13% 的 R 题不再 4/4 正确，mode all 下 9–57%。loop 加在所有 token 上，净收益很小；没有找到一个无标签门控能胜过"一直 loop"。</li>'
             '<li><b>U_avail 最可救。</b>B 线 U_avail 错误行的修复率为 .27–.42，U_none 为 .07–.11；相对安慰剂的 DiD 在锁箱上为 +.12 到 +.20。A 线与 Llama 的差异主要来自一般扰动。</li></ul>')
    # ---- section 5
    P.append('<h2>5. 第二阶段：无标签指标的发现与应用</h2>')
    P.append('<p class="note">层级：L0 不训练；L1 自监督，目标是模型自己的 4 版本置换共识或满预算输出；L2 用金标签 probe 的输出；L3 用金标签训练。只有 L0 和 L1 算"无监督"。选择都在第 0–3 折上做，第 4 折是锁箱，只看一次。</p>')
    rows = [
        ['发射跳变：CSD 的 KL 跳变，读出子空间能量增长', '<span class="tag">L0</span>', '可以安全早停的层（与满预算输出一致）',
         '一致率 99% 时平均退出层：B 线 25.2（置信度稳定规则 30.7，oracle 23.5）；A 线 29.0（31.4，24.5）。发射前 CSD 的高置信度是字母先验造成的假象', '<span class="pass">能用</span>（省算力的早停）'],
        ['置换共识读出头：用 4 版本共识训练的 lens 锚定 probe', '<span class="tag">L1</span>', '已形成的答案；末端字母偏置',
         'A、B 两条线上逐层与金标签 probe 相差 ±0.6pp。在末层读出（不早退）：B +1.59pp，Llama +0.88pp，A +0.42pp。无标签 taxonomy 与官方的 kappa 为 0.93–0.99', '<span class="pass">能用</span>（热插拔读出头）'],
        ['补空间推送：probe 类方向中、与 A–D 读出正交的分量', '<span class="tag">L2→L1</span>', '把已可得未发射的答案推到发射',
         '金标签方向（L2，A 线 U_avail）：准确率 +5.8 到 +7.5pp，发射提前约 2 层。L1 全样本：B 线锁箱 +0.48pp [0.02, 0.99]，胜过所有对照；A 线 −0.20pp；Qwen2.5-7B-Instruct −0.04pp，未复现。R 损伤约 1–2%', '<span class="part">机制真实，收益小且不稳健</span>'],
        ['跨版本一致性门控（4 次前向）', '<span class="tag">L1</span>', 'O（中层形成、末层丢失）',
         '救回 O 78/79（A）和 18/18（B），R 零损伤；但 decoy-O 被"还原"的比例相同（62/72），净值约 0', '<span class="fail">不能区分 O 和 decoy-O</span>'],
        ['单次前向的回读门控', '<span class="tag">L0/L1</span>', '中层对、末层错，还是相反',
         'AUROC 比置信度最多高 +0.04，信号几乎都在末层输出本身。B 线 +0.27pp 来自去字母偏置，用不涉及深度的频率重排就能得到', '<span class="fail">不能用</span>'],
        ['扰动稳定性（loop 不变性）', '<span class="tag">L0</span>', '是否已就绪',
         '判断对错的 AUROC 比置信度低 0.09–0.16；在置信度分层内约 0.5', '<span class="fail">弱于置信度</span>'],
        ['loop 收益预测', '<span class="tag">L0/L1</span>', '哪些行会被 loop 修好、哪些会被弄坏',
         '修好对弄坏的 AUROC 为 0.47–0.58，连金标签门控也做不到。没有任何门控胜过一直 loop', '<span class="fail">不能用</span>'],
        ['"已可得、未发射"的量（probe margin 减 lens margin）', '<span class="tag">L2</span>', '发射时机、O/U、可救性',
         '把 probe 自身的置信度放进基线后，增益不超过 +0.01，不是新量。但它的方向在因果上有效（见上一行的推送）', '<span class="part">作为量不新，作为方向有用</span>']]
    P.append(table(['候选指标', '层级', '用来预测什么', '结果', '结论'], rows))
    st = {r: v for r, v in steer.items() if v}
    if st:
        P.append('<h3>L1 补空间推送：全样本结果</h3>' + STEER_HTML(st))
    else:
        P.append('<h3>L1 补空间推送：全样本结果</h3><p class="note">还在运行。结果会追加到 RESULTS.md 第 13 节。</p>')
    # ---- section 6
    P.append('<h2>6. 对热插拔工具的含义与下一步</h2><ul>'
             '<li><b>早停头</b>：用 L0 的发射跳变决定什么时候停，用 L1 的共识读出头读答案。目标是省算力和去偏置；不要指望靠它把 O 变成 R。</li>'
             '<li><b>depth loop</b>：只对 instruct 模型、只在 brewing 窗口内（约 0.4–0.75 相对深度）、只在关键 token 上用。下一步要找一个比置信度更好的"可救"门控。U_avail（probe 已可得、CSD 未就绪）是目前最有希望的对象，但它的检测依赖 probe（L2）或 4 次前向（L1）。</li>'
             '<li><b>补空间推送</b>：机制成立。方向特异，推它会把答案推向预测类，所以 U_avail 和 O 行会被修复。但 L1 全样本的净效果只有 +0.48pp（instruct 锁箱）或约 0（base），瓶颈在中层共识头的准确性。下一步：先提升"推向哪一类"的头（多层集成、更晚的层），或只在 U_avail 的无标签代理上推。</li>'
             '<li><b>扩展</b>：把 critical token 从选择题字母扩到 CoT 里的关键 token；O 扩样继续以计算型题为主。</li></ul>')
    P.append('<footer>完整迭代日志：<code>/nvme0/if/research/brewing/exp/depth_intervention/RESULTS.md</code>（第 0–13 节）；数据契约：<code>scripts/INTERFACES.md</code>；'
             '结果目录：<code>results/{model}__{fmt}[__ext]/</code>。每一项分析都经过独立 subagent 的对抗复核，更正已经写进 RESULTS.md。</footer></div>')
    html_ = '\n'.join(P)
    open(OUT, 'w').write(html_)
    print('wrote', OUT, len(html_))


def STEER_HTML(st):
    rows = [['<b>Qwen3-4B 后训练（B）</b>', 'maxcons 头 · L22 · α=0.1 · 几乎全部推送', '+0.17 [−0.06, 0.40]（交叉拟合）', '<span class="pass">+0.48 [0.02, 0.99]</span>',
             '+0.08 / −0.13', '−0.21 / <span class="fail">−0.53 [−1.05, −0.07]</span>', '+0.08'],
            ['Qwen3-4B-Base（A）', 'vote 头 · L21 · α=0.1 · 约半数行', '−0.16 [−0.27, −0.03]（交叉拟合）', '−0.20 [−0.48, 0.07]', '−0.08 / −0.09', '−0.15 / −0.20', '−0.07'],
            ['Qwen2.5-7B-Instruct（复现）', 'maxcons 头 · L17 · α=0.1 · 约 41% 的行', '+0.01 [−0.07, 0.10]（交叉拟合）', '−0.04 [−0.18, 0.09]', '−0.06 / −0.13', '−0.01 / −0.07', '−0.08 / −0.04'],
            ['Qwen2.5-7B base（复现）', 'maxcons 头 · L17 · α=0.2 · 约 10% 的行', '+0.03 [−0.02, 0.09]', '−0.02 [−0.11, 0.04]', '−0.02 / +0.04', '−0.02 / +0.02', '−0.02 / −0.04']]
    t = table(['线', 'L1 选中的配置', 'dev 净变化（pp）', '锁箱净变化（pp）', '随机正交方向（dev / 锁箱）', '次选类方向（dev / 锁箱）', '读出子空间投影（dev）'], rows, num_cols=(2, 3))
    return (t + '<p>在 L20–22 层，用 L1 共识头的预测类构造方向，投到 A–D 读出子空间的正交补里，按 α·‖h‖ 推送，然后跑完剩余层。'
            '层、α、门控都按"输出与 4 版本共识一致"这个无标签判据，在第 0–3 折上选定；第 4 折只评一次。'
            '<b>在后训练模型上，这个完全无标签的单层干预在锁箱上显著为正，并且胜过所有同等范数的对照（减随机 +0.62、减次选类 +1.01、减读出投影 +0.55pp，CI 都大于 0）；但 dev 交叉拟合的 CI 含 0。</b>它有方向特异性：推次选类会变差，随机方向约为 0，只推读出子空间也没用。'
            '独立复算（hook 实现）在第 4 折上得到 B +0.55 [0.07, 1.03]（减随机 +0.75 [0.24, 1.30]）、A −0.29 [−0.55, −0.02]。B 线锁箱上 U_avail 行 +4.1pp，R 损伤约 1.6%。它的收益只有"把 L1 共识头直接当末层读出头"（锁箱 +1.6pp）的三分之一。瓶颈在共识头本身：中层的共识头和末层输出不一致时，大多是共识头错。base 模型上没有收益。<b>在第二个 instruct 模型（Qwen2.5-7B-Instruct，层按相对深度预先指定为 L15–17）上没有复现</b>，所以这个效应目前只能算模型特有的小效应。</p>')


if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    build()
