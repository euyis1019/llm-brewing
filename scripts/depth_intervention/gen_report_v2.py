"""Round-1 results report, v2 (report/round1_report_v2.html). Outline agreed with the user on 2026-10-17:
1 data & method, 2 brewing + class shares (5 classes incl. NB), 3 unsupervised indicators, 4 interventions & effect.
Chart numbers are read from result files; prose numbers for stage-2 come from the reviewed RESULTS.md sections."""
import csv, html, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

R = os.path.join(C.ROOT, 'results')
OUT = os.path.join(C.ROOT, 'report', 'round1_report_v2.html')
E = html.escape
RUNS = [('qwen3-4b-base__base_fewshot', 'Qwen3-4B-Base', 'base', 'A'),
        ('qwen3-4b__chat0', 'Qwen3-4B 后训练', 'post', 'B'),
        ('qwen3-4b__base_fewshot', 'Qwen3-4B 后训练 · 1-shot', 'post', '格式对照'),
        ('qwen2.5-7b__base_fewshot', 'Qwen2.5-7B', 'base', 'Q2.5'),
        ('qwen2.5-7b-instruct__chat0', 'Qwen2.5-7B-Instruct', 'post', 'Q2.5-I'),
        ('llama3.1-8b__base_fewshot', 'Llama-3.1-8B', 'base', 'Llama')]
NAME = {r: n for r, n, _, _ in RUNS}


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x, d=1):
    return f'{100 * x:.{d}f}%'


# ----------------------------------------------------------------------------- data
def curves(run):
    ps = jl(f'{R}/{run}/probe/summary.json')
    rows = json.load(open(f'{R}/{run}/extract/rows.json'))
    gold = np.array([r['gold'] for r in rows])
    ps4 = np.asarray(np.load(f'{R}/{run}/extract/csd_raw4.npy', mmap_mode='r'))
    out = dict(L=ps['L'], probe=np.array(ps['acc']['full']), lens=np.array(ps['acc']['lens']),
               patch=(ps4.argmax(-1) == gold[:, None]).mean(0))
    thr = 0.25 + 0.5 * (out['probe'][-1] - 0.25)
    out['hm_probe'] = int(np.argmax(out['probe'] >= thr))
    out['hm_patch'] = int(np.argmax(out['patch'] >= thr))
    return out


def classes(run):
    """Full-set partition: R, O (ready), U, M (available, never ready), NB (never available), X."""
    Q = list(csv.DictReader(open(f'{R}/{run}/classify/questions.csv')))
    oc = np.array([q['outcome_v1_K3'] for q in Q])
    av = np.array([int(q['FPCL_c_K3']) >= 0 for q in Q])
    src = np.array([q['source'] for q in Q])
    lab = np.where(oc == 'R', 'R', np.where(oc == 'O', 'O', np.where(oc == 'X', 'X',
                   np.where(~av, 'NB', np.where(oc == 'M', 'M', 'U')))))
    nb_wrong = (~av) & (oc == 'M')
    fp = np.array([int(q['FPCL_c_K3']) for q in Q]); fj = np.array([int(q['FJC_c_K3']) for q in Q])
    meta = jl(f'{R}/{run}/extract/meta.json')
    return dict(n=len(lab), lab=lab, src=src, nb_wrong=int(nb_wrong.sum()), acc=meta['acc4'],
                fpcl=float(np.median(fp[lab == 'R'][fp[lab == 'R'] >= 0])), fjc=float(np.median(fj[lab == 'R'])))


def direction(run, loop_dir='loop_allq'):
    p = f'{R}/{run}/{loop_dir}/eval_lo/post_classes.csv'
    if not os.path.exists(p):
        return None
    P = list(csv.DictReader(open(p)))
    cols = [c for c in P[0] if c.startswith('last_') and not c.endswith('_none') and '_p4_' not in c]
    pl = [c for c in cols if c.split('_')[1].startswith('p')][0]
    U = [x for x in P if x['last_none'] == 'U']
    sc = lambda c: np.array([1 if x[c] in ('R', 'O') else (-1 if x[c] == 'M' else 0) for x in U], float)
    idx = np.random.default_rng(0).integers(0, len(U), (2000, len(U)))
    sp = sc(pl)
    res = {}
    for c in cols:
        if c == pl:
            continue
        d = sc(c) - sp
        b = d[idx].mean(1)
        res[c.split('_', 1)[1]] = (100 * d.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5))
    return dict(n_U=len(U), placebo=pl.split('_', 1)[1], res=res)


def row_net(run, loop_dir='loop_allq'):
    d = f'{R}/{run}/{loop_dir}'
    rows = json.load(open(f'{R}/{run}/extract/rows.json')); gold = np.array([r['gold'] for r in rows])
    if not os.path.exists(f'{d}/last_none/meta.json'):
        return None
    r0 = np.load(f'{d}/last_none/rows.npy'); ok0 = np.load(f'{d}/last_none/final4.npy').argmax(-1) == gold[r0]
    out = {}
    for c in sorted(os.listdir(d)):
        if c.startswith('last_') and c != 'last_none' and os.path.exists(f'{d}/{c}/meta.json') and '_p4_' not in c:
            ok = np.load(f'{d}/{c}/final4.npy').argmax(-1) == gold[r0]
            out[c[5:]] = 100 * (ok.mean() - ok0.mean())
    return out


# ----------------------------------------------------------------------------- svg helpers
class Fig:
    n = 0


def fid():
    Fig.n += 1
    return f'f{Fig.n}'


def line_chart(series, L, title, y0=0.2, y1=0.7, w=330, h=210, band=None, vline=None, x0=0, yfmt='{:.2f}', ylab=None):
    """series: list of (key, label, values, cls). Crosshair + tooltip via hit rects."""
    pad_l, pad_r, pad_t, pad_b = 38, 10, 24, 26
    X = lambda l: pad_l + (w - pad_l - pad_r) * (l - x0) / max(1, (L - 1 - x0))
    Y = lambda a: pad_t + (h - pad_t - pad_b) * (1 - (min(max(a, y0), y1) - y0) / (y1 - y0))
    i = fid()
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{E(title)}">',
         f'<text x="{pad_l}" y="14" class="ct">{E(title)}</text>']
    ticks = np.linspace(y0, y1, 5)
    for a in ticks:
        s.append(f'<line x1="{pad_l}" x2="{w-pad_r}" y1="{Y(a):.1f}" y2="{Y(a):.1f}" class="grid"/>'
                 f'<text x="{pad_l-5}" y="{Y(a)+3.5:.1f}" class="ax" text-anchor="end">{yfmt.format(a)}</text>')
    step = 6 if L - x0 >= 30 else 4
    xt = [l for l in range(x0, L, step) if (L - 1) - l >= step / 2] + [L - 1]
    for l in xt:
        s.append(f'<text x="{X(l):.1f}" y="{h-9}" class="ax" text-anchor="middle">L{l}</text>')
    if band:
        a, b = band
        s.append(f'<rect x="{X(a):.1f}" y="{pad_t}" width="{max(X(b)-X(a),1):.1f}" height="{h-pad_t-pad_b}" class="band"/>'
                 f'<text x="{(X(a)+X(b))/2:.1f}" y="{pad_t+11}" class="ax" text-anchor="middle">间隔</text>')
    if vline is not None:
        v, vl = vline
        s.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{pad_t}" y2="{h-pad_b}" class="vref"/>'
                 f'<text x="{X(v)+4:.1f}" y="{pad_t+11}" class="ax">{E(vl)}</text>')
    for key, lab, vals, cls in series:
        pts = ' '.join(f'{X(l):.1f},{Y(a):.1f}' for l, a in zip(range(x0, x0 + len(vals)), vals))
        s.append(f'<polyline points="{pts}" class="ln {cls}"/>')
        lv = vals[-1]
        s.append(f'<circle cx="{X(x0+len(vals)-1):.1f}" cy="{Y(lv):.1f}" r="4" class="dot {cls}"/>')
    s.append(f'<line class="xhair" id="{i}x" x1="0" x2="0" y1="{pad_t}" y2="{h-pad_b}" visibility="hidden"/>')
    cw = (w - pad_l - pad_r) / max(1, (L - 1 - x0))
    for l in range(x0, L):
        tip = f'L{l}：' + '；'.join(f'{lab} {yfmt.format(vals[l-x0])}' for _, lab, vals, _ in series if l - x0 < len(vals))
        s.append(f'<rect class="hit" x="{X(l)-cw/2:.1f}" y="{pad_t}" width="{cw:.1f}" height="{h-pad_t-pad_b}" '
                 f'data-tip="{E(tip)}" data-xh="{i}x" data-cx="{X(l):.1f}"/>')
    if ylab:
        s.append(f'<text x="10" y="{pad_t-8}" class="ax">{E(ylab)}</text>')
    s.append('</svg>')
    return ''.join(s)


def stacked(items, keys, cls, labels, w=760):
    """items: list of (name, {key: count}); 100% horizontal stacked bars, 2px surface gaps."""
    pad_l, pad_r, bar, gap = 190, 16, 22, 16
    h = 18 + (bar + gap) * len(items)
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="各配置的类别占比">']
    for j, (name, cnt) in enumerate(items):
        y = 10 + j * (bar + gap)
        tot = sum(cnt[k] for k in keys)
        s.append(f'<text x="{pad_l-10}" y="{y+bar/2+4:.1f}" class="lab" text-anchor="end">{E(name)}</text>')
        x = pad_l
        span = w - pad_l - pad_r
        for k in keys:
            wd = span * cnt[k] / tot
            if wd <= 0:
                continue
            ww = max(wd - 2, 0.6)
            tip = f'{name} · {labels[k]}：{cnt[k]} 题（{100*cnt[k]/tot:.1f}%）'
            s.append(f'<rect x="{x:.1f}" y="{y}" width="{ww:.1f}" height="{bar}" class="{cls[k]}" data-tip="{E(tip)}" tabindex="0"/>')
            txt = f'{k} {100*cnt[k]/tot:.0f}%' if k != 'NB' else f'NB {100*cnt[k]/tot:.0f}%'
            if ww > 9 * len(txt) + 8:
                s.append(f'<text x="{x+ww/2:.1f}" y="{y+bar/2+4:.1f}" class="inbar{" dark" if k == "NB" else ""}" text-anchor="middle">{E(txt)}</text>')
            x += wd
    s.append('</svg>')
    return ''.join(s)


def forest(rows, lo_x=-2.0, hi_x=4.0, w=760):
    """rows: (label, est, lo, hi, cls, bold)."""
    pad_l, pad_r, rh = 230, 16, 21
    h = 30 + rh * len(rows) + 14
    X = lambda v: pad_l + (w - pad_l - pad_r) * (min(max(v, lo_x), hi_x) - lo_x) / (hi_x - lo_x)
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="loop 的方向净值减安慰剂">']
    for v in range(int(lo_x), int(hi_x) + 1):
        s.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="12" y2="{h-22}" class="{"zero" if v == 0 else "grid"}"/>'
                 f'<text x="{X(v):.1f}" y="{h-8}" class="ax" text-anchor="middle">{v:+d}pp</text>')
    for j, (lab, e, lo, hi, cls, bold) in enumerate(rows):
        y = 22 + j * rh
        if lab.startswith('##'):
            s.append(f'<text x="8" y="{y+4}" class="grp">{E(lab[2:])}</text>')
            continue
        s.append(f'<text x="{pad_l-10}" y="{y+4}" class="lab{" b" if bold else ""}" text-anchor="end">{E(lab)}</text>'
                 f'<line x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{y}" y2="{y}" class="ci {cls}"/>'
                 f'<circle cx="{X(e):.1f}" cy="{y}" r="4.5" class="dot {cls}" data-tip="{E(f"{lab}：{e:+.2f}pp [{lo:+.2f}, {hi:+.2f}]")}" tabindex="0"/>')
    s.append('</svg>')
    return ''.join(s)


def grouped_bars(groups, series, vals, w=520, h=230, y0=-3, y1=9, unit='pp'):
    """vals[g][s] = (est, lo, hi); vertical grouped columns from a zero baseline."""
    pad_l, pad_r, pad_t, pad_b = 40, 10, 14, 30
    Y = lambda v: pad_t + (h - pad_t - pad_b) * (1 - (v - y0) / (y1 - y0))
    gw = (w - pad_l - pad_r) / len(groups)
    bw = min(22, (gw - 24) / len(series))
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="推送方向对准确率的影响">']
    for v in range(y0, y1 + 1, 3):
        s.append(f'<line x1="{pad_l}" x2="{w-pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="{"zero" if v == 0 else "grid"}"/>'
                 f'<text x="{pad_l-5}" y="{Y(v)+3.5:.1f}" class="ax" text-anchor="end">{v:+d}</text>')
    for gi, g in enumerate(groups):
        gx = pad_l + gi * gw + (gw - bw * len(series) - 2 * (len(series) - 1)) / 2
        s.append(f'<text x="{pad_l+gi*gw+gw/2:.1f}" y="{h-10}" class="ax" text-anchor="middle">{E(g)}</text>')
        for si, (sk, sl, cls) in enumerate(series):
            e, lo, hi = vals[g][sk]
            x = gx + si * (bw + 2)
            top, bot = (Y(e), Y(0)) if e >= 0 else (Y(0), Y(e))
            s.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(bot-top,1):.1f}" rx="2" class="{cls}" '
                     f'data-tip="{E(f"{g} · {sl}：{e:+.1f}{unit} [{lo:+.1f}, {hi:+.1f}]")}" tabindex="0"/>'
                     f'<line x1="{x+bw/2:.1f}" x2="{x+bw/2:.1f}" y1="{Y(lo):.1f}" y2="{Y(hi):.1f}" class="whisk"/>')
    s.append('</svg>')
    return ''.join(s)


def pareto(panels, w=360, h=230):
    """panels: list of (title, oracle, {series: [(exit, agree), ...]}) -> small multiples."""
    out = []
    for title, oracle, ser in panels:
        pad_l, pad_r, pad_t, pad_b = 44, 12, 24, 30
        x0, x1, y0, y1 = 22, 33, 0.945, 0.995
        X = lambda v: pad_l + (w - pad_l - pad_r) * (v - x0) / (x1 - x0)
        Y = lambda v: pad_t + (h - pad_t - pad_b) * (1 - (v - y0) / (y1 - y0))
        s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{E(title)}">',
             f'<text x="{pad_l}" y="14" class="ct">{E(title)}</text>']
        for v in (0.95, 0.96, 0.97, 0.98, 0.99):
            s.append(f'<line x1="{pad_l}" x2="{w-pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                     f'<text x="{pad_l-5}" y="{Y(v)+3.5:.1f}" class="ax" text-anchor="end">{100*v:.0f}%</text>')
        for v in range(22, 34, 2):
            s.append(f'<text x="{X(v):.1f}" y="{h-12}" class="ax" text-anchor="middle">L{v}</text>')
        s.append(f'<line x1="{X(oracle):.1f}" x2="{X(oracle):.1f}" y1="{pad_t}" y2="{h-pad_b}" class="vref"/>'
                 f'<text x="{X(oracle)+4:.1f}" y="{h-pad_b-6}" class="ax">oracle</text>')
        for key, (lab, cls, pts) in ser.items():
            pp = ' '.join(f'{X(a):.1f},{Y(b):.1f}' for a, b in pts)
            s.append(f'<polyline points="{pp}" class="ln {cls}"/>')
            for a, b in pts:
                s.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="4" class="dot {cls}" '
                         f'data-tip="{E(f"{lab}：一致率 {100*b:.1f}% 时平均停在 L{a:.1f}")}" tabindex="0"/>')
        s.append('</svg>')
        out.append(f'<figure>{"".join(s)}</figure>')
    return '<div class="figs two">' + ''.join(out) + '</div>'


def table(head, rows, num=()):
    hh = ''.join(f'<th class="{"n" if i in num else ""}">{x}</th>' for i, x in enumerate(head))
    bb = ''.join('<tr>' + ''.join(f'<td class="{"n" if i in num else ""}">{x}</td>' for i, x in enumerate(r)) + '</tr>' for r in rows)
    return f'<div class="tbl"><table><thead><tr>{hh}</tr></thead><tbody>{bb}</tbody></table></div>'


def datatable(head, rows, num=()):
    return f'<details class="dt"><summary>数据表</summary>{table(head, rows, num)}</details>'


# ----------------------------------------------------------------------------- diagrams
def diag_versions():
    opts, ans = ['22', '64', '18', '12'], 2
    w, h, cw, rh, x0, y0 = 620, 196, 96, 34, 128, 34
    s = [f'<svg viewBox="0 0 {w} {h}" class="diagram" role="img" aria-label="同一道题的 4 个循环置换版本：正确内容 18 在 A、B、C、D 各出现一次">']
    for i, L in enumerate('ABCD'):
        s.append(f'<text x="{x0+i*cw+cw/2}" y="{y0-12}" class="dl" text-anchor="middle">{L}</text>')
    for v in range(4):
        y = y0 + v * rh
        s.append(f'<text x="{x0-12}" y="{y+rh/2+4}" class="dl" text-anchor="end">版本 {v}</text>')
        for i in range(4):
            o = (i + v) % 4
            hit = o == ans
            s.append(f'<rect x="{x0+i*cw+3}" y="{y+3}" width="{cw-6}" height="{rh-6}" rx="3" class="{"cell hit-cell" if hit else "cell"}"/>'
                     f'<text x="{x0+i*cw+cw/2}" y="{y+rh/2+4}" class="{"dl strong" if hit else "dl"}" text-anchor="middle">{opts[o]}</text>')
    s.append(f'<text x="{x0+4*cw+16}" y="{y0+rh*2+4}" class="dl">正确内容"18"</text>'
             f'<text x="{x0+4*cw+16}" y="{y0+rh*2+20}" class="dl muted">在每一列各出现一次</text></svg>')
    return ''.join(s)


def diag_method():
    w, h = 760, 250
    s = [f'<svg viewBox="0 0 {w} {h}" class="diagram" role="img" aria-label="同一个隐状态的两种读法：probe 看信息是否可得，Patchscope 看是否已能直接输出">',
         '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>']
    # layer stack
    for k in range(9):
        x = 40 + k * 46
        hl = k == 4
        s.append(f'<rect x="{x}" y="96" width="38" height="44" rx="3" class="{"blk hl" if hl else "blk"}"/>')
    s.append('<text x="40" y="86" class="dl">题目 prompt 逐层前向</text>'
             '<text x="59" y="160" class="dl muted" text-anchor="middle">L0</text><text x="243" y="160" class="dl strong" text-anchor="middle">第 l 层</text>'
             '<text x="427" y="160" class="dl muted" text-anchor="middle">末层</text>')
    # probe branch
    s.append('<line x1="243" y1="94" x2="520" y2="44" class="arw" marker-end="url(#ar)"/>'
             '<text x="380" y="56" class="dl" text-anchor="middle">直接读</text>'
             '<rect x="528" y="22" width="210" height="48" rx="4" class="box"/>'
             '<text x="633" y="42" class="dl strong" text-anchor="middle">linear probe</text>'
             '<text x="633" y="60" class="dl muted" text-anchor="middle">答案是否"可得"</text>')
    # patchscope branch
    s.append('<line x1="243" y1="142" x2="520" y2="196" class="arw" marker-end="url(#ar)"/>'
             '<text x="380" y="190" class="dl" text-anchor="middle">注入</text>'
             '<rect x="528" y="172" width="210" height="58" rx="4" class="box"/>'
             '<text x="633" y="192" class="dl strong" text-anchor="middle">Patchscope</text>'
             '<text x="633" y="208" class="dl muted" text-anchor="middle">放进空 prompt "Answer:\\n"</text>'
             '<text x="633" y="223" class="dl muted" text-anchor="middle">跑完 l+1…末层，看输出字母</text>'
             '</svg>')
    return ''.join(s)


def diag_classes():
    w, h = 780, 270
    s = [f'<svg viewBox="0 0 {w} {h}" class="diagram" role="img" aria-label="四类加 NB 的判定顺序">',
         '<defs><marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>']

    def q(x, y, t1, t2):
        s.append(f'<rect x="{x}" y="{y}" width="200" height="46" rx="4" class="box"/>'
                 f'<text x="{x+100}" y="{y+19}" class="dl strong" text-anchor="middle">{t1}</text>'
                 f'<text x="{x+100}" y="{y+36}" class="dl muted" text-anchor="middle">{t2}</text>')

    def leaf(x, y, k, t, cls):
        s.append(f'<rect x="{x}" y="{y}" width="132" height="40" rx="4" class="leaf {cls}"/>'
                 f'<text x="{x+66}" y="{y+17}" class="dl strong" text-anchor="middle">{k}</text>'
                 f'<text x="{x+66}" y="{y+32}" class="dl muted" text-anchor="middle">{t}</text>')

    def arrow(x1, y1, x2, y2, t, tx, ty):
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="arw" marker-end="url(#ar2)"/>'
                 f'<text x="{tx}" y="{ty}" class="dl muted">{t}</text>')

    q(20, 112, '答案曾经可得吗？', 'probe 连续 3 层 4/4 读对')
    q(290, 40, '曾经就绪吗？', 'probe 和 Patchscope 同时 4/4')
    leaf(620, 210, 'NB', '从未可得', 'k-nb')
    arrow(220, 124, 288, 70, '是', 236, 88)
    arrow(220, 148, 618, 228, '否', 300, 196)
    q(530, 6, '最终 4 版本全对？', '真实输出')
    leaf(640, 70, 'R / O', '全对是 R，否则 O', 'k-r')
    arrow(490, 54, 528, 32, '是', 500, 34)
    q(290, 140, '最终一致答错？', '4 版本指向同一个错误内容')
    arrow(390, 86, 390, 138, '否', 398, 116)
    leaf(560, 132, 'M / U', '一致错是 M，否则 U', 'k-u')
    arrow(490, 162, 558, 152, '', 0, 0)
    arrow(630, 52, 680, 68, '', 0, 0)
    s.append('</svg>')
    return ''.join(s)


# ----------------------------------------------------------------------------- page
CSS = """
:root{
  /* Layout: one reading column, figures in a wrapping grid; chart surface = page surface. */
  --paper:#f7f8f6; --ink:#18201c; --ink2:#3c4843; --muted:#647069; --rule:#dde3df; --panel:#ecf0ed;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --nb:#b8c0bb; --grid:#e3e8e5; --base:#c5ccc8;
  --f-head:"Noto Serif SC","Songti SC",serif; --f-body:"Noto Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --paper:#151a18; --ink:#e4e9e6; --ink2:#c2ccc7; --muted:#9aa8a1; --rule:#2a3430; --panel:#1c2421;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --nb:#4b5652; --grid:#26302c; --base:#3a4541; color-scheme:dark}}
:root[data-theme="dark"]{
  --paper:#151a18; --ink:#e4e9e6; --ink2:#c2ccc7; --muted:#9aa8a1; --rule:#2a3430; --panel:#1c2421;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --nb:#4b5652; --grid:#26302c; --base:#3a4541; color-scheme:dark}
body{background:var(--paper);color:var(--ink);font-family:var(--f-body);font-size:15px;line-height:1.75}
.wrap{max-width:1000px;margin:0 auto;padding-block:28px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
header{display:grid;gap:6px;border-bottom:2px solid var(--ink);padding-bottom:14px;margin-bottom:20px}
.eyebrow{font-size:12px;letter-spacing:.08em;color:var(--muted)}
h1{font-family:var(--f-head);font-size:clamp(26px,4vw,36px);line-height:1.25;margin:0;text-wrap:balance}
h2{font-family:var(--f-head);font-size:22px;margin:42px 0 8px;text-wrap:balance}
h3{font-size:16px;margin:24px 0 6px}
p,li{max-width:74ch} p{margin:0 0 10px}
.lede{color:var(--ink2)}
.claims{display:grid;gap:10px;margin:14px 0 6px;padding:0;list-style:none}
.claims li{padding:10px 14px;background:var(--panel);border-radius:3px;max-width:none}
.claims b{margin-right:6px}
.tbl{overflow-x:auto;margin:8px 0}
table{border-collapse:collapse;font-size:13.5px;width:100%;min-width:560px}
th,td{border-bottom:1px solid var(--rule);padding:6px 8px;text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12.5px}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin:10px 0}
.figs.two{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
figure{margin:12px 0;min-width:0}
figcaption{font-size:13px;color:var(--ink2);margin-top:6px;max-width:80ch}
.chart,.diagram{width:100%;height:auto;display:block;font-family:var(--f-body)}
.diagram{color:var(--ink);max-width:780px}
.chart .ct{font-size:12px;font-weight:600;fill:var(--ink)}
.chart .ax{font-size:10.5px;fill:var(--muted);font-variant-numeric:tabular-nums}
.chart .lab{font-size:12px;fill:var(--ink2)} .chart .lab.b{font-weight:600;fill:var(--ink)}
.chart .grp{font-size:12px;font-weight:600;fill:var(--ink)}
.chart .inbar{font-size:11px;fill:#fff;font-weight:600;pointer-events:none} .chart .inbar.dark{fill:var(--ink)}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .zero{stroke:var(--base);stroke-width:1.5}
.chart .vref{stroke:var(--muted);stroke-width:1}
.chart .band{fill:var(--panel)}
.chart .xhair{stroke:var(--muted);stroke-width:1}
.chart .hit{fill:transparent}
.chart .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .dot{stroke:var(--paper);stroke-width:2}
.chart .ci{stroke-width:2.5;stroke-linecap:round}
.chart .whisk{stroke:var(--ink2);stroke-width:1.2}
.c1.ln,.c1.ci{stroke:var(--s1)} .c1.dot,rect.c1{fill:var(--s1)}
.c2.ln,.c2.ci{stroke:var(--s2)} .c2.dot,rect.c2{fill:var(--s2)}
.c3.ln,.c3.ci{stroke:var(--s3)} .c3.dot,rect.c3{fill:var(--s3)}
.c4.ln,.c4.ci{stroke:var(--s4)} .c4.dot,rect.c4{fill:var(--s4)}
.cg.ln,.cg.ci{stroke:var(--muted)} .cg.dot,rect.cg{fill:var(--muted)}
rect.cnb{fill:var(--nb)}
.diagram .dl{font-size:12.5px;fill:currentColor} .diagram .muted{fill:var(--muted)} .diagram .strong{font-weight:600}
.diagram .cell{fill:var(--panel);stroke:none} .diagram .hit-cell{fill:var(--s1);fill-opacity:.22;stroke:var(--s1);stroke-width:1.5}
.diagram .blk{fill:var(--panel);stroke:var(--base)} .diagram .blk.hl{fill:var(--s1);fill-opacity:.25;stroke:var(--s1)}
.diagram .box{fill:var(--paper);stroke:var(--ink2);stroke-width:1}
.diagram .leaf{stroke:none} .diagram .k-r{fill:var(--s1);fill-opacity:.18} .diagram .k-u{fill:var(--s3);fill-opacity:.2} .diagram .k-nb{fill:var(--nb);fill-opacity:.6}
.diagram .arw{stroke:currentColor;stroke-width:1.2}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--ink2);margin:4px 0}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-1px;margin-right:5px}
.sw.c1{background:var(--s1)}.sw.c2{background:var(--s2)}.sw.c3{background:var(--s3)}.sw.c4{background:var(--s4)}.sw.cg{background:var(--muted)}.sw.cnb{background:var(--nb)}
details.dt{margin:2px 0 10px} details.dt summary{cursor:pointer;font-size:12.5px;color:var(--muted)}
details.dt summary:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
.note{font-size:13px;color:var(--muted)}
.tag{font-size:11.5px;padding:1px 6px;border:1px solid var(--rule);border-radius:3px;color:var(--ink2);white-space:nowrap}
.ok{color:var(--s1);font-weight:600} .no{color:var(--ink2);font-weight:600}
code{font-family:var(--f-mono);font-size:12px;background:var(--panel);padding:1px 4px;border-radius:2px;word-break:break-all}
#tip{position:fixed;z-index:10;pointer-events:none;background:var(--ink);color:var(--paper);font-size:12.5px;line-height:1.5;
  padding:6px 9px;border-radius:3px;max-width:320px;opacity:0;transition:opacity .08s}
#tip.on{opacity:1}
[tabindex]:focus-visible{outline:2px solid var(--s1);outline-offset:1px}
footer{margin-top:44px;border-top:1px solid var(--rule);padding-top:12px;font-size:12.5px;color:var(--muted)}
@media (prefers-reduced-motion: reduce){#tip{transition:none}}
"""

JS = """
(()=>{const tip=document.getElementById('tip');let xh=null;
function show(el,x,y){tip.textContent=el.dataset.tip;tip.classList.add('on');
  const r=tip.getBoundingClientRect();let L=x+14,T=y+14;if(L+r.width>innerWidth-8)L=x-r.width-14;if(T+r.height>innerHeight-8)T=y-r.height-14;
  tip.style.left=Math.max(8,L)+'px';tip.style.top=Math.max(8,T)+'px';
  if(el.dataset.xh){const l=document.getElementById(el.dataset.xh);if(l){l.setAttribute('x1',el.dataset.cx);l.setAttribute('x2',el.dataset.cx);l.setAttribute('visibility','visible');xh=l;}}}
function hide(){tip.classList.remove('on');if(xh){xh.setAttribute('visibility','hidden');xh=null;}}
document.addEventListener('pointermove',e=>{const el=e.target.closest&&e.target.closest('[data-tip]');if(el)show(el,e.clientX,e.clientY);else hide();});
document.addEventListener('focusin',e=>{const el=e.target.closest&&e.target.closest('[data-tip]');if(el){const r=el.getBoundingClientRect();show(el,r.right,r.top);}});
document.addEventListener('focusout',hide);document.addEventListener('scroll',hide,{passive:true});})();
"""


def build():
    cv = {r: curves(r) for r, *_ in RUNS}
    cl = {r: classes(r) for r, *_ in RUNS}
    meta = jl(f'{C.ROOT}/data/bench/build_meta.json'); mext = jl(f'{C.ROOT}/data/bench/build_meta_ext.json')
    P = []
    P.append('<title>Brewing 深度干预第一轮</title><meta name="claude:viewport-fit-sides" content="cover">')
    P.append('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400&family=Noto+Sans+SC:wght@400;600&family=Noto+Serif+SC:wght@600;700&display=swap">')
    P.append(f'<style>{CSS}</style><div id="tip" role="tooltip"></div><div class="wrap">')
    P.append('<header><div class="eyebrow">BREWING · 第一轮 · 2026-10-16</div><h1>Brewing 深度干预第一轮</h1>'
             '<p class="lede">关键 token 是四选一题的答案字母。这一轮逐层看它的答案怎样形成，按形成过程把题目分类，再试着找出不需要标签的逐层指标，用来决定早停还是多跑几层。'
             '共 6 个模型或格式配置，每个 22636 条 prompt。所有区间都是按题自助的 95% 区间。</p></header>')
    P.append('<ul class="claims">'
             '<li><b>现象成立。</b>每个模型上，答案都先被 probe 读出，再过 3–5 层才能由模型直接说出来。按全集分类，R 占 15–47%，O 只有 0.3–1.4%，'
             '真正"可得但没就绪"的 U 占 4–8%，从未可得的 NB 占 47–77%。</li>'
             '<li><b>可以抽成无标签指标。</b>Patchscope 读出在"发射层"的跳变是一个完全不用训练的信号；用 4 个置换版本的共识当目标训练的读出头，不用标签也能达到金标签 probe 的读出准确率。</li>'
             '<li><b>用指标做干预，省算力成立，提准确率只成立一部分。</b>早停在一致率 99% 时平均停在 L25–L29（完整为 L35），但不会把错题改对。'
             '准确率的提升来自共识读出头去掉字母偏置（+0.4 到 +1.6pp），以及后训练模型上的 depth loop（+0.4 到 +0.9pp）。用指标决定"何时 loop"还没做成。</li></ul>')

    # ---------------- 1. data & method
    P.append('<h2>1. 数据与方法</h2>')
    rows = [['GSM-MC', '小学数学应用题', '1319', 'test 全量'], ['MATH-MC', '竞赛数学', '1000', '从 4817 道有效 test 题中抽取'],
            ['PythonIO', '读程序判断输出', '1000', '从 1684 道中抽取'], ['MMLU-Pro', '多学科知识', '1000', '10 选抽成 4 选'],
            ['C-Eval', '中文考试题', '1340', 'val 全量，3 道留作示例题']]
    P.append(table(['来源', '题型', '题数', '取法'], rows, num=(2,)))
    P.append(f'<p>核心集共 {meta["n_questions"]} 题。每题做 4 个循环置换版本，共 22636 条 prompt。'
             f'另有一个 O 扩样集（{mext["n_questions"]} 道计算题，45456 条），只在第一对模型上跑，用来给 O 类补样本，不参与下面的分布结论。</p>')
    P.append(f'<figure>{diag_versions()}<figcaption>同一道 GSM 题的 4 个版本。选项按位置循环移动，正确内容"18"在 A、B、C、D 各出现一次。'
             '判定时要求 4 个版本都指向同一个内容，所以模型偏爱某个字母不会被误认成"答对了"。</figcaption></figure>')
    rows = [['Qwen3-4B-Base', 'base', '1-shot 补全', '36'], ['Qwen3-4B（后训练）', '后训练', 'chat（system 指令，非思考）；另跑一次 1-shot 作格式对照', '36'],
            ['Qwen2.5-7B', 'base', '1-shot 补全', '28'], ['Qwen2.5-7B-Instruct', '后训练', 'chat', '28'], ['Llama-3.1-8B', 'base', '1-shot 补全', '32']]
    P.append(table(['模型', '类型', 'prompt 格式', '层数'], rows, num=(3,)))
    P.append(f'<figure>{diag_method()}<figcaption>每一层取关键 token 位置的隐状态，用两种方法读。'
             'linear probe 是每层单独训练的 4 选 1 分类器（在 logit lens 上加一个线性修正，5 折交叉拟合），回答"这一层里答案能不能被读出来"。'
             'Patchscope 不训练：把这一层的状态放进一个只有 "Answer:\\n" 的空 prompt 末尾，跑完剩下的层，看模型说出哪个字母，回答"这一层的状态能不能直接被说出来"。'
             '第一篇的 CSD 会再减去空 prompt 自己的 logits，这里只作敏感性对照。</figcaption></figure>')
    P.append(f'<figure>{diag_classes()}<figcaption>每道题按 4 个版本一起判定。"可得"要求 probe 连续 3 层在 4 个版本上都读对；'
             '"就绪"要求 probe 和 Patchscope 在同一段连续 3 层上都读对。最后一层的 probe 读数用真实输出代替，所以 R 就是"4 个版本最终全对"。'
             '与第一篇一致，M 和 U 都要求答案曾经可得；从未可得的题单列为 NB。</figcaption></figure>')

    # ---------------- 2. brewing & classes
    P.append('<h2>2. brewing 现象与分类占比</h2><h3>2.1 可得早于就绪</h3>')
    P.append('<div class="legend"><span><i class="sw c1"></i>linear probe（可得）</span><span><i class="sw c2"></i>Patchscope（就绪）</span>'
             '<span><i class="sw c3"></i>logit lens</span><span><i class="sw" style="background:var(--panel);border:1px solid var(--rule)"></i>间隔：probe 与 Patchscope 各自达到半程准确率的层之间</span></div>')
    figs = []
    trows = []
    for r, n, kind, short in RUNS:
        c = cv[r]
        figs.append('<figure>' + line_chart([('probe', 'probe', c['probe'], 'c1'), ('patch', 'Patchscope', c['patch'], 'c2'), ('lens', 'lens', c['lens'], 'c3')],
                                            c['L'], n, band=(c['hm_probe'], c['hm_patch'])) + '</figure>')
        trows.append([E(n), f'L{c["hm_probe"]}', f'L{c["hm_patch"]}', f'{c["hm_patch"]-c["hm_probe"]}', f'{100*c["probe"][-1]:.1f}%', f'{100*cl[r]["acc"]:.1f}%'])
    P.append('<div class="figs">' + ''.join(figs) + '</div>')
    P.append('<figcaption>纵轴是单版本准确率，0.25 是随机水平。悬停可看每一层三条曲线的数值。</figcaption>')
    P.append(datatable(['配置', 'probe 半程层', 'Patchscope 半程层', '间隔（层）', '末层 probe', '真实输出准确率'], trows, num=(1, 2, 3, 4, 5)))
    P.append('<ul><li>6 个配置上都是 probe 先读出，Patchscope 晚 3–5 层才读出。lens 和 Patchscope 几乎同时跳起，说明"能说出来"是在发射层一步完成的。</li>'
             '<li>同一家族里，base 和后训练版的可得层、发射层相同，后训练没有改变形成时间。后训练版末层的 probe 比真实输出高约 2pp，差在末层的字母偏置上，B 线输出里 A、B 两个字母各占 31%、32%。</li></ul>')
    P.append('<h3>2.2 全集分类：补上 NB</h3>')
    keys = ['R', 'O', 'U', 'M', 'NB']
    clsmap = {'R': 'c1', 'O': 'c2', 'U': 'c3', 'M': 'c4', 'NB': 'cnb'}
    labels = {'R': 'R 就绪且全对', 'O': 'O 就绪后丢失', 'U': 'U 可得未就绪', 'M': 'M 可得但一致答错', 'NB': 'NB 从未可得'}
    P.append('<div class="legend">' + ''.join(f'<span><i class="sw {clsmap[k]}"></i>{labels[k]}</span>' for k in keys) + '</div>')
    items = []
    trows = []
    for r, n, kind, short in RUNS:
        lab = cl[r]['lab']
        cnt = {k: int((lab == k).sum()) for k in keys}
        items.append((n, cnt))
        tot = sum(cnt.values())
        trows.append([E(n)] + [f'{cnt[k]}（{100*cnt[k]/tot:.1f}%）' for k in keys] + [f'{cl[r]["nb_wrong"]}', f'{int((lab=="X").sum())}'])
    P.append('<figure>' + stacked(items, keys, {k: v for k, v in clsmap.items()}, labels) + '</figure>')
    P.append(datatable(['配置'] + [labels[k] for k in keys] + ['NB 中一致答错', '排除 X'], trows, num=(1, 2, 3, 4, 5, 6, 7)))
    srows = []
    for r in ('qwen3-4b-base__base_fewshot', 'qwen3-4b__chat0'):
        lab, src = cl[r]['lab'], cl[r]['src']
        for s in C.SOURCES:
            m = src == s
            srows.append([E(NAME[r]), s] + [f'{100*np.mean(lab[m]==k):.1f}%' for k in keys])
    P.append('<figcaption>悬停可看每段的题数。O、U、M 太窄放不下标签，具体数字见下方两张表。</figcaption>')
    P.append(datatable(['配置', '来源'] + keys, srows, num=(2, 3, 4, 5, 6)))
    P.append('<ul><li><b>NB 是最大的一类</b>，占 47–77%。上一版报告把它混进了 U 和 M：一致答错的题几乎都属于 NB，答案从来不可得，所以"M 可得但一致答错"接近 0。'
             '这说明一致答错基本是模型不会，而不是先想对、后来改错。</li>'
             '<li><b>真正的 U 只占 4–8%</b>，是 depth loop 理论上的目标。后训练模型的 U 更多（B 线 8.1%），loop 也正是在后训练模型上有效（第 4 节）。</li>'
             '<li><b>O 很少</b>，但高于对照：把正确内容换成任一错误内容、套用同一规则得到的"假 O"，A 线只有 0.56%，真 O 是 1.40%，差 +0.84pp [0.52, 1.16]。'
             'O 在计算题上更多（A 线 79 道里 GSM 占 33 道），这也是另建 O 扩样集的原因。</li></ul>')

    # ---------------- 3. unsupervised indicators
    P.append('<h2>3. 从现象到无监督指标</h2>')
    P.append('<p>"无监督"在这里分两级：<span class="tag">L0</span> 完全不训练，只用模型自己的读出和隐状态；<span class="tag">L1</span> 训练一个小读出头，'
             '但目标来自模型自己：4 个置换版本的共识内容。两级都不用金标签。所有阈值和参数只在第 0–3 折上选，第 4 折作锁箱只评一次。</p>')
    P.append('<h3>3.1 为什么能抽出指标：答案先藏在读出方向之外</h3>')
    g = jl(f'{R}/qwen3-4b-base__base_fewshot/abne/geometry.json')
    Ls = sorted(int(k) for k in g['per_layer'])
    pl = g['per_layer']
    full = [pl[str(l)]['acc']['full_probe'] for l in Ls]
    perp = [pl[str(l)]['acc_probe_parts']['perp_only'] for l in Ls]
    ro = [pl[str(l)]['acc']['readout3'] for l in Ls]
    rnd = [float(np.mean(pl[str(l)]['acc']['random3'])) if isinstance(pl[str(l)]['acc']['random3'], list) else pl[str(l)]['acc']['random3'] for l in Ls]
    P.append('<div class="legend"><span><i class="sw c1"></i>完整 probe</span><span><i class="sw c2"></i>只用读出方向之外的分量</span>'
             '<span><i class="sw c3"></i>只用 A–D 读出方向（3 维）</span><span><i class="sw cg"></i>随机 3 维子空间</span></div>')
    P.append('<figure>' + line_chart([('full', '完整 probe', full, 'c1'), ('perp', '读出方向之外', perp, 'c2'), ('ro', 'A–D 读出方向', ro, 'c3'), ('rnd', '随机 3 维', rnd, 'cg')],
                                     36, 'Qwen3-4B-Base：答案信息在哪个子空间里', y0=0.15, y1=0.7, w=760, h=260, x0=Ls[0], vline=(g['e'], f'发射层 L{g["e"]}')) +
             '<figcaption>A–D 读出方向就是 4 个字母的 unembedding 在隐空间里张成的方向，模型最终靠它们说出字母。发射层之前，完整 probe 已经能读到 .42–.58，'
             '但只用读出方向读不比随机子空间好，信息全在读出方向之外。到发射层 L24，读出方向一下子变得和完整 probe 一样准。'
             'Qwen3-4B 后训练版和 Llama 也是同样的形状。</figcaption></figure>')
    P.append(datatable(['层'] + ['完整 probe', '读出方向之外', 'A–D 读出方向', '随机 3 维'],
                       [[f'L{l}', f'{a:.3f}', f'{b:.3f}', f'{c:.3f}', f'{d:.3f}'] for l, a, b, c, d in zip(Ls, full, perp, ro, rnd)], num=(1, 2, 3, 4)))
    cr = jl(f'{R}/qwen3-4b-base__base_fewshot/abne/causal/report.json')
    groups = ['L20', 'L21', 'L22']
    series = [('perp@0.2', '读出方向之外的类方向', 'c1'), ('par@0.2', 'A–D 读出方向上的投影', 'c2'), ('rand@0.2', '随机方向', 'c3'), ('perp_other@0.2', '另一类的方向', 'c4')]
    vals = {gg: {k: tuple(100 * x for x in cr[f'{gg}|U_avail']['conds'][k]['d_final_is_gold']) for k, _, _ in series} for gg in groups}
    emis = {gg: {k: cr[f'{gg}|U_avail']['conds'][k]['d_csd_commit_layer_k'] for k, _, _ in series} for gg in groups}
    P.append('<h3>3.2 这个方向在因果上起作用</h3>')
    P.append('<div class="legend">' + ''.join(f'<span><i class="sw {c}"></i>{l}</span>' for _, l, c in series) + '</div>')
    P.append('<figure>' + grouped_bars(groups, series, vals) +
             '<figcaption>Qwen3-4B-Base，100 道 U 题（400 行），在第 20–22 层把隐状态沿某个方向推 0.2 倍自身长度，再跑完剩下的层。纵轴是准确率变化（pp），竖线是 95% 区间。'
             '推"probe 认为的那一类在读出方向之外的分量"，准确率 +5.8 到 +7.5pp，发射平均提前约 2 层；同样长度的随机方向无效，读出方向上的投影效果小且不稳定（−0.3 到 +3.0pp），推另一类则略降（不显著）。'
             '这一组实验用的是金标签 probe 的方向（L2），是机制验证，不是无监督干预。</figcaption></figure>')
    P.append(datatable(['层', '方向', '准确率变化（pp）', '发射层变化（层）'],
                       [[gg, l, f'{vals[gg][k][0]:+.1f} [{vals[gg][k][1]:+.1f}, {vals[gg][k][2]:+.1f}]', f'{emis[gg][k][0]:+.2f}'] for gg in groups for k, l, _ in series], num=(2, 3)))
    P.append('<h3>3.3 两个能用的无监督指标</h3>')
    rows = [['发射跳变', '<span class="tag">L0</span>', 'Patchscope 读出在相邻层之间的分布跳变、置信度、连续稳定层数等，不训练任何东西',
             '这一层的读出是否已经和满预算输出一致', '早停（4.1）'],
            ['置换共识读出头', '<span class="tag">L1</span>', '与 probe 同样的结构，目标换成"4 个版本最终输出的多数内容"；推理时只看一个版本',
             '逐层读出准确率与金标签 probe 相差 ±0.6pp 以内（Qwen3 两条线）', '读出去偏（4.2）、推送方向（4.4）']]
    P.append(table(['指标', '级别', '怎么算', '和什么对齐', '用在哪'], rows))
    P.append('<p>其余候选都没有成立。ABNE 量本身就是 probe margin 减去 lens margin，扣掉 probe 自身置信度后不再有预测力，只有它的方向有用。'
             '扰动稳定性指 loop 前后答案是否不变，弱于置信度。区分 O 和"末层改对了"的题，单次前向和 4 次前向都做不到。'
             '预测哪些题会被 loop 修好，AUROC 约 0.5。</p>')

    # ---------------- 4. interventions
    P.append('<h2>4. 用指标做干预：早停、跳层、loop</h2><h3>4.1 早停：省层数，一致率不变</h3>')
    panels = []
    trows = []
    for r, ttl in (('qwen3-4b-base__base_fewshot', 'Qwen3-4B-Base（36 层）'), ('qwen3-4b__chat0', 'Qwen3-4B 后训练（36 层）')):
        pol = jl(f'{R}/{r}/stage2/analysis.json')['exit']['csd']['policy']
        ser = {}
        for key, lab, cls in (('L0', '发射跳变（L0，含 Patchscope 特征）', 'c1'), ('L0nocsd', '只用 lens 与隐状态（L0）', 'c3'), ('base', '置信度稳定规则', 'c2')):
            tts = ('0.95', '0.98', '0.99')
            pts = [(pol[f'{key}@{t}']['mean_exit'], pol[f'{key}@{t}']['agree']) for t in tts]
            ser[key] = (lab, cls, pts)
            for t in tts:
                q = pol[f'{key}@{t}']
                trows.append([ttl, lab, f'{100*float(t):.0f}%', f'L{q["mean_exit"]:.1f}', f'{100*q["agree"]:.2f}%', f'{100*q["acc"]:.1f}%'])
        panels.append((ttl, pol['oracle_mean_exit'], ser))
    P.append('<div class="legend"><span><i class="sw c1"></i>发射跳变（L0，含 Patchscope 特征）</span><span><i class="sw c3"></i>只用 lens 与隐状态（L0）</span>'
             '<span><i class="sw c2"></i>置信度稳定规则</span><span><i class="sw cg"></i>oracle（知道答案何时定型）</span></div>')
    P.append(pareto(panels))
    P.append('<figcaption>横轴是平均停止层，越靠左省得越多；纵轴是停下时的读数与完整前向输出一致的比例。'
             '一致率 99% 时，发射跳变规则在后训练版上平均停在 L25.2，置信度规则要到 L30.7，oracle 是 L23.5。'
             '含 Patchscope 特征的规则需要额外跑剩余层的单 token 读出，只用 lens 和隐状态的版本代价最低，停在 L27.0。</figcaption>')
    P.append(datatable(['模型', '规则', '目标一致率', '平均停止层', '实际一致率', '准确率'], trows, num=(3, 4, 5)))
    P.append('<h3>4.2 早停能不能把错题改对：不能，收益来自读出头</h3>')
    trows = []
    for r, n, kind, short in RUNS:
        ee = jl(f'{R}/{r}/early_exit/summary.json')
        if not ee:
            continue
        A1 = ee.get('A1') or {}
        rules = A1.get('rules', {})
        best = max(rules.items(), key=lambda kv: kv[1]['net_acc_change'][0]) if rules else None
        noex = A1.get('head_at_L1_no_exit', {})
        trows.append([E(n), f'{best[0]} {100*best[1]["net_acc_change"][0]:+.2f}pp' if best else '—',
                      f'{100*noex.get("net_acc_change",[0])[0]:+.2f}pp' if noex else '—', ('未通过' if A1.get('verdict_vs_decoyO') == 'FAIL' else E(A1.get('verdict_vs_decoyO', '')))])
    P.append(table(['配置', '早停规则中净变化最高者', '同一个 probe 放在末层、不早退', '对照"假 O"'], trows, num=(1, 2)))
    P.append('<p>O 确实依赖深度：A 线 probe 在 L24–28 能读对 67/79 道 O，到 L35 只剩 23/79。但末层同样会改掉中层形成的错误内容，这类"假 O"比 O 还多，'
             '早停会把它们一起"还原"，所以没有一个配置在这个对照下通过。和"不早停、只换读出头"相比，6 个配置里 4 个是不早停更高；另外 2 个（后训练版、Llama）早停只多 0.05 和 0.4pp。'
             '所以早停表上的收益基本来自读出头。</p>'
             '<p>换成无标签的置换共识头放在末层读出：后训练版 +1.59pp、Llama +0.88pp、Qwen3-4B-Base +0.42pp；后训练版按无标签规则选出的读出层在锁箱上 +1.60pp。'
             '这是去掉末层字母偏置带来的，和深度无关，但它确实是一个不用标签就能拿到的准确率提升。</p>')
    P.append('<h3>4.3 depth loop：后训练模型上有效</h3>')
    P.append('<p>做法：只在关键 token 位置把一段中间层重复跑一遍或两遍，上下文不变，权重不变。对照是重复后段层的安慰剂（不改变形成过程）。'
             '指标是"方向净值减安慰剂"：每道 U 题变成 R 或 O 记 +1，变成 M 记 −1，取平均后减去安慰剂的同一数值。</p>')
    fr = []
    for r, ttl in (('qwen3-4b__chat0', 'Qwen3-4B 后训练'), ('qwen2.5-7b-instruct__chat0', 'Qwen2.5-7B-Instruct'),
                   ('qwen3-4b-base__base_fewshot', 'Qwen3-4B-Base'), ('qwen2.5-7b__base_fewshot', 'Qwen2.5-7B'), ('llama3.1-8b__base_fewshot', 'Llama-3.1-8B')):
        d = direction(r)
        if not d:
            continue
        post = 'Instruct' in ttl or '后训练' in ttl
        fr.append((f'##{ttl}（{"后训练" if post else "base"}，对照 {d["placebo"]}）', 0, 0, 0, '', False))
        prim = {'qwen3-4b__chat0': 'w18_25_k2', 'qwen3-4b-base__base_fewshot': 'w18_25_k2', 'qwen2.5-7b__base_fewshot': 'w14_19_k2',
                'qwen2.5-7b-instruct__chat0': 'w14_19_k2', 'llama3.1-8b__base_fewshot': 'w16_22_k2'}[r]
        for k in sorted(d['res']):
            e_, lo, hi = d['res'][k]
            fr.append((f'{k}{"（主检验）" if k == prim else ""}', e_, lo, hi, 'c1' if post else 'c2', k == prim))
    P.append('<div class="legend"><span><i class="sw c1"></i>后训练模型</span><span><i class="sw c2"></i>base 模型</span></div>')
    P.append('<figure>' + forest(fr) + '<figcaption>每行一个重复窗口，例如 w18_25_k2 表示把第 18–25 层多跑两遍。点是估计值，横线是 95% 区间，区间在 0 右侧表示比安慰剂更多地把 U 推向正确。'
             '主检验是事先指定的配置。</figcaption></figure>')
    trows = []
    for r, n, kind, short in RUNS:
        rn = row_net(r)
        if rn:
            trows.append([E(n)] + ['；'.join(f'{k} {v:+.2f}' for k, v in sorted(rn.items()))])
    P.append(datatable(['配置', '各窗口的单版本准确率净变化（pp，相对同批次不重复）'], trows))
    P.append('<ul><li>两个后训练模型几乎所有窗口都显著，主检验分别 +1.36pp [0.38, 2.39] 和 +2.11pp [0.95, 3.23]；单版本准确率净 +0.4 到 +0.9pp。'
             'base 模型只有个别窗口显著，Llama 全不显著。</li>'
             '<li>只有覆盖"可得到就绪"那段层的窗口有效。Qwen3 上 L10–17、L12–19 这类更早的窗口没有效果，从 L14 起才开始有效。</li>'
             '<li>代价：3–13% 原本全对的题会被改坏。用 L0/L1 指标挑"该 loop 的题"，没有一个门控胜过"一直 loop"，修好对改坏的 AUROC 约 0.5。</li></ul>')
    P.append('<h3>4.4 沿共识头方向推一把：机制成立，收益小</h3>')
    rows = [['Qwen3-4B 后训练', '+0.17 [−0.06, 0.40]', '<span class="ok">+0.48 [0.02, 0.99]</span>', '胜过随机、次选类和读出投影三种对照'],
            ['Qwen3-4B-Base', '−0.16 [−0.27, −0.03]', '−0.20 [−0.48, 0.07]', '无收益'],
            ['Qwen2.5-7B-Instruct', '+0.01 [−0.07, 0.10]', '−0.04 [−0.18, 0.09]', '没有复现'],
            ['Qwen2.5-7B', '+0.03 [−0.02, 0.09]', '−0.02 [−0.11, 0.04]', '无收益']]
    P.append(table(['模型', 'dev（交叉拟合，pp）', '锁箱（pp）', '说明'], rows))
    P.append('<p>把 3.2 节的推送换成完全无标签的版本：方向来自共识头的预测类，层、幅度、门控都按无标签判据选定，作用在全部题目上。'
             '只有一个模型在锁箱上显著。瓶颈在共识头本身：中间层的共识头和最终输出不一致时，多数是共识头错了，推的方向也就错了。</p>')
    P.append('<h3>4.5 汇总：做成热插拔工具，每个部件现在值多少</h3>')
    rows = [['早停', '发射跳变（L0）', '一致率 99% 时平均停在 L25–L29（完整为 L35）', '准确率持平', '两个 Qwen3 模型'],
            ['读出头', '置换共识头（L1）', '无', '+0.4 到 +1.6pp，后训练模型最大', 'Qwen3、Llama'],
            ['depth loop', '无门控，只在关键 token、brewing 窗口', '每个关键 token 多跑 8–16 层', '+0.4 到 +0.9pp（后训练）；base ≤ 0', '两个后训练模型'],
            ['推送', '共识头方向（L1）', '几乎为 0', '+0.48pp，只在一个模型上成立', 'Qwen3-4B 后训练'],
            ['loop 门控', 'L0 / L1 指标', '—', '没有胜过一直 loop', '暂无']]
    P.append(table(['部件', '用到的指标', '算力', '准确率', '在哪些模型上成立'], rows))

    # ---------------- 5
    P.append('<h2>5. 局限与下一步</h2><ul>'
             '<li>O 太少：最大的一个扩样集也只有 234 道，O 相关的结论区间都很宽。</li>'
             '<li>关键 token 目前只有选择题字母。CoT 中的关键 token 还没做，那里的"可得未就绪"可能更多。</li>'
             '<li>loop 的收益只在后训练模型上出现，原因还没查清。U 的比例更高可能是一部分原因。</li>'
             '<li>下一步最值得做的是找一个比置信度更好的"可救"门控，目标是 U（可得未就绪）。现在 U 的识别还依赖 probe 或 4 次前向。</li></ul>')
    P.append('<footer>完整日志：<code>exp/depth_intervention/RESULTS.md</code>。实验清单：<a href="https://claude.ai/artifact/5zFpwEmeNqK6t8tEx4b9Fc">Brewing 第一轮实验清单</a>。'
             '页面由 <code>scripts/gen_report_v2.py</code> 从结果文件生成；第 3、4 节的部分数字摘自经复核的 RESULTS.md。</footer></div>')
    P.append(f'<script>{JS}</script>')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(P))
    print('wrote', OUT, os.path.getsize(OUT))


if __name__ == '__main__':
    build()
