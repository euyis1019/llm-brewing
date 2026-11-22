"""Round-1 experiment inventory page (report/round1_inventory.html): what ran, when, on what, result, where.
Times come from queue/gpu*.history and log birth/mtime stamps (2026-10-16, machine clock)."""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import html, os

ROOT = str(_experiment_root('depth_intervention'))
OUT = os.path.join(ROOT, 'report', 'round1_inventory.html')
E = html.escape

# (lane, start, end, label, kind)   kind: prep | extract | probe | loop | steer | fail
JOBS = [
    ('GPU0', '19:17', '19:18', '冒烟 A', 'prep'), ('GPU0', '19:20', '19:26', '提取 A', 'extract'),
    ('GPU0', '19:27', '19:29', '冒烟 Llama', 'prep'), ('GPU0', '19:29', '19:40', '提取 Llama', 'extract'),
    ('GPU0', '19:41', '19:54', '提取 A-ext', 'extract'), ('GPU0', '19:57', '20:00', 'probe ×3', 'probe'),
    ('GPU0', '19:54', '20:12', 'loop last A', 'loop'), ('GPU0', '20:12', '20:57', 'loop all A', 'loop'),
    ('GPU0', '20:57', '21:14', 'loop last Llama', 'loop'), ('GPU0', '21:14', '21:15', '', 'fail'),
    ('GPU0', '21:17', '21:29', 'Qwen2.5 流水线', 'extract'), ('GPU0', '21:30', '21:54', 'loop last Q2.5', 'loop'),
    ('GPU0', '22:50', '22:51', '', 'fail'), ('GPU0', '23:01', '23:16', '推送 Q2.5-I', 'steer'),
    ('GPU1', '19:17', '19:20', '冒烟 B、chat 格式', 'prep'), ('GPU1', '19:21', '19:27', '提取 B', 'extract'),
    ('GPU1', '19:27', '19:34', '提取格式对照', 'extract'), ('GPU1', '19:35', '19:47', '提取 B-ext', 'extract'),
    ('GPU1', '19:48', '20:07', 'loop last B', 'loop'), ('GPU1', '19:57', '20:00', 'probe ×3', 'probe'),
    ('GPU1', '20:07', '21:03', 'loop all B', 'loop'), ('GPU1', '21:03', '21:26', '早窗口 B、A', 'loop'),
    ('GPU1', '21:26', '21:37', 'Q2.5-I 流水线', 'extract'), ('GPU1', '21:37', '21:53', 'loop last Q2.5-I', 'loop'),
    ('GPU1', '21:59', '22:49', 'loop all Q2.5-I', 'loop'), ('GPU1', '22:49', '23:24', 'loop all Q2.5', 'loop'),
    ('GPU1', '23:24', '23:41', '推送 Q2.5', 'steer'),
    ('agent', '19:34', '19:42', 'loop 调试', 'loop'), ('agent', '20:27', '20:40', '共识头 ×4', 'probe'),
    ('agent', '20:40', '20:49', 'ABNE 因果', 'steer'), ('agent', '21:35', '21:58', '推送 A', 'steer'),
    ('agent', '21:58', '22:27', '推送 B', 'steer'),
]
T0, T1 = 19 * 60, 23 * 60 + 45


def mins(t):
    h, m = t.split(':')
    return int(h) * 60 + int(m)


def gantt():
    w, lane_h, pad_l, pad_t = 980, 46, 64, 28
    lanes = ['GPU0', 'GPU1', 'agent']
    names = {'GPU0': 'GPU0', 'GPU1': 'GPU1', 'agent': 'subagent'}
    h = pad_t + lane_h * len(lanes) + 26
    X = lambda m: pad_l + (w - pad_l - 10) * (m - T0) / (T1 - T0)
    s = [f'<svg viewBox="0 0 {w} {h}" class="gantt" role="img" aria-label="GPU 时间线">']
    for t in range(T0, T1 + 1, 30):
        s.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{pad_t-6}" y2="{h-22}" class="tick"/>'
                 f'<text x="{X(t):.1f}" y="{h-8}" class="tl" text-anchor="middle">{t//60}:{t%60:02d}</text>')
    for i, ln in enumerate(lanes):
        y = pad_t + i * lane_h
        s.append(f'<text x="{pad_l-8}" y="{y+lane_h/2+4:.1f}" class="ln" text-anchor="end">{names[ln]}</text>')
        rows = [j for j in JOBS if j[0] == ln]
        # two sub-rows so overlapping jobs (probe during loop) stay readable
        ends = [0, 0]
        for lane, a, b, lab, kind in sorted(rows, key=lambda j: mins(j[1])):
            r = 0 if mins(a) >= ends[0] else 1
            ends[r] = mins(b)
            yy = y + 4 + r * 19
            x0, x1 = X(mins(a)), X(max(mins(b), mins(a) + 1))
            s.append(f'<rect x="{x0:.1f}" y="{yy}" width="{x1-x0:.1f}" height="16" rx="2" class="k-{kind}"><title>{E(lab)} {a}–{b}</title></rect>')
            if lab and x1 - x0 > 7 * len(lab) + 6:
                s.append(f'<text x="{x0+4:.1f}" y="{yy+12}" class="bl">{E(lab)}</text>')
    s.append('</svg>')
    return ''.join(s)


def table(head, rows, cls=''):
    h = ''.join(f'<th>{x}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in rows)
    return f'<div class="tbl"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


P = 'results/{model}__{fmt}/'
SECTIONS = [
    ('prep', '准备：数据、格式与核心代码', [
        ['构造 benchmark', '5 个来源', '5659 题 × 4 个循环置换版本 = 22636 条；O 扩样集 11364 题（GSM train + MATH 剩余）', '几分钟（CPU）',
         '5 折按题组划分；每个来源 3 道示例题，答案字母互不相同', '<code>data/bench/</code>'],
        ['few-shot 的 k 选择', 'Qwen3-4B-Base、Llama、Qwen2.5-7B', '每个来源 50 题 × 4 版本 × k∈{0,1,3}', '每个模型约 2 分钟',
         '统一用 1-shot：所有来源格式遵从率 100%，0-shot 时 MATH/MMLU-Pro 只有 96.5%', '<code>data/bench/fewshot_k__*.json</code>'],
        ['chat 格式探索', 'Qwen3-4B 后训练版', '5 种格式 × 5 个来源 × 200 条', '36 秒',
         '只给 user 指令时 GSM/MATH 遵从率仅约 2%；加 system 指令后 100%，定为 chat0', '<code>results/qwen3-4b__chat0/explore_chat_formats.json</code>'],
        ['核心实现校验', 'A、B、Llama、Qwen2.5', '8 条 prompt × 8 层', '每个模型 < 20 秒',
         '前缀 KV 的 CSD 与直接 hook 的 Patchscope argmax 一致率 100%；层序列复现正常前向，差值 0', '<code>results/*/smoke.json</code>'],
    ]),
    ('extract', '读出：提取隐状态、CSD、probe', [
        ['提取 + CSD（每次同时完成）', '8 个运行：A、B、格式对照、Llama、Qwen2.5-7B、Qwen2.5-7B-Instruct、A-ext、B-ext', '核心集每次 22636 条，ext 每次 45456 条；36/28/32 层',
         '6–13 分钟/次', '准确率：A 66.1%，B 56.8%，格式对照 59.1%，Llama 41.0%，Q2.5 62.5%，Q2.5-I 58.6%', '<code>results/*/extract/</code>（4–8GB/次）'],
        ['lens 锚定线性 probe', '同上 8 个运行', '每层 5 折交叉拟合 × 11 个 λ + 3 个 null probe', '2–3 分钟/次',
         'probe 半程层比 CSD 早 3–5 层；null（按自身标签）为 0.244–0.26', '<code>results/*/probe/</code>'],
        ['置换共识 probe（L1，不用标签）', 'A、B、Llama、A-ext、Q2.5、Q2.5-I', '同样的 probe 形式，目标换成 4 版本共识内容', '3–6 分钟/次',
         'Qwen3 上逐层与金标签 probe 相差 ±0.6pp', '<code>results/*/consensus_probe/</code>'],
    ]),
    ('probe', '四类判定与 early exit', [
        ['v1/v0 四类判定 + 对照', '8 个运行 × CSD raw/减基线 × K=1,2,3', '题级判定加 null、decoy、decoy-O 对照，2000 次自助', '每次几秒（CPU）',
         'O 0.3–2.1%，高于 decoy；U 42–76%', '<code>results/*/classify/</code>'],
        ['口径修订后重跑', '同上', '末层 probe 改用真实输出（last_layer=output）', '几分钟',
         '103（A）/ 128（B）道"最终 4/4 正确"的题从 U 移到 R', '<code>classify/</code>，旧版在 <code>classify_lastprobe/</code>'],
        ['early exit（A1）', '8 个运行', 'E-head、E-csd 两种头 × fix、stab、conf 三种规则，按折交叉选参', '每次约 12 秒',
         '字面判据 7/8 通过；对照 decoy-O 8/8 不通过；收益不超过"probe 放末层不早退"', '<code>results/*/early_exit/</code>'],
    ]),
    ('loop', 'depth loop（A2）', [
        ['mode last，全部题', 'A、B（10 个配置）、Llama（6 个）、Q2.5、Q2.5-I（6 个）', '22636 行 × 配置数，每步存 CSD、lens 和隐状态', '17–24 分钟/模型',
         '后训练模型方向净值减安慰剂 +1.1 到 +2.1pp；base 模型和 Llama 不显著', '<code>results/*/loop_allq/</code>'],
        ['更早的窗口', 'A、B', 'w10_17、w12_19、w14_21、安慰剂 p28_33', '11–12 分钟/模型',
         '可得之前的窗口无效；从 L14 起才开始有效', '<code>results/*/loop_allq_early/</code>'],
        ['mode all，目标子集', 'A、B（10 个配置）、Q2.5、Q2.5-I（6 个）', '全部 U 加 R/M/O 各 300 题', '35–56 分钟/模型',
         '后训练模型 4/4 个窗口显著；base 模型不一致；R 损伤更大', '<code>results/*/loop/</code>'],
        ['loop 评估', '上面全部', '旧口径和新口径各算一遍；同批次 none 作参照', '每次约 5 分钟（CPU）',
         '主检验：后训练模型 PASS，base 模型 FAIL', '<code>loop*/eval/</code>、<code>loop*/eval_lo/</code>'],
    ]),
    ('steer', '第二阶段：无标签指标', [
        ['逐层特征与分类任务', 'A、B', '274 个题级特征，L0–L3 分级，交叉拟合', '约 15 分钟/模型（CPU）',
         'L0 发射跳变可提前 2–5 层停止，一致率 99%；单次前向的特征只比置信度多 0.02–0.08', '<code>results/*/stage2/</code>'],
        ['跨版本一致性门控（4 次前向）', 'A、B', '中层 4 版本一致、末层不一致时取中层', '几秒',
         '救回 O 78/79，但 decoy-O 也被同样还原，净值约 0', '<code>scripts/explore/consistency_exit.py</code>'],
        ['回读门控（单次前向）', 'A、B、Llama、A-ext、B-ext、格式对照', '中层对、末层错，还是相反；L0/L1/L3 检测器', '约 30 分钟（CPU）',
         'AUROC 最多比置信度高 +0.04；B 线收益来自去字母偏置', '<code>results/*/retroread/</code>'],
        ['loop 门控与扰动稳定性', 'A、B、Llama', '预测哪些行会被 loop 修好；loop 不变性作为就绪指标', '约 20 分钟（CPU）',
         '修好对弄坏的 AUROC 约 0.5；稳定性弱于置信度', '<code>results/*/loopgate/</code>'],
        ['"已可得、未发射"（ABNE）', 'A、B、Llama、ext', '几何分析、预测、因果推送（200 题 × 3 层 × 4 种方向）', '因果部分约 9 分钟 GPU',
         '不是新的标量；推它的补空间方向会让发射提前约 2 层', '<code>results/*/abne/</code>'],
        ['L1 补空间推送（全样本）', 'A、B、Q2.5、Q2.5-I', '42 个配置打包，22636 行，第 4 折锁箱', '15–30 分钟/模型',
         '只有 B 显著（锁箱 +0.48pp）；Q2.5-I 没有复现', '<code>results/*/steer/</code>'],
    ]),
]

WORKFLOWS = [
    ['脚本编写与复核', '8 个 agent（4 个编写 + 4 个复核），1.8 小时', 'probe、四类判定与 early exit、loop 与评估、第二阶段预览',
     '修正 2 处 bug（X 题计入分母、conf 规则平局取最早层），其余经独立实现核对无误'],
    ['第二阶段深挖', '8 个 agent，1.4 小时', '共识 probe、回读门控、loop 门控、ABNE',
     '没有代码 bug；更正了 7 处表述，例如一致性门控同样会还原 decoy-O、几何 R² 只对版本特异部分成立；补了不看标签的选择流程'],
    ['L1 补空间推送', '2 个 agent，1.5 小时', 'steer.py，A、B 两条线全样本', '没有 bug；独立的 hook 实现复算出锁箱 B +0.55、A −0.29'],
]

ISSUES = [
    ['19:17', 'B 线只给 user 指令时格式遵从率只有 2%', '改成 system 指令加 user（chat0），遵从率 100%；没有换模型，保住同结构对比'],
    ['19:18', '"遵从率 ≥95% 的最小 k"会选 0-shot', '改成统一 1-shot（每个来源遵从率都是 100%）'],
    ['19:31', 'K 连续层规则让 716 道最终 4/4 正确的题落进 U', '连续段在末层截断'],
    ['19:42', 'probe 的子集测试因 GPU 被提取任务占满而 OOM', 'agent 等提取结束后重跑'],
    ['20:20', '早段安慰剂 p4_11 会破坏答案格式', '约束性安慰剂改用 p28_33；p4_11 只报告 X 比例'],
    ['21:10', '最后一层用 probe 判断"读对"，与真实输出冲突（U_final4）', '改成 last_layer=output 并全部重跑；loop 的方向结论随之修正'],
    ['21:14', 'smoke.py 的层号写死，超出 28 层模型的范围', '只保留小于 L 的层，重排 Qwen2.5 流水线'],
    ['21:33', '磁盘只剩约 100GB', '删掉已评估完的 loop 轨迹隐状态，约 150GB；Llama 部分因此无法完整重跑复核'],
    ['22:50', 'steer.py 默认显存上限 11.5GB 装不下 7B 模型', '上限提到 30GB 后重跑'],
]


def build():
    css = """
:root{
  /* Layout: one column run sheet; timeline full width; tables scroll inside their own box. Palette shared with the round-1 report. */
  --paper:#f3f5f2; --ink:#1c2420; --muted:#5b6a63; --rule:#d5ddd8; --panel:#e8ede9;
  --teal:#2f6b5b; --amber:#b9772a; --rose:#a14c4c; --slate:#64748b; --sand:#8c7a52;
  --f-display:"Noto Serif SC","Songti SC",serif; --f-body:"Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --paper:#121816; --ink:#e3e9e5; --muted:#9aaaa2; --rule:#2b3632; --panel:#1a2320;
  --teal:#6fb39c; --amber:#e0a54b; --rose:#d98a8a; --slate:#9aa6b8; --sand:#c4b07e; color-scheme:dark}}
:root[data-theme="dark"]{
  --paper:#121816; --ink:#e3e9e5; --muted:#9aaaa2; --rule:#2b3632; --panel:#1a2320;
  --teal:#6fb39c; --amber:#e0a54b; --rose:#d98a8a; --slate:#9aa6b8; --sand:#c4b07e; color-scheme:dark}
body{background:var(--paper);color:var(--ink);font-family:var(--f-body);font-size:15px;line-height:1.7}
.wrap{max-width:1040px;margin:0 auto;padding-block:28px 60px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
header{display:grid;gap:6px;border-bottom:2px solid var(--ink);padding-bottom:14px;margin-bottom:18px}
.eyebrow{font-family:var(--f-mono);font-size:12px;letter-spacing:.08em;color:var(--muted);text-transform:uppercase}
h1{font-family:var(--f-display);font-size:clamp(26px,4vw,36px);line-height:1.25;margin:0;text-wrap:balance}
h2{font-family:var(--f-display);font-size:21px;margin:34px 0 8px;text-wrap:balance}
p{max-width:76ch;margin:0 0 10px}
.lede{color:var(--muted)}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0 6px}
.fact{border-top:1px solid var(--rule);padding-top:6px;min-width:0}
.fact b{display:block;font-family:var(--f-mono);font-size:20px;font-variant-numeric:tabular-nums}
.fact span{font-size:13px;color:var(--muted)}
.tbl{overflow-x:auto;margin:8px 0}
table{border-collapse:collapse;font-size:13.5px;width:100%;min-width:760px}
th,td{border-bottom:1px solid var(--rule);padding:7px 8px;text-align:left;vertical-align:top}
th{font-size:12.5px;color:var(--muted);font-weight:600}
td:first-child{font-weight:600}
table.runs{min-width:640px} table.issues{min-width:620px} table.issues td:first-child{font-family:var(--f-mono);font-weight:400;white-space:nowrap;color:var(--muted)}
code{font-family:var(--f-mono);font-size:12px;background:var(--panel);padding:1px 4px;border-radius:2px;word-break:break-all}
.gwrap{overflow-x:auto;margin:6px 0 4px}
.gantt{width:100%;min-width:760px;height:auto;display:block}
.gantt .tick{stroke:var(--rule);stroke-width:1}
.gantt .tl{font-size:11px;fill:var(--muted);font-family:var(--f-mono)}
.gantt .ln{font-size:12px;fill:var(--ink);font-family:var(--f-mono)}
.gantt .bl{font-size:10.5px;fill:#fff}
.k-prep{fill:var(--slate)}.k-extract{fill:var(--sand)}.k-probe{fill:var(--amber)}.k-loop{fill:var(--teal)}.k-steer{fill:var(--rose)}.k-fail{fill:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--muted)}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-1px;margin-right:5px}
a{color:var(--teal)}
footer{margin-top:36px;border-top:1px solid var(--rule);padding-top:12px;font-size:12.5px;color:var(--muted)}
"""
    P_ = []
    P_.append('<title>Brewing 第一轮实验清单</title><meta name="claude:viewport-fit-sides" content="cover">')
    P_.append('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Noto+Sans+SC:wght@400;600&family=Noto+Serif+SC:wght@600;700&display=swap">')
    P_.append(f'<style>{css}</style><div class="wrap">')
    P_.append('<header><div class="eyebrow">Brewing · round 1 · 2026-10-16 19:05–23:41</div><h1>Brewing 第一轮实验清单</h1>'
              '<p class="lede">上一轮做过的每一项实验：在什么模型上、跑了多大规模、用了多久、得到什么、产物放在哪里。结论和图表见'
              '<a href="https://claude.ai/artifact/WvVU152bkVa2QyR7oQYzex">第一轮报告</a>，逐步日志见 <code>exp/depth_intervention/RESULTS.md</code>。</p></header>')
    P_.append('<div class="facts">'
              '<div class="fact"><b>4 小时 36 分</b><span>从开始执行（19:05）到最后一次推送（23:41），2 张 4090 48GB 几乎一直满载</span></div>'
              '<div class="fact"><b>6 + 2</b><span>模型或格式配置，加 2 个 O 扩样集</span></div>'
              '<div class="fact"><b>22,636</b><span>核心集每个运行的 prompt 数；扩样集每个运行 45,456 条</span></div>'
              '<div class="fact"><b>11</b><span>次 loop 运行：mode last 7 次（含 2 次早窗口），mode all 4 次，共 80 个配置（含每次的 none 参照）</span></div>'
              '<div class="fact"><b>3 / 18</b><span>subagent 工作流和 agent 数；每项分析都有一个独立的复核 agent</span></div>'
              '<div class="fact"><b>248 GB</b><span>保留的产物；另有约 150GB 轨迹隐状态在评估后删除</span></div></div>')
    runs = [
        ['A', 'Qwen3-4B-Base', '1-shot 补全（base_fewshot）', '核心集', '22,636', '<code>qwen3-4b-base__base_fewshot</code>'],
        ['B', 'Qwen3-4B 后训练版', 'chat0（system 指令 + user，非思考）', '核心集', '22,636', '<code>qwen3-4b__chat0</code>'],
        ['格式对照', 'Qwen3-4B 后训练版', '1-shot 补全（与 A 同一份 prompt）', '核心集', '22,636', '<code>qwen3-4b__base_fewshot</code>'],
        ['Llama', 'Llama-3.1-8B（base）', '1-shot 补全', '核心集', '22,636', '<code>llama3.1-8b__base_fewshot</code>'],
        ['Q2.5', 'Qwen2.5-7B（base）', '1-shot 补全', '核心集', '22,636', '<code>qwen2.5-7b__base_fewshot</code>'],
        ['Q2.5-I', 'Qwen2.5-7B-Instruct', 'chat0', '核心集', '22,636', '<code>qwen2.5-7b-instruct__chat0</code>'],
        ['A-ext', 'Qwen3-4B-Base', '1-shot 补全', 'O 扩样集', '45,456', '<code>qwen3-4b-base__base_fewshot__ext</code>'],
        ['B-ext', 'Qwen3-4B 后训练版', 'chat0', 'O 扩样集', '45,456', '<code>qwen3-4b__chat0__ext</code>']]
    P_.append('<h2>8 个运行</h2><p>页面里的"运行"指一个"模型 × 输入格式 × 数据集"组合：整套 prompt 前向一遍，提取各层隐状态并同时算 CSD，'
              '之后的 probe、四类判定、early exit 也都按运行分别做。产物在 <code>results/</code> 下以运行名为目录。'
              '格式对照用来把"后训练"和"prompt 格式"两个因素分开：它和 A 用同一份 1-shot prompt，和 B 用同一个模型。</p>')
    P_.append(table(['简称', '模型', '输入格式', '数据集', '条数', '目录（results/ 下）'], runs, 'runs'))
    P_.append('<h2>时间线</h2><div class="legend">'
              '<span><i class="sw k-prep"></i>冒烟与格式</span><span><i class="sw k-extract"></i>提取与流水线</span><span><i class="sw k-probe"></i>probe 训练</span>'
              '<span><i class="sw k-loop"></i>depth loop</span><span><i class="sw k-steer"></i>推送与因果实验</span><span><i class="sw k-fail"></i>失败后重排</span></div>')
    P_.append(f'<div class="gwrap">{gantt()}</div>')
    P_.append('<p class="lede">GPU0 和 GPU1 两行来自队列记录和日志时间戳；subagent 一行是 agent 自己启动的 GPU 任务，时间是近似值。CPU 上的分析（四类判定、loop 评估、第二阶段统计）没有画出来。</p>')
    for key, title, rows in SECTIONS:
        P_.append(f'<h2>{E(title)}</h2>')
        P_.append(table(['实验', '对象', '规模', '用时', '结果要点', '产物'], rows))
    P_.append('<h2>subagent 工作流</h2>')
    P_.append(table(['工作流', '规模', '做了什么', '复核结果'], WORKFLOWS))
    P_.append('<h2>问题与处理</h2>')
    P_.append(table(['时间', '问题', '处理'], ISSUES, 'issues'))
    P_.append('<h2>代码与产物位置</h2>'
              '<p>脚本在 <code>exp/depth_intervention/scripts/</code>：common.py（共享核心）、build_bench(_ext).py、smoke.py、extract.py、train_probe.py、classify.py、early_exit.py、loop.py、loop_eval.py、'
              'stage2_features.py、stage2_analysis.py、consensus_probe.py、stage2_retroread.py、stage2_loopgate.py、stage2_abne.py、steer.py、gen_report.py；数据约定在 INTERFACES.md。'
              'GPU 任务都通过 <code>scripts/gpuq.sh</code> 队列运行，记录在 <code>queue/gpu0.history</code> 和 <code>queue/gpu1.history</code>，日志在 <code>logs/</code>，共 235 个文件。</p>'
              '<p>每个运行的产物在 <code>results/{model}__{fmt}[__ext]/</code> 下，按 extract、probe、consensus_probe、classify、early_exit、loop_allq、loop、stage2、retroread、loopgate、abne、steer 分目录。</p>')
    P_.append('<footer>由 <code>scripts/gen_inventory.py</code> 生成。时间取自机器时钟（2026-10-16）。</footer></div>')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(P_))
    print('wrote', OUT)


if __name__ == '__main__':
    build()
