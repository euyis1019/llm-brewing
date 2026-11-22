# 深度维度干预实验：执行计划 v2（2026-10-16）

> **用途**：写给上下文压缩后的我（或接手的 agent）。执行时只依赖这份文档和它引用的文件。
> **时间预算**：一轮完整迭代 8–12 小时，先不求鲁棒，先把整条链跑通：构造 benchmark → 提取隐状态 → 训练 probe → CSD → 四类判定 → 干预 → 对照 → 记录。
> **工作目录**：`/nvme0/if/research/brewing/exp/depth_intervention/`
> - `scripts/`：脚本
> - `data/`：数据
> - `results/`：结果
> - `logs/`：运行日志
> - `RESULTS.md`：迭代日志，每完成一步追加一节
>
> 这个目录不是 git 仓库。
>
> **v2 相对 v1 的改动**（按用户 2026-10-16 的反馈）：
> - 主数据改为**我们新构造的 benchmark**，在上面完整跑一遍 probe 训练和 CSD。7 月的 CF 旧缓存只用来给脚本做冒烟测试。
> - **删掉 Ouro。** 我们的产物是热插拔工具：作用在普通 LLM 上，让它可以早停、可以做 depth loop。所以干预和验收都在普通 LLM 上完成。

---

## 1. 背景与共识

**项目背景**
- Brewing 研究 LLM 逐层处理时答案的生命周期。第一篇已公开：arXiv 2606.17648，CUE-Bench，16 个模型。
- 两个诊断：
  - probe（可得性）：第 l 层最后一个 token 的隐状态里，答案是否线性可读；
  - CSD / Patchscope（就绪性）：把第 l 层状态移植到只含 "Answer:\n" 的空 prompt 里，跑完后面的层，看能否解码出答案。
- 第一篇的四类判定（`/nvme0/if/llm-brewing/source_code/brewing/diagnostics/{metrics,outcome}.py`）：
  - FPCL：probe 第一次读对的层；
  - FJC：probe 和 CSD 在同一层都读对的第一层；
  - 有 FJC 时：最终输出对是 **R**（Resolved），错是 **O**（Overprocessed）；
  - 没有 FJC 时：末尾四分之一层（不低于 3L/4）的 CSD 逐类平均置信度最大值不低于 0.5，是 **M**（Misresolved），否则是 **U**（Unresolved）；
  - FPCL 不存在：NO_BREWING，不进入四类。

**用户目标（2026-10-16）**
1. 在更干净的模型和更多真实 benchmark 上，完整复现层深维度上的现象：生成 critical token（直接决定整题对错的 token；本轮就是 4 选题的答案字母）时，出现 overprocessed 和 unresolved。
2. **第一阶段验收**：一套 evaluation framework，满足两条：
   - 对 O 施加 early exit，使 **O → R**；
   - 对 U 施加 depth loop（layers in the loop），使 **U → O、M、R 三者之一**。

   做到这一步就算验收完成，证明现象真实存在，也为 per-token 的推理干预提供依据。
3. **第二阶段**：基于 hidden state 定义度量，激进版是做 O 对 U 的二分类，最终得到"该加层还是该早停"的 per-token 指标。
4. **最终产物**：热插拔工具，把一个普通 LLM 变成可以早停、可以 depth loop 的模型。早停头、loop 算子和路由信号都来自这里。

**已定的 benchmark 设计**
- 4 选 A–D，单 token 作答。每道题做 4 个循环置换版本，按内容判定。
- 来源：
  - MMLU-Pro：10 选抽成 4 选；
  - C-Eval；
  - GSM-MC、MATH-MC、PythonIO：GitHub `Geralt-Targaryen/MC-Evaluation`，都是 4 选。

**先验证据（2026-10-14 复核过，来自 7 月的 CF 数据，详见 `../../discussion/两阶段重设计_证据与建议_2026-10-14.md`）**
- 内容级的 O 很少：匹配对照后净值只有 0.2–0.4%。U 占 18–28%。
- 发射层：4B 是 L24。probe 起点是题目属性。
- **P6 pilot**（Qwen2.5-Coder-7B，免训练重复中段层）：任何固定区段都没有提高准确率；"救回"的数量与"变成一个预先指定的错误答案"相当，基本是随机翻动。**loop 必须配安慰剂对照。**

**模型与执行者（已确认）**
- **A 线（base）**：Qwen3-4B-Base（稠密），占 GPU0。
- **B 线（instruct，2026-10-16 用户新增）**：本地后训练版 Qwen3-4B（`/nvme0/if/models/Qwen3-4B`），用非思考模式，占 GPU1。
  - 它和 Base 的结构、tokenizer、层数都相同（36 层，hidden 2560，共享嵌入），比较时唯一的变量是后训练。
  - 如果它的格式遵从率不达标，换成 `/nvme0/if/models/huggingface/Qwen/Qwen2.5-7B-Instruct`。
- 第二轮用 Meta-Llama-3.1-8B 复现 base 线（稠密，本地路径 `/nvme0/if/models/Meta-Llama-3.1-8B-nous`）。
- Qwen3.5-9B（混合注意力）不做。
- 执行者是我。

---

## 1.5 模型与时间估算（由用户选定，选定结果记在本节末尾）

**估算依据**
- 7 月实测速度：
  - 4B 提取隐状态每条约 0.17–0.28 秒（batch 8）；
  - 4B 在 1.6 万条上跑 CSD 约 28 分钟，9B 约 50 分钟。
- 新 benchmark 约 5.7k 题 × 4 个版本，约 2.3 万条 prompt。
- loop 的对象：全部 U 加上 R、M、O 各 200–300 题，约 1.4–2k 题 × 4 个版本。

**分步估算**

| 步骤 | Qwen3-4B-Base | Llama-3.1-8B（本地） | Qwen3-8B-Base（需下载约 16GB） |
|---|---|---|---|
| 提取隐状态 | 0.5–1.5 小时 | 1–2.5 小时 | 1–2.5 小时 |
| probe（5 折交叉拟合 + 3 个 null） | 0.3–0.5 小时 | 约 0.5 小时 | 约 0.5 小时 |
| CSD（raw 和减基线同一次跑） | 0.5–1 小时 | 1–1.5 小时 | 1–1.5 小时 |
| loop（L-all 若干配置；L-last 很便宜） | 1.5–4 小时 | 3–6 小时 | 3–6 小时 |
| 单模型 GPU 合计 | 3–7 小时 | 5.5–10 小时 | 5.5–10 小时，另加下载 |

**B 线：** Qwen3-4B 后训练版与 Base 同结构，耗时按 Qwen3-4B-Base 一列估算。chat 0-shot 的 prompt 比 few-shot 短，提取和 loop 会更快。

**公共开销：** 构造 benchmark 1–1.5 小时（CPU，需要在 GPU 开工之前完成）；分类和 early exit 约 1 小时；汇总约 1 小时；首次调试预留 1–2 小时。

**超时时的砍法（按优先级）：**
1. loop 只保留 brewing 窗口块（k=1、2）和两个安慰剂块；
2. U 抽样到 800 题；
3. 阻尼版本和其余窗口块放到下一轮。

**模型选定（用户 2026-10-16 确认）：**
- **第一轮两条线并行，各占一张卡：**
  - A 线：Qwen3-4B-Base，GPU0；
  - B 线：Qwen3-4B 后训练版，非思考模式，GPU1。
  - 每条线的 GPU 时间约 3–7 小时，整轮目标 8–10 小时。
- **第二轮**：用 Llama-3.1-8B 复现 base 线，用同一套脚本。
- Qwen3-8B-Base（同家族扩规模）作为以后的备选。

---

## 2. 关键选择

### 2.1 CSD 用哪个读出：选纯 Patchscope 做主读出，减基线的版本同一次一起算

两种读出都在同一次运行里输出：
- **主读出：raw，不减基线（即纯 Patchscope）。**
- **敏感性读出：减基线。** 基线是只跑目标 prompt "Answer:\n"、不注入时的 logits，这正是第一篇 CSD 的做法。

**选 raw 做主读出的三个理由：**
1. raw 在最终层就等于模型的真实输出。四类判定要拿"最终是否答对"来分，early exit 的效果也要和真实输出比较，这一点必须成立。
2. 7 月的 CF 数据上，减基线之后最终层的预测明显偏离真实输出：内容级 R 在 4B 上从 2507 降到 2389，9B 上从 2911 降到 2412。
3. raw 在早层带字母先验（早层几乎全输出 A），但 4 版本内容级判定能抵消位置先验。

### 2.2 probe 的形式

采用用户的设计，在隐空间等价实现：

> 4 个标签的 logit = 冻结的 lens logit（A–D 四行 unembedding 乘以 final norm 后的 h）+ W·ĥ + b

- ĥ 是做了 RMS 归一化的 h。
- 每层单独训练 W（4×d）和 b。目标是交叉熵加 L2，用全批量 L-BFGS 求解（GPU 上的 torch）。不要用 Adam 加早停。
- 这和"冻结 unembedding 后接一个可训练的词表→标签矩阵"函数类完全相同，参数量少约 60 倍，也不用物化全词表 logits。
- 同时报告两个更低的台阶作为对照：纯 lens，和 lens 加偏置。
- 另训 3 个 null probe：在来源内打乱标签，用来给出机会水平。

### 2.3 隐状态提取：用 hook 取 decoder 块的输出

- 所有层（包括最后一层）都取 **final norm 之前**的残差，这样从根上避免顶层二次归一化的问题。
- 同时存模型真实输出的 4 标签 logits（fp32）。
- 隐状态存成 fp16，只存最后一个 token。

### 2.4 四类判定：两个口径都报

- **v0（与第一篇兼容）：** 单版本，单层命中就算，直接沿用 `classify_outcome`。
- **v1（主口径，内容级稳定版）：**
  - FPCL_c：4 个版本的 probe 都指向正确内容，并且连续保持 K=3 层的第一层；
  - FJC_c：同一层上 probe 和 CSD 的 4 个版本都正确，并且连续 K 层；
  - R：有 FJC_c，且最终 4 个版本的真实输出都正确；
  - O：有 FJC_c，且最终不是 4/4 正确；
  - M：没有 FJC_c，且最终 4 个版本一致指向同一个错误内容；
  - U：其余；
  - 敏感性另报 K 取 1 和 2。
- **内容映射：** 第 v 个版本里第 i 个字母显示的是原题第 (i+v) mod 4 个选项。
- **对照：**
  - null probe 下的 FPCL/FJC 误报率；
  - 匹配 decoy：把正确内容换成某个错误内容，套用同一规则得到的"假 O"。

### 2.5 数据划分：交叉拟合

- 题目少、O 也少，所以不单独留 test。按题组分 5 折：在 4 折上训练 probe，预测剩下的 1 折。这样每道题都有折外读出，所有题都能用来判类、做干预。
- λ 和 early exit 规则的参数，只在被评测那一折以外的数据上定。
- loop 的配置事先固定，不调参。

---

## 3. 干预（作用在普通 LLM 上，也就是热插拔工具的雏形）

### 3.1 early exit（针对 O）

在按规则选出的层直接输出答案。退出层**不能用该样本的金标签来挑**。

两种读出头：
- **E-head**（主）：用训练好的 probe 当退出头。在第 l 层停下，用 probe 解码。这是真正的早停。
- **E-csd**：从第 l 层起丢掉上下文，跑完后面的层（就是 CSD 在第 l 层的读出）。

三种选层规则：
- **fix**：一个固定层，在训练折上按群体准确率选；
- **stab**：从窗口起点开始，top-1 连续 k 层不变、且置信度不低于 τ 就退出；
- **conf**：取窗口内置信度最高的那一层。

窗口起点在 4B 上取 L16，Llama 上取 L14，大约是总层数的 45%。

报告三项：
- O→R 的比例；
- 同一规则作用在全体样本上的净准确率变化，以及把 R 改坏的比例；
- 匹配 decoy 的对照。

这一步只需要 CPU，因为 probe 和 CSD 的逐层读出都已经存好了。

### 3.2 depth loop（针对 U）

把层序列改成：0…b，再 (a…b) 重复 k 次，再 b+1…末层。同一组权重，不训练。有两种作用范围：
- **L-all**：所有位置都走 loop 后的层序列。实现方式是替换 `model.model.layers` 为 ModuleList，同时改 `config.num_hidden_layers`，参考 P6 的 `layer_repeat.py`。
- **L-last**：只在最后一个位置（critical token）重复，上下文的 KV 不变，参考 P6 的 `last_position_repeat.py`。这个版本更贴近以后逐 token 的部署方式。

阻尼版本：重复的块按 `h + β·(layer(h) − h)` 包装，β 取 0.5。

**配置事先固定**（4B 共 36 层；Llama 共 32 层，块位置按比例缩放）：
- brewing 窗口块：[16,23]、[18,25]、[20,27]，k 取 1 和 2；
- 阻尼版本：[18,25]，k=2；
- 安慰剂块：[4,11]、[28,33]，k=1。

**对象：** 全部 U 题（4 个版本），加上 R、M、O 各抽 300 题（用来测伤害和特异性）。

**干预后的状态：**
- 最终真实输出决定 R、M 或 U；
- 扩展序列上的 FJC_c 决定是否变成了 O，读法如下：
  - probe：用产生该位置的原始层的 probe；
  - CSD：把状态注入后，跑完原始的剩余层。

**对照：**
- 安慰剂块的状态改变率；
- 方向：U 转向正确内容（变成 R，或形成正确内容的 O）与转向错误内容（变成 M，或形成某个错误内容）的比例对比；
- 把 R 改坏的比例。

---

## 4. 执行时间线（第一轮：A 线 base 在 GPU0，B 线 instruct 在 GPU1）

所有脚本都接受 `--model`（模型配置）和 `--fmt`（prompt 格式：`base_fewshot` / `chat0`），两条线共用一套代码。

| 时间 | CPU | GPU0：A 线 Qwen3-4B-Base | GPU1：B 线 Qwen3-4B（非思考） |
|---|---|---|---|
| 0:00–1:30 | `build_bench.py`：clone MC-Evaluation，构造约 5.7k 题 × 4 个版本（见 4.1） | 冒烟测试：用 CF eval 的 200 题跑 `extract.py` 和 `csd.py`，顶层读出与真实输出的一致率应约为 97%；用 `loop.py` 的不重复基线配置复现正常前向；对新 benchmark 做 few-shot 的 k 选择（见 4.1） | 冒烟测试：chat 模板和非思考模式下，助手回复的第一个 token 是不是字母（格式遵从率）；同时确认 CSD 的目标 prompt 用助手前缀（见 4.2） |
| 1:30–3:00 | — | `extract.py` | `extract.py` |
| 3:00–4:00 | `train_probe.py`（5 折交叉拟合、λ 网格、3 个 null；用 GPU 上的 torch L-BFGS，插在 CSD 的空隙里跑） | `csd.py`（raw 和减基线两种） | `csd.py` |
| 4:00–5:00 | `classify.py`（v0 和 v1，加对照）；`early_exit.py`。两条线都跑 | — | — |
| 5:00–7:30 | `loop_eval.py` 汇总 | `loop.py`：L-all 跑窗口块 k=1、2 和两个安慰剂块，L-last 跑全部配置 | 同左 |
| 7:30–8:30 | 第二阶段预览（见 4.3）；base 对 instruct 的对比表；汇总 `RESULTS.md`，向用户汇报 | 空余时间补跑：阻尼版本、其余窗口块、K 敏感性 | 空余时间补跑：instruct 模型跑 base 的 few-shot prompt，作为格式对照 |

缓冲时间到 10–12 小时。超时就按 1.5 节的优先级砍配置；B 线可以先只跑 early exit 和 L-last。第二轮换成 Llama-3.1-8B，脚本不变，改模型配置即可：A–D 的 token id 要实测，norm 的 eps 是 1e-5，lm_head 独立。

### 4.1 benchmark 构造细节

- **题量：**
  - GSM-MC：1319 题（test 全量）；
  - MATH-MC：抽 1000 题；
  - PythonIO：抽 1000 题；
  - MMLU-Pro：抽 1000 题，保留正确项，从干扰项里用固定种子抽 3 个；
  - C-Eval：1346 题（本地 normalized 全量）。
- **模板：** `Question:\n…\nChoices:\nA. …\nB. …\nC. …\nD. …\nAnswer:\n`。
- **few-shot（A 线 `base_fewshot` 用；B 线默认不用）：**
  - 示例从同来源、不在评测集里的题中抽；
  - **示例的答案字母要均衡**，不能都是同一个字母；
  - **每个版本的示例同步做同样的第 v 个循环置换**，避免 4 个版本共享同一个字母先验（复核发现，共享示例时早层按位置锁定同一字母的题占 47–89%）。
  - **k 用冒烟测试来定：** 每个来源抽 200 题，比较 0-shot、1-shot、3-shot 下真实输出落在 A–D 的比例（格式遵从率）。取遵从率不低于 95% 的最小 k，可以按来源分别取，长题（PythonIO、MATH）倾向小 k。
  - 格式遵从率作为指标上报。真实输出不在 A–D 的样本单独统计，不进入四类判定。
- **B 线格式 `chat0`：**
  - 用 tokenizer 的 chat 模板，`enable_thinking=False`，0-shot；
  - user 消息是同样的题目和选项，末尾加一句"只回答选项字母（A/B/C/D）"；
  - critical token 是助手回复的第一个 token。
- **元数据：** 写清来源、题组、版本号、内容映射。
- **下载数据的处理：** 放在 `data/raw/` 下单独的目录，读取用 `python -I`。
- **本地已有的数据：**
  - MMLU-Pro 和 C-Eval：`/nvme0/if/llm-brewing/source_code/bench/datasets/benchmark/normalized/{mmlu_pro,ceval}/`
  - MC-Evaluation：`https://github.com/Geralt-Targaryen/MC-Evaluation`，已确认能 clone。

### 4.2 B 线（instruct）的具体做法与预期问题

**模型**
- 主选：Qwen3-4B（后训练版，`/nvme0/if/models/Qwen3-4B`）。
- 备选：Qwen2.5-7B-Instruct。如果主选在冒烟测试里格式遵从率低于 90%，就换成备选。

**格式**
- 用 `chat0`（见 4.1）。critical token 是助手回复的第一个 token。
- 冒烟测试要确认三件事：
  1. 这个位置上 A–D 是不是单 token；
  2. 是否带前导空格；
  3. 实测的 token id。

**CSD 的目标 prompt**
- 改成 chat 格式的空壳：system 和 user 都留空（或只保留"只回答选项字母"这句指令），再加上助手前缀，包括非思考模式下模板插入的空 `<think></think>` 块。
- 在最后一个位置注入状态。
- 减基线时，基线就是这个空壳本身的 logits。

**和 A 线完全相同的部分**
- probe 的形式；
- 内容映射、四类判定（v0 和 v1）；
- 干预、对照和验收口径。

**预期问题与处理**
- **置信度更尖锐**（instruct 模型通常过度自信）：v0 里按 0.5 阈值切 M 和 U 会偏移，主口径用 v1（4 版本一致才算 M），v1 不依赖阈值。
- **字母先验更强**：可能出现"总选同一个字母"的位置偏好，由 4 版本内容级判定处理。另外报告每个字母的输出分布。
- **lens 可读性不同**：后训练可能改变中间层和 unembedding 的对齐。probe 带可训练的修正项，不受影响；纯 lens 那一级的读数要和 base 对比着报告。
- **格式对照**：时间够的话，让 B 线模型也跑一遍 `base_fewshot`，把"后训练的影响"和"prompt 格式的影响"分开。

### 4.3 第二阶段预览（CPU）

- **特征：** 只用不需要标签的逐层特征：probe 和 CSD 的置信度、margin、熵，相邻层 KL，top-1 连续层数，隐状态相邻层余弦，范数。
- **任务：**
  - O 对 U、U 对 M 的逻辑回归，报告 AUROC 和相对单个置信度的增益；
  - 在"判定层"上，预测"再加层会不会让状态改变"（标签来自 3.2 的 loop 结果）。

---

## 5. 资源与路径

- **Python：** `/nvme0/if/anaconda3/envs/qwen3/bin/python`（torch 2.11、transformers 5.13、sklearn 1.9）。
- **机器：** GPU 2 张 4090 48GB；内存 251GB；`/nvme0` 剩约 374GB。
- **Qwen3-4B-Base：** `/nvme0/if/models/Qwen3-4B-Base`
  - 36 层，hidden 2560；
  - 词嵌入与 lm_head 共享；
  - RMSNorm 是 `x / rms * w`，eps 1e-6；
  - "Answer:\n" 之后的 A–D token id 是 32–35。
- **Llama-3.1-8B：** `/nvme0/if/models/Meta-Llama-3.1-8B-nous`
  - 32 层，hidden 4096；
  - lm_head 独立；
  - RMSNorm 的 eps 是 1e-5；
  - token id 要实测。
- **参考代码：**
  - 第一篇的 CSD 和 patchscope：`/nvme0/if/llm-brewing/source_code/brewing/methods/csd.py`、`/nvme0/if/research/brewing/history/llm-brewing/brewing/nnsight_ops.py` 里的 `patchscope_lens`；
  - P6 层重复：`/nvme0/if/research/brewing/Proposal/Claude Code/_pilot/P6_layer_repeat/{layer_repeat.py,last_position_repeat.py}`。
- **冒烟测试用的 7 月 CF 数据：** 先令 `R=/nvme0/if/llm-brewing/source_code/brewing_output/general_prefill_runs/56cf303a0bee12f2__5ffa284ecd741a77`。
  - 样本：`$R/Qwen3-4B-Base/datasets/choicepermutationcf/eval/pooled/seed42/samples.json`；
  - 已知数字：内容级 R 为 2507 / 4044，顶层 lens 与真实输出一致率约 97%。

---

## 6. 必读陷阱

1. **顶层：** HF 的 `hidden_states[-1]` 是 final norm 之后的状态。所以本轮用 hook 取块输出；如果用到旧缓存，顶层不能再归一化。
2. **精度：** 读出一律用 fp32，存 top-2 margin，近似平局记为"不确定"。bf16 的精确并列会决定 O 的成员。
3. **字母先验：** 早层读出带字母先验，判定一律用 4 版本内容级。起点用"从该层起保持到最终层"或"连续 K 层"，不要用单层首次命中。
4. **O 很少：** 必须报告区间。要扩样，就多加计算型题（GSM-MC 和 MATH-MC 的 O 可能更多）。
5. **loop 很可能只是随机翻动：** 没有安慰剂对照和方向对照的"状态改变"不算达标。
6. **B 线（instruct）：**
   - 必须用非思考模式，否则助手回复的第一个 token 是 `<think>`；
   - CSD 的目标 prompt 要换成 chat 空壳，不要沿用 "Answer:\n"；
   - A–D 的 token id 和有没有前导空格要实测，不要照搬 base 的 32–35。
7. **工程规则：**
   - GPU 任务用 `nohup` 起，日志写 `logs/NAME.log`，结束时写 `logs/NAME.done`；
   - 不要用会匹配到自身命令的 `pkill -f`；
   - 不要原地改写正在运行的脚本；
   - 两张卡尽量不要空闲。

---

## 7. 第一轮验收口径

- **A1｜early exit：** 某个不用金标签的规则，用 E-head 满足：
  - O→R 的比例显著高于匹配 decoy，区间不包含 0；
  - 作用在全体样本上时，单版本准确率下降不超过 1 个百分点；超过就单独报告。
- **A2｜depth loop：** 某个 brewing 窗口块的配置满足：
  - U 的状态改变率（变成 O、M、R 之一）高于安慰剂块，区间不包含 0；
  - 朝正确方向的转移多于朝错误方向的。

  U→M 按用户的定义计入，但要和方向对照一起报告。
- 每个模型分别下结论。主表用 v1 口径，v0 附在后面，方便和第一篇比较。

---

## 8. 记录与汇报

- 每个脚本写 `results/*.json`，记下配置、样本数、耗时、脚本 md5。
- 每完成一步，在 `RESULTS.md` 追加一节：做了什么、关键数字、和验收的关系、问题。
- 异常先记录再处理，不要静默跳过。
- 汇报时对话里不写公式，用白话。完整内容以 `RESULTS.md` 为准。
