# Paper 2 — 跨 domain probing:把通用 bench 接进 brewing

> 状态:设计定稿,待实施。工作分支 `brewing`。

## 背景与目标

Paper 1(CUE-Bench,合成任务、单 digit 答案 0-9)已证明 brewing 现象:答案往往在 hidden state
里**先线性可读(available)**,之后才**被模型自身解码出来(ready)**。

Paper 2 要把这个现象**推广到真实公开 benchmark**,证明 "available before ready" 不是合成任务的
artifact,而是普遍结构。数据已硬拷进 `bench/datasets/benchmark/normalized/`,共 9 个 benchmark。

三步目标:

1. **统一 benchmark** —— 把公开 benchmark 转成 brewing 框架能吃的 `Sample`。
2. **先证明 "single next token is enough"(critical token)** —— first-token 命中即算。
3. **收集 probing 训练/验证集**,一次训练出一个跨 domain 可用的 probe。

## 核心设计决定

- **probe 的 label = critical token 的 vocab id**(不是每题 answer_space 的类索引)。
  MC / gen 两族同一口径,与 CSD 的 logit-over-answer-tokens 对齐,brewing gap 可跨 domain 直接比较。
  probe 输出层裁剪成"全域出现过的 answer-token vocab-id 子集"(几百~几千类),而非全 vocab(~15 万)。

- **critical-token 判据 = first-token 命中即算**。greedy 解码 prompt 末位第一个 token,若等于 gold
  target 的首 token,则该样本 "single next token is enough"。论证逻辑:
  1. first token 命中 → 模型极难再生成错误答案;
  2. first token 不对 → 模型极难再生成正确答案。
  因此不需要跑完整生成,与 probe / CSD 的 last-position 口径天然一致。

- **首轮只做 4 个直答型 benchmark**:`mmlu / mmlu_redux / cmmlu / ceval`。

## 为什么首轮只做这 4 个

核对 9 个 benchmark 的 protocol 后,它们按**生成范式**(不只是答案格式)分成三类:

| 类型 | benchmark | 输出范式 | 首轮 |
|---|---|---|---|
| 直答型 MC | mmlu, mmlu_redux, cmmlu, ceval | prompt 以 `Answer: ` 结尾,**第一个生成 token 就是答案字母** | ✅ |
| CoT | gsm8k, math500, bbh, mmlu_pro | 先推理数百 token,再在 `####` / `\boxed{}` 后给答案 | ⏳ 第二阶段 |
| 多 token 精确匹配 | cruxeval | 输出 `[(4,1),...]` 等多 token | ⏳ 第二阶段 |

brewing / probing 依赖 **prompt 末位 hidden state → 第一个 next token**。直答型 MC 的答案就在这个
位置,天然符合口径;CoT / cruxeval 的答案在末位之后,末位 first-token 不是答案,需要另一套
"定位 answer-position" 的设计,留待第二阶段。

## 实施步骤(概要)

### Step 1 — 新 benchmark adapter:`brewing/benchmarks/general_bench/`
仿 `cue_bench/` 结构。`spec.py` 定义 `GENERAL_BENCH`(subsets = 4 个直答型);`loader.py` 读
`bench/datasets/benchmark/normalized/{subset}/*.jsonl`(排除 `__NNNrows` 切片)+ `benchmark_meta.json`,
**复刻** bench 的 MC prompt 口径(`question\nA. …\nB. …\nAnswer: ` + few-shot 前缀)转成 `Sample`:
`prompt` 以 `Answer: ` 结尾,`answer` = gold 字母,`subset` = bench_id,`difficulty.subject` 保留。
复刻而非 import `bench/src`(bench 是独立 harness,避免运行时耦合)。
`__init__.py` 里 `register_benchmark`,`brewing/benchmarks/__init__.py` 追加 import。

### Step 2 — pipeline 解耦硬编码 cue_bench
`pipelines/base.py::resolve_dataset` 的 `data_dir` 分支现在写死 `cue_bench.load_generated_dataset`。
给 `BenchmarkSpec` 加可选 `loader`,`general_bench.spec` 挂上 `load_general_dataset`,`resolve_dataset`
优先用 `benchmark.loader`。`RunConfig._BENCHMARK_PATH_MAP` 加 `"general-bench": "generalbench"`。

### Step 3 — critical-token 检验:`scripts/critical_token_check.py`
仿 `scripts/eval_accuracy.py`(HF 加载、left-padding、`max_new_tokens=1`)。每 sample:末位解码首 token
`pred_first`;gold 首 token = `tokenizer.encode(" "+answer, add_special_tokens=False)[0]`;命中 =
`pred_first == gold_first_id`。输出:每 benchmark 的 **sufficient 比例**(论文证据)+ 每样本
`{sample_id, gold_first_id, pred_first_id, hit}`(供 Step 4 筛样本)。GPU 步骤走 Hope job。

### Step 4 — 收集 train/eval split + 改造 probe 成 vocab-id 口径
- 4 benchmark 合并后按 sample 分层切 train/eval(如 8:2),**只保留 Step 3 命中的样本**
  (通不过 critical 检验的样本无法定义 clean label)。
- 改 `methods/linear_probing.py`:label 从 "answer_space 类索引" 改为 **critical token 的 vocab id**;
  `answer_space` 语义变为 "全域 answer-token vocab-id 子集"(在 `train()` 里由训练集 gold 首 token id
  并集构建,写进 `artifact.metadata`);probe 输出维度 = 子集大小;per-layer `nn.Linear` 结构不变。
  需保证 CUE-Bench 旧 digit 路径行为等价、不回归。
- 跑 `cache_only` 提 4 benchmark hidden state(Hope job),`LinearProbing.train()` 出**一个跨 domain probe**。

## 范围与风险

- 本轮**不动 CoT / cruxeval**;**不改 CSD**(CSD 已是 logit-over-answer-tokens,与 vocab-id 口径兼容,
  answer_space 传字母即可)。
- 主要风险:probe label 从"固定 10 类"改成"vocab-id 子集"动到 `linear_probing.py` 核心,须保证
  CUE-Bench digit 路径不回归。
- GPU 步骤(Step 3 检验、Step 4 cache_only + train)走 Hope job,遵守 vcore≤48 等硬约束。
