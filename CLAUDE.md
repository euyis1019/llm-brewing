# Brewing — Layer-wise Mechanistic Interpretability Framework

## Purpose

Brewing 追踪 LLM 逐层处理代码推理任务时，答案信息从"存在"到"可用"的内部生命周期。

两个互补的诊断函数：

- **Linear Probing (Φ_P)** — information availability：答案是否已在 hidden state 中线性可读
- **CSD (Φ_C)** — information readiness：模型自身能否从该 hidden state 解码出答案

两者的时间差揭示 **brewing-to-resolution** 结构，分化为四类 outcome：
Resolved（成功）/ Overprocessed（曾对后被破坏）/ Misresolved（自信地错）/ Unresolved（没算完）

## 当前目标 vs 长期愿景

**当前**：为 COLM 2026 论文跑完全部实验（`COLM_REQUIREMENTS.md`）。
生产 config 在 `brewing/config/colm/`，每个模型一个 YAML。

```
6 tasks × 9 models × {Probing, CSD} → 108 组 MethodResult → 54 组 DiagnosticResult
```

**长期**：Brewing 要成为一个通用的 layer-wise 可解释性编排框架——不绑定 CUE 论文的特定实验。框架的 7 插槽热插拔架构（数据集适配器、Cache-only 方法、Intervention 方法、三级诊断规则、输出后端）设计为正交可组合。详见 `docs/rfc/RFCandBackground.md`（生态定位、插槽架构、远期 Code Agent Native 愿景）。

**开发时如何权衡**：

- 当前阶段优先保障论文实验能跑通——如果泛化设计和论文需求冲突，先服务论文
- 但接口边界（BenchmarkSpec、AnalysisMethod、MethodResult）要保持通用，不写死 CUE 特有的假设
- 新增功能时：如果只有 CUE 用，实现具体逻辑即可；如果未来其他 benchmark/方法也会需要，则实现为插槽抽象

## 环境

使用 conda 环境 `cue` 运行所有代码：`conda activate cue`。

## Pipeline

```
S0  数据集 resolve/build
S1  Hidden state cache 提取（nnsight / HF 双后端）
S2  方法运行（Probing / CSD）→ MethodResult 落盘
─── pipeline 边界 ───
S3  诊断（独立后处理）：FPCL / FJC / Outcome 分类
```

- S0-S2 由 `Orchestrator` 驱动，S3 完全解耦（`run_diagnostics_from_disk`）
- Probing training 已外部化：主 pipeline 只做 eval_only，训练通过 `LinearProbing.fit()` 单独执行
- CLI：`python -m brewing --config config.yaml [--verbose]`

## 代码布局

```
brewing/
├── orchestrator.py      # S0-S2 调度
├── cli.py               # YAML-config-only CLI
├── resources.py         # 资源管理（resolve-or-build）
├── cache_builder.py     # Hidden state 提取
├── nnsight_ops.py       # NNsight tracing/intervention 封装
├── schema/              # 纯数据结构，无运行逻辑
├── methods/             # linear_probing.py, csd.py
├── diagnostics/         # metrics.py, outcome.py（S3）
└── benchmarks/cue_bench/
    ├── spec.py          # 6 subset 定义
    ├── adapter.py       # datagen → Sample 转换
    ├── builder.py       # 数据集构建
    ├── fixtures.py      # 测试用小样本
    └── datagen/         # 6 个任务的数据生成器
```

## 6 个任务

所有任务统一为 27 配置 × 150 samples/config = 4050 样本。答案均为单 digit (0-9)。


| 任务                            | 类别           | 维度                                   |
| ----------------------------- | ------------ | ------------------------------------ |
| value_tracking (pure_copying) | Data Flow    | mechanism × depth × distractors      |
| computing                     | Data Flow    | structure × steps × operators        |
| conditional                   | Control Flow | branch_type × depth × condition_type |
| function_call                 | Control Flow | mechanism × depth × distractors      |
| loop                          | Data+Control | body_type × iterations × init_offset |
| loop_unrolled                 | Data+Control | body_type × iterations × init_offset |


关键对照：value_tracking vs function_call（值传递 ± 函数内计算），loop vs loop_unrolled（循环 ± 循环语法）。

## 重点强调：

- 用户叫 Eric， 你进行 git 操作时优先使用 gh，然后所有 commit 和 push 的操作的用户不能带上你自己，只能有他。
- 所有模型的实际权重放在：/home/gyf/CUE/models下，需要运行的实验的模型都一定下载好了，如果没有请立刻抛出

- 目前没有必要使用 Trace(nnsight的构建方式)。现在的进展是：如何使用这个半成品框架，去支持我稍微系统性、有组织地将剩余所有实验给完成