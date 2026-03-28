# Brewing 重构方案

**日期**: 2026-03-27  
**范围**: `Brewing/` 目录及其与仓库根目录的边界  
**目标**: 把当前的 skeleton/prototype 形态，整理成更像“框架”的可维护结构

---

## 1. 问题定义

当前 `Brewing/` 的主要问题不是“功能缺失”，而是**边界混杂**：

- 它一边想做框架，一边又直接依赖仓库根目录的 `datagen/`
- 它一边有抽象层（benchmark / method / diagnostics / resources），一边又把 fixture、synthetic cache、legacy fallback 混进主路径
- 它一边有统一 workflow 文档，一边又在实现里保留大量 skeleton 阶段的穿透式写法

这会导致几个直接后果：

- 目录看起来散，因为“框架代码”和“项目专用代码”没有分层
- 文件看起来厚，因为 schema、runtime、adapter、测试替身都堆在同一处
- 后续扩展 benchmark / method 时，容易继续往现有 god file 里加代码
- 使用者很难判断哪些模块是稳定接口，哪些只是当前项目的临时实现

---

## 2. 重构目标

这次重构不追求一次性重写，而是追求四件事：

1. 让 `Brewing` 成为**边界自洽**的 package
2. 让“框架层”和“CUE-Bench 项目层”明确分开
3. 让运行主路径和测试/fixture/原型 fallback 明确分开
4. 让目录结构反映架构文档中的职责划分，而不是反过来靠文档解释代码

---

## 3. 重构原则

- **先整理边界，再整理算法**
- **先拆文件职责，再做抽象升级**
- **先保持行为兼容，再逐步清理 fallback**
- **框架层不直接知道仓库根目录布局**
- **benchmark 是插件式接入，不是写死在 orchestrator 里**

---

## 4. 目标目录结构

建议把 `Brewing/brewing/` 收敛为下面这类结构：

```text
Brewing/
├── pyproject.toml
├── brewing/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── run_config.py
│   ├── schema/
│   │   ├── __init__.py
│   │   ├── sample.py
│   │   ├── benchmark.py
│   │   ├── cache.py
│   │   ├── method_result.py
│   │   └── diagnostics.py
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── resources.py
│   │   └── registry.py
│   ├── cache/
│   │   ├── __init__.py
│   │   └── builder.py
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── linear_probing.py
│   │   └── csd.py
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   ├── brewing_metrics.py
│   │   └── outcome_rules.py
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── model_api.py
│   │   └── nnsight_backend.py
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   └── cue_bench/
│   │       ├── __init__.py
│   │       ├── spec.py
│   │       ├── adapter.py
│   │       ├── fixtures.py
│   │       └── builder.py
│   └── testing/
│       ├── __init__.py
│       ├── synthetic_cache.py
│       └── mock_results.py
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

---

## 5. 目录设计说明

### 5.1 `schema/`

职责：只放数据结构与序列化协议。

这里应该包含：

- `Sample`
- `BenchmarkSpec` / `SubsetSpec`
- `DatasetManifest`
- `HiddenStateCache`
- `FitArtifact`
- `MethodResult`
- `DiagnosticResult`

这里不应该包含：

- 运行逻辑
- registry
- benchmark 构造细节
- diagnostics 计算函数

当前 `data.py` 的问题就是把“类型定义”和“框架其他概念”全塞进一个文件里。拆完后，schema 会更像接口层。

### 5.2 `runtime/`

职责：只负责 workflow 调度与资源生命周期。

这里应该包含：

- `Orchestrator`
- `ResourceManager`
- `Registry`
- config compatibility check

这里不应该包含：

- CUE-Bench 的 fixture fallback
- datagen import 细节
- 模型后端细节

### 5.3 `benchmarks/`

职责：每个 benchmark 作为一个独立子包管理。

`cue_bench/` 内部建议继续拆：

- `spec.py`: subset 定义、benchmark spec
- `fixtures.py`: fixture samples
- `adapter.py`: datagen/raw json -> `Sample`
- `builder.py`: build/load benchmark dataset

这样 CUE-Bench 的项目特定逻辑不会再污染框架层。

### 5.4 `backends/`

职责：封装模型访问与 patch/intervention 后端。

这是当前最缺的一层。现在 `csd.py` 直接混着：

- 方法定义
- nnsight 接口
- legacy hook fallback
- 模型结构假设

应该改成：

- `methods/csd.py` 只定义“CSD 要计算什么”
- `backends/nnsight_backend.py` 负责“如何从某个后端拿 logits / patch hidden states”

这样后面要换 backend，或者兼容不同模型结构，不会继续把方法层写脏。

### 5.5 `testing/`

职责：收纳测试替身和 synthetic helper。

像下面这些都应从主路径中弱化出来：

- synthetic cache
- mock CSD result
- fixture-only shortcut

它们不是框架主能力，只是测试配套。

---

## 6. 文件级重构建议

### 6.1 第一优先级

#### `brewing/data.py`

现状：

- 文件过大
- 数据结构过多
- 接口层不清晰

建议拆分为：

- `schema/sample.py`
- `schema/benchmark.py`
- `schema/cache.py`
- `schema/method_result.py`
- `schema/diagnostics.py`

#### `brewing/benchmarks/cue_bench.py`

现状：

- spec、fixture、adapter、builder 混在一起

建议拆分为：

- `benchmarks/cue_bench/spec.py`
- `benchmarks/cue_bench/fixtures.py`
- `benchmarks/cue_bench/adapter.py`
- `benchmarks/cue_bench/builder.py`

#### `brewing/orchestrator.py`

现状：

- 配置定义、数据集构建、train/eval split、cache resolve、method run、diagnostic run、summary 写盘都在同一个文件

建议拆分为：

- `config/run_config.py`
- `runtime/orchestrator.py`
- `runtime/pipeline_steps.py` 或按私有 helper 保留在 runtime 包内

### 6.2 第二优先级

#### `brewing/methods/csd.py`

现状：

- 方法语义和 backend 细节耦合
- batch path 和 fallback path 语义不一致
- 直接依赖模型内部层路径

建议拆分为：

- `methods/csd.py`: 方法定义、输入输出协议
- `backends/nnsight_backend.py`: patchscope/logit readout
- `backends/model_api.py`: 抽象 layer access / logits access

#### `brewing/cli.py`

现状：

- 通过 `sys.path.insert` 把父仓库暴露进来
- CLI 自己承担 import bootstrap
- 176 行 argparse 参数定义和 RunConfig 之间大量胶水代码

建议调整为：

- `cli/main.py`
- **YAML-config-only**：砍掉所有散装 argparse 参数，只接收 `--config` 路径
- 支持 glob 批量运行（`--config experiments/*.yaml`）
- 只消费已安装的 `brewing` package
- benchmark adapter 作为显式依赖注册，不再靠父目录注入

### 6.3 第三优先级

#### `brewing/cache_builder.py`

建议拆成：

- `cache/builder.py`
- `testing/synthetic_cache.py`

把真实 cache 构建和 synthetic helper 分开。

#### `brewing/diagnostics.py`

建议拆成：

- `diagnostics/brewing_metrics.py`
- `diagnostics/outcome_rules.py`

把 FPCL/FJC 计算和 outcome rule 分开。

---

## 7. 运行边界重构

这是这次最关键的一步。

### 7.1 当前问题

`Brewing` 现在并不是一个真正独立的 package，因为：

- CLI 要向父目录插 `sys.path`
- benchmark builder 直接假设根目录 `datagen` 存在
- 测试也依赖仓库级路径注入

这意味着它其实是“仓库中的子模块”，不是“可安装的框架包”。

### 7.2 目标

把 `Brewing` 改成下面两层关系：

- `brewing`：框架核心包
- `cue_project_adapter`：本仓库里接 CUE datagen 的适配层

如果暂时不想拆成两个分发包，至少也应该在目录职责上模拟这个边界：

- 框架不直接 import 仓库根目录模块
- benchmark adapter 通过明确注册接入
- datagen 缺失时，报清晰错误，而不是悄悄 fallback 到隐式环境

---

## 8. 测试结构重构

建议把测试分三层：

### 8.1 `tests/unit/`

测纯逻辑：

- schema round-trip
- resource path resolution
- FPCL/FJC/outcome 规则
- label encoding / artifact id

### 8.2 `tests/integration/`

测模块协作：

- benchmark builder -> dataset manifest
- cache builder -> method -> diagnostics
- registry -> orchestrator

### 8.3 `tests/e2e/`

测完整 pipeline，但要明确：

- fixture mode
- synthetic mode
- real backend mode

不要把所有 smoke path 都堆在一个 `test_e2e.py` 里。

---

## 9. 分阶段迁移计划

### Phase 1: 纯结构整理，不改行为

目标：先让目录形态像框架。

步骤：

1. 拆 `data.py`
2. 拆 `cue_bench.py`
3. 把 `RunConfig` 从 `orchestrator.py` 拆出
4. 把 `synthetic cache` 从主 cache builder 拆出
5. 保留兼容导出，避免 import 全部断掉

产出：

- 目录更清晰
- import 路径仍兼容
- 单测应基本不变

### Phase 2: 清理边界穿透

目标：让 `Brewing` 成为自洽 package。

步骤：

1. 去掉 CLI/test 中的 `sys.path.insert`
2. datagen 通过 adapter 注册接入
3. benchmark builder 不再默认依赖仓库根目录
4. 把 fixture fallback 改成显式测试模式

产出：

- 包边界稳定
- 安装/运行模型更清晰

### Phase 3: 后端抽象化

目标：把方法层和模型执行层分开。

步骤：

1. 为 CSD 引入 backend API
2. 把 nnsight 访问逻辑迁到 `backends/`
3. 去掉方法中对模型内部结构的硬编码
4. 统一 batch path / fallback path 的语义

产出：

- CSD 更像框架方法，而不是一段实验脚本
- 未来加 activation patching / re-injection 时可复用 backend

### Phase 4: 主路径去原型化

目标：把测试替身和真实运行主路径分开。

步骤：

1. synthetic helper 下沉到 testing support
2. fixture mode 显式标记为 testing-only
3. 把 legacy fallback 改成可配置 backend strategy
4. 把 paper-specific 默认值迁到 config 层

产出：

- 代码阅读时主路径更干净
- 用户能分清哪些是正式接口，哪些是测试设施

---

## 10. 兼容策略

为了避免这次重构成本过高，建议保留一轮兼容层。

### 10.1 import 兼容

例如：

- `brewing.data` 暂时保留，但内部只 re-export 新 schema 模块
- `brewing.orchestrator` 暂时保留，但实际实现迁到 `runtime.orchestrator`
- `brewing.benchmarks.cue_bench` 暂时保留 re-export

### 10.2 路径兼容

不要立刻改资源落盘格式：

- `datasets/`
- `caches/`
- `artifacts/`
- `results/`

这层先不动，避免历史实验产物失效。

### 10.3 CLI 设计决策

**保留 CLI，但大幅精简为 YAML-config-only 模式。**

设计原则：
- CLI 只做一件事：接收 YAML config 路径，交给 Orchestrator
- 砍掉所有 argparse 散装参数（`--model`、`--method` 等），所有实验配置走 YAML
- 批量实验 = 多个 YAML 文件 + shell glob（`brewing run --config experiments/*.yaml`）
- 复现实验 = 分享 YAML 文件

理由：
- 作为框架，CLI 入口是必要的，方便脚本化批量跑实验
- 但 176 行 argparse 胶水代码维护成本高，改一个字段要同步 argparse 和 RunConfig
- YAML 驱动与 RFC 中的"YAML 驱动、7 插槽热插拔"方向一致
- CLI 精简到几十行后基本不需要维护

---

## 11. 不建议现在做的事

下面这些事情有价值，但不建议和本轮目录重构绑定：

- 一次性把所有方法抽象成完美插件系统
- 一次性支持多 benchmark、多 answer type、多 token 全场景
- 一次性重写 CSD 和 probing 的全部科学逻辑
- 一次性重构测试语义和实验配置语义

如果现在同时做这些，重构会从“结构收敛”变成“全面重写”，风险太高。

---

## 12. 推荐执行顺序

建议按下面顺序推进：

1. 拆 `data.py`
2. 拆 `cue_bench.py`
3. 拆 `RunConfig` 与 `orchestrator`
4. 把 `synthetic/testing helpers` 从主路径移走
5. 建立 `backends/` 目录并迁移 CSD 后端细节
6. 去掉 `sys.path.insert`
7. 最后再处理更深层的 API 抽象

---

## 13. 预期结果

如果按上面的方案完成，`Brewing/` 应该会从现在的：

- “研究仓库里的一个可运行子目录”

变成更接近下面这种形态：

- “有清晰 schema、runtime、benchmark adapter、backend、testing support 的框架包”

到那一步，别人看目录时就不需要靠文档脑补结构了，目录本身就能说明它是什么。

---

## 14. 一句话结论

`Brewing` 现在乱，不是因为代码量大，而是因为**框架层、项目层、原型层没有切开**。  
这次重构的核心不是“再设计一套架构”，而是**把文档里已经讲清楚的架构真正体现在目录和模块边界上**。
