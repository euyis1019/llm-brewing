# Brewing 框架实验可用性评估

日期：2026-03-29

## 结论

`Brewing` 当前已经具备完整的实验框架形状，覆盖了：

- S0：数据集解析 / 构建
- S1：hidden state cache 提取与落盘
- S2：方法执行（`linear_probing` / `csd`）
- S3：诊断后处理（FPCL / FJC / Outcome）

但以当前仓库状态来看，它还**不适合直接用于正式整套实验**。主要问题不在于“缺模块”，而在于若干关键路径会在失败时静默退化，或者没有严格绑定到文档声明的实验设定。这会导致实验“能跑完”，但结果并不一定有效。

## 我检查了什么

我主要核对了以下几部分：

- 文档说明：`docs/project_overview.md`、`docs/running_modes.md`、`docs/cuebench_data.md`
- 主流程：`brewing/cli.py`、`brewing/orchestrator.py`、`brewing/pipelines/*.py`
- 方法实现：`brewing/methods/linear_probing.py`、`brewing/methods/csd.py`
- 数据与资源管理：`brewing/benchmarks/cue_bench/*`、`brewing/resources.py`
- 诊断逻辑：`brewing/diagnostics/outcome.py`
- 测试：`tests/test_e2e.py`、`tests/test_pipelines.py`

另外我实际运行了测试：

```bash
pytest -q
```

结果是：

- `80 passed`
- `2 failed`

失败都出现在 CLI 资源判断逻辑上，而不是测试本身写错。

## 主要问题

### 1. 模型不可用时会静默退回 synthetic cache

这是当前最大的问题。

在 CLI 中，如果模型加载失败，代码不会中止，而是只打印 warning：

- `brewing/cli.py:145`

随后在 pipeline 里，如果没有拿到 `model` 和 `tokenizer`，S1 会直接调用测试辅助函数去构造 synthetic cache：

- `brewing/pipelines/base.py:132`
- `tests/helpers.py:10`

具体风险：

- 你可能以为自己在跑真实模型 hidden states
- 实际上框架已经退回到 synthetic cache
- 训练、评估、诊断都会继续进行
- 最终还会正常落盘，看起来像是完整实验结果

这类行为对正式实验是不可接受的。测试环境可以接受 synthetic fallback，正式实验不应该接受。

建议：

- 把 synthetic cache fallback 限制在测试模式
- 正式运行时只要模型缺失、加载失败或 cache 不存在，就直接报错退出

### 2. `train_probing` 没有严格绑定正式 train split

文档里写得很明确，CUE-Bench 的正式切分是每个任务：

- train: 3240
- eval: 810

位置见：

- `docs/cuebench_data.md:26`

但 `train_probing` 的默认配置没有指定 `data_dir`：

- `brewing/config/colm/train_probing_qwen25_coder_7b.yaml:1`

而训练数据解析逻辑是：

1. 先读已落盘数据
2. 否则如果配置了 `data_dir`，就从磁盘数据集读
3. 否则直接动态生成
4. 再不行才退 fixture

实现见：

- `brewing/pipelines/base.py:66`

这意味着默认训练流程很可能不是在文档声明的 `train/` 数据上训练，而是在“重新动态生成的一整批样本”上训练。这样会导致：

- 训练集定义和文档不一致
- probe artifact 的来源不稳定
- 结果难以与后续 eval 对齐
- 难以复现实验

建议：

- `train_probing` 配置显式要求 `data_dir`
- 训练时强制从正式 `train` split 读取
- 不允许默认重新生成来代替正式 train split

### 3. 数据加载函数没有 split 语义，存在 train/eval 混读风险

这是比上一个更细但更危险的问题。

当前加载函数：

- `brewing/benchmarks/cue_bench/builder.py:20`

函数 `load_generated_dataset()` 不接收 `split` 参数。它查找文件时的顺序是：

1. `{data_dir}/{task}.json`
2. `{data_dir}/eval/{task}.json`
3. `{data_dir}/train/{task}.json`

也就是说，它根本不知道当前调用方是想读 train 还是 eval，只是按固定顺序找“第一个存在的文件”。

而 `PipelineBase.resolve_dataset()` 在 train/eval 两种场景里都调用了这个函数，但只传了 `subset_name`，没有传 split：

- `brewing/pipelines/base.py:80`

具体后果：

- 如果目录结构里同时有 `train/` 和 `eval/`
- 训练阶段有可能先匹配到 `eval/`
- 或者不同目录布局下读到不符合预期的文件

这会直接污染实验切分。

建议：

- 给 `load_generated_dataset()` 增加显式 `split` 参数
- 训练阶段只读 `train`
- 评估阶段只读 `eval`
- 如果目标 split 不存在，直接报错，不做隐式回退

### 4. diagnostics 在缺失 eval cache 时会继续运行，导致 outcome 偏差

`diagnostics` 是基于 probe result、CSD result 和模型最终输出一起做分类的。

在 `run_diagnostics_from_disk()` 里，框架会尝试加载 eval cache，以取出：

- `model_predictions`
- `n_layers`

见：

- `brewing/diagnostics/outcome.py:307`

但如果 cache 没有成功加载，代码不会报错，而是继续往下跑：

- `model_predictions = None`
- `n_layers = None`

之后在 `run_diagnostics()` 中，缺失的 `model_output` 会被默认成空字符串：

- `brewing/diagnostics/outcome.py:103`
- `brewing/diagnostics/outcome.py:119`

再结合分类规则：

- 只要 `fjc is not None` 且 `model_output != answer`
- 就会被判成 `overprocessed`

见：

- `brewing/diagnostics/outcome.py:32`

这意味着如果 cache 缺失，不是“诊断失败”，而是“悄悄给你一份系统性偏斜的诊断结果”。

建议：

- diagnostics 模式下默认要求 eval cache 存在
- 缺 cache 时直接失败
- 只有在显式声明“无模型输出分析”时，才允许降级运行

### 5. CLI 的 `needs_model_online()` 当前就有实际错误

我运行 `pytest -q` 后，两个失败都来自这里：

- `brewing/cli.py:41`
- `tests/test_pipelines.py:146`

问题是 `_all_caches_exist()` 调用了 `get_benchmark(config.benchmark)`，但文件顶部没有导入 `get_benchmark`。

这会导致：

- `cache_only` 模式下的资源判断报 `NameError`
- `train_probing` 模式下的资源判断报 `NameError`

虽然这是一个很小的实现错误，但它说明：

- CLI 主入口没有完全跑通
- 关键运行模式的前置判断还不稳定

如果入口层就不稳定，不适合直接做正式大规模实验。

建议：

- 先修掉该问题
- 为 `cache_only` / `train_probing` / `eval` / `diagnostics` 分别补入口级测试

### 6. 当前 e2e 测试主要验证的是 mock path，不是真实实验 path

仓库里虽然有 `test_e2e.py`，但它验证的是：

- fixture 样本
- synthetic cache
- synthetic CSD 结果

见：

- `tests/test_e2e.py:6`

这类测试可以说明：

- 数据结构能串起来
- 方法调用接口基本一致
- 诊断函数可以消费上游产物

但它不能说明：

- 真实模型加载路径稳定
- nnsight / HF cache 提取稳定
- CSD 在真实模型上可跑
- train/eval split 没混
- 正式落盘产物足够支撑复现实验

所以当前测试更接近“框架雏形可用”，不是“真实实验路径已验证”。

建议：

- 增加一条最小真实链路 smoke test
- 至少覆盖：真实 dataset 读取 -> cache 提取 -> probe eval -> diagnostics
- 即使只跑一个 subset、几条样本，也比 synthetic-only e2e 更有价值

## 对整体状态的判断

### 已经具备的能力

- 有明确的 pipeline 分层
- 有统一 schema 和资源落盘规范
- 有 dataset / cache / artifact / result / diagnostic 的持久化机制
- 有 `linear_probing` 与 `csd` 两类方法实现
- 有 YAML 配置、CLI、orchestrator 和 pipeline factory
- 有一定数量的单元测试和 smoke test

### 还不足以支撑正式实验的原因

- 关键路径存在静默 fallback
- train/eval 数据切分约束不严
- diagnostics 可在输入不完整时继续输出结果
- CLI 入口仍有确定性 bug
- e2e 证据主要来自 synthetic 路径

## 修复优先级建议

建议按下面顺序处理：

1. 去掉正式运行中的 synthetic cache fallback
2. 修复 `load_generated_dataset()` 的 split 语义
3. 让 `train_probing` 显式绑定正式 `train` 数据目录
4. 让 diagnostics 缺 cache 时直接失败
5. 修复 CLI 的 `get_benchmark` 导入问题
6. 增加真实实验路径的 smoke test

## 最终判断

如果问题是：

“这个框架模块是不是已经基本齐了？”

答案是：

- 是，已经是一个完整框架，不再是零散脚本

如果问题是：

“它现在是否足以直接支撑我完成整套正式实验？”

答案是：

- 还不行

更准确地说，它现在适合做：

- 框架开发
- 小规模调试
- mock / fixture 验证
- 部分真实流程试跑

但还不适合直接作为正式实验主干去批量产出可信结果，除非先把上面几项关键问题收掉。
