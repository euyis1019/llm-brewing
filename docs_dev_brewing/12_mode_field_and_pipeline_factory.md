# 12. Mode 字段 + Pipeline 工厂：统一运行入口

**日期**: 2026-03-29
**状态**: 设计确认，待实现

## 动机

当前三种运行模式的入口不统一：

| 模式 | 当前入口 | 问题 |
|------|---------|------|
| Probe Training | Python API (`LinearProbing.train()`) | 无 CLI/config 入口，用户需手写脚本 |
| Eval Pipeline | CLI (`python -m brewing --config`) | 唯一有 CLI 入口的模式 |
| Diagnostics | Python API (`run_diagnostics_from_disk()`) | 同上 |
| Cache-only | 不存在 | 想只提 hidden states 得跑完整 pipeline |

问题：
1. 只有 eval 能从 YAML config 驱动，其他模式要写 Python 脚本
2. 框架号称 YAML-config-only，但三分之二的功能没有 config 入口
3. Orchestrator 硬编码了 S0→S1→S2 eval 流程，无法灵活组合 pipeline 阶段

## 设计方案

### 1. RunConfig 新增 `mode` 字段

```yaml
mode: train_probing   # 新字段
model_id: Qwen/Qwen2.5-Coder-7B-Instruct
# ... 其余字段不变
```

合法值：

| mode | 执行阶段 | 需要模型在线 | 产出 |
|------|---------|-------------|------|
| `cache_only` | S0 → S1 | 是 | datasets/ + caches/ |
| `train_probing` | S0 → S1 → fit | 是（提 train cache） | datasets/ + caches/ + artifacts/ |
| `eval` | S0 → S1 → S2 | CSD 需要 | datasets/ + caches/ + results/ |
| `diagnostics` | S3 only | 否 | diagnostics.json |

默认值：`eval`（向后兼容）。

### 2. Pipeline 工厂模式

当前 Orchestrator.run() 硬编码 eval 流程。改为：每个 mode 对应一个独立的 pipeline 执行器，Orchestrator 通过工厂分发。

```
                ┌─────────────────────┐
                │    Orchestrator     │
                │   (调度 + 资源管理)   │
                └─────────┬───────────┘
                          │ match config.mode
            ┌─────────────┼──────────────┬──────────────┐
            ▼             ▼              ▼              ▼
      CacheOnlyPipeline  TrainPipeline  EvalPipeline  DiagPipeline
      S0 → S1           S0 → S1 → fit  S0 → S1 → S2  S3
```

核心代码结构：

```python
# brewing/pipelines/base.py
class PipelineBase(ABC):
    """Pipeline 执行器基类。"""
    def __init__(self, config: RunConfig, resources: ResourceManager, benchmark): ...

    @abstractmethod
    def run(self, model=None, tokenizer=None) -> dict[str, Any]: ...

    # 共享的 S0/S1 逻辑作为 base 方法
    def resolve_eval_dataset(self, subset, key): ...
    def resolve_train_dataset(self, subset, key): ...
    def resolve_hidden_cache(self, key, samples, model, tokenizer): ...


# brewing/pipelines/eval.py
class EvalPipeline(PipelineBase):
    """S0 → S1 → S2：当前 Orchestrator._run_subset 的逻辑迁移至此。"""
    def run(self, model=None, tokenizer=None) -> dict: ...


# brewing/pipelines/train.py
class TrainPipeline(PipelineBase):
    """S0 → S1 → probe training → artifact 落盘。"""
    def run(self, model=None, tokenizer=None) -> dict: ...


# brewing/pipelines/cache_only.py
class CacheOnlyPipeline(PipelineBase):
    """S0 → S1：只提 hidden states。"""
    def run(self, model=None, tokenizer=None) -> dict: ...


# brewing/pipelines/diagnostics.py
class DiagnosticsPipeline(PipelineBase):
    """S3：从落盘的 MethodResult 运行诊断。"""
    def run(self, model=None, tokenizer=None) -> dict: ...
```

工厂分发（在 Orchestrator 或独立函数中）：

```python
# brewing/pipelines/__init__.py
PIPELINE_REGISTRY: dict[str, type[PipelineBase]] = {
    "cache_only": CacheOnlyPipeline,
    "train_probing": TrainPipeline,
    "eval": EvalPipeline,
    "diagnostics": DiagnosticsPipeline,
}

def create_pipeline(config: RunConfig, resources, benchmark) -> PipelineBase:
    cls = PIPELINE_REGISTRY.get(config.mode)
    if cls is None:
        raise ValueError(f"Unknown mode: {config.mode!r}")
    return cls(config, resources, benchmark)
```

### 3. Orchestrator 的角色变化

重构后 Orchestrator 变薄：

```python
class Orchestrator:
    def __init__(self, config: RunConfig):
        self.config = config
        self.resources = ResourceManager(config.output_root)
        self.benchmark = get_benchmark(config.benchmark)

    def run(self, model=None, tokenizer=None) -> dict:
        pipeline = create_pipeline(self.config, self.resources, self.benchmark)
        return pipeline.run(model=model, tokenizer=tokenizer)
```

外部 API 不变——`Orchestrator(config).run(model, tokenizer)` 依旧工作。
CLI 不变——`python -m brewing --config config.yaml` 依旧工作。
只是内部走了工厂分发。

### 4. TrainPipeline 的特殊 config 需求

训练模式需要额外信息（训练数据来源），通过现有字段或 method_configs 传递：

```yaml
mode: train_probing
model_id: Qwen/Qwen2.5-Coder-7B-Instruct
data_dir: /path/to/cue/data/colm_v1/train    # 指向 train split
subsets:
- value_tracking
- computing
method_configs:
  linear_probing:
    probe_params:
      solver: lbfgs
      C: 1.0
      max_iter: 1000
    overwrite: false
```

### 5. needs_model_online 逻辑更新

cli.py 中的 `needs_model_online()` 需要感知 mode：

| mode | 需要模型 |
|------|---------|
| `cache_only` | 是（提 cache 需要 forward pass） |
| `train_probing` | 是（提 train cache） |
| `eval` | 看 methods（CSD 需要，probing 不需要） |
| `diagnostics` | 否 |

## 文件变更预估

| 操作 | 文件 |
|------|------|
| 新建 | `brewing/pipelines/__init__.py` — 工厂 + registry |
| 新建 | `brewing/pipelines/base.py` — PipelineBase（从 Orchestrator 提取 S0/S1） |
| 新建 | `brewing/pipelines/eval.py` — EvalPipeline（现有 Orchestrator 逻辑迁入） |
| 新建 | `brewing/pipelines/train.py` — TrainPipeline |
| 新建 | `brewing/pipelines/cache_only.py` — CacheOnlyPipeline |
| 新建 | `brewing/pipelines/diagnostics.py` — DiagnosticsPipeline |
| 修改 | `brewing/schema/results.py` — RunConfig 加 `mode` 字段 + 校验 |
| 修改 | `brewing/orchestrator.py` — 变薄，委托给 pipeline 工厂 |
| 修改 | `brewing/cli.py` — `needs_model_online` 感知 mode |
| 更新 | `docs/running_modes.md` — 反映新的统一入口 |
| 新增 | 测试：每个 pipeline 的基本行为 |

## 向后兼容

- `mode` 默认值为 `eval`，现有 config 无需改动
- Orchestrator 外部 API 不变
- CLI 命令不变
- 现有测试无需修改（走 eval pipeline）

## 不做的事

- 不做 pipeline DAG / 自定义 stage 编排——四个固定 mode 够用
- 不做 pipeline 之间的自动串联（如 train → eval → diagnostics）——用户自己按顺序跑
- 不引入新的依赖
