# Diagnostics 解耦：从 pipeline 内联到独立后处理

**日期**: 2026-03-28
**范围**: `brewing/orchestrator.py`, `brewing/diagnostics/outcome.py`

---

## 变更内容

### 设计决策

**Diagnostics (S3) 与 Orchestrator (S0-S2) 解耦。**

之前 diagnostics 嵌在 orchestrator 的 `run_subset()` 内，直接从内存中的 `method_results` 字典消费数据，必须在同一次 run 中同时跑完 probing + CSD 才能触发。

现在：

- Orchestrator 只负责 S0→S2：datagen → cache → methods → MethodResult 落盘
- Diagnostics 通过 `run_diagnostics_from_disk()` 从落盘文件独立运行
- 可以先用不同 config.yaml 跑完多个模型的 S0-S2，再统一做 S3 诊断和跨模型对比

### 代码改动

1. `**orchestrator.py`**
  - 移除 S3 diagnostics 代码（原 155-183 行）
  - docstring 更新为 "S0-S2 pipeline"
  - 不再 import `run_diagnostics` / `group_diagnostics_by_difficulty`
2. `**diagnostics/outcome.py**`
  - 新增 `run_diagnostics_from_disk()` 函数（~120 行）
  - 支持两种模式：
    - 按 `model_id` + `eval_dataset_id` 自动解析路径
    - 按显式文件路径指定
  - 从 ResourceManager 加载 MethodResult、HiddenStateCache、Samples
  - 委托 `run_diagnostics()` 做纯计算，最后落盘 DiagnosticResult
3. `**resources.py**`
  - 新增 `resolve_diagnostic()` 和 `save_diagnostic()` 方法
4. `**tests/test_diagnostics.py**`
  - 新增 `TestRunDiagnosticsFromDisk` 测试类（4 个测试）
  - 覆盖：ID 解析、显式路径、缺失文件报错

### 文档更新

- `docs/workflow_and_schema.md` — S3 从 pipeline 内移到解耦边界之后
- `docs/architecture.md` — 架构图标注 pipeline 边界

---

## 设计理由

- 不同 config.yaml 对应不同模型，批量跑 S0-S2 后需要统一做 diagnostics
- Diagnostics 本质上是对已有数据的后处理，不依赖 GPU 或模型在线
- 解耦后可以独立迭代 diagnostics 逻辑，不影响 pipeline 主路径

