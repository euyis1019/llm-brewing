# 14. Experiment Hardening: 消除静默退化路径

**日期**: 2026-03-29

## 背景

Log 13 审计发现框架在若干关键路径失败时会静默退化，导致实验"跑完但结果无效"。本次修改集中收掉这些问题，使框架达到正式批量实验的可靠性要求。

## 已修复（本次之前）

- synthetic cache fallback 已移除（`base.py` raise RuntimeError）
- model load failure 已 raise（`cli.py`）
- CLI `get_benchmark` NameError 已修复

## 本次修改

### 1. `load_generated_dataset()` 添加 split 参数

**文件**: `brewing/benchmarks/cue_bench/builder.py`

- 新增 `split: str | None` 参数
- `split` 指定时只查找 `{data_dir}/{split}/{task}.json`，不存在则 raise FileNotFoundError
- `split=None` 保留原有兜底逻辑（向后兼容）
- 调用方 `PipelineBase.resolve_dataset()` 根据 purpose 自动传递对应 split
- `build_eval_dataset()` 传 `split="eval"`

### 2. Diagnostics 缺 cache 时 fail hard

**文件**: `brewing/diagnostics/outcome.py`

- `run_diagnostics_from_disk()` 新增 `allow_no_cache: bool = False` 参数
- 默认情况下 cache 缺失 raise FileNotFoundError
- 仅 `allow_no_cache=True` 时允许降级运行（附 warning 说明 outcome 偏差）
- 解决问题：之前 model_predictions=None 会导致所有有 FJC 的 sample 被系统性判为 Overprocessed

### 3. RunConfig 校验加强

**文件**: `brewing/schema/results.py`

- `mode="train_probing"` + `data_dir=None` 时发 UserWarning
- `mode="diagnostics"` + `methods` 非空时发 UserWarning
- 均为 warn 不 raise（不阻塞执行）

### 4. DiagnosticsPipeline 汇总输出

**文件**: `brewing/pipelines/diagnostics.py`

- `run()` 结束后写一个汇总 JSON 到 `{output_root}/diagnostics_summary/{model_id_safe}.json`
- 每行包含 model_id, task, outcome_distribution, mean_fpcl, mean_fjc, mean_delta_brew
- 多模型各自一个文件，方便后续合并画图

### 5. 新增测试

**文件**: `tests/test_diagnostics.py`

- `test_missing_cache_raises_by_default`: 验证缺 cache 时 raise
- `test_missing_cache_allowed_with_flag`: 验证 `allow_no_cache=True` 可降级
- `TestLoadGeneratedDatasetSplit`: split="eval" / "train" / None 三种场景
- `TestRunConfigValidation`: train_probing 和 diagnostics 的 warning 验证

## 测试结果

```
87 passed, 4 skipped, 9 warnings
```

## 修改文件

| 文件 | 变更 |
|------|------|
| `brewing/benchmarks/cue_bench/builder.py` | `load_generated_dataset` 新增 split 参数 |
| `brewing/pipelines/base.py` | `resolve_dataset` 传 split |
| `brewing/diagnostics/outcome.py` | cache 缺失 fail hard + `allow_no_cache` |
| `brewing/schema/results.py` | RunConfig 校验 warning |
| `brewing/pipelines/diagnostics.py` | 汇总 JSON 输出 |
| `tests/test_diagnostics.py` | 9 个新测试 |
