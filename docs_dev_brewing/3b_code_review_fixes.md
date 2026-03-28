# 5. Code Review Fixes: BUG-1, DESIGN-1, DESIGN-2

**Date**: 2026-03-27
**Scope**: `orchestrator.py`, `diagnostics.py`
**Fixes**: 3_code_review.md 中的 BUG-1, DESIGN-1, DESIGN-2

---

## Fix 1: Train/Eval 数据分离 (BUG-1)

**问题**: `_run_method` 中 `train_samples = eval_samples`，Probing 在 eval 集上训练又在同一数据上评估。

**修改文件**: `Brewing/brewing/orchestrator.py`

**方案**:
- 在 `RunConfig` 新增 `train_split: float | None` 字段（默认 `None`，运行时 fallback 到 0.8）
- 在 `_run_subset` 中 S0 之后、S1 之前新增 **S0b** 步骤：`_resolve_train_dataset()`
  - 优先级 1: 用户通过 `train_dataset_id` 指定已有训练集
  - 优先级 2/3: 使用 `train_split`（或默认 0.8）从全量数据中切分
  - 使用 `sklearn.model_selection.train_test_split`，`random_state=seed`，确定性
  - train/eval 两份数据各自独立落盘（DatasetManifest + samples.json）
  - 切分后 eval_samples 就地更新为 eval 部分
- 新增 **S1b** 步骤：为 train 数据集独立 resolve/build hidden cache
- `_run_method` 新增 `train_samples`, `train_dataset_id`, `train_cache` 参数
- `_run_method` 中加入 train/eval sample ID 交集检查，若有交集直接报错

**关键决策**:
- split 发生在 `_run_subset` 层面（S0 之后、S2 之前），不在方法内部
- `LinearProbing._fit_probes` 和 `.run` 的 evaluate 部分均不修改
- 如果 train dataset 从之前的 run 已存在于 disk，直接加载不重新 split

---

## Fix 2: `compute_csd_tail_confidence` fallback 语义错误 (DESIGN-1)

**问题**: 当 `layer_confidences is None` 时，fallback 到 `layer_values`（correctness 0/1），把 boolean 当 confidence 用。

**修改文件**: `Brewing/brewing/diagnostics.py`

**方案**:
- 移除 fallback 到 `layer_values` 的代码
- 当 `layer_confidences is None` 时，log warning 并返回 `0.0`（保守归为 Unresolved）
- CSD 的两条路径（batch + per-sample）都已填充 `layer_confidences`，正常运行不会触发 fallback

---

## Fix 3: `eval_dataset_id` 多 subset 共享 (DESIGN-2)

**问题**: 用户指定 `config.eval_dataset_id` 时所有 subset 用同一个 ID，后续 subset 加载第一个 subset 的数据。

**修改文件**: `Brewing/brewing/orchestrator.py`

**方案**:
- 在 `_run_subset` 中：当 `self.config.eval_dataset_id` 非空且 `len(self.subsets) > 1` 时，自动追加 `-{subset_name}` 后缀
- 单 subset 时保持原样使用用户指定的 ID
