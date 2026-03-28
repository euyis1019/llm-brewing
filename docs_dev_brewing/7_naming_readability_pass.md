# 7. Naming & Readability Pass

**Date**: 2026-03-28
**Scope**: `Brewing/brewing/` 全部核心模块 + `Brewing/tests/` + `docs/architecture.md`
**Goal**: 统一术语、消除 nnterp 残留命名、提高新维护者可读性

---

## 1. nnterp 残留命名清理

### cache_builder.py

| 旧名 | 新名 | 原因 |
|------|------|------|
| `is_nnterp` | `is_nnsight_model` | 检测的是 nnsight LanguageModel，不是 nnterp 包 |
| `_extract_nnterp()` | `_extract_nnsight()` | 实际使用 nnsight tracing API |
| docstring "nnterp StandardizedTransformer" | "nnsight LanguageModel or HF AutoModelForCausalLM" | 反映实际类型 |
| comment "nnterp attaches it" | "nnsight LanguageModel exposes .tokenizer" | 准确描述 |

### csd.py

| 旧名 | 新名 | 原因 |
|------|------|------|
| `_get_baseline_logits_nnterp()` | `_get_baseline_probs_nnsight()` | 1) 不是 nnterp 2) 返回的是 probs 不是 logits |
| `baseline_logits` (batch path) | `baseline_probs` | 与函数返回值语义一致 |
| `answer_logits` (batch path) | `answer_probs` | patchscope_lens 返回 probs |

### nnsight_ops.py

- 模块 docstring 更新：明确说明 nnterp 依赖已移除，仅保留 attribution 注释

---

## 2. Orchestrator 命名修正

| 旧名 | 新名 | 原因 |
|------|------|------|
| `_resolve_eval_cache()` | `_resolve_hidden_cache()` | 该方法同时用于 eval 和 train cache |

---

## 3. sys.path.insert 清理

| 文件 | 修改 |
|------|------|
| `cli.py:19` | 移除 `sys.path.insert`，同时移除不再需要的 `import sys` |
| `test_e2e.py:8-10` | 移除 `sys.path.insert` 和对应的 `sys`, `Path` import |

---

## 4. Module-level docstrings 增强

为以下文件增加了 "Responsible for / NOT responsible for" 格式的模块说明：

- `cache_builder.py`: 明确两种 backend（nnsight / HF）的职责
- `orchestrator.py`: 明确 S0→S1→S2 职责，不含 S3
- `resources.py`: 明确 5 种 managed resource types
- `registry.py`: 明确 import-time registration 机制
- `methods/base.py`: 明确方法类型层级（CacheOnly / ModelOnline × Trained / Training-free）
- `methods/linear_probing.py`: 明确 cache-only + training-required
- `methods/csd.py`: 明确两条执行路径（batch vs per-sample），标注已知 BUG-2
- `diagnostics/__init__.py`: 明确 S3 解耦，两个入口点
- `diagnostics/metrics.py`: 列出三个指标的含义
- `diagnostics/outcome.py`: 明确四类 outcome 分类职责
- `schema/__init__.py`: 列出三个子文件的内容
- `schema/types.py`: 列出所有 value objects
- `schema/results.py`: 列出所有输出类型
- `nnsight_ops.py`: 明确作为 tracing/intervention backend 的职责
- `cli.py`: 标注为过渡接口，将来会换成 YAML-config-only

---

## 5. Orchestrator 内联注释

- `_resolve_hidden_cache` 的 synthetic cache fallback 路径增加了 `TESTING-ONLY` 标注

---

## 6. docs/architecture.md 更新

- §5.3 diagram: `nnterp StandardizedTransformer` → `nnsight LanguageModel`
- §6 标题: `nnterp 基座` → `NNsight 基座`
- §6.1: 更新为 nnsight 能力表，增加函数名引用
- §6 顶部: 增加历史说明 block，解释 nnterp → nnsight 迁移
- §7: `nnterp 提供` → `nnsight 提供`

---

## 7. pyproject.toml build backend 修复

`build-backend = "setuptools.backends._legacy:_Backend"` 是非标准路径，导致 `uv run` editable install 失败。改为标准的 `setuptools.build_meta`。

验证：`uv run python -m brewing --help` 和 `uv run --extra dev python -m pytest tests/ -q` 均通过。

---

## 8. cache_builder backend detection docstring 收紧

模块 docstring 原本说 "auto-detected via duck-typing"，暗示了一个比实际更强的兼容范围。实际检测只是 `hasattr(model, "trace") and hasattr(model, "layers")`。

修改为明确描述实际检测方式及其局限性，并引导维护者在需要支持新模型结构时去扩展 `nnsight_ops.get_layers()`。

---

## 测试结果

37/37 tests passed，无行为变化。

---

## 未改动 / 建议下一轮处理

1. **CSD batch vs per-sample 语义不一致** (BUG-2): batch 路径在 log-prob space 做 baseline subtraction，per-sample 在 logit space。本轮仅改了变量名使语义更清晰，未修复算法。
2. **`has_model_online` 在 cli.py 中硬编码检查 `"csd"`**: 应改为查询 method requirements。
3. **synthetic cache 的 `n_layers=28` 硬编码**: 应从 model config 推导。
4. **`SampleMethodResult.from_dict` 的 `layer_predictions` 反序列化逻辑**: 嵌套三元表达式可读性差，建议重写为显式分支。
5. **`N_CLASSES` 常量 in linear_probing.py**: 应从实际 `answer_space` 推导，而非模块级硬编码。
6. **测试中缺少 Overprocessed outcome 覆盖** (GAP-3): `make_synthetic_cache` 的 `model_predictions = answers` 导致永远不会测到 Overprocessed。
7. **YAML-config-only CLI**: 当前 CLI 有 176 行 argparse，待重构为纯 YAML 驱动。
