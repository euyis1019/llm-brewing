# 4. 移除 nnterp 依赖，内联 nnsight 操作

**Date**: 2026-03-27

---

## 背景

Brewing 框架此前直接 `import nnterp`，存在两处运行时依赖：
- `csd.py` — `from nnterp.interventions import patchscope_lens, TargetPrompt`
- `cache_builder.py` — `from nnterp.nnsight_utils import get_token_activations`

决定完全去除 nnterp 作为依赖，将所需代码摘入 Brewing 自有模块。Brewing 只保留对底层 `nnsight` 的依赖。

## 做了什么

### 新建 `Brewing/brewing/nnsight_ops.py`

从 nnterp 摘出 Brewing 实际使用的功能，包括：

| 函数 / 类 | 来源 | 用途 |
|-----------|------|------|
| `get_layers`, `get_num_layers`, `get_layer`, `get_layer_output` | `nnterp.nnsight_utils` | layer access helpers |
| `get_logits`, `get_next_token_probs` | `nnterp.nnsight_utils` | logit / prob 读取 |
| `get_token_activations` | `nnterp.nnsight_utils` | 逐层 hidden state 提取 |
| `TargetPrompt`, `TargetPromptBatch` | `nnterp.interventions` | patchscope 注入位置 dataclass |
| `patchscope_lens` | `nnterp.interventions` | hidden state 注入 + next-token readout |

文件头部保留了对 nnterp 项目的 attribution 注释。

### 更新消费方 import

- `csd.py`: `from nnterp.interventions import ...` → `from brewing.nnsight_ops import ...`
- `cache_builder.py`: `from nnterp.nnsight_utils import ...` → `from brewing.nnsight_ops import ...`

同步更新了两个文件中涉及 nnterp 的 docstring。

## 未改动

- `patchscope_lens` 的内部逻辑保持原样（仍返回 probs），BUG-2 的修复后续单独进行。
- `_extract_nnterp` 函数名及 `is_nnterp` 检测逻辑未改（语义上仍指 nnsight 模型接口，非 nnterp 包依赖）。

## 验证

`grep` 确认 `Brewing/` 下不再有任何 `from nnterp` 或 `import nnterp` 的运行时导入。
