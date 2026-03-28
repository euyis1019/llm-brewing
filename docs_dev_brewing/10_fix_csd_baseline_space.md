# 10 — Fix CSD baseline subtraction space inconsistency (BUG-2)

## 问题

CSD 有两条执行路径，baseline subtraction 在不同空间进行：

| 路径 | 模型输出 | baseline | 减法空间 |
|------|---------|----------|---------|
| Batch (patchscope_lens) | probs (softmax) | probs (nnsight) | `log(p) - log(baseline)` (log-prob) |
| Per-sample (HF hook) | raw logits | raw logits | `logits - baseline` (logit) |

数学上两者差一个全局常数 `C = logZ - logZ_baseline`（对所有 answer token 相同），由于后续 argmax 和 softmax 都是 shift-invariant 的，**预测结果和 confidence 实际等价**。但代码语义混乱，变量名误导，且依赖隐式的 shift-invariance 保证——如果将来有人改了后处理逻辑就会引入真 bug。

## 修复

统一两条路径到 **logit space**：

### `nnsight_ops.py`
- 新增 `get_next_token_logits()` — 返回 `logits[:, -1, :]`（不过 softmax）
- `patchscope_lens()` 新增 `return_logits: bool = False` 参数
  - `True` → 用 `get_next_token_logits`，返回 raw logits
  - `False`（默认）→ 行为不变，返回 probs，保持向后兼容

### `methods/csd.py`
- `_run_batch_patchscope()`: 调用 `patchscope_lens(return_logits=True)`，直接在 logit space 做减法
- `_get_baseline_probs_nnsight()` → `_get_baseline_logits_nnsight()`：用 `get_logits()[:, -1, :]` 替代 `model.next_token_probs`
- 模块 docstring 更新，移除 BUG-2 注释

### `Brewing/CLAUDE.md`
- 已知问题中标记 BUG-2 为已修复

## 改动文件

- `brewing/nnsight_ops.py`
- `brewing/methods/csd.py`
- `Brewing/CLAUDE.md`
