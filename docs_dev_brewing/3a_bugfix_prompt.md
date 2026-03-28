# Code Agent Prompt: Brewing Framework Critical Bugfixes

以下 prompt 分为两批，按依赖关系排序。建议分两次发给 code agent。

---

## Batch 1: 核心正确性修复

```
请根据 code review 文档 `docs_dev_brewing/3_code_review.md` 修复以下问题。

修复前必读的上下文文档：
- `docs/workflow_and_schema.md` §3.3 — Training-required 方法的执行流程
- `docs/workflow_and_schema.md` §2.4.2 — Outcome 判定逻辑
- `Brewing/brewing/orchestrator.py`
- `Brewing/brewing/methods/linear_probing.py`
- `Brewing/brewing/diagnostics.py`
- `Brewing/brewing/methods/csd.py`

---

### Fix 1: Orchestrator 的 train data resolve 链 (BUG-1)

**问题**: `orchestrator.py:_run_method` 里 `train_samples = eval_samples`，
Probing 在 eval 集上训练又在同一数据上评估，构成数据泄漏。

**要求**:
- 在 `_run_method` 中，当 `method.requirements().trained == True` 时，
  实现完整的 S2b train data resolve-or-build 链：
  1. 通过 `train_dataset_id` resolve 独立的训练数据集（走 `resources.resolve_or_build_dataset`）
  2. 通过 `resources.resolve_or_build_cache` resolve 训练集的 hidden state cache
- 训练集的构建方式：
  - 优先级 1: 用户通过 `RunConfig.train_dataset_id` 显式指定已有的训练数据集
  - 优先级 2: 用户通过 `RunConfig.train_split`（新增字段，float，如 0.8）
    指定比例，框架从生成的数据中按比例切分。切出的 train 部分存为独立
    dataset（带独立的 dataset_id），eval 部分保持不变
  - 优先级 3: 如果都没指定，默认 `train_split=0.8`
- split 必须是确定性的（受 seed 控制），用 sklearn 的 `train_test_split`
  或等价逻辑即可，不需要 K-Fold
- split 发生在 `_run_subset` 层面（S0 之后、S2 之前），不是在方法内部
- split 后的 train/eval 两份数据各自独立落盘（各自有 DatasetManifest + samples.json）
- 对应的 train hidden cache 也要独立 resolve-or-build
- `LinearProbing._fit_probes` 不需要改 — 它只管拿到什么就训什么
- `LinearProbing.run` 的 evaluate 部分不需要改 — 它只在 eval_cache 上评估

**验证**: 修完后 `train_cache.sample_ids` 和 `eval_cache.sample_ids`
不能有交集。

---

### Fix 2: `compute_csd_tail_confidence` fallback 语义错误 (DESIGN-1)

**问题**: `diagnostics.py:compute_csd_tail_confidence` 的 fallback 路径
把 correctness flag (0/1) 当 confidence 用。`layer_values` 是 boolean
correctness，不是概率。

**要求**:
- 当 `layer_confidences is None` 时，不应 fallback 到 `layer_values`
- 正确行为：如果没有 `layer_confidences`，应该无法计算 tail confidence，
  返回 0.0（保守地归为 Unresolved），并 log 一个 warning
- 同时确保 CSD 方法总是填充 `layer_confidences`，使得正常运行时不会
  走到这个 fallback

---

### Fix 3: Orchestrator 的 eval_dataset_id 在多 subset 时错误共享 (DESIGN-2)

**问题**: 用户显式设定 `config.eval_dataset_id` 时，所有 subset 用了同一个
dataset_id，第二个 subset 会加载第一个 subset 的数据。

**要求**:
- 当用户指定了 `eval_dataset_id` 且同时跑多个 subset 时，自动加上
  subset 后缀，如 `f"{config.eval_dataset_id}-{subset_name}"`
- 或者：只有当 `subsets` 只有一个元素时才允许用户指定的 `eval_dataset_id`
  原样使用；多 subset 时强制加后缀
```

---

## Batch 2: CSD 实现修复

```
继续根据 `docs_dev_brewing/3_code_review.md` 修复 CSD 方法的实现问题。

修复前必读：
- `Brewing/brewing/methods/csd.py`
- `docs/architecture.md` §2.2 — CSD 定义

---

### Fix 4: CSD batch 路径 baseline subtraction 语义错误 (BUG-2)

**问题**: `_run_batch_patchscope` 中 `patchscope_lens` 返回 probs，
`_get_baseline_logits_nnterp` 也返回 probs。代码在 log 空间做减法
（log-ratio），而非论文要求的 logit 空间减法。
per-sample 路径用的是真正的 logits 做减法，两条路径语义不一致。

**要求**:
- 统一两条路径的 baseline subtraction 到 logit 空间
- batch 路径：要么修改 `patchscope_lens` 调用以获取 logits，
  要么对 probs 取 log 得到 log-probs，baseline 也取 log-probs，
  然后做减法。关键是两条路径必须计算同一个量
- `_get_baseline_logits_nnterp` 的命名也要改 — 如果返回的是 probs
  就不应该叫 `_logits`
- 确认 per-sample 路径的 `_get_baseline_logits_manual` 返回的确实是
  logits（当前实现是对的）

---

### Fix 5: CSD per-sample 路径 dtype 硬编码 (BUG-3)

**问题**: `csd.py:199` 硬编码 `dtype=torch.float16`，但 cache 存的是
float32，模型可能是 bfloat16 或 float32。

**要求**:
- 从模型参数推导 dtype：`next(model.parameters()).dtype`
- 不要硬编码任何 dtype

---

### Fix 6: CSD per-sample 路径硬编码 model.model.layers (BUG-4)

**问题**: `model.model.layers[layer_idx]` 假定了特定架构结构。

**要求**:
- 如果 model 是 nnterp 的 StandardizedTransformer（检查 `hasattr(model, "layers")`），
  用 `model.layers[layer_idx]` 访问
- 否则尝试常见路径：`model.model.layers`，并加上明确的错误信息
  指导用户使用 nnterp 包装模型
- 这是一个 pragmatic fix，完美方案是所有 CSD 走 nnterp batch 路径
```

---

## 通用要求

```
所有修改的通用要求：

1. 每个 fix 改完后跑 `cd Brewing && python -m pytest tests/ -v` 确认
   现有测试不 break
2. 不要添加新的 paper-specific 硬编码
3. 修改后在 `docs_dev_brewing/` 中更新或新建对应的开发日志
4. 不要做 review 文档里没提到的重构或"顺便改进"
```
