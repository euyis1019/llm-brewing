# 3. Code Review: Brewing Framework

**Date**: 2026-03-27
**Scope**: `Brewing/brewing/` 全部核心模块 + `Brewing/tests/`
**Review against**: `docs/workflow_and_schema.md`, `docs/architecture.md`, `COLM_REQUIREMENTS.md`

---

## Findings（按严重程度排序）

---

### BUG-1: Train/Eval 数据未分离 — Probing 用 eval set 同时训练和评估

**严重程度**: **明确 Bug** — 直接影响实验可复现性和论文结果的科学有效性

**位置**: `orchestrator.py:316-319`

```python
# For the default case, use the eval dataset as train dataset
# (with train/test split handled inside the method)
# In production, a separate train dataset should be specified
train_samples = eval_samples
train_cache = eval_cache
```

**问题**: Orchestrator 将 eval_samples/eval_cache 直接传给 LinearProbing 作为 train data。而 `LinearProbing._fit_probes()` (linear_probing.py:189-190) 用**全部** `train_cache.hidden_states` 训练 — 没有做任何 train/test split。评估时 (linear_probing.py:121-149) 又在同一批数据上跑 predict。

这意味着: probe 在训练数据上评估自己，FPCL/FJC 全部基于过拟合的 probe 输出计算。

**docs/workflow_and_schema.md §3.3** 明确写道:
> "训练数据和推理数据是**独立的**" / "Probing 的训练集是单独构造的，推理集与 CSD 共享"

代码注释说 "train/test split handled inside the method"，但方法内部根本没有做 split。

**修复方向**: 要么在 orchestrator 构建真正独立的 train dataset（走 S2b.1-S2b.2），要么在 `_fit_probes` 内部强制 split 并只在 split 后的 test portion 做 eval。前者是 schema 设计意图，后者是 pragmatic shortcut。

---

### BUG-2: CSD batch 路径中 baseline subtraction 在概率空间做减法 — 语义错误

**严重程度**: **明确 Bug** — 影响 CSD 所有 layer_values 的正确性

**位置**: `csd.py:137-144` (batch path)

```python
answer_logits = np.array([
    probs_np[layer_idx, tid] for tid in answer_token_ids
])
# Note: patchscope_lens returns probs, not logits
# Baseline subtraction on log scale
answer_log = np.log(answer_logits + 1e-10)
baseline_log = np.log(baseline_logits + 1e-10)
adjusted = answer_log - baseline_log
```

**问题**: `patchscope_lens` 返回的是概率 (probs)，而 `_get_baseline_logits_nnterp()` (csd.py:256-260) 取的也是概率 (`model.next_token_probs`)。代码在 log 空间做减法 → 实际是在算概率的**比值** (log-ratio)，不是论文定义的 "patched_logits - z_b"（logit 空间减法）。

对比 fallback 路径 `_run_per_sample` (csd.py:227-228):
```python
answer_logits = np.array([logits[tid] for tid in answer_token_ids])
adjusted = answer_logits - baseline_logits
```
fallback 路径用的是真正的 logits，做的是正确的 logit 空间减法。

**两条路径的 baseline subtraction 语义不一致**。Batch 路径的结果和 per-sample 路径的结果不可比较。

**修复方向**: batch 路径要从 patchscope_lens 的输出中提取 logits（而非 probs），或者在 nnterp 基座上直接拿 logits 做减法。

---

### BUG-3: CSD per-sample 路径 dtype 硬编码为 float16 — 潜在精度错误

**严重程度**: **高风险 Bug**

**位置**: `csd.py:199-200`

```python
h = torch.tensor(
    h_all[layer_idx], dtype=torch.float16
).to(next(model.parameters()).device)
```

**问题**: hidden states 在 cache 中以 float32 存储，这里强制转为 float16 再注入。如果模型是以 float32 或 bfloat16 加载的，dtype 不匹配会导致 silent numerical divergence。`cache_builder.py:134` 总是以 float32 存储: `dtype=np.float32`。

**修复**: dtype 应从 model 的参数 dtype 推导，而非硬编码。

---

### BUG-4: CSD per-sample 路径假定 `model.model.layers` 结构

**严重程度**: **高风险 Bug** — 影响跨架构验证

**位置**: `csd.py:217`

```python
hook = model.model.layers[layer_idx].register_forward_hook(
    make_hook(h, patch_pos)
)
```

**问题**: 硬编码 `model.model.layers[layer_idx]` 只适用于 LlamaForCausalLM 架构。Qwen2 使用相同路径，但 DeepSeek-Coder 可能不同。这正是 nnterp 的 `StandardizedTransformer` 要屏蔽的差异。

论文要求跨架构验证（DeepSeek-Coder-6.7B, CodeLlama-7B, Llama-3.1-8B），这条路径会在非标准架构上直接 crash。

---

### DESIGN-1: `compute_csd_tail_confidence` 语义与文档不一致

**严重程度**: **高风险设计问题** — 直接影响 Misresolved vs Unresolved 分类

**位置**: `diagnostics.py:60-77`

**文档定义** (workflow_and_schema.md §2.4.1):
> `csd_tail_confidence`: CSD 在尾部窗口 (>= 3L/4) 的最大平均概率

**代码实现** (当 `layer_confidences` 存在时):
```python
tail_confs = csd_result.layer_confidences[tail_start:]  # (tail_len, C)
mean_per_class = tail_confs.mean(axis=0)  # (C,)
return float(np.max(mean_per_class))
```

这算的是: 对每个 class 在尾部各层取平均，然后取 max class。这衡量的是"是否有某个 class 在尾部持续获得高概率" — 即**稳定收敛到某个答案**。

**但当 `layer_confidences` 为 None 时** (fallback):
```python
tail_vals = csd_result.layer_values[tail_start:]
return float(np.max(tail_vals))
```

`layer_values` 是 correctness flag (0 或 1)，取 max 就是"尾部是否有任何一层 CSD 正确过"。这和 confidence 语义完全不同。

CSD 的 `layer_values` 是 correctness boolean，不是 confidence。Fallback 路径会导致: 只要尾部任何一层碰巧 CSD 正确，就返回 1.0 → 错误分类为 Misresolved。

**影响**: 在 e2e 测试中 CSD 没有 `layer_confidences`（只设了 `layer_values`），这个 fallback 实际上被触发了，但因为测试设计碰巧跳过了这个边界，所以没有 fail。

---

### DESIGN-2: Orchestrator 中 `eval_dataset_id` 在所有 subset 共享

**严重程度**: **高风险设计问题**

**位置**: `orchestrator.py:119-121`

```python
eval_dataset_id = (
    self.config.eval_dataset_id or
    f"cue-{subset_name}-eval-seed{self.config.seed}"
)
```

当用户显式设定 `config.eval_dataset_id = "my-ds"` 时，所有 subset 共享同一个 `eval_dataset_id`。第一个 subset 会创建/加载 `"my-ds"` 并存入 resources；后续 subset 会从 disk 加载同一个 `"my-ds"` — 即**所有 subset 用了第一个 subset 的数据**。

只有 `eval_dataset_id=None`（auto-generate）时才安全，因为会包含 subset_name。

---

### DESIGN-3: `SampleMethodResult.from_dict` 的 `layer_predictions` 反序列化逻辑脆弱

**严重程度**: **高风险设计问题**

**位置**: `data.py:294-310`

```python
layer_predictions=(
    d.get("layer_predictions")
    if isinstance(d.get("layer_predictions"), list) and
       any(isinstance(x, str) for x in d.get("layer_predictions", []))
    else np.array(d["layer_predictions"]) if d.get("layer_predictions") is not None
    else None
),
```

**问题**: 判断是保留 `list[str]` 还是转为 `np.ndarray` 的逻辑是: 检查列表里是否有 str 元素。但如果所有 predictions 恰好是纯数字字符串如 `["0", "1", "2"]`，JSON 反序列化后仍然是 `list[str]`，检测通过。可如果原始数据是 numeric labels 存为 `[0, 1, 2]`，则转为 ndarray。这个判断在跨序列化边界时不稳定。

更关键的是，这个嵌套三元表达式很难 debug — 如果 `layer_predictions` 中混入了 None 或非 str/int 值，行为难以预测。

---

### DESIGN-4: Linear Probing 的 11-class / residual class 硬编码在方法内部

**严重程度**: **中等设计问题** — paper setting 污染框架抽象

**位置**: `linear_probing.py:39-42`

```python
DIGIT_CLASSES = [str(d) for d in range(10)]
RESIDUAL_CLASS = "__residual__"
ALL_CLASSES = DIGIT_CLASSES + [RESIDUAL_CLASS]
N_CLASSES = len(ALL_CLASSES)
```

**问题**: 11-class (0-9 + residual) 是 CUE-Bench 的 setting，不是 Linear Probing 这个方法的固有属性。`_encode_labels` 函数实际上是泛化的（接受 `answer_space` 参数），但模块级常量 `DIGIT_CLASSES` / `N_CLASSES` 给人强烈的"这个方法只适用于 CUE-Bench"的印象。

当前影响有限（论文只用 CUE-Bench），但 `N_CLASSES` 被写入 artifact metadata，如果有人换了 answer_space 但没更新常量，metadata 会不一致。

**判定**: 可接受的 pragmatic shortcut，但应确保 `N_CLASSES` 从实际 `answer_space` 推导而非硬编码常量。

---

### DESIGN-5: `make_synthetic_cache` 的 `model_predictions = answers` — 合成 cache 总是全对

**严重程度**: **中等设计问题** — 影响测试有效性

**位置**: `cache_builder.py:243`

```python
model_predictions=answers,  # synthetic: predictions = answers (all correct)
```

**问题**: 合成 cache 的 model_predictions 和 answers 完全一致，意味着所有测试中 outcome 只能是 Resolved（有 FJC 时）或 Unresolved/Misresolved（无 FJC 时），永远不会出现 Overprocessed。

这导致 Overprocessed outcome 的整条代码路径在所有自动化测试中都没有被真正覆盖。

---

### DESIGN-6: CSD batch 路径逐样本循环 — 没有真正利用 batch

**严重程度**: **低优先级性能问题**

**位置**: `csd.py:114`

```python
for i, sample in enumerate(eval_samples):
```

即使调用了 `patchscope_lens`，也是逐样本调用。如果 eval_samples 有几百个样本 × 28-48 层，这会非常慢。应该考虑把多个样本的 latents 拼起来一次调用。

---

## 测试覆盖缺口

### GAP-1: 没有测试 train ≠ eval 的场景
所有 probing 测试都用 `train_samples=samples, train_cache=cache`。workflow_and_schema.md 的核心设计 — 独立的训练集和评估集 — 没有被测试过。

### GAP-2: CSD 完全没有单元测试
`test_e2e.py` 中 CSD result 是手工构造的 `SampleMethodResult`，没有测试过 `CSD.run()` 的任何实际路径。

### GAP-3: Outcome = Overprocessed 从未在测试中出现
因为 synthetic cache 的 model_predictions = answers（见 DESIGN-5），Overprocessed 不可能出现。应构造 model_predictions ≠ answers 的 case。

### GAP-4: `group_diagnostics_by_difficulty` 没有测试
这个函数会被论文分析高频使用，但完全没有测试覆盖。

### GAP-5: 序列化 round-trip 没有覆盖 edge cases
`SampleMethodResult.from_dict` 的脆弱逻辑（DESIGN-3）没有针对 edge cases 的测试（如 empty layer_predictions, mixed types, None confidences）。

### GAP-6: `ResourceManager.resolve_artifact_with_policy` 没有测试 artifact_id 不匹配的场景
如果用户改了 probe_params 但 artifact_id 相同，现有逻辑会加载旧的不匹配的 artifact。没有 integrity check。

---

## Schema / Workflow 一致性检查

| 检查项 | 状态 | 备注 |
|--------|------|------|
| S3 只消费 MethodResult | **通过** | `diagnostics.py` 输入签名正确 |
| Training 在 S2b 完成 | **通过** | 训练逻辑在 `LinearProbing.run()` 内 |
| Train/Eval 独立数据 | **不通过** | 见 BUG-1 |
| DatasetManifest 字段一致 | **通过** | |
| HiddenStateCache 字段一致 | **通过** | |
| FitArtifact 字段一致 | **通过** | |
| MethodResult 字段一致 | **通过** | |
| DiagnosticResult 字段一致 | **通过** | |
| fit_policy 三种策略 | **通过** | 实现与文档一致 |
| Outcome 判定逻辑 | **有偏差** | 见 DESIGN-1，fallback 路径语义错误 |
| FPCL/FJC 计算 | **通过** | 与文档一致 |

---

## Paper Setting 硬编码评估

| 硬编码位置 | 内容 | 可接受性 |
|-----------|------|---------|
| `cue_bench.py` 全文 | 6 个 task 定义、fixture samples | **正当** — 这就是 benchmark 定义模块 |
| `linear_probing.py:32-42` | DIGIT_CLASSES, N_CLASSES=11 | **可接受** — 只是默认值，实际从 config 读 answer_space |
| `csd.py:33` | DEFAULT_TARGET_PROMPT | **可接受** — 用作默认值，可被 config 覆盖 |
| `orchestrator.py:37-38` | 默认 benchmark="CUE-Bench", model_id | **可接受** — CLI 默认值 |
| `orchestrator.py:267` | `n_layers=28` in synthetic cache | **不可接受** — 如果用户指定了非 Qwen2.5-Coder-7B 模型，合成 cache 的层数是错的 |
| `cli.py:132-133` | `has_model_online` 只检查 "csd" | **不可接受** — 如果未来加其他 model-online 方法，这里不会加载模型 |

---

## 总结

### 必须修复的 Bugs（影响实验正确性）

1. **BUG-1**: Train = Eval 数据泄漏 — Probing 在训练集上评估自己
2. **BUG-2**: CSD batch 路径 baseline subtraction 在 prob 空间操作，与 per-sample 路径不一致
3. **BUG-3**: CSD per-sample 路径 dtype 硬编码 float16

### 必须修复的设计问题（影响正确性或科学有效性）

4. **DESIGN-1**: `csd_tail_confidence` fallback 路径语义错误，影响 Misresolved/Unresolved 分类
5. **DESIGN-2**: 用户指定 eval_dataset_id 时所有 subset 用同一份数据

### 高价值测试缺口

6. GAP-1 + GAP-2 + GAP-3 是最关键的三个：独立 train/eval、CSD 单元测试、Overprocessed outcome 覆盖

### 当前可接受的 Pragmatic Shortcuts

- Paper-specific 常量作为默认值（可被 config 覆盖）
- 6 个 task 的 difficulty_schema 硬编码在 cue_bench.py（这本来就是 benchmark 定义）
- 合成 cache 用于 smoke test（但需要补充更真实的测试数据）
