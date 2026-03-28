# CUE-Bench 数据集

CUE-Bench 包含 6 个代码推理任务，每个任务 4050 个样本，共 24300 个样本。模型需要通过心算执行代码来判断某个变量的最终值（单 digit，0-9）。

---

## 任务与维度

每个任务 3 个正交维度，每个维度 3 个取值，共 27 种配置 × 150 samples/config = 4050 samples。

| 任务 | 类别 | dim1 | dim2 | dim3 |
|------|------|------|------|------|
| **value_tracking** | Data Flow | mechanism: function_chain, container, method_chain | depth: 1, 2, 3 | distractors: 0, 1, 2 |
| **computing** | Data Flow | structure: func_arithmetic, inline_chain, class_method | steps: 2, 3, 4 | operators: add_sub, add_mul, mixed |
| **conditional** | Control Flow | branch_type: if_else, nested_if, elif_chain | depth: 1, 2, 3 | condition_type: numeric, membership, boolean_flag |
| **function_call** | Control Flow | mechanism: arithmetic, container_relay, conditional_return | depth: 1, 2, 3 | distractors: 0, 1, 2 |
| **loop** | Data+Control | body_type: simple_acc, filter_count, dual_var | iterations: 2, 3, 4 | init_offset: "0", "low", "high" |
| **loop_unrolled** | Data+Control | body_type: simple_acc, filter_count, dual_var | iterations: 2, 3, 4 | init_offset: "0", "low", "high" |

**对照组设计**：
- **value_tracking vs function_call** — 值传递 ± 函数内计算
- **loop vs loop_unrolled** — 相同计算 ± 循环语法

---

## 数据量与切分

| 任务 | train | eval | 总计 |
|------|-------|------|------|
| value_tracking | 3240 | 810 | 4050 |
| computing | 3240 | 810 | 4050 |
| conditional | 3240 | 810 | 4050 |
| function_call | 3240 | 810 | 4050 |
| loop | 3240 | 810 | 4050 |
| loop_unrolled | 3240 | 810 | 4050 |
| **合计** | **19440** | **4860** | **24300** |

4:1 均匀随机切分（seed=42），每个任务独立切分。train 用于 probe 训练，eval 用于诊断分析。

---

## 数据位置

```
Brewing/
├── brewing/benchmarks/cue_bench/
│   ├── data/                           # datagen 原始输出
│   │   ├── train/{task}.json           # 3240 samples/task
│   │   └── eval/{task}.json            #  810 samples/task
│   └── datagen/                        # 生成器代码
│
└── brewing_output/datasets/cuebench/   # pipeline 落盘（Brewing Sample 格式）
    ├── train/{task}/seed42/
    │   ├── manifest.json
    │   └── samples.json
    └── eval/{task}/seed42/
        ├── manifest.json
        └── samples.json
```

两份数据是**同一批样本的不同形态**：

- `cue_bench/data/` — datagen 直接输出，保留 `code` 字段，维度在 `metadata` 里
- `brewing_output/` — 经 adapter 转换的 `Sample` 格式，维度拆到 `difficulty` 字段，附 manifest

---

## 数据格式

### datagen 原始格式

```json
{
  "id": "comp_accumulator_s2_a_000",
  "prompt": "def extract(x, lo):\n    ...\n# The value of ret is \"",
  "code": "def extract(x, lo):\n    ...\nret = extract([9, 8, 3], 5)",
  "answer": "1",
  "metadata": {
    "structure": "accumulator",
    "steps": 2,
    "operators": "add_sub",
    "result_var": "ret",
    "sample_idx": 0
  }
}
```

### Brewing Sample 格式

```json
{
  "id": "comp_accumulator_s2_a_000",
  "benchmark": "CUE-Bench",
  "subset": "computing",
  "prompt": "def extract(x, lo):\n    ...\n# The value of ret is \"",
  "answer": "1",
  "difficulty": {
    "structure": "accumulator",
    "steps": 2,
    "operators": "add_sub"
  },
  "metadata": {
    "result_var": "ret",
    "sample_idx": 0
  }
}
```

转换逻辑（`adapter.py`）：维度字段从 `metadata` 拆到 `difficulty`，丢弃 `code`（`prompt` 已含完整代码 + question suffix），添加 `benchmark`/`subset` 标注。

---

## 代码示例

### value_tracking

```python
def merge_val(inp):
    return [inp, 1]

def emit(x):
    return x[0]

token = merge_val(0)
result = emit(token)
# The value of result is "
```

### computing

```python
def extract(x, lo):
    """Aggregate partial results."""
    payload = 0
    for hi in x:
        if hi > lo:
            payload = payload + 1
        elif hi < lo:
            payload = payload - 1
    return payload

ret = extract([9, 8, 3], 5)
# The value of ret is "
```

### function_call

```python
signal = 6

def merge_val(c):
    return c + 3

status = merge_val(signal)
# The value of status is "
```

### loop

```python
def unwrap(c):
    p = 1
    x = 0
    for i in range(c):
        p, x = p + 1, x + p
    return p

ans = unwrap(2)
# The value of ans is "
```

---

## 与 Legacy 数据的差异

Legacy 数据位于 `legacy/{task}/data/dataset.json`，已归档不再使用。

| | Legacy | 当前 (v2) |
|---|--------|-----------|
| **任务** | 5 个（含 short_circuit） | 6 个（short_circuit → function_call） |
| **样本量** | 不统一（180-360/task） | 统一 4050/task |
| **train/eval** | 无切分 | 4:1 切分 |
| **代码风格** | 简单赋值链（`a = 3; b = a`） | 函数/类/容器封装，带 distractor |
| **标识符** | 硬编码（`a`, `b`, `result`） | NamePool 随机化 |
| **维度设计** | 各任务不同，部分耦合 | 全部 3×3×3 正交 |
| **格式** | 各任务自定义 | 统一 schema |
| **exec 验证** | 部分 | 全部 24300 样本通过 |

**short_circuit → function_call**：short_circuit 测试条件短路，与 conditional 重叠；function_call 测试值跨函数边界的传递与变换，与 value_tracking 构成对照组。

---

## 生成参数

| 参数 | 值 |
|------|-----|
| seed | 42 |
| samples_per_config | 150 |
| 答案范围 | 0-9（单 digit） |
| retry 上限 | 200 次/sample |
| 生成器 | `Brewing/brewing/benchmarks/cue_bench/datagen/{task}.py` |
| 入口 | `python -m brewing.benchmarks.cue_bench.datagen.generate` |
