# CUE-Bench: Task Difficulty Design

## Overview

6 个任务，每个任务有 **3 个正交难度维度**，每个维度 **3 个取值**，共 3×3×3 = **27 种配置**。每种配置生成 150 个样本，单任务 4,050 样本，全 benchmark 共 24,300 样本。

所有任务答案均为单 digit (0–9)。

---

## 1. Value Tracking (Data Flow)

**核心能力**：追踪一个值在函数调用 / 容器包装 / 方法链中的**无损传递**——值本身不变，但经过多层间接引用。

| 维度 | 取值 | 含义 |
|------|------|------|
| **mechanism** | `function_chain` | 值通过 N 层嵌套函数原样传递（middleware 风格） |
| | `container` | 值被装入 list/dict/嵌套容器，再取出 |
| | `method_chain` | 类的 builder pattern，`.method().method()` 链式调用 |
| **depth** | `1, 2, 3` | 间接层数（函数 / 容器 / 方法链的嵌套深度） |
| **distractors** | `0, 1, 2` | 每层函数/方法中无关参数的数量 |

**难度梯度示例**：

```python
# EASY: mechanism=function_chain, depth=1, distractors=0
def extract(c):
    return c

result = extract(7)
# → 7
```

```python
# HARD: mechanism=method_chain, depth=3, distractors=2
class Query:
    def __init__(self, dst, force=None, timeout=None):
        self.dst = dst
    def update(self, retries=0, verbose=0):
        self.idx = 9; return self
    def rotate(self, tag=0, label=0):
        self.signal = 2; return self
    def pad(self, encoding=0, quiet=0):
        self.state = 0; return self
    def filter(self):
        return self.dst

config = Query(4, 7, 9).update(0, 4).rotate(7, 4).pad(8, 8).filter()
# → 4
```

---

## 2. Computing (Data Flow)

**核心能力**：执行多步算术运算，追踪中间计算结果。

| 维度 | 取值 | 含义 |
|------|------|------|
| **structure** | `func_arithmetic` | 单函数内多步算术 |
| | `chained_calls` | 嵌套函数调用 `combine(combine(v0, v1), v2)` |
| | `accumulator` | 循环聚合（计数/求和） |
| **steps** | `2, 3, 4` | 运算步数 |
| **operators** | `add` | 仅加法 |
| | `add_sub` | 加减混合 |
| | `add_mul` | 加乘混合（乘数受限以保证答案在 0–9） |

**难度梯度示例**：

```python
# EASY: structure=func_arithmetic, steps=2, operators=add
def process(dst, a, left):
    tmp = dst + a
    return tmp + left

payload = process(0, 5, 4)
# → 9
```

```python
# HARD: structure=accumulator, steps=4, operators=add_mul
def apply_op(n, y):
    score = 1
    for src in n:
        if src > y:
            score = score * src
        else:
            score = score + 1
    return score

tmp = apply_op([2, 3, 2, 0, 1], 2)
# → 9
```

---

## 3. Conditional (Control Flow)

**核心能力**：判断条件分支走向，追踪哪个 branch 会被执行。

| 维度 | 取值 | 含义 |
|------|------|------|
| **branch_type** | `elif_chain` | if/elif/elif.../else 多路分发 |
| | `guard_clause` | early return 守卫 + 最终 fallthrough |
| | `sequential_if` | 顺序 if 语句，中间有状态变异 |
| **depth** | `1, 2, 3` | 分支/守卫层数 |
| **condition_type** | `numeric` | 数值阈值比较 (`>=`, `>`, `<`) |
| | `membership` | 集合成员检查 (`in [...]`) |
| | `boolean_flag` | 布尔标志 (`if flag`, `if not flag`) |

**难度梯度示例**：

```python
# EASY: branch_type=elif_chain, depth=1, condition_type=numeric
def build(k):
    if k >= 96:
        return 6
    elif k >= 66:
        return 5
    else:
        return 7

score = build(128)
# → 6
```

```python
# HARD: branch_type=sequential_if, depth=3, condition_type=boolean_flag
def absorb(suffix, mode, priority):
    ans = 0
    if not suffix:
        ans = ans + 1
        mode = not mode
    if mode:
        ans = ans + 1
        priority = not priority
    if not priority:
        ans = ans + 1
    return ans

msg = absorb(True, True, False)
# → 1
```

**`sequential_if` 的独特难度**：每个 if 不仅改变返回值，还会翻转后续条件用到的 flag，形成状态依赖链。

---

## 4. Function Call (Control Flow)

**核心能力**：追踪值穿过函数边界时的**有损变换**——每层函数都对值做计算。

与 value_tracking 形成对照组：value_tracking 传值不变，function_call 传值有变换。

| 维度 | 取值 | 含义 |
|------|------|------|
| **mechanism** | `arithmetic` | 每层函数做简单算术 (+/−) |
| | `container_relay` | 值经容器传递，每跳有计算 |
| | `conditional_return` | 函数内含条件分支决定返回路径 |
| **depth** | `1, 2, 3` | 函数嵌套深度 |
| **distractors** | `0, 1, 2` | 每层函数中无关参数的数量 |

**设计意图**：depth 和 distractors 与 value_tracking 完全对齐，唯一区别是 mechanism 从"纯传递"变为"带计算的传递"，从而隔离"跨函数计算"这一因素的认知成本。

---

## 5. Loop (Data + Control)

**核心能力**：模拟循环执行，追踪循环变量的多次更新。

| 维度 | 取值 | 含义 |
|------|------|------|
| **body_type** | `simple_acc` | for-range 累加器 (`sum(range(n))`) |
| | `filter_count` | for-each + 条件计数 |
| | `dual_var` | 每次迭代更新两个变量（类 Fibonacci） |
| **iterations** | `2, 3, 4` | 循环迭代次数 |
| **init_offset** | `"0"` | 初始值 = 0 |
| | `"low"` | 初始值 ∈ [1, 2] |
| | `"high"` | 初始值 ∈ [3, 4] |

**难度梯度示例**：

```python
# Loop: body_type=simple_acc, iterations=3, init_offset=0
def transform(k):
    record = 0
    for i in range(k):
        record = record + i
    return record

target = transform(3)
# → 3
```

---

## 6. Loop Unrolled (Data + Control)

**与 Loop 完全相同的逻辑**，但循环被展开为顺序语句。

```python
# Loop Unrolled: body_type=simple_acc, iterations=3, init_offset=0
def extract():
    offset = 0
    offset = offset + 0
    offset = offset + 1
    offset = offset + 2
    return offset

value = extract()
# → 3
```

**设计意图**：Loop vs Loop_unrolled 形成对照组，隔离**循环语法本身**（循环变量、range、终止条件）的认知成本。如果模型在 loop 上表现差而 loop_unrolled 上好，说明难点在循环结构而非计算本身。

---

## 对照组设计

| 对照组 | 任务A | 任务B | 隔离的因素 |
|--------|-------|-------|-----------|
| **Data Flow** | value_tracking | function_call | 跨函数传递时是否有计算 |
| **Data + Control** | loop | loop_unrolled | 循环语法 vs 顺序展开 |

---

## 难度维度的设计哲学

每个任务的 3 个维度扮演不同角色：

| 角色 | 作用 | 各任务对应维度 |
|------|------|---------------|
| **结构维度** | 控制代码的组织形式（质变） | mechanism / structure / branch_type / body_type |
| **深度维度** | 控制嵌套/步数（量变） | depth / steps / iterations |
| **噪声维度** | 增加干扰信息或复杂度 | distractors / operators / condition_type / init_offset |

三个维度正交设计，可以独立分析每个维度对模型内部表示的影响。
