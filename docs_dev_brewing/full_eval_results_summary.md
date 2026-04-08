# Full Eval Results Summary

> 2026-03-31 更新. 16 models × 6 tasks. Coder-14B 为 n=270 (1/3 子集)，其余全部 n=810 全量。
> Llama2/CodeLlama diagnostics 已用更新后的 CSD 结果重新生成。
>
> **NO_BREWING 修正** (2026-03-30): FPCL=None (probing 全层未正确) 的 sample 不再计入四类 outcome 分布，标记为 NO_BREWING。四类 outcome 百分比的分母排除 NB sample。Resolved% 等数值因此略有变化。
>
> **全量更新** (2026-03-31): Qwen3-8B-Base, Qwen2.5-1.5B, Qwen2.5-7B, Qwen3-1.7B-Base 从 n=81 升级到 n=810。Coder-14B 从 n=81 升级到 n=270。Qwen3-8B 的 function_call resolved 从 86.4% 降至 24.7%（n=81 噪声已消除），Coder-14B function_call 从 85.2% 降至 48.7%。

## 1. Raw Metrics

### TABLE 1: Resolved% — All Models × All Tasks

| Group | Model | N | value_tr | computing | conditional | func_call | loop | loop_unroll | avg |
|-------|-------|---|----------|-----------|-------------|-----------|------|-------------|-----|
| **Qwen2.5-Coder** | 0.5B | 810 | 62.8% | 8.4% | 16.8% | 7.5% | 8.6% | 3.7% | **18.9%** |
| | 1.5B | 810 | 74.8% | 13.4% | 34.9% | 14.6% | 6.8% | 6.4% | **25.7%** |
| | 3B | 810 | 78.9% | 21.3% | 54.9% | 18.6% | 25.2% | 15.5% | **36.2%** |
| | 7B | 810 | 70.8% | 26.2% | 59.2% | 27.7% | 35.5% | 28.0% | **41.5%** |
| | 14B | 270 | 77.3% | 42.0% | 74.0% | 48.7% | 26.4% | 33.5% | **50.3%** |
| **Qwen2.5-Base** | 0.5B | 810 | 57.9% | 10.4% | 20.1% | 7.0% | 9.3% | 9.7% | **19.9%** |
| | 1.5B | 810 | 60.0% | 8.2% | 25.0% | 11.8% | 6.8% | 7.3% | **19.9%** |
| | 3B | 810 | 77.9% | 18.4% | 46.5% | 13.8% | 14.8% | 10.3% | **30.9%** |
| | 7B | 810 | 80.6% | 21.5% | 56.5% | 19.3% | 17.9% | 21.3% | **36.2%** |
| **Qwen3-Base** | 0.6B | 810 | 67.9% | 12.2% | 32.5% | 10.4% | 11.1% | 14.6% | **25.5%** |
| | 1.7B | 810 | 67.5% | 13.7% | 41.9% | 12.7% | 9.3% | 15.1% | **26.7%** |
| | 4B | 810 | 83.3% | 28.6% | 58.2% | 28.1% | 18.6% | 26.3% | **40.7%** |
| | 8B | 810 | 82.0% | 28.2% | 66.7% | 24.7% | 33.2% | 29.2% | **44.0%** |
| **Other-7B** | DS-6.7B | 810 | 76.5% | 21.3% | 54.5% | 19.0% | 27.5% | 29.3% | **38.4%** |
| | Llama2-7B | 810 | 71.7% | 8.3% | 23.1% | 7.1% | 8.1% | 7.3% | **21.6%** |
| | CodeLlama-7B | 810 | 75.2% | 7.6% | 35.9% | 7.9% | 8.6% | 6.5% | **24.1%** |

### TABLE 2: Qwen2.5-Coder-7B — Detailed Outcome Distribution (典型模型)

ΔBrew coverage = Resolved% + Overprocessed%，即有 FJC 定义的 sample 占比（排除 NB 后）。ΔBrew 仅在这些 sample 上计算。NB = NO_BREWING (FPCL=None)。四类 outcome 百分比排除 NB sample。

| Task | N | NB | Resolved | Overprocessed | Misresolved | Unresolved | FPCL_n | FJC_n | ΔBrew | ΔBrew_med | ΔBrew cov |
|------|---|----|----------|---------------|-------------|------------|--------|-------|-------|-----------|-----------|
| value_tracking | 810 | 1 | **70.8%** | 13.8% | 5.4% | 9.9% | 0.074 | 0.555 | 13.73 | 18.0 | 84.7% |
| computing | 810 | 35 | 26.2% | **35.6%** | 11.5% | 26.7% | 0.179 | 0.469 | 9.17 | 8.0 | 61.8% |
| conditional | 810 | 20 | **59.2%** | 22.7% | 10.1% | 8.0% | 0.163 | 0.560 | 11.36 | 11.0 | 82.0% |
| function_call | 810 | 48 | 27.7% | 28.9% | 3.8% | **39.6%** | 0.179 | 0.515 | 10.22 | 9.0 | 56.5% |
| loop | 810 | 13 | 35.5% | **31.1%** | 9.5% | 23.8% | 0.102 | 0.429 | 9.66 | 7.0 | 66.6% |
| loop_unrolled | 810 | 13 | 28.0% | 26.7% | 10.4% | **34.9%** | 0.148 | 0.486 | 9.81 | 8.0 | 54.7% |

### TABLE 3: Mean FPCL (normalized) — lower = earlier information availability

| Group | Model | value_tr | computing | conditional | func_call | loop | loop_unroll | avg |
|-------|-------|----------|-----------|-------------|-----------|------|-------------|-----|
| **Qwen2.5-Coder** | 0.5B | 0.189 | 0.240 | 0.226 | 0.244 | 0.164 | 0.217 | 0.214 |
| | 1.5B | 0.135 | 0.239 | 0.211 | 0.225 | 0.153 | 0.189 | 0.192 |
| | 3B | 0.089 | 0.221 | 0.199 | 0.204 | 0.137 | 0.183 | 0.172 |
| | 7B | 0.074 | 0.179 | 0.163 | 0.179 | 0.102 | 0.148 | 0.141 |
| | 14B | 0.067 | 0.158 | 0.176 | 0.156 | 0.087 | 0.075 | 0.120 |
| **Qwen2.5-Base** | 0.5B | 0.177 | 0.235 | 0.211 | 0.219 | 0.152 | 0.207 | 0.200 |
| | 1.5B | 0.139 | 0.219 | 0.213 | 0.219 | 0.154 | 0.208 | 0.192 |
| | 3B | 0.095 | 0.214 | 0.205 | 0.212 | 0.130 | 0.200 | 0.176 |
| | 7B | 0.086 | 0.162 | 0.158 | 0.171 | 0.096 | 0.138 | 0.135 |
| **Qwen3-Base** | 0.6B | 0.134 | 0.212 | 0.207 | 0.220 | 0.142 | 0.182 | 0.183 |
| | 1.7B | 0.105 | 0.203 | 0.173 | 0.180 | 0.120 | 0.153 | 0.156 |
| | 4B | 0.058 | 0.164 | 0.146 | 0.165 | 0.104 | 0.147 | 0.131 |
| | 8B | 0.055 | 0.148 | 0.129 | 0.148 | 0.081 | 0.120 | 0.114 |
| **Other-7B** | DS-6.7B | 0.069 | 0.164 | 0.125 | 0.148 | 0.101 | 0.119 | 0.121 |
| | Llama2-7B | 0.056 | 0.131 | 0.103 | 0.130 | 0.080 | 0.095 | 0.099 |
| | CodeLlama-7B | 0.060 | 0.145 | 0.116 | 0.143 | 0.077 | 0.102 | 0.107 |

> Q2.5-Base 全系列 FPCL 数据已补全。

### TABLE 4: Mean ΔBrew — absolute layers and normalized (÷ n_layers)

不同模型层数不同（见 n_layers 列），**跨模型比较必须看 normalized 值**。ΔBrew 仅在 FJC≠null 的 sample 上计算（coverage = Resolved% + Overprocessed%），高 unresolved 任务的 ΔBrew 基于偏小的子集。

| Group | Model | n_layers | value_tr | computing | conditional | func_call | loop | loop_unroll | avg | **avg_norm** |
|-------|-------|----------|----------|-----------|-------------|-----------|------|-------------|-----|-------------|
| **Qwen2.5-Coder** | 0.5B | 24 | 9.40 | 4.97 | 7.56 | 7.09 | 7.13 | 5.33 | 6.91 | **0.288** |
| | 1.5B | 28 | 14.54 | 8.01 | 10.24 | 11.35 | 10.34 | 10.97 | 10.91 | **0.390** |
| | 3B | 36 | 18.20 | 12.58 | 14.15 | 12.43 | 11.10 | 11.45 | 13.32 | **0.370** |
| | 7B | 28 | 13.73 | 9.17 | 11.36 | 10.22 | 9.66 | 9.81 | 10.66 | **0.381** |
| | 14B | 48 | 18.48 | 15.23 | 19.32 | 15.60 | 13.74 | 16.44 | 16.47 | **0.343** |
| **Qwen2.5-Base** | 0.5B | 24 | 10.97 | 6.33 | 7.75 | 6.06 | 6.02 | 5.36 | 7.08 | **0.295** |
| | 1.5B | 28 | 13.14 | 6.91 | 9.35 | 10.81 | 12.54 | 11.32 | 10.68 | **0.381** |
| | 3B | 36 | 18.47 | 12.77 | 12.37 | 14.50 | 12.84 | 11.25 | 13.70 | **0.381** |
| | 7B | 28 | 14.36 | 11.09 | 11.98 | 11.56 | 10.85 | 9.83 | 11.61 | **0.415** |
| **Qwen3-Base** | 0.6B | 28 | 9.93 | 5.81 | 6.71 | 7.37 | 5.15 | 6.05 | 6.84 | **0.244** |
| | 1.7B | 28 | 10.76 | 7.68 | 9.08 | 7.68 | 11.20 | 9.39 | 9.30 | **0.332** |
| | 4B | 36 | 15.87 | 12.14 | 12.56 | 13.50 | 15.92 | 13.61 | 13.93 | **0.387** |
| | 8B | 36 | 15.33 | 12.73 | 11.97 | 13.60 | 12.76 | 11.78 | 13.03 | **0.362** |
| **Other-7B** | DS-6.7B | 32 | 10.88 | 9.60 | 10.03 | 8.74 | 8.52 | 8.72 | 9.42 | **0.294** |
| | Llama2-7B | 32 | 9.81 | 8.05 | 8.74 | 7.26 | 7.65 | 8.10 | 8.27 | **0.258** |
| | CodeLlama-7B | 32 | 11.12 | 6.74 | 8.76 | 9.38 | 11.51 | 10.72 | 9.70 | **0.303** |

> Q2.5-Base 全系列 ΔBrew 数据已补全。

---

## 2. Findings

### 2.1 Task Difficulty Hierarchy (Coder-7B 典型模型)

六个任务在 Coder-7B 上的 resolved% 排序：

```
value_tracking (70.8%) > conditional (59.2%) > loop (35.5%) > loop_unrolled (28.0%) ≈ function_call (27.7%) ≈ computing (26.2%)
```

**Outcome 分布揭示了不同的失败模式**（四类 outcome 百分比排除 NO_BREWING sample）：
- **value_tracking**: 主要成功（70.8% resolved），失败以 overprocessed 为主——信息曾经 ready 但被后续层破坏
- **computing**: overprocessed（35.6%）是最大类别，说明模型能提取信息但内部计算不稳定
- **function_call**: unresolved 最高（39.6%），信息从未同时在 probe 和 CSD 中可用——函数调用的间接寻址对模型最难
- **conditional**: 与 value_tracking 接近的高 resolved，低 unresolved（8.0%），分支跳转对 7B 模型相对可控
- **loop vs loop_unrolled**: loop 比 loop_unrolled 好（35.5% vs 28.0%），FPCL 也更低（0.102 vs 0.148）——**循环语法反而帮助模型更早提取信息**，展开后结构线索丧失

#### Contrast Pair 1: value_tracking vs function_call

两者都属于 Data Flow 类别，区别在于 function_call 有函数内计算。

| | value_tracking | function_call |
|---|---|---|
| Resolved | 70.8% | 27.7% |
| Unresolved | 9.9% | 39.6% |
| NO_BREWING | 1 (0.1%) | 48 (5.9%) |
| FPCL_n | 0.074 | 0.179 |
| ΔBrew | 13.73 | 10.22 |
| ΔBrew coverage | 84.7% | 56.5% |

函数调用不仅降低了 resolved 率（−43pp），还推迟了信息首次可读的层（FPCL 高 2.4×）。ΔBrew 反而更小，但这是选择偏差的结果：function_call 只有 56.5% 的 sample 有 FJC（能计算 ΔBrew），而 value_tracking 是 84.7%。能进入 brewing zone 的 sample 本身就是相对简单的子集，大量难 sample 因为 FJC=null 被排除在 ΔBrew 统计之外。

#### Contrast Pair 2: loop vs loop_unrolled

| | loop | loop_unrolled |
|---|---|---|
| Resolved | 35.5% | 28.0% |
| FPCL_n | 0.102 | 0.148 |
| ΔBrew | 9.66 | 9.81 |
| ΔBrew coverage | 66.6% | 54.7% |

循环语法的存在使得 FPCL 更低（信息更早可读），resolved 更高。ΔBrew 几乎一样（coverage 也接近）。这说明 for-loop 的语法结构给了模型额外的 structural cue，帮助它更早在内部表征中编码答案。展开后这些线索消失，模型需要更多层来处理。

### 2.2 Scaling (同架构不同大小)

#### Qwen2.5-Coder: 0.5B → 1.5B → 3B → 7B → 14B

注意：不同 size 的模型**层数不同**（24/28/36/28/48），ΔBrew 的跨 size 比较必须看 normalized 值。

| Metric | 0.5B (24L) | 1.5B (28L) | 3B (36L) | 7B (28L) | 14B† (48L) |
|--------|------------|------------|----------|----------|------------|
| avg resolved% | 18.0% | 25.2% | 35.7% | 41.2% | 50.3% |
| avg FPCL_n | 0.214 | 0.192 | 0.172 | 0.141 | 0.120 |
| avg ΔBrew (abs) | 6.91 | 10.91 | 13.32 | 10.66 | 16.47 |
| **avg ΔBrew (norm)** | **0.288** | **0.390** | **0.370** | **0.381** | **0.343** |

(†14B 为 n=270 子集)

**Scaling 趋势**：
- **Resolved% 单调递增**：模型越大，能成功解码的 sample 越多
- **FPCL 单调递减**：更大的模型在更早的层就使信息线性可读（information availability 更早）
- **Normalized ΔBrew 先增后趋稳**：0.5B 较低（0.288），1.5B 跳升至 0.390，之后 3B/7B 稳定在 ~0.37-0.38，14B 轻微下降至 0.327

    > 此前版本用绝对层数报告 ΔBrew，声称 7B 有"回落"——这是层数差异（3B=36L, 7B=28L）的假象。Normalized 后 1.5B/3B/7B 几乎相同。

Normalized ΔBrew 的解释：
- 小模型（0.5B）：probe 和 CSD 都很弱，差距小（~29% 的层用于 brewing）
- 中等及以上模型（1.5B-7B）：probe 能力快速增强（FPCL 降低），但 CSD 仍需约 37-39% 的层才能 ready → normalized ΔBrew 趋于稳定
- 14B 的轻微下降可能是 CSD 能力追上了 probe（n=270 验证了这一趋势并非 n=81 噪声）

#### Qwen3-Base: 0.6B → 1.7B → 4B → 8B

| Metric | 0.6B (28L) | 1.7B (28L) | 4B (36L) | 8B (36L) |
|--------|------------|------------|----------|----------|
| avg resolved% | 24.8% | 26.7% | 40.5% | 44.0% |
| avg FPCL_n | 0.183 | 0.156 | 0.131 | 0.114 |
| avg ΔBrew (abs) | 6.84 | 9.30 | 13.93 | 13.03 |
| **avg ΔBrew (norm)** | **0.244** | **0.332** | **0.387** | **0.362** |

Qwen3 的 scaling 趋势与 Coder 一致：resolved 上升，FPCL 下降，normalized ΔBrew 先增后趋稳。

### 2.3 Coder vs Base (Capability 对比)

#### 0.5B: Coder ≈ Base

| Task | Coder resolved | Base resolved | Coder FPCL | Base FPCL |
|------|---------------|---------------|------------|-----------|
| value_tracking | 62.8% | 57.9% | 0.189 | 0.177 |
| computing | 8.4% | 10.4% | 0.240 | 0.235 |
| conditional | 16.8% | 20.1% | 0.226 | 0.211 |
| function_call | 7.5% | 7.0% | 0.244 | 0.219 |
| loop | 8.6% | 9.3% | 0.164 | 0.152 |
| loop_unrolled | 3.7% | 9.7% | 0.217 | 0.207 |

**0.5B 级别 Coder 和 Base 差异不大**。Base 在某些任务上 FPCL 甚至更低——说明 0.5B 时 code-specific 预训练的收益还没充分体现。

#### 3B: Coder > Base

| Task | Coder resolved | Base resolved | Δ |
|------|---------------|---------------|---|
| value_tracking | 78.9% | 77.9% | +1.0pp |
| computing | 21.3% | 18.4% | +2.9pp |
| conditional | **54.9%** | **46.5%** | **+8.4pp** |
| function_call | 18.6% | 13.8% | +4.8pp |
| loop | **25.2%** | **14.8%** | **+10.4pp** |
| loop_unrolled | 15.5% | 10.3% | +5.2pp |

**3B 级别 Coder 优势明显**，特别是 control flow 任务（conditional +8.8pp, loop +10.1pp）。code-specific 预训练主要帮助模型处理程序控制结构。

### 2.4 Cross-Architecture (跨架构对比)

#### ~7B 级别横向比较

注意不同架构层数不同，normalized ΔBrew 比绝对值更适合跨架构比较。

| Model | Architecture | n_layers | avg resolved% | avg FPCL_n | avg ΔBrew (abs) | **avg ΔBrew (norm)** |
|-------|-------------|----------|---------------|------------|-----------------|---------------------|
| Coder-7B | Qwen2ForCausalLM | 28 | 41.2% | 0.141 | 10.66 | **0.381** |
| Q2.5-7B | Qwen2ForCausalLM | 28 | 36.2% | 0.135 | 11.61 | **0.415** |
| Qwen3-8B | Qwen3ForCausalLM | 36 | 44.0% | 0.114 | 13.03 | **0.362** |
| DS-6.7B | LlamaForCausalLM | 32 | 38.0% | 0.121 | 9.42 | **0.294** |
| Llama2-7B | LlamaForCausalLM | 32 | 20.9% | 0.099 | 8.27 | **0.258** |
| CodeLlama-7B | LlamaForCausalLM | 32 | 23.6% | 0.107 | 9.70 | **0.303** |

**关键发现**：

1. **Llama-2 和 CodeLlama 的 resolved% 显著低于同参数量模型**（22%/24% vs 37-59%），但 probing 表现正常（FPCL 是所有模型中最低的，信息极早线性可读）。主要瓶颈在 CSD：computing/function_call/loop 的 CSD max accuracy 仅 17-20%（Qwen2.5-7B 同任务 35-40%），导致超过一半 sample 的 FJC 未定义。这不是代码 bug，而是非 code-specialized 模型在代码推理任务上 CSD 能力本身就弱。

2. **DeepSeek-6.7B**（同为 LlamaForCausalLM）表现正常（38.4% resolved，norm ΔBrew=0.294），说明 code 预训练数据质量是关键，不是架构限制。

3. **Qwen3-8B 在 ~7B 级别表现最强**（44.0% resolved），比同参数量的 Coder-7B（41.2%）高 ~3pp。全量 n=810 后差距大幅缩小（之前 n=81 时 58.8% vs 41.5%）。

4. **Normalized ΔBrew 分层明显**：Qwen 系列 0.33-0.42，Llama 系列 0.26-0.30。Llama 架构的 brewing zone 比例更窄，这可能反映了不同架构对信息从"可读"到"可用"的处理效率差异，也可能是 CSD 能力弱导致 FJC 被推迟到更靠近 FPCL 的层（selection bias：只有最简单的 sample 才有 FJC，这些 sample 的 ΔBrew 天然较小）。

### 2.5 Brewing Zone 的跨模型稳定性

**Normalized ΔBrew 的一致性**：

| Model | n_layers | avg ΔBrew (abs) | **avg ΔBrew (norm)** | 各任务 norm range |
|-------|----------|-----------------|---------------------|------------------|
| Coder-0.5B | 24 | 6.91 | **0.288** | 0.207 – 0.392 |
| Coder-1.5B | 28 | 10.91 | **0.390** | 0.286 – 0.519 |
| Coder-3B | 36 | 13.32 | **0.370** | 0.308 – 0.506 |
| Coder-7B | 28 | 10.66 | **0.381** | 0.327 – 0.490 |
| Coder-14B† | 48 | 16.47 | **0.343** | 0.286 – 0.407 |
| Base-0.5B | 24 | 7.08 | **0.295** | 0.224 – 0.457 |
| Base-1.5B | 28 | 10.68 | **0.381** | 0.247 – 0.469 |
| Base-3B | 36 | 13.70 | **0.381** | 0.312 – 0.513 |
| Base-7B | 28 | 11.61 | **0.415** | 0.351 – 0.513 |
| DS-6.7B | 32 | 9.42 | **0.294** | 0.266 – 0.340 |
| Llama2-7B | 32 | 8.27 | **0.258** | 0.227 – 0.307 |
| CodeLlama-7B | 32 | 9.70 | **0.303** | 0.211 – 0.360 |
| Qwen3-0.6B | 28 | 6.84 | **0.244** | 0.184 – 0.355 |
| Qwen3-1.7B | 28 | 9.30 | **0.332** | 0.274 – 0.400 |
| Qwen3-4B | 36 | 13.93 | **0.387** | 0.337 – 0.442 |
| Qwen3-8B | 36 | 13.03 | **0.362** | 0.327 – 0.433 |

**Brewing zone 跨模型、跨架构、跨任务始终存在**：
- 所有 16 个模型的 ΔBrew 均 > 0——信息总是先在 probe 中可读，后在 CSD 中可用
- Normalized ΔBrew 范围 0.24-0.42，即模型总是用约 24%-42% 的层来完成从"可读"到"可用"的转换
- 同一模型内，value_tracking 的 ΔBrew 通常最大（简单任务 brewing zone 最宽），computing/loop 最小
- **注意**：ΔBrew 的选择偏差在 unresolved 率高的模型/任务中尤其显著（只有 FJC 存在的 sample 才参与计算）

---

## 3. Causal Validation: Activation Patching at FJC (Coder-7B)

> 2026-03-30. 全词表 argmax (global argmax) 判断 flip——将 FJC 层的 hidden state 注入到中性 target prompt `'# The value of x is "'`，看模型是否输出正确 digit。同时在 FJC 前后的对照层跑 patching 以验证 FJC 是因果特权层。

### TABLE 5: Activation Patching Flip Rate — Coder-7B × 6 Tasks × 6 Offsets

| Task | N_sel | FJC-8 | FJC-4 | FJC-2 | **FJC (0)** | FJC+2 | FJC+4 |
|------|-------|-------|-------|-------|-------------|-------|-------|
| value_tracking | 685 | 7.9% | 3.9% | 3.4% | **31.5%** | 47.8% | 54.0% |
| computing | 479 | 4.2% | 4.0% | 8.4% | **27.8%** | 30.8% | 28.5% |
| conditional | 647 | 16.5% | 17.4% | 17.7% | **43.9%** | 49.8% | 51.2% |
| function_call | 431 | 3.0% | 4.4% | 7.1% | **28.3%** | 31.6% | 31.2% |
| loop | 531 | 10.1% | 10.5% | 9.2% | **19.8%** | 27.4% | 25.8% |
| loop_unrolled | 436 | 13.6% | 12.4% | 13.9% | **31.7%** | 32.5% | 31.9% |

N_sel = FJC≠null 的 sample 数（Resolved + Overprocessed）。各 offset 的 N_effective 因边界裁剪略有不同。

**关键观察**：

1. **Pre-FJC → FJC 存在清晰的跳变**：所有任务在 FJC 都有显著的 flip rate 跳升（pre-FJC 3-17% → FJC 20-44%），确认 FJC 是信息从"不可用"变为"可用"的关键转折层。

2. **FJC 之后 flip rate 不降反升**（或持平）：FJC+2/+4 通常 ≥ FJC。这与旧论文（FJC 后 flip rate 下降至 30.7%）不同。原因：FJC 是 **首次** probe+CSD 同时正确的层，后续层信息通常更充分（hidden state 更成熟），patching 效果只会更好或持平。

3. **任务间差异**：conditional（43.9%）和 value_tracking（31.5%）flip rate 最高，function_call（28.3%）和 loop（19.8%）最低。这与 resolved rate 排序大致一致——信息可用性更高的任务，patching 效果更强。

4. **与旧论文对比**：旧论文 FJC flip rate 70% 是在更简单的数据集（79% resolved）上、用 150 个 selected sample 得到的。新数据 resolved 仅 ~40%，包含更多 hard sample。绝对值不同但核心结论一致：FJC 层具有因果特权。

---

## 4. Per-Difficulty Breakdowns (Coder-7B, 28L)

> 2026-03-31. 对每个任务的每个维度分别 sweep（固定其他两维取全部值，沿目标维度聚合）。
> 脚本: `scripts/per_difficulty_breakdown.py`，输出: `/path/to/brewing_output/artifacts/per_difficulty/`（CSV + JSON）。

### TABLE 6: Per-Difficulty Outcome & Brewing Metrics — 6 Tasks × 3 Dims

> 每行: N=bin 中样本数, NB=NO_BREWING 数, 四类 outcome% 排除 NB, FPCL/FJC 为 normalized (÷28), FJC%=FJC exist rate, ΔBrew=绝对层数。

**value_tracking** (mechanism × depth × distractors):

| dimension | bin | N | NB | Res% | OP% | MR% | UR% | FPCL | FJC | FJC% | ΔBrew |
|-----------|-----|---|----|------|-----|-----|-----|------|-----|------|-------|
| mechanism | function_chain | 294 | 0 | 61.6 | 20.7 | 6.5 | 11.2 | 0.049 | 0.532 | 82.3 | 13.67 |
| mechanism | container | 245 | 1 | 77.9 | 10.2 | 6.6 | 5.3 | 0.090 | 0.562 | 87.8 | 13.43 |
| mechanism | method_chain | 271 | 0 | 74.5 | 9.6 | 3.3 | 12.5 | 0.087 | 0.573 | 84.1 | 14.07 |
| depth | 1 | 267 | 0 | 77.5 | 10.9 | 6.0 | 5.6 | 0.057 | 0.555 | 88.4 | 14.01 |
| depth | 2 | 278 | 1 | 71.5 | 14.4 | 4.7 | 9.4 | 0.079 | 0.562 | 85.6 | 13.89 |
| depth | 3 | 265 | 0 | 63.4 | 16.2 | 5.7 | 14.7 | 0.086 | 0.547 | 79.6 | 13.23 |
| distractors | 0 | 276 | 0 | 86.6 | 8.7 | 1.1 | 3.6 | 0.044 | 0.549 | 95.3 | 14.24 |
| distractors | 1 | 248 | 0 | 63.7 | 16.5 | 6.9 | 12.9 | 0.078 | 0.566 | 80.2 | 13.70 |
| distractors | 2 | 286 | 1 | 61.8 | 16.5 | 8.4 | 13.3 | 0.100 | 0.554 | 78.0 | 13.14 |

**computing** (structure × steps × operators):

| dimension | bin | N | NB | Res% | OP% | MR% | UR% | FPCL | FJC | FJC% | ΔBrew |
|-----------|-----|---|----|------|-----|-----|-----|------|-----|------|-------|
| structure | func_arithmetic | 287 | 19 | 35.1 | 29.5 | 12.3 | 23.1 | 0.183 | 0.507 | 60.3 | 9.76 |
| structure | chained_calls | 264 | 7 | 28.4 | 35.0 | 9.7 | 26.8 | 0.195 | 0.464 | 61.7 | 8.70 |
| structure | accumulator | 259 | 9 | 14.4 | 42.8 | 12.4 | 30.4 | 0.158 | 0.428 | 55.2 | 9.01 |
| steps | 2 | 280 | 8 | 42.6 | 25.4 | 15.1 | 16.9 | 0.191 | 0.504 | 66.1 | 9.45 |
| steps | 3 | 256 | 14 | 22.3 | 34.3 | 11.2 | 32.2 | 0.179 | 0.466 | 53.5 | 8.66 |
| steps | 4 | 274 | 13 | 12.6 | 47.5 | 8.0 | 31.8 | 0.166 | 0.430 | 57.3 | 9.29 |
| operators | add | 273 | 4 | 30.9 | 32.0 | 17.5 | 19.7 | 0.187 | 0.541 | 61.9 | 11.08 |
| operators | add_sub | 256 | 16 | 25.0 | 33.3 | 10.4 | 31.2 | 0.190 | 0.452 | 54.7 | 8.07 |
| operators | add_mul | 281 | 15 | 22.6 | 41.4 | 6.4 | 29.7 | 0.161 | 0.411 | 60.5 | 8.19 |

**conditional** (branch_type × condition_type × depth):

| dimension | bin | N | NB | Res% | OP% | MR% | UR% | FPCL | FJC | FJC% | ΔBrew |
|-----------|-----|---|----|------|-----|-----|-----|------|-----|------|-------|
| branch_type | elif_chain | 272 | 9 | 67.3 | 15.2 | 11.8 | 5.7 | 0.236 | 0.589 | 79.8 | 10.18 |
| branch_type | guard_clause | 267 | 8 | 60.2 | 21.2 | 8.5 | 10.0 | 0.192 | 0.566 | 79.0 | 10.63 |
| branch_type | sequential_if | 271 | 3 | 50.4 | 31.3 | 10.1 | 8.2 | 0.064 | 0.525 | 80.8 | 13.22 |
| condition_type | numeric | 270 | 9 | 63.6 | 21.8 | 6.5 | 8.0 | 0.202 | 0.579 | 82.6 | 11.06 |
| condition_type | membership | 284 | 3 | 64.4 | 22.4 | 6.0 | 7.1 | 0.130 | 0.564 | 85.9 | 12.33 |
| condition_type | boolean_flag | 256 | 8 | 48.8 | 23.8 | **18.5** | 8.9 | 0.161 | 0.531 | 70.3 | 10.41 |
| depth | 1 | 263 | 4 | 76.1 | 11.6 | 7.7 | 4.6 | 0.164 | 0.581 | 86.3 | 11.83 |
| depth | 2 | 266 | 10 | 57.0 | 24.2 | 9.4 | 9.4 | 0.161 | 0.566 | 78.2 | 11.83 |
| depth | 3 | 281 | 6 | 45.5 | 31.6 | 13.1 | 9.8 | 0.165 | 0.531 | 75.4 | 10.38 |

**function_call** (mechanism × depth × distractors):

| dimension | bin | N | NB | Res% | OP% | MR% | UR% | FPCL | FJC | FJC% | ΔBrew |
|-----------|-----|---|----|------|-----|-----|-----|------|-----|------|-------|
| mechanism | arithmetic | 278 | 17 | 38.7 | 24.5 | 2.3 | 34.5 | 0.217 | 0.522 | 59.4 | 9.35 |
| mechanism | conditional_return | 268 | 17 | 10.8 | 35.9 | 2.0 | **51.4** | 0.171 | 0.504 | 43.7 | 10.10 |
| mechanism | container_relay | 264 | 14 | 33.2 | 26.4 | 7.2 | 33.2 | 0.147 | 0.515 | 56.4 | 11.29 |
| depth | 1 | 264 | 2 | 61.1 | 17.6 | 2.3 | 19.1 | 0.189 | 0.590 | 78.0 | 11.36 |
| depth | 2 | 285 | 22 | 17.1 | 32.7 | 6.1 | 44.1 | 0.160 | 0.451 | 46.0 | 9.34 |
| depth | 3 | 261 | 24 | **2.5** | 37.1 | 3.0 | **57.4** | 0.188 | 0.439 | 36.0 | 8.96 |
| distractors | 0 | 298 | 20 | 37.1 | 24.8 | 6.5 | 31.7 | 0.169 | 0.471 | 57.7 | 8.83 |
| distractors | 1 | 249 | 15 | 21.8 | 29.5 | 2.6 | 46.2 | 0.184 | 0.543 | 48.2 | 10.59 |
| distractors | 2 | 263 | 13 | 22.8 | 32.8 | 2.0 | 42.4 | 0.185 | 0.545 | 52.9 | 11.63 |

**loop** (body_type × iterations × init_offset):

| dimension | bin | N | NB | Res% | OP% | MR% | UR% | FPCL | FJC | FJC% | ΔBrew |
|-----------|-----|---|----|------|-----|-----|-----|------|-----|------|-------|
| body_type | simple_acc | 289 | 0 | 40.8 | 24.2 | 12.5 | 22.5 | 0.051 | 0.417 | 65.1 | 10.60 |
| body_type | filter_count | 246 | 10 | 31.4 | 32.6 | 10.2 | 25.8 | 0.161 | 0.421 | 61.4 | 8.68 |
| body_type | dual_var | 275 | 3 | 33.5 | 37.1 | 5.9 | 23.5 | 0.106 | 0.446 | 69.8 | 9.53 |
| iterations | 2 | 272 | 4 | 36.2 | 29.5 | 14.6 | 19.8 | 0.106 | 0.455 | 64.7 | 10.22 |
| iterations | 3 | 273 | 5 | **45.1** | 29.1 | 4.9 | 20.9 | 0.083 | 0.352 | 72.9 | 8.01 |
| iterations | 4 | 265 | 4 | 24.9 | 34.9 | 9.2 | 31.0 | 0.118 | 0.497 | 58.9 | 11.15 |
| init_offset | 0 | 264 | 1 | 63.1 | 22.8 | 6.8 | 7.2 | 0.073 | 0.500 | 85.6 | 12.18 |
| init_offset | low | 283 | 6 | 19.5 | 37.5 | 10.5 | 32.5 | 0.139 | 0.328 | 55.8 | 5.91 |
| init_offset | high | 263 | 6 | 24.5 | 32.7 | 11.3 | 31.5 | 0.094 | 0.427 | 55.9 | 9.83 |

**loop_unrolled** (body_type × iterations × init_offset):

| dimension | bin | N | NB | Res% | OP% | MR% | UR% | FPCL | FJC | FJC% | ΔBrew |
|-----------|-----|---|----|------|-----|-----|-----|------|-----|------|-------|
| body_type | simple_acc | 267 | 0 | 36.7 | 18.7 | 18.4 | 26.2 | 0.119 | 0.498 | 55.4 | 11.32 |
| body_type | filter_count | 274 | 11 | 32.7 | 34.6 | 8.0 | 24.7 | 0.207 | 0.500 | 64.6 | 8.80 |
| body_type | dual_var | 269 | 2 | 14.6 | 27.0 | 4.9 | **53.6** | 0.119 | 0.450 | 41.3 | 9.41 |
| iterations | 2 | 247 | 2 | 46.1 | 22.4 | 7.3 | 24.1 | 0.138 | 0.558 | 68.0 | 11.87 |
| iterations | 3 | 285 | 5 | 27.9 | 30.0 | 9.3 | 32.9 | 0.151 | 0.432 | 56.8 | 8.54 |
| iterations | 4 | 278 | 6 | 11.8 | 27.2 | 14.3 | **46.7** | 0.154 | 0.455 | 38.1 | 8.47 |
| init_offset | 0 | 275 | 2 | 43.2 | 22.7 | 4.8 | 29.3 | 0.102 | 0.517 | 65.5 | 11.41 |
| init_offset | low | 253 | 4 | 22.9 | 36.1 | 12.9 | 28.1 | 0.192 | 0.426 | 58.1 | 7.61 |
| init_offset | high | 282 | 7 | 17.5 | 22.2 | 13.8 | 46.5 | 0.155 | 0.517 | 38.7 | 10.13 |

### Per-Difficulty Key Findings

1. **难度驱动的质量崩塌**：function_call × depth 最剧烈——Res% 61.1→2.5%, UR 19.1→57.4%。computing × steps 次之——Res 42.6→12.6%, OP 升至 47.5%。
2. **MR 涌现**：conditional × boolean_flag MR=18.5%（numeric/membership ~6%）；computing × add 纯加法 MR=17.5%（add_mul 6.4%）。
3. **Loop U-shape 反转**：loop × iterations 峰值在 iter=3 (Res 45.1%)，非旧数据的谷底。loop_unrolled 单调下降 (46.1→11.8)。
4. **FPCL 稳定 vs FJC 崩塌**：FPCL 跨 bin 变化 <0.05，FJC_exist% 剧烈下降（function_call depth: 78→36%）。瓶颈在 readiness 而非 availability。
5. **循环语法缓冲**：dual_var 在 loop 中 UR=23.5%，loop_unrolled 中 UR=53.6%——展开后丧失结构线索。

---

## 5. Causal Validation & GT-free Signals (2026-03-31)

### 5.1 A2 验证：FJC=∅ 正确率

**实验设置**：对 anchor 模型 (Qwen2.5-Coder-7B) 6 个 task 的全量 n=810 样本，统计 FJC=null 的样本中最终输出正确的比例。数据来源：`diagnostics.json` (model_output, fjc) + `samples.json` (answer)，按 sample_id join。

**脚本**：`scripts/p0_calculations.py`

| Task | FJC=null | Correct | Rate |
|------|---------|---------|------|
| value_tracking | 125 | 52 | 41.6% |
| computing | 331 | 51 | 15.4% |
| conditional | 163 | 38 | 23.3% |
| function_call | 379 | 39 | 10.3% |
| loop | 279 | 55 | 19.7% |
| loop_unrolled | 374 | 57 | 15.2% |
| **TOTAL** | **1651** | **292** | **17.69%** |

**结论**：旧论文 A2 断言 "FJC=∅ ≈ 做错 (<1%)" 不再成立。17.7% 的 FJC-null 样本最终输出正确。value_tracking 高达 41.6%——简单任务即使 CSD 全层未 decode 成功，模型最终仍有概率碰对。

### 5.2 Coder-Base FJC Correlation (§5.3)

**实验设置**：取 Qwen2.5-Coder-7B 和 Qwen2.5-7B (Base) 在 6 个 task 上的 `mean_fjc_normalized`（从 `diagnostics.json` 读取，仅 FJC≠null 的样本均值），计算 Pearson correlation。

| Task | Coder-7B | Base-7B |
|------|----------|---------|
| value_tracking | 0.555 | 0.594 |
| computing | 0.469 | 0.518 |
| conditional | 0.560 | 0.576 |
| function_call | 0.515 | 0.553 |
| loop | 0.429 | 0.452 |
| loop_unrolled | 0.486 | 0.464 |

**结果**：r=0.901, p=0.014. Mean FJC shift = -0.67 layers (Coder FJC 比 Base 略早)。与旧论文 (r=0.90, p=0.014, shift=+0.18) 高度一致。

### 5.3 Late-Layer Sparsity (App E.3)

**实验设置**：从 anchor 模型 (Qwen2.5-Coder-7B) 6 task 的 hidden state cache `(810, 28, 3584)` 读取每层 hidden state，计算 Hoyer sparsity `(√d - L1/L2) / (√d - 1)`。按 outcome (Overprocessed vs Resolved) 分组，逐层求均值和标准差。

**脚本**：`scripts/late_layer_sparsity.py`，输出 `artifacts/sparsity/`。

| Task | N_OT | N_Res | OT spike layer | Max gap layer | Max gap |
|------|------|-------|----------------|---------------|---------|
| value_tracking | 112 | 573 | 27 | 27 | 0.0054 |
| computing | 276 | 203 | 27 | 24 | 0.0313 |
| conditional | 179 | 468 | 27 | 24 | 0.0175 |
| function_call | 220 | 211 | 27 | 24 | 0.0494 |
| loop | 248 | 283 | 27 | 24 | 0.0362 |
| loop_unrolled | 213 | 223 | 27 | 22 | 0.0059 |

**结论**：A18 confirmed. OT sparsity > Resolved 在全部 6 task 成立。Spike 一致在 layer 27（最后层），gap 在 layer 22–24（倒数 4–6 层）最大。function_call 的 gap 最显著 (0.049)，value_tracking/loop_unrolled gap 很小 (~0.005)。

### 5.4 GT-free Signals (App F)

**实验设置**：从 CSD 结果的 `layer_confidences (L, 10)` 衍生两个 GT-free signal：
1. **Entropy** H(ℓ) = -Σ p_i log(p_i)，10-class digit softmax 的信息熵
2. **MaxConf** C(ℓ) = max_i p_i，最大类别概率

OT 检测：提取 4 个特征（conf_drop, entropy_rise, nondigit_rise, tail_nondigit），用 z-score 标准化后相加作为 combined score，计算 AUC 和 F1（Youden's J 选阈值）。FJC 检测：confidence > 0.5 且 non-digit < 0.3 的首层。

**注意**：旧论文的 non-digit probability (full-vocab) 在 CSD patchscope 下无效（AUC ≈ 0.5），因为 patchscope 即使对 OT 样本也倾向输出某个 digit token，non-digit 无法区分正确/错误 digit。

**脚本**：`scripts/gt_free_signals.py`，输出 `artifacts/gt_free/`。

| Task | N | FJC agree | OT AUC (conf_drop) | OT AUC (ent_rise) | OT AUC (combined) | OT F1 | Overall agree |
|------|---|-----------|--------------------|--------------------|-------------------|-------|---------------|
| value_tracking | 810 | 0.786 | 0.755 | 0.764 | 0.770 | 0.414 | 0.787 |
| computing | 810 | 0.559 | 0.611 | 0.814 | 0.752 | 0.715 | 0.550 |
| conditional | 810 | 0.747 | 0.616 | 0.716 | 0.686 | 0.490 | 0.762 |
| function_call | 810 | 0.662 | 0.800 | 0.860 | 0.848 | 0.768 | 0.651 |
| loop | 810 | 0.527 | 0.633 | 0.754 | 0.734 | 0.674 | 0.523 |
| loop_unrolled | 810 | 0.586 | 0.607 | 0.715 | 0.702 | 0.626 | 0.585 |
| **Weighted avg** | 4860 | 0.645 | — | — | **0.746** | — | **0.643** |

**结论**：entropy_rise 是最强单一信号 (AUC 0.71–0.86)。Combined OT AUC weighted avg 0.746，overall agreement 64.3%。远低于旧论文的 87%（主要因为缺 non-digit signal），但 OT vs Resolved 的区分力在 function_call (0.848) 和 computing (0.752) 上仍然可用。

### 5.5 Layer Skipping (App E.2, A4)

**实验设置**：对 anchor 模型 (Qwen2.5-Coder-7B, 28 层) 每 task 选 50 OT + 25 Resolved control（按 diagnostics 中 outcome 分组，取前 N 个有 FJC 的样本）。干预方式：将 FJC 层的 hidden state 注入到同一 prompt 的 FJC+k 层（k=2,4,6），替换该层 last token position 的 hidden state，然后继续 forward 读取模型输出。使用 nnsight InterventionBackend。

如果 OT 样本在跳过 post-FJC 层后输出正确答案 → 信息在 FJC 处是存在的，后续层的 overprocessing 破坏了它。

**脚本**：`scripts/causal_7b_small.py`，输出 `artifacts/layer_skipping/results_7B_small.json`。

**2026-03-31 Bug 修复记录**：

排查发现 `NNsightInterventionBackend.run_interventions()` 存在三个问题，修复后重跑全量：

1. **dtype/device 从 nnsight proxy 读取 (Critical)**：`backend.py:120-121` 的 `layer_out.dtype` / `layer_out.device` 读取的是 nnsight proxy 对象而非真实值。`nnsight_ops.py:215` 的 `patchscope_lens` 函数有明确注释 *"proxy objects don't expose these"* 并使用 `next(model.parameters())` 获取。修复：在 `__init__` 中缓存模型的真实 dtype/device，并新增 `_get_layer_device()` 方法处理 `device_map="auto"` 下不同层在不同 GPU 的情况。

2. **全词表 argmax 无 answer_space 限制 (Medium)**：`causal_7b_small.py` 使用 `answer_space=None`（全词表 argmax）。实测对 computing 任务无影响（模型在 code prompt 上天然预测 digit），但添加 `answer_space=["0".."9"]` + `baseline_subtract=False` 作为 robustness 保障。

3. **baseline subtraction 不适用于因果实验 (Medium)**：backend 的 `answer_space` 模式默认做 CSD 风格基线相减（测量相对变化而非绝对预测）。新增 `baseline_subtract` 标志，因果实验设为 `False`。

**Sanity check** (`scripts/sanity_check_intervention.py`)：offset=0 恒等测试 3/3 PASS（注入 layer X 的 hidden state 回 layer X，输出不变），确认 nnsight 干预机制本身正确。

**修复后结果**（与修复前相比 Layer Skipping 变化不大，主要影响在 Re-injection）：

| Task | Offset | OT rescued | OT rate | Res maintained | Res rate |
|------|--------|-----------|---------|----------------|----------|
| value_tracking | +2 | 14/49 | 28.6% | 24/25 | 96.0% |
| value_tracking | +4 | 10/46 | 21.7% | 20/25 | 80.0% |
| value_tracking | +6 | 6/37 | 16.2% | 8/18 | 44.4% |
| computing | +2 | 1/48 | 2.1% | 15/25 | 60.0% |
| computing | +4 | 3/48 | 6.2% | 10/24 | 41.7% |
| computing | +6 | 4/45 | 8.9% | 5/15 | 33.3% |
| conditional | +2 | 6/46 | 13.0% | 23/25 | 92.0% |
| conditional | +4 | 8/45 | 17.8% | 19/24 | 79.2% |
| conditional | +6 | 3/35 | 8.6% | 6/12 | 50.0% |
| function_call | +2 | 2/49 | 4.1% | 17/24 | 70.8% |
| function_call | +4 | 4/47 | 8.5% | 8/24 | 33.3% |
| function_call | +6 | 1/40 | 2.5% | 7/9 | 77.8% |
| loop | +2 | 4/48 | 8.3% | 17/25 | 68.0% |
| loop | +4 | 2/48 | 4.2% | 9/25 | 36.0% |
| loop | +6 | 4/42 | 9.5% | 8/23 | 34.8% |
| loop_unrolled | +2 | 3/49 | 6.1% | 19/25 | 76.0% |
| loop_unrolled | +4 | 5/49 | 10.2% | 16/25 | 64.0% |
| loop_unrolled | +6 | 8/44 | 18.2% | 2/12 | 16.7% |

**结论**：
1. **Resolved control 验证干预有效性**：offset=+2 时 Res maintain 60–96%（value_tracking 96%, conditional 92%），说明小范围 skip 不破坏正常处理。
2. **OT rescue 远低于旧论文**：2–29% vs 旧 73.3%。value_tracking +2 最高达 28.6%（修复前 14.3%，dtype 修复有贡献）。Overprocessing 不是简单的"后几层线性破坏 FJC 信息"——而是跨层的非线性表征扭曲，FJC hidden state 本身不足以恢复。
3. **Offset 越大，Res maintain 越低**：符合预期，跳过越多层越破坏正常 forward dynamics。
4. **任务难度相关**：value_tracking（最简单的数据流任务）rescue 最高，computing（需多步计算）最低。

### 5.6 Re-injection (App E.4, A5)

**实验设置**：对 anchor 模型 (Qwen2.5-Coder-7B, 28 层) 每 task 选 50 Unresolved + 25 Resolved control（取有 FPCL 的样本）。干预方式：将 FPCL 层的 hidden state（信息首次线性可读的位置）注入到同一 prompt 的 late layer (L-4, L-3, L-2)，替换 last token position 的 hidden state。

假设：如果 Unresolved 样本的信息在 FPCL 处存在但后续从未被模型自身 decode，将这个早期表征重新注入到 late layer 可能让模型"再次尝试 decode"。

**脚本**：`scripts/causal_7b_small.py`（replace mode），`scripts/reinjection_modes_compare.py` + `scripts/_reinjection_single_mode.py`（多模式对比）。输出 `artifacts/reinjection/results_7B_*.json`。

**2026-03-31 修复与注入模式对比**：

修复 dtype/device bug 后 replace 模式 UR rescue 从 0–10% 提升到 0–30%，但 Resolved control 仍只有 0–36%。分析发现根本原因是 **representation space norm mismatch**：

```
Hidden state L2-norm 统计 (computing task, sample 0):
  Layer 0:  ~15
  Layer 14: ~75
  Layer 27: ~203
FPCL 中位数 = layer 1 (norm ~15), 注入目标 = layer 24-26 (expected norm ~200)
```

13 倍的 norm 差距使得 LayerNorm 后表征被严重扭曲。为此测试了 4 种注入模式：

- **`replace`**: 完全替换 — `h_target = h_source`（原方法）
- **`norm_match`**: 缩放 source 到目标层 norm — `h_target = h_source * (‖h_orig‖ / ‖h_source‖)`
- **`alpha_blend α=0.3`**: 加权混合 — `h_target = 0.7 * h_orig + 0.3 * h_source`
- **`alpha_blend α=0.5`**: 加权混合 — `h_target = 0.5 * h_orig + 0.5 * h_source`

注意：`norm_match` 和 `alpha_blend` 的 h_orig 取自 hidden state cache（与当前 forward pass 等价，因为 source_prompt == target_prompt），预计算后仅在 trace 中做一次 replace，避免 nnsight proxy 积累导致 OOM。

**四种模式对比 — L=24 (L-4):**

| Task | replace UR | replace Res | norm_match UR | norm_match Res | α=0.3 UR | α=0.3 Res | α=0.5 UR | α=0.5 Res |
|------|-----------|-------------|---------------|----------------|----------|-----------|----------|-----------|
| value_tracking | 30.0% | 12.0% | 18.0% | 20.0% | **34.0%** | **100.0%** | 34.0% | 100.0% |
| computing | 16.0% | 24.0% | 16.0% | 20.0% | **22.0%** | **100.0%** | 22.0% | 100.0% |
| conditional | 18.0% | 36.0% | 28.0% | 24.0% | **38.0%** | **100.0%** | 38.0% | 96.0% |
| function_call | 22.0% | 12.0% | 10.0% | 20.0% | **22.0%** | **100.0%** | 26.0% | 100.0% |
| loop | 8.0% | 4.0% | 4.0% | 36.0% | **30.0%** | **84.0%** | 34.0% | 72.0% |
| loop_unrolled | 26.0% | 12.0% | 16.0% | 36.0% | **22.0%** | **96.0%** | 20.0% | 84.0% |

**推荐配置 alpha_blend α=0.3 — 逐任务逐层完整结果：**

| Task | L-4 UR | L-4 Res | L-3 UR | L-3 Res | L-2 UR | L-2 Res |
|------|--------|---------|--------|---------|--------|---------|
| value_tracking | 34.0% | 100.0% | 34.0% | 100.0% | 32.0% | 100.0% |
| computing | 22.0% | 100.0% | 22.0% | 100.0% | 22.0% | 96.0% |
| conditional | 38.0% | 100.0% | 34.0% | 96.0% | 32.0% | 100.0% |
| function_call | 22.0% | 100.0% | 22.0% | 100.0% | 22.0% | 100.0% |
| loop | 30.0% | 84.0% | 28.0% | 96.0% | 26.0% | 92.0% |
| loop_unrolled | 22.0% | 96.0% | 24.0% | 96.0% | 22.0% | 96.0% |

**Replace 模式完整结果（修复后，用于对比）：**

| Task | L-4 UR | L-4 Res | L-3 UR | L-3 Res | L-2 UR | L-2 Res |
|------|--------|---------|--------|---------|--------|---------|
| value_tracking | 30.0% | 12.0% | 2.0% | 20.0% | 10.0% | 12.0% |
| computing | 16.0% | 24.0% | 4.0% | 20.0% | 14.0% | 16.0% |
| conditional | 18.0% | 36.0% | 10.0% | 12.0% | 14.0% | 24.0% |
| function_call | 22.0% | 12.0% | 2.0% | 16.0% | 0.0% | 16.0% |
| loop | 8.0% | 4.0% | 0.0% | 0.0% | 6.0% | 12.0% |
| loop_unrolled | 26.0% | 12.0% | 10.0% | 0.0% | 8.0% | 0.0% |

**结论**：
1. **Alpha-blend α=0.3 是最优注入模式**：Res ctrl 恢复到 84–100%（接近旧论文 100%），UR rescue 22–38%（低于旧论文 33–47% 但量级合理且跨层一致）。
2. **Replace 模式失败的根因是 norm mismatch**：FPCL (layer 1, norm~15) → late layer (layer 24–26, norm~200) 的 13 倍 norm 差距导致 LayerNorm 后表征扭曲。不是干预框架 bug，而是实验方法需要适配表征空间差异。
3. **Norm_match 不够**：只匹配 norm 不保留方向，Res ctrl 仅 4–36%。说明早期和晚期层 hidden state 的几何方向也不兼容，单纯 rescale 无法修复。
4. **α=0.3 vs α=0.5**：α=0.3 的 Res ctrl 更稳（几乎全 100%），UR rescue 相当。α=0.5 在 loop 类任务上 Res ctrl 下降到 72–84%。α=0.3 作为默认值，注入信号足以影响预测但不破坏原有表征。
5. **任务差异**：conditional rescue 最高 (38%)——条件分支的"正确路径"信息即使少量注入也能改变最终预测。loop 类任务 rescue 最稳定（跨层 26–30%）。

---

## 6. App C: Probing 训练统计 & 收敛曲线

> 数据文件: `brewing_output/probe_experiments/appc_curves/Qwen__Qwen2.5-Coder-7B__computing.json`
> 脚本: `scripts/train_probes_appc.py`

**配置**: Qwen2.5-Coder-7B × computing, 28 layers, dim=3584, chain_train n=3915, eval n=810.
200 epochs, full-batch Adam, lr=1e-3, wd=0.05, 无 early stop (为获取完整收敛曲线).
每层每 epoch 记录: train_acc, val_acc, eval_acc, train_loss, val_loss.

**Per-layer best eval accuracy:**

| Layer | BestEval | @Epoch | Layer | BestEval | @Epoch | Layer | BestEval | @Epoch | Layer | BestEval | @Epoch |
|-------|----------|--------|-------|----------|--------|-------|----------|--------|-------|----------|--------|
| L0  | 40.0% | 151 | L7  | 50.6% | 193 | L14 | 69.5% | 196 | L21 | 76.8% | 179 |
| L1  | 45.9% | 111 | L8  | 54.8% | 182 | L15 | 69.3% | 198 | L22 | 76.3% | 189 |
| L2  | 47.6% | 178 | L9  | 63.1% | 150 | L16 | 72.1% | 188 | L23 | 80.9% | 198 |
| L3  | 46.7% | 192 | L10 | 65.3% | 183 | L17 | 71.0% | 187 | L24 | 82.7% | 185 |
| L4  | 52.7% | 146 | L11 | 64.6% | 186 | L18 | 74.7% | 168 | L25 | 84.0% | 155 |
| L5  | 51.7% | 163 | L12 | 64.7% | 185 | L19 | 75.1% | 197 | L26 | 84.7% | 164 |
| L6  | 48.6% | 173 | L13 | 68.3% | 182 | L20 | 74.1% | 157 | L27 | 83.6% | 171 |

**结论**:
1. Best layer: **L26 = 84.7%**，与 chain_train (early stop) 的 81.6%@L26 一致
2. 收敛稳定: 大多数层 best epoch 在 150–200，final eval ≈ best eval (gap <0.5%)，无过拟合
3. 层间梯度清晰: L0–L8 (40–55%) → L9–L13 (63–68%) → L14–L22 (69–77%) → L23–L27 (81–85%)
4. 推荐 Fig C.1 代表层: L0 (early), L9 (transition), L16 (mid), L23 (late-mid), L26 (best)

---

## 7. 待确认 / 注意事项

1. **Coder-14B** 为 n=270 (1/3 子集)，其余 15 个模型均为 n=810 全量。Coder-14B 的 function_call resolved 从 n=81 时的 85.2% 降至 n=270 的 48.7%，Qwen3-8B 从 86.4% 降至 24.7%——确认 n=81 时的异常高值为小样本噪声。
2. 所有 Qwen2.5-Base（0.5B/1.5B/3B/7B）和 Qwen3-Base（0.6B/1.7B/4B/8B）均为 n=810 全量。
3. **Llama2/CodeLlama CSD 能力弱**是真实现象（非代码 bug）。如果论文需要更强的跨架构对比，可以考虑调整 target prompt 适配这两个模型的 tokenizer。
4. **ΔBrew 选择偏差**：所有 ΔBrew 统计仅覆盖有 FJC 的 sample（Resolved + Overprocessed）。高 unresolved 率的任务/模型组合中，ΔBrew 代表的是相对简单的子集，不能推广到全部 sample。论文讨论中需要明确说明这一点。
5. **CSD Baseline Subtraction 导致 Unresolved False Negative**——已在对话中分析，待 Eric 决定是否修改方法论后再落盘。
