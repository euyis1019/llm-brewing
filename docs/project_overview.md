# Brewing Framework — Project Overview

> 本文档用于快速接手 Brewing 框架。所有 Mermaid 图可在 GitHub / VS Code / 任意 Markdown 预览器中渲染。

---

## 1. 研究背景：一句话版本

> LLM 在逐层处理代码推理任务时，答案信息会经历 **"先存在、后可用"** 的内部生命周期。Brewing 框架通过两个互补的诊断函数来追踪这个过程。

```mermaid
graph LR
    subgraph 两个诊断函数
        P["Φ_P: Linear Probing<br/><b>information availability</b><br/>外部探针能否从 hidden state 读出答案"]
        C["Φ_C: CSD<br/><b>information readiness</b><br/>模型自身能否从 hidden state 解码出答案"]
    end
    P -->|"Φ_P 先变 correct（FPCL）"| Gap["Brewing Gap<br/>Δ_brew = FJC − FPCL"]
    C -->|"Φ_C 后变 correct（FJC）"| Gap
    Gap --> Taxonomy["四类 Outcome"]
```

两者的时间差（层数差）定义了 **brewing-to-resolution** 结构，最终每个样本被分类为四类 outcome 之一：

```mermaid
graph TD
    FJC{"FJC 存在？<br/>(Probe + CSD 同时 correct)"}
    FJC -->|Yes| ModelCorrect{"模型最终输出正确？"}
    FJC -->|No| TailConf{"CSD tail confidence ≥ 0.5？"}
    ModelCorrect -->|Yes| R["✅ Resolved<br/>答案被正确解码"]
    ModelCorrect -->|No| O["⚠️ Overprocessed<br/>曾经对了但后层破坏"]
    TailConf -->|Yes| M["❌ Misresolved<br/>自信地给了错误答案"]
    TailConf -->|No| U["❓ Unresolved<br/>始终没算出来"]
```

---

## 2. Pipeline 架构（S0 → S3）

```mermaid
flowchart TB
    subgraph S0["S0: Dataset Resolve/Build"]
        direction LR
        DataGen["datagen/\n6 个任务生成器"] --> Adapter["adapter.py\nraw dict → Sample"]
        Adapter --> Samples["list[Sample]\n+ DatasetManifest"]
    end

    subgraph S1["S1: Hidden State Cache"]
        direction LR
        Model["Pre-loaded Model\n(nnsight / HF)"] --> CacheBuilder["cache_builder.py"]
        Samples2["Samples"] --> CacheBuilder
        CacheBuilder --> Cache["HiddenStateCache\n(N, L, D) ndarray\n+ model_predictions"]
    end

    subgraph S2["S2: Method Execution"]
        direction LR
        LP["LinearProbing\n(CacheOnly, Trained)\nΦ_P"] --> MR1["MethodResult\n(per_sample)"]
        CSD_M["CSD\n(ModelOnline, Training-free)\nΦ_C"] --> MR2["MethodResult\n(per_sample)"]
    end

    subgraph S3["S3: Diagnostics (解耦)"]
        direction LR
        MR1b["Probe MethodResult"] --> Diag["outcome.py\nrun_diagnostics()"]
        MR2b["CSD MethodResult"] --> Diag
        Preds["model_predictions\n(from Cache)"] --> Diag
        Diag --> DR["DiagnosticResult\nFPCL / FJC / Δ_brew / Outcome\n+ group_by_difficulty"]
    end

    S0 --> S1
    S1 --> S2
    S2 -.->|"从磁盘加载"| S3

    style S0 fill:#e8f4fd,stroke:#2196f3
    style S1 fill:#fff3e0,stroke:#ff9800
    style S2 fill:#e8f5e9,stroke:#4caf50
    style S3 fill:#fce4ec,stroke:#e91e63
```

**关键设计决策**：
- S0-S2 由 `Orchestrator` 统一驱动
- S3 与 pipeline **完全解耦**，通过 `run_diagnostics_from_disk()` 从磁盘文件独立运行
- Probing 的训练（`LinearProbing.train()`）在主 pipeline **之外**单独执行，主 pipeline 只做 eval_only

---

## 3. 代码模块关系

```mermaid
graph TD
    CLI["cli.py<br/>CLI 入口"] --> Orch["orchestrator.py<br/>S0-S2 调度"]
    Orch --> Reg["registry.py<br/>Benchmark/Method 查找"]
    Orch --> RM["resources.py<br/>ResourceManager<br/>resolve-or-build"]
    Orch --> CB["cache_builder.py<br/>Hidden state 提取"]

    Reg --> BenchSpec["benchmarks/cue_bench/<br/>spec.py + builder.py + adapter.py"]
    Reg --> Methods["methods/<br/>linear_probing.py + csd.py"]

    CB --> NNOps["nnsight_ops.py<br/>nnsight tracing 封装"]
    Methods --> NNOps
    Methods --> RM

    DiagMod["diagnostics/<br/>outcome.py + metrics.py<br/>+ group_by_difficulty"] -.->|"读磁盘文件"| RM

    Schema["schema/<br/>types.py + results.py + benchmark.py<br/>纯数据结构"] ---|"被所有模块依赖"| Orch
    Schema ---|"被所有模块依赖"| RM
    Schema ---|"被所有模块依赖"| Methods
    Schema ---|"被所有模块依赖"| DiagMod

    style Schema fill:#f3e5f5,stroke:#9c27b0
    style Orch fill:#e8f5e9,stroke:#4caf50
    style RM fill:#fff3e0,stroke:#ff9800
    style DiagMod fill:#fce4ec,stroke:#e91e63
```

---

## 4. 数据结构全景

### 4.1 类型层次关系

```mermaid
classDiagram
    direction TB

    class Sample {
        +str id
        +str benchmark
        +str subset
        +str prompt
        +str answer
        +dict|None difficulty
        +dict|None metadata
    }

    class DatasetManifest {
        +str dataset_id
        +DatasetPurpose purpose
        +str benchmark
        +str|None subset
        +list~str~ sample_ids
        +dict generation_config
        +int|None seed
        +str|None version
        +str created_at
    }

    class HiddenStateCache {
        +str model_id
        +list~str~ sample_ids
        +ndarray hidden_states  «(N,L,D)»
        +str|list~int~ token_position
        +list~str~ model_predictions
        +dict metadata
        +n_samples() int
        +n_layers() int
        +hidden_dim() int
    }

    class FitArtifact {
        +str artifact_id
        +str method
        +str model_id
        +str train_dataset_id
        +str|None train_cache_id  «always None»
        +dict fit_config
        +dict fit_metrics
        +dict metadata
    }

    class SampleMethodResult {
        +str sample_id
        +ndarray layer_values  «(L,) 0/1 correctness»
        +list~str~|None layer_predictions  «(L,) decoded token»
        +ndarray|None layer_confidences  «(L,C) prob dist»
        +dict extras
    }

    class MethodResult {
        +str method
        +str model_id
        +Granularity granularity
        +str eval_dataset_id
        +list~SampleMethodResult~ sample_results
        +ndarray|None layer_values  «aggregate only»
        +dict extras
        +str|None train_dataset_id
        +int|None train_size
        +str|None fit_artifact_id
        +FitStatus|None fit_status
        +dict|None fit_metrics_summary
    }

    class SampleDiagnostic {
        +str sample_id
        +int|None fpcl
        +int|None fjc
        +int|None delta_brew
        +Outcome|None outcome
        +str|None model_output
        +float|None csd_tail_confidence
    }

    class DiagnosticResult {
        +str model_id
        +str eval_dataset_id
        +str benchmark
        +str|None subset
        +list~SampleDiagnostic~ sample_diagnostics
        +dict outcome_distribution
        +float|None mean_fpcl_normalized
        +float|None mean_fjc_normalized
        +float|None mean_delta_brew
    }

    MethodResult "1" *-- "N" SampleMethodResult
    DiagnosticResult "1" *-- "N" SampleDiagnostic
    DatasetManifest ..> Sample : references by id
    HiddenStateCache ..> Sample : references by id
    MethodResult ..> FitArtifact : references by fit_artifact_id
```

### 4.2 Benchmark 定义结构

```mermaid
classDiagram
    direction LR

    class BenchmarkSpec {
        +str name  «"CUE-Bench"»
        +str domain  «"code_reasoning"»
        +AnswerMeta answer_meta
        +str|None prompt_template
        +list~SubsetSpec~ subsets
        +get_subset(name)
        +subset_names
    }

    class SubsetSpec {
        +str name
        +str category
        +dict difficulty_schema
        +Callable|None generate_fn
        +str|None question_suffix
    }

    class AnswerMeta {
        +AnswerType answer_type  «CATEGORICAL»
        +list~str~|None answer_space  «["0".."9"]»
        +int|None max_answer_tokens  «1»
    }

    BenchmarkSpec "1" *-- "6" SubsetSpec
    BenchmarkSpec "1" *-- "1" AnswerMeta
```

### 4.3 配置与资源定位

```mermaid
classDiagram
    direction TB

    class RunConfig {
        +str mode  «"cache_only"|"train_probing"|"eval"|"diagnostics"»
        +str benchmark
        +list~str~|None subsets
        +str model_id
        +list~str~ methods
        +str output_root
        +str|None data_dir
        +int seed
        +int|None samples_per_config
        +str fit_policy  «"eval_only"»
        +dict method_configs
        +int batch_size
        +str|None device
        +bool use_fixture
        +str|None model_cache_dir
        +str|None quantization
        +benchmark_path_safe() str
    }

    class ResourceKey {
        «frozen»
        +str benchmark
        +str split  «"eval"/"train"»
        +str task
        +int seed
        +str|None model_id
        +str|None method
        +dataset_id() str
        +model_id_safe() str
    }

    class MethodConfig {
        +str method
        +str benchmark
        +dict config  «untyped pass-through»
        +fit_policy() FitPolicy
        +train_dataset_id() str|None
    }

    class MethodRequirements {
        +bool needs_answer_space
        +SingleTokenRequirement single_token_answer
        +bool needs_model_online
        +bool trained
        +dict custom_config_schema
    }

    RunConfig ..> ResourceKey : Orchestrator 构造
    RunConfig ..> MethodConfig : Orchestrator 构造
```

---

## 5. 磁盘布局

`ResourceManager` 管理所有持久化资源，路径由 `ResourceKey` 决定：

```
{output_root}/
├── datasets/{benchmark}/{split}/{task}/seed{seed}/
│   ├── manifest.json          ← DatasetManifest
│   └── samples.json           ← list[Sample]
│
├── caches/{benchmark}/{split}/{task}/seed{seed}/{model_id_safe}/
│   ├── hidden_states.npz      ← HiddenStateCache.hidden_states (N,L,D)
│   └── meta.json              ← model_id, sample_ids, model_predictions
│
├── artifacts/{benchmark}/{task}/{model_id_safe}/{method}/seed{seed}/
│   ├── metadata.json          ← FitArtifact
│   └── model.pkl              ← sklearn probes (pickle)
│
└── results/{benchmark}/{split}/{task}/seed{seed}/{model_id_safe}/
    ├── linear_probing.json    ← MethodResult
    ├── csd.json               ← MethodResult
    └── diagnostics.json       ← DiagnosticResult
```

---

## 6. Method 类型体系

```mermaid
graph TD
    AM["AnalysisMethod\n(abstract)"]
    AM --> COM["CacheOnlyMethod\n不需要 GPU at eval"]
    AM --> MOM["ModelOnlineMethod\n需要模型在线"]
    COM --> LP["LinearProbing\n• trained=True\n• needs_answer_space=True\n• train() 单独调用"]
    MOM --> CSD_C["CSD\n• trained=False\n• single_token=PREFERRED\n• 两条执行路径"]

    subgraph "CSD 双路径"
        CSD_C --> Batch["Batch: patchscope_lens\n(nnsight_ops, 快)"]
        CSD_C --> Fallback["Fallback: manual hooks\n(HF forward, 慢)"]
    end

    subgraph "Probing 生命周期"
        LP --> Train["train()\n外部脚本调用\n→ FitArtifact + model.pkl"]
        LP --> Eval["run()\nOrchestrator 调用\neval_only, 加载 artifact"]
    end

    style AM fill:#f5f5f5,stroke:#999
    style COM fill:#e3f2fd,stroke:#1976d2
    style MOM fill:#fff8e1,stroke:#f9a825
```

---

## 7. 数据流：一个样本的完整旅程

```mermaid
sequenceDiagram
    participant DG as datagen
    participant AD as adapter
    participant CB as cache_builder
    participant LP as LinearProbing
    participant CSD as CSD
    participant DX as diagnostics

    Note over DG: S0: 生成 raw dict
    DG->>AD: {code, answer, mechanism, depth, ...}
    AD->>AD: 转为 Sample(id, prompt, answer, difficulty)

    Note over CB: S1: 提取 hidden states
    CB->>CB: model.forward(prompt)<br/>收集每层 last-token activation
    CB-->>CB: HiddenStateCache(N,L,D)<br/>+ model_predictions

    Note over LP,CSD: S2: 两个方法分别运行
    LP->>LP: 每层: probe.predict(h[l])<br/>→ SampleMethodResult(layer_values=[0,0,1,1,...])
    CSD->>CSD: 每层: inject h[l] into target prompt<br/>→ SampleMethodResult(layer_values=[0,0,0,1,...])

    Note over DX: S3: 诊断（独立后处理）
    DX->>DX: FPCL = first layer probe correct (e.g. layer 2)
    DX->>DX: FJC = first layer both correct (e.g. layer 3)
    DX->>DX: Δ_brew = FJC − FPCL = 1
    DX->>DX: classify → Resolved / Overprocessed / ...
    DX-->>DX: SampleDiagnostic → DiagnosticResult
```

---

## 8. 六个任务

```mermaid
graph LR
    subgraph "Data Flow"
        VT["value_tracking<br/>值传递追踪"]
        CP["computing<br/>多步算术"]
    end
    subgraph "Control Flow"
        CD["conditional<br/>分支选择"]
        FC["function_call<br/>⭐ 新任务<br/>函数内计算"]
    end
    subgraph "Data + Control"
        LP_T["loop<br/>循环累加"]
        LU["loop_unrolled<br/>循环展开"]
    end

    VT <-.->|"对照：值传递 ± 函数内计算"| FC
    LP_T <-.->|"对照：循环 ± 循环语法"| LU
```

每个任务统一：**27 配置 × 150 samples/config = 4,050 样本**，答案均为单 digit (0-9)。

每个任务的 difficulty 由三个维度组成（SubsetSpec.difficulty_schema），维度值的笛卡尔积 = 27 配置：

| 任务 | 维度 1 | 维度 2 | 维度 3 |
|------|--------|--------|--------|
| value_tracking | mechanism (3) | depth (3) | distractors (3) |
| computing | structure (3) | steps (3) | operators (3) |
| conditional | branch_type (3) | depth (3) | condition_type (3) |
| function_call | mechanism (3) | depth (3) | distractors (3) |
| loop | body_type (3) | iterations (3) | init_offset (3) |
| loop_unrolled | body_type (3) | iterations (3) | init_offset (3) |

---

## 9. 论文实验规模

```
6 tasks × 9 models × {Probing, CSD} = 108 组 MethodResult
                                      → 54 组 DiagnosticResult
```

每组 MethodResult 包含 4,050 个 SampleMethodResult，每个有 L 层的逐层指标。

---

## 10. 已知设计问题

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | **MethodResult 双模态**：per_sample 和 aggregate 字段共存于一个 class，大量字段互斥为 None | `schema/results.py` | 可读性差，容易误用 |
| 2 | **训练元数据冗余**：FitArtifact.fit_metrics vs MethodResult.fit_metrics_summary 是同一数据的副本 | `results.py` + `linear_probing.py` | 一致性风险 |
| 3 | **answer_space 散落四处**：BenchmarkSpec / LinearProbing.DIGIT_CLASSES / MethodConfig.config / CSD default | 多个文件 | 应有单一来源 |
| 4 | **model_predictions 只存在 Cache 中**，但 S3 诊断需要它 → S3 必须依赖 cache 文件 | `outcome.py` + `types.py` | 破坏 S3 解耦设计 |
| 5 | **FitArtifact.train_cache_id** 永远是 None | `linear_probing.py:277` | 死字段 |
| 6 | **MethodConfig.config** 是 untyped dict | `results.py` | 无 schema 校验，silent misconfiguration |
| 7 | **N_CLASSES=11 硬编码** | `linear_probing.py` | 应从 answer_space 推导 |
| 8 | **resolve_artifact_with_policy()** 中 auto/force 模式无调用者 | `resources.py:265` | 死代码 |
| 9 | **无训练脚本**：`LinearProbing.train()` 已实现但无配套的 CLI/脚本入口 | 缺失 | 训练需手动写 Python 调用 |
