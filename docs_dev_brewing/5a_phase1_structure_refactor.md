# Phase 1: Structure Refactor

**日期**: 2026-03-27
**范围**: `Brewing/brewing/` 目录结构整理
**目标**: 按 `5_refactor_plan.md` Phase 1 执行纯结构拆分，不改行为

---

## 执行内容

### 1. 拆分 `data.py` → `schema/` 包

原 `data.py`（539 行）包含所有数据结构、枚举、序列化逻辑和兼容性检查。拆为 8 个文件：

| 新文件 | 内容 |
|---|---|
| `schema/enums.py` | 全部 7 个枚举类型 |
| `schema/sample.py` | `Sample`, `save_samples`, `load_samples` |
| `schema/benchmark.py` | `AnswerMeta`, `SubsetSpec`, `BenchmarkSpec` |
| `schema/dataset.py` | `DatasetManifest` |
| `schema/cache.py` | `HiddenStateCache` |
| `schema/artifact.py` | `FitArtifact` |
| `schema/method_result.py` | `SampleMethodResult`, `MethodResult`, `MethodRequirements`, `MethodConfig` |
| `schema/diagnostics.py` | `SampleDiagnostic`, `DiagnosticResult` |
| `schema/compat.py` | `check_compatibility` |
| `schema/__init__.py` | 全部 re-export |

`brewing/data.py` 保留为兼容 shim，仅 re-export `brewing.schema` 中所有名称。

### 2. 拆分 `cue_bench.py` → `benchmarks/cue_bench/` 包

原 `benchmarks/cue_bench.py`（325 行）拆为 4 个文件：

| 新文件 | 内容 |
|---|---|
| `cue_bench/spec.py` | 6 个 SubsetSpec 定义 + `CUE_BENCH` BenchmarkSpec |
| `cue_bench/fixtures.py` | `FIXTURE_SAMPLES`（6 个测试样本） |
| `cue_bench/adapter.py` | datagen ↔ Brewing 转换映射和函数 |
| `cue_bench/builder.py` | `load_generated_dataset`, `generate_and_convert`, `build_eval_dataset` |
| `cue_bench/__init__.py` | 注册 benchmark + 全部 re-export |

原 `cue_bench.py` 重命名为 `_cue_bench_legacy.py` 避免与包目录冲突。

### 3. 提取 `RunConfig` → `config/run_config.py`

`RunConfig` 从 `orchestrator.py` 中拆出，放入 `config/run_config.py`。
`orchestrator.py` 通过 `from .config import RunConfig` 导入并 re-export。

### 4. 提取 `make_synthetic_cache` → `testing/synthetic_cache.py`

`make_synthetic_cache` 从 `cache_builder.py` 移至 `testing/synthetic_cache.py`。
`cache_builder.py` 保留 re-export。

---

## 兼容层

所有旧 import 路径仍然有效：

- `from brewing.data import Sample, ...` → 通过 shim re-export
- `from brewing.orchestrator import RunConfig` → 通过 re-export
- `from brewing.cache_builder import make_synthetic_cache` → 通过 re-export
- `from brewing.benchmarks.cue_bench import FIXTURE_SAMPLES, ...` → 通过包 `__init__.py`

---

## 测试结果

33/33 tests passed，无任何修改测试代码。

---

## 重构后目录结构

```
brewing/
├── __init__.py
├── __main__.py
├── data.py                  # 兼容 shim → schema/
├── cli.py
├── orchestrator.py          # RunConfig 已拆出，re-export 保留
├── resources.py
├── registry.py
├── cache_builder.py         # make_synthetic_cache 已拆出，re-export 保留
├── diagnostics.py
├── nnsight_ops.py
├── schema/                  # NEW: 数据结构按职责拆分
│   ├── __init__.py
│   ├── enums.py
│   ├── sample.py
│   ├── benchmark.py
│   ├── dataset.py
│   ├── cache.py
│   ├── artifact.py
│   ├── method_result.py
│   ├── diagnostics.py
│   └── compat.py
├── config/                  # NEW: 运行配置
│   ├── __init__.py
│   └── run_config.py
├── testing/                 # NEW: 测试辅助
│   ├── __init__.py
│   └── synthetic_cache.py
├── benchmarks/
│   ├── __init__.py
│   ├── _cue_bench_legacy.py # 原文件，已停用
│   └── cue_bench/           # NEW: 拆分为包
│       ├── __init__.py
│       ├── spec.py
│       ├── fixtures.py
│       ├── adapter.py
│       └── builder.py
└── methods/                 # 未动
    ├── __init__.py
    ├── base.py
    ├── linear_probing.py
    └── csd.py
```

---

## 残留问题（留给后续阶段）

1. **`cli.py` 中的 `sys.path.insert`** — Phase 2 内容
2. **`test_e2e.py` 中的 `sys.path.insert`** — Phase 2 内容
3. **`diagnostics.py` 未拆分** — Phase 1 优先级较低，文件 247 行尚可接受
4. **`methods/csd.py` 与 backend 耦合** — Phase 3 内容
5. **`_cue_bench_legacy.py`** — 确认无外部引用后可删除
6. **测试目录未分层** (unit/integration/e2e) — 可作为后续小改进
