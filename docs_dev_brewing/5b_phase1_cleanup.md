# Phase 1 Cleanup: 结构收敛与碎片清理

**日期**: 2026-03-28
**范围**: `Brewing/brewing/` 目录结构清理，承接 `5a_phase1_structure_refactor.md`
**目标**: 消除 5a 遗留的过度拆分、兼容 shim、边界穿透问题

---

## 1. schema/ 合并：9 文件 → 3 文件

5a 将 `data.py`（539 行）拆成 9 个文件，平均每个 ~60 行，过于碎片化。

合并为 3 个文件：

| 文件 | 内容 | 行数 |
|------|------|------|
| `types.py` | 7 个枚举 + `Sample` + `AnswerMeta` + `DatasetManifest` + `HiddenStateCache` + `FitArtifact` + `save/load_samples` | ~243 |
| `results.py` | `SampleMethodResult` + `MethodResult` + `MethodRequirements` + `MethodConfig` + `SampleDiagnostic` + `DiagnosticResult` + `RunConfig` | ~248 |
| `benchmark.py` | `SubsetSpec` + `BenchmarkSpec` + `check_compatibility` | ~71 |

已删除：`enums.py`, `sample.py`, `dataset.py`, `cache.py`, `artifact.py`, `method_result.py`, `diagnostics.py`, `compat.py`

---

## 2. config/ 子包 → YAML 配置目录

`RunConfig` 是一个 dataclass，不需要独立子包。

- `RunConfig` 移入 `schema/results.py`
- 删除 `config/__init__.py` 和 `config/run_config.py`
- `config/` 保留为空目录（`.gitkeep`），未来存放 YAML 实验配置文件

---

## 3. testing/ 子包 → tests/

`make_synthetic_cache` 是测试辅助函数，不应在框架包内。

- `brewing/testing/synthetic_cache.py` 移至 `tests/helpers.py`
- `cache_builder.py` 中的 re-export 已删除
- `brewing/testing/` 目录已删除

---

## 4. data.py 兼容 shim 删除

3000 行的项目不需要维护 shim 层。

- 所有 `from brewing.data import ...` 已全局替换为 `from brewing.schema import ...`
- `brewing/data.py` 已删除

---

## 5. diagnostics.py → diagnostics/ 包

247 行拆为两个文件，为后续因果验证方法（activation patching, layer skipping, re-injection）预留扩展位。

| 文件 | 内容 |
|------|------|
| `metrics.py` | `compute_fpcl`, `compute_fjc`, `compute_csd_tail_confidence`, `TAIL_FRACTION` |
| `outcome.py` | `classify_outcome`, `diagnose_sample`, `run_diagnostics`, `group_diagnostics_by_difficulty`, `MISRESOLVED_THRESHOLD` |
| `__init__.py` | re-export 所有公开函数和常量 |

---

## 6. _cue_bench_legacy.py 删除

5a 重构后遗留的旧文件，无任何 import 引用，直接删除。

---

## 7. datagen/ 移入 benchmarks/cue_bench/

`datagen/` 本质是 CUE-Bench 的数据生成脚本，散落在仓库根目录，靠 `sys.path.insert` hack 才能被 Brewing import。

- 9 个 .py 文件复制到 `benchmarks/cue_bench/datagen/`
- `builder.py` 中 `importlib.import_module("datagen.xxx")` 改为相对 import（`package="brewing.benchmarks.cue_bench"`）
- `cli.py` 和 `test_e2e.py` 中指向 CUE 根目录的 `sys.path.insert` 已删除
- 原始 `/home/gyf/CUE/datagen/` 未删除，由用户自行处理

---

## 重构后目录结构

```
brewing/
├── __init__.py
├── __main__.py
├── cli.py
├── orchestrator.py
├── resources.py
├── registry.py
├── cache_builder.py
├── nnsight_ops.py
├── config/
│   └── .gitkeep                    # 未来存放 YAML 实验配置
├── schema/
│   ├── __init__.py
│   ├── types.py                    # 枚举 + 基础数据容器
│   ├── results.py                  # 方法/诊断结果 + RunConfig
│   └── benchmark.py                # benchmark 定义 + 兼容性检查
├── diagnostics/
│   ├── __init__.py
│   ├── metrics.py                  # FPCL, FJC, tail confidence
│   └── outcome.py                  # outcome 分类 + 聚合
├── methods/
│   ├── __init__.py
│   ├── base.py
│   ├── linear_probing.py
│   └── csd.py
└── benchmarks/
    ├── __init__.py
    └── cue_bench/
        ├── __init__.py
        ├── spec.py                 # SubsetSpec 定义 + CUE_BENCH BenchmarkSpec
        ├── fixtures.py             # 测试用 fixture samples
        ├── adapter.py              # datagen 输出 → Sample 转换
        ├── builder.py              # 数据集加载/生成
        └── datagen/                # 数据生成脚本（从根目录移入）
            ├── __init__.py
            ├── base.py
            ├── generate.py
            ├── value_tracking.py
            ├── computing.py
            ├── conditional.py
            ├── function_call.py
            ├── loop.py
            └── loop_unrolled.py
```

---

## 测试结果

33/33 tests passed，无修改测试逻辑。
