# 1. 初始化数据结构 + Fixture 数据

## 做了什么

创建 Brewing 框架的核心数据结构和最小 fixture 数据（每个 subset 1 条 sample）。

## 文件

| 文件 | 内容 |
|------|------|
| `Brewing/brewing/__init__.py` | 包初始化 |
| `Brewing/brewing/data.py` | 核心数据结构：`Sample`, `BenchmarkSpec`, `SubsetSpec`, `AnswerMeta`, `MethodRequirements`, `MethodConfig`, enums (`AnswerType`, `SingleTokenRequirement`, `Outcome`) |
| `Brewing/brewing/benchmarks/__init__.py` | 包初始化 |
| `Brewing/brewing/benchmarks/cue_bench.py` | CUE-Bench 定义（6 个 SubsetSpec + AnswerMeta）+ 6 条 fixture samples |

## 数据结构来源

字段定义遵循 `docs/workflow_and_schema.md` §2。

## Fixture samples

| subset | id | answer | difficulty |
|--------|----|--------|-----------|
| pure_copying | `pure_copying_fixture_000` | 7 | steps=2, padding=0 |
| computing | `computing_fixture_000` | 5 | steps=2, density=1.0, padding=0 |
| conditional | `conditional_fixture_000` | 2 | nesting=1, cond_type=> |
| function_call | `function_call_fixture_000` | 3 | depth=1, body=identity, distractors=0 |
| loop | `loop_fixture_000` | 4 | iter=3, op=+1 |
| loop_unrolled | `loop_unrolled_fixture_000` | 4 | iter=3, op=+1 |

loop 和 loop_unrolled 的 fixture 是同一道题（a=1, +1 三次 → 4），方便对照。
