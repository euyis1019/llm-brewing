# datasets/benchmark/normalized/

标准化 benchmark 数据目录。存放经过框架格式转换后的 bench JSONL 文件和元数据。

## 目录结构

```
normalized/
  README.md                  ← 本文件
  slim_qwen_test/            ← slim_qwen_test benchmark 的标准化数据
    benchmark_meta.json      ← benchmark 层级结构和 leaf bench 元数据
    mmlu__slim5.jsonl        ← 5 rows, multiple_choice
    mmlu__slim5.manifest.json
    mmlu_pro__slim5.jsonl
    mmlu_pro__slim5.manifest.json
    bbh__slim5.jsonl         ← 5 rows, generate (CoT)
    bbh__slim5.manifest.json
    gsm8k__slim5.jsonl       ← 5 rows, generate
    gsm8k__slim5.manifest.json
    ceval__slim5.jsonl       ← 5 rows, multiple_choice
    ceval__slim5.manifest.json
    cmmlu__slim5.jsonl       ← 5 rows, multiple_choice
    cmmlu__slim5.manifest.json
```

## 约定

- 每个可执行 bench 对应一个 JSONL 文件（每行一个 `SlimBenchRow`）
- 每个 JSONL 旁边有对应的 `.manifest.json`，记录选题规则和来源信息
- `benchmark_meta.json` 描述 benchmark 层级（`BenchmarkMeta` 对象），列出所有 leaf bench 及其 data_path
- 这里的数据是**只读产物**，由 `src/scripts/suite/prepare_bench_data.py` 生成

## 数据来源

原始数据在：

```
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtai/users/guoyifu02/dataset/benchmark_aligned/exports/
```

转换脚本：

```bash
python src/scripts/suite/prepare_bench_data.py --benchmark slim_qwen_test
```

## 与 experiments 的关系

- `experiments/05_framework_evaluate/` 消费这里的数据，不持有原始转换逻辑
- `experiments/05_framework_evaluate/slim_data/` 保留为软链接或副本（向后兼容），但标准化数据的权威位置是本目录

## 新增 benchmark 的流程

1. 在 `src/scripts/suite/prepare_bench_data.py` 里增加新 benchmark 的 preparer 函数
2. 运行 prepare 脚本，把 JSONL 和 manifest 写到 `normalized/<benchmark_name>/`
3. 生成 `benchmark_meta.json`（脚本自动完成）
4. 在 `configs/eval_suites/benchmark/` 下新建 suite YAML，task 的 `data_path` 指向本目录
