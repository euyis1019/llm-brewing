# Brewing Config 目录

## 快速入门

```bash
# 所有命令从 Brewing/ 目录运行
conda activate cue
python -m brewing --config brewing/config/<config>.yaml --verbose
```

## 目录结构

```
config/
├── README.md                       ← 你在这里
├── example_full_reference.yaml     ← 全参数参考（不能直接跑，用于查阅）
├── example_single_task.yaml        ← 最小可运行示例（fixture 小样本）
├── example_probing_tune.yaml       ← Probing eval 示例
├── example_local_model.yaml        ← 指定本地模型路径示例
├── example_14b_int8.yaml           ← int8 量化示例
├── example_anchor.yaml             ← Anchor 方法示例
└── colm/                           ← COLM 2026 论文实验 config
    ├── README.md
    ├── qwen25_coder_*.yaml         ← 各模型的 eval config
    └── train_probing_*.yaml        ← 各模型的 probe 训练 config
```

## 4 种模式 × 典型 config

一个 config 对应一个 mode。按执行顺序：

### 1. `cache_only` — 提取 hidden states

只跑 S0→S1，不做分析。适合批量预提取。

```yaml
mode: cache_only
benchmark: CUE-Bench
subsets: [value_tracking, computing]
model_id: Qwen/Qwen2.5-Coder-7B
model_cache_dir: /path/to/cue/models
output_root: brewing_output
seed: 42
batch_size: 8
```

需要：模型在线（除非 cache 已存在）
产出：`brewing_output/caches/cuebench/{split}/{task}/seed42/{model}/hidden_states.npz`

### 2. `train_probing` — 训练 probe

用 train split 的 cache 训练 per-layer probe，产出 artifact。

```yaml
mode: train_probing
benchmark: CUE-Bench
subsets: [value_tracking]
model_id: Qwen/Qwen2.5-Coder-1.5B-Instruct
model_cache_dir: /path/to/cue/models
output_root: brewing_output
seed: 42
batch_size: 16
method_configs:
  linear_probing:
    probe_type: mlp           # linear | mlp
    probe_params:
      lr: 0.001
      epochs: 2000
      batch_size: 512
      weight_decay: 0.1
      patience: 50
    overwrite: false
    validate_on_eval: true    # 训练后在 eval split 上验证
```

需要：train cache 已存在（否则需要模型提取）
产出：`brewing_output/artifacts/cuebench/{task}/{model}/linear_probing/seed42/model.pkl`

### 3. `eval` — 评估

用已训练好的 probe 和 CSD 在 eval split 上推理。

```yaml
mode: eval                    # 默认值，可以省略
benchmark: CUE-Bench
subsets: [value_tracking, computing, conditional, function_call, loop, loop_unrolled]
model_id: Qwen/Qwen2.5-Coder-7B-Instruct
model_cache_dir: /path/to/cue/models
methods: [linear_probing, csd]
output_root: brewing_output
data_dir: /path/to/cue/data/colm_v1/eval
seed: 42
fit_policy: eval_only
batch_size: 8
```

需要：probe artifact 已训练好；eval cache 已存在（否则需要模型）
产出：`brewing_output/results/cuebench/eval/{task}/seed42/{model}/{method}.json`

### 4. `diagnostics` — S3 诊断

从落盘的 MethodResult 跑 FPCL/FJC/Outcome 分类。不需要模型。

```yaml
mode: diagnostics
benchmark: CUE-Bench
subsets: [value_tracking]
model_id: Qwen/Qwen2.5-Coder-7B-Instruct
output_root: brewing_output
seed: 42
```

需要：MethodResult 已存在
产出：`brewing_output/results/cuebench/eval/{task}/seed42/{model}/diagnostics.json`

## method_configs 参数速查

`method_configs.linear_probing` 下的参数按模式分组：

| 参数 | 适用模式 | 默认值 | 说明 |
|------|----------|--------|------|
| `probe_type` | train_probing | `"linear"` | `"linear"` = nn.Linear, `"mlp"` = 两层 MLP |
| `probe_params.lr` | train_probing | `0.001` | Adam 学习率 |
| `probe_params.epochs` | train_probing | `2000` | 最大训练轮数 |
| `probe_params.batch_size` | train_probing | `512` | probe 训练 mini-batch |
| `probe_params.weight_decay` | train_probing | `0.1` | L2 正则化 |
| `probe_params.patience` | train_probing | `50` | early stopping 耐心 |
| `overwrite` | train_probing | `false` | 覆盖已有 artifact |
| `validate_on_eval` | train_probing | `false` | 训练后在 eval split 验证 |
| `answer_space` | train_probing, eval | digits 0-9 | 分类标签空间 |

## 典型工作流

```bash
# Step 1: 提取 hidden states（train + eval）
python -m brewing --config config_cache_train.yaml
python -m brewing --config config_cache_eval.yaml

# Step 2: 训练 probe
python -m brewing --config config_train_probing.yaml --verbose

# Step 3: 评估
python -m brewing --config config_eval.yaml --verbose

# Step 4: 诊断
python -m brewing --config config_diagnostics.yaml
```

## 注意事项

- 一个 config 对应一个模型。多模型用多个 config 文件。
- `model_cache_dir` 指向本地 HF 权重目录，所有实验模型在 `/path/to/cue/models`。
- `batch_size`（顶层）控制 nnsight forward pass 显存；`probe_params.batch_size` 控制 probe 训练 batch。
- `overwrite: true` 用于重新训练，确认无误后改回 `false`。
