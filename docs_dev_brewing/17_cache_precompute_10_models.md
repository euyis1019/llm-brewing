# 17 — Cache Precompute: 10 Models (Qwen2.5-Coder + Qwen3)

## 目标

为 10 个 base 模型预构建 hidden state cache（6 tasks x 2 splits = 12 caches/model），全部 FP16 加载。

## 模型列表

| 模型 | 参数量 | batch_size | 备注 |
|---|---|---|---|
| Qwen2.5-Coder-0.5B | 0.5B | 64 | 新建 |
| Qwen2.5-Coder-1.5B | 1.5B | 32 | 已有，跳过 |
| Qwen2.5-Coder-3B | 3B | 32 | 已有，跳过 |
| Qwen2.5-Coder-7B | 7B | 16 | 已有，跳过 |
| Qwen2.5-Coder-14B | 14B | 8 | 新建 |
| Qwen3-0.6B-Base | 0.6B | 64 | 新建 |
| Qwen3-1.7B-Base | 1.7B | 32 | 新建 |
| Qwen3-4B-Base | 4B | 16 | 新建 |
| Qwen3-8B-Base | 8B | 16 | 新建 |
| Qwen3-14B-Base | 14B | 8 | 新建 |

## 代码改动

### 1. `CacheOnlyPipeline` 支持 `splits` 参数 (`brewing/pipelines/cache_only.py`)
- 之前 hardcode 只处理 eval split
- 现在通过 `config.splits` 参数指定，默认 `["eval"]`
- `RunConfig` 新增 `splits: list[str] | None` 字段

### 2. `cache_only` 模式使用 HF 直接推理 (`brewing/cli.py`)
- 之前所有模式都用 nnsight `LanguageModel` 加载，trace 编译极慢（0.5B 需 5 分钟，14B 需 40+ 分钟）
- `cache_only` 模式只需提取 hidden states，不需要 intervention，改用 `AutoModelForCausalLM`
- 效果：14B 模型从加载到开始推理仅需 3 分钟（之前 40+ 分钟编译）

### 3. 本地模型路径自动解析 (`brewing/cli.py`)
- 当 `model_cache_dir` 指定时，自动构造 `model_cache_dir/model_id` 作为本地路径
- 避免 HuggingFace Hub 的远程校验（xet-read-token 等），直接从本地磁盘读取

### 4. FP16 无条件默认 (`brewing/cli.py`)
- `dtype=torch.float16` 改为无条件设置，不再仅在无 quantization 时生效
- 14B config 去掉了 `quantization: int8`，文件名从 `qwen25_coder_14b_int8.yaml` 改为 `qwen25_coder_14b.yaml`

### 5. nnsight 0.6.3 兼容 (`brewing/cache_builder.py`)
- `.save()` 返回的 proxy 退出 trace context 后直接变成 tensor
- `logits.value` 改为 `logits.value if hasattr(logits, 'value') else logits`

## 新增 Config 文件

`brewing/config/colm/cache_*.yaml` x 10，统一格式：
- `mode: cache_only`
- `splits: [train, eval]`
- `model_cache_dir: /home/gyf/CUE/models`
- `seed: 42`

## 结果

- 10 模型 x 12 caches = 120 个 cache（加上旧 Instruct 模型共 125 个）
- 总存储：66GB
- 硬件：2x RTX 5090 32GB，双卡并行
- 运行时间：约 2 小时（含调试）

## 经验教训

- nnsight trace 编译时间随模型大小超线性增长，cache_only 模式完全不需要它
- batch_size 需要根据模型大小 + hidden_dim 动态调整，大模型容易 OOM
- `from_pretrained` 用 HF Hub ID + `cache_dir` 仍会做远程校验，直接传本地路径更快
