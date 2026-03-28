# 9. Externalize Probing Training from Evaluation Runs

**Commit**: `ced52e0` — 2026-03-28
**Scope**: 8 files changed, +668 / -344

## 动机

之前 probing 的训练（fit）和评估（eval）耦合在同一个 pipeline run 中：Orchestrator 自动做 train/eval split、构建训练 cache、调 `resolve_artifact_with_policy` 隐式训练。这带来几个问题：

1. **隐式数据切分**：eval run 里自动 `train_test_split`，用户无法控制训练集的构成
2. **pipeline 职责模糊**：Orchestrator 既负责 eval 又负责 fit，`run()` 的语义不清晰
3. **可复现性风险**：同一次 run 可能因为 fit_policy=auto 在不同机器上走不同分支（加载 vs 训练）

## 核心改动

### LinearProbing: fit/eval 彻底分离

- **新增 `train()` 方法**：显式训练入口，接收 train_samples + train_cache，持久化 probe artifact
  - 已有 artifact 时默认拒绝覆盖，需 `overwrite=True`
  - 校验 train_samples 和 train_cache 数量一致
- **`run()` 变为 eval-only**：
  - 只接受 `fit_policy=eval_only`，其他值直接报错
  - 不再接受 train_samples / train_cache 参数
  - 必须有 `train_dataset_id` 以定位 pre-trained artifact
  - 加了 answer_space 和 layer count 的一致性校验（artifact vs eval cache）

### Orchestrator: 移除所有训练相关逻辑

- **删除 `_resolve_train_dataset()`**：~80 行，含 `train_test_split`、train manifest 保存、eval_samples 原地替换
- **删除 S0b（train dataset）和 S1b（train cache）**：不再传 train_samples/train_cache 给 method
- **`_resolve_eval_cache` → `_resolve_hidden_cache`**：重命名，语义更清晰
- **新增 `_train_dataset_id_for_subset()`**：只负责生成 artifact ID 字符串，支持 `{subset}` 模板

### CLI: 全面切换到 YAML-config-only

- **移除所有 argparse flags**（除 `--config` 和 `--verbose`）
- **新增 `load_config()`**：读 YAML → `RunConfig`，含类型校验
- **新增 `needs_model_online()`**：通过 registry 查方法需求，替代原来硬编码 `"csd"` 的判断
- **新增 `build_model_load_kwargs()`**：根据 `quantization` 配置构造 model loading 参数

### RunConfig 变更

- `fit_policy` 默认值: `"auto"` → `"eval_only"`
- `train_split`: 标记为 deprecated，`__post_init__` 中设值直接抛异常
- 新增 `quantization` 字段（`None / "int8" / "int4"`），含白名单校验
- 新增 `__post_init__` 做统一的配置验证

## 新增测试

| 文件 | 行数 | 覆盖内容 |
|------|------|---------|
| `test_cli.py` | 201 | load_config 各种边界、needs_model_online 走 registry、build_model_load_kwargs 三种量化、CLI argparse 集成 |
| `test_linear_probing.py` | 148 | run 无 artifact 报错、非 eval_only 报错、fit 防覆盖、answer_space 不匹配校验 |
| `test_orchestrator.py` | 33 | train_dataset_id 模板格式化、默认 ID 含 subset+seed |
| `test_e2e.py` (改) | — | 适配显式 fit→eval 流程 |

## 设计决策

1. **为什么不保留 fit_policy=auto？** 隐式训练使 eval run 的行为不可预测——同样的 config 在有/无 artifact 的环境下走完全不同的代码路径。eval_only 是唯一合理的 eval-time 策略。
2. **为什么 CLI 不再支持 argparse flags？** 与 doc `8_yaml_config_and_quantization.md` 中的设计一致：一个 YAML config 完整描述一次 run，消除 CLI flag 和 config 之间的优先级歧义。
3. **quantization 放在 RunConfig 而非 CLI？** 量化策略是 run 的一部分，应该在 config 中声明和版本化，不应该作为 transient CLI flag。

## 改动文件清单

- `Brewing/brewing/methods/linear_probing.py` — fit/eval 分离
- `Brewing/brewing/orchestrator.py` — 移除训练逻辑，精简 pipeline
- `Brewing/brewing/schema/results.py` — RunConfig 新字段 + 验证
- `Brewing/brewing/cli.py` — YAML-config-only 重写
- `Brewing/tests/test_cli.py` — 新增
- `Brewing/tests/test_linear_probing.py` — 新增
- `Brewing/tests/test_orchestrator.py` — 新增
- `Brewing/tests/test_e2e.py` — 适配新 API
