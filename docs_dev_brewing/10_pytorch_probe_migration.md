# 10. Probing 训练迁移：sklearn → PyTorch

**日期**：2026-03-29

## 背景

准备批量跑 4 个模型 × 6 任务的 probing 训练（`mode: train_probing`）。
创建了 `brewing/config/colm/train_probing_*.yaml` 共 4 个 config 文件。

## 问题：sklearn LogisticRegression 极慢

首次运行 `train_probing_qwen25_coder_1p5b_base.yaml` 时，第一层就卡了近 3 分钟。

### Benchmark 结果（Layer 14, value_tracking, 1.5B base, 3240 samples × 1536 dim × 10 classes）

| Solver | Scaler | max_iter | 时间 | 备注 |
|--------|--------|----------|------|------|
| lbfgs | 无 | 100 | 59.1s | 未收敛 |
| saga | 无 | 100 | 8.1s | 未收敛 |
| lbfgs | StandardScaler | 200 | 124.5s | 未收敛 |
| saga | StandardScaler | 200 | 15.9s | 未收敛 |
| SGDClassifier (log_loss) | StandardScaler | 1000 | 50.5s | — |

**结论**：sklearn 的所有 solver 在 3240×1536×10 class 规模下都很慢。
28 层 × 60s/层 ≈ 28 分钟/模型，完全不可接受。

## 解决：迁移到 PyTorch nn.Linear

### 方案

- 每层一个 `nn.Linear(hidden_dim, n_classes)` + CrossEntropyLoss
- Adam optimizer，GPU 训练（full-batch 或 mini-batch）
- `StandardScaler` 逻辑内嵌：训练时计算 per-feature mean/std，推理时自动应用
- 包装为 `LinearProbe` 类，暴露 `predict()` / `predict_proba()` / `score()` 与 sklearn 兼容
- 序列化：pickle（与 ResourceManager 的 `save_artifact` / `load_artifact_model` 兼容）

### 速度对比

| 方案 | 设备 | 2000 epochs 单层时间 | 28 层总时间 |
|------|------|---------------------|------------|
| sklearn saga | CPU | ~14s (50 iters) | 28+ min |
| PyTorch | CPU | 24.6s | ~12 min |
| **PyTorch** | **GPU** | **1.1s** | **~1.5 min** |

GPU 加速约 **20×**。

### 正则化探索

初版 PyTorch probe 无正则化，train accuracy 89-91% 但 eval 仅 ~30%。

Grid search 结果（Layer 22, value_tracking, 1.5B base）：

| weight_decay | epochs | train acc | eval acc |
|--------------|--------|-----------|----------|
| 0.01 | 200 | 83.7% | 30.0% |
| 0.01 | 2000 | 89.8% | 30.0% |
| **0.10** | **200** | **63.1%** | **32.1%** |
| 0.10 | 2000 | 63.2% | 32.3% |
| 0.50 | 2000 | 48.7% | 29.8% |
| 1.00 | 2000 | 43.7% | 28.5% |

sklearn C=0.01 + saga 对比：train=50.3%, eval=31.6%。

**结论**：eval ~30% 是 1.5B base 模型在 value_tracking 上的真实上限，不是过拟合问题。
`weight_decay=0.1` 给出最佳 train/eval balance。

### 关于 train/eval distribution shift

调查发现 hidden states 的全局统计量有差异：
- Train L22: mean=0.0255, std=5.91
- Eval L22: mean=0.0233, std=**10.19**

eval 的 activation 方差几乎翻倍。这是 train/eval prompt 结构差异导致的，不是 bug。
per-feature StandardScaler 已经在缓解这个问题。

## 修改文件

| 文件 | 变更 |
|------|------|
| `brewing/methods/linear_probing.py` | sklearn → PyTorch 全面重写：`LinearProbe` 类、`_fit_probes` 用 GPU 训练、`_get_probe_device` 自动检测设备 |
| `brewing/cli.py` | `needs_model_online` 优化：cache 全部就绪时跳过模型加载（新增 `_all_caches_exist`） |
| `brewing/config/colm/train_probing_*.yaml` × 4 | 新建，probe_params 改为 `lr/epochs/batch_size/weight_decay` |
| `brewing/config/colm/qwen25_coder_7b.yaml` | 补充注释 |
| `brewing/pipelines/train.py` | `_validate_on_eval` 加 tqdm 进度条 |

## 当前默认参数

```yaml
probe_params:
  lr: 0.001
  epochs: 2000
  batch_size: 512
  weight_decay: 0.1
```

## 待办

- [ ] 批量跑 4 模型 × 6 任务的 probing 训练
- [ ] 训练完成后将 `overwrite` 改回 `false`
- [ ] 跑 eval 模式验证 probe artifact 能被正确加载和推理
