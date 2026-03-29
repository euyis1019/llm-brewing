# 14. Probing 准确率瓶颈诊断

**日期**：2026-03-29

## 问题

Eric 要求将 probing 在验证集上的最终准确率提升到接近 100%（1.5B 模型至少 >90%）。
指出训练数据量不够（甚至小于参数量）、应当使用深度学习方法、可以尝试非线性 probe。

## 第一步：Train/Eval Split 偏置检查

Eric 担心 train/eval 数据存在分类偏置（未 shuffle、split 节点有特征）。

验证结论：
1. **Split 经过了 shuffle**——ID 在两边交错分布（train 拿 idx 0,2,3,5,6; eval 拿 1,4,15,17…）
2. **Label 分布无显著偏差**——Chi-squared 检验 6 个任务全部 p > 0.05（最低 loop p=0.0545）
3. **每个 config 在两边都有代表**——不存在单边 config
4. **零 overlap**——train 和 eval 的 code 和 id 完全不重叠

**结论：数据 split 无偏。**

## 第二步：Baseline 建立

### 数据规模

每个 task：train=3240, eval=810，共 27 configs × 120/30 samples。

对于 7B（hidden_dim=3584）: n/p = 3240/3584 = 0.9，严重欠定。

### 各方法在 7B / value_tracking 上的 best-layer eval accuracy

| 方法 | Best Eval | Train@best | 耗时 |
|------|-----------|------------|------|
| **PyTorch Linear (wd=0.01)** | 43.8% | ~98% | 13s |
| **PyTorch Linear (wd=0.1)** | 44.4% | ~95% | 14s |
| PyTorch Linear (wd=1.0) | 42.0% | ~80% | 14s |
| PyTorch Linear (wd=10.0) | 36.5% | ~60% | 12s |
| PyTorch Bottleneck MLP (dim=32, wd=0.1) | 44.0% | — | 17s |
| PyTorch Bottleneck MLP (dim=64, wd=0.1) | 43.5% | — | 16s |
| PyTorch Dropout(0.5) + Linear (wd=0.1) | 44.4% | — | 14s |
| PyTorch Dropout(0.5) + Bottleneck(64) | 44.3% | — | 16s |
| sklearn Ridge (α=100) | 46.0% | 98.3% | ~30s/layer |
| sklearn Ridge (α=1000) | 44.3% | 90.2% | ~30s/layer |
| sklearn LDA (Ledoit-Wolf shrinkage) | 45.1% | 89.0% | 260s/layer |
| sklearn LogisticRegression (C=0.01) | 44.7% | 95.3% | ~60s/layer |
| PCA(256) + PyTorch Linear | 16.5% | — | ~10s/layer |
| Gaussian noise augment (×8, σ=0.1) + Ridge | 44.1% | 98.7% | — |
| Cross-task pooling (19440 samples, Ridge) | 41% (vt) | 66% | — |
| Logit Lens (RMSNorm + lm_head, 无训练) | 38.1% | N/A | 即时 |

**所有方法都收敛到 ~44% eval 的 ceiling。**

### MLP 非线性 Probe 对比（3240 train）

| Model | Probe | value_tracking | loop |
|-------|-------|---------------|------|
| 1.5B | linear | 31.5% | 40.6% |
| 1.5B | mlp_small (64) | 31.5% | 45.3% |
| 1.5B | mlp_med (256→64) | 31.2% | 46.5% |
| 1.5B | mlp_large (512→128) | 30.0% | 47.4% |
| 7B | linear | 44.7% | 55.6% |
| 7B | mlp_small (64) | 45.1% | 55.3% |
| 7B | mlp_med (256→64) | 42.1% | 56.5% |
| 7B | mlp_large (512→128) | 44.0% | 57.3% |

**结论：非线性 probe 无显著提升。**

## 第三步：扩充数据排除 overfitting

生成 10× 数据（samples_per_config=1500 → train=32400, eval=8100），重建 7B cache。

| Probe | Best Eval | Train@best | 备注 |
|-------|-----------|------------|------|
| Linear (wd=0.1) | **43.2%** | 48.4% | overfitting 消除（gap 从 50%→5%），但 eval 没提升 |
| MLP (64, BN, Dropout) | 38.1% | 41.0% | 更差 |

**结论：~44% 是 7B/value_tracking 的 representation ceiling，不是 overfitting。**

## 第四步：根因定位

### 模型 next-token 准确率（eval set）

| Model | value_tr | computing | conditional | function | loop | loop_unr | avg |
|-------|----------|-----------|-------------|----------|------|----------|-----|
| 1.5B | 15% | 3% | 9% | 4% | 4% | 4% | **6%** |
| 1.5B-I | 15% | 2% | 11% | 4% | 6% | 10% | **8%** |
| 3B | 25% | 2% | 17% | 8% | 14% | 10% | **13%** |
| 7B | 25% | 3% | 20% | 14% | 22% | 20% | **17%** |

### 关键实验：按模型预测正确/错误分组（全任务 × 1.5B / 7B）

Best-layer linear probe (wd=0.1) 的 eval accuracy，按模型是否预测正确分组：

**1.5B (hidden_dim=1536)**

| Task | All | Model-Correct | N_corr | Model-Incorrect | N_incorr | ModelAcc |
|------|-----|---------------|--------|-----------------|----------|----------|
| value_tracking | 31.2% | **91.8%** | 122 | 20.5% | 688 | 15.1% |
| computing | 17.8% | 9.1% | 22 | 18.0% | 788 | 2.7% |
| conditional | 30.5% | **73.7%** | 76 | 26.0% | 734 | 9.4% |
| function_call | 21.0% | **69.0%** | 29 | 19.2% | 781 | 3.6% |
| loop | 41.6% | **81.2%** | 32 | 40.0% | 778 | 4.0% |
| loop_unrolled | 41.4% | 58.8% | 34 | 40.6% | 776 | 4.2% |

**7B (hidden_dim=3584)**

| Task | All | Model-Correct | N_corr | Model-Incorrect | N_incorr | ModelAcc |
|------|-----|---------------|--------|-----------------|----------|----------|
| value_tracking | 44.7% | **95.1%** | 206 | 27.5% | 604 | 25.4% |
| computing | 18.5% | 23.8% | 21 | 18.4% | 789 | 2.6% |
| conditional | 43.2% | **85.9%** | 163 | 32.5% | 647 | 20.1% |
| function_call | 25.6% | **66.7%** | 111 | 19.0% | 699 | 13.7% |
| loop | 56.3% | **83.5%** | 176 | 48.7% | 634 | 21.7% |
| loop_unrolled | 57.8% | **89.1%** | 165 | 49.8% | 645 | 20.4% |

**观察**：

1. **value_tracking / conditional / loop / loop_unrolled**：model-correct 子集上 probe 准确率 70-95%，说明 probe 能有效提取模型已计算出的信息
2. **computing**：即使 model-correct 的 21 个样本，probe 也只有 9-24%——可能 computing 的答案编码方式不同于其他任务，或者样本量太少（21 个）导致统计噪声
3. **function_call**：7B 的 model-correct 子集上 66.7%（111 个样本），还有提升空间
4. **loop / loop_unrolled**：model-incorrect 子集上也有 40-50% 的 probe 准确率，说明即使模型没有输出正确 token，部分样本的 hidden states 里仍然有答案信息（这正是 "Overprocessed" outcome 的候选）

**根因**：Probe 本身工作正常（model-correct 子集上普遍 70-95%）。整体准确率低是因为模型只在少部分样本上正确推理出了答案——其余样本的 hidden states 里缺乏正确答案信息。

## 涉及的脚本

| 文件 | 用途 |
|------|------|
| `scripts/probe_experiment.py` | v1 实验：baseline + PCA + MLP 对比 |
| `scripts/probe_v2.py` | v2 实验：regularization sweep（weight_decay, bottleneck, dropout, label smoothing） |
| `scripts/rebuild_and_probe.py` | 完整 pipeline：datagen → cache build → probe train/eval |

## 待决定方向

1. **改用 Instruct 模型 + 更明确的 prompt format**
   - 当前 base 模型对 `# The value of x is "` 没有 follow-instruction 的能力
   - Instruct 模型 + chat template 可能大幅提升模型准确率
   - 但 1.5B-Instruct 目前准确率也只有 8%，需要更好的 prompt engineering

2. **降低任务难度**
   - 只保留模型能力范围内的简单配置（depth=1, no distractors）
   - 模型正确率上升 → probe 整体准确率也会上升

3. **使用更大/更强的模型**
   - 14B 模型（已有权重但未 build cache）
   - DeepSeek-Coder-6.7B, CodeLlama-7B 等跨架构模型

4. **重新审视实验设计**
   - Probing 的本意是测量 "information availability"，而非达到 100% 分类准确率
   - 95% 的 model-correct 子集准确率可能本身就是正确的实验结论
   - Brewing-to-resolution 框架的四类 outcome（Resolved / Overprocessed / Misresolved / Unresolved）正是要刻画这种差异

**等待 Eric 确认下一步方向。**
