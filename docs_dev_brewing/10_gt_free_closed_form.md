# 10. GT-free Resolution Index

> 2026-04-01. 仅用 CSD + Probing 的 layer-wise 分布，不访问 ground truth，判定 Resolved vs Rest。
> 16 models × 6 tasks = 74,520 samples。

## 核心结论

**ρ_mult = (1−ĥ)(1−d̂)·â 在全量数据上 AUC = 0.850**（random = 0.500），最佳模型 Coder-7B 达 0.901。

Layer-wise gradient 信号（information flux、Lyapunov stability）被 endpoint statistics 完全吸收——这本身是一个 publishable finding：brewing 过程的全部信息已编码在 terminal state 中。

## §A  Layer-wise 信号


| 信号               | 公式                                                       | 值域           |
| ---------------- | -------------------------------------------------------- | ------------ |
| CSD entropy      | $H_C^\ell = -\sum_t \Phi_C^\ell[t] \ln \Phi_C^\ell[t]$   | $[0, \ln C]$ |
| Probe entropy    | $H_P^\ell = -\sum_t \Phi_P^\ell[t] \ln \Phi_P^\ell[t]$   | $[0, \ln C]$ |
| Probe–CSD JSD    | $D^\ell = \mathrm{JSD}(\Phi_P^\ell \Vert \Phi_C^\ell)$   | $[0, \ln 2]$ |
| Argmax agreement | $A^\ell = \mathbb{1}[\arg\max \Phi_P = \arg\max \Phi_C]$ | $0,1$        |


## §B  Gradient 信号（Savitzky-Golay, w=7, p=3）


| 信号               | 公式                                      | 含义                          |
| ---------------- | --------------------------------------- | --------------------------- |
| Information flux | $\mathcal{J}^\ell = -\partial_\ell H_C$ | 正值 = 熵在减少（信息 crystallising） |
| JSD velocity     | $\partial_\ell D$                       | probe–CSD 分歧变化率             |


## §C  归一化标量

**Endpoint features**（tail = $[\lfloor 0.75L \rfloor, L)$）：


| 特征          | 定义                                | 含义              |
| ----------- | --------------------------------- | --------------- |
| $\hat{h}$   | $\bar{H}_C^{\text{tail}} / \ln C$ | 尾部 CSD 残余不确定性   |
| $\hat{h}_P$ | $\bar{H}_P^{\text{tail}} / \ln C$ | 尾部 Probe 残余不确定性 |
| $\hat{d}$   | $\bar{D}^{\text{tail}} / \ln 2$   | 尾部 probe–CSD 分歧 |
| $\hat{a}$   | $\bar{A}^{\text{tail}}$           | 尾部 argmax 一致率   |


**Gradient features**：


| 特征                           | 定义                                                                                           | 含义                     |
| ---------------------------- | -------------------------------------------------------------------------------------------- | ---------------------- |
| $\hat{\jmath}_{\text{peak}}$ | $\max_\ell \mathcal{J}^\ell / (\ln C / L)$                                                   | 归一化峰值 information flux |
| $\hat{\jmath}_{\text{int}}$  | $\frac{1}{\ln C}\int_0^L \max(\mathcal{J}^\ell, 0) d\ell$                                    | 累积正向信息通量               |
| $\sigma_{\text{tail}}$       | $\mathrm{std}(H_C^{\text{tail}}) / \ln C$                                                    | 尾部熵稳定性                 |
| $\Lambda$                    | $\exp\big(-\mathrm{Var}(H_C^{\text{tail}}) / \sigma_0^2\big)$, $\sigma_0 = \frac{\ln C}{4L}$ | Lyapunov 稳定性           |


## §D  Resolution Functional

### 定义

$$\boxed{\rho = \Phi_{\text{state}} \cdot \left(1 + \beta \Phi_{\text{path}}\right)}$$

**Terminal convergence** — 终态是否达到了 resolution：

$$\Phi_{\text{state}} = \underbrace{(1 - \hat{h})}*{\text{CSD certainty}} \cdot \underbrace{(1 - \hat{d})}*{\text{probe–CSD alignment}} \cdot \underbrace{\hat{a}}_{\text{argmax consensus}}$$

Resolution 需要三个条件 **同时** 成立。乘积天然实现 AND 语义——任一因子趋零则 $\rho \to 0$。

**Path quality** — 是否经历了清晰的 phase transition：

$$\Phi_{\text{path}} = \mathcal{J}^{\alpha} \cdot \Lambda^{1-\alpha}$$

其中：

$$\mathcal{J} = \frac{1}{\ln C}\int_0^L \max\left(-\frac{\partial H_C}{\partial \ell}, 0\right) d\ell \qquad \text{(information flux integral)}$$

$$\Lambda = \exp\left(-\frac{\mathrm{Var}(H_C^{\text{tail}})}{\sigma_0^2}\right) \qquad \text{(Lyapunov stability)}$$

- $\mathcal{J}$：forward pass 过程中被消除的总熵（归一化到 $[0,1]$）。Resolved $\to \mathcal{J} \approx 1$。
- $\Lambda$：尾部熵方差的指数惩罚。收敛到 fixed point $\to \Lambda \approx 1$；overprocessing 导致晚层振荡 $\to \Lambda < 1$。
- $\alpha = 0.5$，$\beta = 0.3$。

### 解读

$\Phi_{\text{state}}$ 认证终点；$\Phi_{\text{path}}$ 认证旅途。$(1 + \beta\Phi_{\text{path}})$ 结构确保 gradient 信息只能 **加分**——终态优秀但动态不清晰的 sample 仍有 $\rho \approx \Phi_{\text{state}}$；同时经历了清晰 phase transition 的 sample 获得额外 boost。

## §E  Feature Ablation


| Feature                                     | Type     | AUC       | Dir | 说明             |
| ------------------------------------------- | -------- | --------- | --- | -------------- |
| $\hat{h}_P$ (probe entropy)                 | endpoint | **0.760** | −   | 最强单信号          |
| $\hat{a}$ (argmax agree)                    | endpoint | **0.743** | +   |                |
| $\hat{h}$ (CSD entropy)                     | endpoint | **0.720** | −   |                |
| $\hat{\jmath}_{\text{int}}$ (flux integral) | gradient | 0.706     | +   | 最强 gradient 信号 |
| $\sigma_{\text{tail}}$ (tail stability)     | gradient | 0.674     | +   |                |
| $\Lambda$ (Lyapunov)                        | gradient | 0.670     | −   |                |
| $\hat{\jmath}_{\text{peak}}$ (peak flux)    | gradient | 0.652     | +   |                |
| $\hat{d}$ (JSD)                             | endpoint | 0.613     | −   |                |
| $\mu$ (monotonicity)                        | gradient | 0.503     | +   | **无用**         |


**关键发现**：Gradient 信号的最佳 AUC (0.706) 低于 endpoint 信号的最佳 (0.760)。组合后 gradient 对 ρ 的增益可忽略 — **endpoint statistics 已经完全 subsume 了 path dynamics 的信息**。

## §F  结果

### 全模型汇总


| Model              | N         | Res%      | ρ AUC     | F1        |
| ------------------ | --------- | --------- | --------- | --------- |
| Qwen2.5-Coder-7B   | 4860      | 40.3%     | **0.901** | 0.794     |
| Qwen2.5-Coder-14B  | 1620      | 49.6%     | 0.883     | **0.827** |
| Qwen3-4B-Base      | 4860      | 39.7%     | 0.882     | 0.748     |
| Qwen3-8B-Base      | 4860      | 43.7%     | 0.877     | 0.795     |
| Qwen2.5-Coder-1.5B | 4860      | 24.3%     | 0.875     | 0.577     |
| CodeLlama-7b       | 4860      | 23.0%     | 0.875     | 0.586     |
| Qwen2.5-7B         | 4860      | 35.3%     | 0.866     | 0.673     |
| Qwen2.5-Coder-3B   | 4860      | 34.8%     | 0.863     | 0.702     |
| Qwen3-1.7B-Base    | 4860      | 25.8%     | 0.863     | 0.613     |
| Qwen2.5-1.5B       | 4860      | 19.0%     | 0.862     | 0.557     |
| **ALL**            | **74520** | **29.2%** | **0.850** | **0.619** |


Random baseline: AUC = 0.500

### Gradient 对 AUC 的影响


| Index                        | 公式                                                                                          | AUC       |
| ---------------------------- | ------------------------------------------------------------------------------------------- | --------- |
| ρ_mult (endpoint only)       | $(1-\hat{h})(1-\hat{d})\hat{a}$                                                             | **0.850** |
| ρ_path (endpoint + gradient) | $\Phi_{\text{state}} \cdot (1 + 0.3 \cdot \mathcal{J}^{0.5} \Lambda^{0.5})$                 | 0.850     |
| ρ_geo (Boltzmann)            | $\exp(-\frac{1}{2}[\hat{h}+\hat{h}_P+\hat{d}])$                                             | 0.836     |
| ρ_dyn (multiplicative gates) | $(1-\hat{h})(1-\hat{d}) \cdot \tanh(\hat\jmath_{\text{peak}} \mu) \cdot e^{-\lambda\Omega}$ | 0.686     |


## §G  Why Gradients Don't Help — and Why That's Interesting

Gradient 信号被 endpoint 吸收的原因不是巧合，而是 brewing process 的结构性必然：

1. **Entropy conservation**：$\hat{\jmath}_{\text{int}} \approx (H_C^{\text{early}} - H_C^{\text{tail}})/\ln C$。Information flux integral 本质上就是 early minus tail entropy — 完全可由 endpoint 差值重建。
2. **Lyapunov stability 退化**：对于 discrete layers（L=24~~48），tail window 仅有 6~~12 层。在如此短的序列上，$\mathrm{Var}(H_C^{\text{tail}})$ 几乎不包含 tail entropy mean 之外的额外信息。
3. **信息论解释**：如果 brewing 是一个 **ergodic process**（几乎所有走向同一终态的路径都是相似的），那么 terminal state 就包含了 path 的全部信息。Endpoint sufficiency 意味着 brewing dynamics 是高度 path-independent 的 — 不同 sample 用不同的 layer 开始 resolve，但终态的 statistical signature 是一致的。

**这本身是一个 publishable finding**：GT-free resolution detection 只需要 terminal-state statistics，不需要显式建模 layer dynamics。

但 gradient 在 **四类判别** 中重新发挥作用——见 §H。

## §H  四类 Outcome 的闭式判别

### 核心观察

Binary Resolved detection 中 gradient 无用，是因为 OT 和 Unresolved 同属 "Rest" 互相抵消。但在 **Rest 内部做 OT vs Unresolved 区分时，gradient 是唯一能区分 "从未降过熵" 和 "降了又升回来" 的信号**。

### 四个判别函数

每个 outcome 对应一个闭式 discriminant function，全部是归一化 $[0,1]$ 量的乘积：

$$\boxed{\rho = (1-\hat{h})(1-\hat{d}) \cdot \hat{a}} \qquad \textbf{Resolution Index}$$

CSD certain × probe aligned × argmax agrees。高 $\rho$ → **Resolved**。

$$\boxed{\mu = (1-\hat{h}) \cdot \hat{d} \cdot (1-\hat{a})} \qquad \textbf{Misresolution Index}$$

CSD certain × probe **divergent** × argmax **disagrees**。$\mu$ 是 $\rho$ 的 "镜像"——共享 $(1-\hat{h})$ certainty factor，但 agreement 项全部取反。高 $\mu$ → **Misresolved**。

$$\boxed{\omega = \hat{\jmath}_{\text{int}} \cdot \hat{a}} \qquad \textbf{Overprocessing Index}$$

Entropy **was** removed (high $\hat\jmath_{\text{int}}$, gradient 信号) × some argmax agreement **persists** (residual $\hat{a}$)。OT 样本 "曾经 resolve 过"——$\hat\jmath_{\text{int}}$ 保留了 brewing 的痕迹，$\hat{a}$ 保留了 alignment 的残余。高 $\omega$ → **Overprocessed**。

**$\omega$ 是 gradient 信号唯一不可替代的位置**：endpoint 无法区分 "ĥ 高因为从未降过" 和 "ĥ 高因为降了又升"，只有 $\hat\jmath_{\text{int}} = \frac{1}{\ln C}\int_0^L \max(-\partial_\ell H_C, 0)\, d\ell$ 能做到。

剩余样本按 $\hat{s}$（peak sharpening speed）分为 **No\_Brewing**（$\hat{s}$ 低）和 **Unresolved**。

### 判别规则

$$\text{outcome} = \begin{cases}
\textbf{Resolved} & \text{if } \rho \geq \tau_\rho \\
\textbf{Misresolved} & \text{if } \mu \geq \tau_\mu \\
\textbf{Overprocessed} & \text{if } \omega \geq \tau_\omega \\
\textbf{No\_Brewing} & \text{if } \hat{s} < \tau_s \\
\textbf{Unresolved} & \text{otherwise}
\end{cases}$$

### 各判别函数的 Binary AUC

| Discriminant | 子任务 | AUC |
|-------------|-------|-----|
| $\rho$ | Resolved vs Rest | **0.888** |
| $\mu$ | Misresolved vs Res+OT | **0.852** |
| $\omega$ | OT vs Unresolved | **0.700** |
| $\hat{s}$ | No\_Brewing vs Unresolved | 0.511 |

$\rho$ 和 $\mu$ 都非常强。$\omega$ 对最难的 OT/Unresolved 问题提供了 0.700 的 AUC（random=0.5），**这是 gradient 信号的独特贡献**。$\hat{s}$ 对 NB/Unresolved 区分力弱——这两类在信号空间中确实高度相似。

### 五类闭式判别结果

74,520 samples（16 models × 6 tasks）：

| | Precision | Recall | F1 | Support |
|---|-----------|--------|-----|---------|
| **Resolved** | 0.72 | 0.78 | 0.75 | 21,755 |
| **Unresolved** | 0.60 | 0.68 | 0.64 | 24,320 |
| **Overprocessed** | 0.41 | 0.35 | 0.38 | 21,337 |
| **Misresolved** | 0.30 | 0.42 | 0.35 | 3,307 |
| **No\_Brewing** | 0.15 | 0.13 | 0.14 | 3,801 |
| **Overall** | **Acc ≈ 0.55** | | **κ ≈ 0.38** | 74,520 |

Random baseline: Acc ≈ 0.200, κ = 0.000

### 结构性分析

公式的对称性值得注意：$\rho$ 和 $\mu$ 是 **对偶** 的——

$$\rho = (1-\hat{h}) \cdot (1-\hat{d}) \cdot \hat{a} \qquad \longleftrightarrow \qquad \mu = (1-\hat{h}) \cdot \hat{d} \cdot (1-\hat{a})$$

两者共享 certainty gate $(1-\hat{h})$，在 alignment 维度上是 complement（$\hat{d} \leftrightarrow 1-\hat{d}$，$\hat{a} \leftrightarrow 1-\hat{a}$）。这反映了 Resolved 和 Misresolved 的对称本质：**CSD 都自信，区别仅在于是否与 probe 对齐**。

而 $\omega = \hat\jmath_{\text{int}} \cdot \hat{a}$ 打破了这种 endpoint 对称——它引入了 path integral $\hat\jmath_{\text{int}}$，编码了 "曾经发生过 brewing" 这一不可从终态完全恢复的信息。

## 文件


| 文件                                                                  | 说明                                     |
| ------------------------------------------------------------------- | -------------------------------------- |
| `scripts/gt_free_resolution_index.py`                               | 四个 resolution index + feature ablation |
| `scripts/gt_free_closed_form.py`                                    | 四类闭式判别（探索性）                            |
| `brewing_output/artifacts/gt_free_v2/resolution_index_results.json` | 结果                                     |


