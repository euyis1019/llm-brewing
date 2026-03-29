# 15. Causal Validation Design Checklist for Brewing

**Date**: 2026-03-29
**Goal**: 为 COLM 论文所需的 3 组因果验证补齐 Brewing 的框架设计清单，并明确哪些地方应借鉴 `pyvene`，哪些地方不应直接照搬。

---

## 1. 结论先说

根据 `COLM_REQUIREMENTS.md`，Brewing 目前缺的不是“某个 activation patching 函数”，而是整套 **causal validation 子系统**。

COLM 需要的 3 组因果验证是：

1. `Activation Patching at FJC`
2. `Layer Skipping (Early Exit)`
3. `Re-injection`

对应原始要求：

| 实验 | 输入 | 输出 | 验证什么 |
|------|------|------|---------|
| Activation Patching at FJC | Model + Samples + FJC | `flip_rate` | FJC 是因果特权层 |
| Layer Skipping (Early Exit) | Model + Overprocessed samples + FJC | `restoration_rate` | Overprocessed 可恢复 |
| Re-injection | Model + Unresolved samples | `rescue_rate_per_round` | Unresolved 含未完成计算 |

这些实验都不适合直接塞进当前 `AnalysisMethod -> MethodResult -> Diagnostics` 主链，因为它们依赖：

- 已有的 `FJC`
- 已有的 `Outcome`
- 在线模型干预
- 干预前后对比指标

也就是说，它们是 **S3 之后的验证层**，而不是普通 S2 方法。

---

## 2. 当前框架为什么还接不住

### 2.1 当前 mode 不包含 causal validation

现在的 `RunConfig.mode` 只有：

- `cache_only`
- `train_probing`
- `eval`
- `diagnostics`

没有任何因果验证专用 mode。

### 2.2 当前结果类型不表达“干预前后对比”

当前 schema 只有两类主结果：

- `MethodResult`: 单个方法在逐层上的输出
- `DiagnosticResult`: FPCL / FJC / Outcome 的后处理结果

但 causal validation 需要表达的是：

- 哪个样本被选中
- 在哪个层、哪个位置做了什么干预
- 干预前输出是什么
- 干预后输出是什么
- 是否发生 flip / restore / rescue
- 按实验聚合后的成功率

这些都不是 `MethodResult` 或 `DiagnosticResult` 现在的职责。

### 2.3 当前 backend 只有“取激活”和“patchscope readout”

`brewing/nnsight_ops.py` 现在已经具备一个很好的起点：

- 取层输出
- 收集中间激活
- 对 target prompt 做 patchscope-style 注入

但还没有统一的“因果干预 backend 抽象”，因此：

- layer skipping 只能临时写 hook
- re-injection 只能临时拼逻辑
- 不同实验会重复写 layer access / patch / collect / generate

### 2.4 这三项不应该继续挂在 `methods/` 下面硬做

`methods/` 里的抽象默认假设输入是：

- `eval_samples`
- `eval_cache`
- 可选 `train_samples` / `train_cache`

而这 3 个验证实验真正需要的输入是：

- `samples`
- `HiddenStateCache`
- `DiagnosticResult`
- `probe_result`
- `csd_result`
- 在线模型
- 实验特定的 intervention config

所以从语义上看，它们更像：

- `validators/`
- `causal/`
- 或 `interventions/`

而不是已有的 `methods/`。

---

## 3. 建议的总设计

建议新增一个独立子系统：

```text
brewing/
├── causal/
│   ├── __init__.py
│   ├── base.py
│   ├── config.py
│   ├── results.py
│   ├── selectors.py
│   ├── activation_patching.py
│   ├── layer_skipping.py
│   ├── reinjection.py
│   └── backend.py
└── pipelines/
    └── causal_validation.py
```

核心原则：

1. 因果验证是新 mode，不是 `eval` mode 的一个 method。
2. 因果验证消费已有 S1/S2/S3 产物，不重复计算 probe / diagnostics。
3. 干预 backend 与实验逻辑分开。
4. 单样本明细和聚合指标都要落盘。

---

## 4. 最少需要新增的抽象

### 4.1 新 mode: `causal_validation`

建议把 `VALID_MODES` 扩成：

```python
("cache_only", "train_probing", "eval", "diagnostics", "causal_validation")
```

这个 mode 的职责：

1. 读取 `DiagnosticResult`
2. 读取 `linear_probing.json`
3. 读取 `csd.json`
4. 读取 `HiddenStateCache`
5. 在线加载模型
6. 根据实验类型筛样本
7. 执行干预
8. 落盘单样本结果和聚合结果

### 4.2 新配置对象: `CausalValidationConfig`

建议不要把所有参数塞进 `method_configs`。新增单独配置更清楚。

最小字段建议：

```python
@dataclass
class CausalValidationConfig:
    experiment: str  # activation_patching_fjc | layer_skipping | reinjection
    selector: dict
    intervention: dict
    decoding: dict
    save_traces: bool = False
```

其中：

- `selector`: 定义如何筛样本
- `intervention`: 定义 patch / skip / reinject 的层位和策略
- `decoding`: 定义如何读取最终输出、是否只看 next token

### 4.3 新结果对象: `CausalValidationResult`

建议新增两个 dataclass：

```python
@dataclass
class SampleCausalResult:
    sample_id: str
    experiment: str
    selected: bool
    reason: str | None
    source_layer: int | None
    target_layer: int | None
    round_idx: int | None
    original_output: str | None
    intervened_output: str | None
    original_correct: bool | None
    intervened_correct: bool | None
    effect_label: str | None  # flipped / restored / rescued / no_effect
    extras: dict = field(default_factory=dict)


@dataclass
class CausalValidationResult:
    experiment: str
    model_id: str
    eval_dataset_id: str
    benchmark: str
    subset: str | None = None
    sample_results: list[SampleCausalResult] = field(default_factory=list)
    summary_metrics: dict[str, float] = field(default_factory=dict)
    config: dict = field(default_factory=dict)
```

这样能表达：

- per-sample 证据
- 论文里的汇总指标
- 运行配置

### 4.4 新 backend 抽象: `InterventionBackend`

当前 `nnsight_ops.py` 已经是很好的原型，但还不够统一。建议提炼为：

```python
class InterventionBackend(ABC):
    def collect_hidden(self, model, prompts, layers, positions): ...
    def patch_hidden(self, model, prompt, layer, position, hidden): ...
    def skip_layer(self, model, prompt, layer): ...
    def reinject_hidden(self, model, prompt, plan): ...
    def decode(self, model, prompt, max_new_tokens=1): ...
```

然后先做一个 `NNSightInterventionBackend`，内部复用现有：

- `get_token_activations`
- `patchscope_lens`
- `get_layer_output`
- `get_logits`

这样做的好处是，三类实验共享一套模型访问抽象。

---

## 5. 三个实验分别该怎么设计

### 5.1 Activation Patching at FJC

#### 实验语义

对每个样本，在它的 `FJC` 层取出正确相关 hidden state，并注入到指定 target context 中，看最终输出是否发生预期 flip。

#### 依赖输入

- `samples`
- `eval_cache`
- `DiagnosticResult` 中的 `fjc`
- 在线模型

#### 样本筛选

只选：

- `fjc is not None`
- answer 为单 token
- cache / diagnostics / method results 对齐

#### 最小配置

```yaml
causal_validation:
  experiment: activation_patching_fjc
  selector:
    require_fjc: true
  intervention:
    source_layer: fjc
    source_position: last
    target_prompt: '# The value of x is "'
    target_position: last
    patch_type: replace
  decoding:
    max_new_tokens: 1
```

#### 输出指标

- `flip_rate`
- `n_selected`
- `n_effective`

#### 需要的 backend 能力

- 单层 hidden replacement
- 干预后 next-token / generation 读取

#### 备注

这项最接近当前 CSD 的 patchscope 逻辑，应该第一个落地。

---

### 5.2 Layer Skipping (Early Exit)

#### 实验语义

对 `Overprocessed` 样本，在 `FJC` 或其附近层做 early exit / skip，观察最终答案是否被恢复。

#### 依赖输入

- `samples`
- `DiagnosticResult` 中的 `outcome`
- `fjc`
- 在线模型

#### 样本筛选

只选：

- `outcome == Overprocessed`
- `fjc is not None`

#### 最小配置

```yaml
causal_validation:
  experiment: layer_skipping
  selector:
    outcome: overprocessed
    require_fjc: true
  intervention:
    skip_from: fjc_plus_1
    skip_to: end
    mode: early_exit
  decoding:
    max_new_tokens: 1
```

#### 输出指标

- `restoration_rate`
- `restored_count`
- `n_selected`

#### 需要的 backend 能力

- 对单层或层区间应用 skip
- 或从某层 hidden 直接接 LM head / 继续最短路径解码

#### 架构提醒

这项不要直接用“hook 后返回原输入”这种临时方案硬做。应该在 backend 层定义清楚：

- `skip single layer`
- `skip layer range`
- `early exit decode`

否则后面不同模型架构会很难统一。

---

### 5.3 Re-injection

#### 实验语义

对 `Unresolved` 样本，从某些候选层读取 hidden state，再按 round 逐步 reinject，观察能否把原本未解出的样本 rescue 成正确答案。

#### 依赖输入

- `samples`
- `eval_cache`
- `DiagnosticResult` 中的 `outcome`
- 在线模型

#### 样本筛选

只选：

- `outcome == Unresolved`

#### 最小配置

```yaml
causal_validation:
  experiment: reinjection
  selector:
    outcome: unresolved
  intervention:
    source_layers: [0, 1, 2, 3, 4, ...]
    target_prompt: '# The value of x is "'
    reinject_rounds: [single, iterative]
    selection_policy: scan_all_layers
  decoding:
    max_new_tokens: 1
```

#### 输出指标

- `rescue_rate_per_round`
- `best_rescue_rate`
- `round_to_first_rescue`

#### 需要的 backend 能力

- 多轮注入计划
- 每轮独立读取输出
- 保留 round-level trace

#### 架构提醒

这项是三者里最不适合塞进普通 `MethodResult` 的，因为它天然是：

- 多轮
- 多候选层
- 多次干预

所以结果对象里必须有 `round_idx` 和 round-level extras。

---

## 6. 资源落盘建议

建议不要把这类结果混进现有 `results/.../{method}.json`。

建议新增路径：

```text
{output_root}/causal/
└── {benchmark}/{split}/{task}/seed{seed}/{model_id_safe}/
    ├── activation_patching_fjc.json
    ├── layer_skipping.json
    └── reinjection.json
```

原因：

- 这些结果不是 method result
- 它们依赖 diagnostics 之后的筛样
- 单独目录更利于论文汇总脚本消费

`ResourceManager` 最少需要新增：

- `causal_result_path(key, experiment)`
- `resolve_causal_result(...)`
- `save_causal_result(...)`

---

## 7. Pipeline 该怎么接

建议新增 `CausalValidationPipeline`，它的主流程应该是：

1. resolve eval dataset
2. resolve eval cache
3. load `linear_probing` result
4. load `csd` result
5. load `diagnostics.json`
6. 根据 `experiment` 选择 validator
7. 调 backend 执行 intervention
8. 保存 `CausalValidationResult`

这个 pipeline 的输入依赖顺序很明确：

```text
S1 cache
 + S2 probing/csd
 + S3 diagnostics
 -> causal validation
```

也就是说，causal validation 是 **S4** 更准确。

---

## 8. 为什么建议单独做 `causal/` 而不是扩展 `methods/`

因为二者的语义不同：

| 维度 | `methods/` | `causal/` |
|------|------------|-----------|
| 目标 | 读出或估计逐层性质 | 做干预并看行为变化 |
| 典型输入 | cache, samples | cache, samples, diagnostics, online model |
| 输出 | `MethodResult` | `CausalValidationResult` |
| 是否依赖 FJC / Outcome | 否 | 是 |
| 是否包含 counterfactual | 否 | 是 |

如果强行并入 `methods/`，会造成两个问题：

1. `AnalysisMethod.run()` 的签名被污染
2. `MethodResult` 变成什么都装的杂项容器

---

## 9. 可以从 pyvene 借的 4 个设计模式

下面是最值得借鉴的地方。

### 9.1 借“声明式干预配置”，不要借整套重量级模型包装

`pyvene` 的一个核心优势是把干预声明成可序列化配置对象。官方文档里有：

- `IntervenableConfig`
- `RepresentationConfig`

这类对象把“在哪个 layer / component / unit 上做什么 intervention”从执行逻辑里剥离出来。

对 Brewing 来说，最值得借的是这个思想：

- 用 dataclass 表达 intervention plan
- plan 可落盘、可复现、可比较

不必一开始就完整引入 `pyvene` 的 `IntervenableModel` 包装。

### 9.2 借“抽象锚点 -> 模型具体模块”的映射层

`pyvene` 文档里明确把各模型的 intervenable modeling 文件定义为：

> abstract naming of intervention anchor points -> actual model module

这对 Brewing 很有启发。我们现在的 `CSD` fallback 还存在模型内部结构硬编码风险，所以 causal validation 更应该先抽象：

- `residual_stream`
- `mlp_output`
- `attn_output`
- `block_output`

然后由 backend 去映射到：

- Qwen
- Llama
- DeepSeek

的真实模块路径。

### 9.3 借“intervention class families”

`pyvene` 已经把很多干预算子类型化了，例如：

- `CollectIntervention`
- `ConstantSourceIntervention`
- `SkipIntervention`
- `AdditionIntervention`
- `ZeroIntervention`

Brewing 不需要全部照搬，但至少应该把自己的最小集合类型化：

- `ReplaceHidden`
- `SkipLayer`
- `ReinjectHidden`
- `CollectHidden`

这样测试和复用都会简单很多。

### 9.4 借“输出对象同时保留 original 与 intervened”

`pyvene` 的输出对象强调同时保留：

- original outputs
- intervened outputs
- collected activations

这正是 causal validation 需要的。Brewing 的 `SampleCausalResult` 也应该保留：

- `original_output`
- `intervened_output`
- `original_correct`
- `intervened_correct`

而不是只留一个聚合率。

---

## 10. 不建议直接照搬 pyvene 的地方

### 10.1 不建议现在直接把 Brewing 完全改造成 pyvene-style model wrapper

原因：

- Brewing 已经有 `nnsight_ops.py`
- CSD 主路径已经围绕 nnsight 建起来了
- 全量切换会把“补齐 COLM 实验”变成“重写底层执行框架”

这不是当前最短路径。

### 10.2 不建议先引入过多 trainable intervention 抽象

COLM 需要的三项里，真正必要的是：

- activation replacement
- layer skipping
- reinjection

而不是：

- 旋转子空间
- trainable masks
- SAE latent interventions

这些可以以后再扩。

### 10.3 不建议让每个实验自己直接碰模型内部路径

这会复制 `CSD` 里已经暴露过的问题：

- dtype
- device
- module path
- tuple output / tensor output

都应该被 backend 封装。

---

## 11. 建议的实现顺序

### Phase 1: 先补 backend 和 schema

- 新增 `CausalValidationResult` / `SampleCausalResult`
- 新增 `causal_validation` mode
- 新增 `CausalValidationPipeline`
- 提炼 `InterventionBackend`

### Phase 2: 先做 Activation Patching at FJC

原因：

- 最接近现有 CSD patchscope
- 工程风险最低
- 最容易尽快打通 end-to-end

### Phase 3: 再做 Layer Skipping

原因：

- 只比 activation patching 多一个 skip operator
- 但仍是单轮干预

### Phase 4: 最后做 Re-injection

原因：

- 需要 round-level 结果
- 需要最复杂的配置和结果结构

### Phase 5: 补测试

至少要有：

- selector 单测
- schema round-trip 单测
- backend smoke tests
- per-experiment fixture tests
- 一条完整 causal pipeline e2e

---

## 12. 我建议的最小 MVP

如果目标是尽快让 COLM 主实验能跑，我建议 MVP 只做：

1. 新增 `causal_validation` mode
2. 新增 `CausalValidationResult`
3. 新增 `NNSightInterventionBackend`
4. 先实现 `Activation Patching at FJC`
5. 结果落盘后再接 `Layer Skipping`
6. `Re-injection` 最后单独补

这样做的原因是：

- 不会打断现在 S0-S3 主链
- 最快验证新架构是否顺手
- 可以先把论文最核心的因果验证入口建立起来

---

## 13. 具体代码改动清单

建议的代码落点：

- `brewing/schema/results.py`
  - 新增 `SampleCausalResult`
  - 新增 `CausalValidationResult`
  - `VALID_MODES` 增加 `causal_validation`

- `brewing/resources.py`
  - 新增 causal result 路径与 save/load

- `brewing/pipelines/causal_validation.py`
  - 新增 causal pipeline

- `brewing/pipelines/__init__.py`
  - 注册 `causal_validation`

- `brewing/causal/base.py`
  - 抽象 validator 接口

- `brewing/causal/backend.py`
  - 抽象 `InterventionBackend`

- `brewing/causal/nnsight_backend.py`
  - 用现有 `nnsight_ops.py` 实现 backend

- `brewing/causal/selectors.py`
  - `select_fjc_samples`
  - `select_overprocessed_samples`
  - `select_unresolved_samples`

- `brewing/causal/activation_patching.py`
  - 实现 `ActivationPatchingAtFJC`

- `brewing/causal/layer_skipping.py`
  - 实现 `LayerSkippingValidation`

- `brewing/causal/reinjection.py`
  - 实现 `ReinjectionValidation`

---

## 14. 参考资料

### Brewing 本地代码

- `COLM_REQUIREMENTS.md`
- `brewing/schema/results.py`
- `brewing/pipelines/__init__.py`
- `brewing/nnsight_ops.py`
- `brewing/methods/csd.py`
- `docs_dev_brewing/5b_phase1_cleanup.md`

### pyvene 官方资料

- GitHub README: https://github.com/stanfordnlp/pyvene
- Intervenable base API: https://stanfordnlp.github.io/pyvene/api/pyvene.models.intervenable_base.html
- IntervenableModel API: https://stanfordnlp.github.io/pyvene/api/pyvene.models.intervenable_base.IntervenableModel.html
- Intervenable config API: https://stanfordnlp.github.io/pyvene/api/pyvene.models.configuration_intervenable_model.html
- RepresentationConfig API: https://stanfordnlp.github.io/pyvene/api/pyvene.models.configuration_intervenable_model.RepresentationConfig.html
- Interventions API: https://stanfordnlp.github.io/pyvene/api/pyvene.models.interventions.html

### 本次参考到的 pyvene 设计点

- README 中强调 intervention 是基础原语、可组合、可序列化
- `build_intervenable_model` 的 factory 设计
- `IntervenableConfig` / `RepresentationConfig` 的声明式配置
- `CollectIntervention` / `SkipIntervention` / `ConstantSourceIntervention` 这类 typed intervention
- `IntervenableModelOutput` 同时保留 original 与 intervened outputs

---

## 15. 最后的判断

Brewing 现在离“能支撑 COLM 的三组因果验证”还差的，不是一个 patch 函数，而是四层东西：

1. mode
2. result schema
3. intervention backend
4. experiment-specific validators

如果只能先做一件事，就先把 **`causal_validation` mode + `CausalValidationResult` + Activation Patching at FJC** 打通。
