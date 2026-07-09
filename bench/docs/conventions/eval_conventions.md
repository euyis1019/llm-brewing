# Eval 约定：唯一入口 submit_eval

> 评测**只走一个入口**：`python -m src.scripts.suite.submit_eval`。
> 不要手写 hope、不要再调任何局部 `generate_eval_hopes.py`（已删）、不要直接调
> `run_suite_src.sh`/`run_suite_external.sh`（旧 server/client 路径，仅作回退）。

## 为什么有这条约定

eval protocol（7 个 bench：5 protocol + 2 code）早就稳定、从未变过，但"哪个 bench
用哪个 driver、要哪些 env、什么资源"这套知识曾散落在脚本注释和人脑里，导致重复犯低级
错误（code bench 套错 driver 缺 `DATASET`/`METRIC`、路径漂移、资源配错）。`submit_eval`
把这套知识收进**一个入口 + 一个 suite YAML**，调用方只给 model/tag/output，其余全自动。

## 协议事实（入口替你守住，无需记忆）

| bench | runner | driver 分派 | 默认资源 |
|---|---|---|---|
| gsm8k, ceval, cmmlu, cruxeval, mmlu | `protocol` | 进程内 `inproc_runner` | TP2 / 65G / 48c / 2gpu |
| humaneval_plus, mbpp_plus | `external` | evalplus `run_code_eval.sh` | TP4 / 400G / 64c / 4gpu |

- 默认引擎 vLLM 的执行层是 `src/eval/infra/run_eval.sh`（单 job、进程内 vLLM、连续批处理打满 GPU）。
  **无 server/client、无 `BENCH_CONCURRENCY`**——历史上"并发调低就 GPU 空转被 kill"的坑在此引擎下不存在。
  另有 FluentLLM 引擎（`--engine fluentllm`，见下文「引擎（engine）」节），是 server/client 架构、用
  `BENCH_CONCURRENCY=500`。上表「默认资源」也是 vLLM 引擎的值；FluentLLM 两类 bench 统一 TP4/128G/36c/4gpu。
- avg@N：进程内 runner 把一轮全题 prompt 提交 `llm.generate()`，按 `ceil(N/batch_cap)` 分批，
  每批每题复制 ≤`batch_cap`（默认 3，suite `spec.batch_cap` 控制）份。avg@3 = 1 批。
- code bench 的 `DATASET`/`METRIC` 由 suite 的 `runner: external` + `framework/dataset/metric`
  字段决定，`src.eval.suites` 强制校验（漏填直接报错）。
- 产物：`OUTPUT_BASE/{model_tag}/{bench_id}/{run_NN/result.json, summary.json}`，
  汇总只看 `summary.json` 顶层 `accuracy_mean`（avg@N 单值；见 memory：不要 std/min/max）。

## 标准用法：submit → cron collect → 汇总

入口分两个**解耦、不阻塞**的子命令。轮询交给 agent 用 cron 编排，入口本身永不挂起。

### 1. submit（一次性，立即返回）

```bash
python -m src.scripts.suite.submit_eval submit \
    --suite base_model_eval_v1 \
    --model-path <绝对路径/到/pruned_model> \
    --model-tag <稳定 tag，如 04b_twostage_r028> \
    --output-dir <绝对路径/eval_output> \
    --engine vllm|fluentllm    # 推理引擎，必填、无默认；见「引擎」节
    [--bench gsm8k mmlu ...]   # 只跑子集，默认全 7 个
    [--tp N --mem MB --gpus N --vcore N]  # 协议默认的逃生口，常态不填
    [--batch-cap N] [--n-runs 3]
    [--dry-run]                # 只生成 hope 不提交，先看 manifest
```

返回一张 `{bench → run_id → 预期 summary.json 路径}` 清单后立即退出。

### 2. collect（幂等，可反复调）

```bash
python -m src.scripts.suite.submit_eval collect \
    --suite base_model_eval_v1 \
    --model-tag <同上> \
    --output-dir <同上>
```

- 齐了：打印 `accuracy_mean` 表（avg@N）+ 末行 AVG，退出码 0。
- 没齐：列出缺哪个 bench，退出码非零。

### 3. 轮询编排（agent 用 cron）

submit 后，agent 用 `CronCreate` 每隔 ~10min（用偏移分钟，避开 :00/:30）调一次 collect；
collect 退出码 0（齐了）就汇总结果、`CronDelete` 停掉 cron。**不要**在入口里 while-sleep 阻塞，
也**不要**短间隔空轮询。

## Suite

`configs/eval_suites/benchmark/{name}.yaml`。标准 7-bench 评测用 `base_model_eval_v1`。新 suite 复用
`src.eval.suites` 格式：`spec.benchmarks` 列 bench（external 必须带 `framework/dataset/metric`），
可选 `spec.resources`（按 runner 覆盖默认）、`spec.batch_cap`。resources/batch_cap 是执行旋钮，
不属于行为契约（prompt/scorer/stop 仍在各 bench 的 `benchmark_meta.json`）。

## 引擎（engine）

`--engine` 选推理引擎，**是整次提交的属性，不是 per-bench**（suite 里每个 bench 不区分引擎）。
**必填、无默认**——提交者必须显式声明（引擎是模型本身的事实：LongCat 必须 `fluentllm`，
Qwen 类用 `vllm`），靠路径猜引擎会静默选错、跑出垃圾分。两种引擎共用同一 suite、同一
`benchmark_meta.json` 行为契约、同一 collect——区别只在执行层（driver / 镜像 / 资源 / 是否起 server）。

| | `vllm`（默认） | `fluentllm` |
|---|---|---|
| 执行模型 | 进程内 vLLM，`llm.generate()` | SGLang server + HTTP client |
| driver | `src/eval/infra/run_eval.sh` | `src/eval/infra/run_eval_fluentllm.sh` |
| 镜像 | `serving_…_vllm_…:1.0.2` | `serving_fluentllm_master_…:1.0.3` |
| 并发 | 无（连续批处理自动打满） | `BENCH_CONCURRENCY=500` |
| protocol 资源 | TP2 / 65G / 48c / 2gpu | TP4 / 128G / 36c / 4gpu |
| external 资源 | TP4 / 400G / 64c / 4gpu | TP4 / 128G / 36c / 4gpu |
| code 生成 | evalplus 直跑（in-proc） | `gen_code_completions.py` 打 endpoint |

- protocol bench：两引擎都经 `src.eval.benchmark.client_runner` 的行为契约
  （vLLM 走 `inproc_runner`，FluentLLM 走 `run_bench_from_meta(mode=remote_vllm_service)`）。
- code bench：两引擎都经 `evalplus` + `src.eval.external.code_eval normalize` 归一化，
  只是 FluentLLM 多一步 `gen_code_completions.py` 向 server 取补全。
- 何时用哪个：默认 vLLM；模型需 FluentLLM/SGLang 专属支持（如特定 MoE/远程代码）时用 `--engine fluentllm`。
- `--engine fluentllm` 时 `--tp/--batch-cap` 不生效（server TP 固定 4，启动参数在 driver 内）。

## 相关文件

- 入口：`src/scripts/suite/submit_eval.py`
- 进程内 runner（vLLM）：`src/eval/benchmark/inproc_runner.py`（复用 `client_runner.build_prompt/score_response`）
- vLLM driver：`src/eval/infra/run_eval.sh`
- FluentLLM driver：`src/eval/infra/run_eval_fluentllm.sh`（server/client，复用 `client_runner` + `code_eval`）
- FluentLLM code 生成：`src/eval/external/gen_code_completions.py`（打 endpoint 取补全）
- code bench worker（vLLM external）：`src/eval/external/run_code_eval.sh`（evalplus，未改）
- suite 校验：`src/eval/suites.py`
- 汇总：`src/eval/infra/compute_summary.py`
- LongCat 模型补丁：`src/eval/infra/{make_patched_model.sh,patch_longcat_model.py}`；evalplus 源：`src/eval/infra/reference_evalplus`（软链）

> 执行层（driver / 汇总 / 模型补丁 / evalplus 源）已统一收口到 `src/eval/infra/`，
> 不再依赖 `experiments/12_.../eval_infra`（旧文件留原地未删，仅作历史参考）。

## 不要做

- ❌ 手写 hope 或拼绝对路径——交给 `submit_eval`。
- ❌ 调旧的 `run_suite_src.sh`/`run_suite_external.sh`（仅回退，新路是 `run_eval.sh`）。
- ❌ 给 code bench 配 `run_suite_src.sh` / 漏 `DATASET`/`METRIC`——入口按 runner 自动注入。
- ❌ 在入口里阻塞轮询——用 cron 调 collect。
