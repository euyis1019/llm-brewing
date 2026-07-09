# eval_benchmark_bundle

一套**协议驱动的 LLM 基准评测 Infra**:7 个 benchmark(5 protocol + 2 code),单一入口
`submit_eval`(submit + collect 两子命令),两套推理引擎(vLLM 进程内 / FluentLLM server-client),
行为契约(prompt/scorer/stop/fewshot)与执行层(driver/资源/汇总)解耦。

> 本 bundle 从一个更大的 model-pruning/merging 实验仓中抽取,**只含评测代码 + 标准化数据集**,
> 不含剪枝/合并代码、不含 PPL 评测、不含校准数据。设计为可迁移到其他平台,迁移时需做少量
> adapt(见末尾「平台适配清单」)。

---

## 1. 目录结构

```
eval_benchmark_bundle/
├── README.md                         ← 本文件
├── src/
│   ├── __init__.py                   ← 包根(submit_eval 以 `python -m src.…` 调用)
│   ├── eval/
│   │   ├── suites.py                 ← Suite YAML 加载 + 校验(单一真相源解析器)
│   │   ├── benchmark/                ← 行为契约层
│   │   │   ├── behavior_catalog.py     prompt_builder / scorer 查表(协议实现)
│   │   │   ├── models.py               EvalRow / BenchmarkProtocol / BenchMeta 数据模型
│   │   │   ├── compiler.py             plan-time 编译:校验 meta、解析 protocol
│   │   │   ├── loader.py               模型解析(pruned 检测、original_path 校验)
│   │   │   ├── client_runner.py        HTTP client runner,build_prompt()/score_response()
│   │   │   ├── inproc_runner.py        vLLM 进程内 runner(复用 client_runner 的 prompt/score)
│   │   │   ├── bench_tree.py           bench 树模型(LeafBench vs BenchmarkNode)
│   │   │   └── slim_runner.py          legacy slim bench runner(仅回退)
│   │   ├── external/                  ← code bench 子系统(evalplus)
│   │   │   ├── code_eval.py            evalplus 适配器,normalize 子命令归一 pass@1→accuracy
│   │   │   ├── gen_code_completions.py FluentLLM 专用,HTTP 打 endpoint 取代码补全
│   │   │   └── run_code_eval.sh        GPU worker:evalplus codegen + evaluate + normalize
│   │   ├── infra/                     ← 执行层
│   │   │   ├── run_eval.sh             ★ vLLM driver(进程内,按 RUNNER 分派)
│   │   │   ├── run_eval_fluentllm.sh   ★ FluentLLM driver(server + HTTP client)
│   │   │   ├── compute_summary.py      N runs → summary.json(mean/std/min/max)
│   │   │   ├── make_patched_model.sh   LongCat 影子模型构建(flock 守护)
│   │   │   ├── patch_longcat_model.py  LongCat remote-code 补丁
│   │   │   └── reference_evalplus/     vendored evalplus 源(实体,已去 .git)
│   │   ├── cli.py                     ← PPL 入口(本 bundle 无 PPL 数据,死代码,见 §6)
│   │   ├── registry.py                ← PPL runner 工厂(同上)
│   │   ├── perplexity.py / ppl/       ← PPL 计算(同上)
│   │   └── __init__.py
│   └── scripts/suite/
│       ├── submit_eval.py             ★★★ 唯一入口(submit + collect)
│       ├── prepare_bench_data.py      一次性:raw → normalized JSONL + benchmark_meta.json
│       ├── normalize_cruxeval.py      一次性:CRUXEval raw → EvalRow
│       ├── plan_benchmark.py          plan-time:生成 execution_plan.json(无 GPU)
│       ├── submit_benchmark.py        legacy deploy+client 架构(仅回退)
│       └── aggregate_repeats.py       legacy 跨 repeat 聚合(现由 compute_summary 承担)
├── configs/
│   └── eval_suites/benchmark/         ← suite YAML(base_model_eval_v1 是标准 7-bench)
├── datasets/
│   └── benchmark/
│       ├── normalized/                ★ 9 个 bench,各含 benchmark_meta.json(行为契约)
│       └── evalplus/                  HumanEval+/MBPP+ 数据
└── docs/
    └── conventions/eval_conventions.md  评测约定(唯一入口说明,权威文档)
```

---

## 2. 核心设计:一条调用链

```
submit_eval (唯一入口)  ──读──▶  Suite YAML  ──校验──▶  suites.py
   │  submit: 每个 bench 生成一个 .hope 提交 GPU 集群
   │  collect: 扫 summary.json 出 accuracy_mean 表(agent 用 cron 轮询)
   ▼
选引擎 --engine vllm|fluentllm  ──▶  driver + 镜像 + 资源
   ▼
driver 按 RUNNER 分派:
   protocol  → inproc_runner (vLLM 进程内)    ── 行为契约 ──▶ benchmark_meta.json
   external  → run_code_eval.sh (evalplus)   ── 归一化  ──▶ code_eval normalize
   ▼
每个 bench 写 run_NN/result.json → compute_summary 汇成 summary.json → collect 读 accuracy_mean
```

**设计原则:入口替你守住协议事实。** 调用方只给 `model-path / model-tag / output-dir / engine`,
driver / env / resource / metric 全自动——这是为避免历史上"code bench 套错 driver 缺
DATASET/METRIC、路径漂移、资源配错"那类低级错误。

---

## 3. 7 个 benchmark

| bench | runner | driver 分派 | 行为契约 |
|---|---|---|---|
| gsm8k, ceval, cmmlu, cruxeval, mmlu | `protocol` | 进程内 `inproc_runner` | `normalized/{id}/benchmark_meta.json` |
| humaneval_plus, mbpp_plus | `external` | evalplus `run_code_eval.sh` | framework 自有 + `code_eval normalize` 归一 |

行为契约字段(prompt_builder_id / scorer_id / stop_tokens / fewshot / generation_kwargs /
fewshot_examples)写在各 bench 的 `benchmark_meta.json`。**suite YAML 不重载行为契约**——
suite 只回答"评测什么",不回答"怎么评"。

> normalized/ 里另有 bbh / math500 / mmlu_pro / mmlu_redux,是历史/扩展 bench,不在标准
> 7-bench suite(base_model_eval_v1)里,但 meta 齐全,可自建 suite 引用。

---

## 4. 用法

### submit(立即返回,不阻塞)

```bash
python -m src.scripts.suite.submit_eval submit \
    --suite base_model_eval_v1 \
    --model-path /abs/path/to/model \
    --model-tag my_model_tag \
    --output-dir /abs/path/to/eval_output \
    --engine vllm            # 或 fluentllm(LongCAT/Flash-3B 模型用)
    [--bench gsm8k mmlu ...] # 默认全 7 个
    [--dry-run]              # 只生成 .hope 不提交
```

### collect(幂等,可反复调)

```bash
python -m src.scripts.suite.submit_eval collect \
    --suite base_model_eval_v1 \
    --model-tag my_model_tag \
    --output-dir /abs/path/to/eval_output
```

齐了 → 打印 `accuracy_mean`(avg@N)表 + 末行 AVG,退出码 0;没齐 → 列缺哪个 bench,退出码非零。

### 轮询编排

submit 后由编排层(原平台用 cron 每 ~10min)反复调 collect,齐了即停。**入口本身不阻塞轮询。**

---

## 5. 两套引擎

`--engine` 是整次提交的属性(不是 per-bench),决定 driver / 镜像 / 资源 / 是否起 server。
两引擎共用同一 suite、同一 benchmark_meta.json、同一 collect。

| | `vllm`(默认) | `fluentllm` |
|---|---|---|
| 执行模型 | 进程内 vLLM,`llm.generate()` | SGLang server + HTTP client |
| driver | `src/eval/infra/run_eval.sh` | `src/eval/infra/run_eval_fluentllm.sh` |
| 并发 | 无(连续批处理自动打满) | `BENCH_CONCURRENCY=500` |
| code 生成 | evalplus 直跑(in-proc) | `gen_code_completions.py` 打 endpoint |

默认资源、镜像 ID 等见 `submit_eval.py` 顶部常量(`RUNNER_DEFAULTS` / `DOCKER_IMAGE`)。

---

## 6. 已知死代码(PPL,不影响 benchmark 评测)

`src/eval/cli.py`、`registry.py`、`perplexity.py`、`ppl/` 是 **PPL(perplexity)评测**子系统,
在原仓里是独立入口(`python -m src.eval.cli run --suite …ppl…`)。本 bundle **未带 PPL 数据
和 PPL suite YAML**,因此这些代码无法运行,但与 benchmark 评测无耦合,留着无害。若要彻底
纯化,可删除这四个文件并清理 `cli.py`/`registry.py` 对 benchmark 的交叉引用——非必须。

---

## 7. 路径约定(已全部相对化)

本 bundle **不含任何硬编码绝对路径**(原始抽取源的 `/mnt/.../guoyifu02/0PM` 已全部消除):

| 位置 | 约定 |
|---|---|
| `submit_eval.py` | `REPO = Path(__file__).resolve().parents[3]`(src/scripts/suite → repo) |
| `run_eval.sh` / `run_eval_fluentllm.sh` | `WORK_DIR=dirname(BASH_SOURCE)`;`REPO = ${WORK_DIR}/../../..`(src/eval/infra → repo) |
| `run_code_eval.sh` | `REPO = ${_SCRIPT_DIR}/../../..`(src/eval/external → repo),可被 env `REPO` 覆盖 |
| `prepare_bench_data.py` | `REPO = parents[3]`;`DEFAULT_DATA_ROOT` 默认 `datasets/benchmark/raw`,可被 `--data-root` / env `BENCH_DATA_ROOT` 覆盖 |

**前提:保持 `src/` 层级不变**,REPO 即自动解析到 bundle 根。所有衍生路径(`datasets/benchmark/
normalized/{id}/benchmark_meta.json`、`src/eval/external/run_code_eval.sh` 等)均相对 REPO。

---

## 8. 产物布局

```
OUTPUT_BASE/
├── {model_tag}/{bench_id}/
│   ├── run_00/result.json … run_NN/result.json
│   └── summary.json          ← collect 扫这个(取顶层 accuracy_mean)
├── {model_tag}/logs/         ← driver 日志(host+timestamp 命名,防覆盖)
└── generated_hopes/{bench}.hope
```

`compute_summary.py` 写出 `accuracy_mean/std/min/max`,但 **collect 只取 `accuracy_mean`**
(avg@N 单值)。

---

## 9. 平台适配清单(迁移时必看)

本 bundle 在原平台(Meituan Hope GPU 集群)上即开即用。迁移到其他平台时,以下几处是
**平台相关的事实**,需 adapt——它们不是 bug,是各自平台的执行后端/环境约定:

### 9.1 任务提交后端(`submit_eval.py`)

入口的 `submit` 子命令把每个 bench 生成一个 `.hope`(Hope 平台 job spec)并 `hope run` 提交。
迁移到别的 GPU 平台时,替换 `_submit()`(`submit_eval.py` 顶部的 `_HOPE_PREFIX`/`_HOPE_SUFFIX`
模板 + `hope run --force` 调用)为目标平台的 job 提交方式。`collect` 子命令纯扫 NFS 上的
`summary.json`,与平台无关,无需改。

平台相关常量(顶部):
- `QUEUE` / `USERGROUP` — Hope 队列与用户组
- `DOCKER_IMAGE[engine]` — 两个引擎各自的镜像(可用 `--docker-image` 覆盖)
- `RUNNER_DEFAULTS[engine][runner]` — 默认 TP/内存/vcore/GPU 数(可被 suite.resources 或 CLI flag 覆盖)

### 9.2 Docker 镜像内的环境路径

- vLLM 镜像:依赖 `transformers 5.8 + vLLM 0.21 + FlashAttention v2`,镜像内已就绪。
- FluentLLM 镜像:`run_eval_fluentllm.sh` 里 `source /home/fluentllmenv/bin/activate`、
  `EPS_HOME=/home/fluentllm/3rdparty/eps/` 是 **FluentLLM 容器内部**的固定路径,不是 repo
  路径——换镜像时按新容器的 env 位置调整这两行。

### 9.3 模型加载

- pruned 模型:`loader.py` 检测 `prune_spec.json` 并校验 `original_path`,剪枝 ckpt 的
  `ffn_hidden_size`/tokenizer 需在评测前核对(原仓 memory 有踩坑记录)。
- LongCat/Flash-3B 模型:raw ckpt 在新 transformers/vLLM 下会崩,`make_patched_model.sh` +
  `patch_longcat_model.py` 在 `OUTPUT_BASE/../_patched_models/{tag}/` 构建影子目录(weights
  软链 + remote-code patch)。两引擎的 protocol/external job 共用同一影子目录。

### 9.4 上游 raw 数据(仅 prepare_bench_data.py 需要)

`prepare_bench_data.py` 是**一次性**数据准备脚本,把上游 raw 转成 `normalized/`。其输入
(`DEFAULT_DATA_ROOT`)指向 bundle **外**的上游 raw-aligned exports——**本 bundle 不含 raw,
只含 normalized 产物**。若要在新平台重新生成 normalized,需自行提供上游 raw(布局见
`prepare_bench_data.py` 各 preparer 的 `src_path`,如 `mmlu/all__test.jsonl`、
`gsm8k/test.jsonl` 等),并通过 `--data-root` 或 env `BENCH_DATA_ROOT` 指定。
**日常评测不涉及此脚本**(normalized 已就绪)。

### 9.5 evalplus 源

`src/eval/infra/reference_evalplus/` 是 vendored 的 evalplus 源(已去 .git)。
`run_code_eval.sh` 用 `pip install -e ${EVALPLUS_SRC}` 就地安装。若目标平台已有 evalplus,
可改用平台版本(env `EVALPLUS_SRC` 指过去)。

---

## 10. 快速自检(无需 GPU)

```bash
# 1. Python 语法 + import 完整性
python -c "import ast,sys; [ast.parse(open(f).read()) for f in __import__('pathlib').Path('src').rglob('*.py')]"

# 2. Suite YAML 可加载 + 校验
python -m src.scripts.suite.submit_eval collect \
    --suite base_model_eval_v1 --model-tag _selftest --output-dir /tmp/_nonexistent

# 3. dry-run 生成 .hope(不提交,检查 manifest)
python -m src.scripts.suite.submit_eval submit \
    --suite base_model_eval_v1 --engine vllm --dry-run \
    --model-path /tmp/any --model-tag _selftest --output-dir /tmp/_selftest_out
```

详见 `docs/conventions/eval_conventions.md`(权威约定文档)。
