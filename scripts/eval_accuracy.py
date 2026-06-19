"""
eval_accuracy.py — CUE-Bench 准确率评测脚本
=============================================
只做 greedy decoding，不走 Brewing cache/probe 流程。
模型对每道题预测下一个 token，对比 ground-truth 单字符答案（0-9）。

用法:
    python scripts/eval_accuracy.py --model /path/to/model [选项]

选项:
    --model         模型路径（必填）
    --data_dir      eval 数据目录（默认: brewing/benchmarks/cue_bench/data/eval）
    --tasks         只跑指定任务，逗号分隔（默认: 全部 6 个）
    --batch_size    推理批大小（默认: 16）
    --max_samples   每任务最多跑多少条，调试用（默认: 全部）
    --output        结果保存路径（默认: 不保存，只打印）
    --dtype         模型精度 bfloat16 / float16 / float32（默认: bfloat16）

示例:
    # 测 Qwen3.5-35B-A3B-Base
    python scripts/eval_accuracy.py \
        --model /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtai/users/guoyifu02/model/big/Qwen/Qwen3.5-35B-A3B-Base \
        --batch_size 8

    # 只测两个任务，快速验证
    python scripts/eval_accuracy.py \
        --model /path/to/model \
        --tasks value_tracking,computing \
        --max_samples 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

ALL_TASKS = [
    "value_tracking",
    "computing",
    "conditional",
    "function_call",
    "loop",
    "loop_unrolled",
]

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "brewing" / "benchmarks" / "cue_bench" / "data" / "eval"


# ---------------------------------------------------------------------------
# 加载数据
# ---------------------------------------------------------------------------

def load_task_data(data_dir: Path, task: str, max_samples: int | None) -> list[dict]:
    path = data_dir / f"{task}.json"
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    with open(path) as f:
        samples = json.load(f)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


# ---------------------------------------------------------------------------
# 批量推理
# ---------------------------------------------------------------------------

def predict_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    device: str,
) -> list[str]:
    """
    对一批 prompt 做 greedy 单步解码，返回预测的下一个 token 字符串列表。
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        # 只需要 1 个新 token
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # output_ids shape: (batch, prompt_len + 1)
    # 取最后一列，即新生成的 token
    new_token_ids = output_ids[:, inputs["input_ids"].shape[1]]
    predictions = [tokenizer.decode([tid]).strip() for tid in new_token_ids.tolist()]
    return predictions


# ---------------------------------------------------------------------------
# 单任务评测
# ---------------------------------------------------------------------------

def eval_task(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: list[dict],
    task: str,
    batch_size: int,
    device: str,
) -> dict:
    """返回该任务的详细结果字典。"""
    prompts = [s["prompt"] for s in samples]
    answers = [s["answer"] for s in samples]
    ids = [s.get("id", str(i)) for i, s in enumerate(samples)]

    all_preds = []
    n = len(prompts)

    print(f"  [{task}] {n} 条样本，batch_size={batch_size}", flush=True)
    t0 = time.time()

    for start in range(0, n, batch_size):
        batch_prompts = prompts[start : start + batch_size]
        preds = predict_batch(model, tokenizer, batch_prompts, device)
        all_preds.extend(preds)

        done = min(start + batch_size, n)
        elapsed = time.time() - t0
        speed = done / elapsed
        print(f"    {done}/{n}  ({speed:.1f} samples/s)", end="\r", flush=True)

    print()  # 换行

    # 计算准确率
    correct = sum(p == a for p, a in zip(all_preds, answers))
    accuracy = correct / n

    # 按 difficulty 分组统计
    breakdown = {}
    for sample, pred, ans in zip(samples, all_preds, answers):
        meta = sample.get("metadata", {})
        # 取第一个 difficulty 维度作为分组 key（每个任务不同）
        dim_key, dim_val = _primary_dimension(task, meta)
        key = f"{dim_key}={dim_val}"
        if key not in breakdown:
            breakdown[key] = {"correct": 0, "total": 0}
        breakdown[key]["total"] += 1
        if pred == ans:
            breakdown[key]["correct"] += 1

    for k, v in breakdown.items():
        v["accuracy"] = v["correct"] / v["total"]

    return {
        "task": task,
        "n_samples": n,
        "n_correct": correct,
        "accuracy": accuracy,
        "breakdown": breakdown,
        "per_sample": [
            {"id": sid, "pred": p, "answer": a, "correct": p == a}
            for sid, p, a in zip(ids, all_preds, answers)
        ],
    }


def _primary_dimension(task: str, meta: dict) -> tuple[str, str]:
    """每个任务取最能代表复杂度的维度用于 breakdown。"""
    dim_map = {
        "value_tracking": "depth",
        "computing": "steps",
        "conditional": "depth",
        "function_call": "depth",
        "loop": "iterations",
        "loop_unrolled": "iterations",
    }
    key = dim_map.get(task, "depth")
    return key, str(meta.get(key, "?"))


# ---------------------------------------------------------------------------
# 打印结果
# ---------------------------------------------------------------------------

def print_results(results: list[dict], model_path: str) -> None:
    print()
    print("=" * 60)
    print(f"  模型: {model_path}")
    print("=" * 60)
    print(f"  {'任务':<20} {'准确率':>8}  {'答对/总数':>12}")
    print("  " + "-" * 44)

    total_correct = 0
    total_n = 0
    for r in results:
        acc_str = f"{r['accuracy']:.1%}"
        cnt_str = f"{r['n_correct']}/{r['n_samples']}"
        print(f"  {r['task']:<20} {acc_str:>8}  {cnt_str:>12}")
        total_correct += r["n_correct"]
        total_n += r["n_samples"]

    print("  " + "-" * 44)
    overall = total_correct / total_n
    print(f"  {'Overall':<20} {overall:.1%}  {total_correct}/{total_n}")
    print("=" * 60)

    # 各任务 difficulty breakdown
    print()
    for r in results:
        print(f"  [{r['task']}] 按难度维度分解:")
        for k, v in sorted(r["breakdown"].items()):
            print(f"    {k:<25} {v['accuracy']:.1%}  ({v['correct']}/{v['total']})")
        print()


# ---------------------------------------------------------------------------
# 模型加载（兼容 text-only 和 VL checkpoint）
# ---------------------------------------------------------------------------

def _load_model(model_path: str, torch_dtype) -> AutoModelForCausalLM:
    """
    智能加载模型：
    - 纯文本 checkpoint（如 Qwen2.5-7B）：直接用 AutoModelForCausalLM
    - VL checkpoint（如 Qwen3.5-9B-Base，权重 key 含 model.language_model.*）：
      用 AutoModelForCausalLM 加载但传入 text_config，或直接用 VL 类做文本推理
    - DeepSeek MoE：同标准 AutoModelForCausalLM 路径，trust_remote_code=True
    """
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    is_vl = hasattr(cfg, "text_config") and cfg.text_config is not None

    load_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if is_vl:
        # VL checkpoint：用 architectures[0] 直接加载，避免 key 前缀不匹配
        arch = (cfg.architectures or [""])[0]
        import transformers as _tf
        model_cls = getattr(_tf, arch, None)
        if model_cls is None:
            print(f"[警告] 找不到架构类 {arch}，回退到 AutoModelForCausalLM", file=sys.stderr)
            model_cls = AutoModelForCausalLM
        print(f"[模型] 检测到 VL checkpoint，使用 {model_cls.__name__} 加载（文本推理模式）")
        return model_cls.from_pretrained(model_path, **load_kwargs)
    else:
        return AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CUE-Bench 准确率评测")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="eval 数据目录")
    parser.add_argument("--tasks", default=None, help="逗号分隔的任务名，默认全部")
    parser.add_argument("--batch_size", type=int, default=16, help="推理批大小")
    parser.add_argument("--max_samples", type=int, default=None, help="每任务最多条数（调试用）")
    parser.add_argument("--output", type=Path, default=None, help="结果 JSON 保存路径")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    # 解析任务列表
    tasks = ALL_TASKS
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
        unknown = set(tasks) - set(ALL_TASKS)
        if unknown:
            print(f"[错误] 未知任务: {unknown}", file=sys.stderr)
            sys.exit(1)

    # 检查数据目录
    if not args.data_dir.exists():
        print(f"[错误] 数据目录不存在: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {device}")
    if device == "cuda":
        n_gpu = torch.cuda.device_count()
        print(f"[GPU]  {n_gpu} 块: " + ", ".join(
            f"{torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory // 1024**3}GB)"
            for i in range(n_gpu)
        ))

    # 加载 tokenizer
    print(f"[模型] 加载 tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 生成任务用 left-padding

    # 加载模型
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    print(f"[模型] 加载权重 (dtype={args.dtype}, device_map=auto)...")
    t_load = time.time()
    model = _load_model(args.model, torch_dtype)
    model.eval()
    print(f"[模型] 加载完成，耗时 {time.time() - t_load:.1f}s")

    # 逐任务评测
    all_results = []
    for task in tasks:
        print(f"\n>> 任务: {task}")
        samples = load_task_data(args.data_dir, task, args.max_samples)
        result = eval_task(model, tokenizer, samples, task, args.batch_size, device)
        all_results.append(result)
        print(f"   准确率: {result['accuracy']:.1%}  ({result['n_correct']}/{result['n_samples']})")

    # 打印汇总
    print_results(all_results, args.model)

    # 保存结果
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "model": args.model,
            "dtype": args.dtype,
            "tasks": all_results,
            "overall_accuracy": sum(r["n_correct"] for r in all_results)
                               / sum(r["n_samples"] for r in all_results),
        }
        # per_sample 数据量大，单独控制是否写入
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n[结果] 已保存到 {args.output}")


if __name__ == "__main__":
    main()
