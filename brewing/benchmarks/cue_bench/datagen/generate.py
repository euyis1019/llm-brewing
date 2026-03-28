#!/usr/bin/env python3
"""
CUE Enhanced Data Generator — CLI entry point.

Usage:
    python -m datagen.generate                  # generate all tasks
    python -m datagen.generate --task loop      # generate one task
    python -m datagen.generate --output ./out   # custom output dir
    python -m datagen.generate --seed 123       # custom seed
"""

import argparse
import sys
from pathlib import Path

from .base import validate_and_save
from . import value_tracking, computing, conditional, function_call, loop, loop_unrolled

TASKS = {
    "value_tracking": (value_tracking, ["mechanism", "depth", "distractors"]),
    "computing": (computing, ["structure", "steps", "operators"]),
    "conditional": (conditional, ["branch_type", "depth", "condition_type"]),
    "function_call": (function_call, ["mechanism", "depth", "distractors"]),
    "loop": (loop, ["body_type", "iterations", "init_offset"]),
    "loop_unrolled": (loop_unrolled, ["body_type", "iterations", "init_offset"]),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate CUE enhanced datasets"
    )
    parser.add_argument(
        "--task", "-t",
        choices=list(TASKS.keys()),
        default=None,
        help="Generate only this task (default: all)",
    )
    default_output = Path(__file__).resolve().parent.parent / "data"
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=default_output,
        help=f"Output directory (default: {default_output})",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
        help="Override samples_per_config for all tasks",
    )
    args = parser.parse_args()

    tasks_to_run = (
        {args.task: TASKS[args.task]}
        if args.task
        else TASKS
    )

    total_samples = 0
    total_issues = 0

    for task_name, (module, dim_names) in tasks_to_run.items():
        kwargs = {"seed": args.seed}
        if args.samples is not None:
            kwargs["samples_per_config"] = args.samples

        dataset = module.generate_dataset(**kwargs)
        result = validate_and_save(dataset, task_name, args.output, dim_names)
        total_samples += result["total"]
        total_issues += result["issues"]

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_samples} samples across {len(tasks_to_run)} tasks")
    if total_issues:
        print(f"  !! {total_issues} issues found — please investigate")
    else:
        print("  All samples passed exec verification")
    print(f"{'=' * 60}")

    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
