"""CUE-Bench subset definitions and benchmark spec."""

from __future__ import annotations

from brewing.schema import (
    AnswerMeta, AnswerType, BenchmarkSpec, SubsetSpec,
)

# ===================================================================
# Subset definitions
# ===================================================================

VALUE_TRACKING = SubsetSpec(
    name="value_tracking",
    category="data_flow",
    difficulty_schema={
        "mechanism": ["function_chain", "container", "method_chain"],
        "depth": [1, 2, 3],
        "distractors": [0, 1, 2],
    },
    question_suffix='# The value of {var} is "',
)

COMPUTING = SubsetSpec(
    name="computing",
    category="data_flow",
    difficulty_schema={
        "structure": ["func_arithmetic", "inline_chain", "class_method"],
        "steps": [2, 3, 4],
        "operators": ["add_sub", "add_mul", "mixed"],
    },
    question_suffix='# The value of {var} is "',
)

CONDITIONAL = SubsetSpec(
    name="conditional",
    category="control_flow",
    difficulty_schema={
        "branch_type": ["if_else", "nested_if", "elif_chain"],
        "depth": [1, 2, 3],
        "condition_type": ["numeric", "membership", "boolean_flag"],
    },
    question_suffix='# The value of {var} is "',
)

FUNCTION_CALL = SubsetSpec(
    name="function_call",
    category="control_flow",
    difficulty_schema={
        "mechanism": ["arithmetic", "container_relay", "conditional_return"],
        "depth": [1, 2, 3],
        "distractors": [0, 1, 2],
    },
    question_suffix='# The value of {var} is "',
)

LOOP = SubsetSpec(
    name="loop",
    category="data_control",
    difficulty_schema={
        "body_type": ["simple_acc", "filter_count", "dual_var"],
        "iterations": [2, 3, 4],
        "init_offset": ["0", "low", "high"],
    },
    question_suffix='# The value of {var} is "',
)

LOOP_UNROLLED = SubsetSpec(
    name="loop_unrolled",
    category="data_control",
    difficulty_schema={
        "body_type": ["simple_acc", "filter_count", "dual_var"],
        "iterations": [2, 3, 4],
        "init_offset": ["0", "low", "high"],
    },
    question_suffix='# The value of {var} is "',
)

# ===================================================================
# Benchmark spec
# ===================================================================

CUE_BENCH = BenchmarkSpec(
    name="CUE-Bench",
    domain="code_reasoning",
    answer_meta=AnswerMeta(
        answer_type=AnswerType.CATEGORICAL,
        answer_space=[str(d) for d in range(10)],
        max_answer_tokens=1,
    ),
    prompt_template="{code}\n{question_suffix}",
    subsets=[VALUE_TRACKING, COMPUTING, CONDITIONAL, FUNCTION_CALL, LOOP, LOOP_UNROLLED],
)
