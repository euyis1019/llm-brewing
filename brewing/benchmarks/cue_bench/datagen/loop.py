"""
Task 5 — Loop (enhanced)

The model must mentally execute loop iterations to determine the final value.
Inspired by SWE-bench patterns:
  - django-11087: topological sort with while + set dependency check
  - django-11885: for loop + dual counter + conditional decrement

Body types:
  simple_acc    — for-range accumulator (sum of loop variable)
  filter_count  — for-each with conditional counting
  dual_var      — two variables updated per iteration (Fibonacci-like)

Dimensions:
  body_type    ∈ {simple_acc, filter_count, dual_var}
  iterations   ∈ {2, 3, 4}    number of iterations
  init_offset  ∈ {0, low, high}    accumulator start: 0 / random 1-2 / random 3-4
"""

import random
from .base import NamePool, make_prompt

TASK_NAME = "loop"

DIMENSIONS = {
    "body_type": ["simple_acc", "filter_count", "dual_var"],
    "iterations": [2, 3, 4],
    "init_offset": ["0", "low", "high"],
}

SAMPLES_PER_CONFIG = 150  # 3×3×3×30 = 810


def _init_offset_value(init_offset: str) -> int:
    """Return a concrete start value for the given init_offset level."""
    if init_offset == "low":
        return random.randint(1, 2)
    elif init_offset == "high":
        return random.randint(3, 4)
    else:  # "0"
        return 0


# ====================================================================
# Body type: simple_acc
# ====================================================================
def _gen_simple_acc(iterations: int, init_offset: str) -> dict | None:
    """For-range accumulator that sums the loop variable.

    With init_offset=low/high, the accumulator starts at a nonzero value
    and is passed as a function parameter.

    iterations=4, init_offset=0::

        def accumulate(n):
            total = 0
            for i in range(n):
                total = total + i
            return total

        result = accumulate(4)
        # 0+1+2+3 = 6

    iterations=3, init_offset=low::

        def accumulate(n, start):
            total = start
            for i in range(n):
                total = total + i
            return total

        result = accumulate(3, 2)
        # 2+0+1+2 = 5
    """
    names = NamePool()
    names.reserve("i")
    fn = names.func()
    n_param = names.param()
    acc = names.var()
    result_var = names.var()

    start_val = _init_offset_value(init_offset)

    answer = start_val + sum(range(iterations))
    if not (0 <= answer <= 9):
        return None

    if init_offset != "0":
        start_param = names.param()
        lines = [
            f"def {fn}({n_param}, {start_param}):",
            f"    {acc} = {start_param}",
            f"    for i in range({n_param}):",
            f"        {acc} = {acc} + i",
            f"    return {acc}",
            "",
            f"{result_var} = {fn}({iterations}, {start_val})",
        ]
    else:
        lines = [
            f"def {fn}({n_param}):",
            f"    {acc} = 0",
            f"    for i in range({n_param}):",
            f"        {acc} = {acc} + i",
            f"    return {acc}",
            "",
            f"{result_var} = {fn}({iterations})",
        ]

    code = "\n".join(lines)
    return {
        "prompt": make_prompt(code, result_var),
        "code": code,
        "answer": str(answer),
        "metadata": {
            "body_type": "simple_acc",
            "iterations": iterations,
            "init_offset": init_offset,
            "result_var": result_var,
        },
    }


# ====================================================================
# Body type: filter_count
# ====================================================================
def _gen_filter_count(iterations: int, init_offset: str) -> dict | None:
    """For-each loop counting items that match a condition.

    With init_offset=low/high, the counter starts at a nonzero value,
    modelling e.g. a running tally that accumulates across batches.

    iterations=4, init_offset=0::

        def count_matches(items, threshold):
            count = 0
            for item in items:
                if item >= threshold:
                    count = count + 1
            return count

        result = count_matches([1, 4, 2, 5], 3)
        # 4,5 >= 3 -> 2
    """
    names = NamePool()
    fn = names.func()
    items_param = names.param()
    threshold_param = names.param()
    counter = names.var()
    item_var = names.param()
    result_var = names.var()

    start_val = _init_offset_value(init_offset)

    for _attempt in range(200):
        items = [random.randint(0, 9) for _ in range(iterations)]
        threshold = random.randint(1, 7)

        cmp_op = random.choice([">=", ">"])
        if cmp_op == ">=":
            match_count = sum(1 for x in items if x >= threshold)
        else:
            match_count = sum(1 for x in items if x > threshold)

        answer = start_val + match_count
        if 0 <= answer <= 9:
            break
    else:
        return None

    if init_offset != "0":
        start_param = names.param()
        lines = [
            f"def {fn}({items_param}, {threshold_param}, {start_param}):",
            f"    {counter} = {start_param}",
            f"    for {item_var} in {items_param}:",
            f"        if {item_var} {cmp_op} {threshold_param}:",
            f"            {counter} = {counter} + 1",
            f"    return {counter}",
            "",
            f"{result_var} = {fn}("
            f"[{', '.join(str(x) for x in items)}], {threshold}, {start_val})",
        ]
    else:
        lines = [
            f"def {fn}({items_param}, {threshold_param}):",
            f"    {counter} = 0",
            f"    for {item_var} in {items_param}:",
            f"        if {item_var} {cmp_op} {threshold_param}:",
            f"            {counter} = {counter} + 1",
            f"    return {counter}",
            "",
            f"{result_var} = {fn}("
            f"[{', '.join(str(x) for x in items)}], {threshold})",
        ]

    code = "\n".join(lines)
    return {
        "prompt": make_prompt(code, result_var),
        "code": code,
        "answer": str(answer),
        "metadata": {
            "body_type": "filter_count",
            "iterations": iterations,
            "init_offset": init_offset,
            "result_var": result_var,
        },
    }


# ====================================================================
# Body type: dual_var
# ====================================================================
def _gen_dual_var(iterations: int, init_offset: str) -> dict | None:
    """Two variables updated each iteration (Fibonacci-style).

    With init_offset=low/high, both initial values are shifted by a
    nonzero amount, producing different sequences.

    iterations=4, init_offset=0::

        def step(n):
            a = 0
            b = 1
            for i in range(n):
                a, b = b, a + b
            return a

        result = step(4)
        # (0,1)->(1,1)->(1,2)->(2,3)->(3,5) -> a=3
    """
    names = NamePool()
    names.reserve("i")
    fn = names.func()
    n_param = names.param()
    var_a = names.param()
    var_b = names.param()
    result_var = names.var()

    if init_offset == "low":
        offset_a = random.randint(1, 2)
        offset_b = random.randint(1, 2)
    elif init_offset == "high":
        offset_a = random.randint(3, 4)
        offset_b = random.randint(3, 4)
    else:
        offset_a = 0
        offset_b = 0

    # Pick from a few dual-variable update rules
    base_rules = [
        # (base_init_a, base_init_b, update_template, update_fn)
        (0, 1, f"{var_a}, {var_b} = {var_b}, {var_a} + {var_b}",
         lambda a, b: (b, a + b)),
        (1, 0, f"{var_a}, {var_b} = {var_a} + 1, {var_b} + {var_a}",
         lambda a, b: (a + 1, b + a)),
        (0, 1, f"{var_a}, {var_b} = {var_b}, {var_a} + 1",
         lambda a, b: (b, a + 1)),
    ]

    for _attempt in range(200):
        base_a, base_b, update_str, update_fn = random.choice(base_rules)
        init_a = base_a + offset_a
        init_b = base_b + offset_b

        a, b = init_a, init_b
        ok = True
        for _ in range(iterations):
            a, b = update_fn(a, b)
            if a > 20 or b > 20 or a < -10 or b < -10:
                ok = False
                break
        if not ok or not (0 <= a <= 9):
            continue

        answer = a

        if init_offset != "0":
            start_a_param = names.param()
            start_b_param = names.param()
            lines = [
                f"def {fn}({n_param}, {start_a_param}, {start_b_param}):",
                f"    {var_a} = {start_a_param}",
                f"    {var_b} = {start_b_param}",
                f"    for i in range({n_param}):",
                f"        {update_str}",
                f"    return {var_a}",
                "",
                f"{result_var} = {fn}({iterations}, {init_a}, {init_b})",
            ]
        else:
            lines = [
                f"def {fn}({n_param}):",
                f"    {var_a} = {init_a}",
                f"    {var_b} = {init_b}",
                f"    for i in range({n_param}):",
                f"        {update_str}",
                f"    return {var_a}",
                "",
                f"{result_var} = {fn}({iterations})",
            ]

        code = "\n".join(lines)
        return {
            "prompt": make_prompt(code, result_var),
            "code": code,
            "answer": str(answer),
            "metadata": {
                "body_type": "dual_var",
                "iterations": iterations,
                "init_offset": init_offset,
                "result_var": result_var,
            },
        }
    return None


# ====================================================================
# Public API
# ====================================================================
_GENERATORS = {
    "simple_acc": _gen_simple_acc,
    "filter_count": _gen_filter_count,
    "dual_var": _gen_dual_var,
}


def generate_dataset(
    seed: int = 42,
    samples_per_config: int = SAMPLES_PER_CONFIG,
) -> list[dict]:
    random.seed(seed)
    dataset: list[dict] = []

    for body_type in DIMENSIONS["body_type"]:
        gen_fn = _GENERATORS[body_type]
        for iterations in DIMENSIONS["iterations"]:
            for init_offset in DIMENSIONS["init_offset"]:
                for idx in range(samples_per_config):
                    for _attempt in range(200):
                        sample = gen_fn(iterations, init_offset)
                        if sample is not None:
                            sample["id"] = (
                                f"loop_{body_type}_i{iterations}"
                                f"_off{init_offset}_{idx:03d}"
                            )
                            sample["metadata"]["sample_idx"] = idx
                            dataset.append(sample)
                            break

    return dataset
