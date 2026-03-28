"""
Task 6 — Loop Unrolled (mirrors Task 5)

Identical computations to the Loop task, but with the loop body written out
as sequential statements. Paired with Loop to isolate the cognitive cost of
loop syntax (iteration variable, range, termination condition).

Body types mirror loop.py:
  simple_acc    — sequential additions of 0,1,2,...
  filter_count  — sequential if-checks on each item
  dual_var      — sequential a,b swaps

Dimensions (3×3×3 = 27 configs):
  body_type    ∈ {simple_acc, filter_count, dual_var}
  iterations   ∈ {2, 3, 4}
  init_offset  ∈ {"0", "low", "high"}
    "0"   — accumulator starts at 0
    "low" — accumulator starts at random value in 1-2
    "high"— accumulator starts at random value in 3-4
"""

import random
from .base import NamePool, make_prompt

TASK_NAME = "loop_unrolled"

DIMENSIONS = {
    "body_type": ["simple_acc", "filter_count", "dual_var"],
    "iterations": [2, 3, 4],
    "init_offset": ["0", "low", "high"],
}

SAMPLES_PER_CONFIG = 150  # 3×3×3×30 = 810


def _resolve_base(init_offset: str) -> int:
    """Return the numeric starting value for the given init_offset level."""
    if init_offset == "0":
        return 0
    elif init_offset == "low":
        return random.randint(1, 2)
    elif init_offset == "high":
        return random.randint(3, 4)
    else:
        raise ValueError(f"Unknown init_offset: {init_offset!r}")


# ====================================================================
# Body type: simple_acc (unrolled)
# ====================================================================
def _gen_simple_acc(iterations: int, init_offset: str) -> dict | None:
    """Unrolled sum of 0..n-1, optionally with a nonzero starting value.

    iterations=4, init_offset="0"::

        def accumulate():
            total = 0
            total = total + 0
            total = total + 1
            total = total + 2
            total = total + 3
            return total

        result = accumulate()   # answer: 6
    """
    names = NamePool()
    fn = names.func()
    acc = names.var()
    result_var = names.var()

    base = _resolve_base(init_offset)

    answer = base + sum(range(iterations))
    if not (0 <= answer <= 9):
        return None

    lines = [f"def {fn}():", f"    {acc} = {base}"]
    for i in range(iterations):
        lines.append(f"    {acc} = {acc} + {i}")
    lines.append(f"    return {acc}")
    lines.append("")
    lines.append(f"{result_var} = {fn}()")

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
# Body type: filter_count (unrolled)
# ====================================================================
def _gen_filter_count(iterations: int, init_offset: str) -> dict | None:
    """Unrolled conditional counting.

    iterations=5, init_offset="0"::

        def count_matches():
            count = 0
            if 1 >= 3:
                count = count + 1
            if 4 >= 3:
                count = count + 1
            ...
            return count

        result = count_matches()   # answer: 3
    """
    names = NamePool()
    fn = names.func()
    counter = names.var()
    result_var = names.var()

    base = _resolve_base(init_offset)

    for _attempt in range(200):
        items = [random.randint(0, 9) for _ in range(iterations)]
        threshold = random.randint(1, 7)
        cmp_op = random.choice([">=", ">"])

        if cmp_op == ">=":
            match_count = sum(1 for x in items if x >= threshold)
        else:
            match_count = sum(1 for x in items if x > threshold)

        answer = base + match_count
        if 0 <= answer <= 9:
            break
    else:
        return None

    lines = [f"def {fn}():", f"    {counter} = {base}"]
    for item in items:
        lines.append(f"    if {item} {cmp_op} {threshold}:")
        lines.append(f"        {counter} = {counter} + 1")
    lines.append(f"    return {counter}")
    lines.append("")
    lines.append(f"{result_var} = {fn}()")

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
# Body type: dual_var (unrolled)
# ====================================================================
def _gen_dual_var(iterations: int, init_offset: str) -> dict | None:
    """Unrolled dual-variable update (Fibonacci-style).

    For dual_var, init_offset shifts init_a by the offset value.

    iterations=4, init_offset="0"::

        def step():
            a = 0
            b = 1
            a, b = b, a + b
            a, b = b, a + b
            a, b = b, a + b
            a, b = b, a + b
            return a

        result = step()   # answer: 3
    """
    names = NamePool()
    fn = names.func()
    var_a = names.param()
    var_b = names.param()
    result_var = names.var()

    offset = _resolve_base(init_offset)

    rules = [
        # (base_init_a, init_b, update_template, update_fn)
        (0, 1, f"{var_a}, {var_b} = {var_b}, {var_a} + {var_b}",
         lambda a, b: (b, a + b)),
        (1, 0, f"{var_a}, {var_b} = {var_a} + 1, {var_b} + {var_a}",
         lambda a, b: (a + 1, b + a)),
        (0, 1, f"{var_a}, {var_b} = {var_b}, {var_a} + 1",
         lambda a, b: (b, a + 1)),
    ]

    for _attempt in range(200):
        base_a, init_b, update_str, update_fn = random.choice(rules)
        init_a = base_a + offset
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
        lines = [
            f"def {fn}():",
            f"    {var_a} = {init_a}",
            f"    {var_b} = {init_b}",
        ]
        for _ in range(iterations):
            lines.append(f"    {update_str}")
        lines.append(f"    return {var_a}")
        lines.append("")
        lines.append(f"{result_var} = {fn}()")

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
                                f"lu_{body_type}_i{iterations}"
                                f"_o{init_offset}_{idx:03d}"
                            )
                            sample["metadata"]["sample_idx"] = idx
                            dataset.append(sample)
                            break

    return dataset
