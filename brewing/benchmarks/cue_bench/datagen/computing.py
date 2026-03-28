"""
Task 2 — Computing (enhanced)

The model must trace arithmetic through realistic code structures.
Inspired by SWE-bench patterns:
  - sympy-13031: reduce(row_join) accumulates matrix columns
  - django-10999: parse_duration with sign + timedelta arithmetic
  - django-14792: timezone offset sign flip

Structures:
  func_arithmetic   — function that takes params, does arithmetic, returns result
  chained_calls     — nested combine(combine(0, 1), 2) pattern
  accumulator       — function with loop that counts/sums matching items

Dimensions:
  structure  in {func_arithmetic, chained_calls, accumulator}
  steps      in {2, 3, 4}             number of computation steps
  operators  in {"add", "add_sub", "add_mul"}  operator set
"""

import random
from .base import NamePool, make_prompt

TASK_NAME = "computing"

DIMENSIONS = {
    "structure": ["func_arithmetic", "chained_calls", "accumulator"],
    "steps": [2, 3, 4],
    "operators": ["add", "add_sub", "add_mul"],
}

SAMPLES_PER_CONFIG = 150  # 3x3x3x30 = 810


# ====================================================================
# Operator helpers
# ====================================================================
def _pick_ops(n: int, operators: str) -> list[str]:
    """Return a list of n operator characters for the given operator set."""
    if operators == "add":
        return ["+"] * n
    elif operators == "add_sub":
        return [random.choice(["+", "-"]) for _ in range(n)]
    else:  # add_mul
        return [random.choice(["+", "*"]) for _ in range(n)]


def _pick_values(n: int, operators: str, ops: list[str]) -> list[int]:
    """Pick n operand values appropriate for the operator set.

    For add_mul, operands feeding into * are kept small (0-3)
    to avoid blowing past single-digit answers.
    """
    if operators == "add_mul":
        values = []
        for i in range(n):
            # If this value will be used with a * (either as left or right
            # operand), constrain to 0-3
            feeds_mul = False
            if i > 0 and i - 1 < len(ops) and ops[i - 1] == "*":
                feeds_mul = True
            if i < len(ops) and ops[i] == "*":
                feeds_mul = True
            if feeds_mul:
                values.append(random.randint(0, 3))
            else:
                values.append(random.randint(0, 5))
        return values
    else:
        return [random.randint(0, 5) for _ in range(n)]


def _eval_chain(values: list[int], ops: list[str]) -> int | None:
    """Evaluate a left-to-right chain: v0 op0 v1 op1 v2 ...

    Returns None if any intermediate or final result is outside 0-9.
    """
    running = values[0]
    for i, op in enumerate(ops):
        v = values[i + 1]
        if op == "+":
            running = running + v
        elif op == "-":
            running = running - v
        else:  # "*"
            running = running * v
        if running < 0 or running > 9:
            return None
    return running


def _op_symbol(op: str) -> str:
    """Human-readable symbol (identity, used directly in generated code)."""
    return op


# ====================================================================
# Distractor / decoration helpers for realism
# ====================================================================
_DOC_STRINGS = [
    '    """Process the input values."""',
    '    """Apply transformation to arguments."""',
    '    """Compute the derived metric."""',
    '    """Evaluate the expression and return the result."""',
    '    """Reduce inputs to a single output value."""',
    '    """Aggregate partial results."""',
    '    """Combine operands according to the configured rule."""',
]

_COMMENTS = [
    "# apply the operation",
    "# accumulate partial result",
    "# combine with next value",
    "# update running total",
    "# fold in next operand",
    "# reduce step",
]


def _maybe_docstring() -> str | None:
    """Return a random docstring 40% of the time, else None."""
    if random.random() < 0.4:
        return random.choice(_DOC_STRINGS)
    return None


def _maybe_comment() -> str | None:
    """Return an inline comment 30% of the time, else None."""
    if random.random() < 0.3:
        return random.choice(_COMMENTS)
    return None


# ====================================================================
# Structure: func_arithmetic
# ====================================================================
def _gen_func_arithmetic(steps: int, operators: str) -> dict | None:
    """Function with multi-step arithmetic on its parameters.

    Generates code like::

        def compute(a, b, c):
            '''Process the input values.'''
            tmp = a + b
            return tmp + c

        result = compute(1, 3, 2)

    With add_mul, operators can include * with constrained operands.
    """
    names = NamePool()
    fn = names.func()
    result_var = names.var()

    params = [names.param() for _ in range(steps + 1)]
    ops = _pick_ops(steps, operators)

    for _attempt in range(300):
        values = _pick_values(steps + 1, operators, ops)
        running = _eval_chain(values, ops)
        if running is None:
            # re-pick ops too for variety on retry
            if _attempt % 20 == 19:
                ops = _pick_ops(steps, operators)
            continue

        # Build function body
        lines: list[str] = []
        sig = ", ".join(params)

        # Optionally add a distractor parameter with default
        distractor_param = None
        if random.random() < 0.35:
            distractor_param = names.distractor()
            default_val = random.choice(["None", "False", "0", '""'])
            sig += f", {distractor_param}={default_val}"

        lines.append(f"def {fn}({sig}):")

        doc = _maybe_docstring()
        if doc:
            lines.append(doc)

        tmp_vars = [names.var() for _ in range(steps - 1)]
        prev = params[0]
        for i in range(steps):
            expr = f"{prev} {ops[i]} {params[i + 1]}"
            comment = _maybe_comment()
            suffix = f"  {comment}" if comment else ""
            if i < steps - 1:
                tv = tmp_vars[i]
                lines.append(f"    {tv} = {expr}{suffix}")
                prev = tv
            else:
                lines.append(f"    return {expr}{suffix}")
        lines.append("")

        call_args = ", ".join(str(v) for v in values)
        lines.append(f"{result_var} = {fn}({call_args})")

        code = "\n".join(lines)
        return {
            "prompt": make_prompt(code, result_var),
            "code": code,
            "answer": str(running),
            "metadata": {
                "structure": "func_arithmetic",
                "steps": steps,
                "operators": operators,
                "result_var": result_var,
            },
        }
    return None


# ====================================================================
# Structure: chained_calls
# ====================================================================
def _gen_chained_calls(steps: int, operators: str) -> dict | None:
    """Nested function calls: combine(combine(combine(0, 1), 2), 3).

    Inspired by sympy-13031 reduce(row_join, matrices) pattern.

    For mixed operators, generates separate named functions::

        def merge(a, b):
            return a + b

        def scale(a, b):
            return a * b

        result = scale(merge(merge(1, 2), 1), 2)
    """
    names = NamePool()
    result_var = names.var()
    p1, p2 = names.param(), names.param()

    ops = _pick_ops(steps, operators)

    for _attempt in range(300):
        values = _pick_values(steps + 1, operators, ops)
        running = _eval_chain(values, ops)
        if running is None:
            if _attempt % 20 == 19:
                ops = _pick_ops(steps, operators)
            continue

        unique_ops = sorted(set(ops))

        if len(unique_ops) == 1:
            # Single function for all steps
            fn = names.func()
            lines = [f"def {fn}({p1}, {p2}):"]
            doc = _maybe_docstring()
            if doc:
                lines.append(doc)
            lines.append(f"    return {p1} {unique_ops[0]} {p2}")
            lines.append("")

            # Build nested call expression
            expr = str(values[0])
            for i in range(steps):
                expr = f"{fn}({expr}, {values[i + 1]})"
            lines.append(f"{result_var} = {expr}")
        else:
            # Separate function per operator type
            op_to_fn: dict[str, str] = {}
            fn_defs: list[str] = []
            for op in unique_ops:
                fn_name = names.func()
                op_to_fn[op] = fn_name
                fn_defs.append(f"def {fn_name}({p1}, {p2}):")
                doc = _maybe_docstring()
                if doc:
                    fn_defs.append(doc)
                fn_defs.append(f"    return {p1} {op} {p2}")
                fn_defs.append("")

            lines = fn_defs
            expr = str(values[0])
            for i in range(steps):
                f = op_to_fn[ops[i]]
                expr = f"{f}({expr}, {values[i + 1]})"
            lines.append(f"{result_var} = {expr}")

        code = "\n".join(lines)
        return {
            "prompt": make_prompt(code, result_var),
            "code": code,
            "answer": str(running),
            "metadata": {
                "structure": "chained_calls",
                "steps": steps,
                "operators": operators,
                "result_var": result_var,
            },
        }
    return None


# ====================================================================
# Structure: accumulator
# ====================================================================
def _gen_accumulator(steps: int, operators: str) -> dict | None:
    """Function with a loop that accumulates/counts over a list.

    Inspired by django parse_duration sign arithmetic and sympy reduce.

    - add: count items matching a condition
    - add_sub: count up for matches, count down for anti-matches
    - add_mul: multiply accumulator by weight for matches, add offset otherwise

    steps determines list length (steps + 1 items).

    Example (add)::

        def count_valid(items, threshold):
            total = 0
            for item in items:
                if item > threshold:
                    total = total + 1
            return total

        result = count_valid([3, 1, 4, 1, 5], 2)

    Example (add_mul)::

        def score_items(items, threshold):
            '''Aggregate partial results.'''
            total = 1
            for item in items:
                if item > threshold:
                    total = total * item
                else:
                    total = total + 1
            return total

        result = score_items([2, 0, 3, 1], 1)
    """
    names = NamePool()
    fn = names.func()
    result_var = names.var()
    items_param = names.param()
    threshold_param = names.param()
    acc_var = names.var()
    item_var = names.param()

    list_len = steps + 1

    for _attempt in range(300):
        threshold = random.randint(1, 6)
        items = [random.randint(0, 9) for _ in range(list_len)]

        if operators == "add":
            # Count items above threshold
            answer = sum(1 for x in items if x > threshold)
        elif operators == "add_sub":
            # +1 for above, -1 for below
            answer = 0
            for x in items:
                if x > threshold:
                    answer += 1
                elif x < threshold:
                    answer -= 1
        else:  # add_mul
            # Multiply by small factor for matches, add 1 for non-matches
            # Use small items to keep products in range
            items = [random.randint(0, 3) for _ in range(list_len)]
            threshold = random.randint(0, 2)
            answer = 1  # start at 1 for multiplicative identity
            for x in items:
                if x > threshold:
                    answer = answer * x
                else:
                    answer = answer + 1
            # If answer is still 1 and nothing happened, that's boring
            # but still valid

        if not (0 <= answer <= 9):
            continue

        lines: list[str] = []
        lines.append(f"def {fn}({items_param}, {threshold_param}):")

        doc = _maybe_docstring()
        if doc:
            lines.append(doc)

        if operators == "add_mul":
            lines.append(f"    {acc_var} = 1")
        else:
            lines.append(f"    {acc_var} = 0")

        lines.append(f"    for {item_var} in {items_param}:")

        if operators == "add":
            lines.append(f"        if {item_var} > {threshold_param}:")
            lines.append(f"            {acc_var} = {acc_var} + 1")
        elif operators == "add_sub":
            lines.append(f"        if {item_var} > {threshold_param}:")
            lines.append(f"            {acc_var} = {acc_var} + 1")
            lines.append(f"        elif {item_var} < {threshold_param}:")
            lines.append(f"            {acc_var} = {acc_var} - 1")
        else:  # add_mul
            lines.append(f"        if {item_var} > {threshold_param}:")
            lines.append(f"            {acc_var} = {acc_var} * {item_var}")
            lines.append(f"        else:")
            lines.append(f"            {acc_var} = {acc_var} + 1")

        lines.append(f"    return {acc_var}")
        lines.append("")
        items_str = "[" + ", ".join(str(x) for x in items) + "]"
        lines.append(f"{result_var} = {fn}({items_str}, {threshold})")

        code = "\n".join(lines)
        return {
            "prompt": make_prompt(code, result_var),
            "code": code,
            "answer": str(answer),
            "metadata": {
                "structure": "accumulator",
                "steps": steps,
                "operators": operators,
                "result_var": result_var,
            },
        }
    return None


# ====================================================================
# Public API
# ====================================================================
_GENERATORS = {
    "func_arithmetic": _gen_func_arithmetic,
    "chained_calls": _gen_chained_calls,
    "accumulator": _gen_accumulator,
}


def generate_dataset(
    seed: int = 42,
    samples_per_config: int = SAMPLES_PER_CONFIG,
) -> list[dict]:
    random.seed(seed)
    dataset: list[dict] = []

    for structure in DIMENSIONS["structure"]:
        gen_fn = _GENERATORS[structure]
        for steps in DIMENSIONS["steps"]:
            for operators in DIMENSIONS["operators"]:
                for idx in range(samples_per_config):
                    sample = gen_fn(steps, operators)
                    if sample is not None:
                        op_tag = operators[0]
                        sample["id"] = (
                            f"comp_{structure}_s{steps}_{op_tag}_{idx:03d}"
                        )
                        sample["metadata"]["sample_idx"] = idx
                        dataset.append(sample)

    return dataset
