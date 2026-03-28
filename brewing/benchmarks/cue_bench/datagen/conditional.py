"""
Task 3 — Conditional (enhanced)

The model must determine which branch executes for a given input.
Inspired by SWE-bench patterns:
  - django-15732: _delete_composed_index mismatched PK filter (membership)
  - sympy-15599: Mod.doit() multi-layer guard clause chain
  - django-13925: _check_default_pk inherited PK false positive

Branch types:
  elif_chain    — multi-level elif dispatch (HTTP status style)
  guard_clause  — early-return guards with final fallthrough
  sequential_if — sequential ifs that mutate state between checks

Condition types:
  numeric       — numeric threshold comparisons (>=, >, <)
  membership    — set/list membership checks (in [...])
  boolean_flag  — boolean flag conditions (if flag, if not flag)

Dimensions:
  branch_type    ∈ {elif_chain, guard_clause, sequential_if}
  depth          ∈ {1, 2, 3}    number of branch/guard levels
  condition_type ∈ {numeric, membership, boolean_flag}
"""

import random
from .base import NamePool, make_prompt

TASK_NAME = "conditional"

DIMENSIONS = {
    "branch_type": ["elif_chain", "guard_clause", "sequential_if"],
    "depth": [1, 2, 3],
    "condition_type": ["numeric", "membership", "boolean_flag"],
}

SAMPLES_PER_CONFIG = 150  # 3×3×3×30 = 810


# ====================================================================
# Membership-check word pools (realistic CLI / API style groups)
# ====================================================================
_MEMBERSHIP_GROUPS = [
    (["get", "fetch", "read"], ["set", "put", "write"], ["delete", "remove", "drop"]),
    (["info", "debug"], ["warn", "error"], ["critical", "fatal"]),
    (["json", "xml"], ["csv", "tsv"], ["yaml", "toml"]),
    (["admin", "root"], ["editor", "manager"], ["viewer", "guest"]),
    (["pending", "queued"], ["running", "active"], ["done", "complete"]),
    (["tcp", "udp"], ["http", "https"], ["ws", "wss"]),
    (["png", "jpg", "gif"], ["mp4", "avi", "mkv"], ["mp3", "wav", "flac"]),
    (["north", "south"], ["east", "west"], ["up", "down"]),
    (["red", "crimson"], ["blue", "navy"], ["green", "lime"]),
    (["apple", "pear"], ["carrot", "celery"], ["salmon", "tuna"]),
]


def _pick_membership_groups(n: int) -> list[list[str]]:
    """Pick *n* disjoint membership groups from the pool."""
    pool = random.choice(_MEMBERSHIP_GROUPS)
    groups = list(pool)
    random.shuffle(groups)
    return groups[:n]


# ====================================================================
# Branch type: elif_chain
# ====================================================================
def _gen_elif_chain_numeric(depth: int) -> dict | None:
    """elif_chain + numeric: threshold-based dispatch."""
    names = NamePool()
    fn = names.func()
    param = names.param()
    result_var = names.var()

    n_branches = depth + 1  # depth elif + else
    returns = random.sample(range(0, 10), min(n_branches + 1, 10))

    thresholds = sorted(
        random.sample(range(10, 100), n_branches),
        reverse=True,
    )

    target_branch = random.randint(0, n_branches)

    if target_branch == 0:
        input_val = random.randint(thresholds[0], thresholds[0] + 50)
    elif target_branch < n_branches:
        hi = thresholds[target_branch - 1] - 1
        lo = thresholds[target_branch]
        if lo > hi:
            return None
        input_val = random.randint(lo, hi)
    else:
        input_val = random.randint(0, thresholds[-1] - 1)

    answer = None
    for i, t in enumerate(thresholds):
        if input_val >= t:
            answer = returns[i]
            break
    if answer is None:
        answer = returns[n_branches]

    if not (0 <= answer <= 9):
        return None

    lines: list[str] = []
    lines.append(f"def {fn}({param}):")
    for i, t in enumerate(thresholds):
        kw = "if" if i == 0 else "elif"
        lines.append(f"    {kw} {param} >= {t}:")
        lines.append(f"        return {returns[i]}")
    lines.append(f"    else:")
    lines.append(f"        return {returns[n_branches]}")
    lines.append("")
    lines.append(f"{result_var} = {fn}({input_val})")

    code = "\n".join(lines)
    return {
        "prompt": make_prompt(code, result_var),
        "code": code,
        "answer": str(answer),
        "metadata": {
            "branch_type": "elif_chain",
            "depth": depth,
            "condition_type": "numeric",
            "result_var": result_var,
        },
    }


def _gen_elif_chain_membership(depth: int) -> dict | None:
    """elif_chain + membership: in-list dispatch.

    Uses *depth* membership branches (if / elif / elif...) + an else fallback.
    """
    names = NamePool()
    fn = names.func()
    param = names.param()
    result_var = names.var()

    # depth membership branches + 1 else = depth+1 return values needed
    n_in_branches = depth
    returns = random.sample(range(0, 10), min(n_in_branches + 1, 10))

    groups = _pick_membership_groups(n_in_branches)
    if len(groups) < n_in_branches:
        return None

    # Pick which branch to hit (0..depth-1 = in-branch, depth = else)
    target_branch = random.randint(0, n_in_branches)

    if target_branch < n_in_branches:
        input_val = random.choice(groups[target_branch])
    else:
        # Hit the else — pick a value not in any group
        all_vals = [v for g in groups for v in g]
        input_val = "unknown"
        for candidate in ["none", "other", "default", "misc", "null"]:
            if candidate not in all_vals:
                input_val = candidate
                break

    # Compute expected answer
    answer = None
    for i, group in enumerate(groups):
        if input_val in group:
            answer = returns[i]
            break
    if answer is None:
        answer = returns[n_in_branches]

    if not (0 <= answer <= 9):
        return None

    lines: list[str] = []
    lines.append(f"def {fn}({param}):")
    for i, group in enumerate(groups):
        kw = "if" if i == 0 else "elif"
        group_str = ", ".join(f'"{v}"' for v in group)
        lines.append(f"    {kw} {param} in [{group_str}]:")
        lines.append(f"        return {returns[i]}")
    lines.append(f"    else:")
    lines.append(f"        return {returns[n_in_branches]}")
    lines.append("")
    lines.append(f'{result_var} = {fn}("{input_val}")')

    code = "\n".join(lines)
    return {
        "prompt": make_prompt(code, result_var),
        "code": code,
        "answer": str(answer),
        "metadata": {
            "branch_type": "elif_chain",
            "depth": depth,
            "condition_type": "membership",
            "result_var": result_var,
        },
    }


def _gen_elif_chain_boolean(depth: int) -> dict | None:
    """elif_chain + boolean_flag: flag-based dispatch."""
    names = NamePool()
    fn = names.func()
    result_var = names.var()

    n_branches = depth + 1
    returns = random.sample(range(0, 10), min(n_branches + 1, 10))

    # Generate boolean flag parameters
    flag_names = [names.distractor() for _ in range(n_branches)]
    # Each branch checks one flag; negate randomly for variety
    negations = [random.choice([True, False]) for _ in range(n_branches)]
    # Flag values (True/False) — we want exactly one branch to trigger
    target_branch = random.randint(0, n_branches)

    flag_vals: list[bool] = []
    for i in range(n_branches):
        if i == target_branch:
            # This branch should trigger: if negated, flag must be False; otherwise True
            flag_vals.append(False if negations[i] else True)
        elif i < target_branch:
            # Earlier branches must NOT trigger
            flag_vals.append(True if negations[i] else False)
        else:
            # Later branches: don't matter, randomize
            flag_vals.append(random.choice([True, False]))

    # Compute expected answer
    answer = None
    for i in range(n_branches):
        condition_met = (not flag_vals[i]) if negations[i] else flag_vals[i]
        if condition_met:
            answer = returns[i]
            break
    if answer is None:
        answer = returns[n_branches]

    if not (0 <= answer <= 9):
        return None

    sig = ", ".join(flag_names)
    lines: list[str] = []
    lines.append(f"def {fn}({sig}):")
    for i in range(n_branches):
        kw = "if" if i == 0 else "elif"
        cond = f"not {flag_names[i]}" if negations[i] else flag_names[i]
        lines.append(f"    {kw} {cond}:")
        lines.append(f"        return {returns[i]}")
    lines.append(f"    else:")
    lines.append(f"        return {returns[n_branches]}")
    lines.append("")
    call_args = ", ".join(str(v) for v in flag_vals)
    lines.append(f"{result_var} = {fn}({call_args})")

    code = "\n".join(lines)
    return {
        "prompt": make_prompt(code, result_var),
        "code": code,
        "answer": str(answer),
        "metadata": {
            "branch_type": "elif_chain",
            "depth": depth,
            "condition_type": "boolean_flag",
            "result_var": result_var,
        },
    }


# ====================================================================
# Branch type: guard_clause
# ====================================================================
def _gen_guard_clause_numeric(depth: int) -> dict | None:
    """guard_clause + numeric: early-return guards with numeric checks."""
    names = NamePool()
    fn = names.func()
    p1, p2 = names.param(), names.param()
    result_var = names.var()

    guard_templates = [
        ("{b} == 0", random.randint(0, 9), None, 0),
        ("{a} == 0", random.randint(0, 9), 0, None),
        ("{a} == {b}", random.randint(0, 9), "eq", "eq"),
        ("{a} < {b}", random.randint(0, 9), "lt", None),
        ("{b} < 0", random.randint(0, 9), None, "neg"),
    ]

    random.shuffle(guard_templates)
    guards = guard_templates[:depth]

    fallthrough_ops = [
        ("{a} % {b}", lambda a, b: a % b if b != 0 else None),
        ("{a} + {b}", lambda a, b: a + b),
        ("({a} + {b}) % 10", lambda a, b: (a + b) % 10),
        ("{a} - {b}", lambda a, b: a - b),
    ]

    for _attempt in range(200):
        ft_template, ft_fn = random.choice(fallthrough_ops)
        a_val = random.randint(1, 8)
        b_val = random.randint(1, 8)

        skip = False
        for cond_fmt, ret_val, a_c, b_c in guards:
            cond_str = cond_fmt.format(a=a_val, b=b_val)
            try:
                if eval(cond_str):
                    skip = True
                    break
            except Exception:
                skip = True
                break

        if skip:
            continue

        answer = ft_fn(a_val, b_val)
        if answer is None or not (0 <= answer <= 9):
            continue

        lines: list[str] = []
        lines.append(f"def {fn}({p1}, {p2}):")
        for cond_fmt, ret_val, _, _ in guards:
            cond = cond_fmt.format(a=p1, b=p2)
            lines.append(f"    if {cond}:")
            lines.append(f"        return {ret_val}")
        fallthrough = ft_template.format(a=p1, b=p2)
        lines.append(f"    return {fallthrough}")
        lines.append("")
        lines.append(f"{result_var} = {fn}({a_val}, {b_val})")

        code = "\n".join(lines)
        return {
            "prompt": make_prompt(code, result_var),
            "code": code,
            "answer": str(answer),
            "metadata": {
                "branch_type": "guard_clause",
                "depth": depth,
                "condition_type": "numeric",
                "result_var": result_var,
            },
        }
    return None


def _gen_guard_clause_membership(depth: int) -> dict | None:
    """guard_clause + membership: early-return guards with in-checks."""
    names = NamePool()
    fn = names.func()
    param = names.param()
    data_param = names.param()
    result_var = names.var()

    # Each guard checks if param is in a reject-set, returning early
    groups = _pick_membership_groups(depth)
    if len(groups) < depth:
        return None
    guard_returns = random.sample(range(0, 10), depth)

    # Fallthrough: return data_param % 10
    # Pick a data value and a param value that doesn't match any guard
    all_guard_vals = [v for g in groups for v in g]
    safe_vals = []
    for candidate in ["none", "other", "default", "misc", "null",
                       "ok", "pass", "skip", "noop", "valid"]:
        if candidate not in all_guard_vals:
            safe_vals.append(candidate)

    if not safe_vals:
        return None

    for _attempt in range(200):
        input_cmd = random.choice(safe_vals)
        data_val = random.randint(10, 99)
        answer = data_val % 10

        if not (0 <= answer <= 9):
            continue

        # Also generate samples that hit a guard
        hit_guard = random.random() < 0.5
        if hit_guard:
            guard_idx = random.randint(0, depth - 1)
            input_cmd = random.choice(groups[guard_idx])
            answer = guard_returns[guard_idx]

        lines: list[str] = []
        lines.append(f"def {fn}({param}, {data_param}):")
        for i in range(depth):
            group_str = ", ".join(f'"{v}"' for v in groups[i])
            lines.append(f"    if {param} in [{group_str}]:")
            lines.append(f"        return {guard_returns[i]}")
        lines.append(f"    return {data_param} % 10")
        lines.append("")
        lines.append(f'{result_var} = {fn}("{input_cmd}", {data_val})')

        code = "\n".join(lines)
        return {
            "prompt": make_prompt(code, result_var),
            "code": code,
            "answer": str(answer),
            "metadata": {
                "branch_type": "guard_clause",
                "depth": depth,
                "condition_type": "membership",
                "result_var": result_var,
            },
        }
    return None


def _gen_guard_clause_boolean(depth: int) -> dict | None:
    """guard_clause + boolean_flag: early-return guards with boolean flags.

    Example (depth=2)::

        def process(data, strict, force):
            if not force:
                return 0
            if strict:
                return 1
            return data % 10

        result = process(37, False, True)
    """
    names = NamePool()
    fn = names.func()
    data_param = names.param()
    result_var = names.var()

    flag_names = [names.distractor() for _ in range(depth)]
    negations = [random.choice([True, False]) for _ in range(depth)]
    guard_returns = random.sample(range(0, 10), depth)

    for _attempt in range(200):
        data_val = random.randint(10, 99)
        fallthrough_answer = data_val % 10

        # Decide: hit a guard or fall through
        hit_guard = random.random() < 0.5
        if hit_guard:
            target = random.randint(0, depth - 1)
        else:
            target = depth  # fall through

        flag_vals: list[bool] = []
        for i in range(depth):
            if i < target:
                # Must NOT trigger: if negated, flag=True; else flag=False
                flag_vals.append(True if negations[i] else False)
            elif i == target and target < depth:
                # Must trigger: if negated, flag=False; else flag=True
                flag_vals.append(False if negations[i] else True)
            else:
                flag_vals.append(random.choice([True, False]))

        # Verify logic
        answer = None
        for i in range(depth):
            cond = (not flag_vals[i]) if negations[i] else flag_vals[i]
            if cond:
                answer = guard_returns[i]
                break
        if answer is None:
            answer = fallthrough_answer

        if not (0 <= answer <= 9):
            continue

        sig = ", ".join([data_param] + flag_names)
        lines: list[str] = []
        lines.append(f"def {fn}({sig}):")
        for i in range(depth):
            cond_expr = f"not {flag_names[i]}" if negations[i] else flag_names[i]
            lines.append(f"    if {cond_expr}:")
            lines.append(f"        return {guard_returns[i]}")
        lines.append(f"    return {data_param} % 10")
        lines.append("")
        call_args = ", ".join([str(data_val)] + [str(v) for v in flag_vals])
        lines.append(f"{result_var} = {fn}({call_args})")

        code = "\n".join(lines)
        return {
            "prompt": make_prompt(code, result_var),
            "code": code,
            "answer": str(answer),
            "metadata": {
                "branch_type": "guard_clause",
                "depth": depth,
                "condition_type": "boolean_flag",
                "result_var": result_var,
            },
        }
    return None


# ====================================================================
# Branch type: sequential_if (state mutation between checks)
# ====================================================================
def _gen_sequential_if_numeric(depth: int) -> dict | None:
    """sequential_if + numeric: threshold checks with state mutation."""
    names = NamePool()
    fn = names.func()
    param = names.param()
    result_var = names.var()
    counter = names.var()

    thresholds = sorted(
        [random.randint(1, 6) for _ in range(depth)],
        reverse=True,
    )
    subtracts = [max(1, t - random.randint(0, 1)) for t in thresholds]

    for _attempt in range(200):
        input_val = random.randint(0, 15)
        x = input_val
        flag = 0
        for i in range(depth):
            if x > thresholds[i]:
                flag += 1
                if i < depth - 1:
                    x = x - subtracts[i]

        if not (0 <= flag <= 9):
            continue

        lines: list[str] = []
        lines.append(f"def {fn}({param}):")
        lines.append(f"    {counter} = 0")
        for i in range(depth):
            lines.append(f"    if {param} > {thresholds[i]}:")
            lines.append(f"        {counter} = {counter} + 1")
            if i < depth - 1:
                lines.append(f"        {param} = {param} - {subtracts[i]}")
        lines.append(f"    return {counter}")
        lines.append("")
        lines.append(f"{result_var} = {fn}({input_val})")

        code = "\n".join(lines)
        return {
            "prompt": make_prompt(code, result_var),
            "code": code,
            "answer": str(flag),
            "metadata": {
                "branch_type": "sequential_if",
                "depth": depth,
                "condition_type": "numeric",
                "result_var": result_var,
            },
        }
    return None


def _gen_sequential_if_membership(depth: int) -> dict | None:
    """sequential_if + membership: sequential membership checks mutating state.

    Each step checks if param is in a group; if so, increments counter
    and remaps param to a new value for the next check.
    """
    names = NamePool()
    fn = names.func()
    param = names.param()
    result_var = names.var()
    counter = names.var()

    groups = _pick_membership_groups(min(depth, 3))
    while len(groups) < depth:
        groups.append(random.choice(groups)[:])

    all_pool = list({v for g in groups for v in g})

    for _attempt in range(200):
        input_val = random.choice(all_pool + ["unknown", "other", "none"])

        # Pre-choose remap targets for each step
        remaps: list[str] = []
        for i in range(depth - 1):
            if random.random() < 0.6 and i + 1 < len(groups):
                remaps.append(random.choice(groups[i + 1]))
            else:
                remaps.append("done")

        # Build code
        lines: list[str] = []
        lines.append(f"def {fn}({param}):")
        lines.append(f"    {counter} = 0")
        for i in range(depth):
            group_str = ", ".join(f'"{v}"' for v in groups[i])
            lines.append(f"    if {param} in [{group_str}]:")
            lines.append(f"        {counter} = {counter} + 1")
            if i < depth - 1:
                lines.append(f'        {param} = "{remaps[i]}"')
        lines.append(f"    return {counter}")
        lines.append("")
        lines.append(f'{result_var} = {fn}("{input_val}")')

        code = "\n".join(lines)
        try:
            ns: dict = {"__builtins__": {"True": True, "False": False, "None": None}}
            exec(code, ns, ns)
            actual = ns[result_var]
            if not (0 <= actual <= 9):
                continue
            return {
                "prompt": make_prompt(code, result_var),
                "code": code,
                "answer": str(actual),
                "metadata": {
                    "branch_type": "sequential_if",
                    "depth": depth,
                    "condition_type": "membership",
                    "result_var": result_var,
                },
            }
        except Exception:
            continue

    return None


def _gen_sequential_if_boolean(depth: int) -> dict | None:
    """sequential_if + boolean_flag: sequential boolean checks with state toggling.

    Example (depth=2)::

        def check(enable, verbose):
            count = 0
            if enable:
                count = count + 1
                verbose = not verbose
            if verbose:
                count = count + 1
            return count

        result = check(True, False)  # answer: 2
    """
    names = NamePool()
    fn = names.func()
    result_var = names.var()
    counter = names.var()

    flag_names = [names.distractor() for _ in range(depth)]
    # Each step: if flag_i (possibly negated), counter += 1, toggle next flag
    negations = [random.choice([True, False]) for _ in range(depth)]

    for _attempt in range(200):
        flag_vals = [random.choice([True, False]) for _ in range(depth)]

        # Simulate
        flags = list(flag_vals)
        count = 0
        for i in range(depth):
            cond = (not flags[i]) if negations[i] else flags[i]
            if cond:
                count += 1
                # Toggle the next flag if exists
                if i < depth - 1:
                    flags[i + 1] = not flags[i + 1]

        if not (0 <= count <= 9):
            continue

        sig = ", ".join(flag_names)
        lines: list[str] = []
        lines.append(f"def {fn}({sig}):")
        lines.append(f"    {counter} = 0")
        for i in range(depth):
            cond_expr = f"not {flag_names[i]}" if negations[i] else flag_names[i]
            lines.append(f"    if {cond_expr}:")
            lines.append(f"        {counter} = {counter} + 1")
            if i < depth - 1:
                lines.append(f"        {flag_names[i + 1]} = not {flag_names[i + 1]}")
        lines.append(f"    return {counter}")
        lines.append("")
        call_args = ", ".join(str(v) for v in flag_vals)
        lines.append(f"{result_var} = {fn}({call_args})")

        code = "\n".join(lines)
        return {
            "prompt": make_prompt(code, result_var),
            "code": code,
            "answer": str(count),
            "metadata": {
                "branch_type": "sequential_if",
                "depth": depth,
                "condition_type": "boolean_flag",
                "result_var": result_var,
            },
        }
    return None


# ====================================================================
# Dispatch table
# ====================================================================
_GENERATORS = {
    ("elif_chain", "numeric"): _gen_elif_chain_numeric,
    ("elif_chain", "membership"): _gen_elif_chain_membership,
    ("elif_chain", "boolean_flag"): _gen_elif_chain_boolean,
    ("guard_clause", "numeric"): _gen_guard_clause_numeric,
    ("guard_clause", "membership"): _gen_guard_clause_membership,
    ("guard_clause", "boolean_flag"): _gen_guard_clause_boolean,
    ("sequential_if", "numeric"): _gen_sequential_if_numeric,
    ("sequential_if", "membership"): _gen_sequential_if_membership,
    ("sequential_if", "boolean_flag"): _gen_sequential_if_boolean,
}


# ====================================================================
# Public API
# ====================================================================
def generate_dataset(
    seed: int = 42,
    samples_per_config: int = SAMPLES_PER_CONFIG,
) -> list[dict]:
    random.seed(seed)
    dataset: list[dict] = []

    for branch_type in DIMENSIONS["branch_type"]:
        for depth in DIMENSIONS["depth"]:
            for condition_type in DIMENSIONS["condition_type"]:
                gen_fn = _GENERATORS[(branch_type, condition_type)]
                for idx in range(samples_per_config):
                    for _attempt in range(200):
                        sample = gen_fn(depth)
                        if sample is not None:
                            sample["id"] = (
                                f"cond_{branch_type}_{condition_type}_d{depth}_{idx:03d}"
                            )
                            sample["metadata"]["sample_idx"] = idx
                            dataset.append(sample)
                            break

    return dataset
