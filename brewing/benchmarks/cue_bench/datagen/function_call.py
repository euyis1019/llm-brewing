"""
Task 4 — Function Call (redesigned)

Controlled pair with Value Tracking: value crosses function boundaries WITH
computation at each level. Value Tracking passes values WITHOUT computation;
Function Call always transforms the value.

Inspired by SWE-bench patterns:
  - django-13195: encode/decode through container layers (container_relay)
  - django-14792: sign flip across function boundaries (arithmetic)
  - xarray-7229: parameter threaded through call stack (conditional_return)

Dimensions:
  mechanism    ∈ {arithmetic, container_relay, conditional_return}
  depth        ∈ {1, 2, 3}        function nesting depth
  distractors  ∈ {0, 1, 2}        unused parameters per function

27 configs × 30 samples = 810 total
"""

import random
from .base import NamePool, make_prompt

TASK_NAME = "function_call"

DIMENSIONS = {
    "mechanism": ["arithmetic", "container_relay", "conditional_return"],
    "depth": [1, 2, 3],
    "distractors": [0, 1, 2],
}

SAMPLES_PER_CONFIG = 150  # 3×3×3×30 = 810


# ====================================================================
# Mechanism: arithmetic
# ====================================================================
def _gen_arithmetic(depth: int, distractors: int) -> dict | None:
    """Each function layer does a simple arithmetic operation.

    depth=2, distractors=1 example::

        val = 1
        def handle(x, verbose):
            return x + 3
        def process(y, timeout):
            return handle(y, False) + 1
        result = process(val, 8)
        # answer: 5
    """
    names = NamePool()
    result_var = names.var()
    init_var = names.var()

    func_names = [names.func() for _ in range(depth)]
    param_names = [names.param() for _ in range(depth)]
    dist_names = [
        [names.distractor() for _ in range(distractors)]
        for _ in range(depth)
    ]

    for _attempt in range(200):
        # Each layer adds or subtracts a small delta (never zero)
        deltas = []
        ops = []
        for _ in range(depth):
            op = random.choice(["+", "-"])
            delta = random.randint(1, 3)
            deltas.append(delta)
            ops.append(op)

        # Find a starting value that keeps all intermediates in 0-9
        init_value = random.randint(0, 9)
        running = init_value
        ok = True
        for i in range(depth):
            if ops[i] == "+":
                running += deltas[i]
            else:
                running -= deltas[i]
            if running < 0 or running > 9:
                ok = False
                break
        if not ok:
            continue
        answer = running

        lines: list[str] = []
        lines.append(f"{init_var} = {init_value}")
        lines.append("")

        # Define functions from innermost to outermost
        for i in range(depth):
            fn = func_names[i]
            param = param_names[i]
            d_params = dist_names[i]
            sig = ", ".join([param] + d_params)

            lines.append(f"def {fn}({sig}):")
            if i == 0:
                # Innermost function: direct arithmetic on param
                lines.append(f"    return {param} {ops[i]} {deltas[i]}")
            else:
                # Calls previous (inner) function, then applies own op
                prev_fn = func_names[i - 1]
                prev_d_vals = [
                    str(random.choice([True, False, random.randint(0, 9)]))
                    for _ in dist_names[i - 1]
                ]
                inner_call_args = ", ".join([param] + prev_d_vals)
                lines.append(
                    f"    return {prev_fn}({inner_call_args}) {ops[i]} {deltas[i]}"
                )
            lines.append("")

        # Call outermost function
        outer_fn = func_names[-1]
        outer_d_vals = [str(random.randint(0, 9)) for _ in dist_names[-1]]
        call_args = ", ".join([init_var] + outer_d_vals)
        lines.append(f"{result_var} = {outer_fn}({call_args})")

        code = "\n".join(lines)
        return {
            "prompt": make_prompt(code, result_var),
            "code": code,
            "answer": str(answer),
            "metadata": {
                "mechanism": "arithmetic",
                "depth": depth,
                "distractors": distractors,
                "result_var": result_var,
            },
        }
    return None


# ====================================================================
# Mechanism: container_relay
# ====================================================================
def _gen_container_relay(depth: int, distractors: int) -> dict | None:
    """Value passes through containers with computation at each hop.

    depth=2, distractors=1 example::

        val = 5
        def pack(x, tag):
            return {"data": x + 1, "tag": tag}
        def relay(packet):
            return packet["data"] + 2
        result = relay(pack(val, "info"))
        # answer: 8
    """
    names = NamePool()
    result_var = names.var()
    init_var = names.var()

    func_names = [names.func() for _ in range(depth)]
    param_names = [names.param() for _ in range(depth)]
    key_names = [names.key() for _ in range(depth)]
    dist_names = [
        [names.distractor() for _ in range(distractors)]
        for _ in range(depth)
    ]

    for _attempt in range(200):
        # Each layer adds a small positive delta when packing
        deltas = [random.randint(1, 2) for _ in range(depth)]

        init_value = random.randint(0, 9)
        running = init_value
        ok = True
        for delta in deltas:
            running += delta
            if running > 9:
                ok = False
                break
        if not ok:
            continue
        answer = running

        lines: list[str] = []
        lines.append(f"{init_var} = {init_value}")
        lines.append("")

        # Build functions: each takes a value (or container), computes,
        # and returns a container (except the last which extracts)
        for i in range(depth):
            fn = func_names[i]
            param = param_names[i]
            d_params = dist_names[i]
            data_key = key_names[i]

            if i == 0:
                # First function: takes raw value, packs into dict with computation
                sig = ", ".join([param] + d_params)
                lines.append(f"def {fn}({sig}):")
                # Add distractor keys to the dict
                extra_entries = []
                for dp in d_params:
                    ek = names.key()
                    extra_entries.append(f'"{ek}": {dp}')
                main_entry = f'"{data_key}": {param} + {deltas[i]}'
                all_entries = [main_entry] + extra_entries
                random.shuffle(all_entries)
                dict_body = "{" + ", ".join(all_entries) + "}"
                lines.append(f"    return {dict_body}")
            elif i < depth - 1:
                # Middle function: extracts from previous container,
                # computes, repacks
                prev_key = key_names[i - 1]
                sig = ", ".join([param] + d_params)
                lines.append(f"def {fn}({sig}):")
                extract_expr = f'{param}["{prev_key}"]'
                extra_entries = []
                for dp in d_params:
                    ek = names.key()
                    extra_entries.append(f'"{ek}": {dp}')
                main_entry = f'"{data_key}": {extract_expr} + {deltas[i]}'
                all_entries = [main_entry] + extra_entries
                random.shuffle(all_entries)
                dict_body = "{" + ", ".join(all_entries) + "}"
                lines.append(f"    return {dict_body}")
            else:
                # Last function: extracts from previous container and
                # adds final delta (returns plain int)
                prev_key = key_names[i - 1]
                sig = ", ".join([param] + d_params)
                lines.append(f"def {fn}({sig}):")
                lines.append(
                    f'    return {param}["{prev_key}"] + {deltas[i]}'
                )
            lines.append("")

        # Build the call chain
        # For depth=1: result = func0(init_var, dist_vals...)
        # For depth=2: result = func1(func0(init_var, d0...), d1...)
        # For depth=3: result = func2(func1(func0(init_var, d0...), d1...), d2...)
        if depth == 1:
            d_vals = [str(random.randint(0, 9)) for _ in dist_names[0]]
            call_args = ", ".join([init_var] + d_vals)
            # depth=1: single function packs and returns dict; we need the
            # answer to be an int.  Restructure: depth=1 function takes raw
            # value, does computation, returns plain int.
            # Re-generate the function to return plain int instead of dict.
            lines_new: list[str] = []
            lines_new.append(f"{init_var} = {init_value}")
            lines_new.append("")
            fn = func_names[0]
            param = param_names[0]
            d_params = dist_names[0]
            sig = ", ".join([param] + d_params)
            lines_new.append(f"def {fn}({sig}):")
            lines_new.append(f"    return {param} + {deltas[0]}")
            lines_new.append("")
            d_vals = [str(random.randint(0, 9)) for _ in d_params]
            call_args = ", ".join([init_var] + d_vals)
            lines_new.append(f"{result_var} = {fn}({call_args})")
            code = "\n".join(lines_new)
        else:
            # Build nested call expression
            d_vals_0 = [str(random.randint(0, 9)) for _ in dist_names[0]]
            inner_args = ", ".join([init_var] + d_vals_0)
            expr = f"{func_names[0]}({inner_args})"
            for i in range(1, depth):
                d_vals_i = [str(random.randint(0, 9)) for _ in dist_names[i]]
                args = ", ".join([expr] + d_vals_i)
                expr = f"{func_names[i]}({args})"
            lines.append(f"{result_var} = {expr}")
            code = "\n".join(lines)

        return {
            "prompt": make_prompt(code, result_var),
            "code": code,
            "answer": str(answer),
            "metadata": {
                "mechanism": "container_relay",
                "depth": depth,
                "distractors": distractors,
                "result_var": result_var,
            },
        }
    return None


# ====================================================================
# Mechanism: conditional_return
# ====================================================================
def _gen_conditional_return(depth: int, distractors: int) -> dict | None:
    """Function contains conditional branch determining return path.

    depth=2, distractors=1 example::

        val = 3
        def route(x, mode):
            if mode > 0:
                return x + 2
            return x
        def dispatch(y, flag):
            return route(y, flag) + 1
        result = dispatch(val, 1)
        # answer: 6
    """
    names = NamePool()
    result_var = names.var()
    init_var = names.var()

    func_names = [names.func() for _ in range(depth)]
    param_names = [names.param() for _ in range(depth)]
    cond_params = [names.param() for _ in range(depth)]
    dist_names = [
        [names.distractor() for _ in range(distractors)]
        for _ in range(depth)
    ]

    for _attempt in range(200):
        # For each layer: condition determines which branch
        # true_delta and false_delta are different, both nonzero
        layer_specs = []
        for _ in range(depth):
            threshold = random.randint(1, 5)
            true_delta = random.randint(1, 3)
            false_delta = random.randint(1, 2)
            # Make the branches distinguishable: one adds, other subtracts
            # or different magnitudes
            branch_type = random.choice(["add_vs_sub", "diff_add"])
            if branch_type == "add_vs_sub":
                true_op, false_op = "+", "-"
            else:
                true_op = "+"
                false_op = "+"
                # Ensure deltas differ
                false_delta = true_delta + 1 if true_delta < 3 else true_delta - 1
            layer_specs.append({
                "threshold": threshold,
                "true_delta": true_delta,
                "true_op": true_op,
                "false_delta": false_delta,
                "false_op": false_op,
            })

        # Pick condition values and compute answer
        init_value = random.randint(0, 9)
        cond_values = [random.randint(0, 9) for _ in range(depth)]
        running = init_value
        ok = True

        for i in range(depth):
            spec = layer_specs[i]
            if cond_values[i] > spec["threshold"]:
                op, delta = spec["true_op"], spec["true_delta"]
            else:
                op, delta = spec["false_op"], spec["false_delta"]
            if op == "+":
                running += delta
            else:
                running -= delta
            if running < 0 or running > 9:
                ok = False
                break

        if not ok:
            continue
        answer = running

        lines: list[str] = []
        lines.append(f"{init_var} = {init_value}")
        lines.append("")

        # Define functions from innermost to outermost
        for i in range(depth):
            fn = func_names[i]
            param = param_names[i]
            cond_p = cond_params[i]
            d_params = dist_names[i]
            sig = ", ".join([param, cond_p] + d_params)
            spec = layer_specs[i]

            lines.append(f"def {fn}({sig}):")
            if i == 0:
                # Innermost: conditional on cond_p
                lines.append(f"    if {cond_p} > {spec['threshold']}:")
                lines.append(
                    f"        return {param} {spec['true_op']} {spec['true_delta']}"
                )
                lines.append(
                    f"    return {param} {spec['false_op']} {spec['false_delta']}"
                )
            else:
                # Outer: calls inner function, then applies own conditional
                prev_fn = func_names[i - 1]
                prev_cond_val = cond_values[i - 1]
                prev_d_vals = [
                    str(random.choice([True, False, random.randint(0, 9)]))
                    for _ in dist_names[i - 1]
                ]
                inner_call_args = ", ".join(
                    [param, str(prev_cond_val)] + prev_d_vals
                )
                inner_call = f"{prev_fn}({inner_call_args})"
                tmp = names.var()
                lines.append(f"    {tmp} = {inner_call}")
                lines.append(f"    if {cond_p} > {spec['threshold']}:")
                lines.append(
                    f"        return {tmp} {spec['true_op']} {spec['true_delta']}"
                )
                lines.append(
                    f"    return {tmp} {spec['false_op']} {spec['false_delta']}"
                )
            lines.append("")

        # Call outermost function
        outer_fn = func_names[-1]
        outer_cond = cond_values[-1]
        outer_d_vals = [str(random.randint(0, 9)) for _ in dist_names[-1]]
        call_args = ", ".join(
            [init_var, str(outer_cond)] + outer_d_vals
        )
        lines.append(f"{result_var} = {outer_fn}({call_args})")

        code = "\n".join(lines)
        return {
            "prompt": make_prompt(code, result_var),
            "code": code,
            "answer": str(answer),
            "metadata": {
                "mechanism": "conditional_return",
                "depth": depth,
                "distractors": distractors,
                "result_var": result_var,
            },
        }
    return None


# ====================================================================
# Public API
# ====================================================================
_GENERATORS = {
    "arithmetic": _gen_arithmetic,
    "container_relay": _gen_container_relay,
    "conditional_return": _gen_conditional_return,
}


def generate_dataset(
    seed: int = 42,
    samples_per_config: int = SAMPLES_PER_CONFIG,
) -> list[dict]:
    random.seed(seed)
    dataset: list[dict] = []

    for mechanism in DIMENSIONS["mechanism"]:
        gen_fn = _GENERATORS[mechanism]
        for depth in DIMENSIONS["depth"]:
            for distractors in DIMENSIONS["distractors"]:
                for idx in range(samples_per_config):
                    for _attempt in range(200):
                        sample = gen_fn(depth, distractors)
                        if sample is not None:
                            sample["id"] = (
                                f"fc_{mechanism}_d{depth}_x{distractors}"
                                f"_{idx:03d}"
                            )
                            sample["metadata"]["sample_idx"] = idx
                            dataset.append(sample)
                            break

    return dataset
