"""
Task 1 — Value Tracking (enhanced pure_copying)

The model must track a value as it passes through realistic code constructs
without being transformed. Inspired by SWE-bench patterns:
  - django-13195: data encoded/decoded through container layers
  - django-13344: object passed through middleware chain
  - xarray-7229: parameter threaded through 5-level call stack

Mechanisms:
  function_chain  — value passes through N nested function calls
  container       — value packed into list/dict, extracted by index/key
  method_chain    — class with builder-pattern method chaining

Dimensions:
  mechanism     in {function_chain, container, method_chain}
  depth         in {1, 2, 3}        layers of indirection
  distractors   in {0, 1, 2}        unused params / fields
"""

import random
from .base import NamePool, make_prompt

TASK_NAME = "value_tracking"

DIMENSIONS = {
    "mechanism": ["function_chain", "container", "method_chain"],
    "depth": [1, 2, 3],
    "distractors": [0, 1, 2],
}

SAMPLES_PER_CONFIG = 150  # 3x3x3x30 = 810


# ====================================================================
# Realistic code templates — give functions context-appropriate bodies
# ====================================================================

# Patterns for function_chain: each is a (def_template, call_template) pair.
# The def body must return the value parameter unchanged.
# {fn} = func name, {param} = value param, {dp} = distractor params (comma-separated)
# {prev_call} = call to previous function in chain

_FC_BODY_STYLES = [
    # Style 0: simple passthrough (baseline)
    "passthrough",
    # Style 1: validate-and-return (like Django middleware)
    "validate",
    # Style 2: log-and-forward (like logging middleware)
    "log_forward",
    # Style 3: conditional no-op (guard clause that always passes)
    "guard",
]


def _fc_make_body(style: str, param: str, dist_params: list[str],
                  inner_call: str | None, names: NamePool) -> list[str]:
    """Generate function body lines (indented) for function_chain."""
    if style == "passthrough":
        if inner_call is not None:
            return [f"    return {inner_call}"]
        return [f"    return {param}"]

    if style == "validate":
        # Validates param type/range but always returns it unchanged
        body = []
        if dist_params:
            body.append(f"    if {dist_params[0]}:")
            body.append(f"        _ = type({param})")
        if inner_call is not None:
            body.append(f"    return {inner_call}")
        else:
            body.append(f"    return {param}")
        return body

    if style == "log_forward":
        # Simulates a log call, then passes value through
        body = []
        local = names.var() if random.random() < 0.5 else None
        if local:
            if inner_call is not None:
                body.append(f"    {local} = {inner_call}")
                body.append(f"    return {local}")
            else:
                body.append(f"    {local} = {param}")
                body.append(f"    return {local}")
        else:
            if inner_call is not None:
                body.append(f"    return {inner_call}")
            else:
                body.append(f"    return {param}")
        return body

    if style == "guard":
        # Guard clause that never triggers, then return
        body = []
        sentinel = random.randint(100, 999)
        body.append(f"    if {param} == {sentinel}:")
        body.append(f"        return {random.randint(0, 9)}")
        if inner_call is not None:
            body.append(f"    return {inner_call}")
        else:
            body.append(f"    return {param}")
        return body

    # fallback
    if inner_call is not None:
        return [f"    return {inner_call}"]
    return [f"    return {param}"]


# ====================================================================
# Mechanism: function_chain
# ====================================================================
def _gen_function_chain(depth: int, distractors: int) -> dict | None:
    """Value passes through *depth* nested function calls unchanged.

    Inspired by Django middleware chains and xarray parameter threading.
    Functions may include validation, logging, or guard clauses that
    never alter the tracked value.

    depth=2, distractors=1 example::

        def validate(src, verbose):
            if verbose:
                _ = type(src)
            return src

        def dispatch(y, timeout):
            return validate(y, False)

        output = dispatch(7, 3)
    """
    names = NamePool()
    answer = random.randint(0, 9)

    func_names = [names.func() for _ in range(depth)]
    param_names = [names.param() for _ in range(depth)]
    dist_names = [[names.distractor() for _ in range(distractors)]
                  for _ in range(depth)]
    result_var = names.var()

    # Pick a body style for this sample
    style = random.choice(_FC_BODY_STYLES)

    lines: list[str] = []
    for i in range(depth):
        fn = func_names[i]
        param = param_names[i]
        d_params = dist_names[i]
        sig = ", ".join([param] + d_params)

        if i == 0:
            # Innermost function: no inner call
            lines.append(f"def {fn}({sig}):")
            body = _fc_make_body(style, param, d_params, None, names)
            lines.extend(body)
        else:
            prev_fn = func_names[i - 1]
            prev_d_vals = [str(random.randint(0, 9)) for _ in dist_names[i - 1]]
            call_args = ", ".join([param] + prev_d_vals)
            inner_call = f"{prev_fn}({call_args})"
            lines.append(f"def {fn}({sig}):")
            body = _fc_make_body(style, param, d_params, inner_call, names)
            lines.extend(body)
        lines.append("")

    outer = func_names[-1]
    outer_d_vals = [str(random.randint(0, 9)) for _ in dist_names[-1]]
    call_args = ", ".join([str(answer)] + outer_d_vals)
    lines.append(f"{result_var} = {outer}({call_args})")

    code = "\n".join(lines)
    return {
        "prompt": make_prompt(code, result_var),
        "code": code,
        "answer": str(answer),
        "metadata": {
            "mechanism": "function_chain",
            "depth": depth,
            "distractors": distractors,
            "result_var": result_var,
        },
    }


# ====================================================================
# Container styles for richer pack/unpack patterns
# ====================================================================
_CONTAINER_STYLES = ["list", "dict", "nested_dict", "tuple_like"]


def _make_container_layer(style: str, val_param: str,
                          dist_params: list[str],
                          names: NamePool) -> tuple[str, list[str], str]:
    """Build a container packing expression and its access path.

    Returns (body_expr, extra_sig_params, access_suffix).
    body_expr uses val_param and dist_params.
    access_suffix is applied to the packed result to retrieve the value.
    """
    if style == "list":
        n_slots = 2 + len(dist_params)
        val_pos = random.randint(0, n_slots - 1)
        slots = [str(random.randint(0, 9)) for _ in range(n_slots)]
        slots[val_pos] = val_param
        # Fill distractor params into random slots
        for dp in dist_params:
            free = [k for k in range(n_slots)
                    if slots[k] != val_param and slots[k].isdigit()]
            if free:
                pos = random.choice(free)
                slots[pos] = dp
        body = f"[{', '.join(slots)}]"
        access = f"[{val_pos}]"
        return body, [], access

    if style == "dict":
        val_key = names.key()
        entries = [f'"{val_key}": {val_param}']
        for dp in dist_params:
            dk = names.key()
            entries.append(f'"{dk}": {dp}')
        extra_key = names.key()
        entries.append(f'"{extra_key}": {random.randint(0, 9)}')
        random.shuffle(entries)
        body = "{" + ", ".join(entries) + "}"
        access = f'["{val_key}"]'
        return body, [], access

    if style == "nested_dict":
        # Value nested one level deeper: {"outer": {"inner": val}}
        outer_key = names.key()
        inner_key = names.key()
        inner_dict = '{' + f'"{inner_key}": {val_param}' + '}'
        entries = [f'"{outer_key}": {inner_dict}']
        for dp in dist_params:
            dk = names.key()
            entries.append(f'"{dk}": {dp}')
        random.shuffle(entries)
        body = "{" + ", ".join(entries) + "}"
        access = f'["{outer_key}"]["{inner_key}"]'
        return body, [], access

    if style == "tuple_like":
        # Use a list but access with negative index or len-based
        n_slots = 2 + len(dist_params)
        val_pos = random.randint(0, n_slots - 1)
        slots = [str(random.randint(0, 9)) for _ in range(n_slots)]
        slots[val_pos] = val_param
        for dp in dist_params:
            free = [k for k in range(n_slots)
                    if slots[k] != val_param and slots[k].isdigit()]
            if free:
                pos = random.choice(free)
                slots[pos] = dp
        body = f"[{', '.join(slots)}]"
        # Use negative indexing half the time for variety
        if random.random() < 0.5 and val_pos > 0:
            neg_idx = val_pos - n_slots
            access = f"[{neg_idx}]"
        else:
            access = f"[{val_pos}]"
        return body, [], access

    # Fallback to list
    body = f"[{val_param}]"
    return body, [], "[0]"


# ====================================================================
# Mechanism: container (list / dict pack-unpack)
# ====================================================================
def _gen_container(depth: int, distractors: int) -> dict | None:
    """Value packed into containers and extracted back.

    Inspired by Django's encode/decode through container layers and
    xarray's nested data structure access patterns.

    depth=2, distractors=1 example::

        def encode(msg, tag):
            return {"payload": msg, "tag": tag}

        def wrap(data, status):
            return [status, data]

        def extract(packet):
            return packet[1]["payload"]

        raw = encode(3, 8)
        wrapped = wrap(raw, 0)
        result = extract(wrapped)
    """
    names = NamePool()
    answer = random.randint(0, 9)
    result_var = names.var()

    layers: list[dict] = []
    # Pick container styles, avoiding nested_dict if we have many layers
    # to keep access paths tractable
    available_styles = list(_CONTAINER_STYLES)

    for layer_idx in range(depth):
        fn_pack = names.func()
        val_param = names.param()
        dist_p = [names.distractor() for _ in range(distractors)]

        style = random.choice(available_styles)
        body, _, access = _make_container_layer(
            style, val_param, dist_p, names
        )

        layers.append({
            "fn": fn_pack,
            "val_param": val_param,
            "dist_params": dist_p,
            "sig": ", ".join([val_param] + dist_p),
            "body": body,
            "access": access,
        })

    # -- generate code --
    lines: list[str] = []
    for layer in layers:
        lines.append(f"def {layer['fn']}({layer['sig']}):")
        lines.append(f"    return {layer['body']}")
        lines.append("")

    # Compose extraction function
    extract_fn = names.func()
    outer_param = names.param()
    access_chain = outer_param
    for layer in reversed(layers):
        access_chain += layer["access"]
    lines.append(f"def {extract_fn}({outer_param}):")
    lines.append(f"    return {access_chain}")
    lines.append("")

    # Build call chain with intermediate variables
    intermediate_vars: list[str] = []
    for i, layer in enumerate(layers):
        iv = names.var()
        intermediate_vars.append(iv)
        dist_vals = [str(random.randint(0, 9)) for _ in layer["dist_params"]]
        if i == 0:
            call_args = ", ".join([str(answer)] + dist_vals)
        else:
            call_args = ", ".join([intermediate_vars[i - 1]] + dist_vals)
        lines.append(f"{iv} = {layer['fn']}({call_args})")

    last_iv = intermediate_vars[-1] if intermediate_vars else str(answer)
    lines.append(f"{result_var} = {extract_fn}({last_iv})")

    code = "\n".join(lines)
    return {
        "prompt": make_prompt(code, result_var),
        "code": code,
        "answer": str(answer),
        "metadata": {
            "mechanism": "container",
            "depth": depth,
            "distractors": distractors,
            "result_var": result_var,
        },
    }


# ====================================================================
# Method chain styles for richer builder patterns
# ====================================================================
_MC_STYLES = [
    "builder",       # classic builder: each method returns new instance
    "fluent_self",   # fluent API: methods return self (mutate in place)
    "wrapper",       # wrapper class: methods create new wrapper around value
]


# ====================================================================
# Mechanism: method_chain (builder-pattern class)
# ====================================================================
def _gen_method_chain(depth: int, distractors: int) -> dict | None:
    """Class with chained methods that pass the value through.

    Inspired by Django QuerySet chaining and xarray's fluent API.
    Three sub-styles for variety:
      - builder: each method returns a new instance (immutable pattern)
      - fluent_self: methods return self after no-op processing
      - wrapper: methods wrap/unwrap through a new wrapper object

    depth=2, distractors=1 example (builder style)::

        class Pipeline:
            def __init__(self, src, debug=None):
                self.src = src
            def filter(self, timeout=0):
                return Pipeline(self.src)
            def advance(self, retries=0):
                return Pipeline(self.src)
            def execute(self):
                return self.src

        result = Pipeline(7, 3).filter(5).advance(1).execute()
    """
    names = NamePool()
    answer = random.randint(0, 9)
    result_var = names.var()
    cls_name = names.cls()
    val_attr = names.param()   # internal attribute name
    get_method = names.method()

    method_names = [names.method() for _ in range(depth)]
    style = random.choice(_MC_STYLES)

    lines: list[str] = []
    lines.append(f"class {cls_name}:")

    # __init__
    init_dist = [names.distractor() for _ in range(distractors)]
    init_sig = ", ".join(["self", val_attr] + [f"{d}=None" for d in init_dist])
    lines.append(f"    def __init__({init_sig}):")
    lines.append(f"        self.{val_attr} = {val_attr}")
    lines.append("")

    # chain methods
    for mn in method_names:
        m_dist = [names.distractor() for _ in range(distractors)]
        m_sig = ", ".join(["self"] + [f"{d}=0" for d in m_dist])
        lines.append(f"    def {mn}({m_sig}):")

        if style == "builder":
            # Return a new instance with same value
            lines.append(f"        return {cls_name}(self.{val_attr})")
        elif style == "fluent_self":
            # No-op on value, return self
            if distractors > 0 and random.random() < 0.4:
                # Add a harmless attribute set for realism
                dummy_attr = names.var()
                lines.append(f"        self.{dummy_attr} = {random.randint(0, 9)}")
            lines.append(f"        return self")
        elif style == "wrapper":
            # Create new instance, simulating a pipeline stage
            local = names.var()
            lines.append(f"        {local} = self.{val_attr}")
            lines.append(f"        return {cls_name}({local})")
        lines.append("")

    # terminal get method
    lines.append(f"    def {get_method}(self):")
    lines.append(f"        return self.{val_attr}")
    lines.append("")

    # Build call chain
    init_dist_vals = [str(random.randint(0, 9)) for _ in range(distractors)]
    init_args = ", ".join([str(answer)] + init_dist_vals)
    chain = f"{cls_name}({init_args})"
    for mn in method_names:
        m_dist_vals = [str(random.randint(0, 9)) for _ in range(distractors)]
        m_args = ", ".join(m_dist_vals) if m_dist_vals else ""
        chain += f".{mn}({m_args})"
    chain += f".{get_method}()"

    lines.append(f"{result_var} = {chain}")

    code = "\n".join(lines)
    return {
        "prompt": make_prompt(code, result_var),
        "code": code,
        "answer": str(answer),
        "metadata": {
            "mechanism": "method_chain",
            "depth": depth,
            "distractors": distractors,
            "result_var": result_var,
        },
    }


# ====================================================================
# Public API
# ====================================================================
_GENERATORS = {
    "function_chain": _gen_function_chain,
    "container": _gen_container,
    "method_chain": _gen_method_chain,
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
                                f"vt_{mechanism}_d{depth}_x{distractors}_{idx:03d}"
                            )
                            sample["metadata"]["sample_idx"] = idx
                            dataset.append(sample)
                            break

    return dataset
