"""
Shared infrastructure for CUE enhanced data generators.

Provides name randomisation, exec-based validation, and dataset I/O.
"""

import random
import json
from pathlib import Path
from collections import Counter
from typing import Any


# ====================================================================
# Name pools — realistic engineering-style identifiers
# ====================================================================
_FUNC_NAMES = [
    "process", "handle", "compute", "transform", "validate", "parse",
    "encode", "decode", "convert", "extract", "resolve", "dispatch",
    "evaluate", "classify", "normalize", "aggregate", "apply_op",
    "fetch", "lookup", "check", "verify", "sanitize", "format_val",
    "prepare", "finalize", "build", "compose", "unwrap", "wrap",
    "load_val", "store_val", "merge_val", "split_val", "reduce_val",
    "map_val", "collect", "emit", "flush", "absorb", "relay",
]

_VAR_NAMES = [
    "result", "value", "total", "count", "output", "data",
    "status", "score", "level", "idx", "flag",
    "acc", "current", "target", "offset",
    "response", "payload", "config", "state", "buffer", "item",
    "msg", "record", "entry", "token", "signal", "metric",
    "raw", "val", "tmp", "ret", "out", "ans",
]

_PARAM_NAMES = [
    "x", "y", "z", "n", "k", "v", "w", "m", "p", "q",
    "a", "b", "c", "d", "e",
    "src", "dst", "lo", "hi",
    "left", "right", "start", "end",
    "inp", "arg", "elem", "part",
]

_CLASS_NAMES = [
    "Query", "Builder", "Config", "Handler", "Processor",
    "Node", "Context", "Pipeline", "Wrapper", "Container",
    "Tracker", "Scanner", "Resolver", "Adapter", "Registry",
    "Encoder", "Decoder", "Filter", "Mapper", "Reducer",
]

_DISTRACTOR_PARAMS = [
    "verbose", "debug", "timeout", "retries", "encoding",
    "mode", "cache", "strict", "force", "quiet",
    "dry_run", "prefix", "suffix", "tag", "label",
    "priority", "channel", "region", "version", "locale",
]

_METHOD_NAMES = [
    "step", "apply", "run", "execute", "advance",
    "update", "push", "pop", "shift", "rotate",
    "filter", "select", "reject", "trim", "pad",
]

_DICT_KEYS = [
    "timeout", "retries", "max_size", "threshold", "interval",
    "rate", "limit", "offset", "depth", "width",
    "priority", "weight", "capacity", "duration", "delay",
]


class NamePool:
    """Picks unique names within a single sample to avoid collisions."""

    def __init__(self):
        self._used: set[str] = set()
        self._pools = {
            "func": list(_FUNC_NAMES),
            "var": list(_VAR_NAMES),
            "param": list(_PARAM_NAMES),
            "cls": list(_CLASS_NAMES),
            "distractor": list(_DISTRACTOR_PARAMS),
            "method": list(_METHOD_NAMES),
            "key": list(_DICT_KEYS),
        }
        for pool in self._pools.values():
            random.shuffle(pool)

    def func(self) -> str:
        return self._pick("func")

    def var(self) -> str:
        return self._pick("var")

    def param(self) -> str:
        return self._pick("param")

    def cls(self) -> str:
        return self._pick("cls")

    def distractor(self) -> str:
        return self._pick("distractor")

    def method(self) -> str:
        return self._pick("method")

    def key(self) -> str:
        return self._pick("key")

    def _pick(self, kind: str) -> str:
        pool = self._pools[kind]
        for i, name in enumerate(pool):
            if name not in self._used:
                self._used.add(name)
                pool.pop(i)
                return name
        j = 0
        while f"_{kind}{j}" in self._used:
            j += 1
        name = f"_{kind}{j}"
        self._used.add(name)
        return name

    def reserve(self, name: str):
        """Mark a name as used so it won't be picked."""
        self._used.add(name)


# ====================================================================
# Safe execution environment
# ====================================================================
_SAFE_BUILTINS = {
    "range": range, "len": len, "sum": sum, "min": min, "max": max,
    "abs": abs, "int": int, "str": str, "float": float, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "enumerate": enumerate, "zip": zip, "sorted": sorted,
    "all": all, "any": any, "isinstance": isinstance, "type": type,
    "True": True, "False": False, "None": None,
    "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError,
    "print": lambda *a, **kw: None,
    "__build_class__": __build_class__,
    "__name__": "__exec__",
}


def exec_verify(code: str, result_var: str = "result") -> Any:
    """Execute *code* in a sandbox and return the value of *result_var*.

    Uses a single namespace so that function definitions are visible
    to other functions defined in the same code block.
    """
    ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    exec(code, ns, ns)
    if result_var not in ns:
        raise KeyError(f"'{result_var}' not found after execution")
    return ns[result_var]


# ====================================================================
# Correctness check (shared across all tasks)
# ====================================================================
def is_correct(output: str, expected: str) -> bool:
    """Check if model output matches expected single-digit answer."""
    cleaned = output.strip()
    if not cleaned:
        return False
    if cleaned == expected or cleaned[0] == expected:
        return True
    if len(cleaned) >= 2 and cleaned[0] in "\"'" and cleaned[-1] in "\"'":
        if cleaned[1:-1].strip() == expected:
            return True
    return False


# ====================================================================
# Prompt construction
# ====================================================================
def make_prompt(code: str, result_var: str = "result") -> str:
    """Append the standard CSD-compatible query suffix."""
    return code.rstrip() + f'\n# The value of {result_var} is "'


# ====================================================================
# Dataset validation & I/O
# ====================================================================
def validate_and_save(
    dataset: list[dict],
    task_name: str,
    output_dir: Path,
    dim_names: list[str],
) -> dict:
    """Exec-verify every sample, print stats, save JSON. Returns summary."""
    issues: list[str] = []
    for sample in dataset:
        sid = sample["id"]
        # exec check
        try:
            rv = sample["metadata"].get("result_var", "result")
            actual = exec_verify(sample["code"], rv)
            if str(actual) != sample["answer"]:
                issues.append(f"{sid}: exec={actual}, want={sample['answer']}")
        except Exception as e:
            issues.append(f"{sid}: exec error — {e}")
        # range check
        ans = sample["answer"]
        if not (ans.isdigit() and 0 <= int(ans) <= 9):
            issues.append(f"{sid}: answer '{ans}' not in 0-9")

    answer_dist = Counter(s["answer"] for s in dataset)

    print(f"\n{'=' * 60}")
    print(f"{task_name}: {len(dataset)} samples, {len(issues)} issues")
    print(f"{'=' * 60}")
    if issues:
        for iss in issues[:15]:
            print(f"  !! {iss}")
        if len(issues) > 15:
            print(f"  ... and {len(issues) - 15} more")
    print(f"  Answers: {dict(sorted(answer_dist.items()))}")
    for dn in dim_names:
        dist = Counter(str(s["metadata"].get(dn, "?")) for s in dataset)
        print(f"  {dn}: {dict(sorted(dist.items()))}")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{task_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"  -> {path}")

    return {"total": len(dataset), "issues": len(issues)}
