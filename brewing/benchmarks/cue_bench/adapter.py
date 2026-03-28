"""Datagen integration — mapping from datagen output to Sample objects."""

from __future__ import annotations

from typing import Any

from brewing.schema import Sample

# Map from datagen task names to CUE-Bench subset names.
# After the rename, these are now identity mappings.
_DATAGEN_TO_SUBSET = {
    "value_tracking": "value_tracking",
    "computing": "computing",
    "conditional": "conditional",
    "function_call": "function_call",
    "loop": "loop",
    "loop_unrolled": "loop_unrolled",
}

# Map from subset names to difficulty dimension keys (matching datagen metadata)
_DIFFICULTY_KEYS: dict[str, list[str]] = {
    "value_tracking": ["mechanism", "depth", "distractors"],
    "computing": ["structure", "steps", "operators"],
    "conditional": ["branch_type", "depth", "condition_type"],
    "function_call": ["mechanism", "depth", "distractors"],
    "loop": ["body_type", "iterations", "init_offset"],
    "loop_unrolled": ["body_type", "iterations", "init_offset"],
}


def datagen_sample_to_brewing(
    raw: dict[str, Any],
    task_name: str,
    subset_name: str,
) -> Sample:
    """Convert a datagen dict to a Brewing Sample."""
    meta = raw.get("metadata", {})
    difficulty_keys = _DIFFICULTY_KEYS.get(subset_name, [])
    difficulty = {k: meta[k] for k in difficulty_keys if k in meta}
    remaining_meta = {k: v for k, v in meta.items() if k not in difficulty_keys}

    return Sample(
        id=raw["id"],
        benchmark="CUE-Bench",
        subset=subset_name,
        prompt=raw["prompt"],
        answer=raw["answer"],
        difficulty=difficulty if difficulty else None,
        metadata=remaining_meta if remaining_meta else None,
    )


def get_subset_for_datagen(datagen_name: str) -> str | None:
    """Return the CUE-Bench subset name for a datagen task name."""
    return _DATAGEN_TO_SUBSET.get(datagen_name)


def get_datagen_for_subset(subset_name: str) -> str | None:
    """Return the datagen task name for a CUE-Bench subset name."""
    for dg_name, sub_name in _DATAGEN_TO_SUBSET.items():
        if sub_name == subset_name:
            return dg_name
    return None


def get_datagen_task_names() -> list[str]:
    """Return all known datagen task names."""
    return list(_DATAGEN_TO_SUBSET.keys())
