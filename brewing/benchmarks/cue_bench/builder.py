"""Dataset building and loading for CUE-Bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brewing.schema import DatasetManifest, DatasetPurpose, Sample

from .adapter import (
    _DATAGEN_TO_SUBSET,
    datagen_sample_to_brewing,
    get_datagen_for_subset,
)
from .fixtures import FIXTURE_SAMPLES
from .spec import CUE_BENCH


def load_generated_dataset(
    data_dir: Path,
    task_name: str | None = None,
) -> list[Sample]:
    """Load datagen output files and convert to Brewing Samples.

    Args:
        data_dir: Directory containing JSON files from datagen
        task_name: If given, load only this task. Otherwise load all.

    Returns:
        List of Sample objects
    """
    samples: list[Sample] = []
    tasks = [task_name] if task_name else list(_DATAGEN_TO_SUBSET.keys())

    for tn in tasks:
        subset_name = _DATAGEN_TO_SUBSET.get(tn, tn)
        # Try direct path first, then eval/ and train/ subdirectories
        path = data_dir / f"{tn}.json"
        if not path.exists():
            path = data_dir / "eval" / f"{tn}.json"
        if not path.exists():
            path = data_dir / "train" / f"{tn}.json"
        if not path.exists():
            continue
        with open(path) as f:
            raw_samples = json.load(f)
        for raw in raw_samples:
            samples.append(datagen_sample_to_brewing(raw, tn, subset_name))

    return samples


def generate_and_convert(
    subset_name: str,
    seed: int = 42,
    samples_per_config: int | None = None,
) -> list[Sample]:
    """Run datagen for a subset and convert to Brewing Samples.

    Imports datagen modules dynamically to avoid hard dependency.
    """
    datagen_name = get_datagen_for_subset(subset_name)
    if datagen_name is None:
        raise ValueError(f"No datagen module for subset '{subset_name}'")

    try:
        import importlib
        mod = importlib.import_module(
            f".datagen.{datagen_name}",
            package="brewing.benchmarks.cue_bench",
        )
    except ImportError:
        raise ImportError(
            f"datagen.{datagen_name} not found in "
            f"brewing.benchmarks.cue_bench.datagen."
        )

    kwargs: dict[str, Any] = {"seed": seed}
    if samples_per_config is not None:
        kwargs["samples_per_config"] = samples_per_config

    raw_samples = mod.generate_dataset(**kwargs)
    return [
        datagen_sample_to_brewing(raw, datagen_name, subset_name)
        for raw in raw_samples
    ]


def build_eval_dataset(
    subsets: list[str] | None = None,
    seed: int = 42,
    samples_per_config: int | None = None,
    data_dir: Path | None = None,
) -> tuple[DatasetManifest, list[Sample]]:
    """Build or load an eval dataset for CUE-Bench.

    Tries to load from data_dir first; falls back to generation.
    """
    if subsets is None:
        subsets = CUE_BENCH.subset_names

    all_samples: list[Sample] = []

    for subset_name in subsets:
        # Try loading from disk first
        if data_dir is not None:
            loaded = load_generated_dataset(data_dir, subset_name)
            if loaded:
                all_samples.extend(loaded)
                continue

        # Fall back to generation
        try:
            generated = generate_and_convert(
                subset_name, seed=seed,
                samples_per_config=samples_per_config,
            )
            all_samples.extend(generated)
        except (ImportError, ValueError):
            # If datagen is not available, use fixture
            for s in FIXTURE_SAMPLES:
                if s.subset == subset_name:
                    all_samples.append(s)

    version = f"eval-seed{seed}"
    if samples_per_config is not None:
        version += f"-n{samples_per_config}"

    manifest = DatasetManifest(
        dataset_id=f"cue-bench-{'-'.join(subsets)}-{version}",
        purpose=DatasetPurpose.EVAL,
        benchmark="CUE-Bench",
        subset=subsets[0] if len(subsets) == 1 else None,
        sample_ids=[s.id for s in all_samples],
        generation_config={"seed": seed, "samples_per_config": samples_per_config},
        seed=seed,
        version=version,
    )

    return manifest, all_samples
