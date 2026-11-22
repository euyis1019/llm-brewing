"""Locate data, results and model weights for the study scripts.

The repository root is the default workspace: study inputs live in
experiments/<study>/data and runs write to experiments/<study>/results.
BREWING_WORKSPACE selects another directory with the same layout.
BREWING_MODEL_ROOT selects the directory holding model weights.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STUDIES = ('depth_intervention', 'gen_depth')


def workspace_root() -> Path:
    value = os.environ.get('BREWING_WORKSPACE')
    return Path(value).expanduser().resolve() if value else REPO


def experiment_root(name: str) -> Path:
    if name not in STUDIES:
        raise ValueError(f'Unknown study: {name}')
    return workspace_root() / 'experiments' / name


def model_root() -> Path:
    value = os.environ.get('BREWING_MODEL_ROOT')
    return Path(value).expanduser().resolve() if value else REPO / 'models'


def normalized_benchmark_root() -> Path:
    return workspace_root() / 'assets/data/general_benchmarks/benchmark/normalized'
