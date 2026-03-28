"""CUE-Bench: 6 code-reasoning tasks for mechanistic interpretability.

Provides:
- BenchmarkSpec with all 6 subset definitions
- Fixture samples for testing (1 per subset)
- Integration with datagen/ for real dataset generation
"""

from brewing.registry import register_benchmark

from .spec import CUE_BENCH
from .fixtures import FIXTURE_SAMPLES
from .adapter import (
    datagen_sample_to_brewing,
    get_datagen_for_subset,
    get_subset_for_datagen,
    get_datagen_task_names,
)
from .builder import (
    build_eval_dataset,
    generate_and_convert,
    load_generated_dataset,
)

# Auto-register
register_benchmark(CUE_BENCH)

__all__ = [
    "CUE_BENCH",
    "FIXTURE_SAMPLES",
    "build_eval_dataset",
    "generate_and_convert",
    "load_generated_dataset",
    "datagen_sample_to_brewing",
    "get_datagen_for_subset",
    "get_subset_for_datagen",
    "get_datagen_task_names",
]
