"""Tests for explicit linear probing fit/eval separation."""

import pytest

from brewing.methods.linear_probing import LinearProbing
from brewing.resources import ResourceKey, ResourceManager
from brewing.schema import MethodConfig, Sample
from tests.helpers import make_synthetic_cache


def _samples() -> list[Sample]:
    return [
        Sample(
            id=f"s_{i}",
            benchmark="CUE-Bench",
            subset="computing",
            prompt=f"p{i}",
            answer=str(i % 3),
        )
        for i in range(6)
    ]


def _artifact_key(**overrides) -> ResourceKey:
    defaults = dict(
        benchmark="cuebench", split="train", task="computing",
        seed=42, model_id="synthetic", method="linear_probing",
    )
    defaults.update(overrides)
    return ResourceKey(**defaults)


def _config(*, fit_policy: str = "eval_only") -> MethodConfig:
    return MethodConfig(
        method="linear_probing",
        benchmark="CUE-Bench",
        config={
            "eval_dataset_id": "cuebench-computing-eval-seed42",
            "answer_space": [str(d) for d in range(10)],
            "fit_policy": fit_policy,
            "resource_key_benchmark": "cuebench",
            "resource_key_task": "computing",
            "resource_key_seed": 42,
        },
    )


def test_linear_probing_run_requires_existing_artifact(tmp_path):
    samples = _samples()
    cache = make_synthetic_cache(
        n_samples=len(samples),
        n_layers=4,
        hidden_dim=16,
        sample_ids=[s.id for s in samples],
        answers=[s.answer for s in samples],
    )
    rm = ResourceManager(tmp_path / "output")

    with pytest.raises(FileNotFoundError, match="Train it first"):
        LinearProbing().run(
            config=_config(),
            eval_samples=samples,
            eval_cache=cache,
            resources=rm,
        )


def test_linear_probing_run_rejects_non_eval_only_policy(tmp_path):
    samples = _samples()
    cache = make_synthetic_cache(
        n_samples=len(samples),
        n_layers=4,
        hidden_dim=16,
        sample_ids=[s.id for s in samples],
        answers=[s.answer for s in samples],
    )
    rm = ResourceManager(tmp_path / "output")

    with pytest.raises(ValueError, match="evaluation-only"):
        LinearProbing().run(
            config=_config(fit_policy="auto"),
            eval_samples=samples,
            eval_cache=cache,
            resources=rm,
        )


def test_linear_probing_fit_requires_explicit_overwrite(tmp_path):
    samples = _samples()
    cache = make_synthetic_cache(
        n_samples=len(samples),
        n_layers=4,
        hidden_dim=16,
        sample_ids=[s.id for s in samples],
        answers=[s.answer for s in samples],
    )
    rm = ResourceManager(tmp_path / "output")
    probing = LinearProbing()
    key = _artifact_key()

    probing.train(
        resources=rm,
        train_samples=samples,
        train_cache=cache,
        artifact_key=key,
    )

    with pytest.raises(FileExistsError, match="overwrite=True"):
        probing.train(
            resources=rm,
            train_samples=samples,
            train_cache=cache,
            artifact_key=key,
        )

    artifact, _ = probing.train(
        resources=rm,
        train_samples=samples,
        train_cache=cache,
        artifact_key=key,
        overwrite=True,
    )
    assert artifact.train_dataset_id == key.dataset_id


def test_linear_probing_run_checks_answer_space_against_artifact(tmp_path):
    samples = _samples()
    cache = make_synthetic_cache(
        n_samples=len(samples),
        n_layers=4,
        hidden_dim=16,
        sample_ids=[s.id for s in samples],
        answers=[s.answer for s in samples],
    )
    rm = ResourceManager(tmp_path / "output")
    probing = LinearProbing()
    key = _artifact_key()

    probing.train(
        resources=rm,
        train_samples=samples,
        train_cache=cache,
        artifact_key=key,
        answer_space=["0", "1", "2"],
    )

    with pytest.raises(ValueError, match="answer_space"):
        probing.run(
            config=_config(),
            eval_samples=samples,
            eval_cache=cache,
            resources=rm,
        )
