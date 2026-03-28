"""Tests for resource management and lifecycle."""

import numpy as np
import pytest

from brewing.schema import (
    DatasetManifest, DatasetPurpose, FitArtifact, FitPolicy,
    HiddenStateCache, Sample,
)
from brewing.resources import ResourceKey, ResourceManager


@pytest.fixture
def rm(tmp_path):
    return ResourceManager(tmp_path / "output")


def _ds_key(**overrides) -> ResourceKey:
    defaults = dict(benchmark="cuebench", split="eval", task="computing", seed=42)
    defaults.update(overrides)
    return ResourceKey(**defaults)


def _cache_key(**overrides) -> ResourceKey:
    defaults = dict(benchmark="cuebench", split="eval", task="computing", seed=42, model_id="test-model")
    defaults.update(overrides)
    return ResourceKey(**defaults)


def _artifact_key(**overrides) -> ResourceKey:
    defaults = dict(benchmark="cuebench", split="train", task="computing", seed=42,
                    model_id="m", method="linear_probing")
    defaults.update(overrides)
    return ResourceKey(**defaults)


@pytest.fixture
def sample_data():
    samples = [
        Sample(id=f"s_{i}", benchmark="B", subset="sub",
               prompt=f"p{i}", answer=str(i % 10))
        for i in range(5)
    ]
    manifest = DatasetManifest(
        dataset_id="cuebench-computing-eval-seed42",
        purpose=DatasetPurpose.EVAL,
        benchmark="B",
        sample_ids=[s.id for s in samples],
    )
    return manifest, samples


class TestResourceKey:
    def test_dataset_id(self):
        key = ResourceKey(benchmark="cuebench", split="eval", task="computing", seed=42)
        assert key.dataset_id == "cuebench-computing-eval-seed42"

    def test_model_id_safe(self):
        key = ResourceKey(benchmark="cuebench", split="eval", task="computing",
                          model_id="Qwen/Qwen2.5-Coder-7B")
        assert key.model_id_safe == "Qwen__Qwen2.5-Coder-7B"

    def test_model_id_safe_raises_when_none(self):
        key = ResourceKey(benchmark="cuebench", split="eval", task="computing")
        with pytest.raises(ValueError, match="model_id is None"):
            key.model_id_safe


class TestDatasetLifecycle:
    def test_save_and_resolve(self, rm, sample_data):
        manifest, samples = sample_data
        key = _ds_key()
        rm.save_dataset(key, manifest, samples)

        loaded = rm.resolve_dataset(key)
        assert loaded is not None
        loaded_manifest, loaded_samples = loaded
        assert loaded_manifest.dataset_id == "cuebench-computing-eval-seed42"
        assert len(loaded_samples) == 5

    def test_resolve_missing(self, rm):
        assert rm.resolve_dataset(_ds_key(task="nonexistent")) is None

    def test_resolve_or_build(self, rm, sample_data):
        manifest, samples = sample_data
        key = _ds_key()
        call_count = 0

        def build():
            nonlocal call_count
            call_count += 1
            return manifest, samples

        # First call: builds
        m1, s1 = rm.resolve_or_build_dataset(key, build)
        assert call_count == 1
        assert len(s1) == 5

        # Second call: resolves from disk
        m2, s2 = rm.resolve_or_build_dataset(key, build)
        assert call_count == 1  # not called again


class TestCacheLifecycle:
    def test_save_and_resolve(self, rm):
        cache = HiddenStateCache(
            model_id="test-model",
            sample_ids=["a", "b"],
            hidden_states=np.random.randn(2, 4, 8).astype(np.float32),
            model_predictions=["1", "2"],
        )
        key = _cache_key()
        rm.save_cache(key, cache)

        loaded = rm.resolve_cache(key)
        assert loaded is not None
        assert loaded.n_samples == 2
        np.testing.assert_allclose(
            loaded.hidden_states, cache.hidden_states, atol=1e-5
        )

    def test_resolve_or_build(self, rm):
        call_count = 0

        def build():
            nonlocal call_count
            call_count += 1
            return HiddenStateCache(
                model_id="m", sample_ids=["a"],
                hidden_states=np.zeros((1, 2, 4), dtype=np.float32),
            )

        key = _cache_key(model_id="m")
        c1 = rm.resolve_or_build_cache(key, build)
        assert call_count == 1

        c2 = rm.resolve_or_build_cache(key, build)
        assert call_count == 1  # reused


class TestArtifactLifecycle:
    def _make_artifact(self):
        return FitArtifact(
            artifact_id="art-1",
            method="linear_probing",
            model_id="m",
            train_dataset_id="train-1",
            fit_config={"C": 1.0},
            fit_metrics={"accuracy": 0.9},
        )

    def test_fit_policy_auto_no_existing(self, rm):
        """auto: no existing artifact -> trains."""
        art = self._make_artifact()
        model_data = {"weights": [1, 2, 3]}
        key = _artifact_key()

        result_art, result_model, was_trained = rm.resolve_artifact_with_policy(
            key=key,
            fit_policy=FitPolicy.AUTO,
            fit_fn=lambda: (art, model_data),
        )
        assert was_trained is True
        assert result_art.artifact_id == "art-1"

    def test_fit_policy_auto_existing(self, rm):
        """auto: existing artifact -> loads."""
        art = self._make_artifact()
        model_data = {"weights": [1, 2, 3]}
        key = _artifact_key()
        rm.save_artifact(key, art, model_data)

        result_art, result_model, was_trained = rm.resolve_artifact_with_policy(
            key=key,
            fit_policy=FitPolicy.AUTO,
            fit_fn=lambda: (art, {"weights": [99]}),  # should NOT be called
        )
        assert was_trained is False
        assert result_model == {"weights": [1, 2, 3]}

    def test_fit_policy_force(self, rm):
        """force: always trains, even if existing."""
        art = self._make_artifact()
        key = _artifact_key()
        rm.save_artifact(key, art, {"old": True})

        new_art = self._make_artifact()
        result_art, result_model, was_trained = rm.resolve_artifact_with_policy(
            key=key,
            fit_policy=FitPolicy.FORCE,
            fit_fn=lambda: (new_art, {"new": True}),
        )
        assert was_trained is True
        assert result_model == {"new": True}

    def test_fit_policy_eval_only_missing(self, rm):
        """eval_only: missing artifact -> error."""
        key = _artifact_key(task="missing")
        with pytest.raises(FileNotFoundError):
            rm.resolve_artifact_with_policy(
                key=key,
                fit_policy=FitPolicy.EVAL_ONLY,
                fit_fn=lambda: None,
            )

    def test_fit_policy_eval_only_existing(self, rm):
        """eval_only: existing artifact -> loads."""
        art = self._make_artifact()
        key = _artifact_key()
        rm.save_artifact(key, art, {"weights": [1]})

        result_art, result_model, was_trained = rm.resolve_artifact_with_policy(
            key=key,
            fit_policy=FitPolicy.EVAL_ONLY,
            fit_fn=lambda: None,
        )
        assert was_trained is False
