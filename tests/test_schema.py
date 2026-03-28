"""Tests for core data structures and schema round-trips."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from brewing.schema import (
    DatasetManifest, DatasetPurpose, DiagnosticResult, FitArtifact,
    FitPolicy, FitStatus, Granularity, HiddenStateCache,
    MethodResult, Outcome, Sample, SampleDiagnostic, SampleMethodResult,
    check_compatibility, load_samples, save_samples,
    AnswerMeta, AnswerType, BenchmarkSpec, MethodRequirements,
    SingleTokenRequirement,
)


class TestSample:
    def test_round_trip(self):
        s = Sample(
            id="test_001", benchmark="CUE-Bench", subset="computing",
            prompt="a = 1\nb = a + 2", answer="3",
            difficulty={"steps": 1}, metadata={"init": 1},
        )
        d = s.to_dict()
        s2 = Sample.from_dict(d)
        assert s2.id == s.id
        assert s2.answer == s.answer
        assert s2.difficulty == s.difficulty

    def test_save_load(self, tmp_path):
        samples = [
            Sample(id=f"s_{i}", benchmark="B", subset="sub",
                   prompt=f"p{i}", answer=str(i))
            for i in range(5)
        ]
        path = tmp_path / "samples.json"
        save_samples(samples, path)
        loaded = load_samples(path)
        assert len(loaded) == 5
        assert loaded[0].id == "s_0"


class TestDatasetManifest:
    def test_round_trip(self, tmp_path):
        m = DatasetManifest(
            dataset_id="test-ds-v1",
            purpose=DatasetPurpose.EVAL,
            benchmark="CUE-Bench",
            subset="computing",
            sample_ids=["s_0", "s_1"],
            seed=42,
        )
        path = tmp_path / "manifest.json"
        m.save(path)
        m2 = DatasetManifest.load(path)
        assert m2.dataset_id == m.dataset_id
        assert m2.purpose == DatasetPurpose.EVAL
        assert m2.sample_ids == ["s_0", "s_1"]


class TestHiddenStateCache:
    def test_round_trip(self, tmp_path):
        hs = np.random.randn(3, 4, 8).astype(np.float32)
        cache = HiddenStateCache(
            model_id="test-model",
            sample_ids=["a", "b", "c"],
            hidden_states=hs,
            model_predictions=["1", "2", "3"],
            metadata={"n_layers": 4},
        )
        npz_path = tmp_path / "hidden_states.npz"
        meta_path = tmp_path / "meta.json"
        cache.save(npz_path, meta_path)
        loaded = HiddenStateCache.load(npz_path, meta_path)

        assert loaded.model_id == "test-model"
        assert loaded.sample_ids == ["a", "b", "c"]
        assert loaded.model_predictions == ["1", "2", "3"]
        np.testing.assert_allclose(loaded.hidden_states, hs, atol=1e-5)
        assert loaded.n_samples == 3
        assert loaded.n_layers == 4
        assert loaded.hidden_dim == 8

    def test_round_trip_default_meta_path(self, tmp_path):
        """When meta_path is not given, uses meta.json in the same dir."""
        hs = np.random.randn(2, 3, 4).astype(np.float32)
        cache = HiddenStateCache(
            model_id="m", sample_ids=["x", "y"], hidden_states=hs,
        )
        npz_path = tmp_path / "subdir" / "hidden_states.npz"
        cache.save(npz_path)
        loaded = HiddenStateCache.load(npz_path)
        assert loaded.model_id == "m"
        np.testing.assert_allclose(loaded.hidden_states, hs, atol=1e-5)

    def test_get_sample_states(self):
        hs = np.random.randn(3, 4, 8).astype(np.float32)
        cache = HiddenStateCache(
            model_id="m", sample_ids=["a", "b", "c"],
            hidden_states=hs,
        )
        states = cache.get_sample_states("b")
        np.testing.assert_array_equal(states, hs[1])


class TestFitArtifact:
    def test_make_artifact_id(self):
        aid = FitArtifact.make_artifact_id(
            method="linear_probing",
            model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
            train_dataset_id="train-v1",
            fit_config={"C": 1.0},
        )
        assert "linear_probing" in aid
        assert "Qwen" in aid
        assert "__" in aid

    def test_round_trip(self, tmp_path):
        art = FitArtifact(
            artifact_id="test-art",
            method="linear_probing",
            model_id="test-model",
            train_dataset_id="train-v1",
            fit_config={"C": 1.0},
            fit_metrics={"accuracy": 0.95},
        )
        path = tmp_path / "meta.json"
        art.save_metadata(path)
        loaded = FitArtifact.load_metadata(path)
        assert loaded.artifact_id == "test-art"
        assert loaded.fit_metrics["accuracy"] == 0.95


class TestMethodResult:
    def test_per_sample_round_trip(self, tmp_path):
        sr = SampleMethodResult(
            sample_id="s_0",
            layer_values=np.array([0.0, 0.0, 1.0, 1.0]),
            layer_predictions=["3", "5", "7", "7"],
            layer_confidences=np.random.rand(4, 10).astype(np.float32),
        )
        mr = MethodResult(
            method="linear_probing",
            model_id="test-model",
            granularity=Granularity.PER_SAMPLE,
            eval_dataset_id="eval-v1",
            sample_results=[sr],
            train_dataset_id="train-v1",
            fit_status=FitStatus.TRAINED,
        )
        path = tmp_path / "result.json"
        mr.save(path)
        loaded = MethodResult.load(path)

        assert loaded.method == "linear_probing"
        assert loaded.granularity == Granularity.PER_SAMPLE
        assert len(loaded.sample_results) == 1
        assert loaded.sample_results[0].sample_id == "s_0"
        np.testing.assert_array_equal(
            loaded.sample_results[0].layer_values,
            np.array([0.0, 0.0, 1.0, 1.0]),
        )
        assert loaded.fit_status == FitStatus.TRAINED


class TestDiagnosticResult:
    def test_round_trip(self, tmp_path):
        sd = SampleDiagnostic(
            sample_id="s_0", fpcl=5, fjc=10, delta_brew=5,
            outcome=Outcome.RESOLVED, model_output="7",
            csd_tail_confidence=0.9,
        )
        dr = DiagnosticResult(
            model_id="test-model",
            eval_dataset_id="eval-v1",
            benchmark="CUE-Bench",
            subset="computing",
            sample_diagnostics=[sd],
            outcome_distribution={"resolved": 1.0},
        )
        path = tmp_path / "diag.json"
        dr.save(path)
        loaded = DiagnosticResult.load(path)

        assert loaded.sample_diagnostics[0].outcome == Outcome.RESOLVED
        assert loaded.sample_diagnostics[0].fpcl == 5


class TestCompatibility:
    def test_cue_bench_probing(self):
        bench = BenchmarkSpec(
            name="CUE-Bench", domain="code",
            answer_meta=AnswerMeta(
                answer_type=AnswerType.CATEGORICAL,
                answer_space=[str(d) for d in range(10)],
                max_answer_tokens=1,
            ),
        )
        reqs = MethodRequirements(
            needs_answer_space=True,
            single_token_answer=SingleTokenRequirement.NOT_NEEDED,
            trained=True,
        )
        issues = check_compatibility(reqs, bench)
        assert len(issues) == 0  # fully compatible

    def test_incompatible_no_answer_space(self):
        bench = BenchmarkSpec(
            name="HumanEval", domain="code",
            answer_meta=AnswerMeta(
                answer_type=AnswerType.CODE,
            ),
        )
        reqs = MethodRequirements(needs_answer_space=True)
        issues = check_compatibility(reqs, bench)
        assert any(sev == "INCOMPATIBLE" for sev, _ in issues)
