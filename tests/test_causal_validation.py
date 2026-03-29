"""Tests for the causal validation subsystem.

Covers:
  - Schema round-trip (SampleCausalResult, CausalValidationResult)
  - ResourceManager causal result paths, save/load
  - Selector: select_fjc_samples
  - Validator: ActivationPatchingFJC with FakeInterventionBackend
  - Pipeline: CausalValidationPipeline with mocked dependencies
  - CLI: needs_model_online for causal_validation mode
  - RunConfig: causal_validation mode validation
"""

import json
import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import brewing.benchmarks  # noqa: F401
import brewing.methods.linear_probing  # noqa: F401
import brewing.methods.csd  # noqa: F401

from brewing.schema import (
    CausalValidationResult,
    DatasetManifest,
    DatasetPurpose,
    DiagnosticResult,
    Granularity,
    HiddenStateCache,
    MethodResult,
    Outcome,
    RunConfig,
    Sample,
    SampleCausalResult,
    SampleDiagnostic,
    SampleMethodResult,
)
from brewing.resources import ResourceKey, ResourceManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(sid: str = "s_0", answer: str = "7") -> Sample:
    return Sample(id=sid, benchmark="B", subset="sub", prompt="x=7", answer=answer)


def _make_diagnostic_result(
    sample_diagnostics: list[SampleDiagnostic],
    model_id: str = "test-model",
) -> DiagnosticResult:
    return DiagnosticResult(
        model_id=model_id,
        eval_dataset_id="cuebench-sub-eval-seed42",
        benchmark="B",
        subset="sub",
        sample_diagnostics=sample_diagnostics,
    )


def _make_cache(
    sample_ids: list[str],
    n_layers: int = 8,
    hidden_dim: int = 16,
    model_id: str = "test-model",
    predictions: list[str] | None = None,
    seed: int = 42,
) -> HiddenStateCache:
    n = len(sample_ids)
    if predictions is None:
        predictions = ["7"] * n
    rng = np.random.RandomState(seed)
    return HiddenStateCache(
        model_id=model_id,
        sample_ids=sample_ids,
        hidden_states=rng.randn(n, n_layers, hidden_dim).astype(np.float32),
        model_predictions=predictions,
    )


# ---------------------------------------------------------------------------
# 1. Schema round-trip tests
# ---------------------------------------------------------------------------

class TestSampleCausalResultRoundTrip:
    def test_selected_sample(self):
        sr = SampleCausalResult(
            sample_id="s_0",
            selected=True,
            source_layer=5,
            target_layer=5,
            original_output="0",
            intervened_output="7",
            original_correct=None,
            intervened_correct=True,
            effect_label="flipped",
            extras={"note": "test"},
        )
        d = sr.to_dict()
        restored = SampleCausalResult.from_dict(d)
        assert restored.sample_id == "s_0"
        assert restored.selected is True
        assert restored.source_layer == 5
        assert restored.effect_label == "flipped"
        assert restored.extras == {"note": "test"}

    def test_skipped_sample(self):
        sr = SampleCausalResult(
            sample_id="s_1",
            selected=False,
            skip_reason="fjc_is_none",
        )
        d = sr.to_dict()
        restored = SampleCausalResult.from_dict(d)
        assert restored.selected is False
        assert restored.skip_reason == "fjc_is_none"
        assert restored.source_layer is None


class TestCausalValidationResultRoundTrip:
    def test_save_load(self, tmp_path):
        result = CausalValidationResult(
            experiment="activation_patching_fjc",
            model_id="test-model",
            eval_dataset_id="cuebench-sub-eval-seed42",
            benchmark="B",
            subset="sub",
            sample_results=[
                SampleCausalResult(
                    sample_id="s_0", selected=True,
                    source_layer=5, target_layer=5,
                    original_output="0", intervened_output="7",
                    original_correct=None, intervened_correct=True,
                    effect_label="flipped",
                ),
                SampleCausalResult(
                    sample_id="s_1", selected=False,
                    skip_reason="fjc_is_none",
                ),
            ],
            summary={"flip_rate": 1.0, "n_selected": 1, "n_effective": 1, "n_flipped": 1},
        )

        path = tmp_path / "causal_result.json"
        result.save(path)
        assert path.exists()

        loaded = CausalValidationResult.load(path)
        assert loaded.experiment == "activation_patching_fjc"
        assert loaded.model_id == "test-model"
        assert len(loaded.sample_results) == 2
        assert loaded.sample_results[0].effect_label == "flipped"
        assert loaded.sample_results[1].skip_reason == "fjc_is_none"
        assert loaded.summary["flip_rate"] == 1.0


# ---------------------------------------------------------------------------
# 2. ResourceManager causal result tests
# ---------------------------------------------------------------------------

class TestResourceManagerCausal:
    def test_causal_result_path(self, tmp_path):
        rm = ResourceManager(tmp_path / "out")
        key = ResourceKey(
            benchmark="cuebench", split="eval", task="sub",
            seed=42, model_id="test/model",
        )
        path = rm.causal_result_path(key, "activation_patching_fjc")
        expected = (
            tmp_path / "out" / "causal" / "cuebench" / "eval" / "sub" /
            "seed42" / "test__model" / "activation_patching_fjc.json"
        )
        assert path == expected

    def test_save_and_load(self, tmp_path):
        rm = ResourceManager(tmp_path / "out")
        key = ResourceKey(
            benchmark="cuebench", split="eval", task="sub",
            seed=42, model_id="test-model",
        )
        result = CausalValidationResult(
            experiment="activation_patching_fjc",
            model_id="test-model",
            eval_dataset_id="ds",
            benchmark="B",
            subset="sub",
            summary={"flip_rate": 0.5},
        )

        rm.save_causal_result(key, "activation_patching_fjc", result)
        loaded = rm.resolve_causal_result(key, "activation_patching_fjc")
        assert loaded is not None
        assert loaded.summary["flip_rate"] == 0.5

    def test_resolve_missing_returns_none(self, tmp_path):
        rm = ResourceManager(tmp_path / "out")
        key = ResourceKey(
            benchmark="cuebench", split="eval", task="sub",
            seed=42, model_id="test-model",
        )
        assert rm.resolve_causal_result(key, "nonexistent") is None


# ---------------------------------------------------------------------------
# 3. Selector tests
# ---------------------------------------------------------------------------

class TestSelectFJCSamples:
    def test_selects_fjc_samples(self):
        from brewing.causal.selectors import select_fjc_samples

        samples = [_make_sample("s_0", "7"), _make_sample("s_1", "3")]
        diagnostics = _make_diagnostic_result([
            SampleDiagnostic(sample_id="s_0", fjc=5, outcome=Outcome.RESOLVED),
            SampleDiagnostic(sample_id="s_1", fjc=None, outcome=Outcome.UNRESOLVED),
        ])
        cache = _make_cache(["s_0", "s_1"])

        selected, skipped = select_fjc_samples(samples, diagnostics, cache)

        assert len(selected) == 1
        assert selected[0].sample.id == "s_0"
        assert selected[0].source_layer == 5

        assert len(skipped) == 1
        assert skipped[0].sample_id == "s_1"
        assert skipped[0].skip_reason == "fjc_is_none"

    def test_skip_no_diagnostic(self):
        from brewing.causal.selectors import select_fjc_samples

        samples = [_make_sample("s_0", "7"), _make_sample("s_missing", "3")]
        diagnostics = _make_diagnostic_result([
            SampleDiagnostic(sample_id="s_0", fjc=5, outcome=Outcome.RESOLVED),
        ])
        cache = _make_cache(["s_0", "s_missing"])

        selected, skipped = select_fjc_samples(samples, diagnostics, cache)
        assert len(selected) == 1
        assert len(skipped) == 1
        assert skipped[0].skip_reason == "no_diagnostic"

    def test_skip_not_in_cache(self):
        from brewing.causal.selectors import select_fjc_samples

        samples = [_make_sample("s_0", "7"), _make_sample("s_nocache", "3")]
        diagnostics = _make_diagnostic_result([
            SampleDiagnostic(sample_id="s_0", fjc=5, outcome=Outcome.RESOLVED),
            SampleDiagnostic(sample_id="s_nocache", fjc=3, outcome=Outcome.RESOLVED),
        ])
        cache = _make_cache(["s_0"])  # s_nocache not in cache

        selected, skipped = select_fjc_samples(samples, diagnostics, cache)
        assert len(selected) == 1
        assert len(skipped) == 1
        assert skipped[0].skip_reason == "not_in_cache"


# ---------------------------------------------------------------------------
# 4. Validator unit test (ActivationPatchingFJC with fake backend)
# ---------------------------------------------------------------------------

class TestActivationPatchingFJC:
    def test_reads_real_hidden_state_and_injects_into_target(self):
        """Core test: validator reads real FJC hidden states from cache
        and injects them into target prompt; does NOT zero-ablate."""
        from brewing.causal.activation_patching import ActivationPatchingFJC
        from brewing.causal.backend import FakeInterventionBackend, InterventionRequest

        samples = [_make_sample("s_0", "7"), _make_sample("s_1", "3")]
        diagnostics = _make_diagnostic_result([
            SampleDiagnostic(sample_id="s_0", fjc=5, outcome=Outcome.RESOLVED, model_output="7"),
            SampleDiagnostic(sample_id="s_1", fjc=None, outcome=Outcome.UNRESOLVED),
        ])
        cache = _make_cache(["s_0", "s_1"])

        # Track what the backend receives
        received_requests: list[InterventionRequest] = []
        class TrackingBackend(FakeInterventionBackend):
            def run_interventions(self, requests):
                received_requests.extend(requests)
                return super().run_interventions(requests)

        # predictions: s_0's FJC hidden injects into target → produces "7"
        backend = TrackingBackend(predictions={"s_0": "7"}, baseline_output="0")

        validator = ActivationPatchingFJC()
        result = validator.run(samples, cache, diagnostics, backend)

        # Verify the request uses real hidden state, not zeros
        assert len(received_requests) == 1
        req = received_requests[0]
        assert req.sample_id == "s_0"
        # source_hidden should be the real FJC hidden state from cache
        expected_hidden = cache.hidden_states[0, 5]  # sample 0, layer 5
        np.testing.assert_array_equal(req.source_hidden, expected_hidden)
        # Intervention should target a neutral target prompt, not the original
        assert req.target_prompt != samples[0].prompt
        assert req.source_prompt == samples[0].prompt

    def test_flip_when_target_produces_correct_answer(self):
        """flipped = target prompt output matches correct answer after injection."""
        from brewing.causal.activation_patching import ActivationPatchingFJC
        from brewing.causal.backend import FakeInterventionBackend

        samples = [
            _make_sample("s_0", "7"),
            _make_sample("s_1", "3"),
            _make_sample("s_2", "5"),
        ]
        diagnostics = _make_diagnostic_result([
            SampleDiagnostic(sample_id="s_0", fjc=5, outcome=Outcome.RESOLVED, model_output="7"),
            SampleDiagnostic(sample_id="s_1", fjc=None, outcome=Outcome.UNRESOLVED),
            SampleDiagnostic(sample_id="s_2", fjc=3, outcome=Outcome.RESOLVED, model_output="5"),
        ])
        cache = _make_cache(["s_0", "s_1", "s_2"], predictions=["7", "3", "5"])

        # Fake: injecting s_0's hidden → target outputs "7" (correct)
        #        injecting s_2's hidden → target outputs "5" (correct)
        backend = FakeInterventionBackend(
            predictions={"s_0": "7", "s_2": "5"},
            baseline_output="0",
        )

        validator = ActivationPatchingFJC()
        result = validator.run(samples, cache, diagnostics, backend)

        assert result.experiment == "activation_patching_fjc"
        assert result.summary["n_selected"] == 2  # s_0, s_2
        assert result.summary["n_effective"] == 2
        assert result.summary["n_flipped"] == 2  # both produced correct answer
        assert result.summary["flip_rate"] == 1.0

        result_map = {sr.sample_id: sr for sr in result.sample_results}
        assert result_map["s_0"].selected is True
        assert result_map["s_0"].effect_label == "flipped"
        assert result_map["s_0"].intervened_correct is True
        assert result_map["s_1"].selected is False
        assert result_map["s_1"].skip_reason == "fjc_is_none"

    def test_no_effect_when_target_does_not_produce_answer(self):
        from brewing.causal.activation_patching import ActivationPatchingFJC
        from brewing.causal.backend import FakeInterventionBackend

        samples = [_make_sample("s_0", "7")]
        diagnostics = _make_diagnostic_result([
            SampleDiagnostic(sample_id="s_0", fjc=5, outcome=Outcome.RESOLVED, model_output="7"),
        ])
        cache = _make_cache(["s_0"], predictions=["7"])

        # Fake: injection does NOT produce correct answer
        backend = FakeInterventionBackend(predictions={"s_0": "3"}, baseline_output="0")

        validator = ActivationPatchingFJC()
        result = validator.run(samples, cache, diagnostics, backend)

        assert result.summary["n_flipped"] == 0
        assert result.summary["flip_rate"] == 0.0
        r0 = [sr for sr in result.sample_results if sr.sample_id == "s_0"][0]
        assert r0.effect_label == "no_effect"
        assert r0.intervened_correct is False

    def test_all_skipped(self):
        from brewing.causal.activation_patching import ActivationPatchingFJC
        from brewing.causal.backend import FakeInterventionBackend

        samples = [_make_sample("s_0", "7")]
        diagnostics = _make_diagnostic_result([
            SampleDiagnostic(sample_id="s_0", fjc=None, outcome=Outcome.UNRESOLVED),
        ])
        cache = _make_cache(["s_0"])

        backend = FakeInterventionBackend()
        validator = ActivationPatchingFJC()
        result = validator.run(samples, cache, diagnostics, backend)

        assert result.summary["n_selected"] == 0
        assert result.summary["flip_rate"] == 0.0

    def test_config_overrides_target_prompt(self):
        """Per-experiment config can override target_prompt."""
        from brewing.causal.activation_patching import ActivationPatchingFJC
        from brewing.causal.backend import FakeInterventionBackend, InterventionRequest

        samples = [_make_sample("s_0", "7")]
        diagnostics = _make_diagnostic_result([
            SampleDiagnostic(sample_id="s_0", fjc=5, outcome=Outcome.RESOLVED),
        ])
        cache = _make_cache(["s_0"])

        received: list[InterventionRequest] = []
        class TrackingBackend(FakeInterventionBackend):
            def run_interventions(self, requests):
                received.extend(requests)
                return super().run_interventions(requests)

        backend = TrackingBackend(predictions={"s_0": "7"})
        validator = ActivationPatchingFJC()

        custom_prompt = "The answer is: "
        result = validator.run(
            samples, cache, diagnostics, backend,
            config={"intervention": {"target_prompt": custom_prompt}},
        )

        assert len(received) == 1
        assert received[0].target_prompt == custom_prompt


# ---------------------------------------------------------------------------
# 5. Pipeline integration test
# ---------------------------------------------------------------------------

class TestCausalValidationPipeline:
    def _setup_disk_artifacts(self, tmp_path, model_id="test-model"):
        """Persist all required artifacts to disk."""
        rm = ResourceManager(tmp_path)
        samples = [_make_sample("s_0", "7"), _make_sample("s_1", "3")]

        ds_key = ResourceKey(benchmark="cuebench", split="eval", task="sub", seed=42)
        cache_key = ResourceKey(benchmark="cuebench", split="eval", task="sub", seed=42, model_id=model_id)

        # Save dataset
        manifest = DatasetManifest(
            dataset_id=ds_key.dataset_id,
            purpose=DatasetPurpose.EVAL,
            benchmark="B", subset="sub",
            sample_ids=[s.id for s in samples],
        )
        rm.save_dataset(ds_key, manifest, samples)

        # Save cache
        cache = _make_cache(["s_0", "s_1"], model_id=model_id, predictions=["7", "3"])
        rm.save_cache(cache_key, cache)

        # Save probing result
        probe_key = ResourceKey(benchmark="cuebench", split="eval", task="sub", seed=42, model_id=model_id, method="linear_probing")
        probe_mr = MethodResult(
            method="linear_probing", model_id=model_id,
            granularity=Granularity.PER_SAMPLE,
            eval_dataset_id=ds_key.dataset_id,
            sample_results=[
                SampleMethodResult(sample_id="s_0", layer_values=np.array([0,0,1,1,1,1,1,1], dtype=float)),
                SampleMethodResult(sample_id="s_1", layer_values=np.array([0,0,0,0,0,0,0,0], dtype=float)),
            ],
        )
        rm.save_result(probe_key, probe_mr)

        # Save CSD result
        csd_key = ResourceKey(benchmark="cuebench", split="eval", task="sub", seed=42, model_id=model_id, method="csd")
        csd_mr = MethodResult(
            method="csd", model_id=model_id,
            granularity=Granularity.PER_SAMPLE,
            eval_dataset_id=ds_key.dataset_id,
            sample_results=[
                SampleMethodResult(sample_id="s_0", layer_values=np.array([0,0,0,0,1,1,1,1], dtype=float)),
                SampleMethodResult(sample_id="s_1", layer_values=np.array([0,0,0,0,0,0,0,0], dtype=float)),
            ],
        )
        rm.save_result(csd_key, csd_mr)

        # Save diagnostics
        diag = DiagnosticResult(
            model_id=model_id,
            eval_dataset_id=ds_key.dataset_id,
            benchmark="B", subset="sub",
            sample_diagnostics=[
                SampleDiagnostic(sample_id="s_0", fpcl=2, fjc=4, outcome=Outcome.RESOLVED, model_output="7"),
                SampleDiagnostic(sample_id="s_1", fpcl=None, fjc=None, outcome=Outcome.UNRESOLVED, model_output="3"),
            ],
        )
        rm.save_diagnostic(cache_key, diag)

        return rm, samples

    def test_pipeline_runs_with_fake_backend(self, tmp_path):
        model_id = "test-model"
        rm, samples = self._setup_disk_artifacts(tmp_path, model_id)

        rc = RunConfig(
            mode="causal_validation",
            output_root=str(tmp_path),
            model_id=model_id,
            subsets=["sub"],
            causal_validation={"experiments": ["activation_patching_fjc"]},
        )

        from brewing.pipelines import create_pipeline
        from brewing.registry import get_benchmark
        from brewing.causal.backend import FakeInterventionBackend

        bm = get_benchmark("CUE-Bench")
        pipeline = create_pipeline(rc, rm, bm)

        # Monkey-patch _build_backend to use fake
        fake_backend = FakeInterventionBackend(
            predictions={"s_0": "7", "s_1": "3"},
            baseline_output="0",
        )
        pipeline._build_backend = lambda model, tokenizer: fake_backend

        result = pipeline.run(model=None, tokenizer=None)

        assert "subsets" in result
        assert "sub" in result["subsets"]
        sub_result = result["subsets"]["sub"]
        assert "activation_patching_fjc" in sub_result
        summary = sub_result["activation_patching_fjc"]
        assert summary["n_selected"] == 1  # only s_0 has fjc
        assert "flip_rate" in summary

        # Verify persisted result
        eval_key = ResourceKey(benchmark="cuebench", split="eval", task="sub", seed=42, model_id=model_id)
        loaded = rm.resolve_causal_result(eval_key, "activation_patching_fjc")
        assert loaded is not None
        assert loaded.experiment == "activation_patching_fjc"

    def test_pipeline_missing_dataset_raises(self, tmp_path):
        """Pipeline should fail if eval dataset is missing — NEVER rebuild."""
        model_id = "test-model"
        rm = ResourceManager(tmp_path)

        rc = RunConfig(
            mode="causal_validation",
            output_root=str(tmp_path),
            model_id=model_id,
            subsets=["sub"],
            causal_validation={"experiments": ["activation_patching_fjc"]},
        )

        from brewing.pipelines import create_pipeline
        from brewing.registry import get_benchmark
        from brewing.causal.backend import FakeInterventionBackend

        bm = get_benchmark("CUE-Bench")
        pipeline = create_pipeline(rc, rm, bm)
        pipeline._build_backend = lambda m, t: FakeInterventionBackend()

        result = pipeline.run(model=None, tokenizer=None)
        # Error should be captured in subset result, NOT silently regenerated
        assert "error" in result["subsets"]["sub"]
        assert "Eval dataset not found" in result["subsets"]["sub"]["error"]

    def test_pipeline_missing_diagnostics_raises(self, tmp_path):
        """Pipeline should fail if diagnostics are missing."""
        model_id = "test-model"
        rm, samples = self._setup_disk_artifacts(tmp_path, model_id)

        # Delete diagnostics file
        diag_key = ResourceKey(benchmark="cuebench", split="eval", task="sub", seed=42, model_id=model_id)
        diag_path = rm.diagnostic_path(diag_key)
        diag_path.unlink()

        rc = RunConfig(
            mode="causal_validation",
            output_root=str(tmp_path),
            model_id=model_id,
            subsets=["sub"],
            causal_validation={"experiments": ["activation_patching_fjc"]},
        )

        from brewing.pipelines import create_pipeline
        from brewing.registry import get_benchmark
        from brewing.causal.backend import FakeInterventionBackend

        bm = get_benchmark("CUE-Bench")
        pipeline = create_pipeline(rc, rm, bm)
        pipeline._build_backend = lambda m, t: FakeInterventionBackend()

        result = pipeline.run(model=None, tokenizer=None)
        assert "error" in result["subsets"]["sub"]

    def test_pipeline_passes_experiment_config(self, tmp_path):
        """Per-experiment config from YAML should flow to validator."""
        model_id = "test-model"
        rm, samples = self._setup_disk_artifacts(tmp_path, model_id)

        rc = RunConfig(
            mode="causal_validation",
            output_root=str(tmp_path),
            model_id=model_id,
            subsets=["sub"],
            causal_validation={
                "experiments": ["activation_patching_fjc"],
                "activation_patching_fjc": {
                    "intervention": {
                        "target_prompt": "custom prompt: ",
                    },
                },
            },
        )

        from brewing.pipelines import create_pipeline
        from brewing.registry import get_benchmark
        from brewing.causal.backend import FakeInterventionBackend, InterventionRequest

        bm = get_benchmark("CUE-Bench")
        pipeline = create_pipeline(rc, rm, bm)

        received: list[InterventionRequest] = []
        class TrackingBackend(FakeInterventionBackend):
            def run_interventions(self, requests):
                received.extend(requests)
                return super().run_interventions(requests)

        pipeline._build_backend = lambda m, t: TrackingBackend(
            predictions={"s_0": "7"}, baseline_output="0",
        )

        pipeline.run(model=None, tokenizer=None)

        assert len(received) == 1
        assert received[0].target_prompt == "custom prompt: "


# ---------------------------------------------------------------------------
# 6. CLI / config smoke tests
# ---------------------------------------------------------------------------

class TestCausalValidationConfig:
    def test_mode_valid(self):
        rc = RunConfig(mode="causal_validation")
        assert rc.mode == "causal_validation"

    def test_mode_in_valid_modes(self):
        from brewing.schema.results import VALID_MODES
        assert "causal_validation" in VALID_MODES

    def test_causal_validation_field_default(self):
        rc = RunConfig()
        assert rc.causal_validation == {}

    def test_causal_validation_field_set(self):
        rc = RunConfig(
            mode="causal_validation",
            causal_validation={"experiments": ["activation_patching_fjc"]},
        )
        assert rc.causal_validation["experiments"] == ["activation_patching_fjc"]


class TestCLINeedsModelCausalValidation:
    def test_needs_model_causal_validation(self):
        from brewing.cli import needs_model_online
        rc = RunConfig(mode="causal_validation")
        assert needs_model_online(rc) is True


# ---------------------------------------------------------------------------
# 7. Pipeline registry test
# ---------------------------------------------------------------------------

class TestPipelineRegistryCausal:
    def test_registry_has_causal_validation(self):
        from brewing.pipelines import PIPELINE_REGISTRY, CausalValidationPipeline
        assert "causal_validation" in PIPELINE_REGISTRY
        assert PIPELINE_REGISTRY["causal_validation"] is CausalValidationPipeline

    def test_create_pipeline_causal(self, tmp_path):
        from brewing.pipelines import create_pipeline, CausalValidationPipeline
        from brewing.registry import get_benchmark

        rm = ResourceManager(tmp_path / "out")
        bm = get_benchmark("CUE-Bench")
        rc = RunConfig(mode="causal_validation", output_root=str(tmp_path / "out"))
        pipeline = create_pipeline(rc, rm, bm)
        assert isinstance(pipeline, CausalValidationPipeline)


# ---------------------------------------------------------------------------
# 8. Validator registry test
# ---------------------------------------------------------------------------

class TestValidatorRegistry:
    def test_get_activation_patching(self):
        from brewing.causal import get_validator, ActivationPatchingFJC
        v = get_validator("activation_patching_fjc")
        assert isinstance(v, ActivationPatchingFJC)

    def test_get_unknown_raises(self):
        from brewing.causal import get_validator
        with pytest.raises(ValueError, match="Unknown causal validator"):
            get_validator("nonexistent_experiment")
