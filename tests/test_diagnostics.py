"""Tests for diagnostics engine and dataset loading."""

import json
import warnings

import numpy as np
import pytest

from brewing.schema import (
    DatasetManifest, DatasetPurpose, Granularity, HiddenStateCache,
    MethodResult, Outcome, RunConfig, Sample, SampleMethodResult, save_samples,
)
from brewing.diagnostics import (
    classify_outcome, compute_csd_tail_confidence, compute_fjc,
    compute_fpcl, diagnose_sample, run_diagnostics, run_diagnostics_from_disk,
)
from brewing.resources import ResourceKey, ResourceManager


def _make_sample(sid: str = "s_0", answer: str = "7") -> Sample:
    return Sample(id=sid, benchmark="B", subset="sub", prompt="p", answer=answer)


def _make_smr(
    sid: str = "s_0",
    layer_values: list[float] | None = None,
    n_layers: int = 8,
) -> SampleMethodResult:
    if layer_values is None:
        layer_values = [0.0] * n_layers
    return SampleMethodResult(
        sample_id=sid,
        layer_values=np.array(layer_values),
    )


class TestFPCL:
    def test_found(self):
        smr = _make_smr(layer_values=[0, 0, 0, 1, 1, 1, 1, 1])
        assert compute_fpcl(smr) == 3

    def test_not_found(self):
        smr = _make_smr(layer_values=[0, 0, 0, 0, 0, 0, 0, 0])
        assert compute_fpcl(smr) is None


class TestFJC:
    def test_found(self):
        probe = _make_smr(layer_values=[0, 0, 1, 1, 1, 1, 1, 1])
        csd = _make_smr(layer_values=[0, 0, 0, 0, 0, 1, 1, 1])
        assert compute_fjc(probe, csd) == 5

    def test_not_found(self):
        probe = _make_smr(layer_values=[0, 0, 1, 1, 1, 0, 0, 0])
        csd = _make_smr(layer_values=[0, 0, 0, 0, 0, 1, 1, 1])
        assert compute_fjc(probe, csd) is None

    def test_simultaneous(self):
        probe = _make_smr(layer_values=[0, 0, 0, 1, 1, 1, 1, 1])
        csd = _make_smr(layer_values=[0, 0, 0, 1, 1, 1, 1, 1])
        assert compute_fjc(probe, csd) == 3


class TestOutcome:
    def test_resolved(self):
        assert classify_outcome(fjc=5, model_output="7", answer="7", csd_tail_confidence=0.9) == Outcome.RESOLVED

    def test_overprocessed(self):
        assert classify_outcome(fjc=5, model_output="3", answer="7", csd_tail_confidence=0.9) == Outcome.OVERPROCESSED

    def test_misresolved(self):
        assert classify_outcome(fjc=None, model_output="3", answer="7", csd_tail_confidence=0.8) == Outcome.MISRESOLVED

    def test_unresolved(self):
        assert classify_outcome(fjc=None, model_output="3", answer="7", csd_tail_confidence=0.3) == Outcome.UNRESOLVED


class TestRunDiagnostics:
    def test_basic(self):
        samples = [_make_sample("s_0", "7"), _make_sample("s_1", "3")]

        probe_results = [
            _make_smr("s_0", [0, 0, 1, 1, 1, 1, 1, 1]),
            _make_smr("s_1", [0, 0, 0, 0, 0, 0, 0, 0]),
        ]
        csd_results = [
            _make_smr("s_0", [0, 0, 0, 0, 1, 1, 1, 1]),
            _make_smr("s_1", [0, 0, 0, 0, 0, 0, 0, 0]),
        ]

        probe_mr = MethodResult(
            method="probing", model_id="m",
            granularity=Granularity.PER_SAMPLE,
            eval_dataset_id="eval",
            sample_results=probe_results,
        )
        csd_mr = MethodResult(
            method="csd", model_id="m",
            granularity=Granularity.PER_SAMPLE,
            eval_dataset_id="eval",
            sample_results=csd_results,
        )

        diag = run_diagnostics(
            samples=samples,
            probe_result=probe_mr,
            csd_result=csd_mr,
            model_predictions={"s_0": "7", "s_1": "5"},
            n_layers=8,
        )

        assert len(diag.sample_diagnostics) == 2

        # s_0: FPCL=2, FJC=4, Resolved (model_output=7=answer)
        d0 = diag.sample_diagnostics[0]
        assert d0.fpcl == 2
        assert d0.fjc == 4
        assert d0.delta_brew == 2
        assert d0.outcome == Outcome.RESOLVED

        # s_1: no FPCL, no FJC -> Unresolved (csd_tail_confidence low)
        d1 = diag.sample_diagnostics[1]
        assert d1.fpcl is None
        assert d1.fjc is None
        assert d1.outcome == Outcome.UNRESOLVED

        # Aggregates
        assert diag.outcome_distribution["resolved"] == 0.5
        assert diag.outcome_distribution["unresolved"] == 0.5


class TestRunDiagnosticsFromDisk:
    """Tests for run_diagnostics_from_disk — the decoupled disk-based entry point."""

    def _persist_test_data(self, tmp_path):
        """Persist samples, cache, and method results to disk via ResourceManager."""
        rm = ResourceManager(tmp_path)
        model_id = "test-model"

        samples = [_make_sample("s_0", "7"), _make_sample("s_1", "3")]

        # Build keys
        ds_key = ResourceKey(benchmark="cuebench", split="eval", task="sub", seed=42)
        cache_key = ResourceKey(benchmark="cuebench", split="eval", task="sub", seed=42, model_id=model_id)
        probe_key = ResourceKey(benchmark="cuebench", split="eval", task="sub", seed=42, model_id=model_id, method="linear_probing")
        csd_key = ResourceKey(benchmark="cuebench", split="eval", task="sub", seed=42, model_id=model_id, method="csd")

        # Save dataset
        manifest = DatasetManifest(
            dataset_id=ds_key.dataset_id,
            purpose=DatasetPurpose.EVAL,
            benchmark="B",
            subset="sub",
            sample_ids=[s.id for s in samples],
        )
        rm.save_dataset(ds_key, manifest, samples)

        # Save cache with model_predictions
        cache = HiddenStateCache(
            model_id=model_id,
            sample_ids=["s_0", "s_1"],
            hidden_states=np.zeros((2, 8, 16), dtype=np.float32),
            model_predictions=["7", "5"],
        )
        rm.save_cache(cache_key, cache)

        # Save probe result
        probe_mr = MethodResult(
            method="linear_probing", model_id=model_id,
            granularity=Granularity.PER_SAMPLE,
            eval_dataset_id=ds_key.dataset_id,
            sample_results=[
                _make_smr("s_0", [0, 0, 1, 1, 1, 1, 1, 1]),
                _make_smr("s_1", [0, 0, 0, 0, 0, 0, 0, 0]),
            ],
        )
        rm.save_result(probe_key, probe_mr)

        # Save CSD result
        csd_mr = MethodResult(
            method="csd", model_id=model_id,
            granularity=Granularity.PER_SAMPLE,
            eval_dataset_id=ds_key.dataset_id,
            sample_results=[
                _make_smr("s_0", [0, 0, 0, 0, 1, 1, 1, 1]),
                _make_smr("s_1", [0, 0, 0, 0, 0, 0, 0, 0]),
            ],
        )
        rm.save_result(csd_key, csd_mr)

        return rm, model_id, ds_key, cache_key, probe_key, csd_key

    def test_from_disk_with_key(self, tmp_path):
        """Load everything via ResourceKey resolution."""
        rm, model_id, ds_key, cache_key, probe_key, csd_key = self._persist_test_data(tmp_path)

        diag = run_diagnostics_from_disk(
            results_dir=tmp_path,
            key=cache_key,  # has model_id, split, task, seed, benchmark
        )

        assert len(diag.sample_diagnostics) == 2
        d0 = diag.sample_diagnostics[0]
        assert d0.fpcl == 2
        assert d0.fjc == 4
        assert d0.outcome == Outcome.RESOLVED

        d1 = diag.sample_diagnostics[1]
        assert d1.outcome == Outcome.UNRESOLVED

        # Verify it was persisted
        diag_path = rm.diagnostic_path(cache_key)
        assert diag_path.exists()

    def test_from_disk_with_explicit_paths(self, tmp_path):
        """Load everything via explicit file paths."""
        rm, model_id, ds_key, cache_key, probe_key, csd_key = self._persist_test_data(tmp_path)

        output_path = tmp_path / "custom_diag.json"
        diag = run_diagnostics_from_disk(
            results_dir=tmp_path,
            probe_result_path=rm.result_path(probe_key),
            csd_result_path=rm.result_path(csd_key),
            cache_path=rm.cache_path(cache_key),
            samples_path=rm.samples_path(ds_key),
            output_path=output_path,
        )

        assert len(diag.sample_diagnostics) == 2
        assert output_path.exists()

    def test_missing_probe_result_raises(self, tmp_path):
        """Error when probe result doesn't exist."""
        missing_key = ResourceKey(
            benchmark="cuebench", split="eval", task="nonexistent",
            seed=42, model_id="nonexistent",
        )
        with pytest.raises(FileNotFoundError):
            run_diagnostics_from_disk(
                results_dir=tmp_path,
                key=missing_key,
            )

    def test_missing_ids_raises(self, tmp_path):
        """Error when neither explicit paths nor key are provided."""
        with pytest.raises(ValueError, match="model_id and eval_dataset_id"):
            run_diagnostics_from_disk(results_dir=tmp_path)

    def test_missing_cache_raises_by_default(self, tmp_path):
        """Without cache, diagnostics should raise FileNotFoundError."""
        rm, model_id, ds_key, cache_key, probe_key, csd_key = self._persist_test_data(tmp_path)

        # Delete the cache file so it can't be found
        cache_file = rm.cache_path(cache_key)
        cache_file.unlink()
        meta_file = cache_file.with_suffix(".meta.json")
        if meta_file.exists():
            meta_file.unlink()

        with pytest.raises(FileNotFoundError, match="Eval cache not found"):
            run_diagnostics_from_disk(
                results_dir=tmp_path,
                key=cache_key,
            )

    def test_missing_cache_allowed_with_flag(self, tmp_path):
        """With allow_no_cache=True, diagnostics should run (with bias warning)."""
        rm, model_id, ds_key, cache_key, probe_key, csd_key = self._persist_test_data(tmp_path)

        # Delete the cache
        cache_file = rm.cache_path(cache_key)
        cache_file.unlink()
        meta_file = cache_file.with_suffix(".meta.json")
        if meta_file.exists():
            meta_file.unlink()

        diag = run_diagnostics_from_disk(
            results_dir=tmp_path,
            key=cache_key,
            allow_no_cache=True,
        )
        # Should complete but all FJC samples become Overprocessed (empty model_output)
        assert len(diag.sample_diagnostics) == 2


class TestLoadGeneratedDatasetSplit:
    """Tests for load_generated_dataset split parameter."""

    def _write_samples(self, path, samples):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(samples))

    def _sample_data(self, answer="5"):
        return [{
            "id": "test_001",
            "prompt": "x = 5\n# What is x?",
            "code": "x = 5",
            "answer": answer,
            "metadata": {},
        }]

    def test_split_eval_reads_only_eval(self, tmp_path):
        from brewing.benchmarks.cue_bench import load_generated_dataset

        self._write_samples(tmp_path / "eval" / "value_tracking.json", self._sample_data("5"))
        self._write_samples(tmp_path / "train" / "value_tracking.json", self._sample_data("9"))

        samples = load_generated_dataset(tmp_path, "value_tracking", split="eval")
        assert len(samples) == 1
        assert samples[0].answer == "5"

    def test_split_train_reads_only_train(self, tmp_path):
        from brewing.benchmarks.cue_bench import load_generated_dataset

        self._write_samples(tmp_path / "eval" / "value_tracking.json", self._sample_data("5"))
        self._write_samples(tmp_path / "train" / "value_tracking.json", self._sample_data("9"))

        samples = load_generated_dataset(tmp_path, "value_tracking", split="train")
        assert len(samples) == 1
        assert samples[0].answer == "9"

    def test_split_raises_if_not_found(self, tmp_path):
        from brewing.benchmarks.cue_bench import load_generated_dataset

        with pytest.raises(FileNotFoundError, match="split 'eval'"):
            load_generated_dataset(tmp_path, "value_tracking", split="eval")

    def test_no_split_uses_legacy_fallback(self, tmp_path):
        from brewing.benchmarks.cue_bench import load_generated_dataset

        # Only train/ exists — legacy fallback should find it
        self._write_samples(tmp_path / "train" / "value_tracking.json", self._sample_data("9"))

        samples = load_generated_dataset(tmp_path, "value_tracking", split=None)
        assert len(samples) == 1
        assert samples[0].answer == "9"


class TestRunConfigValidation:
    """Tests for RunConfig validation warnings."""

    def test_train_probing_no_data_dir_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RunConfig(
                model_id="test",
                mode="train_probing",
                data_dir=None,
            )
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert any("train_probing" in str(x.message) for x in user_warnings)

    def test_diagnostics_with_methods_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RunConfig(
                model_id="test",
                mode="diagnostics",
                methods=["linear_probing", "csd"],
            )
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert any("diagnostics" in str(x.message) for x in user_warnings)

    def test_eval_mode_no_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RunConfig(model_id="test", mode="eval")
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0
