"""End-to-end smoke test: explicit probe fit -> eval -> diagnostics."""

import numpy as np


def test_e2e_fixture_smoke(tmp_path):
    """Explicitly fit probes, then evaluate them on fixture data."""
    # Import and register
    import brewing.benchmarks  # noqa: F401
    import brewing.methods.linear_probing  # noqa: F401
    import brewing.methods.csd  # noqa: F401
    from brewing.benchmarks.cue_bench import FIXTURE_SAMPLES
    from tests.helpers import make_synthetic_cache
    from brewing.schema import Granularity, MethodConfig
    from brewing.methods.linear_probing import LinearProbing
    from brewing.resources import ResourceKey, ResourceManager

    # Use ALL fixture samples to get multiple answer classes
    samples = FIXTURE_SAMPLES
    assert len(samples) >= 3

    cache = make_synthetic_cache(
        n_samples=len(samples),
        n_layers=8,
        hidden_dim=32,
        sample_ids=[s.id for s in samples],
        answers=[s.answer for s in samples],
    )

    rm = ResourceManager(tmp_path / "output")
    probing = LinearProbing()
    artifact_key = ResourceKey(
        benchmark="cuebench", split="train", task="fixture",
        seed=42, model_id="synthetic", method="linear_probing",
    )
    artifact, _ = probing.train(
        resources=rm,
        train_samples=samples,
        train_cache=cache,
        artifact_key=artifact_key,
    )
    assert artifact.train_dataset_id == artifact_key.dataset_id
    result = probing.run(
        config=MethodConfig(
            method="linear_probing",
            benchmark="CUE-Bench",
            config={
                "eval_dataset_id": "cuebench-fixture-eval-seed42",
                "answer_space": [str(d) for d in range(10)],
                "fit_policy": "eval_only",
                "resource_key_benchmark": "cuebench",
                "resource_key_task": "fixture",
                "resource_key_seed": 42,
            },
        ),
        eval_samples=samples,
        eval_cache=cache,
        resources=rm,
    )

    assert result.method == "linear_probing"
    assert result.granularity == Granularity.PER_SAMPLE
    assert len(result.sample_results) == len(samples)
    assert result.fit_status.value == "loaded"


def test_e2e_probing_and_diagnostics(tmp_path):
    """Test probing + synthetic CSD -> diagnostics."""
    import brewing.benchmarks  # noqa: F401
    import brewing.methods.linear_probing  # noqa: F401
    from brewing.benchmarks.cue_bench import FIXTURE_SAMPLES
    from tests.helpers import make_synthetic_cache
    from brewing.schema import (
        Granularity, MethodConfig, MethodResult, SampleMethodResult,
    )
    from brewing.diagnostics import run_diagnostics
    from brewing.resources import ResourceKey, ResourceManager

    # Use ALL fixture samples to get multiple answer classes for probing
    samples = FIXTURE_SAMPLES
    assert len(samples) >= 3

    # Create synthetic cache
    cache = make_synthetic_cache(
        n_samples=len(samples),
        n_layers=8,
        hidden_dim=32,
        sample_ids=[s.id for s in samples],
        answers=[s.answer for s in samples],
    )

    # Run probing
    from brewing.methods.linear_probing import LinearProbing
    rm = ResourceManager(tmp_path / "output")

    probing = LinearProbing()
    artifact_key = ResourceKey(
        benchmark="cuebench", split="train", task="test",
        seed=42, model_id="synthetic", method="linear_probing",
    )
    probing.train(
        resources=rm,
        train_samples=samples,
        train_cache=cache,
        artifact_key=artifact_key,
    )
    probe_result = probing.run(
        config=MethodConfig(
            method="linear_probing",
            benchmark="CUE-Bench",
            config={
                "eval_dataset_id": "cuebench-test-eval-seed42",
                "answer_space": [str(d) for d in range(10)],
                "fit_policy": "eval_only",
                "resource_key_benchmark": "cuebench",
                "resource_key_task": "test",
                "resource_key_seed": 42,
            },
        ),
        eval_samples=samples,
        eval_cache=cache,
        resources=rm,
    )

    assert probe_result.method == "linear_probing"
    assert probe_result.granularity == Granularity.PER_SAMPLE
    assert len(probe_result.sample_results) == len(samples)

    # Create synthetic CSD result (simulate)
    csd_results = []
    for s in samples:
        # Simulate CSD getting correct at later layers
        vals = [0.0] * 4 + [1.0] * 4
        csd_results.append(SampleMethodResult(
            sample_id=s.id,
            layer_values=np.array(vals),
        ))

    csd_mr = MethodResult(
        method="csd", model_id="synthetic",
        granularity=Granularity.PER_SAMPLE,
        eval_dataset_id="cuebench-test-eval-seed42",
        sample_results=csd_results,
    )

    # Run diagnostics
    model_preds = {s.id: s.answer for s in samples}  # all correct
    diag = run_diagnostics(
        samples=samples,
        probe_result=probe_result,
        csd_result=csd_mr,
        model_predictions=model_preds,
        n_layers=8,
    )

    assert len(diag.sample_diagnostics) == len(samples)
    # With correct model output and FJC existing, should be Resolved
    for sd in diag.sample_diagnostics:
        if sd.fjc is not None:
            assert sd.outcome.value in ["resolved", "overprocessed"]
