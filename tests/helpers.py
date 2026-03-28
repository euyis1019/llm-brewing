"""Test helpers — synthetic caches and mock results."""

from __future__ import annotations

import numpy as np

from brewing.schema import HiddenStateCache


def make_synthetic_cache(
    n_samples: int,
    n_layers: int,
    hidden_dim: int,
    sample_ids: list[str] | None = None,
    model_id: str = "synthetic",
    answers: list[str] | None = None,
    seed: int = 42,
) -> HiddenStateCache:
    """Create a synthetic cache for testing.

    Generates random hidden states with structure that makes
    probing meaningful: later layers have stronger answer signal.
    """
    rng = np.random.RandomState(seed)

    if sample_ids is None:
        sample_ids = [f"sample_{i:04d}" for i in range(n_samples)]
    if answers is None:
        answers = [str(rng.randint(0, 10)) for _ in range(n_samples)]

    hidden_states = rng.randn(n_samples, n_layers, hidden_dim).astype(np.float32)

    # Add answer signal that grows with layer depth
    for i, ans in enumerate(answers):
        ans_int = int(ans) if ans.isdigit() else 0
        signal = np.zeros(hidden_dim)
        signal[ans_int * (hidden_dim // 10):(ans_int + 1) * (hidden_dim // 10)] = 1.0
        for layer_idx in range(n_layers):
            strength = (layer_idx / n_layers) * 2.0  # grows from 0 to 2
            hidden_states[i, layer_idx] += signal * strength

    return HiddenStateCache(
        model_id=model_id,
        sample_ids=sample_ids,
        hidden_states=hidden_states,
        token_position="last",
        model_predictions=answers,  # synthetic: predictions = answers (all correct)
        metadata={
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "dtype": "float32",
            "synthetic": True,
        },
    )
