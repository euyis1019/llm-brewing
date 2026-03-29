"""Activation Patching at FJC — causal validation experiment.

For each sample with a valid FJC, extracts the real hidden state at
the FJC layer from the eval cache and injects it into a neutral target
prompt via patchscope-style intervention.  If the target prompt's
output matches the sample's correct answer, the FJC layer is confirmed
as the causally privileged layer for that sample.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from brewing.schema import (
    CausalValidationResult,
    DiagnosticResult,
    HiddenStateCache,
    Sample,
    SampleCausalResult,
)
from .backend import InterventionBackend, InterventionRequest
from .base import CausalValidator, register_validator
from .selectors import select_fjc_samples

logger = logging.getLogger(__name__)

DEFAULT_TARGET_PROMPT = '# The value of x is "'
DEFAULT_TARGET_POSITION = -1  # last token


@register_validator
class ActivationPatchingFJC(CausalValidator):
    """Activation patching at FJC layer.

    For each sample with a valid FJC:
    1. Read the real hidden state at the FJC layer from the eval cache
    2. Inject it into a neutral target prompt at the designated position
    3. Decode the target prompt's next-token output after injection
    4. Compare with the sample's correct answer

    If the injected target prompt produces the correct answer, the FJC
    layer contains causally sufficient information for that sample.

    ``effect_label`` semantics:
      - "flipped": target prompt output matches the correct answer
        (information transfer confirmed)
      - "no_effect": target prompt output does NOT match the correct
        answer (information insufficient at this layer)
    """

    name = "activation_patching_fjc"

    def run(
        self,
        samples: list[Sample],
        cache: HiddenStateCache,
        diagnostics: DiagnosticResult,
        backend: InterventionBackend,
        *,
        config: dict | None = None,
        **kwargs: Any,
    ) -> CausalValidationResult:
        config = config or {}
        intervention_cfg = config.get("intervention", {})
        target_prompt = intervention_cfg.get("target_prompt", DEFAULT_TARGET_PROMPT)
        target_position = intervention_cfg.get("target_position", DEFAULT_TARGET_POSITION)

        # Select eligible samples
        selected, skipped_results = select_fjc_samples(samples, diagnostics, cache)

        logger.info(
            "Activation patching at FJC: %d selected, %d skipped",
            len(selected), len(skipped_results),
        )

        # Build intervention requests — inject real FJC hidden state
        # into the target prompt
        requests: list[InterventionRequest] = []
        for sel in selected:
            # Read the real hidden state at the FJC layer from cache
            # cache.hidden_states shape: (N, L, D)
            fjc_hidden = cache.hidden_states[sel.cache_index, sel.source_layer]

            requests.append(InterventionRequest(
                sample_id=sel.sample.id,
                source_prompt=sel.sample.prompt,
                target_prompt=target_prompt,
                source_hidden=fjc_hidden,
                target_layer=sel.source_layer,
                target_position=target_position,
            ))

        # Run interventions
        responses = backend.run_interventions(requests)

        # Build per-sample results for selected samples
        response_map = {r.sample_id: r for r in responses}
        sample_results: list[SampleCausalResult] = list(skipped_results)

        n_effective = len(selected)
        n_flipped = 0

        for sel in selected:
            resp = response_map.get(sel.sample.id)
            if resp is None:
                n_effective -= 1
                sample_results.append(SampleCausalResult(
                    sample_id=sel.sample.id,
                    selected=True,
                    skip_reason="intervention_failed",
                    source_layer=sel.source_layer,
                ))
                continue

            # "flipped" means the target prompt produced the correct answer
            # after receiving the FJC hidden state — confirming causal sufficiency
            intervened_correct = resp.intervened_output == sel.sample.answer
            if intervened_correct:
                n_flipped += 1

            effect_label = "flipped" if intervened_correct else "no_effect"

            sample_results.append(SampleCausalResult(
                sample_id=sel.sample.id,
                selected=True,
                source_layer=sel.source_layer,
                target_layer=sel.source_layer,
                original_output=resp.original_output,
                intervened_output=resp.intervened_output,
                original_correct=None,  # target prompt has no "correct" baseline
                intervened_correct=intervened_correct,
                effect_label=effect_label,
            ))

        # Summary metrics
        flip_rate = n_flipped / n_effective if n_effective > 0 else 0.0
        summary = {
            "n_selected": len(selected),
            "n_effective": n_effective,
            "n_flipped": n_flipped,
            "flip_rate": flip_rate,
        }

        logger.info(
            "Activation patching at FJC complete: %d selected, %d effective, "
            "%d flipped (flip_rate=%.3f)",
            len(selected), n_effective, n_flipped, flip_rate,
        )

        return CausalValidationResult(
            experiment=self.name,
            model_id=cache.model_id,
            eval_dataset_id=diagnostics.eval_dataset_id,
            benchmark=diagnostics.benchmark,
            subset=diagnostics.subset,
            sample_results=sample_results,
            summary=summary,
        )
