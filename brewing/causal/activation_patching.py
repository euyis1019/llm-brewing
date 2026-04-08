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
DEFAULT_ANSWER_SPACE = [str(d) for d in range(10)]  # digits 0-9, same as CSD


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
        # answer_space: None = global argmax (genuine causal sufficiency test).
        # Setting to ["0".."9"] would use CSD-consistent comparison but is
        # tautological at FJC (CSD is correct there by definition).
        answer_space = intervention_cfg.get("answer_space", None)
        # layer_offsets: run patching at FJC + offset for each offset.
        # Default [0] = FJC only. Use e.g. [-5,-2,0,2,5] for contrast.
        layer_offsets = intervention_cfg.get("layer_offsets", [0])

        # Select eligible samples
        selected, skipped_results = select_fjc_samples(samples, diagnostics, cache)

        logger.info(
            "Activation patching at FJC: %d selected, %d skipped, offsets=%s",
            len(selected), len(skipped_results), layer_offsets,
        )

        n_layers = cache.n_layers
        all_offset_results: dict[int, dict] = {}
        primary_response_map: dict[str, Any] = {}

        for offset in layer_offsets:
            requests: list[InterventionRequest] = []
            offset_selected = []

            for sel in selected:
                layer = sel.source_layer + offset
                if layer < 0 or layer >= n_layers:
                    continue
                fjc_hidden = cache.hidden_states[sel.cache_index, layer]
                offset_selected.append(sel)

                requests.append(InterventionRequest(
                    sample_id=sel.sample.id,
                    source_prompt=sel.sample.prompt,
                    target_prompt=target_prompt,
                    source_hidden=fjc_hidden,
                    target_layer=layer,
                    target_position=target_position,
                    answer_space=answer_space,
                ))

            responses = backend.run_interventions(requests)
            response_map = {r.sample_id: r for r in responses}

            n_eff = 0
            n_flip = 0
            for sel in offset_selected:
                resp = response_map.get(sel.sample.id)
                if resp is None:
                    continue
                n_eff += 1
                if resp.intervened_output == sel.sample.answer:
                    n_flip += 1

            flip_rate = n_flip / n_eff if n_eff > 0 else 0.0
            all_offset_results[offset] = {
                "n_effective": n_eff,
                "n_flipped": n_flip,
                "flip_rate": flip_rate,
            }
            logger.info(
                "  offset=%+d: %d effective, %d flipped (flip_rate=%.3f)",
                offset, n_eff, n_flip, flip_rate,
            )

            if offset == 0:
                primary_response_map = response_map

        # Build per-sample results from offset=0
        sample_results: list[SampleCausalResult] = list(skipped_results)

        for sel in selected:
            resp = primary_response_map.get(sel.sample.id)
            if resp is None:
                sample_results.append(SampleCausalResult(
                    sample_id=sel.sample.id,
                    selected=True,
                    skip_reason="intervention_failed",
                    source_layer=sel.source_layer,
                ))
                continue

            intervened_correct = resp.intervened_output == sel.sample.answer
            effect_label = "flipped" if intervened_correct else "no_effect"

            sample_results.append(SampleCausalResult(
                sample_id=sel.sample.id,
                selected=True,
                source_layer=sel.source_layer,
                target_layer=sel.source_layer,
                original_output=resp.original_output,
                intervened_output=resp.intervened_output,
                original_correct=None,
                intervened_correct=intervened_correct,
                effect_label=effect_label,
            ))

        # Summary
        summary: dict[str, Any] = {"n_selected": len(selected)}
        for k, v in all_offset_results.items():
            summary[f"offset_{k}"] = v
        if 0 in all_offset_results:
            summary["n_effective"] = all_offset_results[0]["n_effective"]
            summary["n_flipped"] = all_offset_results[0]["n_flipped"]
            summary["flip_rate"] = all_offset_results[0]["flip_rate"]

        logger.info(
            "Activation patching complete: %d selected, offsets=%s",
            len(selected), list(all_offset_results.keys()),
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
