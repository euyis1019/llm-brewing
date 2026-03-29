"""Causal validation subsystem for Brewing.

Provides intervention-based causal verification of diagnostic findings.
Three planned experiments:
  - activation_patching_fjc: Patch hidden state at FJC layer (MVP)
  - layer_skipping: Early exit at FJC for Overprocessed samples (future)
  - reinjection: Re-inject hidden states for Unresolved samples (future)
"""

from .base import CausalValidator, VALIDATOR_REGISTRY, get_validator
from .selectors import select_fjc_samples
from .activation_patching import ActivationPatchingFJC

__all__ = [
    "CausalValidator",
    "VALIDATOR_REGISTRY",
    "get_validator",
    "select_fjc_samples",
    "ActivationPatchingFJC",
]
