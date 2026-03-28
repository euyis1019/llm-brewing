"""DiagnosticsPipeline — S3 only.

Runs diagnostics from persisted MethodResult files on disk.
Does not require the model to be online.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from brewing.diagnostics.outcome import run_diagnostics_from_disk
from .base import PipelineBase

logger = logging.getLogger(__name__)


class DiagnosticsPipeline(PipelineBase):
    """S3: run diagnostics from disk-persisted results."""

    def run(self, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
        results_summary: dict[str, Any] = {"subsets": {}}

        for subset_name in self.subsets:
            logger.info("=" * 60)
            logger.info("[diagnostics] Processing subset: %s", subset_name)
            logger.info("=" * 60)

            try:
                key = self.make_key(subset_name, "eval")
                diag = run_diagnostics_from_disk(
                    results_dir=self.config.output_root,
                    key=key,
                )
                results_summary["subsets"][subset_name] = {
                    "outcome_distribution": diag.outcome_distribution,
                    "mean_fpcl_normalized": diag.mean_fpcl_normalized,
                    "mean_fjc_normalized": diag.mean_fjc_normalized,
                    "mean_delta_brew": diag.mean_delta_brew,
                }
            except Exception as e:
                logger.error("Failed on subset '%s': %s", subset_name, e, exc_info=True)
                results_summary["subsets"][subset_name] = {"error": str(e)}

        return results_summary
