"""CLI entry point for the Brewing framework.

Responsible for: parsing CLI arguments, loading YAML config, constructing
RunConfig, optionally loading a model, and delegating to Orchestrator.run().

NOT responsible for: any analysis logic or result interpretation.

Usage:
    python -m brewing --config path/to/config.yaml           # standard run
    python -m brewing --config path/to/config.yaml --verbose  # debug logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from brewing.schema import RunConfig
from brewing.registry import get_method_class


def load_config(config_path: str | Path) -> RunConfig:
    """Read a YAML config file and return a validated RunConfig."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    if not isinstance(config_dict, dict):
        raise ValueError(
            f"YAML config must be a mapping, got {type(config_dict).__name__}"
        )
    return RunConfig(**config_dict)


def needs_model_online(config: RunConfig) -> bool:
    """Check whether any requested method requires the model to be loaded."""
    for method_name in config.methods:
        method_cls = get_method_class(method_name)
        if method_cls().requirements().needs_model_online:
            return True
    return False


def build_model_load_kwargs(config: RunConfig) -> dict[str, Any]:
    """Build kwargs for AutoModelForCausalLM.from_pretrained based on config."""
    import torch

    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "output_hidden_states": True,
    }

    if config.quantization == "int8":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif config.quantization == "int4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.float16

    return load_kwargs


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Brewing: Layer-wise Mechanistic Interpretability Framework"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args(argv)

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Ensure benchmarks and methods are registered
    import brewing.benchmarks  # noqa: F401
    import brewing.methods.linear_probing  # noqa: F401
    import brewing.methods.csd  # noqa: F401

    # Load config
    config = load_config(args.config)

    # Load model if needed
    model = None
    tokenizer = None

    if needs_model_online(config):
        logging.info("Loading model: %s", config.model_id)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.model_id)
            load_kwargs = build_model_load_kwargs(config)
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id, **load_kwargs
            )
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.warning(
                "Could not load model: %s. Running with synthetic cache.", e
            )

    # Run
    from brewing.orchestrator import Orchestrator

    orchestrator = Orchestrator(config)
    results = orchestrator.run(model=model, tokenizer=tokenizer)

    # Print summary
    print("\n" + "=" * 60)
    print("Brewing Run Complete")
    print("=" * 60)
    for subset_name, subset_result in results.get("subsets", {}).items():
        print(f"\n  {subset_name}:")
        if isinstance(subset_result, dict):
            for k, v in subset_result.items():
                if k == "diagnostics" and isinstance(v, dict):
                    print(f"    {k}:")
                    for dk, dv in v.items():
                        print(f"      {dk}: {dv}")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"    {subset_result}")

    print(f"\nOutput: {config.output_root}/")


if __name__ == "__main__":
    main()
