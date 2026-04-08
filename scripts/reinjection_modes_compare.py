#!/usr/bin/env python3
"""Re-injection with multiple injection modes, one mode per subprocess.

Each mode gets a fresh model load to avoid nnsight memory accumulation.
"""

import json
import logging
import sys
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent / "brewing_output"
OUT_DIR = BASE / "artifacts" / "reinjection"

MODES = [
    {"name": "replace",     "injection_mode": "replace",     "alpha": 1.0},
    {"name": "norm_match",  "injection_mode": "norm_match",  "alpha": 1.0},
    {"name": "alpha_0.3",   "injection_mode": "alpha_blend",  "alpha": 0.3},
    {"name": "alpha_0.5",   "injection_mode": "alpha_blend",  "alpha": 0.5},
]


def run_single_mode(mode_cfg):
    """Run a single mode in a subprocess."""
    cmd = [
        sys.executable, str(Path(__file__).parent / "_reinjection_single_mode.py"),
        "--mode", mode_cfg["injection_mode"],
        "--alpha", str(mode_cfg["alpha"]),
        "--output", str(OUT_DIR / f"results_7B_{mode_cfg['name']}.json"),
    ]
    logger.info("Running mode=%s ...", mode_cfg["name"])
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error("Mode %s failed with exit code %d", mode_cfg["name"], result.returncode)
        return False
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for mode_cfg in MODES:
        run_single_mode(mode_cfg)

    # Print combined summary
    print("\n=== Re-injection Mode Comparison (7B, 50 UR + 25 Res per task) ===")
    print(f"{'Task':<18} {'Mode':<12} {'Layer':>6} {'UR_rescue':>15} {'Res_maintain':>15}")
    for mode_cfg in MODES:
        path = OUT_DIR / f"results_7B_{mode_cfg['name']}.json"
        if not path.exists():
            continue
        with open(path) as f:
            results = json.load(f)
        for r in results:
            for rnd, d in r["rounds"].items():
                ur_s = f"{d['ur_rescued']}/{d['ur_n']} ({d['ur_rate']*100:.1f}%)" if d['ur_rate'] is not None else "N/A"
                re_s = f"{d['res_maintained']}/{d['res_n']} ({d['res_rate']*100:.1f}%)" if d['res_rate'] is not None else "N/A"
                print(f"  {r['task']:<18} {mode_cfg['name']:<12} L={d['target_layer']:>3} {ur_s:>15} {re_s:>15}")
        print()


if __name__ == "__main__":
    main()
