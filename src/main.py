# src/main.py
"""Entry-point that orchestrates the full experimental workflow.

Usage examples:

    # Smoke test only
    uv run python -m src.main --smoke-test

    # Full experiment only
    uv run python -m src.main --full-experiment

If neither flag is provided, a two-phase execution is performed: the
smoke test is run first and, if (and only if) it succeeds, the full
experiment is executed afterwards.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import yaml

from .evaluate import EXPERIMENT_DISPATCH

# -------------------------------------------------------------------
#  configuration helpers
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

SMOKE_CFG_PATH = CONFIG_DIR / "smoke_test.yaml"
FULL_CFG_PATH = CONFIG_DIR / "full_experiment.yaml"


def _load_cfg(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file '{path}' not found.")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # inject HF token so that downstream modules do not need to read env-vars again
    import os

    cfg["_hf_token"] = os.getenv(cfg.get("hf_token_env", "HF_TOKEN"))
    return cfg


# -------------------------------------------------------------------
#  experiment execution function
# -------------------------------------------------------------------

def _run_experiments(cfg: Dict, smoke: bool):
    for exp_name in cfg["experiments"]:
        func = EXPERIMENT_DISPATCH.get(exp_name)
        if func is None:
            print(f"[WARN] Unknown experiment '{exp_name}' – skipping.")
            continue
        try:
            func(cfg, smoke)
        except RuntimeError as e:
            print(f"[ERROR] Experiment '{exp_name}' terminated: {e}")
            sys.exit(1)


# -------------------------------------------------------------------
#  CLI handling & main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HYPERION-SHIELD experimental pipeline")
    parser.add_argument("--smoke-test", action="store_true", help="Run only the smoke test configuration")
    parser.add_argument("--full-experiment", action="store_true", help="Run the full experimental configuration")
    args = parser.parse_args()

    if args.smoke_test and args.full_experiment:
        parser.error("The flags --smoke-test and --full-experiment are mutually exclusive.")

    if args.smoke_test:
        cfg = _load_cfg(SMOKE_CFG_PATH)
        _run_experiments(cfg, smoke=True)
        return

    if args.full_experiment:
        cfg = _load_cfg(FULL_CFG_PATH)
        _run_experiments(cfg, smoke=False)
        return

    # default two-phase execution -------------------------------------------------
    try:
        cfg_smoke = _load_cfg(SMOKE_CFG_PATH)
        _run_experiments(cfg_smoke, smoke=True)
    except SystemExit as e:
        # propagate failure directly – do NOT attempt full if smoke failed
        raise e

    cfg_full = _load_cfg(FULL_CFG_PATH)
    _run_experiments(cfg_full, smoke=False)


if __name__ == "__main__":
    main()
