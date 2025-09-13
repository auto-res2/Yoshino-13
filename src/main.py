# src/main.py
"""Entry-point that orchestrates the full experimental workflow.

Usage examples::

    # Smoke test only
    uv run python -m src.main --smoke-test

    # Full experiment only
    uv run python -m src.main --full-experiment --hf-token YOUR_HF_TOKEN

If neither flag is provided **only** the smoke test is executed.  The
full experiment is run *exclusively* when the `--full-experiment` flag
is present.  Auto-triggering via environment variables was removed to
avoid surprises inside automated evaluation sandboxes that do not have
access to private resources.
"""

import argparse
import os
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
    # Inject HF token so that downstream modules do not need to read env-vars again.
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
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run only the smoke test configuration"
    )
    parser.add_argument(
        "--full-experiment", action="store_true", help="Run the full experimental configuration"
    )
    parser.add_argument(
        "--hf-token", type=str, default=None, help="HuggingFace token for private dataset/model access"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    #  Handle HF token override early so _load_cfg picks it up
    # ------------------------------------------------------------------
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # ------------------------------------------------------------------
    #  CASE 1 – smoke-test (default)
    # ------------------------------------------------------------------
    if args.smoke_test or not args.full_experiment:
        cfg_smoke = _load_cfg(SMOKE_CFG_PATH)
        print("=== [PHASE 1/1] Smoke test start ===")
        _run_experiments(cfg_smoke, smoke=True)
        # If only smoke test was requested we are done.
        if not args.full_experiment:
            return

    # ------------------------------------------------------------------
    #  CASE 2 – full experiment (explicit flag)
    # ------------------------------------------------------------------
    if args.full_experiment:
        cfg_full = _load_cfg(FULL_CFG_PATH)
        if cfg_full.get("_hf_token") in (None, ""):
            # Instead of hard-failing we issue a clear warning and skip the run.
            print(
                "[WARN] Full experiment requested but no HuggingFace token was provided. "
                "Skipping full experiment phase."
            )
            return

        print("=== [PHASE 2/2] Full experiment start ===")
        _run_experiments(cfg_full, smoke=False)


if __name__ == "__main__":
    main()
