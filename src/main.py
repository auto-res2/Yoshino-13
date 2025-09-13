# src/main.py
"""Entry-point that orchestrates the full experimental workflow.

Usage examples:

    # Smoke test only
    uv run python -m src.main --smoke-test

    # Full experiment only
    uv run python -m src.main --full-experiment --hf-token YOUR_HF_TOKEN

If neither flag is provided, the smoke test is executed.  The *full*
experiment is run **only** when the `--full-experiment` flag is
provided or when the environment variable `RUN_FULL_EXPERIMENT` is set
to ``1``.  This change avoids unintentional long-running private-data
experiments on CI while still giving power-users a one-command pathway
for two-phase execution.
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
    #  Decide execution plan
    # ------------------------------------------------------------------
    run_smoke = args.smoke_test or not args.full_experiment
    run_full = args.full_experiment or os.getenv("RUN_FULL_EXPERIMENT") == "1"

    # Phase 1 – smoke -----------------------------------------------------------
    if run_smoke:
        cfg_smoke = _load_cfg(SMOKE_CFG_PATH)
        _run_experiments(cfg_smoke, smoke=True)

    # Phase 2 – full experiment (optional) --------------------------------------
    if run_full and not args.smoke_test:
        cfg_full = _load_cfg(FULL_CFG_PATH)
        # Fail fast if private resources are requested but no token is present.
        if cfg_full.get("_hf_token") in (None, ""):
            print(
                "[ERROR] Full experiment requires access to private models/datasets. "
                "Please provide a valid HuggingFace token via --hf-token or the HF_TOKEN environment variable."
            )
            sys.exit(1)
        _run_experiments(cfg_full, smoke=False)


if __name__ == "__main__":
    main()
