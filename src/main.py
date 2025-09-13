# src/main.py
"""Entry-point orchestrating the experimental workflow."""

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
    # Inject HF token (may be None/"") so downstream modules do not need to read env-vars.
    cfg["_hf_token"] = os.getenv(cfg.get("hf_token_env", "HF_TOKEN"))
    return cfg


# -------------------------------------------------------------------
#  experiment execution
# -------------------------------------------------------------------

def _run_experiments(cfg: Dict, smoke: bool):
    for exp_name in cfg["experiments"]:
        func = EXPERIMENT_DISPATCH.get(exp_name)
        if func is None:
            print(f"[WARN] Unknown experiment '{exp_name}' – skipping.")
            continue
        func(cfg, smoke)


# -------------------------------------------------------------------
#  CLI handling & main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HYPERION-SHIELD experimental pipeline")
    parser.add_argument("--smoke-test", action="store_true", help="Run the smoke-test configuration only")
    parser.add_argument("--full-experiment", action="store_true", help="Run the full experimental configuration")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token for private access")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    #  HF token override (must be set *before* configurations are loaded)
    # ------------------------------------------------------------------
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # ------------------------------------------------------------------
    #  Phase 1 – smoke test (always runs unless the user explicitly opts out)
    # ------------------------------------------------------------------
    if args.smoke_test or not args.full_experiment:
        cfg_smoke = _load_cfg(SMOKE_CFG_PATH)
        print("=== [PHASE 1/1] Smoke test start ===")
        _run_experiments(cfg_smoke, smoke=True)
        if not args.full_experiment:  # User only wanted the smoke test
            return

    # ------------------------------------------------------------------
    #  Phase 2 – full experiment (optional – may require HF token)
    # ------------------------------------------------------------------
    if args.full_experiment:
        cfg_full = _load_cfg(FULL_CFG_PATH)
        if cfg_full.get("_hf_token") in (None, ""):
            print("[WARN] No HuggingFace token detected – proceeding with public resources only.")
        print("=== [PHASE 2/2] Full experiment start ===")
        _run_experiments(cfg_full, smoke=False)


if __name__ == "__main__":
    main()
