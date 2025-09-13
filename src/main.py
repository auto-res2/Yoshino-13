"""src/main.py
Entry-point with CLI flags for running the TRACS experiments (iteration-9).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

from .evaluate import run_experiment_1

logger = logging.getLogger("tracs_runner.main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

###############################################################################
#   Config helpers                                                             #
###############################################################################

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_SMOKE_CFG = _CONFIG_DIR / "smoke_test.yaml"
_FULL_CFG = _CONFIG_DIR / "full_experiment.yaml"


class ExperimentConfig:  # pylint: disable=too-few-public-methods
    __slots__ = (
        "name",
        "description",
        "datasets",
        "models",
        "safety_stacks",
        "seeds",
        "hyper",
        "output_dir",
    )

    def __init__(self, raw):
        for k, v in raw.items():
            setattr(self, k, v)

    @staticmethod
    def load(path: Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as fp:
            raw = yaml.safe_load(fp)
        return ExperimentConfig(raw)

###############################################################################
#   CLI parsing                                                                #
###############################################################################

def _parse_args():
    p = argparse.ArgumentParser(description="TRACS benchmark runner")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--smoke-test", action="store_true", help="Run the small CI test")
    g.add_argument("--full-experiment", action="store_true", help="Run the full benchmark")
    return p.parse_args()

###############################################################################
#   Main orchestration                                                         #
###############################################################################

def _run_cfg(cfg_path: Path):
    if not cfg_path.exists():
        logger.error("Config file %s not found", cfg_path)
        sys.exit(1)

    cfg = ExperimentConfig.load(cfg_path)

    # Enable mixed-precision tweaks on GPU
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if cfg.name == "experiment_1":
        run_experiment_1(cfg)
    else:
        logger.error("Unknown experiment name %s", cfg.name)
        sys.exit(1)


def main():
    args = _parse_args()

    if args.smoke_test:
        _run_cfg(_SMOKE_CFG)
        return

    if args.full_experiment:
        _run_cfg(_FULL_CFG)
        return

    # Default: two-phase (smoke then full)
    logger.info("No flag provided – executing smoke-test followed by full experiment …")
    _run_cfg(_SMOKE_CFG)
    _run_cfg(_FULL_CFG)


if __name__ == "__main__":
    main()
