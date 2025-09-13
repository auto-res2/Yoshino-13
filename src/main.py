"""main.py – command-line entry point.
Supports two operational modes via argparse:
    --smoke-test        run only the quick sanity configuration
    --full-experiment   run full experiment *after* a successful smoke test
If --full-experiment is given without --smoke-test, a smoke test is still
executed first in compliance with the spec.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import yaml

from .train import SimpleTrainer


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def run_config(cfg_path: Path) -> None:
    """Run every experiment block inside a YAML config file."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    for exp_name, exp_cfg in cfg["experiments"].items():
        print("\n============================================")
        print(f"Launching experiment: {exp_name}")
        print("============================================\n")
        trainer = SimpleTrainer(exp_name, exp_cfg)
        trainer.train()


def main(argv=None):
    parser = argparse.ArgumentParser(description="ORACLE research experiments")
    parser.add_argument("--smoke-test", action="store_true", help="run smoke test only")
    parser.add_argument("--full-experiment", action="store_true", help="run full experiment (executes smoke test first)")
    args = parser.parse_args(argv)

    if not args.smoke_test and not args.full_experiment:
        parser.error("At least one of --smoke-test or --full-experiment must be set.")

    # Always run smoke test first whenever full experiment is requested
    if args.smoke_test or args.full_experiment:
        smoke_cfg = CONFIG_DIR / "smoke_test.yaml"
        print("\n=== Running SMOKE TEST ===")
        run_config(smoke_cfg)

    # Run full only if specifically requested *and* smoke succeeded
    if args.full_experiment:
        full_cfg = CONFIG_DIR / "full_experiment.yaml"
        print("\n=== Running FULL EXPERIMENT ===")
        run_config(full_cfg)


if __name__ == "__main__":
    main()
