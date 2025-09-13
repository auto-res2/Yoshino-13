"""src/evaluate.py
Evaluation utilities – Experiment-1 (iteration-9).

Updates versus iteration-8
-------------------------
1. Paths now point to `.research/iteration9/**` in compliance with the latest
   storage policy.
2. All other logic remains identical apart from minor housekeeping tweaks.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

# Use a head-less backend **before** importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
import seaborn as sns  # noqa: E402  pylint: disable=wrong-import-position
import torch  # noqa: E402  pylint: disable=wrong-import-position

from .preprocess import ensure_dataset
from .train import GatedRepoAccessError, GenerationConfig, load_guard, load_model

logger = logging.getLogger("tracs_runner.evaluate")

###############################################################################
#   Metric helpers                                                            #
###############################################################################

def compute_asr(outputs):
    """Attack-success-rate – expects each output dict to have key *violation*."""
    n = len(outputs)
    return 0.0 if n == 0 else 100.0 * sum(o["violation"] for o in outputs) / n


def median(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return 0.0
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2.0

###############################################################################
#   Plot helpers – images must reside in .research/iteration9/images           #
###############################################################################

def _save_bar(fig_name: str, labels: List[str], numbers: List[float], ylabel: str) -> str:
    """Save bar-plot under the mandated research directory and return its path."""

    images_dir = Path(".research/iteration9/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = images_dir / f"{fig_name}.pdf"

    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=labels, y=numbers, palette="Blues_d")
    for i, v in enumerate(numbers):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom")
    plt.ylabel(ylabel)
    plt.xlabel("Safety Stack")
    plt.title(fig_name.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    return str(pdf_path)

###############################################################################
#   Core experimental routine                                                 #
###############################################################################

def run_experiment_1(cfg):
    """Execute Experiment-1 and persist outputs below *cfg.output_dir*."""

    logger.info("Running Experiment 1 – %s", cfg.description.split("\n")[0])

    # ------------------------------------------------------------------
    # 1) dataset – LongFuse
    # ------------------------------------------------------------------
    lf_ds = ensure_dataset("longfuse", cfg.datasets["longfuse"])["train"]

    # ------------------------------------------------------------------
    # 2) iterate over models & guard stacks
    # ------------------------------------------------------------------
    results_all: Dict[str, Dict[str, Any]] = {}
    for model_key, model_id in cfg.models.items():
        try:
            tokenizer, model = load_model(model_id)
        except GatedRepoAccessError:
            logger.warning("Skipping %s (gated repo, no HF_TOKEN).", model_id)
            continue
        except Exception as e:  # Any other issue → fail-fast
            logger.error("Unhandled error while loading %s: %s", model_id, e)
            raise

        device = next(model.parameters()).device  # universal device getter

        model_res: Dict[str, Any] = {}
        for stack in cfg.safety_stacks:
            guard = load_guard(stack, model)
            stack_outputs = []

            for seed in cfg.seeds:
                torch.manual_seed(seed)
                for ex in lf_ds:
                    prompt = ex.get("prompt", ex.get("text", ""))
                    inp = tokenizer(prompt, return_tensors="pt").to(device)

                    gen_cfg = GenerationConfig(
                        do_sample=True,
                        top_p=0.7,
                        temperature=1.0,
                        max_new_tokens=32,  # keep smoke-test light-weight
                    )
                    t0 = time.perf_counter()
                    if stack == "no-guard":
                        output_ids = model.generate(**inp, generation_config=gen_cfg)
                        violation = False
                    else:
                        output_ids, info = guard.generate(prompt, generation_config=gen_cfg)
                        violation = info.get("violation", False)

                    latency = (time.perf_counter() - t0) / max(
                        1, output_ids.shape[1] - inp["input_ids"].shape[1]
                    )
                    stack_outputs.append({"violation": bool(violation), "latency": latency * 1000.0})

            asr = compute_asr(stack_outputs)
            median_latency = median([o["latency"] for o in stack_outputs])
            model_res[stack] = {"ASR": asr, "median_latency_ms": median_latency}
            logger.info(
                "%s – %s: ASR=%.2f, median latency=%.2f ms", model_key, stack, asr, median_latency
            )
        results_all[model_key] = model_res

    if not results_all:
        logger.error("All models were skipped – no results produced.")
        return

    # ------------------------------------------------------------------
    # 3) persist results JSON – one file per experiment under cfg.output_dir
    # ------------------------------------------------------------------
    base_dir = Path(cfg.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / "experiment_1_results.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(results_all, fp, indent=2)

    # ------------------------------------------------------------------
    # 4) figures
    # ------------------------------------------------------------------
    fig_files = []
    for model_key, model_res in results_all.items():
        labels, vals = zip(*[(k, v["ASR"]) for k, v in model_res.items()])
        pdf = _save_bar(f"long_horizon_asr_{model_key}", list(labels), list(vals), "ASR % (↓)")
        fig_files.append(pdf)

    # ------------------------------------------------------------------
    # 5) stdout trace for CI visibility
    # ------------------------------------------------------------------
    print("\n>>> Numerical Results (ASR & Latency)")
    print(json.dumps(results_all, indent=2))

    print("\nFigures generated:")
    for fig in fig_files:
        print(" –", fig)

    print("\nFull JSON saved to:", out_path)
    with open(out_path, "r", encoding="utf-8") as fp:
        print(fp.read())
