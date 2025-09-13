"""evaluate.py – plotting / evaluation helpers.
Currently only contains the training-curve visualisation that was
originally embedded inside SimpleTrainer.  Splitting it out keeps
train.py focused on optimisation logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_training_curve(loss_history: List[float], fig_path: Path) -> None:
    """Render and save a simple loss-vs-step curve."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, label="train_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
