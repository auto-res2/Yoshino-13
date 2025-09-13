"""train.py – model definitions and lightweight trainer extracted
from the original monolithic script.  No new logic introduced; only minimal
modifications for package re-structuring, device-safety and path handling.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import (
    T5ForConditionalGeneration,
    WhisperForConditionalGeneration,
)

from .evaluate import save_training_curve  # plotting util
from .preprocess import build_dataloader


# -----------------------------------------------------------------------------
#  Differentiable Top-k (Gumbel-Sinkhorn) – identical to original implementation
#    but now robust to the case k > len(scores)
# -----------------------------------------------------------------------------

def gumbel_sinkhorn_topk(scores: torch.Tensor, k: int, tau: float) -> torch.Tensor:
    """Return soft k-hot selection vector via Gumbel-Softmax.

    The original implementation assumed k ≤ len(scores).  If k is larger than
    the number of available items (e.g. small batch sizes in smoke tests),
    ``torch.topk`` throws an error.  We therefore clamp *k* to the valid range
    while keeping the gradient-flow intact.  This change preserves the
    mathematical behaviour in the valid regime and gracefully degrades to a
    full-support softmax when k exceeds the support size.
    """
    # 1. Sample Gumbel noise & compute relaxed category probabilities
    gumbel_noise = -torch.empty_like(scores).exponential_().log()  # ~Gumbel(0,1)
    y = (scores + gumbel_noise) / tau
    probs = F.softmax(y, dim=-1)

    # 2. Select (soft) top-k entries.  Guard against k > |scores|.
    support_size = probs.shape[-1]
    k_eff = min(k, support_size)
    if k_eff == support_size:
        return probs  # nothing to mask – keep differentiability

    topk_vals, topk_idx = probs.topk(k_eff, dim=-1)
    mask = torch.zeros_like(probs)
    mask.scatter_(-1, topk_idx, topk_vals)
    return mask


# -----------------------------------------------------------------------------
#  ORACLE  Heads / Backbones  (vision, text, audio)
# -----------------------------------------------------------------------------


class OracleHead(nn.Module):
    def __init__(self, in_dim: int, k: int = 5, tau: float = 0.05):
        super().__init__()
        self.score = nn.Linear(in_dim, 1)
        self.k, self.tau = k, tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_dim)
        logits = self.score(x).squeeze(-1)  # (B,)
        soft_topk = gumbel_sinkhorn_topk(
            logits.unsqueeze(0), k=self.k, tau=self.tau
        ).squeeze(0)
        return soft_topk


class OracleVision(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.backbone = timm.create_model(
            cfg["backbone"], pretrained=True, num_classes=0
        )
        hdim = self.backbone.num_features
        self.head = OracleHead(hdim, k=cfg["k"], tau=cfg["tau"])
        self.criterion = nn.L1Loss()

    def forward(self, batch):
        device = next(self.parameters()).device
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True).float()
        z = self.backbone(x)
        y_pred = self.head(z)
        loss = self.criterion(y_pred, y)
        return loss, y_pred.detach().cpu(), y.cpu()


class OracleText(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        # smaller checkpoint for CI friendliness
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.head = OracleHead(
            self.model.config.d_model, k=cfg["k"], tau=cfg["tau"]
        )
        self.criterion = nn.L1Loss()

    def forward(self, batch):
        device = next(self.parameters()).device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["labels"].to(device).float()
        enc = (
            self.model.encoder(input_ids=input_ids, attention_mask=attn_mask)
            .last_hidden_state.mean(1)
        )
        y_pred = self.head(enc)
        loss = self.criterion(y_pred, y)
        return loss, y_pred.detach().cpu(), y.cpu()


class OracleAudio(nn.Module):
    """Lightweight audio variant leveraging Whisper-tiny for fast CI."""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-tiny"
        )
        self.head = OracleHead(
            self.model.config.d_model, k=cfg["k"], tau=cfg["tau"]
        )
        self.criterion = nn.L1Loss()
        # freeze Whisper to cut memory & time
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, batch):
        device = next(self.parameters()).device
        mel = batch["mel"].to(device, non_blocking=True)
        y = batch["labels"].to(device).float()
        enc_out = self.model.model.encoder(input_features=mel).last_hidden_state
        enc = enc_out.mean(1)  # (B, d_model)
        y_pred = self.head(enc)
        loss = self.criterion(y_pred, y)
        return loss, y_pred.detach().cpu(), y.cpu()


# -----------------------------------------------------------------------------
#  SimpleTrainer  (lightweight – no external lightning dependency)
# -----------------------------------------------------------------------------

class SimpleTrainer:
    """Single-GPU (or CPU fallback) training loop."""

    def __init__(self, exp_name: str, cfg: Dict):
        self.exp_name = exp_name
        self.cfg = cfg

        # device selection ----------------------------------------------------
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(0)

        # model ---------------------------------------------------------------
        model_type = cfg["model"].lower()
        if model_type == "oracle_vision":
            self.model = OracleVision(cfg)
        elif model_type == "oracle_text":
            self.model = OracleText(cfg)
        elif model_type == "oracle_audio":
            self.model = OracleAudio(cfg)
        else:
            raise RuntimeError(f"Unknown model {model_type}")
        self.model.to(self.device)

        # optimiser -----------------------------------------------------------
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=float(cfg["lr"]))

        # data ----------------------------------------------------------------
        self.dl_train = build_dataloader(cfg["dataset"], split="train")

        # bookkeeping ---------------------------------------------------------
        root = Path(__file__).resolve().parent.parent
        self.img_dir = root / ".research" / "iteration9" / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.res_dir = root / ".research" / "iteration9"
        self.res_dir.mkdir(parents=True, exist_ok=True)
        self.loss_history: List[float] = []

    # ----------------------------------------------------------------------
    #  Training loop
    # ----------------------------------------------------------------------
    def train(self):
        self.model.train()
        max_steps = self.cfg.get("max_steps", 100)
        for step, batch in enumerate(self.dl_train):
            self.opt.zero_grad(set_to_none=True)
            loss, *_ = self.model(batch)
            loss.backward()
            self.opt.step()
            self.loss_history.append(loss.item())

            if step % 10 == 0:
                print(f"[exp:{self.exp_name}] step {step:04d} loss={loss.item():.4f}")
            if step + 1 >= max_steps:
                break

        # ---------------- results persistence ------------------------------
        fig_path = self.img_dir / f"{self.exp_name}_training_loss.pdf"
        save_training_curve(self.loss_history, fig_path)

        res = {
            "experiment": self.exp_name,
            "final_train_loss": self.loss_history[-1],
            "mean_train_loss": float(sum(self.loss_history) / len(self.loss_history)),
            "steps": len(self.loss_history),
            "figure": str(fig_path.relative_to(self.res_dir.parent)),
        }
        res_path = self.res_dir / f"{self.exp_name}_results.json"
        json.dump(res, open(res_path, "w"), indent=2)

        # mandatory stdout ---------------------------------------------------
        print("\n=== Numerical Results ===")
        print(json.dumps(res, indent=2))
        print("\nFigure saved →", res["figure"])
