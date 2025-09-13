"""src/train.py
Utility functions for loading language models and attaching the optional
safety-guard stacks (CAI, HiLMAS, TRACS).  Pure helpers – no training is
performed because the benchmark only evaluates inference-time guards.
"""
from __future__ import annotations

import logging
import sys
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("tracs_runner.train")

###############################################################################
# Model helpers
###############################################################################

def load_model(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Return (tokenizer, model) placed on an appropriate device."""

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Prefer GPU when available, otherwise fall back to CPU
    device_map = "auto" if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map=device_map,
    )
    return tokenizer, model

###############################################################################
# Safety-Guard loader
###############################################################################

def load_guard(stack_name: str, model):
    """Return *model* wrapped in the requested safety-stack implementation.

    When *stack_name* == "no-guard" the raw model is returned.  Any missing
    third-party guard library aborts the run – in line with the STRICT
    NO-FALLBACK RULE.
    """

    stack_name = stack_name.lower()
    if stack_name == "no-guard":
        return model

    try:
        if stack_name == "cai":
            from cai_safety_filter import CaiSafetyGuard

            return CaiSafetyGuard(model)
        if stack_name == "hilmas":
            from hilmas_guard import HiLMASGuard

            return HiLMASGuard(model)
        if stack_name == "tracs":
            from tracs_guard import TracsGuard

            return TracsGuard(model)
    except ModuleNotFoundError as e:
        logger.error(
            "Requested safety stack '%s' but the required Python package is missing.\n%s",
            stack_name,
            e,
        )
        sys.exit(1)

    logger.error("Unknown safety stack: %s", stack_name)
    sys.exit(1)
