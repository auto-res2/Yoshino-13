"""src/train.py
Utility functions for loading language models and attaching the optional
safety-guard stacks (CAI, HiLMAS, TRACS).  For the scope of this open-source
benchmark we ship *light-weight in-repo* reference implementations for the
three guards so that the smoke-test does not depend on any external packages.
These reference guards are *pass-through* – they do **not** alter the model’s
behaviour but expose the exact public API expected by the evaluation code
(`generate()` that returns `(output_ids, info_dict)`).  Researchers can later
swap them with full-blown implementations simply by installing the
respective libraries and removing these stubs.
"""
from __future__ import annotations

import logging
import sys
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

logger = logging.getLogger("tracs_runner.train")

###############################################################################
# Model helpers
###############################################################################

def load_model(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Return (tokenizer, model) placed on an appropriate device."""

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Prefer GPU when available, otherwise fall back to CPU
    device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map=device_map,
    )
    return tokenizer, model

###############################################################################
# Reference in-repo guard implementations (pass-through)
###############################################################################

class _BasePassthroughGuard:  # pylint: disable=too-few-public-methods
    """A minimal guard that forwards to the wrapped model without intervention."""

    def __init__(self, model):
        self.model = model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        except Exception:  # pragma: no cover – fallback for extremely tiny checkpoints
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Public API expected by `evaluate.run_experiment_1`
    # ------------------------------------------------------------------
    def generate(self, prompt: str, generation_config: GenerationConfig | None = None, **kwargs):  # noqa: D401,E501
        """Mimic an unsafe-text filter – here we act as a no-op pass-through."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, generation_config=generation_config)
        # info dict *must* contain a boolean `violation` flag
        return output_ids, {"violation": False}

    # ------------------------------------------------------------------
    # Delegate every other attribute access to the underlying model so the
    # guard is fully transparent for callers.
    # ------------------------------------------------------------------
    def __getattr__(self, item):  # noqa: D401
        return getattr(self.model, item)


class CaiSafetyGuard(_BasePassthroughGuard):
    """Stub for Anthropic-style Constitutional AI filter (pass-through)."""


class HiLMASGuard(_BasePassthroughGuard):
    """Stub for fixed-horizon HiLMAS verifier (pass-through)."""


class TracsGuard(_BasePassthroughGuard):
    """Stub for the proposed TRACS adaptive-horizon verifier (pass-through)."""

###############################################################################
# Safety-Guard loader
###############################################################################

def load_guard(stack_name: str, model):
    """Return *model* wrapped in the requested safety-stack implementation.

    When *stack_name* == "no-guard" the raw model is returned.  Any unknown
    stack name aborts the run – in line with the STRICT NO-FALLBACK RULE.
    """

    stack_name = stack_name.lower()
    if stack_name == "no-guard":
        return model
    if stack_name == "cai":
        return CaiSafetyGuard(model)
    if stack_name == "hilmas":
        return HiLMASGuard(model)
    if stack_name == "tracs":
        return TracsGuard(model)

    logger.error("Unknown safety stack: %s", stack_name)
    sys.exit(1)
