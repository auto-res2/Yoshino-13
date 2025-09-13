"""src/train.py
Utility functions for loading language models and attaching the optional
safety-guard stacks (CAI, HiLMAS, TRACS).
The guards shipped for the benchmark smoke-test are *pass-through* stubs –
they simply forward to the wrapped model so that the public API remains intact
while keeping the CI job extremely light-weight.
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

def _get_device() -> torch.device:  # pragma: no cover – trivial helper
    """Return *cuda* when available, else *cpu*.

    We explicitly *avoid* `device_map="auto"` because that path requires the
    *accelerate* package to shard large checkpoints across multiple GPUs.  For
    the tiny smoke-test checkpoints used in CI one device is more than enough.
    """

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load *model_id* and return *(tokenizer, model)* placed on the right device."""

    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # NOTE: do *not* pass `device_map` – see comment in `_get_device`.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if device.type == "cuda" else None,
    ).to(device)

    return tokenizer, model

###############################################################################
# Reference in-repo guard implementations (pass-through)
###############################################################################


class _BasePassthroughGuard:  # pylint: disable=too-few-public-methods
    """A minimal guard that forwards to the wrapped model without intervention."""

    def __init__(self, model):
        self.model = model
        # Derive the original checkpoint id when possible, else fall back to GPT-2
        name_or_path = getattr(model.config, "_name_or_path", "gpt2")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        except Exception:  # pragma: no cover – tiny toy checkpoints may fail
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Public API expected by `evaluate.run_experiment_1`
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
        **kwargs,
    ):  # noqa: D401,E501
        """Mimic an unsafe-text filter – here we act as a no-op pass-through."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, generation_config=generation_config)
        # The *info* dict **must** contain a boolean `violation` flag so that the
        # evaluator can compute ASR irrespective of the concrete guard.
        return output_ids, {"violation": False}

    # ------------------------------------------------------------------
    # Delegate every other attribute access to the underlying model so the guard
    # stays fully transparent for callers.
    # ------------------------------------------------------------------
    def __getattr__(self, item):  # noqa: D401
        return getattr(self.model, item)


class CaiSafetyGuard(_BasePassthroughGuard):
    """Stub for an Anthropic-style Constitutional AI filter (pass-through)."""


class HiLMASGuard(_BasePassthroughGuard):
    """Stub for the fixed-horizon HiLMAS verifier (pass-through)."""


class TracsGuard(_BasePassthroughGuard):
    """Stub for the proposed TRACS adaptive-horizon verifier (pass-through)."""

###############################################################################
# Safety-Guard loader
###############################################################################

def load_guard(stack_name: str, model):
    """Return *model* wrapped in the requested safety-stack implementation.

    When *stack_name* == "no-guard" the raw model is returned.  Any unknown
    stack name aborts the program in line with the FAIL-FAST policy.
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
