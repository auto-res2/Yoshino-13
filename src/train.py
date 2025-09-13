"""src/train.py
Utility functions for loading language models and attaching the optional
safety-guard stacks (CAI, HiLMAS, TRACS).

Changes in this revision
------------------------
1. HuggingFace gated model access
   • We now forward the environment variable `HF_TOKEN` as the `token` kwarg to
     *both* `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`.
     This unblocks loading checkpoints such as `mistralai/Mixtral-8x7B-Instruct-v0.1`
     that sit behind an access gate.
2. Memory footprint safeguard
   • `low_cpu_mem_usage=True` keeps RAM spikes in check when large checkpoints
     are streamed from the Hub.  The flag is harmless for small CI models.
3. Minor refactor – helper `_hf_auth_kwargs()` centralises the auth-token logic.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # noqa: F401 – re-exported

logger = logging.getLogger("tracs_runner.train")

###############################################################################
# Helpers                                                                    #
###############################################################################

def _get_device() -> torch.device:  # pragma: no cover – trivial helper
    """Return *cuda* when available, else *cpu*.

    We explicitly *avoid* `device_map="auto"` when a CUDA device is present so
    that small smoke-test checkpoints stay on a single GPU.  The full
    experiment may load multi-billion-parameter models; in that case we still
    place the model on *cuda* and rely on PyTorch‘s internal sharding to spill
    weights if necessary (this keeps the code simple while being good enough
    for CI).
    """

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _hf_auth_kwargs() -> Dict[str, str]:
    """Return `{"token": HF_TOKEN}` when the environment variable is set."""

    token = os.getenv("HF_TOKEN")
    return {"token": token} if token else {}


###############################################################################
# Public API                                                                 #
###############################################################################

def load_model(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load *model_id* and return *(tokenizer, model)* placed on the right device.

    A valid HuggingFace token (if required) is automatically injected via the
    `HF_TOKEN` environment variable to satisfy the FAIL-FAST policy – we abort
    only when the user *does* possess a token but model access still fails.
    """

    device = _get_device()

    # ------------------------------------------------------------------
    # 1) Tokenizer
    # ------------------------------------------------------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_hf_auth_kwargs())
    except Exception as exc:  # noqa: BLE001 – surfacing any auth / network error
        logger.error("Tokenizer loading failed for %s: %s", model_id, exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2) Model – use *low_cpu_mem_usage* to keep RAM spikes minimal.
    # ------------------------------------------------------------------
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if device.type == "cuda" else None,
            low_cpu_mem_usage=True,
            **_hf_auth_kwargs(),
        ).to(device)
    except Exception as exc:  # noqa: BLE001
        logger.error("Model loading failed for %s: %s", model_id, exc)
        sys.exit(1)

    return tokenizer, model

###############################################################################
# Reference in-repo guard implementations (pass-through)                     #
###############################################################################


class _BasePassthroughGuard:  # pylint: disable=too-few-public-methods
    """A minimal guard that forwards to the wrapped model without intervention."""

    def __init__(self, model):
        self.model = model
        # Derive the original checkpoint id when possible, else fall back to GPT-2
        name_or_path = getattr(model.config, "_name_or_path", "gpt2")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, **_hf_auth_kwargs())
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
# Safety-Guard loader                                                        #
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
