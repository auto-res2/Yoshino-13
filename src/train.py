"""src/train.py
Utility functions for loading language models and attaching the optional
safety-guard stacks (CAI, HiLMAS, TRACS).

Revision (iteration-7)
----------------------
1.  Gated-repo awareness
    • Introduce `GatedRepoAccessError` – a custom exception signalling that a
      checkpoint requires authentication which is currently not available via
      the `HF_TOKEN` environment variable.  Down-stream callers can *catch &
      skip* such models while still failing fast for every other error.
2.  Token forwarding
    • Continue to forward `token=<HF_TOKEN>` to both *tokenizer* and *model* –
      unchanged from the previous revision.
3.  Minor: renamed deprecated `torch_dtype` → `dtype` to silence warnings on
      the latest `transformers` release.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # noqa: F401 – re-exported

logger = logging.getLogger("tracs_runner.train")

__all__ = [
    "load_model",
    "load_guard",
    "GenerationConfig",  # re-export so callers can avoid importing *transformers*
    "GatedRepoAccessError",
]

###############################################################################
# Exceptions                                                                  #
###############################################################################


class GatedRepoAccessError(RuntimeError):
    """Raised when a HuggingFace repo is gated *and* no valid token is present."""

    def __init__(self, model_id: str):
        super().__init__(
            f"Model '{model_id}' requires gated-repo access but no HF_TOKEN is set."
        )
        self.model_id = model_id


###############################################################################
# Helpers                                                                     #
###############################################################################


def _get_device() -> torch.device:  # pragma: no cover – trivial helper
    """Return *cuda* when available, else *cpu*."""

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _hf_auth_kwargs() -> Dict[str, str]:
    """Return `{"token": HF_TOKEN}` when the environment variable is set."""

    token = os.getenv("HF_TOKEN")
    return {"token": token} if token else {}


###############################################################################
# Public API                                                                  #
###############################################################################


def _maybe_raise_gated(exc: Exception, model_id: str):
    """Detect a gated-repo auth error and raise *GatedRepoAccessError* instead."""

    msg = str(exc).lower()
    if "gated repo" in msg or "access is restricted" in msg:
        # The repo is gated – only re-raise as *GatedRepoAccessError* when the
        # user did *not* supply a token; otherwise propagate the original error
        # so we still fail-fast on invalid credentials.
        if os.getenv("HF_TOKEN") is None:
            raise GatedRepoAccessError(model_id) from exc
    # Fallback – propagate the original exception → triggers fail-fast.
    raise exc


def load_model(model_id: str):  # -> Tuple[AutoTokenizer, AutoModelForCausalLM]
    """Load *model_id* and return *(tokenizer, model)* on the correct device.

    When the repository is gated **and** no access token is available we raise
    *GatedRepoAccessError* which callers may catch to *skip* the model.
    """

    device = _get_device()

    # ------------------------------------------------------------------
    # 1) Tokenizer
    # ------------------------------------------------------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_hf_auth_kwargs())
    except Exception as exc:  # noqa: BLE001 – surfacing any auth / network error
        _maybe_raise_gated(exc, model_id)
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
        _maybe_raise_gated(exc, model_id)
        logger.error("Model loading failed for %s: %s", model_id, exc)
        sys.exit(1)

    return tokenizer, model


###############################################################################
# Reference pass-through safety-guard implementations                         #
###############################################################################


class _BasePassthroughGuard:  # pylint: disable=too-few-public-methods
    """A minimal guard that forwards to the wrapped model without intervention."""

    def __init__(self, model):
        self.model = model
        name_or_path = getattr(model.config, "_name_or_path", "gpt2")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, **_hf_auth_kwargs())
        except Exception:  # pragma: no cover – tiny toy checkpoints may fail
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Public API expected by `evaluate.run_experiment_1`
    # ------------------------------------------------------------------
    def generate(self, prompt: str, generation_config: GenerationConfig | None = None, **kwargs):
        """Pass-through generation – *violation* flag is always *False*."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, generation_config=generation_config)
        return output_ids, {"violation": False}

    # Delegate attribute access
    def __getattr__(self, item):  # noqa: D401
        return getattr(self.model, item)


class CaiSafetyGuard(_BasePassthroughGuard):
    """Stub for an Anthropic-style Constitutional AI filter (pass-through)."""


class HiLMASGuard(_BasePassthroughGuard):
    """Stub for the fixed-horizon HiLMAS verifier (pass-through)."""


class TracsGuard(_BasePassthroughGuard):
    """Stub for the proposed TRACS adaptive-horizon verifier (pass-through)."""


###############################################################################
# Safety-Guard loader                                                         #
###############################################################################

def load_guard(stack_name: str, model):
    """Return *model* wrapped in the requested safety-stack implementation."""

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
