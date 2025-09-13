# src/train.py
"""Model-related helpers.
Currently our experiments do **not** fine-tune models; they only have to
load pre-trained checkpoints from the Hugging Face Hub.  All model
handling code therefore lives here so that the evaluation module can
re-use it easily.
"""
from pathlib import Path
from typing import Union

import torch
from transformers import AutoModelForCausalLM

__all__ = ["load_model"]

# NOTE: Starting with *transformers* 4.56 the preferred keyword for the
# expected tensor dtype in `from_pretrained` was renamed from
# `torch_dtype` → `dtype`.  To stay backward-compatible with older
# releases we populate *both* keys – redundant entries are ignored by
# the underlying implementation, so this is safe across versions.

def _build_dtype_kwargs(dtype: torch.dtype):
    """Return a kwargs dict that sets the model dtype in a
    version-agnostic way (supports both `torch_dtype` and the new
    `dtype` argument introduced around v4.56)."""

    return {"torch_dtype": dtype, "dtype": dtype}


def load_model(repo: str, hf_token: Union[str, None] = None, dtype: torch.dtype = torch.float16):
    """Download (if necessary) and load a causal-LM checkpoint.

    *   If CUDA is available we delegate device placement to *transformers* by
        passing ``device_map=\"auto\"`` (which internally relies on *accelerate*).
    *   If CUDA is **not** available we load the weights on CPU **without** a
        ``device_map`` argument – this removes the hard dependency on the
        *accelerate* package for CPU-only CI environments.
    *   FP16 tensors are only supported on GPU.  When running on CPU we
        automatically promote the dtype to ``torch.float32``.
    """

    has_cuda = torch.cuda.is_available()

    # ------------------------------------------------------------------
    # ensure dtype is supported on the target device
    # ------------------------------------------------------------------
    if not has_cuda and dtype == torch.float16:
        dtype = torch.float32

    # ------------------------------------------------------------------
    # resolve synthetic:// URIs – used by smoke tests to load tiny local ckpts
    # ------------------------------------------------------------------
    if repo.startswith("synthetic://"):
        local_dir = Path("data") / repo[len("synthetic://") :].replace("/", "__")
        if not local_dir.exists():
            raise RuntimeError(
                f"Synthetic model directory '{local_dir}' not found. Did the smoke-test assets get generated?"
            )
        repo = str(local_dir)

    # ------------------------------------------------------------------
    # build kwargs for `from_pretrained`
    # ------------------------------------------------------------------
    kwargs = _build_dtype_kwargs(dtype)
    kwargs["token"] = hf_token  # auth token may be *None*

    if has_cuda:
        # Requires the *accelerate* dependency which is already part of
        # our `pyproject.toml`.  On a CPU-only runner we purposefully do
        # *not* pass this argument to avoid the import.
        kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
        # On CPU we explicitly move the model even if the checkpoint was
        # saved with CUDA tensors (edge-case for tiny synthetic models).
        if not has_cuda:
            model.to(torch.device("cpu"))
        model.eval()
        return model
    except Exception as e:  # pragma: no cover – fail fast, propagate reason
        raise RuntimeError(f"Cannot load model '{repo}': {e}") from e
