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


def load_model(repo: str, hf_token: Union[str, None] = None, dtype: torch.dtype = torch.float16):
    """Download (if necessary) and load a causal-LM checkpoint.

    *   If CUDA is available we delegate device placement to *transformers* by
        passing ``device_map="auto"`` (this uses *accelerate* under the hood).
    *   If CUDA is **not** available we load the weights on CPU **without** a
        ``device_map`` argument – this removes the hard dependency on the
        *accelerate* package for CPU-only CI environments.
    *   FP16 tensors are only supported on GPU.  When running on CPU we
        automatically promote the dtype to ``torch.float32``.
    """

    has_cuda = torch.cuda.is_available()

    # ---------------------------------------------------------------------
    # ensure dtype is supported on the target device
    # ---------------------------------------------------------------------
    if not has_cuda and dtype == torch.float16:
        dtype = torch.float32

    # ---------------------------------------------------------------------
    # resolve synthetic:// URIs – used by smoke tests to load tiny local ckpts
    # ---------------------------------------------------------------------
    if repo.startswith("synthetic://"):
        local_dir = Path("data") / repo[len("synthetic://") :].replace("/", "__")
        if not local_dir.exists():
            raise RuntimeError(
                f"Synthetic model directory '{local_dir}' not found. Did the smoke-test assets get generated?"
            )
        repo = str(local_dir)

    # ---------------------------------------------------------------------
    # build kwargs for `from_pretrained`
    # ---------------------------------------------------------------------
    kwargs = {
        "torch_dtype": dtype,  # `torch_dtype` is still accepted by >=4.56
        "token": hf_token,
    }
    if has_cuda:
        kwargs["device_map"] = "auto"  # requires *accelerate*

    try:
        model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
        # When running on CPU, explicitly move the model even if the checkpoint
        # carries CUDA tensors (edge-case for tiny synthetic models saved on GPU)
        if not has_cuda:
            model.to(torch.device("cpu"))
        model.eval()
        return model
    except Exception as e:  # pragma: no cover – fail fast, propagate reason
        raise RuntimeError(f"Cannot load model '{repo}': {e}") from e
