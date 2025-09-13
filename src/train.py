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
    """Download (if necessary) and load a causal-LM checkpoint onto the
    first available CUDA device (or CPU if CUDA is not available).
    A thin wrapper so that all model creation logic is centralised
    inside *train.py*.
    """
    # IMPORTANT: allocate on CPU by default – many CI runners do not expose GPUs
    device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}

    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=dtype,
            device_map=device_map,
            token=hf_token,
        )
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Cannot load model '{repo}': {e}")
