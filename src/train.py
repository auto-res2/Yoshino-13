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
        # In smoke-test mode we often pass a *synthetic* identifier that
        # refers to a **local** tiny model stored inside the repository
        # (e.g. ``synthetic://tiny-gpt2``).  Such identifiers are NOT
        # hosted on the HF Hub and therefore have to be resolved via the
        # filesystem.  We canonicalise those URIs here so that down-stream
        # code never has to special-case them.
        if repo.startswith("synthetic://"):
            local_dir = Path("data") / repo[len("synthetic://") :].replace("/", "__")
            if not local_dir.exists():
                raise RuntimeError(
                    f"Synthetic model directory '{local_dir}' not found. Did the smoke-test assets get generated?"
                )
            repo = str(local_dir)

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
