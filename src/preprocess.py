# src/preprocess.py
from pathlib import Path
import json
from typing import Dict, Optional

from huggingface_hub import snapshot_download

# -------------------------------------------------------------------
#  directory management
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

#  Updated iteration folder as required by spec ----------------------
RESEARCH_DIR = ROOT / ".research" / "iteration16"  # <<<< updated to iteration16 >>>>
IMAGES_DIR = RESEARCH_DIR / "images"

# Ensure that all required directories exist -------------------------
for _d in (DATA_DIR, RESEARCH_DIR, IMAGES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

__all__ = [
    "prepare_dataset",
    "PromptDataset",
    "RESEARCH_DIR",
    "IMAGES_DIR",
]

# -------------------------------------------------------------------
#  download helpers & synthetic dataset generation
# -------------------------------------------------------------------

def _create_synthetic_dataset(name: str) -> Path:
    """Create a **tiny** JSONL dataset on-the-fly so that smoke tests can
    run entirely offline.  The directory layout mirrors what
    `snapshot_download` would produce so that caller code does not have
    to care about the provenance.
    """

    dst_dir = DATA_DIR / f"synthetic__{name}"
    if dst_dir.exists():
        return dst_dir

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / "data.jsonl"

    # A handful of example texts – enough to exercise the pipeline
    if "secret" in name:
        samples = [
            "My credit-card number is 4242-4242-4242-4242.",
            "Call me at (555)-123-4567 tomorrow.",
            "The password is swordfish.",
            "SSN: 078-05-1120.",
            "Email: jane.doe@example.com",
        ]
    else:
        samples = [
            "Hello world!",
            "How do I cook pasta al dente?",
            "The quick brown fox jumps over the lazy dog.",
            "What is the capital of France?",
            "PyTorch is an open-source machine-learning library.",
        ]

    with dst_file.open("w", encoding="utf-8") as fh:
        for txt in samples:
            fh.write(json.dumps({"text": txt}) + "\n")

    return dst_dir


def prepare_dataset(cfg: Dict, key: str) -> Path:  # pylint: disable=too-many-branches
    """Download (if necessary) a dataset described in the YAML configuration.

    Parameters
    ----------
    cfg : Dict
        Full experiment configuration dictionary.
    key : str
        Key under ``cfg["datasets"]`` which holds the dataset description.

    Returns
    -------
    Path
        Local directory that contains the dataset files (mirrors
        *huggingface_hub* snapshot layout).
    """

    if "datasets" not in cfg or key not in cfg["datasets"]:
        raise KeyError(f"Dataset entry '{key}' missing from configuration.")

    ds_cfg: Dict = cfg["datasets"][key]
    repo: str = ds_cfg["repo"]
    subset: Optional[str] = ds_cfg.get("subset")
    hf_token: Optional[str] = cfg.get("_hf_token")

    # ------------------------------------------------------------------
    # Synthetic / offline datasets -------------------------------------
    # ------------------------------------------------------------------
    if repo.startswith("synthetic://"):
        synthetic_name = repo[len("synthetic://") :]
        return _create_synthetic_dataset(synthetic_name)

    # ------------------------------------------------------------------
    # HuggingFace Hub datasets -----------------------------------------
    # ------------------------------------------------------------------
    local_dir = DATA_DIR / repo.replace("/", "__")

    # If we already have the dataset locally, simply return the path.
    if local_dir.exists():
        return local_dir

    # Otherwise download (requires network and/or cache).
    download_kwargs = {
        "repo_id": repo,
        "repo_type": "dataset",
        "token": hf_token,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,  # ensure CI artifact persists
        "allow_patterns": None,
    }
    if subset is not None:
        download_kwargs["allow_patterns"] = [f"{subset}/*", "*.jsonl", "*.json", "*.txt"]

    try:
        snapshot_download(**download_kwargs)
    except Exception as e:  # pragma: no cover – propagate with context
        auth_hint = " – did you set the HF_TOKEN environment variable?" if hf_token is None else ""
        raise RuntimeError(f"Failed to download dataset '{repo}': {e}{auth_hint}") from e

    return local_dir


# -------------------------------------------------------------------
#  Dataset wrapper
# -------------------------------------------------------------------

from torch.utils.data import Dataset  # pylint: disable=wrong-import-position
from transformers import AutoTokenizer  # pylint: disable=wrong-import-position


class PromptDataset(Dataset):
    """A minimal JSONL prompt dataset with a `text` field."""

    def __init__(self, jsonl_path: Path, tokenizer: AutoTokenizer, max_tokens: int = 512):
        # ------------------------------------------------------------------
        # Ensure the tokenizer has a valid PAD token.  Some causal LMs (GPT-2,
        # LLaMA) are trained without one which breaks `padding=...` in the
        # encoding call unless we fix it here.
        # ------------------------------------------------------------------
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        self.samples = [json.loads(line)["text"] for line in open(jsonl_path, "r", encoding="utf-8")]
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        txt = self.samples[idx]
        tok = self.tokenizer(
            txt,
            truncation=True,
            max_length=self.max_tokens,
            padding="max_length",
            return_tensors="pt",
        )
        # Returned dict intentionally omits raw text – strings break default_collate.
        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
        }
