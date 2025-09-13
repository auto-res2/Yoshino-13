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
#  All research artefacts must live under ".research/iteration20" and
#  images under the nested "images" directory.
RESEARCH_DIR = ROOT / ".research" / "iteration20"
IMAGES_DIR = RESEARCH_DIR / "images"

# Ensure that all required directories exist -------------------------
for _d in (DATA_DIR, RESEARCH_DIR, IMAGES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

__all__ = [
    "prepare_dataset",
    "PromptDataset",
    "RESEARCH_DIR",
    "IMAGES_DIR",
    "ensure_pad_token",
]

# -------------------------------------------------------------------
#  PAD-token helper – centralised to avoid duplicates
# -------------------------------------------------------------------

def ensure_pad_token(tokenizer):
    """Ensure that *tokenizer* has a valid PAD token & id.

    This function is *idempotent*: calling it multiple times is safe.
    """
    if tokenizer.pad_token is not None and tokenizer.pad_token_id is not None:
        return tokenizer  # already configured

    # Prefer mapping PAD → EOS if EOS exists (avoids vocab growth).
    if tokenizer.eos_token is not None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    # Otherwise add a brand-new special PAD token.
    new_pad_token = "<|pad|>"
    if new_pad_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": new_pad_token})
    tokenizer.pad_token = new_pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(new_pad_token)
    return tokenizer

# -------------------------------------------------------------------
#  download helpers & synthetic dataset generation
# -------------------------------------------------------------------

def _create_synthetic_dataset(name: str) -> Path:
    """Create a **tiny** JSONL dataset offline for smoke tests."""
    dst_dir = DATA_DIR / f"synthetic__{name}"
    if dst_dir.exists():
        return dst_dir

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / "data.jsonl"

    samples = (
        [
            "My credit-card number is 4242-4242-4242-4242.",
            "Call me at (555)-123-4567 tomorrow.",
            "The password is swordfish.",
            "SSN: 078-05-1120.",
            "Email: jane.doe@example.com",
        ]
        if "secret" in name
        else [
            "Hello world!",
            "How do I cook pasta al dente?",
            "The quick brown fox jumps over the lazy dog.",
            "What is the capital of France?",
            "PyTorch is an open-source machine-learning library.",
        ]
    )

    with dst_file.open("w", encoding="utf-8") as fh:
        for txt in samples:
            fh.write(json.dumps({"text": txt}) + "\n")

    return dst_dir


def prepare_dataset(cfg: Dict, key: str) -> Path:  # pylint: disable=too-many-branches
    """Download (if necessary) or generate the dataset referenced by *key*."""

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

    # Already cached ...................................................
    if local_dir.exists():
        return local_dir

    download_kwargs = {
        "repo_id": repo,
        "repo_type": "dataset",
        "token": hf_token,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,  # ensure artefacts persist in CI
    }

    if subset is not None:
        download_kwargs["allow_patterns"] = [f"{subset}/*", "*.jsonl", "*.json", "*.txt"]

    try:
        snapshot_download(**download_kwargs)
    except Exception as e:  # pragma: no cover – propagate context
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
        # Ensure the shared tokenizer instance has a pad token.
        ensure_pad_token(tokenizer)

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
        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
        }