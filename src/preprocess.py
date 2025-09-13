# src/preprocess.py
"""Data loading, preprocessing and Hugging-Face Hub download helpers."""
import shutil
from pathlib import Path
from typing import Dict

from huggingface_hub import hf_hub_download, snapshot_download

# -------------------------------------------------------------------
#  directory management
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESEARCH_DIR = ROOT / ".research" / "iteration2"  # updated as per spec
IMAGES_DIR = RESEARCH_DIR / "images"

for _d in (DATA_DIR, IMAGES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
#  download helpers
# -------------------------------------------------------------------


def _hf_download(repo_id: str, filename: str | None = None, subfolder: str | None = None, token: str | None = None) -> Path:
    """Download *either* a single file (via `hf_hub_download`) *or* an entire
    repository snapshot (via `snapshot_download`).  Any exception is
    converted into a `RuntimeError` so that callers fail fast and do not
    silently proceed with partial or missing data.
    """
    try:
        if filename is None:
            # full repo download
            path = snapshot_download(repo_id=repo_id, token=token, cache_dir=str(DATA_DIR))
            return Path(path)
        # single file
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            token=token,
            cache_dir=str(DATA_DIR),
        )
        return Path(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download {repo_id}/{filename or ''}: {e}") from e


def prepare_dataset(cfg: Dict, repo_key: str) -> Path:
    """Ensure that the dataset referenced by *repo_key* in the YAML config is
    present locally.  The complete repository snapshot is copied into an
    organised folder below *data/* so that downstream code never needs to
    re-query the Hub.
    """
    info = cfg["datasets"][repo_key]
    repo = info["repo"]

    local_dir = DATA_DIR / repo.replace("/", "__")
    if local_dir.exists():
        return local_dir

    # populate HF cache (either returns directory or file path depending on arguments)
    src_path = _hf_download(repo_id=repo, token=cfg.get("_hf_token"))

    # `src_path` may be a directory (snapshot_download) or a file path
    src_folder = src_path if src_path.is_dir() else src_path.parent
    shutil.copytree(src_folder, local_dir)
    return local_dir


# -------------------------------------------------------------------
#  dataset classes
# -------------------------------------------------------------------

from torch.utils.data import Dataset  # pylint: disable=wrong-import-position
from transformers import AutoTokenizer  # pylint: disable=wrong-import-position


class PromptDataset(Dataset):
    """A minimal JSONL prompt dataset with a `text` field."""

    def __init__(self, jsonl_path: Path, tokenizer: AutoTokenizer, max_tokens: int = 512):
        import json

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
            "raw_text": txt,
        }
