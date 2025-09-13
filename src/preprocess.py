# src/preprocess.py
"""Data loading, preprocessing and Hugging-Face Hub download helpers."""
import json
import shutil
from pathlib import Path
from typing import Dict

from huggingface_hub import hf_hub_download, snapshot_download

# -------------------------------------------------------------------
#  directory management
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESEARCH_DIR = ROOT / ".research" / "iteration3"  # updated as per spec
IMAGES_DIR = RESEARCH_DIR / "images"

for _d in (DATA_DIR, IMAGES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

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
    present locally.  The function now supports three URI schemes:

    1. ``synthetic://<name>`` – a tiny dataset generated on-the-fly for CI.
    2. ``file://<absolute-or-relative-path>`` – an already present folder.
    3. Any other string → treated as HuggingFace Hub repo ID.
    """
    info = cfg["datasets"][repo_key]
    repo = info["repo"]

    # -------------------------------------------------------------
    # 1) synthetic datasets (used for smoke tests)
    # -------------------------------------------------------------
    if repo.startswith("synthetic://"):
        name = repo[len("synthetic://") :]
        return _create_synthetic_dataset(name)

    # -------------------------------------------------------------
    # 2) explicit file path
    # -------------------------------------------------------------
    if repo.startswith("file://"):
        path = Path(repo[len("file://") :]).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"Local dataset path '{path}' not found.")
        return path

    # -------------------------------------------------------------
    # 3) regular HF download (default case)
    # -------------------------------------------------------------
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
        import json as _json

        self.samples = [_json.loads(line)["text"] for line in open(jsonl_path, "r", encoding="utf-8")]
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
