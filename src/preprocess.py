"""preprocess.py – data acquisition & DataLoader construction.
Exact logic copied from the original src/data.py with only
minor path / typing tweaks.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
from typing import Dict

import requests
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchaudio import datasets as aud_datasets, transforms as aud_transforms
from datasets import load_dataset

# -----------------------------------------------------------------------------
#  Helper – deterministic SHA-256 for downloaded files (integrity check)
# -----------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
#  Generic HTTP(S) downloader – left here for completeness (unused by loaders)
# -----------------------------------------------------------------------------

def download_url(url: str, target: Path, expected_sha256: str | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if expected_sha256 and _sha256(target) == expected_sha256:
            return  # already verified
        target.unlink()
    print(f"Downloading {url} → {target}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    if expected_sha256 and _sha256(target) != expected_sha256:
        raise RuntimeError(f"Hash mismatch for {url}")


# -----------------------------------------------------------------------------
#  LiveBench / SeasonBench streaming datasets via 🤗datasets
# -----------------------------------------------------------------------------
_LIVEBENCH_MAP = {
    "livebench-image-23": "livebench/livebench-image-23",
    "livebench-text-12": "livebench/livebench-text-12",
    "livebench-audio-pilot-8": "livebench/livebench-audio-pilot-8",
    "seasonbench": "oracle-research/seasonbench",
}


def load_livebench(name: str, split: str = "train") -> Dataset:
    if name not in _LIVEBENCH_MAP:
        raise RuntimeError(f"Unknown LiveBench stream {name}")
    repo_id = _LIVEBENCH_MAP[name]
    try:
        ds = load_dataset(repo_id, split=split, use_auth_token=os.getenv("HF_TOKEN"))
    except Exception as e:
        raise RuntimeError(
            f"Could not load required dataset '{repo_id}'. Original error: {e}.\n"
            "Per STRICT NO-FALLBACK rule we abort."
        )
    return ds


# -----------------------------------------------------------------------------
#  Public mini-datasets for smoke testing (CIFAR-10, AG-News, SpeechCommands)
# -----------------------------------------------------------------------------

class SmokeImage(Dataset):
    def __init__(self, train: bool = True):
        tfm = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        self.ds = datasets.CIFAR10("data/cifar10", train=train, download=True, transform=tfm)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return {"pixel_values": x, "labels": y}


class SmokeText(Dataset):
    def __init__(self, split: str):
        ag_news = load_dataset("ag_news", split=split)
        self.texts = ag_news["text"][:2000]  # cap for quick run
        self.labels = ag_news["label"][:2000]
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained("t5-base")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": torch.tensor(self.labels[idx]),
        }


class SmokeAudio(Dataset):
    def __init__(self, subset: str = "training"):
        self.ds = aud_datasets.SPEECHCOMMANDS(
            "data/speech_cmd", subset=subset, download=True
        )
        self.mel = aud_transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
        # simple label mapping (folder name → int)
        self.label2idx = {
            label: i for i, label in enumerate(sorted(list(set(self.ds._walker))))
        }

    def __len__(self):
        return 256  # restrict for smoke test speed

    def __getitem__(self, idx):
        waveform, sr, label, *_ = self.ds[idx]
        mel = self.mel(waveform).squeeze(0)
        y = self.label2idx[label]
        return {"mel": mel, "labels": torch.tensor(y)}


# -----------------------------------------------------------------------------
#  Public factory used by trainers
# -----------------------------------------------------------------------------

def build_dataloader(cfg: Dict, split: str = "train") -> DataLoader:
    """Return torch.utils.data.DataLoader according to cfg.dataset section."""
    name = cfg["name"].lower()
    batch_size = cfg.get("batch_size", 32)

    # HuggingFace streaming datasets ----------------------------------
    if name.startswith("livebench") or name == "seasonbench":
        ds = load_livebench(name, split=split)
        ds.set_format(type="torch")  # HF → torch tensors
        return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))

    # smoke test datasets ---------------------------------------------
    if name == "cifar10":
        ds = SmokeImage(train=(split == "train"))
    elif name == "ag_news":
        ds = SmokeText(split="train" if split == "train" else "test")
    elif name == "speechcommands":
        ds = SmokeAudio(subset="training" if split == "train" else "testing")
    else:
        raise RuntimeError(f"Unsupported dataset {name}")

    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
