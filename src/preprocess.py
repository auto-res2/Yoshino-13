"""src/preprocess.py
Dataset acquisition & caching utilities.  Downloads are stored under ./data
and re-used on subsequent runs.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset, DatasetDict

logger = logging.getLogger("tracs_runner.preprocess")

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
# Dataset helper
###############################################################################

def _load_local_jsonl(path: Path) -> DatasetDict:
    """Helper – load a local JSONL file via 🤗 Datasets."""
    if not path.exists():
        logger.error("Local dataset %s not found", path)
        sys.exit(1)
    return load_dataset("json", data_files=str(path))


def ensure_dataset(name: str, spec: Dict[str, Any]) -> DatasetDict:
    """Download (if required) and return a 🤗 *DatasetDict* for *name*.

    Aborts the run with exit-code 1 on any error – signalling CI failure.
    """

    ds_path = DATA_DIR / name
    if ds_path.exists():
        # Cached – prefer local copy irrespective of *spec* to guarantee
        # reproducibility and avoid flaky network.
        logger.info("Using cached dataset '%s'", name)
        local_file = ds_path / "data.jsonl"
        return _load_local_jsonl(local_file)

    try:
        ds_path.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # 1) HuggingFace Hub repository
        # ------------------------------------------------------------------
        if spec.get("type") == "hf":
            repo_id = spec["repo"]
            token = os.getenv("HF_TOKEN")
            return load_dataset(repo_id, token=token)

        # ------------------------------------------------------------------
        # 2) Remote JSONL file (HTTP/HTTPS)
        # ------------------------------------------------------------------
        if spec.get("type") == "jsonl":
            url = spec["url"]
            import requests

            local_file = ds_path / "data.jsonl"
            logger.info("Downloading %s → %s", url, local_file)
            resp = requests.get(url, timeout=120)
            if resp.status_code != 200:
                logger.error("Failed to download %s (status %s)", url, resp.status_code)
                sys.exit(1)
            with open(local_file, "wb") as f:
                f.write(resp.content)
            return _load_local_jsonl(local_file)

        # ------------------------------------------------------------------
        # 3) Local path already supplied – zero network I/O
        # ------------------------------------------------------------------
        if spec.get("type") == "local":
            local_file = Path(spec["path"])
            return _load_local_jsonl(local_file)

    except Exception as e:
        logger.error("Dataset acquisition failed for %s: %s", name, e)
        sys.exit(1)

    logger.error("Unknown dataset spec for %s", name)
    sys.exit(1)