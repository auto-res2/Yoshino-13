"""src/preprocess.py
Dataset acquisition & caching utilities.  Downloads are stored under ./data and
re-used on subsequent runs.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger("tracs_runner.preprocess")

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
# Internal helpers                                                             #
###############################################################################

def _load_local_jsonl(path: Path) -> DatasetDict:
    if not path.exists():
        logger.error("Local dataset %s not found", path)
        sys.exit(1)
    return load_dataset("json", data_files=str(path))


def _write_jsonl(file_path: Path, rows: List[Dict[str, Any]]):
    with open(file_path, "w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

###############################################################################
# Public API                                                                   #
###############################################################################

def ensure_dataset(name: str, spec: Dict[str, Any]) -> DatasetDict:
    """Download (if required) and return a 🤗 *DatasetDict* for *name*."""

    ds_path = DATA_DIR / name
    cached_file = ds_path / "data.jsonl"
    if cached_file.exists():
        logger.info("Using cached dataset '%s'", name)
        return _load_local_jsonl(cached_file)

    ds_path.mkdir(parents=True, exist_ok=True)

    try:
        if spec.get("type") == "hf":
            repo_id = spec["repo"]
            token = os.getenv("HF_TOKEN")
            return load_dataset(repo_id, token=token)

        if spec.get("type") == "jsonl":
            url = spec["url"]
            import requests

            logger.info("Downloading %s → %s", url, cached_file)
            resp = requests.get(url, timeout=120)
            if resp.status_code != 200:
                logger.error("Failed to download %s (status %s)", url, resp.status_code)
                sys.exit(1)
            cached_file.write_bytes(resp.content)
            return _load_local_jsonl(cached_file)

        if spec.get("type") == "local":
            local_file = Path(spec["path"])
            return _load_local_jsonl(local_file)

        if spec.get("type") == "inline":
            rows: List[Dict[str, Any]] = spec.get("data", [])
            if not rows:
                logger.error("Inline dataset for '%s' is empty", name)
                sys.exit(1)
            _write_jsonl(cached_file, rows)
            ds = Dataset.from_list(rows)
            return DatasetDict({"train": ds})

    except Exception as e:  # noqa: BLE001 – catch *anything* and abort
        logger.error("Dataset acquisition failed for %s: %s", name, e)
        sys.exit(1)

    logger.error("Unknown dataset spec for %s", name)
    sys.exit(1)
