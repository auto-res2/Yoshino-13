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

def ensure_dataset(name: str, spec: Dict[str, Any]) -> DatasetDict:
    """Download (if required) and return a 🤗 *DatasetDict* for *name*.

    Aborts the run with exit-code 1 on any error – signalling CI failure.
    """

    ds_path = DATA_DIR / name
    if ds_path.exists():
        logger.info("Using cached dataset '%s'", name)

    try:
        if spec.get("type") == "hf":
            repo_id = spec["repo"]
            token = os.getenv("HF_TOKEN")
            return load_dataset(repo_id, token=token)

        if spec.get("type") == "jsonl":
            url = spec["url"]
            local_file = ds_path / "data.jsonl"
            if not local_file.exists():
                import requests

                logger.info("Downloading %s → %s", url, local_file)
                resp = requests.get(url, timeout=120)
                if resp.status_code != 200:
                    logger.error("Failed to download %s (status %s)", url, resp.status_code)
                    sys.exit(1)
                ds_path.mkdir(parents=True, exist_ok=True)
                with open(local_file, "wb") as f:
                    f.write(resp.content)
            return load_dataset("json", data_files=str(local_file))
    except Exception as e:
        logger.error("Dataset acquisition failed for %s: %s", name, e)
        sys.exit(1)

    logger.error("Unknown dataset spec for %s", name)
    sys.exit(1)
