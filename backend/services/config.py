from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


BASE_DIR = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_config(config_path: str | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else BASE_DIR / "config.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=1)
def get_project_paths() -> Dict[str, Path]:
    config = get_config()
    paths = config.get("paths", {})
    return {
        "models": BASE_DIR / paths.get("models", "models"),
        "artifacts": BASE_DIR / paths.get("artifacts", "artifacts"),
        "logs": BASE_DIR / paths.get("logs", "logs"),
        "uploads": BASE_DIR / paths.get("uploads", "data/uploads"),
        "reports": BASE_DIR / paths.get("reports", "artifacts/reports"),
        "data": BASE_DIR / "data",
        # Runtime state (cases, pipeline runs, operating threshold); not committed.
        "state": BASE_DIR / os.getenv("SGCC_STATE_DIR", "artifacts/state"),
    }
