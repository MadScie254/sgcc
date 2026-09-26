from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]

# Settings from a local .env (DATABASE_URL, API_KEYS, ...; see .env.example and
# scripts/setup_database.py). Variables already set in the environment win.
load_dotenv(BASE_DIR / ".env", override=False)


@lru_cache(maxsize=1)
def get_config() -> Dict[str, Any]:
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=1)
def get_paths() -> Dict[str, Path]:
    """Absolute paths the API reads, and the local state directory (never committed)."""
    config = get_config()
    paths = config["paths"]
    return {
        "model_file": BASE_DIR / paths["model_file"],
        "artifacts": BASE_DIR / paths["artifacts"],
        "baselines": BASE_DIR / paths["models"] / "baselines" / "comparison_results.json",
        "serving_data": BASE_DIR / config["data"]["serving_data_path"],
        # SQLite database and local blobs when DATABASE_URL / S3_BUCKET are not set.
        "state": BASE_DIR / os.getenv("SGCC_STATE_DIR", "artifacts/state"),
    }


def environment() -> str:
    return os.getenv("ENV", "development").lower()


def scoring_interval_minutes() -> float:
    """Minutes between scheduled scoring runs; 0 disables the schedule."""
    try:
        return max(float(os.getenv("SCORING_INTERVAL_MINUTES", "0")), 0.0)
    except ValueError:
        return 0.0
