"""Shared pytest fixtures."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATASET = REPO_ROOT / "data" / "sgcc_demo.csv.gz"

# Keep everything the API writes (cases, runs, uploads, reports) out of the repo, and
# out of the working database: tests use TEST_DATABASE_URL, or SQLite in a temporary
# folder. The values are set before the backend loads .env, which never overrides them.
os.environ.setdefault("SGCC_STATE_DIR", tempfile.mkdtemp(prefix="sgcc-state-"))
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL") or dotenv_values(REPO_ROOT / ".env").get("TEST_DATABASE_URL") or ""
os.environ["ENV"] = "development"
os.environ["API_KEYS"] = ""
os.environ["S3_BUCKET"] = ""


@pytest.fixture(scope="session")
def demo_dataset_path() -> Path:
    """The unlabelled operational population the API scores (committed to the repo)."""
    assert DEMO_DATASET.is_file(), "data/sgcc_demo.csv.gz is missing; run python -m src.train"
    return DEMO_DATASET


def sgcc_frame(days: int = 120) -> pd.DataFrame:
    """
    A small frame in the raw SGCC layout: CONS_NO, FLAG, then date columns in
    lexicographic (not chronological) order, as in the public dump.

    A consumes day-of-series kWh (strictly increasing); B is constant until a
    mid-series drop to zero; C has a long gap.
    """
    dates = pd.date_range("2014-01-01", periods=days, freq="D")
    half, third = days // 2, days // 3
    values = {
        "A": np.arange(1, days + 1, dtype=float),
        "B": np.r_[np.full(half, 10.0), np.zeros(days - half)],
        "C": np.r_[np.full(third, 5.0), np.full(third, np.nan), np.full(days - 2 * third, 5.0)],
    }
    frame = pd.DataFrame(values, index=[f"{d.year}/{d.month}/{d.day}" for d in dates]).T
    frame = frame[sorted(frame.columns)]
    frame.insert(0, "FLAG", [0, 1, 0])
    frame.insert(0, "CONS_NO", frame.index)
    return frame.reset_index(drop=True)


@pytest.fixture
def sgcc_csv(tmp_path) -> Path:
    path = tmp_path / "sgcc.csv"
    sgcc_frame().to_csv(path, index=False)
    return path
