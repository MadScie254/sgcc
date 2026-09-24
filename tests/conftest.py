"""Shared pytest fixtures."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Keep API runtime state (cases, pipeline runs, published threshold) out of the repo.
os.environ.setdefault("SGCC_STATE_DIR", tempfile.mkdtemp(prefix="sgcc-state-"))
DEMO_DATASET = REPO_ROOT / "data" / "sgcc_demo.csv.gz"


@pytest.fixture(scope="session")
def demo_dataset_path() -> Path:
    """The held-out customer sample the API serves (committed to the repo)."""
    assert DEMO_DATASET.is_file(), "data/sgcc_demo.csv.gz is missing; run python -m src.train"
    return DEMO_DATASET


@pytest.fixture
def sgcc_csv(tmp_path) -> Path:
    """
    A small file in the raw SGCC layout: CONS_NO, FLAG, then date columns in
    lexicographic (not chronological) order, as in the public dump.

    Customer A consumes day-of-year kWh (strictly increasing); B is constant
    until a mid-series drop to zero; C has a long gap.
    """
    dates = pd.date_range("2014-01-01", periods=120, freq="D")
    values = {
        "A": np.arange(1, 121, dtype=float),
        "B": np.r_[np.full(60, 10.0), np.zeros(60)],
        "C": np.r_[np.full(40, 5.0), np.full(40, np.nan), np.full(40, 5.0)],
    }
    frame = pd.DataFrame(values, index=[f"{d.year}/{d.month}/{d.day}" for d in dates]).T
    frame = frame[sorted(frame.columns)]  # "2014/1/1", "2014/1/10", ...
    frame.insert(0, "FLAG", [0, 1, 0])
    frame.insert(0, "CONS_NO", frame.index)
    path = tmp_path / "sgcc.csv"
    frame.to_csv(path, index=False)
    return path
