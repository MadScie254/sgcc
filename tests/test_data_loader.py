"""Tests for src.data_loader."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import frame_to_wide, is_consumption_frame, load_wide
from tests.conftest import sgcc_frame


def test_load_wide_sorts_date_columns_chronologically(sgcc_csv):
    wide, labels = load_wide(str(sgcc_csv))

    assert isinstance(wide.columns, pd.DatetimeIndex)
    assert wide.columns.is_monotonic_increasing
    assert wide.shape == (3, 120)
    np.testing.assert_array_equal(wide.loc["A"].to_numpy(), np.arange(1, 121))
    assert labels.to_dict() == {"A": 0, "B": 1, "C": 0}
    assert labels.name == "label"


def test_frame_to_wide_labels_optional():
    frame = sgcc_frame().drop(columns="FLAG")
    wide, labels = frame_to_wide(frame, require_labels=False)
    assert labels is None
    assert list(wide.index) == ["A", "B", "C"]
    with pytest.raises(ValueError, match="label column"):
        frame_to_wide(frame)


def test_frame_to_wide_rejects_bad_input():
    with pytest.raises(ValueError, match="customer id"):
        frame_to_wide(sgcc_frame().drop(columns="CONS_NO"))
    with pytest.raises(ValueError, match="at least 30"):
        frame_to_wide(sgcc_frame(days=10))
    bad = sgcc_frame()
    bad.loc[0, "FLAG"] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        frame_to_wide(bad)


def test_frame_to_wide_drops_duplicate_customers():
    frame = pd.concat([sgcc_frame(), sgcc_frame().iloc[[0]]], ignore_index=True)
    wide, labels = frame_to_wide(frame)
    assert list(wide.index) == ["A", "B", "C"]
    assert len(labels) == 3


def test_is_consumption_frame():
    assert is_consumption_frame(sgcc_frame())
    assert not is_consumption_frame(pd.DataFrame({"CONS_NO": ["a"], "mean": [1.0]}))


def test_demo_dataset_loads(demo_dataset_path):
    wide, labels = load_wide(str(demo_dataset_path))

    assert len(wide) == len(labels) > 100
    assert set(labels.unique()) == {0, 1}
    assert wide.columns.is_monotonic_increasing
    observed = wide.to_numpy()[~np.isnan(wide.to_numpy())]
    assert (observed >= 0).all()
