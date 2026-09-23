"""Tests for src.data_loader."""

import numpy as np
import pandas as pd

from src.data_loader import load_processed_features, load_raw, load_wide, save_processed_features


def test_load_wide_sorts_date_columns_chronologically(sgcc_csv):
    wide, labels = load_wide(str(sgcc_csv))

    assert isinstance(wide.columns, pd.DatetimeIndex)
    assert wide.columns.is_monotonic_increasing
    assert wide.shape == (3, 120)
    np.testing.assert_array_equal(wide.loc["A"].to_numpy(), np.arange(1, 121))
    assert labels.to_dict() == {"A": 0, "B": 1, "C": 0}
    assert labels.name == "label"


def test_load_wide_positional_layout(tmp_path):
    frame = pd.DataFrame([[1.0, 2.0, "id1", 0], [3.0, np.nan, "id2", 1]], columns=["d0", "d1", "cid", "y"])
    path = tmp_path / "positional.csv"
    frame.to_csv(path, index=False)

    wide, labels = load_wide(str(path))

    assert list(wide.index) == ["id1", "id2"]
    assert labels.tolist() == [0, 1]
    assert np.isnan(wide.loc["id2"].iloc[1])


def test_load_wide_drops_duplicate_customers(tmp_path):
    frame = pd.DataFrame({"CONS_NO": ["a", "a", "b"], "FLAG": [0, 0, 1], "2014/1/1": [1, 1, 2], "2014/1/2": [3, 3, 4]})
    path = tmp_path / "dups.csv"
    frame.to_csv(path, index=False)

    wide, labels = load_wide(str(path))

    assert list(wide.index) == ["a", "b"]
    assert len(labels) == 2


def test_load_raw_long_format(sgcc_csv):
    df_long, labels = load_raw(str(sgcc_csv))

    assert list(df_long.columns) == ["customer_id", "day_index", "consumption_kwh"]
    assert len(df_long) == 3 * 120
    first = df_long[df_long["customer_id"] == "A"].sort_values("day_index")
    np.testing.assert_array_equal(first["consumption_kwh"].to_numpy(), np.arange(1, 121))
    assert df_long["customer_id"].nunique() == len(labels)


def test_demo_dataset_loads(demo_dataset_path):
    wide, labels = load_wide(str(demo_dataset_path))

    assert len(wide) == len(labels) > 100
    assert set(labels.unique()) == {0, 1}
    assert wide.columns.is_monotonic_increasing
    assert (wide.to_numpy()[~np.isnan(wide.to_numpy())] >= 0).all()


def test_processed_features_round_trip(tmp_path):
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.5, np.nan]}, index=["c1", "c2"])
    y = pd.Series([0, 1], index=X.index, name="label")
    path = tmp_path / "features.csv"

    save_processed_features(X, y, str(path))
    X2, y2 = load_processed_features(str(path))

    pd.testing.assert_frame_equal(X, X2)
    assert y2.tolist() == [0, 1]
    assert list(X2.index) == ["c1", "c2"]
