"""Tests for src.features."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_wide
from src.features import (
    build_features,
    build_features_wide,
    compute_anomaly_features,
    compute_other_features,
    compute_statistical_features,
    compute_temporal_features,
    compute_trend_features,
)


def test_statistical_features():
    features = compute_statistical_features(np.array([10, 20, 15, 25, 30, 0, 35, 40], dtype=float))
    assert features["min"] == 0
    assert features["max"] == 40
    assert features["range"] == 40
    assert features["mean"] == pytest.approx(21.875)


def test_trend_features_detect_increase():
    features = compute_trend_features(np.arange(100, dtype=float) * 2)
    assert features["slope_full"] == pytest.approx(2.0)
    assert features["slope_last_30d"] == pytest.approx(2.0)


def test_anomaly_features():
    features = compute_anomaly_features(np.array([10, 10, 2, 10, 0, 0, 10], dtype=float))
    assert features["zero_day_count"] == 2
    assert features["sudden_drop_count"] == 2  # 10 -> 2 and 10 -> 0


def test_temporal_features():
    features = compute_temporal_features(np.tile([10, 10, 10, 10, 10, 5, 5], 4).astype(float))
    assert features["weekday_vs_weekend_ratio"] == pytest.approx(2.0)
    assert 0 <= features["peak_day_ratio"] <= 1


def test_other_features_counts_missing_runs():
    consumption = np.array([1, 2, np.nan, np.nan, np.nan, np.nan, 3, 4, np.nan, 5], dtype=float)
    assert compute_other_features(consumption)["missing_sequences_count"] == 1


def test_build_features_wide_matches_single_series_helpers(sgcc_csv):
    wide, _ = load_wide(str(sgcc_csv))
    X = build_features_wide(wide)

    assert list(X.index) == ["A", "B", "C"]
    assert X.loc["A", "slope_full"] == pytest.approx(1.0, rel=1e-5)
    assert X.loc["B", "zero_day_count"] == 60
    assert X.loc["B", "longest_zero_run"] == 60
    assert X.loc["B", "sudden_drop_count"] == 1
    assert X.loc["C", "longest_missing_run"] == 40
    assert X.loc["C", "missing_ratio"] == pytest.approx(40 / 120)
    # Monthly profile: B's latest month is all zeros relative to its mean.
    assert X.loc["B", "month_lag_00"] == 0
    for name in ("mean", "std", "zero_day_count", "sudden_drop_count"):
        single = {**compute_statistical_features(wide.loc["B"].to_numpy(float)),
                  **compute_anomaly_features(wide.loc["B"].to_numpy(float))}
        assert X.loc["B", name] == pytest.approx(single[name], rel=1e-5)


def test_build_features_wide_handles_empty_customer():
    wide = pd.DataFrame([[np.nan] * 60, list(range(60))], index=["empty", "ok"],
                        columns=pd.date_range("2014-01-01", periods=60), dtype="float32")
    X = build_features_wide(wide)

    assert X.loc["empty", "missing_ratio"] == 1.0
    assert not np.isinf(X.to_numpy()).any()


def test_build_features_from_long_format():
    rng = np.random.default_rng(0)
    df_long = pd.DataFrame({
        "customer_id": ["C1"] * 100 + ["C2"] * 100,
        "day_index": list(range(100)) * 2,
        "consumption_kwh": rng.random(200) * 50,
    })
    labels = pd.Series([0, 1], index=["C1", "C2"], name="label")

    X, y = build_features(df_long, labels)

    assert list(X.index) == ["C1", "C2"]
    assert y.tolist() == [0, 1]
    assert X.shape[1] >= 50


def test_build_features_interleaved_rows():
    """Rows interleaved across customers give the same features as contiguous rows."""
    rng = np.random.default_rng(0)
    contiguous = pd.DataFrame({
        "customer_id": ["C1"] * 60 + ["C2"] * 60 + ["C3"] * 60,
        "day_index": list(range(60)) * 3,
        "consumption_kwh": rng.random(180) * 50,
    })
    interleaved = contiguous.sort_values(["day_index", "customer_id"], kind="stable")
    labels = pd.Series([0, 1, 0], index=["C1", "C2", "C3"], name="label")

    X_contiguous, y_contiguous = build_features(contiguous, labels)
    X_interleaved, y_interleaved = build_features(interleaved, labels)

    pd.testing.assert_frame_equal(X_contiguous, X_interleaved)
    pd.testing.assert_series_equal(y_contiguous, y_interleaved)
