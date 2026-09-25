"""Tests for src.features and src.feature_catalog."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_wide
from src.feature_catalog import CORE_FEATURES, FEATURE_GROUPS, feature_group, feature_label, format_feature_value
from src.features import build_features_wide


def test_build_features_wide_values(sgcc_csv):
    wide, _ = load_wide(str(sgcc_csv))
    X = build_features_wide(wide)

    assert list(X.index) == ["A", "B", "C"]
    assert X.shape[1] == 87
    assert X.loc["A", "slope_full"] == pytest.approx(1.0, rel=1e-5)
    assert X.loc["A", "mean"] == pytest.approx(60.5)
    assert X.loc["B", "zero_day_count"] == 60
    assert X.loc["B", "longest_zero_run"] == 60
    assert X.loc["B", "sudden_drop_count"] == 1
    assert X.loc["C", "longest_missing_run"] == 40
    assert X.loc["C", "missing_ratio"] == pytest.approx(40 / 120)
    assert X.loc["B", "month_lag_00"] == 0  # latest month is all zeros relative to the mean
    assert X.loc["A", "slope_last_180d"] == pytest.approx(1.0, rel=1e-5)
    assert X.loc["A", "trend_direction_changes"] == 0  # always rising


def test_build_features_wide_handles_empty_customer():
    wide = pd.DataFrame([[np.nan] * 60, list(range(60))], index=["empty", "ok"],
                        columns=pd.date_range("2014-01-01", periods=60), dtype="float32")
    X = build_features_wide(wide)

    assert X.loc["empty", "missing_ratio"] == 1.0
    assert not np.isinf(X.to_numpy()).any()


def test_build_features_wide_without_dates_uses_configured_calendar(sgcc_csv):
    wide, _ = load_wide(str(sgcc_csv))
    undated = wide.copy()
    undated.columns = range(undated.shape[1])
    pd.testing.assert_frame_equal(build_features_wide(wide), build_features_wide(undated))


def test_feature_catalog():
    assert feature_label("longest_missing_run") == "Longest reporting gap"
    assert feature_label("month_lag_00") == "Latest month vs average"
    assert feature_label("month_lag_07") == "Use 7 months ago vs average"
    assert format_feature_value("missing_ratio", 0.25) == "25%"
    assert format_feature_value("longest_missing_run", 573.0) == "573 days"
    assert format_feature_value("last30_vs_mean", None) == "no readings"
    assert format_feature_value("last_obs_frac", 460 / 1034, date(2014, 1, 1), 1034) == "day 460 · Apr 2015"


def test_core_features_and_groups(sgcc_csv):
    wide, _ = load_wide(str(sgcc_csv))
    X = build_features_wide(wide)
    assert len(CORE_FEATURES) == len(set(CORE_FEATURES)) == 25
    assert set(CORE_FEATURES) <= set(X.columns)
    assert {feature_group(name) for name in X.columns} == set(FEATURE_GROUPS)
    assert {feature_group(name) for name in CORE_FEATURES} == set(FEATURE_GROUPS)


def test_trend_direction_changes_counts_reversals():
    up, down = np.arange(30, dtype=float), np.arange(30, 0, -1, dtype=float)
    wide = pd.DataFrame([np.r_[up, down, up, np.full(30, 5.0)]], columns=pd.date_range("2014-01-01", periods=120))
    assert build_features_wide(wide).iloc[0]["trend_direction_changes"] == 2  # up, down, up; the flat window is skipped
