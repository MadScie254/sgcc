"""Tests for src.preprocessing, src.resampling and src.pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.data_loader import load_wide
from src.features import build_features_wide
from src.modeling import make_folds
from src.pipeline import model_input
from src.preprocessing import clean_series
from src.resampling import Treatment, diagnostics


def test_clean_series_applies_each_rule_and_logs_it():
    wide = pd.DataFrame([[1, np.nan, np.nan, 4, np.nan, np.nan, np.nan, np.nan, 9, 1],
                         [np.nan] * 10,
                         [-1, 2, 2, 2, 20000, 2, 2, 2, 2, 2]], dtype=float)
    cleaned, log = clean_series(wide, {"outlier_sigma": 100})

    np.testing.assert_allclose(cleaned.loc[0], [1, 2, 3, 4, 4, 4, 4, 4, 9, 1])  # short gap interpolated, long one filled
    assert (cleaned.loc[1] == 0).all()  # no readings at all
    assert cleaned.loc[2, 0] == 2 and cleaned.loc[2, 4] == 10_000  # negative -> median, capped
    assert not cleaned.isna().any().any()
    assert log["short_gap_cells_interpolated"] == 2 and log["long_gap_cells_filled"] == 4
    assert log["negative_replaced"] == 1 and log["capped_above_limit"] == 1 and log["customers_without_readings"] == 1


def test_clean_series_replaces_sigma_outliers_with_median():
    row = np.full(40, 5.0)
    row[10] = 500.0
    cleaned, log = clean_series(pd.DataFrame([row]))
    assert cleaned.iloc[0, 10] == 5.0 and log["outliers_replaced"] == 1


def test_missingness_features_survive_cleaning(sgcc_csv):
    wide, _ = load_wide(str(sgcc_csv))
    raw = build_features_wide(wide)
    clean = model_input(wide, {"preprocessing": "clean"})
    for name in ("missing_ratio", "longest_missing_run", "first_obs_frac", "last_obs_frac"):
        pd.testing.assert_series_equal(raw[name], clean[name])
    assert clean.loc["C", "longest_missing_run"] == 40


def test_model_input_imputes_with_training_medians(sgcc_csv):
    wide, _ = load_wide(str(sgcc_csv))
    X = model_input(wide, {"preprocessing": "raw", "impute": {"last30_vs_mean": -1.0}})
    raw = build_features_wide(wide)
    assert X["last30_vs_mean"].notna().all()
    assert (X["last30_vs_mean"][raw["last30_vs_mean"].isna()] == -1.0).all()


@pytest.fixture(scope="module")
def imbalanced():
    X, y = make_classification(n_samples=600, n_features=5, n_informative=3, weights=[0.9, 0.1], random_state=1)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    X.iloc[::11, 0] = np.nan
    return X, pd.Series(y)


def test_smote_enn_counts_and_units(imbalanced):
    X, y = imbalanced
    treatment = Treatment("smote_enn", random_state=0)
    X_res, y_res = treatment.fit_resample(X, y)
    stats = treatment.stats_

    assert stats["after_smote"]["theft"] == int(0.5 * stats["before"]["honest"])
    assert stats["after"] == {"honest": int((y_res == 0).sum()), "theft": int((y_res == 1).sum())}
    removed = stats["synthetic_removed_by_enn"] + stats["honest_removed_by_enn"] + stats["theft_removed_by_enn"]
    assert len(y_res) == len(y) + stats["synthetic_created"] - removed
    assert not X_res.isna().any().any()
    assert list(X_res.columns) == list(X.columns)
    # Back in the original units, not standardised.
    assert X_res["f1"].mean() == pytest.approx(X["f1"].mean(), abs=1.0)
    assert treatment.transform(X).isna().sum().sum() == 0


def test_no_treatment_is_identity(imbalanced):
    X, y = imbalanced
    treatment = Treatment("none")
    X_res, y_res = treatment.fit_resample(X, y)
    assert X_res is X and y_res is y and treatment.transform(X) is X


def test_resampling_stays_inside_training_folds(imbalanced):
    X, y = imbalanced
    folds = make_folds(X, y, cv=3, random_state=0, treatment="smote_enn")
    covered = np.sort(np.concatenate([val_idx for *_, val_idx in folds]))
    np.testing.assert_array_equal(covered, np.arange(len(y)))  # every real row validated exactly once
    for X_tr, y_tr, X_val, val_idx in folds:
        assert len(X_val) == len(val_idx)  # validation rows are never resampled
        np.testing.assert_allclose(X_val["f1"], X.iloc[val_idx]["f1"])
        assert (y_tr == 1).mean() > (y == 1).mean()  # training rows were


def test_diagnostics(imbalanced):
    from sklearn.preprocessing import StandardScaler

    X, y = imbalanced
    filled = X.fillna(X.median())
    stats = diagnostics(filled, y, StandardScaler().fit(filled))
    assert -1 <= stats["silhouette"] <= 1
    assert 0 <= stats["boundary_noise"] <= 1 and 0 <= stats["boundary_noise_theft"] <= 1
    assert stats["fisher_ratio_max"] >= stats["fisher_ratio_mean"] >= 0
