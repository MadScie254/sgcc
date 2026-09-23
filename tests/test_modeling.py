"""Tests for src.modeling and src.eval."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.eval import classification_metrics, feature_importance
from src.modeling import cross_val_proba, get_xgb_model, load_model, save_model, select_threshold, tune_xgb


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=400, n_features=6, n_informative=4, weights=[0.85, 0.15], random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    X.iloc[::17, 0] = np.nan  # the model must cope with missing values
    return X, pd.Series(y)


SMALL_SPACE = {"n_estimators": {"low": 20, "high": 40, "step": 10}, "max_depth": {"low": 2, "high": 3}}


def test_cross_val_proba_covers_every_row(data):
    X, y = data
    oof = cross_val_proba({"n_estimators": 20}, X, y, cv=3)
    assert oof.shape == (len(y),)
    assert ((oof >= 0) & (oof <= 1)).all()
    assert classification_metrics(y, oof)["auc"] > 0.7


def test_tune_xgb_returns_params_in_search_space(data):
    X, y = data
    params, study = tune_xgb(X, y, n_trials=2, cv=3, search_space=SMALL_SPACE)
    assert len(study.trials) == 2
    assert 20 <= params["n_estimators"] <= 40
    assert 2 <= params["max_depth"] <= 3


def test_select_threshold_strategies():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.6, 0.7, 0.9])
    assert select_threshold(y, proba, "f1") == pytest.approx(0.35)
    assert select_threshold(y, proba, "precision", min_precision=1.0) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        select_threshold(y, proba, "nope")


def test_classification_metrics_single_class_confusion():
    metrics = classification_metrics([0, 0, 0], [0.1, 0.9, 0.2], threshold=0.5)
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 0, "tp": 0}


def test_model_round_trip_native_format(data, tmp_path):
    X, y = data
    model = get_xgb_model({"n_estimators": 20}).fit(X, y)
    path = tmp_path / "model.ubj"

    save_model(model, str(path))
    loaded = load_model(str(path))

    np.testing.assert_allclose(model.predict_proba(X), loaded.predict_proba(X), rtol=1e-6)
    assert loaded.get_booster().feature_names == list(X.columns)
    importance = feature_importance(loaded, X.columns)
    assert importance["importance"].sum() == pytest.approx(1.0)
