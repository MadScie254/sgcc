"""Tests for the modeling module."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.modeling import (
    build_cv_pipeline,
    evaluate_pipeline_cv,
    fit_final_pipeline,
    get_classifier,
    transform_for_classifier,
)


def test_build_cv_pipeline_structure() -> None:
    """The CV pipeline should include scaler, resampler, and classifier steps."""
    pipeline = build_cv_pipeline(
        {
            "smote": {"k_neighbors": 3, "sampling_strategy": "auto"},
            "enn": {"n_neighbors": 4},
        },
        {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "scale_pos_weight": 2.0,
            "min_child_weight": 1,
            "gamma": 0.0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": -1,
        },
        random_state=42,
    )

    assert list(pipeline.named_steps) == ["scaler", "resample", "clf"]
    assert pipeline.named_steps["resample"].smote.k_neighbors == 3
    assert pipeline.named_steps["resample"].enn.n_neighbors == 4
    assert pipeline.named_steps["clf"].get_params()["max_depth"] == 6


def test_evaluate_pipeline_cv_uses_one_pipeline_per_fold() -> None:
    """Each CV split should receive a freshly built pipeline instance."""
    X_array, y_array = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.75, 0.25],
        random_state=42,
    )
    X = pd.DataFrame(X_array, columns=[f"feature_{i}" for i in range(X_array.shape[1])])
    y = pd.Series(y_array)

    factory_calls = 0
    fit_calls = 0

    class DummyPipeline:
        def fit(self, X_fold, y_fold):
            nonlocal fit_calls
            fit_calls += 1
            self._train_size = len(X_fold)
            self._label_count = len(y_fold)
            return self

        def predict(self, X_fold):
            return np.zeros(len(X_fold), dtype=int)

    def pipeline_factory():
        nonlocal factory_calls
        factory_calls += 1
        return DummyPipeline()

    scores = evaluate_pipeline_cv(
        pipeline_factory,
        X,
        y,
        cv=4,
        random_state=42,
    )

    assert factory_calls == 4
    assert fit_calls == 4
    assert len(scores) == 4


def test_final_pipeline_scores_raw_features_like_its_classifier() -> None:
    """The saved pipeline must apply its own scaling: no train/serve skew."""
    X_array, y_array = make_classification(
        n_samples=300, n_features=5, weights=[0.85, 0.15], random_state=0
    )
    # Very different feature scales, so a missing scaler would change predictions
    X = pd.DataFrame(X_array * [1, 10, 100, 1000, 10000], columns=[f"f{i}" for i in range(5)])
    y = pd.Series(y_array)

    pipeline = fit_final_pipeline(
        X,
        y,
        xgb_params={"n_estimators": 30, "max_depth": 3, "random_state": 0},
        smote_enn_params={"smote": {"k_neighbors": 3, "sampling_strategy": 0.8}, "enn": {"n_neighbors": 3}},
        random_state=0,
    )

    X_model = transform_for_classifier(pipeline, X)
    assert list(X_model.columns) == list(X.columns)
    assert X_model.index.equals(X.index)
    assert X_model.to_numpy().min() >= -1e-9 and X_model.to_numpy().max() <= 1 + 1e-9

    classifier = get_classifier(pipeline)
    np.testing.assert_allclose(
        pipeline.predict_proba(X)[:, 1],
        classifier.predict_proba(X_model)[:, 1],
    )
    assert list(pipeline.feature_names_in_) == list(X.columns)


def test_helpers_pass_through_a_bare_classifier() -> None:
    """get_classifier/transform_for_classifier are no-ops without a pipeline."""
    X = pd.DataFrame({"a": [1.0, 2.0]})
    classifier = object()
    assert get_classifier(classifier) is classifier
    assert transform_for_classifier(classifier, X) is X
