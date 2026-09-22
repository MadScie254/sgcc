"""
Consistency tests for the deployed model artifacts and the backend serving path.

These guard against the failures found in the 2026-09 audit: metrics.json
describing a different model than the one deployed, and the backend scoring
unscaled features with a model trained on scaled ones.
"""

import hashlib
import json

import numpy as np
import pytest

from tests.conftest import REPO_ROOT, require_data_file


@pytest.fixture(scope="module")
def artifacts():
    for path in [
        "models/xgb_best.joblib",
        "artifacts/test_data.pkl",
        "artifacts/features.csv",
        "data/datasetsmall.csv",
    ]:
        require_data_file(path)

    import joblib

    from src.modeling import load_model

    metrics = json.loads((REPO_ROOT / "artifacts/metrics.json").read_text(encoding="utf-8"))
    test_data = joblib.load(REPO_ROOT / "artifacts/test_data.pkl")
    return {
        "model": load_model(str(REPO_ROOT / "models/xgb_best.joblib")),
        "metrics": metrics,
        "X_test": test_data["X_test"],
        "y_test": test_data["y_test"],
    }


def test_deployed_model_is_a_pipeline_with_named_features(artifacts):
    model = artifacts["model"]
    assert list(model.named_steps) == ["scaler", "resample", "clf"]

    feature_names = json.loads((REPO_ROOT / "artifacts/feature_names.json").read_text(encoding="utf-8"))
    assert list(model.feature_names_in_) == feature_names
    assert list(artifacts["X_test"].columns) == feature_names


def test_metrics_json_describes_the_deployed_model(artifacts):
    from src.eval import evaluate_model

    stored = artifacts["metrics"]
    recomputed = evaluate_model(
        artifacts["model"], artifacts["X_test"], artifacts["y_test"], threshold=stored["threshold"]
    )

    assert recomputed["confusion_matrix"] == stored["confusion_matrix"]
    assert recomputed["support"] == stored["support"]
    for key in ["recall", "precision", "f1", "accuracy", "auc", "gmean", "mcc"]:
        assert recomputed[key] == pytest.approx(stored[key], abs=1e-12), key
    assert stored["split"]["n_test"] == len(artifacts["y_test"])
    assert stored["quick_mode"] is False


def test_metrics_provenance_matches_committed_data():
    require_data_file("data/datasetsmall.csv")
    require_data_file("artifacts/features.csv")
    stored = json.loads((REPO_ROOT / "artifacts/metrics.json").read_text(encoding="utf-8"))

    data_path = REPO_ROOT / stored["data"]["path"]
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == stored["data"]["sha256"]

    from src.data_loader import load_processed_features

    X, _ = load_processed_features(str(REPO_ROOT / "artifacts/features.csv"))
    assert len(X) == stored["data"]["n_customers"]


@pytest.fixture(scope="module")
def serving(artifacts):
    from backend.services import model as model_service

    return model_service


def test_served_probability_equals_pipeline_probability(serving, artifacts):
    """Compare the API's answer with the pipeline loaded independently from disk."""
    from backend.services.data import get_feature_matrix

    X, _ = get_feature_matrix()
    reference = artifacts["model"]
    feature_names = list(reference.feature_names_in_)
    for customer_id in X.index[:5]:
        served = serving.predict_for_customer(str(customer_id))["probability"]
        expected = reference.predict_proba(X.loc[[customer_id], feature_names])[:, 1][0]
        assert served == pytest.approx(expected, abs=1e-9)


def test_shap_explains_the_served_probability(serving, artifacts):
    """SHAP additivity only holds if SHAP sees the same scaled inputs the model scores."""
    from backend.services.data import get_feature_matrix

    X, _ = get_feature_matrix()
    customer_id = str(X.index[0])
    details = serving.get_local_shap_details(customer_id)
    reference = artifacts["model"]
    expected = reference.predict_proba(X.loc[[X.index[0]], list(reference.feature_names_in_)])[:, 1][0]

    margin = details["base_value"] + float(np.sum(details["shap_values"]))
    probability = 1.0 / (1.0 + np.exp(-margin))
    assert details["probability"] == pytest.approx(expected, abs=1e-9)
    assert probability == pytest.approx(expected, abs=1e-5)

    raw_row = X.loc[[X.index[0]], serving.get_feature_names()].iloc[0]
    assert details["feature_values"] == pytest.approx(raw_row.astype(float).tolist())


def test_metrics_endpoint_reports_the_test_split(serving, artifacts):
    payload = serving.get_model_metrics()
    stored = artifacts["metrics"]
    assert payload["confusion_matrix"] == stored["confusion_matrix"]
    assert payload["support"] == stored["support"]
    assert payload["model_version"] == stored["model_version"]


def test_operating_point_is_consistent_with_the_model(serving, artifacts):
    stored = artifacts["metrics"]
    operating_point = stored.get("operating_point")
    if operating_point is None:
        pytest.skip("metrics.json predates operating_point")

    assert stored["threshold"] == operating_point["threshold"]
    assert operating_point["medium_threshold"] <= operating_point["threshold"]

    scores = artifacts["model"].predict_proba(artifacts["X_test"])[:, 1]
    assert float((scores >= stored["threshold"]).mean()) == pytest.approx(operating_point["test_flag_rate"], abs=1e-12)

    assert serving.get_decision_threshold() == stored["threshold"]
    assert serving.get_risk_tier_thresholds() == {
        "high": operating_point["threshold"],
        "medium": operating_point["medium_threshold"],
    }
