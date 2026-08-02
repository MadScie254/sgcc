from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, cast

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from src.eval import evaluate_model
from src.modeling import load_model

from .config import get_config, get_project_paths
from .data import get_customer_timeseries, get_feature_matrix


@lru_cache(maxsize=1)
def get_model_path() -> Path:
    return get_project_paths()["models"] / "xgb_best.joblib"


@lru_cache(maxsize=1)
def get_trained_model():
    return load_model(str(get_model_path()))


@lru_cache(maxsize=1)
def get_feature_names() -> List[str]:
    X, _ = get_feature_matrix()
    return X.columns.tolist()


@lru_cache(maxsize=1)
def get_feature_importance(limit: int = 20) -> List[Dict[str, object]]:
    model = get_trained_model()
    feature_names = get_feature_names()
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        raise RuntimeError("Model does not expose feature_importances_")

    ranking = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(limit)
    )
    return [
        {"feature": str(row.feature), "importance": float(cast(Any, row.importance))}
        for row in ranking.itertuples(index=False)
    ]


@lru_cache(maxsize=1)
def get_shap_explainer():
    model = get_trained_model()
    return shap.TreeExplainer(model)


@lru_cache(maxsize=1)
def get_test_data() -> Dict[str, object]:
    artifact_path = get_project_paths()["artifacts"] / "test_data.pkl"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Test data not found: {artifact_path}")
    return joblib.load(artifact_path)


def _align_feature_row(features: Dict[str, float]) -> pd.DataFrame:
    feature_names = get_feature_names()
    aligned = {name: float(features.get(name, 0.0)) for name in feature_names}
    return pd.DataFrame([aligned], columns=feature_names)


@lru_cache(maxsize=128)
def get_customer_feature_row(customer_id: str) -> pd.DataFrame:
    X, _ = get_feature_matrix()
    if str(customer_id) not in X.index.astype(str):
        raise KeyError(f"Unknown customer_id: {customer_id}")
    row = X.loc[str(customer_id)]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return pd.DataFrame([row.to_dict()], columns=get_feature_names())


def predict_from_features(features: Dict[str, float], threshold: float = 0.5) -> Dict[str, object]:
    model = get_trained_model()
    feature_frame = _align_feature_row(features)
    probability = float(model.predict_proba(feature_frame)[:, 1][0])
    prediction = int(probability >= threshold)
    top_reasons = get_top_reasons(feature_frame)
    return {
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold,
        "top_reasons": top_reasons,
    }


def predict_for_customer(customer_id: str, threshold: float = 0.5) -> Dict[str, object]:
    model = get_trained_model()
    feature_frame = get_customer_feature_row(customer_id)
    probability = float(model.predict_proba(feature_frame)[:, 1][0])
    prediction = int(probability >= threshold)
    top_reasons = get_top_reasons(feature_frame)
    return {
        "customer_id": str(customer_id),
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold,
        "top_reasons": top_reasons,
    }


def get_top_reasons(feature_frame: pd.DataFrame, top_n: int = 3) -> List[Dict[str, float]]:
    explainer = get_shap_explainer()
    shap_values = explainer.shap_values(feature_frame)

    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[1 if len(shap_values) > 1 else 0])
    else:
        shap_array = np.asarray(shap_values)

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    values = shap_array[0]
    top_indices = np.argsort(np.abs(values))[::-1][:top_n]
    row = feature_frame.iloc[0]

    reasons = []
    for idx in top_indices:
        feature_name = feature_frame.columns[idx]
        reasons.append({
            "feature": feature_name,
            "value": float(row.iloc[idx]),
            "shap_value": float(values[idx]),
        })
    return reasons


def _extract_shap_values(explainer, feature_frame: pd.DataFrame) -> tuple[float, List[float]]:
    shap_values = explainer.shap_values(feature_frame)

    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[1 if len(shap_values) > 1 else 0])
    else:
        shap_array = np.asarray(shap_values)

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, tuple, np.ndarray)):
        base_value = base_value[1 if len(base_value) > 1 else 0]

    return float(base_value), shap_array[0].astype(float).tolist()


@lru_cache(maxsize=1)
def get_global_shap_sample(sample_count: int = 200) -> Dict[str, object]:
    model = get_trained_model()
    X, _ = get_feature_matrix()
    sample_frame = X.head(sample_count).copy()
    explainer = get_shap_explainer()
    shap_values = explainer.shap_values(sample_frame)

    if isinstance(shap_values, list):
        shap_values = shap_values[1 if len(shap_values) > 1 else 0]

    shap_array = np.asarray(shap_values, dtype=float)
    feature_values = sample_frame.astype(float).values.tolist()

    return {
        "feature_names": sample_frame.columns.tolist(),
        "shap_values": shap_array.tolist(),
        "feature_values": feature_values,
        "sample_count": int(len(sample_frame)),
    }


@lru_cache(maxsize=128)
def get_local_shap_details(customer_id: str) -> Dict[str, object]:
    model = get_trained_model()
    explainer = get_shap_explainer()
    feature_frame = get_customer_feature_row(customer_id)
    probability = float(model.predict_proba(feature_frame)[:, 1][0])
    base_value, shap_values = _extract_shap_values(explainer, feature_frame)

    feature_names = feature_frame.columns.tolist()
    feature_values = feature_frame.iloc[0].astype(float).tolist()
    top_indices = np.argsort(np.abs(np.asarray(shap_values)))[::-1][:5]
    top_reasons = [
        {
            "feature": feature_names[index],
            "value": float(feature_values[index]),
            "shap_value": float(shap_values[index]),
        }
        for index in top_indices
    ]

    return {
        "customer_id": str(customer_id),
        "feature_names": feature_names,
        "feature_values": feature_values,
        "shap_values": shap_values,
        "base_value": base_value,
        "probability": probability,
        "top_reasons": top_reasons,
    }


def threshold_preview(threshold: float) -> Dict[str, object]:
    model = get_trained_model()
    test_data = get_test_data()
    X_test = pd.DataFrame(cast(Any, test_data["X_test"]))
    y_test = pd.Series(cast(Any, test_data["y_test"]))
    probabilities = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
    predictions = np.asarray(probabilities >= threshold, dtype=int)
    metrics = {
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "auc": float(roc_auc_score(y_test, probabilities)),
    }
    cm = confusion_matrix(y_test, predictions)
    return {
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }
