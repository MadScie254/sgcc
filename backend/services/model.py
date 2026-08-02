from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from src.eval import evaluate_model
from src.modeling import load_model

from .config import get_config, get_project_paths
from .data import get_customer_timeseries, get_feature_matrix


HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4


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


def risk_tier_for_probability(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "high"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


@lru_cache(maxsize=1)
def get_model_config() -> Dict[str, Any]:
    config = get_config()
    feature_groups = {
        key: value
        for key, value in config.get("features", {}).items()
        if isinstance(value, list)
    }
    feature_parameters = {
        key: value
        for key, value in config.get("features", {}).items()
        if not isinstance(value, list)
    }
    return {
        "feature_groups": feature_groups,
        "feature_parameters": feature_parameters,
        "model": config.get("model", {}),
        "preprocessing": config.get("preprocessing", {}),
        "evaluation": config.get("evaluation", {}),
    }


@lru_cache(maxsize=1)
def get_model_metrics() -> Dict[str, Any]:
    metrics_path = get_project_paths()["artifacts"] / "metrics.json"
    metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = cast(Dict[str, Any], json.load(handle))

    X, y = get_feature_matrix()
    model = get_trained_model()
    probabilities = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    threshold = float(metrics.get("threshold", 0.5))
    predictions = np.asarray(probabilities >= threshold, dtype=int)
    confusion = confusion_matrix(y, predictions)
    support = metrics.get("support") or {
        "class_0": int((y == 0).sum()),
        "class_1": int((y == 1).sum()),
    }
    total_support = max(int(support.get("class_0", 0)) + int(support.get("class_1", 0)), 1)

    return {
        "threshold": threshold,
        "metrics": {
            "recall": float(metrics.get("recall", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "f1": float(metrics.get("f1", 0.0)),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "auc": float(metrics.get("auc", 0.0)),
            "gmean": float(metrics.get("gmean", 0.0)),
            "mcc": float(metrics.get("mcc", 0.0)),
        },
        "support": {
            "class_0": int(support.get("class_0", 0)),
            "class_1": int(support.get("class_1", 0)),
        },
        "confusion_matrix": {
            "tn": int(confusion[0, 0]),
            "fp": int(confusion[0, 1]),
            "fn": int(confusion[1, 0]),
            "tp": int(confusion[1, 1]),
        },
        "customers_monitored": int(len(X)),
        "flagged_today": int(predictions.sum()),
        "current_mean_probability": float(probabilities.mean()) if len(probabilities) else 0.0,
        "base_rate": float(int(support.get("class_1", 0)) / total_support),
        "risk_tier_distribution": {
            "high": int((probabilities >= HIGH_RISK_THRESHOLD).sum()),
            "medium": int(((probabilities >= MEDIUM_RISK_THRESHOLD) & (probabilities < HIGH_RISK_THRESHOLD)).sum()),
            "low": int((probabilities < MEDIUM_RISK_THRESHOLD).sum()),
        },
    }


@lru_cache(maxsize=1)
def get_customer_rankings() -> List[Dict[str, Any]]:
    X, _ = get_feature_matrix()
    model = get_trained_model()
    probabilities = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    frame = pd.DataFrame(
        {
            "customer_id": X.index.astype(str),
            "risk_score": probabilities,
        }
    )
    frame["threshold"] = get_model_metrics()["threshold"]
    frame["predicted_label"] = (frame["risk_score"] >= frame["threshold"]).astype(int)
    frame["risk_tier"] = frame["risk_score"].apply(risk_tier_for_probability)
    frame["rank"] = frame["risk_score"].rank(method="first", ascending=False).astype(int)
    return frame.sort_values("risk_score", ascending=False).to_dict(orient="records")


def get_customer_table(
    search: Optional[str] = None,
    risk_tier: Optional[str] = None,
    sort_by: str = "risk_score",
    sort_dir: str = "desc",
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    rankings = pd.DataFrame(get_customer_rankings())
    if search:
        rankings = rankings[rankings["customer_id"].str.contains(str(search), case=False, na=False)]
    if risk_tier:
        rankings = rankings[rankings["risk_tier"] == risk_tier]

    if sort_by not in rankings.columns:
        sort_by = "risk_score"

    ascending = sort_dir == "asc"
    rankings = rankings.sort_values(sort_by, ascending=ascending, kind="mergesort")
    rankings = rankings.reset_index(drop=True)
    rankings["rank"] = rankings.index + 1
    total = int(len(rankings))
    start = max((page - 1) * page_size, 0)
    end = start + page_size
    items = rankings.iloc[start:end].to_dict(orient="records")
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "search": search,
        "sort_by": sort_by,
        "sort_dir": sort_dir,
        "risk_tier": risk_tier,
    }


def get_feature_importance_from_csv(limit: int = 15) -> List[Dict[str, Any]]:
    csv_path = get_project_paths()["artifacts"] / "feature_importance.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature importance file not found: {csv_path}")

    ranking = pd.read_csv(csv_path).sort_values("importance", ascending=False).head(limit)
    return ranking.to_dict(orient="records")


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
    index_as_str = X.index.astype(str)
    if str(customer_id) not in index_as_str:
        raise KeyError(f"Unknown customer_id: {customer_id}")
    actual_index = X.index[index_as_str == str(customer_id)][0]
    row = X.loc[actual_index]
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
