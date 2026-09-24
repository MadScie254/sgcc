from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.modeling import load_model

from . import data as data_service
from .config import BASE_DIR, get_config, get_project_paths
from .data import finite_or_none, get_feature_matrix


# Probabilities at or above this are "high" risk; between the model's decision
# threshold and this they are "medium"; below the threshold "low".
HIGH_RISK_PROBABILITY = 0.6

FEATURE_GROUPS = {
    "statistical": ("mean", "median", "std", "coef_var", "min", "max", "range", "skewness", "kurtosis", "q"),
    "missing_and_zero": ("missing", "zero", "longest", "first_obs", "last_obs"),
    "day_to_day": ("sudden_drop", "diff_", "autocorr"),
    "trend": ("slope", "last", "first", "yoy", "changepoint"),
    "calendar": ("weekday", "peak"),
    "monthly": ("monthly", "low_months", "max_monthly"),
    "monthly_profile": ("month_lag",),
}


@lru_cache(maxsize=1)
def get_model_path() -> Path:
    return BASE_DIR / get_config()["paths"]["model_file"]


@lru_cache(maxsize=1)
def get_trained_model():
    return load_model(str(get_model_path()))


@lru_cache(maxsize=1)
def get_feature_names() -> List[str]:
    names = get_trained_model().get_booster().feature_names
    if names:
        return list(names)
    X, _ = get_feature_matrix()
    return X.columns.tolist()


@lru_cache(maxsize=1)
def get_saved_metrics() -> Dict[str, Any]:
    metrics_path = get_project_paths()["artifacts"] / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def get_trained_threshold() -> float:
    """Threshold chosen during training (max out-of-fold F1)."""
    return float(get_saved_metrics().get("threshold", 0.5))


def _operating_threshold_path() -> Path:
    return get_project_paths()["state"] / "operating_threshold.json"


@lru_cache(maxsize=1)
def get_operating_threshold_override() -> Optional[float]:
    path = _operating_threshold_path()
    if not path.is_file():
        return None
    try:
        value = float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
    except (ValueError, KeyError, TypeError):
        return None
    return value if 0.0 < value < 1.0 else None


def set_operating_threshold(threshold: Optional[float]) -> None:
    """Publish an operating threshold for scoring, or None to return to the trained one."""
    path = _operating_threshold_path()
    if threshold is None:
        path.unlink(missing_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"threshold": float(threshold)}), encoding="utf-8")
    clear_caches()


def get_decision_threshold() -> float:
    """Threshold in service: a published operating threshold, else the trained one."""
    override = get_operating_threshold_override()
    return override if override is not None else get_trained_threshold()


def risk_tier_for_probability(probability: float, threshold: Optional[float] = None) -> str:
    threshold = get_decision_threshold() if threshold is None else threshold
    if probability >= max(HIGH_RISK_PROBABILITY, threshold):
        return "high"
    if probability >= threshold:
        return "medium"
    return "low"


def clear_caches() -> None:
    """Drop cached model, data, and derived results (e.g. after retraining)."""
    for module in (data_service, sys.modules[__name__]):
        for value in vars(module).values():
            if callable(getattr(value, "cache_clear", None)):
                value.cache_clear()


def _group_features(names: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {group: [] for group in FEATURE_GROUPS}
    for name in names:
        for group, prefixes in FEATURE_GROUPS.items():
            if name.startswith(prefixes):
                groups[group].append(name)
                break
        else:
            groups.setdefault("other", []).append(name)
    return {group: members for group, members in groups.items() if members}


@lru_cache(maxsize=1)
def get_model_config() -> Dict[str, Any]:
    config = get_config()
    return {
        "feature_groups": _group_features(get_feature_names()),
        "feature_parameters": dict(config.get("features", {})),
        "model": {**config.get("model", {}), "decision_threshold": get_decision_threshold()},
        "preprocessing": {"missing_values": "kept as NaN (handled natively by XGBoost)", "resampling": "none"},
        "evaluation": config.get("evaluation", {}),
    }


def _predict(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(get_trained_model().predict_proba(frame[get_feature_names()])[:, 1], dtype=float)


@lru_cache(maxsize=1)
def get_population_probabilities() -> pd.Series:
    X, _ = get_feature_matrix()
    return pd.Series(_predict(X), index=X.index.astype(str))


@lru_cache(maxsize=1)
def get_model_metrics() -> Dict[str, Any]:
    """Hold-out metrics from training plus live counts over the served (held-out) customers."""
    saved = get_saved_metrics()
    _, y = get_feature_matrix()
    probabilities = get_population_probabilities().to_numpy()
    threshold = get_decision_threshold()
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y.to_numpy(), predictions, labels=[0, 1]).ravel()
    support = saved.get("support") or {"class_0": int((y == 0).sum()), "class_1": int((y == 1).sum())}
    total_support = max(int(support.get("class_0", 0)) + int(support.get("class_1", 0)), 1)
    tiers = [risk_tier_for_probability(p, threshold) for p in probabilities]

    return {
        "threshold": threshold,
        "trained_threshold": get_trained_threshold(),
        "model_version": str(saved.get("model_version", get_config().get("model", {}).get("version", "unknown"))),
        "trained_at": saved.get("trained_at"),
        "metrics": {
            key: float(saved.get(key, 0.0))
            for key in ("recall", "precision", "f1", "accuracy", "auc", "pr_auc", "gmean", "mcc")
        },
        "support": {"class_0": int(support.get("class_0", 0)), "class_1": int(support.get("class_1", 0))},
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "customers_monitored": int(len(probabilities)),
        "flagged_today": int(predictions.sum()),
        "current_mean_probability": float(probabilities.mean()) if len(probabilities) else 0.0,
        "base_rate": float(int(support.get("class_1", 0)) / total_support),
        "risk_tier_distribution": {tier: tiers.count(tier) for tier in ("high", "medium", "low")},
    }


@lru_cache(maxsize=1)
def get_customer_rankings() -> List[Dict[str, Any]]:
    probabilities = get_population_probabilities()
    threshold = get_decision_threshold()
    frame = pd.DataFrame({"customer_id": probabilities.index, "risk_score": probabilities.to_numpy()})
    frame["threshold"] = threshold
    frame["predicted_label"] = (frame["risk_score"] >= threshold).astype(int)
    frame["risk_tier"] = [risk_tier_for_probability(p, threshold) for p in frame["risk_score"]]
    frame = frame.sort_values("risk_score", ascending=False, kind="mergesort").reset_index(drop=True)
    frame["rank"] = frame.index + 1
    return frame.to_dict(orient="records")


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
        rankings = rankings[rankings["customer_id"].str.contains(str(search), case=False, na=False, regex=False)]
    if risk_tier:
        rankings = rankings[rankings["risk_tier"] == risk_tier]

    if sort_by not in rankings.columns:
        sort_by = "risk_score"

    rankings = rankings.sort_values(sort_by, ascending=sort_dir == "asc", kind="mergesort")
    total = int(len(rankings))
    start = max((page - 1) * page_size, 0)
    items = rankings.iloc[start:start + page_size].to_dict(orient="records")
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
def get_shap_explainer():
    return shap.TreeExplainer(get_trained_model())


def _align_feature_row(features: Dict[str, float]) -> pd.DataFrame:
    # Features the caller leaves out are treated as missing, as in training.
    names = get_feature_names()
    return pd.DataFrame([[float(features.get(name, np.nan)) for name in names]], columns=names)


@lru_cache(maxsize=256)
def get_customer_feature_row(customer_id: str) -> pd.DataFrame:
    X, _ = get_feature_matrix()
    customer_id = str(customer_id)
    if customer_id not in X.index:
        raise KeyError(f"Unknown customer_id: {customer_id}")
    return X.loc[[customer_id], get_feature_names()].reset_index(drop=True)


def _shap_row(feature_frame: pd.DataFrame) -> tuple[float, np.ndarray]:
    explainer = get_shap_explainer()
    values = explainer.shap_values(feature_frame)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values, dtype=float).reshape(len(feature_frame), -1)
    base = explainer.expected_value
    if isinstance(base, (list, tuple, np.ndarray)):
        base = np.asarray(base).ravel()[-1]
    return float(base), values[0]


def _top_reasons(feature_frame: pd.DataFrame, shap_values: np.ndarray, top_n: int) -> List[Dict[str, Any]]:
    row = feature_frame.iloc[0]
    return [
        {
            "feature": feature_frame.columns[idx],
            "value": finite_or_none(row.iloc[idx]),
            "shap_value": float(shap_values[idx]),
        }
        for idx in np.argsort(np.abs(shap_values))[::-1][:top_n]
    ]


def _prediction_payload(feature_frame: pd.DataFrame, threshold: Optional[float]) -> Dict[str, Any]:
    threshold = get_decision_threshold() if threshold is None else threshold
    probability = float(_predict(feature_frame)[0])
    _, shap_values = _shap_row(feature_frame)
    return {
        "prediction": int(probability >= threshold),
        "probability": probability,
        "threshold": threshold,
        "risk_tier": risk_tier_for_probability(probability, threshold),
        "top_reasons": _top_reasons(feature_frame, shap_values, top_n=3),
    }


def predict_from_features(features: Dict[str, float], threshold: Optional[float] = None) -> Dict[str, Any]:
    return _prediction_payload(_align_feature_row(features), threshold)


def predict_for_customer(customer_id: str, threshold: Optional[float] = None) -> Dict[str, Any]:
    return {"customer_id": str(customer_id), **_prediction_payload(get_customer_feature_row(customer_id), threshold)}


def predict_frame(frame: pd.DataFrame) -> np.ndarray:
    """Probabilities for a frame of feature columns; absent or non-numeric values count as missing."""
    names = get_feature_names()
    aligned = frame.reindex(columns=names).apply(pd.to_numeric, errors="coerce")
    return _predict(aligned)


@lru_cache(maxsize=8)
def get_global_shap_sample(sample_count: int = 200) -> Dict[str, object]:
    X, _ = get_feature_matrix()
    sample_frame = X[get_feature_names()].sample(n=min(sample_count, len(X)), random_state=0)
    shap_values = get_shap_explainer().shap_values(sample_frame)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    return {
        "feature_names": sample_frame.columns.tolist(),
        "shap_values": np.asarray(shap_values, dtype=float).tolist(),
        "feature_values": [[finite_or_none(v) for v in row] for row in sample_frame.to_numpy()],
        "sample_count": int(len(sample_frame)),
    }


@lru_cache(maxsize=128)
def get_local_shap_details(customer_id: str) -> Dict[str, object]:
    feature_frame = get_customer_feature_row(customer_id)
    base_value, shap_values = _shap_row(feature_frame)
    return {
        "customer_id": str(customer_id),
        "feature_names": feature_frame.columns.tolist(),
        "feature_values": [finite_or_none(v) for v in feature_frame.iloc[0].tolist()],
        "shap_values": shap_values.tolist(),
        "base_value": base_value,
        "probability": float(_predict(feature_frame)[0]),
        "top_reasons": _top_reasons(feature_frame, shap_values, top_n=5),
    }


def threshold_preview(threshold: float) -> Dict[str, object]:
    """Metrics at ``threshold`` over the served customers, who were held out from training."""
    _, y = get_feature_matrix()
    y_true = y.to_numpy()
    probabilities = get_population_probabilities().reindex(y.index.astype(str)).to_numpy()
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "metrics": {
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "auc": float(roc_auc_score(y_true, probabilities)) if len(set(y_true)) > 1 else 0.0,
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


@lru_cache(maxsize=1)
def operating_curve() -> List[Dict[str, float]]:
    """Confusion counts, precision and recall across thresholds on the served (held-out) customers."""
    _, y = get_feature_matrix()
    y_true = y.to_numpy()
    probabilities = get_population_probabilities().reindex(y.index.astype(str)).to_numpy()
    points = []
    for threshold in np.round(np.arange(0.02, 0.99, 0.02), 2):
        predicted = probabilities >= threshold
        tp = int((predicted & (y_true == 1)).sum())
        fp = int((predicted & (y_true == 0)).sum())
        fn = int((~predicted & (y_true == 1)).sum())
        tn = int((~predicted & (y_true == 0)).sum())
        points.append({
            "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        })
    return points


@lru_cache(maxsize=1)
def score_distribution(bins: int = 20) -> Dict[str, Any]:
    """Histogram of theft probabilities, split by the dataset label."""
    _, y = get_feature_matrix()
    probabilities = get_population_probabilities().reindex(y.index.astype(str)).to_numpy()
    labels = y.to_numpy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    return {
        "edges": edges.round(4).tolist(),
        "honest": np.histogram(probabilities[labels == 0], bins=edges)[0].astype(int).tolist(),
        "theft": np.histogram(probabilities[labels == 1], bins=edges)[0].astype(int).tolist(),
        "threshold": get_decision_threshold(),
    }


@lru_cache(maxsize=1)
def get_flagged_drivers() -> Dict[str, Dict[str, Any]]:
    """Strongest SHAP driver (pushing towards theft) for every customer at or above the threshold."""
    probabilities = get_population_probabilities()
    flagged = probabilities[probabilities >= get_decision_threshold()].index
    if len(flagged) == 0:
        return {}
    X, _ = get_feature_matrix()
    frame = X.loc[flagged, get_feature_names()]
    values = get_shap_explainer().shap_values(frame)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values, dtype=float)
    drivers = {}
    for row, customer_id in enumerate(flagged):
        idx = int(np.argmax(values[row]))
        drivers[str(customer_id)] = {
            "feature": frame.columns[idx],
            "value": finite_or_none(frame.iat[row, idx]),
            "shap_value": float(values[row, idx]),
        }
    return drivers
