from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.data_loader import load_wide
from src.features import build_features_wide, compute_anomaly_features

from .config import BASE_DIR, get_config


def finite_or_none(value: Any) -> float | None:
    """JSON-safe float: NaN/inf (e.g. undefined features) become None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


@lru_cache(maxsize=1)
def get_wide_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Customer-by-day consumption matrix the API serves, with labels."""
    path = BASE_DIR / get_config()["data"]["serving_data_path"]
    return load_wide(str(path))


@lru_cache(maxsize=1)
def get_feature_matrix() -> Tuple[pd.DataFrame, pd.Series]:
    wide, labels = get_wide_data()
    X = build_features_wide(wide, get_config().get("features"))
    return X, labels.reindex(X.index).astype(int)


@lru_cache(maxsize=1)
def get_dataset_summary() -> Dict[str, object]:
    wide, labels = get_wide_data()
    X, _ = get_feature_matrix()
    values = wide.to_numpy()
    observed = ~np.isnan(values)
    class_distribution = labels.value_counts().sort_index().astype(int).to_dict()
    missing_values = {column: int(value) for column, value in X.isna().sum().items() if int(value) > 0}

    return {
        "dataset_path": get_config()["data"]["serving_data_path"],
        "total_customers": int(labels.shape[0]),
        "total_rows": int(values.size),
        "feature_count": int(X.shape[1]),
        "class_distribution": {str(key): int(value) for key, value in class_distribution.items()},
        "missing_values": missing_values,
        "zero_consumption_pct": float((values[observed] == 0).mean() * 100) if observed.any() else 0.0,
    }


@lru_cache(maxsize=128)
def get_feature_distribution(feature_name: str) -> Dict[str, object]:
    X, _ = get_feature_matrix()
    if feature_name not in X.columns:
        raise KeyError(f"Unknown feature: {feature_name}")

    series = pd.to_numeric(X[feature_name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {"feature": feature_name, "count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "bins": [], "counts": []}

    bin_count = min(20, max(5, int(np.sqrt(len(series)))))
    counts, bins = np.histogram(series.to_numpy(), bins=bin_count)
    return {
        "feature": feature_name,
        "count": int(series.shape[0]),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
        "bins": bins.round(6).tolist(),
        "counts": counts.astype(int).tolist(),
    }


@lru_cache(maxsize=1)
def get_correlation_matrix() -> Dict[str, object]:
    X, _ = get_feature_matrix()
    corr = X.select_dtypes(include=[np.number]).corr().fillna(0.0).round(6)
    return {"features": corr.columns.tolist(), "matrix": corr.values.tolist()}


@lru_cache(maxsize=128)
def get_customer_timeseries(customer_id: str) -> Dict[str, object]:
    wide, labels = get_wide_data()
    customer_id = str(customer_id)
    if customer_id not in wide.index:
        raise KeyError(f"Unknown customer_id: {customer_id}")

    threshold = float(get_config()["features"]["sudden_drop_threshold"])
    values = wide.loc[customer_id].to_numpy(dtype=float)
    dates = wide.columns.strftime("%Y-%m-%d") if isinstance(wide.columns, pd.DatetimeIndex) else [None] * len(values)

    points = []
    prev = None
    for day_index, (day, value) in enumerate(zip(dates, values)):
        score = 0.0
        if prev is not None and prev > 0 and not math.isnan(value):
            score = max(0.0, (prev - value) / prev)
        points.append({
            "day_index": day_index,
            "date": day,
            "consumption_kwh": finite_or_none(value),
            "sudden_drop": score > threshold,
            "anomaly_score": score,
        })
        prev = value

    return {
        "customer_id": customer_id,
        "label": int(labels.loc[customer_id]),
        "points": points,
        "summary": compute_anomaly_features(values, sudden_drop_threshold=threshold),
    }
