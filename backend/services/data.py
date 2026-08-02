from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data_loader import load_processed_features, load_raw
from src.features import build_features, compute_anomaly_features

from .config import BASE_DIR, get_config


@lru_cache(maxsize=1)
def get_long_data() -> Tuple[pd.DataFrame, pd.Series]:
    config = get_config()
    raw_path = config["data"]["raw_data_path"]
    return load_raw(raw_path)


@lru_cache(maxsize=1)
def get_feature_matrix() -> Tuple[pd.DataFrame, pd.Series]:
    config = get_config()
    features_path = Path(config["data"]["processed_features_path"])

    if features_path.exists():
        return load_processed_features(str(features_path))

    df_long, labels = get_long_data()
    feature_config = {
        "sudden_drop_threshold": config["features"]["sudden_drop_threshold"],
        "peak_day_percentile": config["features"]["peak_day_percentile"],
        "missing_sequence_threshold": config["features"]["missing_sequence_threshold"],
    }
    return build_features(df_long, labels, config=feature_config)


@lru_cache(maxsize=1)
def get_dataset_summary() -> Dict[str, object]:
    config = get_config()
    df_long, labels = get_long_data()
    X, _ = get_feature_matrix()
    class_distribution = labels.value_counts().sort_index().astype(int).to_dict()
    zero_consumption_pct = float((df_long["consumption_kwh"] == 0).mean() * 100)
    missing_values = {
        column: int(value)
        for column, value in X.isna().sum().items()
        if int(value) > 0
    }

    return {
        "dataset_path": config["data"]["raw_data_path"],
        "total_customers": int(labels.shape[0]),
        "total_rows": int(df_long.shape[0]),
        "feature_count": int(X.shape[1]),
        "class_distribution": {str(key): int(value) for key, value in class_distribution.items()},
        "missing_values": missing_values,
        "zero_consumption_pct": zero_consumption_pct,
    }


@lru_cache(maxsize=128)
def get_feature_distribution(feature_name: str) -> Dict[str, object]:
    X, _ = get_feature_matrix()
    if feature_name not in X.columns:
        raise KeyError(f"Unknown feature: {feature_name}")

    series = pd.to_numeric(X[feature_name], errors="coerce").dropna()
    if series.empty:
        return {
            "feature": feature_name,
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "bins": [],
            "counts": [],
        }

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
    numeric_X = X.select_dtypes(include=[np.number])
    corr = numeric_X.corr().fillna(0.0).round(6)
    return {
        "features": corr.columns.tolist(),
        "matrix": corr.values.tolist(),
    }


@lru_cache(maxsize=128)
def get_customer_timeseries(customer_id: str) -> Dict[str, object]:
    config = get_config()
    df_long, labels = get_long_data()
    customer_mask = df_long["customer_id"].astype(str) == str(customer_id)
    customer_df = df_long.loc[customer_mask].sort_values("day_index").copy()

    if customer_df.empty:
        raise KeyError(f"Unknown customer_id: {customer_id}")

    threshold = float(config["features"]["sudden_drop_threshold"])
    values = customer_df["consumption_kwh"].astype(float).fillna(0.0).to_numpy()
    prev_value = None
    sudden_drop_flags = []
    anomaly_scores = []

    for value in values:
        if prev_value is None or prev_value <= 0:
            sudden_drop_flags.append(False)
            anomaly_scores.append(0.0)
        else:
            anomaly_score = max(0.0, float((prev_value - value) / prev_value))
            sudden_drop_flags.append(anomaly_score > threshold)
            anomaly_scores.append(anomaly_score)
        prev_value = value

    customer_df["sudden_drop"] = sudden_drop_flags
    customer_df["anomaly_score"] = anomaly_scores
    summary = compute_anomaly_features(values, sudden_drop_threshold=threshold)
    label_value = int(labels.loc[str(customer_id)]) if str(customer_id) in labels.index else None

    return {
        "customer_id": str(customer_id),
        "label": label_value,
        "points": customer_df[["day_index", "consumption_kwh", "sudden_drop", "anomaly_score"]].to_dict(orient="records"),
        "summary": summary,
    }
