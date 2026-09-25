"""
SGCC Theft Detector - Evaluation Module

Hold-out metrics, the proposal's baseline classifiers, and computational cost.
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> Dict:
    """Threshold and ranking metrics for binary probabilities."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, proba)) if y_true.any() else 0.0,
        "gmean": float(np.sqrt(recall_score(y_true, y_pred, zero_division=0) * (tn / (tn + fp) if (tn + fp) else 0.0))),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "support": {"class_0": int((y_true == 0).sum()), "class_1": int((y_true == 1).sum())},
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> Dict:
    """Evaluate a fitted classifier on held-out data."""
    metrics = classification_metrics(y_test, model.predict_proba(X_test)[:, 1], threshold)
    logger.info(
        "Test AUC %.4f | PR-AUC %.4f | recall %.4f | precision %.4f | F1 %.4f @ %.3f",
        metrics["auc"], metrics["pr_auc"], metrics["recall"], metrics["precision"], metrics["f1"], threshold,
    )
    return metrics


def feature_importance(model, feature_names) -> pd.DataFrame:
    """Gain-based importance, normalised to sum to 1, sorted descending."""
    gain = model.get_booster().get_score(importance_type="gain")
    values = np.array([gain.get(name, 0.0) for name in feature_names], dtype=float)
    total = values.sum()
    return (
        pd.DataFrame({"feature": list(feature_names), "importance": values / total if total else values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def baseline_model(name: str, random_state: int = 42):
    """The proposal's reference classifiers; both are trained on SMOTE-treated rows."""
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=random_state)
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    raise ValueError(f"Unknown baseline {name!r}")


def model_size_mb(model) -> float:
    """Size of the fitted model as it would be stored."""
    if hasattr(model, "get_booster"):
        return len(model.get_booster().save_raw("ubj")) / 1e6
    return len(pickle.dumps(model)) / 1e6


def inference_ms_per_customer(model, X: pd.DataFrame, repeats: int = 3) -> float:
    """Best-of-``repeats`` wall time of scoring ``X``, per row, in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_proba(X)
        best = min(best, time.perf_counter() - start)
    return best / max(len(X), 1) * 1000


def save_json(payload: Dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved %s", output_path)
