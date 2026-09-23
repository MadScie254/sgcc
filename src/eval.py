"""
SGCC Theft Detector - Evaluation Module

Hold-out metrics for the XGBoost model and for simple baselines.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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


def evaluate_baselines(X_train, y_train, X_test, y_test, random_state: int = 42) -> Dict[str, Dict]:
    """Fit simple reference models on the same split and report hold-out metrics at threshold 0.5."""
    models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
                n_jobs=-1, random_state=random_state,
            ),
        ),
    }
    results = {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        metrics = classification_metrics(y_test, model.predict_proba(X_test)[:, 1])
        metrics["training_time"] = round(time.perf_counter() - start, 2)
        results[name] = metrics
        logger.info("Baseline %s: AUC %.4f PR-AUC %.4f", name, metrics["auc"], metrics["pr_auc"])
    return results


def save_json(payload: Dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved %s", output_path)
