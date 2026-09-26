"""The research record: how the pipelines did on the test customers, and why.

Read-only views of what training and scripts/significance.py saved. The test
customers were scored once, after every choice (hyperparameters, early stopping,
calibration, thresholds, which pipeline to serve) was made on training and
validation customers. Every response names the population it describes.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from src.eval import classification_metrics

from .config import get_paths
from .data import get_pipeline_spec, load_predictions

LIMITATION = ("Customers were split at random, so the test set measures performance on unseen customers "
              "from the same utility and period. There is no evidence yet of performance on later periods "
              "or on another utility's customers.")


def _artifact(name: str) -> Dict[str, Any]:
    path = get_paths()["artifacts"] / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _comparison() -> Dict[str, Any]:
    path = get_paths()["baselines"]
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def test_population() -> Dict[str, Any]:
    y = load_predictions("test")["label"]
    return {"split": "test", "customers": int(len(y)), "theft": int(y.sum()),
            "description": f"test set: {len(y):,} customers ({int(y.sum()):,} thieves) held out from training and "
                           "validation, scored once", "limitation": LIMITATION}


def evaluation() -> Dict[str, Any]:
    """The served pipeline on the test customers at its trained threshold."""
    spec = get_pipeline_spec()
    test = load_predictions("test")
    result = classification_metrics(test["label"], test[spec["name"]], spec["threshold"])
    calibration = _artifact("calibration.json").get("pipelines", {}).get(spec["name"], {})
    return {
        "population": test_population(),
        "pipeline": spec["name"], "pipeline_label": spec["label"], "model_version": spec["model_version"],
        "threshold": spec["threshold"],
        "metrics": {key: result[key] for key in ("auc", "pr_auc", "precision", "recall", "f1", "mcc", "gmean", "specificity", "accuracy")},
        "confusion_matrix": result["confusion_matrix"],
        "calibration": {"raw": calibration.get("test", {}).get("raw"), "platt": calibration.get("test", {}).get("platt")},
    }


def model_comparison() -> List[Dict[str, Any]]:
    """
    Every pipeline on the same test customers at its own validation-chosen threshold, with
    computational cost and, when scripts/significance.py has run, the bootstrap interval of its
    difference from the proposed pipeline and the Holm-adjusted p-values.
    """
    significance = _artifact("significance.json")
    comparisons = significance.get("comparisons", {})
    intervals = significance.get("pipelines", {})
    keys = ("threshold", "auc", "pr_auc", "precision", "recall", "f1", "gmean", "mcc",
            "training_time", "inference_ms_per_customer", "model_size_mb")
    rows = []
    for name, result in _comparison().items():
        vs = comparisons.get(name, {})
        rows.append({
            "model": name, "label": result["label"], "served": bool(result.get("served")),
            "preprocessing": result["preprocessing"], "treatment": result["treatment"],
            **{key: float(result[key]) for key in keys},
            "pr_auc_ci": [intervals[name]["metrics"]["pr_auc"][k] for k in ("ci_low", "ci_high")] if name in intervals else None,
            "f1_ci": [intervals[name]["metrics"]["f1"][k] for k in ("ci_low", "ci_high")] if name in intervals else None,
            "p_value_pr_auc": vs.get("pr_auc", {}).get("p_holm"),
            "p_value_f1": vs.get("f1", {}).get("p_holm"),
            "p_value_mcnemar": vs.get("mcnemar", {}).get("p_holm"),
        })
    return rows


def significance() -> Dict[str, Any]:
    return _artifact("significance.json")


def operating_curve() -> Dict[str, Any]:
    """Precision and recall of the served pipeline on the test customers at thresholds 0.02-0.98."""
    spec = get_pipeline_spec()
    test = load_predictions("test")
    y, p = test["label"].to_numpy(dtype=int) == 1, test[spec["name"]].to_numpy(dtype=float)
    points = []
    for threshold in np.round(np.arange(0.02, 0.99, 0.02), 2):
        flagged = p >= threshold
        tp, fp = int((flagged & y).sum()), int((flagged & ~y).sum())
        fn, tn = int((~flagged & y).sum()), int((~flagged & ~y).sum())
        points.append({"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                       "precision": tp / (tp + fp) if tp + fp else 1.0, "recall": tp / (tp + fn) if tp + fn else 0.0})
    return {"population": test_population(), "threshold": spec["threshold"], "points": points}


def score_distribution() -> Dict[str, Any]:
    """20-bin histogram of the served pipeline's calibrated test probabilities, by true class."""
    spec = get_pipeline_spec()
    test = load_predictions("test")
    labels, p = test["label"].to_numpy(dtype=int), test[spec["name"]].to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, 21)
    return {
        "population": test_population(),
        "edges": edges.round(4).tolist(),
        "honest": np.histogram(p[labels == 0], bins=edges)[0].astype(int).tolist(),
        "theft": np.histogram(p[labels == 1], bins=edges)[0].astype(int).tolist(),
        "threshold": spec["threshold"],
    }


def calibration() -> Dict[str, Any]:
    """Brier score, log-loss and ECE before and after calibration, and the served pipeline's reliability table."""
    saved = _artifact("calibration.json")
    pipelines = saved.get("pipelines", {})
    served = saved.get("served")
    return {
        "population": test_population(),
        "method": saved.get("method"), "fitted_on": saved.get("fitted_on"), "served": served,
        "pipelines": [
            {"model": name, "label": _comparison().get(name, {}).get("label", name),
             **{f"{split}_{kind}": value[split][kind] for split in ("validation", "test") for kind in ("raw", "platt", "isotonic")}}
            for name, value in pipelines.items()
        ],
        "reliability": pipelines.get(served, {}).get("reliability_test", {}),
    }


def resampling_effect() -> Dict[str, Any]:
    """What SMOTE and SMOTE+ENN did to the training data (Objective 1), from training."""
    return _artifact("resampling.json")


def training_summary() -> Dict[str, Any]:
    saved = _artifact("metrics.json")
    spec = get_pipeline_spec()
    keys = ("pipeline", "pipeline_label", "model_version", "trained_at", "quick_mode", "device", "n_trials", "cv_metric",
            "cv_best_score", "cv_fold_scores", "train_customers", "validation_customers", "test_customers", "n_features")
    return {
        **{key: saved.get(key) for key in keys},
        "stages": saved.get("stages", []),
        "best_params": _artifact("best_params.json"),
        "provenance": spec.get("provenance", {}),
        "manifest": {key: value for key, value in _artifact("manifest.json").items() if key != "files"},
    }

