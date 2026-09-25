from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.feature_catalog import feature_label, format_feature_value
from src.modeling import load_model

from . import data as data_service
from .config import get_config, get_paths
from .data import finite_or_none, get_feature_matrix, series_calendar
from .errors import NotFoundError

# Probabilities at or above this are "high" risk; between the decision threshold
# and this they are "medium"; below the threshold "low".
HIGH_RISK_PROBABILITY = 0.6


# ---------------------------------------------------------------------------
# Model and threshold
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_trained_model():
    return load_model(str(get_paths()["model_file"]))


@lru_cache(maxsize=1)
def get_feature_names() -> List[str]:
    return list(get_trained_model().get_booster().feature_names)


@lru_cache(maxsize=1)
def get_saved_metrics() -> Dict[str, Any]:
    path = get_paths()["artifacts"] / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def get_trained_threshold() -> float:
    """Threshold chosen during training (max out-of-fold F1)."""
    return float(get_saved_metrics().get("threshold", 0.5))


def _operating_threshold_path():
    return get_paths()["state"] / "operating_threshold.json"


@lru_cache(maxsize=1)
def _operating_threshold_override() -> Optional[float]:
    path = _operating_threshold_path()
    if not path.is_file():
        return None
    try:
        value = float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
    except (ValueError, KeyError, TypeError):
        return None
    return value if 0.0 < value < 1.0 else None


def get_decision_threshold() -> float:
    """Threshold in service: a published operating threshold, else the trained one."""
    override = _operating_threshold_override()
    return override if override is not None else get_trained_threshold()


def set_operating_threshold(threshold: Optional[float]) -> None:
    """Publish an operating threshold, or None to return to the trained one."""
    path = _operating_threshold_path()
    if threshold is None:
        path.unlink(missing_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"threshold": float(threshold)}), encoding="utf-8")
    clear_caches()


def risk_tier(probability: float, threshold: float) -> str:
    if probability >= max(HIGH_RISK_PROBABILITY, threshold):
        return "high"
    if probability >= threshold:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def predict_proba(features: pd.DataFrame) -> np.ndarray:
    """Probabilities for rows of model features; absent or non-numeric values count as missing."""
    aligned = features.reindex(columns=get_feature_names()).apply(pd.to_numeric, errors="coerce")
    impute = data_service.get_pipeline_spec().get("impute")
    if impute:  # the model was trained on gap-free (resampled) rows
        aligned = aligned.fillna(pd.Series(impute, dtype=float))
    return np.asarray(get_trained_model().predict_proba(aligned)[:, 1], dtype=float)


@lru_cache(maxsize=1)
def get_population_probabilities() -> pd.Series:
    """Theft probability of every served customer, highest first."""
    X, _ = get_feature_matrix()
    return pd.Series(predict_proba(X), index=X.index.astype(str)).sort_values(ascending=False, kind="mergesort")


def _confusion(y_true: np.ndarray, predicted: np.ndarray) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


@lru_cache(maxsize=1)
def get_model_metrics() -> Dict[str, Any]:
    """Hold-out metrics from training plus live counts over the served (held-out) customers."""
    saved = get_saved_metrics()
    _, y = get_feature_matrix()
    probabilities = get_population_probabilities().reindex(y.index.astype(str)).to_numpy()
    threshold = get_decision_threshold()
    tiers = pd.Series([risk_tier(p, threshold) for p in probabilities])
    return {
        "threshold": threshold,
        "trained_threshold": get_trained_threshold(),
        "model_version": str(saved.get("model_version", get_config()["model"]["version"])),
        "trained_at": saved.get("trained_at"),
        "metrics": {key: float(saved.get(key, 0.0)) for key in ("recall", "precision", "f1", "accuracy", "auc", "pr_auc", "mcc")},
        "confusion_matrix": _confusion(y.to_numpy(), (probabilities >= threshold).astype(int)),
        "customers_monitored": int(len(probabilities)),
        "flagged": int((probabilities >= threshold).sum()),
        "base_rate": float(y.mean()),
        "risk_tier_distribution": {tier: int((tiers == tier).sum()) for tier in ("high", "medium", "low")},
    }


def list_customers(search: Optional[str], tier: Optional[str], page: int, page_size: int) -> Dict[str, Any]:
    """Served customers ranked by theft probability."""
    probabilities = get_population_probabilities()
    threshold = get_decision_threshold()
    rows = [
        {"customer_id": cid, "rank": rank, "risk_score": float(p), "risk_tier": risk_tier(p, threshold)}
        for rank, (cid, p) in enumerate(probabilities.items(), start=1)
    ]
    if search:
        rows = [r for r in rows if search.lower() in r["customer_id"].lower()]
    if tier:
        rows = [r for r in rows if r["risk_tier"] == tier]
    start = (page - 1) * page_size
    return {"items": rows[start:start + page_size], "total": len(rows), "page": page, "page_size": page_size, "threshold": threshold}


def threshold_preview(threshold: float) -> Dict[str, Any]:
    """Metrics at any threshold over the served customers, who were held out from training."""
    _, y = get_feature_matrix()
    y_true = y.to_numpy()
    probabilities = get_population_probabilities().reindex(y.index.astype(str)).to_numpy()
    predicted = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "metrics": {
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, predicted)),
            "auc": float(roc_auc_score(y_true, probabilities)),
        },
        "confusion_matrix": _confusion(y_true, predicted),
    }


@lru_cache(maxsize=1)
def operating_curve() -> List[Dict[str, float]]:
    """Confusion counts, precision and recall at thresholds 0.02–0.98 over the served customers."""
    _, y = get_feature_matrix()
    y_true = y.to_numpy() == 1
    probabilities = get_population_probabilities().reindex(y.index.astype(str)).to_numpy()
    points = []
    for threshold in np.round(np.arange(0.02, 0.99, 0.02), 2):
        flagged = probabilities >= threshold
        tp, fp = int((flagged & y_true).sum()), int((flagged & ~y_true).sum())
        fn, tn = int((~flagged & y_true).sum()), int((~flagged & ~y_true).sum())
        points.append({
            "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        })
    return points


@lru_cache(maxsize=1)
def score_distribution() -> Dict[str, Any]:
    """20-bin histogram of theft probabilities, split by the dataset label."""
    _, y = get_feature_matrix()
    probabilities = get_population_probabilities().reindex(y.index.astype(str)).to_numpy()
    labels = y.to_numpy()
    edges = np.linspace(0.0, 1.0, 21)
    return {
        "edges": edges.round(4).tolist(),
        "honest": np.histogram(probabilities[labels == 0], bins=edges)[0].astype(int).tolist(),
        "theft": np.histogram(probabilities[labels == 1], bins=edges)[0].astype(int).tolist(),
        "threshold": get_decision_threshold(),
    }


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_shap_explainer():
    return shap.TreeExplainer(get_trained_model())


def _shap_matrix(features: pd.DataFrame) -> np.ndarray:
    values = get_shap_explainer().shap_values(features[get_feature_names()])
    if isinstance(values, list):
        values = values[-1]
    return np.asarray(values, dtype=float).reshape(len(features), -1)


def _base_value() -> float:
    base = get_shap_explainer().expected_value
    return float(np.asarray(base).ravel()[-1])


def reason(name: str, value: Any, shap_value: float) -> Dict[str, Any]:
    start, n_days = series_calendar()
    value = finite_or_none(value)
    return {
        "feature": name,
        "label": feature_label(name),
        "value": value,
        "display_value": format_feature_value(name, value, start, n_days),
        "shap_value": float(shap_value),
    }


def _reasons(features: pd.Series, contributions: np.ndarray, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    return [reason(features.index[i], features.iloc[i], contributions[i]) for i in order]


def _customer_features(customer_id: str) -> pd.DataFrame:
    X, _ = get_feature_matrix()
    if customer_id not in X.index:
        raise NotFoundError(f"Unknown customer_id: {customer_id}")
    return X.loc[[customer_id], get_feature_names()]


def explain_customer(customer_id: str) -> Dict[str, Any]:
    """Every feature's SHAP contribution (log-odds), largest first; base + sum = logit(probability)."""
    features = _customer_features(customer_id)
    contributions = _shap_matrix(features)[0]
    return {
        "customer_id": customer_id,
        "probability": float(predict_proba(features)[0]),
        "base_value": _base_value(),
        "contributions": _reasons(features.iloc[0], contributions),
    }


def _prediction(features: pd.DataFrame, threshold: Optional[float]) -> Dict[str, Any]:
    threshold = get_decision_threshold() if threshold is None else threshold
    probability = float(predict_proba(features)[0])
    return {
        "probability": probability,
        "prediction": int(probability >= threshold),
        "threshold": threshold,
        "risk_tier": risk_tier(probability, threshold),
        "reasons": _reasons(features.iloc[0], _shap_matrix(features)[0], top_n=3),
    }


def predict_customer(customer_id: str, threshold: Optional[float] = None) -> Dict[str, Any]:
    return {"customer_id": customer_id, **_prediction(_customer_features(customer_id), threshold)}


def predict_features(features: Dict[str, float], threshold: Optional[float] = None) -> Dict[str, Any]:
    names = get_feature_names()
    frame = pd.DataFrame([[features.get(name, np.nan) for name in names]], columns=names, dtype=float)
    return {"customer_id": None, **_prediction(frame, threshold)}


@lru_cache(maxsize=1)
def get_flagged_drivers() -> Dict[str, Dict[str, Any]]:
    """Strongest SHAP driver towards theft for every customer at or above the threshold."""
    probabilities = get_population_probabilities()
    flagged = probabilities[probabilities >= get_decision_threshold()].index
    if len(flagged) == 0:
        return {}
    X, _ = get_feature_matrix()
    frame = X.loc[flagged, get_feature_names()]
    values = _shap_matrix(frame)
    drivers = {}
    for row, customer_id in enumerate(flagged):
        i = int(np.argmax(values[row]))
        drivers[customer_id] = reason(frame.columns[i], frame.iat[row, i], values[row, i])
    return drivers


@lru_cache(maxsize=1)
def global_drivers(sample_size: int = 500, top_n: int = 15) -> Dict[str, Any]:
    """Mean |SHAP| per feature over a fixed sample of served customers, with the direction of effect."""
    X, _ = get_feature_matrix()
    sample = X[get_feature_names()].sample(n=min(sample_size, len(X)), random_state=0)
    values = _shap_matrix(sample)
    drivers = []
    for j, name in enumerate(sample.columns):
        column = sample[name].to_numpy(dtype=float)
        observed = ~np.isnan(column)
        # Sign of the covariance between feature value and contribution: does more of it push towards theft?
        if observed.sum() > 1 and np.std(column[observed]) > 0:
            direction = "higher" if np.cov(column[observed], values[observed, j])[0, 1] >= 0 else "lower"
        else:
            direction = "unclear"
        drivers.append({"feature": name, "label": feature_label(name), "mean_abs_shap": float(np.abs(values[:, j]).mean()), "risk_when": direction})
    drivers.sort(key=lambda d: d["mean_abs_shap"], reverse=True)
    return {"sample_size": int(len(sample)), "drivers": drivers[:top_n]}


# ---------------------------------------------------------------------------
# Training record
# ---------------------------------------------------------------------------

def _artifact(name: str) -> Dict[str, Any]:
    path = get_paths()["artifacts"] / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def model_comparison() -> List[Dict[str, Any]]:
    """
    Every pipeline compared in training, scored on the same test customers at its own
    validation-chosen threshold, with computational cost and, when scripts/significance.py
    has run, the Holm-adjusted paired t-test p-value against the proposed pipeline.
    """
    path = get_paths()["baselines"]
    results = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    tests = _artifact("significance.json").get("tests", {})
    keys = ("threshold", "auc", "pr_auc", "precision", "recall", "f1", "gmean", "mcc",
            "training_time", "inference_ms_per_customer", "model_size_mb")
    rows = []
    for name, result in results.items():
        p_values = {metric: tests.get(metric, {}).get(name, {}).get("t_test_p_holm") for metric in ("pr_auc", "f1")}
        rows.append({
            "model": name, "label": result["label"], "served": bool(result.get("served")),
            "preprocessing": result["preprocessing"], "treatment": result["treatment"],
            **{key: float(result[key]) for key in keys},
            "p_value_pr_auc": p_values["pr_auc"], "p_value_f1": p_values["f1"],
        })
    return rows


def resampling_effect() -> Dict[str, Any]:
    """What SMOTE and SMOTE+ENN did to the training data (Objective 1), from training."""
    return _artifact("resampling.json")


def training_summary() -> Dict[str, Any]:
    saved = get_saved_metrics()
    keys = ("pipeline", "pipeline_label", "model_version", "trained_at", "quick_mode", "n_trials", "cv_metric",
            "cv_best_score", "train_customers", "validation_customers", "test_customers", "n_features",
            "auc", "pr_auc", "precision", "recall", "f1", "threshold")
    return {
        **{key: saved.get(key) for key in keys},
        "stages": saved.get("stages", []),
        "best_params": _artifact("best_params.json"),
    }


# ---------------------------------------------------------------------------
# Second opinion: LIME (proposal section 3.12)
# ---------------------------------------------------------------------------

EXPLANATION_CHECK_TOP_N = 5


@lru_cache(maxsize=1)
def get_lime_explainer():
    """LIME explainer over the served customers (gaps filled with medians, as LIME needs)."""
    from lime.lime_tabular import LimeTabularExplainer

    X, _ = get_feature_matrix()
    background = X[get_feature_names()]
    medians = background.median().fillna(0.0)
    # Decile bins: with LIME's default quartiles its surrogate leans on magnitude features and agrees
    # with SHAP on fewer cases (8 vs 14 of the 30 highest-risk served customers).
    explainer = LimeTabularExplainer(background.fillna(medians).to_numpy(), feature_names=get_feature_names(),
                                     class_names=["honest", "theft"], mode="classification", discretizer="decile",
                                     random_state=0)
    return explainer, medians


@lru_cache(maxsize=256)
def explanation_check(customer_id: str) -> Dict[str, Any]:
    """
    Compare SHAP's and LIME's strongest features for one customer. They agree when at least
    3 of the top 5 coincide and SHAP's strongest feature is in LIME's top 5 with the same
    direction; otherwise the case should be reviewed by a person before acting on either.
    """
    names = get_feature_names()
    shap_top = explain_customer(customer_id)["contributions"][:EXPLANATION_CHECK_TOP_N]
    explainer, medians = get_lime_explainer()
    row = _customer_features(customer_id).iloc[0].fillna(medians)

    def class_probabilities(rows: np.ndarray) -> np.ndarray:
        theft = predict_proba(pd.DataFrame(rows, columns=names))
        return np.column_stack([1 - theft, theft])

    lime = explainer.explain_instance(row.to_numpy(dtype=float), class_probabilities,
                                      num_features=EXPLANATION_CHECK_TOP_N, num_samples=5000)
    lime_weights = {names[i]: float(w) for i, w in lime.as_map()[1]}
    lime_top = sorted(lime_weights, key=lambda f: -abs(lime_weights[f]))
    shared = [c["feature"] for c in shap_top if c["feature"] in lime_weights]
    strongest = shap_top[0]
    same_direction = strongest["feature"] in lime_weights and np.sign(lime_weights[strongest["feature"]]) == np.sign(strongest["shap_value"])
    agrees = len(shared) >= 3 and bool(same_direction)
    return {
        "customer_id": customer_id,
        "top_n": EXPLANATION_CHECK_TOP_N,
        "shap": [{"feature": c["feature"], "label": c["label"], "weight": c["shap_value"]} for c in shap_top],
        "lime": [{"feature": f, "label": feature_label(f), "weight": lime_weights[f]} for f in lime_top],
        "shared": shared,
        "agrees": agrees,
        "message": (f"SHAP and LIME agree on {len(shared)} of the top {EXPLANATION_CHECK_TOP_N} signals."
                    if agrees else
                    f"SHAP and LIME share only {len(shared)} of the top {EXPLANATION_CHECK_TOP_N} signals"
                    f"{'' if same_direction else ' and disagree on the strongest one'}: review this case manually."),
    }


_CACHED = (
    data_service.get_wide_data, data_service.get_feature_matrix, get_trained_model, get_feature_names,
    get_saved_metrics, _operating_threshold_override, get_population_probabilities, get_model_metrics,
    operating_curve, score_distribution, get_shap_explainer, get_flagged_drivers, global_drivers,
    data_service.get_pipeline_spec, get_lime_explainer, explanation_check,
)


def clear_caches() -> None:
    """Reload model, data and every derived result on next use."""
    for function in _CACHED:
        function.cache_clear()
