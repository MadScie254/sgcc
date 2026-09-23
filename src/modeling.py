"""
SGCC Theft Detector - Modeling Module

XGBoost classifier tuned with Optuna on cross-validated PR-AUC.

Class imbalance (~8.5% theft) is left to the model rather than resampled:
on SGCC, SMOTE+ENN lowered cross-validated PR-AUC from ~0.53 to ~0.42. The
decision threshold is picked afterwards from out-of-fold predictions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

FIXED_PARAMS: Dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
}

DEFAULT_SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "n_estimators": {"low": 200, "high": 1200, "step": 100},
    "learning_rate": {"low": 0.01, "high": 0.1, "log": True},
    "max_depth": {"low": 3, "high": 9},
    "min_child_weight": {"low": 1, "high": 20, "log": True},
    "subsample": {"low": 0.6, "high": 1.0},
    "colsample_bytree": {"low": 0.3, "high": 1.0},
    "gamma": {"low": 0.0, "high": 5.0},
    "reg_alpha": {"low": 1e-3, "high": 10.0, "log": True},
    "reg_lambda": {"low": 1e-2, "high": 20.0, "log": True},
    "scale_pos_weight": {"low": 1.0, "high": 5.0},
}

_INT_PARAMS = {"n_estimators", "max_depth"}

SCORERS = {
    "average_precision": average_precision_score,
    "roc_auc": roc_auc_score,
}


def get_xgb_model(params: Optional[Dict[str, Any]] = None, random_state: int = 42, n_jobs: int = -1) -> xgb.XGBClassifier:
    """XGBClassifier with the fixed objective settings plus ``params``."""
    return xgb.XGBClassifier(**{**FIXED_PARAMS, "random_state": random_state, "n_jobs": n_jobs, **(params or {})})


def cross_val_proba(
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Out-of-fold positive-class probabilities for every row of X."""
    y_arr = np.asarray(y)
    oof = np.zeros(len(y_arr), dtype=float)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    for train_idx, val_idx in skf.split(X, y_arr):
        model = get_xgb_model(params, random_state=random_state)
        model.fit(X.iloc[train_idx], y_arr[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return oof


def _suggest(trial: optuna.Trial, name: str, spec: Dict[str, Any]):
    if name in _INT_PARAMS:
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))


def tune_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 40,
    cv: int = 5,
    random_state: int = 42,
    search_space: Optional[Dict[str, Dict[str, Any]]] = None,
    metric: str = "average_precision",
    timeout: Optional[int] = None,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Tune XGBoost hyperparameters with Optuna (TPE) on cross-validated ``metric``.

    Returns:
        (best_params, study). ``best_params`` holds only the tuned parameters.
    """
    if metric not in SCORERS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(SCORERS)}")
    scorer = SCORERS[metric]
    space = search_space or DEFAULT_SEARCH_SPACE

    def objective(trial: optuna.Trial) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in space.items()}
        oof = cross_val_proba(params, X, y, cv=cv, random_state=random_state)
        return float(scorer(y, oof))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    logger.info("Tuning XGBoost: %d trials, %d-fold CV on %s", n_trials, cv, metric)
    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        logger.info("Trial %d/%d: %s=%.4f (best %.4f)", trial.number + 1, n_trials, metric, trial.value, study.best_value)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[log_trial])
    logger.info("Best CV %s: %.4f with %s", metric, study.best_value, study.best_params)
    return dict(study.best_params), study


def select_threshold(y_true, proba, strategy: str = "f1", min_precision: float = 0.5) -> float:
    """
    Pick a decision threshold from (ideally out-of-fold) probabilities.

    strategy:
        "f1": maximise F1.
        "precision": the lowest threshold whose precision is at least ``min_precision``
            (maximises recall at that precision); falls back to "f1" if unreachable.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    if strategy == "precision":
        ok = np.flatnonzero(precision >= min_precision)
        if len(ok):
            return float(thresholds[ok[0]])
    elif strategy != "f1":
        raise ValueError(f"Unknown threshold strategy {strategy!r}")
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def save_model(model: xgb.XGBClassifier, model_path: str = "models/xgb_best.ubj") -> None:
    """Save in XGBoost's native format (no pickle, portable across library versions)."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    logger.info("Saved model to %s", model_path)


def load_model(model_path: str = "models/xgb_best.ubj") -> xgb.XGBClassifier:
    """Load a model saved by ``save_model``."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model
