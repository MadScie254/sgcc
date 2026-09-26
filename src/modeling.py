"""
SGCC Theft Detector - Modeling Module

XGBoost classifier tuned with Optuna on the mean per-fold PR-AUC.

A data-level treatment (``src.resampling.Treatment``: none, SMOTE or SMOTE+ENN)
is fitted inside every cross-validation fold on that fold's training rows only,
so no synthetic or cleaned row ever leaks into the rows it is scored on.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .resampling import Treatment

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

Fold = Tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]

_INT_PARAMS = {"n_estimators", "max_depth"}

SCORERS = {
    "average_precision": average_precision_score,
    "roc_auc": roc_auc_score,
}


def get_xgb_model(params: Optional[Dict[str, Any]] = None, random_state: int = 42, n_jobs: int = -1,
                  device: str = "cpu") -> xgb.XGBClassifier:
    """
    XGBClassifier with the fixed objective settings plus ``params``. ``device="cuda"`` trains on an
    NVIDIA GPU; results differ slightly from CPU (floating-point order), so the CPU run is the reference.
    """
    return xgb.XGBClassifier(**{**FIXED_PARAMS, "random_state": random_state, "n_jobs": n_jobs, "device": device, **(params or {})})


def to_cpu(model):
    """Point a fitted XGBoost model at the CPU, so it scores CPU data without device transfers."""
    if isinstance(model, xgb.XGBModel):
        model.set_params(device="cpu")
    return model


def make_folds(X: pd.DataFrame, y: pd.Series, cv: int = 5, random_state: int = 42,
               treatment: str = "none", treatment_config: Optional[dict] = None) -> List[Fold]:
    """
    Stratified folds as (train rows after the treatment, their labels, prepared validation rows,
    validation positions). The treatment is fitted on each fold's training rows only.
    """
    y = pd.Series(np.asarray(y).astype(int), index=X.index)
    folds = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    for train_idx, val_idx in skf.split(X, y):
        fitted = Treatment(treatment, treatment_config, random_state)
        X_tr, y_tr = fitted.fit_resample(X.iloc[train_idx], y.iloc[train_idx])
        folds.append((X_tr, y_tr, fitted.transform(X.iloc[val_idx]), val_idx))
    return folds


def cross_val_scores(params: Dict[str, Any], folds: List[Fold], y, scorer=average_precision_score,
                     random_state: int = 42, device: str = "cpu") -> np.ndarray:
    """
    ``scorer`` on each fold's validation rows, one value per fold. The folds are scored separately
    and averaged by the caller: pooling out-of-fold probabilities from separately fitted models
    mixes their score scales, so a pooled PR-AUC does not measure any one model.
    """
    y = np.asarray(y).astype(int)
    scores = []
    for X_tr, y_tr, X_val, val_idx in folds:
        model = get_xgb_model(params, random_state=random_state, device=device)
        model.fit(X_tr, y_tr)
        scores.append(float(scorer(y[val_idx], model.predict_proba(X_val)[:, 1])))
    return np.asarray(scores)


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
    treatment: str = "none",
    treatment_config: Optional[dict] = None,
    initial_params: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Tune XGBoost hyperparameters with Optuna (TPE) on the mean of per-fold ``metric``,
    with ``treatment`` applied inside every fold.

    Returns:
        (best_params, study). ``best_params`` holds only the tuned parameters.
        ``initial_params`` fixes some parameters of the first trial (the rest are sampled),
        e.g. scale_pos_weight at the class ratio, as the proposal initialises it.
    """
    if metric not in SCORERS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(SCORERS)}")
    scorer = SCORERS[metric]
    space = search_space or DEFAULT_SEARCH_SPACE
    # Resampling does not depend on the hyperparameters, so each fold is treated once.
    folds = make_folds(X, y, cv=cv, random_state=random_state, treatment=treatment, treatment_config=treatment_config)

    def objective(trial: optuna.Trial) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in space.items()}
        scores = cross_val_scores(params, folds, y, scorer, random_state=random_state, device=device)
        trial.set_user_attr("fold_scores", [round(float(v), 6) for v in scores])
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    if initial_params:
        study.enqueue_trial(initial_params, skip_if_exists=True)
    logger.info("Tuning XGBoost (%s): %d trials, %d-fold CV on %s", treatment, n_trials, cv, metric)
    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        logger.info("Trial %d/%d: %s=%.4f (best %.4f)", trial.number + 1, n_trials, metric, trial.value, study.best_value)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[log_trial])
    logger.info("Best CV %s: %.4f with %s", metric, study.best_value, study.best_params)
    return dict(study.best_params), study


def fit_with_early_stopping(params: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series, rounds: int = 50,
                            random_state: int = 42, device: str = "cpu") -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Find the best number of trees by early stopping on validation PR-AUC (patience ``rounds``),
    then refit with exactly that many trees, so the saved model and its SHAP explanations
    use the same trees. Returns (model, learning curve of training and validation PR-AUC).
    """
    probe = get_xgb_model({**params, "early_stopping_rounds": rounds}, random_state=random_state, device=device)
    probe.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    curves = probe.evals_result()
    best = int(probe.best_iteration) + 1
    model = get_xgb_model({**params, "n_estimators": best}, random_state=random_state, device=device).fit(X_train, y_train)
    return model, {
        "best_iteration": best,
        "train_pr_auc": [float(v) for v in curves["validation_0"]["aucpr"]],
        "validation_pr_auc": [float(v) for v in curves["validation_1"]["aucpr"]],
    }


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
