"""
SGCC Theft Detector - The pipelines compared in the study

One definition of each candidate, shared by training (``src.train``), the
significance tests (``scripts/significance.py``) and the ablation study:

- proposed: cleaned series, SMOTE+ENN, tuned XGBoost (the proposal's framework)
- xgboost: raw series, no resampling, tuned XGBoost ("standard XGBoost")
- xgboost_default: raw series, no resampling, XGBoost with library defaults and no
  imbalance weighting (what class imbalance does to an untreated learner; RQ1, RQ3)
- random_forest_smote, logistic_regression_smote: the proposal's baselines

Only the two tuned XGBoost pipelines can be served; the one with the higher
validation PR-AUC is published.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .eval import baseline_model
from .modeling import fit_with_early_stopping, get_xgb_model, to_cpu
from .resampling import Treatment

CANDIDATES: Dict[str, Dict[str, Any]] = {
    "proposed": {"label": "SMOTE+ENN + XGBoost (proposed)", "preprocessing": "clean", "treatment": "smote_enn",
                 "learner": "xgboost", "tuned": True},
    "xgboost": {"label": "XGBoost, no resampling", "preprocessing": "raw", "treatment": "none",
                "learner": "xgboost", "tuned": True},
    "xgboost_default": {"label": "XGBoost, default settings", "preprocessing": "raw", "treatment": "none",
                        "learner": "xgboost", "tuned": False},
    "random_forest_smote": {"label": "Random forest + SMOTE", "preprocessing": "clean", "treatment": "smote",
                            "learner": "random_forest", "tuned": False},
    "logistic_regression_smote": {"label": "Logistic regression + SMOTE", "preprocessing": "clean", "treatment": "smote",
                                  "learner": "logistic_regression", "tuned": False},
}
SERVABLE = ("proposed", "xgboost")


@dataclass
class Fitted:
    name: str
    model: Any
    treatment: Treatment
    fit_seconds: float
    learning_curve: Optional[Dict[str, Any]] = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Theft probabilities for untreated rows (the treatment's median fill is applied)."""
        return self.model.predict_proba(self.treatment.transform(X))[:, 1]


def fit_candidate(name: str, X_train: pd.DataFrame, y_train: pd.Series, params: Optional[Dict[str, Any]] = None,
                  X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                  resampling_config: Optional[dict] = None, early_stopping_rounds: int = 50,
                  random_state: int = 42, device: str = "cpu") -> Fitted:
    """
    Treat the training rows and fit the candidate's learner. Tuned XGBoost stops early on the
    validation rows when they are given. ``fit_seconds`` covers resampling and fitting.
    XGBoost trains on ``device``; the fitted model is returned on the CPU, where it is scored.
    """
    spec = CANDIDATES[name]
    start = time.perf_counter()
    treatment = Treatment(spec["treatment"], resampling_config, random_state)
    X_tr, y_tr = treatment.fit_resample(X_train, y_train)
    curve = None
    if spec["learner"] != "xgboost":
        model = baseline_model(spec["learner"], random_state).fit(X_tr, y_tr)
    elif not spec["tuned"]:
        model = get_xgb_model(random_state=random_state, device=device).fit(X_tr, y_tr)
    elif X_val is not None:
        model, curve = fit_with_early_stopping(params or {}, X_tr, y_tr, treatment.transform(X_val), y_val,
                                               rounds=early_stopping_rounds, random_state=random_state, device=device)
    else:
        model = get_xgb_model(params, random_state=random_state, device=device).fit(X_tr, y_tr)
    return Fitted(name, to_cpu(model), treatment, time.perf_counter() - start, curve)
