"""
SGCC Theft Detector - Probability calibration

A model trained on resampled or re-weighted rows ranks customers well, but its
scores overstate how often theft occurs: SMOTE+ENN trains on roughly one theft
per two honest customers, and ``scale_pos_weight`` has the same effect. Platt
scaling fits ``p = sigmoid(a * logit(score) + b)`` on the validation customers,
so a calibrated 0.3 means about 30 in 100 such customers were thieves.

With ``a > 0`` the map is strictly increasing: the ranking, ROC-AUC, PR-AUC and
the case queue are unchanged, and only the probability scale moves. The two
numbers are stored in JSON (no pickle). Isotonic regression is reported next to
it for comparison, but not served: it is a step function and ties customers.
"""

from typing import Dict, List

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

_EPS = 1e-6


def _logit(score) -> np.ndarray:
    p = np.clip(np.asarray(score, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def fit_platt(score, y) -> Dict[str, float]:
    """Platt parameters ``{"a", "b"}`` fitted on (score, label) pairs by maximum likelihood."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(score).reshape(-1, 1), np.asarray(y).astype(int))
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    if a <= 0:
        raise ValueError(f"Platt slope {a:.4f} is not positive: the scores do not rank thefts above honest customers")
    return {"a": a, "b": b}


def apply_platt(score, params: Dict[str, float]) -> np.ndarray:
    """Calibrated probabilities for raw model scores."""
    return 1.0 / (1.0 + np.exp(-(params["a"] * _logit(score) + params["b"])))


def fit_isotonic(score, y) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(score, dtype=float), np.asarray(y).astype(int))


def reliability(y, p, bins: int = 10) -> List[Dict[str, float]]:
    """Equal-width bins of predicted probability: mean prediction, observed theft rate and size."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        if mask.any():
            rows.append({"low": float(edges[b]), "high": float(edges[b + 1]), "count": int(mask.sum()),
                         "mean_predicted": float(p[mask].mean()), "observed_rate": float(y[mask].mean())})
    return rows


def calibration_metrics(y, p, bins: int = 10) -> Dict[str, float]:
    """Brier score, log-loss and expected calibration error (ECE, equal-width bins)."""
    y, p = np.asarray(y).astype(int), np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    table = reliability(y, p, bins)
    ece = sum(row["count"] * abs(row["mean_predicted"] - row["observed_rate"]) for row in table) / len(y)
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": float(ece),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
    }
