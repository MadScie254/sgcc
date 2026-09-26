"""
SGCC Theft Detector - Shared set-up for the evidence scripts beyond the main training run

The scripts in ``scripts/`` that extend the study (resampling sensitivity, robustness,
literature metrics, history length, deep baseline) all need the same things: the
customers and labels, feature matrices on raw and cleaned readings, the training
split, and a way to fit a pipeline, calibrate it and set its threshold on validation
customers before scoring the test customers once. That protocol is the one
``src.train`` follows; this module repeats it without touching the served model.

Feature matrices take about a minute to build, so they are cached in ``data/cache``
(gitignored), keyed by the data file's size and modification time and the settings.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .calibration import apply_platt, fit_platt
from .data_loader import load_wide
from .eval import classification_metrics, inference_ms_per_customer, model_size_mb
from .experiment import ALL_CANDIDATES, fit_candidate
from .modeling import select_threshold
from .pipeline import features_for
from .train import load_config, split_customers

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Study:
    config: Dict[str, Any]
    wide: pd.DataFrame
    y: pd.Series
    X_by: Dict[str, pd.DataFrame]
    train_idx: pd.Index
    val_idx: pd.Index
    test_idx: pd.Index
    artifacts: Path = field(default=ROOT / "artifacts")

    def split(self, seed: int):
        """The 70/15/15 stratified customer split for ``seed`` (42 is the study's split)."""
        evaluation = self.config["evaluation"]
        return split_customers(self.y, float(evaluation["validation_size"]), float(evaluation["test_size"]), seed)

    def tuned_params(self, name: str) -> Dict[str, Any]:
        """Hyperparameters training tuned for ``name``, with the trees early stopping kept."""
        tuning = json.loads((self.artifacts / "tuning.json").read_text())
        curves = json.loads((self.artifacts / "learning_curves.json").read_text())
        return {**tuning[name]["params"], "n_estimators": curves[name]["best_iteration"]}


def _cache_key(path: Path, *settings: Any) -> str:
    stat = path.stat()
    blob = json.dumps([path.name, stat.st_size, int(stat.st_mtime), *settings], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def features_cached(wide: pd.DataFrame, preprocessing: str, config: Dict[str, Any], key: str,
                    feature_config: Optional[dict] = None) -> pd.DataFrame:
    feature_config = feature_config if feature_config is not None else config.get("features")
    cache = ROOT / "data" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps([key, preprocessing, config.get("preprocessing"), feature_config],
                                       sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = cache / f"features-{preprocessing}-{digest}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    X = features_for(wide, preprocessing, config.get("preprocessing"), feature_config)
    X.to_pickle(path)
    return X


def load_study(config_path: str = "config.yaml") -> Study:
    config = load_config(config_path)
    source = ROOT / config["data"]["training_data_path"]
    if not source.exists():
        raise SystemExit(f"Training data not found: {source}. Run: python scripts/download_data.py")
    wide, labels = load_wide(str(source))
    y = labels.reindex(wide.index).astype(int)
    key = _cache_key(source)
    X_by = {p: features_cached(wide, p, config, key) for p in ("raw", "clean")}
    evaluation = config["evaluation"]
    train_idx, val_idx, test_idx = split_customers(y, float(evaluation["validation_size"]),
                                                   float(evaluation["test_size"]), int(config["random_state"]))
    return Study(config, wide, y, X_by, train_idx, val_idx, test_idx, ROOT / config["paths"]["artifacts"])


def fit_and_score(study: Study, name: str, params: Optional[Dict[str, Any]], train_idx, val_idx, test_idx,
                  X: Optional[pd.DataFrame] = None, device: str = "cpu", random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Fit pipeline ``name`` on the training customers (tuned XGBoost stops early on validation),
    calibrate with Platt scaling on validation, choose the F1 threshold on the calibrated
    validation probabilities, then score the test customers once.
    """
    spec = ALL_CANDIDATES[name]
    config, evaluation = study.config, study.config["evaluation"]
    random_state = int(config["random_state"]) if random_state is None else random_state
    X = study.X_by[spec["preprocessing"]] if X is None else X
    y = study.y
    fitted = fit_candidate(name, X.loc[train_idx], y.loc[train_idx], params=params, X_val=X.loc[val_idx],
                           y_val=y.loc[val_idx], resampling_config=config.get("resampling"),
                           early_stopping_rounds=int(evaluation["early_stopping_rounds"]),
                           random_state=random_state, device=device)
    raw_val = fitted.predict(X.loc[val_idx])
    platt = fit_platt(raw_val, y.loc[val_idx])
    cal_val = apply_platt(raw_val, platt)
    threshold = select_threshold(y.loc[val_idx], cal_val, strategy=evaluation.get("threshold_strategy", "f1"),
                                 min_precision=float(evaluation.get("min_precision", 0.5)))
    X_test = fitted.treatment.transform(X.loc[test_idx])
    raw_test = fitted.model.predict_proba(X_test)[:, 1]
    cal_test = apply_platt(raw_test, platt)
    return {
        "name": name, "label": spec["label"], "threshold": float(threshold), "platt": platt,
        "validation_pr_auc": float(average_precision_score(y.loc[val_idx], raw_val)),
        "metrics": classification_metrics(y.loc[test_idx], cal_test, threshold),
        "training_time": round(fitted.fit_seconds, 2),
        "inference_ms_per_customer": inference_ms_per_customer(fitted.model, X_test),
        "model_size_mb": model_size_mb(fitted.model),
        "raw_val": raw_val, "cal_val": cal_val, "raw_test": raw_test, "cal_test": cal_test,
        "learning_curve": fitted.learning_curve,
    }


def public(result: Dict[str, Any]) -> Dict[str, Any]:
    """A fit_and_score result without its score arrays, for JSON."""
    return {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}


def save_extension_predictions(study: Study, split: str, index, labels: pd.Series,
                               columns: Dict[str, np.ndarray]) -> Path:
    """
    Add columns (raw score ``<name>_raw`` and calibrated ``<name>``) to
    artifacts/predictions/extensions_<split>.csv.gz, keeping columns other scripts wrote.
    """
    path = study.artifacts / "predictions" / f"extensions_{split}.csv.gz"
    frame = pd.DataFrame({"customer_id": pd.Index(index).astype(str), "label": labels.to_numpy(dtype=int)})
    if path.exists():
        old = pd.read_csv(path, dtype={"customer_id": str})
        if list(old["customer_id"]) == list(frame["customer_id"]):
            frame = old.drop(columns=[c for c in columns if c in old.columns])
    for name, values in columns.items():
        frame[name] = np.round(np.asarray(values, dtype=float), 6)
    frame.to_csv(path, index=False, compression="gzip")
    return path
