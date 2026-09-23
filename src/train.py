"""
SGCC Theft Detector - Training Pipeline

    python -m src.train            # full run (Optuna trials from config.yaml)
    python -m src.train --quick    # smoke run on a sample with a few trials

Steps: load data -> features -> stratified customer split -> Optuna tuning on
the training split -> threshold from out-of-fold predictions -> final fit ->
hold-out evaluation and baselines -> write model, metrics, and the demo
dataset the API serves (a sample of held-out customers).
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .data_loader import load_wide
from .eval import evaluate_baselines, evaluate_model, feature_importance, save_json
from .features import build_features_wide
from .modeling import cross_val_proba, get_xgb_model, save_model, select_threshold, tune_xgb

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        path = BASE_DIR / path
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else BASE_DIR / p


def write_demo_dataset(wide: pd.DataFrame, labels: pd.Series, path: Path) -> None:
    """Write customers in the SGCC layout (CONS_NO, FLAG, one column per date)."""
    frame = wide.copy()
    if isinstance(frame.columns, pd.DatetimeIndex):
        frame.columns = frame.columns.strftime("%Y-%m-%d")
    frame.insert(0, "FLAG", labels.reindex(frame.index).astype(int).values)
    frame.insert(0, "CONS_NO", frame.index.astype(str))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6g", compression={"method": "gzip", "mtime": 0})
    logger.info("Wrote %d demo customers to %s (%.1f MB)", len(frame), path, path.stat().st_size / 1e6)


def train_pipeline(config_path: str = "config.yaml", quick_mode: bool = False,
                   data_path: Optional[str] = None) -> Dict[str, Any]:
    config = load_config(config_path)
    random_state = int(config["random_state"])
    data_cfg, model_cfg, eval_cfg = config["data"], config["model"], config["evaluation"]
    paths = {key: _resolve(value) for key, value in config["paths"].items()}

    # 1. Data
    source = _resolve(data_path or data_cfg["training_data_path"])
    if not source.exists():
        raise FileNotFoundError(f"Training data not found: {source}. Run: python scripts/download_data.py")
    wide, labels = load_wide(str(source))

    if quick_mode:
        fraction = float(model_cfg["quick_train"]["sample_fraction"])
        keep, _ = train_test_split(labels.index, train_size=fraction, stratify=labels, random_state=random_state)
        wide, labels = wide.loc[keep], labels.loc[keep]
        logger.info("Quick mode: sampled %d customers", len(labels))

    # 2. Features
    X = build_features_wide(wide, config.get("features"))
    y = labels.reindex(X.index).astype(int)
    logger.info("Feature matrix: %d customers x %d features", *X.shape)

    # 3. Split by customer
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(eval_cfg["test_size"]), stratify=y, random_state=random_state,
    )

    # 4. Tune
    tuning = model_cfg["quick_train"] if quick_mode else model_cfg["optuna"]
    optuna_cfg = model_cfg["optuna"]
    best_params, study = tune_xgb(
        X_train, y_train,
        n_trials=int(tuning["n_trials"]),
        cv=int(tuning["cv_folds"]),
        random_state=random_state,
        search_space=optuna_cfg.get("search_space"),
        metric=optuna_cfg.get("metric", "average_precision"),
        timeout=optuna_cfg.get("timeout"),
    )

    # 5. Threshold from out-of-fold predictions on the training split
    oof = cross_val_proba(best_params, X_train, y_train, cv=int(tuning["cv_folds"]), random_state=random_state)
    threshold = select_threshold(
        y_train, oof,
        strategy=eval_cfg.get("threshold_strategy", "f1"),
        min_precision=float(eval_cfg.get("min_precision", 0.5)),
    )
    logger.info("Decision threshold: %.4f", threshold)

    # 6. Final fit and hold-out evaluation
    model = get_xgb_model(best_params, random_state=random_state)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
    metrics.update({
        "model_version": str(model_cfg.get("version", "unknown")),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick_mode": quick_mode,
        "cv_best_score": float(study.best_value),
        "cv_metric": optuna_cfg.get("metric", "average_precision"),
        "n_trials": len(study.trials),
        "train_customers": int(len(X_train)),
        "test_customers": int(len(X_test)),
        "n_features": int(X.shape[1]),
    })

    # 7. Persist artifacts
    save_model(model, str(paths["model_file"]))
    save_json(metrics, str(paths["artifacts"] / "metrics.json"))
    save_json(best_params, str(paths["artifacts"] / "best_params.json"))
    feature_importance(model, X.columns).to_csv(paths["artifacts"] / "feature_importance.csv", index=False)

    baselines = evaluate_baselines(X_train, y_train, X_test, y_test, random_state=random_state)
    save_json(baselines, str(paths["models"] / "baselines" / "comparison_results.json"))

    demo_size = min(int(config["serving"]["demo_customers"]), len(X_test))
    demo_ids, _ = train_test_split(
        X_test.index, train_size=demo_size, stratify=y_test, random_state=random_state,
    ) if demo_size < len(X_test) else (X_test.index, None)
    write_demo_dataset(wide.loc[demo_ids], labels, _resolve(data_cfg["serving_data_path"]))

    return {
        "best_score": float(study.best_value),
        "best_params": best_params,
        "threshold": threshold,
        "test_metrics": {k: metrics[k] for k in ("auc", "pr_auc", "recall", "precision", "f1")},
        "baselines": {name: {k: m[k] for k in ("auc", "pr_auc")} for name, m in baselines.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SGCC theft detector")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--data", default=None, help="Override data.training_data_path")
    parser.add_argument("--quick", action="store_true", help="Sample customers and run few trials")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = train_pipeline(config_path=args.config, quick_mode=args.quick, data_path=args.data)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
