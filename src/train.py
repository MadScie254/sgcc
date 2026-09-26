"""
SGCC Theft Detector - Training Pipeline

    python -m src.train                  # full run (Optuna trials from config.yaml)
    python -m src.train --quick          # smoke run on a sample with a few trials, written to artifacts/quick/
    python -m src.train --device cuda    # XGBoost on an NVIDIA GPU (the rest stays on the CPU)

Follows the proposal's protocol:

1. Load the SGCC matrix and clean each series (section 3.7).
2. Build the features on raw and on cleaned readings (section 3.8).
3. Split customers 70 / 15 / 15 into training, validation and test, stratified by label.
4. Measure what SMOTE+ENN does to the training data (Objective 1, section 3.9).
5. Tune the two XGBoost pipelines with Optuna, resampling inside every CV fold (section 3.10).
6. Fit every candidate in ``src.experiment``; tuned XGBoost stops early on validation.
   Each candidate is Platt-calibrated on validation (``src.calibration``), and its decision
   threshold maximises F1 on the calibrated validation probabilities.
7. Serve the XGBoost pipeline with the higher validation PR-AUC.
8. Score the test customers once: effectiveness, calibration and computational cost (section 3.11).
   Every candidate's validation and test predictions are saved, so the significance tests
   (scripts/significance.py) and the research API never refit or rescore anything.
9. Publish (``src.publish``): the model, its frozen pipeline spec (feature settings, feature
   list, calibration, threshold, code and data fingerprints), all results, and an unlabelled
   operational population, followed by a manifest of their SHA-256 hashes.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .calibration import apply_platt, calibration_metrics, fit_isotonic, fit_platt, reliability
from .data_loader import load_wide
from .eval import classification_metrics, feature_importance, inference_ms_per_customer, model_size_mb
from .experiment import CANDIDATES, SERVABLE, fit_candidate
from .features import build_features_wide
from .modeling import save_model, select_threshold, tune_xgb
from .preprocessing import clean_series
from .publish import Publisher, git_revision, library_versions, sha256
from .resampling import Treatment, diagnostics

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


def write_population(wide: pd.DataFrame, path: Path) -> None:
    """Write customers in the SGCC layout without labels (CONS_NO, one column per date)."""
    frame = wide.copy()
    if isinstance(frame.columns, pd.DatetimeIndex):
        frame.columns = frame.columns.strftime("%Y-%m-%d")
    frame.insert(0, "CONS_NO", frame.index.astype(str))
    frame.to_csv(path, index=False, float_format="%.6g", compression={"method": "gzip", "mtime": 0})
    logger.info("Wrote %d unlabelled customers to %s (%.1f MB)", len(frame), path, path.stat().st_size / 1e6)


def write_predictions(ids, y: pd.Series, raw: Dict[str, np.ndarray], calibrated: Dict[str, np.ndarray], path: Path) -> None:
    """One row per customer: label, then each pipeline's raw score and calibrated probability."""
    frame = pd.DataFrame({"customer_id": list(map(str, ids)), "label": y.loc[ids].astype(int).to_numpy()})
    for name in raw:
        frame[f"{name}_raw"] = raw[name]
        frame[name] = calibrated[name]
    frame.to_csv(path, index=False, float_format="%.8g", compression={"method": "gzip", "mtime": 0})


def calibration_report(y_val, y_test, raw_val, raw_test, platt: Dict[str, float]) -> Dict[str, Any]:
    """Brier, log-loss and ECE before and after Platt scaling (served) and isotonic regression (compared)."""
    iso = fit_isotonic(raw_val, y_val)
    report: Dict[str, Any] = {"platt": platt}
    for split, y, raw in (("validation", y_val, raw_val), ("test", y_test, raw_test)):
        report[split] = {
            "raw": calibration_metrics(y, raw),
            "platt": calibration_metrics(y, apply_platt(raw, platt)),
            "isotonic": calibration_metrics(y, iso.predict(raw)),
        }
    report["reliability_test"] = {"raw": reliability(y_test, raw_test), "platt": reliability(y_test, apply_platt(raw_test, platt))}
    return report


def split_customers(y: pd.Series, validation_size: float, test_size: float, random_state: int):
    """Stratified customer split into training, validation and test indices."""
    train_idx, rest_idx = train_test_split(y.index, test_size=validation_size + test_size, stratify=y, random_state=random_state)
    val_idx, test_idx = train_test_split(rest_idx, test_size=test_size / (validation_size + test_size),
                                         stratify=y.loc[rest_idx], random_state=random_state)
    return train_idx, val_idx, test_idx


def resampling_effect(X: pd.DataFrame, y: pd.Series, resampling_config: dict, random_state: int) -> Dict[str, Any]:
    """Class counts, separability and boundary noise before and after SMOTE and SMOTE+ENN (Objective 1)."""
    filled = X.fillna(X.median().fillna(0.0))
    scaler = StandardScaler().fit(filled)
    result: Dict[str, Any] = {"config": resampling_config, "before": diagnostics(filled, y, scaler, random_state=random_state)}
    for kind in ("smote", "smote_enn"):
        treatment = Treatment(kind, resampling_config, random_state)
        X_res, y_res = treatment.fit_resample(X, y)
        result[kind] = {"counts": treatment.stats_, "diagnostics": diagnostics(X_res, y_res, scaler, random_state=random_state)}
    return result


def train_pipeline(config_path: str = "config.yaml", quick_mode: bool = False,
                   data_path: Optional[str] = None, device: Optional[str] = None) -> Dict[str, Any]:
    config = load_config(config_path)
    device = device or str(config["model"].get("device", "cpu"))
    logger.info("XGBoost device: %s", device)
    random_state = int(config["random_state"])
    data_cfg, model_cfg, eval_cfg = config["data"], config["model"], config["evaluation"]
    resampling_cfg = config.get("resampling", {})
    stages = []
    clock = [time.perf_counter()]

    def mark(name: str) -> None:
        now = time.perf_counter()
        stages.append({"name": name, "seconds": round(now - clock[0], 2)})
        clock[0] = now

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
    mark("Load")

    cleaned, cleaning_log = clean_series(wide, config.get("preprocessing"))
    logger.info("Cleaning: %s", cleaning_log)
    mark("Clean")

    # 2. Features on raw and on cleaned readings (missingness always from the raw gaps)
    feature_cfg = config.get("features")
    X_by = {
        "raw": build_features_wide(wide, feature_cfg),
        "clean": build_features_wide(cleaned, feature_cfg, missing_mask=wide.isna().to_numpy()),
    }
    y = labels.reindex(wide.index).astype(int)
    logger.info("Feature matrix: %d customers x %d features", *X_by["raw"].shape)
    mark("Features")

    # 3. Split by customer
    train_idx, val_idx, test_idx = split_customers(
        y, float(eval_cfg["validation_size"]), float(eval_cfg["test_size"]), random_state)
    y_train, y_val, y_test = y.loc[train_idx], y.loc[val_idx], y.loc[test_idx]
    logger.info("Split: %d train / %d validation / %d test", len(train_idx), len(val_idx), len(test_idx))
    mark("Split")

    # 4. Effect of the treatment on the training data
    resampling = resampling_effect(X_by["clean"].loc[train_idx], y_train, resampling_cfg, random_state)
    mark("Resample")

    # 5. Tune both XGBoost pipelines
    tuning = model_cfg["quick_train"] if quick_mode else model_cfg["optuna"]
    optuna_cfg = model_cfg["optuna"]
    class_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    tuned: Dict[str, Dict[str, Any]] = {}
    for name in SERVABLE:
        spec = CANDIDATES[name]
        space = {**optuna_cfg["search_space"], "scale_pos_weight": optuna_cfg["scale_pos_weight"][spec["treatment"]]}
        initial = {"scale_pos_weight": min(class_ratio, space["scale_pos_weight"]["high"])} if spec["treatment"] == "none" else None
        params, study = tune_xgb(
            X_by[spec["preprocessing"]].loc[train_idx], y_train,
            n_trials=int(tuning["n_trials"]), cv=int(tuning["cv_folds"]), random_state=random_state,
            search_space=space, metric=optuna_cfg.get("metric", "average_precision"), timeout=optuna_cfg.get("timeout"),
            treatment=spec["treatment"], treatment_config=resampling_cfg, initial_params=initial, device=device,
        )
        tuned[name] = {"params": params, "cv_best_score": float(study.best_value),
                       "fold_scores": study.best_trial.user_attrs["fold_scores"],
                       "trials": [float(t.value) for t in study.trials if t.value is not None]}
    mark("Tune")

    # 6. Fit every candidate; calibration and thresholds from validation
    fitted, validation = {}, {}
    raw_val, cal_val = {}, {}
    for name, spec in CANDIDATES.items():
        X = X_by[spec["preprocessing"]]
        fitted[name] = fit_candidate(
            name, X.loc[train_idx], y_train, params=tuned.get(name, {}).get("params"),
            X_val=X.loc[val_idx], y_val=y_val, resampling_config=resampling_cfg,
            early_stopping_rounds=int(eval_cfg["early_stopping_rounds"]), random_state=random_state, device=device,
        )
        raw_val[name] = fitted[name].predict(X.loc[val_idx])
        platt = fit_platt(raw_val[name], y_val)
        cal_val[name] = apply_platt(raw_val[name], platt)
        validation[name] = {
            "pr_auc": float(average_precision_score(y_val, raw_val[name])),
            "platt": platt,
            "threshold": select_threshold(y_val, cal_val[name], strategy=eval_cfg.get("threshold_strategy", "f1"),
                                          min_precision=float(eval_cfg.get("min_precision", 0.5))),
        }
        logger.info("%s: validation PR-AUC %.4f", name, validation[name]["pr_auc"])
    winner = max(SERVABLE, key=lambda name: validation[name]["pr_auc"])
    logger.info("Serving %s", winner)
    mark("Validate")

    # 7. Test set, once
    comparison, calibration = {}, {}
    raw_test, cal_test = {}, {}
    for name, spec in CANDIDATES.items():
        X_test = fitted[name].treatment.transform(X_by[spec["preprocessing"]].loc[test_idx])
        raw_test[name] = fitted[name].model.predict_proba(X_test)[:, 1]
        cal_test[name] = apply_platt(raw_test[name], validation[name]["platt"])
        calibration[name] = calibration_report(y_val, y_test, raw_val[name], raw_test[name], validation[name]["platt"])
        comparison[name] = {
            **spec,
            **classification_metrics(y_test, cal_test[name], validation[name]["threshold"]),
            "at_threshold_0_5": classification_metrics(y_test, cal_test[name], 0.5),
            "validation_pr_auc": validation[name]["pr_auc"],
            "training_time": round(fitted[name].fit_seconds, 2),
            "inference_ms_per_customer": inference_ms_per_customer(fitted[name].model, X_test),
            "model_size_mb": model_size_mb(fitted[name].model),
            "served": name == winner,
        }
    mark("Evaluate")

    # 8. Publish: stage every file, move them into place, then write the manifest.
    #    A quick run publishes under artifacts/quick/ and never replaces the served model.
    root = _resolve(config["paths"]["artifacts"]) / "quick" if quick_mode else BASE_DIR
    rel = {"model": config["paths"]["model_file"], "artifacts": config["paths"]["artifacts"],
           "models": config["paths"]["models"], "population": data_cfg["serving_data_path"]}
    out = Publisher(root)
    served = fitted[winner]
    spec = CANDIDATES[winner]
    X_served = X_by[spec["preprocessing"]]
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_model(served.model, str(out.path(rel["model"])))
    out.json(f"{rel['artifacts']}/pipeline.json", {
        "name": winner,
        "label": spec["label"],
        "model_version": str(model_cfg.get("version", "unknown")),
        "trained_at": trained_at,
        "preprocessing": spec["preprocessing"],
        "cleaning": config.get("preprocessing") if spec["preprocessing"] == "clean" else None,
        "feature_config": feature_cfg,
        "features": list(X_served.columns),
        "impute": None if served.treatment.medians_ is None else {k: float(v) for k, v in served.treatment.medians_.items()},
        "calibration": {"method": "platt", **validation[winner]["platt"]},
        "threshold": validation[winner]["threshold"],
        "provenance": {
            "code": git_revision(BASE_DIR),
            "data_file": source.name,
            "data_sha256": sha256(source),
            "quick_mode": quick_mode,
            "device": device,
            "libraries": library_versions(),
        },
    })
    out.json(f"{rel['artifacts']}/calibration.json", {
        "method": "platt", "fitted_on": "validation", "served": winner, "pipelines": calibration})
    out.json(f"{rel['artifacts']}/best_params.json", tuned[winner]["params"])
    out.json(f"{rel['artifacts']}/tuning.json", tuned)
    out.json(f"{rel['models']}/baselines/comparison_results.json", comparison)
    out.json(f"{rel['artifacts']}/resampling.json", resampling)
    out.json(f"{rel['artifacts']}/preprocessing_log.json", cleaning_log)
    out.json(f"{rel['artifacts']}/learning_curves.json", {name: fitted[name].learning_curve for name in SERVABLE})
    feature_importance(served.model, X_served.columns).to_csv(out.path(f"{rel['artifacts']}/feature_importance.csv"), index=False)
    write_predictions(val_idx, y, raw_val, cal_val, out.path(f"{rel['artifacts']}/predictions/validation.csv.gz"))
    write_predictions(test_idx, y, raw_test, cal_test, out.path(f"{rel['artifacts']}/predictions/test.csv.gz"))

    # The operational population: a sample of test customers, without their labels.
    size = min(int(config["serving"]["population_customers"]), len(test_idx))
    population = train_test_split(test_idx, train_size=size, stratify=y_test, random_state=random_state)[0] \
        if size < len(test_idx) else test_idx
    write_population(wide.loc[population], out.path(rel["population"]))
    mark("Publish")

    metrics = {
        **{k: v for k, v in comparison[winner].items() if k not in CANDIDATES[winner]},
        "pipeline": winner,
        "pipeline_label": spec["label"],
        "model_version": str(model_cfg.get("version", "unknown")),
        "trained_at": trained_at,
        "quick_mode": quick_mode,
        "device": device,
        "cv_best_score": tuned[winner]["cv_best_score"],
        "cv_fold_scores": tuned[winner]["fold_scores"],
        "cv_metric": optuna_cfg.get("metric", "average_precision"),
        "n_trials": len(tuned[winner]["trials"]),
        "train_customers": int(len(train_idx)),
        "validation_customers": int(len(val_idx)),
        "test_customers": int(len(test_idx)),
        "population_customers": int(len(population)),
        "n_features": int(X_served.shape[1]),
        "calibration": {split: calibration[winner][split]["platt"] for split in ("validation", "test")},
        "stages": stages,
    }
    out.json(f"{rel['artifacts']}/metrics.json", metrics)
    manifest = out.commit({"model_version": metrics["model_version"], "pipeline": winner, "quick_mode": quick_mode})
    logger.info("Published %d files under %s", len(manifest["files"]), root)
    return {
        "served": winner,
        "published_to": str(root),
        "validation_pr_auc": {name: round(v["pr_auc"], 4) for name, v in validation.items()},
        "test": {name: {k: round(c[k], 4) for k in ("auc", "pr_auc", "recall", "precision", "f1", "mcc")}
                 for name, c in comparison.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SGCC theft detector")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--data", default=None, help="Override data.training_data_path")
    parser.add_argument("--quick", action="store_true", help="Sample customers, run few trials, publish to artifacts/quick/")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None,
                        help="Where XGBoost trains (default: model.device in config.yaml)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = train_pipeline(config_path=args.config, quick_mode=args.quick, data_path=args.data, device=args.device)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
