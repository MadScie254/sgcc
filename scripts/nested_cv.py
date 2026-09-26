"""
Nested cross-validation of the whole hybrid procedure, so that no reported number depends on
the study's test customers.

    pip install -r requirements-research.txt
    python scripts/nested_cv.py             # ~1 hour on 4 CPU cores; --device cuda on a GPU
    python scripts/nested_cv.py --plot-only

The 42,372 customers are split into five stratified outer folds. For each fold, the other four
folds are split 70:15 (stratified) into training and validation customers, and the procedure of
src.train is repeated from scratch on them: Optuna tuning of standard XGBoost (40 trials, 5-fold
PR-AUC on the training customers), early stopping, Platt calibration and the F1 threshold on
validation; the Wide & Deep CNN trained with early stopping on validation and Platt-calibrated;
the blend weight chosen on validation PR-AUC, the blend recalibrated and thresholded on
validation. The held-out fold is then scored once. Every customer is scored exactly once, by
models that never saw them and whose every choice was made without them.

Writes artifacts/nested_cv.json and docs/thesis-figures/fig-5-27-nested-cv.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

from src import figstyle  # noqa: E402
from src.calibration import apply_platt, fit_platt  # noqa: E402
from src.eval import classification_metrics  # noqa: E402
from src.figstyle import ORANGE, SLATE, plt  # noqa: E402
from src.modeling import select_threshold, tune_xgb  # noqa: E402
from src.stats import paired_comparison  # noqa: E402
from src.study import fit_and_score, load_study  # noqa: E402

OUT = ROOT / "artifacts" / "nested_cv.json"
METRICS = ("pr_auc", "auc", "recall", "precision", "f1", "mcc", "gmean")
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)
TEAL = "#1baf7a"
NAMES = ("hybrid", "cnn", "xgboost")
SMOKE = bool(os.environ.get("SGCC_SMOKE"))  # tiny run to test the script end to end
if SMOKE:
    OUT = Path(os.environ["SGCC_SMOKE"]) / "nested_cv.json"


def blend_weight(y_val, cnn_val, xgb_val) -> float:
    scores = [average_precision_score(y_val, w * cnn_val + (1 - w) * xgb_val) for w in WEIGHTS]
    return float(WEIGHTS[int(np.argmax(scores))])


def run_fold(study, values, mask, position, dev_idx, test_idx, fold, device):
    from src.sequence import predict_network, train_network

    config = study.config
    y = study.y
    evaluation, optuna_cfg = config["evaluation"], config["model"]["optuna"]
    val_share = float(evaluation["validation_size"]) / (1 - float(evaluation["test_size"]))
    train_idx, val_idx = train_test_split(dev_idx, test_size=val_share, stratify=y.loc[dev_idx],
                                          random_state=int(config["random_state"]))
    y_train = y.loc[train_idx]
    ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    space = {**optuna_cfg["search_space"], "scale_pos_weight": optuna_cfg["scale_pos_weight"]["none"]}
    trials, cv = (2, 2) if SMOKE else (int(optuna_cfg["n_trials"]), int(optuna_cfg["cv_folds"]))
    started = time.time()
    params, tuning = tune_xgb(study.X_by["raw"].loc[train_idx], y_train, n_trials=trials,
                              cv=cv, random_state=int(config["random_state"]),
                              search_space=space, metric=optuna_cfg.get("metric", "average_precision"),
                              initial_params={"scale_pos_weight": min(ratio, space["scale_pos_weight"]["high"])},
                              device=device)
    tune_seconds = time.time() - started
    xgb = fit_and_score(study, "xgboost", params, train_idx, val_idx, test_idx, device=device)

    rows = {k: np.array([position[c] for c in v]) for k, v in
            (("train", train_idx), ("val", val_idx), ("test", test_idx))}
    yy = y.to_numpy(dtype=int)
    y_val, y_test = yy[rows["val"]], yy[rows["test"]]
    started = time.time()
    model, history = train_network(values[rows["train"]], mask[rows["train"]], yy[rows["train"]],
                                   values[rows["val"]], mask[rows["val"]], y_val, device, 42,
                                   {**config["model"]["hybrid"].get("sequence", {}), **({"max_epochs": 1} if SMOKE else {})},
                                   log=lambda line: print(f"fold {fold} {line}", flush=True))
    cnn_seconds = time.time() - started
    raw_val = predict_network(model, values[rows["val"]], mask[rows["val"]], device)
    platt = fit_platt(raw_val, y_val)
    cnn_val = apply_platt(raw_val, platt)
    cnn_test = apply_platt(predict_network(model, values[rows["test"]], mask[rows["test"]], device), platt)

    weight = blend_weight(y_val, cnn_val, xgb["cal_val"])
    blend_val = weight * cnn_val + (1 - weight) * xgb["cal_val"]
    blend_test = weight * cnn_test + (1 - weight) * xgb["cal_test"]
    platt_h = fit_platt(blend_val, y_val)
    hybrid_val, hybrid_test = apply_platt(blend_val, platt_h), apply_platt(blend_test, platt_h)
    strategy = evaluation.get("threshold_strategy", "f1")
    thresholds = {"hybrid": float(select_threshold(y_val, hybrid_val, strategy=strategy)),
                  "cnn": float(select_threshold(y_val, cnn_val, strategy=strategy)),
                  "xgboost": float(xgb["threshold"])}
    scores = {"hybrid": hybrid_test, "cnn": cnn_test, "xgboost": xgb["cal_test"]}
    result = {
        "fold": fold, "customers": int(len(test_idx)), "theft": int(y_test.sum()),
        "train_customers": int(len(train_idx)), "validation_customers": int(len(val_idx)),
        "weight_cnn": weight, "thresholds": thresholds, "xgboost_params": params,
        "cv_best_score": float(tuning.best_value), "cnn_epochs": len(history),
        "seconds": {"tuning": round(tune_seconds, 1), "cnn": round(cnn_seconds, 1)},
        "validation_pr_auc": {"hybrid": float(average_precision_score(y_val, hybrid_val)),
                              "cnn": float(average_precision_score(y_val, cnn_val)),
                              "xgboost": float(average_precision_score(y_val, xgb["cal_val"]))},
        "test": {n: {m: float(v) for m, v in classification_metrics(y_test, scores[n], thresholds[n]).items()
                     if m in METRICS} for n in NAMES},
    }
    predictions = pd.DataFrame({"customer_id": pd.Index(test_idx).astype(str), "label": y_test, "fold": fold,
                                **{n: scores[n] for n in NAMES},
                                **{f"{n}_flag": (scores[n] >= thresholds[n]).astype(int) for n in NAMES}})
    return result, predictions


def summarise(folds, predictions) -> dict:
    y = predictions["label"].to_numpy(dtype=int)
    summary = {}
    for n in NAMES:
        vals = {m: np.array([f["test"][n][m] for f in folds]) for m in METRICS}
        summary[n] = {"mean": {m: float(v.mean()) for m, v in vals.items()},
                      "sd": {m: float(v.std(ddof=1)) for m, v in vals.items()},
                      "pooled_pr_auc": float(average_precision_score(y, predictions[n]))}
    scores = {n: predictions[n].to_numpy(dtype=float) for n in NAMES}
    flags = {n: predictions[f"{n}_flag"].to_numpy(dtype=bool) for n in NAMES}
    pipelines, comparisons = paired_comparison(y, scores, flags, "hybrid")
    wins = {o: int(sum(f["test"]["hybrid"]["pr_auc"] > f["test"][o]["pr_auc"] for f in folds)) for o in ("cnn", "xgboost")}
    return {"per_model": summary, "pooled": {"pipelines": pipelines, "comparisons": comparisons},
            "hybrid_higher_pr_auc_than": wins, "customers": int(len(y)), "theft": int(y.sum())}


def plot(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    folds = data["folds"]
    x = np.arange(len(folds))
    for name, color, label in (("hybrid", TEAL, "Hybrid: CNN + XGBoost"), ("cnn", SLATE, "Wide & Deep CNN"),
                               ("xgboost", ORANGE, "XGBoost, no resampling")):
        vals = [f["test"][name]["pr_auc"] for f in folds]
        ax.plot(x, vals, marker="o", color=color, label=f"{label} ({np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(f["fold"]) for f in folds])
    ax.set_xlabel("Outer fold (each scored once, by models that never saw it)")
    ax.set_ylabel("PR-AUC on the held-out fold")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    ax.set_title("Nested cross-validation of the whole procedure, all 42,372 customers", pad=10)
    if not SMOKE:
        figstyle.save(fig, "fig-5-27-nested-cv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested cross-validation of the hybrid procedure")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        from src.sequence import prepare, torch_available

        if not torch_available():
            raise SystemExit("PyTorch is not installed. Run: pip install -r requirements-research.txt")
        study = load_study()
        values, mask = prepare(study.wide, study.config.get("preprocessing"))
        position = {c: i for i, c in enumerate(study.wide.index)}
        y = study.y.loc[study.wide.index]
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(study.config["random_state"]))
        folds, frames = [], []
        for fold, (dev, test) in enumerate(outer.split(y.index, y), start=1):
            result, predictions = run_fold(study, values, mask, position, y.index[dev], y.index[test], fold, args.device)
            folds.append(result)
            frames.append(predictions)
            if SMOKE and fold == 2:
                break
            t = result["test"]
            print(f"fold {fold}: w={result['weight_cnn']:.2f} hybrid {t['hybrid']['pr_auc']:.4f}, "
                  f"cnn {t['cnn']['pr_auc']:.4f}, xgboost {t['xgboost']['pr_auc']:.4f}", flush=True)
        predictions = pd.concat(frames, ignore_index=True)
        data = {"outer_folds": 5, "folds": folds, "summary": summarise(folds, predictions)}
        OUT.write_text(json.dumps(data, indent=2))
        s = data["summary"]["per_model"]
        print({n: (round(s[n]["mean"]["pr_auc"], 4), round(s[n]["sd"]["pr_auc"], 4), round(s[n]["pooled_pr_auc"], 4))
               for n in NAMES}, "wins", data["summary"]["hybrid_higher_pr_auc_than"])
    plot(json.loads(OUT.read_text()))


if __name__ == "__main__":
    main()
