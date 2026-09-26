"""
Is SMOTE+ENN itself the problem? A fair test and a sensitivity study (Objective 1, RQ2).

    python scripts/resampling_study.py                 # ~30 min on 4 cores; --device cuda on a GPU
    python scripts/resampling_study.py --skip-grid     # only the fair test
    python scripts/resampling_study.py --plot-only     # redraw the figure from the JSON

The proposed pipeline differs from standard XGBoost in two ways: cleaned readings and
SMOTE+ENN. This script separates them.

1. Fair test. "SMOTE+ENN + XGBoost, raw readings" (src.experiment.EXTRA_CANDIDATES) is
   tuned exactly as training tunes the other two XGBoost pipelines (Optuna, 40 trials,
   mean PR-AUC of five separately scored folds with SMOTE+ENN inside each fold, early
   stopping on validation), calibrated and thresholded on validation customers, and
   scored once on the same test customers. It is compared with standard XGBoost and
   with the proposed pipeline by paired stratified bootstrap and McNemar's test (Holm
   across the two comparisons). Its predictions are added to
   artifacts/predictions/extensions_{validation,test}.csv.gz.
2. Sensitivity grid. 5-fold CV on the 36,016 development customers (test customers
   untouched), raw features, standard XGBoost's tuned hyperparameters held fixed
   (scale_pos_weight = 1 when the data is resampled, since resampling already rebalances
   it). Treatments: none; SMOTE and SMOTE+ENN at several ratios and ENN neighbourhoods;
   SMOTE-Tomek, Borderline-SMOTE, ADASYN and random undersampling. Each fitted on each
   fold's training rows only.

Writes artifacts/resampling_study.json and docs/thesis-figures/fig-5-21-resampling-sensitivity.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src import figstyle  # noqa: E402
from src.experiment import ALL_CANDIDATES  # noqa: E402
from src.figstyle import BLUE, INK2, ORANGE, VIOLET, plt  # noqa: E402
from src.modeling import cross_val_scores, make_folds, tune_xgb  # noqa: E402
from src.stats import paired_comparison  # noqa: E402
from src.study import fit_and_score, load_study, public, save_extension_predictions  # noqa: E402

NAME = "xgboost_smote_enn_raw"
OUT = ROOT / "artifacts" / "resampling_study.json"

# (label, treatment, config overrides)
GRID = [
    ("No resampling", "none", {}),
    ("SMOTE 0.2", "smote", {"sampling_strategy": 0.2}),
    ("SMOTE 0.35", "smote", {"sampling_strategy": 0.35}),
    ("SMOTE 0.5", "smote", {"sampling_strategy": 0.5}),
    ("SMOTE 1.0", "smote", {"sampling_strategy": 1.0}),
    ("SMOTE+ENN 0.2", "smote_enn", {"sampling_strategy": 0.2}),
    ("SMOTE+ENN 0.5 (proposed setting)", "smote_enn", {"sampling_strategy": 0.5}),
    ("SMOTE+ENN 1.0", "smote_enn", {"sampling_strategy": 1.0}),
    ("SMOTE+ENN 0.5, ENN k = 5", "smote_enn", {"sampling_strategy": 0.5, "enn_n_neighbors": 5}),
    ("SMOTE-Tomek 0.5", "smote_tomek", {"sampling_strategy": 0.5}),
    ("Borderline-SMOTE 0.5", "borderline_smote", {"sampling_strategy": 0.5}),
    ("ADASYN 0.5", "adasyn", {"sampling_strategy": 0.5}),
    ("Random undersampling 0.5", "undersample", {"sampling_strategy": 0.5}),
]


def fair_test(study, device: str, trials: int) -> dict:
    config, spec = study.config, ALL_CANDIDATES[NAME]
    optuna_cfg = config["model"]["optuna"]
    space = {**optuna_cfg["search_space"], "scale_pos_weight": optuna_cfg["scale_pos_weight"][spec["treatment"]]}
    X, y = study.X_by[spec["preprocessing"]], study.y
    start = time.perf_counter()
    params, tuning = tune_xgb(X.loc[study.train_idx], y.loc[study.train_idx], n_trials=trials,
                              cv=int(optuna_cfg["cv_folds"]), random_state=int(config["random_state"]),
                              search_space=space, metric=optuna_cfg.get("metric", "average_precision"),
                              treatment=spec["treatment"], treatment_config=config.get("resampling"), device=device)
    tuning_seconds = time.perf_counter() - start
    result = fit_and_score(study, NAME, params, study.train_idx, study.val_idx, study.test_idx, device=device)
    save_extension_predictions(study, "validation", study.val_idx, y.loc[study.val_idx],
                               {f"{NAME}_raw": result["raw_val"], NAME: result["cal_val"]})
    save_extension_predictions(study, "test", study.test_idx, y.loc[study.test_idx],
                               {f"{NAME}_raw": result["raw_test"], NAME: result["cal_test"]})

    # Paired comparison on the test customers with the two pipelines it separates.
    saved = pd.read_csv(study.artifacts / "predictions" / "test.csv.gz", dtype={"customer_id": str}).set_index("customer_id")
    saved = saved.loc[pd.Index(study.test_idx).astype(str)]
    thresholds = json.loads((ROOT / "models" / "baselines" / "comparison_results.json").read_text())
    y_test = y.loc[study.test_idx].to_numpy(dtype=int)
    assert (saved["label"].to_numpy(dtype=int) == y_test).all(), "saved predictions do not match the split"
    scores = {NAME: result["cal_test"], "xgboost": saved["xgboost"].to_numpy(), "proposed": saved["proposed"].to_numpy()}
    flags = {NAME: scores[NAME] >= result["threshold"],
             "xgboost": scores["xgboost"] >= thresholds["xgboost"]["threshold"],
             "proposed": scores["proposed"] >= thresholds["proposed"]["threshold"]}
    pipelines, comparisons = paired_comparison(y_test, scores, flags, NAME)
    return {
        "pipeline": NAME, "label": spec["label"],
        "tuning": {"params": params, "cv_best_score": float(tuning.best_value),
                   "fold_scores": tuning.best_trial.user_attrs["fold_scores"],
                   "trials": [float(t.value) for t in tuning.trials if t.value is not None],
                   "seconds": round(tuning_seconds, 1)},
        "result": public(result),
        "bootstrap": {"reference": NAME, "resamples": 10_000, "pipelines": pipelines, "comparisons": comparisons},
    }


def grid(study, device: str) -> dict:
    # Development customers in file order, as scripts/ablation.py has them, so the folds are the same.
    y = study.y.drop(index=study.test_idx)
    X = study.X_by["raw"].loc[y.index]
    base = study.tuned_params("xgboost")
    base_config = study.config.get("resampling") or {}
    variants = {}
    for label, treatment, overrides in GRID:
        params = base if treatment == "none" else {**base, "scale_pos_weight": 1.0}
        config = {**base_config, **overrides}
        start = time.perf_counter()
        folds = make_folds(X, y, cv=5, random_state=42, treatment=treatment, treatment_config=config)
        rows = [len(f[1]) for f in folds]
        theft_share = [float(np.mean(f[1])) for f in folds]
        scores = {"roc_auc": cross_val_scores(params, folds, y, roc_auc_score, device=device),
                  "pr_auc": cross_val_scores(params, folds, y, average_precision_score, device=device)}
        variants[label] = {
            "treatment": treatment, "config": config if treatment != "none" else {},
            **{k: float(v.mean()) for k, v in scores.items()},
            **{f"{k}_sd": float(v.std(ddof=1)) for k, v in scores.items()},
            "folds": {k: [round(float(x), 6) for x in v] for k, v in scores.items()},
            "training_rows": int(np.mean(rows)), "training_theft_share": round(float(np.mean(theft_share)), 4),
            "seconds": round(time.perf_counter() - start, 1),
        }
        print(f"{label}: PR-AUC {variants[label]['pr_auc']:.4f} ± {variants[label]['pr_auc_sd']:.4f}", flush=True)
    return {"customers": int(len(y)), "base_rate": float(y.mean()), "folds": 5,
            "hyperparameters": "standard XGBoost's tuned values and trees; scale_pos_weight = 1 when resampled",
            "variants": variants}


def plot(data: dict) -> None:
    variants = data["grid"]["variants"]
    labels = list(variants)
    means = np.array([variants[k]["pr_auc"] for k in labels])
    sds = np.array([variants[k]["pr_auc_sd"] for k in labels])
    reference = variants["No resampling"]["pr_auc"]
    colors = [ORANGE if variants[k]["treatment"] == "none" else
              BLUE if "proposed" in k else VIOLET if variants[k]["treatment"] == "smote_enn" else "#8a88c4"
              for k in labels]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    y = np.arange(len(labels))[::-1]
    ax.axvline(reference, color=ORANGE, lw=1, ls="--", zorder=1)
    for yi, m, s, c in zip(y, means, sds, colors):
        ax.errorbar(m, yi, xerr=s, fmt="o", color=c, ecolor=c, elinewidth=1.4, capsize=3, markersize=6, zorder=3)
        # Values in a column to the right of the plot, clear of the marks.
        ax.text(1.02, yi, f"{m:.3f} ± {s:.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK2)
    ax.text(1.02, len(labels) - 0.2, "PR-AUC", transform=ax.get_yaxis_transform(), fontsize=8, color=INK2,
            fontweight="bold", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlim(float((means - sds).min()) - 0.01, float((means + sds).max()) + 0.01)
    ax.set_xlabel("PR-AUC, mean ± SD over 5 folds of the development customers (dashed: no resampling)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Resampling choices against no resampling\n(raw readings, fixed hyperparameters)", pad=10)
    figstyle.save(fig, "fig-5-21-resampling-sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair SMOTE+ENN test and resampling sensitivity")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--trials", type=int, default=None, help="Optuna trials (default: config, 40)")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-fair-test", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    if not args.plot_only:
        study = load_study()
        if not args.skip_fair_test:
            trials = args.trials or int(study.config["model"]["optuna"]["n_trials"])
            data["fair_test"] = fair_test(study, args.device, trials)
            OUT.write_text(json.dumps(data, indent=2))
            m = data["fair_test"]["result"]["metrics"]
            print(f"fair test: PR-AUC {m['pr_auc']:.4f}, recall {m['recall']:.4f}, precision {m['precision']:.4f}")
        if not args.skip_grid:
            data["grid"] = grid(study, args.device)
            OUT.write_text(json.dumps(data, indent=2))
    if "grid" in data:
        plot(data)


if __name__ == "__main__":
    main()
