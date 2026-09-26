"""
How much meter history does the model need?

    python scripts/history_length.py          # ~15 min on 4 cores; --device cuda on a GPU
    python scripts/history_length.py --plot-only

A utility scoring its customers today has only the readings it has collected so far.
Here each customer is described by only the most recent 3, 6, 12, 18, 24 or all 34
months of readings (ending 31 October 2016); the monthly profile keeps as many months as
the window holds. For each window, standard XGBoost (the served pipeline, its tuned
hyperparameters held fixed) is refitted on the training customers with early stopping on
validation, calibrated and thresholded on validation customers, and scored once on the
same test customers. The 34-month window is the served model's setting.

Writes artifacts/history_length.json and docs/thesis-figures/fig-5-24-history-length.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import figstyle  # noqa: E402
from src.figstyle import BLUE, INK2, ORANGE, plt  # noqa: E402
from src.study import _cache_key, features_cached, fit_and_score, load_study  # noqa: E402

OUT = ROOT / "artifacts" / "history_length.json"
WINDOWS = (3, 6, 12, 18, 24, 34)


def window(wide: pd.DataFrame, months: int) -> pd.DataFrame:
    """The readings of the last ``months`` calendar months up to the last day in ``wide``."""
    end = wide.columns.max()
    start = end - pd.DateOffset(months=months) + pd.Timedelta(days=1)
    return wide.loc[:, wide.columns >= start]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detection quality by length of meter history")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if not args.plot_only:
        study = load_study()
        params = json.loads((study.artifacts / "tuning.json").read_text())["xgboost"]["params"]
        key = _cache_key(ROOT / study.config["data"]["training_data_path"])
        results = {}
        for months in WINDOWS:
            wide = window(study.wide, months)
            feature_config = {**study.config.get("features", {}),
                              "monthly_profile_months": min(months, int(study.config["features"]["monthly_profile_months"]))}
            X = features_cached(wide, "raw", study.config, f"{key}-last{months}m", feature_config)
            r = fit_and_score(study, "xgboost", params, study.train_idx, study.val_idx, study.test_idx, X=X,
                              device=args.device)
            results[str(months)] = {"months": months, "days": int(wide.shape[1]), "features": int(X.shape[1]),
                                    "first_day": str(wide.columns.min().date()), "threshold": r["threshold"],
                                    "validation_pr_auc": r["validation_pr_auc"],
                                    **{m: float(r["metrics"][m]) for m in ("pr_auc", "auc", "recall", "precision", "f1", "mcc")}}
            print(f"last {months} months ({wide.shape[1]} days): PR-AUC {results[str(months)]['pr_auc']:.4f}, "
                  f"recall {results[str(months)]['recall']:.4f}", flush=True)
        OUT.write_text(json.dumps({"windows": results, "end": str(study.wide.columns.max().date()),
                                   "pipeline": "xgboost", "test_customers": int(len(study.test_idx))}, indent=2))
    plot(json.loads(OUT.read_text()))


def plot(data: dict) -> None:
    rows = sorted(data["windows"].values(), key=lambda r: r["months"])
    months = [r["months"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for key, color, label in (("pr_auc", ORANGE, "PR-AUC"), ("recall", BLUE, "Recall at the validation threshold"),
                              ("precision", "#1baf7a", "Precision at the validation threshold")):
        vals = [r[key] for r in rows]
        ax.plot(months, vals, marker="o", color=color, label=label)
        for x, v in zip(months, vals):
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(months)
    ax.set_xlabel("Months of meter history used (most recent, ending 31 Oct 2016)")
    ax.set_ylabel("Test customers")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Detection quality by length of meter history (standard XGBoost)", pad=10)
    figstyle.save(fig, "fig-5-24-history-length")


if __name__ == "__main__":
    main()
