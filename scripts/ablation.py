"""
Ablation study: what each change to the pipeline is worth.

    pip install matplotlib imbalanced-learn
    python scripts/ablation.py              # ~10 min on 4 cores
    python scripts/ablation.py --plot-only  # redraw the figure from artifacts/ablation.json

5-fold stratified cross-validation on all 42,372 customers, with the tuned
XGBoost hyperparameters (artifacts/best_params.json) held fixed across variants:

  A. 17 original features, day columns in file order (lexicographic dates, the original bug)
  B. 17 original features, days sorted chronologically
  C. 85 features, day columns in file order
  D. 85 features (this repo's feature set)
  E. 85 features + SMOTE-ENN resampling inside each training fold (the original pipeline)

Writes artifacts/ablation.json and docs/thesis-figures/fig-5-6-ablation.png/.pdf.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from imblearn.combine import SMOTEENN  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from imblearn.under_sampling import EditedNearestNeighbours  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

from src.data_loader import load_wide  # noqa: E402
from src.features import build_features_wide  # noqa: E402
from src.modeling import get_xgb_model  # noqa: E402

LEGACY = ["mean", "median", "std", "coef_var", "min", "max", "range", "skewness", "slope_full", "slope_last_30d",
          "slope_last_90d", "zero_day_count", "sudden_drop_count", "weekday_vs_weekend_ratio", "peak_day_ratio",
          "autocorr_lag1", "missing_sequences_count"]


def smote_enn_fit(params, X, y):
    """Scale, resample with SMOTE-ENN as the original pipeline did, and fit (NaN filled with 0 as it did)."""
    Xf = X.fillna(0.0)
    scaler = MinMaxScaler().fit(Xf)
    Xr, yr = SMOTEENN(
        smote=SMOTE(k_neighbors=7, sampling_strategy=0.8, random_state=42),
        enn=EditedNearestNeighbours(n_neighbors=5, sampling_strategy="all"), random_state=42,
    ).fit_resample(scaler.transform(Xf), y)
    model = get_xgb_model(params).fit(Xr, yr)
    return lambda Xv: model.predict_proba(scaler.transform(Xv.fillna(0.0)))[:, 1]


def cross_validate(X, y, params, resample=False):
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        if resample:
            oof[va] = smote_enn_fit(params, X.iloc[tr], y[tr])(X.iloc[va])
        else:
            oof[va] = get_xgb_model(params).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    return {"roc_auc": float(roc_auc_score(y, oof)), "pr_auc": float(average_precision_score(y, oof))}


def main() -> None:
    if "--plot-only" in sys.argv:
        plot(json.loads((ROOT / "artifacts" / "ablation.json").read_text()), 3615 / 42372)
        return
    data_path = ROOT / "data" / "sgcc_full.csv"
    if not data_path.exists():
        sys.exit("data/sgcc_full.csv not found: run python scripts/download_data.py first")
    params = json.loads((ROOT / "artifacts" / "best_params.json").read_text())

    wide, labels = load_wide(str(data_path))
    y = labels.to_numpy()
    X = build_features_wide(wide)

    # Undo the chronological sort: the raw file's header order ("2014/1/1", "2014/1/10", ...).
    raw_header = pd.read_csv(data_path, nrows=0).columns.drop(["CONS_NO", "FLAG"])
    file_order = pd.to_datetime(raw_header, format="%Y/%m/%d")
    unsorted = wide[file_order]
    unsorted.columns = range(unsorted.shape[1])
    X_unsorted = build_features_wide(unsorted)

    variants = {
        "A. 17 features,\ndates unsorted": (X_unsorted[LEGACY], False),
        "B. 17 features,\ndates sorted": (X[LEGACY], False),
        "C. 85 features,\ndates unsorted": (X_unsorted, False),
        "D. 85 features,\ndates sorted": (X, False),
        "E. 85 features\n+ SMOTE-ENN": (X, True),
    }
    results = {}
    for name, (features, resample) in variants.items():
        results[name] = cross_validate(features, y, params, resample)
        print(name.replace("\n", " "), results[name], flush=True)

    (ROOT / "artifacts" / "ablation.json").write_text(json.dumps(
        {k.replace("\n", " "): v for k, v in results.items()}, indent=2))
    plot(results, float(labels.mean()))


def plot(results: dict, base_rate: float) -> None:
    blue, orange, ink2 = "#2a78d6", "#eb6834", "#52514e"
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    names = list(results)
    x = np.arange(len(names))
    for offset, key, color, label in ((-0.19, "roc_auc", blue, "ROC-AUC"), (0.19, "pr_auc", orange, "PR-AUC")):
        vals = [results[n][key] for n in names]
        ax.bar(x + offset, vals, width=0.36, color=color, label=label)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=ink2)
    ax.axhline(base_rate, color="#b9b8b2", lw=1, ls="--", label=f"PR-AUC of random guessing ({base_rate:.3f})")
    ax.set(xticks=x, ylim=(0, 1.08), ylabel="5-fold cross-validated score")
    ax.set_title("Ablation: effect of date order, features and resampling (42,372 customers)", fontweight="bold", fontsize=11, pad=14)
    ax.set_xticklabels([n.replace(", ", ",\n", 1).replace(" + ", "\n+ ") if "\n" not in n else n for n in names], fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    out = ROOT / "docs" / "thesis-figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "fig-5-6-ablation.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig-5-6-ablation.pdf", bbox_inches="tight")
    print("wrote fig-5-6-ablation")

if __name__ == "__main__":
    main()
