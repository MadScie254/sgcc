"""
Ablation study: what each part of the pipeline is worth (the proposal's variables).

    python scripts/ablation.py                # after python -m src.train; ~20 min on 4 cores
    python scripts/ablation.py --device cuda  # XGBoost fits on an NVIDIA GPU
    python scripts/ablation.py --plot-only    # redraw the figure from artifacts/ablation.json

5-fold stratified cross-validation on the development customers (training plus
validation, 36,016): the test customers stay untouched, as in training. Each
fold is scored separately and the table reports the mean and standard deviation
over the five folds. Each variant changes one thing relative to its neighbour,
with the hyperparameters training tuned (artifacts/tuning.json, trees from early
stopping) held fixed:

  A. raw readings, the proposal's 25 core features, no resampling
  B. raw readings, all features, no resampling (standard XGBoost)
  C. cleaned readings (section 3.7), all features, no resampling
  D. cleaned readings, all features, SMOTE
  E. cleaned readings, all features, SMOTE+ENN (the proposed framework)
  F. as B, but scale_pos_weight = 1 (no algorithm-level imbalance correction)
  G. as B, but XGBoost's default hyperparameters (no tuning)

Treatments are fitted inside each fold on its training rows only. Writes
artifacts/ablation.json and docs/thesis-figures/fig-5-6-ablation.png/.pdf.
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.data_loader import load_wide  # noqa: E402
from src.feature_catalog import CORE_FEATURES  # noqa: E402
from src.modeling import cross_val_scores, make_folds  # noqa: E402
from src.pipeline import features_for  # noqa: E402
from src.train import load_config, split_customers  # noqa: E402

BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e6e5e1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study of the pipeline's parts")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Where XGBoost trains")
    parser.add_argument("--plot-only", action="store_true", help="Redraw the figure from artifacts/ablation.json")
    args = parser.parse_args()
    out = ROOT / "artifacts" / "ablation.json"
    if args.plot_only:
        plot(json.loads(out.read_text()))
        return
    config = load_config()
    artifacts = ROOT / config["paths"]["artifacts"]
    tuning = json.loads((artifacts / "tuning.json").read_text())
    curves = json.loads((artifacts / "learning_curves.json").read_text())
    params = {name: {**tuning[name]["params"], "n_estimators": curves[name]["best_iteration"]} for name in tuning}

    wide, labels = load_wide(str(ROOT / config["data"]["training_data_path"]))
    y = labels.reindex(wide.index).astype(int)
    evaluation = config["evaluation"]
    _, _, test_idx = split_customers(y, float(evaluation["validation_size"]), float(evaluation["test_size"]),
                                     int(config["random_state"]))
    wide, y = wide.drop(index=test_idx), y.drop(index=test_idx)
    raw = features_for(wide, "raw", feature_config=config.get("features"))
    clean = features_for(wide, "clean", config.get("preprocessing"), config.get("features"))
    resampling = config.get("resampling")

    variants = {
        "A. Raw, 25 core features": (raw[list(CORE_FEATURES)], "none", params["xgboost"]),
        "B. Raw, all features (standard XGBoost)": (raw, "none", params["xgboost"]),
        "C. Cleaned, all features": (clean, "none", params["xgboost"]),
        "D. Cleaned + SMOTE": (clean, "smote", params["proposed"]),
        "E. Cleaned + SMOTE+ENN (proposed)": (clean, "smote_enn", params["proposed"]),
        "F. B with scale_pos_weight = 1": (raw, "none", {**params["xgboost"], "scale_pos_weight": 1.0}),
        "G. B with default hyperparameters": (raw, "none", {}),
    }
    results = {}
    for name, (X, treatment, p) in variants.items():
        folds = make_folds(X, y, cv=5, random_state=42, treatment=treatment, treatment_config=resampling)
        scores = {"roc_auc": cross_val_scores(p, folds, y, roc_auc_score, device=args.device),
                  "pr_auc": cross_val_scores(p, folds, y, average_precision_score, device=args.device)}
        results[name] = {
            **{key: float(v.mean()) for key, v in scores.items()},
            **{f"{key}_sd": float(v.std(ddof=1)) for key, v in scores.items()},
            "folds": {key: [round(float(x), 6) for x in v] for key, v in scores.items()},
            "features": int(X.shape[1]), "treatment": treatment,
        }
        print(name, {k: round(results[name][k], 4) for k in ("roc_auc", "roc_auc_sd", "pr_auc", "pr_auc_sd")}, flush=True)
    out.write_text(json.dumps({"customers": int(len(y)), "base_rate": float(y.mean()), "variants": results}, indent=2))
    plot(json.loads(out.read_text()))


def plot(data: dict) -> None:
    results, base_rate = data["variants"], data["base_rate"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    names = list(results)
    x = np.arange(len(names))
    for offset, key, color, label in ((-0.2, "roc_auc", BLUE, "ROC-AUC"), (0.2, "pr_auc", ORANGE, "PR-AUC")):
        vals = [results[n][key] for n in names]
        sds = [results[n][f"{key}_sd"] for n in names]
        ax.bar(x + offset, vals, width=0.38, color=color, label=f"{label} (mean ± SD over 5 folds)",
               yerr=sds, capsize=2.5, error_kw={"elinewidth": 0.9, "ecolor": INK2})
        for xi, v, sd in zip(x + offset, vals, sds):
            ax.text(xi, v + sd + 0.012, f"{v:.3f}", ha="center", fontsize=7.5, color=INK2)
    ax.axhline(base_rate, color="#b9b8b2", lw=1, ls="--", label=f"PR-AUC of random guessing ({base_rate:.3f})")
    ax.set(xticks=x, ylim=(0, 1.1), ylabel="Score per fold, mean ± SD")
    ax.set_xticklabels([textwrap.fill(n, 16) for n in names], fontsize=7.5)
    ax.set_title(f"Ablation: features, cleaning, resampling and tuning ({data['customers']:,} development customers)",
                 fontweight="bold", fontsize=11, pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    target = ROOT / "docs" / "thesis-figures"
    target.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target / "fig-5-6-ablation.png", dpi=300, bbox_inches="tight")
    fig.savefig(target / "fig-5-6-ablation.pdf", bbox_inches="tight")
    print("wrote fig-5-6-ablation")


if __name__ == "__main__":
    main()
