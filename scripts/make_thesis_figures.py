"""
Regenerate the thesis result figures from the committed model and the full dataset.

    python scripts/download_data.py          # once: data/sgcc_full.csv
    pip install matplotlib
    python scripts/make_thesis_figures.py    # writes docs/thesis-figures/*.png and *.pdf

Rebuilds the exact 80/20 customer split used in training (config.yaml
random_state), scores the held-out customers with models/xgb_best.ubj, refits
the two baselines on the same split, and plots the results.
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
import yaml  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data_loader import load_wide  # noqa: E402
from src.features import build_features_wide  # noqa: E402
from src.modeling import load_model  # noqa: E402

OUT = ROOT / "docs" / "thesis-figures"

LABELS = {
    "longest_missing_run": "Longest reporting gap", "std": "Consumption volatility (std)", "range": "Range of daily readings",
    "missing_ratio_first_third": "Missing reads, first third", "missing_ratio_last_third": "Missing reads, last third",
    "max": "Highest daily reading", "q10_rel": "Low-use days vs average", "q90_rel": "High-use days vs average",
    "sudden_drop_rate": "Sudden-drop rate", "sudden_drop_count": "Sudden drops (>50%)", "autocorr_lag7": "Weekly consistency",
    "autocorr_lag1": "Day-to-day consistency", "slope_full": "Overall trend", "last30_vs_mean": "Last 30 days vs average",
    "last_obs_frac": "Last reading position", "first_obs_frac": "First reading position", "mean": "Average daily use",
    "median": "Median daily use", "missing_ratio": "Missing reads", "zero_ratio": "Share of zero readings",
    "coef_var": "Relative volatility", "kurtosis": "Spikiness of readings", "skewness": "Skew of readings",
    "monthly_cv": "Month-to-month volatility", "yoy_last_12m": "Last 12 months vs year before",
    "changepoint_min_ratio": "Use after vs before change point", "first180_vs_mean": "First 180 days vs average",
}


def feature_label(name: str) -> str:
    if name.startswith("month_lag_"):
        return f"Use {int(name.rsplit('_', 1)[1])} months ago vs average"
    return LABELS.get(name, name.replace("_", " "))

# Validated categorical slots (blue, orange, aqua) and text/grid tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK2, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 300, "lines.linewidth": 2,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    data_path = ROOT / config["data"]["training_data_path"]
    if not data_path.exists():
        sys.exit("data/sgcc_full.csv not found: run python scripts/download_data.py first")

    wide, labels = load_wide(str(data_path))
    X = build_features_wide(wide, config.get("features"))
    y = labels.reindex(X.index).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(config["evaluation"]["test_size"]), stratify=y, random_state=int(config["random_state"]))

    model = load_model(str(ROOT / config["paths"]["model_file"]))
    metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    threshold = float(metrics["threshold"])
    p_xgb = model.predict_proba(X_test[model.get_booster().feature_names])[:, 1]

    baselines = {
        "Random forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
        "Logistic regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(class_weight="balanced", max_iter=2000)),
    }
    scores = {"XGBoost": p_xgb}
    for name, pipe in baselines.items():
        scores[name] = pipe.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    colors = {"XGBoost": BLUE, "Random forest": ORANGE, "Logistic regression": AQUA}
    yt = y_test.to_numpy()
    base_rate = yt.mean()

    # ROC
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for name, p in scores.items():
        fpr, tpr, _ = roc_curve(yt, p)
        ax.plot(fpr, tpr, color=colors[name], label=f"{name} (AUC {roc_auc_score(yt, p):.3f})")
    ax.plot([0, 1], [0, 1], color="#b9b8b2", lw=1, ls="--", label="Random (AUC 0.500)")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", xlim=(0, 1), ylim=(0, 1.01),
           title=f"ROC curves, hold-out test set (n = {len(yt):,})")
    ax.legend(loc="lower right")
    save(fig, "fig-5-1-roc-curves")

    # Precision-recall
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for name, p in scores.items():
        prec, rec, _ = precision_recall_curve(yt, p)
        ax.plot(rec, prec, color=colors[name], label=f"{name} (PR-AUC {average_precision_score(yt, p):.3f})")
    ax.axhline(base_rate, color="#b9b8b2", lw=1, ls="--", label=f"Random (base rate {base_rate:.3f})")
    ax.set(xlabel="Recall (thefts caught)", ylabel="Precision (hit rate of flags)", xlim=(0, 1), ylim=(0, 1.01),
           title="Precision-recall curves, hold-out test set")
    ax.legend(loc="upper right")
    save(fig, "fig-5-2-precision-recall-curves")

    # Confusion matrix at the trained threshold
    tn, fp, fn, tp = confusion_matrix(yt, (p_xgb >= threshold).astype(int), labels=[0, 1]).ravel()
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.grid(False)
    cells = np.array([[tp, fn], [fp, tn]])
    names = np.array([["True positive\n(theft caught)", "False negative\n(theft missed)"],
                      ["False positive\n(wasted visit)", "True negative\n(correctly cleared)"]])
    fills = np.array([[BLUE, "#f0efec"], ["#f0efec", "#cde2fb"]])
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.98, 0.98, color=fills[i, j]))
            ink = "white" if fills[i, j] == BLUE else INK
            ax.text(j + 0.49, 1 - i + 0.58, f"{cells[i, j]:,}", ha="center", va="center", fontsize=18, color=ink, weight="bold")
            ax.text(j + 0.49, 1 - i + 0.27, names[i, j], ha="center", va="center", fontsize=8.5, color=ink)
    ax.set(xlim=(0, 2), ylim=(0, 2), xticks=[0.49, 1.49], yticks=[0.49, 1.49],
           xticklabels=["Flagged (theft)", "Not flagged"], yticklabels=["Actually honest", "Actually theft"],
           title=f"Confusion matrix at τ = {threshold:.3f}")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig-5-3-confusion-matrix")

    # Operating curve: precision and recall vs threshold
    ts = np.linspace(0.02, 0.98, 97)
    prec = [((p_xgb >= t) & (yt == 1)).sum() / max((p_xgb >= t).sum(), 1) for t in ts]
    rec = [((p_xgb >= t) & (yt == 1)).sum() / (yt == 1).sum() for t in ts]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.plot(ts, prec, color=BLUE, label="Precision (hit rate)")
    ax.plot(ts, rec, color=ORANGE, label="Recall (thefts caught)")
    ax.axvline(threshold, color=INK, lw=1, ls="--")
    ax.text(threshold + 0.01, 0.04, f"τ = {threshold:.3f}\n(max OOF F1)", color=INK, fontsize=8.5)
    ax.set(xlabel="Decision threshold τ", ylabel="Rate", xlim=(0, 1), ylim=(0, 1.02),
           title="Precision and recall across thresholds, hold-out test set")
    ax.legend(loc="center right")
    save(fig, "fig-5-4-threshold-tradeoff")

    # Mean |SHAP| top 15 on the test set
    import shap
    sample = X_test[model.get_booster().feature_names].sample(n=min(2000, len(X_test)), random_state=0)
    sv = np.asarray(shap.TreeExplainer(model).shap_values(sample))
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=sample.columns).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.barh([feature_label(n) for n in mean_abs.index], mean_abs.values, color=BLUE, height=0.62)
    for i, v in enumerate(mean_abs.values):
        ax.text(v + mean_abs.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Mean |SHAP value| (log-odds)", title=f"Top 15 features by mean |SHAP|, {len(sample):,} test customers")
    save(fig, "fig-5-5-shap-importance")

    # Example consumption: one honest and one theft customer (monthly means)
    rng = np.random.default_rng(7)
    test_ids = X_test.index
    miss_test = X_test["missing_ratio"].to_numpy()
    last_obs = X_test["last_obs_frac"].to_numpy()
    # A complete honest meter vs a flagged theft whose meter falls silent: the pattern
    # behind the model's strongest features (longest gap, last reading position).
    honest_id = rng.choice(test_ids[(yt == 0) & (p_xgb < 0.05) & (miss_test < 0.02)])
    silent = (yt == 1) & (p_xgb > 0.9) & (last_obs < 0.7) & (miss_test < 0.7)
    theft_id = test_ids[silent][np.argmax(p_xgb[silent])] if silent.any() else test_ids[(yt == 1) & (p_xgb > 0.9)][0]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    for ax, cid, label, color in ((axes[0], honest_id, "Honest customer", BLUE), (axes[1], theft_id, "Theft customer", ORANGE)):
        monthly = wide.loc[cid].groupby(wide.columns.to_period("M")).mean()
        idx = monthly.index.to_timestamp()
        ax.bar(idx, monthly.values, width=24, color=color)
        missing = monthly.isna()
        for t in idx[missing.to_numpy()]:
            ax.axvspan(t, t + pd.Timedelta(days=30), color="#f0efec", lw=0)
        p = p_xgb[list(test_ids).index(cid)]
        ax.set_title(f"{label} · model probability {p:.3f}", loc="left")
        ax.set_ylabel("kWh / day")
    axes[0].text(1.0, 1.08, "shaded months: no meter readings", transform=axes[0].transAxes, ha="right", fontsize=8, color=INK2)
    save(fig, "fig-3-2-example-consumption")

    # Class balance and missing-data signal
    miss = X["missing_ratio"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.linspace(0, 1, 21)
    centers, width = (bins[:-1] + bins[1:]) / 2, (bins[1] - bins[0]) * 0.42
    for cls, color, name, shift in ((0, BLUE, "Honest", -0.5), (1, ORANGE, "Theft", 0.5)):
        counts, _ = np.histogram(miss[y == cls], bins=bins)
        ax.bar(centers + shift * (width + 0.002), counts / counts.sum(), width=width, color=color,
               label=f"{name} (n = {(y == cls).sum():,})")
    ax.set(xlabel="Share of days with no meter reading", ylabel="Share of customers",
           title="Missing readings by class, all 42,372 customers")
    ax.legend()
    save(fig, "fig-3-3-missing-readings-by-class")

    print(f"\nTest set: {len(yt):,} customers; XGBoost AUC {roc_auc_score(yt, p_xgb):.4f}, "
          f"PR-AUC {average_precision_score(yt, p_xgb):.4f}; tp {tp} fp {fp} fn {fn} tn {tn}")


if __name__ == "__main__":
    main()
