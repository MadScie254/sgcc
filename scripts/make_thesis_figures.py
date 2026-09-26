"""
Regenerate the thesis result figures from the trained pipeline and the full dataset.

    python scripts/download_data.py          # once: data/sgcc_full.csv
    python -m src.train                      # model, artifacts and saved predictions
    python scripts/significance.py           # bootstrap intervals and tests (figure 5.16)
    python scripts/make_thesis_figures.py    # writes docs/thesis-figures/*.png and *.pdf

Every test-set figure uses the predictions training saved
(artifacts/predictions/test.csv.gz): the test customers were scored once, and
nothing is refitted here, so no GPU is needed. The full dataset is read again only
for the data figures (3.2, 3.3), the SHAP plots of the served model and the PCA view
of SMOTE+ENN.
"""

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data_loader import load_wide  # noqa: E402
from src.experiment import CANDIDATES  # noqa: E402
from src.feature_catalog import feature_label  # noqa: E402
from src.modeling import load_model  # noqa: E402
from src.pipeline import features_for, model_input  # noqa: E402
from src.resampling import Treatment  # noqa: E402
from src.train import load_config, split_customers  # noqa: E402

OUT = ROOT / "docs" / "thesis-figures"


# Validated categorical slots in fixed order (blue, orange, aqua, yellow, magenta) and text/grid tokens.
# Aqua, yellow and magenta sit below 3:1 on white, so every mark in those colours carries a direct label.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
COLORS = dict(zip(CANDIDATES, (BLUE, ORANGE, AQUA, YELLOW, MAGENTA)))

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


def wrap(text: str, width: int = 14) -> str:
    return textwrap.fill(text, width)


def label_bars(ax, bars, fmt="{:.3f}", fontsize=7.5):
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=fontsize, color=INK2)


def main() -> None:
    config = load_config()
    data_path = ROOT / config["data"]["training_data_path"]
    if not data_path.exists():
        sys.exit("data/sgcc_full.csv not found: run python scripts/download_data.py first")
    artifacts = ROOT / config["paths"]["artifacts"]
    read = lambda name: json.loads((artifacts / name).read_text())  # noqa: E731
    spec, tuning, curves, resampling = read("pipeline.json"), read("tuning.json"), read("learning_curves.json"), read("resampling.json")
    calibration = read("calibration.json")
    comparison = json.loads((ROOT / config["paths"]["models"] / "baselines" / "comparison_results.json").read_text())
    eval_cfg, random_state = config["evaluation"], int(config["random_state"])

    wide, labels = load_wide(str(data_path))
    y = labels.reindex(wide.index).astype(int)
    X_by = {prep: features_for(wide, prep, config.get("preprocessing"), spec["feature_config"]) for prep in ("raw", "clean")}
    train_idx, _, test_idx = split_customers(y, float(eval_cfg["validation_size"]), float(eval_cfg["test_size"]), random_state)

    # The saved test predictions (calibrated probabilities), in the order of the split.
    test = pd.read_csv(artifacts / "predictions" / "test.csv.gz", dtype={"customer_id": str}).set_index("customer_id").loc[test_idx]
    yt = test["label"].to_numpy()
    assert (yt == y.loc[test_idx].to_numpy()).all(), "saved predictions do not match the training split"
    base_rate = yt.mean()
    scores = {name: test[name].to_numpy() for name in CANDIDATES}
    for name, p in scores.items():
        print(f"{name}: test PR-AUC {average_precision_score(yt, p):.4f}")
    served = spec["name"]
    threshold = float(spec["threshold"])
    p_served = scores[served]
    X_served = model_input(wide.loc[test_idx], spec, spec["feature_config"])[spec["features"]]
    label = {name: s_["label"] for name, s_ in CANDIDATES.items()}

    # 5.1 ROC and 5.2 precision-recall, every candidate
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for name, p in scores.items():
        fpr, tpr, _ = roc_curve(yt, p)
        ax.plot(fpr, tpr, color=COLORS[name], lw=1.8, label=f"{label[name]} ({roc_auc_score(yt, p):.3f})")
    ax.plot([0, 1], [0, 1], color="#b9b8b2", lw=1, ls="--", label="Random (0.500)")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", xlim=(0, 1), ylim=(0, 1.01),
           title=f"ROC curves, test customers (n = {len(yt):,})")
    ax.legend(loc="lower right", fontsize=7.5, title="ROC-AUC", title_fontsize=8)
    save(fig, "fig-5-1-roc-curves")

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for name, p in scores.items():
        prec, rec, _ = precision_recall_curve(yt, p)
        ax.plot(rec, prec, color=COLORS[name], lw=1.8, label=f"{label[name]} ({average_precision_score(yt, p):.3f})")
    ax.axhline(base_rate, color="#b9b8b2", lw=1, ls="--", label=f"Random ({base_rate:.3f})")
    ax.set(xlabel="Recall (thefts caught)", ylabel="Precision (hit rate of flags)", xlim=(0, 1), ylim=(0, 1.01),
           title="Precision-recall curves, test customers")
    ax.legend(loc="upper right", fontsize=7.5, title="PR-AUC", title_fontsize=8)
    save(fig, "fig-5-2-precision-recall-curves")

    # 5.3 Confusion matrix of the served pipeline
    tn, fp, fn, tp = confusion_matrix(yt, (p_served >= threshold).astype(int), labels=[0, 1]).ravel()
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
           title=f"{label[served]}, τ = {threshold:.3f}")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "fig-5-3-confusion-matrix")

    # 5.4 Precision and recall across thresholds, served pipeline
    ts = np.linspace(0.02, 0.98, 97)
    prec = [((p_served >= t) & (yt == 1)).sum() / max((p_served >= t).sum(), 1) for t in ts]
    rec = [((p_served >= t) & (yt == 1)).sum() / (yt == 1).sum() for t in ts]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.plot(ts, prec, color=BLUE, label="Precision (hit rate)")
    ax.plot(ts, rec, color=ORANGE, label="Recall (thefts caught)")
    ax.axvline(threshold, color=INK, lw=1, ls="--")
    ax.text(threshold + 0.01, 0.04, f"τ = {threshold:.3f}\n(max F1 on validation)", color=INK, fontsize=8.5)
    ax.set(xlabel="Decision threshold τ (calibrated probability)", ylabel="Rate", xlim=(0, 1), ylim=(0, 1.02),
           title="Precision and recall across thresholds, test customers")
    ax.legend(loc="center right")
    save(fig, "fig-5-4-threshold-tradeoff")

    # 5.5 Mean |SHAP| top 15 and 5.8 SHAP summary (top 20) on the test set, raw score of the served model
    model = load_model(str(ROOT / config["paths"]["model_file"]))
    sample = X_served.sample(n=min(2000, len(X_served)), random_state=0)
    sv = np.asarray(shap.TreeExplainer(model).shap_values(sample))
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=sample.columns).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.barh([feature_label(n) for n in mean_abs.index], mean_abs.values, color=BLUE, height=0.62)
    for i, v in enumerate(mean_abs.values):
        ax.text(v + mean_abs.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Mean |SHAP value| (log-odds)", title=f"Top 15 features by mean |SHAP|, {len(sample):,} test customers")
    save(fig, "fig-5-5-shap-importance")

    plt.figure()
    shap.summary_plot(sv, sample.rename(columns=feature_label), max_display=20, show=False, plot_size=(7.4, 7.2))
    fig = plt.gcf()
    fig.axes[0].set_title(f"SHAP summary, top 20 features ({len(sample):,} test customers)", fontsize=11, fontweight="bold")
    save(fig, "fig-5-8-shap-summary")

    # 5.9 Objective 1: what SMOTE and SMOTE+ENN do to the training data
    stages = [("Original", resampling["before"], resampling["smote_enn"]["counts"]["before"]),
              ("After SMOTE", resampling["smote"]["diagnostics"], resampling["smote"]["counts"]["after"]),
              ("After SMOTE+ENN", resampling["smote_enn"]["diagnostics"], resampling["smote_enn"]["counts"]["after"])]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    x = np.arange(len(stages))
    ax = axes[0]
    for offset, cls, color in ((-0.2, "honest", BLUE), (0.2, "theft", ORANGE)):
        label_bars(ax, ax.bar(x + offset, [c[cls] for _, _, c in stages], width=0.38, color=color, label=cls.title()), "{:,.0f}", 7)
    ax.set(xticks=x, title="Training customers by class", ylabel="Rows", ylim=(0, 34000))
    ax.set_xticklabels([s_[0].replace(" ", "\n", 1) for s_ in stages], fontsize=8)
    ax.legend(fontsize=8, ncol=2, loc="upper center")
    for ax, key, title in zip(axes[1:], ("silhouette", "fisher_ratio_mean", "boundary_noise_theft"),
                              ("Silhouette score\n(class separation)", "Mean Fisher ratio\n(feature separability)",
                               "Boundary noise, theft rows\n(share with honest-majority neighbours)")):
        bars = ax.bar(x, [d[key] for _, d, _ in stages], width=0.55, color=INK2)
        label_bars(ax, bars, "{:.3f}")
        ax.set(xticks=x, title=title)
        ax.set_xticklabels([s_[0].replace(" ", "\n", 1) for s_ in stages], fontsize=8)
    fig.suptitle("Objective 1: effect of SMOTE+ENN on the training data (cleaned features, standardised)", fontweight="bold", fontsize=11)
    save(fig, "fig-5-9-resampling-effect")

    # 5.10 The same, in two dimensions (PCA fitted on the original training rows)
    X_clean_train = X_by["clean"].loc[train_idx]
    treatment = Treatment("smote_enn", config.get("resampling"), random_state)
    X_res, y_res = treatment.fit_resample(X_clean_train, y.loc[train_idx])
    filled = X_clean_train.fillna(treatment.medians_)
    scaler = StandardScaler().fit(filled)
    pca = PCA(n_components=2, random_state=0).fit(scaler.transform(filled))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    for ax, (Xp, yp, title) in zip(axes, ((filled, y.loc[train_idx].to_numpy(), "Before"), (X_res, y_res.to_numpy(), "After SMOTE+ENN"))):
        Z = pca.transform(scaler.transform(Xp))
        rng = np.random.default_rng(0)
        for cls, color, name in ((0, BLUE, "Honest"), (1, ORANGE, "Theft")):
            idx = np.flatnonzero(yp == cls)
            idx = rng.choice(idx, size=min(len(idx), 3000), replace=False)
            ax.scatter(Z[idx, 0], Z[idx, 1], s=8, color=color, alpha=0.55, edgecolors=SURFACE, linewidths=0.4,
                       label=f"{name} ({(yp == cls).sum():,})")
        ax.set(title=f"{title} (up to 3,000 points per class)", xlabel="Principal component 1")
        ax.legend(fontsize=8, markerscale=1.5)
    axes[0].set_ylabel("Principal component 2")
    lo, hi = np.percentile(pca.transform(scaler.transform(filled)), [1, 99], axis=0)
    axes[0].set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    save(fig, "fig-5-10-resampling-pca")

    # 5.11 Effectiveness of every pipeline at its validation-chosen threshold
    measures = [("recall", "Recall"), ("precision", "Precision"), ("f1", "F1"), ("pr_auc", "PR-AUC"), ("gmean", "G-Mean"), ("mcc", "MCC")]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    width = 0.8 / len(CANDIDATES)
    x = np.arange(len(measures))
    for k, name in enumerate(CANDIDATES):
        bars = ax.bar(x - 0.4 + width * (k + 0.5), [comparison[name][m] for m, _ in measures], width=width * 0.92,
                      color=COLORS[name], label=label[name])
        label_bars(ax, bars, "{:.2f}", 6.5)
    ax.set(xticks=x, ylim=(0, 1.05), ylabel="Test-set score", title="Pipelines compared on the test customers (each at its own validation threshold)")
    ax.set_xticklabels([m for _, m in measures])
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    save(fig, "fig-5-11-model-comparison")

    # 5.12 Computational cost
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (key, title, fmt) in zip(axes, (("training_time", "Training time (s)\nresampling + fitting", "{:.1f}"),
                                            ("inference_ms_per_customer", "Inference time per customer (ms)", "{:.3f}"),
                                            ("model_size_mb", "Model size (MB)", "{:.2f}"))):
        order = list(CANDIDATES)
        bars = ax.bar(np.arange(len(order)), [comparison[n][key] for n in order], color=[COLORS[n] for n in order])
        label_bars(ax, bars, fmt, 7.5)
        ax.set(xticks=np.arange(len(order)), title=title)
        ax.set_xticklabels([wrap(label[n], 12) for n in order], fontsize=7)
    fig.suptitle("Computational cost on a 4-core CPU (no GPU)", fontweight="bold", fontsize=11)
    save(fig, "fig-5-12-computational-cost")

    # 5.13 Accuracy paradox (RQ1): accuracy hides missed thefts
    all_honest = {"accuracy": 1 - base_rate, "recall": 0.0, "f1": 0.0}
    default_05 = comparison["xgboost_default"]["at_threshold_0_5"]
    rows = [("Flag nobody", all_honest), ("XGBoost, defaults,\np ≥ 0.5", default_05), (f"{label[served]},\nτ = {threshold:.3f}", comparison[served])]
    fig, ax = plt.subplots(figsize=(7.6, 4))
    x = np.arange(len(rows))
    for offset, key, color, name in ((-0.27, "accuracy", INK2, "Accuracy"), (0, "recall", ORANGE, "Recall"), (0.27, "f1", BLUE, "F1")):
        label_bars(ax, ax.bar(x + offset, [r[key] for _, r in rows], width=0.26, color=color, label=name), "{:.3f}")
    ax.set(xticks=x, ylim=(0, 1.1), title="Class imbalance: high accuracy, few thefts caught (test customers)")
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.legend(ncol=3, loc="upper center", fontsize=8.5)
    save(fig, "fig-5-13-accuracy-paradox")

    # 5.14 Learning curves (overfitting check, section 3.10) and 5.15 tuning history
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, name in zip(axes, curves):
        c = curves[name]
        ax.plot(np.arange(1, len(c["train_pr_auc"]) + 1), c["train_pr_auc"], color=BLUE, label="Training rows")
        ax.plot(np.arange(1, len(c["validation_pr_auc"]) + 1), c["validation_pr_auc"], color=ORANGE, label="Validation rows")
        ax.axvline(c["best_iteration"], color=INK, lw=1, ls="--")
        ax.text(c["best_iteration"], 0.6, f" kept {c['best_iteration']} trees", fontsize=8, color=INK)
        ax.set(title=label[name], xlabel="Boosting rounds (trees)", ylim=(0, 1.02))
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("PR-AUC")
    fig.suptitle("Learning curves with early stopping on validation PR-AUC", fontweight="bold", fontsize=11)
    save(fig, "fig-5-14-learning-curves")

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for name in tuning:
        trials = np.array(tuning[name]["trials"])
        ax.plot(np.arange(1, len(trials) + 1), trials, "o", ms=4, color=COLORS[name], alpha=0.45)
        ax.plot(np.arange(1, len(trials) + 1), np.maximum.accumulate(trials), color=COLORS[name], label=f"{label[name]} (best {trials.max():.3f})")
    ax.set(xlabel="Optuna trial", ylabel="Mean PR-AUC over 5 folds", title="Hyperparameter search (5-fold CV on training customers)")
    ax.legend(fontsize=8.5, title="line: best so far · dots: each trial", title_fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, "fig-5-15-tuning-history")

    # 5.16 Paired bootstrap on the test customers (significance.py): difference from the proposed pipeline
    significance_path = artifacts / "significance.json"
    if significance_path.exists():
        sig = json.loads(significance_path.read_text())
        others = [n for n in CANDIDATES if n != sig["reference"]]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
        for ax, metric, name_ in zip(axes, ("pr_auc", "f1", "mcc"), ("PR-AUC", "F1", "MCC")):
            for k, name in enumerate(others):
                row = sig["comparisons"][name][metric]
                ax.errorbar(row["difference"], k, xerr=[[row["difference"] - row["ci_low"]], [row["ci_high"] - row["difference"]]],
                            fmt="o", color=COLORS[name], ecolor=COLORS[name], elinewidth=2, capsize=4, ms=7)
                ax.text(row["ci_high"], k + 0.22, f"  p = {row['p_holm']:.3g}{' *' if row['significant'] else ''}", fontsize=7.5, color=INK2)
            ax.axvline(0, color=INK, lw=1, ls="--")
            ax.set(title=f"{name_}: proposed minus other", xlabel="Difference (95% bootstrap interval)")
            ax.grid(axis="y", visible=False)
        axes[0].set(yticks=np.arange(len(others)), ylim=(-0.6, len(others) - 0.3))
        axes[0].set_yticklabels([wrap(label[n], 22) for n in others], fontsize=8)
        axes[0].invert_yaxis()
        fig.suptitle(f"Paired stratified bootstrap on the {sig['population']['customers']:,} test customers "
                     f"({sig['resamples']:,} resamples; Holm-adjusted p, * p < 0.05). Right of 0: the proposed pipeline is better.",
                     fontweight="bold", fontsize=10)
        save(fig, "fig-5-16-bootstrap-comparison")

    # 5.17 Reliability diagram: raw scores and Platt-calibrated probabilities, test customers
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, name in zip(axes, (served, "proposed" if served != "proposed" else "xgboost")):
        report = calibration["pipelines"][name]
        for kind, color, text in (("raw", ORANGE, "Raw score"), ("platt", BLUE, "Calibrated (Platt)")):
            bins = report["reliability_test"][kind]
            ece = report["test"][kind]["ece"]
            ax.plot([b["mean_predicted"] for b in bins], [b["observed_rate"] for b in bins], "o-", color=color, ms=5,
                    label=f"{text}, ECE {ece:.3f}")
        ax.plot([0, 1], [0, 1], color="#b9b8b2", lw=1, ls="--", label="Perfect calibration")
        ax.set(title=f"{label[name]}{' (served)' if name == served else ''}", xlabel="Predicted probability", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Observed theft rate")
    fig.suptitle("Calibration on the test customers (Platt scaling fitted on validation customers)", fontweight="bold", fontsize=11)
    save(fig, "fig-5-17-reliability")

    # 5.18 Inspection workload against thefts found, every pipeline (cumulative gains on the test customers)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    share = np.arange(1, len(yt) + 1) / len(yt)
    for name, p in scores.items():
        found = np.cumsum(yt[np.argsort(-p, kind="mergesort")]) / yt.sum()
        ax.plot(share, found, color=COLORS[name], lw=1.8, label=label[name])
    flagged = (p_served >= threshold).mean()
    ax.plot([0, 1], [0, 1], color="#b9b8b2", lw=1, ls="--", label="Random inspection")
    ax.axvline(flagged, color=INK, lw=1, ls=":")
    ax.text(flagged + 0.01, 0.05, f"served τ: inspect {flagged:.1%},\nfind {(p_served >= threshold)[yt == 1].mean():.1%} of thefts", fontsize=8, color=INK)
    ax.set(xlabel="Share of customers inspected (highest probability first)", ylabel="Share of thefts found", xlim=(0, 1), ylim=(0, 1.01),
           title="Inspection workload against thefts found, test customers")
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "fig-5-18-workload")

    # 3.2 Example consumption: one honest and one theft customer (monthly means)
    rng = np.random.default_rng(7)
    X_test = X_by["raw"].loc[test_idx]
    miss_test, last_obs = X_test["missing_ratio"].to_numpy(), X_test["last_obs_frac"].to_numpy()
    honest_id = rng.choice(test_idx[(yt == 0) & (p_served <= np.quantile(p_served, 0.3)) & (miss_test < 0.02)])
    silent = (yt == 1) & (p_served >= np.quantile(p_served, 0.98)) & (last_obs < 0.7) & (miss_test < 0.7)
    theft_id = test_idx[silent][np.argmax(p_served[silent])] if silent.any() else test_idx[(yt == 1)][np.argmax(p_served[yt == 1])]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    for ax, cid, name, color in ((axes[0], honest_id, "Honest customer", BLUE), (axes[1], theft_id, "Theft customer", ORANGE)):
        monthly = wide.loc[cid].groupby(wide.columns.to_period("M")).mean()
        idx = monthly.index.to_timestamp()
        ax.bar(idx, monthly.values, width=24, color=color)
        for t in idx[monthly.isna().to_numpy()]:
            ax.axvspan(t, t + pd.Timedelta(days=30), color="#f0efec", lw=0)
        ax.set_title(f"{name} · model probability {p_served[list(test_idx).index(cid)]:.3f}", loc="left")
        ax.set_ylabel("kWh / day")
    axes[0].text(1.0, 1.08, "shaded months: no meter readings", transform=axes[0].transAxes, ha="right", fontsize=8, color=INK2)
    save(fig, "fig-3-2-example-consumption")

    # 3.3 Missing readings by class
    miss = X_by["raw"]["missing_ratio"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.linspace(0, 1, 21)
    centers, width = (bins[:-1] + bins[1:]) / 2, (bins[1] - bins[0]) * 0.42
    for cls, color, name, shift in ((0, BLUE, "Honest", -0.5), (1, ORANGE, "Theft", 0.5)):
        counts, _ = np.histogram(miss[y == cls], bins=bins)
        ax.bar(centers + shift * (width + 0.002), counts / counts.sum(), width=width, color=color, label=f"{name} (n = {(y == cls).sum():,})")
    ax.set(xlabel="Share of days with no meter reading", ylabel="Share of customers", title="Missing readings by class, all 42,372 customers")
    ax.legend()
    save(fig, "fig-3-3-missing-readings-by-class")

    print(f"\nServed: {served}; test PR-AUC {average_precision_score(yt, p_served):.4f}; tp {tp} fp {fp} fn {fn} tn {tn}")


if __name__ == "__main__":
    main()
