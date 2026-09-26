"""
Robustness of the comparison: other random splits, and how much the model relies on missing readings.

    python scripts/robustness.py                 # ~25 min on 4 cores (after resampling_study.py)
    python scripts/robustness.py --skip-splits   # only the missing-data study
    python scripts/robustness.py --plot-only

1. Repeated splits. The study's results come from one stratified 70/15/15 customer split
   (seed 42). Here every pipeline is refitted on five more splits (seeds 1-5) with its
   tuned hyperparameters held fixed (they were tuned on the seed-42 training customers;
   re-tuning per split is left out for time and stated as a limitation). Early stopping,
   Platt calibration and the F1 threshold use each split's own validation customers, and
   each split's test customers are scored once. Seed 42 reproduces the study's numbers.
   The extra pipeline "SMOTE+ENN + XGBoost, raw readings" is included when
   artifacts/resampling_study.json holds its tuned hyperparameters.
2. Missing-reading dependence. 5-fold CV on the 36,016 development customers, standard
   XGBoost's tuned hyperparameters held fixed:
     all features (raw readings)                         - the served model's inputs
     without the missing-reading features                - gaps still show as missing months
     consumption behaviour only (cleaned, gaps filled,
       no missing-reading features)                      - no trace of missingness left
     missing-reading features only

Writes artifacts/robustness.json and docs/thesis-figures/fig-5-22-split-stability and
fig-5-23-missingness.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src import figstyle  # noqa: E402
from src.experiment import ALL_CANDIDATES, CANDIDATES  # noqa: E402
from src.figstyle import INK2, ORANGE, PIPELINE_COLORS, plt  # noqa: E402
from src.modeling import cross_val_scores, make_folds  # noqa: E402
from src.study import fit_and_score, load_study  # noqa: E402

OUT = ROOT / "artifacts" / "robustness.json"
SEEDS = (42, 1, 2, 3, 4, 5)
METRICS = ("pr_auc", "auc", "recall", "precision", "f1", "mcc", "gmean")
MISSING_FEATURES = ("missing_ratio", "missing_ratio_first_third", "missing_ratio_middle_third",
                    "missing_ratio_last_third", "longest_missing_run", "missing_sequences_count",
                    "first_obs_frac", "last_obs_frac")


def pipeline_params(study) -> dict:
    # The tuned values as training used them: early stopping on each split's validation customers
    # picks the trees, so seed 42 reproduces the study's models.
    tuning = json.loads((study.artifacts / "tuning.json").read_text())
    params = {name: tuning[name]["params"] for name in ("proposed", "xgboost")}
    extra = ROOT / "artifacts" / "resampling_study.json"
    if extra.exists() and "fair_test" in json.loads(extra.read_text()):
        params["xgboost_smote_enn_raw"] = json.loads(extra.read_text())["fair_test"]["tuning"]["params"]
    return params


def repeated_splits(study, device: str) -> dict:
    params = pipeline_params(study)
    names = list(CANDIDATES) + [n for n in ("xgboost_smote_enn_raw",) if n in params]
    per_seed = {n: {} for n in names}
    for seed in SEEDS:
        train_idx, val_idx, test_idx = study.split(seed)
        for name in names:
            spec = ALL_CANDIDATES[name]
            result = fit_and_score(study, name, params.get(name) if spec["tuned"] else None,
                                   train_idx, val_idx, test_idx, device=device)
            per_seed[name][seed] = {m: float(result["metrics"][m]) for m in METRICS}
            per_seed[name][seed]["threshold"] = result["threshold"]
            print(f"seed {seed} {name}: PR-AUC {per_seed[name][seed]['pr_auc']:.4f}", flush=True)
    summary = {}
    for name in names:
        values = {m: np.array([per_seed[name][s][m] for s in SEEDS]) for m in METRICS}
        summary[name] = {"label": ALL_CANDIDATES[name]["label"],
                         "per_seed": {str(s): per_seed[name][s] for s in SEEDS},
                         "mean": {m: float(v.mean()) for m, v in values.items()},
                         "sd": {m: float(v.std(ddof=1)) for m, v in values.items()},
                         "min": {m: float(v.min()) for m, v in values.items()},
                         "max": {m: float(v.max()) for m, v in values.items()}}
    first = {n: 0 for n in names}
    for s in SEEDS:
        first[max(names, key=lambda n: per_seed[n][s]["pr_auc"])] += 1
    beats = {n: sum(per_seed["xgboost"][s]["pr_auc"] > per_seed[n][s]["pr_auc"] for s in SEEDS)
             for n in names if n != "xgboost"}
    return {"seeds": list(SEEDS), "pipelines": summary, "ranked_first_on_pr_auc": first,
            "standard_xgboost_higher_pr_auc_than": beats,
            "note": "Hyperparameters tuned once on the seed-42 training customers and held fixed."}


def missingness(study, device: str) -> dict:
    # Development customers in file order, as scripts/ablation.py has them, so the folds are the same.
    y = study.y.drop(index=study.test_idx)
    raw, clean = study.X_by["raw"].loc[y.index], study.X_by["clean"].loc[y.index]
    missing = [c for c in MISSING_FEATURES if c in raw.columns]
    variants = {
        "All features (served model's inputs)": raw,
        "Without missing-reading features": raw.drop(columns=missing),
        "Consumption behaviour only": clean.drop(columns=missing),
        "Missing-reading features only": raw[missing],
    }
    params = study.tuned_params("xgboost")
    out = {}
    for label, X in variants.items():
        folds = make_folds(X, y, cv=5, random_state=42)
        scores = {"roc_auc": cross_val_scores(params, folds, y, roc_auc_score, device=device),
                  "pr_auc": cross_val_scores(params, folds, y, average_precision_score, device=device)}
        out[label] = {"features": int(X.shape[1]),
                      **{k: float(v.mean()) for k, v in scores.items()},
                      **{f"{k}_sd": float(v.std(ddof=1)) for k, v in scores.items()},
                      "folds": {k: [round(float(x), 6) for x in v] for k, v in scores.items()}}
        print(f"{label}: PR-AUC {out[label]['pr_auc']:.4f} ± {out[label]['pr_auc_sd']:.4f}", flush=True)
    return {"customers": int(len(y)), "base_rate": float(y.mean()), "missing_features": missing, "variants": out}


def plot_splits(data: dict) -> None:
    pipes = data["pipelines"]
    names = sorted(pipes, key=lambda n: -pipes[n]["mean"]["pr_auc"])
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, name in enumerate(names):
        vals = [pipes[name]["per_seed"][str(s)]["pr_auc"] for s in data["seeds"]]
        color = PIPELINE_COLORS.get(name, INK2)
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(i + jitter, vals, s=22, color=color, zorder=3)
        ax.scatter([i + jitter[0]], [vals[0]], s=60, facecolors="none", edgecolors=INK2, zorder=4)
        mean = pipes[name]["mean"]["pr_auc"]
        ax.hlines(mean, i - 0.25, i + 0.25, color=color, lw=2.2, zorder=2)
        ax.text(i + 0.28, mean, f"{mean:.3f}\n± {pipes[name]['sd']['pr_auc']:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([figstyle_wrap(pipes[n]["label"]) for n in names], fontsize=8)
    ax.set_ylabel("Test-set PR-AUC")
    ax.set_xlim(-0.5, len(names) - 0.1)
    ax.set_title(f"PR-AUC on {len(data['seeds'])} different random splits (circled: the study's split)", pad=10)
    figstyle.save(fig, "fig-5-22-split-stability")


def figstyle_wrap(text: str, width: int = 16) -> str:
    import textwrap
    return textwrap.fill(text, width)


def plot_missingness(data: dict) -> None:
    variants = data["variants"]
    labels = list(variants)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    y = np.arange(len(labels))[::-1]
    means = [variants[k]["pr_auc"] for k in labels]
    sds = [variants[k]["pr_auc_sd"] for k in labels]
    colors = [ORANGE, "#f2a27f", "#2a78d6", "#9a98cf"]
    ax.barh(y, means, xerr=sds, color=colors, height=0.6, error_kw={"elinewidth": 0.9, "ecolor": INK2, "capsize": 2.5})
    for yi, m, s, k in zip(y, means, sds, labels):
        ax.text(m + s + 0.006, yi, f"{m:.3f}  ({variants[k]['features']} features)", va="center", fontsize=8, color=INK2)
    ax.axvline(data["base_rate"], color="#b9b8b2", ls="--", lw=1)
    ax.text(data["base_rate"], len(labels) - 0.4, " random", fontsize=7.5, color=INK2, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, max(means) + 0.15)
    ax.set_xlabel("PR-AUC, mean ± SD over 5 folds of the development customers")
    ax.grid(axis="y", visible=False)
    ax.set_title("How much does the model rely on missing readings?", pad=10)
    figstyle.save(fig, "fig-5-23-missingness")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated splits and missing-reading dependence")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument("--skip-missingness", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    if not args.plot_only:
        study = load_study()
        if not args.skip_missingness:
            data["missingness"] = missingness(study, args.device)
            OUT.write_text(json.dumps(data, indent=2))
        if not args.skip_splits:
            data["splits"] = repeated_splits(study, args.device)
            OUT.write_text(json.dumps(data, indent=2))
    if "splits" in data:
        plot_splits(data["splits"])
    if "missingness" in data:
        plot_missingness(data["missingness"])


if __name__ == "__main__":
    main()
