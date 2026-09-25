"""
Statistical significance of the model comparison (proposal section 3.11).

    python scripts/significance.py                  # after python -m src.train; ~20 min on 4 cores
    python scripts/significance.py --device cuda    # XGBoost fits on an NVIDIA GPU

10-fold stratified cross-validation on the training and validation customers
(the test customers stay untouched). Every candidate pipeline in src.experiment
is refitted in every fold with the settings training chose: the tuned
hyperparameters, the number of trees early stopping found, and each
candidate's decision threshold. Treatments are fitted inside each fold.

The proposed pipeline is compared with every other candidate on each metric by
a paired t-test over the 10 folds (alpha = 0.05) and a Wilcoxon signed-rank
test, with Holm's correction across the comparisons for that metric.

Writes artifacts/significance.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_wide  # noqa: E402
from src.eval import classification_metrics  # noqa: E402
from src.experiment import CANDIDATES, fit_candidate  # noqa: E402
from src.pipeline import features_for  # noqa: E402
from src.train import load_config, split_customers  # noqa: E402

FOLDS = 10
ALPHA = 0.05
METRICS = ("recall", "precision", "f1", "auc", "pr_auc", "gmean", "mcc")
REFERENCE = "proposed"


def holm(p_values):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="10-fold paired comparison of the pipelines")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Where XGBoost trains")
    device = parser.parse_args().device
    config = load_config()
    eval_cfg = config["evaluation"]
    random_state = int(config["random_state"])
    artifacts = ROOT / config["paths"]["artifacts"]
    tuning = json.loads((artifacts / "tuning.json").read_text())
    curves = json.loads((artifacts / "learning_curves.json").read_text())
    comparison = json.loads((ROOT / config["paths"]["models"] / "baselines" / "comparison_results.json").read_text())

    wide, labels = load_wide(str(ROOT / config["data"]["training_data_path"]))
    y_all = labels.reindex(wide.index).astype(int)
    train_idx, val_idx, _ = split_customers(y_all, float(eval_cfg["validation_size"]), float(eval_cfg["test_size"]), random_state)
    development = train_idx.append(val_idx)
    y = y_all.loc[development]
    X_by = {prep: features_for(wide.loc[development], prep, config.get("preprocessing"), config.get("features"))
            for prep in ("raw", "clean")}

    params = {name: {**tuning[name]["params"], "n_estimators": curves[name]["best_iteration"]} for name in tuning}
    scores = {name: {m: [] for m in METRICS} for name in CANDIDATES}
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=random_state)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        for name, spec in CANDIDATES.items():
            X = X_by[spec["preprocessing"]]
            fitted = fit_candidate(name, X.iloc[tr], y.iloc[tr], params=params.get(name),
                                   resampling_config=config.get("resampling"), random_state=random_state, device=device)
            result = classification_metrics(y.iloc[te], fitted.predict(X.iloc[te]), comparison[name]["threshold"])
            for m in METRICS:
                scores[name][m].append(result[m])
        print(f"fold {fold}/{FOLDS}: " + ", ".join(f"{n} PR-AUC {scores[n]['pr_auc'][-1]:.3f}" for n in CANDIDATES), flush=True)

    others = [name for name in CANDIDATES if name != REFERENCE]
    tests = {}
    for m in METRICS:
        ref = np.array(scores[REFERENCE][m])
        t_p = [float(stats.ttest_rel(ref, scores[o][m]).pvalue) for o in others]
        w_p = [float(stats.wilcoxon(ref, scores[o][m]).pvalue) if np.any(ref != np.array(scores[o][m])) else 1.0 for o in others]
        t_adj, w_adj = holm(t_p), holm(w_p)
        tests[m] = {
            o: {
                "mean_difference": float(ref.mean() - np.mean(scores[o][m])),
                "t_test_p": t_p[k], "t_test_p_holm": float(t_adj[k]),
                "wilcoxon_p": w_p[k], "wilcoxon_p_holm": float(w_adj[k]),
                "significant": bool(t_adj[k] < ALPHA),
            }
            for k, o in enumerate(others)
        }

    summary = {name: {m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for m, v in per.items()}
               for name, per in scores.items()}
    out = {"folds": FOLDS, "alpha": ALPHA, "reference": REFERENCE, "customers": int(len(y)),
           "summary": summary, "per_fold": scores, "tests": tests}
    (artifacts / "significance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    table = pd.DataFrame({n: {m: f"{summary[n][m]['mean']:.3f} ± {summary[n][m]['sd']:.3f}" for m in METRICS} for n in CANDIDATES})
    print(table.to_string())


if __name__ == "__main__":
    main()
