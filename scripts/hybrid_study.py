"""
Hybrid ensemble: standard XGBoost on the engineered features blended with the Wide & Deep CNN
on the daily readings, checked on the six random splits of scripts/robustness.py.

    pip install -r requirements-research.txt
    python scripts/hybrid_study.py          # ~30 min on 4 CPU cores; --device cuda on a GPU
    python scripts/hybrid_study.py --plot-only

On each split both models are fitted on the training customers (XGBoost with its tuned
hyperparameters and early stopping, the CNN as in scripts/deep_baseline.py), and each is
Platt-calibrated on the validation customers. The hybrid's probability is
w * CNN + (1 - w) * XGBoost, with w chosen on the validation customers (grid of 0.05, highest
validation PR-AUC), recalibrated with Platt scaling on validation, and thresholded at the F1
maximum on validation. The test customers are scored once. Nothing is chosen on test customers.

``--significance`` adds, from the served model's saved test predictions (nothing refitted), a
paired stratified bootstrap and McNemar's test of the hybrid against standard XGBoost and the CNN.

Writes artifacts/hybrid_study.json and docs/thesis-figures/fig-5-26-hybrid-splits.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.metrics import average_precision_score  # noqa: E402

from src import figstyle  # noqa: E402
from src.calibration import apply_platt, fit_platt  # noqa: E402
from src.eval import classification_metrics  # noqa: E402
from src.figstyle import ORANGE, SLATE, plt  # noqa: E402
from src.modeling import select_threshold  # noqa: E402
from src.stats import paired_comparison  # noqa: E402
from src.study import fit_and_score, load_study  # noqa: E402

OUT = ROOT / "artifacts" / "hybrid_study.json"
SEEDS = (42, 1, 2, 3, 4, 5)
METRICS = ("pr_auc", "auc", "recall", "precision", "f1", "mcc", "gmean")
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)
TEAL = "#1baf7a"


def blend_weight(y_val, cnn_val, xgb_val) -> float:
    scores = [average_precision_score(y_val, w * cnn_val + (1 - w) * xgb_val) for w in WEIGHTS]
    return float(WEIGHTS[int(np.argmax(scores))])


def served_significance() -> dict:
    """The served hybrid against its two parts on the study's test customers, from the saved predictions."""
    import pandas as pd

    comparison = json.loads((ROOT / "models" / "baselines" / "comparison_results.json").read_text())
    if "hybrid" not in comparison:
        raise SystemExit("The served model is not the hybrid: run python -m src.train with PyTorch installed.")
    test = pd.read_csv(ROOT / "artifacts" / "predictions" / "test.csv.gz")
    y = test["label"].to_numpy(dtype=int)
    names = ("hybrid", "xgboost", "wide_deep_cnn")
    scores = {n: test[n].to_numpy(dtype=float) for n in names}
    flags = {n: scores[n] >= comparison[n]["threshold"] for n in names}
    pipelines, comparisons = paired_comparison(y, scores, flags, "hybrid")
    return {"reference": "hybrid", "resamples": 10_000, "customers": int(len(y)), "theft": int(y.sum()),
            "weight_sequence": comparison["hybrid"].get("weight_sequence"),
            "thresholds": {n: comparison[n]["threshold"] for n in names},
            "confusion_matrix": {n: comparison[n]["confusion_matrix"] for n in names},
            "pipelines": pipelines, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN + XGBoost hybrid on six random splits")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--significance", action="store_true", help="Only the served hybrid's paired tests")
    args = parser.parse_args()
    if args.significance:
        data = json.loads(OUT.read_text()) if OUT.exists() else {}
        data["served_split"] = served_significance()
        OUT.write_text(json.dumps(data, indent=2))
        for other, c in data["served_split"]["comparisons"].items():
            print(f"hybrid - {other}: " + ", ".join(
                f"{m} {c[m]['difference']:+.3f} [{c[m]['ci_low']:+.3f}, {c[m]['ci_high']:+.3f}] p={c[m]['p_holm']:.4f}"
                for m in ("pr_auc", "auc", "recall", "precision", "f1", "mcc")) + f"; McNemar p={c['mcnemar']['p_holm']:.3g}")
        return
    if not args.plot_only:
        import deep_baseline as cnn

        if cnn.torch is None:
            raise SystemExit("PyTorch is not installed. Run: pip install -r requirements-research.txt")
        study = load_study()
        values, mask = cnn.prepare(study.wide, study.config.get("preprocessing"))
        position = {c: i for i, c in enumerate(study.wide.index)}
        params = json.loads((study.artifacts / "tuning.json").read_text())["xgboost"]["params"]
        strategy = study.config["evaluation"].get("threshold_strategy", "f1")
        per_seed = {}
        for seed in SEEDS:
            train_idx, val_idx, test_idx = study.split(seed)
            xgb = fit_and_score(study, "xgboost", params, train_idx, val_idx, test_idx, device=args.device)
            rows = {k: np.array([position[c] for c in v]) for k, v in
                    (("train", train_idx), ("val", val_idx), ("test", test_idx))}
            y = study.y.to_numpy(dtype=int)
            y_val, y_test = y[rows["val"]], y[rows["test"]]
            model, _ = cnn.train(values[rows["train"]], mask[rows["train"]], y[rows["train"]],
                                 values[rows["val"]], mask[rows["val"]], y_val, args.device)
            raw_val = cnn.predict(model, values[rows["val"]], mask[rows["val"]], args.device)
            platt = fit_platt(raw_val, y_val)
            cnn_val = apply_platt(raw_val, platt)
            cnn_test = apply_platt(cnn.predict(model, values[rows["test"]], mask[rows["test"]], args.device), platt)
            weight = blend_weight(y_val, cnn_val, xgb["cal_val"])
            blend_val = weight * cnn_val + (1 - weight) * xgb["cal_val"]
            blend_test = weight * cnn_test + (1 - weight) * xgb["cal_test"]
            platt_h = fit_platt(blend_val, y_val)
            hybrid_val, hybrid_test = apply_platt(blend_val, platt_h), apply_platt(blend_test, platt_h)
            cnn_threshold = select_threshold(y_val, cnn_val, strategy=strategy)
            threshold = select_threshold(y_val, hybrid_val, strategy=strategy)
            per_seed[str(seed)] = {
                "weight_cnn": weight, "threshold": float(threshold),
                "hybrid": {m: float(v) for m, v in classification_metrics(y_test, hybrid_test, threshold).items() if m in METRICS},
                "xgboost": {m: float(xgb["metrics"][m]) for m in METRICS},
                "cnn": {m: float(v) for m, v in classification_metrics(y_test, cnn_test, cnn_threshold).items() if m in METRICS},
                "validation_pr_auc": {"hybrid": float(average_precision_score(y_val, hybrid_val)),
                                      "xgboost": float(xgb["validation_pr_auc"]),
                                      "cnn": float(average_precision_score(y_val, cnn_val))},
            }
            r = per_seed[str(seed)]
            print(f"seed {seed}: w={weight:.2f} hybrid {r['hybrid']['pr_auc']:.4f}, xgboost {r['xgboost']['pr_auc']:.4f}, "
                  f"cnn {r['cnn']['pr_auc']:.4f}", flush=True)
        summary = {}
        for model_name in ("hybrid", "xgboost", "cnn"):
            vals = {m: np.array([per_seed[str(s)][model_name][m] for s in SEEDS]) for m in METRICS}
            summary[model_name] = {"mean": {m: float(v.mean()) for m, v in vals.items()},
                                   "sd": {m: float(v.std(ddof=1)) for m, v in vals.items()}}
        wins = {other: int(sum(per_seed[str(s)]["hybrid"]["pr_auc"] > per_seed[str(s)][other]["pr_auc"] for s in SEEDS))
                for other in ("xgboost", "cnn")}
        OUT.write_text(json.dumps({"seeds": list(SEEDS), "weights_grid": WEIGHTS.tolist(), "per_seed": per_seed,
                                   "summary": summary, "hybrid_higher_pr_auc_than": wins}, indent=2))
        print("hybrid", {m: round(summary["hybrid"]["mean"][m], 4) for m in ("pr_auc", "auc", "recall", "precision")},
              "wins", wins)
    plot(json.loads(OUT.read_text()))


def plot(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    seeds = [str(s) for s in data["seeds"]]
    x = np.arange(len(seeds))
    for name, color, label in (("hybrid", TEAL, "Hybrid: CNN + XGBoost"), ("cnn", SLATE, "Wide & Deep CNN"),
                               ("xgboost", ORANGE, "XGBoost, no resampling")):
        vals = [data["per_seed"][s][name]["pr_auc"] for s in seeds]
        ax.plot(x, vals, marker="o", color=color, label=f"{label} ({np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(study)" if s == "42" else s for s in seeds])
    ax.set_xlabel("Random split (seed)")
    ax.set_ylabel("Test-set PR-AUC")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False)
    weights = sorted({data["per_seed"][s]["weight_cnn"] for s in seeds})
    chosen = f"CNN weight {weights[0]:.2f} on every split" if len(weights) == 1 else "CNN weight chosen per split"
    ax.set_title(f"Hybrid ensemble against its two parts on six random splits\n({chosen})", pad=10)
    figstyle.save(fig, "fig-5-26-hybrid-splits")


if __name__ == "__main__":
    main()
