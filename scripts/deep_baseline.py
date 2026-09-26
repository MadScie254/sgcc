"""
Deep-learning baseline: a Wide & Deep CNN in the style of Zheng et al. (2018), on a CPU.

    pip install -r requirements-research.txt     # PyTorch (optional; not needed by the app)
    python scripts/deep_baseline.py              # ~15-30 min on 4 CPU cores; --device cuda on a GPU
    python scripts/deep_baseline.py --splits     # the five other splits of robustness.py (~20 min)
    python scripts/deep_baseline.py --plot-only

Zheng et al. released the SGCC data with a model that reads the daily series directly:
a "wide" fully connected part on the one-dimensional series and a "deep" convolutional
part on the series folded into weeks (a two-dimensional weeks x days image), whose
outputs are joined for the prediction. This re-implementation follows that design at
a size a CPU trains in minutes; it is a baseline, not a reproduction of their numbers.

- Input: readings cleaned as in section 3.7 (src.preprocessing.clean_series), scaled to
  [0, 1] per customer, padded from 1,034 to 1,036 days (148 weeks); the deep part also
  sees a second channel marking which readings were missing before cleaning.
- Wide: 1,036 -> 60 units. Deep: three 3x3 convolutions (15 filters), pooling over weeks,
  -> 60 units. Joined -> 1 logit.
- Training: the same training customers; binary cross-entropy weighted by the class
  ratio; Adam; early stopping on validation PR-AUC (patience 6, at most 60 epochs); the
  best epoch's weights are kept. Seeds fixed.
- Evaluation as for every other pipeline: Platt calibration and the F1 threshold on
  validation customers, one scoring of the test customers, a paired bootstrap and
  McNemar's test against standard XGBoost, and training time, inference time and size.

Writes artifacts/deep_baseline.json, adds "wide_deep_cnn" to
artifacts/predictions/extensions_{validation,test}.csv.gz, and draws
docs/thesis-figures/fig-5-25-extension-pr-curves (precision-recall curves of the
served, proposed, raw-readings SMOTE+ENN and CNN pipelines).
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

from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from src import figstyle  # noqa: E402
from src.calibration import apply_platt, fit_platt  # noqa: E402
from src.eval import classification_metrics  # noqa: E402
from src.figstyle import INK2, MUTED, PIPELINE_COLORS, plt  # noqa: E402
from src.modeling import select_threshold  # noqa: E402
from src.stats import paired_comparison  # noqa: E402
from src.study import load_study, save_extension_predictions  # noqa: E402

NAME, LABEL = "wide_deep_cnn", "Wide & Deep CNN (Zheng et al., 2018)"
OUT = ROOT / "artifacts" / "deep_baseline.json"
METRICS = ("pr_auc", "auc", "recall", "precision", "f1", "mcc", "gmean")

from src.sequence import DAYS, build_network, predict_network, prepare, train_network  # noqa: E402,F401

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def WideDeep(filters: int = 15, hidden: int = 60):  # noqa: N802 - kept for the tests' and scripts' use
    return build_network(filters, hidden)


def predict(model, values, mask, device, batch: int = 2048) -> np.ndarray:
    return predict_network(model, values, mask, device, batch)


def train(values, mask, y, val_values, val_mask, y_val, device: str, seed: int = 42, max_epochs: int = 60,
          patience: int = 6, batch: int = 256):
    return train_network(values, mask, y, val_values, val_mask, y_val, device, seed,
                         {"max_epochs": max_epochs, "patience": patience, "batch_size": batch},
                         log=lambda line: print(line, flush=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Wide & Deep CNN baseline (Zheng et al., 2018) on the study's split")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--splits", action="store_true",
                        help="Also train on the five other random splits of scripts/robustness.py (seeds 1-5)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.splits:
        other_splits(args)
    elif not args.plot_only:
        if torch is None:
            raise SystemExit("PyTorch is not installed. Run: pip install -r requirements-research.txt")
        torch.set_num_threads(max(1, torch.get_num_threads()))
        study = load_study()
        values, mask = prepare(study.wide, study.config.get("preprocessing"))
        position = {c: i for i, c in enumerate(study.wide.index)}
        rows = {s: np.array([position[c] for c in idx]) for s, idx in
                (("train", study.train_idx), ("val", study.val_idx), ("test", study.test_idx))}
        y = study.y.to_numpy(dtype=int)
        start = time.perf_counter()
        model, history = train(values[rows["train"]], mask[rows["train"]], y[rows["train"]],
                               values[rows["val"]], mask[rows["val"]], y[rows["val"]], args.device,
                               max_epochs=args.max_epochs)
        fit_seconds = time.perf_counter() - start
        raw_val = predict(model, values[rows["val"]], mask[rows["val"]], args.device)
        platt = fit_platt(raw_val, y[rows["val"]])
        cal_val = apply_platt(raw_val, platt)
        threshold = select_threshold(y[rows["val"]], cal_val, strategy=study.config["evaluation"].get("threshold_strategy", "f1"))
        start = time.perf_counter()
        raw_test = predict(model, values[rows["test"]], mask[rows["test"]], args.device)
        inference_ms = 1000 * (time.perf_counter() - start) / len(rows["test"])
        cal_test = apply_platt(raw_test, platt)
        y_test = y[rows["test"]]
        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        save_extension_predictions(study, "validation", study.val_idx, study.y.loc[study.val_idx],
                                   {f"{NAME}_raw": raw_val, NAME: cal_val})
        save_extension_predictions(study, "test", study.test_idx, study.y.loc[study.test_idx],
                                   {f"{NAME}_raw": raw_test, NAME: cal_test})

        saved = pd.read_csv(study.artifacts / "predictions" / "test.csv.gz", dtype={"customer_id": str}).set_index("customer_id")
        saved = saved.loc[pd.Index(study.test_idx).astype(str)]
        assert (saved["label"].to_numpy(dtype=int) == y_test).all(), "saved predictions do not match the split"
        served_threshold = json.loads((ROOT / "models" / "baselines" / "comparison_results.json").read_text())["xgboost"]["threshold"]
        scores = {NAME: cal_test, "xgboost": saved["xgboost"].to_numpy()}
        flags = {NAME: cal_test >= threshold, "xgboost": scores["xgboost"] >= served_threshold}
        pipelines, comparisons = paired_comparison(y_test, scores, flags, NAME)
        result = {
            "pipeline": NAME, "label": LABEL,
            "architecture": {"wide": f"{DAYS} -> 60", "deep": "3 conv 3x3 (15 filters) on 148x7 weeks, 2 channels -> 60",
                             "parameters": int(sum(p.numel() for p in model.parameters()))},
            "training": {"epochs_run": len(history), "best_epoch": int(np.argmax([h["validation_pr_auc"] for h in history]) + 1),
                         "history": history, "device": args.device, "torch": torch.__version__},
            "result": {"threshold": float(threshold), "platt": platt,
                       "validation_pr_auc": float(average_precision_score(y[rows["val"]], raw_val)),
                       "metrics": classification_metrics(y_test, cal_test, threshold),
                       "training_time": round(fit_seconds, 1), "inference_ms_per_customer": inference_ms,
                       "model_size_mb": round(size_mb, 3)},
            "bootstrap": {"reference": NAME, "resamples": 10_000, "pipelines": pipelines, "comparisons": comparisons},
        }
        OUT.write_text(json.dumps(result, indent=2))
        m = result["result"]["metrics"]
        print(f"test: PR-AUC {m['pr_auc']:.4f}, ROC-AUC {m['auc']:.4f}, recall {m['recall']:.4f}, "
              f"precision {m['precision']:.4f}; {fit_seconds:.0f} s training")
    plot()


def other_splits(args) -> None:
    """
    The same network trained, calibrated and thresholded on each of the other random splits used in
    scripts/robustness.py, next to standard XGBoost's results on the same splits (from robustness.json).
    """
    if torch is None:
        raise SystemExit("PyTorch is not installed. Run: pip install -r requirements-research.txt")
    data = json.loads(OUT.read_text())
    robustness = json.loads((ROOT / "artifacts" / "robustness.json").read_text())["splits"]
    study = load_study()
    values, mask = prepare(study.wide, study.config.get("preprocessing"))
    position = {c: i for i, c in enumerate(study.wide.index)}
    y = study.y.to_numpy(dtype=int)
    per_seed = {"42": {m: float(data["result"]["metrics"][m]) for m in METRICS}}
    for seed in robustness["seeds"]:
        if seed == 42:
            continue
        idx = dict(zip(("train", "val", "test"), study.split(seed)))
        rows = {k: np.array([position[c] for c in v]) for k, v in idx.items()}
        model, history = train(values[rows["train"]], mask[rows["train"]], y[rows["train"]],
                               values[rows["val"]], mask[rows["val"]], y[rows["val"]], args.device,
                               max_epochs=args.max_epochs)
        raw_val = predict(model, values[rows["val"]], mask[rows["val"]], args.device)
        platt = fit_platt(raw_val, y[rows["val"]])
        threshold = select_threshold(y[rows["val"]], apply_platt(raw_val, platt),
                                     strategy=study.config["evaluation"].get("threshold_strategy", "f1"))
        cal_test = apply_platt(predict(model, values[rows["test"]], mask[rows["test"]], args.device), platt)
        metrics = classification_metrics(y[rows["test"]], cal_test, threshold)
        per_seed[str(seed)] = {m: float(metrics[m]) for m in METRICS}
        per_seed[str(seed)]["epochs_run"] = len(history)
        print(f"seed {seed}: PR-AUC {metrics['pr_auc']:.4f} (standard XGBoost "
              f"{robustness['pipelines']['xgboost']['per_seed'][str(seed)]['pr_auc']:.4f})", flush=True)
    seeds = [str(s) for s in robustness["seeds"]]
    xgb = robustness["pipelines"]["xgboost"]["per_seed"]
    values_by = {m: np.array([per_seed[s][m] for s in seeds]) for m in METRICS}
    data["splits"] = {
        "seeds": robustness["seeds"], "per_seed": per_seed,
        "mean": {m: float(v.mean()) for m, v in values_by.items()},
        "sd": {m: float(v.std(ddof=1)) for m, v in values_by.items()},
        "difference_from_standard_xgboost": {m: [per_seed[s][m] - xgb[s][m] for s in seeds] for m in METRICS},
        "higher_pr_auc_than_standard_xgboost": int(sum(per_seed[s]["pr_auc"] > xgb[s]["pr_auc"] for s in seeds)),
        "note": "Same architecture and training settings on every split; standard XGBoost from robustness.json.",
    }
    OUT.write_text(json.dumps(data, indent=2))
    print(f"mean PR-AUC {data['splits']['mean']['pr_auc']:.4f} ± {data['splits']['sd']['pr_auc']:.4f}; "
          f"above standard XGBoost on {data['splits']['higher_pr_auc_than_standard_xgboost']} of {len(seeds)} splits")


def plot() -> None:
    test = pd.read_csv(ROOT / "artifacts" / "predictions" / "test.csv.gz", dtype={"customer_id": str}).set_index("customer_id")
    path = ROOT / "artifacts" / "predictions" / "extensions_test.csv.gz"
    extra = pd.read_csv(path, dtype={"customer_id": str}).set_index("customer_id").loc[test.index] if path.exists() else pd.DataFrame()
    y = test["label"].to_numpy(int)
    curves = [("xgboost", "XGBoost, no resampling (served)", test), ("proposed", "SMOTE+ENN + XGBoost (proposed)", test),
              ("xgboost_smote_enn_raw", "SMOTE+ENN + XGBoost, raw readings", extra), (NAME, LABEL, extra)]
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for name, label, frame in curves:
        if name not in frame.columns:
            continue
        p = frame[name].to_numpy(float)
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, color=PIPELINE_COLORS[name], lw=1.8,
                label=f"{label} ({average_precision_score(y, p):.3f})")
    ax.axhline(y.mean(), color=MUTED, ls="--", lw=1, label=f"Random ({y.mean():.3f})")
    ax.set(xlabel="Recall (thefts caught)", ylabel="Precision (hit rate of flags)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Precision-recall on the test customers, with the added pipelines", pad=10)
    ax.text(0.01, 0.01, "PR-AUC in brackets", transform=ax.transAxes, fontsize=7.5, color=INK2)
    figstyle.save(fig, "fig-5-25-extension-pr-curves")


if __name__ == "__main__":
    main()
