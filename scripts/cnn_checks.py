"""
Two checks of the sequence model behind the served hybrid, on the study's split.

    pip install -r requirements-research.txt
    python scripts/cnn_checks.py            # ~40 min on 4 CPU cores; --device cuda on a GPU
    python scripts/cnn_checks.py --plot-only

1. Training seeds. The CNN is retrained with five seeds (42, the served one, and 1-4); each is
   Platt-calibrated on validation and blended with the served standard XGBoost (saved calibrated
   predictions) exactly as training does: weight chosen on validation PR-AUC, the blend
   recalibrated and thresholded on validation, the test customers scored once.
2. Missing readings. The CNN is retrained without the channel that marks missing readings
   (it then sees only the cleaned, gap-filled series), and again with September 2016 (the month
   around the date absent from the release) also replaced by each customer's median observed
   reading. Each variant is blended with the served XGBoost, and the second is also blended with
   XGBoost on consumption behaviour only (cleaned readings, no missing-reading features; tuned
   hyperparameters held fixed), a hybrid with no access to where readings are missing.

Writes artifacts/cnn_checks.json and docs/thesis-figures/fig-5-28-cnn-checks.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.metrics import average_precision_score  # noqa: E402

from src import figstyle  # noqa: E402
from src.calibration import apply_platt, fit_platt  # noqa: E402
from src.eval import classification_metrics  # noqa: E402
from src.figstyle import ORANGE, SLATE, plt  # noqa: E402
from src.modeling import select_threshold  # noqa: E402
from src.study import fit_and_score, load_study  # noqa: E402

OUT = ROOT / "artifacts" / "cnn_checks.json"
METRICS = ("pr_auc", "auc", "recall", "precision", "f1", "mcc")
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)
SEEDS = (42, 1, 2, 3, 4)
SMOKE = bool(os.environ.get("SGCC_SMOKE"))  # tiny run to test the script end to end
if SMOKE:
    OUT = Path(os.environ["SGCC_SMOKE"]) / "cnn_checks.json"
    SEEDS = (42, 1)
TEAL = "#1baf7a"
MISSING_FEATURES = ("missing_ratio", "missing_ratio_first_third", "missing_ratio_middle_third",
                    "missing_ratio_last_third", "longest_missing_run", "missing_sequences_count",
                    "first_obs_frac", "last_obs_frac")


def blend(y_val, y_test, cnn_val, cnn_test, xgb_val, xgb_test):
    scores = [average_precision_score(y_val, w * cnn_val + (1 - w) * xgb_val) for w in WEIGHTS]
    w = float(WEIGHTS[int(np.argmax(scores))])
    platt = fit_platt(w * cnn_val + (1 - w) * xgb_val, y_val)
    val = apply_platt(w * cnn_val + (1 - w) * xgb_val, platt)
    test = apply_platt(w * cnn_test + (1 - w) * xgb_test, platt)
    threshold = float(select_threshold(y_val, val, strategy="f1"))
    return {"weight_cnn": w, "threshold": threshold,
            **{m: float(v) for m, v in classification_metrics(y_test, test, threshold).items() if m in METRICS}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed and missing-reading checks of the CNN")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        from src.sequence import predict_network, prepare, torch_available, train_network

        if not torch_available():
            raise SystemExit("PyTorch is not installed. Run: pip install -r requirements-research.txt")
        study = load_study()
        values, mask = prepare(study.wide, study.config.get("preprocessing"))
        position = {c: i for i, c in enumerate(study.wide.index)}
        rows = {k: np.array([position[c] for c in v]) for k, v in
                (("train", study.train_idx), ("val", study.val_idx), ("test", study.test_idx))}
        yy = study.y.loc[study.wide.index].to_numpy(dtype=int)
        y_train, y_val, y_test = (yy[rows[k]] for k in ("train", "val", "test"))
        settings = {**study.config["model"]["hybrid"].get("sequence", {}), **({"max_epochs": 1} if SMOKE else {})}

        def saved(split):
            frame = pd.read_csv(study.artifacts / "predictions" / f"{split}.csv.gz", dtype={"customer_id": str})
            return frame.set_index("customer_id")
        xgb_val = saved("validation")["xgboost"].reindex(pd.Index(study.val_idx).astype(str)).to_numpy()
        xgb_test = saved("test")["xgboost"].reindex(pd.Index(study.test_idx).astype(str)).to_numpy()
        assert not np.isnan(xgb_val).any() and not np.isnan(xgb_test).any()

        def cnn(v, m, seed):
            model, history = train_network(v[rows["train"]], m[rows["train"]], y_train, v[rows["val"]], m[rows["val"]],
                                           y_val, args.device, seed, settings, log=lambda line: None)
            raw_val = predict_network(model, v[rows["val"]], m[rows["val"]], args.device)
            platt = fit_platt(raw_val, y_val)
            cal_val = apply_platt(raw_val, platt)
            cal_test = apply_platt(predict_network(model, v[rows["test"]], m[rows["test"]], args.device), platt)
            threshold = float(select_threshold(y_val, cal_val, strategy="f1"))
            metrics = {m_: float(x) for m_, x in classification_metrics(y_test, cal_test, threshold).items() if m_ in METRICS}
            return cal_val, cal_test, {"epochs": len(history), "threshold": threshold, **metrics}

        seeds = {}
        for seed in SEEDS:
            cal_val, cal_test, cnn_metrics = cnn(values, mask, seed)
            seeds[str(seed)] = {"cnn": cnn_metrics, "hybrid": blend(y_val, y_test, cal_val, cal_test, xgb_val, xgb_test)}
            print(f"seed {seed}: cnn {cnn_metrics['pr_auc']:.4f}, hybrid {seeds[str(seed)]['hybrid']['pr_auc']:.4f}", flush=True)

        no_mask = np.zeros_like(mask)
        september = np.flatnonzero((study.wide.columns >= "2016-09-01") & (study.wide.columns <= "2016-09-30"))
        typical = np.array([np.median(v[m < 0.5]) if (m < 0.5).any() else 0.0 for v, m in zip(values, mask)],
                           dtype=np.float32)
        no_september = values.copy()
        no_september[:, september] = typical[:, None]

        clean = study.X_by["clean"]
        behaviour = clean.drop(columns=[c for c in MISSING_FEATURES if c in clean.columns])
        xgb_b = fit_and_score(study, "xgboost", study.tuned_params("xgboost"), study.train_idx, study.val_idx,
                              study.test_idx, X=behaviour, device=args.device)
        variants = {}
        for key, label, v, m in (("full", "Served inputs (readings and missing-reading channel)", values, mask),
                                 ("no_mask", "No missing-reading channel", values, no_mask),
                                 ("no_mask_no_september", "No missing-reading channel, September 2016 replaced",
                                  no_september, no_mask)):
            cal_val, cal_test, cnn_metrics = cnn(v, m, 42)
            entry = {"label": label, "cnn": cnn_metrics,
                     "hybrid": blend(y_val, y_test, cal_val, cal_test, xgb_val, xgb_test)}
            if key == "no_mask_no_september":
                entry["hybrid_behaviour_only"] = blend(y_val, y_test, cal_val, cal_test, xgb_b["cal_val"], xgb_b["cal_test"])
            variants[key] = entry
            print(f"{label}: cnn {cnn_metrics['pr_auc']:.4f}, hybrid {entry['hybrid']['pr_auc']:.4f}", flush=True)
        summary = {part: {m: {"mean": float(np.mean([seeds[s][part][m] for s in seeds])),
                              "sd": float(np.std([seeds[s][part][m] for s in seeds], ddof=1))} for m in METRICS}
                   for part in ("cnn", "hybrid")}
        data = {"seeds": seeds, "seed_summary": summary, "variants": variants,
                "xgboost_behaviour_only": {m: float(v) for m, v in xgb_b["metrics"].items() if m in METRICS},
                "september_days": int(len(september)), "missing_features_removed": list(MISSING_FEATURES)}
        OUT.write_text(json.dumps(data, indent=2))
        print("seed summary", {p: (round(summary[p]["pr_auc"]["mean"], 4), round(summary[p]["pr_auc"]["sd"], 4))
                               for p in summary}, flush=True)
    plot(json.loads(OUT.read_text()))


def plot(data: dict) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
    seeds = list(data["seeds"])
    x = np.arange(len(seeds))
    for part, color, label in (("hybrid", TEAL, "Hybrid"), ("cnn", SLATE, "CNN alone")):
        vals = [data["seeds"][s][part]["pr_auc"] for s in seeds]
        left.plot(x, vals, marker="o", color=color, label=f"{label} ({np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f})")
    left.set_xticks(x)
    left.set_xticklabels([f"{s}\n(served)" if s == "42" else s for s in seeds])
    left.set_xlabel("CNN training seed")
    left.set_ylabel("Test-set PR-AUC")
    left.set_title("Five training seeds", pad=8)
    left.legend(fontsize=8.5, loc="lower center", frameon=False)

    rows = [(v["label"], v["cnn"]["pr_auc"], v["hybrid"]["pr_auc"]) for v in data["variants"].values()]
    rows.append(("Behaviour only: no channel, no September\n2016, XGBoost without missing-reading features",
                 None, data["variants"]["no_mask_no_september"]["hybrid_behaviour_only"]["pr_auc"]))
    y = np.arange(len(rows))[::-1]
    for yi, (label, c, h) in zip(y, rows):
        if c is not None:
            right.plot([c], [yi + 0.12], "o", color=SLATE)
            right.annotate(f"{c:.3f}", (c, yi + 0.12), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        right.plot([h], [yi - 0.12], "o", color=TEAL)
        right.annotate(f"{h:.3f}", (h, yi - 0.12), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8)
    right.axvline(0.513, color=ORANGE, lw=1, ls="--")
    right.text(0.513, y[0] + 0.45, "XGBoost alone 0.513", color=ORANGE, fontsize=8, ha="center")
    right.set_yticks(y)
    right.set_yticklabels([r[0] for r in rows], fontsize=8)
    right.set_ylim(-0.6, len(rows) - 0.3)
    right.set_xlabel("Test-set PR-AUC (grey: CNN alone; green: hybrid)")
    right.set_title("Without the missing-reading information", pad=8)
    fig.suptitle("Checks of the sequence model behind the served hybrid", y=1.02)
    if not SMOKE:
        figstyle.save(fig, "fig-5-28-cnn-checks")


if __name__ == "__main__":
    main()
