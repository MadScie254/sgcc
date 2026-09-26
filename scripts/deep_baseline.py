"""
Deep-learning baseline: a Wide & Deep CNN in the style of Zheng et al. (2018), on a CPU.

    pip install -r requirements-research.txt     # PyTorch (optional; not needed by the app)
    python scripts/deep_baseline.py              # ~15-30 min on 4 CPU cores; --device cuda on a GPU
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
from src.preprocessing import clean_series  # noqa: E402
from src.stats import paired_comparison  # noqa: E402
from src.study import load_study, save_extension_predictions  # noqa: E402

NAME, LABEL = "wide_deep_cnn", "Wide & Deep CNN (Zheng et al., 2018)"
OUT = ROOT / "artifacts" / "deep_baseline.json"
DAYS, WEEKS = 1036, 148

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


def prepare(wide: pd.DataFrame, cleaning: dict) -> tuple:
    """(values, missing mask) as float32 arrays of shape (customers, 1036)."""
    cleaned, _ = clean_series(wide, cleaning)
    values = cleaned.to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0)
    low, high = values.min(axis=1, keepdims=True), values.max(axis=1, keepdims=True)
    values = np.divide(values - low, high - low, out=np.zeros_like(values), where=(high - low) > 0)
    mask = wide.isna().to_numpy(dtype=np.float32)
    pad = DAYS - values.shape[1]
    values = np.pad(values, ((0, 0), (0, pad)))
    mask = np.pad(mask, ((0, 0), (0, pad)), constant_values=1.0)
    return values, mask


if nn is not None:
    class WideDeep(nn.Module):
        def __init__(self, filters: int = 15, hidden: int = 60):
            super().__init__()
            self.wide = nn.Sequential(nn.Linear(DAYS, hidden), nn.ReLU())
            self.deep = nn.Sequential(
                nn.Conv2d(2, filters, 3, padding=1), nn.ReLU(),
                nn.Conv2d(filters, filters, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(filters, filters, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Flatten(), nn.Linear(filters * (WEEKS // 4) * 7, hidden), nn.ReLU(), nn.Dropout(0.2),
            )
            self.head = nn.Linear(2 * hidden, 1)

        def forward(self, values, mask):
            image = torch.stack([values, mask], dim=1).view(-1, 2, WEEKS, 7)
            return self.head(torch.cat([self.wide(values), self.deep(image)], dim=1)).squeeze(1)


def predict(model, values, mask, device, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(values), batch):
            v = torch.from_numpy(values[i:i + batch]).to(device)
            m = torch.from_numpy(mask[i:i + batch]).to(device)
            out.append(torch.sigmoid(model(v, m)).cpu().numpy())
    return np.concatenate(out)


def train(values, mask, y, val_values, val_mask, y_val, device: str, seed: int = 42, max_epochs: int = 60,
          patience: int = 6, batch: int = 256):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WideDeep().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio, device=device))
    rng = np.random.default_rng(seed)
    best, best_state, history, waited = -1.0, None, [], 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(y))
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            v = torch.from_numpy(values[idx]).to(device)
            m = torch.from_numpy(mask[idx]).to(device)
            target = torch.from_numpy(y[idx].astype(np.float32)).to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(v, m), target)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(idx)
        score = float(average_precision_score(y_val, predict(model, val_values, val_mask, device)))
        history.append({"epoch": epoch, "train_loss": total / len(y), "validation_pr_auc": score})
        print(f"epoch {epoch}: loss {total / len(y):.4f}, validation PR-AUC {score:.4f}", flush=True)
        if score > best:
            best, waited = score, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Wide & Deep CNN baseline (Zheng et al., 2018) on the study's split")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if not args.plot_only:
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
