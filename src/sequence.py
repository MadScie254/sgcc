"""
SGCC Theft Detector - The sequence model: a Wide & Deep CNN on the daily readings

A network in the style of Zheng et al. (2018) that reads each customer's daily series
directly: a "wide" dense layer on the 1-D series and a "deep" convolutional part on the
series folded into a weeks x days image (with a second channel marking missing readings),
joined for one logit. It complements XGBoost, which reads 87 engineered features; the
two are blended into the hybrid pipeline (src.train).

Training needs PyTorch (requirements-research.txt) and is imported lazily. Serving does
not: training exports the network to ONNX, and ``OnnxNetwork`` scores it with
onnxruntime. ``prepare`` (numpy) is shared by both, so a customer is seen the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .preprocessing import clean_series

DAYS, WEEKS = 1036, 148  # 1,034 SGCC days padded to 148 whole weeks
TRAINING_DEFAULTS = {"max_epochs": 60, "patience": 6, "batch_size": 256, "learning_rate": 1e-3,
                     "weight_decay": 1e-5, "filters": 15, "hidden": 60, "dropout": 0.2}


# ---------------------------------------------------------------------------
# Input: shared by training and serving
# ---------------------------------------------------------------------------

def _fit_length(array: np.ndarray, fill: float) -> np.ndarray:
    """Keep the most recent DAYS days; a shorter history is padded at the start (as missing)."""
    if array.shape[1] >= DAYS:
        return array[:, -DAYS:] if array.shape[1] > DAYS else array
    return np.pad(array, ((0, 0), (DAYS - array.shape[1], 0)), constant_values=fill)


def prepare(wide: pd.DataFrame, cleaning: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    (values, missing mask), float32 arrays of shape (customers, 1036): readings cleaned as in
    section 3.7 and scaled to [0, 1] per customer; the mask marks readings missing before cleaning.
    The SGCC series (1,034 days) gets two padding days at the end, marked missing.
    """
    cleaned, _ = clean_series(wide, cleaning)
    values = np.nan_to_num(cleaned.to_numpy(dtype=np.float32), nan=0.0)
    low, high = values.min(axis=1, keepdims=True), values.max(axis=1, keepdims=True)
    values = np.divide(values - low, high - low, out=np.zeros_like(values), where=(high - low) > 0)
    mask = wide.isna().to_numpy(dtype=np.float32)
    if values.shape[1] < DAYS and values.shape[1] >= DAYS - 6:  # the SGCC layout: pad the final week
        pad = DAYS - values.shape[1]
        values = np.pad(values, ((0, 0), (0, pad)))
        mask = np.pad(mask, ((0, 0), (0, pad)), constant_values=1.0)
    return _fit_length(values, 0.0).astype(np.float32), _fit_length(mask, 1.0).astype(np.float32)


def day_index(n_days: int) -> np.ndarray:
    """For each of the DAYS input positions, the index of the original day it holds (-1 = padding)."""
    original = np.arange(n_days)
    if DAYS - 6 <= n_days < DAYS:
        return np.r_[original, np.full(DAYS - n_days, -1)]
    if n_days >= DAYS:
        return original[-DAYS:]
    return np.r_[np.full(DAYS - n_days, -1), original]


def sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logit, dtype=float)))


# ---------------------------------------------------------------------------
# Serving: ONNX
# ---------------------------------------------------------------------------

class OnnxNetwork:
    """The exported network, scored with onnxruntime (no PyTorch)."""

    def __init__(self, path: str | Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def logits(self, values: np.ndarray, mask: np.ndarray, batch: int = 2048) -> np.ndarray:
        out = [self.session.run(["logit"], {"values": values[i:i + batch].astype(np.float32),
                                            "mask": mask[i:i + batch].astype(np.float32)})[0]
               for i in range(0, len(values), batch)]
        return np.concatenate(out).reshape(-1) if out else np.zeros(0)

    def predict(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Raw theft score (sigmoid of the logit), before calibration."""
        return sigmoid(self.logits(values, mask))


def week_effects(network: "OnnxNetwork", values: np.ndarray, mask: np.ndarray) -> List[Dict[str, float]]:
    """
    How much each week of one customer's readings raised the network's raw score: the score minus
    the score with that week replaced by the customer's typical (median observed) day, marked as
    observed. Positive = the week pushed towards theft.
    """
    observed = values[mask < 0.5]
    typical = float(np.median(observed)) if observed.size else 0.0
    rows_v = np.repeat(values[None, :], WEEKS + 1, axis=0)
    rows_m = np.repeat(mask[None, :], WEEKS + 1, axis=0)
    for week in range(WEEKS):
        rows_v[week + 1, week * 7:(week + 1) * 7] = typical
        rows_m[week + 1, week * 7:(week + 1) * 7] = 0.0
    scores = network.predict(rows_v, rows_m)
    return [{"week": week, "effect": float(scores[0] - scores[week + 1])} for week in range(WEEKS)]


# ---------------------------------------------------------------------------
# Training: PyTorch (lazy)
# ---------------------------------------------------------------------------

def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is not installed: pip install -r requirements-research.txt") from exc
    return torch


def torch_available() -> bool:
    try:
        _torch()
        return True
    except RuntimeError:
        return False


def build_network(filters: int = 15, hidden: int = 60, dropout: float = 0.2):
    torch = _torch()
    nn = torch.nn

    class WideDeep(nn.Module):
        def __init__(self):
            super().__init__()
            self.wide = nn.Sequential(nn.Linear(DAYS, hidden), nn.ReLU())
            self.deep = nn.Sequential(
                nn.Conv2d(2, filters, 3, padding=1), nn.ReLU(),
                nn.Conv2d(filters, filters, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(filters, filters, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Flatten(), nn.Linear(filters * (WEEKS // 4) * 7, hidden), nn.ReLU(), nn.Dropout(dropout),
            )
            self.head = nn.Linear(2 * hidden, 1)

        def forward(self, values, mask):
            image = torch.stack([values, mask], dim=1).view(-1, 2, WEEKS, 7)
            return self.head(torch.cat([self.wide(values), self.deep(image)], dim=1)).squeeze(1)

    return WideDeep()


def predict_network(model, values: np.ndarray, mask: np.ndarray, device: str = "cpu", batch: int = 2048) -> np.ndarray:
    """Raw theft score of a PyTorch network (sigmoid of the logit)."""
    torch = _torch()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(values), batch):
            v = torch.from_numpy(values[i:i + batch]).to(device)
            m = torch.from_numpy(mask[i:i + batch]).to(device)
            out.append(torch.sigmoid(model(v, m)).cpu().numpy())
    return np.concatenate(out)


def train_network(values: np.ndarray, mask: np.ndarray, y: np.ndarray, val_values: np.ndarray, val_mask: np.ndarray,
                  y_val: np.ndarray, device: str = "cpu", seed: int = 42, settings: Optional[dict] = None,
                  log=print) -> Tuple[Any, List[Dict[str, float]]]:
    """
    Train on the training customers with a class-weighted loss and Adam; early stopping on validation
    PR-AUC keeps the best epoch's weights. Returns (model, per-epoch history).
    """
    torch = _torch()
    s = {**TRAINING_DEFAULTS, **(settings or {})}
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_network(s["filters"], s["hidden"], s["dropout"]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=s["learning_rate"], weight_decay=s["weight_decay"])
    ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio, device=device))
    rng = np.random.default_rng(seed)
    best, best_state, history, waited = -1.0, None, [], 0
    for epoch in range(1, int(s["max_epochs"]) + 1):
        model.train()
        order = rng.permutation(len(y))
        total = 0.0
        for i in range(0, len(order), int(s["batch_size"])):
            idx = order[i:i + int(s["batch_size"])]
            v = torch.from_numpy(values[idx]).to(device)
            m = torch.from_numpy(mask[idx]).to(device)
            target = torch.from_numpy(y[idx].astype(np.float32)).to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(v, m), target)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        score = float(average_precision_score(y_val, predict_network(model, val_values, val_mask, device)))
        history.append({"epoch": epoch, "train_loss": total / len(y), "validation_pr_auc": score})
        log(f"epoch {epoch}: loss {total / len(y):.4f}, validation PR-AUC {score:.4f}")
        if score > best:
            best, waited = score, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            waited += 1
            if waited >= int(s["patience"]):
                break
    model.load_state_dict(best_state)
    return model.cpu(), history


def export_onnx(model, path: str | Path) -> None:
    """Write the network as ONNX (inputs "values" and "mask", output "logit", any batch size)."""
    torch = _torch()
    model = model.cpu().eval()
    example = (torch.zeros(2, DAYS), torch.zeros(2, DAYS))
    kwargs = dict(input_names=["values", "mask"], output_names=["logit"], opset_version=17,
                  dynamic_axes={"values": {0: "batch"}, "mask": {0: "batch"}, "logit": {0: "batch"}})
    try:
        torch.onnx.export(model, example, str(path), dynamo=False, **kwargs)
    except TypeError:  # older PyTorch without the dynamo switch
        torch.onnx.export(model, example, str(path), **kwargs)


def parameter_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))
