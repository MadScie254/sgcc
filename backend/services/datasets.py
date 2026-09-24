"""Uploaded datasets: parsing, scoring and the upload catalogue.

Two upload formats are accepted:
- consumption: SGCC layout, a customer id column (CONS_NO / CUSTOMER_ID /
  customer_id), an optional FLAG/label column and one column per date. Features
  are built exactly as in training.
- features: one row per customer with model feature columns (at least half of
  them), an optional id column and an optional label column.
"""

from __future__ import annotations

import io
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import UploadFile
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_loader import ID_COLUMNS, LABEL_COLUMNS, frame_to_wide, is_consumption_frame
from src.features import build_features_wide

from .config import get_config, get_paths
from .errors import NotFoundError
from .model import get_decision_threshold, get_feature_names, predict_proba, risk_tier
from .storage import read_json, write_json

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_ROWS = 200_000
MAX_COLUMNS = 2_000
MAX_DATASETS = 100  # older uploads are deleted
_ID_PATTERN = re.compile(r"[0-9]{14}-[0-9a-f]{8}")
_CATALOG_LOCK = Lock()


class DatasetError(ValueError):
    """The upload is not a dataset the model can score; the message says why."""


class UploadTooLargeError(DatasetError):
    pass


@dataclass
class ScoredDataset:
    format: str
    customer_ids: List[str]
    probabilities: np.ndarray
    labels: Optional[np.ndarray]
    days: Optional[int]
    features_found: int


async def read_upload(file: UploadFile) -> bytes:
    """Read an upload, stopping as soon as it exceeds MAX_UPLOAD_BYTES."""
    chunks, size = [], 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise UploadTooLargeError(f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        chunks.append(chunk)
    return b"".join(chunks)


def parse_csv(content: bytes) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(content), nrows=MAX_ROWS + 1, low_memory=False)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Could not read the file as CSV: {exc}") from exc
    if frame.empty:
        raise DatasetError("The CSV has no data rows")
    if len(frame) > MAX_ROWS or frame.shape[1] > MAX_COLUMNS:
        raise UploadTooLargeError(f"The CSV exceeds {MAX_ROWS:,} rows or {MAX_COLUMNS:,} columns")
    return frame


def _labels_from(frame: pd.DataFrame) -> Optional[np.ndarray]:
    column = next((c for c in LABEL_COLUMNS if c in frame.columns), None)
    if column is None:
        return None
    labels = pd.to_numeric(frame[column], errors="coerce").to_numpy()
    if np.isnan(labels).any() or not set(np.unique(labels)).issubset({0, 1}):
        raise DatasetError(f"Label column {column!r} must contain only 0 and 1")
    return labels.astype(int)


def score_frame(frame: pd.DataFrame) -> ScoredDataset:
    """Detect the format, build features if needed, and score every customer."""
    if is_consumption_frame(frame):
        try:
            wide, labels = frame_to_wide(frame, require_labels=False)
        except ValueError as exc:
            raise DatasetError(str(exc)) from exc
        features = build_features_wide(wide, get_config().get("features"))
        return ScoredDataset(
            format="consumption", customer_ids=list(features.index.astype(str)),
            probabilities=predict_proba(features),
            labels=None if labels is None else labels.reindex(features.index).to_numpy(),
            days=int(wide.shape[1]), features_found=len(get_feature_names()),
        )

    names = get_feature_names()
    found = [name for name in names if name in frame.columns]
    if len(found) < len(names) / 2:
        raise DatasetError(
            "Unrecognised layout. Upload either SGCC meter data (a CONS_NO column, an optional FLAG "
            "column and one column per date) or model features (one column per feature, "
            f"at least {len(names) // 2 + 1} of the {len(names)})."
        )
    id_column = next((c for c in ID_COLUMNS if c in frame.columns), None)
    ids = frame[id_column].astype(str).tolist() if id_column else [f"row-{i + 1}" for i in range(len(frame))]
    return ScoredDataset(
        format="features", customer_ids=ids, probabilities=predict_proba(frame),
        labels=_labels_from(frame), days=None, features_found=len(found),
    )


def summarize(scored: ScoredDataset, threshold: float, top_n: int = 10) -> Dict[str, Any]:
    p = scored.probabilities
    tiers = [risk_tier(x, threshold) for x in p]
    summary: Dict[str, Any] = {
        "format": scored.format,
        "customers": int(len(p)),
        "days": scored.days,
        "features_found": scored.features_found,
        "features_expected": len(get_feature_names()),
        "threshold": threshold,
        "flagged": int((p >= threshold).sum()),
        "tiers": {t: tiers.count(t) for t in ("high", "medium", "low")},
        "mean_probability": float(p.mean()),
        "labelled": scored.labels is not None,
        "label_metrics": None,
    }
    if scored.labels is not None:
        y = scored.labels
        predicted = p >= threshold
        tp = int((predicted & (y == 1)).sum())
        summary["label_metrics"] = {
            "theft": int(y.sum()),
            "caught": tp,
            "precision": tp / int(predicted.sum()) if predicted.any() else None,
            "recall": tp / int(y.sum()) if y.any() else None,
            "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            "pr_auc": float(average_precision_score(y, p)) if y.any() else None,
        }
    order = np.argsort(p)[::-1][:top_n]
    summary["top"] = [
        {"customer_id": scored.customer_ids[i], "probability": float(p[i]), "risk_tier": tiers[i],
         "label": None if scored.labels is None else int(scored.labels[i])}
        for i in order
    ]
    return summary


def scores_csv(scored: ScoredDataset, threshold: float) -> str:
    """One row per customer: id, probability, prediction, tier, and the label when present."""
    out = pd.DataFrame({
        "customer_id": scored.customer_ids,
        "probability": np.round(scored.probabilities, 6),
        "prediction": (scored.probabilities >= threshold).astype(int),
        "risk_tier": [risk_tier(x, threshold) for x in scored.probabilities],
    })
    if scored.labels is not None:
        out["label"] = scored.labels
    # Neutralise spreadsheet formula injection in echoed ids.
    out["customer_id"] = out["customer_id"].map(lambda v: "'" + v if v[:1] in ("=", "+", "-", "@", "\t", "\r") else v)
    return out.sort_values("probability", ascending=False).to_csv(index=False)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

def _uploads_dir():
    path = get_paths()["uploads"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_catalog() -> List[Dict[str, Any]]:
    return read_json(_uploads_dir() / "catalog.json", [])


def register_upload(filename: str, content: bytes) -> Dict[str, Any]:
    """Score an uploaded CSV, keep the file, and add it to the catalogue."""
    scored = score_frame(parse_csv(content))
    dataset_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    (_uploads_dir() / f"{dataset_id}.csv").write_bytes(content)
    item = {
        "dataset_id": dataset_id,
        "filename": (filename or "upload.csv")[:200],
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summarize(scored, get_decision_threshold()),
    }
    with _CATALOG_LOCK:
        catalog = [item, *_read_catalog()]
        for expired in catalog[MAX_DATASETS:]:
            if _ID_PATTERN.fullmatch(str(expired.get("dataset_id"))):
                (_uploads_dir() / f"{expired['dataset_id']}.csv").unlink(missing_ok=True)
        write_json(_uploads_dir() / "catalog.json", catalog[:MAX_DATASETS])
    return item


def list_datasets() -> List[Dict[str, Any]]:
    with _CATALOG_LOCK:
        return _read_catalog()


def get_dataset(dataset_id: str) -> Dict[str, Any]:
    item = next((i for i in list_datasets() if i.get("dataset_id") == dataset_id), None)
    if item is None:
        raise NotFoundError(f"Unknown dataset_id: {dataset_id}")
    return item


def load_dataset(dataset_id: str) -> ScoredDataset:
    """Re-score a stored upload with the model and threshold in service now."""
    if not _ID_PATTERN.fullmatch(dataset_id):
        raise NotFoundError(f"Unknown dataset_id: {dataset_id}")
    path = _uploads_dir() / f"{dataset_id}.csv"
    if not path.is_file():
        raise NotFoundError(f"Dataset file missing: {dataset_id}")
    return score_frame(parse_csv(path.read_bytes()))
