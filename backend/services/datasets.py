"""Uploaded datasets: parsing, scoring and the upload catalogue.

Two upload formats are accepted:
- consumption: SGCC layout, a customer id column (CONS_NO / CUSTOMER_ID /
  customer_id), an optional FLAG/label column and one column per date written
  year first. Features are built exactly as in training.
- features: one row per customer with every model feature, an optional id column
  and an optional label column.

Repeated customer ids, ambiguous dates and missing features are rejected with the
reason. The catalogue is in the database and the files in the blob store. A
consumption upload can be promoted to the operational population.
"""

from __future__ import annotations

import io
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import UploadFile
from sklearn.metrics import average_precision_score, roc_auc_score
from sqlalchemy import delete, insert, select

from src.data_loader import ID_COLUMNS, LABEL_COLUMNS, frame_to_wide, is_consumption_frame

from . import db
from .blobstore import get_blobstore
from .data import active_population, build_model_input
from .errors import NotFoundError
from .model import (
    NEEDS_READINGS, FeatureInputError, get_decision_threshold, get_feature_names, is_hybrid, predict_proba, require_features, risk_tier,
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_ROWS = 200_000
MAX_COLUMNS = 2_000
_ID_PATTERN = re.compile(r"[0-9]{14}-[0-9a-f]{8}")


class DatasetError(ValueError):
    """The upload is not a dataset the model can score; the message says why."""


class UploadTooLargeError(DatasetError):
    pass


class DatasetInUseError(RuntimeError):
    """The dataset is the operational population and cannot be deleted."""


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
        features = build_model_input(wide)
        return ScoredDataset(
            format="consumption", customer_ids=list(features.index.astype(str)),
            probabilities=predict_proba(features, wide),
            labels=None if labels is None else labels.reindex(features.index).to_numpy(),
            days=int(wide.shape[1]), features_found=len(get_feature_names()),
        )

    if not any(name in frame.columns for name in get_feature_names()):
        raise DatasetError(
            "Unrecognised layout. Upload either SGCC meter data (a CONS_NO column, an optional FLAG "
            "column and one column per date, written year first) or model features (one column per "
            f"feature, all {len(get_feature_names())} of them)."
        )
    if is_hybrid():
        raise DatasetError(NEEDS_READINGS)
    try:
        require_features(frame.columns)
    except FeatureInputError as exc:
        raise DatasetError(str(exc)) from exc
    id_column = next((c for c in ID_COLUMNS if c in frame.columns), None)
    ids = frame[id_column].astype(str).str.strip().tolist() if id_column else [f"row-{i + 1}" for i in range(len(frame))]
    repeated = pd.Index(ids)[pd.Index(ids).duplicated()].unique()
    if len(repeated):
        raise DatasetError(f"{len(repeated)} customer ids appear more than once: {', '.join(map(str, repeated[:10]))}")
    return ScoredDataset(
        format="features", customer_ids=ids, probabilities=predict_proba(frame),
        labels=_labels_from(frame), days=None, features_found=len(get_feature_names()),
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

def _blob_key(dataset_id: str) -> str:
    return f"uploads/{dataset_id}.csv"


def register_upload(filename: str, content: bytes, actor: str) -> Dict[str, Any]:
    """Score an uploaded CSV, keep the file, and add it to the catalogue."""
    scored = score_frame(parse_csv(content))
    dataset_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    get_blobstore().put(_blob_key(dataset_id), content, "text/csv")
    item = {
        "dataset_id": dataset_id,
        "filename": (filename or "upload.csv")[:200],
        "uploaded_at": db.now(),
        "uploaded_by": actor,
        "summary": summarize(scored, get_decision_threshold()),
    }
    with db.transaction() as connection:
        connection.execute(insert(db.datasets).values(**item, blob_key=_blob_key(dataset_id)))
    return item


def _public(row) -> Dict[str, Any]:
    return {key: row[key] for key in ("dataset_id", "filename", "uploaded_at", "uploaded_by", "summary")}


def list_datasets(limit: int = 100) -> List[Dict[str, Any]]:
    with db.transaction() as connection:
        rows = connection.execute(select(db.datasets).order_by(db.datasets.c.uploaded_at.desc()).limit(limit)).mappings().all()
    return [_public(row) for row in rows]


def _row(dataset_id: str):
    if not _ID_PATTERN.fullmatch(dataset_id):
        raise NotFoundError(f"Unknown dataset_id: {dataset_id}")
    with db.transaction() as connection:
        row = connection.execute(select(db.datasets).where(db.datasets.c.dataset_id == dataset_id)).mappings().first()
    if row is None:
        raise NotFoundError(f"Unknown dataset_id: {dataset_id}")
    return row


def get_dataset(dataset_id: str) -> Dict[str, Any]:
    return _public(_row(dataset_id))


def load_dataset(dataset_id: str) -> ScoredDataset:
    """Re-score a stored upload with the model and threshold in service now."""
    return score_frame(parse_csv(get_blobstore().get(_row(dataset_id)["blob_key"])))


def delete_dataset(dataset_id: str) -> None:
    row = _row(dataset_id)
    if active_population()["id"] == dataset_id:
        raise DatasetInUseError("This dataset is the operational population; switch the population first")
    get_blobstore().delete(row["blob_key"])
    with db.transaction() as connection:
        connection.execute(delete(db.datasets).where(db.datasets.c.dataset_id == dataset_id))


def promote_dataset(dataset_id: str, actor: str) -> Dict[str, Any]:
    """Make an uploaded meter-data file the operational population (its labels, if any, are ignored)."""
    row = _row(dataset_id)
    if row["summary"]["format"] != "consumption":
        raise DatasetError("Only meter-data uploads (one column per day) can become the operational population")
    db.set_setting("population", {"dataset_id": dataset_id, "filename": row["filename"], "blob_key": row["blob_key"]}, actor)
    return active_population()


def reset_population(actor: str) -> Dict[str, Any]:
    db.set_setting("population", None, actor)
    return active_population()


def retention_days() -> float:
    try:
        return max(float(os.getenv("RETENTION_DAYS", "90")), 0.0)
    except ValueError:
        return 90.0


def purge_expired_datasets() -> int:
    """Delete uploads older than RETENTION_DAYS (0 keeps them), except the operational population."""
    days = retention_days()
    if not days:
        return 0
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    keep = active_population()["id"]
    with db.transaction() as connection:
        expired = connection.execute(select(db.datasets.c.dataset_id, db.datasets.c.blob_key)
                                     .where(db.datasets.c.uploaded_at < cutoff, db.datasets.c.dataset_id != keep)).all()
    for dataset_id, blob_key in expired:
        get_blobstore().delete(blob_key)
        with db.transaction() as connection:
            connection.execute(delete(db.datasets).where(db.datasets.c.dataset_id == dataset_id))
    return len(expired)
