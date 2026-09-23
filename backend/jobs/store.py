from __future__ import annotations

import copy
import logging
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import yaml

from backend.services.config import get_config
from backend.services.model import clear_caches
from src.train import train_pipeline

logger = logging.getLogger(__name__)

MAX_JOB_HISTORY = 50


@dataclass
class TrainingJobRecord:
    job_id: str
    mode: str
    status: str = "queued"
    current_step: str = "queued"
    best_score: Optional[float] = None
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TrainingBusyError(RuntimeError):
    pass


_JOB_LOCK = Lock()
_JOBS: Dict[str, TrainingJobRecord] = {}


def _update(job: TrainingJobRecord, **changes: Any) -> None:
    with _JOB_LOCK:
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = datetime.now(timezone.utc)


def create_training_job(mode: str) -> TrainingJobRecord:
    """Register a job; only one may be queued or running at a time."""
    with _JOB_LOCK:
        if any(job.status in {"queued", "running"} for job in _JOBS.values()):
            raise TrainingBusyError("A training job is already in progress")
        job = TrainingJobRecord(job_id=str(uuid.uuid4()), mode=mode)
        _JOBS[job.job_id] = job
        for stale in sorted(_JOBS.values(), key=lambda j: j.created_at)[:-MAX_JOB_HISTORY]:
            _JOBS.pop(stale.job_id, None)
    return job


def get_training_job(job_id: str) -> TrainingJobRecord:
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise KeyError(f"Unknown training job: {job_id}")
    return job


def _write_temp_config(overrides: Dict[str, Any]) -> Path:
    config = copy.deepcopy(get_config())
    optuna_cfg = config["model"]["optuna"]
    quick_cfg = config["model"]["quick_train"]
    for key in ("n_trials", "cv_folds"):
        if overrides.get(key) is not None:
            optuna_cfg[key] = quick_cfg[key] = overrides[key]
    if overrides.get("timeout_seconds") is not None:
        optuna_cfg["timeout"] = overrides["timeout_seconds"]
    if overrides.get("test_size") is not None:
        config["evaluation"]["test_size"] = overrides["test_size"]

    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="sgcc_train_", delete=False, encoding="utf-8")
    with handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return Path(handle.name)


def run_training_job(job_id: str, mode: str, overrides: Optional[Dict[str, Any]] = None) -> None:
    job = get_training_job(job_id)
    _update(job, status="running", current_step="running training pipeline")

    temp_config = None
    try:
        overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
        temp_config = _write_temp_config(overrides) if overrides else None
        result = train_pipeline(config_path=str(temp_config or "config.yaml"), quick_mode=mode == "quick")
        clear_caches()
        _update(
            job, status="succeeded", current_step="completed", result=result,
            best_score=float(result.get("best_score", 0.0)), message="Training completed successfully",
        )
    except Exception as exc:  # surfaced via the job status, not raised
        logger.exception("Training job %s failed", job_id)
        _update(job, status="failed", current_step="failed", message=str(exc), error=type(exc).__name__)
    finally:
        if temp_config is not None:
            temp_config.unlink(missing_ok=True)


def serialize_training_job(job: TrainingJobRecord) -> Dict[str, Any]:
    with _JOB_LOCK:
        payload = asdict(job)
    payload["created_at"] = job.created_at.isoformat()
    payload["updated_at"] = job.updated_at.isoformat()
    return payload
