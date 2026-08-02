from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional
import tempfile
import traceback
import uuid

import yaml

from src.train import train_pipeline

from backend.services.config import get_config


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


_JOB_LOCK = Lock()
_JOBS: Dict[str, TrainingJobRecord] = {}


def _touch(job: TrainingJobRecord) -> None:
    job.updated_at = datetime.now(timezone.utc)


def create_training_job(mode: str, config_overrides: Optional[Dict[str, Any]] = None) -> TrainingJobRecord:
    job = TrainingJobRecord(job_id=str(uuid.uuid4()), mode=mode)
    with _JOB_LOCK:
        _JOBS[job.job_id] = job
    return job


def get_training_job(job_id: str) -> TrainingJobRecord:
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise KeyError(f"Unknown training job: {job_id}")
    return job


def _write_temp_config(config_overrides: Optional[Dict[str, Any]]) -> Path:
    config = get_config()
    if config_overrides:
        for key, value in config_overrides.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value

    temp_file = Path(tempfile.gettempdir()) / f"sgcc_train_{uuid.uuid4().hex}.yaml"
    with open(temp_file, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return temp_file


def run_training_job(job_id: str, mode: str, config_overrides: Optional[Dict[str, Any]] = None) -> None:
    job = get_training_job(job_id)
    job.status = "running"
    job.current_step = "preparing configuration"
    _touch(job)

    temp_config = None
    try:
        if config_overrides:
            temp_config = _write_temp_config(config_overrides)
            config_path = str(temp_config)
        else:
            config_path = "config.yaml"

        job.current_step = "running training pipeline"
        _touch(job)
        quick_mode = mode == "quick"
        result = train_pipeline(config_path=config_path, quick_mode=quick_mode)

        job.status = "succeeded"
        job.current_step = "completed"
        job.best_score = float(result.get("best_score", 0.0)) if result else None
        job.result = result
        job.message = "Training completed successfully"
        _touch(job)
    except Exception as exc:  # pragma: no cover - surfaced via API response
        job.status = "failed"
        job.current_step = "failed"
        job.error = f"{exc}\n{traceback.format_exc()}"
        job.message = str(exc)
        _touch(job)
    finally:
        if temp_config and temp_config.exists():
            temp_config.unlink(missing_ok=True)


def serialize_training_job(job: TrainingJobRecord) -> Dict[str, Any]:
    payload = asdict(job)
    payload["created_at"] = job.created_at.isoformat()
    payload["updated_at"] = job.updated_at.isoformat()
    return payload
