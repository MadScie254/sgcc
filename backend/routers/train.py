from __future__ import annotations

import asyncio
import json
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from backend.jobs.store import (
    TrainingBusyError,
    create_training_job,
    get_training_job,
    run_training_job,
    serialize_training_job,
)
from backend.schemas.api import TrainingJobCreateRequest, TrainingJobStatusResponse

router = APIRouter(prefix="/api/train", tags=["train"])


def _training_enabled() -> bool:
    # Retraining overwrites the served model, so it is opt-in outside development.
    if os.getenv("ENV", "development").lower() == "development":
        return True
    return os.getenv("ENABLE_TRAINING_API", "").lower() in {"1", "true", "yes"}


@router.post("/jobs", response_model=TrainingJobStatusResponse)
def create_job(payload: TrainingJobCreateRequest, background_tasks: BackgroundTasks) -> TrainingJobStatusResponse:
    if not _training_enabled():
        raise HTTPException(status_code=403, detail="Training API is disabled; set ENABLE_TRAINING_API=1 to allow it")
    try:
        job = create_training_job(payload.mode)
    except TrainingBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    overrides = payload.config_overrides.model_dump() if payload.config_overrides else None
    background_tasks.add_task(run_training_job, job.job_id, payload.mode, overrides)
    return TrainingJobStatusResponse(**serialize_training_job(job))


@router.get("/jobs/{job_id}", response_model=TrainingJobStatusResponse)
def get_job(job_id: str) -> TrainingJobStatusResponse:
    try:
        job = get_training_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TrainingJobStatusResponse(**serialize_training_job(job))


@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    async def event_stream():
        while True:
            try:
                job = serialize_training_job(get_training_job(job_id))
            except KeyError:
                yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
                break

            yield f"data: {json.dumps(job, default=str)}\n\n"
            if job["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
