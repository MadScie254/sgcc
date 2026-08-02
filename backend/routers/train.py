from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from backend.jobs.store import create_training_job, get_training_job, run_training_job, serialize_training_job
from backend.schemas.api import TrainingJobCreateRequest, TrainingJobStatusResponse

router = APIRouter(prefix="/api/train", tags=["train"])


@router.post("/jobs", response_model=TrainingJobStatusResponse)
def create_job(payload: TrainingJobCreateRequest, background_tasks: BackgroundTasks) -> TrainingJobStatusResponse:
    job = create_training_job(payload.mode, payload.config_overrides)
    background_tasks.add_task(run_training_job, job.job_id, payload.mode, payload.config_overrides)
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

            yield f"data: {json.dumps(job)}\n\n"
            if job["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
