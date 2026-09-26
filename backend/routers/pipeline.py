from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from backend.dependencies.auth import User, current_user
from backend.schemas import PipelineConfig, PipelineRun
from backend.services import db
from backend.services.config import environment, scoring_interval_minutes
from backend.services.operations import PipelineBusyError, list_runs, run_scoring_pipeline

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.get("/runs", response_model=List[PipelineRun])
def runs(limit: int = Query(default=20, ge=1, le=100)):
    return list_runs(limit)


@router.post("/runs", response_model=PipelineRun)
async def start_run(user: User = Depends(current_user)):
    db.audit(user.name, user.role, "pipeline.run")
    try:
        return await run_in_threadpool(run_scoring_pipeline, "manual", user.name)
    except PipelineBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/config", response_model=PipelineConfig)
def config():
    return {"run_on_startup": True, "scoring_interval_minutes": scoring_interval_minutes(), "environment": environment()}
