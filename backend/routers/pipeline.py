from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from backend.services.operations import list_runs, run_scoring_pipeline, training_summary

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.get("/runs")
def runs(limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
    return list_runs(limit)


@router.post("/runs")
async def start_run() -> dict:
    try:
        return await run_in_threadpool(run_scoring_pipeline, "manual")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/training")
def training() -> dict:
    return training_summary()


@router.get("/config")
def config() -> dict:
    """Automation settings, as read from the environment at startup."""
    from backend.main import SCORING_INTERVAL_MINUTES
    from backend.routers.train import _training_enabled

    return {
        "run_on_startup": True,
        "scoring_interval_minutes": SCORING_INTERVAL_MINUTES,
        "training_api_enabled": _training_enabled(),
        "environment": os.getenv("ENV", "development").lower(),
    }
