"""The model in operations: counts, thresholds and drivers over the operational population (no labels)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.concurrency import run_in_threadpool

from backend.dependencies.auth import User, require_supervisor
from backend.schemas import GlobalDrivers, ModelMetrics, OperatingCurve, ScoreDistribution, ThresholdPreview, ThresholdUpdate
from backend.services import db, model
from backend.services.operations import PipelineBusyError, run_scoring_pipeline

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/metrics", response_model=ModelMetrics)
def metrics():
    return model.get_model_metrics()


@router.get("/operating-curve", response_model=OperatingCurve)
def operating_curve(
    capacity: Optional[int] = Query(default=None, ge=1, le=10_000_000, description="Inspections the team can make"),
    cost_per_visit: float = Query(default=0.0, ge=0.0, le=1e9),
    value_per_theft: float = Query(default=0.0, ge=0.0, le=1e12),
):
    return model.operating_curve(capacity, cost_per_visit, value_per_theft)


@router.get("/threshold-preview", response_model=ThresholdPreview)
def threshold_preview(threshold: float = Query(..., ge=0.0, le=1.0)):
    return model.threshold_preview(threshold)


@router.get("/score-distribution", response_model=ScoreDistribution)
def score_distribution():
    return model.score_distribution()


@router.get("/drivers", response_model=GlobalDrivers)
def drivers():
    return model.global_drivers()


@router.put("/threshold", response_model=ModelMetrics)
async def publish_threshold(payload: ThresholdUpdate, user: User = Depends(require_supervisor)):
    model.set_operating_threshold(payload.threshold, user.name)
    db.audit(user.name, user.role, "threshold.publish", detail="trained threshold" if payload.threshold is None else f"{payload.threshold:.4f}")
    try:
        # Re-score so cases open for newly flagged customers straight away.
        await run_in_threadpool(run_scoring_pipeline, "threshold", user.name)
    except PipelineBusyError:
        pass  # the run in progress reads the threshold from the database
    return model.get_model_metrics()
