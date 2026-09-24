from __future__ import annotations

from typing import List

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from backend.schemas import ComparisonRow, GlobalDrivers, ModelMetrics, OperatingPoint, ScoreDistribution, ThresholdUpdate, TrainingSummary
from backend.services import model
from backend.services.operations import PipelineBusyError, run_scoring_pipeline

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/metrics", response_model=ModelMetrics)
def metrics():
    return model.get_model_metrics()


@router.get("/operating-curve", response_model=List[OperatingPoint])
def operating_curve():
    return model.operating_curve()


@router.get("/score-distribution", response_model=ScoreDistribution)
def score_distribution():
    return model.score_distribution()


@router.get("/comparison", response_model=List[ComparisonRow])
def comparison():
    return model.model_comparison()


@router.get("/drivers", response_model=GlobalDrivers)
def drivers():
    return model.global_drivers()


@router.get("/training", response_model=TrainingSummary)
def training():
    return model.training_summary()


@router.put("/threshold", response_model=ModelMetrics)
async def publish_threshold(payload: ThresholdUpdate):
    model.set_operating_threshold(payload.threshold)
    try:
        # Re-score so cases open for newly flagged customers straight away.
        await run_in_threadpool(run_scoring_pipeline, "threshold")
    except PipelineBusyError:
        pass  # the run in progress picks up the new threshold when it reads it
    return model.get_model_metrics()
