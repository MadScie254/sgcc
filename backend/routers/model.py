from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.schemas.api import FeatureImportanceItem
from backend.schemas.api import ModelConfigResponse, ModelMetricsResponse
from backend.services.model import (
    get_feature_importance_from_csv,
    get_model_config,
    get_model_metrics,
    operating_curve,
    score_distribution,
    set_operating_threshold,
)
from backend.services.operations import run_scoring_pipeline

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/metrics", response_model=ModelMetricsResponse)
def metrics() -> ModelMetricsResponse:
    return ModelMetricsResponse(**get_model_metrics())


@router.get("/config", response_model=ModelConfigResponse)
def config() -> ModelConfigResponse:
    try:
        return ModelConfigResponse(**get_model_config())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/feature-importance", response_model=list[FeatureImportanceItem])
def feature_importance(limit: int = Query(default=15, ge=1, le=100)) -> list[FeatureImportanceItem]:
    try:
        return [FeatureImportanceItem(**item) for item in get_feature_importance_from_csv(limit=limit)]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

class ThresholdUpdate(BaseModel):
    # None returns to the threshold chosen during training.
    threshold: float | None = Field(default=None, gt=0.0, lt=1.0)


@router.get("/operating-curve")
def get_operating_curve() -> list[dict]:
    return operating_curve()


@router.get("/score-distribution")
def get_score_distribution() -> dict:
    return score_distribution()


@router.put("/threshold", response_model=ModelMetricsResponse)
def put_threshold(payload: ThresholdUpdate) -> ModelMetricsResponse:
    set_operating_threshold(payload.threshold)
    try:
        # Re-score so cases open for newly flagged customers straight away.
        run_scoring_pipeline("threshold")
    except RuntimeError:
        pass  # a run already in progress picks the new threshold up
    return ModelMetricsResponse(**get_model_metrics())
