from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.api import FeatureImportanceItem
from backend.schemas.api import ModelConfigResponse, ModelMetricsResponse
from backend.services.model import get_feature_importance_from_csv, get_model_config, get_model_metrics

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
def feature_importance(limit: int = 15) -> list[FeatureImportanceItem]:
    try:
        return [FeatureImportanceItem(**item) for item in get_feature_importance_from_csv(limit=limit)]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc