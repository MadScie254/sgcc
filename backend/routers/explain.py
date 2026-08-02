from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.schemas.api import FeatureImportanceItem, GlobalShapResponse, LocalShapResponse
from backend.services.model import get_customer_feature_row, get_feature_importance, get_global_shap_sample, get_local_shap_details

router = APIRouter(prefix="/api/explain", tags=["explain"])


@router.get("/feature-importance", response_model=list[FeatureImportanceItem])
def feature_importance(limit: int = Query(default=20, ge=1, le=100)) -> list[FeatureImportanceItem]:
    return [FeatureImportanceItem(**item) for item in get_feature_importance(limit=limit)]


@router.get("/global-shap", response_model=GlobalShapResponse)
def global_shap(sample_count: int = Query(default=200, ge=10, le=1000)) -> GlobalShapResponse:
    return GlobalShapResponse(**get_global_shap_sample(sample_count=sample_count))


@router.get("/local-shap/{customer_id}", response_model=LocalShapResponse)
def local_shap(customer_id: str) -> LocalShapResponse:
    try:
        return LocalShapResponse(**get_local_shap_details(customer_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc