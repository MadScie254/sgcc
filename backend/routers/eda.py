from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.schemas.api import (
    CorrelationMatrixResponse,
    CustomerTimeseriesResponse,
    DatasetSummary,
    FeatureDistributionResponse,
)
from backend.services.data import (
    get_correlation_matrix,
    get_customer_timeseries,
    get_dataset_summary,
    get_feature_distribution,
)

router = APIRouter(prefix="/api/eda", tags=["eda"])


@router.get("/summary", response_model=DatasetSummary)
def summary() -> DatasetSummary:
    return DatasetSummary(**get_dataset_summary())


@router.get("/feature-distributions", response_model=FeatureDistributionResponse)
def feature_distributions(feature: str = Query(..., min_length=1)) -> FeatureDistributionResponse:
    try:
        return FeatureDistributionResponse(**get_feature_distribution(feature))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/correlation-matrix", response_model=CorrelationMatrixResponse)
def correlation_matrix() -> CorrelationMatrixResponse:
    return CorrelationMatrixResponse(**get_correlation_matrix())


@router.get("/customers/{customer_id}/timeseries", response_model=CustomerTimeseriesResponse)
def customer_timeseries(customer_id: str) -> CustomerTimeseriesResponse:
    try:
        return CustomerTimeseriesResponse(**get_customer_timeseries(customer_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
