from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.schemas.api import CustomerTimeseriesResponse, CustomersResponse, LocalShapResponse
from backend.services.data import get_customer_timeseries
from backend.services.model import get_customer_table, get_local_shap_details

router = APIRouter(prefix="/api/customers", tags=["customers"])


@router.get("", response_model=CustomersResponse)
def list_customers(
    search: str | None = Query(default=None, min_length=1),
    risk_tier: str | None = Query(default=None),
    sort_by: str = Query(default="risk_score"),
    sort_dir: str = Query(default="desc", pattern="^(asc|desc)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> CustomersResponse:
    return CustomersResponse(**get_customer_table(search, risk_tier, sort_by, sort_dir, page, page_size))


@router.get("/{customer_id}/shap", response_model=LocalShapResponse)
def customer_shap(customer_id: str) -> LocalShapResponse:
    try:
        return LocalShapResponse(**get_local_shap_details(customer_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{customer_id}/timeseries", response_model=CustomerTimeseriesResponse)
def customer_timeseries(customer_id: str) -> CustomerTimeseriesResponse:
    try:
        return CustomerTimeseriesResponse(**get_customer_timeseries(customer_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc