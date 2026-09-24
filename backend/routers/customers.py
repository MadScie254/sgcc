from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from backend.schemas import CustomerList, Explanation, TimeSeries
from backend.services import model
from backend.services.data import get_customer_timeseries

router = APIRouter(prefix="/api/customers", tags=["customers"])


@router.get("", response_model=CustomerList)
def list_customers(
    search: Optional[str] = Query(default=None, min_length=1, max_length=64),
    tier: Optional[str] = Query(default=None, pattern="^(high|medium|low)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
):
    return model.list_customers(search, tier, page, page_size)


@router.get("/{customer_id}/timeseries", response_model=TimeSeries)
def timeseries(customer_id: str):
    return get_customer_timeseries(customer_id)


@router.get("/{customer_id}/explanation", response_model=Explanation)
def explanation(customer_id: str):
    return model.explain_customer(customer_id)
