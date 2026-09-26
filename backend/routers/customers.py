from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool

from backend.schemas import CustomerList, Explanation, ExplanationCheck, TimeSeries
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


@router.get("/{customer_id}/explanation-check", response_model=ExplanationCheck)
async def explanation_check(customer_id: str):
    """LIME's view of the same prediction: whether the explanation is consistent, not whether it is causal."""
    return await run_in_threadpool(model.explanation_check, customer_id)
