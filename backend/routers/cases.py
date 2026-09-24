from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from backend.schemas import CaseDetail, CaseList, CaseStatus, CaseUpdate
from backend.services.operations import get_case, list_cases, update_case

router = APIRouter(prefix="/api/cases", tags=["cases"])


@router.get("", response_model=CaseList)
def cases(
    status: Optional[CaseStatus] = None,
    tier: Optional[str] = Query(default=None, pattern="^(high|medium|low)$"),
    search: Optional[str] = Query(default=None, min_length=1, max_length=64),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
):
    return list_cases(status=status, tier=tier, search=search, page=page, page_size=page_size)


@router.get("/{customer_id}", response_model=CaseDetail)
def case(customer_id: str):
    return get_case(customer_id)


@router.patch("/{customer_id}", response_model=CaseDetail)
def patch_case(customer_id: str, payload: CaseUpdate):
    return update_case(customer_id, status=payload.status, note=payload.note)
