from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.services.operations import get_case, list_cases, update_case

router = APIRouter(prefix="/api/cases", tags=["cases"])

Status = Literal["new", "reviewing", "dispatched", "confirmed", "cleared"]


class CaseUpdate(BaseModel):
    status: Optional[Status] = None
    note: Optional[str] = Field(default=None, max_length=4000)


@router.get("")
def cases(
    status: Optional[Status] = None,
    tier: Optional[Literal["high", "medium"]] = None,
    search: Optional[str] = Query(default=None, min_length=1, max_length=64),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
) -> dict:
    return list_cases(status=status, tier=tier, search=search, page=page, page_size=page_size)


@router.get("/{customer_id}")
def case(customer_id: str) -> dict:
    try:
        return get_case(customer_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{customer_id}")
def patch_case(customer_id: str, payload: CaseUpdate) -> dict:
    try:
        return update_case(customer_id, status=payload.status, note=payload.note)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
