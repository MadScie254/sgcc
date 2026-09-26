"""Who is calling, what they did, and which readings are scored."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool

from backend.dependencies.auth import User, current_user, require_supervisor
from backend.schemas import AuditEntry, Me, Population, PopulationRequest
from backend.services import db
from backend.services.data import active_population
from backend.services.datasets import promote_dataset, reset_population
from backend.services.operations import PipelineBusyError, run_scoring_pipeline

router = APIRouter(prefix="/api", tags=["admin"])


@router.get("/me", response_model=Me)
def me(request: Request, user: User = Depends(current_user)):
    return {"name": user.name, "role": user.role, "auth_required": request.app.state.api_keys is not None}


@router.get("/audit", response_model=List[AuditEntry])
def audit(limit: int = Query(default=100, ge=1, le=1000), _: User = Depends(require_supervisor)):
    return db.list_audit(limit)


@router.get("/population", response_model=Population)
def population():
    return active_population()


async def _rescore(user: User) -> None:
    try:
        await run_in_threadpool(run_scoring_pipeline, "population", user.name)
    except PipelineBusyError:
        pass  # the run in progress reads the population from the database


@router.post("/population", response_model=Population)
async def promote(payload: PopulationRequest, user: User = Depends(require_supervisor)):
    """Score an uploaded meter-data file from now on, instead of the training sample."""
    result = promote_dataset(payload.dataset_id, user.name)
    db.audit(user.name, user.role, "population.promote", payload.dataset_id)
    await _rescore(user)
    return result


@router.delete("/population", response_model=Population)
async def reset(user: User = Depends(require_supervisor)):
    result = reset_population(user.name)
    db.audit(user.name, user.role, "population.reset")
    await _rescore(user)
    return result
