from __future__ import annotations

from typing import List

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from backend.schemas import Report, ReportRequest
from backend.services.reports import generate_report, list_reports, report_path

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("", response_model=List[Report])
def reports(limit: int = Query(default=50, ge=1, le=500)):
    return list_reports(limit)


@router.post("", response_model=Report, status_code=201)
async def create_report(request: ReportRequest):
    subject_id = request.dataset_id if request.kind == "dataset" else request.customer_id
    return await run_in_threadpool(generate_report, request.kind, subject_id)


@router.get("/{report_id}/pdf")
def download(report_id: str) -> FileResponse:
    return FileResponse(report_path(report_id), media_type="application/pdf", filename=f"{report_id}.pdf")
