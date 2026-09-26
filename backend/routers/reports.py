from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from backend.dependencies.auth import User, current_user, require_supervisor
from backend.schemas import Report, ReportRequest
from backend.services import db
from backend.services.reports import delete_report, generate_report, list_reports, report_pdf

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("", response_model=List[Report])
def reports(limit: int = Query(default=50, ge=1, le=500)):
    return list_reports(limit)


@router.post("", response_model=Report, status_code=201)
async def create_report(request: ReportRequest, user: User = Depends(current_user)):
    subject_id = request.dataset_id if request.kind == "dataset" else request.customer_id
    entry = await run_in_threadpool(generate_report, request.kind, subject_id, user.name)
    db.audit(user.name, user.role, "report.create", entry["report_id"], entry["subject"])
    return entry


@router.get("/{report_id}/pdf")
def download(report_id: str, user: User = Depends(current_user)) -> Response:
    content = report_pdf(report_id)
    db.audit(user.name, user.role, "report.download", report_id)
    return Response(content=content, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{report_id}.pdf"'})


@router.delete("/{report_id}", status_code=204)
def remove(report_id: str, user: User = Depends(require_supervisor)) -> None:
    delete_report(report_id)
    db.audit(user.name, user.role, "report.delete", report_id)
