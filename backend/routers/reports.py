from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.api import ReportRequest, ReportResponse
from backend.services.reporting import build_report_file_response, generate_report, load_report_index

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.post("/generate", response_model=ReportResponse)
def create_report(request: ReportRequest) -> ReportResponse:
    try:
        return ReportResponse(**generate_report(request.dataset_id, request.country_code, request.latitude, request.longitude))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/latest", response_model=list[dict])
def latest_reports() -> list[dict]:
    return load_report_index()[:10]


@router.get("/{report_id}/download")
def download_report(report_id: str):
    try:
        return build_report_file_response(report_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc