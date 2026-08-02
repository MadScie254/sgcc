from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.api import AnalyticsDashboardResponse
from backend.services.data import get_dataset_summary
from backend.services.model import get_model_metrics
from backend.services.public_apis import get_country_context, get_public_holidays, get_weather_context
from backend.services.reporting import load_report_index, load_upload_catalog

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/dashboard", response_model=AnalyticsDashboardResponse)
def dashboard() -> AnalyticsDashboardResponse:
    return AnalyticsDashboardResponse(
        dataset_summary=get_dataset_summary(),
        model_metrics=get_model_metrics(),
        uploads=load_upload_catalog()[:8],
        reports=load_report_index()[:8],
        context={
            "country": get_country_context(),
            "holidays": get_public_holidays()[:8],
            "weather": get_weather_context(),
        },
    )