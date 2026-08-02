from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from backend.schemas.api import CompareResponse
from backend.services.config import get_project_paths

router = APIRouter(prefix="/api/compare", tags=["compare"])


@router.get("/baselines", response_model=CompareResponse)
def baselines() -> CompareResponse:
    artifacts = get_project_paths()["artifacts"]
    xgb_path = artifacts / "metrics.json"
    baseline_path = get_project_paths()["models"] / "baselines" / "comparison_results.json"

    xgboost_metrics = {}
    baselines_metrics = {}

    if xgb_path.exists():
        xgboost_metrics = json.loads(xgb_path.read_text(encoding="utf-8"))
    if baseline_path.exists():
        baselines_metrics = json.loads(baseline_path.read_text(encoding="utf-8"))

    return CompareResponse(xgboost=xgboost_metrics, baselines=baselines_metrics)
