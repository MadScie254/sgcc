from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from backend.schemas.api import MonitorResponse
from backend.services.config import get_project_paths

router = APIRouter(prefix="/api/monitor", tags=["monitor"])


def _load_latest_monitoring_log() -> dict:
    monitoring_path = get_project_paths()["artifacts"] / "monitoring_log.json"
    if not monitoring_path.exists():
        return {
            "timestamp": None,
            "data_drift": {},
            "concept_drift": {},
            "alerts": [],
            "recommendations": [],
        }

    payload = json.loads(monitoring_path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload:
        return payload[-1]
    if isinstance(payload, dict):
        return payload
    return {
        "timestamp": None,
        "data_drift": {},
        "concept_drift": {},
        "alerts": [],
        "recommendations": [],
    }


@router.get("/drift", response_model=MonitorResponse)
def drift() -> MonitorResponse:
    return MonitorResponse(**_load_latest_monitoring_log())


@router.get("/alerts")
def alerts() -> dict:
    log = _load_latest_monitoring_log()
    return {
        "timestamp": log.get("timestamp"),
        "alerts": log.get("alerts", []),
        "recommendations": log.get("recommendations", []),
    }
