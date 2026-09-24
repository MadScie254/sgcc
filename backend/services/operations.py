"""Operational workflow: scoring-pipeline runs and investigation cases.

State lives in small JSON files under the state directory (``artifacts/state/``
by default, not committed), so a restart keeps run history, case statuses and
analyst notes. Writes are atomic (temp file + rename) and serialised by a lock.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from . import data as data_service
from . import model as model_service
from .config import get_paths
from .errors import NotFoundError
from .storage import read_json, write_json

CASE_STATUSES = ("new", "reviewing", "dispatched", "confirmed", "cleared")
MAX_RUNS = 100
MAX_HISTORY = 50

_RUN_LOCK = Lock()
_STATE_LOCK = Lock()


class PipelineBusyError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read(name: str, default):
    return read_json(get_paths()["state"] / name, default)


def _write(name: str, payload) -> None:
    write_json(get_paths()["state"] / name, payload)


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

def list_runs(limit: int = 20) -> List[Dict[str, Any]]:
    with _STATE_LOCK:
        runs = _read("pipeline_runs.json", [])
    return list(reversed(runs))[:limit]


def run_scoring_pipeline(trigger: str) -> Dict[str, Any]:
    """Reload data and model, then ingest → features → score → explain → route, timing each stage."""
    if not _RUN_LOCK.acquire(blocking=False):
        raise PipelineBusyError("A scoring run is already in progress")
    try:
        record: Dict[str, Any] = {
            "run_id": uuid.uuid4().hex[:12], "trigger": trigger, "started_at": _now(),
            "status": "running", "stages": [],
        }
        model_service.clear_caches()
        started = time.perf_counter()

        def stage(key: str, name: str, work: Callable[[], str]) -> None:
            t0 = time.perf_counter()
            detail = work()
            record["stages"].append({"key": key, "name": name, "seconds": round(time.perf_counter() - t0, 3), "detail": detail})

        def ingest() -> str:
            wide, _ = data_service.get_wide_data()
            return f"{wide.shape[0]:,} customers × {wide.shape[1]:,} days"

        def features() -> str:
            X, _ = data_service.get_feature_matrix()
            return f"{X.shape[1]} features per customer"

        def score() -> str:
            probabilities = model_service.get_population_probabilities()
            threshold = model_service.get_decision_threshold()
            return f"{int((probabilities >= threshold).sum())} at or above threshold {threshold:.3f}"

        def explain() -> str:
            return f"SHAP drivers for {len(model_service.get_flagged_drivers())} flagged customers"

        def route() -> str:
            return f"{sync_cases()} new cases opened"

        try:
            stage("ingest", "Ingest reads", ingest)
            stage("features", "Build features", features)
            stage("score", "Score", score)
            stage("explain", "Explain", explain)
            stage("route", "Route cases", route)
            record["status"] = "succeeded"
            metrics = model_service.get_model_metrics()
            record["summary"] = {
                "customers": metrics["customers_monitored"],
                "flagged": metrics["flagged"],
                "tiers": metrics["risk_tier_distribution"],
                "threshold": metrics["threshold"],
                "model_version": metrics["model_version"],
            }
        except Exception as exc:  # recorded on the run, then re-raised
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["finished_at"] = _now()
            record["seconds"] = round(time.perf_counter() - started, 3)
            with _STATE_LOCK:
                runs = _read("pipeline_runs.json", [])
                runs.append(record)
                _write("pipeline_runs.json", runs[-MAX_RUNS:])
        return record
    finally:
        _RUN_LOCK.release()


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def sync_cases() -> int:
    """Open a case for every flagged customer that has none; returns how many were opened."""
    probabilities = model_service.get_population_probabilities()
    flagged = probabilities[probabilities >= model_service.get_decision_threshold()]
    opened = 0
    with _STATE_LOCK:
        cases = _read("cases.json", {})
        for customer_id, probability in flagged.items():
            if customer_id not in cases:
                cases[customer_id] = {
                    "status": "new", "note": "", "opened_at": _now(), "updated_at": _now(),
                    "history": [{"at": _now(), "event": f"Flagged by scoring run (p = {probability:.3f})"}],
                }
                opened += 1
        _write("cases.json", cases)
    return opened


def _row(customer_id: str, probability: float, rank: int, threshold: float,
         case: Optional[Dict[str, Any]], driver: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    case = case or {}
    return {
        "customer_id": customer_id,
        "rank": rank,
        "risk_score": float(probability),
        "risk_tier": model_service.risk_tier(float(probability), threshold),
        "flagged": bool(probability >= threshold),
        "status": case.get("status", "new"),
        "note": case.get("note", ""),
        "updated_at": case.get("updated_at"),
        "top_driver": driver,
    }


def list_cases(status: Optional[str], tier: Optional[str], search: Optional[str], page: int, page_size: int) -> Dict[str, Any]:
    """
    Cases for every flagged customer, plus any case already worked on (status past
    "new") even if a later threshold no longer flags it, so work in progress never
    disappears. Ordered by theft probability.
    """
    probabilities = model_service.get_population_probabilities()
    threshold = model_service.get_decision_threshold()
    drivers = model_service.get_flagged_drivers()
    with _STATE_LOCK:
        cases = _read("cases.json", {})

    rows = []
    for rank, (customer_id, probability) in enumerate(probabilities.items(), start=1):
        case = cases.get(customer_id)
        if probability >= threshold or (case and case.get("status") != "new"):
            rows.append(_row(customer_id, probability, rank, threshold, case, drivers.get(customer_id)))

    counts = {name: sum(1 for r in rows if r["status"] == name) for name in CASE_STATUSES}
    if status:
        rows = [r for r in rows if r["status"] == status]
    if tier:
        rows = [r for r in rows if r["risk_tier"] == tier]
    if search:
        rows = [r for r in rows if search.lower() in r["customer_id"].lower()]
    start = (page - 1) * page_size
    return {
        "items": rows[start:start + page_size],
        "total": len(rows),
        "page": page,
        "page_size": page_size,
        "status_counts": counts,
        "threshold": threshold,
    }


def get_case(customer_id: str) -> Dict[str, Any]:
    probabilities = model_service.get_population_probabilities()
    if customer_id not in probabilities.index:
        raise NotFoundError(f"Unknown customer_id: {customer_id}")
    with _STATE_LOCK:
        case = _read("cases.json", {}).get(customer_id)
    rank = int(probabilities.index.get_loc(customer_id)) + 1
    row = _row(customer_id, probabilities[customer_id], rank, model_service.get_decision_threshold(), case,
               model_service.get_flagged_drivers().get(customer_id))
    row["history"] = (case or {}).get("history", [])
    row["population"] = int(len(probabilities))
    return row


def update_case(customer_id: str, status: Optional[str] = None, note: Optional[str] = None) -> Dict[str, Any]:
    if customer_id not in model_service.get_population_probabilities().index:
        raise NotFoundError(f"Unknown customer_id: {customer_id}")
    if status is not None and status not in CASE_STATUSES:
        raise ValueError(f"Unknown status {status!r}")
    with _STATE_LOCK:
        cases = _read("cases.json", {})
        case = cases.setdefault(customer_id, {"status": "new", "note": "", "opened_at": _now(), "history": []})
        if status is not None and status != case["status"]:
            case["history"].append({"at": _now(), "event": f"Status: {case['status']} → {status}"})
            case["status"] = status
        if note is not None and note != case.get("note"):
            case["note"] = note
            case["history"].append({"at": _now(), "event": "Analyst note updated"})
        case["history"] = case["history"][-MAX_HISTORY:]
        case["updated_at"] = _now()
        _write("cases.json", cases)
    return get_case(customer_id)
