"""Operational workflow: scoring-pipeline runs and investigation cases.

State lives in small JSON files under ``artifacts/state/`` (not committed), so a
restart keeps run history, case statuses and analyst notes.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from . import data as data_service
from . import model as model_service
from .config import get_project_paths

CASE_STATUSES = ("new", "reviewing", "dispatched", "confirmed", "cleared")
MAX_RUNS = 100
MAX_HISTORY = 50

_RUN_LOCK = Lock()
_STATE_LOCK = Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _state_path(name: str) -> Path:
    return get_project_paths()["state"] / name


def _read(name: str, default):
    path = _state_path(name)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(name: str, payload) -> None:
    path = _state_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

def list_runs(limit: int = 20) -> List[Dict[str, Any]]:
    with _STATE_LOCK:
        runs = _read("pipeline_runs.json", [])
    return list(reversed(runs))[:limit]


def run_scoring_pipeline(trigger: str = "manual") -> Dict[str, Any]:
    """Reload data and model, then ingest -> features -> score -> explain -> route, timing each stage."""
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("A scoring run is already in progress")
    try:
        record: Dict[str, Any] = {
            "run_id": uuid.uuid4().hex[:12], "trigger": trigger, "started_at": _now(),
            "status": "running", "stages": [],
        }
        model_service.clear_caches()
        started = time.perf_counter()

        def stage(key: str, name: str, fn):
            t0 = time.perf_counter()
            detail = fn()
            record["stages"].append({"key": key, "name": name, "seconds": round(time.perf_counter() - t0, 3), "detail": detail})

        try:
            def ingest():
                wide, labels = data_service.get_wide_data()
                return f"{wide.shape[0]:,} customers × {wide.shape[1]:,} days"

            def features():
                X, _ = data_service.get_feature_matrix()
                return f"{X.shape[1]} features per customer"

            def score():
                model_service.get_trained_model()
                probabilities = model_service.get_population_probabilities()
                flagged = int((probabilities >= model_service.get_decision_threshold()).sum())
                return f"{flagged} at or above τ {model_service.get_decision_threshold():.3f}"

            def explain():
                drivers = model_service.get_flagged_drivers()
                return f"SHAP drivers for {len(drivers)} flagged customers"

            def route():
                opened = sync_cases()
                return f"{opened} new cases opened"

            stage("ingest", "Ingest reads", ingest)
            stage("features", "Build features", features)
            stage("score", "Score", score)
            stage("explain", "Explain", explain)
            stage("route", "Route cases", route)
            record["status"] = "succeeded"
        except Exception as exc:  # recorded on the run, then re-raised
            record["status"] = "failed"
            record["error"] = str(exc)
            raise
        finally:
            record["finished_at"] = _now()
            record["seconds"] = round(time.perf_counter() - started, 3)
            if record["status"] == "succeeded":
                metrics = model_service.get_model_metrics()
                record["summary"] = {
                    "customers": metrics["customers_monitored"],
                    "flagged": metrics["flagged_today"],
                    "tiers": metrics["risk_tier_distribution"],
                    "threshold": metrics["threshold"],
                    "model_version": metrics["model_version"],
                }
            with _STATE_LOCK:
                runs = _read("pipeline_runs.json", [])
                runs.append(record)
                _write("pipeline_runs.json", runs[-MAX_RUNS:])
        return record
    finally:
        _RUN_LOCK.release()


def training_summary() -> Dict[str, Any]:
    metrics = model_service.get_saved_metrics()
    params_path = get_project_paths()["artifacts"] / "best_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.is_file() else {}
    keys = ("model_version", "trained_at", "quick_mode", "n_trials", "cv_metric", "cv_best_score",
            "train_customers", "test_customers", "n_features", "auc", "pr_auc", "precision", "recall", "f1", "threshold")
    return {**{key: metrics.get(key) for key in keys}, "stages": metrics.get("stages", []), "best_params": params}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def _load_cases() -> Dict[str, Dict[str, Any]]:
    return _read("cases.json", {})


def sync_cases() -> int:
    """Open a case for every flagged customer that has none; returns how many were opened."""
    probabilities = model_service.get_population_probabilities()
    threshold = model_service.get_decision_threshold()
    flagged = probabilities[probabilities >= threshold]
    opened = 0
    with _STATE_LOCK:
        cases = _load_cases()
        for customer_id, probability in flagged.items():
            if customer_id not in cases:
                cases[customer_id] = {
                    "status": "new", "note": "", "opened_at": _now(), "updated_at": _now(),
                    "history": [{"at": _now(), "event": f"Flagged by scoring run (p = {probability:.3f})"}],
                }
                opened += 1
        _write("cases.json", cases)
    return opened


def _case_row(customer_id: str, probability: float, threshold: float, case: Optional[Dict[str, Any]],
              driver: Optional[Dict[str, Any]], rank: int) -> Dict[str, Any]:
    return {
        "customer_id": customer_id,
        "rank": rank,
        "risk_score": float(probability),
        "risk_tier": model_service.risk_tier_for_probability(float(probability), threshold),
        "status": (case or {}).get("status", "new"),
        "note": (case or {}).get("note", ""),
        "updated_at": (case or {}).get("updated_at"),
        "top_driver": driver,
    }


def list_cases(status: Optional[str] = None, tier: Optional[str] = None, search: Optional[str] = None,
               page: int = 1, page_size: int = 25) -> Dict[str, Any]:
    probabilities = model_service.get_population_probabilities().sort_values(ascending=False)
    threshold = model_service.get_decision_threshold()
    flagged = probabilities[probabilities >= threshold]
    drivers = model_service.get_flagged_drivers()
    with _STATE_LOCK:
        cases = _load_cases()

    rows = [
        _case_row(cid, p, threshold, cases.get(cid), drivers.get(cid), rank)
        for rank, (cid, p) in enumerate(flagged.items(), start=1)
    ]
    counts = {name: 0 for name in CASE_STATUSES}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
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
        raise KeyError(f"Unknown customer_id: {customer_id}")
    threshold = model_service.get_decision_threshold()
    ranked = probabilities.rank(ascending=False, method="first")
    with _STATE_LOCK:
        case = _load_cases().get(customer_id)
    row = _case_row(customer_id, probabilities[customer_id], threshold, case,
                    model_service.get_flagged_drivers().get(customer_id), int(ranked[customer_id]))
    row["flagged"] = bool(probabilities[customer_id] >= threshold)
    row["history"] = (case or {}).get("history", [])
    row["population"] = int(len(probabilities))
    return row


def update_case(customer_id: str, status: Optional[str] = None, note: Optional[str] = None) -> Dict[str, Any]:
    if customer_id not in model_service.get_population_probabilities().index:
        raise KeyError(f"Unknown customer_id: {customer_id}")
    if status is not None and status not in CASE_STATUSES:
        raise ValueError(f"Unknown status {status!r}")
    with _STATE_LOCK:
        cases = _load_cases()
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
