"""Operational workflow: scoring-pipeline runs and investigation cases.

State lives in the database (``backend.services.db``). A scoring run holds the
scoring lock, so only one instance scores at a time.

Cases follow a fixed workflow; every change records who made it:

    new -> reviewing -> dispatched -> confirmed | cleared
               \\-> cleared                  (nothing to inspect)
    confirmed | cleared -> reviewing        (reopen)

Analysts review and dispatch. Resolving (confirmed or cleared) and reopening need a
supervisor, and resolving needs a reason and a reference to the evidence (an
inspection report number, a photo id, ...): a flag is a reason to inspect, not a finding.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import delete, insert, select, update

from . import db
from . import model as model_service
from .data import get_feature_matrix, get_wide_data
from .errors import NotFoundError

CASE_STATUSES = ("new", "reviewing", "dispatched", "confirmed", "cleared")
RESOLVED = ("confirmed", "cleared")
# from -> {to: role needed}
TRANSITIONS: Dict[str, Dict[str, str]] = {
    "new": {"reviewing": "analyst"},
    "reviewing": {"dispatched": "analyst", "cleared": "supervisor"},
    "dispatched": {"confirmed": "supervisor", "cleared": "supervisor"},
    "confirmed": {"reviewing": "supervisor"},
    "cleared": {"reviewing": "supervisor"},
}
MAX_RUNS = 200


class PipelineBusyError(RuntimeError):
    pass


class CaseTransitionError(ValueError):
    """The requested status change is not a step of the workflow."""


class CasePermissionError(PermissionError):
    """The step needs the supervisor role."""


def allowed_transitions(status: str, role: str) -> List[str]:
    return [to for to, needed in TRANSITIONS.get(status, {}).items() if needed == "analyst" or role == "supervisor"]


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

def list_runs(limit: int = 20) -> List[Dict[str, Any]]:
    with db.transaction() as connection:
        rows = connection.execute(select(db.pipeline_runs.c.record)
                                  .order_by(db.pipeline_runs.c.started_at.desc()).limit(limit)).scalars().all()
    return list(rows)


def run_scoring_pipeline(trigger: str, actor: str = "system") -> Dict[str, Any]:
    """Reload data and model, then ingest → features → score → explain → route, timing each stage."""
    with db.scoring_lock() as held:
        if not held:
            raise PipelineBusyError("A scoring run is already in progress")
        record: Dict[str, Any] = {
            "run_id": uuid.uuid4().hex[:12], "trigger": trigger, "actor": actor, "started_at": db.now(),
            "status": "running", "stages": [],
        }
        model_service.clear_caches()
        started = time.perf_counter()

        def stage(key: str, name: str, work: Callable[[], str]) -> None:
            t0 = time.perf_counter()
            detail = work()
            record["stages"].append({"key": key, "name": name, "seconds": round(time.perf_counter() - t0, 3), "detail": detail})

        def ingest() -> str:
            wide = get_wide_data()
            return f"{wide.shape[0]:,} customers × {wide.shape[1]:,} days"

        def features() -> str:
            return f"{get_feature_matrix().shape[1]} features per customer"

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
                "expected_thefts_flagged": metrics["expected_thefts_flagged"],
                "tiers": metrics["risk_tier_distribution"],
                "threshold": metrics["threshold"],
                "model_version": metrics["model_version"],
                "population": metrics["population"]["id"],
            }
        except Exception as exc:  # recorded on the run, then re-raised
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["finished_at"] = db.now()
            record["seconds"] = round(time.perf_counter() - started, 3)
            with db.transaction() as connection:
                connection.execute(insert(db.pipeline_runs).values(run_id=record["run_id"], started_at=record["started_at"], record=record))
                keep = select(db.pipeline_runs.c.run_id).order_by(db.pipeline_runs.c.started_at.desc()).limit(MAX_RUNS)
                connection.execute(delete(db.pipeline_runs).where(db.pipeline_runs.c.run_id.not_in(keep.scalar_subquery())))
        return record


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def _event(connection, customer_id: str, actor: str, event: str) -> None:
    connection.execute(insert(db.case_events).values(customer_id=customer_id, at=db.now(), actor=actor, event=event))


def sync_cases() -> int:
    """Open a case for every flagged customer that has none; returns how many were opened."""
    probabilities = model_service.get_population_probabilities()
    flagged = probabilities[probabilities >= model_service.get_decision_threshold()]
    with db.transaction() as connection:
        existing = set(connection.execute(select(db.cases.c.customer_id)).scalars())
        new = [(cid, float(p)) for cid, p in flagged.items() if cid not in existing]
        if new:
            at = db.now()
            connection.execute(insert(db.cases), [
                {"customer_id": cid, "status": "new", "note": "", "opened_at": at, "updated_at": at,
                 "updated_by": "scoring", "opened_probability": p} for cid, p in new])
            connection.execute(insert(db.case_events), [
                {"customer_id": cid, "at": at, "actor": "scoring", "event": f"Flagged by scoring run (p = {p:.3f})"}
                for cid, p in new])
    return len(new)


def _all_cases() -> Dict[str, Dict[str, Any]]:
    with db.transaction() as connection:
        return {row["customer_id"]: dict(row) for row in connection.execute(select(db.cases)).mappings()}


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
        "updated_by": case.get("updated_by"),
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
    cases = _all_cases()

    rows = []
    for rank, (customer_id, probability) in enumerate(probabilities.items(), start=1):
        case = cases.get(customer_id)
        if probability >= threshold or (case and case["status"] != "new"):
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


def get_case(customer_id: str, role: str = "analyst") -> Dict[str, Any]:
    probabilities = model_service.get_population_probabilities()
    if customer_id not in probabilities.index:
        raise NotFoundError(f"Unknown customer_id: {customer_id}")
    with db.transaction() as connection:
        case = connection.execute(select(db.cases).where(db.cases.c.customer_id == customer_id)).mappings().first()
        history = connection.execute(select(db.case_events.c.at, db.case_events.c.actor, db.case_events.c.event)
                                     .where(db.case_events.c.customer_id == customer_id)
                                     .order_by(db.case_events.c.id)).mappings().all()
    case = dict(case) if case else None
    rank = int(probabilities.index.get_loc(customer_id)) + 1
    row = _row(customer_id, probabilities[customer_id], rank, model_service.get_decision_threshold(), case,
               model_service.get_flagged_drivers().get(customer_id))
    row["history"] = [dict(h) for h in history]
    row["population"] = int(len(probabilities))
    row["allowed_transitions"] = allowed_transitions(row["status"], role)
    row["resolution"] = None if not case or not case["resolved_at"] else {
        "outcome": case["status"], "reason": case["resolution_reason"], "evidence": case["resolution_evidence"],
        "by": case["resolved_by"], "at": case["resolved_at"],
    }
    return row


def update_case(customer_id: str, actor: str, role: str, status: Optional[str] = None, note: Optional[str] = None,
                reason: Optional[str] = None, evidence: Optional[str] = None) -> Dict[str, Any]:
    probabilities = model_service.get_population_probabilities()
    if customer_id not in probabilities.index:
        raise NotFoundError(f"Unknown customer_id: {customer_id}")
    reason, evidence = (reason or "").strip(), (evidence or "").strip()
    with db.transaction() as connection:
        case = connection.execute(select(db.cases).where(db.cases.c.customer_id == customer_id)).mappings().first()
        if case is None:
            at = db.now()
            connection.execute(insert(db.cases).values(customer_id=customer_id, status="new", note="", opened_at=at,
                                                       updated_at=at, updated_by=actor,
                                                       opened_probability=float(probabilities[customer_id])))
            _event(connection, customer_id, actor, "Case opened by hand")
            current = "new"
        else:
            current = case["status"]
        changes: Dict[str, Any] = {}
        if status is not None and status != current:
            needed = TRANSITIONS.get(current, {}).get(status)
            if needed is None:
                allowed = ", ".join(TRANSITIONS.get(current, {})) or "none"
                raise CaseTransitionError(f"A case cannot move from {current} to {status} (next steps: {allowed})")
            if needed == "supervisor" and role != "supervisor":
                raise CasePermissionError(f"Moving a case from {current} to {status} needs the supervisor role")
            if status in RESOLVED and (not reason or not evidence):
                raise CaseTransitionError(f"Marking a case {status} needs a reason and an evidence reference")
            if current in RESOLVED and not reason:
                raise CaseTransitionError("Reopening a case needs a reason")
            changes["status"] = status
            if status in RESOLVED:
                changes.update(resolution_reason=reason, resolution_evidence=evidence, resolved_by=actor, resolved_at=db.now())
            elif current in RESOLVED:
                changes.update(resolution_reason=None, resolution_evidence=None, resolved_by=None, resolved_at=None)
            detail = f" ({reason}; evidence: {evidence})" if status in RESOLVED else (f" ({reason})" if reason else "")
            _event(connection, customer_id, actor, f"Status: {current} → {status}{detail}")
        if note is not None and note != (case["note"] if case else ""):
            changes["note"] = note
            _event(connection, customer_id, actor, "Analyst note updated")
        if changes:
            connection.execute(update(db.cases).where(db.cases.c.customer_id == customer_id)
                               .values(**changes, updated_at=db.now(), updated_by=actor))
    return get_case(customer_id, role)
