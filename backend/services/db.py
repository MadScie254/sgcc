"""Operational state in a relational database (SQLAlchemy Core).

``DATABASE_URL`` selects PostgreSQL (``postgres://``, ``postgresql://`` or
``postgresql+psycopg://``; the psycopg 3 driver is used). Without it, state goes
to SQLite in the state directory, which suits development and tests but not
several instances.

Tables: cases and their events, scoring runs, settings (operating threshold and
population), uploaded datasets, generated reports, and the audit log. The schema
is created on startup. Files (uploads, PDFs) live in the blob store, not here.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional, Tuple

from sqlalchemy import (
    JSON, Column, Engine, Float, Integer, MetaData, String, Table, Text, create_engine, delete, event, insert, select,
    text,
)

from .config import get_paths

metadata = MetaData()

cases = Table(
    "cases", metadata,
    Column("customer_id", String(64), primary_key=True),
    Column("status", String(16), nullable=False),
    Column("note", Text, nullable=False, default=""),
    Column("opened_at", String(32), nullable=False),
    Column("updated_at", String(32), nullable=False),
    Column("updated_by", String(64)),
    Column("opened_probability", Float),
    Column("resolution_reason", Text),
    Column("resolution_evidence", Text),
    Column("resolved_by", String(64)),
    Column("resolved_at", String(32)),
)

case_events = Table(
    "case_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("customer_id", String(64), nullable=False, index=True),
    Column("at", String(32), nullable=False),
    Column("actor", String(64), nullable=False),
    Column("event", Text, nullable=False),
)

pipeline_runs = Table(
    "pipeline_runs", metadata,
    Column("run_id", String(32), primary_key=True),
    Column("started_at", String(32), nullable=False, index=True),
    Column("record", JSON, nullable=False),
)

settings = Table(
    "settings", metadata,
    Column("key", String(64), primary_key=True),
    Column("value", JSON, nullable=False),
    Column("updated_at", String(32), nullable=False),
    Column("updated_by", String(64), nullable=False),
)

datasets = Table(
    "datasets", metadata,
    Column("dataset_id", String(40), primary_key=True),
    Column("filename", String(200), nullable=False),
    Column("uploaded_at", String(32), nullable=False, index=True),
    Column("uploaded_by", String(64), nullable=False),
    Column("blob_key", String(200), nullable=False),
    Column("summary", JSON, nullable=False),
)

reports = Table(
    "reports", metadata,
    Column("report_id", String(48), primary_key=True),
    Column("kind", String(16), nullable=False),
    Column("title", String(200), nullable=False),
    Column("subject", String(200), nullable=False),
    Column("created_at", String(32), nullable=False, index=True),
    Column("created_by", String(64), nullable=False),
    Column("bytes", Integer, nullable=False),
    Column("blob_key", String(200), nullable=False),
)

audit_log = Table(
    "audit_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("at", String(32), nullable=False, index=True),
    Column("actor", String(64), nullable=False),
    Column("role", String(16), nullable=False),
    Column("action", String(48), nullable=False),
    Column("target", String(200)),
    Column("detail", Text),
)


def now() -> str:
    """UTC timestamp in ISO format; milliseconds keep "newest first" well defined."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        state = get_paths()["state"]
        state.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{state / 'gridsentinel.db'}"
    # Hosted Postgres hands out postgres:// URLs; SQLAlchemy needs the dialect and the driver.
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def is_postgres() -> bool:
    return get_engine().dialect.name == "postgresql"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = database_url()
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(connection, _record):  # concurrent readers while one thread writes
            cursor = connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()
        return engine
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def init_db() -> None:
    metadata.create_all(get_engine())


@contextmanager
def transaction() -> Iterator[Any]:
    with get_engine().begin() as connection:
        yield connection


# ---------------------------------------------------------------------------
# Settings, read through a short cache so every instance sees a change within seconds
# ---------------------------------------------------------------------------

SETTINGS_TTL_SECONDS = 5.0
_SETTINGS_CACHE: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}


def get_setting(key: str) -> Optional[Dict[str, Any]]:
    """The setting's row (value, updated_at, updated_by), or None when it was never set."""
    cached = _SETTINGS_CACHE.get(key)
    if cached and time.monotonic() - cached[0] < SETTINGS_TTL_SECONDS:
        return cached[1]
    with transaction() as connection:
        row = connection.execute(select(settings).where(settings.c.key == key)).mappings().first()
    value = dict(row) if row else None
    _SETTINGS_CACHE[key] = (time.monotonic(), value)
    return value


def set_setting(key: str, value: Optional[Dict[str, Any]], actor: str) -> None:
    """Store a setting, or delete it with ``value=None``."""
    with transaction() as connection:
        connection.execute(delete(settings).where(settings.c.key == key))
        if value is not None:
            connection.execute(insert(settings).values(key=key, value=value, updated_at=now(), updated_by=actor))
    _SETTINGS_CACHE.pop(key, None)


# ---------------------------------------------------------------------------
# One scoring run at a time, across every instance
# ---------------------------------------------------------------------------

_SCORING_KEY = 0x5C0_12E  # arbitrary, fixed advisory-lock id
_LOCAL_LOCK = threading.Lock()


@contextmanager
def scoring_lock() -> Iterator[bool]:
    """
    Yields True when this caller holds the scoring lock, False when a run is already in
    progress. PostgreSQL uses a session advisory lock, so it holds across instances; SQLite
    (one instance) uses a process lock.
    """
    if not _LOCAL_LOCK.acquire(blocking=False):
        yield False
        return
    try:
        if not is_postgres():
            yield True
            return
        with get_engine().connect() as connection:
            held = bool(connection.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": _SCORING_KEY}).scalar())
            try:
                yield held
            finally:
                if held:
                    connection.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _SCORING_KEY})
                connection.commit()
    finally:
        _LOCAL_LOCK.release()


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def audit(actor: str, role: str, action: str, target: Optional[str] = None, detail: Optional[str] = None) -> None:
    with transaction() as connection:
        connection.execute(insert(audit_log).values(at=now(), actor=actor, role=role, action=action,
                                                    target=None if target is None else str(target)[:200],
                                                    detail=detail))


def list_audit(limit: int = 100) -> list:
    with transaction() as connection:
        rows = connection.execute(select(audit_log).order_by(audit_log.c.id.desc()).limit(limit)).mappings().all()
    return [dict(row) for row in rows]

