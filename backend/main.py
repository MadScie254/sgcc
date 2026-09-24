from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.dependencies.auth import load_api_key, require_api_key
from backend.routers import cases, customers, datasets, model, pipeline, predict, reports
from backend.schemas import Health
from backend.services.config import environment, scoring_interval_minutes
from backend.services.datasets import DatasetError, UploadTooLargeError
from backend.services.errors import NotFoundError
from backend.services.model import get_model_metrics, get_trained_model
from backend.services.operations import run_scoring_pipeline
from backend.services.reports import ReportsUnavailable, reports_status

logger = logging.getLogger(__name__)

IS_DEVELOPMENT = environment() == "development"


async def _scheduled_scoring(interval_minutes: float) -> None:
    while True:
        await asyncio.sleep(interval_minutes * 60)
        try:
            await asyncio.to_thread(run_scoring_pipeline, "schedule")
        except Exception:
            logger.exception("Scheduled scoring run failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.api_key = load_api_key()
    status = reports_status()
    if not status["available"]:
        logger.warning(status["detail"])
    try:
        # Warm every cache and record the run, so the pipeline view starts with real timings.
        await asyncio.to_thread(run_scoring_pipeline, "startup")
    except Exception:
        logger.exception("Startup scoring run failed")
    interval = scoring_interval_minutes()
    scheduler = asyncio.create_task(_scheduled_scoring(interval)) if interval > 0 else None
    yield
    if scheduler is not None:
        scheduler.cancel()


app = FastAPI(
    title="GridSentinel API",
    version="2.0.0",
    lifespan=lifespan,
    # Interactive docs expose the whole API surface; keep them to development.
    docs_url="/api/docs" if IS_DEVELOPMENT else None,
    redoc_url=None,
    openapi_url="/api/openapi.json" if IS_DEVELOPMENT else None,
)

allowed_origins = [o.strip() for o in os.getenv("FRONTEND_DEV_ORIGIN", "http://localhost:5173").split(",") if o.strip()]
if IS_DEVELOPMENT:
    allowed_origins.append("http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH"],
    allow_headers=["Content-Type", "X-API-Key"],
    expose_headers=["Content-Disposition"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


for module in (model, customers, cases, pipeline, predict, datasets, reports):
    app.include_router(module.router, dependencies=[Depends(require_api_key)])


@app.get("/api/health", response_model=Health)
def health():
    try:
        get_trained_model()
        payload = {"status": "ok", "model_loaded": True, "model_version": get_model_metrics()["model_version"]}
    except Exception:
        logger.exception("Health check could not load the model")
        payload = {"status": "degraded", "model_loaded": False, "model_version": "unknown"}
    payload["reports"] = reports_status()
    return payload if payload["model_loaded"] else JSONResponse(status_code=503, content=payload)


@app.exception_handler(NotFoundError)
async def not_found(request: Request, exc: NotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(UploadTooLargeError)
async def upload_too_large(request: Request, exc: UploadTooLargeError):
    return JSONResponse(status_code=413, content={"detail": str(exc)})


@app.exception_handler(DatasetError)
async def dataset_error(request: Request, exc: DatasetError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ReportsUnavailable)
async def reports_unavailable(request: Request, exc: ReportsUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    error_id = uuid.uuid4().hex[:12]
    logger.exception("Unhandled error %s on %s %s", error_id, request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal error (reference {error_id})"})


frontend_dist = (Path(__file__).resolve().parents[1] / "frontend" / "dist").resolve()


def resolve_frontend_file(path: str) -> Path | None:
    """Map a URL path to a file inside the frontend build, or None if it would escape it."""
    try:
        candidate = (frontend_dist / path.lstrip("/")).resolve()
    except (OSError, ValueError):
        return None
    if candidate.is_relative_to(frontend_dist) and candidate.is_file():
        return candidate
    return None


if (frontend_dist / "index.html").is_file():
    if (frontend_dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def serve_frontend(path: str):
        if path == "api" or path.startswith("api/"):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        file_path = resolve_frontend_file(path)
        if file_path is not None:
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")
