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
from backend.routers.analytics import router as analytics_router
from backend.routers.cases import router as cases_router
from backend.routers.compare import router as compare_router
from backend.routers.customers import router as customers_router
from backend.routers.datasets import router as datasets_router
from backend.routers.eda import router as eda_router
from backend.routers.explain import router as explain_router
from backend.routers.model import router as model_router
from backend.routers.monitor import router as monitor_router
from backend.routers.pipeline import router as pipeline_router
from backend.routers.predict import router as predict_router
from backend.routers.reports import router as reports_router
from backend.routers.train import router as train_router
from backend.services.model import get_model_config, get_trained_model
from backend.services.operations import run_scoring_pipeline

logger = logging.getLogger(__name__)

IS_DEVELOPMENT = os.getenv("ENV", "development").lower() == "development"


# Minutes between automatic scoring runs; 0 runs only at startup.
SCORING_INTERVAL_MINUTES = float(os.getenv("SCORING_INTERVAL_MINUTES", "0"))


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
    try:
        # Warm every cache and record the run, so the pipeline view starts with real timings.
        await asyncio.to_thread(run_scoring_pipeline, "startup")
    except Exception:
        logger.exception("Startup scoring run failed")
    scheduler = asyncio.create_task(_scheduled_scoring(SCORING_INTERVAL_MINUTES)) if SCORING_INTERVAL_MINUTES > 0 else None
    yield
    if scheduler is not None:
        scheduler.cancel()


app = FastAPI(
    title="SGCC Theft Detector API",
    version="0.2.0",
    lifespan=lifespan,
    # Interactive docs expose the whole API surface; keep them to development.
    docs_url="/api/docs" if IS_DEVELOPMENT else None,
    redoc_url=None,
    openapi_url="/api/openapi.json" if IS_DEVELOPMENT else None,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_DEV_ORIGIN", "http://localhost:5173").split(",")
    if origin.strip()
]
if IS_DEVELOPMENT:
    allowed_origins.append("http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


secured_router_kwargs = {"dependencies": [Depends(require_api_key)]}

for router in (
    eda_router, train_router, predict_router, customers_router, model_router, explain_router,
    compare_router, monitor_router, analytics_router, datasets_router, reports_router, pipeline_router, cases_router,
):
    app.include_router(router, **secured_router_kwargs)


@app.get("/api/health")
def health():
    try:
        model_loaded = get_trained_model() is not None
    except Exception:
        model_loaded = False

    try:
        model_version = get_model_config().get("model", {}).get("version", "unknown")
    except Exception:
        model_version = "unknown"

    payload = {
        "status": "ok" if model_loaded else "degraded",
        "service": "sgcc-backend",
        "model_loaded": model_loaded,
        "model_version": model_version,
    }
    return payload if model_loaded else JSONResponse(status_code=503, content=payload)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    error_id = str(uuid.uuid4())
    logger.exception("Unhandled error %s on %s %s", error_id, request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal error", "error_id": error_id})


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


if frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def serve_frontend(path: str):
        if path == "api" or path.startswith("api/"):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})

        file_path = resolve_frontend_file(path)
        if file_path is not None:
            return FileResponse(file_path)

        index_file = frontend_dist / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        return JSONResponse(status_code=404, content={"detail": "Frontend build not found"})
