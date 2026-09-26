from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.dependencies.auth import current_user, load_api_keys, make_rate_limiter
from backend.routers import admin, cases, customers, datasets, model, pipeline, predict, reports, research
from backend.schemas import Health
from backend.services import db
from backend.services.blobstore import get_blobstore
from backend.services.config import environment, scoring_interval_minutes
from backend.services.data import ModelUnavailable, get_pipeline_spec
from backend.services.datasets import DatasetError, DatasetInUseError, UploadTooLargeError, purge_expired_datasets
from backend.services.errors import NotFoundError
from backend.services.model import FeatureInputError, get_trained_model, integrity_problems
from backend.services.operations import CasePermissionError, CaseTransitionError, PipelineBusyError, run_scoring_pipeline
from backend.services.reports import ReportsUnavailable, purge_expired_reports, reports_status

logger = logging.getLogger(__name__)

IS_DEVELOPMENT = environment() == "development"
MAINTENANCE_HOURS = 6


def _purge() -> None:
    removed = purge_expired_datasets() + purge_expired_reports()
    if removed:
        logger.info("Retention: deleted %d expired uploads and reports", removed)


async def _every(minutes: float, work, name: str) -> None:
    while True:
        await asyncio.sleep(minutes * 60)
        try:
            await asyncio.to_thread(work)
        except PipelineBusyError:
            pass  # another instance is scoring
        except Exception:
            logger.exception("%s failed", name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.api_keys = load_api_keys()
    app.state.rate_limiter = make_rate_limiter()
    db.init_db()
    status = reports_status()
    if not status["available"]:
        logger.warning(status["detail"])
    problems = integrity_problems()
    if problems:
        logger.error("Published model files failed verification; scoring is disabled: %s", "; ".join(problems))
    else:
        try:
            # Warm every cache and record the run, so the pipeline view starts with real timings.
            await asyncio.to_thread(run_scoring_pipeline, "startup")
        except PipelineBusyError:
            pass
        except Exception:
            logger.exception("Startup scoring run failed")
    await asyncio.to_thread(_purge)
    tasks = [asyncio.create_task(_every(MAINTENANCE_HOURS * 60, _purge, "Retention purge"))]
    interval = scoring_interval_minutes()
    if interval > 0:
        tasks.append(asyncio.create_task(_every(interval, lambda: run_scoring_pipeline("schedule"), "Scheduled scoring run")))
    yield
    for task in tasks:
        task.cancel()


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
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
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


for module in (model, research, customers, cases, pipeline, predict, datasets, reports, admin):
    app.include_router(module.router, dependencies=[Depends(current_user)])


@app.get("/api/health", response_model=Health)
def health():
    payload = {"database": db.get_engine().dialect.name, "blob_store": get_blobstore().kind, "reports": reports_status()}
    try:
        get_trained_model()
        payload.update(status="ok", model_loaded=True, model_version=get_pipeline_spec()["model_version"], problems=[])
    except ModelUnavailable as exc:
        payload.update(status="degraded", model_loaded=False, model_version="unknown", problems=integrity_problems() or [str(exc)])
    except Exception:
        logger.exception("Health check could not load the model")
        payload.update(status="degraded", model_loaded=False, model_version="unknown", problems=["The model could not be loaded"])
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


@app.exception_handler(FeatureInputError)
async def feature_input_error(request: Request, exc: FeatureInputError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ModelUnavailable)
async def model_unavailable(request: Request, exc: ModelUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(CaseTransitionError)
async def case_transition(request: Request, exc: CaseTransitionError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(CasePermissionError)
async def case_permission(request: Request, exc: CasePermissionError):
    return JSONResponse(status_code=403, content={"detail": str(exc)})


@app.exception_handler(DatasetInUseError)
async def dataset_in_use(request: Request, exc: DatasetInUseError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


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


CONSOLE_NOT_BUILT = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>GridSentinel API</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:40rem;margin:3rem auto;padding:0 1rem;color:#14161b}
code{background:#f1efea;padding:.1rem .3rem;border-radius:4px}</style></head>
<body><h1>GridSentinel API is running</h1>
<p>Health: <a href="/api/health">/api/health</a>.{docs}</p>
<p>The console is not built into this server. Either run it separately:</p>
<pre><code>cd frontend
npm ci
npm run dev</code></pre>
<p>and open <a href="http://localhost:5173">http://localhost:5173</a>, or build it once with
<code>npm run build</code> in <code>frontend/</code>, restart uvicorn and reload this page.</p>
</body></html>"""


if (frontend_dist / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def serve_frontend(path: str):
    if path == "api" or path.startswith("api/"):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    if not (frontend_dist / "index.html").is_file():
        # No console build: say how to open it instead of a bare 404.
        if path:
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        docs = ' API docs: <a href="/api/docs">/api/docs</a>.' if IS_DEVELOPMENT else ""
        return HTMLResponse(CONSOLE_NOT_BUILT.replace("{docs}", docs))
    file_path = resolve_frontend_file(path)
    if file_path is not None:
        return FileResponse(file_path)
    return FileResponse(frontend_dist / "index.html")
