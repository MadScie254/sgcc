from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.routers.compare import router as compare_router
from backend.routers.analytics import router as analytics_router
from backend.routers.datasets import router as datasets_router
from backend.routers.customers import router as customers_router
from backend.routers.eda import router as eda_router
from backend.routers.explain import router as explain_router
from backend.routers.model import router as model_router
from backend.routers.monitor import router as monitor_router
from backend.routers.reports import router as reports_router
from backend.routers.predict import router as predict_router
from backend.routers.train import router as train_router
from backend.dependencies.auth import load_api_key, require_api_key
from backend.services.model import get_feature_names, get_model_config, get_shap_explainer, get_trained_model


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.api_key = load_api_key()
    try:
        get_trained_model()
        get_shap_explainer()
        get_feature_names()
    except Exception:
        logger.exception("Backend startup warmup failed")
    yield

app = FastAPI(title="SGCC Theft Detector API", version="0.1.0", lifespan=lifespan)

frontend_origin = os.getenv("FRONTEND_DEV_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin, "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

secured_router_kwargs = {"dependencies": [Depends(require_api_key)]}

app.include_router(eda_router, **secured_router_kwargs)
app.include_router(train_router, **secured_router_kwargs)
app.include_router(predict_router, **secured_router_kwargs)
app.include_router(customers_router, **secured_router_kwargs)
app.include_router(model_router, **secured_router_kwargs)
app.include_router(explain_router, **secured_router_kwargs)
app.include_router(compare_router, **secured_router_kwargs)
app.include_router(monitor_router, **secured_router_kwargs)
app.include_router(analytics_router, **secured_router_kwargs)
app.include_router(datasets_router, **secured_router_kwargs)
app.include_router(reports_router, **secured_router_kwargs)


@app.get("/api/health")
def health() -> dict:
    try:
        model = get_trained_model()
        model_loaded = model is not None
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
    if not model_loaded:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    error_id = str(uuid.uuid4())
    logger.exception("Unhandled error %s on %s %s", error_id, request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal error", "error_id": error_id})


frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{path:path}")
    def serve_frontend(path: str):
        if path.startswith("api/"):
            return {"detail": "Not Found"}

        candidate = frontend_dist / path
        if candidate.is_file():
            return FileResponse(candidate)

        index_file = frontend_dist / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

        return {"detail": "Frontend build not found"}
