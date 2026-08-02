from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

app = FastAPI(title="SGCC Theft Detector API", version="0.1.0")

frontend_origin = os.getenv("FRONTEND_DEV_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin, "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(eda_router)
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(customers_router)
app.include_router(model_router)
app.include_router(explain_router)
app.include_router(compare_router)
app.include_router(monitor_router)
app.include_router(analytics_router)
app.include_router(datasets_router)
app.include_router(reports_router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "sgcc-backend"}


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
