from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from backend.schemas import Prediction, PredictionRequest, ThresholdPreview
from backend.services import model
from backend.services.datasets import parse_csv, read_upload, score_frame, scores_csv

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("/single", response_model=Prediction)
def predict_single(request: PredictionRequest):
    if request.customer_id:
        return model.predict_customer(request.customer_id, threshold=request.threshold)
    if request.features:
        return model.predict_features(request.features, threshold=request.threshold)
    raise HTTPException(status_code=400, detail="Provide either customer_id or features")


@router.post("/batch")
async def predict_batch(file: UploadFile = File(...)) -> Response:
    """Score a CSV (SGCC meter data or model features); returns one row per customer."""
    content = await read_upload(file)
    scored = await run_in_threadpool(lambda: score_frame(parse_csv(content)))
    return Response(
        content=scores_csv(scored, model.get_decision_threshold()),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'},
    )


@router.get("/threshold-preview", response_model=ThresholdPreview)
def threshold_preview(threshold: float = Query(..., ge=0.0, le=1.0)):
    return model.threshold_preview(threshold)
