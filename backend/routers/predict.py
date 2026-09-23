from __future__ import annotations

import io

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from backend.schemas.api import (
    SinglePredictionRequest,
    SinglePredictionResponse,
    ThresholdPreviewResponse,
)
from backend.services.model import (
    get_decision_threshold,
    predict_for_customer,
    predict_frame,
    predict_from_features,
    threshold_preview,
)
from backend.services.uploads import read_csv_upload

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("/single", response_model=SinglePredictionResponse)
def predict_single(request: SinglePredictionRequest) -> SinglePredictionResponse:
    if request.customer_id:
        try:
            payload = predict_for_customer(request.customer_id, threshold=request.threshold)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SinglePredictionResponse(**payload)

    if request.features:
        return SinglePredictionResponse(**predict_from_features(request.features, threshold=request.threshold))

    raise HTTPException(status_code=400, detail="Provide either customer_id or features")


@router.post("/batch")
async def predict_batch(file: UploadFile = File(...)) -> StreamingResponse:
    """Score a CSV of feature columns; returns the rows with prediction and probability appended."""
    frame, _ = await read_csv_upload(file)
    threshold = get_decision_threshold()
    probabilities = predict_frame(frame)

    result = frame.copy()
    result["probability"] = probabilities
    result["prediction"] = (probabilities >= threshold).astype(int)
    # Neutralise spreadsheet formula injection in echoed text cells.
    for column in result.select_dtypes(include="object").columns:
        result[column] = result[column].map(
            lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r") else v
        )

    output = io.StringIO()
    result.to_csv(output, index=False)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'},
    )


@router.get("/threshold-preview", response_model=ThresholdPreviewResponse)
def preview_threshold(threshold: float = Query(..., ge=0.0, le=1.0)) -> ThresholdPreviewResponse:
    return ThresholdPreviewResponse(**threshold_preview(threshold))
