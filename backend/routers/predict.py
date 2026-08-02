from __future__ import annotations

import csv
import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from backend.schemas.api import (
    SinglePredictionRequest,
    SinglePredictionResponse,
    ThresholdPreviewResponse,
)
from backend.services.model import (
    get_feature_names,
    predict_for_customer,
    predict_from_features,
    threshold_preview,
)

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
        payload = predict_from_features(request.features, threshold=request.threshold)
        return SinglePredictionResponse(**payload)

    raise HTTPException(status_code=400, detail="Provide either customer_id or features")


@router.post("/batch")
async def predict_batch(file: UploadFile = File(...)) -> StreamingResponse:
    contents = await file.read()
    frame = pd.read_csv(io.BytesIO(contents))
    feature_names = get_feature_names()
    aligned = frame.reindex(columns=feature_names, fill_value=0.0)
    model_payloads = []
    from backend.services.model import get_trained_model

    model = get_trained_model()
    probabilities = model.predict_proba(aligned)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    output = io.StringIO()
    writer = csv.writer(output)
    header = list(frame.columns) + ["prediction", "probability"]
    writer.writerow(header)
    for idx, row in frame.iterrows():
        writer.writerow(list(row.values) + [int(predictions[idx]), float(probabilities[idx])])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'},
    )


@router.get("/threshold-preview", response_model=ThresholdPreviewResponse)
def preview_threshold(threshold: float = Query(..., ge=0.0, le=1.0)) -> ThresholdPreviewResponse:
    payload = threshold_preview(threshold)
    return ThresholdPreviewResponse(**payload)
