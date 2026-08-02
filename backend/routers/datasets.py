from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.schemas.api import DatasetCatalogResponse, DatasetUploadResponse
from backend.services.reporting import get_uploaded_dataset, load_upload_catalog, register_uploaded_dataset

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("/catalog", response_model=DatasetCatalogResponse)
def catalog() -> DatasetCatalogResponse:
    return DatasetCatalogResponse(items=load_upload_catalog())


@router.get("/{dataset_id}", response_model=DatasetUploadResponse)
def dataset(dataset_id: str) -> DatasetUploadResponse:
    try:
        return DatasetUploadResponse(item=get_uploaded_dataset(dataset_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload(file: UploadFile = File(...)) -> DatasetUploadResponse:
    try:
        item = await register_uploaded_dataset(file)
        return DatasetUploadResponse(item=item)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc