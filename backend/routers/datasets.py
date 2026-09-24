from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.schemas import Dataset
from backend.services.datasets import get_dataset, list_datasets, read_upload, register_upload

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("", response_model=List[Dataset])
def datasets():
    return list_datasets()


@router.post("", response_model=Dataset, status_code=201)
async def upload(file: UploadFile = File(...)):
    content = await read_upload(file)
    return await run_in_threadpool(register_upload, file.filename or "upload.csv", content)


@router.get("/{dataset_id}", response_model=Dataset)
def dataset(dataset_id: str):
    return get_dataset(dataset_id)
