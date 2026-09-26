from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.dependencies.auth import User, current_user, require_supervisor
from backend.schemas import Dataset
from backend.services import db
from backend.services.datasets import delete_dataset, get_dataset, list_datasets, read_upload, register_upload

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("", response_model=List[Dataset])
def datasets():
    return list_datasets()


@router.post("", response_model=Dataset, status_code=201)
async def upload(file: UploadFile = File(...), user: User = Depends(current_user)):
    content = await read_upload(file)
    item = await run_in_threadpool(register_upload, file.filename or "upload.csv", content, user.name)
    db.audit(user.name, user.role, "dataset.upload", item["dataset_id"], item["filename"])
    return item


@router.get("/{dataset_id}", response_model=Dataset)
def dataset(dataset_id: str):
    return get_dataset(dataset_id)


@router.delete("/{dataset_id}", status_code=204)
def remove(dataset_id: str, user: User = Depends(require_supervisor)) -> None:
    delete_dataset(dataset_id)
    db.audit(user.name, user.role, "dataset.delete", dataset_id)
