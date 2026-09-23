from __future__ import annotations

import io
import os
from typing import Tuple

import pandas as pd
from fastapi import HTTPException, UploadFile

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_UPLOAD_ROWS = 200_000
MAX_UPLOAD_COLUMNS = 2_000


async def read_upload_bytes(file: UploadFile) -> bytes:
    """Read an upload, rejecting anything over MAX_UPLOAD_BYTES without buffering it all."""
    chunks, size = [], 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        chunks.append(chunk)
    return b"".join(chunks)


def parse_csv(contents: bytes) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(contents), nrows=MAX_UPLOAD_ROWS + 1)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc
    if frame.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows")
    if len(frame) > MAX_UPLOAD_ROWS or frame.shape[1] > MAX_UPLOAD_COLUMNS:
        raise HTTPException(
            status_code=413,
            detail=f"CSV exceeds {MAX_UPLOAD_ROWS} rows or {MAX_UPLOAD_COLUMNS} columns",
        )
    return frame


async def read_csv_upload(file: UploadFile) -> Tuple[pd.DataFrame, bytes]:
    contents = await read_upload_bytes(file)
    return parse_csv(contents), contents
