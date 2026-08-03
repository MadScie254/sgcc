from __future__ import annotations

import os

from fastapi import Header, HTTPException, Request, status


def load_api_key() -> str | None:
    env_name = os.getenv("ENV", "development").lower()
    api_key = os.getenv("API_KEY")

    if env_name != "development" and not api_key:
        raise RuntimeError("API_KEY must be set unless ENV=development")

    return api_key


def require_api_key(request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    api_key = getattr(request.app.state, "api_key", None)
    if not api_key:
        return

    if x_api_key != api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")