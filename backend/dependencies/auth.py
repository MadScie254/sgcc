from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, Request, status


def load_api_key() -> str | None:
    env_name = os.getenv("ENV", "development").lower()
    api_key = os.getenv("API_KEY") or None

    if env_name != "development" and not api_key:
        raise RuntimeError("API_KEY must be set unless ENV=development")

    return api_key


def require_api_key(request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    api_key = getattr(request.app.state, "api_key", None)
    if not api_key:
        return

    # Constant-time comparison so response timing does not leak the key.
    if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), api_key.encode()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
