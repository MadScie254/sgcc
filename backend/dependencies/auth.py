"""Named API keys with roles.

``API_KEYS`` holds a JSON list of ``{"name", "role", "sha256"}``: the SHA-256 of
each key, never the key itself (``python scripts/make_api_key.py <name> <role>``
prints both). Roles:

- analyst: views scores and explanations, reviews and dispatches cases, writes
  notes, uploads datasets, generates reports and starts scoring runs;
- supervisor: also resolves and reopens cases, publishes the operating threshold,
  promotes an uploaded dataset to the operational population and deletes files.

With ``ENV=development`` and no ``API_KEYS``, requests are not authenticated and act
as the supervisor "developer". Every request is rate-limited per key.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Literal, Optional

from fastapi import Depends, Header, HTTPException, Request, status

from backend.services.config import environment

Role = Literal["analyst", "supervisor"]
ROLES = ("analyst", "supervisor")


@dataclass(frozen=True)
class User:
    name: str
    role: Role

    @property
    def is_supervisor(self) -> bool:
        return self.role == "supervisor"


DEVELOPER = User("developer", "supervisor")


@dataclass(frozen=True)
class KeyEntry:
    user: User
    digest: bytes


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def load_api_keys() -> Optional[List[KeyEntry]]:
    """Parsed ``API_KEYS``; None means no authentication (development only)."""
    raw = os.getenv("API_KEYS", "").strip()
    if not raw:
        if environment() != "development":
            raise RuntimeError("API_KEYS must be set unless ENV=development (see scripts/make_api_key.py)")
        return None
    try:
        entries = json.loads(raw)
        keys = []
        for entry in entries:
            if entry["role"] not in ROLES:
                raise ValueError(f"role must be one of {ROLES}")
            digest = bytes.fromhex(entry["sha256"])
            if len(digest) != 32:
                raise ValueError("sha256 must be 64 hex characters")
            keys.append(KeyEntry(User(str(entry["name"])[:64], entry["role"]), digest))
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"API_KEYS is not a JSON list of {{name, role, sha256}}: {exc}") from exc
    if not keys:
        raise RuntimeError("API_KEYS is empty")
    if len({k.user.name for k in keys}) != len(keys):
        raise RuntimeError("API_KEYS names must be unique")
    return keys


def _match(keys: List[KeyEntry], presented: str) -> Optional[User]:
    digest = hashlib.sha256(presented.encode()).digest()
    found = None
    for entry in keys:  # compare against every entry, in constant time each
        if hmac.compare_digest(digest, entry.digest):
            found = entry.user
    return found


class RateLimiter:
    """Sliding one-minute window per identity, in memory (per instance)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, identity: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            hits = self._hits[identity]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            return True


def make_rate_limiter() -> RateLimiter:
    try:
        return RateLimiter(int(os.getenv("RATE_LIMIT_PER_MINUTE", "600")))
    except ValueError:
        return RateLimiter(600)


def current_user(request: Request, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> User:
    keys = getattr(request.app.state, "api_keys", None)
    limiter: RateLimiter = request.app.state.rate_limiter
    if keys is None:
        user = DEVELOPER
    else:
        user = _match(keys, x_api_key) if x_api_key else None
        if user is None:
            client = request.client.host if request.client else "unknown"
            if not limiter.allow(f"failed:{client}"):
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    if not limiter.allow(f"user:{user.name}"):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests")
    return user


def require_supervisor(user: User = Depends(current_user)) -> User:
    if not user.is_supervisor:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This action needs the supervisor role")
    return user
