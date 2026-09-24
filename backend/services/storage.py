"""JSON files under the runtime state directory, written atomically."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any) -> Any:
    """The file's content, or `default` when it is missing, unreadable or of another type."""
    if not path.is_file():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default
    return payload if isinstance(payload, type(default)) else default


def write_json(path: Path, payload: Any) -> None:
    """Write via a temporary file and rename, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)
