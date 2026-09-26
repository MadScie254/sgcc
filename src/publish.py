"""
SGCC Theft Detector - Publishing a trained model

Training writes every output into a staging directory first. ``Publisher.commit``
removes the old manifest, moves each staged file into place, and only then writes
``artifacts/manifest.json`` with the SHA-256 of every file. A run that dies half
way therefore leaves no manifest (the API reports "degraded" and refuses to
score) instead of a model that silently disagrees with its own pipeline spec.

``verify_manifest`` is what the API runs at startup.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List

MANIFEST = "artifacts/manifest.json"
LIBRARIES = ("xgboost", "scikit-learn", "imbalanced-learn", "numpy", "pandas", "scipy", "optuna")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(repo: Path) -> Dict[str, Any]:
    """Commit the code was trained from, and whether the working tree had uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def library_versions() -> Dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


class Publisher:
    """Collects a run's outputs under ``root/.staging`` and moves them into ``root`` at once."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.staging = self.root / ".staging"
        shutil.rmtree(self.staging, ignore_errors=True)
        self.files: List[str] = []

    def path(self, relative: str) -> Path:
        """Staged location for ``relative`` (a path relative to ``root``); the parent is created."""
        if relative not in self.files:
            self.files.append(relative)
        target = self.staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def json(self, relative: str, payload: Any) -> None:
        self.path(relative).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def commit(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Move every staged file into place, then write the manifest last."""
        manifest_path = self.root / MANIFEST
        manifest_path.unlink(missing_ok=True)
        entries = {}
        for relative in self.files:
            source, target = self.staging / relative, self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            entries[relative] = {"sha256": sha256(source), "bytes": source.stat().st_size}
            os.replace(source, target)
        manifest = {"published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info, "files": entries}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, manifest_path)
        shutil.rmtree(self.staging, ignore_errors=True)
        return manifest


def verify_manifest(root: Path) -> List[str]:
    """Problems with the published files under ``root``: missing manifest, missing or changed files."""
    path = Path(root) / MANIFEST
    if not path.is_file():
        return [f"{MANIFEST} is missing: retrain, or restore the published artifacts"]
    try:
        files = json.loads(path.read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError):
        return [f"{MANIFEST} is unreadable"]
    problems = []
    for relative, entry in files.items():
        target = Path(root) / relative
        if not target.is_file():
            problems.append(f"{relative} is missing")
        elif sha256(target) != entry["sha256"]:
            problems.append(f"{relative} does not match the manifest (changed after training)")
    return problems
