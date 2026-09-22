"""
Shared pytest fixtures.

Data files are stored in Git LFS. A checkout without `git lfs pull` leaves
small pointer files in their place, which still pass `Path.exists()`, so
data-dependent tests must check the content, not just the path.
"""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path) -> bool:
    """Return True if the file is an un-fetched Git LFS pointer."""
    with open(path, "rb") as handle:
        return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def require_data_file(relative_path: str) -> Path:
    """
    Resolve a repo data file, skipping the test if it is missing or an LFS pointer.

    Set REQUIRE_LFS=1 (as CI does) to fail instead of skip, so a broken LFS
    fetch cannot silently turn into skipped tests.
    """
    path = REPO_ROOT / relative_path
    if not path.exists():
        reason = f"{relative_path} is missing"
    elif is_lfs_pointer(path):
        reason = f"{relative_path} is a Git LFS pointer; run `git lfs pull`"
    else:
        return path

    if os.environ.get("REQUIRE_LFS") == "1":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def small_dataset_path() -> Path:
    """Path to data/datasetsmall.csv, fetched from LFS."""
    return require_data_file("data/datasetsmall.csv")
