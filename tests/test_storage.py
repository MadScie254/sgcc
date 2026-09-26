"""Tests for the blob stores (local and S3 through moto) and the database helpers."""

import boto3
import pytest
from moto import mock_aws

from backend.services import db
from backend.services.blobstore import LocalBlobStore, S3BlobStore
from backend.services.errors import NotFoundError


def roundtrip(store) -> None:
    store.put("uploads/20260101000000-abcdef12.csv", b"CONS_NO\n1\n", "text/csv")
    assert store.get("uploads/20260101000000-abcdef12.csv") == b"CONS_NO\n1\n"
    store.delete("uploads/20260101000000-abcdef12.csv")
    with pytest.raises(NotFoundError):
        store.get("uploads/20260101000000-abcdef12.csv")
    for key in ("../secret.csv", "uploads/../../x.csv", "/etc/passwd", "other/a.csv", "uploads/a.exe"):
        with pytest.raises(ValueError):
            store.put(key, b"", "text/plain")


def test_local_blob_store(tmp_path):
    roundtrip(LocalBlobStore(tmp_path))
    assert not any(tmp_path.rglob("*.tmp"))


def test_s3_blob_store(monkeypatch):
    for name, value in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(name, value)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="gridsentinel")
        store = S3BlobStore("gridsentinel", prefix="prod")
        roundtrip(store)
        store.put("reports/research-20260101000000-abcdef12.pdf", b"%PDF", "application/pdf")
        keys = [o["Key"] for o in boto3.client("s3").list_objects_v2(Bucket="gridsentinel")["Contents"]]
        assert keys == ["prod/reports/research-20260101000000-abcdef12.pdf"]


def test_settings_are_shared_and_deletable():
    db.init_db()
    db.set_setting("test-setting", {"value": 1}, "amina")
    row = db.get_setting("test-setting")
    assert row["value"] == {"value": 1} and row["updated_by"] == "amina"
    db.set_setting("test-setting", None, "amina")
    assert db.get_setting("test-setting") is None


def test_scoring_lock_is_exclusive():
    with db.scoring_lock() as first:
        with db.scoring_lock() as second:
            assert first and not second
    with db.scoring_lock() as again:
        assert again


def test_database_url_normalises_hosted_postgres(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@host:5432/gridsentinel")
    assert db.database_url() == "postgresql+psycopg://u:p@host:5432/gridsentinel"
