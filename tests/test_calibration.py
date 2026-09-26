"""Tests for src.calibration and src.publish."""

import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.calibration import apply_platt, calibration_metrics, fit_platt, reliability
from src.publish import MANIFEST, Publisher, verify_manifest


@pytest.fixture(scope="module")
def overconfident():
    """Scores that rank well but overstate theft, as a model trained on resampled rows does."""
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(int)
    logit = np.where(y == 1, 1.0, -0.5) + rng.normal(0, 1, len(y))
    return 1 / (1 + np.exp(-logit)), y


def test_platt_keeps_ranking_and_improves_calibration(overconfident):
    score, y = overconfident
    params = fit_platt(score, y)
    calibrated = apply_platt(score, params)
    assert params["a"] > 0
    assert np.array_equal(np.argsort(score, kind="stable"), np.argsort(calibrated, kind="stable"))
    assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, score))
    before, after = calibration_metrics(y, score), calibration_metrics(y, calibrated)
    assert after["ece"] < before["ece"] / 3
    assert after["brier"] < before["brier"]
    assert after["mean_predicted"] == pytest.approx(y.mean(), abs=0.01)


def test_platt_rejects_scores_that_do_not_rank():
    y = np.r_[np.zeros(100), np.ones(100)]
    with pytest.raises(ValueError, match="not positive"):
        fit_platt(np.r_[np.full(100, 0.9), np.full(100, 0.1)], y)


def test_reliability_bins_cover_every_row(overconfident):
    score, y = overconfident
    table = reliability(y, score, bins=10)
    assert sum(row["count"] for row in table) == len(y)
    assert all(row["low"] <= row["mean_predicted"] <= row["high"] for row in table)


def test_publisher_writes_manifest_last_and_detects_changes(tmp_path):
    out = Publisher(tmp_path)
    out.json("artifacts/pipeline.json", {"name": "xgboost"})
    out.path("models/model.bin").write_bytes(b"trees")
    assert not (tmp_path / MANIFEST).exists()  # nothing is live before commit
    manifest = out.commit({"pipeline": "xgboost"})

    assert set(manifest["files"]) == {"artifacts/pipeline.json", "models/model.bin"}
    assert json.loads((tmp_path / MANIFEST).read_text())["pipeline"] == "xgboost"
    assert not (tmp_path / ".staging").exists()
    assert verify_manifest(tmp_path) == []

    (tmp_path / "models/model.bin").write_bytes(b"other trees")
    assert verify_manifest(tmp_path) == ["models/model.bin does not match the manifest (changed after training)"]
    (tmp_path / "artifacts/pipeline.json").unlink()
    assert "artifacts/pipeline.json is missing" in verify_manifest(tmp_path)
    (tmp_path / MANIFEST).unlink()
    assert "missing" in verify_manifest(tmp_path)[0]
