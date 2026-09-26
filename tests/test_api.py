"""End-to-end tests of the FastAPI app against the committed model, predictions and population."""

import gzip
import io
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

import backend.main as main
import backend.services.reports as reports_service
from backend.dependencies.auth import KeyEntry, RateLimiter, User, hash_key
from backend.main import app, resolve_frontend_file
from backend.services import db
from backend.services import model as model_service
from backend.services.config import get_paths
from backend.services.data import get_feature_matrix, get_pipeline_spec, get_wide_data, load_predictions
from backend.services.datasets import purge_expired_datasets
from src.calibration import apply_platt
from src.pipeline import model_input
from tests.conftest import DEMO_DATASET, sgcc_frame

SUPERVISOR = {"X-API-Key": "supervisor-key"}
ANALYST = {"X-API-Key": "analyst-key"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def named_keys(client):
    """Switch the app from development (no auth) to two named keys for one test."""
    app.state.api_keys = [KeyEntry(User("amina", "supervisor"), bytes.fromhex(hash_key("supervisor-key"))),
                          KeyEntry(User("otieno", "analyst"), bytes.fromhex(hash_key("analyst-key")))]
    try:
        yield
    finally:
        app.state.api_keys = None


@pytest.fixture(scope="module")
def top_customer(client):
    return client.get("/api/customers", params={"page_size": 1}).json()["items"][0]["customer_id"]


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


def population_rows(n: int) -> pd.DataFrame:
    with gzip.open(DEMO_DATASET, "rt") as handle:
        return pd.read_csv(handle, nrows=n)


def labelled_rows(n: int) -> pd.DataFrame:
    """Population customers with their true labels, which only the saved test predictions hold."""
    rows = population_rows(n)
    labels = load_predictions("test").set_index("customer_id")["label"]
    rows.insert(1, "FLAG", labels.reindex(rows["CONS_NO"].astype(str)).to_numpy())
    return rows


# --- Health, identity, integrity, static files ----------------------------------

def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True and body["problems"] == []
    assert body["database"] in {"sqlite", "postgresql"} and body["blob_store"] in {"local", "s3"}
    assert body["reports"]["available"] is True


def test_named_keys_and_roles(client, named_keys):
    assert client.get("/api/model/metrics").status_code == 401
    assert client.get("/api/model/metrics", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/me", headers=ANALYST).json() == {"name": "otieno", "role": "analyst", "auth_required": True}
    assert client.get("/api/model/metrics", headers=ANALYST).status_code == 200
    # Publishing a threshold, reading the audit log and switching the population need a supervisor.
    assert client.put("/api/model/threshold", json={"threshold": None}, headers=ANALYST).status_code == 403
    assert client.get("/api/audit", headers=ANALYST).status_code == 403
    assert client.delete("/api/population", headers=ANALYST).status_code == 403
    assert client.put("/api/model/threshold", json={"threshold": None}, headers=SUPERVISOR).status_code == 200
    latest = client.get("/api/audit", params={"limit": 1}, headers=SUPERVISOR).json()[0]
    assert (latest["actor"], latest["role"], latest["action"]) == ("amina", "supervisor", "threshold.publish")


def test_development_mode_acts_as_developer(client):
    assert client.get("/api/me").json() == {"name": "developer", "role": "supervisor", "auth_required": False}


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(app.state, "rate_limiter", RateLimiter(3))
    codes = [client.get("/api/me").status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_changed_model_files_disable_scoring(client, monkeypatch):
    monkeypatch.setattr(model_service, "verify_manifest", lambda root: ["models/xgb_best.ubj does not match the manifest"])
    model_service.clear_caches()
    try:
        health = client.get("/api/health")
        assert health.status_code == 503
        assert health.json()["problems"] == ["models/xgb_best.ubj does not match the manifest"]
        refused = client.post("/api/pipeline/runs")
        assert refused.status_code == 503 and "Scoring is disabled" in refused.json()["detail"]
    finally:
        monkeypatch.undo()
        model_service.clear_caches()
    assert client.get("/api/health").status_code == 200


def test_root_explains_how_to_open_the_console_until_it_is_built(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "frontend_dist", (tmp_path / "missing").resolve())
    page = client.get("/")
    assert page.status_code == 200 and "npm run dev" in page.text and "/api/health" in page.text
    assert client.get("/cases").status_code == 404
    assert client.get("/api/nope").status_code == 404

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<div id=root></div>")
    monkeypatch.setattr(main, "frontend_dist", dist.resolve())
    assert client.get("/").text == "<div id=root></div>"
    assert client.get("/cases/123").text == "<div id=root></div>"  # client-side routes get the app
    assert client.get("/api/nope").status_code == 404


def test_frontend_paths_cannot_escape_build_dir(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("index")
    (tmp_path / "secret.txt").write_text("secret")
    monkeypatch.setattr(main, "frontend_dist", dist.resolve())

    assert resolve_frontend_file("index.html") == (dist / "index.html").resolve()
    for path in ("../secret.txt", "/etc/hostname", "assets/../../secret.txt", str(tmp_path / "secret.txt")):
        assert resolve_frontend_file(path) is None


# --- Operations: no labels ------------------------------------------------------------

def test_operational_population_has_no_labels(client, top_customer):
    with gzip.open(DEMO_DATASET, "rt") as handle:
        header = handle.readline().strip().split(",")
    assert header[0] == "CONS_NO" and "FLAG" not in header and "label" not in header
    assert "label" not in client.get(f"/api/customers/{top_customer}/timeseries").json()
    metrics = client.get("/api/model/metrics").json()
    assert not {"confusion_matrix", "metrics", "base_rate"} & set(metrics)


def test_model_metrics(client):
    metrics = client.get("/api/model/metrics").json()
    assert metrics["population"]["source"] == "sample" and metrics["customers_monitored"] == 3000
    assert metrics["pipeline"] == get_pipeline_spec()["name"]
    assert 0 < metrics["expected_thefts_flagged"] < metrics["flagged"]
    assert metrics["expected_thefts_flagged"] < metrics["expected_thefts_total"] < metrics["customers_monitored"]
    assert sum(metrics["risk_tier_distribution"].values()) == metrics["customers_monitored"]


def test_threshold_preview_uses_validation_customers(client):
    trained = client.get("/api/model/metrics").json()["trained_threshold"]
    preview = client.get("/api/model/threshold-preview", params={"threshold": trained}).json()
    validation = preview["validation"]
    assert validation["customers"] == len(load_predictions("validation")) == 6356
    assert validation["tp"] + validation["fp"] + validation["fn"] + validation["tn"] == validation["customers"]
    assert validation["precision"] > 0.4 and validation["recall"] > 0.3
    assert preview["population_flagged"] == client.get("/api/model/metrics").json()["flagged"]


def test_operating_curve_with_cost_model(client):
    curve = client.get("/api/model/operating-curve", params={"capacity": 50, "cost_per_visit": 40, "value_per_theft": 500}).json()
    points = curve["points"]
    assert curve["validation_customers"] == 6356 and curve["capacity"] == 50
    recalls = [p["recall"] for p in points]
    assert recalls == sorted(recalls, reverse=True)
    assert all(p["visits"] == min(p["population_flagged"], 50) for p in points)
    assert all(p["net_value"] == pytest.approx(p["expected_thefts_found"] * 500 - p["visits"] * 40) for p in points)
    # With a capacity, lowering the threshold past the capacity adds no visits.
    assert points[0]["visits"] == 50 and points[0]["expected_thefts_found"] == pytest.approx(points[1]["expected_thefts_found"])
    dist = client.get("/api/model/score-distribution").json()
    assert sum(dist["counts"]) == 3000


def test_serving_matches_training_pipeline():
    # The committed spec must describe the committed model, or serving would score differently.
    spec = get_pipeline_spec()
    assert spec["features"] == list(model_service.get_trained_model().get_booster().feature_names)
    assert spec["provenance"]["code"]["commit"] and len(spec["provenance"]["data_sha256"]) == 64
    wide = get_wide_data()
    pd.testing.assert_frame_equal(get_feature_matrix(), model_input(wide, spec, spec["feature_config"]))


def test_publish_threshold_rescores_and_resets(client):
    before = client.get("/api/model/metrics").json()
    raised = client.put("/api/model/threshold", json={"threshold": 0.6}).json()
    assert raised["threshold"] == 0.6
    assert raised["flagged"] < before["flagged"]
    run = client.get("/api/pipeline/runs", params={"limit": 1}).json()[0]
    assert (run["trigger"], run["actor"]) == ("threshold", "developer")

    restored = client.put("/api/model/threshold", json={"threshold": None}).json()
    assert restored["threshold"] == before["trained_threshold"]
    assert restored["flagged"] == before["flagged"]
    assert client.put("/api/model/threshold", json={"threshold": 1.5}).status_code == 422


# --- Research: the test set ------------------------------------------------------------

def test_research_evaluation(client):
    body = client.get("/api/research/evaluation").json()
    assert body["population"]["customers"] == 6356 and body["population"]["theft"] == 542
    assert "no evidence yet" in body["population"]["limitation"]
    assert body["metrics"]["auc"] > 0.8 and body["metrics"]["pr_auc"] > 0.4
    assert sum(body["confusion_matrix"].values()) == 6356
    assert body["calibration"]["platt"]["ece"] < body["calibration"]["raw"]["ece"]


def test_research_comparison_and_significance(client):
    rows = client.get("/api/research/comparison").json()
    assert [r["model"] for r in rows] == ["proposed", "xgboost", "xgboost_default", "random_forest_smote", "logistic_regression_smote"]
    served = [r for r in rows if r["served"]]
    assert len(served) == 1 and served[0]["model"] in {"proposed", "xgboost"}
    assert all(r["inference_ms_per_customer"] > 0 and r["model_size_mb"] > 0 and 0 < r["threshold"] < 1 for r in rows)
    assert all(r["pr_auc_ci"][0] < r["pr_auc"] < r["pr_auc_ci"][1] for r in rows)
    assert rows[0]["p_value_pr_auc"] is None and all(0 <= r["p_value_mcnemar"] <= 1 for r in rows[1:])
    significance = client.get("/api/research/significance").json()
    assert significance["population"] == {"split": "test", "customers": 6356, "theft": 542}
    assert significance["resamples"] == 10_000


def test_research_curves_calibration_training(client):
    curve = client.get("/api/research/operating-curve").json()
    assert all(p["tp"] + p["fp"] + p["fn"] + p["tn"] == 6356 for p in curve["points"])
    dist = client.get("/api/research/score-distribution").json()
    assert sum(dist["honest"]) + sum(dist["theft"]) == 6356 and sum(dist["theft"]) == 542

    calibration = client.get("/api/research/calibration").json()
    assert len(calibration["pipelines"]) == 5 and calibration["fitted_on"] == "validation"
    assert sum(b["count"] for b in calibration["reliability"]["platt"]) == 6356

    resampling = client.get("/api/research/resampling").json()
    counts = resampling["smote_enn"]["counts"]
    assert counts["after"]["theft"] > counts["before"]["theft"]

    training = client.get("/api/research/training").json()
    assert training["pipeline"] == served_pipeline()
    assert training["train_customers"] > training["validation_customers"] == training["test_customers"] > 0
    assert len(training["cv_fold_scores"]) == 5
    assert training["cv_best_score"] == pytest.approx(np.mean(training["cv_fold_scores"]), abs=1e-5)
    assert [s["name"] for s in training["stages"]] == [
        "Load", "Clean", "Features", "Split", "Resample", "Tune", "Validate", "Evaluate", "Publish"]
    assert training["manifest"]["pipeline"] == training["pipeline"]


def served_pipeline() -> str:
    return get_pipeline_spec()["name"]


# --- Customers, explanations and predictions ---------------------------------------

def test_customers_ranked(client):
    body = client.get("/api/customers", params={"page_size": 5}).json()
    scores = [item["risk_score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert [item["rank"] for item in body["items"]] == [1, 2, 3, 4, 5]
    assert client.get("/api/customers", params={"tier": "nope"}).status_code == 422


def test_timeseries(client, top_customer):
    body = json.loads(client.get(f"/api/customers/{top_customer}/timeseries").text)  # strict: no NaN
    assert body["points"][0]["date"] == "2014-01-01"
    assert len(body["points"]) == 1034
    assert client.get("/api/customers/nope/timeseries").status_code == 404


def test_explanation_adds_up_to_the_raw_score(client, top_customer):
    body = client.get(f"/api/customers/{top_customer}/explanation").json()
    assert len(body["contributions"]) == 87
    logit = body["base_value"] + sum(c["shap_value"] for c in body["contributions"])
    assert 1 / (1 + math.exp(-logit)) == pytest.approx(body["raw_score"], abs=1e-4)
    assert body["probability"] == pytest.approx(float(apply_platt([body["raw_score"]], get_pipeline_spec()["calibration"])[0]))
    assert all(c["label"] and c["display_value"] for c in body["contributions"])


def test_explanation_consistency(client, top_customer):
    check = client.get(f"/api/customers/{top_customer}/explanation-check").json()
    assert len(check["shap"]) == len(check["lime"]) == check["top_n"] == 5
    assert set(check["shared"]) <= {a["feature"] for a in check["shap"]}
    assert not check["consistent"] or len(check["shared"]) >= 3
    assert "not that it is causal" in check["note"]
    assert client.get("/api/customers/nope/explanation-check").status_code == 404


def test_predict_single(client, top_customer):
    by_id = client.post("/api/predict/single", json={"customer_id": top_customer}).json()
    listed = client.get("/api/customers", params={"page_size": 1}).json()["items"][0]
    assert by_id["probability"] == pytest.approx(listed["risk_score"])
    assert len(by_id["reasons"]) == 3

    row = get_feature_matrix().loc[top_customer]
    features = {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
    by_features = client.post("/api/predict/single", json={"features": features, "threshold": 0.3}).json()
    assert by_features["probability"] == pytest.approx(by_id["probability"], abs=1e-6)
    assert by_features["threshold"] == 0.3 and by_features["customer_id"] is None

    partial = client.post("/api/predict/single", json={"features": {"missing_ratio": 0.4}})
    assert partial.status_code == 400 and "86 of the 87 model features are missing" in partial.json()["detail"]
    assert client.post("/api/predict/single", json={}).status_code == 400
    assert client.post("/api/predict/single", json={"customer_id": "nope"}).status_code == 404


def test_batch_scores_meter_data_and_features(client):
    rows = population_rows(20)
    response = client.post("/api/predict/batch", files={"file": ("meter.csv", csv_bytes(rows), "text/csv")})
    assert response.status_code == 200
    scored = pd.read_csv(io.StringIO(response.text))
    assert list(scored.columns) == ["customer_id", "probability", "prediction", "risk_tier"]
    assert len(scored) == 20

    # The same customers as feature rows must score identically.
    ids = rows["CONS_NO"].astype(str).tolist()
    features = get_feature_matrix().loc[ids].reset_index()
    by_features = pd.read_csv(io.StringIO(client.post(
        "/api/predict/batch", files={"file": ("features.csv", csv_bytes(features), "text/csv")}).text))
    merged = scored.merge(by_features, on="customer_id", suffixes=("_meter", "_features"))
    assert len(merged) == 20
    assert (merged["probability_meter"] - merged["probability_features"]).abs().max() < 1e-5

    incomplete = client.post("/api/predict/batch", files={"file": ("f.csv", csv_bytes(features.drop(columns="missing_ratio")), "text/csv")})
    assert incomplete.status_code == 400 and "1 of the 87 model features are missing: missing_ratio" in incomplete.json()["detail"]


def test_batch_rejects_unusable_files(client):
    junk = client.post("/api/predict/batch", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert junk.status_code == 400 and "Unrecognised layout" in junk.json()["detail"]
    assert client.post("/api/predict/batch", files={"file": ("x.csv", b"\x00\x01", "text/csv")}).status_code == 400

    repeated = pd.concat([sgcc_frame(), sgcc_frame().iloc[[1]]])
    duplicate = client.post("/api/predict/batch", files={"file": ("d.csv", csv_bytes(repeated), "text/csv")})
    assert duplicate.status_code == 400 and "appear more than once: B" in duplicate.json()["detail"]

    ambiguous = sgcc_frame()
    ambiguous.columns = [c if c in ("CONS_NO", "FLAG") else pd.Timestamp(c).strftime("%d/%m/%Y") for c in ambiguous.columns]
    response = client.post("/api/predict/batch", files={"file": ("a.csv", csv_bytes(ambiguous), "text/csv")})
    assert response.status_code == 400 and "year first" in response.json()["detail"]


# --- Pipeline and cases ----------------------------------------------------------------

def test_pipeline_run(client):
    run = client.post("/api/pipeline/runs").json()
    assert run["status"] == "succeeded"
    assert [s["key"] for s in run["stages"]] == ["ingest", "features", "score", "explain", "route"]
    assert run["summary"]["flagged"] == client.get("/api/model/metrics").json()["flagged"]
    assert client.get("/api/pipeline/runs", params={"limit": 1}).json()[0]["run_id"] == run["run_id"]
    assert client.get("/api/pipeline/config").json()["run_on_startup"] is True


def test_scoring_runs_one_at_a_time(client):
    with db.scoring_lock() as held:
        assert held
        assert client.post("/api/pipeline/runs").status_code == 409


def test_case_workflow(client, named_keys):
    cases = client.get("/api/cases", params={"page_size": 100, "status": "new"}, headers=ANALYST).json()["items"]
    cid = cases[-1]["customer_id"]

    def move(headers, **payload):
        return client.patch(f"/api/cases/{cid}", json=payload, headers=headers)

    assert client.get(f"/api/cases/{cid}", headers=ANALYST).json()["allowed_transitions"] == ["reviewing"]
    skipped = move(ANALYST, status="confirmed", reason="x", evidence="y")
    assert skipped.status_code == 409 and "cannot move from new to confirmed" in skipped.json()["detail"]
    assert move(ANALYST, status="reviewing", note="usage fell after meter swap").status_code == 200
    reviewing = move(ANALYST, status="dispatched").json()
    assert reviewing["status"] == "dispatched" and reviewing["updated_by"] == "otieno"
    assert reviewing["allowed_transitions"] == []  # resolving needs a supervisor
    assert move(ANALYST, status="confirmed", reason="bypass found", evidence="INS-1").status_code == 403
    no_evidence = move(SUPERVISOR, status="confirmed", reason="bypass found")
    assert no_evidence.status_code == 409 and "evidence reference" in no_evidence.json()["detail"]

    resolved = move(SUPERVISOR, status="confirmed", reason="bypass found at the meter", evidence="INS-2024-118").json()
    assert resolved["resolution"]["by"] == "amina" and resolved["resolution"]["evidence"] == "INS-2024-118"
    assert move(SUPERVISOR, status="reviewing").status_code == 409  # reopening needs a reason
    reopened = move(SUPERVISOR, status="reviewing", reason="customer appealed").json()
    assert reopened["status"] == "reviewing" and reopened["resolution"] is None
    actors = [(e["actor"], e["event"]) for e in reopened["history"]]
    assert ("otieno", "Status: new → reviewing") in actors and ("otieno", "Analyst note updated") in actors
    assert ("amina", "Status: confirmed → reviewing (customer appealed)") in actors

    # Raising the threshold un-flags most customers, but a case in progress stays listed.
    client.put("/api/model/threshold", json={"threshold": 0.95}, headers=SUPERVISOR)
    try:
        listed = client.get("/api/cases", params={"status": "reviewing"}, headers=ANALYST).json()
        assert cid in {r["customer_id"] for r in listed["items"]}
    finally:
        client.put("/api/model/threshold", json={"threshold": None}, headers=SUPERVISOR)

    assert move(ANALYST, status="bogus").status_code == 422
    assert client.patch("/api/cases/nope", json={"status": "reviewing"}, headers=ANALYST).status_code == 404


# --- Datasets, population and reports ----------------------------------------------

def test_dataset_upload_meter_data_with_labels(client):
    response = client.post("/api/datasets", files={"file": ("meter.csv", csv_bytes(labelled_rows(300)), "text/csv")})
    assert response.status_code == 201
    item = response.json()
    summary = item["summary"]
    assert summary["format"] == "consumption" and summary["customers"] == 300 and summary["days"] == 1034
    assert summary["labelled"] and summary["label_metrics"]["roc_auc"] > 0.7
    assert item["uploaded_by"] == "developer"
    assert client.get(f"/api/datasets/{item['dataset_id']}").json()["dataset_id"] == item["dataset_id"]
    assert client.get("/api/datasets").json()[0]["dataset_id"] == item["dataset_id"]


def test_dataset_upload_errors(client, monkeypatch):
    unlabelled = sgcc_frame().drop(columns="FLAG")
    ok = client.post("/api/datasets", files={"file": ("small.csv", csv_bytes(unlabelled), "text/csv")})
    assert ok.status_code == 201 and ok.json()["summary"]["labelled"] is False
    assert client.post("/api/datasets", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}).status_code == 400
    monkeypatch.setattr("backend.services.datasets.MAX_UPLOAD_BYTES", 100)
    assert client.post("/api/datasets", files={"file": ("big.csv", b"x" * 1000, "text/csv")}).status_code == 413


def test_promote_upload_to_population(client, named_keys):
    upload = client.post("/api/datasets", files={"file": ("new.csv", csv_bytes(population_rows(200)), "text/csv")},
                         headers=ANALYST).json()
    assert client.post("/api/population", json={"dataset_id": upload["dataset_id"]}, headers=ANALYST).status_code == 403
    promoted = client.post("/api/population", json={"dataset_id": upload["dataset_id"]}, headers=SUPERVISOR)
    assert promoted.status_code == 200 and promoted.json()["source"] == "upload"
    try:
        metrics = client.get("/api/model/metrics", headers=ANALYST).json()
        assert metrics["customers_monitored"] == 200 and metrics["population"]["promoted_by"] == "amina"
        assert client.get("/api/pipeline/runs", params={"limit": 1}, headers=ANALYST).json()[0]["summary"]["population"] == upload["dataset_id"]
        in_use = client.delete(f"/api/datasets/{upload['dataset_id']}", headers=SUPERVISOR)
        assert in_use.status_code == 409
    finally:
        reset = client.delete("/api/population", headers=SUPERVISOR).json()
    assert reset["source"] == "sample"
    assert client.get("/api/model/metrics", headers=ANALYST).json()["customers_monitored"] == 3000
    features = client.post("/api/datasets", files={"file": ("f.csv", csv_bytes(get_feature_matrix().head(5).reset_index()), "text/csv")},
                           headers=ANALYST).json()
    assert client.post("/api/population", json={"dataset_id": features["dataset_id"]}, headers=SUPERVISOR).status_code == 400


def test_delete_and_retention(client, named_keys, monkeypatch):
    first, second = (client.post("/api/datasets", files={"file": ("s.csv", csv_bytes(sgcc_frame()), "text/csv")},
                                 headers=ANALYST).json() for _ in range(2))
    assert client.delete(f"/api/datasets/{first['dataset_id']}", headers=ANALYST).status_code == 403
    assert client.delete(f"/api/datasets/{first['dataset_id']}", headers=SUPERVISOR).status_code == 204
    assert client.get(f"/api/datasets/{first['dataset_id']}", headers=ANALYST).status_code == 404

    old = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat(timespec="seconds")
    with db.transaction() as connection:
        connection.execute(update(db.datasets).where(db.datasets.c.dataset_id == second["dataset_id"]).values(uploaded_at=old))
    monkeypatch.setenv("RETENTION_DAYS", "30")
    assert purge_expired_datasets() >= 1
    assert client.get(f"/api/datasets/{second['dataset_id']}", headers=ANALYST).status_code == 404
    assert not (get_paths()["state"] / "blobs" / "uploads" / f"{second['dataset_id']}.csv").exists()


@pytest.mark.parametrize("kind", ["portfolio", "dataset", "case", "research"])
def test_reports_generate_and_download(client, top_customer, kind):
    payload = {"kind": kind}
    if kind == "dataset":
        payload["dataset_id"] = client.post(
            "/api/datasets", files={"file": ("meter.csv", csv_bytes(population_rows(50)), "text/csv")}).json()["dataset_id"]
    if kind == "case":
        payload["customer_id"] = top_customer
    created = client.post("/api/reports", json=payload)
    assert created.status_code == 201, created.text
    report = created.json()
    assert report["kind"] == kind and report["bytes"] > 1000 and report["created_by"] == "developer"

    pdf = client.get(f"/api/reports/{report['report_id']}/pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")
    assert client.get("/api/reports").json()[0]["report_id"] == report["report_id"]
    assert db.list_audit(1)[0]["action"] == "report.download"


def test_report_delete_and_errors(client, monkeypatch):
    report = client.post("/api/reports", json={"kind": "portfolio"}).json()
    assert client.delete(f"/api/reports/{report['report_id']}").status_code == 204
    assert client.get(f"/api/reports/{report['report_id']}/pdf").status_code == 404

    missing = client.post("/api/reports", json={"kind": "dataset"})
    assert missing.status_code == 422 and "dataset_id is required" in missing.text
    assert client.post("/api/reports", json={"kind": "case"}).status_code == 422
    unknown = client.post("/api/reports", json={"kind": "case", "customer_id": "nope"})
    assert unknown.status_code == 404 and unknown.json()["detail"] == "Unknown customer_id: nope"
    assert client.post("/api/reports", json={"kind": "dataset", "dataset_id": "nope"}).status_code == 404
    assert client.post("/api/reports", json={"kind": "other"}).status_code == 422
    assert client.get("/api/reports/..%2f..%2fetc%2fpasswd/pdf").status_code == 404

    monkeypatch.setattr(reports_service, "_fpdf_version", lambda: "1.7.2")
    unavailable = client.post("/api/reports", json={"kind": "portfolio"})
    assert unavailable.status_code == 503
    assert "pip install -r requirements.lock" in unavailable.json()["detail"]
    assert client.get("/api/health").json()["reports"]["available"] is False


def test_old_pdf_library_disables_reports_not_the_api(tmp_path):
    """PyFPDF 1.x (the old package that shares fpdf2's import name) must not stop the API from starting."""
    stub = tmp_path / "fpdf"
    stub.mkdir()
    (stub / "__init__.py").write_text('FPDF_VERSION = "1.7.2"\n\nclass FPDF:\n    pass\n')
    code = ("import backend.main\n"
            "from backend.services.reports import reports_status\n"
            "status = reports_status()\n"
            "assert status['available'] is False and status['fpdf_version'] == '1.7.2', status\n"
            "assert 'pip install -r requirements.lock' in status['detail']\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(Path(__file__).resolve().parents[1])])}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
