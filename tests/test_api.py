"""End-to-end tests of the FastAPI app against the committed model and demo data."""

import gzip
import io
import json
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.main as main
import backend.services.reports as reports_service
from backend.main import app, resolve_frontend_file
from backend.services.config import get_config, get_paths
from backend.services.data import get_feature_matrix, get_pipeline_spec, get_wide_data
from src.pipeline import model_input
from tests.conftest import DEMO_DATASET, sgcc_frame


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def top_customer(client):
    return client.get("/api/customers", params={"page_size": 1}).json()["items"][0]["customer_id"]


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


def demo_rows(n: int) -> pd.DataFrame:
    with gzip.open(DEMO_DATASET, "rt") as handle:
        return pd.read_csv(handle, nrows=n)


# --- Health, auth, static files -----------------------------------------------

def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["reports"]["available"] is True


def test_api_key_enforced(client):
    app.state.api_key = "s3cret"
    try:
        assert client.get("/api/model/metrics").status_code == 401
        assert client.get("/api/model/metrics", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/api/model/metrics", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert client.get("/api/health").status_code == 200
    finally:
        app.state.api_key = None


def test_frontend_paths_cannot_escape_build_dir(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("index")
    (tmp_path / "secret.txt").write_text("secret")
    monkeypatch.setattr(main, "frontend_dist", dist.resolve())

    assert resolve_frontend_file("index.html") == (dist / "index.html").resolve()
    for path in ("../secret.txt", "/etc/hostname", "assets/../../secret.txt", str(tmp_path / "secret.txt")):
        assert resolve_frontend_file(path) is None


# --- Model ---------------------------------------------------------------------

def test_model_metrics_and_quality(client):
    metrics = client.get("/api/model/metrics").json()
    cm = metrics["confusion_matrix"]
    assert metrics["customers_monitored"] == sum(cm.values()) > 100
    assert metrics["flagged"] == cm["tp"] + cm["fp"]
    assert metrics["metrics"]["auc"] > 0.8
    preview = client.get("/api/predict/threshold-preview", params={"threshold": metrics["trained_threshold"]}).json()
    assert preview["metrics"]["auc"] > 0.8
    assert preview["metrics"]["precision"] > 0.4


def test_operating_curve_and_distribution(client):
    population = client.get("/api/model/metrics").json()["customers_monitored"]
    curve = client.get("/api/model/operating-curve").json()
    assert all(p["tp"] + p["fp"] + p["fn"] + p["tn"] == population for p in curve)
    recalls = [p["recall"] for p in curve]
    assert recalls == sorted(recalls, reverse=True)
    dist = client.get("/api/model/score-distribution").json()
    assert sum(dist["honest"]) + sum(dist["theft"]) == population


def test_comparison_drivers_training(client):
    rows = client.get("/api/model/comparison").json()
    assert [r["model"] for r in rows] == ["proposed", "xgboost", "xgboost_default", "random_forest_smote", "logistic_regression_smote"]
    served = [r for r in rows if r["served"]]
    assert len(served) == 1 and served[0]["model"] in {"proposed", "xgboost"}
    assert all(r["inference_ms_per_customer"] > 0 and r["model_size_mb"] > 0 and 0 < r["threshold"] < 1 for r in rows)
    assert {r["treatment"] for r in rows} == {"none", "smote", "smote_enn"}

    resampling = client.get("/api/model/resampling").json()
    counts = resampling["smote_enn"]["counts"]
    assert counts["after"]["theft"] > counts["before"]["theft"]
    assert resampling["config"]["sampling_strategy"] == 0.5

    drivers = client.get("/api/model/drivers").json()
    values = [d["mean_abs_shap"] for d in drivers["drivers"]]
    assert values == sorted(values, reverse=True) and len(values) == 15
    assert all(d["label"] and d["risk_when"] in {"higher", "lower", "unclear"} for d in drivers["drivers"])

    training = client.get("/api/model/training").json()
    assert training["pipeline"] == served[0]["model"]
    assert training["train_customers"] > training["validation_customers"] == training["test_customers"] > 0
    assert [s["name"] for s in training["stages"]] == [
        "Load", "Clean", "Features", "Split", "Resample", "Tune", "Validate", "Evaluate", "Publish"]


def test_explanation_check(client, top_customer):
    check = client.get(f"/api/customers/{top_customer}/explanation-check").json()
    assert len(check["shap"]) == len(check["lime"]) == check["top_n"] == 5
    assert set(check["shared"]) <= {a["feature"] for a in check["shap"]}
    assert not check["agrees"] or len(check["shared"]) >= 3
    assert ("review" in check["message"]) != check["agrees"]
    assert client.get("/api/customers/nope/explanation-check").status_code == 404


def test_serving_matches_training_pipeline(client):
    # The committed spec must describe the committed model, or serving would score differently.
    assert (get_paths()["artifacts"] / "pipeline.json").is_file()
    assert get_pipeline_spec()["name"] == client.get("/api/model/training").json()["pipeline"]
    wide, _ = get_wide_data()
    X, _ = get_feature_matrix()
    pd.testing.assert_frame_equal(X, model_input(wide, get_pipeline_spec(), get_config().get("features")))


def test_publish_threshold_rescores_and_resets(client):
    before = client.get("/api/model/metrics").json()
    raised = client.put("/api/model/threshold", json={"threshold": 0.6}).json()
    assert raised["threshold"] == 0.6
    assert raised["flagged"] < before["flagged"]
    assert client.get("/api/pipeline/runs", params={"limit": 1}).json()[0]["trigger"] == "threshold"

    restored = client.put("/api/model/threshold", json={"threshold": None}).json()
    assert restored["threshold"] == before["trained_threshold"]
    assert restored["flagged"] == before["flagged"]
    assert client.put("/api/model/threshold", json={"threshold": 1.5}).status_code == 422


# --- Customers and predictions -------------------------------------------------

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


def test_explanation_adds_up(client, top_customer):
    body = client.get(f"/api/customers/{top_customer}/explanation").json()
    assert len(body["contributions"]) == 87
    logit = body["base_value"] + sum(c["shap_value"] for c in body["contributions"])
    assert 1 / (1 + math.exp(-logit)) == pytest.approx(body["probability"], abs=1e-4)
    assert all(c["label"] and c["display_value"] for c in body["contributions"])


def test_predict_single(client, top_customer):
    by_id = client.post("/api/predict/single", json={"customer_id": top_customer}).json()
    listed = client.get("/api/customers", params={"page_size": 1}).json()["items"][0]
    assert by_id["probability"] == pytest.approx(listed["risk_score"])
    assert len(by_id["reasons"]) == 3

    partial = client.post("/api/predict/single", json={"features": {"missing_ratio": 0.4}, "threshold": 0.3}).json()
    assert partial["threshold"] == 0.3 and partial["customer_id"] is None

    assert client.post("/api/predict/single", json={}).status_code == 400
    assert client.post("/api/predict/single", json={"customer_id": "nope"}).status_code == 404


def test_batch_scores_meter_data_and_features(client):
    rows = demo_rows(20)
    response = client.post("/api/predict/batch", files={"file": ("meter.csv", csv_bytes(rows), "text/csv")})
    assert response.status_code == 200
    scored = pd.read_csv(io.StringIO(response.text))
    assert list(scored.columns) == ["customer_id", "probability", "prediction", "risk_tier", "label"]
    assert len(scored) == 20

    # The same customers as feature rows must score identically.
    X, _ = get_feature_matrix()
    ids = rows["CONS_NO"].astype(str).tolist()
    features = X.loc[ids].reset_index()
    by_features = pd.read_csv(io.StringIO(client.post(
        "/api/predict/batch", files={"file": ("features.csv", csv_bytes(features), "text/csv")}).text))
    merged = scored.merge(by_features, on="customer_id", suffixes=("_meter", "_features"))
    assert len(merged) == 20
    assert (merged["probability_meter"] - merged["probability_features"]).abs().max() < 1e-5


def test_batch_rejects_unusable_files(client):
    junk = client.post("/api/predict/batch", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert junk.status_code == 400
    assert "Unrecognised layout" in junk.json()["detail"]
    assert client.post("/api/predict/batch", files={"file": ("x.csv", b"\x00\x01", "text/csv")}).status_code == 400


# --- Cases and pipeline ----------------------------------------------------------

def test_pipeline_run(client):
    run = client.post("/api/pipeline/runs").json()
    assert run["status"] == "succeeded"
    assert [s["key"] for s in run["stages"]] == ["ingest", "features", "score", "explain", "route"]
    assert run["summary"]["flagged"] == client.get("/api/model/metrics").json()["flagged"]
    assert client.get("/api/pipeline/runs", params={"limit": 1}).json()[0]["run_id"] == run["run_id"]
    assert client.get("/api/pipeline/config").json()["run_on_startup"] is True


def test_case_workflow_survives_threshold_change(client):
    listing = client.get("/api/cases", params={"page_size": 100}).json()
    flagged = client.get("/api/model/metrics").json()["flagged"]
    assert listing["total"] == flagged
    lowest = listing["items"][-1] if listing["total"] <= 100 else client.get(
        "/api/cases", params={"page": math.ceil(flagged / 100), "page_size": 100}).json()["items"][-1]
    assert lowest["top_driver"]["shap_value"] > 0

    updated = client.patch(f"/api/cases/{lowest['customer_id']}", json={"status": "dispatched", "note": "check seal"}).json()
    assert updated["status"] == "dispatched" and updated["note"] == "check seal"
    assert any("dispatched" in e["event"] for e in updated["history"])

    # Raising the threshold un-flags this customer, but the case in progress stays listed.
    client.put("/api/model/threshold", json={"threshold": 0.9})
    try:
        cases = client.get("/api/cases", params={"status": "dispatched"}).json()
        row = next(r for r in cases["items"] if r["customer_id"] == lowest["customer_id"])
        assert row["flagged"] is False
    finally:
        client.put("/api/model/threshold", json={"threshold": None})

    assert client.patch(f"/api/cases/{lowest['customer_id']}", json={"status": "bogus"}).status_code == 422
    assert client.patch("/api/cases/nope", json={"status": "cleared"}).status_code == 404


# --- Datasets and reports --------------------------------------------------------

def test_dataset_upload_meter_data_with_labels(client):
    response = client.post("/api/datasets", files={"file": ("meter.csv", csv_bytes(demo_rows(300)), "text/csv")})
    assert response.status_code == 201
    summary = response.json()["summary"]
    assert summary["format"] == "consumption" and summary["customers"] == 300 and summary["days"] == 1034
    assert summary["labelled"] and summary["label_metrics"]["roc_auc"] > 0.7
    dataset_id = response.json()["dataset_id"]
    assert client.get(f"/api/datasets/{dataset_id}").json()["dataset_id"] == dataset_id
    assert client.get("/api/datasets").json()[0]["dataset_id"] == dataset_id


def test_dataset_upload_errors(client, monkeypatch):
    unlabelled = sgcc_frame().drop(columns="FLAG")
    ok = client.post("/api/datasets", files={"file": ("small.csv", csv_bytes(unlabelled), "text/csv")})
    assert ok.status_code == 201 and ok.json()["summary"]["labelled"] is False
    assert client.post("/api/datasets", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}).status_code == 400
    monkeypatch.setattr("backend.services.datasets.MAX_UPLOAD_BYTES", 100)
    assert client.post("/api/datasets", files={"file": ("big.csv", b"x" * 1000, "text/csv")}).status_code == 413


def test_dataset_retention(client, monkeypatch):
    monkeypatch.setattr("backend.services.datasets.MAX_DATASETS", 1)
    first, second = (client.post("/api/datasets", files={"file": ("s.csv", csv_bytes(sgcc_frame()), "text/csv")}).json() for _ in range(2))
    assert [d["dataset_id"] for d in client.get("/api/datasets").json()] == [second["dataset_id"]]
    assert client.get(f"/api/datasets/{first['dataset_id']}").status_code == 404
    assert not (get_paths()["uploads"] / f"{first['dataset_id']}.csv").exists()


@pytest.mark.parametrize("kind", ["portfolio", "dataset", "case"])
def test_reports_generate_and_download(client, top_customer, kind):
    payload = {"kind": kind}
    if kind == "dataset":
        payload["dataset_id"] = client.post(
            "/api/datasets", files={"file": ("meter.csv", csv_bytes(demo_rows(50)), "text/csv")}).json()["dataset_id"]
    if kind == "case":
        payload["customer_id"] = top_customer
    created = client.post("/api/reports", json=payload)
    assert created.status_code == 201, created.text
    report = created.json()
    assert report["kind"] == kind and report["bytes"] > 1000

    pdf = client.get(f"/api/reports/{report['report_id']}/pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")
    assert client.get("/api/reports").json()[0]["report_id"] == report["report_id"]


def test_report_errors(client, monkeypatch):
    missing = client.post("/api/reports", json={"kind": "dataset"})
    assert missing.status_code == 422 and "dataset_id is required" in missing.text
    assert client.post("/api/reports", json={"kind": "case"}).status_code == 422
    unknown = client.post("/api/reports", json={"kind": "case", "customer_id": "nope"})
    assert unknown.status_code == 404 and unknown.json()["detail"] == "Unknown customer_id: nope"
    assert client.post("/api/reports", json={"kind": "dataset", "dataset_id": "nope"}).status_code == 404
    assert client.post("/api/reports", json={"kind": "other"}).status_code == 422
    assert client.get("/api/reports/..%2f..%2fetc%2fpasswd/pdf").status_code == 404

    monkeypatch.setattr(reports_service.fpdf, "FPDF_VERSION", "1.7.2")
    unavailable = client.post("/api/reports", json={"kind": "portfolio"})
    assert unavailable.status_code == 503
    assert "pip install -r requirements.lock" in unavailable.json()["detail"]
    assert client.get("/api/health").json()["reports"]["available"] is False
