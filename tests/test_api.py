"""End-to-end tests of the FastAPI app against the committed model and demo data."""

import json

import pytest
from fastapi.testclient import TestClient

import backend.main as main
from backend.main import app, resolve_frontend_file


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def customer_id(client):
    return client.get("/api/customers", params={"page_size": 1}).json()["items"][0]["customer_id"]


def test_health(client):
    body = client.get("/api/health").json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"


def test_customers_ranked_with_tuned_threshold(client):
    body = client.get("/api/customers", params={"page_size": 5}).json()
    scores = [item["risk_score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert body["total"] > 100
    metrics = client.get("/api/model/metrics").json()
    assert all(item["threshold"] == metrics["threshold"] for item in body["items"])
    assert metrics["metrics"]["auc"] > 0.7


def test_timeseries_is_valid_json_with_missing_readings(client, customer_id):
    response = client.get(f"/api/customers/{customer_id}/timeseries")
    assert response.status_code == 200
    body = json.loads(response.text)  # strict: NaN would fail to serialise
    assert body["points"][0]["date"] == "2014-01-01"


def test_predict_single(client, customer_id):
    by_id = client.post("/api/predict/single", json={"customer_id": customer_id}).json()
    assert 0 <= by_id["probability"] <= 1
    assert len(by_id["top_reasons"]) == 3

    partial = client.post("/api/predict/single", json={"features": {"missing_ratio": 0.4}, "threshold": 0.3})
    assert partial.status_code == 200
    assert partial.json()["threshold"] == 0.3

    assert client.post("/api/predict/single", json={}).status_code == 400
    assert client.post("/api/predict/single", json={"customer_id": "nope"}).status_code == 404


def test_batch_predict_rejects_garbage(client):
    response = client.post("/api/predict/batch", files={"file": ("x.csv", b"\x00\x01\x02", "text/csv")})
    assert response.status_code == 400


def test_explanations(client, customer_id):
    local = client.get(f"/api/explain/local-shap/{customer_id}").json()
    assert len(local["feature_names"]) == len(local["shap_values"]) == len(local["feature_values"])
    assert client.get("/api/explain/global-shap", params={"sample_count": 20}).status_code == 200
    assert client.get("/api/predict/threshold-preview", params={"threshold": 0.4}).status_code == 200
    assert client.get("/api/eda/summary").json()["feature_count"] > 50


def test_input_validation(client):
    assert client.post("/api/reports/generate", json={"country_code": "../x"}).status_code == 422
    assert client.post("/api/reports/generate", json={"latitude": 500}).status_code == 422
    assert client.post("/api/train/jobs", json={"mode": "quick", "config_overrides": {"paths": {"models": "/tmp"}}}).status_code == 422
    assert client.get("/api/reports/..%2f..%2fetc%2fpasswd/download").status_code == 404


def test_api_key_enforced(client):
    app.state.api_key = "s3cret"
    try:
        assert client.get("/api/customers").status_code == 401
        assert client.get("/api/customers", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/api/customers", headers={"X-API-Key": "s3cret"}).status_code == 200
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
    for path in ("../secret.txt", "/etc/hostname", "..%2fsecret.txt", "assets/../../secret.txt", str(tmp_path / "secret.txt")):
        assert resolve_frontend_file(path) is None


def test_upload_and_pdf_report(client, tmp_path, monkeypatch):
    import backend.services.config as config_service
    import backend.services.reporting as reporting
    from backend.services.data import get_feature_matrix

    paths = {**config_service.get_project_paths(), "uploads": tmp_path / "uploads", "reports": tmp_path / "reports"}
    monkeypatch.setattr(reporting, "get_project_paths", lambda: paths)
    monkeypatch.setattr(reporting, "get_country_context", lambda code: {"name": "Algeria", "capital": "Algiers"})
    monkeypatch.setattr(reporting, "get_public_holidays", lambda code: [])
    monkeypatch.setattr(reporting, "get_weather_context", lambda lat, lon: {})

    X, y = get_feature_matrix()
    frame = X.head(20).copy()
    frame["label"] = y.head(20).to_numpy()
    upload = client.post("/api/datasets/upload", files={"file": ("données.csv", frame.to_csv().encode(), "text/csv")})
    assert upload.status_code == 200
    item = upload.json()["item"]
    assert item["status"] == "verified"
    assert item["summary"]["verification_accuracy"] is not None

    report = client.post("/api/reports/generate", json={"dataset_id": item["dataset_id"]})
    assert report.status_code == 200
    pdf = client.get(report.json()["download_url"])
    assert pdf.status_code == 200
    assert pdf.content.startswith(b"%PDF")


def test_scoring_pipeline_run_records_stages(client):
    run = client.post("/api/pipeline/runs").json()
    assert run["status"] == "succeeded"
    assert [s["key"] for s in run["stages"]] == ["ingest", "features", "score", "explain", "route"]
    assert run["summary"]["flagged"] == client.get("/api/model/metrics").json()["flagged_today"]
    assert client.get("/api/pipeline/runs", params={"limit": 1}).json()[0]["run_id"] == run["run_id"]
    assert client.get("/api/pipeline/config").json()["run_on_startup"] is True
    assert client.get("/api/pipeline/training").json()["n_features"] == 85


def test_case_workflow(client, customer_id):
    listing = client.get("/api/cases", params={"page_size": 5}).json()
    assert listing["total"] == client.get("/api/model/metrics").json()["flagged_today"]
    assert listing["items"][0]["top_driver"]["shap_value"] > 0

    updated = client.patch(f"/api/cases/{customer_id}", json={"status": "dispatched", "note": "check seal"}).json()
    assert updated["status"] == "dispatched"
    assert updated["note"] == "check seal"
    assert any("dispatched" in event["event"] for event in updated["history"])
    assert client.get("/api/cases", params={"status": "dispatched"}).json()["total"] >= 1

    assert client.patch(f"/api/cases/{customer_id}", json={"status": "bogus"}).status_code == 422
    assert client.patch("/api/cases/nope", json={"status": "cleared"}).status_code == 404


def test_publish_threshold_rescores_and_resets(client):
    before = client.get("/api/model/metrics").json()
    raised = client.put("/api/model/threshold", json={"threshold": 0.6}).json()
    assert raised["threshold"] == 0.6
    assert raised["flagged_today"] < before["flagged_today"]
    assert raised["trained_threshold"] == before["trained_threshold"]

    restored = client.put("/api/model/threshold", json={"threshold": None}).json()
    assert restored["threshold"] == before["trained_threshold"]
    assert restored["flagged_today"] == before["flagged_today"]
    assert client.put("/api/model/threshold", json={"threshold": 1.5}).status_code == 422


def test_operating_curve_and_distribution_match_population(client):
    curve = client.get("/api/model/operating-curve").json()
    population = client.get("/api/model/metrics").json()["customers_monitored"]
    assert all(p["tp"] + p["fp"] + p["fn"] + p["tn"] == population for p in curve)
    recalls = [p["recall"] for p in curve]
    assert recalls == sorted(recalls, reverse=True)
    dist = client.get("/api/model/score-distribution").json()
    assert sum(dist["honest"]) + sum(dist["theft"]) == population


def test_served_model_quality(client):
    """The integrated model keeps its hold-out quality on the customers the API serves."""
    trained = client.get("/api/model/metrics").json()["trained_threshold"]
    preview = client.get("/api/predict/threshold-preview", params={"threshold": trained}).json()
    assert preview["metrics"]["auc"] > 0.8
    assert preview["metrics"]["precision"] > 0.4
