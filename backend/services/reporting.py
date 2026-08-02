from __future__ import annotations

import csv
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import UploadFile
from fastapi.responses import FileResponse
from fpdf import FPDF

from backend.services.config import get_project_paths
from backend.services.model import get_feature_names, get_model_metrics, get_trained_model, get_customer_rankings, risk_tier_for_probability
from backend.services.public_apis import get_country_context, get_public_holidays, get_weather_context


UPLOAD_INDEX_FILENAME = "catalog.json"
REPORT_INDEX_FILENAME = "reports.json"


def _paths() -> Dict[str, Path]:
    paths = get_project_paths()
    uploads_dir = paths["uploads"]
    reports_dir = paths["reports"]
    uploads_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return {"uploads": uploads_dir, "reports": reports_dir, "artifacts": paths["artifacts"]}


def _catalog_path() -> Path:
    return _paths()["uploads"] / UPLOAD_INDEX_FILENAME


def _report_index_path() -> Path:
    return _paths()["reports"] / REPORT_INDEX_FILENAME


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    return []


def _write_json_list(path: Path, payload: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "dataset"


def load_upload_catalog() -> List[Dict[str, Any]]:
    return sorted(_read_json_list(_catalog_path()), key=lambda item: item.get("uploaded_at", ""), reverse=True)


def load_report_index() -> List[Dict[str, Any]]:
    return sorted(_read_json_list(_report_index_path()), key=lambda item: item.get("generated_at", ""), reverse=True)


def _save_upload_catalog(items: List[Dict[str, Any]]) -> None:
    _write_json_list(_catalog_path(), items)


def _save_report_index(items: List[Dict[str, Any]]) -> None:
    _write_json_list(_report_index_path(), items)


def _load_csv_frame(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.BytesIO(file_bytes))


def _coerce_numeric(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    coerced = frame.reindex(columns=columns, fill_value=0.0).copy()
    for column in columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce").fillna(0.0)
    return coerced


def _detect_label_column(frame: pd.DataFrame) -> Optional[str]:
    for candidate in ("label", "target", "is_theft", "theft"):
        if candidate in frame.columns:
            return candidate
    return None


def verify_dataset_frame(frame: pd.DataFrame) -> Dict[str, Any]:
    feature_names = get_feature_names()
    model = get_trained_model()
    metrics = get_model_metrics()
    threshold = float(metrics["threshold"])

    aligned = _coerce_numeric(frame, feature_names)
    probabilities = np.asarray(model.predict_proba(aligned)[:, 1], dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    label_column = _detect_label_column(frame)
    verification_accuracy = None
    mismatches = None
    if label_column:
        labels = pd.to_numeric(frame[label_column], errors="coerce").fillna(0).astype(int).to_numpy()
        verification_accuracy = float((predictions == labels).mean())
        mismatches = int((predictions != labels).sum())

    missing_features = [feature for feature in feature_names if feature not in frame.columns]
    extra_columns = [column for column in frame.columns if column not in feature_names]

    top_rows = frame.copy()
    top_rows["prediction_probability"] = probabilities
    top_rows["prediction"] = predictions
    top_rows["risk_tier"] = top_rows["prediction_probability"].apply(risk_tier_for_probability)
    sort_columns = ["prediction_probability"]
    if "customer_id" in top_rows.columns:
        sort_columns.insert(0, "customer_id")

    top_risk_rows = top_rows.sort_values("prediction_probability", ascending=False).head(10)

    summary = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "feature_columns_present": int(len(feature_names) - len(missing_features)),
        "missing_features": missing_features,
        "extra_columns": extra_columns,
        "threshold": threshold,
        "predicted_positive_rows": int(predictions.sum()),
        "mean_probability": float(probabilities.mean()) if len(probabilities) else 0.0,
        "median_probability": float(np.median(probabilities)) if len(probabilities) else 0.0,
        "high_risk_rows": int((probabilities >= 0.7).sum()),
        "medium_risk_rows": int(((probabilities >= 0.4) & (probabilities < 0.7)).sum()),
        "low_risk_rows": int((probabilities < 0.4).sum()),
        "label_column": label_column,
        "verification_accuracy": verification_accuracy,
        "verification_mismatches": mismatches,
    }

    return {
        "summary": summary,
        "top_risk_rows": top_risk_rows.replace({np.nan: None}).to_dict(orient="records"),
        "probabilities": probabilities.tolist(),
        "predictions": predictions.tolist(),
    }


async def register_uploaded_dataset(file: UploadFile) -> Dict[str, Any]:
    upload_bytes = await file.read()
    frame = _load_csv_frame(upload_bytes)
    dataset_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    stored_name = f"{dataset_id}-{_slugify(file.filename or 'dataset')}.csv"
    stored_path = _paths()["uploads"] / stored_name
    stored_path.write_bytes(upload_bytes)

    verification = verify_dataset_frame(frame)
    item = {
        "dataset_id": dataset_id,
        "original_filename": file.filename or stored_name,
        "stored_filename": stored_name,
        "stored_path": str(stored_path),
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "rows": verification["summary"]["rows"],
        "columns": verification["summary"]["columns"],
        "status": "verified" if not verification["summary"]["missing_features"] else "needs_review",
        "summary": verification["summary"],
        "top_risk_rows": verification["top_risk_rows"],
    }

    catalog = load_upload_catalog()
    catalog = [entry for entry in catalog if entry.get("dataset_id") != dataset_id]
    catalog.insert(0, item)
    _save_upload_catalog(catalog)
    return item


def get_uploaded_dataset(dataset_id: str) -> Dict[str, Any]:
    for item in load_upload_catalog():
        if item.get("dataset_id") == dataset_id:
            return item
    raise KeyError(f"Unknown dataset_id: {dataset_id}")


def _load_frame_from_catalog_item(item: Dict[str, Any]) -> pd.DataFrame:
    stored_path = Path(item["stored_path"])
    if not stored_path.exists():
        raise FileNotFoundError(f"Uploaded dataset missing on disk: {stored_path}")
    return pd.read_csv(stored_path)


def _report_pdf_path(report_id: str) -> Path:
    return _paths()["reports"] / f"{report_id}.pdf"


def _render_pdf(report: Dict[str, Any], pdf_path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "SGCC Analytics Report", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Generated: {report['generated_at']}", ln=True)
    pdf.cell(0, 8, f"Dataset: {report['dataset_label']}", ln=True)
    pdf.ln(4)

    def section(title: str, lines: List[str]) -> None:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, title, ln=True)
        pdf.set_font("Arial", "", 10)
        for line in lines:
            pdf.multi_cell(0, 6, line)
        pdf.ln(2)

    section(
        "Model snapshot",
        [
            f"Recall: {report['model_metrics']['metrics'].get('recall', 0.0):.3f}",
            f"Precision: {report['model_metrics']['metrics'].get('precision', 0.0):.3f}",
            f"Customers monitored: {report['model_metrics'].get('customers_monitored', 0)}",
            f"Flagged customers: {report['model_metrics'].get('flagged_today', 0)}",
        ],
    )
    section(
        "Dataset verification",
        [
            f"Rows: {report['dataset_summary'].get('rows', 0)}",
            f"Columns: {report['dataset_summary'].get('columns', 0)}",
            f"Average probability: {report['dataset_summary'].get('mean_probability', 0.0):.3f}",
            f"Verification accuracy: {report['dataset_summary'].get('verification_accuracy') if report['dataset_summary'].get('verification_accuracy') is not None else 'n/a'}",
        ],
    )
    section(
        "Public context",
        [
            f"Country: {report['context']['country'].get('name', 'N/A')}",
            f"Capital: {report['context']['country'].get('capital', 'N/A')}",
            f"Weather temperature: {report['context']['weather'].get('temperature_2m', 'n/a')}",
            f"Holiday count this year: {len(report['context']['holidays'])}",
        ],
    )
    section(
        "Top risk rows",
        [
            f"{row.get('customer_id', row.get('row_index', 'row'))}: probability={row.get('prediction_probability', row.get('risk_score', 0.0)):.3f}, tier={row.get('risk_tier', 'low')}"
            for row in report.get("top_risk_rows", [])[:10]
        ] or ["No rows available"],
    )
    pdf.output(str(pdf_path))


def _store_report_index(entry: Dict[str, Any]) -> None:
    index = load_report_index()
    index = [item for item in index if item.get("report_id") != entry["report_id"]]
    index.insert(0, entry)
    _save_report_index(index)


def generate_report(dataset_id: Optional[str] = None, country_code: str = "DZ", latitude: float = 36.7538, longitude: float = 3.0588) -> Dict[str, Any]:
    if dataset_id:
        dataset_item = get_uploaded_dataset(dataset_id)
        frame = _load_frame_from_catalog_item(dataset_item)
        verification = verify_dataset_frame(frame)
        dataset_label = dataset_item["original_filename"]
    else:
        from backend.services.data import get_dataset_summary

        dataset_item = None
        frame = None
        verification = {
            "summary": get_dataset_summary(),
            "top_risk_rows": [
                {
                    **row,
                    "prediction_probability": row.get("risk_score", 0.0),
                }
                for row in get_customer_rankings()[:10]
            ],
        }
        dataset_label = get_dataset_summary()["dataset_path"]

    report_id = f"report-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    generated_at = datetime.utcnow().isoformat() + "Z"
    model_metrics = get_model_metrics()
    country = get_country_context(country_code)
    holidays = get_public_holidays(country_code)
    weather = get_weather_context(latitude, longitude)

    report = {
        "report_id": report_id,
        "dataset_id": dataset_id,
        "dataset_label": dataset_label,
        "generated_at": generated_at,
        "model_metrics": model_metrics,
        "dataset_summary": verification["summary"],
        "top_risk_rows": verification["top_risk_rows"],
        "context": {
            "country": country,
            "holidays": holidays[:10],
            "weather": weather,
        },
    }

    pdf_path = _report_pdf_path(report_id)
    _render_pdf(report, pdf_path)

    entry = {
        "report_id": report_id,
        "dataset_id": dataset_id,
        "dataset_label": dataset_label,
        "generated_at": generated_at,
        "pdf_path": str(pdf_path),
    }
    _store_report_index(entry)

    report.update(
        {
            "pdf_path": str(pdf_path),
            "download_url": f"/api/reports/{report_id}/download",
        }
    )
    return report


def get_report_entry(report_id: str) -> Dict[str, Any]:
    for item in load_report_index():
        if item.get("report_id") == report_id:
            return item
    raise KeyError(f"Unknown report_id: {report_id}")


def build_report_file_response(report_id: str) -> FileResponse:
    entry = get_report_entry(report_id)
    pdf_path = Path(entry["pdf_path"])
    if not pdf_path.exists():
        raise FileNotFoundError(f"Report PDF missing on disk: {pdf_path}")
    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)