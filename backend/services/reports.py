"""PDF reports.

Operations (no labels): the portfolio overview, an uploaded dataset's results, and a
customer's case file. Research: the test-set evaluation, calibration and the
statistical comparison of the pipelines. The two are never mixed in one report.

The page layout (fpdf2) is in ``pdf_layout``, imported only when a report is built:
an old or missing PDF library disables reports and /api/health says how to fix it.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, insert, select

from . import db
from . import research
from .blobstore import get_blobstore
from .data import get_customer_timeseries
from .datasets import get_dataset, load_dataset, retention_days, summarize
from .errors import NotFoundError
from .model import explain_customer, explanation_check, get_decision_threshold, get_model_metrics, threshold_preview
from .operations import get_case, list_cases

MIN_FPDF_VERSION = (2, 7, 8)
_ID_PATTERN = re.compile(r"(portfolio|dataset|case|research)-[0-9]{14}-[0-9a-f]{8}")


class ReportsUnavailable(RuntimeError):
    """PDF reports cannot be produced in this environment; the message says how to fix it."""


def _version(text: str) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def _fpdf_version() -> str:
    """Version of the installed PDF library: fpdf2, or the older PyFPDF that shares its name."""
    try:
        import fpdf
    except ImportError:
        return "not installed"
    return str(getattr(fpdf, "FPDF_VERSION", "unknown"))


def reports_status() -> Dict[str, Any]:
    installed = _fpdf_version()
    ok = bool(_version(installed)) and _version(installed) >= MIN_FPDF_VERSION
    return {
        "available": ok,
        "fpdf_version": installed,
        "detail": None if ok else (
            f"PDF reports need fpdf2 >= {'.'.join(map(str, MIN_FPDF_VERSION))} (installed: {installed}). "
            "Run: pip install -r requirements.lock"
        ),
    }


def _pct(value: Optional[float], digits: int = 1) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def _num(value: Optional[float], digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Report content
# ---------------------------------------------------------------------------

def _document(title: str, subtitle: str, footer_note: str):
    """A new report page; the PDF library is imported here, after reports_status() allowed it."""
    from .pdf_layout import Report

    return Report(title, subtitle, footer_note)


RESPONSIBLE_USE = ("A flag is a reason to inspect, not evidence of theft. Conclusions rest on the field "
                   "investigation; a customer found honest is cleared and can appeal (proposal section 3.13).")
PROBABILITY_NOTE = ("Probabilities are calibrated on validation customers: among customers scored 0.30, about 30 in "
                    "100 were thieves there. Expected counts are sums of these probabilities, not observed outcomes.")


def _footer(metrics: Dict[str, Any]) -> str:
    return f"Model {metrics['pipeline']} v{metrics['model_version']}  ·  threshold in service {metrics['threshold']:.3f}"


def _portfolio(generated: str) -> tuple:
    metrics = get_model_metrics()
    cases = list_cases(status=None, tier=None, search=None, page=1, page_size=30)
    preview = threshold_preview(metrics["threshold"])["validation"]
    population = metrics["population"]
    pdf = _document("Portfolio risk report", f"Generated {generated}  ·  {metrics['customers_monitored']:,} customers", _footer(metrics))

    pdf.heading("Summary")
    pdf.paragraph(
        f"{metrics['flagged']:,} of {metrics['customers_monitored']:,} customers are at or above the operating "
        f"threshold ({metrics['threshold']:.3f}); {metrics['risk_tier_distribution']['high']:,} are high risk. "
        f"Inspecting every flagged customer should find about {metrics['expected_thefts_flagged']:.0f} thefts "
        f"(an estimate from calibrated probabilities)."
    )
    pdf.facts([
        ("Customers monitored", f"{metrics['customers_monitored']:,}"), ("Flagged", f"{metrics['flagged']:,}"),
        ("High risk", f"{metrics['risk_tier_distribution']['high']:,}"), ("Medium risk", f"{metrics['risk_tier_distribution']['medium']:,}"),
        ("Expected thefts, flagged", f"{metrics['expected_thefts_flagged']:.1f}"), ("Expected thefts, all", f"{metrics['expected_thefts_total']:.1f}"),
        ("Threshold in service", f"{metrics['threshold']:.3f}"), ("Trained threshold", f"{metrics['trained_threshold']:.3f}"),
        ("Population", population["filename"]), ("Source", "training sample" if population["source"] == "sample" else "uploaded readings"),
    ])
    pdf.paragraph(PROBABILITY_NOTE)

    pdf.heading("What the threshold means (validation customers)")
    pdf.paragraph("Measured on the validation customers the threshold was chosen on; the research report has the "
                  "test-set evaluation.")
    pdf.facts([
        ("Precision", _num(preview["precision"])), ("Recall", _num(preview["recall"])),
        ("Thefts caught", f"{preview['tp']:,} of {preview['tp'] + preview['fn']:,}"), ("Wasted visits", f"{preview['fp']:,}"),
    ])

    pdf.heading("Case workflow")
    counts = cases["status_counts"]
    pdf.facts([(status.capitalize(), f"{counts[status]:,}") for status in counts], columns=2)

    pdf.heading(f"Top {len(cases['items'])} cases by theft probability")
    pdf.grid(["Rank", "Customer", "Probability", "Tier", "Status", "Strongest signal"],
             [[c["rank"], c["customer_id"][:16], f"{c['risk_score']:.3f}", c["risk_tier"], c["status"],
               f"{c['top_driver']['label']} ({c['top_driver']['display_value']})" if c["top_driver"] else "-"]
              for c in cases["items"]],
             widths=[12, 34, 20, 16, 22, 74], align=["RIGHT", "LEFT", "RIGHT", "LEFT", "LEFT", "LEFT"])
    pdf.paragraph(RESPONSIBLE_USE)
    return pdf, f"{metrics['customers_monitored']:,} customers at threshold {metrics['threshold']:.3f}"


_LABEL_NAMES = {0: "honest", 1: "theft"}


def _dataset(generated: str, dataset_id: str) -> tuple:
    item = get_dataset(dataset_id)
    threshold = get_decision_threshold()
    summary = summarize(load_dataset(dataset_id), threshold, top_n=20)
    metrics = get_model_metrics()
    pdf = _document("Dataset scoring report", f"Generated {generated}  ·  {item['filename']}", _footer(metrics))

    pdf.heading("Dataset")
    pdf.facts([
        ("File", item["filename"]), ("Uploaded", f"{item['uploaded_at'].replace('T', ' ')[:16]} by {item['uploaded_by']}"),
        ("Format", "SGCC meter data" if summary["format"] == "consumption" else "Model features"),
        ("Customers", f"{summary['customers']:,}"),
        ("Days of readings" if summary["days"] else "Features", f"{summary['days']:,}" if summary["days"] else f"{summary['features_expected']}"),
        ("Labels", "yes" if summary["labelled"] else "no"),
    ])

    pdf.heading("Results at the operating threshold")
    pdf.facts([
        ("Threshold", f"{threshold:.3f}"), ("Flagged", f"{summary['flagged']:,} ({_pct(summary['flagged'] / max(summary['customers'], 1))})"),
        ("High risk", f"{summary['tiers']['high']:,}"), ("Medium risk", f"{summary['tiers']['medium']:,}"),
        ("Mean probability", _num(summary["mean_probability"])), ("", ""),
    ])
    if summary["label_metrics"]:
        lm = summary["label_metrics"]
        pdf.heading("Agreement with the file's own labels")
        pdf.facts([
            ("Labelled theft", f"{lm['theft']:,}"), ("Theft caught", f"{lm['caught']:,}"),
            ("Precision", _num(lm["precision"])), ("Recall", _num(lm["recall"])),
            ("ROC-AUC", _num(lm["roc_auc"])), ("PR-AUC", _num(lm["pr_auc"])),
        ])

    pdf.heading(f"Top {len(summary['top'])} customers by theft probability")
    headings = ["#", "Customer", "Probability", "Tier"] + (["Label"] if summary["labelled"] else [])
    rows = [[i + 1, t["customer_id"][:32], f"{t['probability']:.3f}", t["risk_tier"]] + ([_LABEL_NAMES[t["label"]]] if summary["labelled"] else [])
            for i, t in enumerate(summary["top"])]
    pdf.grid(headings, rows, widths=[10, 80, 30, 30] + ([28] if summary["labelled"] else []))
    pdf.paragraph(PROBABILITY_NOTE)
    return pdf, item["filename"]


def _case(generated: str, customer_id: str) -> tuple:
    case = get_case(customer_id)
    explanation = explain_customer(customer_id)
    series = get_customer_timeseries(customer_id)
    metrics = get_model_metrics()
    pdf = _document("Case file", f"Generated {generated}  ·  customer {customer_id}", _footer(metrics))

    points = series["points"]
    observed = [p for p in points if p["kwh"] is not None]
    longest = run = 0
    for p in points:
        run = run + 1 if p["kwh"] is None else 0
        longest = max(longest, run)
    months: Dict[str, List[float]] = {}
    for p in points:
        months.setdefault(p["date"][:7], [])
        if p["kwh"] is not None:
            months[p["date"][:7]].append(p["kwh"])

    pdf.heading("Assessment")
    pdf.facts([
        ("Customer", customer_id), ("Theft probability", f"{case['risk_score']:.3f}"),
        ("Rank", f"{case['rank']:,} of {case['population']:,}"), ("Risk tier", case["risk_tier"]),
        ("Flagged", "yes" if case["flagged"] else "no"), ("Case status", case["status"]),
    ])
    if case["resolution"]:
        r = case["resolution"]
        pdf.paragraph(f"Resolved as {r['outcome']} by {r['by']} on {r['at'].replace('T', ' ')[:16]} UTC: {r['reason']} "
                      f"(evidence: {r['evidence']}).")
    if case["note"]:
        pdf.paragraph(f"Analyst note: {case['note']}")

    pdf.heading("Meter readings")
    pdf.facts([
        ("First reading", observed[0]["date"] if observed else "-"), ("Last reading", observed[-1]["date"] if observed else "-"),
        ("Days read", f"{len(observed):,} of {len(points):,} ({_pct(len(observed) / max(len(points), 1), 0)})"),
        ("Longest gap", f"{longest:,} days"),
        ("Use while reporting", f"{sum(p['kwh'] for p in observed) / len(observed):.2f} kWh/day" if observed else "-"),
        ("", ""),
    ])
    pdf.monthly_chart([(key, sum(v) / len(v) if v else None) for key, v in months.items()])

    pdf.heading("Why the model scored this customer")
    parts = explanation.get("parts")
    if parts:
        pdf.paragraph("The model in service is a hybrid: " + " and ".join(
            f"{p['label']} ({p['probability']:.3f}, weight {p['weight']:.2f})" for p in parts)
            + f", blended and recalibrated to the probability {explanation['probability']:.3f}. The signals below "
            "explain the XGBoost part; the console's case page also shows which weeks of readings raised the "
            "sequence model's score.")
        pdf.paragraph("SHAP contributions to the XGBoost part's raw score, in log-odds: positive values push towards "
                      f"theft, negative away from it. From the base value ({explanation['base_value']:+.2f}) they add up "
                      f"to its raw score {explanation['raw_score']:.3f} (calibrated: {explanation['tree_probability']:.3f}).")
    else:
        pdf.paragraph("SHAP contributions to the model's raw score, in log-odds: positive values push towards theft, "
                      f"negative away from it. From the base value ({explanation['base_value']:+.2f}) they add up to the raw "
                      f"score {explanation['raw_score']:.3f}, which calibration maps to the probability "
                      f"{explanation['probability']:.3f} without changing the ranking.")
    pdf.grid(["Signal", "Value", "Contribution"],
             [[c["label"], c["display_value"], f"{c['shap_value']:+.3f}"] for c in explanation["contributions"][:10]],
             widths=[82, 62, 34], align=["LEFT", "LEFT", "RIGHT"])

    check = explanation_check(customer_id)
    pdf.heading("Explanation consistency (LIME)")
    pdf.paragraph(f"{check['message']} LIME's strongest signals: "
                  + ", ".join(f"{a['label']} ({a['weight']:+.3f})" for a in check["lime"]) + f". {check['note']}")

    pdf.heading("Case history")
    history = case["history"] or [{"at": "", "actor": "", "event": "No case activity yet."}]
    pdf.grid(["When (UTC)", "Who", "Event"], [[h["at"].replace("T", " ")[:16], h["actor"], h["event"]] for h in history],
             widths=[34, 28, 116])
    pdf.paragraph(RESPONSIBLE_USE)
    return pdf, customer_id


def _interval(bounds: Optional[List[float]]) -> str:
    return "-" if not bounds else f"{bounds[0]:.3f}-{bounds[1]:.3f}"


def _research(generated: str) -> tuple:
    evaluation = research.evaluation()
    population = evaluation["population"]
    m, cm = evaluation["metrics"], evaluation["confusion_matrix"]
    pdf = _document("Research evaluation", f"Generated {generated}  ·  {population['description']}",
                  f"Research record  ·  model {evaluation['pipeline']} v{evaluation['model_version']}")

    pdf.heading("Population")
    pdf.paragraph(f"All figures in this report describe the {population['description']}. {population['limitation']}")

    pdf.heading(f"Served pipeline: {evaluation['pipeline_label']}")
    pdf.facts([
        ("ROC-AUC", _num(m["auc"])), ("PR-AUC", _num(m["pr_auc"])),
        ("Precision", _num(m["precision"])), ("Recall", _num(m["recall"])),
        ("F1", _num(m["f1"])), ("MCC", _num(m["mcc"])),
        ("Threshold", _num(evaluation["threshold"])), ("G-mean", _num(m["gmean"])),
    ])
    pdf.grid(["", "Actually theft", "Actually honest"],
             [["Flagged", f"{cm['tp']:,} caught", f"{cm['fp']:,} wasted visits"],
              ["Not flagged", f"{cm['fn']:,} missed", f"{cm['tn']:,} correctly cleared"]],
             widths=[40, 69, 69])

    rows = research.model_comparison()
    if rows:
        pdf.heading("Pipelines compared (each at its own validation-chosen threshold)")
        pdf.grid(["Pipeline", "PR-AUC", "95% CI", "F1", "MCC", "p PR-AUC", "p McNemar"],
                 [[("* " if r["served"] else "") + r["label"], _num(r["pr_auc"]), _interval(r["pr_auc_ci"]), _num(r["f1"]),
                   _num(r["mcc"]), _num(r["p_value_pr_auc"]), "-" if r["p_value_mcnemar"] is None else f"{r['p_value_mcnemar']:.2g}"]
                  for r in rows],
                 widths=[58, 16, 26, 14, 14, 22, 28], align=["LEFT"] + ["RIGHT"] * 6)
        significance = research.significance()
        if significance:
            pdf.paragraph(f"* the pipeline in service. p-values compare each pipeline with the proposed one: a paired "
                          f"stratified bootstrap ({significance['resamples']:,} resamples of the test customers) for PR-AUC, "
                          "and McNemar's exact test on the flag decisions, both Holm-corrected across the four comparisons.")

    calibration = research.calibration()
    if calibration["pipelines"]:
        pdf.heading("Calibration on the test customers")
        pdf.grid(["Pipeline", "Brier raw", "Brier Platt", "ECE raw", "ECE Platt", "ECE isotonic"],
                 [[r["label"], _num(r["test_raw"]["brier"], 4), _num(r["test_platt"]["brier"], 4), _num(r["test_raw"]["ece"], 4),
                   _num(r["test_platt"]["ece"], 4), _num(r["test_isotonic"]["ece"], 4)] for r in calibration["pipelines"]],
                 widths=[62, 22, 24, 22, 22, 26], align=["LEFT"] + ["RIGHT"] * 5)
        pdf.paragraph("Platt scaling is fitted on the validation customers and served; isotonic regression is shown for comparison.")
    return pdf, population["description"]


# ---------------------------------------------------------------------------
# Catalogue and files
# ---------------------------------------------------------------------------

def _blob_key(report_id: str) -> str:
    return f"reports/{report_id}.pdf"


def generate_report(kind: str, subject_id: Optional[str], actor: str) -> Dict[str, Any]:
    """Render and store a report. `subject_id` is the dataset id or customer id it covers."""
    status = reports_status()
    if not status["available"]:
        raise ReportsUnavailable(status["detail"])
    now = datetime.now(timezone.utc)
    generated = now.strftime("%d %b %Y %H:%M UTC")
    builders = {"portfolio": lambda: _portfolio(generated), "dataset": lambda: _dataset(generated, str(subject_id)),
                "case": lambda: _case(generated, str(subject_id)), "research": lambda: _research(generated)}
    if kind not in builders:
        raise ValueError(f"Unknown report kind {kind!r}")
    pdf, subject = builders[kind]()

    report_id = f"{kind}-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    content = bytes(pdf.output())
    get_blobstore().put(_blob_key(report_id), content, "application/pdf")
    entry = {"report_id": report_id, "kind": kind, "title": pdf.report_title, "subject": subject[:200],
             "created_at": now.isoformat(timespec="milliseconds"), "created_by": actor, "bytes": len(content)}
    with db.transaction() as connection:
        connection.execute(insert(db.reports).values(**entry, blob_key=_blob_key(report_id)))
    return entry


def list_reports(limit: int = 50) -> List[Dict[str, Any]]:
    with db.transaction() as connection:
        rows = connection.execute(select(db.reports).order_by(db.reports.c.created_at.desc()).limit(limit)).mappings().all()
    return [{key: row[key] for key in ("report_id", "kind", "title", "subject", "created_at", "created_by", "bytes")} for row in rows]


def _row(report_id: str):
    if not _ID_PATTERN.fullmatch(report_id):
        raise NotFoundError(f"Unknown report: {report_id}")
    with db.transaction() as connection:
        row = connection.execute(select(db.reports).where(db.reports.c.report_id == report_id)).mappings().first()
    if row is None:
        raise NotFoundError(f"Unknown report: {report_id}")
    return row


def report_pdf(report_id: str) -> bytes:
    """A generated report's PDF; NotFoundError if the id is malformed or unknown."""
    return get_blobstore().get(_row(report_id)["blob_key"])


def delete_report(report_id: str) -> None:
    row = _row(report_id)
    get_blobstore().delete(row["blob_key"])
    with db.transaction() as connection:
        connection.execute(delete(db.reports).where(db.reports.c.report_id == report_id))


def purge_expired_reports() -> int:
    """Delete reports older than RETENTION_DAYS (0 keeps them)."""
    days = retention_days()
    if not days:
        return 0
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    with db.transaction() as connection:
        expired = connection.execute(select(db.reports.c.report_id, db.reports.c.blob_key)
                                     .where(db.reports.c.created_at < cutoff)).all()
    for report_id, blob_key in expired:
        get_blobstore().delete(blob_key)
        with db.transaction() as connection:
            connection.execute(delete(db.reports).where(db.reports.c.report_id == report_id))
    return len(expired)
